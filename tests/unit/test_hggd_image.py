"""The network input of HGGD: Pillow's resize in tensor form, and the image layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from grasppanda.models.detectors.hggd.image import (
    encode_image,
    pil_bicubic_weights,
    resize_colour,
    resize_depth,
)
from grasppanda.structures import ContractError


def pillow_colour(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``(H, W, 3)`` uint8 → Pillow's default resize to ``(height, width)``."""
    return np.array(Image.fromarray(image).resize((size[1], size[0])))


def pillow_depth(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(
        Image.fromarray(image.astype(np.float32)).resize((size[1], size[0])), np.float32
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ((72, 128), (36, 64)),
        ((40, 64), (20, 32)),
        ((30, 50), (17, 23)),
        ((16, 16), (16, 8)),
        ((12, 20), (24, 40)),
    ],
)
def test_colour_resize_is_pillows_bit_for_bit(source, target):
    generator = np.random.default_rng(0)
    image = generator.integers(0, 256, size=(*source, 3), dtype=np.uint8)
    levels = torch.from_numpy(image).permute(2, 0, 1).double()
    mine = resize_colour(levels, target).permute(1, 2, 0).numpy()
    assert np.array_equal(mine, pillow_colour(image, target).astype(np.float64))
    assert mine.min() >= 0 and mine.max() <= 255  # clipped like 8-bit pixels


def test_colour_resize_rounds_after_each_pass():
    """Pillow stores 8 bits between its two passes; a single float pass differs."""
    generator = np.random.default_rng(1)
    image = generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    levels = torch.from_numpy(image).permute(2, 0, 1).double()
    horizontal = pil_bicubic_weights(64, 32)
    vertical = pil_bicubic_weights(48, 24)
    one_pass = torch.einsum("vh,chw,uw->cvu", vertical, levels, horizontal)
    two_pass = resize_colour(levels, (24, 32))
    assert not torch.equal(torch.round(one_pass).clamp(0, 255), two_pass)
    assert (torch.round(one_pass) - two_pass).abs().max() <= 2


@pytest.mark.parametrize(("source", "target"), [((72, 128), (36, 64)), ((30, 50), (17, 23))])
def test_depth_resize_is_pillows_float_mode(source, target):
    generator = np.random.default_rng(2)
    depth = generator.uniform(300.0, 900.0, size=source).astype(np.float32)
    depth[generator.random(source) < 0.1] = 0.0  # holes are blended like any value
    mine = resize_depth(torch.from_numpy(depth), target).numpy()
    reference = pillow_depth(depth, target)
    assert mine.dtype == np.float32
    assert np.abs(mine - reference).max() <= 1e-4  # millimetres: a float32 rounding at most
    assert (mine == reference).mean() > 0.99


def test_filter_weights_are_normalised_and_antialiased():
    weights = pil_bicubic_weights(128, 64)
    assert weights.shape == (64, 128) and weights.dtype == torch.float64
    assert torch.allclose(weights.sum(dim=1), torch.ones(64, dtype=torch.float64))
    assert int((weights[10] != 0).sum()) == 8  # the kernel is stretched by the reduction
    fixed = pil_bicubic_weights(128, 64, fixed_point=True)
    assert torch.equal(fixed, torch.round(fixed)) and float(fixed.abs().max()) < 2**22
    with pytest.raises(ValueError, match="sizes must be positive"):
        pil_bicubic_weights(0, 4)


def test_image_is_depth_then_colour_in_width_height_order():
    generator = np.random.default_rng(3)
    colour = generator.integers(0, 256, size=(2, 80, 128, 3), dtype=np.uint8)
    depth = generator.uniform(0.3, 0.9, size=(2, 80, 128)).astype(np.float32)
    rgb = torch.from_numpy(colour).permute(0, 3, 1, 2).float() / 255.0
    image = encode_image(rgb, torch.from_numpy(depth), (64, 40))
    assert tuple(image.shape) == (2, 4, 64, 40) and image.dtype == torch.float32  # (B, 4, U, V)
    for frame in range(2):
        reference = pillow_colour(colour[frame], (40, 64)).astype(np.float32) / 255.0
        assert np.array_equal(image[frame, 1:].numpy(), reference.transpose(2, 1, 0))
        small = pillow_depth(depth[frame] * 1000.0, (40, 64)) / 1000.0
        shifted = np.clip(small - small.mean(), -1, 1)
        assert np.abs(image[frame, 0].numpy() - shifted.T).max() < 1e-6
        assert abs(float(image[frame, 0].mean())) < 1e-6  # shifted by its own mean


def test_depth_channel_is_clipped_around_the_mean():
    depth = torch.full((1, 80, 128), 0.5)
    depth[0, :40] = 4.0  # half the image is 3.5 m further away
    rgb = torch.zeros(1, 3, 80, 128)
    image = encode_image(rgb, depth, (64, 40))
    assert float(image[0, 0].max()) == 1.0 and float(image[0, 0].min()) == -1.0
    loose = encode_image(rgb, depth, (64, 40), depth_clip=5.0)
    assert float(loose[0, 0].max()) > 1.5


def test_torch_resize_is_a_different_named_variant():
    generator = np.random.default_rng(4)
    rgb = torch.from_numpy(generator.random((1, 3, 80, 128), dtype=np.float32))
    rgb = torch.round(rgb * 255) / 255
    depth = torch.full((1, 80, 128), 0.5)
    reference = encode_image(rgb, depth, (64, 40))
    variant = encode_image(rgb, depth, (64, 40), resize="torch_bicubic")
    assert variant.shape == reference.shape
    assert not torch.allclose(variant, reference, atol=1e-3)
    assert (variant - reference).abs().max() < 0.1
    with pytest.raises(ValueError, match="unknown resize 'nearest'"):
        encode_image(rgb, depth, (64, 40), resize="nearest")


def test_mismatched_images_are_rejected():
    rgb, depth = torch.zeros(1, 3, 80, 128), torch.zeros(1, 80, 128)
    with pytest.raises(ContractError, match=r"expected rgb \(B, 3, H, W\)"):
        encode_image(rgb[0], depth, (64, 40))
    with pytest.raises(ContractError, match="not images of the same frames"):
        encode_image(rgb, torch.zeros(1, 40, 64), (64, 40))
    with pytest.raises(ContractError, match="not images of the same frames"):
        encode_image(rgb, torch.zeros(2, 80, 128), (64, 40))
