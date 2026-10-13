"""DepthNormals and CoordinateFeatures: the normal of a pixel against a per-pixel oracle
and against planes with a known normal, the two encodings, and how the features travel
through sampling and augmentation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError, SampleKey
from grasppanda.transforms.frame import PointSample
from grasppanda.transforms.normals import CoordinateFeatures, DepthNormals, depth_normal_map
from grasppanda.transforms.points import RandomFlip, RandomSample

from .conftest import REPO

INTRINSICS = np.array([[40.0, 0.0, 5.5], [0.0, 42.0, 4.0], [0.0, 0.0, 1.0]])
HEIGHT, WIDTH = 9, 12


def back_project(depth: np.ndarray) -> np.ndarray:
    v, u = np.mgrid[: depth.shape[0], : depth.shape[1]]
    z = depth.astype(np.float64)
    x = (u - INTRINSICS[0, 2]) * z / INTRINSICS[0, 0]
    y = (v - INTRINSICS[1, 2]) * z / INTRINSICS[1, 1]
    return np.stack([x, y, z], axis=-1)


def plane_depth(normal: np.ndarray, offset: float) -> np.ndarray:
    """Depth image of the plane ``normal · p = offset`` seen through INTRINSICS."""
    v, u = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack(
        [
            (u - INTRINSICS[0, 2]) / INTRINSICS[0, 0],
            (v - INTRINSICS[1, 2]) / INTRINSICS[1, 1],
            np.ones((HEIGHT, WIDTH)),
        ],
        axis=-1,
    )
    return offset / (rays @ normal)


def sample_of(depth: np.ndarray) -> PointSample:
    depth = depth.astype(np.float32)
    pixels = np.flatnonzero(depth > 0)
    return PointSample(
        key=SampleKey("unit", 3, "none", 1),
        intrinsics=INTRINSICS,
        points=back_project(depth).reshape(-1, 3)[pixels],
        pixels=pixels,
        image_size=depth.shape,
        depth=depth,
        rng=np.random.default_rng(0),
    )


def random_depth(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.4, 0.9, size=(HEIGHT, WIDTH)).astype(np.float32)
    depth[rng.random((HEIGHT, WIDTH)) < 0.15] = 0.0  # missing pixels
    return depth


# -------------------------------------------------------------------- the estimator


def normal_oracle(depth: np.ndarray, step: int, towards_camera: bool) -> np.ndarray:
    """Equation 5 of the paper, one pixel at a time, with clamped neighbours."""
    cloud = back_project(depth)
    height, width = depth.shape
    out = np.zeros((height, width, 3))
    for v in range(height):
        for u in range(width):
            right = cloud[v, min(u + step, width - 1)]
            left = cloud[v, max(u - step, 0)]
            lower = cloud[min(v + step, height - 1), u]
            upper = cloud[max(v - step, 0), u]
            normal = np.cross(right - left, lower - upper)
            length = np.linalg.norm(normal)
            if length > 0:
                out[v, u] = normal / length * (-1.0 if towards_camera else 1.0)
    return out


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("towards_camera", [True, False])
def test_normal_map_is_the_cross_product_of_the_two_image_tangents(step, towards_camera):
    depth = random_depth()
    normals = depth_normal_map(depth, INTRINSICS, step, towards_camera)
    assert normals.shape == (HEIGHT, WIDTH, 3) and normals.dtype == np.float64
    assert np.allclose(normals, normal_oracle(depth, step, towards_camera), atol=1e-12)
    lengths = np.linalg.norm(normals, axis=-1)
    assert np.all((np.abs(lengths - 1) < 1e-12) | (lengths == 0))


def test_a_plane_has_its_own_normal_everywhere_and_faces_the_camera():
    normal = np.array([0.3, -0.2, 1.0])
    normal /= np.linalg.norm(normal)
    depth = plane_depth(normal, offset=0.6)
    towards = depth_normal_map(depth, INTRINSICS)
    # Tangents of a plane lie in it, so central and one-sided differences agree: every pixel.
    assert np.allclose(towards, -normal, atol=1e-9)
    assert np.all(towards[..., 2] < 0)  # towards the camera at the origin
    assert np.allclose(depth_normal_map(depth, INTRINSICS, towards_camera=False), normal, atol=1e-9)
    assert np.allclose(depth_normal_map(depth, INTRINSICS, step=3), -normal, atol=1e-9)
    flat = depth_normal_map(np.full((HEIGHT, WIDTH), 0.5), INTRINSICS)
    assert np.allclose(flat, [0.0, 0.0, -1.0], atol=1e-12)


def test_undefined_normals_are_zero_and_arguments_are_checked():
    empty = depth_normal_map(np.zeros((4, 5)), INTRINSICS)
    assert not empty.any()  # every pixel on the camera centre: no tangent
    with pytest.raises(ValueError, match=r"depth must be \(H, W\)"):
        depth_normal_map(np.zeros((4, 5, 1)), INTRINSICS)
    with pytest.raises(ValueError, match="step must be at least 1"):
        depth_normal_map(np.zeros((4, 5)), INTRINSICS, step=0)
    for options, message in (
        ({"step": 0}, "step must be at least 1"),
        ({"encoding": "rgb"}, "encoding must be 'image8' or 'unit'"),
    ):
        with pytest.raises(ValueError, match=message):
            DepthNormals(**options)


# -------------------------------------------------------------------- the transform


def test_features_are_the_normals_of_the_points_in_the_8_bit_image_encoding():
    depth = random_depth(1)
    sample = DepthNormals()(sample_of(depth))
    normals = depth_normal_map(depth, INTRINSICS).reshape(-1, 3)[sample.pixels]
    assert sample.feats.shape == (len(sample), 3) and sample.feats.dtype == np.float32
    levels = np.floor((normals + 1.0) / 2.0 * 255.0)  # what an 8-bit normal image stores
    assert np.array_equal(sample.feats, (levels / 255.0).astype(np.float32))
    assert sample.feats.min() >= 0.0 and sample.feats.max() <= 1.0
    assert np.array_equal(np.rint(sample.feats * 255), levels)
    # A surface that faces the camera squarely: n = (0, 0, -1) -> (127, 127, 0) / 255.
    facing = DepthNormals()(sample_of(np.full((HEIGHT, WIDTH), 0.5)))
    assert np.array_equal(facing.feats, np.tile(np.float32([127 / 255, 127 / 255, 0.0]), (108, 1)))
    away = DepthNormals(towards_camera=False)(sample_of(np.full((HEIGHT, WIDTH), 0.5)))
    assert np.array_equal(away.feats[0], np.float32([127 / 255, 127 / 255, 1.0]))


def test_unit_encoding_keeps_the_vector():
    depth = random_depth(2)
    sample = DepthNormals(encoding="unit", step=2)(sample_of(depth))
    normals = depth_normal_map(depth, INTRINSICS, step=2).reshape(-1, 3)[sample.pixels]
    assert np.array_equal(sample.feats, normals.astype(np.float32))
    assert sample.feats.min() < 0


def test_it_needs_the_depth_image_before_sampling_and_consumes_it():
    sample = sample_of(random_depth())
    sample.depth = None
    with pytest.raises(ContractError, match=r"needs the depth image .* `images: \[depth\]`"):
        DepthNormals()(sample)
    sampled = RandomSample(20)(sample_of(random_depth()))
    with pytest.raises(ContractError, match="must run before the cloud .* is sampled"):
        DepthNormals()(sampled)
    wrong = sample_of(random_depth())
    wrong.depth = wrong.depth[:-1]
    with pytest.raises(ContractError, match="the cloud was taken from an image of"):
        DepthNormals()(wrong)
    assert DepthNormals()(sample_of(random_depth())).depth is None
    assert DepthNormals(keep_depth=True)(sample_of(random_depth())).depth is not None


def test_features_follow_sampling_and_stay_behind_when_the_cloud_is_mirrored():
    """The reference flips the cloud and the object poses and leaves the normals as they
    were; the features of this pipeline do the same."""
    sample = DepthNormals()(sample_of(random_depth(3)))
    feats, points = sample.feats.copy(), sample.points.copy()
    index = np.array([5, 0, 17, 5])
    sample = RandomSample(4).apply(sample, index)
    assert np.array_equal(sample.feats, feats[index])
    assert np.array_equal(sample.points, points[index])
    sample = RandomFlip("x", 1.0).apply(sample, True)
    assert np.array_equal(sample.points[:, 0], -points[index, 0])
    assert np.array_equal(sample.feats, feats[index])  # not mirrored
    kept = DepthNormals(keep_depth=True)(sample_of(random_depth(3)))
    with pytest.raises(ContractError, match="would leave the image-space data"):
        RandomFlip("x", 1.0).apply(kept, True)


def test_coordinate_features_go_in_front_of_whatever_is_there():
    sample = DepthNormals()(sample_of(random_depth(4)))
    normals = sample.feats.copy()
    sample = CoordinateFeatures()(RandomFlip("x", 1.0).apply(sample, True))
    assert sample.feats.shape == (len(sample), 6) and sample.feats.dtype == np.float32
    assert np.array_equal(
        sample.feats[:, :3], sample.points.astype(np.float32)
    )  # the flipped cloud
    assert np.array_equal(sample.feats[:, 3:], normals)
    alone = sample_of(random_depth(4))
    alone.depth = None
    alone = CoordinateFeatures()(alone)
    assert np.array_equal(alone.feats, alone.points.astype(np.float32))
    alone.feats = alone.feats[:-1]
    with pytest.raises(ContractError, match="feature rows for"):
        CoordinateFeatures()(alone)


def test_the_parity_recorder_writes_the_normal_image_this_transform_computes():
    """The reference dataset reads normal files that are not published; the recorder of
    the parity fixtures writes them itself, with code of its own (it cannot import this
    package). Both must be the same function of a depth image."""
    path = REPO / "tools" / "oracle" / "finegrasp_reference.py"
    spec = importlib.util.spec_from_file_location("finegrasp_reference", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rng = np.random.default_rng(5)
    raw = rng.integers(350, 900, size=(HEIGHT, WIDTH)).astype(np.uint16)  # millimetres
    raw[rng.random((HEIGHT, WIDTH)) < 0.15] = 0
    image = oracle.normal_image(raw, INTRINSICS, 1000.0)
    assert image.dtype == np.uint8 and image.shape == (HEIGHT, WIDTH, 3)
    metres = (raw / np.float32(1000.0)).astype(np.float32)  # as the dataset serves the depth
    sample = DepthNormals()(sample_of(metres))
    assert np.array_equal(sample.feats, (image.reshape(-1, 3)[sample.pixels] / 255.0).astype("f4"))
    away = oracle.normal_image(raw, INTRINSICS, 1000.0, towards_camera=False)
    sample = DepthNormals(towards_camera=False)(sample_of(metres))
    assert np.array_equal(np.rint(sample.feats * 255), away.reshape(-1, 3)[sample.pixels])


def test_both_are_registered_and_describe_themselves():
    step = TRANSFORMS.build({"type": "DepthNormals", "step": 2, "encoding": "unit"})
    assert step.config() == {
        "type": "DepthNormals",
        "version": 1,
        "step": 2,
        "towards_camera": True,
        "encoding": "unit",
        "keep_depth": False,
    }
    assert TRANSFORMS.build({"type": "CoordinateFeatures"}).config() == {
        "type": "CoordinateFeatures",
        "version": 1,
    }
