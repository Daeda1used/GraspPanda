"""Point painting: image features for the scene encoders of the seed-based family."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from grasppanda.combine.compose import ComposeError, compose, is_adapter
from grasppanda.combine.grid import METHODS
from grasppanda.models.backbones.image_painting import (
    ImagePainted,
    ImagePaintedSubsampled,
    sample_maps,
)
from grasppanda.models.image_encoders import ImageEncoder
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, Observation, PointSet, SampleKey

DENSE = {"type": "PointNeXt", "radius": 0.04}
SUBSAMPLED = {"type": "PointNet2", "npoints": [64, 32, 16, 8]}


class Identity(ImageEncoder):
    """Returns the image and a half-resolution copy: what is sampled is known."""

    input_layout = "rgb"
    in_channels = 3
    out_channels = (3, 3)
    out_strides = (1, 2)

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, image):
        return [image * self.scale, nn.functional.avg_pool2d(image, 2) * self.scale]


def scene(frames: int = 2, points: int = 800, size=(40, 64)) -> Observation:
    """Points that project onto known pixels of a colour ramp."""
    generator = torch.Generator().manual_seed(0)
    height, width = size
    intrinsics = torch.tensor([[60.0, 0.0, 31.5], [0.0, 60.0, 19.5], [0.0, 0.0, 1.0]])
    clouds, images = [], []
    for _ in range(frames):
        u = torch.randint(0, width, (points,), generator=generator).float()
        v = torch.randint(0, height, (points,), generator=generator).float()
        z = 0.4 + 0.3 * torch.rand(points, generator=generator)
        clouds.append(torch.stack([(u - 31.5) * z / 60.0, (v - 19.5) * z / 60.0, z], dim=1))
        ramp_u = torch.arange(width).float().expand(height, width) / width
        ramp_v = torch.arange(height).float()[:, None].expand(height, width) / height
        images.append(torch.stack([ramp_u, ramp_v, torch.zeros(height, width)]))
    return Observation(
        keys=[SampleKey("paint", 0, "realsense", index) for index in range(frames)],
        intrinsics=intrinsics.expand(frames, 3, 3).clone(),
        rgb=torch.stack(images),
        points=PointSet.cat([PointSet(cloud) for cloud in clouds]),
    )


def test_a_point_reads_the_feature_of_its_own_pixel():
    observation = scene()
    rgb = observation.rgb
    sampled = sample_maps([rgb], observation.points, observation.intrinsics, rgb.shape[-2:])
    coords = observation.points.coords
    u = (60.0 * coords[:, 0] / coords[:, 2] + 31.5).round().long()
    v = (60.0 * coords[:, 1] / coords[:, 2] + 19.5).round().long()
    assert torch.allclose(sampled[:, 0], u.float() / 64, atol=1e-4)
    assert torch.allclose(sampled[:, 1], v.float() / 40, atol=1e-4)
    # A coarser map is read at the same place; a point outside the image reads zero.
    half = nn.functional.avg_pool2d(rgb, 2)
    coarse = sample_maps([half], observation.points, observation.intrinsics, rgb.shape[-2:])
    assert (coarse[:, 0] - sampled[:, 0]).abs().max() < 1.5 / 64
    outside = PointSet(torch.tensor([[5.0, 0.0, 0.5]]))
    assert sample_maps([rgb[:1]], outside, observation.intrinsics[:1], (40, 64)).abs().sum() == 0
    world = PointSet(coords[:10], frame="table")
    with pytest.raises(ContractError, match="the cloud is in frame 'table'"):
        sample_maps([rgb[:1]], world, observation.intrinsics[:1], (40, 64))


@pytest.mark.parametrize("fusion", ["input", "output"])
@pytest.mark.parametrize(
    ("cls", "inner"), [(ImagePainted, DENSE), (ImagePaintedSubsampled, SUBSAMPLED)]
)
def test_painted_encoders_keep_the_contract_of_what_they_wrap(cls, inner, fusion):
    torch.manual_seed(0)
    observation = scene()
    wrapped = {**inner, "in_channels": 11 if fusion == "input" else 3}
    model = cls(backbone=wrapped, image_encoder=Identity(), fusion=fusion, paint_channels=8)
    plain = BACKBONES.build({**inner, "in_channels": 3})
    assert model.output == plain.output and model.out_channels == plain.out_channels
    assert model.in_channels == 3 and model.uses_observation
    assert model.backbone.in_channels == (11 if fusion == "input" else 3)
    points = observation.points
    out = model(points, torch.ones(len(points), 3), observation)
    assert out.feats.shape[1] == model.out_channels
    if model.output == "dense":
        assert out.points.id == points.id
    else:
        assert out.points.parent_id == points.id and len(out.points) < len(points)
    out.feats.sum().backward()
    assert model.project.weight.grad is not None
    if fusion == "input":  # the image network is trained through the 3-D network
        assert model.image_encoder.scale.grad.abs() > 0
    with pytest.raises(ContractError, match=r"Serve them with `images: \[rgb\]`"):
        model(points, torch.ones(len(points), 3), None)


def test_misuse_is_refused_with_the_way_out():
    with pytest.raises(ValueError, match="PointNet2 declares output='subsampled'. Use ImageP"):
        ImagePainted(backbone={**SUBSAMPLED, "in_channels": 32}, image_encoder=Identity())
    rgbd = Identity()
    rgbd.input_layout = "rgbd"
    with pytest.raises(ContractError, match="paints from the colour image"):
        ImagePainted(backbone=DENSE, image_encoder=rgbd)
    with pytest.raises(ValueError, match=r"image_scale must be in \(0, 1\]"):
        ImagePainted(backbone=DENSE, image_encoder=Identity(), image_scale=2.0)
    with pytest.raises(ValueError, match="Set its in_channels to what the detector feeds plus"):
        ImagePainted(backbone={**DENSE, "in_channels": 3}, image_encoder=Identity())
    problems = list(ImagePainted.check_config({"backbone": SUBSAMPLED}))
    assert any("needs an `image_encoder` node" in problem for problem in problems)
    assert any("PointNet2 returns `subsampled`" in problem for problem in problems)
    # They wrap an encoder, so compose treats them as adapters and never inserts them
    # by itself: without an image encoder they fit nothing.
    assert is_adapter(ImagePainted) and is_adapter(ImagePaintedSubsampled)
    with pytest.raises(ComposeError, match="is an adapter"):
        compose(METHODS["GSNet"], ["backbone=ImagePainted"])


def test_the_skeleton_hands_the_observation_to_a_painted_encoder():
    from grasppanda.registry import DETECTORS

    from .test_gsnet_detector import batch, tiny_config

    painted = {
        "type": "ImagePainted",
        "backbone": {"type": "PointNeXt", "radius": 0.04, "in_channels": 7},
        "image_encoder": Identity(),
        "paint_channels": 4,
    }
    model = DETECTORS.build(tiny_config(backbone=painted))
    assert {"points", "rgb", "intrinsics"} <= model.required_inputs
    assert model.backbone.backbone.in_channels == 7
    # The observation reaches the encoder: one without a colour image is refused by it.
    observation, _ = batch(2)
    with pytest.raises(ContractError, match=r"ImagePainted reads the colour image"):
        model.eval().predict(observation)
