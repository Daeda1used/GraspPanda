"""Swapping the scene encoder of GSNet is one override, and only meaningful swaps pass.

Each dense encoder is put under the unchanged GSNet proposal, view stage, grouper and
head through a config override, built, and run on a synthetic tabletop cloud. The
networks are untrained, so the number of grasps is arbitrary (often zero); what is
asserted is that the override validates, the detector builds with the injected channel
counts, and prediction returns valid grasp groups. An encoder whose features live on a
sub-sampled set must be refused when the config is checked.
"""

from __future__ import annotations

import pytest
import torch

from grasppanda.config import Config, ConfigError, available_configs
from grasppanda.registry import DETECTORS
from grasppanda.structures import Observation, PointSet, SampleKey

pytest.importorskip("spconv", reason="needs the spconv extra")
pytestmark = pytest.mark.gpu

BASE = "gsnet/gsnet_sunet14_g1b-realsense_10e"
DENSE = {
    "SparseUNet14": [],
    "PTv3": ["model.backbone={type: PTv3, in_channels: 3, grid_size: 0.005}"],
    "PointNet2Dense": ["model.backbone={type: PointNet2Dense, in_channels: 3}"],
    "FeaturePropagation": [
        "model.backbone={type: FeaturePropagation, backbone: {type: PointNet2, in_channels: 3}}"
    ],
    "PointNeXt": ["model.backbone={type: PointNeXt, in_channels: 3}"],
    "LitePT": ["model.backbone={type: LitePT, in_channels: 3, grid_size: 0.005}"],
    "OACNNs": [
        "model.backbone={type: OACNNs, in_channels: 3, out_channels: 512, grid_size: 0.005}"
    ],
    "SpUNet": [
        "model.backbone={type: SpUNet, in_channels: 3, out_channels: 512, grid_size: 0.005}"
    ],
}
BASELINE = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"
EXPERIMENTAL = [name for name in available_configs() if name.startswith("experimental/")]


def tabletop(count: int = 6000) -> Observation:
    generator = torch.Generator().manual_seed(0)
    cloud = torch.rand(1, count, 3, generator=generator) * torch.tensor([0.5, 0.4, 0.15])
    cloud = cloud + torch.tensor([-0.25, -0.2, 0.4])
    return Observation(
        keys=[SampleKey("synthetic", 0, "none", 0)],
        points=PointSet.from_dense(cloud),
        point_feats=torch.ones(count, 3),
    )


@pytest.mark.parametrize("name", sorted(DENSE))
def test_dense_encoders_drop_in_under_gsnet(name, cuda):
    config = Config.load(BASE, [*DENSE[name], "model.proposal.num_seeds=256"])
    model = DETECTORS.build(config["model"]).to(cuda).eval()
    assert model.backbone.output == "dense"
    with torch.no_grad():
        (grasps,) = model.predict(tabletop().to(cuda))
    grasps.validate()
    assert len(grasps) <= 256 and grasps.key == SampleKey("synthetic", 0, "none", 0)


def test_a_subsampled_encoder_is_refused_with_the_way_out():
    with pytest.raises(ConfigError) as error:
        Config.load(BASE, ["model.backbone={type: PointNet2, in_channels: 3}"])
    message = str(error.value)
    assert (
        "requires 'dense' scene features" in message and "PointNet2 returns 'subsampled'" in message
    )
    assert "FeaturePropagation" in message  # the error names the explicit adapter


def test_a_dense_encoder_serves_the_baseline_and_keeps_the_reference_seeds(cuda):
    """The other direction, at full size: GraspNet-baseline refuses a sparse U-Net, names
    the sub-sampling adapter, and with the reference sampling rules the adapted encoder
    hands over the very 1024 points the reference PointNet++ ends on."""
    with pytest.raises(ConfigError) as error:
        Config.load(BASELINE, ["model.backbone={type: SparseUNet14}"])
    message = str(error.value)
    assert "requires 'subsampled' scene features" in message
    assert "SparseUNet14 returns 'dense'" in message and "FurthestPointSubsample" in message
    adapted = (
        "model.backbone={type: FurthestPointSubsample, fps_tie_break: pointnet2_cuda, "
        "backbone: {type: SparseUNet14, out_channels: 256}}"
    )
    model = DETECTORS.build(Config.load(BASELINE, [adapted])["model"]).to(cuda).eval()
    reference = DETECTORS.build(Config.load(BASELINE)["model"]).to(cuda).eval()
    observation = tabletop().to(cuda)
    with torch.no_grad():
        kept = model.backbone(observation.points, observation.point_feats)
        seeds = reference.backbone(observation.points, None)
        (grasps,) = model.predict(observation)
    assert model.backbone.output == "subsampled" and kept.feats.shape == (1024, 256)
    assert torch.equal(kept.points.parent_index, seeds.points.parent_index)
    assert torch.equal(kept.points.coords, seeds.points.coords)
    grasps.validate()
    assert len(grasps) <= 1024


@pytest.mark.parametrize("name", EXPERIMENTAL)
def test_experimental_configs_build_and_predict_at_full_size(name, cuda):
    """Every shipped swap config, as written: full-size model, one synthetic frame."""
    config = Config.load(name)
    model = DETECTORS.build(config["model"]).to(cuda).eval()
    with torch.no_grad():
        (grasps,) = model.predict(tabletop().to(cuda))
    grasps.validate()
    assert len(grasps) <= model.proposal.num_seeds
    assert EXPERIMENTAL and "experimental" in str(config.path)
