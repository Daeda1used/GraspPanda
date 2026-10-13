"""FeaturePropagation: the one explicit bridge from subsampled to dense features."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch
from torch import Tensor

from grasppanda.config import Config, ConfigError
from grasppanda.models import Backbone
from grasppanda.models.backbones.feature_propagation import FeaturePropagation
from grasppanda.registry import BACKBONES, BuildError
from grasppanda.structures import ContractError, PointFeatures, PointSet

from .conftest import TBackbone

INNER = {
    "type": "PointNet2",
    "in_channels": 2,
    "npoints": [32, 16],
    "radii": [0.3, 0.6],
    "nsamples": [6, 6],
    "sa_mlps": [[8, 12], [12, 16]],
    "fp_mlps": [[14]],
}


@BACKBONES.register()
class EveryThirdPoint(Backbone):
    """A stand-in encoder: keeps every third point and returns chosen features on them."""

    output = "subsampled"

    def __init__(self, in_channels: int = 2, out_channels: int = 5, detach: bool = False) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.detach = in_channels, out_channels, detach

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        bounds = points.batch_offsets.tolist()
        kept = [torch.arange(start, stop, 3) for start, stop in pairwise(bounds)]
        index = torch.cat(kept)
        offsets = torch.tensor([0, *np.cumsum([len(rows) for rows in kept])])
        values = torch.cos(points.coords[index] @ torch.arange(15.0).reshape(3, 5))
        if self.detach:  # features on a set that is not a child of the input
            return PointFeatures(values, PointSet(points.coords[index], offsets))
        return PointFeatures(values, points.subset(index, offsets))


def batch(sizes: tuple[int, ...] = (50, 71), seed: int = 0) -> tuple[PointSet, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    clouds = [PointSet(torch.rand(size, 3, generator=generator)) for size in sizes]
    points = PointSet.cat(clouds)
    return points, torch.sin(points.coords[:, :2] * 5)


def test_dense_features_are_interpolated_per_cloud_then_refined():
    model = FeaturePropagation(EveryThirdPoint(), mlp=[9, 7], neighbours=3).eval()
    assert (model.output, model.in_channels, model.out_channels) == ("dense", 2, 7)
    points, feats = batch()
    with torch.no_grad():
        out = model(points, feats)
        coarse = model.backbone(points, feats)
    assert out.points is points and out.feats.shape == (121, 7)

    # Inverse-distance interpolation over the three nearest subsampled points of the
    # same cloud, written out; then the adapter's own MLP on [interpolated, input].
    lifted = np.zeros((121, 5))
    coords, known, values = (
        points.coords.numpy(),
        coarse.points.coords.numpy(),
        coarse.feats.numpy(),
    )
    cloud_of_known = coarse.points.batch_index.numpy()
    for row, cloud in enumerate(points.batch_index.numpy()):
        mine = np.flatnonzero(cloud_of_known == cloud)
        distance = np.sqrt(((known[mine] - coords[row]) ** 2).sum(-1))
        nearest = np.argsort(distance, kind="stable")[:3]
        weight = 1.0 / (distance[nearest] + 1e-8)
        lifted[row] = (weight[:, None] * values[mine[nearest]]).sum(0) / weight.sum()
    merged = torch.cat([torch.from_numpy(lifted).float(), feats], dim=1)
    with torch.no_grad():
        expected = model.fp.mlp.apply_rows(merged[:, None, :])[:, 0, :]
    assert torch.allclose(out.feats, expected, atol=1e-5)


def test_wraps_a_real_subsampled_backbone_from_a_config_node():
    model = BACKBONES.build({"type": "FeaturePropagation", "backbone": INNER, "mlp": [20, 12]})
    assert model.backbone.output == "subsampled" and model.out_channels == 12
    assert model.fp.mlp.layer0.conv.in_channels == 14 + 2  # interpolated + skipped input
    points, feats = batch()
    out = model.eval()(points, feats)
    assert out.feats.shape == (121, 12) and out.points is points

    without_skip = FeaturePropagation(INNER, mlp=[20], skip_input=False)
    assert without_skip.fp.mlp.layer0.conv.in_channels == 14


def test_a_dense_backbone_is_rejected_before_and_at_build_time():
    node = {"type": "FeaturePropagation", "backbone": {"type": "TBackbone"}}
    with pytest.raises(ConfigError) as error:
        Config.from_dict(
            {
                "model": {
                    "type": "TDetector",
                    "backbone": node,
                    "head": {"type": "THead", "num_angle": 12},
                }
            }
        )
    messages = [issue.message for issue in error.value.issues]
    assert any("TBackbone declares output='dense'. Use TBackbone directly." in m for m in messages)
    with pytest.raises(ValueError, match="wraps backbones that return subsampled features"):
        FeaturePropagation(TBackbone())


def test_in_channels_is_passed_down_and_must_agree():
    inner = {key: value for key, value in INNER.items() if key != "in_channels"}
    model = BACKBONES.build({"type": "FeaturePropagation", "backbone": inner}, in_channels=3)
    assert model.in_channels == model.backbone.in_channels == 3
    with pytest.raises(BuildError, match="PointNet2.in_channels is 2 in the config"):
        BACKBONES.build({"type": "FeaturePropagation", "backbone": INNER}, in_channels=3)
    with pytest.raises(ValueError, match="wrapped EveryThirdPoint has in_channels=2"):
        FeaturePropagation(EveryThirdPoint(), in_channels=4)


def test_misuse_is_rejected_with_an_explanation():
    points, feats = batch()
    stray = FeaturePropagation(EveryThirdPoint(detach=True)).eval()
    with pytest.raises(ContractError, match="to return features on a subsample of the point set"):
        stray(points, feats)
    greedy = FeaturePropagation(EveryThirdPoint(), neighbours=20).eval()
    with pytest.raises(ContractError, match="returned only 17 points for cloud 0"):
        greedy(points, feats)
    with pytest.raises(ValueError, match="neighbours must be positive"):
        FeaturePropagation(EveryThirdPoint(), neighbours=0)
    with pytest.raises(ValueError, match="mlp needs at least one layer"):
        FeaturePropagation(EveryThirdPoint(), mlp=[])
