"""``FurthestPointSubsample``: the explicit way from dense features to seeds.

The general backbone contract is checked in ``test_backbone_contract.py``. Here: what
the adapter promises on top of it. It keeps the points a PointNet++ encoder would have
ended on, hands over the wrapped backbone's features unchanged, and is the way out the
config checker names when a dense encoder meets a proposal for sub-sampled features.
"""

from __future__ import annotations

import pytest
import torch

from grasppanda.config import Config, ConfigError
from grasppanda.registry import BACKBONES, BuildError
from grasppanda.structures import PointSet

POINTNET2 = {
    "npoints": [48, 24, 12, 6],
    "radii": [0.2, 0.4, 0.6, 0.9],
    "nsamples": [8, 8, 4, 4],
    "sa_mlps": [[8, 8], [8, 16], [16, 16], [16, 16]],
}
DENSE = {"type": "PointNet2Dense", **POINTNET2, "fp_mlps": [[16], [16], [16], [12]]}


def clouds(*sizes: int) -> PointSet:
    generator = torch.Generator().manual_seed(3)
    return PointSet.cat([PointSet(torch.rand(size, 3, generator=generator)) for size in sizes])


@pytest.mark.parametrize("tie_break", ["lowest_index", "pointnet2_cuda"])
@pytest.mark.parametrize("skip", [True, False])
def test_it_keeps_the_points_pointnet2_would_end_on(tie_break, skip):
    """Furthest point samples are nested, so sampling 24 points directly gives the set a
    PointNet++ encoder reaches through its 48-point level, in the same order."""
    rules = {"fps_skip_near_origin": skip, "fps_tie_break": tie_break}
    torch.manual_seed(0)
    encoder = BACKBONES.build(
        {"type": "PointNet2", **POINTNET2, "fp_mlps": [[16], [16]], **rules}
    ).eval()
    adapter = BACKBONES.build(
        {
            "type": "FurthestPointSubsample",
            "backbone": {**DENSE, "fps_skip_near_origin": skip},
            "npoint": 24,
            **rules,
        }
    ).eval()
    points = clouds(140, 97)
    with torch.no_grad():
        reference, kept = encoder(points, None), adapter(points, None)
    assert torch.equal(kept.points.parent_index, reference.points.parent_index)
    assert torch.equal(kept.points.coords, reference.points.coords)
    assert kept.points.batch_offsets.tolist() == [0, 24, 48]


def test_the_features_are_those_of_the_wrapped_backbone_at_the_kept_points():
    torch.manual_seed(0)
    adapter = BACKBONES.build({"type": "FurthestPointSubsample", "backbone": DENSE, "npoint": 20})
    adapter.eval()
    points = clouds(120, 80)
    with torch.no_grad():
        kept = adapter(points, None)
        dense = adapter.backbone(points, None)
    assert adapter.output == "subsampled" and adapter.backbone.output == "dense"
    assert adapter.out_channels == adapter.backbone.out_channels == 12
    assert adapter.in_channels == adapter.backbone.in_channels == 0
    assert torch.equal(kept.feats, dense.feats[kept.points.parent_index])
    assert not list(adapter.parameters(recurse=False))  # nothing learned by the adapter


def test_a_subsampling_backbone_is_refused_before_and_at_build_time():
    node = {
        "type": "FurthestPointSubsample",
        "backbone": {"type": "PointNet2", **POINTNET2, "fp_mlps": [[16]]},
    }
    model = {"type": "TDetector", "backbone": node, "head": {"type": "THead", "num_angle": 12}}
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": model})
    message = error.value.issues[0].message
    assert "wraps backbones that return dense features" in message
    assert (
        "PointNet2 declares output='subsampled'" in message and "Use PointNet2 directly" in message
    )
    with pytest.raises(ValueError, match="wraps backbones that return dense features"):
        BACKBONES.build(node)


def test_injected_channels_reach_the_wrapped_backbone_and_must_agree():
    adapter = BACKBONES.build(
        {"type": "FurthestPointSubsample", "backbone": DENSE, "npoint": 8}, in_channels=3
    )
    assert adapter.in_channels == adapter.backbone.in_channels == 3
    with pytest.raises(BuildError, match="in_channels"):
        BACKBONES.build(
            {
                "type": "FurthestPointSubsample",
                "backbone": {**DENSE, "in_channels": 2},
                "in_channels": 3,
            }
        )


def test_the_two_adapters_are_inverse_in_kind_and_never_stack_silently():
    """Dense → subsampled → dense is allowed (it is what PointNet2 + propagation does);
    wrapping the wrong kind is an error naming the direct use."""
    round_trip = BACKBONES.build(
        {
            "type": "FeaturePropagation",
            "backbone": {"type": "FurthestPointSubsample", "backbone": DENSE, "npoint": 16},
            "mlp": [8],
        }
    ).eval()
    points = clouds(90)
    with torch.no_grad():
        out = round_trip(points, None)
    assert round_trip.output == "dense" and out.points is points and out.feats.shape == (90, 8)
    with pytest.raises(ValueError, match="Use FurthestPointSubsample directly"):
        BACKBONES.build(
            {
                "type": "FurthestPointSubsample",
                "backbone": {"type": "FurthestPointSubsample", "backbone": DENSE},
            }
        )
