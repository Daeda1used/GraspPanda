"""The backbone contract, checked the same way for every registered scene encoder.

A backbone is swappable only if it keeps the contract of
``grasppanda.models.Backbone``: it says where its features live, binds them to that
point set, treats every cloud of a batch on its own, and refuses misuse loudly. Each
case below is a small configuration of one registered backbone; the tests run on CPU.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from grasppanda.config import Config
from grasppanda.models import Backbone
from grasppanda.registry import BACKBONES, component_parameters
from grasppanda.structures import ContractError, PointSet

HAS_SPCONV = importlib.util.find_spec("spconv") is not None
needs_spconv = pytest.mark.skipif(not HAS_SPCONV, reason="needs the spconv extra")

#: Stated tolerance of "a batch gives the same per-cloud result as separate runs".
#: The arithmetic per cloud is the same; only the layout of the batched tensors
#: differs, which moves float32 sums by a few units in the last place.
BATCH_TOLERANCE = 2e-5


@dataclass
class Case:
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    #: Smallest cloud the configuration accepts, and one that is too small (0 = none).
    sizes: tuple[int, int] = (96, 131)
    too_few: int = 0
    #: Parameters that exist for checkpoint compatibility but take no part in forward.
    unused: tuple[str, ...] = ()
    marks: tuple[Any, ...] = ()


POINTNET2 = {
    "npoints": [48, 24, 12, 6],
    "radii": [0.2, 0.4, 0.6, 0.9],
    "nsamples": [8, 8, 4, 4],
    "sa_mlps": [[8, 8], [8, 16], [16, 16], [16, 16]],
}
PTV3 = {
    "grid_size": 0.05,
    "stride": [2, 2],
    "enc_depths": [1, 1, 2],
    "enc_channels": [12, 24, 48],
    "enc_num_head": [2, 4, 8],
    "enc_patch_size": [16, 16, 16],
}
PTV3_DECODER = {
    "dec_depths": [1, 1],
    "dec_channels": [12, 24],
    "dec_num_head": [2, 4],
    "dec_patch_size": [16, 16],
}

CASES = {
    "PointNet2": Case("PointNet2", {**POINTNET2, "fp_mlps": [[16, 16], [16, 20]]}, too_few=47),
    "PointNet2+feats": Case(
        "PointNet2", {**POINTNET2, "in_channels": 3, "fp_mlps": [[16]]}, too_few=47
    ),
    "PointNet2Dense": Case(
        "PointNet2Dense",
        {**POINTNET2, "in_channels": 2, "fp_mlps": [[16], [16], [16], [12, 12]]},
        too_few=47,
    ),
    "FeaturePropagation": Case(
        "FeaturePropagation",
        {
            "backbone": {"type": "PointNet2", **POINTNET2, "in_channels": 3, "fp_mlps": [[16]]},
            "mlp": [16, 10],
        },
        too_few=47,
    ),
    "FurthestPointSubsample": Case(
        "FurthestPointSubsample",
        {
            "backbone": {
                "type": "PointNet2Dense",
                **POINTNET2,
                "in_channels": 2,
                "fp_mlps": [[16], [16], [16], [12, 12]],
            },
            "npoint": 24,
        },
        too_few=23,
    ),
    "PTv3": Case("PTv3", {**PTV3, **PTV3_DECODER, "in_channels": 3}, marks=(needs_spconv,)),
    "PTv3+rpe": Case(
        "PTv3",
        {**PTV3, **PTV3_DECODER, "in_channels": 3, "enable_rpe": True, "pre_norm": False},
        marks=(needs_spconv,),
    ),
    "PTv3Sonata": Case(
        "PTv3Sonata",
        {**PTV3, "in_channels": 3, "concat_stages": 2, "layer_scale": 0.1},
        unused=("embedding.mask_token",),
        marks=(needs_spconv,),
    ),
    "PTv3Sonata+decoder": Case(
        "PTv3Sonata",
        {**PTV3, **PTV3_DECODER, "in_channels": 3, "enc_mode": False, "mask_token": False},
        marks=(needs_spconv,),
    ),
    "PTv3Utonia": Case(
        "PTv3Utonia",
        {**PTV3, "in_channels": 3, "concat_stages": 3},
        unused=("embedding.mask_token",),
        marks=(needs_spconv,),
    ),
    # The published layout in small: convolution in the fine stages, attention in the
    # coarsest one, a decoder that only unpools.
    "LitePT": Case(
        "LitePT",
        {
            "in_channels": 3,
            "grid_size": 0.05,
            "stride": [2, 2],
            "enc_depths": [1, 1, 2],
            "enc_channels": [12, 24, 48],
            "enc_num_head": [2, 4, 8],
            "enc_patch_size": [16, 16, 16],
            "enc_conv": [True, True, False],
            "enc_attn": [False, False, True],
            "enc_rope_freq": [100.0, 100.0, 100.0],
            "dec_depths": [0, 0],
            "dec_channels": [12, 24],
            "dec_num_head": [2, 4],
            "dec_patch_size": [16, 16],
            "dec_conv": [False, False],
            "dec_attn": [False, False],
            "dec_rope_freq": [100.0, 100.0],
        },
        marks=(needs_spconv,),
    ),
    # Everything else a stage can be: convolution and attention together, attention from
    # the input level on, decoder blocks of both kinds, post-norm.
    "LitePT+decoder": Case(
        "LitePT",
        {
            "in_channels": 3,
            "grid_size": 0.05,
            "stride": [2, 2],
            "enc_depths": [1, 1, 2],
            "enc_channels": [12, 24, 48],
            "enc_num_head": [2, 4, 8],
            "enc_patch_size": [16, 16, 16],
            "enc_conv": [False, True, False],
            "enc_attn": [True, True, True],
            "enc_rope_freq": [100.0, 50.0, 10.0],
            "dec_depths": [1, 2],
            "dec_channels": [12, 24],
            "dec_num_head": [2, 4],
            "dec_patch_size": [16, 16],
            "dec_conv": [True, False],
            "dec_attn": [False, True],
            "dec_rope_freq": [100.0, 50.0],
            "pre_norm": False,
        },
        marks=(needs_spconv,),
    ),
    "PointNeXt-S": Case(
        "PointNeXt",
        {
            "in_channels": 3,
            "width": 8,
            "strides": [1, 2, 2, 2],
            "blocks": [1, 1, 1, 1],
            "radius": 0.2,
            "nsample": 8,
        },
        too_few=23,
    ),
    "PointNeXt-B": Case(
        "PointNeXt",
        {
            "in_channels": 3,
            "width": 8,
            "strides": [1, 2, 2],
            "blocks": [2, 2, 3],
            "radius": 0.2,
            "nsample": 8,
            "sa_layers": 1,
            "sa_use_res": False,
            "expansion": 2,
        },
        too_few=11,
    ),
    "OACNNs": Case(
        "OACNNs",
        {
            "in_channels": 3,
            "out_channels": 12,
            "grid_size": 0.05,
            "embed_channels": 8,
            "enc_channels": [8, 12],
            "enc_depth": [1, 2],
            "dec_channels": [8, 12],
            "point_grid_size": [[2, 4], [2, 3]],
        },
        marks=(needs_spconv,),
    ),
    "OACNNs+every-voxel": Case(
        "OACNNs",
        {
            "in_channels": 3,
            "out_channels": 12,
            "grid_size": 0.05,
            "embed_channels": 8,
            "enc_channels": [8, 12, 16],
            "enc_depth": [1, 1, 1],
            "dec_channels": [8, 12, 12],
            "point_grid_size": [[3, 5], [2, 3], [2, 4]],
            "crop_odd_extent": False,
        },
        marks=(needs_spconv,),
    ),
    "SpUNet": Case(
        "SpUNet",
        {
            "in_channels": 3,
            "out_channels": 12,
            "grid_size": 0.05,
            "base_channels": 6,
            "channels": [8, 12, 10, 8],
            "layers": [1, 2, 1, 1],
        },
        marks=(needs_spconv,),
    ),
    "SpUNet+no-final": Case(
        "SpUNet",
        {
            "in_channels": 3,
            "out_channels": None,
            "grid_size": 0.05,
            "base_channels": 6,
            "channels": [8, 12, 16, 12, 10, 8],
            "layers": [1, 1, 2, 1, 1, 2],
        },
        marks=(needs_spconv,),
    ),
}


def cases() -> list[Any]:
    return [pytest.param(case, id=name, marks=case.marks) for name, case in CASES.items()]


def build(case: Case) -> Backbone:
    torch.manual_seed(0)
    model = BACKBONES.build({"type": case.type, **case.config})
    # Give batch norm non-trivial running statistics so that evaluation mode is a real test.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.uniform_(-0.2, 0.2)
            module.running_var.uniform_(0.5, 1.5)
    return model


def cloud(count: int, seed: int) -> torch.Tensor:
    return torch.rand(count, 3, generator=torch.Generator().manual_seed(seed))


def features(model: Backbone, coords: torch.Tensor) -> torch.Tensor | None:
    """Input features that depend on the point only, so a batch can be split and compared."""
    if model.in_channels == 0:
        return None
    columns = [torch.sin(coords[:, index % 3] * (index + 2)) for index in range(model.in_channels)]
    return torch.stack(columns, dim=1)


def batch_of(clouds: list[torch.Tensor]) -> PointSet:
    return PointSet.cat([PointSet(coords) for coords in clouds])


@pytest.mark.parametrize("case", cases())
def test_backbone_declares_a_checkable_contract(case):
    cls = BACKBONES.get(case.type)
    assert issubclass(cls, Backbone)
    assert cls.output in ("dense", "subsampled")
    assert cls.status in ("tested", "weights-verified")
    assert cls.__doc__ and inspect.getmodule(cls).__doc__
    params, var_kwargs = component_parameters(cls)
    assert not var_kwargs
    assert all(p.annotation is not inspect.Parameter.empty for p in params.values())
    # The static validator accepts the configuration without constructing anything.
    Config.from_dict(
        {
            "model": {
                "type": "TDetector",
                "backbone": {"type": case.type, **case.config},
                "head": {"type": "THead", "num_angle": 12},
            }
        }
    )
    model = build(case)
    assert isinstance(model.in_channels, int) and isinstance(model.out_channels, int)


@pytest.mark.parametrize("case", cases())
def test_features_are_bound_to_the_declared_point_set(case):
    model = build(case).eval()
    clouds = [cloud(size, seed) for seed, size in enumerate(case.sizes)]
    points = batch_of(clouds)
    with torch.no_grad():
        out = model(points, features(model, points.coords))
    assert out.feats.shape[1] == model.out_channels
    assert torch.isfinite(out.feats).all()
    if model.output == "dense":
        assert out.points is points
        assert out.feats.shape[0] == len(points)
        out.expect(points, user="test")
        return
    child = out.points
    assert child.parent_id == points.id and child.id != points.id
    assert torch.equal(child.coords, points.coords[child.parent_index])
    assert child.batch_size == points.batch_size
    assert out.feats.shape[0] == len(child) < len(points)
    with pytest.raises(ContractError, match="add an explicit propagation step"):
        out.expect(points, user="a stage that needs dense features")
    # Every sampled row stays inside its own cloud.
    assert torch.equal(points.batch_index[child.parent_index], child.batch_index)


@pytest.mark.parametrize("case", cases())
def test_gradients_reach_every_parameter(case):
    model = build(case).train()
    points = batch_of([cloud(size, seed + 10) for seed, size in enumerate(case.sizes)])
    out = model(points, features(model, points.coords))
    out.feats.pow(2).mean().backward()
    for name, parameter in model.named_parameters():
        if name in case.unused:
            assert parameter.grad is None, f"{name} is expected to be unused"
            continue
        assert parameter.grad is not None, f"no gradient reaches {name}"
        assert torch.isfinite(parameter.grad).all(), name
    untouched = [
        name
        for name, parameter in model.named_parameters()
        if name not in case.unused and not parameter.grad.any()
    ]
    assert not untouched, f"all-zero gradients: {untouched}"


@pytest.mark.parametrize("case", cases())
def test_evaluation_is_deterministic(case):
    model = build(case).eval()
    points = batch_of([cloud(size, seed + 20) for seed, size in enumerate(case.sizes)])
    feats = features(model, points.coords)
    with torch.no_grad():
        first, second = model(points, feats), model(points, feats)
    assert torch.equal(first.feats, second.feats)
    if model.output == "subsampled":
        assert torch.equal(first.points.parent_index, second.points.parent_index)


@pytest.mark.parametrize("case", cases())
@pytest.mark.parametrize("sizes", ["equal", "ragged"])
def test_a_batch_gives_the_same_result_as_its_clouds_alone(case, sizes):
    """Batch independence in evaluation mode (training-mode batch norm mixes clouds by design)."""
    model = build(case).eval()
    counts = (case.sizes[1], case.sizes[1]) if sizes == "equal" else case.sizes
    clouds = [cloud(size, seed + 30) for seed, size in enumerate(counts)]
    points = batch_of(clouds)
    with torch.no_grad():
        together = model(points, features(model, points.coords))
        alone = [model(PointSet(coords), features(model, coords)) for coords in clouds]
    expected = torch.cat([out.feats for out in alone])
    assert together.feats.shape == expected.shape
    assert torch.allclose(together.feats, expected, atol=BATCH_TOLERANCE, rtol=0)
    if model.output == "subsampled":
        starts = points.batch_offsets[:-1].tolist()
        index = torch.cat([out.points.parent_index + start for out, start in zip(alone, starts)])
        assert torch.equal(together.points.parent_index, index)


@pytest.mark.parametrize("case", cases())
def test_wrong_channel_count_is_rejected(case):
    model = build(case).eval()
    points = batch_of([cloud(case.sizes[1], 40)])
    wrong = torch.zeros(len(points), model.in_channels + 1)
    with pytest.raises(ContractError, match="in_channels"):
        model(points, wrong)
    if model.in_channels:
        with pytest.raises(ContractError, match="received no features"):
            model(points, None)
        with pytest.raises(ContractError, match="expects features of shape"):
            model(points, torch.zeros(len(points) - 1, model.in_channels))


@pytest.mark.parametrize("case", [c for c in cases() if c.values[0].too_few])
def test_too_few_points_for_the_configured_sampling_is_rejected(case):
    model = build(case).eval()
    points = batch_of([cloud(case.sizes[1], 50), cloud(case.too_few, 51)])
    with pytest.raises(ContractError, match=rf"cloud 1 of the batch has (only )?{case.too_few}"):
        model(points, features(model, points.coords))
