"""Point-based scene encoders taken from existing code: the shared backbone contract
on tiny clouds, plus what is particular to each (layouts, sizes, misuse messages).

The contract checks are the functions of ``test_backbone_contract``, run on the cases
below: features bound to the declared point set, gradients to every parameter,
deterministic evaluation, batch independence, loud refusal of wrong inputs.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

from . import test_backbone_contract as contract
from .test_backbone_contract import Case

HAS_PYG = importlib.util.find_spec("torch_geometric") is not None
needs_pyg = pytest.mark.skipif(not HAS_PYG, reason="needs the torch_geometric extra")

_MSG = {
    "npoints": [48, 24, 12, 6],
    "radii": [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.9, 1.0]],
    "nsamples": [[6, 8], [6, 8], [4, 4], [4, 4]],
    "sa_mlps": [[[8], [8, 8]], [[8], [16]], [[16], [8, 8]], [[16], [16]]],
}

CASES: dict[str, Case] = {
    "PointNet2MSG": Case("PointNet2MSG", {**_MSG, "fp_mlps": [[16, 16], [16, 20]]}, too_few=47),
    "PointNet2MSG+feats": Case(
        "PointNet2MSG", {**_MSG, "in_channels": 3, "fp_mlps": [[16]]}, too_few=47
    ),
    "PointNet2MSGDense": Case(
        "PointNet2MSGDense",
        {**_MSG, "in_channels": 2, "fp_mlps": [[16], [16], [16], [12, 12]]},
        too_few=47,
    ),
    "PointMetaBase": Case(
        "PointMetaBase",
        {
            "in_channels": 3,
            "width": 8,
            "strides": [1, 2, 2],
            "blocks": [1, 2, 3],
            "radius": 0.2,
            "nsample": 8,
            "expansion": 2,
        },
        too_few=11,
    ),
    "PointMetaBase+plain-stage": Case(
        "PointMetaBase",
        {
            "in_channels": 2,
            "width": 8,
            "strides": [1, 2, 2, 2],
            "blocks": [1, 1, 2, 1],
            "radius": 0.2,
            "nsample": 8,
        },
        too_few=23,
    ),
    "PointVector": Case(
        "PointVector",
        {
            "in_channels": 3,
            "width": 8,
            "strides": [1, 2, 2],
            "blocks": [1, 2, 2],
            "radius": 0.2,
            "nsample": 8,
            "block_nsample": 4,
        },
        too_few=11,
    ),
    "DGCNN": Case(
        "DGCNN",
        {"in_channels": 3, "k": 6, "width": 8, "emb_dims": 16, "out_channels": 12},
        too_few=6,
    ),
    "DGCNN+static-local": Case(
        "DGCNN",
        {"in_channels": 0, "k": 5, "width": 8, "emb_dims": 0, "out_channels": 10, "dynamic": False},
        too_few=5,
    ),
    "DGCNN+feats-only": Case(
        "DGCNN",
        {"in_channels": 4, "k": 5, "width": 8, "emb_dims": 8, "coords_as_features": False},
    ),
    "ASSANet": Case(
        "ASSANet",
        {
            "in_channels": 3,
            "width": 6,
            "strides": [2, 2],
            "radii": [[0.2, 0.3], [0.4, 0.5]],
            "nsamples": [[6, 8], [6, 6]],
            "mlps": [[[6, 6, 9], [9, 9, 12]], [[12, 12, 12], [12, 12, 18]]],
            "fp_mlps": [[12, 12], [16, 16]],
        },
        too_few=11,
    ),
    "ASSANet+max-no-stem": Case(
        "ASSANet",
        {
            "in_channels": 2,
            "width": 6,
            "stem_layers": 0,
            "strides": [2, 2],
            "radii": [[0.3], [0.5]],
            "nsamples": [[6], [6]],
            "mlps": [[[6, 6]], [[9, 9, 9]]],
            "fp_mlps": [[10], [12]],
            "reduction": "max",
        },
        too_few=11,
    ),
    "DeLA": Case(
        "DeLA",
        {
            "in_channels": 3,
            "grid_size": [0.1, 0.3, 0.5],
            "k": [6, 6, 4],
            "depths": [2, 2, 1],
            "dims": [8, 12, 16],
            "nbr_dims": [8, 8],
            "head_dim": 12,
            "zero_init_residual": False,
        },
    ),
    "DeLA+coords-only": Case(
        "DeLA",
        {
            "in_channels": 0,
            "grid_size": [0.1, 0.4],
            "k": [5, 40],
            "depths": [1, 2],
            "dims": [8, 8],
            "nbr_dims": [8, 8],
            "head_dim": 10,
            "zero_init_residual": False,
        },
    ),
    "PointMLP": Case(
        "PointMLP",
        {
            "in_channels": 3,
            "embed_dim": 8,
            "dim_expansion": [2, 1],
            "pre_blocks": [1, 1],
            "pos_blocks": [1, 2],
            "k_neighbors": [6, 4],
            "reducers": [2, 2],
            "de_dims": [16, 12],
            "de_blocks": [1, 2],
            "gmp_dim": 8,
            "out_channels": 12,
        },
        too_few=11,
    ),
    "PointMLP+no-context": Case(
        "PointMLP",
        {
            "embed_dim": 8,
            "dim_expansion": [2],
            "pre_blocks": [1],
            "pos_blocks": [1],
            "k_neighbors": [5],
            "reducers": [4],
            "de_dims": [8],
            "de_blocks": [1],
            "gmp_dim": 0,
            "out_channels": 10,
        },
        too_few=11,
    ),
    "RandLANet": Case(
        "RandLANet",
        {
            "in_channels": 3,
            "num_neighbors": 4,
            "sub_sampling_ratio": [2, 2],
            "dim_features": 4,
            "dim_output": [4, 8],
            "out_channels": 12,
        },
        too_few=7,
    ),
    "RandLANet+coords-only": Case(
        "RandLANet",
        {"num_neighbors": 5, "sub_sampling_ratio": [2, 2, 2], "dim_output": [4, 4, 8]},
        too_few=19,
    ),
    **{
        f"PyG-{conv}": Case(
            "PyGPointEncoder",
            {"in_channels": 3, "conv": conv, "channels": channels, "out_channels": 12, "k": 6},
            too_few=5,
            marks=(needs_pyg,),
        )
        for conv, channels in [
            ("pointnet", [8, 8, 16]),
            ("edge", [8, 16]),
            ("point_transformer", [8, 16]),
            ("pointgnn", [8, 8]),
        ]
    },
    "PyG-coords-only": Case(
        "PyGPointEncoder",
        {
            "conv": "pointnet",
            "channels": [8],
            "out_channels": 10,
            "k": 4,
            "conv_args": {"aggr": "mean"},
        },
        marks=(needs_pyg,),
    ),
}


def cases() -> list:
    return [pytest.param(case, id=name, marks=case.marks) for name, case in CASES.items()]


@pytest.mark.parametrize("case", cases())
def test_declares_a_checkable_contract(case):
    contract.test_backbone_declares_a_checkable_contract(case)
    cls = BACKBONES.get(case.type)
    assert "source" in vars(cls) and cls.source


@pytest.mark.parametrize("case", cases())
def test_features_are_bound_to_the_declared_point_set(case):
    contract.test_features_are_bound_to_the_declared_point_set(case)


@pytest.mark.parametrize("case", cases())
def test_gradients_reach_every_parameter(case):
    contract.test_gradients_reach_every_parameter(case)


@pytest.mark.parametrize("case", cases())
def test_evaluation_is_deterministic(case):
    contract.test_evaluation_is_deterministic(case)


@pytest.mark.parametrize("case", cases())
@pytest.mark.parametrize("sizes", ["equal", "ragged"])
def test_a_batch_gives_the_same_result_as_its_clouds_alone(case, sizes):
    contract.test_a_batch_gives_the_same_result_as_its_clouds_alone(case, sizes)


@pytest.mark.parametrize("case", cases())
def test_wrong_channel_count_is_rejected(case):
    contract.test_wrong_channel_count_is_rejected(case)


@pytest.mark.parametrize("case", [c for c in cases() if c.values[0].too_few])
def test_too_few_points_is_rejected(case):
    contract.test_too_few_points_for_the_configured_sampling_is_rejected(case)


# ------------------------------------------------------------------ PointNet++ MSG


def test_msg_defaults_are_a_drop_in_for_the_baseline_encoder():
    model = BACKBONES.build({"type": "PointNet2MSG"})
    assert model.output == "subsampled" and model.out_channels == 256
    assert model.npoints == (2048, 1024, 512, 256)
    # Two scales per level: the widths of both are concatenated.
    assert [layer.out_channels for layer in model.SA_modules] == [192, 384, 512, 512]
    state = model.state_dict()
    assert state["SA_modules.0.mlps.1.layer0.conv.weight"].shape == (64, 3, 1, 1)
    assert state["SA_modules.1.mlps.0.layer0.conv.weight"].shape == (64, 192 + 3, 1, 1)
    dense = BACKBONES.build({"type": "PointNet2MSGDense"})
    assert dense.output == "dense" and dense.out_channels == 128
    assert sum(p.numel() for p in dense.parameters()) > sum(p.numel() for p in model.parameters())


def test_msg_misuse_is_named():
    with pytest.raises(ValueError, match="takes a list per level"):
        BACKBONES.build({"type": "PointNet2MSG", **{**_MSG, "radii": [[0.2], [0.4], [0.6], [0.9]]}})
    with pytest.raises(ValueError, match="Use PointNet2MSGDense"):
        BACKBONES.build({"type": "PointNet2MSG", **_MSG, "fp_mlps": [[8]] * 4})
    with pytest.raises(ValueError, match="Use PointNet2MSG for features on a subsample"):
        BACKBONES.build({"type": "PointNet2MSGDense", **_MSG, "fp_mlps": [[8]]})
    model = BACKBONES.build({"type": "PointNet2MSG", **_MSG})
    with pytest.raises(ContractError, match="has only 20"):
        model(PointSet(torch.rand(20, 3)), None)


# ------------------------------------------------- PointMetaBase and PointVector


def _with_seg_head(model) -> float:
    """Millions of parameters with the official 13-class segmentation head."""
    width = model.out_channels
    head = width * width + 2 * width + width * 13 + 13
    return round((sum(p.numel() for p in model.parameters()) + head) / 1e6, 3)


@pytest.mark.parametrize(
    ("config", "published_millions"),
    [
        # "Params.(M)" in the headers of cfgs/s3dis/pointmetabase-{l,xl,xxl}.yaml of the
        # official repository (the XL header has its two columns swapped).
        ({}, 2.738),
        ({"blocks": [1, 4, 7, 4, 4], "width": 64}, 15.284),
        ({"blocks": [1, 5, 9, 5, 5], "width": 64}, 19.672),
    ],
)
def test_pointmetabase_has_the_published_size(config, published_millions):
    model = BACKBONES.build({"type": "PointMetaBase", **config})
    assert _with_seg_head(model) == published_millions
    assert model.output == "dense" and model.in_channels == 4


def test_pointvector_defaults_are_the_scene_network():
    model = BACKBONES.build({"type": "PointVector"})
    assert model.out_channels == 32 and model.block_nsample == 8
    assert _with_seg_head(model) == 4.215  # counted here; no config header states it
    state = model.state_dict()
    # Each channel's 3-vector goes back to a scalar through one group of the convolution.
    assert state["encoder.encoder.1.1.tf_zx.0.weight"].shape == (64, 3, 1)
    assert model.block_radii == [0.1, 0.2, 0.4, 0.8, 1.6]


@pytest.mark.parametrize("name", ["PointMetaBase", "PointVector"])
def test_family_misuse_is_named(name):
    with pytest.raises(ValueError, match="the stem stage has exactly one block"):
        BACKBONES.build({"type": name, "blocks": [2, 1, 1], "strides": [1, 2, 2]})
    with pytest.raises(ValueError, match="the first stride must be 1"):
        BACKBONES.build({"type": name, "blocks": [1, 1, 1], "strides": [2, 2, 2]})
    with pytest.raises(ValueError, match="at least one input feature channel"):
        BACKBONES.build({"type": name, "in_channels": 0})


# --------------------------------------------------------------------------- DGCNN


def test_dgcnn_defaults_are_the_reference_segmentation_network():
    model = BACKBONES.build({"type": "DGCNN", "in_channels": 6})
    state = model.state_dict()
    # 9 input channels as in the S3DIS model of the reference: edges carry twice that.
    assert state["conv1.0.weight"].shape == (64, 18, 1, 1)
    assert state["conv6.0.weight"].shape == (1024, 192, 1)
    assert state["conv7.0.weight"].shape == (512, 1216, 1)
    assert model.out_channels == 256 and model.output == "dense"


def test_dgcnn_graph_is_dynamic_and_stays_inside_a_cloud():
    from grasppanda.models.backbones.dgcnn import _knn_rows

    values = torch.rand(50, 4)
    rows = _knn_rows(values, [20, 30], 5, chunk=7)
    assert rows.shape == (50, 5)
    assert (rows[:20] < 20).all() and (rows[20:] >= 20).all()
    assert torch.equal(rows[:, 0], torch.arange(50))  # a point is its own nearest neighbour
    assert torch.equal(rows, _knn_rows(values, [20, 30], 5, chunk=50))

    points = PointSet(torch.rand(80, 3, generator=torch.Generator().manual_seed(0)))
    torch.manual_seed(0)
    dynamic = BACKBONES.build({"type": "DGCNN", "k": 6, "width": 8, "emb_dims": 0}).eval()
    static = BACKBONES.build(
        {"type": "DGCNN", "k": 6, "width": 8, "emb_dims": 0, "dynamic": False}
    ).eval()
    static.load_state_dict(dynamic.state_dict())
    with torch.no_grad():
        assert not torch.allclose(dynamic(points).feats, static(points).feats)


def test_dgcnn_misuse_is_named():
    with pytest.raises(ValueError, match="needs coords_as_features=true"):
        BACKBONES.build({"type": "DGCNN", "coords_as_features": False})


# -------------------------------------------------------------------------- ASSANet


def test_assanet_has_the_published_size_and_separable_layout():
    model = BACKBONES.build({"type": "ASSANet"})
    # "Params.(M)" in the header of cfgs/s3dis/assanet.yaml of the PointNeXt repository.
    assert _with_seg_head(model) == 2.425
    assert model.out_channels == 64 and model.output == "dense"
    first = model.SA_modules[0].local_aggregations[0]
    # 16 -> 16 -> ceil(16 / 3) = 6 before grouping; three offset coordinates per channel after.
    assert first.pre[1][0].weight.shape == (6, 16, 1)
    assert first.post[0][0].weight.shape == (32, 18, 1)
    assert first.skip_layer.weight.shape == (32, 6, 1)
    # The second scale of a level reads the output of the first.
    assert model.SA_modules[0].local_aggregations[1].pre[0][0].weight.shape == (32, 32, 1)
    with pytest.raises(ValueError, match="one entry per level"):
        BACKBONES.build({"type": "ASSANet", "strides": [4, 4]})
    with pytest.raises(ValueError, match="widths of a propagation level must be equal"):
        BACKBONES.build(
            {"type": "ASSANet", "fp_mlps": [[64, 32], [128, 128], [256, 256], [512, 512]]}
        )


# ----------------------------------------------------------------------------- DeLA


def test_dela_pyramid_is_a_grid_and_keeps_clouds_apart():
    from grasppanda.models.backbones.dela import _grid_subsample, _knn_padded

    coords = torch.tensor(
        [
            [0.01, 0.01, 0.0],
            [0.02, 0.03, 0.0],
            [0.26, 0.0, 0.0],
            [0.27, 0.01, 0.0],
            [0.01, 0.02, 0.0],
        ]
    )
    keep, counts = _grid_subsample(coords, [4, 1], 0.1)
    # One point per occupied cell, the first in input order; the second cloud keeps its own.
    assert keep.tolist() == [0, 2, 4] and counts == [2, 1]
    rows = _knn_padded(coords, [4, 1], 3)
    assert rows.shape == (5, 3) and (rows[:4] < 4).all()
    assert rows[4].tolist() == [4, 4, 4]  # a cloud smaller than k repeats its own point


def test_dela_defaults_are_the_official_s3dis_model():
    model = BACKBONES.build({"type": "DeLA"})
    state = model.state_dict()
    assert state["stages.0.nbr_embed.0.weight"].shape == (16, 7)  # offsets and 4 features
    assert state["stages.1.nbr_embed.0.weight"].shape == (8, 3)
    assert state["stages.1.nbr_proj.weight"].shape == (128, 32)
    assert state["stages.3.postproj.1.weight"].shape == (256, 512)
    assert len(model.stages[2].blk.lfps) == 8 and len(model.stages[2].blk.mlps) == 4
    assert model.out_channels == 256 and model.coord_scale == pytest.approx(40.0)
    # The residual branches start at the constants of the official code.
    assert float(model.stages[0].blk.lfps[0].bn.weight[0]) == 0.0
    assert float(model.stages[0].nbr_bn.weight[0]) == pytest.approx(0.8)
    model.eval()
    points = PointSet(torch.rand(300, 3))
    with torch.no_grad():
        assert torch.isfinite(model(points, torch.rand(300, 4)).feats).all()
    with pytest.raises(ValueError, match="grow from stage to stage"):
        BACKBONES.build({"type": "DeLA", "grid_size": [0.04, 0.04, 0.16, 0.32]})


# ------------------------------------------------------------------------- PointMLP


def test_pointmlp_defaults_are_the_part_segmentation_layout():
    model = BACKBONES.build({"type": "PointMLP", "in_channels": 3})
    state = model.state_dict()
    assert state["embedding.conv.weight"].shape == (64, 6)  # coordinates and three features
    # A group holds the normalised neighbour (features and offsets) and the anchor feature.
    assert state["stages.0.transfer.conv.weight"].shape == (128, 2 * 64 + 3)
    assert state["stages.0.affine_alpha"].shape == (64 + 3,)
    assert state["stages.3.pos.1.net2.conv.weight"].shape == (1024, 1024)
    assert state["decode_list.0.fuse.conv.weight"].shape == (512, 1024 + 512)
    assert state["decode_list.3.fuse.conv.weight"].shape == (128, 128 + 64)
    assert state["head.conv.weight"].shape == (128, 128 + 64)
    assert model.output == "dense"


def test_pointmlp_scales_a_group_by_the_spread_of_its_own_cloud():
    """The geometric affine makes the features of a cloud independent of its size in space."""
    torch.manual_seed(0)
    config = CASES["PointMLP+no-context"].config
    model = BACKBONES.build({"type": "PointMLP", **config}).eval()
    coords = torch.rand(64, 3)
    with torch.no_grad():
        # Zero the coordinate input of the embedding: what is left enters through groups.
        model.embedding.conv.weight.zero_()
        small = model(PointSet(coords)).feats
        large = model(PointSet(coords * 8.0)).feats
    assert torch.allclose(small, large, atol=1e-4)
    with pytest.raises(ValueError, match="one entry per stage"):
        BACKBONES.build({"type": "PointMLP", "reducers": [4, 4]})


# ----------------------------------------------------------------------- RandLA-Net


def test_randlanet_defaults_are_the_published_layout():
    model = BACKBONES.build({"type": "RandLANet", "in_channels": 3})
    state = model.state_dict()
    assert state["fc0.conv.weight"].shape == (8, 6)  # coordinates, then the features
    assert state["encoder.0.lse1.conv.weight"].shape == (8, 10)  # distance, offset, 2 positions
    assert state["encoder.3.mlp2.conv.weight"].shape == (512, 256)
    assert state["decoder.0.conv.weight"].shape == (256, 256 + 512)
    assert state["decoder.3.conv.weight"].shape == (32, 32 + 32)
    assert model.out_channels == 32 and model.output == "dense"


def test_randlanet_sampling_is_random_in_training_and_seeded_in_evaluation():
    model = BACKBONES.build(
        {
            "type": "RandLANet",
            "num_neighbors": 4,
            "sub_sampling_ratio": [2, 2],
            "dim_output": [4, 8],
        }
    )
    device = torch.device("cpu")
    model.eval()
    assert torch.equal(model._order([30, 40], device), model._order([30, 40], device))
    alone = model._order([40], device)
    assert torch.equal(model._order([30, 40], device)[30:] - 30, alone)  # a function of the size
    assert sorted(model._order([30, 40], device).tolist()) == list(range(70))
    model.train()
    assert not torch.equal(model._order([200], device), model._order([200], device))
    with pytest.raises(ValueError, match="one entry per level"):
        BACKBONES.build({"type": "RandLANet", "sub_sampling_ratio": [4, 4]})


# ------------------------------------------------------- torch_geometric bridge


@pytest.mark.skipif(HAS_PYG, reason="checks the message of the missing extra")
def test_pyg_bridge_names_the_extra_when_the_library_is_absent():
    from grasppanda.utils.imports import MissingDependencyError

    with pytest.raises(MissingDependencyError, match=r"grasppanda\[torch_geometric\]"):
        BACKBONES.build({"type": "PyGPointEncoder"})


def test_pyg_bridge_declares_its_extra():
    cls = BACKBONES.get("PyGPointEncoder")
    assert cls.status == "bridge" and cls.extras == ("torch_geometric",)


def test_pyg_bridge_uses_the_library_layers_and_names_misuse():
    nn = pytest.importorskip("torch_geometric.nn")
    wanted = {
        "pointnet": nn.PointNetConv,
        "edge": nn.EdgeConv,
        "point_transformer": nn.PointTransformerConv,
        "pointgnn": nn.PointGNNConv,
    }
    for conv, layer in wanted.items():
        model = BACKBONES.build({"type": "PyGPointEncoder", "conv": conv, "channels": [8, 8]})
        assert all(isinstance(module, layer) for module in model.convs), conv
        assert model.out_channels == 256
    with pytest.raises(ValueError, match="all channels must be equal for conv=pointgnn"):
        BACKBONES.build({"type": "PyGPointEncoder", "conv": "pointgnn", "channels": [8, 16]})
    with pytest.raises(ValueError, match="need coords_as_features=true"):
        BACKBONES.build({"type": "PyGPointEncoder", "coords_as_features": False})
