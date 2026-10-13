"""The interchangeable groupers: contract, a backward pass and the likely misuse of each.

``LocalGroup`` and ``DepthLocalGroup`` (shape, encoder and pooling as options), the RoI
pooling groupers adapted from OpenPCDet, the PyTorch Geometric bridge, the two depth-axis
adapters, ``MultiRadiusGroup``, ``DepthCylinderProjectionGroup`` and the box query.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from grasppanda.models.groupers.local import crop_members, masked_pool
from grasppanda.ops import approach_to_rotation, box_query
from grasppanda.registry import GROUPERS
from grasppanda.structures import ContractError, PointFeatures, PointSet

CHANNELS, WIDTH = 5, 12
CROP = {"radius": 0.3, "hmin": -0.1, "hmax": 0.25}


def seeds(batch: int = 2, per_frame: int = 30, channels: int = CHANNELS, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    points = PointSet.from_dense(torch.rand(batch, per_frame, 3, generator=generator))
    feats = torch.randn(batch * per_frame, channels, generator=generator)
    rotation = approach_to_rotation(
        torch.randn(batch * per_frame, 3, generator=generator),
        torch.rand(batch * per_frame, generator=generator) * math.pi,
    )
    return PointFeatures(feats, points), points.coords.clone(), rotation


def scene(batch: int = 2, per_frame: int = 30, points: int = 200, channels: int = 0):
    """A scene cloud without features and ``per_frame`` seeds taken from it."""
    generator = torch.Generator().manual_seed(1)
    coords = torch.rand(batch, points, 3, generator=generator)
    cloud = PointSet.from_dense(coords)
    feats = torch.randn(batch * points, channels, generator=generator)
    centres = coords[:, :per_frame].reshape(-1, 3).clone()
    rotation = approach_to_rotation(
        torch.randn(batch * per_frame, 3, generator=generator),
        torch.rand(batch * per_frame, generator=generator) * math.pi,
    )
    seed_feats = torch.randn(batch * per_frame, CHANNELS, generator=generator)
    return PointFeatures(feats, cloud), centres, rotation, seed_feats


def build(node: dict, channels: int = CHANNELS):
    torch.manual_seed(0)
    return GROUPERS.build(node, in_channels=channels)


def learns(stage, out: torch.Tensor) -> None:
    """A backward pass reaches every parameter of the stage."""
    out.square().mean().backward()
    missing = [name for name, p in stage.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in stage.parameters())


# ----------------------------------------------------------------------------- queries


def test_box_query_matches_a_written_out_box():
    support, _, rotation = seeds(batch=1, per_frame=60)
    coords = support.points.dense()
    lower, upper = (-0.1, -0.3, -0.2), (0.25, 0.3, 0.2)
    index, hits = box_query(coords, coords, rotation[None], lower, upper, 6)
    for row in range(60):
        local = (coords[0] - coords[0, row]) @ rotation[row]
        inside = ((local > torch.tensor(lower)) & (local < torch.tensor(upper))).all(-1)
        members = torch.nonzero(inside)[:, 0][:6]
        assert hits[0, row] == len(members)
        assert index[0, row, : len(members)].tolist() == members.tolist()
        assert (index[0, row, len(members) :] == members[0]).all()  # padded with the first
    with pytest.raises(ValueError, match="rotation must be"):
        box_query(coords, coords, rotation[None, :5], lower, upper, 6)


@pytest.mark.parametrize("shape", ["ball", "knn", "box", "cylinder"])
def test_crop_members_marks_real_slots_and_cyclic_padding_repeats_in_turn(shape):
    support, _, rotation = seeds(batch=1, per_frame=40)
    coords, frame = support.points.dense(), rotation[None]
    index, mask = crop_members(shape, coords, coords, frame, 0.12, -0.05, 0.1, 8)
    assert index.shape == mask.shape == (1, 40, 8) and mask[..., 0].all()
    assert (index[~mask] == index[..., :1].expand_as(index)[~mask]).all()
    cyclic, everything = crop_members(shape, coords, coords, frame, 0.12, -0.05, 0.1, 8, "cyclic")
    assert everything.all()
    for row in range(40):
        real = int(mask[0, row].sum())
        assert cyclic[0, row].tolist() == [int(index[0, row, k % real]) for k in range(8)]
    if shape == "knn":
        assert mask.all() and (index[0, :, 0] == torch.arange(40)).all()  # itself first


def test_masked_pool_ignores_padding_and_empty_rows_are_zero():
    hidden = torch.tensor([[[1.0, -5.0], [3.0, -1.0], [100.0, 100.0]]]).repeat(2, 1, 1)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    assert masked_pool(hidden, mask, "max", None).tolist() == [[3.0, -1.0], [0.0, 0.0]]
    assert masked_pool(hidden, mask, "mean", None).tolist() == [[2.0, -3.0], [0.0, 0.0]]
    uniform = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(uniform.weight)
    attentive = masked_pool(hidden, mask, "attentive", uniform)
    assert torch.allclose(attentive, torch.tensor([[2.0, -3.0], [0.0, 0.0]]), atol=1e-4)


# -------------------------------------------------------------------------- LocalGroup

LOCAL_VARIANTS = [
    {"shape": shape, "encoder": encoder, "pool": pool}
    for shape, encoder, pool in itertools.product(
        ["ball", "knn", "box", "cylinder"],
        ["mlp", "edge", "invres"],
        ["max", "mean", "attentive", "vector_attention"],
    )
]


@pytest.mark.parametrize("variant", LOCAL_VARIANTS, ids=lambda v: "-".join(v.values()))
def test_local_group_contract_for_every_combination_of_parts(variant):
    stage = build(
        {"type": "LocalGroup", **CROP, "nsample": 6, "mlp_channels": [8, WIDTH], **variant}
    )
    assert (stage.support, stage.out_axes, stage.out_channels) == (
        "seeds",
        ("seed", "channel"),
        WIDTH,
    )
    support, centres, rotation = seeds()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "channel") and out.feats.shape == (60, WIDTH)
    assert torch.isfinite(out.feats).all()
    learns(stage, out.feats)


def test_local_group_options_change_the_crop_and_frames_stay_apart():
    support, centres, rotation = seeds(batch=1, per_frame=20)
    node = {"type": "LocalGroup", **CROP, "nsample": 6, "mlp_channels": [8, WIDTH]}
    stage = build({**node, "geometry": "offset+distance"}).eval()
    geometry, feats, mask, own = stage.group(support, centres, rotation)
    assert geometry.shape == (20, 6, 4) and feats.shape == (20, 6, CHANNELS)
    assert torch.allclose(geometry[..., 3], geometry[..., :3].norm(dim=-1))
    assert (geometry[..., 3][mask] < 1.0).all()  # a ball of one radius, in units of the radius
    assert own is support.feats
    # Two copies of a frame in one batch give the same crops: nothing leaks across frames.
    twice = PointSet.from_dense(support.points.coords.reshape(1, 20, 3).repeat(2, 1, 1))
    both = stage(
        PointFeatures(support.feats.repeat(2, 1), twice), twice.coords, rotation.repeat(2, 1, 1)
    )
    assert torch.allclose(both.feats[:20], both.feats[20:], atol=1e-6)
    assert torch.allclose(both.feats[:20], stage(support, centres, rotation).feats, atol=1e-5)


def test_local_group_misuse_is_named():
    stage = build({"type": "LocalGroup", **CROP})
    support, centres, rotation = seeds(channels=CHANNELS + 1)
    with pytest.raises(ContractError, match=r"built for 5-channel support features, got 6"):
        stage(support, centres, rotation)
    support, centres, rotation = seeds()
    with pytest.raises(ContractError, match="one centre .* per support point with M = 60"):
        stage(support, centres[:10], rotation[:10])
    for wrong, fragment in (
        ({"shape": "sphere"}, "unknown shape 'sphere'"),
        ({"pool": "sum"}, "unknown pool 'sum'"),
        ({"encoder": "kpconv"}, "unknown encoder 'kpconv'"),
        ({"padding": "zeros"}, "unknown padding 'zeros'"),
        ({"radius": 0.0}, "needs radius > 0"),
    ):
        with pytest.raises(Exception, match=fragment):
            build({"type": "LocalGroup", **wrong})


# --------------------------------------------------------------------- DepthLocalGroup


@pytest.mark.parametrize("shape", ["ball", "knn", "box", "cylinder"])
@pytest.mark.parametrize("pool", ["max", "mean", "attentive"])
def test_depth_local_group_contract(shape, pool):
    node = {
        "type": "DepthLocalGroup",
        "shape": shape,
        "pool": pool,
        "radius": 0.3,
        "hmin": -0.1,
        "hmax": [0.05, 0.1, 0.25],
        "nsample": 12,
        "mlp_channels": [8, WIDTH],
        "seed_residual": "projected",
    }
    stage = build(node)
    assert (stage.support, stage.out_axes) == ("scene", ("seed", "depth", "channel"))
    assert stage.uses_seed_features and stage.num_depth == 3
    support, centres, rotation, seed_feats = scene()
    out = stage(support, centres, rotation, seed_feats)
    assert out.axes == ("seed", "depth", "channel") and out.feats.shape == (60, 3, WIDTH)
    learns(stage, out.feats)


def test_depth_local_group_slices_one_crop_by_depth():
    node = {
        "type": "DepthLocalGroup",
        "shape": "cylinder",
        "radius": 0.3,
        "hmin": -0.1,
        "hmax": [0.05, 0.25],
        "nsample": 200,
        "mlp_channels": [WIDTH],
    }
    stage = build(node).eval()
    assert not stage.uses_seed_features
    support, centres, rotation, seed_feats = scene(batch=1)
    out = stage(support, centres, rotation).feats
    # A deeper bin pools a superset of the members of a shallower one.
    assert (out[:, 1] >= out[:, 0] - 1e-6).all() and (out[:, 1] > out[:, 0] + 1e-6).any()
    with pytest.raises(ContractError, match="seed_residual: none and would ignore them"):
        stage(support, centres, rotation, seed_feats)
    with pytest.raises(ValueError, match=r"hmin < hmax\[0\] < hmax\[1\]"):
        build({**node, "hmax": [0.2, 0.1]})
    with pytest.raises(ValueError, match="pools with max, mean or attentive"):
        build({**node, "pool": "vector_attention"})
    projected = build({**node, "seed_residual": "projected"})
    with pytest.raises(ContractError, match=r"needs the \(30, 5\) features of the seeds"):
        projected(support, centres, rotation)


# ------------------------------------------------------------------------- RoI pooling


def test_roi_grid_group_lays_the_grid_in_the_crop_frame():
    node = {
        "type": "RoIGridGroup",
        **CROP,
        "grid_size": 2,
        "pool_radius": 0.2,
        "nsample": 4,
        "mlp_channels": [8],
        "fc_channels": [16, WIDTH],
    }
    stage = build(node)
    grid = stage.grid_points(torch.zeros(1))
    assert grid.shape == (8, 3)
    assert torch.allclose(grid.amin(0), torch.tensor([-0.0125, -0.15, -0.15]))
    assert torch.allclose(grid.amax(0), torch.tensor([0.1625, 0.15, 0.15]))
    support, centres, rotation = seeds()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "channel") and out.feats.shape == (60, WIDTH)
    learns(stage, out.feats)
    with pytest.raises(ValueError, match="grid_size >= 1"):
        build({**node, "grid_size": 0})
    with pytest.raises(ContractError, match="built for 5-channel support features"):
        stage(*seeds(channels=3))


def test_roi_aware_group_pools_every_cell_and_keeps_empty_cells_zero():
    node = {
        "type": "RoIAwareGroup",
        **CROP,
        "grid_size": 2,
        "nsample": 30,
        "cell_channels": 6,
        "fc_channels": [WIDTH],
    }
    stage = build(node)
    support, centres, rotation = seeds(batch=1, per_frame=30)
    cells = stage.pool(support, centres, rotation)
    assert cells.shape == (30, 8, CHANNELS + 3)
    coords, feats = support.points.coords, support.feats
    lower, upper = torch.tensor([-0.1, -0.3, -0.3]), torch.tensor([0.25, 0.3, 0.3])
    for row in (0, 7, 19):
        local = (coords - coords[row]) @ rotation[row]
        inside = ((local > lower) & (local < upper)).all(-1)
        cell = ((local - lower) / (upper - lower) * 2).long().clamp(0, 1)
        cell = (cell[:, 0] * 2 + cell[:, 1]) * 2 + cell[:, 2]
        for number in range(8):
            members = inside & (cell == number)
            if members.any():
                assert torch.allclose(cells[row, number, :CHANNELS], feats[members].amax(0))
                mean = (local[members] / 0.3).mean(0)
                assert torch.allclose(cells[row, number, CHANNELS:], mean, atol=1e-5)
            else:
                assert (cells[row, number] == 0).all()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "channel") and out.feats.shape == (30, WIDTH)
    learns(stage, out.feats)


def test_voxel_set_abstraction_group_fuses_scene_balls_with_the_seed():
    node = {
        "type": "VoxelSetAbstractionGroup",
        "radii": [0.2, 0.4],
        "nsample": [8, 8],
        "mlp_channels": [8],
        "out_channels": WIDTH,
    }
    stage = build(node)
    assert (stage.support, stage.out_axes, stage.uses_seed_features) == (
        "scene",
        ("seed", "channel"),
        True,
    )
    support, centres, rotation, seed_feats = scene()
    out = stage(support, centres, rotation, seed_feats)
    assert out.axes == ("seed", "channel") and out.feats.shape == (60, WIDTH)
    learns(stage, out.feats)
    with pytest.raises(ContractError, match=r"features of the\s+seeds, got None"):
        stage(support, centres, rotation)
    featured = build({**node, "scene_features": True})
    with pytest.raises(ContractError, match="under a sub-sampling encoder set scene_features"):
        featured(support, centres, rotation, seed_feats)
    dense, centres, rotation, seed_feats = scene(channels=CHANNELS)
    assert featured(dense, centres, rotation, seed_feats).feats.shape == (60, WIDTH)
    with pytest.raises(ValueError, match="one nsample >= 1 per radius"):
        build({**node, "nsample": [8]})


# ------------------------------------------------------------- PyTorch Geometric bridge


@pytest.mark.parametrize(
    "conv", ["PointNetConv", "PointTransformerConv", "EdgeConv", "GATv2Conv", "TransformerConv"]
)
def test_pyg_conv_group_contract(conv):
    pytest.importorskip("torch_geometric")
    node = {
        "type": "PyGConvGroup",
        "conv": conv,
        **CROP,
        "nsample": 6,
        "hidden_channels": 8,
        "out_channels": WIDTH,
    }
    stage = build(node)
    support, centres, rotation = seeds()
    feats, offsets, edges = stage.graph(support, centres, rotation)
    assert feats.shape[1] == CHANNELS and offsets.shape == (len(feats), 3)
    assert edges.shape == (2, len(feats)) and edges[1].unique().numel() == 60  # no empty crop
    assert (offsets[:, 1:].norm(dim=1) < 1).all()  # inside the cylinder, in units of its radius
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "channel") and out.feats.shape == (60, WIDTH)
    learns(stage, out.feats)


def test_pyg_conv_group_misuse_is_named():
    pytest.importorskip("torch_geometric")
    with pytest.raises(Exception, match="unknown conv 'GCNConv'"):
        build({"type": "PyGConvGroup", "conv": "GCNConv"})
    wide = build({"type": "PyGConvGroup", "conv": "GATv2Conv", "conv_args": {"heads": 2}})
    assert wide.conv.heads == 2
    with pytest.raises(ContractError, match="built for 5-channel support features"):
        wide(*seeds(channels=3))


def test_pyg_conv_group_without_the_library_says_how_to_install_it(monkeypatch):
    import importlib

    real = importlib.import_module

    def missing(name, *args, **kwargs):
        if name.startswith("torch_geometric"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(Exception, match=r'pip install "grasppanda\[torch_geometric\]"'):
        build({"type": "PyGConvGroup"})


# ------------------------------------------------------------------ depth-axis adapters


def test_depth_embed_group_adds_a_depth_axis():
    inner = {"type": "LocalGroup", **CROP, "nsample": 6, "mlp_channels": [WIDTH]}
    stage = build({"type": "DepthEmbedGroup", "grouper": inner, "num_depth": 3})
    assert (stage.support, stage.out_axes) == ("seeds", ("seed", "depth", "channel"))
    support, centres, rotation = seeds()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "depth", "channel") and out.feats.shape == (60, 3, WIDTH)
    assert not torch.allclose(out.feats[:, 0], out.feats[:, 1])  # the bins differ
    learns(stage, out.feats)
    scene_crop = {"type": "DepthCylinderGroup"}
    with pytest.raises(ContractError, match="DepthCylinderGroup declares support = 'scene'"):
        build({"type": "DepthEmbedGroup", "grouper": scene_crop})
    problems = list(GROUPERS.get("DepthEmbedGroup").check_config({"grouper": scene_crop}))
    assert len(problems) == 2 and "emits ('seed', 'depth', 'channel')" in problems[1]
    with pytest.raises(Exception, match="needs it as a config node"):
        build({"type": "DepthEmbedGroup", "grouper": "LocalGroup"})


@pytest.mark.parametrize("pool", ["max", "mean"])
def test_depth_pool_group_removes_the_depth_axis(pool):
    inner = {
        "type": "DepthLocalGroup",
        "radius": 0.3,
        "hmax": [0.05, 0.25],
        "nsample": 12,
        "mlp_channels": [WIDTH],
        "seed_residual": "projected",
    }
    stage = build({"type": "DepthPoolGroup", "grouper": inner, "pool": pool})
    assert (stage.support, stage.out_axes, stage.uses_seed_features) == (
        "scene",
        ("seed", "channel"),
        True,
    )
    support, centres, rotation, seed_feats = scene()
    out = stage(support, centres, rotation, seed_feats)
    assert out.axes == ("seed", "channel") and out.feats.shape == (60, WIDTH)
    learns(stage, out.feats)
    with pytest.raises(ContractError, match="LocalGroup declares support = 'seeds'"):
        build({"type": "DepthPoolGroup", "grouper": {"type": "LocalGroup"}})


# ------------------------------------------------------------- options of ported groupers


@pytest.mark.parametrize("fusion", ["linear", "max", "mean"])
def test_multi_radius_group_fuses_without_attention(fusion):
    branch = {"type": "LocalGroup", "shape": "cylinder", "nsample": 6, "mlp_channels": [WIDTH]}
    node = {
        "type": "MultiRadiusGroup",
        "branch": branch,
        "radius": 0.3,
        "radius_scales": [0.5, 1.0],
        "fusion": fusion,
    }
    stage = build(node)
    assert (stage.fusion is None) == (fusion != "linear")
    assert stage.radii == pytest.approx((0.15, 0.3)) and len(stage.branches) == 2
    support, centres, rotation = seeds()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "channel") and out.feats.shape == (60, WIDTH)
    learns(stage, out.feats)
    with pytest.raises(Exception, match="unknown fusion 'attention'"):
        build({**node, "fusion": "attention"})
    with pytest.raises(Exception, match="DepthCylinderGroup declares support = 'scene'"):
        build({**node, "branch": {"type": "DepthCylinderGroup"}})


def test_depth_cylinder_projection_group_takes_seed_features_of_another_width():
    node = {
        "radius": 0.3,
        "hmax": [0.05, 0.25],
        "nsample": 8,
        "mlp_channels": [WIDTH],
    }
    gated = GROUPERS.get("DepthCylinderGroup")
    assert gated.projected_variant == "DepthCylinderProjectionGroup"
    assert gated.required_seed_channels({**node, "seed_residual": "gated"}) == WIDTH
    assert gated.required_seed_channels({**node, "seed_residual": "none"}) is None
    with pytest.raises(Exception, match="Use DepthCylinderProjectionGroup"):
        build({"type": "DepthCylinderGroup", **node, "seed_residual": "gated"})
    projected = GROUPERS.get("DepthCylinderProjectionGroup")
    assert projected.required_seed_channels(node) is None and projected.projected_variant is None
    stage = build({"type": "DepthCylinderProjectionGroup", **node, "radius_scales": [0.5, 1.0]})
    assert stage.uses_seed_features and stage.in_channels == CHANNELS
    assert (stage.support, stage.out_axes) == ("scene", ("seed", "depth", "channel"))
    support, centres, rotation, seed_feats = scene()
    out = stage(support, centres, rotation, seed_feats)
    assert out.axes == ("seed", "depth", "channel") and out.feats.shape == (60, 2, WIDTH)
    learns(stage, out.feats)
    with pytest.raises(ContractError, match=r"needs the \(M, 5\) features of the seeds, got None"):
        stage(support, centres, rotation)


def test_the_width_of_an_encoder_is_learnt_without_its_released_weights(monkeypatch):
    """The seed-width rule builds an encoder to learn its width. A preset that starts from
    released weights must get the same verdict whether or not the file is on the machine."""
    import json
    from types import SimpleNamespace

    from grasppanda.models.detectors import seed_based

    def build(node):
        if node.get("pretrained"):
            raise FileNotFoundError("the released weights are not on this machine")
        return SimpleNamespace(out_channels=12)

    monkeypatch.setattr(seed_based.BACKBONES, "build", build)
    seed_based._built_width.cache_clear()
    try:
        node = json.dumps({"type": "Released", "pretrained": "some-file"})
        assert seed_based._built_width(node) == 12
        assert seed_based._built_width(json.dumps({"type": "Released"})) == 12
        wrapped = {"type": "Adapter", "backbone": {"type": "Released", "pretrained": "f"}}
        monkeypatch.setattr(
            seed_based.BACKBONES, "build", lambda node: build(node.get("backbone", node))
        )
        assert seed_based._built_width(json.dumps(wrapped)) == 12
    finally:
        seed_based._built_width.cache_clear()
