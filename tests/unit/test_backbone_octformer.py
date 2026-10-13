"""OctFormer through the optional ``ocnn`` library: contract, octree geometry, misuse.

Skipped without ``ocnn``. The encoder is a bridge (the octree, its convolutions and the
interpolation are the library's), so it is not part of ``test_backbone_contract``; the
checks here are the same ones except batch independence, which the reference does not
have: its attention groups are cut from the nodes of the whole batch."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ocnn", reason="needs the ocnn extra")

from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

TINY = {
    "in_channels": 2,
    "out_channels": 10,
    "voxel_size": 0.05,
    "octree_depth": 6,
    "channels": [8, 16],
    "num_blocks": [2, 2],
    "num_heads": [2, 4],
    "patch_size": 4,
    "dilation": 2,
    "drop_path": 0.0,
    "fpn_channel": 8,
}


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def batch() -> PointSet:
    generator = torch.Generator().manual_seed(0)
    return PointSet.cat(
        [
            PointSet(torch.rand(size, 3, generator=generator) + shift)
            for size, shift in ((90, 0.0), (131, 4.0))
        ]
    )


def test_features_on_every_point_and_gradients_everywhere():
    model = BACKBONES.build({"type": "OctFormer", **TINY})
    assert (model.output, model.status, model.extras) == ("dense", "bridge", ("ocnn",))
    points = batch()
    out = model(points, torch.rand(len(points), 2))
    assert out.points is points and out.feats.shape == (221, 10)
    assert torch.isfinite(out.feats).all()
    out.feats.pow(2).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_evaluation_is_deterministic_and_clouds_are_centred_on_their_own_box():
    model = BACKBONES.build({"type": "OctFormer", **TINY}).eval()
    points = batch()
    feats = torch.sin(points.coords[:, :2] * 3)
    with torch.no_grad():
        first, second = model(points, feats).feats, model(points, feats).feats
        moved = PointSet(points.coords + 2.5, points.batch_offsets)  # 50 leaves: same octree
        shifted = model(moved, feats).feats
    assert torch.equal(first, second)
    assert torch.allclose(first, shifted, atol=1e-4)


def test_defaults_are_the_scannet_model_and_convolutions_avoid_triton():
    model = BACKBONES.build({"type": "OctFormer"})
    assert (model.in_channels, model.out_channels, model.octree_depth) == (3, 168, 11)
    state = model.state_dict()
    assert state["backbone.patch_embed.convs.0.conv.weights"].shape == (27, 6, 24)  # 3 + local
    assert state["backbone.layers.2.blocks.17.attention.qkv.weight"].shape == (1152, 384)
    assert state["backbone.layers.0.blocks.1.attention.rpe.rpe_table"].shape == (3 * 83, 6)
    assert state["head.conv1x1.0.weight"].shape == (168, 384)
    methods = {m.method for m in model.modules() if hasattr(m, "method")}
    assert methods and "triton" not in methods


def test_misuse_is_refused_with_the_way_out():
    with pytest.raises(ValueError, match="it must be at least 5"):
        BACKBONES.build({"type": "OctFormer", **{**TINY, "octree_depth": 4}})
    with pytest.raises(ValueError, match="unknown conv_method 'fast'"):
        BACKBONES.build({"type": "OctFormer", **TINY, "conv_method": "fast"})
    model = BACKBONES.build({"type": "OctFormer", **TINY}).eval()
    wide = PointSet(torch.rand(50, 3) * 4.0)  # 4 m in a cube of 0.05 * 2**6 = 3.2 m
    with pytest.raises(ContractError, match=r"more than the 3.20 m cube .* Raise octree_depth"):
        model(wide, torch.rand(50, 2))
    with pytest.raises(ContractError, match="in_channels"):
        model(batch(), torch.rand(221, 3))
