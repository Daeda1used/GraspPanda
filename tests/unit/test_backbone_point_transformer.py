"""Point Transformer V1 and V2: layout of the Pointcept models, pooling, misuse.

The shared contract (binding, gradients, batch independence, channel checks) is in
``test_backbone_contract``. No weights were released for either model, so there is no
checkpoint test."""

from __future__ import annotations

import pytest
import torch

from grasppanda.models.backbones.point_transformer import (
    PointTransformerV1,
    PointTransformerV2,
    _neighbours,
)
from grasppanda.structures import ContractError, PointSet


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_v1_defaults_are_seg26_with_the_pointcept_tensor_shapes():
    model = PointTransformerV1()
    assert (model.output, model.in_channels, model.out_channels) == ("dense", 6, 32)
    state = model.state_dict()
    assert state["down.0.linear.weight"].shape == (32, 6)  # stride 1: a linear embedding
    assert state["down.1.linear.weight"].shape == (64, 32 + 3)  # grouped offsets + features
    assert state["enc.4.0.transformer.linear_p.3.weight"].shape == (512, 3)
    assert state["enc.4.0.transformer.linear_w.2.weight"].shape == (64, 512)  # share_planes 8
    assert state["up.4.linear1.0.weight"].shape == (512, 1024)  # coarsest: feature + cloud mean
    assert state["up.0.linear2.0.weight"].shape == (32, 64)
    # PointTransformer-Seg26 of Pointcept @ 1342eda without its `cls` head has this many
    # parameters, in tensors of the same shapes (counted on that class, 2026-10-13).
    assert sum(p.numel() for p in model.parameters()) == 4_852_596
    seg50 = PointTransformerV1(blocks=[1, 2, 3, 5, 2])
    assert [len(blocks) for blocks in seg50.enc] == [1, 2, 3, 5, 2]


def test_v1_refuses_sizes_that_do_not_fit():
    with pytest.raises(ValueError, match="multiple of share_planes=8"):
        PointTransformerV1(planes=[32, 60], blocks=[1, 1], strides=[1, 4], nsamples=[8, 16])
    with pytest.raises(ValueError, match="need one entry per stage"):
        PointTransformerV1(planes=[32, 64])
    with pytest.raises(ValueError, match="first stride must be 1"):
        PointTransformerV1(strides=[2, 4, 4, 4, 4])
    model = PointTransformerV1(
        in_channels=1, planes=[8, 8], blocks=[1, 1], strides=[1, 4], nsamples=[4, 4]
    )
    with pytest.raises(ContractError, match="every cloud needs at least 4 points"):
        model(PointSet(torch.rand(3, 3)), torch.ones(3, 1))


def test_v2_defaults_are_the_scannet_model_and_grid_sizes_are_metres():
    model = PointTransformerV2()
    assert (model.output, model.in_channels, model.out_channels) == ("dense", 6, 48)
    state = model.state_dict()
    assert state["patch_embed.proj.0.weight"].shape == (48, 6)
    assert state["enc_stages.0.down.fc.weight"].shape == (96, 48)
    assert state["enc_stages.3.blocks.blocks.1.attn.weight_encoding.0.weight"].shape == (64, 512)
    assert state["dec_stages.0.up.proj_skip.0.weight"].shape == (48, 48)
    assert not any("linear_p_multiplier" in key for key in state)  # PT-v2m2: bias only
    # PT-v2m2 of Pointcept @ 1342eda with `num_classes=0` and these grid sizes: the same
    # keys, shapes and count (compared with that class, 2026-10-13).
    assert sum(p.numel() for p in model.parameters()) == 11_320_376
    assert [stage.down.grid_size for stage in model.enc_stages] == [0.06, 0.15, 0.375, 0.9375]
    assert PointTransformerV2(out_channels=512).out_channels == 512

    # Two points per cell of 0.5: the pooling halves the cloud, cloud by cloud.
    tiny = PointTransformerV2(
        in_channels=1,
        patch_embed_channels=6,
        patch_embed_groups=2,
        patch_embed_neighbours=2,
        enc_depths=[1],
        enc_channels=[8],
        enc_groups=[2],
        enc_neighbours=[2],
        dec_depths=[1],
        dec_channels=[6],
        dec_groups=[2],
        dec_neighbours=[2],
        grid_sizes=[0.5],
    ).eval()
    corners = torch.tensor([[x, y, 0.0] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5)])
    coords = torch.cat([corners + 0.1, corners + 0.3])  # 6 cells, 2 points each
    seen = []
    tiny.enc_stages[0].down.register_forward_hook(lambda m, a, out: seen.append(out[2]))
    points = PointSet.cat([PointSet(coords), PointSet(coords[:8] + 5.0)])
    with torch.no_grad():
        out = tiny(points, torch.ones(len(points), 1))
    assert seen == [[6, 6]]  # the second cloud is shifted: its grid starts at its own corner
    assert out.feats.shape == (20, 6)


def test_v2_refuses_sizes_that_do_not_fit():
    with pytest.raises(ValueError, match="every per-stage option needs 4 entries"):
        PointTransformerV2(grid_sizes=[0.06, 0.12])
    with pytest.raises(ValueError, match="cannot be split into 7 attention groups"):
        PointTransformerV2(patch_embed_groups=7)


def test_a_cloud_smaller_than_the_neighbourhood_is_padded_and_masked():
    coords = torch.rand(7, 3)
    _, index, valid = _neighbours(coords, [3, 4], coords, [3, 4], 4)
    assert index.shape == (7, 4) and valid is not None
    assert valid[:3].sum(dim=1).tolist() == [3, 3, 3] and valid[3:].all()
    assert (index[:3] < 3).all() and (index[3:] >= 3).all()  # never across clouds
    assert torch.equal(index[:3, 3], index[:3, 0])  # the padding repeats the nearest
    assert _neighbours(coords, [3, 4], coords, [3, 4], 3)[2] is None
