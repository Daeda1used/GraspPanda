"""OACNNs on tiny inputs: published size, the block written out, the bounding-box rule, misuse.

Agreement with the official implementation is the subject of
``tests/integration/test_backbone_sparse_cnn_oracle.py``; the tests here need no fixture.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("spconv", reason="needs the spconv extra")

from grasppanda.models.backbones.oacnns import OACNNs, _run
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

#: Parameters of the official model built from ``semseg-oacnns-v1m1-0-base``.
PUBLISHED_PARAMETERS = 51_494_484

GRID = 0.25  # cell centres are multiples of 1/8: exact in float32


@pytest.fixture(autouse=True)
def single_thread():
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny(**overrides) -> OACNNs:
    """Two stages, with weights of order one: the official initialisation (a standard
    deviation of 0.02) would shrink every effect tested here below float32 resolution.

    The layers that produce the voting logits are scaled up further. Votes are
    ``exp(logit - largest) / (sum + 1e-6)``; only when logits spread over many units do
    the shift and the constant change anything, and they are part of what is tested.
    """
    config = {
        "type": "OACNNs",
        "in_channels": 3,
        "out_channels": 6,
        "grid_size": GRID,
        "embed_channels": 4,
        "enc_channels": [6, 8],
        "enc_depth": [2, 1],
        "dec_channels": [4, 6],
        "point_grid_size": [[2, 8], [4]],
    }
    model = BACKBONES.build({**config, **overrides})
    generator = torch.Generator().manual_seed(0)
    for name, tensor in model.state_dict().items():
        if tensor.ndim >= 2:
            tensor.normal_(0.0, (tensor.numel() // tensor.shape[0]) ** -0.5, generator=generator)
            if ".weight." in name:  # enc.<stage>.blocks.<block>.weight.<grid>.weight
                tensor.mul_(60.0)
        elif name.endswith(("running_var", "weight")):
            tensor.uniform_(0.6, 1.4, generator=generator)
        elif tensor.is_floating_point():
            tensor.uniform_(-0.3, 0.3, generator=generator)
    return model.eval()


def box(extent: tuple[int, int, int], seed: int, fill: float = 0.5) -> torch.Tensor:
    """Random cells of a box, with two opposite corners occupied so that it is this box."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.nonzero(torch.rand(extent, generator=generator) < fill)
    corners = torch.tensor([[0, 0, 0], [extent[0] - 1, extent[1] - 1, extent[2] - 1]])
    return torch.unique(torch.cat([cells, corners]), dim=0)


def notched(extent: tuple[int, int, int], seed: int) -> torch.Tensor:
    """A box with an odd last axis whose two lowest layers along the first axis are
    occupied in the top layer of the last axis only. The official model loses that layer
    at the first down-sampling, so its next level starts at cell 1, not at cell 0."""
    assert extent[2] % 2 == 1
    cells = box(extent, seed)
    cells = cells[(cells[:, 0] >= 2) | (cells[:, 2] == extent[2] - 1)]
    corners = torch.tensor([[0, 0, extent[2] - 1], [extent[0] - 1, extent[1] - 1, 0], [4, 0, 0]])
    return torch.unique(torch.cat([cells, corners]), dim=0)


def centres(cells: torch.Tensor) -> torch.Tensor:
    return (cells.float() + 0.5) * GRID


def feats_of(cells: torch.Tensor) -> torch.Tensor:
    """Features that belong to the cell, so that clouds sharing cells share inputs."""
    return torch.sin(cells.float() * torch.tensor([0.7, 1.3, 2.1]) + 0.5)


def run(model: OACNNs, cells: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(PointSet(centres(cells)), feats_of(cells)).feats


def test_default_architecture_is_the_published_model():
    torch.manual_seed(0)
    net = BACKBONES.build({"type": "OACNNs"})
    assert sum(p.numel() for p in net.parameters()) == PUBLISHED_PARAMETERS
    assert (net.in_channels, net.out_channels, net.grid_size) == (9, 20, 0.02)
    assert net.output == "dense" and net.total_stride == 16 and net.crop_odd_extent
    assert (OACNNs.status, OACNNs.extras) == ("tested", ("spconv",))
    assert "Peng et al., CVPR 2024" in OACNNs.source
    # The official initialisation: truncated normal for linear layers and 3³ convolutions.
    assert net.stem[3].weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert net.enc[2].blocks[0].fuse[0].weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert net.final.weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert not net.final.bias.any() and torch.equal(net.stem[1].weight, torch.ones(64))
    # The down-sampling convolutions keep the sparse library's: uniform within 1/sqrt(fan in).
    down, bound = net.enc[0].down[0].weight, (64 * 8) ** -0.5
    assert down.abs().max().item() <= bound
    assert down.std().item() == pytest.approx(bound / 3**0.5, rel=0.05)


def test_state_dict_follows_the_official_layout():
    state = {key: tuple(value.shape) for key, value in tiny().state_dict().items()}
    assert state["stem.0.weight"] == (4, 3, 3, 3, 3)
    assert state["stem.7.running_var"] == (4,)
    assert state["enc.0.down.0.weight"] == (6, 2, 2, 2, 4)
    assert state["enc.0.blocks.1.proj.2.0.weight"] == (6, 6)  # two grids and the voxel itself
    assert state["enc.0.blocks.0.l_w.1.1.running_mean"] == (6,)
    assert state["enc.0.blocks.0.weight.1.weight"] == (6, 6)
    assert state["enc.0.blocks.0.adaptive.weight"] == (2, 6)
    assert state["enc.1.blocks.0.adaptive.weight"] == (1, 8)
    assert state["enc.0.blocks.0.fuse.0.weight"] == (6, 12)
    assert state["enc.0.blocks.0.voxel_block.3.weight"] == (6, 3, 3, 3, 6)
    assert state["dec.1.up.0.weight"] == (6, 2, 2, 2, 8)
    assert state["dec.0.up.0.weight"] == (4, 2, 2, 2, 6)
    assert state["dec.0.fuse.0.weight"] == (4, 8) and state["dec.0.fuse.3.bias"] == (4,)
    assert state["final.weight"] == (6, 1, 1, 1, 4) and state["final.bias"] == (6,)
    biased = {key.rsplit(".", 1)[0] for key in state if key.endswith(".bias")}
    convolutions = {key.rsplit(".", 1)[0] for key, shape in state.items() if len(shape) == 5}
    assert biased & convolutions == {"final"}


def test_points_of_a_voxel_share_a_feature_and_the_grid_starts_at_the_cloud():
    net = tiny()
    cells = box((8, 8, 8), seed=1)
    coords, feats = centres(cells), feats_of(cells)
    twin = torch.cat([coords, coords[:1] + 0.1])  # a second point in the first cell
    with torch.no_grad():
        base = net(PointSet(coords), feats).feats
        doubled = net(PointSet(twin), torch.cat([feats, feats[:1] + 5.0])).feats
        moved = net(PointSet(coords + 3 * GRID), feats).feats
        shifted = net(PointSet(coords + GRID / 2), feats).feats
    assert base.shape == (len(cells), 6)
    # The voxel is represented by its first point; the second one only receives.
    assert torch.equal(doubled[-1], doubled[0]) and torch.allclose(doubled[:-1], base, atol=1e-6)
    # Cells are counted from the lowest occupied one, so where the cloud sits is irrelevant.
    assert torch.allclose(moved, base, atol=1e-6)
    assert torch.allclose(shifted, base, atol=1e-6)


def test_block_equals_the_aggregation_written_out_cube_by_cube():
    """The first block of the network against plain loops over pooling cubes."""
    net = tiny()
    block, sizes = net.enc[0].blocks[0], net.enc[0].grids
    seen = {}
    block.register_forward_hook(lambda module, args, out: seen.update(args=args, out=out))
    first, second = notched((8, 7, 9), seed=2), box((5, 8, 6), seed=3)
    points = PointSet.cat([PointSet(centres(first)), PointSet(centres(second))])
    with torch.no_grad():
        net(points, torch.cat([feats_of(first), feats_of(second)]))
        x, _, batch, _ = seen["args"]
        feat, cells = x.features, x.indices[:, 1:].long()
        # The notch worked: the first cloud starts at cell 1 of this level.
        assert cells[batch == 0].amin(dim=0).tolist() == [1, 0, 0]
        cubes_of = []  # per pooling grid: the rows of every cube
        for size in sizes:
            cubes: dict[tuple[int, ...], list[int]] = {}
            for row in range(len(feat)):
                lowest = cells[batch == batch[row]].amin(dim=0)
                key = (int(batch[row]), *((cells[row] - lowest) // size).tolist())
                cubes.setdefault(key, []).append(row)
            cubes_of.append(list(cubes.values()))
        assert any(len(rows) > 3 for rows in cubes_of[0]) and len(cubes_of[1]) == 2

        gathered = []
        for index, cubes in enumerate(cubes_of):
            logit = block.l_w[index](feat)
            centred = torch.empty_like(logit)
            for rows in cubes:
                centred[rows] = logit[rows] - logit[rows].mean(dim=0)
            logit = block.weight[index](centred)
            projected = block.proj[index](feat)
            pooled = torch.empty_like(projected)
            for rows in cubes:
                top = logit[batch == batch[rows[0]]].max()  # of the cloud, all channels
                vote = torch.exp(logit[rows] - top)
                vote = vote / (vote.sum(dim=0) + 1e-6)
                pooled[rows] = (projected[rows] * vote).sum(dim=0)
            gathered.append(pooled)
        mix = torch.softmax(block.adaptive(feat), dim=1)
        context = sum(mix[:, index, None] * pooled for index, pooled in enumerate(gathered))
        fused = block.fuse(torch.cat([block.proj[-1](feat), context], dim=1)) + feat
        expected = torch.relu(_run(block.voxel_block, x.replace_feature(fused)).features + fused)
    assert expected.abs().max() > 0.5
    assert torch.allclose(seen["out"].features, expected, atol=1e-5)


def test_bounding_box_rule_loses_exactly_the_last_cell_of_an_odd_axis():
    """A box of 4 x 8 x 8 cells halves twice without remainder. One more voxel at cell 8
    of the first axis makes that axis nine cells long: officially its last cell is lost."""
    cloud = box((4, 8, 8), seed=4)
    extra = torch.cat([cloud, torch.tensor([[8, 3, 3]])])
    official, every_voxel = tiny(), tiny(crop_odd_extent=False)
    # Without an odd axis the rule has nothing to do.
    assert torch.allclose(run(official, cloud), run(every_voxel, cloud), atol=1e-6)
    # The lost voxel is five cells away from the rest, out of reach of the stem (three).
    # With the rule nothing else connects it to the cloud; without it, the pooling grids do.
    assert torch.allclose(run(official, extra)[:-1], run(official, cloud), atol=1e-6)
    assert (run(every_voxel, extra)[:-1] - run(every_voxel, cloud)).abs().max() > 1e-2
    assert torch.isfinite(run(official, extra)[-1]).all()


def test_a_cloud_thinner_than_the_total_stride_needs_every_voxel_kept():
    thin = box((8, 8, 3), seed=5)
    with pytest.raises(ContractError, match=r"cloud 0 .* no voxel left after 2 of 2 .*at least 4"):
        run(tiny(), thin)
    assert torch.isfinite(run(tiny(crop_odd_extent=False), thin)).all()


@pytest.mark.parametrize("crop_odd_extent", [True, False])
def test_clouds_with_different_boxes_do_not_interact(crop_odd_extent):
    net = tiny(crop_odd_extent=crop_odd_extent)
    clouds = [box((9, 8, 11), seed=6), box((5, 6, 4), seed=7), box((12, 5, 7), seed=8)]
    points = PointSet.cat([PointSet(centres(cells)) for cells in clouds])
    with torch.no_grad():
        together = net(points, torch.cat([feats_of(cells) for cells in clouds])).feats
    alone = torch.cat([run(net, cells) for cells in clouds])
    assert together.abs().max() > 0.5
    assert torch.allclose(together, alone, atol=2e-5)


def test_misuse_is_rejected_with_an_explanation():
    net = tiny()
    cells = box((8, 8, 8), seed=9)
    points = PointSet(centres(cells))
    with pytest.raises(ContractError, match=r"expects features of shape \(\d+, 3\)"):
        net(points, torch.ones(len(cells), 4))
    with pytest.raises(ContractError, match="received no features"):
        net(points, None)
    with pytest.raises(ContractError, match="empty cloud"):
        net(PointSet(torch.zeros(0, 3)), torch.ones(0, 3))
    with pytest.raises(ValueError, match="one entry per stage"):
        OACNNs(enc_channels=(8, 16, 32))
    with pytest.raises(ValueError, match="at least one pooling grid"):
        tiny(point_grid_size=[[2], []])
    with pytest.raises(ValueError, match="grid_size must be positive"):
        OACNNs(grid_size=0.0)
