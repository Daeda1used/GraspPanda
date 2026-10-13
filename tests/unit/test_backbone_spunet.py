"""SpUNet on tiny inputs: published size, state-dict layout, 1×1 kernels, voxel semantics.

Agreement with the official implementation is the subject of
``tests/integration/test_backbone_sparse_cnn_oracle.py``; the tests here need no fixture.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("spconv", reason="needs the spconv extra")

from grasppanda.models.backbones.spunet import SpUNet, convert_pointcept_state
from grasppanda.ops import sparse
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

#: Parameters of the official model built from ``semseg-spunet-v1m1-0-base``, and of the
#: same model without its final layer (``num_classes=0``).
PUBLISHED_PARAMETERS = 39_157_844
WITHOUT_FINAL = 39_155_904

GRID = 0.25  # cell centres are multiples of 1/8: exact in float32


@pytest.fixture(autouse=True)
def single_thread():
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny(**overrides) -> SpUNet:
    """Two stages, with weights of order one instead of the official 0.02."""
    config = {
        "type": "SpUNet",
        "in_channels": 3,
        "out_channels": 5,
        "grid_size": GRID,
        "base_channels": 4,
        "channels": [6, 8, 7, 5],
        "layers": [1, 2, 2, 1],
    }
    model = BACKBONES.build({**config, **overrides})
    generator = torch.Generator().manual_seed(0)
    for name, tensor in model.state_dict().items():
        if tensor.ndim >= 2:
            tensor.normal_(0.0, (tensor.numel() // tensor.shape[0]) ** -0.5, generator=generator)
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


def centres(cells: torch.Tensor) -> torch.Tensor:
    return (cells.float() + 0.5) * GRID


def feats_of(cells: torch.Tensor) -> torch.Tensor:
    return torch.sin(cells.float() * torch.tensor([0.7, 1.3, 2.1]) + 0.5)


def run(model: SpUNet, cells: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(PointSet(centres(cells)), feats_of(cells)).feats


def test_default_architecture_is_the_published_model():
    torch.manual_seed(0)
    net = BACKBONES.build({"type": "SpUNet"})
    assert sum(p.numel() for p in net.parameters()) == PUBLISHED_PARAMETERS
    assert (net.in_channels, net.out_channels, net.grid_size) == (6, 20, 0.02)
    assert net.output == "dense" and net.total_stride == 16
    assert (SpUNet.status, SpUNet.extras) == ("tested", ("spconv",))
    assert "MinkUNet-34C" in SpUNet.source and "Pointcept" in SpUNet.source
    assert [len(blocks) for blocks in net.enc] == [2, 3, 4, 6]
    assert [len(blocks) for blocks in net.dec] == [2, 2, 2, 2]
    # The official initialisation: truncated normal for 3³/5³ convolutions and 1×1 layers.
    assert net.conv_input[0].weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert net.enc[3].block5.conv2.weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert net.dec[0].block0.proj[0].weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert net.final.weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert not net.final.bias.any() and torch.equal(net.down[0][1].weight, torch.ones(32))
    # The down-sampling convolutions keep the sparse library's: uniform within 1/sqrt(fan in).
    down, bound = net.down[1][0].weight, (32 * 8) ** -0.5
    assert down.abs().max().item() <= bound
    assert down.std().item() == pytest.approx(bound / 3**0.5, rel=0.05)


def test_without_final_layer_the_last_decoder_stage_is_the_output():
    net = BACKBONES.build({"type": "SpUNet", "out_channels": None})
    assert net.final is None and net.out_channels == 96
    assert sum(p.numel() for p in net.parameters()) == WITHOUT_FINAL
    small = tiny(out_channels=None)
    cells = box((6, 5, 7), seed=1)
    assert small.out_channels == 5 and run(small, cells).shape == (len(cells), 5)


def test_state_dict_follows_the_official_layout():
    state = {key: tuple(value.shape) for key, value in tiny().state_dict().items()}
    assert state["conv_input.0.weight"] == (4, 5, 5, 5, 3)
    assert state["conv_input.1.running_var"] == (4,)
    assert state["down.0.0.weight"] == (6, 2, 2, 2, 4)  # the stride-2 convolution widens
    assert state["down.1.0.weight"] == (8, 2, 2, 2, 6)
    assert state["enc.1.block1.conv2.weight"] == (8, 3, 3, 3, 8)
    assert state["enc.1.block1.bn2.running_mean"] == (8,)
    assert not any(key.startswith("enc.") and ".proj." in key for key in state)
    assert state["up.1.0.weight"] == (7, 2, 2, 2, 8)  # from the bottleneck
    assert state["up.0.0.weight"] == (5, 2, 2, 2, 7)
    assert state["dec.1.block0.conv1.weight"] == (7, 3, 3, 3, 7 + 6)  # joined with level 1
    assert state["dec.1.block0.proj.0.weight"] == (7, 7 + 6)
    assert state["dec.1.block1.conv1.weight"] == (7, 3, 3, 3, 7)
    assert "dec.1.block1.proj.0.weight" not in state
    assert state["dec.0.block0.proj.0.weight"] == (5, 5 + 4)  # joined with the stem
    assert state["dec.0.block0.proj.1.running_var"] == (5,)
    assert state["final.weight"] == (5, 5) and state["final.bias"] == (5,)
    biased = {key.rsplit(".", 1)[0] for key in state if key.endswith(".bias")}
    kernels = {key.rsplit(".", 1)[0] for key, shape in state.items() if len(shape) in (2, 5)}
    assert biased & kernels == {"final"}


def test_converted_1x1_kernel_computes_what_the_sparse_library_computes():
    """spconv multiplies by its ``(out, 1, 1, 1, in)`` parameter viewed as ``(in, out)``."""
    spconv = pytest.importorskip("spconv.pytorch")
    generator = torch.Generator().manual_seed(0)
    official = spconv.SubMConv3d(5, 7, kernel_size=1, bias=True)
    with torch.no_grad():
        official.weight.copy_(torch.randn(official.weight.shape, generator=generator))
        official.bias.copy_(torch.randn(7, generator=generator))
    kernel = torch.randn(4, 3, 3, 3, 5, generator=generator)
    state = {"final.weight": official.weight.detach(), "final.bias": official.bias.detach()}
    converted = convert_pointcept_state(
        {**state, "conv.weight": kernel, "bn.weight": torch.ones(4)}
    )
    assert list(converted) == ["final.weight", "final.bias", "conv.weight", "bn.weight"]
    assert converted["final.weight"].shape == (7, 5)
    assert converted["conv.weight"] is kernel  # every other kernel already has the layout

    feats = torch.randn(40, 5, generator=generator)
    voxels = torch.cat([torch.zeros(40, 1, dtype=torch.long), box((4, 4, 4), 1, 1.0)[:40]], dim=1)
    x, _ = sparse.make_tensor(feats, voxels, batch_size=1)
    linear = sparse.Linear(5, 7)
    linear.load_state_dict({"weight": converted["final.weight"], "bias": converted["final.bias"]})
    with torch.no_grad():
        expected = official(x).features
        assert torch.allclose(sparse.features(linear(x)), expected, atol=1e-6)
        # The reading the shape suggests is a different map.
        linear.weight.copy_(official.weight.reshape(7, 5))
        assert not torch.allclose(sparse.features(linear(x)), expected, atol=1e-2)


def test_points_of_a_voxel_share_a_feature_and_the_grid_starts_at_the_cloud():
    net = tiny()
    cells = box((8, 7, 9), seed=2)
    coords, feats = centres(cells), feats_of(cells)
    twin = torch.cat([coords, coords[:1] + 0.1])  # a second point in the first cell
    with torch.no_grad():
        base = net(PointSet(coords), feats).feats
        doubled = net(PointSet(twin), torch.cat([feats, feats[:1] + 5.0])).feats
        moved = net(PointSet(coords + 3 * GRID), feats).feats
    assert base.shape == (len(cells), 5) and base.abs().max() > 0.1
    assert torch.equal(doubled[-1], doubled[0]) and torch.allclose(doubled[:-1], base, atol=1e-6)
    # Cells are counted from the lowest occupied one, so where the cloud sits is irrelevant.
    assert torch.allclose(moved, base, atol=1e-6)


def test_no_voxel_is_lost_whatever_the_extent():
    """Odd extents and clouds thinner than the total stride are ordinary inputs here."""
    net = tiny()
    for extent in ((9, 7, 5), (8, 8, 1), (1, 1, 1)):
        cells = box(extent, seed=3)
        out = run(net, cells)
        assert out.shape == (len(cells), 5) and torch.isfinite(out).all()
    # A voxel in the last cell of an odd axis takes part in the coarse levels. Four cells
    # from the rest, it is out of reach at full resolution (5³ stem) and at half
    # resolution (a gap of one cell), and next to it at quarter resolution.
    cloud = box((5, 8, 8), seed=4)
    extra = torch.cat([cloud, torch.tensor([[8, 3, 3]])])
    near = cloud[:, 0] == 4
    assert (run(net, extra)[:-1][near] - run(net, cloud)[near]).abs().max() > 1e-3


def test_clouds_of_a_batch_do_not_interact():
    net = tiny()
    clouds = [box((9, 8, 11), seed=5), box((5, 6, 4), seed=6), box((12, 5, 7), seed=7)]
    points = PointSet.cat([PointSet(centres(cells)) for cells in clouds])
    with torch.no_grad():
        together = net(points, torch.cat([feats_of(cells) for cells in clouds])).feats
    alone = torch.cat([run(net, cells) for cells in clouds])
    assert together.abs().max() > 0.1
    assert torch.allclose(together, alone, atol=2e-5)


def test_misuse_is_rejected_with_an_explanation():
    net = tiny()
    cells = box((8, 8, 8), seed=8)
    points = PointSet(centres(cells))
    with pytest.raises(ContractError, match=r"expects features of shape \(\d+, 3\)"):
        net(points, torch.ones(len(cells), 4))
    with pytest.raises(ContractError, match="received no features"):
        net(points, None)
    with pytest.raises(ContractError, match="empty cloud"):
        net(PointSet(torch.zeros(0, 3)), torch.ones(0, 3))
    with pytest.raises(ValueError, match="as many encoder as decoder stages"):
        SpUNet(channels=(32, 64, 96), layers=(2, 2, 2))
    with pytest.raises(ValueError, match="one entry per stage"):
        SpUNet(layers=(2, 3, 4, 6))
    with pytest.raises(ValueError, match="at least one block"):
        SpUNet(layers=(2, 3, 4, 6, 2, 2, 0, 2))
    with pytest.raises(ValueError, match="grid_size must be positive"):
        SpUNet(grid_size=-0.02)
