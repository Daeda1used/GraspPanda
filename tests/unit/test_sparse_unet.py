"""SparseUNet14 on tiny inputs: contract, voxel semantics, grid alignment, misuse."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("spconv", reason="needs the spconv extra")

from grasppanda.models.backbones.sparse_unet import SparseUNet14, VoxelMap
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

#: Trainable parameters of the reference MinkUNet14D(3 → 512), counted in its environment.
REFERENCE_PARAMETERS = 14_646_816


@pytest.fixture(autouse=True)
def single_thread():
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny(**overrides) -> SparseUNet14:
    torch.manual_seed(0)
    config = {
        "type": "SparseUNet14",
        "out_channels": 12,
        "stem_channels": 4,
        "encoder_channels": [4, 6, 8, 10],
        "decoder_channels": [8, 6, 6, 6],
        "voxel_size": 0.25,
    }
    return BACKBONES.build({**config, **overrides})


def cloud(batch: int, count: int, seed: int = 0) -> torch.Tensor:
    """Points on a 1/8 grid, so that shifts by whole voxels are exact in float32."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-40, 40, (batch, count, 3), generator=generator).float() / 8


def test_forward_returns_one_feature_per_input_point_and_backpropagates():
    net = tiny()
    points = PointSet.from_dense(cloud(2, 200))
    feats = torch.ones(len(points), 3)
    out = net(points, feats)
    assert net.output == "dense" and net.out_channels == 12 and net.in_channels == 3
    assert out.feats.shape == (400, 12) and out.points is points
    out.expect(points, user="test")  # bound to the very set that came in
    out.feats.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert (
        sum(float(p.grad.abs().sum()) > 0 for p in net.parameters())
        > len(list(net.parameters())) // 2
    )


def test_points_in_one_voxel_share_a_feature_and_frames_do_not_interact():
    net = tiny().eval()
    coords = cloud(2, 150)
    coords[0, 1] = coords[0, 0] + 0.1  # same voxel (size 0.25), different point
    points = PointSet.from_dense(coords)
    with torch.no_grad():
        both = net(points, torch.ones(300, 3)).feats
        alone = net(PointSet.from_dense(coords[:1]), torch.ones(150, 3)).feats
    assert torch.equal(both[0], both[1])
    assert torch.allclose(both[:150], alone, atol=1e-5)


def test_grid_is_aligned_to_absolute_coordinates():
    """Moving the cloud by the total stride changes nothing; moving it by one voxel does."""
    net = tiny().eval()
    assert net.total_stride == 16
    coords = cloud(1, 300)
    ones = torch.ones(300, 3)
    with torch.no_grad():
        base = net(PointSet.from_dense(coords), ones).feats
        aligned = net(PointSet.from_dense(coords + 16 * 0.25), ones).feats
        shifted = net(PointSet.from_dense(coords + 1 * 0.25), ones).feats
    assert torch.allclose(aligned, base, atol=1e-5)
    assert not torch.allclose(shifted, base, atol=1e-3)


def test_variable_size_frames_are_supported():
    net = tiny().eval()
    first, second = cloud(1, 120, 1)[0], cloud(1, 70, 2)[0]
    points = PointSet.cat([PointSet(first), PointSet(second)])
    with torch.no_grad():
        out = net(points, torch.ones(190, 3)).feats
        alone = net(PointSet(second), torch.ones(70, 3)).feats
    assert out.shape == (190, 12) and torch.allclose(out[120:], alone, atol=1e-5)


def test_voxel_map_reduces_features_and_refuses_other_point_sets():
    coords = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.6, 0.1, 0.1], [-0.1, 0.1, 0.1]])
    points = PointSet(coords)
    voxel_map = VoxelMap.build(points, 0.5)
    assert len(voxel_map.voxels) == 3 and voxel_map.inverse[0] == voxel_map.inverse[1]
    feats = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    first, mean = voxel_map.reduce(feats, "first"), voxel_map.reduce(feats, "mean")
    shared = voxel_map.inverse[0]
    assert first[shared].item() == 1.0 and mean[shared].item() == 2.0
    assert torch.equal(
        voxel_map.devoxelize(mean, points).feats[:, 0], torch.tensor([2.0, 2.0, 5.0, 7.0])
    )
    with pytest.raises(ContractError, match="cannot map features onto point set"):
        voxel_map.devoxelize(mean, PointSet(coords.clone()))
    with pytest.raises(ContractError, match="one feature row per voxel"):
        voxel_map.devoxelize(mean[:2], points)


def test_mean_reduction_changes_the_result_only_for_non_constant_features():
    coords = cloud(1, 200) / 8  # dense enough for voxels to hold several points
    points = PointSet.from_dense(coords)
    nets = {rule: tiny(voxel_features=rule).eval() for rule in ("first", "mean")}
    nets["mean"].load_state_dict(nets["first"].state_dict())
    varied = torch.rand(200, 3, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        constant = [net(points, torch.ones(200, 3)).feats for net in nets.values()]
        different = [net(points, varied).feats for net in nets.values()]
    assert torch.allclose(constant[0], constant[1], atol=1e-6)
    assert not torch.allclose(different[0], different[1], atol=1e-4)


def test_misuse_is_rejected_with_an_explanation():
    net = tiny()
    points = PointSet.from_dense(cloud(1, 50))
    with pytest.raises(ContractError, match=r"expects input features of shape \(50, 3\)"):
        net(points, torch.ones(50, 4))
    with pytest.raises(ContractError, match="empty point set"):
        net(PointSet(torch.zeros(0, 3)), torch.ones(0, 3))
    with pytest.raises(ValueError, match="four down-sampling and four up-sampling stages"):
        SparseUNet14(encoder_channels=(8, 16, 32))
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        SparseUNet14(voxel_size=0.0)


def test_default_architecture_has_the_reference_size_and_initialisation():
    torch.manual_seed(0)
    net = BACKBONES.build({"type": "SparseUNet14"})
    assert sum(p.numel() for p in net.parameters()) == REFERENCE_PARAMETERS
    assert net.voxel_size == 0.005 and net.out_channels == 512
    # Layer-type-wise statistics of the reference initialisation.
    assert net.stem.conv.weight.std().item() == pytest.approx(math.sqrt(2 / (32 * 125)), rel=0.05)
    assert net.down[0].conv.weight.std().item() == pytest.approx(math.sqrt(2 / (32 * 8)), rel=0.05)
    bound = 1 / math.sqrt(192 * 8)
    assert net.up[0].conv.weight.abs().max().item() <= bound
    assert net.up[0].conv.weight.std().item() == pytest.approx(bound / math.sqrt(3), rel=0.05)
    assert net.project.weight.std().item() == pytest.approx(math.sqrt(2 / 192), rel=0.05)
    assert net.project.bias.abs().max().item() <= 1 / math.sqrt(192)
    shortcut = net.encoder[1].shortcut
    assert shortcut.conv.weight.std().item() == pytest.approx(math.sqrt(2 / 32), rel=0.05)
    assert net.encoder[0].shortcut is None  # 32 → 32 needs no projection
    assert torch.equal(net.stem.norm.weight, torch.ones(32))
