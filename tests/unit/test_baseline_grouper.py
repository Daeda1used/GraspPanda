"""DepthCylinderGroup against a per-crop brute-force oracle, its contract and its misuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.groupers.depth_cylinder import DepthCylinderGroup
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import GROUPERS
from grasppanda.structures import ContractError, PointFeatures, PointSet

RADIUS, HMIN, HMAX, NSAMPLE = 0.3, -0.1, (0.05, 0.15, 0.3), 6
BATCH, POINTS, SEEDS = 2, 120, 9


def grouper(**overrides) -> DepthCylinderGroup:
    torch.manual_seed(0)
    config = {
        "type": "DepthCylinderGroup",
        "radius": RADIUS,
        "hmin": HMIN,
        "hmax": list(HMAX),
        "nsample": NSAMPLE,
        "mlp_channels": [5, 7],
    }
    return GROUPERS.build({**config, **overrides}, in_channels=7)


def inputs(seed: int = 0, channels: int = 0):
    """A scene, and per frame a few crop centres that are not scene points."""
    generator = torch.Generator().manual_seed(seed)
    scene = PointSet.from_dense(torch.rand(BATCH, POINTS, 3, generator=generator))
    support = PointFeatures(torch.zeros(BATCH * POINTS, channels), scene)
    centres = torch.rand(BATCH * SEEDS, 3, generator=generator)
    centres[0] = 5.0  # a crop far away from every point: empty
    rotation = approach_to_rotation(
        torch.randn(BATCH * SEEDS, 3, generator=generator),
        torch.rand(BATCH * SEEDS, generator=generator) * math.pi,
    )
    return support, centres, rotation


def crop_oracle(coords, centre, rotation, hmax):
    """One crop, written out: members in index order, padded with the first member;
    scene point 0 stands in when the cylinder is empty."""
    local = (coords - centre) @ rotation  # rows are Rᵀ (p - c)
    inside = (
        (local[:, 1] ** 2 + local[:, 2] ** 2 < RADIUS**2)
        & (local[:, 0] > HMIN)
        & (local[:, 0] < hmax)
    )
    members = np.flatnonzero(inside)[:NSAMPLE]
    index = np.full(NSAMPLE, members[0] if len(members) else 0)
    index[: len(members)] = members
    return index, local[index], int(inside.sum())


def test_every_depth_bin_gathers_its_own_cylinder_from_the_scene():
    stage = grouper()
    support, centres, rotation = inputs()
    neighbours = stage.neighbours(support, centres, rotation)
    grouped = stage.group(support, centres, rotation)
    assert neighbours.shape == (BATCH, SEEDS, 1, len(HMAX), NSAMPLE)  # one radius
    assert grouped.shape == (BATCH * SEEDS, 1, len(HMAX), NSAMPLE, 3)
    coords = support.points.coords.double().numpy().reshape(BATCH, POINTS, 3)
    sizes = []
    for seed in range(BATCH * SEEDS):
        frame = seed // SEEDS  # crops never cross frames
        for depth, hmax in enumerate(HMAX):
            index, local, size = crop_oracle(
                coords[frame], centres[seed].double().numpy(), rotation[seed].double().numpy(), hmax
            )
            sizes.append(size)
            assert neighbours[frame, seed % SEEDS, 0, depth].tolist() == index.tolist(), (
                seed,
                depth,
            )
            assert np.allclose(grouped[seed, 0, depth].numpy(), local, atol=1e-5), (seed, depth)
    # the test covers empty, padded and truncated crops, and crops that grow with depth
    assert min(sizes) == 0 and max(sizes) > NSAMPLE and any(0 < s < NSAMPLE for s in sizes)
    counts = np.array(sizes).reshape(-1, len(HMAX))
    assert (np.diff(counts, axis=1) >= 0).all() and (counts[:, -1] > counts[:, 0]).any()


def test_offsets_are_metres_unless_normalised():
    support, centres, rotation = inputs()
    raw = grouper().group(support, centres, rotation)
    scaled = grouper(normalize_offsets=True).group(support, centres, rotation)
    assert torch.allclose(scaled * RADIUS, raw, atol=1e-6)


def test_forward_is_a_shared_mlp_and_a_max_over_neighbours_with_a_depth_axis():
    stage = grouper().eval()
    support, centres, rotation = inputs()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "depth", "channel") == stage.out_axes and stage.support == "scene"
    assert out.feats.shape == (BATCH * SEEDS, len(HMAX), 7) and out.size("depth") == 3
    assert stage.out_channels == 7 and stage.num_depth == 3
    hidden = stage.group(support, centres, rotation).reshape(-1, 3)
    for layer in stage.scales[
        0
    ]:  # linear (no bias) → batch norm → ReLU, the same for all neighbours
        hidden = torch.relu(layer.norm(layer.linear(hidden)))
    pooled = hidden.reshape(BATCH * SEEDS, len(HMAX), NSAMPLE, 7).amax(dim=2)
    assert torch.allclose(out.feats, pooled, atol=1e-5)
    first = stage.scales[0][0].linear
    assert first.in_features == 3 and first.bias is None and stage.fuse is None


def test_the_crop_is_geometry_only_and_batch_norm_sees_every_neighbour():
    stage = grouper().train()
    support, centres, rotation = inputs()
    with_features = PointFeatures(torch.randn(BATCH * POINTS, 4), support.points)
    torch.manual_seed(1)
    first = stage(support, centres, rotation).feats
    assert torch.equal(first, stage(with_features, centres, rotation).feats)  # features unused
    assert int(stage.scales[0][0].norm.num_batches_tracked) == 2
    first.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters())


def test_recorded_neighbours_replace_the_query():
    stage = grouper().eval()
    support, centres, rotation = inputs()
    natural = stage.neighbours(support, centres, rotation)
    assert torch.equal(
        stage(support, centres, rotation, neighbours=natural).feats,
        stage(support, centres, rotation).feats,
    )
    other = natural.flip(dims=(1,))
    assert not torch.allclose(
        stage(support, centres, rotation, neighbours=other).feats,
        stage(support, centres, rotation).feats,
    )
    with pytest.raises(ContractError, match=r"neighbours must be \(2, 9, 1, 3, 6\)"):
        stage(support, centres, rotation, neighbours=natural[:, :, :, :2])


def test_contract_violations_are_explained():
    stage = grouper()
    support, centres, rotation = inputs()
    with pytest.raises(ContractError, match=r"groups over the scene .* each of the 2 frame"):
        stage(support, centres[:7], rotation[:7])
    with pytest.raises(ContractError, match="one frame"):
        stage(support, centres, rotation[:, :2])
    uneven = PointSet.cat([PointSet(torch.rand(12, 3)), PointSet(torch.rand(228, 3))])
    with pytest.raises(ContractError, match="equally sized samples"):
        stage(PointFeatures(torch.zeros(240, 0), uneven), centres, rotation)


def test_defaults_are_the_reference_crop_and_bad_values_are_rejected():
    stage = GROUPERS.build({"type": "DepthCylinderGroup"}, in_channels=256)
    assert (stage.radius, stage.hmin, stage.nsample) == (0.05, -0.02, 64)
    assert stage.hmax == (0.01, 0.02, 0.03, 0.04) and stage.num_depth == 4
    assert stage.out_channels == 256 and not stage.normalize_offsets
    assert len(stage.scales) == 1 and stage.radii == (0.05,) and not stage.uses_seed_features
    assert [layer.linear.in_features for layer in stage.scales[0]] == [3, 64, 128]
    for layer, fan_in in zip(
        stage.scales[0][1:], (64, 128)
    ):  # the reference's shared-MLP initialisation
        assert layer.linear.weight.std().item() == pytest.approx(math.sqrt(2 / fan_in), rel=0.1)
    bad_values = (
        {"radius": 0.0},
        {"nsample": 0},
        {"mlp_channels": []},
        {"hmax": []},
        {"hmax": [-0.05, 0.01]},  # the first cylinder would be empty by construction
        {"hmax": [0.02, 0.02]},
        {"hmax": [0.03, 0.01]},
    )
    for bad in bad_values:
        with pytest.raises(ValueError, match="DepthCylinderGroup needs radius > 0"):
            GROUPERS.build({"type": "DepthCylinderGroup", **bad}, in_channels=256)
