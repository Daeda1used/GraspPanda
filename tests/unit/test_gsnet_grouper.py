"""CylinderGroup against a per-seed brute-force crop, its contract and its misuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.groupers.cylinder import CylinderGroup
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import GROUPERS, BuildError
from grasppanda.structures import ContractError, PointFeatures, PointSet

RADIUS, HMIN, HMAX, NSAMPLE = 0.3, -0.1, 0.25, 5


def grouper(channels: int = 4, **overrides) -> CylinderGroup:
    torch.manual_seed(0)
    config = {
        "type": "CylinderGroup",
        "radius": RADIUS,
        "hmin": HMIN,
        "hmax": HMAX,
        "nsample": NSAMPLE,
        "mlp_channels": [6, 7],
    }
    return GROUPERS.build({**config, **overrides}, in_channels=channels)


def inputs(batch: int = 2, per_frame: int = 40, channels: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    points = PointSet.from_dense(torch.rand(batch, per_frame, 3, generator=generator))
    feats = torch.randn(batch * per_frame, channels, generator=generator)
    rotation = approach_to_rotation(
        torch.randn(batch * per_frame, 3, generator=generator),
        torch.rand(batch * per_frame, generator=generator) * math.pi,
    )
    return PointFeatures(feats, points), points.coords.clone(), rotation


def crop_oracle(coords, feats, centre, rotation):
    """One crop, written out: members in index order, padded with the first member."""
    local = (coords - centre) @ rotation  # rows are Rᵀ (p - c)
    inside = (
        (local[:, 1] ** 2 + local[:, 2] ** 2 < RADIUS**2)
        & (local[:, 0] > HMIN)
        & (local[:, 0] < HMAX)
    )
    members = np.flatnonzero(inside)[:NSAMPLE]
    index = np.full(NSAMPLE, members[0])
    index[: len(members)] = members
    return np.concatenate([local[index] / RADIUS, feats[index]], axis=1), int(inside.sum())


def test_group_gathers_the_cylinder_neighbourhood_of_every_seed():
    stage = grouper()
    support, centres, rotation = inputs()
    grouped = stage.group(support, centres, rotation)
    assert grouped.shape == (80, NSAMPLE, 3 + 4)
    coords = support.points.coords.double().numpy()
    feats = support.feats.double().numpy()
    sizes = []
    for seed in range(80):
        frame = slice(0, 40) if seed < 40 else slice(40, 80)  # crops never cross frames
        expected, size = crop_oracle(
            coords[frame], feats[frame], coords[seed], rotation[seed].double().numpy()
        )
        sizes.append(size)
        assert np.allclose(grouped[seed].numpy(), expected, atol=1e-5), seed
    assert 1 <= min(sizes) < NSAMPLE < max(sizes)  # the test covers padded and truncated crops


def test_crops_do_not_see_other_frames():
    stage = grouper()
    support, _, rotation = inputs(batch=1, per_frame=20)
    twice = PointSet.from_dense(support.points.coords.reshape(1, 20, 3).repeat(2, 1, 1))
    feats = torch.cat([support.feats, support.feats + 100.0])  # frame 1 is recognisable
    grouped = stage.group(PointFeatures(feats, twice), twice.coords, rotation.repeat(2, 1, 1))
    assert grouped[:20, :, 3:].max() < 50 and grouped[20:, :, 3:].min() > 50
    assert torch.allclose(grouped[:20, :, :3], grouped[20:, :, :3], atol=1e-6)


def test_forward_is_a_shared_mlp_and_a_max_over_neighbours():
    stage = grouper().eval()
    support, centres, rotation = inputs()
    out = stage(support, centres, rotation)
    assert out.axes == ("seed", "channel") == stage.out_axes and stage.support == "seeds"
    assert out.feats.shape == (80, 7) and stage.out_channels == 7 and out.size("channel") == 7
    grouped = stage.group(support, centres, rotation)
    hidden = grouped
    for layer in stage.mlp:  # linear (no bias) → batch norm → ReLU, the same for all neighbours
        hidden = torch.relu(layer.norm(layer.linear(hidden).transpose(1, 2)).transpose(1, 2))
    assert torch.allclose(out.feats, hidden.amax(dim=1), atol=1e-5)
    assert stage.mlp[0].linear.in_features == 3 + 4 and stage.mlp[0].linear.bias is None


def test_gradients_reach_the_support_features_and_the_mlp():
    stage = grouper().train()
    support, centres, rotation = inputs()
    support.feats.requires_grad_(True)
    stage(support, centres, rotation).feats.square().sum().backward()
    assert support.feats.grad.abs().sum() > 0
    assert all(p.grad is not None for p in stage.parameters())


def test_training_crop_centres_and_frames_may_differ_from_the_seeds():
    stage = grouper()
    support, centres, rotation = inputs()
    moved = stage.group(support, centres + 0.05, rotation)
    assert not torch.allclose(moved, stage.group(support, centres, rotation))


def test_contract_violations_are_explained():
    stage = grouper()
    support, centres, rotation = inputs()
    with pytest.raises(ContractError, match="built for 4-channel support features, got 9"):
        stage(PointFeatures(torch.zeros(80, 9), support.points), centres, rotation)
    with pytest.raises(ContractError, match=r"groups among the seeds .* M = 80"):
        stage(support, centres[:30], rotation[:30])
    uneven = PointSet.cat([PointSet(torch.rand(12, 3)), PointSet(torch.rand(68, 3))])
    with pytest.raises(ContractError, match="equally sized samples"):
        stage(PointFeatures(support.feats, uneven), centres, rotation)


def test_defaults_are_the_reference_cylinder_and_bad_values_are_rejected():
    stage = GROUPERS.build({"type": "CylinderGroup"}, in_channels=512)
    assert (stage.radius, stage.hmin, stage.hmax, stage.nsample) == (0.05, -0.02, 0.04, 16)
    assert stage.out_channels == 256 and stage.mlp[0].linear.in_features == 515
    for layer, fan_in in zip(stage.mlp, (515, 256)):  # the reference's shared-MLP initialisation
        assert layer.linear.weight.std().item() == pytest.approx(math.sqrt(2 / fan_in), rel=0.05)
    with pytest.raises(BuildError, match="needs in_channels"):
        GROUPERS.build({"type": "CylinderGroup"})
    for bad in ({"radius": 0.0}, {"hmin": 0.1, "hmax": 0.0}, {"nsample": 0}, {"mlp_channels": []}):
        with pytest.raises(ValueError, match="CylinderGroup needs radius > 0"):
            GROUPERS.build({"type": "CylinderGroup", **bad}, in_channels=8)
