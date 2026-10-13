"""SWADHead and its codec: grid order, targets, loss, decoding, encode → decode round
trip, and the inputs the bundle must refuse."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.heads.codec import AngleDepthCodec
from grasppanda.models.heads.swad import SWADHead, SWADPredictions
from grasppanda.ops import approach_to_rotation, fibonacci_views
from grasppanda.registry import HEADS, BuildError
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    Gripper,
    GroupFeatures,
    PointSet,
    SeedSet,
)

ANGLES, DEPTHS = 3, 2


def head(channels: int = 8, **overrides) -> SWADHead:
    torch.manual_seed(0)
    config = {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS, **overrides}
    return HEADS.build(config, in_channels=channels, gripper=Gripper())


def labels(count: int = 6, seed: int = 0, valid: torch.Tensor | None = None) -> MatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    levels = torch.tensor([-1.0, 0.0, 0.2, 0.4, 0.5, 1.0])
    friction = levels[torch.randint(0, 6, (count, ANGLES, DEPTHS), generator=generator)]
    width = torch.rand(count, ANGLES, DEPTHS, generator=generator) * 0.13  # some beyond 0.1 m
    zeros = torch.zeros(count, 3)
    return MatchedLabels(
        valid=torch.ones(count, dtype=torch.bool) if valid is None else valid,
        label_point=zeros,
        distance=torch.zeros(count),
        object_index=torch.zeros(count, dtype=torch.long),
        crop_centre=zeros,
        crop_rotation=torch.eye(3).expand(count, 3, 3),
        view_friction=torch.zeros(count, 4, ANGLES, DEPTHS),
        friction=friction,
        width=width,
    )


def seeds_and_approach(count: int = 6):
    points = PointSet(torch.rand(count, 3, generator=torch.Generator().manual_seed(9)))
    direction = -fibonacci_views(count)
    return SeedSet(points=points), ApproachSet(direction=direction)


# -------------------------------------------------------------------------- codec


def test_codec_owns_grid_order_and_units():
    codec = AngleDepthCodec()
    assert codec.grid == (12, 4) and codec.num_cells == 48
    grid = torch.arange(48).reshape(12, 4)
    flat = codec.flatten(grid)
    assert flat.tolist() == list(range(48)) and torch.equal(codec.unflatten(flat), grid)
    angle, depth = codec.split_cell(torch.tensor([0, 5, 47]))  # angle-major: cell = a * D + d
    assert angle.tolist() == [0, 1, 11] and depth.tolist() == [0, 1, 3]
    assert torch.equal(codec.cell(angle, depth), torch.tensor([0, 5, 47]))
    assert codec.angle(torch.tensor([0, 6, 11])).tolist() == pytest.approx(
        [0.0, math.pi / 2, 11 * math.pi / 12]
    )
    assert codec.depth(torch.tensor([0, 3])).tolist() == pytest.approx([0.01, 0.04])
    assert codec.encode_width(torch.tensor([0.05])).item() == pytest.approx(0.5)
    decoded = codec.decode_width(torch.tensor([-0.3, 0.5, 2.0]), max_width=0.1)
    assert decoded.tolist() == pytest.approx([0.0, 0.06, 0.1])  # ×1.2 / 10, clamped to the gripper
    with pytest.raises(
        ContractError, match=r"grid of \(3, 2\), but the head is configured for \(12, 4\)"
    ):
        codec.expect_grid(torch.zeros(5, 3, 2), what="the labels", user="test")
    with pytest.raises(ValueError, match="must be positive"):
        AngleDepthCodec(num_angle=0)
    with pytest.raises(AttributeError):  # frozen: nobody can re-bin behind the bundle's back
        codec.num_angle = 6


# ------------------------------------------------------------------------ forward


def test_forward_emits_score_and_width_grids_in_codec_order():
    bundle = head()
    feats = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    raw = bundle(GroupFeatures(feats, ("seed", "channel")))
    assert raw.score.shape == raw.width.shape == (5, ANGLES, DEPTHS)
    flat = bundle.output(F.relu(bundle.hidden(feats)))
    cells = ANGLES * DEPTHS
    assert torch.equal(bundle.codec.flatten(raw.score), flat[:, :cells])  # score channels first
    assert torch.equal(bundle.codec.flatten(raw.width), flat[:, cells:])
    assert bundle.in_axes == ("seed", "channel") and bundle.score_kind == "predicted_quality"
    assert bundle.hidden.in_features == bundle.hidden.out_features == 8


def test_features_with_other_axes_or_channels_are_refused():
    bundle = head()
    with_depth = GroupFeatures(torch.zeros(5, 4, 8), ("seed", "depth", "channel"))
    with pytest.raises(ContractError, match=r"expects features with axes \('seed', 'channel'\)"):
        bundle(with_depth)
    with pytest.raises(ContractError, match="built for 8-channel crop features, got 3"):
        bundle(GroupFeatures(torch.zeros(5, 3), ("seed", "channel")))


# ------------------------------------------------------------------------ targets


def score_oracle(friction, width, valid, worst, best, max_width=0.1):
    usable = valid[:, None, None] & (friction > 0) & (width <= max_width)
    target = torch.zeros_like(friction)
    target[usable] = torch.log(worst / friction[usable]) / (torch.log(worst / best) + 1e-6)
    return target, usable


def test_score_targets_are_normalised_with_the_extrema_of_the_batch():
    bundle = head()
    matched = labels()
    targets = bundle.encode_targets(matched)
    everything = torch.ones(6, dtype=torch.bool)
    usable = (matched.friction > 0) & (matched.width <= 0.1)
    worst, best = matched.friction[usable].max(), matched.friction[usable].min()
    expected, _ = score_oracle(matched.friction, matched.width, everything, worst, best)
    assert torch.allclose(targets.score, expected, atol=1e-6)
    assert targets.score.max().item() == pytest.approx(1.0, abs=1e-5)  # the best grasp of the batch
    assert bool(targets.score_mask.all())
    assert torch.equal(targets.width_mask, targets.score > 0)
    # The worst usable friction maps to exactly 0, so it does not supervise the width either.
    assert (targets.score[usable & (matched.friction == worst)] == 0).all()
    assert torch.allclose(targets.width, matched.width * 10.0)

    # The same labels in a batch with different extrema get different targets ...
    harder = labels()
    harder.friction[harder.friction == 1.0] = -1.0
    assert not torch.allclose(
        bundle.encode_targets(harder).score[usable & (matched.friction < 1)],
        targets.score[usable & (matched.friction < 1)],
    )
    # ... unless the controlled variant with a fixed friction range is selected.
    fixed = head(score_target="log_friction_fixed_range", friction_range=[0.1, 1.0])
    full, _ = score_oracle(
        matched.friction, matched.width, everything, torch.tensor(1.0), torch.tensor(0.1)
    )
    assert torch.allclose(fixed.encode_targets(matched).score, full, atol=1e-6)
    keep = harder.friction > 0
    assert torch.allclose(
        fixed.encode_targets(harder).score[keep], fixed.encode_targets(matched).score[keep]
    )


def test_unsupervised_seeds_shape_neither_targets_nor_extrema():
    bundle = head()
    valid = torch.tensor([True, True, False, True, False, True])
    matched = labels(valid=valid)
    matched.friction[2] = 0.05  # would be the best friction of the batch if it counted
    targets = bundle.encode_targets(matched)
    usable = valid[:, None, None] & (matched.friction > 0) & (matched.width <= 0.1)
    expected, _ = score_oracle(
        matched.friction,
        matched.width,
        valid,
        matched.friction[usable].max(),
        matched.friction[usable].min(),
    )
    assert (
        torch.allclose(targets.score, expected, atol=1e-6)
        and targets.score[~valid].abs().sum() == 0
    )
    assert not targets.score_mask[~valid].any() and not targets.width_mask[~valid].any()


def test_labels_on_another_grid_are_refused():
    matched = labels()
    with pytest.raises(
        ContractError, match=r"friction labels are laid out on an angle × depth grid of \(3, 2\)"
    ):
        head(num_angle=12, num_depth=4).encode_targets(matched)


# --------------------------------------------------------------------------- loss


def test_loss_terms_match_hand_computation():
    bundle = head()
    valid = torch.tensor([True, True, False, True, True, True])
    matched = labels(seed=3, valid=valid)
    generator = torch.Generator().manual_seed(4)
    raw = SWADPredictions(
        score=torch.randn(6, ANGLES, DEPTHS, generator=generator, requires_grad=True),
        width=torch.randn(6, ANGLES, DEPTHS, generator=generator, requires_grad=True),
    )
    losses = bundle.loss(raw, matched)
    targets = bundle.encode_targets(matched)
    assert set(losses) == {"loss/score", "loss/width"}
    assert losses["loss/score"].item() == pytest.approx(
        15.0 * F.smooth_l1_loss(raw.score[valid], targets.score[valid]).item(), rel=1e-6
    )
    positive = targets.score > 0
    assert losses["loss/width"].item() == pytest.approx(
        10.0 * F.smooth_l1_loss(raw.width[positive], matched.width[positive] * 10.0).item(),
        rel=1e-6,
    )
    (losses["loss/score"] + losses["loss/width"]).backward()
    assert raw.score.grad[2].abs().sum() == 0 and raw.width.grad[~positive].abs().sum() == 0

    nothing = labels()
    nothing.friction.fill_(-1.0)  # no usable grasp anywhere: zero width loss, not NaN
    assert bundle.loss(raw, nothing)["loss/width"].item() == 0.0


def test_losses_are_configurable_primitives():
    bundle = head(score_loss={"type": "L1", "weight": 2.0})
    assert type(bundle.score_loss).__name__ == "L1" and bundle.score_loss.weight == 2.0
    assert type(bundle.width_loss).__name__ == "SmoothL1" and bundle.width_loss.weight == 10.0
    assert head().score_loss.weight == 15.0


# ------------------------------------------------------------------------- decode


def test_decode_takes_the_best_cell_of_every_seed():
    bundle = head()
    seeds, approach = seeds_and_approach()
    generator = torch.Generator().manual_seed(5)
    raw = SWADPredictions(
        score=torch.randn(6, ANGLES, DEPTHS, generator=generator),
        width=torch.rand(6, ANGLES, DEPTHS, generator=generator),
    )
    grasps = bundle.decode(raw, seeds, approach).validate()
    assert len(grasps) == 6 and grasps.score_kind == "predicted_quality"
    for seed in range(6):
        best = raw.score[seed].flatten().argmax().item()
        angle, depth = best // DEPTHS, best % DEPTHS
        assert grasps.score[seed].item() == pytest.approx(raw.score[seed].max().item())
        assert grasps.depth[seed].item() == pytest.approx((depth + 1) * 0.01)
        expected_width = min(max(1.2 * raw.width[seed, angle, depth].item() / 10.0, 0.0), 0.1)
        assert grasps.width[seed].item() == pytest.approx(expected_width, abs=1e-7)
        rotation = approach_to_rotation(
            approach.direction[seed], torch.tensor(angle * math.pi / ANGLES)
        )
        assert torch.allclose(grasps.rotation[seed], rotation, atol=1e-6)
    assert torch.equal(grasps.translation, seeds.points.coords)
    assert torch.allclose(grasps.height, torch.full((6,), 0.02)) and (grasps.object_id == -1).all()
    assert grasps.gripper is bundle.gripper and grasps.frame == seeds.points.frame
    with pytest.raises(ContractError, match="do not line up"):
        bundle.decode(raw, seeds, ApproachSet(direction=approach.direction[:4]))


def test_encode_then_decode_recovers_the_best_labelled_grasp():
    """Targets fed back as predictions decode to the lowest-friction cell and its width."""
    bundle = head(width_decode_gain=1.0, score_target="log_friction_fixed_range")
    matched = labels(count=40, seed=6)
    targets = bundle.encode_targets(matched)
    seeds, approach = seeds_and_approach(count=40)
    grasps = bundle.decode(SWADPredictions(targets.score, targets.width), seeds, approach)
    usable = (matched.friction > 0) & (matched.width <= 0.1)
    checked = 0
    for seed in range(40):
        if not (usable[seed] & (matched.friction[seed] < 1.0)).any():
            continue  # no grasp with a positive score target: nothing to recover
        friction = torch.where(
            usable[seed], matched.friction[seed], torch.full_like(matched.friction[seed], 9.0)
        )
        best = friction.flatten().argmin().item()  # ties resolve to the first cell on both sides
        angle, depth = best // DEPTHS, best % DEPTHS
        assert grasps.depth[seed].item() == pytest.approx((depth + 1) * 0.01)
        assert grasps.width[seed].item() == pytest.approx(
            matched.width[seed, angle, depth].item(), abs=1e-6
        )
        expected = approach_to_rotation(
            approach.direction[seed], torch.tensor(angle * math.pi / ANGLES)
        )
        assert torch.allclose(grasps.rotation[seed], expected, atol=1e-6)
        checked += 1
    assert checked > 20
    # With the reference gain the decoder opens the gripper 20 % wider than regressed.
    wider = head(score_target="log_friction_fixed_range").decode(
        SWADPredictions(targets.score, targets.width), seeds, approach
    )
    inside = grasps.width * 1.2 < 0.1
    assert torch.allclose(wider.width[inside], grasps.width[inside] * 1.2, atol=1e-6)
    assert wider.width.max() <= 0.1 + 1e-7


def test_the_gripper_is_injected_and_bounds_labels_and_widths():
    with pytest.raises(BuildError, match="needs gripper"):
        HEADS.build({"type": "SWADHead"}, in_channels=8)
    with pytest.raises(BuildError, match="gripper must be a Gripper"):
        HEADS.build({"type": "SWADHead"}, in_channels=8, gripper={"max_width": 0.1})
    with pytest.raises(BuildError, match="in the config, but the parent component provides 8"):
        HEADS.build({"type": "SWADHead", "in_channels": 4}, in_channels=8, gripper=Gripper())
    narrow = HEADS.build(
        {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS},
        in_channels=8,
        gripper=Gripper(max_width=0.05),
    )
    matched = labels()
    wide = (matched.width > 0.05) & (matched.width <= 0.1) & (matched.friction > 0)
    assert wide.any() and (narrow.encode_targets(matched).score[wide] == 0).all()
    seeds, approach = seeds_and_approach()
    raw = SWADPredictions(torch.zeros(6, ANGLES, DEPTHS), torch.full((6, ANGLES, DEPTHS), 5.0))
    assert narrow.decode(raw, seeds, approach).width.tolist() == pytest.approx([0.05] * 6)
    with pytest.raises(ValueError, match="unknown score_target"):
        SWADHead(score_target="softmax", in_channels=8, gripper=Gripper())
    with pytest.raises(ValueError, match="friction_range must be"):
        SWADHead(friction_range=(1.0, 0.1), in_channels=8, gripper=Gripper())
