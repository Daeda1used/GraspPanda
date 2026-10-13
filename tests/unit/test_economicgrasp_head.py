"""EconomicHead: token flow, targets, losses, the decoder and its reference quirks,
encode → decode round trip, and the inputs the bundle must refuse."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.models.heads.economic import TOKENS, EconomicHead, EconomicPredictions
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

ANGLES, DEPTHS, SCORES = 3, 2, 6


def head(channels: int = 8, **overrides) -> EconomicHead:
    torch.manual_seed(0)
    config = {
        "type": "EconomicHead",
        "num_angle": ANGLES,
        "num_depth": DEPTHS,
        "token_channels": 4,
        **overrides,
    }
    return HEADS.build(config, in_channels=channels, gripper=Gripper())


def labels(count: int = 12, seed: int = 0, **overrides) -> CompactMatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    zeros = torch.zeros(count, 3)
    labelled = torch.rand(count, generator=generator) < 0.8
    near = torch.rand(count, generator=generator) < 0.7
    fields = {
        "valid": labelled & near,
        "view_valid": torch.ones(count, dtype=torch.bool),
        "label_point": zeros,
        "distance": torch.zeros(count),
        "object_index": torch.zeros(count, dtype=torch.long),
        "crop_centre": zeros,
        "crop_rotation": torch.eye(3).expand(count, 3, 3),
        "view_graspness": torch.zeros(count, 4),
        "view_max_friction": 0.6,
        "view_labelled": labelled,
        "angle": torch.randint(0, ANGLES, (count,), generator=generator),
        "depth": torch.randint(0, DEPTHS, (count,), generator=generator),
        "score": torch.randint(0, 11, (count,), generator=generator) / 10,
        "width": torch.randint(0, 150, (count,), generator=generator) / 1000,
    }
    fields.update(overrides)
    for name in ("angle", "depth"):  # what the assigner hands over for seeds without a label
        fields[name] = torch.where(fields["view_labelled"], fields[name], -1)
    return CompactMatchedLabels(**fields)


def predictions(count: int = 12, seed: int = 1, grad: bool = False) -> EconomicPredictions:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, requires_grad=grad)

    return EconomicPredictions(
        angle=randn(count, ANGLES + 1),
        depth=randn(count, DEPTHS + 1),
        score=randn(count, SCORES),
        width=randn(count),
    )


def seeds_and_approach(count: int = 12):
    points = PointSet(torch.rand(count, 3, generator=torch.Generator().manual_seed(9)))
    return SeedSet(points=points), ApproachSet(direction=-fibonacci_views(count))


# ------------------------------------------------------------------------ forward


def test_forward_projects_four_tokens_lets_them_interact_and_predicts():
    bundle = head().eval()
    feats = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        raw = bundle(GroupFeatures(feats, ("seed", "channel")))
        tokens = torch.stack([bundle.project[t](feats) for t in TOKENS], dim=1)
        mixed = bundle.interaction(tokens)
    assert TOKENS == ("angle", "depth", "width", "score") and tokens.shape == (5, 4, 4)
    assert raw.angle.shape == (5, ANGLES + 1) and raw.depth.shape == (5, DEPTHS + 1)
    assert raw.score.shape == (5, SCORES) and raw.width.shape == (5,)
    assert torch.equal(raw.angle, bundle.predict["angle"](mixed[:, 0]))
    assert torch.equal(raw.depth, bundle.predict["depth"](mixed[:, 1]))
    assert torch.equal(raw.width, bundle.predict["width"](mixed[:, 2])[:, 0])
    assert torch.equal(raw.score, bundle.predict["score"](mixed[:, 3]))
    # Every prediction depends on every token: that is the interaction.
    bundle.zero_grad()
    bundle(GroupFeatures(feats, ("seed", "channel"))).width.sum().backward()
    assert all(bundle.project[t].weight.grad.abs().sum() > 0 for t in TOKENS)
    assert bundle.in_axes == ("seed", "channel") and bundle.label_format == "compact"


def test_features_with_other_axes_or_channels_are_refused():
    bundle = head()
    with_depth = GroupFeatures(torch.zeros(5, 4, 8), ("seed", "depth", "channel"))
    with pytest.raises(ContractError, match=r"expects features with axes \('seed', 'channel'\)"):
        bundle(with_depth)
    with pytest.raises(ContractError, match="built for 8-channel crop features, got 3"):
        bundle(GroupFeatures(torch.zeros(5, 3), ("seed", "channel")))


def test_defaults_are_the_reference_head():
    bundle = HEADS.build({"type": "EconomicHead"}, in_channels=256, gripper=Gripper())
    assert bundle.codec.grid == (12, 4) and bundle.num_score == 6
    assert bundle.predict["angle"].out_features == 13 and bundle.predict["depth"].out_features == 5
    assert bundle.predict["score"].out_features == 6 and bundle.predict["width"].out_features == 1
    assert bundle.interaction.channels == 64 and bundle.interaction.dropout.p == 0.05
    assert sum(p.numel() for p in bundle.parameters()) == 84_185  # as the reference head
    assert bundle.score_values.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert "score_values" not in bundle.state_dict()  # derived, not a weight
    assert (bundle.decode_argmax, bundle.score_decode) == ("all_logits", "logit_weighted")
    assert bundle.score_kind == "weighted_quality_logits"
    weights = {name: getattr(bundle, f"{name}_loss").weight for name in TOKENS}
    assert weights == {"angle": 1.0, "depth": 1.0, "width": 10.0, "score": 1.0}
    assert type(bundle.width_loss).__name__ == "SmoothL1"
    assert type(bundle.score_loss).__name__ == "CrossEntropy"


# ------------------------------------------------------------------------ targets


def test_targets_are_classes_and_a_scaled_width_under_the_validity_mask():
    bundle = head()
    matched = labels()
    targets = bundle.encode_targets(matched)
    labelled = matched.view_labelled
    assert torch.equal(targets.mask, matched.valid)
    assert torch.equal(targets.angle[labelled], matched.angle[labelled])
    assert torch.equal(targets.depth[labelled], matched.depth[labelled])
    # Seeds without a label sit on the extra class; the mask never lets them count.
    assert (targets.angle[~labelled] == ANGLES).all() and (targets.depth[~labelled] == DEPTHS).all()
    assert not targets.mask[~labelled].any()
    assert torch.allclose(targets.width, matched.width * 10.0)
    assert targets.angle.dtype == targets.score.dtype == torch.long


def test_two_label_levels_share_a_quality_class():
    bundle = head()
    levels = torch.arange(11) / 10  # the label qualities: 0 = no grasp, 1.0 = friction 0.1
    matched = labels(count=11, score=levels, view_labelled=torch.ones(11, dtype=torch.bool))
    classes = bundle.encode_targets(matched).score
    assert classes.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert classes.tolist() == [level // 2 for level in range(11)]
    assert torch.equal(classes, (levels * 10 / 2).long())  # the reference's arithmetic
    coarse = head(num_score=3).encode_targets(matched).score
    assert coarse.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]


def test_without_the_extra_class_unlabelled_seeds_are_parked_on_a_real_bin():
    bundle = head(unmatched_class=False)
    assert bundle.predict["angle"].out_features == ANGLES
    assert bundle.predict["depth"].out_features == DEPTHS
    matched = labels()
    targets = bundle.encode_targets(matched)
    assert targets.angle.max() < ANGLES and (targets.angle[~matched.view_labelled] == 0).all()


def test_labels_that_do_not_fit_are_refused():
    matched = labels()
    with pytest.raises(ContractError, match="angle bin 2, but the head is configured for 2 angle"):
        head(num_angle=2).encode_targets(matched)
    with pytest.raises(ContractError, match="depth bin 1, but the head is configured for 1 depth"):
        head(num_depth=1).encode_targets(matched)
    with pytest.raises(ContractError, match=r"label qualities must lie in \[0, 1\]"):
        head().encode_targets(labels(score=torch.full((12,), 1.5)))
    dense = MatchedLabels(*(torch.zeros(1) for _ in range(9)))
    with pytest.raises(ContractError, match="supervised with compact labels .* MatchedLabels"):
        head().loss(predictions(), dense)


# --------------------------------------------------------------------------- loss


def test_loss_terms_match_hand_computation():
    bundle = head()
    matched = labels(seed=3)
    raw = predictions(grad=True)
    losses = bundle.loss(raw, matched)
    targets = bundle.encode_targets(matched)
    valid = matched.valid
    assert 0 < valid.sum() < len(valid)
    assert {k for k in losses if k.startswith("loss/")} == {
        "loss/angle",
        "loss/depth",
        "loss/score",
        "loss/width",
    }
    for what in ("angle", "depth", "score"):
        logits, target = getattr(raw, what), getattr(targets, what)
        expected = F.cross_entropy(logits[valid], target[valid])
        assert losses[f"loss/{what}"].item() == pytest.approx(expected.item(), rel=1e-6)
        accuracy = (logits[valid].argmax(-1) == target[valid]).float().mean()
        assert losses[f"metric/{what}_accuracy"].item() == pytest.approx(accuracy.item())
    width = 10.0 * F.smooth_l1_loss(raw.width[valid], matched.width[valid] * 10.0)
    assert losses["loss/width"].item() == pytest.approx(width.item(), rel=1e-6)
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    for tensor in (raw.angle, raw.depth, raw.score, raw.width):
        assert tensor.grad[~valid].abs().sum() == 0 and tensor.grad[valid].abs().sum() > 0


def test_a_batch_without_supervised_seeds_gives_zero_losses_not_nan():
    bundle = head()
    nothing = labels(valid=torch.zeros(12, dtype=torch.bool))
    raw = predictions(grad=True)
    losses = bundle.loss(raw, nothing)
    assert all(value.item() == 0.0 for value in losses.values())
    sum(losses[f"loss/{what}"] for what in TOKENS).backward()  # still part of the graph
    assert raw.angle.grad.abs().sum() == 0


def test_losses_are_configurable_primitives():
    bundle = head(width_loss={"type": "L1", "weight": 2.0}, score_loss={"type": "CrossEntropy"})
    assert type(bundle.width_loss).__name__ == "L1" and bundle.width_loss.weight == 2.0
    assert bundle.angle_loss.weight == 1.0 and head().width_loss.weight == 10.0


# ------------------------------------------------------------------------- decode


def test_decode_takes_the_most_likely_bins_the_width_and_a_logit_weighted_score():
    bundle = head()
    seeds, approach = seeds_and_approach()
    raw = predictions()
    grasps = bundle.decode(raw, seeds, approach).validate()
    assert len(grasps) == 12 and grasps.score_kind == "weighted_quality_logits"
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for seed in range(12):
        angle, depth = raw.angle[seed].argmax().item(), raw.depth[seed].argmax().item()
        score = sum(c * logit for c, logit in zip(values, raw.score[seed].tolist()))
        assert grasps.score[seed].item() == pytest.approx(score, abs=1e-5)  # raw logits: unbounded
        assert grasps.depth[seed].item() == pytest.approx((depth + 1) * 0.01)
        expected_width = min(max(1.2 * raw.width[seed].item() / 10.0, 0.0), 0.1)
        assert grasps.width[seed].item() == pytest.approx(expected_width, abs=1e-7)
        rotation = approach_to_rotation(
            approach.direction[seed], torch.tensor(angle * math.pi / ANGLES)
        )
        assert torch.allclose(grasps.rotation[seed], rotation, atol=1e-6)
    assert torch.equal(grasps.translation, seeds.points.coords)
    assert torch.allclose(grasps.height, torch.full((12,), 0.02)) and (grasps.object_id == -1).all()
    assert grasps.gripper is bundle.gripper and grasps.frame == seeds.points.frame
    assert grasps.score.min() < 0 or grasps.score.max() > 1  # not a probability
    with pytest.raises(ContractError, match="do not line up"):
        bundle.decode(raw, seeds, ApproachSet(direction=approach.direction[:4]))


def test_the_extra_class_takes_part_in_the_arg_max_unless_told_otherwise():
    """Upstream 7361bb6 (default) against upstream 4119bdc (`supervised`)."""
    seeds, approach = seeds_and_approach()
    raw = predictions()
    raw.angle[:6, ANGLES] = 50.0  # the never-supervised class wins for half of the seeds
    raw.depth[3:9, DEPTHS] = 50.0
    pinned = head().decode(raw, seeds, approach)
    later = head(decode_argmax="supervised").decode(raw, seeds, approach)
    # all_logits: the extra angle class is the angle π, the extra depth one step too deep.
    spun = approach_to_rotation(approach.direction[:6], torch.full((6,), math.pi))
    assert torch.allclose(pinned.rotation[:6], spun, atol=1e-6)
    assert torch.allclose(pinned.depth[3:9], torch.full((6,), (DEPTHS + 1) * 0.01))
    # supervised: the best real bin instead.
    assert (
        torch.equal(later.depth, (raw.depth[:, :DEPTHS].argmax(-1) + 1) * 0.01)
        and later.depth.max() <= DEPTHS * 0.01 + 1e-9
    )
    best = raw.angle[:, :ANGLES].argmax(-1) * math.pi / ANGLES
    assert torch.allclose(later.rotation, approach_to_rotation(approach.direction, best), atol=1e-6)
    # Where the extra class does not win, the two decoders agree; nothing else differs.
    assert torch.equal(pinned.rotation[9:], later.rotation[9:])
    assert torch.equal(pinned.depth[9:], later.depth[9:])
    assert torch.equal(pinned.score, later.score) and torch.equal(pinned.width, later.width)


def test_softmax_expectation_is_the_controlled_score_variant():
    seeds, approach = seeds_and_approach()
    raw = predictions()
    variant = head(score_decode="softmax_expectation")
    grasps = variant.decode(raw, seeds, approach)
    values = torch.arange(SCORES) / (SCORES - 1)
    assert torch.allclose(grasps.score, (raw.score.softmax(-1) * values).sum(-1), atol=1e-6)
    assert grasps.score.min() >= 0 and grasps.score.max() <= 1
    assert grasps.score_kind == "expected_quality" and variant.score_kind == "expected_quality"
    reference = head().decode(raw, seeds, approach)
    assert not torch.allclose(grasps.score, reference.score)
    assert torch.equal(grasps.rotation, reference.rotation)


def test_encode_then_decode_recovers_the_labelled_grasp():
    """One-hot logits of the targets decode to the labelled bins and width."""
    bundle = head(width_decode_gain=1.0, score_decode="softmax_expectation")
    everywhere = torch.ones(40, dtype=torch.bool)
    matched = labels(count=40, seed=6, view_labelled=everywhere, valid=everywhere)
    targets = bundle.encode_targets(matched)
    raw = EconomicPredictions(
        angle=F.one_hot(targets.angle, ANGLES + 1) * 30.0,
        depth=F.one_hot(targets.depth, DEPTHS + 1) * 30.0,
        score=F.one_hot(targets.score, SCORES) * 30.0,
        width=targets.width,
    )
    seeds, approach = seeds_and_approach(count=40)
    grasps = bundle.decode(raw, seeds, approach)
    assert torch.allclose(grasps.depth, (matched.depth + 1) * 0.01)
    assert torch.allclose(grasps.width, matched.width.clamp(max=0.1), atol=1e-6)
    expected = approach_to_rotation(approach.direction, matched.angle * math.pi / ANGLES)
    assert torch.allclose(grasps.rotation, expected, atol=1e-6)
    # The decoded score is the class value: the label quality rounded down to a class.
    assert torch.allclose(grasps.score, torch.floor(matched.score * 5 + 1e-4) / 5, atol=1e-5)
    # With the reference gain the decoder opens the gripper 20 % wider than regressed.
    wider = head(score_decode="softmax_expectation").decode(raw, seeds, approach)
    inside = grasps.width * 1.2 < 0.1
    assert torch.allclose(wider.width[inside], grasps.width[inside] * 1.2, atol=1e-6)
    assert wider.width.max() <= 0.1 + 1e-7


def test_the_gripper_is_injected_and_options_are_checked():
    with pytest.raises(BuildError, match="needs gripper"):
        HEADS.build({"type": "EconomicHead"}, in_channels=8)
    with pytest.raises(BuildError, match="gripper must be a Gripper"):
        HEADS.build({"type": "EconomicHead"}, in_channels=8, gripper={"max_width": 0.1})
    with pytest.raises(BuildError, match="in the config, but the parent component provides 8"):
        HEADS.build({"type": "EconomicHead", "in_channels": 4}, in_channels=8, gripper=Gripper())
    narrow = HEADS.build(
        {"type": "EconomicHead", "num_angle": ANGLES, "num_depth": DEPTHS},
        in_channels=8,
        gripper=Gripper(max_width=0.05),
    )
    seeds, approach = seeds_and_approach()
    raw = predictions()
    raw.width.fill_(5.0)
    assert narrow.decode(raw, seeds, approach).width.tolist() == pytest.approx([0.05] * 12)
    for bad, message in (
        ({"decode_argmax": "first"}, "unknown decode_argmax"),
        ({"score_decode": "sigmoid"}, "unknown score_decode"),
        ({"num_score": 1}, "num_score must be at least 2"),
        ({"num_angle": 0}, "must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            EconomicHead(**bad, in_channels=8, gripper=Gripper())
