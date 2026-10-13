"""OperationHead: layout of its outputs, targets and losses against the definition
written out in loops, decoding, and the inputs it must refuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.heads.friction import log_friction_score, worst_friction
from grasppanda.models.heads.operation import OperationHead, OperationPredictions
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import HEADS, BuildError
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    Gripper,
    GroupFeatures,
    PointSet,
    SeedSet,
)

SEEDS, VIEWS, ANGLES, DEPTHS, CHANNELS = 14, 6, 5, 3, 8
MAX_WIDTH, MAX_TOLERANCE = 0.1, 0.05


def head(**overrides) -> OperationHead:
    torch.manual_seed(0)
    config = {
        "type": "OperationHead",
        "num_angle": ANGLES,
        "num_depth": DEPTHS,
        "hidden_channels": 6,
    }
    return HEADS.build({**config, **overrides}, in_channels=CHANNELS, gripper=Gripper())


def groups(seed: int = 0) -> GroupFeatures:
    generator = torch.Generator().manual_seed(seed)
    feats = torch.randn(SEEDS, DEPTHS, CHANNELS, generator=generator)
    return GroupFeatures(feats, ("seed", "depth", "channel"))


def matched(seed: int = 0, tolerance: bool = True) -> MatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    levels = torch.tensor([-1.0, 0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    shape = (SEEDS, VIEWS, ANGLES, DEPTHS)
    view_friction = levels[torch.randint(0, len(levels), shape, generator=generator)]
    view_width = torch.rand(shape, generator=generator) * 0.12  # some wider than the gripper
    chosen = torch.randint(0, VIEWS, (SEEDS,), generator=generator)
    rows = torch.arange(SEEDS)
    valid = torch.rand(SEEDS, generator=generator) < 0.7
    return MatchedLabels(
        valid=valid,
        label_point=torch.zeros(SEEDS, 3),
        distance=torch.zeros(SEEDS),
        object_index=torch.zeros(SEEDS, dtype=torch.long),
        crop_centre=torch.zeros(SEEDS, 3),
        crop_rotation=torch.eye(3).expand(SEEDS, 3, 3),
        view_friction=view_friction,
        friction=view_friction[rows, chosen],
        width=view_width[rows, chosen],
        on_object=valid,
        view_width=view_width,
        tolerance=(
            torch.randint(0, 51, (SEEDS, ANGLES, DEPTHS), generator=generator) * 0.001
            if tolerance
            else None
        ),
    )


def huber(error: float) -> float:
    return 0.5 * error * error if abs(error) <= 1 else abs(error) - 0.5


# --------------------------------------------------------------------------- forward


def test_outputs_are_grids_over_angle_and_depth_in_the_reference_channel_layout():
    stage = head().eval()
    feats = groups()
    raw = stage(feats)
    for grid in (raw.score, raw.angle, raw.width, raw.tolerance):
        assert grid.shape == (SEEDS, ANGLES, DEPTHS)
    # one vector per (seed, depth); output channels are [score | angle logits | width] × angle
    flat = feats.feats.reshape(SEEDS * DEPTHS, CHANNELS)
    operation = stage.operation(flat).reshape(SEEDS, DEPTHS, 3 * ANGLES)
    tolerance = stage.tolerance(flat).reshape(SEEDS, DEPTHS, ANGLES)
    for seed, angle, depth in ((0, 0, 0), (3, 4, 2), (13, 2, 1)):
        assert raw.score[seed, angle, depth] == operation[seed, depth, angle]
        assert raw.angle[seed, angle, depth] == operation[seed, depth, ANGLES + angle]
        assert raw.width[seed, angle, depth] == operation[seed, depth, 2 * ANGLES + angle]
        assert raw.tolerance[seed, angle, depth] == tolerance[seed, depth, angle]
    assert stage.in_axes == ("seed", "depth", "channel") and stage.codec.grid == (ANGLES, DEPTHS)
    assert stage.codec.width_scale == 1.0  # widths are regressed in metres


def test_batch_norm_statistics_are_taken_over_seeds_and_depths():
    stage = head().train()
    stage(groups())
    assert int(stage.operation.norm1.num_batches_tracked) == 1
    with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
        single = GroupFeatures(torch.randn(1, 1, CHANNELS), ("seed", "depth", "channel"))
        HEADS.build(
            {"type": "OperationHead", "num_depth": 1}, in_channels=CHANNELS, gripper=Gripper()
        ).train()(single)


# --------------------------------------------------------------------------- targets


def test_targets_are_the_best_angle_per_depth_and_the_labels_at_it():
    stage = head()
    labels = matched()
    targets = stage.encode_targets(labels)
    worst = float(labels.view_friction.max())  # over every view of every seed, usable or not
    assert worst == 1.0
    friction, width = labels.friction.numpy(), labels.width.numpy()
    supervised = 0
    for seed in range(SEEDS):
        for depth in range(DEPTHS):
            scores = [
                math.log(worst / friction[seed, a, depth])
                if friction[seed, a, depth] > 0 and width[seed, a, depth] <= MAX_WIDTH
                else 0.0
                for a in range(ANGLES)
            ]
            best = int(np.argmax(scores))  # the first maximum
            assert int(targets.angle[seed, depth]) == best
            assert float(targets.score[seed, depth]) == pytest.approx(scores[best], abs=1e-6)
            assert targets.width[seed, depth] == labels.width[seed, best, depth]
            assert targets.tolerance[seed, depth] == labels.tolerance[seed, best, depth]
            expected = bool(labels.valid[seed]) and scores[best] > 0.1
            assert bool(targets.mask[seed, depth]) == expected
            supervised += expected
    assert 0 < supervised < SEEDS * DEPTHS


def test_the_score_scale_is_a_named_option():
    labels = matched()
    labels.view_friction = labels.view_friction.clamp(max=0.5)  # the worst level is absent
    labels.friction = labels.friction.clamp(max=0.5)
    batch = head().encode_targets(labels).score
    fixed = head(score_target="log_friction_fixed_max").encode_targets(labels).score
    positive = batch > 0
    assert positive.any()
    assert torch.allclose(fixed[positive], batch[positive] + math.log(1.0 / 0.5), atol=1e-6)
    other = head(score_target="log_friction_fixed_max", max_friction=2.0).encode_targets(labels)
    assert torch.allclose(other.score[positive], batch[positive] + math.log(2.0 / 0.5), atol=1e-6)
    assert float(worst_friction(labels, "log_friction_batch_max", 1.0)) == 0.5
    unusable = log_friction_score(
        torch.tensor([0.5, 0.5, -1.0, 0.0]),
        torch.tensor([0.05, 0.2, 0.05, 0.05]),
        torch.tensor(1.0),
        0.1,
    )
    assert unusable.tolist() == pytest.approx([math.log(2.0), 0.0, 0.0, 0.0])


# ------------------------------------------------------------------------------ loss


def test_losses_are_the_reference_terms_on_the_supervised_seed_depth_pairs():
    stage = head().train()
    labels = matched()
    raw = stage(groups())
    losses = stage.loss(raw, labels)
    targets = stage.encode_targets(labels)
    total = {"score": 0.0, "angle": 0.0, "width": 0.0, "tolerance": 0.0}
    hits = [0, 0, 0]
    count = 0
    logits = raw.angle.detach().double()
    for seed in range(SEEDS):
        for depth in range(DEPTHS):
            if not targets.mask[seed, depth]:
                continue
            count += 1
            angle = int(targets.angle[seed, depth])
            total["score"] += huber(
                float(raw.score[seed, angle, depth] - targets.score[seed, depth])
            )
            total["angle"] += float(
                torch.logsumexp(logits[seed, :, depth], 0) - logits[seed, angle, depth]
            )
            total["width"] += huber(
                float(raw.width[seed, angle, depth] - targets.width[seed, depth]) / MAX_WIDTH
            )
            total["tolerance"] += huber(
                float(raw.tolerance[seed, angle, depth] - targets.tolerance[seed, depth])
                / MAX_TOLERANCE
            )
            error = abs(int(logits[seed, :, depth].argmax()) - angle)
            for bins in range(3):
                hits[bins] += min(error, ANGLES - error) <= bins
    assert count > 3
    for name, value in total.items():  # the reference weights the four grasp terms by 0.2
        assert float(losses[f"loss/{name}"]) == pytest.approx(0.2 * value / count, rel=1e-5)
    for bins in range(3):
        assert float(losses[f"metric/angle_accuracy_within_{bins}"]) == pytest.approx(
            hits[bins] / count
        )
    assert float(losses["metric/supervised_depths"]) == pytest.approx(count / (SEEDS * DEPTHS))
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    assert all(p.grad is not None for p in stage.parameters())


def test_loss_weights_and_primitives_are_configurable_and_empty_selections_are_zero():
    labels = matched()
    stage = head(score_loss={"type": "L1", "weight": 3.0}, angle_loss={"type": "CrossEntropy"})
    raw = stage(groups())
    assert stage.score_loss.weight == 3.0 and stage.angle_loss.weight == 1.0
    assert stage.width_loss.weight == stage.tolerance_loss.weight == 0.2
    default = head().loss(raw, labels)
    custom = stage.loss(raw, labels)
    assert float(custom["loss/angle"]) == pytest.approx(
        5.0 * float(default["loss/angle"]), rel=1e-5
    )
    assert float(custom["loss/width"]) == pytest.approx(float(default["loss/width"]), rel=1e-6)
    labels.valid = torch.zeros(SEEDS, dtype=torch.bool)
    empty = stage.loss(raw, labels)
    assert all(float(empty[f"loss/{k}"]) == 0.0 for k in ("score", "angle", "width", "tolerance"))
    assert all(torch.isfinite(value) for value in empty.values())


# ---------------------------------------------------------------------------- decode


def decode_inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    template = ViewTemplate.fibonacci(VIEWS)
    index = torch.randint(0, VIEWS, (SEEDS,), generator=generator)
    approach = ApproachSet(template.approach(index), template.rotation(index), index)
    points = PointSet(torch.rand(SEEDS, 3, generator=generator))
    scores = {"objectness": torch.rand(SEEDS, generator=generator)}
    shape = (SEEDS, ANGLES, DEPTHS)
    raw = OperationPredictions(
        score=torch.randn(shape, generator=generator),
        angle=torch.randn(shape, generator=generator),
        width=torch.randn(shape, generator=generator) * 0.08,
        tolerance=torch.rand(shape, generator=generator) * 0.05,
    )
    return raw, SeedSet(points, scores=scores), approach


def test_decode_takes_the_best_angle_class_per_depth_and_then_the_best_depth():
    stage = head()
    raw, seeds, approach = decode_inputs()
    grasps = stage.decode(raw, seeds, approach).validate()
    assert len(grasps) == SEEDS and grasps.score_kind == "predicted_quality_x_tolerance"
    assert torch.equal(grasps.translation, seeds.points.coords)
    for seed in range(SEEDS):
        angles = [int(raw.angle[seed, :, depth].argmax()) for depth in range(DEPTHS)]
        candidates = [float(raw.score[seed, angles[depth], depth]) for depth in range(DEPTHS)]
        depth = int(np.argmax(candidates))
        angle = angles[depth]
        tolerance = float(raw.tolerance[seed, angle, depth])
        assert float(grasps.score[seed]) == pytest.approx(
            candidates[depth] * tolerance / MAX_TOLERANCE, rel=1e-5, abs=1e-7
        )
        assert float(grasps.depth[seed]) == pytest.approx(0.01 * (depth + 1))
        width = min(max(1.2 * float(raw.width[seed, angle, depth]), 0.0), MAX_WIDTH)
        assert float(grasps.width[seed]) == pytest.approx(width, abs=1e-7)
        rotation = approach_to_rotation(
            approach.direction[seed], torch.tensor(angle * math.pi / ANGLES)
        )
        assert torch.allclose(grasps.rotation[seed], rotation, atol=1e-6)
    assert (grasps.width == 0).any() and (grasps.width == MAX_WIDTH).any()  # both clamps occur


def test_decoded_score_and_width_gain_are_named_options():
    raw, seeds, approach = decode_inputs()
    reference = head().decode(raw, seeds, approach)
    plain = head(score_decode="score", width_decode_gain=1.0).decode(raw, seeds, approach)
    assert plain.score_kind == "predicted_quality"
    assert torch.equal(plain.depth, reference.depth)  # the choice of cell does not change
    free = (reference.width > 0) & (reference.width < MAX_WIDTH)
    assert torch.allclose(reference.width[free], 1.2 * plain.width[free], atol=1e-7)
    chosen = raw.score.gather(1, raw.angle.argmax(dim=1, keepdim=True)).squeeze(1).amax(dim=-1)
    assert torch.equal(plain.score, chosen)


# ---------------------------------------------------------------------------- misuse


def test_contract_violations_are_explained():
    stage = head()
    flat = GroupFeatures(torch.randn(SEEDS, CHANNELS), ("seed", "channel"))
    with pytest.raises(ContractError, match=r"expects features with axes \('seed', 'depth', 'ch"):
        stage(flat)
    with pytest.raises(ContractError, match="built for 8-channel crop features, got 5"):
        stage(GroupFeatures(torch.randn(SEEDS, DEPTHS, 5), stage.in_axes))
    with pytest.raises(ContractError, match=r"num_depth = 3, but the crop\s+features have 4"):
        stage(GroupFeatures(torch.randn(SEEDS, 4, CHANNELS), stage.in_axes))
    raw = stage(groups())
    with pytest.raises(ContractError, match="supervises a grasp tolerance.*prepare"):
        stage.loss(raw, matched(tolerance=False))
    other_grid = HEADS.build(
        {"type": "OperationHead", "num_angle": 4, "num_depth": DEPTHS},
        in_channels=CHANNELS,
        gripper=Gripper(),
    )
    with pytest.raises(ContractError, match=r"grid of\s+\(5, 3\), but the head is configured"):
        other_grid.encode_targets(matched())
    _, _, approach = decode_inputs()
    with pytest.raises(ContractError, match="do not line up"):
        stage.decode(raw, SeedSet(PointSet(torch.rand(3, 3))), approach)


def test_defaults_are_the_reference_head_and_bad_values_are_rejected():
    stage = HEADS.build({"type": "OperationHead"}, in_channels=256, gripper=Gripper())
    assert stage.codec.grid == (12, 4) and stage.codec.width_decode_gain == 1.2
    assert stage.operation.hidden1.out_features == 128 and stage.operation.output.out_features == 36
    assert stage.tolerance.output.out_features == 12 and stage.max_tolerance == 0.05
    assert stage.min_score_target == 0.1 and stage.score_target == "log_friction_batch_max"
    for loss in (stage.score_loss, stage.angle_loss, stage.width_loss, stage.tolerance_loss):
        assert loss.weight == 0.2
    assert stage.score_loss.beta == 1.0  # the reference's Huber loss with delta 1
    with pytest.raises(BuildError, match="needs in_channels, gripper"):
        HEADS.build({"type": "OperationHead"})
    with pytest.raises(BuildError, match="gripper must be a Gripper"):
        HEADS.build({"type": "OperationHead"}, in_channels=8, gripper={"max_width": 0.1})
    for bad, message in (
        ({"score_target": "log_friction_batch_norm"}, "unknown score_target"),
        ({"max_friction": 0.0}, "max_friction must be positive"),
        ({"score_decode": "tolerance"}, "unknown score_decode"),
        ({"max_tolerance": 0.0}, "must be positive"),
        ({"num_angle": 0}, "num_angle and num_depth must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            HEADS.build({"type": "OperationHead", **bad}, in_channels=8, gripper=Gripper())
