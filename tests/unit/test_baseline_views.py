"""SubsampledSeeds and ObjectnessViewStage: what they emit, their targets against the
definition written out in loops, their losses, and the inputs they must refuse."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.losses.squared import MSE
from grasppanda.models.proposals.subsampled import SubsampledSeeds
from grasppanda.models.views.objectness_view import ObjectnessApproach, ObjectnessViewStage
from grasppanda.registry import LOSSES, PROPOSALS, VIEWS, BuildError
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    Gripper,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    SeedSet,
    Supervision,
)

SEEDS, VIEW_COUNT, ANGLES, DEPTHS, CHANNELS = 16, 7, 4, 3, 6


def stage(**overrides) -> ObjectnessViewStage:
    torch.manual_seed(0)
    config = {"type": "ObjectnessViewStage", "num_views": VIEW_COUNT}
    return VIEWS.build({**config, **overrides}, in_channels=CHANNELS, gripper=Gripper())


def seeds(seed: int = 0) -> SeedSet:
    generator = torch.Generator().manual_seed(seed)
    points = PointSet(torch.rand(SEEDS, 3, generator=generator))
    return SeedSet(points, torch.randn(SEEDS, CHANNELS, generator=generator))


def matched(seed: int = 0, **changes) -> MatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    levels = torch.tensor([-1.0, 0.0, 0.1, 0.3, 0.6, 1.0])
    shape = (SEEDS, VIEW_COUNT, ANGLES, DEPTHS)
    view_friction = levels[torch.randint(0, len(levels), shape, generator=generator)]
    view_width = torch.rand(shape, generator=generator) * 0.13  # some wider than the gripper
    on_object = torch.rand(SEEDS, generator=generator) < 0.6
    fields = {
        "valid": on_object,
        "label_point": torch.zeros(SEEDS, 3),
        "distance": torch.zeros(SEEDS),
        "object_index": torch.zeros(SEEDS, dtype=torch.long),
        "crop_centre": torch.zeros(SEEDS, 3),
        "crop_rotation": torch.eye(3).expand(SEEDS, 3, 3),
        "view_friction": view_friction,
        "friction": view_friction[:, 0],
        "width": view_width[:, 0],
        "on_object": on_object,
        "view_width": view_width,
    }
    return MatchedLabels(**{**fields, **changes})


# ------------------------------------------------------------------ SubsampledSeeds


def backbone_output(per_frame: int = 8, frames: int = 2):
    scene = PointSet.from_dense(torch.rand(frames, 4 * per_frame, 3))
    keep = torch.arange(0, len(scene), 4)
    offsets = torch.arange(frames + 1) * per_frame
    feats = PointFeatures(torch.randn(len(keep), CHANNELS), scene.subset(keep, offsets))
    observation = Observation(
        keys=[SampleKey("unit", 0, "none", f) for f in range(frames)], points=scene
    )
    return feats, observation


def test_the_backbone_subsample_becomes_the_seed_set_unchanged():
    proposal = PROPOSALS.build({"type": "SubsampledSeeds", "num_seeds": 8}, in_channels=CHANNELS)
    feats, observation = backbone_output()
    found, scores = proposal(feats, observation)
    assert scores is None and proposal.loss(scores, Supervision()) == {}
    assert found.points is feats.points and found.feats is feats.feats
    assert bool(found.valid.all()) and found.source == "backbone_subsample"
    assert found.scores == {} and proposal.requires == "subsampled"
    assert list(proposal.parameters()) == []


def test_subsampled_seeds_refuses_dense_features_and_other_seed_counts():
    proposal = SubsampledSeeds(num_seeds=8)
    feats, observation = backbone_output()
    dense = PointFeatures(torch.randn(len(observation.points), CHANNELS), observation.points)
    with pytest.raises(ContractError, match="requires 'subsampled' scene features.*dense features"):
        proposal(dense, observation)
    with pytest.raises(ContractError, match="needs features on a subsample of point set"):
        proposal(PointFeatures(feats.feats, PointSet(feats.points.coords)), observation)
    with pytest.raises(ContractError, match=r"configured for 6 seeds per frame.*\[8, 8\]"):
        SubsampledSeeds(num_seeds=6)(feats, observation)
    with pytest.raises(ContractError, match="needs observation.points"):
        proposal(feats, Observation(keys=observation.keys))
    with pytest.raises(ValueError, match="num_seeds must be positive"):
        SubsampledSeeds(num_seeds=0)


# -------------------------------------------------------------------------- forward


def test_forward_scores_objectness_and_views_with_one_head():
    view = stage().eval()
    given = seeds()
    approach, returned = view(given)
    assert isinstance(approach, ObjectnessApproach)
    assert approach.objectness.shape == (SEEDS, 2) and approach.logits.shape == (SEEDS, VIEW_COUNT)
    hidden = F.relu(view.norm1(view.hidden1(given.feats)))
    hidden = F.relu(view.norm2(view.hidden2(hidden)))
    output = view.output(hidden)  # rows 0-1: objectness, rows 2…: views
    assert torch.equal(approach.objectness, output[:, :2])
    assert torch.equal(approach.logits, output[:, 2:])
    assert view.hidden2.out_features == view.output.in_features == 2 + VIEW_COUNT
    assert torch.equal(approach.view_index, approach.logits.argmax(dim=-1))
    assert torch.equal(approach.direction, view.template.approach(approach.view_index))
    assert torch.equal(approach.rotation, view.template.rotation(approach.view_index))
    assert returned.feats is given.feats  # no residual: seed features pass through
    probability = approach.objectness.softmax(dim=-1)[:, 1]
    assert torch.allclose(returned.scores["objectness"], probability)
    assert not returned.scores["objectness"].requires_grad


def test_the_best_view_is_chosen_in_training_too_and_a_recorded_choice_can_be_replayed():
    view = stage().train()
    given = seeds()
    first, _ = view(given)
    assert torch.equal(first.view_index, first.logits.argmax(dim=-1))  # no sampling
    replay = torch.arange(SEEDS) % VIEW_COUNT
    replayed, _ = view(given, view_index=replay)
    assert torch.equal(replayed.view_index, replay)
    assert torch.equal(replayed.rotation, view.template.rotation(replay))
    with pytest.raises(ContractError, match=r"one view per seed \(16,\)"):
        view(given, view_index=replay[:5])


def test_background_seeds_are_rejected_in_evaluation_mode_only():
    view = stage()
    given = seeds()
    with torch.no_grad():  # make the decision depend on the sign of the first feature
        view.output.weight[:2].zero_()
        view.output.bias[:2] = torch.tensor([0.0, -1.0])
        view.hidden2.weight.zero_()
        view.hidden2.bias.zero_()
    _, training = view.train()(given)
    assert training.valid is None  # nothing is rejected while training
    _, evaluated = view.eval()(given)
    assert not evaluated.valid.any()  # objectness logits (0, -1): every seed is background
    with torch.no_grad():
        view.output.bias[:2] = torch.tensor([0.0, 1.0])
    _, kept = view(given)
    assert kept.valid.all()
    padded = SeedSet(given.points, given.feats, valid=torch.arange(SEEDS) < 4)
    assert view(padded)[1].valid.sum() == 4  # an earlier invalid mark is never lifted
    _, unfiltered = stage(reject_background=False).eval()(given)
    assert unfiltered.valid is None


# -------------------------------------------------------------------------- targets


def test_view_targets_are_the_best_usable_grasp_score_of_every_view():
    view = stage()
    labels = matched()
    targets = view.encode_targets(labels)
    friction, width = labels.view_friction, labels.view_width
    worst = float(friction.max())
    for seed in range(SEEDS):
        for index in range(VIEW_COUNT):
            best = 0.0
            for angle in range(ANGLES):
                for depth in range(DEPTHS):
                    mu = float(friction[seed, index, angle, depth])
                    if mu > 0 and float(width[seed, index, angle, depth]) <= 0.1:
                        best = max(best, math.log(worst / mu))
            assert float(targets.view[seed, index]) == pytest.approx(best, abs=1e-6)
    assert torch.equal(targets.objectness, labels.on_object.long())
    assert torch.equal(targets.view_mask, labels.valid)
    narrow = VIEWS.build(
        {"type": "ObjectnessViewStage", "num_views": VIEW_COUNT},
        in_channels=CHANNELS,
        gripper=Gripper(max_width=0.05),
    )
    assert (narrow.encode_targets(labels).view <= targets.view).all()
    assert (narrow.encode_targets(labels).view < targets.view).any()  # the gripper is injected


def test_loss_is_cross_entropy_on_all_seeds_and_squared_error_on_object_seeds():
    view = stage().train()
    labels = matched()
    approach, _ = view(seeds())
    losses = view.loss(approach, labels)
    targets = view.encode_targets(labels)
    objectness = F.cross_entropy(approach.objectness, labels.on_object.long())
    squared = (approach.logits - targets.view).square()[labels.on_object].mean()
    assert float(losses["loss/objectness"]) == pytest.approx(float(objectness), rel=1e-6)
    assert float(losses["loss/view"]) == pytest.approx(float(squared), rel=1e-6)
    predicted = approach.objectness.argmax(dim=-1) == 1
    assert float(losses["metric/objectness_accuracy"]) == pytest.approx(
        float((predicted == labels.on_object).float().mean())
    )
    good = (approach.logits >= 0.7) & labels.on_object[:, None]
    assert float(losses["metric/good_view_predictions"]) == float(good.sum())
    (losses["loss/objectness"] + losses["loss/view"]).backward()
    assert all(p.grad is not None for p in view.parameters())
    assert isinstance(view.view_loss, MSE) and view.view_loss.weight == 1.0


def test_losses_are_configurable_and_supervision_follows_the_assigner_mask():
    view = stage(view_loss={"type": "SmoothL1", "weight": 2.0}).train()
    approach, _ = view(seeds())
    labels = matched()
    none = matched(valid=torch.zeros(SEEDS, dtype=torch.bool))
    assert float(view.loss(approach, none)["loss/view"]) == 0.0  # nothing supervised: zero, not NaN
    assert float(view.loss(approach, none)["loss/objectness"]) > 0  # objectness still learns
    assert float(view.loss(approach, labels)["loss/view"]) > 0
    squared = LOSSES.build({"type": "MSE", "weight": 3.0})
    values = squared(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))
    assert float(values) == pytest.approx(3.0 * (1.0 + 4.0) / 2)


# --------------------------------------------------------------------------- misuse


def test_contract_violations_are_explained():
    view = stage()
    approach, _ = view(seeds())
    with pytest.raises(ContractError, match=r"needs \(M, 6\) seed features, got \(16, 9\)"):
        view(SeedSet(seeds().points, torch.zeros(SEEDS, 9)))
    with pytest.raises(ContractError, match="supervise: object_seeds.*view_widths: true"):
        view.loss(approach, matched(on_object=None))
    with pytest.raises(ContractError, match="supervise: object_seeds.*view_widths: true"):
        view.loss(approach, matched(view_width=None))
    plain = ApproachSet(approach.direction, approach.rotation, approach.view_index, approach.logits)
    with pytest.raises(ContractError, match="objectness logits of its own forward"):
        view.loss(plain, matched())
    other = VIEWS.build(
        {"type": "ObjectnessViewStage", "num_views": 9}, in_channels=CHANNELS, gripper=Gripper()
    )
    with pytest.raises(ContractError, match="describe 7 views, but the view template has 9"):
        other.encode_targets(matched())


def test_defaults_are_the_reference_stage_and_bad_values_are_rejected():
    view = VIEWS.build({"type": "ObjectnessViewStage"}, in_channels=256, gripper=Gripper())
    assert view.template.num_views == 300 and view.output.out_features == 302
    assert view.reject_background and view.score_target == "log_friction_batch_max"
    assert view.objectness_loss.weight == view.view_loss.weight == 1.0
    assert (
        sum(p.numel() for p in view.parameters())
        == 256 * 257 + 2 * 256 + 302 * 257 + 2 * 302 + 302 * 303
    )
    with pytest.raises(BuildError, match="needs in_channels, gripper"):
        VIEWS.build({"type": "ObjectnessViewStage"})
    with pytest.raises(ValueError, match="unknown score_target"):
        VIEWS.build(
            {"type": "ObjectnessViewStage", "score_target": "graspness"},
            in_channels=8,
            gripper=Gripper(),
        )
