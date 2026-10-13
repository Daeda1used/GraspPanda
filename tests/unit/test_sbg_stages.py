"""What Scale-Balanced-Grasp adds to the stages of GraspNet-baseline, each against the
definition written out in loops: crops at several radii with a gated seed residual,
scale-balanced seed weights, the graspable target, weighted losses and the decoded score."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.assigners.scale_balance import (
    HISTOGRAM_KEY,
    ScaleBalancedAssigner,
    best_grasp_width,
    bin_weights,
    read_histogram,
    scale_balanced_weights,
)
from grasppanda.models.heads.operation import OperationPredictions
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import ASSIGNERS, GROUPERS, HEADS, LOSSES, VIEWS, BuildError
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    Gripper,
    GroupFeatures,
    ObjectGraspLabels,
    PointFeatures,
    PointSet,
    SeedSet,
    Supervision,
)

SEEDS, VIEWS_, ANGLES, DEPTHS = 14, 6, 4, 3
EDGES = torch.tensor([0.01, 0.03, 0.05, 0.07, 0.09], dtype=torch.float64)
COUNTS = torch.tensor([10, 40, 80, 20])


def labels(seed: int = 0, **changes) -> MatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    levels = torch.tensor([-1.0, 0.0, 0.1, 0.3, 0.6, 1.0])
    shape = (SEEDS, VIEWS_, ANGLES, DEPTHS)
    view_friction = levels[torch.randint(0, len(levels), shape, generator=generator)]
    view_friction[0] = -1.0  # a label point without any usable grasp
    view_width = torch.rand(shape, generator=generator) * 0.12
    on_object = torch.rand(SEEDS, generator=generator) < 0.7
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
        "tolerance": torch.rand(SEEDS, ANGLES, DEPTHS, generator=generator) * 0.05,
        "weight": torch.rand(SEEDS, generator=generator) + 0.5,
    }
    return MatchedLabels(**{**fields, **changes})


def huber(error: float) -> float:
    return 0.5 * error * error if abs(error) <= 1 else abs(error) - 0.5


# ------------------------------------------------------------ crops at several radii

RADIUS, HMIN, HMAX, NSAMPLE, SCALES, CHANNELS = 0.4, -0.1, (0.1, 0.3), 5, (0.5, 1.0), 6


def grouper(**overrides):
    torch.manual_seed(0)
    config = {
        "type": "DepthCylinderGroup",
        "radius": RADIUS,
        "hmin": HMIN,
        "hmax": list(HMAX),
        "nsample": NSAMPLE,
        "mlp_channels": [5, CHANNELS],
        "radius_scales": list(SCALES),
        "seed_residual": "gated",
    }
    return GROUPERS.build({**config, **overrides}, in_channels=CHANNELS)


def scene(seeds: int = 8, points: int = 80):
    generator = torch.Generator().manual_seed(0)
    cloud = PointSet.from_dense(torch.rand(1, points, 3, generator=generator))
    support = PointFeatures(torch.zeros(points, 0), cloud)
    centres = torch.rand(seeds, 3, generator=generator)
    rotation = approach_to_rotation(
        torch.randn(seeds, 3, generator=generator), torch.rand(seeds, generator=generator)
    )
    return support, centres, rotation, torch.randn(seeds, CHANNELS, generator=generator)


def test_every_radius_has_its_own_cylinders_and_its_own_mlp():
    stage = grouper()
    support, centres, rotation, _ = scene()
    neighbours = stage.neighbours(support, centres, rotation)
    grouped = stage.group(support, centres, rotation)
    assert neighbours.shape == (1, 8, 2, 2, NSAMPLE) and grouped.shape == (8, 2, 2, NSAMPLE, 3)
    assert stage.radii == (0.2, 0.4) and len(stage.scales) == 2
    coords = support.points.coords.double().numpy()
    grew = False
    for seed in range(8):
        local = (coords - centres[seed].double().numpy()) @ rotation[seed].double().numpy()
        for scale, radius in enumerate(stage.radii):
            for depth, hmax in enumerate(HMAX):
                inside = (
                    (local[:, 1] ** 2 + local[:, 2] ** 2 < radius**2)
                    & (local[:, 0] > HMIN)
                    & (local[:, 0] < hmax)
                )
                members = np.flatnonzero(inside)[:NSAMPLE]
                index = np.full(NSAMPLE, members[0] if len(members) else 0)
                index[: len(members)] = members
                assert neighbours[0, seed, scale, depth].tolist() == index.tolist()
                assert np.allclose(grouped[seed, scale, depth].numpy(), local[index], atol=1e-5)
                grew |= scale == 1 and inside.sum() > 0
    assert grew
    assert stage.scales[0][0].linear.weight is not stage.scales[1][0].linear.weight
    normalised = grouper(normalize_offsets=True).group(support, centres, rotation)
    assert torch.allclose(normalised[:, 0] * 0.2, grouped[:, 0], atol=1e-6)  # by its own radius
    assert torch.allclose(normalised[:, 1] * 0.4, grouped[:, 1], atol=1e-6)


def test_radii_are_fused_and_the_seed_feature_comes_back_through_a_gate():
    stage = grouper().eval()
    support, centres, rotation, seed_feats = scene()
    out = stage(support, centres, rotation, seed_feats)
    assert out.axes == ("seed", "depth", "channel") and out.feats.shape == (8, 2, CHANNELS)
    assert stage.uses_seed_features and stage.fuse.in_features == 2 * CHANNELS
    grouped = stage.group(support, centres, rotation)
    pooled = []
    for scale, mlp in enumerate(stage.scales):  # smallest radius first
        hidden = mlp(grouped[:, scale].reshape(-1, 3)).reshape(8, 2, NSAMPLE, CHANNELS)
        pooled.append(hidden.amax(dim=2))
    fused = stage.fuse(torch.cat(pooled, dim=-1))
    residual = torch.sigmoid(stage.gate(seed_feats)) * seed_feats  # the same for every depth
    assert torch.allclose(out.feats, fused + residual[:, None, :], atol=1e-5)
    out.feats.sum().backward()
    assert all(p.grad is not None for p in stage.parameters())


def test_seed_features_are_required_exactly_when_they_are_used():
    support, centres, rotation, seed_feats = scene()
    gated = grouper()
    with pytest.raises(ContractError, match=r"seed_residual: gated needs the\s+\(8, 6\) features"):
        gated(support, centres, rotation)
    with pytest.raises(ContractError, match=r"got \(8, 3\)"):
        gated(support, centres, rotation, seed_feats[:, :3])
    plain = grouper(seed_residual="none")
    assert not plain.uses_seed_features and plain.gate is None and plain.fuse is not None
    assert plain(support, centres, rotation).feats.shape == (8, 2, CHANNELS)
    with pytest.raises(ContractError, match="seed_residual: none and would ignore them"):
        plain(support, centres, rotation, seed_feats)
    with pytest.raises(ValueError, match="adds 9-channel seed\\s+features to 6-channel crop"):
        GROUPERS.build(
            {"type": "DepthCylinderGroup", "mlp_channels": [5, 6], "seed_residual": "gated"},
            in_channels=9,
        )
    with pytest.raises(BuildError, match="needs in_channels"):
        GROUPERS.build({"type": "DepthCylinderGroup"})
    for bad in ({"radius_scales": []}, {"radius_scales": [0.5, 0.0]}):
        with pytest.raises(ValueError, match="positive radius_scales"):
            grouper(**bad)
    with pytest.raises(ValueError, match="unknown seed_residual 'sum'"):
        grouper(seed_residual="sum")


# ----------------------------------------------------------- scale-balanced weights


def test_bin_weights_are_one_minus_the_log_of_the_relative_count():
    weights = bin_weights(COUNTS)
    expected = [1 - math.log(count / 80) for count in COUNTS.tolist()]
    assert weights.tolist() == pytest.approx(expected, rel=1e-6)
    assert float(weights.min()) == 1.0  # the most frequent scale
    with pytest.raises(ContractError, match="positive count in every bin"):
        bin_weights(torch.tensor([3, 0, 5]))


def test_seed_weight_is_the_weight_of_the_bin_of_its_best_grasp():
    matched = labels()
    width = best_grasp_width(matched.view_friction, matched.view_width, 0.1)
    weight = scale_balanced_weights(matched.view_friction, matched.view_width, 0.1, EDGES, COUNTS)
    table = bin_weights(COUNTS)
    friction = matched.view_friction.reshape(SEEDS, -1)
    widths = matched.view_width.reshape(SEEDS, -1)
    edges = EDGES.float()
    bins = set()
    for seed in range(SEEDS):
        best, best_score = 0, 0.0  # without a usable grasp: the first cell
        for cell in range(friction.shape[1]):
            mu, w = float(friction[seed, cell]), float(widths[seed, cell])
            score = math.log(1.0 / mu) if mu > 0 and w <= 0.1 else 0.0
            if score > best_score:  # the first of equal maxima
                best, best_score = cell, score
        assert float(width[seed]) == float(widths[seed, best]), seed
        index = 0  # a width in no bin falls into bin 0
        for k in range(len(COUNTS)):
            if edges[k] < widths[seed, best] < edges[k + 1]:
                index = k
        bins.add(index)
        assert float(weight[seed]) == float(table[index]), seed
    assert float(width[0]) == float(widths[0, 0])  # the seed without a usable grasp
    assert len(bins) >= 3


def test_widths_on_an_edge_or_outside_the_histogram_take_bin_zero():
    friction = torch.full((5, 1, 1, 1), 0.2)
    width = torch.tensor([0.005, 0.03, 0.0300001, 0.095, 0.04]).reshape(5, 1, 1, 1)
    weight = scale_balanced_weights(friction, width, 0.1, EDGES, COUNTS)
    table = bin_weights(COUNTS)
    # below the range, exactly on an edge, just above it, above the range, inside bin 1
    assert weight.tolist() == [table[0], table[0], table[1], table[0], table[1]]


def make_batch(template):
    generator = torch.Generator().manual_seed(0)
    scene_points = PointSet.from_dense(torch.rand(1, 30, 3, generator=generator))
    keep = torch.arange(0, 30, 3)
    seeds = SeedSet(scene_points.subset(keep, torch.tensor([0, 10])))
    index = torch.randint(0, VIEWS_, (10,), generator=generator)
    approach = ApproachSet(template.approach(index), template.rotation(index), index)
    shape = (12, VIEWS_, ANGLES, DEPTHS)
    obj = ObjectGraspLabels(
        object_id=3,
        pose=torch.eye(4),
        points=torch.rand(12, 3, generator=generator),
        scores=torch.rand(shape, generator=generator),
        widths=torch.rand(shape, generator=generator) * 0.1,
    )
    histogram = {"edges": EDGES, "counts": COUNTS}
    supervision = Supervision(
        objects=[[obj]],
        point_labels={"objectness": torch.ones(30, dtype=torch.long)},
        extras={HISTOGRAM_KEY: [histogram]},
    )
    return seeds, approach, supervision


def test_the_assigner_returns_the_weights_with_the_matched_labels():
    template = ViewTemplate.fibonacci(VIEWS_)
    assigner = ASSIGNERS.build(
        {"type": "ScaleBalancedAssigner"}, template=template, gripper=Gripper()
    )
    assert isinstance(assigner, ScaleBalancedAssigner)
    assert (assigner.crop_centre, assigner.supervise, assigner.view_widths) == (
        "label_point",
        "object_seeds",
        True,
    )
    seeds, approach, supervision = make_batch(template)
    matched = assigner(seeds, approach, supervision)
    expected = scale_balanced_weights(matched.view_friction, matched.view_width, 0.1, EDGES, COUNTS)
    assert matched.weight.shape == (10,) and torch.equal(matched.weight, expected)
    assert matched.view_width is not None and bool(matched.on_object.all())
    narrow = ASSIGNERS.build(
        {"type": "ScaleBalancedAssigner"}, template=template, gripper=Gripper(max_width=0.02)
    )
    assert not torch.equal(narrow(seeds, approach, supervision).weight, matched.weight)


def test_the_histogram_must_come_with_the_batch():
    template = ViewTemplate.fibonacci(VIEWS_)
    assigner = ScaleBalancedAssigner(template=template, gripper=Gripper())
    seeds, approach, supervision = make_batch(template)
    bare = Supervision(supervision.objects, supervision.point_labels)
    with pytest.raises(ContractError, match="scale_statistic.*prepare"):
        assigner(seeds, approach, bare)
    other = {"edges": EDGES, "counts": COUNTS + 1}
    with pytest.raises(ContractError, match="carry different scale histograms"):
        read_histogram(
            Supervision(
                extras={HISTOGRAM_KEY: [other, other, supervision.extras[HISTOGRAM_KEY][0]]}
            ),
            user="x",
        )
    with pytest.raises(ContractError, match="K \\+ 1 increasing edges"):
        read_histogram(
            Supervision(extras={HISTOGRAM_KEY: [{"edges": EDGES[:-1], "counts": COUNTS}]}), user="x"
        )
    with pytest.raises(BuildError, match="needs template, gripper"):
        ASSIGNERS.build({"type": "ScaleBalancedAssigner"})
    with pytest.raises(BuildError, match="gripper must be a Gripper"):
        ASSIGNERS.build({"type": "ScaleBalancedAssigner"}, template=template, gripper=None)


# ------------------------------------------------------------------ weighted losses


def test_loss_primitives_take_a_weighted_mean():
    loss = LOSSES.build({"type": "SmoothL1", "weight": 2.0})
    prediction, target = torch.tensor([0.0, 1.0, 3.0, 5.0]), torch.zeros(4)
    mask = torch.tensor([True, True, True, False])
    weights = torch.tensor([1.0, 3.0, 0.5, 100.0])
    expected = 2.0 * (1 * 0.0 + 3 * 0.5 + 0.5 * 2.5) / 4.5
    assert float(loss(prediction, target, mask, weights)) == pytest.approx(expected)
    assert float(loss(prediction, target, mask)) == pytest.approx(2.0 * (0.0 + 0.5 + 2.5) / 3)
    assert float(loss(prediction, target, None, torch.zeros(4))) == 0.0  # nothing weighted
    with pytest.raises(ContractError, match="weights .* does not match the penalty"):
        loss(prediction, target, mask, weights[:2])
    entropy = LOSSES.build({"type": "CrossEntropy"})
    logits, classes = torch.randn(5, 3), torch.tensor([0, 1, 2, 1, 0])
    row = F.cross_entropy(logits, classes, reduction="none")
    weight = torch.tensor([1.0, 2.0, 0.0, 4.0, 0.5])
    assert float(entropy(logits, classes, None, weight)) == pytest.approx(
        float((row * weight).sum() / weight.sum()), rel=1e-6
    )


def view_stage(**overrides):
    torch.manual_seed(0)
    config = {"type": "ObjectnessViewStage", "num_views": VIEWS_}
    return VIEWS.build({**config, **overrides}, in_channels=5, gripper=Gripper())


def test_graspable_seeds_need_enough_graspable_views():
    matched = labels()
    plain, strict = view_stage(), view_stage(min_graspable_views=2, graspable_score=0.6)
    base, targets = plain.encode_targets(matched), strict.encode_targets(matched)
    assert torch.equal(base.objectness, matched.on_object.long())
    views = (targets.view > 0.6).sum(dim=-1)
    assert 0 < int((views > 2).sum()) < SEEDS  # the rule separates the seeds of this batch
    graspable = matched.on_object & (views > 2)
    assert torch.equal(targets.objectness, graspable.long())
    assert torch.equal(targets.view_mask, graspable) and torch.equal(targets.view, base.view)
    assert int(targets.objectness.sum()) < int(base.objectness.sum())
    with pytest.raises(ValueError, match="min_graspable_views must be null or at least 0"):
        view_stage(min_graspable_views=-1)


def test_view_loss_is_weighted_by_the_seed_weights():
    stage = view_stage(min_graspable_views=2, graspable_score=0.6).train()
    matched = labels()
    approach, _ = stage(SeedSet(PointSet(torch.rand(SEEDS, 3)), torch.randn(SEEDS, 5)))
    losses = stage.loss(approach, matched)
    targets = stage.encode_targets(matched)
    squared = (approach.logits - targets.view).square().sum(dim=1)  # per seed, all views
    weight = matched.weight * targets.view_mask
    expected = (squared * weight).sum() / (weight.sum() * VIEWS_)
    assert float(losses["loss/view"]) == pytest.approx(float(expected), rel=1e-5)
    unweighted = stage.loss(approach, labels(weight=None))
    assert float(unweighted["loss/view"]) != pytest.approx(float(expected), rel=1e-3)
    assert float(losses["loss/objectness"]) == pytest.approx(
        float(F.cross_entropy(approach.objectness, targets.objectness)), rel=1e-6
    )  # the objectness loss is not weighted


def head(**overrides):
    torch.manual_seed(0)
    config = {
        "type": "OperationHead",
        "num_angle": ANGLES,
        "num_depth": DEPTHS,
        "hidden_channels": 6,
    }
    return HEADS.build({**config, **overrides}, in_channels=8, gripper=Gripper())


def test_head_losses_are_weighted_and_the_score_covers_every_depth_of_a_supervised_seed():
    stage = head(score_supervision="seeds_with_a_supervised_depth").train()
    matched = labels()
    raw = stage(GroupFeatures(torch.randn(SEEDS, DEPTHS, 8), stage.in_axes))
    losses = stage.loss(raw, matched)
    targets = stage.encode_targets(matched)
    assert torch.equal(targets.score_mask, targets.mask.any(dim=1, keepdim=True).expand(-1, DEPTHS))
    assert int(targets.score_mask.sum()) > int(targets.mask.sum()) > 0
    total = {"score": 0.0, "width": 0.0, "angle": 0.0}
    norm = {"score": 0.0, "width": 0.0, "angle": 0.0}
    for seed in range(SEEDS):
        weight = float(matched.weight[seed])
        for depth in range(DEPTHS):
            angle = int(targets.angle[seed, depth])
            if targets.score_mask[seed, depth]:
                error = float(raw.score[seed, angle, depth] - targets.score[seed, depth])
                total["score"] += weight * huber(error)
                norm["score"] += weight
            if targets.mask[seed, depth]:
                error = float(raw.width[seed, angle, depth] - targets.width[seed, depth]) / 0.1
                total["width"] += weight * huber(error)
                logits = raw.angle[seed, :, depth].detach().double()
                total["angle"] += weight * float(torch.logsumexp(logits, 0) - logits[angle])
                norm["width"] += weight
                norm["angle"] += weight
    for name in total:
        assert float(losses[f"loss/{name}"]) == pytest.approx(
            0.2 * total[name] / norm[name], rel=1e-5
        )
    baseline = head().encode_targets(matched)
    assert torch.equal(baseline.score_mask, baseline.mask)  # the default supervises alike
    with pytest.raises(ValueError, match="unknown score_supervision 'all'"):
        head(score_supervision="all")


def test_decoded_score_can_include_the_objectness_probability():
    generator = torch.Generator().manual_seed(0)
    template = ViewTemplate.fibonacci(VIEWS_)
    index = torch.randint(0, VIEWS_, (SEEDS,), generator=generator)
    approach = ApproachSet(template.approach(index), template.rotation(index), index)
    shape = (SEEDS, ANGLES, DEPTHS)
    raw = OperationPredictions(
        score=torch.randn(shape, generator=generator),
        angle=torch.randn(shape, generator=generator),
        width=torch.rand(shape, generator=generator) * 0.08,
        tolerance=torch.rand(shape, generator=generator) * 0.05,
    )
    probability = torch.rand(SEEDS, generator=generator)
    points = PointSet(torch.rand(SEEDS, 3, generator=generator))
    seeds = SeedSet(points, scores={"objectness": probability})
    with_tolerance = head().decode(raw, seeds, approach)
    stage = head(score_decode="score_x_objectness_x_tolerance")
    grasps = stage.decode(raw, seeds, approach)
    assert grasps.score_kind == "predicted_quality_x_objectness_x_tolerance"
    assert torch.equal(grasps.depth, with_tolerance.depth)  # the chosen cell is the same
    plain = head(score_decode="score").decode(raw, seeds, approach)
    cell_tolerance = with_tolerance.score / plain.score * 0.05
    expected = plain.score * probability * cell_tolerance / 0.05
    assert torch.allclose(grasps.score, expected, atol=1e-6)
    with pytest.raises(ContractError, match="carry no 'objectness' score"):
        stage.decode(raw, SeedSet(points), approach)
