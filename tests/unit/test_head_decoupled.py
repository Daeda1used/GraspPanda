"""DecoupledGridHead: outputs per option, targets from the best cell, loss, decoding, and
what it refuses. Tiny synthetic grids on CPU."""

from __future__ import annotations

import math

import pytest
import torch

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.models.detectors.seed_based import stage_contract_problems
from grasppanda.models.heads.decoupled import DecoupledGridHead
from grasppanda.ops import fibonacci_views
from grasppanda.registry import HEADS
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    Gripper,
    GroupFeatures,
    PointSet,
    SeedSet,
)

ANGLES, DEPTHS, BINS = 3, 2, 5
VARIANTS = [
    {},
    {"depth": "regression"},
    {"width": "class", "width_bins": BINS},
    {"width": "bins", "width_bins": BINS},
    {"depth": "regression", "width": "bins", "width_bins": BINS},
]


def head(channels: int = 8, **overrides) -> DecoupledGridHead:
    torch.manual_seed(0)
    config = {
        "type": "DecoupledGridHead",
        "num_angle": ANGLES,
        "num_depth": DEPTHS,
        "hidden_channels": 6,
        **overrides,
    }
    return HEADS.build(config, in_channels=channels, gripper=Gripper())


def labels(friction: torch.Tensor, width: torch.Tensor, valid: torch.Tensor | None = None):
    count = len(friction)
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


def random_labels(count: int = 6, seed: int = 0) -> MatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    levels = torch.tensor([-1.0, 0.0, 0.2, 0.4, 0.5, 1.0])
    friction = levels[torch.randint(0, 6, (count, ANGLES, DEPTHS), generator=generator)]
    width = torch.rand(count, ANGLES, DEPTHS, generator=generator) * 0.13  # some beyond 0.1 m
    return labels(friction, width)


def crops(count: int = 6, channels: int = 8) -> GroupFeatures:
    feats = torch.randn(count, channels, generator=torch.Generator().manual_seed(1))
    return GroupFeatures(feats, ("seed", "channel"))


def seeds_and_approach(count: int = 6):
    points = PointSet(torch.rand(count, 3, generator=torch.Generator().manual_seed(9)))
    return SeedSet(points=points), ApproachSet(direction=-fibonacci_views(count))


def test_outputs_follow_the_depth_and_width_options():
    raw = head()(crops())
    assert raw.score.shape == (6, ANGLES, DEPTHS) and raw.angle.shape == (6, ANGLES)
    assert raw.depth.shape == (6, DEPTHS) and raw.width.shape == (6, DEPTHS)
    raw = head(depth="regression", width="bins", width_bins=BINS)(crops())
    assert raw.depth.shape == (6,) and raw.width.shape == (6, DEPTHS, BINS)
    bundle = head()
    assert bundle.in_axes == ("seed", "channel") and bundle.label_format == "dense"
    assert bundle.score_kind == "predicted_quality" and bundle.applies_seed_weights
    assert DecoupledGridHead.loss_terms_for({}) == DecoupledGridHead.loss_terms
    assert DecoupledGridHead.loss_terms_for({"depth": "regression", "width": "bins"}) == {
        "score_loss": "unit",
        "angle_loss": "class",
        "depth_loss": "value",
        "width_loss": "bins",
    }
    assert type(head(width="bins").width_loss).__name__ == "DistributionFocal"
    assert type(head(width="class").width_loss).__name__ == "CrossEntropy"
    assert type(head(depth="regression").depth_loss).__name__ == "SmoothL1"


def test_targets_are_the_score_grid_the_best_cell_and_the_widths_at_its_angle():
    friction = torch.tensor(
        [
            [[1.0, 0.5], [0.2, -1.0], [0.4, 0.1]],  # best: angle 2, depth 1 (friction 0.1)
            [[0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],  # no usable grasp
            [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]],  # all equal: the first cell wins
        ]
    )
    width = torch.full((3, ANGLES, DEPTHS), 0.05)
    width[0, 2] = torch.tensor([0.02, 0.08])
    width[2, 0, 1] = 0.2  # wider than the gripper: not usable
    targets = head().encode_targets(labels(friction, width))
    scale = math.log(10.0) + 1e-6
    assert targets.score[0, 2, 1].item() == pytest.approx(math.log(1.0 / 0.1) / scale)
    assert targets.score[0, 0, 0].item() == 0.0 and targets.score[0, 1, 1].item() == 0.0
    assert targets.angle.tolist()[0] == 2 and targets.depth.tolist()[0] == 1
    assert targets.seed_mask.tolist() == [True, False, True]
    assert (targets.angle[2].item(), targets.depth[2].item()) == (0, 0)
    # Widths at the best angle, per depth bin, in regression units (metres x 10).
    assert targets.width[0].tolist() == pytest.approx([0.2, 0.8])
    assert targets.width_mask.tolist() == [[True, True], [False, False], [True, False]]
    assert bool(targets.score_mask.all())
    # As bin positions: a fraction of the gripper opening times (bins - 1).
    binned = head(width="bins", width_bins=BINS).encode_targets(labels(friction, width))
    assert binned.width[0].tolist() == pytest.approx([0.02 / 0.1 * 4, 0.08 / 0.1 * 4])
    # An unsupervised seed supervises nothing.
    valid = torch.tensor([False, True, True])
    masked = head().encode_targets(labels(friction, width, valid))
    assert masked.seed_mask.tolist() == [False, False, True]
    assert not bool(masked.score_mask[0].any()) and float(masked.score[0].abs().sum()) == 0.0
    # The batch-normalised target of SWAD is an option.
    batch = head(score_target="log_friction_batch_norm").encode_targets(labels(friction, width))
    assert batch.score.max().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("options", VARIANTS, ids=lambda o: "+".join(o) or "defaults")
def test_loss_reaches_every_output_and_decode_gives_one_grasp_per_seed(options):
    bundle = head(**options)
    feats = crops()
    feats.feats.requires_grad_(True)
    raw = bundle(feats)
    losses = bundle.loss(raw, random_labels())
    terms = {key: value for key, value in losses.items() if key.startswith("loss/")}
    assert set(terms) == {"loss/score", "loss/angle", "loss/depth", "loss/width"}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert all(float(value) > 0 for value in terms.values()), terms
    sum(terms.values()).backward()
    for layer in (bundle.score, bundle.angle, bundle.depth_output, bundle.width_output):
        assert layer.weight.grad is not None and bool(layer.weight.grad.any())
    assert torch.isfinite(feats.feats.grad).all()
    seeds, approach = seeds_and_approach()
    grasps = bundle.decode(raw, seeds, approach)
    grasps.validate()
    assert len(grasps) == 6 and grasps.score_kind == "predicted_quality"
    assert torch.equal(grasps.translation, seeds.points.coords)
    assert bool((grasps.width >= 0).all()) and bool((grasps.width <= 0.1).all())
    assert bool((grasps.depth >= 0.01 - 1e-6).all()) and bool((grasps.depth <= 0.02 + 1e-6).all())
    # The score is read from the grid at the decoded angle and depth.
    angle = raw.angle.argmax(dim=-1)
    if options.get("depth") == "regression":
        depth = (raw.depth.clamp(1.0, float(DEPTHS)).round() - 1).long()
    else:
        depth = raw.depth.argmax(dim=-1)
        assert torch.allclose(grasps.depth, (depth + 1) * 0.01)
    assert torch.equal(grasps.score, raw.score[torch.arange(6), angle, depth])


def test_width_decoding_per_option():
    seeds, approach = seeds_and_approach(1)
    # Regression: metres x 10, opened by the decode gain.
    bundle = head(width_decode_gain=1.2)
    raw = bundle(crops(1))
    raw.depth = torch.tensor([[0.0, 5.0]])
    raw.width = torch.tensor([[0.9, 0.5]])
    assert bundle.decode(raw, seeds, approach).width.item() == pytest.approx(1.2 * 0.05)
    # Bins: the expectation of the distribution over the bin positions.
    bundle = head(width="bins", width_bins=BINS, width_decode_gain=1.0)
    raw = bundle(crops(1))
    raw.depth = torch.tensor([[5.0, 0.0]])
    logits = torch.full((1, DEPTHS, BINS), -1e9)
    logits[0, 0, 1] = logits[0, 0, 2] = 0.0  # half the mass on bins 1 and 2
    raw.width = logits
    assert bundle.decode(raw, seeds, approach).width.item() == pytest.approx(0.1 * 1.5 / 4)
    # Class: the centre of the most likely bin.
    bundle = head(width="class", width_bins=BINS, width_decode_gain=1.0)
    logits[0, 0, 2] = 1.0
    raw = bundle(crops(1))
    raw.depth, raw.width = torch.tensor([[5.0, 0.0]]), logits
    assert bundle.decode(raw, seeds, approach).width.item() == pytest.approx(0.1 * 2 / 4)
    # A regressed depth is decoded continuously inside the range of the bins.
    bundle = head(depth="regression")
    raw = bundle(crops(1))
    for output, expected in ((1.4, 0.014), (0.2, 0.01), (7.0, 0.02)):
        raw.depth = torch.tensor([output])
        assert bundle.decode(raw, seeds, approach).depth.item() == pytest.approx(expected)


def test_a_logit_score_loss_is_decoded_through_its_sigmoid():
    bundle = head(score_loss={"type": "QualityFocal", "weight": 15.0})
    raw = bundle(crops())
    seeds, approach = seeds_and_approach()
    score = bundle.decode(raw, seeds, approach).score
    assert float(score.min()) > 0.0 and float(score.max()) < 1.0
    assert torch.isfinite(bundle.loss(raw, random_labels())["loss/score"])


def test_what_the_bundle_refuses():
    with pytest.raises(ValueError, match="unknown width 'histogram'"):
        head(width="histogram")
    with pytest.raises(ValueError, match="unknown depth 'bins'"):
        head(depth="bins")
    with pytest.raises(ValueError, match="width_bins at least 2"):
        head(width="bins", width_bins=1)
    bundle = head()
    with pytest.raises(ContractError, match=r"expects features with axes \('seed', 'channel'\)"):
        bundle(GroupFeatures(torch.zeros(5, 4, 8), ("seed", "depth", "channel")))
    with pytest.raises(ContractError, match="built for 8-channel crop features, got 3"):
        bundle(GroupFeatures(torch.zeros(5, 3), ("seed", "channel")))
    compact = CompactMatchedLabels.__new__(CompactMatchedLabels)
    with pytest.raises(ContractError, match="supervised with the dense label grids"):
        bundle.encode_targets(compact)
    wrong_grid = labels(torch.zeros(2, 12, 4), torch.zeros(2, 12, 4))
    with pytest.raises(ContractError, match=r"grid of \(12, 4\), but the head is configured"):
        bundle.encode_targets(wrong_grid)
    seeds, approach = seeds_and_approach(4)
    with pytest.raises(ContractError, match="do not line up"):
        bundle.decode(bundle(crops()), seeds, approach)


def test_the_config_check_knows_the_kind_of_every_term_per_option():
    """A width over bins takes a loss for bins, a regressed one a regression loss: the
    most likely misuse is refused before anything is built, with the losses that fit."""
    base = {"type": "DecoupledGridHead", "width": "bins"}
    assert stage_contract_problems(head={**base, "width_loss": {"type": "DistributionFocal"}}) == []
    (problem,) = stage_contract_problems(head={**base, "width_loss": {"type": "SmoothL1"}})
    assert problem.startswith("DecoupledGridHead.width_loss is a `bins` term")
    assert "but SmoothL1 fits `value` or `unit` terms" in problem
    assert "Losses for a `bins` term: DistributionFocal" in problem
    (problem,) = stage_contract_problems(
        head={"type": "DecoupledGridHead", "width_loss": {"type": "DistributionFocal"}}
    )
    assert "is a `value` term" in problem and "but DistributionFocal fits `bins` terms" in problem
    (problem,) = stage_contract_problems(
        head={
            "type": "DecoupledGridHead",
            "depth": "regression",
            "depth_loss": {"type": "CrossEntropy"},
        }
    )
    assert problem.startswith("DecoupledGridHead.depth_loss is a `value` term")
    # A built bundle gives the same verdict (the detector checks its built stages).
    assert stage_contract_problems(head=head(width="bins", width_bins=BINS)) == []
    # Next to its neighbours it has the contracts of SWADHead.
    assert (
        stage_contract_problems(
            grouper={"type": "CylinderGroup"}, head={"type": "DecoupledGridHead"}
        )
        == []
    )
    (problem,) = stage_contract_problems(
        assigner={"type": "CompactLabelAssigner"}, head={"type": "DecoupledGridHead"}
    )
    assert "must agree on the label set" in problem
