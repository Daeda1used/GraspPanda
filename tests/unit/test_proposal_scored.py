"""ScoredSeeds and SampledSeeds: selection rules, the sampler slot, too few points, misuse."""

from __future__ import annotations

import pytest
import torch

from grasppanda.config import Config, ConfigError
from grasppanda.models.proposals.scored import SampledSeeds, ScoredSeeds
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import PROPOSALS, SAMPLERS
from grasppanda.structures import (
    ContractError,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    Supervision,
)


def scored(num_seeds: int = 8, **options) -> ScoredSeeds:
    """A proposal whose scores are its input features: [logit0, logit1, graspness]."""
    stage = PROPOSALS.build(
        {"type": "ScoredSeeds", "num_seeds": num_seeds, **options}, in_channels=3
    )
    with torch.no_grad():
        stage.score.weight.copy_(torch.eye(3))
        stage.score.bias.zero_()
    return stage


def scene(sizes: list[int], seed: int = 0) -> tuple[Observation, PointFeatures]:
    generator = torch.Generator().manual_seed(seed)
    sets = [PointSet(torch.rand(n, 3, generator=generator) + 0.5) for n in sizes]
    points = PointSet.cat(sets) if len(sets) > 1 else sets[0]
    scores = torch.randn(len(points), 3, generator=generator)
    keys = [SampleKey("unit", 0, "none", i) for i in range(len(sizes))]
    return Observation(keys=keys, points=points), PointFeatures(scores, points)


def test_the_defaults_select_as_graspness_fps_without_the_kernel_quirks():
    observation, feats = scene([120, 90])
    seeds, scores = scored()(feats, observation)
    reference = PROPOSALS.build(
        {
            "type": "GraspnessFPS",
            "num_seeds": 8,
            "fps_skip_near_origin": False,
            "fps_tie_break": "lowest_index",
        },
        in_channels=3,
    )
    with torch.no_grad():
        reference.score.weight.copy_(torch.eye(3))
        reference.score.bias.zero_()
    expected, _ = reference(feats, observation)
    assert torch.equal(seeds.points.parent_index, expected.points.parent_index)
    assert torch.equal(seeds.valid, expected.valid) and bool(seeds.valid.all())
    assert seeds.source == "scored_seeds" and seeds.points.parent_id == observation.points.id
    assert seeds.points.batch_offsets.tolist() == [0, 8, 16]
    assert torch.equal(seeds.feats, feats.feats[seeds.points.parent_index])
    assert set(seeds.scores) == set(ScoredSeeds.seed_scores) == {"graspness", "objectness"}
    assert scores.num_candidates.tolist() == expected_counts(feats, [120, 90])


def expected_counts(feats: PointFeatures, sizes: list[int]) -> list[int]:
    mask = (feats.feats[:, 1] > feats.feats[:, 0]) & (feats.feats[:, 2] > 0.1)
    return [int(part.sum()) for part in mask.split(sizes)]


@pytest.mark.parametrize(
    ("select_by", "rule"),
    [
        ("graspness_and_objectness", lambda p, g: (p > 0.5) & (g > 0.1)),
        ("graspness", lambda p, g: g > 0.1),
        ("objectness", lambda p, g: p > 0.5),
        ("product", lambda p, g: p * g.clamp(min=0) > 0.1),
    ],
)
def test_the_score_that_selects_is_an_option(select_by, rule):
    observation, feats = scene([150])
    stage = scored(select_by=select_by)
    seeds, scores = stage(feats, observation)
    is_object = feats.feats[:, :2].softmax(-1)[:, 1]
    mask = rule(is_object, feats.feats[:, 2])
    assert torch.equal(stage.candidates(scores), mask)
    assert bool(mask[seeds.points.parent_index].all()) and bool(seeds.valid.all())
    members = torch.nonzero(mask).flatten()
    assert torch.equal(
        seeds.points.parent_index,
        members[furthest_point_sample(observation.points.coords[members], 8)],
    )


def test_a_rank_cut_replaces_the_thresholds():
    observation, feats = scene([100, 60])
    stage = scored(select_by="graspness", top_fraction=0.2)
    seeds, scores = stage(feats, observation)
    assert scores.num_candidates.tolist() == [20, 12]
    graspness = feats.feats[:, 2]
    assert graspness[seeds.points.parent_index[:8]].min() >= graspness[:100].topk(20).values.min()
    with pytest.raises(Exception, match=r"top_fraction must lie in \(0, 1\]"):
        scored(top_fraction=0.0)


def test_the_sampler_is_a_component_and_receives_scores_and_features():
    observation, feats = scene([200])
    top = scored(sampler={"type": "TopK"}, select_by="graspness")
    seeds, _ = top(feats, observation)
    assert torch.equal(seeds.points.parent_index, feats.feats[:, 2].topk(8).indices)
    for name in ("FeatureFPS", "FusionFPS", "ScoreWeightedFPS", "RandomSampler", "FPSThenTopK"):
        stage = scored(sampler={"type": name})
        seeds, scores = stage(feats, observation)
        assert isinstance(stage.sampler, SAMPLERS.get(name))
        assert bool(stage.candidates(scores)[seeds.points.parent_index].all())
        assert len(seeds.points.parent_index.unique()) == 8 and bool(seeds.valid.all())
        assert name in repr(stage)


def test_too_few_candidates_are_padded_or_filled_with_the_best_of_the_rest():
    observation, feats = scene([40, 40], seed=2)
    feats.feats[:40, 2] = -1.0  # frame 0: nothing is graspable
    feats.feats[40:, 2] = -1.0
    feats.feats[40:43, 1:] = 5.0  # frame 1: three candidates
    padded, scores = scored(too_few="pad")(feats, observation)
    assert scores.num_candidates.tolist() == [0, 3]
    assert padded.valid.reshape(2, 8).sum(dim=1).tolist() == [0, 3]
    assert padded.points.parent_index[:8].tolist() == [0] * 8  # the frame's first point
    assert set(padded.points.parent_index[8:].tolist()) == {40, 41, 42}

    feats.feats[:40, 2] = torch.linspace(-0.9, -0.1, 40)
    filled, _ = scored(too_few="best_rest", select_by="graspness")(feats, observation)
    assert bool(filled.valid.all()) and len(filled.points.parent_index.unique()) == 16
    # The selecting score is never negative: below zero all are equal, the first ones win.
    assert filled.points.parent_index[:8].tolist() == list(range(8))
    assert set(filled.points.parent_index[8:11].tolist()) == {40, 41, 42}
    assert bool((filled.points.parent_index[8:] >= 40).all())  # never across frames

    tiny_observation, tiny_feats = scene([5])
    tiny, _ = scored(too_few="best_rest")(tiny_feats, tiny_observation)
    assert int(tiny.valid.sum()) == 5 and len(tiny) == 8  # a frame smaller than the seed count


def test_a_variable_count_sampler_leaves_marked_padding():
    observation, feats = scene([300])
    feats.feats[:, 1:] = 5.0  # every point is a candidate
    stage = scored(num_seeds=64, sampler={"type": "ScoreNMS", "min_distance": 0.4})
    seeds, _ = stage(feats, observation)
    real = int(seeds.valid.sum())
    assert 0 < real < 64 and not bool(seeds.valid[real:].any())
    assert len(seeds.points.parent_index[:real].unique()) == real


def test_scored_seeds_trains_its_scorer_and_replays_recorded_seeds():
    observation, feats = scene([60, 50])
    feats = PointFeatures(feats.feats.clone().requires_grad_(), feats.points)
    stage = scored(sampler={"type": "ScoreWeightedFPS"})
    seeds, scores = stage(feats, observation)
    labels = {
        "objectness": (torch.arange(110) % 2).long(),
        "graspness": torch.rand(110),
    }
    losses = stage.loss(scores, Supervision(point_labels=labels))
    assert {"loss/objectness", "loss/graspness"} <= set(losses)
    (losses["loss/objectness"] + losses["loss/graspness"] + seeds.feats.sum()).backward()
    assert stage.score.weight.grad is not None and bool(feats.feats.grad.abs().sum() > 0)
    recorded = seeds.points.parent_index.reshape(2, 8)
    replayed, _ = stage(feats, observation, seed_index=recorded)
    assert replayed.source == "replayed"
    assert torch.equal(replayed.points.parent_index, seeds.points.parent_index)


def test_sampled_seeds_scores_nothing_and_samples_every_frame():
    observation, feats = scene([120, 90])
    stage = PROPOSALS.build({"type": "SampledSeeds", "num_seeds": 8}, in_channels=3)
    assert isinstance(stage, SampledSeeds) and stage.requires == "dense"
    assert stage.seed_scores == () and not list(stage.parameters())
    seeds, scores = stage(feats, observation)
    assert scores is None and stage.loss(None, Supervision()) == {}
    assert seeds.scores == {} and seeds.source == "sampled_seeds" and bool(seeds.valid.all())
    assert seeds.points.batch_offsets.tolist() == [0, 8, 16]
    coords = observation.points.coords
    assert torch.equal(seeds.points.parent_index[:8], furthest_point_sample(coords[:120], 8))
    assert torch.equal(seeds.points.parent_index[8:], furthest_point_sample(coords[120:], 8) + 120)
    # Features follow the points and stay attached to the encoder.
    live = PointFeatures(feats.feats.clone().requires_grad_(), feats.points)
    by_feature = PROPOSALS.build(
        {"type": "SampledSeeds", "num_seeds": 8, "sampler": {"type": "FeatureFPS"}}, in_channels=3
    )
    seeds, _ = by_feature(live, observation)
    seeds.feats.sum().backward()
    assert int((live.feats.grad.abs().sum(dim=1) > 0).sum()) == 16
    small_observation, small_feats = scene([5])
    small, _ = stage(small_feats, small_observation)
    assert small.valid.tolist() == [True] * 5 + [False] * 3


def test_misuse_is_refused_with_the_fix():
    # A score-driven sampler inside the proposal that predicts no score: refused by the
    # config check, before anything is built, and by the constructor.
    node = {
        "type": "GSNet",
        "backbone": {"type": "PointNet2Dense"},
        "proposal": {"type": "SampledSeeds", "sampler": {"type": "TopK"}},
        "view": {"type": "TemplateViewStage"},
        "assigner": {"type": "NearestLabelAssigner"},
        "grouper": {"type": "CylinderGroup"},
        "head": {"type": "SWADHead"},
    }
    with pytest.raises(ConfigError) as caught:
        Config.from_dict({"model": node})
    messages = [issue.message for issue in caught.value.issues]
    assert any(
        "TopK needs scores per candidate point, but SampledSeeds predicts no score" in message
        and "FeatureFPS" in message
        and "ScoredSeeds" in message
        for message in messages
    ), messages
    with pytest.raises(Exception, match="SampledSeeds predicts no score, but TopK needs scores"):
        PROPOSALS.build({"type": "SampledSeeds", "sampler": {"type": "TopK"}}, in_channels=3)
    assert (
        list(ScoredSeeds.check_config({"type": "ScoredSeeds", "sampler": {"type": "TopK"}})) == []
    )
    # Unknown option values name the choices.
    with pytest.raises(Exception, match="unknown select_by 'score'"):
        scored(select_by="score")
    with pytest.raises(Exception, match="unknown too_few 'drop'"):
        scored(too_few="drop")
    # Features on a subsample: both proposals need dense features.
    observation, feats = scene([40])
    child = observation.points.subset(torch.arange(10))
    for stage in (
        scored(),
        PROPOSALS.build({"type": "SampledSeeds", "num_seeds": 8}, in_channels=3),
    ):
        with pytest.raises(ContractError, match="requires 'dense' scene features"):
            stage(PointFeatures(feats.feats[:10], child), observation)
