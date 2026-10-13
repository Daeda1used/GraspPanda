"""GraspnessFPS: the selection rule against its definition, padding, replay, loss, misuse."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.proposals.graspness import GraspnessFPS
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import PROPOSALS, BuildError
from grasppanda.structures import (
    ContractError,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    Supervision,
)


def proposal(num_seeds: int = 8, **overrides) -> GraspnessFPS:
    """A proposal whose scores are its input features: [logit0, logit1, graspness]."""
    stage = PROPOSALS.build(
        {"type": "GraspnessFPS", "num_seeds": num_seeds, **overrides}, in_channels=3
    )
    with torch.no_grad():
        stage.score.weight.copy_(torch.eye(3))
        stage.score.bias.zero_()
    return stage


def scene(sizes: list[int], seed: int = 0) -> tuple[Observation, PointFeatures]:
    """Frames of the given sizes with random points (away from the origin) and scores."""
    generator = torch.Generator().manual_seed(seed)
    sets = [PointSet(torch.rand(n, 3, generator=generator) + 0.5) for n in sizes]
    points = PointSet.cat(sets) if len(sets) > 1 else sets[0]
    scores = torch.randn(len(points), 3, generator=generator)
    keys = [SampleKey("unit", 0, "none", i) for i in range(len(sizes))]
    return Observation(keys=keys, points=points), PointFeatures(scores, points)


def test_seeds_are_fps_over_the_points_that_pass_both_tests():
    stage = proposal(num_seeds=8)
    observation, feats = scene([120, 90])
    seeds, scores = stage(feats, observation)
    assert stage.requires == "dense" and len(seeds) == 16 and seeds.source == "graspness_fps"
    assert seeds.points.parent_id == observation.points.id
    assert seeds.points.batch_offsets.tolist() == [0, 8, 16]
    assert torch.equal(scores.objectness, feats.feats[:, :2])
    assert torch.equal(scores.graspness, feats.feats[:, 2])

    is_object = feats.feats[:, 1] > feats.feats[:, 0]
    mask = is_object & (feats.feats[:, 2] > 0.1)
    assert torch.equal(stage.candidates(scores), mask)
    start = 0
    for frame, size in enumerate([120, 90]):
        members = torch.nonzero(mask[start : start + size]).flatten() + start
        picks = furthest_point_sample(observation.points.coords[members], 8)  # the definition
        expected = members[picks]
        rows = slice(frame * 8, frame * 8 + 8)
        assert torch.equal(seeds.points.parent_index[rows], expected)
        assert scores.num_candidates[frame] == len(members)
        start += size
    index = seeds.points.parent_index
    assert bool(seeds.valid.all()) and len(set(index.tolist())) == 16
    assert torch.equal(seeds.points.coords, observation.points.coords[index])
    assert torch.equal(seeds.feats, feats.feats[index])  # features follow the points
    assert torch.equal(seeds.scores["graspness"], feats.feats[index, 2])
    assert torch.allclose(seeds.scores["objectness"], feats.feats[index, :2].softmax(-1)[:, 1])


def test_selection_does_not_depend_on_how_frames_are_batched():
    stage = proposal(num_seeds=6)
    observation, feats = scene([70, 110], seed=3)
    together, _ = stage(feats, observation)
    second = PointSet(observation.points.coords[70:])
    alone, _ = stage(
        PointFeatures(feats.feats[70:], second),
        Observation(keys=observation.keys[1:], points=second),
    )
    assert torch.equal(together.points.parent_index[6:] - 70, alone.points.parent_index)


def test_frames_with_too_few_candidates_are_padded_and_marked():
    stage = proposal(num_seeds=6)
    observation, _ = scene([40, 40, 40], seed=1)
    scores = torch.zeros(120, 3)
    scores[:, 0] = 1.0  # nothing is an object ...
    scores[[3, 17, 29], 1] = 5.0  # ... except three points of frame 0
    scores[[3, 17, 29], 2] = 1.0
    scores[40:80, 1], scores[40:80, 2] = 5.0, 1.0  # frame 1: everything; frame 2: nothing
    seeds, point_scores = stage(PointFeatures(scores, observation.points), observation)
    valid = seeds.valid.reshape(3, 6)
    index = seeds.points.parent_index.reshape(3, 6)
    assert point_scores.num_candidates.tolist() == [3, 40, 0]
    assert valid.sum(dim=1).tolist() == [3, 6, 0]
    assert (
        sorted(index[0, :3].tolist()) == [3, 17, 29]
        and valid[0].tolist() == [True] * 3 + [False] * 3
    )
    assert (index[0, 3:] == index[0, 0]).all()  # padding repeats the first seed of the frame
    assert (index[2] == 80).all()  # no candidate at all: the frame's first point, all invalid
    assert len(seeds) == 18  # the seed count per frame never changes


def test_repeated_coordinates_are_not_counted_twice():
    stage = proposal(num_seeds=4)
    coords = torch.tensor([[1.0, 1, 1], [1.0, 1, 1], [2.0, 1, 1], [2.0, 1, 1], [1.0, 3, 1]])
    points = PointSet(coords)
    scores = torch.tensor([[0.0, 1.0, 1.0]]).repeat(5, 1)
    observation = Observation(keys=[SampleKey("unit", 0, "none", 0)], points=points)
    seeds, _ = stage(PointFeatures(scores, points), observation)
    assert seeds.valid.tolist() == [True, True, True, False]  # three distinct locations
    assert len({tuple(c) for c in seeds.points.coords[seeds.valid].tolist()}) == 3


def test_points_next_to_the_origin_are_skipped_like_the_reference_kernel():
    coords = torch.tensor([[0.5, 0.5, 0.5], [0.01, 0.01, 0.01], [0.6, 0.5, 0.5], [0.5, 0.7, 0.5]])
    points = PointSet(coords)
    scores = torch.tensor([[0.0, 1.0, 1.0]]).repeat(4, 1)
    observation = Observation(keys=[SampleKey("unit", 0, "none", 0)], points=points)
    feats = PointFeatures(scores, points)
    reference, _ = proposal(num_seeds=3)(feats, observation)
    plain, _ = proposal(num_seeds=3, fps_skip_near_origin=False)(feats, observation)
    assert 1 not in reference.points.parent_index.tolist()
    assert plain.points.parent_index.tolist()[1] == 1  # the farthest point by distance alone


def test_equally_distant_candidates_follow_the_configured_tie_order():
    coords = torch.rand(40, 3, generator=torch.Generator().manual_seed(2)) * 0.05 + 1.0
    coords[0] = 1.0
    tied = [1, 2, 3, 4, 5]  # all exactly 0.5 away from point 0, everything else is close to it
    coords[tied] = coords[0] + torch.tensor(
        [[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5]]
    )
    points = PointSet(coords)
    feats = PointFeatures(torch.tensor([[0.0, 1.0, 1.0]]).repeat(40, 1), points)
    observation = Observation(keys=[SampleKey("unit", 0, "none", 0)], points=points)
    picked = {}
    for rule in ("pointnet2_cuda", "lowest_index"):
        seeds, _ = proposal(num_seeds=2, fps_tie_break=rule)(feats, observation)
        expected = furthest_point_sample(coords, 2, skip_near_origin=True, tie_break=rule)
        assert torch.equal(seeds.points.parent_index, expected)
        picked[rule] = seeds.points.parent_index[1].item()
    assert picked == {"pointnet2_cuda": 4, "lowest_index": 1}  # 4: first in bit-reversed order
    assert proposal().fps_tie_break == "pointnet2_cuda"  # the reference kernel is the default
    with pytest.raises(ValueError, match="unknown fps_tie_break 'random'"):
        GraspnessFPS(fps_tie_break="random", in_channels=3)


def test_recorded_seed_indices_can_be_replayed():
    stage = proposal(num_seeds=4)
    observation, feats = scene([30, 30], seed=5)
    recorded = torch.tensor([[0, 5, 9, 29], [30, 31, 45, 59]])
    seeds, _ = stage(feats, observation, seed_index=recorded)
    assert torch.equal(seeds.points.parent_index, recorded.reshape(-1))
    assert bool(seeds.valid.all()) and seeds.source == "replayed"
    with pytest.raises(ContractError, match=r"seed_index must be \(2, 4\)"):
        stage(feats, observation, seed_index=recorded[:, :3])
    with pytest.raises(ContractError, match="outside the frame of its row"):
        stage(feats, observation, seed_index=recorded.flip(0))


def test_features_on_a_subsampled_set_are_refused():
    """A PointNet++-style backbone returns features on a child set: not what this stage needs."""
    stage = proposal()
    observation, _ = scene([64])
    child = observation.points.subset(torch.arange(0, 64, 4))
    with pytest.raises(ContractError) as error:
        stage(PointFeatures(torch.zeros(16, 3), child), observation)
    message = str(error.value)
    assert "requires 'dense' scene features" in message and "subsample" in message
    with pytest.raises(ContractError, match="built for 3-channel features, got 5"):
        stage(PointFeatures(torch.zeros(64, 5), observation.points), observation)
    with pytest.raises(ContractError, match="needs observation.points"):
        stage(PointFeatures(torch.zeros(64, 3), observation.points), Observation(keys=[]))


def test_loss_is_cross_entropy_plus_graspness_on_object_points():
    stage = proposal()
    observation, feats = scene([50, 30], seed=7)
    feats.feats.requires_grad_(True)
    _, scores = stage(feats, observation)
    generator = torch.Generator().manual_seed(8)
    objectness = torch.randint(0, 3, (80,), generator=generator)  # instance ids: non-zero = object
    graspness = torch.rand(80, 1, generator=generator)  # a trailing unit axis is accepted
    supervision = Supervision(point_labels={"objectness": objectness, "graspness": graspness})
    losses = stage.loss(scores, supervision)

    is_object = objectness != 0
    expected_objectness = F.cross_entropy(feats.feats[:, :2], is_object.long())
    expected_graspness = 10.0 * F.smooth_l1_loss(feats.feats[is_object, 2], graspness[is_object, 0])
    assert losses["loss/objectness"].item() == pytest.approx(expected_objectness.item(), rel=1e-6)
    assert losses["loss/graspness"].item() == pytest.approx(expected_graspness.item(), rel=1e-6)
    predicted = feats.feats[:, 1] > feats.feats[:, 0]
    assert losses["metric/objectness_accuracy"].item() == pytest.approx(
        (predicted == is_object).float().mean().item()
    )
    assert losses["metric/objectness_recall"].item() == pytest.approx(
        (predicted & is_object).sum().item() / is_object.sum().item()
    )
    (losses["loss/objectness"] + losses["loss/graspness"]).backward()
    assert feats.feats.grad.abs().sum() > 0


def test_loss_needs_aligned_point_labels():
    stage = proposal()
    observation, feats = scene([20])
    _, scores = stage(feats, observation)
    with pytest.raises(ContractError, match=r"needs supervision.point_labels\['graspness'\]"):
        stage.loss(scores, Supervision(point_labels={"objectness": torch.zeros(20)}))
    labels = {"objectness": torch.zeros(19), "graspness": torch.zeros(20)}
    with pytest.raises(ContractError, match="must be aligned with observation.points"):
        stage.loss(scores, Supervision(point_labels=labels))


def test_configured_losses_and_injection():
    stage = PROPOSALS.build(
        {"type": "GraspnessFPS", "graspness_loss": {"type": "L1", "weight": 3.0}}, in_channels=16
    )
    assert type(stage.graspness_loss).__name__ == "L1" and stage.graspness_loss.weight == 3.0
    assert stage.objectness_loss.weight == 1.0 and stage.num_seeds == 1024
    assert stage.graspness_threshold == 0.1 and stage.score.in_features == 16
    with pytest.raises(BuildError, match="needs in_channels"):
        PROPOSALS.build({"type": "GraspnessFPS"})
    with pytest.raises(ValueError, match="num_seeds must be positive"):
        GraspnessFPS(num_seeds=0, in_channels=4)
