"""Point augmentations adapted from Pointcept: what moves, what stays, what is refused."""

from __future__ import annotations

import numpy as np
import pytest

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError
from grasppanda.transforms.pointcept import RandomDropout, RandomJitter, RandomShift
from grasppanda.transforms.points import RandomSample

from .test_transforms import make_sample


def test_the_augmentations_are_registered_and_fingerprintable():
    assert {"RandomDropout", "RandomJitter", "RandomShift"} <= set(TRANSFORMS.names())
    step = TRANSFORMS.build({"type": "RandomShift", "shift": [[-1, 1], [0, 0], [0, 2]]})
    assert step.config() == {
        "type": "RandomShift",
        "version": 1,
        "shift": [[-1.0, 1.0], [0.0, 0.0], [0.0, 2.0]],
    }
    assert RandomJitter().config()["sigma"] == 0.001
    assert RandomDropout().config() == {
        "type": "RandomDropout",
        "version": 1,
        "ratio": 0.2,
        "prob": 0.5,
    }


def test_jitter_perturbs_the_input_points_only():
    sample = make_sample()
    points = sample.points.copy()
    poses = [annotation.pose.copy() for annotation in sample.objects]
    labels = {name: values.copy() for name, values in sample.point_labels.items()}
    RandomJitter(sigma=0.01, clip=0.02)(sample)
    moved = sample.points - points
    assert sample.points.shape == points.shape and sample.points.dtype == np.float64
    assert 0 < np.abs(moved).max() <= 0.02 + 1e-12 and 0.005 < moved.std() < 0.015
    assert np.array_equal(sample.full_points, points)  # the clean cloud stays
    for annotation, pose in zip(sample.objects, poses):
        assert np.array_equal(annotation.pose, pose)
    for name, values in labels.items():
        assert np.array_equal(sample.point_labels[name], values)
    assert np.allclose(RandomJitter(sigma=0.0).draw(5, sample.rng), 0.0)


def test_dropout_removes_points_with_their_labels_and_sampling_restores_the_count():
    sample = make_sample(count=30)
    points, labels = sample.points.copy(), sample.point_labels["graspness"].copy()
    RandomDropout(ratio=0.2, prob=1.0)(sample)
    assert len(sample) == 24 and len(sample.pixels) == 24
    assert np.array_equal(sample.index, np.sort(sample.index))
    assert np.array_equal(sample.points, points[sample.index])
    assert np.array_equal(sample.point_labels["graspness"], labels[sample.index])
    assert len(sample.full_points) == 30  # the masked cloud is not thinned
    RandomSample(30)(sample)
    assert len(sample) == 30 and set(sample.index) <= set(range(30))
    untouched = make_sample(count=30)
    RandomDropout(ratio=0.2, prob=0.0)(untouched)
    assert len(untouched) == 30 and untouched.index is None
    rng = np.random.default_rng(0)
    applied = [RandomDropout(0.2, 0.5).draw(10, rng) is not None for _ in range(2000)]
    assert 0.45 < np.mean(applied) < 0.55


def test_shift_translates_points_and_object_poses_together():
    sample = make_sample()
    points = sample.points.copy()
    poses = [annotation.pose.copy() for annotation in sample.objects]
    offset = np.array([0.1, -0.2, 0.05])
    RandomShift([[0, 0], [0, 0], [0, 0]]).apply(sample, offset)
    assert np.allclose(sample.points, points + offset)
    assert sample.full_points is sample.points
    for annotation, pose in zip(sample.objects, poses):
        assert annotation.pose.dtype == np.float32
        assert np.array_equal(annotation.pose[:3, :3], pose[:3, :3])
        assert np.allclose(annotation.pose[:3, 3], pose[:3, 3] + offset, atol=1e-6)
    sampled = make_sample()
    RandomSample(8)(sampled)
    RandomShift([[0, 0], [0, 0], [0, 0]]).apply(sampled, offset)
    assert np.allclose(sampled.points, sampled.full_points[sampled.index])
    rng = np.random.default_rng(0)
    step = RandomShift([[-0.1, 0.1], [0.0, 0.0], [-0.05, 0.1]])
    draws = np.array([step.draw(rng) for _ in range(500)])
    assert draws[:, 0].min() >= -0.1 and draws[:, 0].max() <= 0.1 and not draws[:, 1].any()
    assert draws[:, 2].min() >= -0.05 and draws[:, 2].max() <= 0.1


def test_draws_are_seeded_by_the_sample():
    one, two = make_sample(seed=3), make_sample(seed=3)
    for step in (
        RandomDropout(0.2, 0.5),
        RandomJitter(),
        RandomShift([[-0.1, 0.1], [-0.1, 0.1], [0, 0]]),
    ):
        one, two = step(one), step(two)
    assert np.array_equal(one.points, two.points)


def test_moving_points_next_to_images_is_refused_and_bad_arguments_are_named():
    sample = make_sample()
    sample.depth = np.zeros(sample.image_size, dtype=np.float32)
    for step in (RandomJitter(), RandomShift([[0, 1], [0, 1], [0, 1]])):
        with pytest.raises(ContractError, match=r"moves points.*\(depth\)"):
            step(sample)
    RandomDropout(0.2, 1.0)(sample)  # removes points, moves none
    with pytest.raises(ValueError, match="three \\[min, max\\] pairs"):
        RandomShift([[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="min <= max"):
        RandomShift([[1, 0], [0, 1], [0, 1]])
    with pytest.raises(ValueError, match="sigma >= 0 and clip > 0"):
        RandomJitter(clip=0.0)
    with pytest.raises(ValueError, match="0 <= ratio < 1"):
        RandomDropout(ratio=1.0)
