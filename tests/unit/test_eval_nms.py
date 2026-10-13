"""Evaluator NMS against a brute-force statement of the benchmark rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b import nms
from grasppanda.evaluation.graspnet1b.nms import grasp_nms
from grasppanda.evaluation.graspnet1b.protocol import GRASPNET1B_SV_V1 as PROTOCOL

T, R = PROTOCOL.nms_translation, PROTOCOL.nms_rotation


def brute_force(translation, rotation, score, t=T, r=R):
    """The rule, pair by pair: greedy in descending score, both tests strict."""
    order = sorted(range(len(score)), key=lambda i: (-score[i], i))
    kept = []
    for i in order:
        suppressed = False
        for j in kept:
            d = translation[j] - translation[i]
            distance = math.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2])
            trace = float(np.trace(rotation[j] @ rotation[i].T))
            angle = math.acos(0.5 * (min(max(trace, -1.0), 3.0) - 1.0))
            if distance < t and angle < r:
                suppressed = True
                break
        if not suppressed:
            kept.append(i)
    return np.array(kept, dtype=np.int64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def random_set(count: int, seed: int, spread: float = 0.06):
    rng = np.random.default_rng(seed)
    translation = rng.uniform(-spread, spread, size=(count, 3))
    rotation = np.linalg.qr(rng.normal(size=(count, 3, 3)))[0]
    # a third of the grasps are near-copies of others, so that suppression really happens
    copies = rng.integers(0, count, size=count // 3)
    translation[: len(copies)] = translation[copies] + rng.normal(0, 0.004, size=(len(copies), 3))
    twist = np.array([rotation_z(a) for a in rng.normal(0, 0.2, size=len(copies))]).reshape(
        -1, 3, 3
    )
    rotation[: len(copies)] = rotation[copies] @ twist
    return translation, rotation, rng.permutation(count).astype(np.float64)


def run(translation, rotation, score):
    return grasp_nms(translation, rotation, score, translation_threshold=T, rotation_threshold=R)


@pytest.mark.parametrize("count,seed", [(1, 0), (2, 1), (40, 2), (300, 3), (900, 4)])
def test_matches_brute_force(count, seed):
    translation, rotation, score = random_set(count, seed)
    keep = run(translation, rotation, score)
    expected = brute_force(translation, rotation, score)
    assert np.array_equal(keep, expected)
    assert len(expected) < count or count < 3  # the fixture does exercise suppression
    assert (np.diff(score[keep]) < 0).all()  # best first


def test_small_blocks_give_the_same_answer(monkeypatch):
    translation, rotation, score = random_set(400, 5, spread=0.02)
    expected = brute_force(translation, rotation, score)
    monkeypatch.setattr(nms, "_PAIR_BUDGET", 7)  # many blocks, some with a single grasp
    assert np.array_equal(run(translation, rotation, score), expected)


def test_empty_input():
    keep = run(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0))
    assert keep.shape == (0,)


def test_both_thresholds_are_strict():
    eye = np.eye(3)
    for distance, angle, suppressed in [
        (T * 0.5, R * 0.5, True),
        (T * 1.5, R * 0.5, False),  # far apart
        (T * 0.5, R * 1.5, False),  # rotated apart
        (0.0, R + 1e-9, False),
        (0.0, R - 1e-9, True),
    ]:
        translation = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        rotation = np.stack([eye, rotation_z(angle)])
        keep = run(translation, rotation, np.array([1.0, 0.5]))
        assert (len(keep) == 1) is suppressed, (distance, angle)
    # exactly at the translation threshold: not suppressed (0.03 - 0 == 0.03 exactly)
    translation = np.array([[0.0, 0.0, 0.0], [0.0, T, 0.0]])
    assert len(run(translation, np.stack([eye, eye]), np.array([1.0, 0.5]))) == 2


def test_suppression_is_decided_by_kept_grasps_only():
    """B is suppressed by A, so B cannot suppress C even though B and C are close."""
    eye = np.eye(3)
    translation = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]])
    keep = run(translation, np.stack([eye] * 3), np.array([0.9, 0.8, 0.7]))
    assert keep.tolist() == [0, 2]


def test_ties_keep_row_order():
    eye = np.eye(3)
    translation = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.5, 0, 0], [0.501, 0, 0]])
    score = np.array([0.7, 0.7, 0.7, 0.7])
    keep = run(translation, np.stack([eye] * 4), score)
    assert keep.tolist() == [0, 2]  # the earlier row of each tied pair survives
    reversed_keep = run(translation[::-1], np.stack([eye] * 4), score)
    assert reversed_keep.tolist() == [0, 2]


def test_rotations_are_used_as_given():
    """A scaled matrix is not re-normalised: its trace is clipped, as in the reference."""
    eye = np.eye(3)
    translation = np.zeros((2, 3))
    keep = run(translation, np.stack([eye, 5.0 * eye]), np.array([1.0, 0.5]))
    assert len(keep) == 1  # trace 15 clips to 3, angle 0
    keep = run(translation, np.stack([eye, np.full((3, 3), np.nan)]), np.array([1.0, 0.5]))
    assert len(keep) == 2  # a NaN angle is never below the threshold


def test_rejects_unrankable_input():
    eye = np.stack([np.eye(3)] * 2)
    with pytest.raises(ValueError, match="finite"):
        run(np.zeros((2, 3)), eye, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match=r"\(G, 3\)"):
        run(np.zeros((2, 2)), eye, np.array([1.0, 0.5]))
