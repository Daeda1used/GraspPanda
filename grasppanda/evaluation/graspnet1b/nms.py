"""Grasp non-maximum suppression with the benchmark's exact rule, in NumPy.

The reference tool calls a compiled extension (``grasp_nms``). Its rule: visit grasps
in descending score order; a kept grasp removes every later grasp whose translation
distance is ``< translation_threshold`` *and* whose rotation angle is
``< rotation_threshold`` (both strict), where::

    distance = sqrt((dx*dx + dy*dy) + dz*dz)
    angle    = acos(0.5 * (clip(trace(R1 @ R2.T), -1, 3) - 1))

This module is the evaluator's own pinned copy (``grasppanda.ops.grasp_nms`` is a model
post-processor and may evolve). Differences from the extension, none of which change a
result on tie-free input:

* Ties. The extension orders grasps with ``std::sort``, which is unstable. Here equal
  scores keep their row order (earlier row first).
* Cost. The extension compares every pair, O(G²). Here a k-d tree proposes the pairs
  that are close enough in translation; the two tests are then evaluated for those
  pairs with exactly the arithmetic above, so the tree only prunes and never decides.
* Memory. Candidate pairs are processed in bounded blocks, and nothing is kept between
  calls (the extension never frees its buffers).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["grasp_nms"]

#: Candidate pairs evaluated at once; bounds the working memory to a few tens of MB.
_PAIR_BUDGET = 200_000
#: The tree search uses a slightly larger radius than the exact test that follows it.
_RADIUS_SLACK = 1.0 + 1e-9
#: Angles this close to the threshold are re-evaluated with the C library's ``acos``.
_ANGLE_GUARD = 1e-12


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle between rotations given as ``(P, 9)`` row-major matrices, and its cosine."""
    rows = [
        (first[:, r] * second[:, r] + first[:, r + 1] * second[:, r + 1])
        + first[:, r + 2] * second[:, r + 2]
        for r in (0, 3, 6)
    ]
    cosine = 0.5 * (np.clip((rows[0] + rows[1]) + rows[2], -1.0, 3.0) - 1.0)
    with np.errstate(invalid="ignore"):  # a NaN rotation yields a NaN angle: never suppressed
        return np.arccos(cosine), cosine


def _suppressed_pairs(
    translation: np.ndarray,
    rotation: np.ndarray,
    rows: np.ndarray,
    partners: np.ndarray,
    translation_threshold: float,
    rotation_threshold: float,
) -> np.ndarray:
    """Mask of candidate pairs ``(rows[i], partners[i])`` that satisfy both strict tests."""
    delta = translation[rows] - translation[partners]
    squared = (delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]) + delta[:, 2] * delta[:, 2]
    close = np.sqrt(squared) < translation_threshold
    angle, cosine = _rotation_angle(rotation[rows], rotation[partners])
    aligned = angle < rotation_threshold
    # NumPy may evaluate arccos with a vector routine that differs from libm in the last
    # bit; decide the (practically never occurring) borderline pairs with libm itself.
    for index in np.flatnonzero(close & (np.abs(angle - rotation_threshold) < _ANGLE_GUARD)):
        aligned[index] = math.acos(cosine[index]) < rotation_threshold
    return close & aligned


def grasp_nms(
    translation: np.ndarray,
    rotation: np.ndarray,
    score: np.ndarray,
    *,
    translation_threshold: float,
    rotation_threshold: float,
) -> np.ndarray:
    """Indices of the grasps that survive suppression, in descending score order.

    Args:
        translation: ``(G, 3)`` float64, metres.
        rotation: ``(G, 3, 3)`` float64. Matrices are used as given; they are not
            re-orthonormalised.
        score: ``(G,)``; must be finite. Equal scores are ranked by row index.
        translation_threshold: Metres, strict.
        rotation_threshold: Radians, strict.
    """
    translation = np.ascontiguousarray(translation, dtype=np.float64)
    count = len(translation)
    flat = np.ascontiguousarray(rotation, dtype=np.float64).reshape(count, 9)
    score = np.asarray(score, dtype=np.float64)
    if translation.shape != (count, 3) or score.shape != (count,):
        raise ValueError(
            f"expected (G, 3) translations, (G, 3, 3) rotations and (G,) scores, got "
            f"{translation.shape}, {np.shape(rotation)} and {score.shape}"
        )
    if not (np.isfinite(score).all() and np.isfinite(translation).all()):
        raise ValueError("grasp scores and translations must be finite to be ranked")
    order = np.argsort(-score, kind="stable")
    if count < 2:
        return order
    translation, flat = translation[order], flat[order]

    tree = cKDTree(translation)
    radius = translation_threshold * _RADIUS_SLACK
    neighbours = tree.query_ball_point(translation, radius, return_length=True)
    alive = np.ones(count, dtype=bool)
    start = 0
    while start < count:
        # Grow the block until its candidate pairs reach the budget (always >= 1 grasp).
        stop = start + 1
        pairs = int(neighbours[start])
        while stop < count and pairs + int(neighbours[stop]) <= _PAIR_BUDGET:
            pairs += int(neighbours[stop])
            stop += 1
        members = start + np.flatnonzero(alive[start:stop])
        lists = tree.query_ball_point(translation[members], radius)
        lengths = np.fromiter((len(found) for found in lists), dtype=np.int64, count=len(lists))
        rows = np.repeat(members, lengths)
        partners = np.fromiter(
            itertools.chain.from_iterable(lists), dtype=np.int64, count=int(lengths.sum())
        )
        later = partners > rows  # a grasp can only suppress grasps ranked below it
        rows, partners = rows[later], partners[later]
        hit = _suppressed_pairs(
            translation, flat, rows, partners, translation_threshold, rotation_threshold
        )
        rows, partners = rows[hit], partners[hit]
        ranked = np.argsort(rows, kind="stable")
        rows, partners = rows[ranked], partners[ranked]
        first = np.searchsorted(rows, members, side="left")
        last = np.searchsorted(rows, members, side="right")
        for member, lo, hi in zip(members.tolist(), first.tolist(), last.tolist()):
            if alive[member] and hi > lo:
                alive[partners[lo:hi]] = False
        start = stop
    return order[alive]
