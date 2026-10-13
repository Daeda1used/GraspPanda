"""Evaluator-side collision and emptiness check against ground-truth geometry.

This is part of *scoring*: it uses the true object models (and, for GraspNet-1Billion,
a synthetic table). It is unrelated to the model-side collision *filter*, which works
on the observed cloud and is part of inference.

The gripper is three boxes in the grasp frame (x: approach, y: closing, z: height):
two fingers of thickness ``finger_width`` outside ``|y| = width / 2`` reaching from
``-depth_base`` to the grasp depth, and a palm of thickness ``finger_width`` behind
them. A grasp collides when any scene point lies strictly inside a box, and is empty
when fewer than ``empty_threshold`` points lie between the fingers. The predicted
``height`` column is ignored; the gripper's own finger height is used.

All comparisons are strict and evaluated in float64, as in the reference.
"""

from __future__ import annotations

import numpy as np

from grasppanda.structures.common import Gripper

from .arithmetic import rotate_rows

__all__ = ["crop_to_box", "gripper_collision"]

#: Safety factor on the pruning radius; rounding errors are about ten orders smaller.
_RADIUS_SLACK = 1.0 + 1e-6


def crop_to_box(scene: np.ndarray, anchor: np.ndarray, margin: float) -> np.ndarray:
    """Scene points strictly inside the bounding box of ``anchor`` grown by ``margin``."""
    lower = anchor.min(axis=0) - margin
    upper = anchor.max(axis=0) + margin
    inside = (scene[:, 0] > lower[0]) & (scene[:, 0] < upper[0])
    for axis in (1, 2):
        inside &= (scene[:, axis] > lower[axis]) & (scene[:, axis] < upper[axis])
    return scene[inside]


def _within_reach(
    scene: np.ndarray, translation: np.ndarray, rotation: np.ndarray, extent: np.ndarray
) -> np.ndarray:
    """Mask of the points that could lie inside a gripper whose half-extents are ``extent``.

    Pruning only, never a decision: a point with gripper-frame coordinates ``q`` and
    scene offset ``d`` satisfies ``|q| >= s * |d|`` where ``s`` is the smallest singular
    value of the orientation matrix, so points further than ``|extent| / s`` cannot be
    inside any gripper region. For a matrix that is not a rotation the bound adapts;
    if it collapses a direction (``s = 0``) nothing is pruned.
    """
    smallest = np.linalg.svd(rotation, compute_uv=False)[-1]
    radius = np.sqrt((extent * extent).sum()) * _RADIUS_SLACK
    if not np.isfinite(smallest) or smallest * 1e12 <= radius:
        return np.ones(len(scene), dtype=bool)
    radius = radius / smallest * _RADIUS_SLACK
    offset = scene - translation
    return (offset * offset).sum(axis=1) <= radius * radius


def gripper_collision(
    translation: np.ndarray,
    rotation: np.ndarray,
    depth: np.ndarray,
    width: np.ndarray,
    scene: np.ndarray,
    *,
    gripper: Gripper,
    empty_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Test grasps against scene points.

    Args:
        translation: ``(G, 3)`` grasp points, same frame as ``scene``.
        rotation: ``(G, 3, 3)`` grasp orientations.
        depth: ``(G,)`` finger insertion depths.
        width: ``(G,)`` jaw openings.
        scene: ``(P, 3)`` points to test against.
        gripper: Finger height, finger width and palm offset.
        empty_threshold: Minimum number of points between the fingers.

    Returns:
        ``collision`` and ``empty``, both ``(G,)`` bool. ``collision`` is true for
        colliding *or* empty grasps, as in the reference; ``empty`` flags the latter.
    """
    count = len(translation)
    half_height = gripper.finger_height / 2
    back = gripper.depth_base + gripper.finger_width
    hit = np.zeros(count, dtype=bool)
    between = np.zeros(count, dtype=np.int64)
    for grasp in range(count):
        half_width = width[grasp] / 2
        outer = half_width + gripper.finger_width
        extent = np.array([max(abs(depth[grasp]), back), outer, half_height])
        near = scene[_within_reach(scene, translation[grasp], rotation[grasp], extent)]
        local = rotate_rows(near - translation[grasp], rotation[grasp])
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        in_height = (z > -half_height) & (z < half_height)
        along = (x > -gripper.depth_base) & (x < depth[grasp])
        left = (y > -outer) & (y < -half_width)
        right = (y < outer) & (y > half_width)
        palm = (y > -outer) & (y < outer) & (x > -back) & (x < -gripper.depth_base)
        hit[grasp] = (in_height & ((along & (left | right)) | palm)).any()
        between[grasp] = (in_height & along & ~(y < -half_width) & ~(y > half_width)).sum()
    empty = between < empty_threshold
    return hit | empty, empty
