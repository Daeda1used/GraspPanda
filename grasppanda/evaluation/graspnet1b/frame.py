"""Scoring of one frame: the order of operations that defines the benchmark.

1. Clip predicted widths to ``[0, gripper.max_width]``.
2. Grasp NMS over the whole frame, in descending score order.
3. Assign each surviving grasp to the object that owns the nearest model point.
4. Per object, keep the ``per_object_top`` highest-scoring grasps.
5. Keep grasps whose score is at least the score at global rank ``top_k``
   (so ties with that score are all kept).
6. Collision and emptiness against the ground-truth models (plus the table), cropped
   to the assigned object's bounding box grown by ``crop_margin``.
7. Force closure on the assigned object's signed-distance grid for grasps that
   passed 6; the result is the smallest friction that holds, or ``-1``.
8. Rank by predicted score; ``precision[k, f]`` is the share of the first ``k + 1``
   grasps with ``0 < friction score <= frictions[f]``. Missing ranks count as failures.

**Tie rule.** Wherever grasps are ranked by score (steps 2, 4, 8), equal scores are
ordered by their row in the prediction file, earlier row first. The reference sorts
with unstable algorithms, so its result for tied input depends on platform and
library version; for tie-free input both agree.

**Empty input.** A prediction with no rows scores an all-zero table. The reference
raises ``IndexError`` there; the zero table is what every maintained fork returns.

**Malformed input.** Non-finite values are rejected. A grasp that reaches step 7 with
a closing axis that is not a unit vector (within ``axis_tolerance``) is rejected with
the same condition under which the reference raises ``ValueError``.

Numeric fine print reproduced from the reference: widths are clipped in the file's
own precision and everything afterwards runs in float64; object poses are float32;
the object-frame grasp centre uses ``np.linalg.inv`` of the float32 pose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .arithmetic import matmul3, norm3, transform_points
from .cache import ObjectAsset
from .collision import crop_to_box, gripper_collision
from .force_closure import NO_FORCE_CLOSURE, friction_scores
from .nms import grasp_nms
from .protocol import Protocol

__all__ = [
    "GRASP_COLUMNS",
    "FrameResult",
    "evaluate_frame",
    "nearest_point",
    "precision_table",
    "prepare_prediction",
    "select_grasps",
]

#: ``[score, width, height, depth, R(9, row-major), t(3), object_id]``
GRASP_COLUMNS = 17
_SCORE, _WIDTH, _DEPTH = 0, 1, 3
_ROTATION = slice(4, 13)
_TRANSLATION = slice(13, 16)
#: Two model points this close in distance (relative) are compared exactly, not by the tree.
_NEAREST_SLACK = 1e-9


@dataclass(frozen=True, eq=False)
class FrameResult:
    """Scores of one frame together with every intermediate decision.

    Row indices refer to the prediction array as read from disk.

    Attributes:
        precision: ``(top_k, num_friction)`` precision-at-k table.
        selected: ``(K,)`` rows surviving suppression, best first.
        owner: ``(K,)`` position (in the scene's object list) of each selected grasp's object.
        kept: ``(J,)`` rows left after the per-object and global cuts, grouped by object
            in scene order and best first within an object.
        kept_owner: ``(J,)`` object position of each kept grasp.
        collision: ``(J,)`` true when the grasp collides or is empty.
        empty: ``(J,)`` true when too few points lie between the fingers.
        friction: ``(J,)`` smallest friction with force closure, ``-1`` if none or colliding.
        ranking: ``(J,)`` permutation of ``kept`` into final rank order.
    """

    precision: np.ndarray
    selected: np.ndarray
    owner: np.ndarray
    kept: np.ndarray
    kept_owner: np.ndarray
    collision: np.ndarray
    empty: np.ndarray
    friction: np.ndarray
    ranking: np.ndarray


def prepare_prediction(prediction: np.ndarray, max_width: float) -> np.ndarray:
    """Validate a ``(G, 17)`` prediction and return it as float64 with clipped widths.

    The clip is applied in the array's own floating-point type before widening,
    because that is where the reference applies it: a float32 file is clipped to
    ``float32(max_width)``, which is not the float64 value of ``max_width``.
    """
    prediction = np.asarray(prediction)
    if prediction.ndim != 2 or prediction.shape[1] != GRASP_COLUMNS:
        raise ValueError(
            f"a prediction must have shape (G, {GRASP_COLUMNS}), got {prediction.shape}"
        )
    if prediction.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"a prediction must be stored as float32 or float64, got {prediction.dtype}"
        )
    if not np.isfinite(prediction[:, :16]).all():
        bad = int((~np.isfinite(prediction[:, :16])).any(axis=1).sum())
        raise ValueError(f"{bad} grasp(s) contain NaN or Inf")
    width = prediction[:, _WIDTH].copy()
    width[width < 0] = 0
    limit = prediction.dtype.type(max_width)
    width[width > limit] = limit
    grasps = prediction.astype(np.float64)
    grasps[:, _WIDTH] = width
    return grasps


def nearest_point(queries: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of, and distance to, the nearest of ``points`` for each query.

    Equivalent to the brute-force ``argmin`` over ``sqrt((dx² + dy²) + dz²)`` with the
    lowest index winning ties. A k-d tree finds the answer; whenever the two nearest
    candidates are closer in distance than rounding could resolve, that query is
    decided by the brute-force formula itself.
    """
    if len(points) == 0:
        raise ValueError("cannot assign grasps: the scene has no model points")
    index = np.zeros(len(queries), dtype=np.int64)
    if len(points) > 1 and len(queries):
        distance, candidate = cKDTree(points).query(queries, k=2)
        index = candidate[:, 0].astype(np.int64)
        unsure = distance[:, 1] <= distance[:, 0] * (1.0 + _NEAREST_SLACK) + 1e-300
        for row in np.flatnonzero(unsure):
            index[row] = int(np.argmin(norm3(queries[row] - points)))
    return index, norm3(queries - points[index])


def select_grasps(
    grasps: np.ndarray,
    selected: np.ndarray,
    owner: np.ndarray,
    num_objects: int,
    protocol: Protocol,
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 4 and 5: per-object top grasps, then the global score cut.

    Args:
        grasps: ``(G, 17)`` float64.
        selected: ``(K,)`` rows surviving suppression, best first.
        owner: ``(K,)`` object position per selected grasp.
        num_objects: Number of objects in the scene.
        protocol: Supplies ``per_object_top`` and ``top_k``.

    Returns:
        ``kept`` rows grouped by object and their ``kept_owner``.
    """
    groups = [
        selected[owner == position][: protocol.per_object_top] for position in range(num_objects)
    ]
    candidates = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
    if len(candidates) == 0:
        return candidates, np.zeros(0, dtype=np.int64)
    scores = np.sort(grasps[candidates, _SCORE])[::-1]
    threshold = scores[min(protocol.top_k - 1, len(scores) - 1)]
    kept, kept_owner = [], []
    for position, group in enumerate(groups):
        group = group[grasps[group, _SCORE] >= threshold]
        kept.append(group)
        kept_owner.append(np.full(len(group), position, dtype=np.int64))
    return np.concatenate(kept), np.concatenate(kept_owner)


def precision_table(ranked_friction: np.ndarray, protocol: Protocol) -> np.ndarray:
    """Step 8: ``(top_k, num_friction)`` precision of friction scores given in rank order."""
    hits = np.zeros((protocol.top_k, len(protocol.frictions)), dtype=np.int64)
    head = np.asarray(ranked_friction, dtype=np.float64)[: protocol.top_k]
    limits = np.array(protocol.frictions, dtype=np.float64)
    hits[: len(head)] = (head[:, None] <= limits[None, :]) & (head[:, None] > 0)
    return np.cumsum(hits, axis=0) / np.arange(1, protocol.top_k + 1)[:, None]


def _object_frame_grasps(
    grasps: np.ndarray, pose: np.ndarray, protocol: Protocol
) -> tuple[np.ndarray, np.ndarray]:
    """Fingertip midpoints and unit closing axes of camera-frame grasps in one object's frame."""
    rotation = grasps[:, _ROTATION].reshape(-1, 3, 3)
    depth = grasps[:, _DEPTH]
    # Rotate the gripper-frame point (depth, 0, 0); the zero terms are kept so that the
    # sum is rounded exactly as a full matrix-vector product.
    tip = (rotation[:, :, 0] * depth[:, None] + rotation[:, :, 1] * 0.0) + rotation[:, :, 2] * 0.0
    center = transform_points(np.linalg.inv(pose), tip + grasps[:, _TRANSLATION])
    in_object = matmul3(pose[:3, :3].T.astype(np.float64)[None], rotation)
    axis = in_object[:, :, 1]
    if protocol.normalize_closing_axis:
        # The reference of this protocol rescales first and still normalises again below.
        axis = axis / norm3(axis)[:, None]
    length = norm3(axis)
    off = np.abs(length - 1.0) > protocol.axis_tolerance
    if off.any():
        worst = float(np.abs(length - 1.0).max())
        raise ValueError(
            f"{int(off.sum())} grasp(s) have a closing axis (second rotation column) whose "
            f"norm differs from 1 by up to {worst:.2e} (tolerance {protocol.axis_tolerance:.0e}); "
            "the force-closure test, like the reference evaluator, requires unit axes"
        )
    return center, axis / length[:, None]


def evaluate_frame(
    prediction: np.ndarray,
    objects: Sequence[ObjectAsset],
    poses: Sequence[np.ndarray],
    table: np.ndarray | None,
    protocol: Protocol,
) -> FrameResult:
    """Score one frame.

    Args:
        prediction: ``(G, 17)`` grasps in the camera frame, float32 or float64.
        objects: Scoring data of the scene's objects, in annotation order.
        poses: Per object the ``(4, 4)`` object-to-camera pose of this frame.
        table: ``(T, 3)`` table points in the camera frame, or ``None``.
        protocol: Constants of the benchmark.
    """
    if len(objects) != len(poses):
        raise ValueError(f"{len(objects)} objects but {len(poses)} poses")
    grasps = prepare_prediction(prediction, protocol.gripper.max_width)
    empty_index = np.zeros(0, dtype=np.int64)
    if len(grasps) == 0:
        return FrameResult(
            precision=precision_table(np.zeros(0), protocol),
            selected=empty_index,
            owner=empty_index,
            kept=empty_index,
            kept_owner=empty_index,
            collision=np.zeros(0, dtype=bool),
            empty=np.zeros(0, dtype=bool),
            friction=np.zeros(0),
            ranking=empty_index,
        )

    posed = [transform_points(pose, asset.model_points) for asset, pose in zip(objects, poses)]
    scene = np.concatenate(posed)
    segment = np.repeat(np.arange(len(posed)), [len(points) for points in posed])

    rows = np.arange(len(grasps))
    if protocol.pre_nms_max_distance is not None:
        _, distance = nearest_point(grasps[:, _TRANSLATION], scene)
        rows = rows[distance < protocol.pre_nms_max_distance]

    order = grasp_nms(
        grasps[rows, _TRANSLATION],
        grasps[rows, _ROTATION].reshape(-1, 3, 3),
        grasps[rows, _SCORE],
        translation_threshold=protocol.nms_translation,
        rotation_threshold=protocol.nms_rotation,
    )
    selected = rows[order]
    owner = segment[nearest_point(grasps[selected, _TRANSLATION], scene)[0]]
    kept, kept_owner = select_grasps(grasps, selected, owner, len(posed), protocol)

    collision_scene = scene if table is None else np.concatenate([scene, table])
    collision = np.zeros(len(kept), dtype=bool)
    empty = np.zeros(len(kept), dtype=bool)
    friction = np.full(len(kept), NO_FORCE_CLOSURE)
    for position in np.unique(kept_owner):
        members = np.flatnonzero(kept_owner == position)
        mine = grasps[kept[members]]
        workspace = crop_to_box(collision_scene, posed[position], protocol.crop_margin)
        collision[members], empty[members] = gripper_collision(
            mine[:, _TRANSLATION],
            mine[:, _ROTATION].reshape(-1, 3, 3),
            mine[:, _DEPTH],
            mine[:, _WIDTH],
            workspace,
            gripper=protocol.gripper,
            empty_threshold=protocol.empty_threshold,
        )
        # The reference converts every non-empty grasp to the object frame, so a bad
        # closing axis is an error even when the grasp collides.
        filled = ~empty[members]
        if not filled.any():
            continue
        center, axis = _object_frame_grasps(mine[filled], poses[position], protocol)
        free = ~collision[members][filled]
        if free.any():
            target = members[filled][free]
            friction[target] = friction_scores(
                objects[position].sdf,
                center[free],
                axis[free],
                mine[filled][free, _WIDTH],
                protocol.frictions,
            )

    ranking = np.lexsort((kept, -grasps[kept, _SCORE]))
    return FrameResult(
        precision=precision_table(friction[ranking], protocol),
        selected=selected,
        owner=owner,
        kept=kept,
        kept_owner=kept_owner,
        collision=collision,
        empty=empty,
        friction=friction,
        ranking=ranking,
    )
