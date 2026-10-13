"""Where an average precision comes from: an exact split of AP by object and by cause.

A GraspNet-style frame score is the mean of a precision-at-k table. Written out, the
precision of one friction column is a weighted sum over the ranked grasps::

    AP_f = (1 / K) * sum_k precision[k, f] = sum_i  w_i * hit[i, f],
    w_i  = (1 / K) * sum_{k = i..K} 1 / k,          sum_i w_i = 1

so every rank ``i`` owns the share ``w_i`` of the frame's AP: rank 1 owns 9 % of it for
``K = 50``, rank 50 owns 0.04 %. A rank earns its share when its grasp succeeds at the
friction of the column and loses it otherwise, for exactly one reason:

``collision``         the gripper collides with the scene (ground-truth models);
``empty``             too few points lie between the fingers;
``no_force_closure``  the grasp is free but not force-closed at this friction;
``unfilled``          the frame has fewer ranked grasps than ``K``.

Per column, ``gain + collision + empty + no_force_closure + unfilled = 1`` for every
frame. Summing over frames and dividing by their number gives the same identity for a
split, with ``gain`` equal to the split's AP of that column. Because a ranked grasp
belongs to one object, the shares add up per object as well.

This says nothing new about the score; it attributes it. It is computed from the
decisions the evaluator already made, so it cannot disagree with the precision table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Tally", "frame_tallies", "rank_weights"]


def rank_weights(top_k: int) -> np.ndarray:
    """``(top_k,)`` share of a frame's AP that each rank owns; sums to one."""
    return np.cumsum((1.0 / np.arange(1, top_k + 1))[::-1])[::-1] / top_k


@dataclass
class Tally:
    """AP shares of a set of ranked grasps, summed over frames.

    Attributes:
        frames: Frames that contributed.
        ranked: Grasps that held one of the ``top_k`` ranks.
        gain: ``(F,)`` share earned, per friction column.
        collision: Share lost to collisions; the same for every column.
        empty: Share lost to grasps with nothing between the fingers.
        no_force_closure: ``(F,)`` share lost by collision-free grasps without force
            closure at the column's friction.
        unfilled: Share lost because ranks were not filled. Kept per frame set only; an
            unfilled rank belongs to no object.
    """

    frames: int = 0
    ranked: int = 0
    gain: np.ndarray | None = None
    collision: float = 0.0
    empty: float = 0.0
    no_force_closure: np.ndarray | None = None
    unfilled: float = 0.0

    def add(self, other: Tally) -> Tally:
        """Add ``other`` to this tally, in place."""
        self.frames += other.frames
        self.ranked += other.ranked
        self.collision += other.collision
        self.empty += other.empty
        self.unfilled += other.unfilled
        for name in ("gain", "no_force_closure"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if theirs is not None:
                setattr(self, name, theirs.copy() if mine is None else mine + theirs)
        return self

    def to_dict(self, columns: int, *, with_unfilled: bool) -> dict[str, Any]:
        """Plain data for a result record; ``columns`` is the number of friction columns."""

        def per_column(values: np.ndarray | None) -> list[float]:
            # An object that never held a rank has earned and lost nothing in any column.
            return [0.0] * columns if values is None else [float(v) for v in values]

        out: dict[str, Any] = {
            "frames": int(self.frames),
            "ranked": int(self.ranked),
            "gain": per_column(self.gain),
            "collision": float(self.collision),
            "empty": float(self.empty),
            "no_force_closure": per_column(self.no_force_closure),
        }
        if with_unfilled:
            out["unfilled"] = float(self.unfilled)
        return out


def frame_tallies(
    *,
    owner: np.ndarray,
    collision: np.ndarray,
    empty: np.ndarray,
    friction: np.ndarray,
    ranking: np.ndarray,
    object_ids: Sequence[int],
    top_k: int,
    frictions: Sequence[float],
) -> tuple[Tally, dict[int, Tally]]:
    """Split the AP of one frame.

    Args:
        owner: ``(J,)`` position, in ``object_ids``, of the object each kept grasp belongs to.
        collision: ``(J,)`` true when the grasp collides or is empty.
        empty: ``(J,)`` true when too few points lie between the fingers.
        friction: ``(J,)`` smallest friction with force closure; not positive if none.
        ranking: ``(J,)`` permutation of the kept grasps into rank order.
        object_ids: Dataset ids of the scene's objects, by position.
        top_k: Number of ranks in the precision table.
        frictions: Friction of each column, ascending.

    Returns:
        The frame's tally and one tally per object id that has a ranked grasp. The
        object tallies have ``frames == 0``: the caller counts in how many frames an
        object was present, ranked or not.
    """
    weights = rank_weights(top_k)
    limits = np.asarray(frictions, dtype=np.float64)
    head = np.asarray(ranking)[:top_k]
    share = weights[: len(head)]
    is_empty = np.asarray(empty, dtype=bool)[head]
    collides = np.asarray(collision, dtype=bool)[head] & ~is_empty
    free = ~is_empty & ~collides
    value = np.asarray(friction, dtype=np.float64)[head]
    hit = free[:, None] & (value[:, None] > 0) & (value[:, None] <= limits[None, :])
    miss = free[:, None] & ~hit
    positions = np.asarray(owner)[head]

    def tally(rows: np.ndarray) -> Tally:
        return Tally(
            ranked=int(rows.sum()),
            gain=(share[rows, None] * hit[rows]).sum(axis=0),
            collision=float(share[rows][collides[rows]].sum()),
            empty=float(share[rows][is_empty[rows]].sum()),
            no_force_closure=(share[rows, None] * miss[rows]).sum(axis=0),
        )

    whole = tally(np.ones(len(head), dtype=bool))
    whole.frames = 1
    whole.unfilled = float(weights[len(head) :].sum())
    objects: dict[int, Tally] = {}
    for position in np.unique(positions):
        part = tally(positions == position)
        object_id = int(object_ids[int(position)])
        objects[object_id] = objects[object_id].add(part) if object_id in objects else part
    return whole, objects
