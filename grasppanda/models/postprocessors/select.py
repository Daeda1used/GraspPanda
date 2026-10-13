"""Selection by score and by spread: top-k, score threshold, furthest grasp sampling.

These read nothing but the grasps, so they follow every detector. None of them changes
a pose or a score.
"""

from __future__ import annotations

from typing import Literal

import torch

from grasppanda.models.base import PostProcessor
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation

__all__ = ["FurthestGraspSampling", "ScoreThreshold", "TopK"]


@POSTPROCESSORS.register()
class TopK(PostProcessor):
    """Keep the ``k`` best-scoring grasps, best first.

    Ties keep their input order. AnyGrasp hands "the top-100 ranking grasp poses" to
    its collision check; the GraspNet-1Billion evaluator scores the best 50 per frame.

    Args:
        k: Number of grasps to keep; a frame with fewer keeps all of them.
    """

    status = "tested"
    source = "Generic; the top-k step of AnyGrasp (Fang et al., T-RO, arXiv 2212.08333, section IV-D) and of GPD"
    primary = "k"

    def __init__(self, k: int = 50) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        return grasps.sort_by_score()[: self.k]


@POSTPROCESSORS.register()
class ScoreThreshold(PostProcessor):
    """Keep grasps whose score reaches a threshold.

    A score means what ``GraspGroup.score_kind`` says and its scale differs between
    methods (GSNet's is unbounded, EconomicGrasp's is a sum of logits, HGGD's lies in
    ``[0, 1]``), so an ``absolute`` threshold is tuned per method. ``quantile`` and
    ``relative`` need no such tuning.

    Args:
        threshold: The limit, read according to ``mode``.
        mode: ``absolute``: keep ``score >= threshold``. ``quantile``: keep the best
            ``1 - threshold`` share of the frame (``0.8`` keeps the best fifth).
            ``relative``: keep ``score >= min + threshold * (max - min)`` of the frame.
        min_keep: Keep at least this many of the best grasps, whatever their score, so
            that a robot is not left without a candidate.
    """

    status = "tested"
    source = "Generic; `choose_fn: better_than_threshold` of 6-DOF GraspNet (ICCV 2019)"
    primary = "threshold"

    def __init__(
        self,
        threshold: float = 0.5,
        mode: Literal["absolute", "quantile", "relative"] = "absolute",
        min_keep: int = 0,
    ) -> None:
        if mode not in ("absolute", "quantile", "relative"):
            raise ValueError(f"mode must be 'absolute', 'quantile' or 'relative', got {mode!r}")
        if mode != "absolute" and not 0 <= threshold <= 1:
            raise ValueError(f"with mode '{mode}' the threshold is a share in [0, 1]: {threshold}")
        if min_keep < 0:
            raise ValueError(f"min_keep must not be negative, got {min_keep}")
        self.threshold = threshold
        self.mode = mode
        self.min_keep = min_keep

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        score = grasps.score
        if self.mode == "absolute":
            limit = score.new_tensor(self.threshold)
        elif self.mode == "quantile":
            limit = torch.quantile(score.double(), self.threshold).to(score.dtype)
        else:
            low, high = score.min(), score.max()
            limit = low + self.threshold * (high - low)
        keep = score >= limit
        if self.min_keep and int(keep.sum()) < self.min_keep:
            best = torch.argsort(score, descending=True, stable=True)[: self.min_keep]
            keep[best] = True
        return grasps[keep]


@POSTPROCESSORS.register()
class FurthestGraspSampling(PostProcessor):
    """Pick grasps that are spread over the scene: furthest point sampling of grasp points.

    A detector's best grasps often sit on one object. This step first takes the best
    ``candidates`` grasps by score, then selects ``num`` of them whose grasp points are
    mutually far apart, starting from the best grasp. The result is sorted by score.
    It is the diversity step of Contact-GraspNet's ``select_grasps`` (furthest point
    sampling among confident grasps), written here on the toolbox's own sampling
    operator.

    Args:
        num: Number of grasps to return.
        candidates: Only the best-scoring ``candidates`` grasps take part; ``None``
            lets every grasp take part (then a poor but remote grasp can be selected).
    """

    status = "tested"
    source = (
        "Generic, on grasppanda.ops.furthest_point_sample; the idea is the grasp "
        "selection of Contact-GraspNet (Sundermeyer et al., ICRA 2021)"
    )
    primary = "num"

    def __init__(self, num: int = 50, candidates: int | None = 300) -> None:
        if num < 1 or (candidates is not None and candidates < num):
            raise ValueError(
                f"num must be at least 1 and candidates at least num; got {num} and {candidates}"
            )
        self.num = num
        self.candidates = candidates

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        grasps = grasps.sort_by_score()
        if self.candidates is not None:
            grasps = grasps[: self.candidates]
        if len(grasps) <= self.num:
            return grasps
        picked = furthest_point_sample(grasps.translation.float(), self.num)
        return grasps[torch.sort(picked).values]
