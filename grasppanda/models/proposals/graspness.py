"""Graspness proposal: score every scene point, keep the graspable ones, spread seeds.

One linear layer predicts, per scene point, two objectness logits and a graspness
value. Points classified as object whose graspness exceeds a threshold are candidates;
furthest point sampling over the candidates of each frame yields a fixed number of
seeds. Seeds are selected from the *predictions* in training as well as in inference,
as in the Graspness reference implementation.

The stage owns the seed count and the graspness threshold
(``docs/design/ARCHITECTURE.md`` §8.6) and its own targets and loss.

Padding policy. Every frame yields exactly ``num_seeds`` seeds so that later stages can
work on dense ``(B, M, …)`` tensors. A frame with fewer distinct candidates than
``num_seeds`` is padded with repeats of its first seed (which is what the reference
sampling kernel produces once all candidates are used up); a frame with no candidate
at all is padded with its first scene point. Padding is marked ``valid = False`` and
is never turned into a grasp by the detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Loss, Proposal
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import INJECTED, LOSSES, PROPOSALS, Cfg
from grasppanda.structures import (
    ContractError,
    Observation,
    PointFeatures,
    PointSet,
    SeedSet,
    Supervision,
)

__all__ = ["GraspnessFPS", "PointScores"]


@dataclass(eq=False)
class PointScores:
    """Per-point predictions of a proposal stage, bound to the scene point set.

    Attributes:
        points: The scene point set the rows refer to.
        objectness: ``(N, 2)`` logits, class 1 = the point lies on an object.
        graspness: ``(N,)`` predicted point-wise graspness.
        num_candidates: ``(B,)`` points per frame that passed the selection rule.
    """

    points: PointSet
    objectness: Tensor
    graspness: Tensor
    num_candidates: Tensor | None = None


def _first_occurrence(index: Tensor) -> Tensor:
    """``(B, M)`` mask: true where a value appears for the first time in its row."""
    same = index[:, :, None] == index[:, None, :]
    return ~same.tril(diagonal=-1).any(dim=-1)


@PROPOSALS.register()
class GraspnessFPS(Proposal):
    """Objectness + graspness per point, threshold, furthest point sampling.

    Args:
        num_seeds: Seeds per frame.
        graspness_threshold: A candidate needs a predicted graspness above this value
            (and an objectness arg-max of "object").
        fps_skip_near_origin: Replicate the PointNet++ CUDA sampling kernel, which never
            picks a point whose squared norm is ≤ 1e-3. Reference value: true.
        fps_tie_break: Which candidate wins when several are equally far from the seeds
            chosen so far. ``pointnet2_cuda`` (reference) reproduces the order of that
            kernel, ``lowest_index`` takes the first. Depth-camera clouds lie on a
            lattice, so exact ties do occur. The kernel order is reproduced exactly for
            frames with at least 512 candidates; below that the kernel runs fewer
            threads than the batched call here assumes.
        objectness_loss: Loss on the objectness logits of every point. ``None`` selects
            the reference, ``CrossEntropy`` with weight 1.
        graspness_loss: Loss on the graspness of points labelled as object. ``None``
            selects the reference, ``SmoothL1`` with weight 10.
        in_channels: Channels of the scene features; injected by the detector.
    """

    requires = "dense"
    status = "tested"
    source = "GSNet (Wang et al., ICCV 2021)"
    seed_scores = ("graspness", "objectness")

    def __init__(
        self,
        num_seeds: int = 1024,
        graspness_threshold: float = 0.1,
        fps_skip_near_origin: bool = True,
        fps_tie_break: Literal["lowest_index", "pointnet2_cuda"] = "pointnet2_cuda",
        objectness_loss: Cfg[Loss] | None = None,
        graspness_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if num_seeds < 1:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}")
        if fps_tie_break not in ("lowest_index", "pointnet2_cuda"):
            raise ValueError(f"unknown fps_tie_break '{fps_tie_break}'")
        self.num_seeds = num_seeds
        self.graspness_threshold = graspness_threshold
        self.fps_skip_near_origin = fps_skip_near_origin
        self.fps_tie_break = fps_tie_break
        self.in_channels = in_channels
        #: Rows 0-1: objectness logits; row 2: graspness.
        self.score = nn.Linear(in_channels, 3)
        self.objectness_loss = LOSSES.build(
            objectness_loss if objectness_loss is not None else {"type": "CrossEntropy"}
        )
        self.graspness_loss = LOSSES.build(
            graspness_loss if graspness_loss is not None else {"type": "SmoothL1", "weight": 10.0}
        )

    # --------------------------------------------------------------------- forward

    def forward(
        self,
        feats: PointFeatures,
        observation: Observation,
        seed_index: Tensor | None = None,
    ) -> tuple[SeedSet, PointScores]:
        """Score the scene points and select the seeds.

        Args:
            feats: Scene features. They must live on ``observation.points`` itself.
            observation: The batch being processed.
            seed_index: ``(B, num_seeds)`` indices into the scene point set. Given, they
                replace the selection (all are treated as valid); this replays a seed
                set recorded elsewhere, for instance by a reference implementation.

        Returns:
            The seeds, and the per-point scores the loss needs.
        """
        name = type(self).__name__
        points = observation.points
        if points is None:
            raise ContractError(f"{name} needs observation.points")
        feats.expect(points, user=f"{name} (requires '{self.requires}' scene features)")
        if feats.channels != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel features, got {feats.channels}"
            )
        prediction = self.score(feats.feats)
        scores = PointScores(points, prediction[:, :2], prediction[:, 2])
        if seed_index is None:
            index, valid, scores.num_candidates = self._select(scores)
        else:
            index, valid = self._replay(points, seed_index)
        flat = index.reshape(-1)
        offsets = torch.arange(index.shape[0] + 1, device=flat.device) * self.num_seeds
        seeds = SeedSet(
            points=points.subset(flat, offsets),
            feats=feats.feats[flat],
            scores={
                "graspness": scores.graspness.detach()[flat],
                "objectness": scores.objectness.detach().softmax(dim=-1)[flat, 1],
            },
            valid=valid.reshape(-1),
            source="graspness_fps" if seed_index is None else "replayed",
        )
        return seeds, scores

    def candidates(self, scores: PointScores) -> Tensor:
        """``(N,)`` mask of the scene points that may become seeds."""
        is_object = scores.objectness.argmax(dim=-1) == 1
        return is_object & (scores.graspness > self.graspness_threshold)

    @torch.no_grad()
    def _select(self, scores: PointScores) -> tuple[Tensor, Tensor, Tensor]:
        points = scores.points
        mask = self.candidates(scores)
        offsets = _offsets(points)
        chosen = []
        for start, stop in pairwise(offsets):
            if stop == start:
                raise ContractError(f"{type(self).__name__} received a frame without points")
            chosen.append(torch.nonzero(mask[start:stop]).flatten() + start)
        counts = torch.tensor([len(c) for c in chosen], device=mask.device)
        width = max(int(counts.max()), self.num_seeds)
        # Pad every frame with repeats of its first candidate. A repeat has distance zero
        # to the selected set from the first step on, so it can only be picked once every
        # candidate is used up, and then position 0 (the original) wins under either tie
        # order. Frames are therefore sampled in one batched call.
        padded = torch.stack(
            [
                torch.cat([c, c[:1].expand(width - len(c))])
                if len(c)
                else torch.full((width,), start, device=mask.device)
                for c, start in zip(chosen, offsets[:-1])
            ]
        )
        picks = furthest_point_sample(
            points.coords[padded],
            self.num_seeds,
            skip_near_origin=self.fps_skip_near_origin,
            tie_break=self.fps_tie_break,
        )
        index = torch.gather(padded, 1, picks)
        valid = _first_occurrence(picks) & (counts[:, None] > 0)
        return index, valid, counts

    def _replay(self, points: PointSet, seed_index: Tensor) -> tuple[Tensor, Tensor]:
        if seed_index.ndim != 2 or tuple(seed_index.shape) != (
            points.batch_size,
            self.num_seeds,
        ):
            raise ContractError(
                f"seed_index must be ({points.batch_size}, {self.num_seeds}) indices into the "
                f"scene point set, got {tuple(seed_index.shape)}"
            )
        offsets = _offsets(points)
        index = seed_index.to(points.coords.device).long()
        low = torch.as_tensor(offsets[:-1], device=index.device)[:, None]
        high = torch.as_tensor(offsets[1:], device=index.device)[:, None]
        if ((index < low) | (index >= high)).any():
            raise ContractError("seed_index refers to points outside the frame of its row")
        return index, torch.ones_like(index, dtype=torch.bool)

    # ------------------------------------------------------------------------ loss

    def loss(self, scores: PointScores, supervision: Supervision) -> dict[str, Tensor]:
        """Objectness and graspness losses against the per-point labels.

        Reads ``supervision.point_labels['objectness']`` (``(N,)``, non-zero = object)
        and ``['graspness']`` (``(N,)`` or ``(N, 1)``), both aligned with the scene
        point set. Graspness is supervised on points labelled as object only.
        """
        count = len(scores.points)
        labels = {}
        for key in ("objectness", "graspness"):
            if key not in supervision.point_labels:
                raise ContractError(
                    f"{type(self).__name__}.loss needs supervision.point_labels['{key}'] aligned "
                    f"with observation.points; available: {sorted(supervision.point_labels)}"
                )
            label = supervision.point_labels[key]
            if label.shape[0] != count or label.numel() != count:
                raise ContractError(
                    f"point label '{key}' has shape {tuple(label.shape)}, but the scene has "
                    f"{count} points; labels must be aligned with observation.points"
                )
            labels[key] = label.reshape(count)
        is_object = labels["objectness"] != 0
        target = is_object.long()
        predicted = scores.objectness.detach().argmax(dim=-1) == 1
        correct = predicted == is_object
        return {
            "loss/objectness": self.objectness_loss(scores.objectness, target),
            "loss/graspness": self.graspness_loss(
                scores.graspness, labels["graspness"].to(scores.graspness.dtype), is_object
            ),
            "metric/objectness_accuracy": correct.float().mean(),
            "metric/objectness_precision": _ratio(correct & predicted, predicted),
            "metric/objectness_recall": _ratio(correct & is_object, is_object),
        }


def _offsets(points: PointSet) -> list[int]:
    if points.batch_offsets is None:
        return [0, len(points)]
    return [int(v) for v in points.batch_offsets.tolist()]


def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator.sum().float() / denominator.sum().clamp(min=1).float()
