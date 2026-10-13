"""SWAD head bundle: Score and Width over the in-plane Angle × Depth grid.

For every seed the head regresses two ``num_angle × num_depth`` grids from the crop
feature: a grasp quality score and a gripper width per (in-plane angle, depth) cell.
Decoding keeps the best-scoring cell of each seed, so every seed yields one grasp.

The bundle owns the prediction layers, the target encoder, the loss and the decoder,
all tied to one :class:`AngleDepthCodec`. None of the four can be swapped alone.

Targets (reference behaviour, each a named option):

* ``score_target = log_friction_batch_norm``: a label friction ``μ`` (lower is better)
  becomes ``log(μ_max / μ) / (log(μ_max / μ_min) + 1e-6)``, where the extrema are taken
  over *all supervised cells of the current batch*. The target of a given grasp
  therefore depends on what else is in the batch. ``log_friction_fixed_range`` uses
  the constant ``friction_range`` instead and is the controlled variant.
* Cells without a usable grasp (``μ ≤ 0``, or a width beyond the gripper) get score
  target 0. The width is supervised only where the score target is positive, which
  also excludes the worst friction level present (its target is exactly 0).
* Width targets are metres × ``width_target_scale``; the decoder additionally applies
  ``width_decode_gain`` (see the codec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.base import HeadBundle, Loss
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import HEADS, INJECTED, LOSSES, Cfg
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    GraspGroup,
    Gripper,
    GroupFeatures,
    SeedSet,
)

from .codec import AngleDepthCodec

__all__ = ["SWADHead", "SWADPredictions", "SWADTargets"]

#: Guard of the score normalisation, as in the reference implementation.
_SCORE_EPS = 1e-6


@dataclass(eq=False)
class SWADPredictions:
    """Raw outputs of :class:`SWADHead`; they keep gradients.

    Attributes:
        score: ``(M, num_angle, num_depth)`` predicted grasp quality.
        width: ``(M, num_angle, num_depth)`` predicted width in regression units.
    """

    score: Tensor
    width: Tensor


@dataclass(eq=False)
class SWADTargets:
    """Regression targets on the same grids as :class:`SWADPredictions`.

    Attributes:
        score: ``(M, num_angle, num_depth)``; 0 where there is no usable grasp.
        width: ``(M, num_angle, num_depth)`` in regression units.
        score_mask: ``(M, num_angle, num_depth)`` cells that supervise the score.
        width_mask: ``(M, num_angle, num_depth)`` cells that supervise the width.
    """

    score: Tensor
    width: Tensor
    score_mask: Tensor
    width_mask: Tensor


@HEADS.register()
class SWADHead(HeadBundle):
    """Score and width grids per seed, regressed; one decoded grasp per seed.

    Args:
        num_angle: In-plane angle bins of the grid (owned by the codec).
        num_depth: Depth bins of the grid (owned by the codec).
        depth_step: Metres between depth bins.
        width_target_scale: Width targets are metres times this factor.
        width_decode_gain: Factor on predicted widths in the decoder only. Reference
            value 1.2; 1.0 decodes exactly what was regressed.
        score_target: How label frictions become score targets, see the module
            docstring. Reference value ``log_friction_batch_norm``.
        friction_range: ``(μ_min, μ_max)`` used by ``log_friction_fixed_range``. The
            default is the range of the GraspNet-1Billion labels.
        score_loss: Loss on the score grid of supervised seeds. ``None`` selects the
            reference, ``SmoothL1`` with weight 15.
        width_loss: Loss on the width grid where the score target is positive.
            ``None`` selects the reference, ``SmoothL1`` with weight 10.
        in_channels: Channels of the crop features; injected by the detector.
        gripper: Gripper the decoded grasps refer to; injected by the detector. Its
            ``max_width`` bounds usable labels and decoded widths.
    """

    in_axes = ("seed", "channel")
    score_kind = "predicted_quality"
    status = "tested"
    source = "GSNet (Wang et al., ICCV 2021)"
    loss_terms: ClassVar[dict[str, str]] = {"score_loss": "unit", "width_loss": "value"}

    def __init__(
        self,
        num_angle: int = 12,
        num_depth: int = 4,
        depth_step: float = 0.01,
        width_target_scale: float = 10.0,
        width_decode_gain: float = 1.2,
        score_target: Literal[
            "log_friction_batch_norm", "log_friction_fixed_range"
        ] = "log_friction_batch_norm",
        friction_range: tuple[float, float] = (0.1, 1.0),
        score_loss: Cfg[Loss] | None = None,
        width_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
        gripper: Gripper = INJECTED,
    ) -> None:
        super().__init__()
        if not isinstance(gripper, Gripper):
            raise TypeError(f"gripper must be a Gripper, got {type(gripper).__name__}")
        if score_target not in ("log_friction_batch_norm", "log_friction_fixed_range"):
            raise ValueError(f"unknown score_target '{score_target}'")
        low, high = friction_range
        if not 0 < low < high:
            raise ValueError(
                f"friction_range must be (min, max) with 0 < min < max, got {low}, {high}"
            )
        self.codec = AngleDepthCodec(
            num_angle, num_depth, depth_step, width_target_scale, width_decode_gain
        )
        self.score_target = score_target
        self.friction_range = (float(low), float(high))
        self.in_channels = in_channels
        self.gripper = gripper
        self.hidden = nn.Linear(in_channels, in_channels)
        #: Output channels: [score grid | width grid], each flattened by the codec.
        self.output = nn.Linear(in_channels, 2 * self.codec.num_cells)
        self.score_loss = LOSSES.build(
            score_loss if score_loss is not None else {"type": "SmoothL1", "weight": 15.0}
        )
        self.width_loss = LOSSES.build(
            width_loss if width_loss is not None else {"type": "SmoothL1", "weight": 10.0}
        )

    # --------------------------------------------------------------------- forward

    def forward(self, groups: GroupFeatures) -> SWADPredictions:
        name = type(self).__name__
        groups.expect(self.in_axes, user=name)
        if groups.size("channel") != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel crop features, got "
                f"{groups.size('channel')}"
            )
        output = self.output(F.relu(self.hidden(groups.feats)))
        grids = self.codec.unflatten(output.unflatten(-1, (2, self.codec.num_cells)))
        return SWADPredictions(score=grids[:, 0], width=grids[:, 1])

    # --------------------------------------------------------------------- targets

    @torch.no_grad()
    def encode_targets(self, matched: MatchedLabels) -> SWADTargets:
        """Score and width targets from the labels matched at the chosen view."""
        name = type(self).__name__
        self.codec.expect_grid(matched.friction, what="the matched friction labels", user=name)
        self.codec.expect_grid(matched.width, what="the matched width labels", user=name)
        supervised = matched.valid[:, None, None].expand_as(matched.friction)
        usable = supervised & (matched.friction > 0) & (matched.width <= self.gripper.max_width)
        friction = torch.where(usable, matched.friction, torch.zeros_like(matched.friction))
        score = torch.zeros_like(friction)
        positive = friction > 0
        if positive.any():
            if self.score_target == "log_friction_batch_norm":
                worst, best = friction.max(), friction[positive].min()
            else:
                best, worst = (friction.new_tensor(v) for v in self.friction_range)
            score[positive] = torch.log(worst / friction[positive]) / (
                torch.log(worst / best) + _SCORE_EPS
            )
        return SWADTargets(
            score=score,
            width=self.codec.encode_width(matched.width),
            score_mask=supervised,
            width_mask=supervised & (score > 0),
        )

    # ------------------------------------------------------------------------ loss

    def loss(self, raw: SWADPredictions, matched: MatchedLabels) -> dict[str, Tensor]:
        """Score loss on supervised seeds; width loss where the score target is positive."""
        targets = self.encode_targets(matched)
        score_mask = None if bool(targets.score_mask.all()) else targets.score_mask
        return {
            "loss/score": self.score_loss(raw.score, targets.score, score_mask),
            "loss/width": self.width_loss(raw.width, targets.width, targets.width_mask),
        }

    # ---------------------------------------------------------------------- decode

    @torch.no_grad()
    def decode(self, raw: SWADPredictions, seeds: SeedSet, approach: ApproachSet) -> GraspGroup:
        """One grasp per seed: the best cell of its score grid.

        The returned group is aligned with ``seeds`` row by row and still contains the
        grasps of padded seeds; the detector drops those.
        """
        if not len(raw.score) == len(seeds) == len(approach):
            raise ContractError(
                f"{type(self).__name__}.decode: {len(raw.score)} predictions, {len(seeds)} seeds "
                f"and {len(approach)} approaches do not line up"
            )
        # A score loss that trains logits (QualityFocal, Varifocal, ...) is decoded
        # through its sigmoid; the best cell is the same either way.
        score, cell = self.codec.flatten(self.score_loss.as_value(raw.score)).max(dim=-1)
        angle_index, depth_index = self.codec.split_cell(cell)
        width = self.codec.flatten(raw.width).gather(1, cell[:, None]).squeeze(1)
        return GraspGroup(
            translation=seeds.points.coords,
            rotation=approach_to_rotation(approach.direction, self.codec.angle(angle_index)),
            width=self.codec.decode_width(width, self.gripper.max_width),
            depth=self.codec.depth(depth_index),
            score=score,
            frame=seeds.points.frame,
            gripper=self.gripper,
            score_kind=self.score_kind,
        )

    def extra_repr(self) -> str:
        return f"codec={self.codec}, score_target={self.score_target!r}"
