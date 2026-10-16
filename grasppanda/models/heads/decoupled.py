"""Decoupled grid head bundle: a score grid, and the grasp read from separate outputs.

``SWADHead`` regresses a score and a width for every (in-plane angle, depth) cell and
takes the arg-max cell. Here the questions are separated, as in the grasp head of
FGC-GraspNet: the *score grid* still says how good every cell is, but *which* angle and
*which* depth to use is answered by two outputs of their own, and the width is predicted
once per depth bin instead of once per cell. The bundle is written for this toolbox from
that idea (no upstream code); what each quantity is, a class or a regressed number, is
an option, so the same bundle also answers "classification or regression" for depth and
width:

* ``depth: class`` classifies the depth bin, ``regression`` regresses the depth in
  units of ``depth_step`` and decodes a continuous depth;
* ``width: regression`` regresses one width per depth bin, ``class`` classifies it into
  ``width_bins`` equal bins of the gripper opening and decodes the bin centre, ``bins``
  predicts a distribution over the same bins, trains it with a distribution focal loss
  and decodes its expectation.

Targets, all from the dense label grid of the chosen view:

* the score target of a cell is that of ``SWADHead`` (option ``score_target``);
* the *best cell* of a seed is the cell with the highest score target (the first one
  among equals). Its angle bin and its depth bin are the targets of the angle and the
  depth output; a seed without any usable grasp supervises neither;
* the width target of depth bin ``d`` is the label width at (best angle, ``d``), where
  that cell holds a usable grasp.

Decoding reads the angle and the depth from their outputs, the score from the grid at
that cell and the width at that depth, so every seed yields one grasp.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

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

__all__ = ["DecoupledGridHead", "DecoupledPredictions", "DecoupledTargets"]

#: Guard of the score normalisation, as in ``SWADHead``.
_SCORE_EPS = 1e-6
#: ``width`` / ``depth`` option → kind of the loss term it makes.
_WIDTH_TERMS = {"regression": "value", "class": "class", "bins": "bins"}
_DEPTH_TERMS = {"class": "class", "regression": "value"}


@dataclass(eq=False)
class DecoupledPredictions:
    """Raw outputs of :class:`DecoupledGridHead`; they keep gradients.

    Attributes:
        score: ``(M, num_angle, num_depth)`` predicted grasp quality per cell.
        angle: ``(M, num_angle)`` logits over the in-plane angle bins.
        depth: ``(M, num_depth)`` logits over the depth bins, or ``(M,)`` regressed
            depth in units of ``depth_step`` with ``depth: regression``.
        width: ``(M, num_depth)`` width per depth bin in regression units, or
            ``(M, num_depth, width_bins)`` logits with ``width: class`` or ``bins``.
    """

    score: Tensor
    angle: Tensor
    depth: Tensor
    width: Tensor


@dataclass(eq=False)
class DecoupledTargets:
    """Targets of :class:`DecoupledGridHead`.

    Attributes:
        score: ``(M, num_angle, num_depth)``; 0 where there is no usable grasp.
        score_mask: ``(M, num_angle, num_depth)`` cells that supervise the score.
        angle: ``(M,)`` angle bin of the best cell.
        depth: ``(M,)`` depth bin of the best cell (a class, or the regression target in
            units of ``depth_step`` after adding 1).
        seed_mask: ``(M,)`` seeds that supervise angle and depth.
        width: ``(M, num_depth)`` label width at the best angle, in the units of the
            width term: regression units, or a position in ``[0, width_bins - 1]``.
        width_mask: ``(M, num_depth)`` entries that supervise the width.
    """

    score: Tensor
    score_mask: Tensor
    angle: Tensor
    depth: Tensor
    seed_mask: Tensor
    width: Tensor
    width_mask: Tensor


@HEADS.register()
class DecoupledGridHead(HeadBundle):
    """Score grid plus separate angle, depth and width outputs per seed; one grasp per seed.

    Args:
        num_angle: In-plane angle bins of the grid (owned by the codec).
        num_depth: Depth bins of the grid (owned by the codec).
        depth_step: Metres between depth bins.
        hidden_channels: Width of the two hidden layers.
        depth: ``class`` classifies the depth bin; ``regression`` regresses the depth in
            units of ``depth_step`` and decodes a continuous depth within the range of
            the bins.
        width: ``regression`` regresses one width per depth bin; ``class`` classifies
            it into ``width_bins`` bins; ``bins`` predicts a distribution over the bins
            and decodes its expectation. See the module docstring.
        width_bins: Number of width bins between 0 and the gripper opening, for
            ``width: class`` and ``bins``. The bin positions are ``k / (width_bins - 1)``
            of the opening.
        width_target_scale: Width targets of ``width: regression`` are metres times this
            factor.
        width_decode_gain: Factor on decoded widths. 1.2 is what the reference
            implementations of this family apply; 1.0 decodes what was learned.
        score_target: How label frictions become score targets, as in ``SWADHead``.
            ``log_friction_fixed_range`` (the default here) normalises with the constant
            ``friction_range``; ``log_friction_batch_norm`` with the extrema of the
            supervised cells of the batch.
        friction_range: ``(min, max)`` friction of ``log_friction_fixed_range``. The
            default is the range of the GraspNet-1Billion labels.
        score_loss: Loss on the score grid of supervised seeds. ``None`` selects
            ``SmoothL1`` with weight 15, the weight of ``SWADHead``.
        angle_loss: Loss on the angle logits. ``None`` selects ``CrossEntropy``, weight 1.
        depth_loss: Loss on the depth output. ``None`` selects ``CrossEntropy`` with
            weight 1, or ``SmoothL1`` with weight 1 for ``depth: regression``.
        width_loss: Loss on the width output. ``None`` selects ``SmoothL1`` with weight
            10 (``regression``), ``CrossEntropy`` with weight 1 (``class``) or
            ``DistributionFocal`` with weight 1 (``bins``).
        in_channels: Channels of the crop features; injected by the detector.
        gripper: Gripper the decoded grasps refer to; injected by the detector. Its
            ``max_width`` bounds usable labels, decoded widths and the width bins.
    """

    in_axes = ("seed", "channel")
    score_kind = "predicted_quality"
    status = "tested"
    source = (
        "after the grasp head of FGC-GraspNet (Lu et al., ICRA 2022), written here; width "
        "bins with the distribution focal loss of Generalized Focal Loss (Li et al., "
        "NeurIPS 2020)"
    )
    #: A weight per seed from the assigner weights all four losses.
    applies_seed_weights = True
    loss_terms: ClassVar[dict[str, str]] = {
        "score_loss": "unit",
        "angle_loss": "class",
        "depth_loss": "class",
        "width_loss": "value",
    }

    @classmethod
    def loss_terms_for(cls, options: Mapping[str, Any]) -> dict[str, str]:
        """The kinds of the four loss terms for the ``depth`` and ``width`` options given."""
        return {
            **cls.loss_terms,
            "depth_loss": _DEPTH_TERMS.get(options.get("depth", "class"), "class"),
            "width_loss": _WIDTH_TERMS.get(options.get("width", "regression"), "value"),
        }

    @classmethod
    def option_problems(cls, options: Mapping[str, Any]) -> list[str]:
        """Option values the constructor would refuse, told without building."""
        limits = options.get("friction_range")
        try:
            low, high = limits
            fine = 0 < low < high
        except (TypeError, ValueError):
            fine = False
        if fine:
            return []
        return [
            f"{cls.__name__} has friction_range = {list(limits) if limits else limits}; it "
            "must be [min, max] with 0 < min < max (the friction coefficients the score "
            "target is scaled between), e.g. [0.1, 1.0]."
        ]

    def __init__(
        self,
        num_angle: int = 12,
        num_depth: int = 4,
        depth_step: float = 0.01,
        hidden_channels: int = 128,
        depth: Literal["class", "regression"] = "class",
        width: Literal["regression", "class", "bins"] = "regression",
        width_bins: int = 16,
        width_target_scale: float = 10.0,
        width_decode_gain: float = 1.2,
        score_target: Literal[
            "log_friction_batch_norm", "log_friction_fixed_range"
        ] = "log_friction_fixed_range",
        friction_range: tuple[float, float] = (0.1, 1.0),
        score_loss: Cfg[Loss] | None = None,
        angle_loss: Cfg[Loss] | None = None,
        depth_loss: Cfg[Loss] | None = None,
        width_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
        gripper: Gripper = INJECTED,
    ) -> None:
        super().__init__()
        if not isinstance(gripper, Gripper):
            raise TypeError(f"gripper must be a Gripper, got {type(gripper).__name__}")
        if depth not in _DEPTH_TERMS:
            raise ValueError(f"unknown depth '{depth}'; choose class or regression")
        if width not in _WIDTH_TERMS:
            raise ValueError(f"unknown width '{width}'; choose regression, class or bins")
        if score_target not in ("log_friction_batch_norm", "log_friction_fixed_range"):
            raise ValueError(f"unknown score_target '{score_target}'")
        low, high = friction_range
        if not 0 < low < high:
            raise ValueError(
                f"friction_range must be (min, max) with 0 < min < max, got {low}, {high}"
            )
        if hidden_channels < 1 or width_bins < 2:
            raise ValueError("hidden_channels must be positive and width_bins at least 2")
        self.codec = AngleDepthCodec(
            num_angle, num_depth, depth_step, width_target_scale, width_decode_gain
        )
        self.depth = depth
        self.width = width
        self.width_bins = width_bins
        self.hidden_channels = hidden_channels
        self.score_target = score_target
        self.friction_range = (float(low), float(high))
        self.in_channels = in_channels
        self.gripper = gripper
        self.hidden1 = nn.Linear(in_channels, hidden_channels)
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels)
        self.score = nn.Linear(hidden_channels, self.codec.num_cells)
        self.angle = nn.Linear(hidden_channels, num_angle)
        self.depth_output = nn.Linear(hidden_channels, num_depth if depth == "class" else 1)
        per_depth = 1 if width == "regression" else width_bins
        self.width_output = nn.Linear(hidden_channels, num_depth * per_depth)
        #: Fraction of the gripper opening each width bin stands for.
        self.register_buffer(
            "bin_positions", torch.arange(width_bins) / (width_bins - 1), persistent=False
        )
        defaults = {
            "score": {"type": "SmoothL1", "weight": 15.0},
            "angle": {"type": "CrossEntropy", "weight": 1.0},
            "depth": {"type": "CrossEntropy" if depth == "class" else "SmoothL1", "weight": 1.0},
            "width": {
                "regression": {"type": "SmoothL1", "weight": 10.0},
                "class": {"type": "CrossEntropy", "weight": 1.0},
                "bins": {"type": "DistributionFocal", "weight": 1.0},
            }[width],
        }
        self.score_loss = LOSSES.build(score_loss if score_loss is not None else defaults["score"])
        self.angle_loss = LOSSES.build(angle_loss if angle_loss is not None else defaults["angle"])
        self.depth_loss = LOSSES.build(depth_loss if depth_loss is not None else defaults["depth"])
        self.width_loss = LOSSES.build(width_loss if width_loss is not None else defaults["width"])

    # --------------------------------------------------------------------- forward

    def forward(self, groups: GroupFeatures) -> DecoupledPredictions:
        name = type(self).__name__
        groups.expect(self.in_axes, user=name)
        if groups.size("channel") != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel crop features, got "
                f"{groups.size('channel')}"
            )
        hidden = F.relu(self.hidden2(F.relu(self.hidden1(groups.feats))))
        depth = self.depth_output(hidden)
        width = self.width_output(hidden)
        if self.width != "regression":
            width = width.unflatten(-1, (self.codec.num_depth, self.width_bins))
        return DecoupledPredictions(
            score=self.codec.unflatten(self.score(hidden)),
            angle=self.angle(hidden),
            depth=depth if self.depth == "class" else depth[:, 0],
            width=width,
        )

    # --------------------------------------------------------------------- targets

    @torch.no_grad()
    def encode_targets(self, matched: MatchedLabels) -> DecoupledTargets:
        """Score grid, best cell and widths at the best angle, from the chosen view."""
        name = type(self).__name__
        if not isinstance(matched, MatchedLabels):
            raise ContractError(
                f"{name} is supervised with the dense label grids, but the assigner returned "
                f"{type(matched).__name__}. Use NearestLabelAssigner with a dataset that serves "
                "dense grasp labels."
            )
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
            score.clamp_(min=0.0, max=1.0)  # frictions outside a fixed range
        top, cell = self.codec.flatten(score).max(dim=-1)  # the first maximum wins
        angle, depth = self.codec.split_cell(cell)
        seed_mask = matched.valid & (top > 0)
        at_angle = angle[:, None, None].expand(-1, 1, self.codec.num_depth)
        width = matched.width.gather(1, at_angle).squeeze(1)  # (M, D), metres
        width_mask = seed_mask[:, None] & (score.gather(1, at_angle).squeeze(1) > 0)
        if self.width == "regression":
            width = self.codec.encode_width(width)
        else:
            width = (width / self.gripper.max_width).clamp(0.0, 1.0) * (self.width_bins - 1)
        return DecoupledTargets(
            score=score,
            score_mask=supervised,
            angle=angle,
            depth=depth,
            seed_mask=seed_mask,
            width=width,
            width_mask=width_mask,
        )

    # ------------------------------------------------------------------------ loss

    def loss(self, raw: DecoupledPredictions, matched: MatchedLabels) -> dict[str, Tensor]:
        """Score loss on every cell of the supervised seeds; angle and depth at the best
        cell; width per depth bin at the best angle.

        With a weight per seed from the assigner, every loss is the weighted mean over its
        supervised entries, each entry weighted by its seed, as in ``OperationHead``.
        """
        targets = self.encode_targets(matched)
        seeds = targets.seed_mask
        score_mask = None if bool(targets.score_mask.all()) else targets.score_mask
        depth_target = targets.depth if self.depth == "class" else (targets.depth + 1).float()
        width, width_target, width_mask = raw.width, targets.width, targets.width_mask
        weight = matched.weight
        cells = per_depth = None
        if weight is not None:
            cells = weight[:, None, None].expand_as(raw.score)
            per_depth = weight[:, None].expand_as(width_mask)
        if self.width != "regression":  # one row of logits per (seed, depth bin)
            width = width.reshape(-1, self.width_bins)
            width_target, width_mask = width_target.reshape(-1), width_mask.reshape(-1)
            per_depth = None if per_depth is None else per_depth.reshape(-1)
            if self.width == "class":
                width_target = width_target.round().long()
        supervised = seeds.sum().clamp(min=1)
        hit = raw.angle.detach().argmax(dim=-1) == targets.angle
        return {
            "loss/score": self.score_loss(raw.score, targets.score, score_mask, cells),
            "loss/angle": self.angle_loss(raw.angle, targets.angle, seeds, weight),
            "loss/depth": self.depth_loss(raw.depth, depth_target, seeds, weight),
            "loss/width": self.width_loss(width, width_target, width_mask, per_depth),
            "metric/angle_accuracy": (hit & seeds).sum().float() / supervised,
            "metric/graspable_seeds": seeds.float().mean(),
        }

    # ---------------------------------------------------------------------- decode

    @torch.no_grad()
    def decode(
        self, raw: DecoupledPredictions, seeds: SeedSet, approach: ApproachSet
    ) -> GraspGroup:
        """One grasp per seed: angle and depth from their outputs, the score of that cell.

        The returned group is aligned with ``seeds`` row by row and still contains the
        grasps of padded seeds; the detector drops those.
        """
        if not len(raw.score) == len(seeds) == len(approach):
            raise ContractError(
                f"{type(self).__name__}.decode: {len(raw.score)} predictions, {len(seeds)} seeds "
                f"and {len(approach)} approaches do not line up"
            )
        angle_index = raw.angle.argmax(dim=-1)
        if self.depth == "class":
            depth_index = raw.depth.argmax(dim=-1)
            depth = self.codec.depth(depth_index)
        else:  # a continuous depth inside the range of the bins; the score of the nearest bin
            steps = raw.depth.clamp(min=1.0, max=float(self.codec.num_depth))
            depth_index = (steps.round() - 1).long()
            depth = steps * self.codec.depth_step
        cell = self.codec.cell(angle_index, depth_index)
        score = self.codec.flatten(self.score_loss.as_value(raw.score))
        score = score.gather(1, cell[:, None]).squeeze(1)
        if self.width == "regression":
            width = raw.width.gather(1, depth_index[:, None]).squeeze(1)
            width = self.codec.decode_width(width, self.gripper.max_width)
        else:
            at_depth = depth_index[:, None, None].expand(-1, 1, self.width_bins)
            logits = raw.width.gather(1, at_depth).squeeze(1)  # (M, width_bins)
            if self.width == "bins":
                fraction = (logits.softmax(dim=-1) * self.bin_positions).sum(dim=-1)
            else:
                fraction = self.bin_positions[logits.argmax(dim=-1)]
            width = (self.codec.width_decode_gain * fraction * self.gripper.max_width).clamp(
                min=0.0, max=self.gripper.max_width
            )
        return GraspGroup(
            translation=seeds.points.coords,
            rotation=approach_to_rotation(approach.direction, self.codec.angle(angle_index)),
            width=width,
            depth=depth,
            score=score,
            frame=seeds.points.frame,
            gripper=self.gripper,
            score_kind=self.score_kind,
        )

    def extra_repr(self) -> str:
        return (
            f"codec={self.codec}, depth={self.depth!r}, width={self.width!r}, "
            f"score_target={self.score_target!r}"
        )
