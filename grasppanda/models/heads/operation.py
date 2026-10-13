"""Operation + tolerance head bundle of the GraspNet-baseline family.

The bundle consumes crop features with an explicit depth axis, one vector per
(seed, depth bin). Two small networks run on every such vector:

* the *operation* network predicts, for each in-plane angle bin, a grasp score, an
  angle-class logit and a gripper width;
* the *tolerance* network predicts, for each angle bin, the grasp tolerance: how far
  the grasp point may move while the grasp stays good.

All four outputs live on the ``num_angle × num_depth`` grid of one
:class:`AngleDepthCodec`. The bundle owns the prediction layers, the target encoder,
the loss and the decoder; none of the four can be swapped alone.

Targets (reference behaviour, each a named option):

* The score target of a cell is ``log(μ_worst / μ)`` for a usable grasp and 0 otherwise
  (:mod:`.friction`, option ``score_target``).
* Per (seed, depth bin) only the best angle is supervised: the angle bin with the
  highest score target is the class target of the angle logits, and the score, width
  and tolerance are regressed *at that angle only*. Equal targets are resolved towards
  the lowest angle bin, and they are common because label frictions are quantised.
* A (seed, depth bin) is supervised when the seed is supervised by the assigner and its
  best score target exceeds ``min_score_target``.
* Width and tolerance errors are measured in units of the gripper's maximum width and
  of ``max_tolerance``. The network regresses both in metres; ``width_decode_gain``
  widens the gripper in the decoder only.
* When the assigner provides a weight per seed (Scale-Balanced-Grasp), every loss is the
  weighted mean over its supervised entries. That method also supervises the score on
  *every* depth bin of a seed that has at least one supervised depth bin
  (``score_supervision``).

Decoding picks, per (seed, depth bin), the arg-max angle class, then the depth bin with
the highest score at that angle, so every seed yields one grasp. ``score_decode`` says
what the emitted score is: the reference multiplies the regressed score by the
predicted tolerance over ``max_tolerance``; Scale-Balanced-Grasp also multiplies in the
probability that the seed is graspable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

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
from .friction import check_score_target, log_friction_score, worst_friction

__all__ = ["OperationHead", "OperationPredictions", "OperationTargets"]

#: ``score_decode`` → the meaning of the emitted score.
_SCORE_KINDS = {
    "score_x_tolerance": "predicted_quality_x_tolerance",
    "score_x_objectness_x_tolerance": "predicted_quality_x_objectness_x_tolerance",
    "score": "predicted_quality",
}


@dataclass(eq=False)
class OperationPredictions:
    """Raw outputs of :class:`OperationHead`; they keep gradients.

    All tensors are ``(M, num_angle, num_depth)``.

    Attributes:
        score: Predicted grasp score.
        angle: Angle-class logits; the class axis is the angle axis, per depth bin.
        width: Predicted gripper width, metres.
        tolerance: Predicted grasp tolerance, metres.
    """

    score: Tensor
    angle: Tensor
    width: Tensor
    tolerance: Tensor


@dataclass(eq=False)
class OperationTargets:
    """Targets per (seed, depth bin); all tensors are ``(M, num_depth)``.

    Attributes:
        angle: Angle bin with the best score target; the class target.
        score: Score target at that angle.
        width: Label width at that angle, metres.
        tolerance: Label tolerance at that angle, metres.
        mask: Which (seed, depth bin) entries are supervised.
        score_mask: Which entries supervise the score; ``mask`` unless the head is
            configured otherwise.
        weight: Loss weight of every entry (the weight of its seed), or ``None``.
    """

    angle: Tensor
    score: Tensor
    width: Tensor
    tolerance: Tensor
    mask: Tensor
    score_mask: Tensor
    weight: Tensor | None = None


class _PerCropMLP(nn.Module):
    """Two hidden layers with batch norm and a linear output, shared by all crops."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, feats: Tensor) -> Tensor:
        hidden = F.relu(self.norm1(self.hidden1(feats)))
        hidden = F.relu(self.norm2(self.hidden2(hidden)))
        return self.output(hidden)


@HEADS.register()
class OperationHead(HeadBundle):
    """Score, angle class, width and tolerance per (angle, depth); one grasp per seed.

    Args:
        num_angle: In-plane angle bins of the grid (owned by the codec).
        num_depth: Depth bins of the grid (owned by the codec). The crop features must
            have exactly this many entries on their depth axis.
        depth_step: Metres between depth bins.
        hidden_channels: Width of the hidden layers of both networks.
        width_decode_gain: Factor on predicted widths in the decoder only. Reference
            value 1.2; 1.0 decodes exactly what was regressed.
        max_tolerance: Unit of the tolerance error and of the tolerance factor in the
            decoded score, metres. It is the largest value the tolerance labels take.
        score_target: Which friction level maps to score 0, see :mod:`.friction`.
            Reference value ``log_friction_batch_max``. The view stage must use the
            same setting.
        max_friction: The level used by ``log_friction_fixed_max``.
        min_score_target: A (seed, depth bin) is supervised only if its best score
            target exceeds this value. Reference value 0.1.
        score_supervision: Which entries supervise the score. ``supervised_depths``
            (GraspNet-baseline): the supervised (seed, depth bin) entries, as for the
            other terms. ``seeds_with_a_supervised_depth`` (Scale-Balanced-Grasp): all
            depth bins of a seed that has at least one supervised depth bin.
        score_decode: What the decoder emits as score. ``score_x_tolerance``
            (GraspNet-baseline) is the regressed score times the predicted tolerance
            over ``max_tolerance``; ``score_x_objectness_x_tolerance``
            (Scale-Balanced-Grasp) also multiplies in the seed's probability of being an
            object, ``SeedSet.scores['objectness']``; ``score`` is the regressed score
            alone.
        score_loss: Loss on the score at the target angle. ``None`` selects the
            reference, ``SmoothL1`` with weight 0.2.
        angle_loss: Loss on the angle-class logits. ``None`` selects the reference,
            ``CrossEntropy`` with weight 0.2.
        width_loss: Loss on the width at the target angle, in units of the gripper's
            maximum width. ``None`` selects the reference, ``SmoothL1`` with weight 0.2.
        tolerance_loss: Loss on the tolerance at the target angle, in units of
            ``max_tolerance``. ``None`` selects the reference, ``SmoothL1`` with weight 0.2.
        in_channels: Channels of the crop features; injected by the detector.
        gripper: Gripper the decoded grasps refer to; injected by the detector. Its
            ``max_width`` bounds usable labels and decoded widths.
    """

    in_axes = ("seed", "depth", "channel")
    score_kind = "predicted_quality_x_tolerance"
    status = "tested"
    source = "GraspNet-baseline (Fang et al., CVPR 2020)"
    applies_seed_weights = True

    @classmethod
    def required_seed_scores(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        """The objectness of the seeds, when the decoded score multiplies it in."""
        decode = options.get("score_decode")
        return ("objectness",) if decode == "score_x_objectness_x_tolerance" else ()

    def __init__(
        self,
        num_angle: int = 12,
        num_depth: int = 4,
        depth_step: float = 0.01,
        hidden_channels: int = 128,
        width_decode_gain: float = 1.2,
        max_tolerance: float = 0.05,
        score_target: Literal[
            "log_friction_batch_max", "log_friction_fixed_max"
        ] = "log_friction_batch_max",
        max_friction: float = 1.0,
        min_score_target: float = 0.1,
        score_supervision: Literal[
            "supervised_depths", "seeds_with_a_supervised_depth"
        ] = "supervised_depths",
        score_decode: Literal[
            "score_x_tolerance", "score_x_objectness_x_tolerance", "score"
        ] = "score_x_tolerance",
        score_loss: Cfg[Loss] | None = None,
        angle_loss: Cfg[Loss] | None = None,
        width_loss: Cfg[Loss] | None = None,
        tolerance_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
        gripper: Gripper = INJECTED,
    ) -> None:
        super().__init__()
        if not isinstance(gripper, Gripper):
            raise TypeError(f"gripper must be a Gripper, got {type(gripper).__name__}")
        check_score_target(score_target, max_friction)
        if score_decode not in _SCORE_KINDS:
            raise ValueError(f"unknown score_decode '{score_decode}'")
        if score_supervision not in ("supervised_depths", "seeds_with_a_supervised_depth"):
            raise ValueError(f"unknown score_supervision '{score_supervision}'")
        if max_tolerance <= 0 or hidden_channels < 1:
            raise ValueError("max_tolerance and hidden_channels must be positive")
        # The network regresses widths in metres: one width unit per metre.
        self.codec = AngleDepthCodec(num_angle, num_depth, depth_step, 1.0, width_decode_gain)
        self.max_tolerance = float(max_tolerance)
        self.score_target = score_target
        self.max_friction = float(max_friction)
        self.min_score_target = min_score_target
        self.score_supervision = score_supervision
        self.score_decode = score_decode
        self.score_kind = _SCORE_KINDS[score_decode]
        self.in_channels = in_channels
        self.gripper = gripper
        #: Output channels per (seed, depth): [score | angle logits | width], one per angle.
        self.operation = _PerCropMLP(in_channels, hidden_channels, 3 * num_angle)
        #: Output channels per (seed, depth): tolerance, one per angle.
        self.tolerance = _PerCropMLP(in_channels, hidden_channels, num_angle)
        reference = {"type": "SmoothL1", "weight": 0.2}
        self.score_loss = LOSSES.build(score_loss if score_loss is not None else reference)
        self.angle_loss = LOSSES.build(
            angle_loss if angle_loss is not None else {"type": "CrossEntropy", "weight": 0.2}
        )
        self.width_loss = LOSSES.build(width_loss if width_loss is not None else reference)
        self.tolerance_loss = LOSSES.build(
            tolerance_loss if tolerance_loss is not None else reference
        )

    # --------------------------------------------------------------------- forward

    def forward(self, groups: GroupFeatures) -> OperationPredictions:
        name = type(self).__name__
        groups.expect(self.in_axes, user=name)
        if groups.size("channel") != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel crop features, got "
                f"{groups.size('channel')}"
            )
        seeds, depths = groups.size("seed"), groups.size("depth")
        if depths != self.codec.num_depth:
            raise ContractError(
                f"{name} is configured for num_depth = {self.codec.num_depth}, but the crop "
                f"features have {depths} entries on their depth axis. The grouper must make one "
                "crop per depth bin of the head."
            )
        angles = self.codec.num_angle
        feats = groups.feats.reshape(seeds * depths, self.in_channels)
        # (seed · depth, quantity · angle) → (seed, angle, depth) per quantity
        operation = self.operation(feats).reshape(seeds, depths, 3, angles).permute(2, 0, 3, 1)
        tolerance = self.tolerance(feats).reshape(seeds, depths, angles).transpose(1, 2)
        return OperationPredictions(
            score=operation[0], angle=operation[1], width=operation[2], tolerance=tolerance
        )

    # --------------------------------------------------------------------- targets

    @torch.no_grad()
    def encode_targets(self, matched: MatchedLabels) -> OperationTargets:
        """Best angle per (seed, depth bin) and the labels at it, from the chosen view."""
        name = type(self).__name__
        if matched.tolerance is None:
            raise ContractError(
                f"{name} supervises a grasp tolerance, but the matched labels carry none. "
                "Configure the dataset to serve tolerance labels (`tolerance:` section; build "
                "the asset with `grasppanda prepare <dataset> --tolerance`)."
            )
        self.codec.expect_grid(matched.friction, what="the matched friction labels", user=name)
        self.codec.expect_grid(matched.width, what="the matched width labels", user=name)
        self.codec.expect_grid(matched.tolerance, what="the matched tolerance labels", user=name)
        worst = worst_friction(matched, self.score_target, self.max_friction)
        score = log_friction_score(matched.friction, matched.width, worst, self.gripper.max_width)
        best, angle = score.max(dim=1)  # over the angle axis; the first maximum wins
        index = angle[:, None, :]
        mask = matched.valid[:, None] & (best > self.min_score_target)
        score_mask = mask
        if self.score_supervision == "seeds_with_a_supervised_depth":
            score_mask = mask.any(dim=1, keepdim=True).expand_as(mask)
        return OperationTargets(
            angle=angle,
            score=best,
            width=matched.width.gather(1, index).squeeze(1),
            tolerance=matched.tolerance.gather(1, index).squeeze(1),
            mask=mask,
            score_mask=score_mask,
            weight=None if matched.weight is None else matched.weight[:, None].expand_as(mask),
        )

    # ------------------------------------------------------------------------ loss

    def loss(self, raw: OperationPredictions, matched: MatchedLabels) -> dict[str, Tensor]:
        """Score, angle-class, width and tolerance losses on the supervised (seed, depth)s."""
        targets = self.encode_targets(matched)
        index, mask, weight = targets.angle[:, None, :], targets.mask, targets.weight
        flat_weight = None if weight is None else weight.reshape(-1)
        angles = self.codec.num_angle

        def at_target(grid: Tensor) -> Tensor:
            return grid.gather(1, index).squeeze(1)

        logits = raw.angle.transpose(1, 2).reshape(-1, angles)
        classes = targets.angle.reshape(-1)
        width_unit, tolerance_unit = self.gripper.max_width, self.max_tolerance
        error = (logits.detach().argmax(dim=-1) - classes).abs()[mask.reshape(-1)]
        error = torch.minimum(error, angles - error)  # angle bins wrap around
        count = error.numel()
        accuracy = {
            f"metric/angle_accuracy_within_{bins}": (
                (error <= bins).float().mean() if count else error.new_zeros((), dtype=torch.float)
            )
            for bins in (0, 1, 2)
        }
        return {
            "loss/score": self.score_loss(
                at_target(raw.score), targets.score, targets.score_mask, weight
            ),
            "loss/angle": self.angle_loss(logits, classes, mask.reshape(-1), flat_weight),
            "loss/width": self.width_loss(
                at_target(raw.width) / width_unit, targets.width / width_unit, mask, weight
            ),
            "loss/tolerance": self.tolerance_loss(
                at_target(raw.tolerance) / tolerance_unit,
                targets.tolerance / tolerance_unit,
                mask,
                weight,
            ),
            **accuracy,
            "metric/supervised_depths": mask.float().mean(),
        }

    # ---------------------------------------------------------------------- decode

    @torch.no_grad()
    def decode(
        self, raw: OperationPredictions, seeds: SeedSet, approach: ApproachSet
    ) -> GraspGroup:
        """One grasp per seed: the arg-max angle class per depth, then the best depth.

        The returned group is aligned with ``seeds`` row by row and still contains the
        grasps of invalid seeds; the detector drops those.
        """
        if not len(raw.score) == len(seeds) == len(approach):
            raise ContractError(
                f"{type(self).__name__}.decode: {len(raw.score)} predictions, {len(seeds)} seeds "
                f"and {len(approach)} approaches do not line up"
            )
        angle_index = raw.angle.argmax(dim=1, keepdim=True)  # (M, 1, D)
        score, depth_index = raw.score.gather(1, angle_index).squeeze(1).max(dim=-1)

        def chosen(grid: Tensor) -> Tensor:
            return grid.gather(1, angle_index).squeeze(1).gather(1, depth_index[:, None])[:, 0]

        if self.score_decode == "score_x_objectness_x_tolerance":
            objectness = seeds.scores.get("objectness")
            if objectness is None:
                raise ContractError(
                    f"{type(self).__name__} is configured with score_decode: "
                    f"{self.score_decode}, but the seeds carry no 'objectness' score. Use a "
                    "view stage that predicts objectness, or another score_decode."
                )
            score = score * objectness
        if self.score_decode != "score":
            score = score * chosen(raw.tolerance) / self.max_tolerance
        angle = self.codec.angle(angle_index[:, 0].gather(1, depth_index[:, None])[:, 0])
        return GraspGroup(
            translation=seeds.points.coords,
            rotation=approach_to_rotation(approach.direction, angle),
            width=self.codec.decode_width(chosen(raw.width), self.gripper.max_width),
            depth=self.codec.depth(depth_index),
            score=score,
            frame=seeds.points.frame,
            gripper=self.gripper,
            score_kind=self.score_kind,
        )

    def extra_repr(self) -> str:
        return (
            f"codec={self.codec}, score_target={self.score_target!r}, "
            f"score_decode={self.score_decode!r}"
        )
