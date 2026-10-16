"""EconomicGrasp head bundle: four interacting task tokens, one grasp per seed.

The crop feature of a seed is projected into four tokens, one per quantity of the
grasp: in-plane angle, insertion depth, gripper width and quality. The tokens attend
to each other once (:class:`~grasppanda.models.interaction.TokenInteraction`); then
each predicts its own quantity. Angle, depth and quality are *classified*, the width is
regressed. There is no angle × depth grid: the labels hold one grasp per point and
view (:class:`~grasppanda.structures.CompactGraspLabels`), and the head predicts that
one grasp.

The bundle owns the prediction layers, the target encoder, the loss and the decoder,
all tied to one :class:`AngleDepthCodec`. None of the four can be swapped alone.

Reference behaviour, each a named option:

* ``unmatched_class``: the angle and depth classifiers have one output more than there
  are bins. The reference gives that extra class to seeds whose view has no label and
  then masks exactly those seeds out of the loss, so the class is never a target; it
  only competes in the softmax.
* ``decode_argmax = all_logits``: the decoder takes the arg-max over all outputs,
  including the extra one. A seed decoded to it gets the angle ``π`` or a depth one
  step beyond the deepest bin. ``supervised`` restricts the arg-max to the real bins.
* ``score_decode = logit_weighted``: the decoded score is ``Σ_k c_k · logit_k`` with
  the class values ``c_k = k / (num_score - 1)``: a weighted sum of *raw logits*,
  unbounded and not an expectation. ``softmax_expectation`` applies the softmax first.
* Quality classes are coarser than the labels: a label quality ``q`` in ``[0, 1]`` goes
  to class ``⌊q · (num_score - 1)⌋``. With six classes and labels in steps of 0.1 two
  neighbouring label levels share a class, "no usable grasp" shares class 0 with the
  worst level, and only the best level has a class of its own.
* The width is supervised on every supervised seed, including those whose view holds no
  usable grasp; targets are metres × ``width_target_scale`` and the decoder additionally
  applies ``width_decode_gain`` (see the codec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.assigners.matched import CompactMatchedLabels
from grasppanda.models.base import HeadBundle, Loss
from grasppanda.models.interaction import TokenInteraction
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

__all__ = ["EconomicHead", "EconomicPredictions", "EconomicTargets"]

#: The task tokens, in the order in which they enter the interaction.
TOKENS = ("angle", "depth", "width", "score")
#: Guard of the quality binning: label levels sit exactly on class boundaries.
_BIN_EPS = 1e-4


@dataclass(eq=False)
class EconomicPredictions:
    """Raw outputs of :class:`EconomicHead`; they keep gradients.

    Attributes:
        angle: ``(M, num_angle [+ 1])`` logits over the in-plane angle bins.
        depth: ``(M, num_depth [+ 1])`` logits over the depth bins.
        score: ``(M, num_score)`` logits over the quality classes.
        width: ``(M,)`` predicted width in regression units.
    """

    angle: Tensor
    depth: Tensor
    score: Tensor
    width: Tensor


@dataclass(eq=False)
class EconomicTargets:
    """Targets of :class:`EconomicHead`, one row per seed.

    Attributes:
        angle: ``(M,)`` angle class.
        depth: ``(M,)`` depth class.
        score: ``(M,)`` quality class.
        width: ``(M,)`` width in regression units.
        mask: ``(M,)`` seeds that supervise all four predictions.
    """

    angle: Tensor
    depth: Tensor
    score: Tensor
    width: Tensor
    mask: Tensor


@HEADS.register()
class EconomicHead(HeadBundle):
    """Angle, depth and quality classes plus a width per seed; one decoded grasp per seed.

    Args:
        num_angle: In-plane angle bins (owned by the codec).
        num_depth: Depth bins (owned by the codec).
        depth_step: Metres between depth bins.
        num_score: Quality classes.
        width_target_scale: Width targets are metres times this factor.
        width_decode_gain: Factor on predicted widths in the decoder only. Reference
            value 1.2; 1.0 decodes exactly what was regressed.
        unmatched_class: Give the angle and depth classifiers one extra, never
            supervised output each, see the module docstring. Reference value: true.
        decode_argmax: Outputs the decoder's arg-max runs over, see the module
            docstring. ``all_logits`` replicates upstream commit ``7361bb6``,
            ``supervised`` upstream commit ``4119bdc``. Without ``unmatched_class`` the
            two are the same.
        score_decode: How quality logits become the decoded score, see the module
            docstring. Reference value ``logit_weighted``.
        token_channels: Width of the task tokens.
        attention_heads: Heads of the token interaction.
        attention_dropout: Dropout on its attention weights during training.
        angle_loss: Loss on the angle logits. ``None`` selects the reference,
            ``CrossEntropy`` with weight 1.
        depth_loss: Loss on the depth logits. ``None`` selects the reference,
            ``CrossEntropy`` with weight 1.
        score_loss: Loss on the quality logits. ``None`` selects the reference,
            ``CrossEntropy`` with weight 1.
        width_loss: Loss on the width. ``None`` selects the reference, ``SmoothL1``
            with weight 10.
        in_channels: Channels of the crop features; injected by the detector.
        gripper: Gripper the decoded grasps refer to; injected by the detector. Its
            ``max_width`` bounds decoded widths.
    """

    in_axes = ("seed", "channel")
    #: Kind of matched labels the bundle consumes; the assigner must return the same.
    label_format = "compact"
    status = "tested"
    source = "EconomicGrasp (Wu et al., ECCV 2024)"
    #: A weight per seed from the assigner (``CompactMatchedLabels.weight``) weights all
    #: four losses.
    applies_seed_weights = True
    loss_terms: ClassVar[dict[str, str]] = {
        "angle_loss": "class",
        "depth_loss": "class",
        "score_loss": "class",
        "width_loss": "value",
    }

    def __init__(
        self,
        num_angle: int = 12,
        num_depth: int = 4,
        depth_step: float = 0.01,
        num_score: int = 6,
        width_target_scale: float = 10.0,
        width_decode_gain: float = 1.2,
        unmatched_class: bool = True,
        decode_argmax: Literal["all_logits", "supervised"] = "all_logits",
        score_decode: Literal["logit_weighted", "softmax_expectation"] = "logit_weighted",
        token_channels: int = 64,
        attention_heads: int = 1,
        attention_dropout: float = 0.05,
        angle_loss: Cfg[Loss] | None = None,
        depth_loss: Cfg[Loss] | None = None,
        score_loss: Cfg[Loss] | None = None,
        width_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
        gripper: Gripper = INJECTED,
    ) -> None:
        super().__init__()
        if not isinstance(gripper, Gripper):
            raise TypeError(f"gripper must be a Gripper, got {type(gripper).__name__}")
        if decode_argmax not in ("all_logits", "supervised"):
            raise ValueError(f"unknown decode_argmax '{decode_argmax}'")
        if score_decode not in ("logit_weighted", "softmax_expectation"):
            raise ValueError(f"unknown score_decode '{score_decode}'")
        if num_score < 2:
            raise ValueError(f"num_score must be at least 2, got {num_score}")
        self.codec = AngleDepthCodec(
            num_angle, num_depth, depth_step, width_target_scale, width_decode_gain
        )
        self.num_score = num_score
        self.unmatched_class = unmatched_class
        self.decode_argmax = decode_argmax
        self.score_decode = score_decode
        self.score_kind = (
            "weighted_quality_logits" if score_decode == "logit_weighted" else "expected_quality"
        )
        self.in_channels = in_channels
        self.gripper = gripper
        extra = 1 if unmatched_class else 0
        outputs = {
            "angle": num_angle + extra,
            "depth": num_depth + extra,
            "width": 1,
            "score": num_score,
        }
        self.project = nn.ModuleDict({t: nn.Linear(in_channels, token_channels) for t in TOKENS})
        self.interaction = TokenInteraction(token_channels, attention_heads, attention_dropout)
        self.predict = nn.ModuleDict({t: nn.Linear(token_channels, outputs[t]) for t in TOKENS})
        #: Value each quality class stands for when a score is decoded.
        self.register_buffer(
            "score_values", torch.arange(num_score) / (num_score - 1), persistent=False
        )
        default = {"type": "CrossEntropy", "weight": 1.0}
        self.angle_loss = LOSSES.build(angle_loss if angle_loss is not None else default)
        self.depth_loss = LOSSES.build(depth_loss if depth_loss is not None else default)
        self.score_loss = LOSSES.build(score_loss if score_loss is not None else default)
        self.width_loss = LOSSES.build(
            width_loss if width_loss is not None else {"type": "SmoothL1", "weight": 10.0}
        )

    # --------------------------------------------------------------------- forward

    def forward(self, groups: GroupFeatures) -> EconomicPredictions:
        name = type(self).__name__
        groups.expect(self.in_axes, user=name)
        if groups.size("channel") != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel crop features, got "
                f"{groups.size('channel')}"
            )
        tokens = torch.stack([self.project[t](groups.feats) for t in TOKENS], dim=1)
        tokens = self.interaction(tokens)
        raw = {t: self.predict[t](tokens[:, slot]) for slot, t in enumerate(TOKENS)}
        return EconomicPredictions(
            angle=raw["angle"], depth=raw["depth"], score=raw["score"], width=raw["width"][:, 0]
        )

    # --------------------------------------------------------------------- targets

    @torch.no_grad()
    def encode_targets(self, matched: CompactMatchedLabels) -> EconomicTargets:
        """Class and width targets from the compact labels matched at the chosen view."""
        name = type(self).__name__
        if not isinstance(matched, CompactMatchedLabels):
            raise ContractError(
                f"{name} is supervised with compact labels (one grasp per seed at the chosen "
                f"view), but the assigner returned {type(matched).__name__}. Use "
                "CompactLabelAssigner together with a dataset that serves `compact_labels`."
            )
        labelled = matched.view_labelled
        bins = {"angle": self.codec.num_angle, "depth": self.codec.num_depth}
        classes = {}
        for what, count in bins.items():
            label = getattr(matched, what)
            if labelled.any() and int(label[labelled].max()) >= count:
                raise ContractError(
                    f"{name}: the matched labels use {what} bin {int(label[labelled].max())}, "
                    f"but the head is configured for {count} {what} bins. The label set and "
                    "the head bundle must use the same grid."
                )
            # Seeds without a label are never supervised; the reference parks them on the
            # extra class.
            parked = count if self.unmatched_class else 0
            classes[what] = torch.where(labelled, label, torch.full_like(label, parked))
        if (
            matched.score.numel()
            and not 0.0 <= float(matched.score.min()) <= float(matched.score.max()) <= 1.0
        ):
            raise ContractError(f"{name}: label qualities must lie in [0, 1]")
        quality = (matched.score * (self.num_score - 1) + _BIN_EPS).floor().long()
        return EconomicTargets(
            angle=classes["angle"],
            depth=classes["depth"],
            score=quality.clamp(max=self.num_score - 1),
            width=self.codec.encode_width(matched.width),
            mask=matched.valid,
        )

    # ------------------------------------------------------------------------ loss

    def loss(self, raw: EconomicPredictions, matched: CompactMatchedLabels) -> dict[str, Tensor]:
        """Three classification losses and the width loss, all on the supervised seeds.

        With a weight per seed from the assigner, each is the weighted mean over the
        supervised seeds, as in ``OperationHead``.
        """
        targets = self.encode_targets(matched)
        mask = targets.mask
        weight = getattr(matched, "weight", None)
        supervised = mask.sum().clamp(min=1)
        losses = {
            "loss/angle": self.angle_loss(raw.angle, targets.angle, mask, weight),
            "loss/depth": self.depth_loss(raw.depth, targets.depth, mask, weight),
            "loss/score": self.score_loss(raw.score, targets.score, mask, weight),
            "loss/width": self.width_loss(raw.width, targets.width, mask, weight),
        }
        for what in ("angle", "depth", "score"):
            hit = getattr(raw, what).detach().argmax(dim=-1) == getattr(targets, what)
            losses[f"metric/{what}_accuracy"] = (hit & mask).sum().float() / supervised
        return losses

    # ---------------------------------------------------------------------- decode

    @torch.no_grad()
    def decode(self, raw: EconomicPredictions, seeds: SeedSet, approach: ApproachSet) -> GraspGroup:
        """One grasp per seed: its most likely angle and depth, its width and its score.

        The returned group is aligned with ``seeds`` row by row and still contains the
        grasps of padded seeds; the detector drops those.
        """
        if not len(raw.score) == len(seeds) == len(approach):
            raise ContractError(
                f"{type(self).__name__}.decode: {len(raw.score)} predictions, {len(seeds)} seeds "
                f"and {len(approach)} approaches do not line up"
            )
        angle, depth = raw.angle, raw.depth
        if self.decode_argmax == "supervised":
            angle, depth = angle[:, : self.codec.num_angle], depth[:, : self.codec.num_depth]
        score = raw.score.softmax(dim=-1) if self.score_decode != "logit_weighted" else raw.score
        return GraspGroup(
            translation=seeds.points.coords,
            rotation=approach_to_rotation(
                approach.direction, self.codec.angle(angle.argmax(dim=-1))
            ),
            width=self.codec.decode_width(raw.width, self.gripper.max_width),
            depth=self.codec.depth(depth.argmax(dim=-1)),
            score=(score * self.score_values).sum(dim=-1),
            frame=seeds.points.frame,
            gripper=self.gripper,
            score_kind=self.score_kind,
        )

    def extra_repr(self) -> str:
        return (
            f"codec={self.codec}, num_score={self.num_score}, "
            f"decode_argmax={self.decode_argmax!r}, score_decode={self.score_decode!r}"
        )
