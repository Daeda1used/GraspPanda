"""View stage of the GraspNet-baseline family: objectness and view scores from one head.

Two hidden layers with batch norm and one output layer map the feature of every seed
to ``2 + V`` values: two objectness logits ("does this seed lie on an object?") and one
score per template view. The best-scoring view is chosen in training and in inference
alike.

Objectness lives here, not in a proposal stage, for two reasons. The reference network
computes both from a shared trunk and one output layer, so they cannot be given to two
stages without changing the function. And nothing is *selected* with it: every seed
goes through the crop and the head; objectness only masks losses and, in evaluation
mode, marks background seeds so that no grasp is emitted for them.

Targets (reference behaviour):

* objectness: the label of the scene point the seed was taken from;
* view score: the best score target of the view over its angle × depth grid, with the
  score target of :mod:`grasppanda.models.heads.friction`. It is regressed for every
  view of the supervised seeds (seeds on an object).

Scale-Balanced-Grasp narrows "object" to "graspable" (``min_graspable_views``): a seed
is a positive, and its view scores are supervised, only if more than that many of its
views have a target above ``graspable_score``. When the assigner provides a weight per
seed, the view loss is the weighted mean.

The seed features are passed on unchanged: this stage adds no residual to them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.base import Loss, ViewStage
from grasppanda.models.heads.friction import (
    check_score_target,
    log_friction_score,
    worst_friction,
)
from grasppanda.registry import INJECTED, LOSSES, VIEWS, Cfg
from grasppanda.structures import ApproachSet, ContractError, Gripper, SeedSet

from .template import ViewTemplate

__all__ = ["ObjectnessApproach", "ObjectnessViewStage", "ObjectnessViewTargets"]


@dataclass(eq=False)
class ObjectnessApproach(ApproachSet):
    """An approach per seed plus the objectness logits predicted with it.

    Attributes:
        objectness: ``(M, 2)`` logits, class 1 = the seed lies on an object.
    """

    objectness: Tensor | None = None


@dataclass(eq=False)
class ObjectnessViewTargets:
    """Targets of :class:`ObjectnessViewStage`.

    Attributes:
        objectness: ``(M,)`` class index, 1 = on an object (and graspable, when the
            stage asks for a minimum number of graspable views).
        view: ``(M, V)`` target score of every view.
        view_mask: ``(M,)`` seeds whose view scores are supervised.
    """

    objectness: Tensor
    view: Tensor
    view_mask: Tensor


@VIEWS.register()
class ObjectnessViewStage(ViewStage):
    """Classify every seed as object or background and choose its best template view.

    Args:
        num_views: Size of the view template. The label set must use the same lattice.
            The second hidden layer is ``2 + num_views`` wide, as in the reference.
        reject_background: In evaluation mode, mark a seed whose objectness arg-max is
            "background" as invalid, so that no grasp is decoded for it. Reference
            value: true. Training never rejects: losses are masked by labels.
        score_target: Which friction level maps to score 0 in the view target, see
            :mod:`grasppanda.models.heads.friction`. Reference value
            ``log_friction_batch_max``. The head bundle must use the same setting.
        max_friction: The level used by ``log_friction_fixed_max``.
        good_view_score: A predicted view score at or above this value on an object
            seed is counted in ``metric/good_view_predictions``. Logging only; the
            reference counts with 0.7.
        min_graspable_views: With a number ``n``, a seed counts as a positive of the
            objectness target, and has its view scores supervised, only if it lies on an
            object *and* more than ``n`` of its views have a target above
            ``graspable_score``. ``None`` (GraspNet-baseline) uses the object label
            alone; Scale-Balanced-Grasp uses 10.
        graspable_score: View target above which a view counts for
            ``min_graspable_views``. Reference value 0.1.
        objectness_loss: Loss on the objectness logits of every seed. ``None`` selects
            the reference, ``CrossEntropy`` with weight 1.
        view_loss: Loss between predicted and target view scores, over all views of the
            supervised seeds. ``None`` selects the reference, ``MSE`` with weight 1.
        in_channels: Channels of the seed features; injected by the detector.
        gripper: Gripper the labels are read for; injected by the detector. A labelled
            grasp wider than its ``max_width`` does not count towards a view target.
    """

    status = "tested"
    source = "GraspNet-baseline (Fang et al., CVPR 2020)"
    seed_scores = ("objectness",)
    needs_matched = ("on_object", "view_width")
    applies_seed_weights = True

    def __init__(
        self,
        num_views: int = 300,
        reject_background: bool = True,
        score_target: Literal[
            "log_friction_batch_max", "log_friction_fixed_max"
        ] = "log_friction_batch_max",
        max_friction: float = 1.0,
        good_view_score: float = 0.7,
        min_graspable_views: int | None = None,
        graspable_score: float = 0.1,
        objectness_loss: Cfg[Loss] | None = None,
        view_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
        gripper: Gripper = INJECTED,
    ) -> None:
        super().__init__()
        if not isinstance(gripper, Gripper):
            raise TypeError(f"gripper must be a Gripper, got {type(gripper).__name__}")
        check_score_target(score_target, max_friction)
        self.reject_background = reject_background
        self.score_target = score_target
        self.max_friction = float(max_friction)
        self.good_view_score = good_view_score
        if min_graspable_views is not None and min_graspable_views < 0:
            raise ValueError(
                f"min_graspable_views must be null or at least 0, got {min_graspable_views}"
            )
        self.min_graspable_views = min_graspable_views
        self.graspable_score = graspable_score
        self.in_channels = in_channels
        self.gripper = gripper
        self.template = ViewTemplate.fibonacci(num_views)
        width = 2 + num_views
        self.hidden1 = nn.Linear(in_channels, in_channels)
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.hidden2 = nn.Linear(in_channels, width)
        self.norm2 = nn.BatchNorm1d(width)
        #: Rows 0-1: objectness logits; rows 2…: one score per template view.
        self.output = nn.Linear(width, width)
        self.objectness_loss = LOSSES.build(
            objectness_loss if objectness_loss is not None else {"type": "CrossEntropy"}
        )
        self.view_loss = LOSSES.build(view_loss if view_loss is not None else {"type": "MSE"})

    # --------------------------------------------------------------------- forward

    def forward(
        self, seeds: SeedSet, view_index: Tensor | None = None
    ) -> tuple[ObjectnessApproach, SeedSet]:
        """Choose one approach per seed and score its objectness.

        Args:
            seeds: Seeds with ``(M, in_channels)`` features.
            view_index: ``(M,)`` template indices to use instead of the arg-max. This
                replays a recorded choice.

        Returns:
            The approach (direction, zero-angle template frame, view index, raw view
            scores, objectness logits) and the seeds with ``scores['objectness']`` (the
            probability of "object") and, in evaluation mode with ``reject_background``,
            background seeds marked invalid. Seed features are unchanged.
        """
        name = type(self).__name__
        if seeds.feats is None or seeds.feats.shape[-1] != self.in_channels:
            got = None if seeds.feats is None else tuple(seeds.feats.shape)
            raise ContractError(f"{name} needs (M, {self.in_channels}) seed features, got {got}")
        hidden = F.relu(self.norm1(self.hidden1(seeds.feats)))
        hidden = F.relu(self.norm2(self.hidden2(hidden)))
        output = self.output(hidden)
        objectness, logits = output[:, :2], output[:, 2:]
        if view_index is None:
            view_index = logits.detach().argmax(dim=-1)
        elif tuple(view_index.shape) != (len(seeds),):
            raise ContractError(
                f"view_index must hold one view per seed ({len(seeds)},), got "
                f"{tuple(view_index.shape)}"
            )
        view_index = view_index.to(logits.device).long()
        approach = ObjectnessApproach(
            direction=self.template.approach(view_index),
            rotation=self.template.rotation(view_index),
            view_index=view_index,
            logits=logits,
            objectness=objectness,
        )
        detached = objectness.detach()
        valid = seeds.valid
        if self.reject_background and not self.training:
            is_object = detached.argmax(dim=-1) == 1
            valid = is_object if valid is None else valid & is_object
        scores = {**seeds.scores, "objectness": detached.softmax(dim=-1)[:, 1]}
        return approach, replace(seeds, scores=scores, valid=valid)

    # --------------------------------------------------------------------- targets

    @torch.no_grad()
    def encode_targets(self, matched: MatchedLabels) -> ObjectnessViewTargets:
        """Objectness classes and view scores from the matched labels of every view."""
        name = type(self).__name__
        if matched.on_object is None or matched.view_width is None:
            raise ContractError(
                f"{name} reads the objectness label of every seed and the label widths of "
                "every view from the matched labels; configure the assigner with "
                "`supervise: object_seeds` and `view_widths: true`"
            )
        views = matched.view_friction.shape[1]
        if views != self.template.num_views:
            raise ContractError(
                f"the matched labels describe {views} views, but the view template has "
                f"{self.template.num_views}"
            )
        worst = worst_friction(matched, self.score_target, self.max_friction)
        score = log_friction_score(
            matched.view_friction, matched.view_width, worst, self.gripper.max_width
        )
        view = score.flatten(-2).amax(dim=-1)
        positive, supervised = matched.on_object, matched.valid
        if self.min_graspable_views is not None:
            graspable = (view > self.graspable_score).sum(dim=-1) > self.min_graspable_views
            positive, supervised = positive & graspable, supervised & graspable
        return ObjectnessViewTargets(objectness=positive.long(), view=view, view_mask=supervised)

    # ------------------------------------------------------------------------ loss

    def loss(self, approach: ApproachSet, matched: MatchedLabels) -> dict[str, Tensor]:
        """Objectness loss on every seed; view-score loss on the supervised seeds."""
        name = type(self).__name__
        objectness = getattr(approach, "objectness", None)
        if approach.logits is None or objectness is None:
            raise ContractError(
                f"{name}.loss needs the view scores and objectness logits of its own forward"
            )
        targets = self.encode_targets(matched)
        mask = targets.view_mask[:, None].expand_as(targets.view)
        weights = None if matched.weight is None else matched.weight[:, None].expand_as(mask)
        predicted = objectness.detach().argmax(dim=-1) == 1
        is_object = targets.objectness == 1
        correct = predicted == is_object
        good = (approach.logits.detach() >= self.good_view_score) & is_object[:, None]
        return {
            "loss/objectness": self.objectness_loss(objectness, targets.objectness),
            "loss/view": self.view_loss(approach.logits, targets.view, mask, weights),
            "metric/objectness_accuracy": correct.float().mean(),
            "metric/objectness_precision": _ratio(correct & predicted, predicted),
            "metric/objectness_recall": _ratio(correct & is_object, is_object),
            "metric/good_view_predictions": good.sum().float(),
        }

    def extra_repr(self) -> str:
        return f"score_target={self.score_target!r}, reject_background={self.reject_background}"


def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator.sum().float() / denominator.sum().clamp(min=1).float()
