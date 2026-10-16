"""View stage of the Graspness family: score every template view, pick one per seed.

A hidden layer refines the seed features and a second layer regresses one score per
template view (the "view-wise graspness"). The refined features are added back to the
seed features that later stages consume, so the stage returns the updated seeds
together with the chosen approach.

Selection follows ``nn.Module.training``: inference takes the best-scoring view;
training follows ``train_selection``. The reference *samples* the training view from
the predicted scores, which exposes the crop and the grasp head to more views than the
arg-max would.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.models.base import Loss, ViewStage
from grasppanda.registry import INJECTED, LOSSES, VIEWS, Cfg
from grasppanda.structures import ApproachSet, ContractError, SeedSet

from .template import ViewTemplate

__all__ = ["TemplateViewStage"]

#: Guards of the two min-max normalisations, as in the reference implementation.
_SELECTION_EPS = 1e-8
_TARGET_EPS = 1e-5


@VIEWS.register()
class TemplateViewStage(ViewStage):
    """Regress a score for each view of a Fibonacci template and choose one view.

    Args:
        num_views: Size of the view template. The label set must use the same lattice.
        train_selection: How the view is chosen in training mode. ``multinomial`` (the
            reference) draws one view per seed with probability proportional to the
            detached scores, min-max normalised per seed; ``argmax`` takes the best
            view, as inference always does.
        max_friction: A labelled grasp counts towards the target of its view when its
            friction coefficient lies in ``(0, max_friction]``.
        loss: Loss between predicted and target view scores, over all views of the
            supervised seeds. ``None`` selects the reference, ``SmoothL1`` with weight 100.
        in_channels: Channels of the seed features; injected by the detector.
    """

    status = "tested"
    source = "GSNet (Wang et al., ICCV 2021)"
    label_formats = ("dense", "compact")
    #: A weight per seed from the assigner weights the views of that seed in the loss.
    applies_seed_weights = True
    #: The view scores are only ranked and min-max normalised, so a loss that trains them
    #: as logits needs no activation here.
    loss_terms: ClassVar[dict[str, str]] = {"loss": "unit"}

    def __init__(
        self,
        num_views: int = 300,
        train_selection: Literal["multinomial", "argmax"] = "multinomial",
        max_friction: float = 0.6,
        loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if train_selection not in ("multinomial", "argmax"):
            raise ValueError(f"unknown train_selection '{train_selection}'")
        self.train_selection = train_selection
        self.max_friction = max_friction
        self.in_channels = in_channels
        self.template = ViewTemplate.fibonacci(num_views)
        self.hidden = nn.Linear(in_channels, in_channels)
        self.score = nn.Linear(in_channels, num_views)
        self.loss_fn = LOSSES.build(
            loss if loss is not None else {"type": "SmoothL1", "weight": 100.0}
        )

    def forward(
        self, seeds: SeedSet, view_index: Tensor | None = None
    ) -> tuple[ApproachSet, SeedSet]:
        """Choose one approach per seed.

        Args:
            seeds: Seeds with ``(M, in_channels)`` features.
            view_index: ``(M,)`` template indices to use instead of selecting. This
                replays a recorded choice, e.g. the draw of a reference run.

        Returns:
            The approach (direction, zero-angle template frame, view index, raw view
            scores) and the seeds with the stage's residual added to their features.
        """
        if seeds.feats is None or seeds.feats.shape[-1] != self.in_channels:
            got = None if seeds.feats is None else tuple(seeds.feats.shape)
            raise ContractError(
                f"{type(self).__name__} needs (M, {self.in_channels}) seed features, got {got}"
            )
        residual = F.relu(self.hidden(seeds.feats))
        logits = self.score(residual)
        if view_index is None:
            view_index = self.select(logits)
        elif tuple(view_index.shape) != (len(seeds),):
            raise ContractError(
                f"view_index must hold one view per seed ({len(seeds)},), got "
                f"{tuple(view_index.shape)}"
            )
        view_index = view_index.to(logits.device).long()
        approach = ApproachSet(
            direction=self.template.approach(view_index),
            rotation=self.template.rotation(view_index),
            view_index=view_index,
            logits=logits,
        )
        return approach, replace(seeds, feats=seeds.feats + residual)

    @torch.no_grad()
    def select(self, logits: Tensor) -> Tensor:
        """``(M,)`` chosen view per seed from ``(M, V)`` scores."""
        if not self.training or self.train_selection == "argmax":
            return logits.argmax(dim=-1)
        low = logits.amin(dim=-1, keepdim=True)
        high = logits.amax(dim=-1, keepdim=True)
        weight = (logits - low) / (high - low + _SELECTION_EPS)
        # A seed whose scores are all equal has no preference (and an all-zero row cannot
        # be sampled from): every view is then equally likely.
        weight = torch.where(high > low, weight, torch.ones_like(weight))
        return torch.multinomial(weight, 1).squeeze(-1)

    def encode_targets(self, view_friction: Tensor) -> Tensor:
        """Target view scores ``(M, V)`` from friction grids ``(M, V, A, D)``.

        The graspness of a view is the fraction of its angle × depth cells holding a
        grasp with friction in ``(0, max_friction]``; it is then min-max normalised
        over the views of each seed.
        """
        good = (view_friction > 0) & (view_friction <= self.max_friction)
        cells = view_friction.shape[-2] * view_friction.shape[-1]
        graspness = good.flatten(-2).sum(dim=-1).to(view_friction.dtype) / cells
        low = graspness.amin(dim=-1, keepdim=True)
        high = graspness.amax(dim=-1, keepdim=True)
        return (graspness - low) / (high - low + _TARGET_EPS)

    def targets(self, matched: MatchedLabels | CompactMatchedLabels) -> tuple[Tensor, Tensor]:
        """Target view scores ``(M, V)`` and the ``(M,)`` seeds they supervise.

        Dense labels hand over friction grids, which are encoded here. Compact labels
        no longer hold the grids and carry the encoded scores instead; those must have
        been encoded with this stage's ``max_friction``.
        """
        compact = isinstance(matched, CompactMatchedLabels)
        labels = matched.view_graspness if compact else matched.view_friction
        if labels.shape[1] != self.template.num_views:
            raise ContractError(
                f"the matched labels describe {labels.shape[1]} views, but the view template "
                f"has {self.template.num_views}"
            )
        if not compact:
            return self.encode_targets(labels), matched.valid
        if not math.isclose(matched.view_max_friction, self.max_friction, rel_tol=1e-9):
            raise ContractError(
                f"the compact labels encode view scores with max_friction = "
                f"{matched.view_max_friction}, but {type(self).__name__} is configured with "
                f"{self.max_friction}. Generate the label asset with the value of the view "
                "stage, or set the view stage to the value of the asset."
            )
        return labels, matched.view_valid

    def loss(
        self, approach: ApproachSet, matched: MatchedLabels | CompactMatchedLabels
    ) -> dict[str, Tensor]:
        """View-score regression loss on the supervised seeds.

        With a weight per seed from the assigner, the loss is the weighted mean over the
        views of the supervised seeds, every view weighted by its seed.
        """
        if approach.logits is None:
            raise ContractError(f"{type(self).__name__}.loss needs the view scores of its forward")
        target, supervised = self.targets(matched)
        mask = None if bool(supervised.all()) else supervised[:, None].expand_as(target)
        weight = getattr(matched, "weight", None)
        views = None if weight is None else weight[:, None].expand_as(target)
        return {"loss/view": self.loss_fn(approach.logits, target, mask, views)}
