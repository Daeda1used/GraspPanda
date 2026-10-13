"""View stages that score the views of a lattice other than the one the labels use.

GraspNet-1Billion labels are indexed by 300 Fibonacci views, and the assigners look
labels up by an index into that lattice. The stages here keep the two apart: they own
the *label lattice* (``template``, handed to the assigner, ``label_views`` views) and a
*prediction lattice* of their own (``lattice``). Every prediction view is tied to the
label view nearest to it, so

* the target of a prediction view is the graspness of its nearest label view;
* ``ApproachSet.view_index`` is the nearest label view of the chosen direction, which
  is what the assigner needs; ``ApproachSet.direction`` and ``rotation`` are the chosen
  direction itself and its zero-angle frame, which is what the crop (at inference) and
  the head's decoder use.

A coarser lattice therefore reads a subset of the label columns and a finer one reads
some columns twice: the labels carry no more than their own 300 directions. In training
the crop frame comes from the labels (the snapped view); at inference it is the frame
of the prediction view, off by at most the spacing of the coarser of the two lattices.

On top of the lattice the stages offer what :class:`TemplateViewStage` does not:
classification of the best view instead of regression of all scores, a soft-arg-max
read-out that returns a direction between lattice views, sampling among the ``top_k``
views while training, and seed weights in the loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.models.base import Loss, ViewStage
from grasppanda.ops import (
    approach_to_rotation,
    fibonacci_views,
    healpix_views,
    icosphere_views,
    nearest_view,
)
from grasppanda.registry import INJECTED, LOSSES, VIEWS, Cfg
from grasppanda.structures import ApproachSet, ContractError, SeedSet

from .angular_cost import AngularCost
from .targets import view_graspness
from .template import ViewTemplate

__all__ = ["HealpixViewStage", "IcosphereViewStage", "LatticeViewStage"]

_EPS = 1e-8
_TARGET_EPS = 1e-5
_TARGETS = ("score", "best_view")
_SELECTIONS = ("argmax", "soft_argmax")
_TRAIN_SELECTIONS = ("multinomial", "argmax", "topk")
_DEFAULT_LOSS = {
    "score": {"type": "SmoothL1", "weight": 100.0},
    "best_view": {"type": "CrossEntropy", "weight": 1.0},
}


class _LatticeStage(ViewStage):
    """Shared body of the lattice stages; subclasses only say which views they score."""

    label_formats = ("dense", "compact")
    applies_seed_weights = True
    #: ``target: score`` regresses a normalised graspness per view (a `unit` term, read
    #: only by ranking, min-max and softmax, so a loss may train it as a logit);
    #: ``target: best_view`` classifies the best view (a `class` term).
    loss_terms: ClassVar[dict[str, str]] = {"loss": "unit"}

    @classmethod
    def loss_terms_for(cls, options: Mapping[str, Any]) -> dict[str, str]:
        """The kind of the view loss for the ``target`` option given."""
        return {"loss": "class" if options.get("target", "score") == "best_view" else "unit"}

    def _setup(
        self,
        views: Tensor,
        *,
        hemisphere: bool,
        target: str,
        selection: str,
        train_selection: str,
        top_k: int | None,
        temperature: float,
        angular_weight: float,
        label_views: int,
        max_friction: float,
        residual: bool,
        loss: Cfg[Loss] | None,
        in_channels: int,
    ) -> None:
        name = type(self).__name__
        for option, value, known in (
            ("target", target, _TARGETS),
            ("selection", selection, _SELECTIONS),
            ("train_selection", train_selection, _TRAIN_SELECTIONS),
        ):
            if value not in known:
                raise ValueError(f"{name}: unknown {option} '{value}'; choose one of {known}")
        if temperature <= 0:
            raise ValueError(f"{name}: temperature must be positive, got {temperature}")
        if hemisphere:
            # The camera looks along +z: an approach away from the camera has z > 0, and
            # the viewpoint it starts from (the negated approach) has z < 0.
            views = views[views[:, 2] < 0]
        if top_k is not None and not 1 <= top_k <= len(views):
            raise ValueError(
                f"{name}: top_k must be null or between 1 and the {len(views)} views of the "
                f"lattice, got {top_k}"
            )
        if train_selection == "topk" and top_k is None:
            raise ValueError(f"{name}: train_selection 'topk' needs top_k")
        if angular_weight < 0:
            raise ValueError(f"{name}: angular_weight must not be negative, got {angular_weight}")
        if angular_weight and target != "best_view":
            raise ValueError(
                f"{name}: angular_weight needs target 'best_view' (the angular cost is a loss "
                f"over view classes), got target '{target}'"
            )
        self.hemisphere = hemisphere
        self.target = target
        self.selection = selection
        self.train_selection = train_selection
        self.top_k = top_k
        self.temperature = float(temperature)
        self.angular_weight = float(angular_weight)
        self.label_views = label_views
        self.max_friction = max_friction
        self.residual = residual
        self.in_channels = in_channels
        #: The label lattice: what the assigner indexes labels by.
        self.template = ViewTemplate.fibonacci(label_views)
        #: The prediction lattice: what this stage scores.
        self.lattice = ViewTemplate(views)
        to_label = nearest_view(self.lattice.views, self.template.views)
        self.register_buffer("to_label", to_label, persistent=False)
        self.identity = len(views) == label_views and bool(
            torch.equal(to_label, torch.arange(label_views))
        )
        self.hidden = nn.Linear(in_channels, in_channels)
        self.score = nn.Linear(in_channels, len(views))
        self.loss_fn = LOSSES.build(loss if loss is not None else _DEFAULT_LOSS[target])
        self.angular_cost = AngularCost(-self.lattice.views) if target == "best_view" else None

    # --------------------------------------------------------------------- forward

    def forward(
        self, seeds: SeedSet, view_index: Tensor | None = None
    ) -> tuple[ApproachSet, SeedSet]:
        """Choose one approach per seed.

        Args:
            seeds: Seeds with ``(M, in_channels)`` features.
            view_index: ``(M,)`` indices into this stage's *prediction* lattice to use
                instead of selecting (a replayed choice).

        Returns:
            The approach: ``direction`` (M, 3) unit vectors, ``rotation`` (M, 3, 3) their
            zero-angle frames, ``view_index`` (M,) the nearest view of the label lattice,
            ``logits`` (M, V) one score per prediction view. And the seeds, with the
            stage's residual added to their features when ``residual`` is set.
        """
        name = type(self).__name__
        if seeds.feats is None or seeds.feats.shape[-1] != self.in_channels:
            got = None if seeds.feats is None else tuple(seeds.feats.shape)
            raise ContractError(f"{name} needs (M, {self.in_channels}) seed features, got {got}")
        refined = F.relu(self.hidden(seeds.feats))
        logits = self.score(refined)
        soft = view_index is None and not self.training and self.selection == "soft_argmax"
        if soft:
            direction = self.soft_direction(logits)
            label_index = nearest_view(direction, -self.template.views)
            rotation = approach_to_rotation(direction)
        else:
            if view_index is None:
                view_index = self.select(logits)
            elif tuple(view_index.shape) != (len(seeds),):
                raise ContractError(
                    f"view_index must hold one view per seed ({len(seeds)},), got "
                    f"{tuple(view_index.shape)}"
                )
            view_index = view_index.to(logits.device).long()
            direction = self.lattice.approach(view_index)
            rotation = self.lattice.rotation(view_index)
            label_index = self.to_label[view_index]
        approach = ApproachSet(
            direction=direction, rotation=rotation, view_index=label_index, logits=logits
        )
        if self.residual:
            seeds = replace(seeds, feats=seeds.feats + refined)
        return approach, seeds

    @torch.no_grad()
    def select(self, logits: Tensor) -> Tensor:
        """``(M,)`` chosen prediction view per seed from ``(M, V)`` scores."""
        if not self.training or self.train_selection == "argmax":
            return logits.argmax(dim=-1)
        low = logits.amin(dim=-1, keepdim=True)
        high = logits.amax(dim=-1, keepdim=True)
        weight = (logits - low) / (high - low + _EPS)
        weight = torch.where(high > low, weight, torch.ones_like(weight))
        if self.train_selection == "topk":
            kept = torch.zeros_like(weight)
            index = weight.topk(self.top_k, dim=-1).indices
            # The floor keeps a row samplable when its k-th score is the row minimum.
            kept.scatter_(1, index, weight.gather(1, index).clamp(min=_EPS))
            weight = kept
        return torch.multinomial(weight, 1).squeeze(-1)

    @torch.no_grad()
    def soft_direction(self, logits: Tensor) -> Tensor:
        """``(M, 3)`` soft-arg-max direction: the softmax-weighted mean of the view
        directions (of the ``top_k`` best views when set), renormalised.

        Where the mean vanishes (opposite views of equal weight) the best view is used.
        """
        index = None
        if self.top_k is not None:
            logits, index = logits.topk(self.top_k, dim=-1)
        weight = (logits / self.temperature).softmax(dim=-1)
        approach = -self.lattice.views
        directions = approach.expand(len(logits), -1, -1) if index is None else approach[index]
        mean = (weight[..., None] * directions).sum(dim=1)
        norm = mean.norm(dim=-1, keepdim=True)
        best = directions[torch.arange(len(logits), device=logits.device), weight.argmax(dim=-1)]
        return torch.where(norm > 1e-6, mean / norm.clamp(min=_EPS), best)

    # --------------------------------------------------------------------- targets

    def targets(self, matched: MatchedLabels | CompactMatchedLabels) -> tuple[Tensor, ...]:
        """Target, supervised seeds ``(M,)`` and seed weights (or ``None``).

        ``target: score`` gives ``(M, V)`` graspness of the prediction views, min-max
        normalised per seed; ``target: best_view`` gives the ``(M,)`` index of the best
        prediction view and supervises only seeds that have a graspable view.
        """
        name = type(self).__name__
        graspness, supervised, weight = view_graspness(
            matched, self.max_friction, self.template.num_views, name
        )
        scores = graspness if self.identity else graspness[:, self.to_label]
        if self.target == "best_view":
            return scores.argmax(dim=-1), supervised & (scores.amax(dim=-1) > 0), weight
        if not (self.identity and isinstance(matched, CompactMatchedLabels)):
            low = scores.amin(dim=-1, keepdim=True)
            high = scores.amax(dim=-1, keepdim=True)
            scores = (scores - low) / (high - low + _TARGET_EPS)
        return scores, supervised, weight

    def loss(
        self, approach: ApproachSet, matched: MatchedLabels | CompactMatchedLabels
    ) -> dict[str, Tensor]:
        """``loss/view`` on the supervised seeds (weighted per seed when the assigner
        returns weights). With ``target: best_view`` also ``metric/view_accuracy``, the
        mean angle between the predicted and the best view (``metric/view_error_deg``)
        and, with ``angular_weight``, ``loss/view_angular``."""
        if approach.logits is None:
            raise ContractError(f"{type(self).__name__}.loss needs the view scores of its forward")
        target, supervised, weight = self.targets(matched)
        if self.target == "best_view":
            logits = approach.logits
            hit = (logits.detach().argmax(dim=-1) == target) & supervised
            error = self.angular_cost.mean_error(logits, target, supervised)
            losses = {
                "loss/view": self.loss_fn(logits, target, supervised, weight),
                "metric/view_accuracy": hit.sum().float() / supervised.sum().clamp(min=1).float(),
                "metric/view_error_deg": torch.rad2deg(error),
            }
            if self.angular_weight:
                cost = self.angular_cost(logits, target, supervised, weight)
                losses["loss/view_angular"] = self.angular_weight * cost
            return losses
        mask = supervised[:, None].expand_as(target)
        weights = None if weight is None else weight[:, None].expand_as(target)
        return {"loss/view": self.loss_fn(approach.logits, target, mask, weights)}

    def extra_repr(self) -> str:
        return (
            f"views={self.lattice.num_views}, label_views={self.label_views}, "
            f"target={self.target!r}, selection={self.selection!r}"
        )


_OPTIONS = """hemisphere: Keep only the views whose approach points away from the camera (camera
            ``z`` of the approach above 0), as RGB Matters does with 60 of 120 views.
        target: ``score`` regresses a score for every view (the view-wise graspness of
            GSNet, min-max normalised per seed); ``best_view`` classifies which view is
            the best one (softmax over the views).
        selection: Read-out in evaluation mode. ``argmax`` takes the best view;
            ``soft_argmax`` returns the softmax-weighted mean direction of the ``top_k``
            best views (all views when ``top_k`` is null), a direction between lattice
            points.
        train_selection: Choice in training mode. ``multinomial`` draws a view with
            probability proportional to the detached, min-max normalised scores (the
            GSNet reference); ``topk`` does the same among the ``top_k`` best views only;
            ``argmax`` takes the best view.
        top_k: Number of best views used by ``soft_argmax`` and ``topk``.
        temperature: Softmax temperature of ``soft_argmax``. Scores regressed with
            ``target: score`` lie in about [0, 1], so use a small value (0.05) there;
            1.0 suits the logits of ``target: best_view``.
        angular_weight: With ``target: best_view``, add this weight times the expected
            angular cost of the predicted view distribution (``loss/view_angular``, the
            class loss of RGB Matters): a wrong view costs ``(1 - cos)/2`` of its angle to
            the best view, so near misses are cheap. 0 switches it off.
        label_views: Size of the Fibonacci lattice the grasp labels are indexed by (300
            for GraspNet-1Billion and GraspClutter6D). Not a free choice.
        max_friction: A labelled grasp counts towards the graspness of its view when its
            friction coefficient lies in ``(0, max_friction]``. Compact labels must have
            been encoded with the same value.
        residual: Add the hidden features of the stage to the seed features that the
            grouper and the head consume, as GSNet does.
        loss: Loss of the view scores. ``None`` selects ``SmoothL1`` with weight 100 for
            ``target: score`` and ``CrossEntropy`` with weight 1 for ``best_view``.
        in_channels: Channels of the seed features; injected by the detector."""


@VIEWS.register()
class LatticeViewStage(_LatticeStage):
    __doc__ = f"""Score a Fibonacci lattice of any size; labels stay on their own lattice.

    With the defaults this computes what :class:`TemplateViewStage` computes; the
    options change the size of the lattice, the kind of target, the read-out and the
    training-time sampling. See the module docstring for how a lattice that is not the
    label lattice is supervised.

    Args:
        num_views: Size of the prediction lattice (before ``hemisphere``).
        {_OPTIONS}
    """

    status = "tested"
    source = (
        "view scoring of GraspNet-1Billion (Fang et al., CVPR 2020) and GSNet (Wang et al., "
        "ICCV 2021) on a free lattice, written for GraspPanda; hemisphere template and "
        "angular class loss of RGB Matters (Gou et al., ICRA 2021), the loss adapted from "
        "GouMinghao/rgb_matters@2e176728f6, MIT"
    )

    def __init__(
        self,
        num_views: int = 300,
        hemisphere: bool = False,
        target: Literal["score", "best_view"] = "score",
        selection: Literal["argmax", "soft_argmax"] = "argmax",
        train_selection: Literal["multinomial", "argmax", "topk"] = "multinomial",
        top_k: int | None = None,
        temperature: float = 1.0,
        angular_weight: float = 0.0,
        label_views: int = 300,
        max_friction: float = 0.6,
        residual: bool = True,
        loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self._setup(
            fibonacci_views(num_views),
            hemisphere=hemisphere,
            target=target,
            selection=selection,
            train_selection=train_selection,
            top_k=top_k,
            temperature=temperature,
            angular_weight=angular_weight,
            label_views=label_views,
            max_friction=max_friction,
            residual=residual,
            loss=loss,
            in_channels=in_channels,
        )


@VIEWS.register()
class IcosphereViewStage(_LatticeStage):
    __doc__ = f"""Score the vertices of a subdivided icosahedron (through ``trimesh``).

    12, 42, 162, 642 or 2562 nearly evenly spaced views for ``subdivisions`` 0 to 4.
    Every vertex has five or six neighbours at nearly the same distance, which a
    Fibonacci lattice does not offer; the views are supervised from the 300-view labels.

    Args:
        subdivisions: Subdivision level of the icosahedron: ``10 · 4**n + 2`` views.
        {_OPTIONS}
    """

    status = "bridge"
    source = "icosphere lattice via trimesh (trimesh.creation.icosphere, MIT)"
    extras = ("trimesh",)

    def __init__(
        self,
        subdivisions: int = 2,
        hemisphere: bool = False,
        target: Literal["score", "best_view"] = "score",
        selection: Literal["argmax", "soft_argmax"] = "argmax",
        train_selection: Literal["multinomial", "argmax", "topk"] = "multinomial",
        top_k: int | None = None,
        temperature: float = 1.0,
        angular_weight: float = 0.0,
        label_views: int = 300,
        max_friction: float = 0.6,
        residual: bool = True,
        loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        self.subdivisions = subdivisions
        self._setup(
            icosphere_views(subdivisions),
            hemisphere=hemisphere,
            target=target,
            selection=selection,
            train_selection=train_selection,
            top_k=top_k,
            temperature=temperature,
            angular_weight=angular_weight,
            label_views=label_views,
            max_friction=max_friction,
            residual=residual,
            loss=loss,
            in_channels=in_channels,
        )


@VIEWS.register()
class HealpixViewStage(_LatticeStage):
    __doc__ = f"""Score the pixel centres of a HEALPix grid (through ``healpix``).

    ``12 · nside**2`` views of exactly equal area on rings of constant latitude
    (Górski et al., ApJ 2005): 48, 192, 300, 768 for ``nside`` 2, 4, 5, 8. ``nside: 5``
    has as many views as the label lattice, at other positions.

    Args:
        nside: Resolution parameter of the grid.
        {_OPTIONS}
    """

    status = "bridge"
    source = "HEALPix (Górski et al., ApJ 2005) via healpix (BSD-3-Clause)"
    extras = ("healpix",)

    def __init__(
        self,
        nside: int = 4,
        hemisphere: bool = False,
        target: Literal["score", "best_view"] = "score",
        selection: Literal["argmax", "soft_argmax"] = "argmax",
        train_selection: Literal["multinomial", "argmax", "topk"] = "multinomial",
        top_k: int | None = None,
        temperature: float = 1.0,
        angular_weight: float = 0.0,
        label_views: int = 300,
        max_friction: float = 0.6,
        residual: bool = True,
        loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        self.nside = nside
        self._setup(
            healpix_views(nside),
            hemisphere=hemisphere,
            target=target,
            selection=selection,
            train_selection=train_selection,
            top_k=top_k,
            temperature=temperature,
            angular_weight=angular_weight,
            label_views=label_views,
            max_friction=max_friction,
            residual=residual,
            loss=loss,
            in_channels=in_channels,
        )
