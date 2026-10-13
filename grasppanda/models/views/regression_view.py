"""View stages that regress the approach instead of scoring a lattice of views.

Three stages share one body:

* :class:`DirectionViewStage` regresses the approach vector (3 numbers per seed);
* :class:`TwoVectorViewStage` regresses two vectors and orthonormalises them by
  Gram-Schmidt into a full grasp frame (approach, closing direction, their cross
  product), in the order of the 6-D representation or in the order Contact-GraspNet uses;
* :class:`RomaRotationViewStage` regresses a full rotation through one of the mappings
  of the RoMa library (6-D, 9-D with SVD, quaternion, rotation vector).

What the neighbours receive is the same in all three, and the same as from a stage
that scores a lattice: ``ApproachSet.direction`` is the regressed unit approach,
``rotation`` its zero-angle frame (``approach_to_rotation``), and ``view_index`` the
view of the 300-view label lattice nearest to it, by which the assigner looks labels
up. The head bundle still predicts the in-plane angle, depth and width in the frame of
that direction, so every registered head bundle works behind them. The in-plane part of
a regressed frame is an auxiliary output (``RegressedApproach.frame``): it is
supervised, logged and available to a caller, and it does not replace the head's angle.

Targets, from the labels the assigner matched:

* direction: the approach of the label view with the largest graspness (the fraction
  of its angle × depth cells with friction in ``(0, max_friction]``); seeds without
  any graspable view are not supervised. Loss ``1 - cos``.
* frame (the two stages with a frame): the label frame of the view the seed was
  snapped to, rotated in-plane to its best grasp (dense labels: the cell with the
  lowest positive friction; compact labels: the labelled angle bin). A parallel-jaw
  grasp turned by half a turn about its approach is the same grasp, so the loss is
  the smaller of the squared Frobenius distances to the target and to that copy.

Training crops use the label frame of the snapped view and inference crops the frame
of the regressed direction; they differ by at most the spacing of the label lattice
(about 9° for 300 views).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.models.base import ViewStage
from grasppanda.ops import approach_to_rotation, nearest_view, two_vector_rotation
from grasppanda.registry import INJECTED, VIEWS
from grasppanda.structures import ApproachSet, ContractError, SeedSet
from grasppanda.utils.imports import require

from .targets import view_graspness
from .template import ViewTemplate

__all__ = [
    "DirectionViewStage",
    "RegressedApproach",
    "RomaRotationViewStage",
    "TwoVectorViewStage",
]

_EPS = 1e-8


@dataclass(eq=False)
class RegressedApproach(ApproachSet):
    """A regressed approach per seed.

    Attributes:
        raw: ``(M, K)`` unconstrained output of the stage, kept for the loss.
        frame: ``(M, 3, 3)`` regressed grasp frame (columns: approach, closing
            direction, their cross product) of the stages that regress one.
    """

    raw: Tensor | None = None
    frame: Tensor | None = None


class _RegressionStage(ViewStage):
    """Shared body: a hidden layer, an unconstrained output, a mapping to a direction."""

    label_formats = ("dense", "compact")
    applies_seed_weights = True

    def _setup(
        self,
        out_channels: int,
        *,
        label_views: int,
        max_friction: float,
        direction_weight: float,
        frame_weight: float,
        num_angle: int,
        residual: bool,
        in_channels: int,
    ) -> None:
        self.label_views = label_views
        self.max_friction = max_friction
        self.direction_weight = float(direction_weight)
        self.frame_weight = float(frame_weight)
        self.num_angle = num_angle
        self.residual = residual
        self.in_channels = in_channels
        #: The label lattice: what the assigner indexes labels by.
        self.template = ViewTemplate.fibonacci(label_views)
        self.hidden = nn.Linear(in_channels, in_channels)
        self.output = nn.Linear(in_channels, out_channels)

    def to_frame(self, raw: Tensor) -> tuple[Tensor, Tensor | None]:
        """``(M, 3)`` approach vectors (any length) and the ``(M, 3, 3)`` frame, if any."""
        raise NotImplementedError

    def forward(
        self, seeds: SeedSet, view_index: Tensor | None = None
    ) -> tuple[RegressedApproach, SeedSet]:
        """Regress one approach per seed.

        Args:
            seeds: Seeds with ``(M, in_channels)`` features.
            view_index: ``(M,)`` indices into the label lattice. A replayed choice: the
                approach is then that view, and the regression is kept for the loss only.

        Returns:
            The approach (``direction`` (M, 3) unit, ``rotation`` (M, 3, 3) zero-angle
            frame of the direction, ``view_index`` (M,) nearest label view, ``raw``, and
            ``frame`` when the stage regresses one) and the seeds, with the stage's
            residual added to their features when ``residual`` is set.
        """
        name = type(self).__name__
        if seeds.feats is None or seeds.feats.shape[-1] != self.in_channels:
            got = None if seeds.feats is None else tuple(seeds.feats.shape)
            raise ContractError(f"{name} needs (M, {self.in_channels}) seed features, got {got}")
        refined = F.relu(self.hidden(seeds.feats))
        raw = self.output(refined)
        vector, frame = self.to_frame(raw)
        direction = vector / vector.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        if view_index is None:
            chosen = direction.detach()
            view_index = nearest_view(chosen, -self.template.views)
        elif tuple(view_index.shape) != (len(seeds),):
            raise ContractError(
                f"view_index must hold one view per seed ({len(seeds)},), got "
                f"{tuple(view_index.shape)}"
            )
        else:
            view_index = view_index.to(raw.device).long()
            chosen = self.template.approach(view_index)
        approach = RegressedApproach(
            direction=chosen,
            rotation=approach_to_rotation(chosen),
            view_index=view_index,
            raw=raw,
            frame=frame,
        )
        if self.residual:
            seeds = replace(seeds, feats=seeds.feats + refined)
        return approach, seeds

    # --------------------------------------------------------------------- targets

    def direction_target(
        self, matched: MatchedLabels | CompactMatchedLabels
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """``(M, 3)`` approach of the best label view, the ``(M,)`` seeds that have a
        graspable view and are supervised, and the seed weights (or ``None``)."""
        graspness, supervised, weight = view_graspness(
            matched, self.max_friction, self.template.num_views, type(self).__name__
        )
        best = graspness.argmax(dim=-1)
        return self.template.approach(best), supervised & (graspness.amax(dim=-1) > 0), weight

    def frame_target(self, matched: MatchedLabels | CompactMatchedLabels) -> tuple[Tensor, Tensor]:
        """``(M, 3, 3)`` label frame of the snapped view at its best in-plane angle and
        the ``(M,)`` seeds for which that frame exists."""
        if isinstance(matched, CompactMatchedLabels):
            usable = matched.valid & matched.view_labelled & (matched.angle >= 0)
            angle_bin, bins = matched.angle.clamp(min=0), self.num_angle
        else:
            friction = matched.friction  # (M, A, D)
            bins = friction.shape[1]
            if bins != self.num_angle:
                raise ContractError(
                    f"the matched labels have {bins} in-plane angle bins, but "
                    f"{type(self).__name__} is configured with num_angle = {self.num_angle}"
                )
            good = (friction > 0) & (friction <= self.max_friction)
            cost = torch.where(good, friction, torch.full_like(friction, float("inf")))
            angle_bin = cost.amin(dim=-1).argmin(dim=-1)
            usable = matched.valid & good.flatten(1).any(dim=-1)
        base = matched.crop_rotation
        # A mirrored object pose (flip augmentation) yields a reflection: no rotation target.
        usable = usable & (torch.linalg.det(base) > 0)
        angle = angle_bin.to(base.dtype) * (math.pi / bins)
        cos, sin = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones_like(cos), torch.zeros_like(cos)
        spin = torch.stack([one, zero, zero, zero, cos, -sin, zero, sin, cos], dim=-1)
        return base @ spin.reshape(-1, 3, 3), usable

    # ------------------------------------------------------------------------ loss

    def loss(
        self, approach: ApproachSet, matched: MatchedLabels | CompactMatchedLabels
    ) -> dict[str, Tensor]:
        """``loss/view`` (``1 - cos`` to the best label view), ``loss/view_frame`` for a
        stage that regresses a frame, and the mean angular errors in degrees as metrics."""
        name = type(self).__name__
        raw = getattr(approach, "raw", None)
        if raw is None:
            raise ContractError(f"{name}.loss needs the approach of its own forward")
        vector, frame = self.to_frame(raw)
        direction = vector / vector.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        target, supervised, weight = self.direction_target(matched)
        cosine = (direction * target).sum(dim=-1)
        error = torch.rad2deg(torch.acos(cosine.detach().clamp(-1.0, 1.0)))
        losses = {
            "loss/view": self.direction_weight * _mean(1.0 - cosine, supervised, weight),
            "metric/view_error_deg": _mean(error, supervised, None),
        }
        if frame is not None:
            goal, usable = self.frame_target(matched)
            flipped = goal * goal.new_tensor([1.0, -1.0, -1.0])
            distance = torch.minimum(
                (frame - goal).square().sum(dim=(-2, -1)),
                (frame - flipped).square().sum(dim=(-2, -1)),
            )
            losses["loss/view_frame"] = self.frame_weight * _mean(distance, usable, weight)
            # ‖R - T‖² = 4 (1 - cos θ) for two rotations θ apart.
            angle = torch.acos((1.0 - distance.detach() / 4.0).clamp(-1.0, 1.0))
            losses["metric/view_frame_error_deg"] = _mean(torch.rad2deg(angle), usable, None)
        return losses


def _mean(values: Tensor, mask: Tensor, weight: Tensor | None) -> Tensor:
    """Mean (weighted when ``weight`` is given) of ``values`` over ``mask``; 0 when empty."""
    if not bool(mask.any()):
        return values.sum() * 0.0
    values = values[mask]
    if weight is None:
        return values.mean()
    weight = weight[mask]
    total = weight.sum()
    return (values * weight).sum() / total if bool(total > 0) else values.sum() * 0.0


_SHARED = """label_views: Size of the Fibonacci lattice the grasp labels are indexed by (300
            for GraspNet-1Billion and GraspClutter6D). Not a free choice.
        max_friction: A labelled grasp counts towards the graspness of its view when its
            friction coefficient lies in ``(0, max_friction]``. Compact labels must have
            been encoded with the same value.
        direction_weight: Weight of ``loss/view``.
        residual: Add the hidden features of the stage to the seed features that the
            grouper and the head consume, as GSNet's view stage does.
        in_channels: Channels of the seed features; injected by the detector."""
_FRAME = """frame_weight: Weight of ``loss/view_frame``.
        num_angle: In-plane angle bins of the label set; the head bundle must use the
            same number (the config check compares them)."""


@VIEWS.register()
class DirectionViewStage(_RegressionStage):
    __doc__ = f"""Regress the approach vector of every seed and snap it to the label lattice.

    The alternative the GraspNet-1Billion paper ablates against view classification,
    and what the approach network of Generalizing-Grasp (Ma et al., CVPR 2024) ships:
    three numbers per seed, normalised, with a cosine loss to the best label view.

    Args:
        concentration: Predict a fourth number, the concentration ``κ`` of a von
            Mises-Fisher distribution around the regressed direction, and train with its
            negative log-likelihood instead of ``1 - cos`` (the direction likelihood of
            vMF-Contact, Shi et al., ICRA 2025; the closed form is the textbook
            one). ``κ`` is a per-seed confidence of the direction; it is logged, not
            used by the decoder. The loss is a density and may be negative.
        {_SHARED}
    """

    status = "tested"
    source = (
        "direct regression of the approach vector (ablation of GraspNet-1Billion, Fang et "
        "al., CVPR 2020; Generalizing-Grasp, Ma et al., CVPR 2024), optional von Mises-Fisher "
        "likelihood (vMF-Contact, Shi et al., ICRA 2025); written for GraspPanda"
    )

    def __init__(
        self,
        label_views: int = 300,
        max_friction: float = 0.6,
        direction_weight: float = 1.0,
        concentration: bool = False,
        residual: bool = True,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        self.concentration = concentration
        self._setup(
            4 if concentration else 3,
            label_views=label_views,
            max_friction=max_friction,
            direction_weight=direction_weight,
            frame_weight=0.0,
            num_angle=0,
            residual=residual,
            in_channels=in_channels,
        )
        del self.num_angle, self.frame_weight  # not options of this stage

    def to_frame(self, raw: Tensor) -> tuple[Tensor, None]:
        return raw[:, :3], None

    def loss(
        self, approach: ApproachSet, matched: MatchedLabels | CompactMatchedLabels
    ) -> dict[str, Tensor]:
        """``loss/view``: ``1 - cos`` to the best label view, or with ``concentration``
        the negative log-likelihood of that view under the predicted von Mises-Fisher
        distribution; plus the mean angular error (and concentration) as metrics."""
        if not self.concentration:
            return super().loss(approach, matched)
        raw = getattr(approach, "raw", None)
        if raw is None or raw.shape[-1] != 4:
            raise ContractError(f"{type(self).__name__}.loss needs the approach of its own forward")
        direction = raw[:, :3] / raw[:, :3].norm(dim=-1, keepdim=True).clamp(min=_EPS)
        kappa = F.softplus(raw[:, 3]) + 1e-3
        target, supervised, weight = self.direction_target(matched)
        cosine = (direction * target).sum(dim=-1)
        # log p = κ cos θ + log C(κ), with log C(κ) = log κ - log 2π - κ - log(1 - exp(-2κ)),
        # written with (cos θ - 1) so that the two terms of size κ cancel.
        log_p = (
            kappa * (cosine - 1.0)
            + torch.log(kappa)
            - math.log(2.0 * math.pi)
            - torch.log1p(-torch.exp(-2.0 * kappa))
        )
        error = torch.rad2deg(torch.acos(cosine.detach().clamp(-1.0, 1.0)))
        return {
            "loss/view": self.direction_weight * _mean(-log_p, supervised, weight),
            "metric/view_error_deg": _mean(error, supervised, None),
            "metric/view_concentration": _mean(kappa.detach(), supervised, None),
        }


@VIEWS.register()
class TwoVectorViewStage(_RegressionStage):
    __doc__ = f"""Regress two vectors per seed and orthonormalise them into a grasp frame.

    Six numbers per seed: an approach vector and a baseline (closing direction).
    ``keep: approach`` normalises the approach and projects the baseline onto its
    normal plane, which is the 6-D rotation representation (Zhou et al., CVPR 2019);
    ``keep: baseline`` normalises the baseline and projects the approach, the order of
    Contact-GraspNet (Sundermeyer et al., ICRA 2021), whose idea this is (none of its
    code is used).

    Args:
        keep: Which of the two vectors keeps its direction in the Gram-Schmidt step.
        {_SHARED}
        {_FRAME}
    """

    status = "tested"
    source = (
        "two-vector Gram-Schmidt frame: 6-D rotation representation (Zhou et al., CVPR "
        "2019) and the approach/baseline parameterisation of Contact-GraspNet (Sundermeyer "
        "et al., ICRA 2021); written for GraspPanda"
    )

    def __init__(
        self,
        keep: Literal["approach", "baseline"] = "approach",
        label_views: int = 300,
        max_friction: float = 0.6,
        direction_weight: float = 1.0,
        frame_weight: float = 1.0,
        num_angle: int = 12,
        residual: bool = True,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if keep not in ("approach", "baseline"):
            raise ValueError(f"unknown keep '{keep}'; choose 'approach' or 'baseline'")
        self.keep = keep
        self._setup(
            6,
            label_views=label_views,
            max_friction=max_friction,
            direction_weight=direction_weight,
            frame_weight=frame_weight,
            num_angle=num_angle,
            residual=residual,
            in_channels=in_channels,
        )

    def to_frame(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        approach, baseline = raw[:, :3], raw[:, 3:]
        if self.keep == "approach":
            frame = two_vector_rotation(approach, baseline)
        else:
            swapped = two_vector_rotation(baseline, approach)  # columns b, a', b × a'
            frame = torch.stack([swapped[..., 1], swapped[..., 0], -swapped[..., 2]], dim=-1)
        return frame[..., 0], frame


#: representation → (output channels, shape handed to RoMa, RoMa function).
_ROMA = {
    "6d": (6, (3, 2), "special_gramschmidt"),
    "9d": (9, (3, 3), "special_procrustes"),
    "quaternion": (4, (4,), "unitquat_to_rotmat"),
    "rotvec": (3, (3,), "rotvec_to_rotmat"),
}


@VIEWS.register()
class RomaRotationViewStage(_RegressionStage):
    __doc__ = f"""Regress a full rotation per seed through a mapping of the RoMa library.

    ``representation`` picks the mapping from the unconstrained output to SO(3): ``6d``
    (Gram-Schmidt on two columns, Zhou et al., CVPR 2019), ``9d`` (special Procrustes:
    the nearest rotation by SVD, Levinson et al., arXiv 2006.14616), ``quaternion``
    (normalised, XYZW) or ``rotvec`` (axis-angle). The first column of the rotation is
    the approach.

    Args:
        representation: The mapping, see above.
        {_SHARED}
        {_FRAME}
    """

    status = "bridge"
    source = (
        "rotation mappings via RoMa (Brégier, 3DV 2021; BSD-3-Clause): 6-D (Zhou et al., CVPR 2019), "
        "9-D + SVD (Levinson et al., arXiv 2006.14616), quaternion, rotation vector"
    )
    extras = ("roma",)

    def __init__(
        self,
        representation: Literal["6d", "9d", "quaternion", "rotvec"] = "6d",
        label_views: int = 300,
        max_friction: float = 0.6,
        direction_weight: float = 1.0,
        frame_weight: float = 1.0,
        num_angle: int = 12,
        residual: bool = True,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if representation not in _ROMA:
            raise ValueError(
                f"unknown representation '{representation}'; choose one of {tuple(_ROMA)}"
            )
        self.representation = representation
        require("roma", extra="roma", purpose="RomaRotationViewStage")
        self._setup(
            _ROMA[representation][0],
            label_views=label_views,
            max_friction=max_friction,
            direction_weight=direction_weight,
            frame_weight=frame_weight,
            num_angle=num_angle,
            residual=residual,
            in_channels=in_channels,
        )

    def to_frame(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        _, shape, function = _ROMA[self.representation]
        value = raw.reshape(len(raw), *shape)
        if self.representation == "quaternion":
            value = value / value.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        roma = require("roma", extra="roma", purpose="RomaRotationViewStage")
        frame = getattr(roma, function)(value)
        return frame[..., 0], frame
