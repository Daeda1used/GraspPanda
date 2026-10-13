# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for GraspPanda from https://github.com/open-mmlab/mmdetection
# @ cfd5d3a985b0249de009b67d04f37263e11cdf3d (Apache-2.0):
# mmdet/models/losses/{gfocal_loss,varifocal_loss,ghm_loss,balanced_l1_loss}.py.
"""Losses of dense object detectors, as primitives of grasp stages.

The formulas are those of MMDetection. What changed is the frame around them: every
class here is a primitive in the sense of :mod:`.elementwise` (a penalty per element, a
mask, optional weights, ``weight x mean``), the reductions by "number of positives" of
the detection code are gone, and each class states the kind of loss term it fits.

Three of them read the prediction as a *logit* (``prediction = "logit"``): quality focal,
varifocal and GHM-C. They fit ``unit`` terms only, i.e. targets that are qualities in
``[0, 1]``, and the stage that owns the output hands it on through ``Loss.as_value``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from grasppanda.registry import LOSSES
from grasppanda.structures import ContractError

from .elementwise import DIFFERENCE_TERMS, ElementwiseLoss, check_class_term

__all__ = [
    "GHMC",
    "GHMR",
    "BalancedL1",
    "DistributionFocal",
    "QualityFocal",
    "SetLoss",
    "Varifocal",
    "check_unit_target",
]


def check_unit_target(name: str, target: Tensor) -> None:
    """Refuse targets outside ``[0, 1]`` for a loss that reads the prediction as a logit."""
    if target.numel() and not 0.0 <= float(target.min()) <= float(target.max()) <= 1.0:
        raise ContractError(
            f"{name} reads the prediction as a logit and needs targets in [0, 1], got "
            f"[{float(target.min()):.3g}, {float(target.max()):.3g}]. This is a `value` term "
            "(a width, a tolerance, an unnormalised score): a sigmoid cross-entropy has no "
            "minimum there. Use SmoothL1, Huber, L1, MSE, BalancedL1 or GHMR."
        )


class SetLoss(ElementwiseLoss):
    """Base of primitives whose penalty depends on the *set* of supervised elements.

    The mask is applied first and :meth:`penalty` sees the selected elements only, so a
    statistic over them (a histogram of gradient norms, a ranking) never counts a masked
    cell.
    """

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        self._check(prediction, target)
        rows = target.shape
        for name, extra in (("mask", mask), ("weights", weights)):
            if extra is not None and extra.shape != rows:
                raise ContractError(
                    f"{type(self).__name__}: {name} {tuple(extra.shape)} does not match the "
                    f"targets {tuple(rows)}"
                )
        if mask is not None:
            prediction, target = prediction[mask], target[mask]
            weights = None if weights is None else weights[mask]
        if target.numel() == 0:
            return prediction.sum() * 0.0
        values = self.penalty(prediction, target)
        if weights is None:
            return self.weight * values.mean()
        total = weights.sum()
        if not bool(total > 0):
            return prediction.sum() * 0.0
        return self.weight * (values * weights).sum() / total


@LOSSES.register()
class QualityFocal(ElementwiseLoss):
    """Quality focal loss: sigmoid cross-entropy to a soft target, focused on large errors.

    ``|y - sigmoid(x)|^beta x BCE(x, y)`` per element, for a target quality ``y`` in
    ``[0, 1]``. The prediction is a logit. In a grasp head the target is the normalised
    grasp score of a cell (0 where there is no usable grasp), so the many empty cells of
    a score grid are down-weighted once they are predicted as empty.

    Args:
        weight: Weight of the term.
        beta: Exponent of the modulating factor; 0 is the plain soft-target cross-entropy.
    """

    source = (
        "Generalized Focal Loss (Li et al., NeurIPS 2020); adapted from "
        "open-mmlab/mmdetection@cfd5d3a, Apache-2.0"
    )
    status = "tested"
    accepts = ("unit",)
    prediction = "logit"

    def __init__(self, weight: float = 1.0, beta: float = 2.0) -> None:
        super().__init__(weight)
        if beta < 0:
            raise ValueError(f"beta must not be negative, got {beta}")
        self.beta = beta

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        super()._check(prediction, target)
        check_unit_target(type(self).__name__, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Upstream treats the cells with target 0 apart; the expression is the same one.
        target = target.type_as(prediction)
        scale = (target - prediction.sigmoid()).abs().pow(self.beta)
        return F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * scale


@LOSSES.register()
class Varifocal(ElementwiseLoss):
    """Varifocal loss: positives weighted by their target quality, negatives focused.

    A sigmoid cross-entropy to a soft target ``y`` in ``[0, 1]`` whose weight is ``y``
    where ``y > 0`` and ``alpha x |sigmoid(x) - y|^gamma`` where ``y = 0``. The prediction
    is a logit. Good grasps dominate the term; a cell with a poor but usable grasp counts
    little, which is the intent of the loss and the thing to know before using it.

    Args:
        weight: Weight of the term.
        alpha: Factor on the cells with target 0.
        gamma: Exponent of their modulating factor.
        quality_weighted: Weight the cells with a positive target by that target
            (upstream ``iou_weighted``); false weights them by 1.
    """

    source = (
        "VarifocalNet (Zhang et al., CVPR 2021); adapted from "
        "open-mmlab/mmdetection@cfd5d3a, Apache-2.0"
    )
    status = "tested"
    accepts = ("unit",)
    prediction = "logit"

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
        quality_weighted: bool = True,
    ) -> None:
        super().__init__(weight)
        self.alpha = alpha
        self.gamma = gamma
        self.quality_weighted = quality_weighted

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        super()._check(prediction, target)
        check_unit_target(type(self).__name__, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = target.type_as(prediction)
        positive = (target > 0.0).to(prediction.dtype)
        focal = self.alpha * (prediction.sigmoid() - target).abs().pow(self.gamma)
        focal_weight = (target if self.quality_weighted else 1.0) * positive + focal * (
            1.0 - positive
        )
        return (
            F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
            * focal_weight.detach()
        )


@LOSSES.register()
class DistributionFocal(ElementwiseLoss):
    """Distribution focal loss: cross-entropy to the two bins around a continuous target.

    The prediction is a distribution over ``C`` bins, ``(N, C)`` logits; the target is a
    position ``y`` in ``[0, C - 1]`` in units of bins. The loss is
    ``(y_r - y) x CE(left bin) + (y - y_l) x CE(right bin)``, so the expectation of the
    learned distribution is pulled to ``y``. A head that owns such a term decodes the
    expectation (``DecoupledGridHead`` with ``width: bins``).

    Args:
        weight: Weight of the term.
    """

    source = (
        "Generalized Focal Loss (Li et al., NeurIPS 2020); adapted from "
        "open-mmlab/mmdetection@cfd5d3a, Apache-2.0"
    )
    status = "tested"
    accepts = ("bins",)

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        name = type(self).__name__
        if prediction.ndim != 2 or target.shape != prediction.shape[:1]:
            raise ContractError(
                f"{name} expects (N, C) logits over bins and (N,) bin positions, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        last = prediction.shape[1] - 1
        if last < 1:
            raise ContractError(f"{name} needs at least two bins, got {last + 1}")
        if target.numel() and not 0.0 <= float(target.min()) <= float(target.max()) <= last:
            raise ContractError(
                f"{name}: bin positions must lie in [0, {last}] for {last + 1} bins, got "
                f"[{float(target.min()):.3g}, {float(target.max()):.3g}]"
            )

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = target.type_as(prediction)
        # Upstream clamps its targets below the last bin; here the last bin is a target.
        left = target.long().clamp(max=prediction.shape[1] - 2)
        right = left + 1
        weight_left = right.to(target.dtype) - target
        weight_right = target - left.to(target.dtype)
        return (
            F.cross_entropy(prediction, left, reduction="none") * weight_left
            + F.cross_entropy(prediction, right, reduction="none") * weight_right
        )


@LOSSES.register()
class BalancedL1(ElementwiseLoss):
    """Balanced L1: Smooth-L1 with a larger gradient for small errors.

    Below ``beta`` the gradient grows as ``alpha x ln(b |d| / beta + 1)``, above it is the
    constant ``gamma``; inliers therefore contribute more than under Smooth-L1 while
    outliers stay clipped.

    Args:
        weight: Weight of the term.
        beta: Error at which the penalty turns linear.
        alpha: Smaller values give inliers more gradient.
        gamma: Gradient of the linear part.
    """

    source = (
        "Libra R-CNN (Pang et al., CVPR 2019); adapted from "
        "open-mmlab/mmdetection@cfd5d3a, Apache-2.0"
    )
    status = "tested"
    accepts = DIFFERENCE_TERMS

    def __init__(
        self, weight: float = 1.0, beta: float = 1.0, alpha: float = 0.5, gamma: float = 1.5
    ) -> None:
        super().__init__(weight)
        if beta <= 0 or alpha <= 0:
            raise ValueError(f"beta and alpha must be positive, got {beta} and {alpha}")
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = (prediction - target).abs()
        b = math.e ** (self.gamma / self.alpha) - 1
        return torch.where(
            diff < self.beta,
            self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1)
            - self.alpha * diff,
            self.gamma * diff + self.gamma / b - self.alpha * self.beta,
        )


class _GradientHarmonised(SetLoss):
    """Shared part of GHM-C and GHM-R: weights from the density of gradient norms."""

    def __init__(self, weight: float, bins: int, momentum: float) -> None:
        super().__init__(weight)
        if bins < 1:
            raise ValueError(f"bins must be positive, got {bins}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1), got {momentum}")
        self.bins = bins
        self.momentum = momentum
        #: Running number of elements per bin; used (and checkpointed) with momentum > 0.
        self.register_buffer("acc_sum", torch.zeros(bins))

    def harmonise(self, norm: Tensor, last_edge: float) -> Tensor:
        """Weight per element: total / (elements in its bin x occupied bins)."""
        edges = torch.arange(self.bins + 1, device=norm.device, dtype=norm.dtype) / self.bins
        edges[-1] = last_edge
        weights = torch.zeros_like(norm)
        total = max(float(norm.numel()), 1.0)
        occupied = 0
        for index in range(self.bins):
            inside = (norm >= edges[index]) & (norm < edges[index + 1])
            count = int(inside.sum())
            if count == 0:
                continue
            occupied += 1
            if self.momentum > 0:
                self.acc_sum[index] = (
                    self.momentum * self.acc_sum[index] + (1 - self.momentum) * count
                )
                weights[inside] = total / float(self.acc_sum[index])
            else:
                weights[inside] = total / count
        return weights / occupied if occupied else weights


@LOSSES.register()
class GHMC(_GradientHarmonised):
    """Gradient harmonising, classification form: sigmoid cross-entropy re-weighted so
    that every range of gradient norms contributes alike.

    The gradient norm of an element is ``|sigmoid(x) - y|``. Elements are counted in
    ``bins`` ranges of it over the supervised set of the call, and each is weighted by
    the inverse of the population of its range: the mass of easy elements and the few
    very hard ones (often label noise) are both turned down. On a ``class`` term the
    indices are expanded to one-hot targets, one binary problem per class.

    Args:
        weight: Weight of the term.
        bins: Ranges of the gradient norm.
        momentum: With a value above 0 the populations are a running mean over calls
            (a buffer of the module, so it is checkpointed).
    """

    source = (
        "Gradient Harmonized Single-stage Detector (Li et al., AAAI 2019); adapted from "
        "open-mmlab/mmdetection@cfd5d3a, Apache-2.0"
    )
    status = "tested"
    accepts = ("unit", "class")
    prediction = "logit"

    def __init__(self, weight: float = 1.0, bins: int = 10, momentum: float = 0.0) -> None:
        super().__init__(weight, bins, momentum)

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape == target.shape:
            check_unit_target(type(self).__name__, target)
        else:
            check_class_term(type(self).__name__, prediction, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        classes = prediction.shape != target.shape
        if classes:
            target = F.one_hot(target.long(), prediction.shape[1])
        target = target.type_as(prediction)
        norm = (prediction.sigmoid().detach() - target).abs()
        weights = self.harmonise(norm, 1.0 + 1e-6)
        values = F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * weights
        return values.sum(dim=1) if classes else values


@LOSSES.register()
class GHMR(_GradientHarmonised):
    """Gradient harmonising, regression form: a smooth L1 re-weighted so that every
    range of gradient norms contributes alike.

    The penalty is ``sqrt(d^2 + mu^2) - mu`` (its gradient norm ``|d| / sqrt(d^2 + mu^2)``
    lies in ``[0, 1)``); the weights are those of :class:`GHMC`.

    Args:
        weight: Weight of the term.
        mu: Error scale at which the penalty turns from quadratic to linear, in the
            units of the target. The detection default 0.02 suits unit-normalised
            targets; a width in metres x 10 has the same order of magnitude.
        bins: Ranges of the gradient norm.
        momentum: Running mean of the populations over calls, see :class:`GHMC`.
    """

    source = (
        "Gradient Harmonized Single-stage Detector (Li et al., AAAI 2019); adapted from "
        "open-mmlab/mmdetection@cfd5d3a, Apache-2.0"
    )
    status = "tested"
    accepts = DIFFERENCE_TERMS

    def __init__(
        self, weight: float = 1.0, mu: float = 0.02, bins: int = 10, momentum: float = 0.0
    ) -> None:
        super().__init__(weight, bins, momentum)
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        self.mu = mu

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        root = torch.sqrt(diff * diff + self.mu * self.mu)
        norm = (diff / root).abs().detach()
        return (root - self.mu) * self.harmonise(norm, 1e3)
