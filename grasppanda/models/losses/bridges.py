"""Losses that run the code of an optional library: torchvision and kornia.

Each class is a thin adapter: it reshapes what a grasp stage hands to a primitive into
what the library function expects, calls it without reduction, and leaves mask, weights
and the mean to :class:`~grasppanda.models.losses.elementwise.ElementwiseLoss`. The
library is imported when the loss is built, so the core install is enough to import
this module; a missing library is reported with the extra to install.
"""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

from grasppanda.registry import LOSSES
from grasppanda.structures import ContractError
from grasppanda.utils.imports import require

from .detection import SetLoss, check_unit_target
from .elementwise import DIFFERENCE_TERMS, ElementwiseLoss, check_class_term

__all__ = ["DiceLoss", "FocalLoss", "LovaszSoftmaxLoss", "RobustLoss", "SoftmaxFocalLoss"]

#: ``RobustLoss.kind`` → the function of ``kornia.losses`` that implements it.
ROBUST_KINDS = ("charbonnier", "welsch", "cauchy", "geman_mcclure")


@LOSSES.register()
class FocalLoss(ElementwiseLoss):
    """Sigmoid focal loss: cross-entropy that turns down well-classified elements.

    Runs ``torchvision.ops.sigmoid_focal_loss``. On a ``class`` term the indices are
    expanded to one-hot targets and every class is a binary problem of its own (the
    one-vs-all form of RetinaNet); the penalty of an element is the sum over its classes.
    On a ``unit`` term the prediction is a logit and the target a value in ``[0, 1]``;
    the modulating factor is exact for binary targets, for graded ones ``QualityFocal``
    is the loss designed for it.

    Args:
        weight: Weight of the term.
        alpha: Balance between positives (``alpha``) and negatives (``1 - alpha``);
            ``-1`` switches the balance off.
        gamma: Exponent of the modulating factor ``(1 - p_t)^gamma``; 0 is the plain
            sigmoid cross-entropy.
    """

    source = "Focal Loss for Dense Object Detection (Lin et al., ICCV 2017), via torchvision.ops"
    status = "bridge"
    extras = ("torchvision",)
    accepts = ("class", "unit")
    prediction = "logit"

    def __init__(self, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__(weight)
        if gamma < 0:
            raise ValueError(f"gamma must not be negative, got {gamma}")
        self.alpha = alpha
        self.gamma = gamma
        self._function = require(
            "torchvision.ops", extra="torchvision", purpose="FocalLoss"
        ).sigmoid_focal_loss

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape == target.shape:
            check_unit_target(type(self).__name__, target)
        else:
            check_class_term(type(self).__name__, prediction, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        classes = prediction.shape != target.shape
        if classes:
            target = F.one_hot(target.long(), prediction.shape[1])
        values = self._function(
            prediction, target.type_as(prediction), self.alpha, self.gamma, "none"
        )
        return values.sum(dim=1) if classes else values


@LOSSES.register()
class SoftmaxFocalLoss(ElementwiseLoss):
    """Softmax focal loss: ``(1 - p_t)^gamma x cross-entropy`` over mutually exclusive classes.

    Runs ``kornia.losses.focal_loss``. Unlike :class:`FocalLoss` the classes compete in
    one softmax, as they do under ``CrossEntropy``, so it is the drop-in choice for angle,
    depth and quality classes.

    Args:
        weight: Weight of the term.
        gamma: Exponent of the modulating factor; 0 is the plain cross-entropy.
        alpha: Factor on every class but the first (kornia's convention: class 0 gets
            ``1 - alpha``); ``None`` weights all classes alike.
    """

    source = "Focal Loss for Dense Object Detection (Lin et al., ICCV 2017), via kornia.losses"
    status = "bridge"
    extras = ("kornia",)
    accepts = ("class",)

    def __init__(self, weight: float = 1.0, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__(weight)
        if gamma < 0:
            raise ValueError(f"gamma must not be negative, got {gamma}")
        self.gamma = gamma
        self.alpha = alpha
        self._function = require(
            "kornia.losses", extra="kornia", purpose="SoftmaxFocalLoss"
        ).focal_loss

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        check_class_term(type(self).__name__, prediction, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        # (N, C) per-class terms, zero off the target class.
        return self._function(prediction, target.long(), self.alpha, self.gamma, "none").sum(dim=1)


class _Overlap(SetLoss):
    """Segmentation losses of kornia on a ``class`` term: one value for the selected set."""

    accepts = ("class",)
    applies_weights = False

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        check_class_term(type(self).__name__, prediction, target)

    def overlap(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Loss of ``(1, C, n, 1)`` logits against ``(1, n, 1)`` labels, a scalar."""
        raise NotImplementedError

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        if weights is not None:
            raise ContractError(
                f"{type(self).__name__} measures the overlap of two sets and cannot weight "
                "single elements, but the stage passes a weight per element (an assigner "
                "with seed weights). Use CrossEntropy, FocalLoss or SoftmaxFocalLoss there."
            )
        return super().forward(prediction, target, mask)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        # The library reads images: the selected elements become one column of pixels.
        logits = prediction.transpose(0, 1)[None, :, :, None]
        return self.overlap(logits, target.long()[None, :, None]).reshape(1)


@LOSSES.register()
class DiceLoss(_Overlap):
    """Dice loss: one minus the overlap of predicted and labelled classes over the set.

    Runs ``kornia.losses.dice_loss`` on the softmax of the logits. It is a property of
    the whole supervised set, not a mean of per-element penalties: its value does not
    shrink when one class is rare, which is why segmentation uses it for small
    foregrounds (the objectness of scene points or seeds is such a term).

    Args:
        weight: Weight of the term.
        average: ``micro`` computes one overlap across all classes, ``macro`` one per
            class and averages them.
        eps: Guard of the division.
    """

    source = "V-Net (Milletari et al., arXiv 1606.04797), via kornia.losses"
    status = "bridge"
    extras = ("kornia",)

    def __init__(
        self,
        weight: float = 1.0,
        average: Literal["micro", "macro"] = "micro",
        eps: float = 1e-8,
    ) -> None:
        super().__init__(weight)
        if average not in ("micro", "macro"):
            raise ValueError(f"unknown average '{average}'")
        self.average = average
        self.eps = eps
        self._function = require("kornia.losses", extra="kornia", purpose="DiceLoss").dice_loss

    def overlap(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self._function(logits, labels, average=self.average, eps=self.eps)


@LOSSES.register()
class LovaszSoftmaxLoss(_Overlap):
    """Lovász-Softmax: a convex surrogate of the mean intersection-over-union of the set.

    Runs ``kornia.losses.lovasz_softmax_loss``. Like :class:`DiceLoss` it scores the
    supervised set as a whole; it optimises the Jaccard index of every class directly.

    Args:
        weight: Weight of the term.
    """

    source = "The Lovász-Softmax loss (Berman et al., CVPR 2018), via kornia.losses"
    status = "bridge"
    extras = ("kornia",)

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._function = require(
            "kornia.losses", extra="kornia", purpose="LovaszSoftmaxLoss"
        ).lovasz_softmax_loss

    def overlap(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self._function(logits, labels)


@LOSSES.register()
class RobustLoss(ElementwiseLoss):
    """Robust regression penalties of kornia: Charbonnier, Welsch, Cauchy, Geman-McClure.

    All four are quadratic for small errors and grow slowly (Charbonnier: linearly;
    Cauchy: logarithmically) or not at all (Welsch, Geman-McClure: bounded) for large
    ones, so a wrong label pulls less than under ``SmoothL1``. The scale of "small" is 1
    in the units of the target; rescale with the target scale of the head, not here.

    Args:
        weight: Weight of the term.
        kind: ``charbonnier``, ``welsch``, ``cauchy`` or ``geman_mcclure``.
    """

    source = (
        "A General and Adaptive Robust Loss Function (Barron, CVPR 2019), special cases, "
        "via kornia.losses"
    )
    status = "bridge"
    extras = ("kornia",)
    accepts = DIFFERENCE_TERMS

    def __init__(
        self,
        weight: float = 1.0,
        kind: Literal["charbonnier", "welsch", "cauchy", "geman_mcclure"] = "charbonnier",
    ) -> None:
        super().__init__(weight)
        if kind not in ROBUST_KINDS:
            raise ValueError(f"unknown kind '{kind}'; kinds are {', '.join(ROBUST_KINDS)}")
        self.kind = kind
        losses = require("kornia.losses", extra="kornia", purpose="RobustLoss")
        self._function = getattr(losses, f"{kind}_loss")

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self._function(prediction, target, reduction="none")
