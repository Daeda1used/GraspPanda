"""Primitive losses: an element-wise penalty, a mask, a mean and a weight.

Stages own their targets and decide *which* elements count; the primitive only defines
the penalty per element. That keeps a swap such as ``SmoothL1`` → ``L1`` safe: it can
change neither the target structure nor the set of supervised elements.

Every primitive returns ``weight × mean`` over the selected elements, so a stage can put
the result under a ``loss/…`` key directly. With nothing selected the result is a zero
that still depends on the prediction (the reference implementations would produce NaN
from an empty mean and poison the step). A stage may also pass per-element ``weights``;
the mean is then the weighted mean over the selected elements.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from grasppanda.models.base import Loss
from grasppanda.registry import LOSSES
from grasppanda.structures import ContractError

__all__ = ["L1", "CrossEntropy", "ElementwiseLoss", "Huber", "SmoothL1"]

#: Kinds of loss term a penalty on the difference of two equally shaped tensors fits.
DIFFERENCE_TERMS = ("value", "unit")


class ElementwiseLoss(Loss):
    """Base of the primitives: subclasses define :meth:`penalty` only."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Unreduced penalty, one value per supervised element."""
        raise NotImplementedError

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ContractError(
                f"{type(self).__name__}: prediction {tuple(prediction.shape)} and target "
                f"{tuple(target.shape)} differ in shape; losses never broadcast"
            )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Weighted mean penalty over the elements where ``mask`` is true (all if omitted).

        ``weights`` gives every element a non-negative weight; the result is then
        ``Σ w · penalty / Σ w`` over the selected elements instead of their plain mean.
        """
        self._check(prediction, target)
        values = self.penalty(prediction, target)
        for name, extra in (("mask", mask), ("weights", weights)):
            if extra is not None and extra.shape != values.shape:
                raise ContractError(
                    f"{type(self).__name__}: {name} {tuple(extra.shape)} does not match the "
                    f"penalty {tuple(values.shape)}"
                )
        if mask is not None:
            values = values[mask]
            weights = None if weights is None else weights[mask]
        if values.numel() == 0:
            return prediction.sum() * 0.0
        if weights is None:
            return self.weight * values.mean()
        total = weights.sum()
        if not bool(total > 0):
            return prediction.sum() * 0.0
        return self.weight * (values * weights).sum() / total

    def extra_repr(self) -> str:
        return f"weight={self.weight}"


@LOSSES.register()
class SmoothL1(ElementwiseLoss):
    """Huber-style penalty: quadratic below ``beta``, linear above."""

    source = "PyTorch smooth_l1_loss"
    accepts = DIFFERENCE_TERMS

    def __init__(self, weight: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(weight)
        self.beta = beta

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(prediction, target, reduction="none", beta=self.beta)


@LOSSES.register()
class L1(ElementwiseLoss):
    """Absolute error."""

    source = "PyTorch l1_loss"
    accepts = DIFFERENCE_TERMS

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).abs()


@LOSSES.register()
class Huber(ElementwiseLoss):
    """Huber penalty: quadratic below ``delta``, linear with slope ``delta`` above.

    ``Huber(delta) = delta x SmoothL1(beta=delta)``: the two agree at ``delta = 1`` and
    differ in the scale of the linear part elsewhere, so a loss weight tuned for one does
    not carry over to the other.
    """

    source = "PyTorch huber_loss"
    status = "tested"
    accepts = DIFFERENCE_TERMS

    def __init__(self, weight: float = 1.0, delta: float = 1.0) -> None:
        super().__init__(weight)
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = delta

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(prediction, target, reduction="none", delta=self.delta)


@LOSSES.register()
class CrossEntropy(ElementwiseLoss):
    """Softmax cross-entropy between ``(N, C)`` logits and ``(N,)`` class indices."""

    source = "PyTorch cross_entropy"
    accepts = ("class",)

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        check_class_term(type(self).__name__, prediction, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(prediction, target.long(), reduction="none")


def check_class_term(name: str, prediction: Tensor, target: Tensor) -> None:
    """Refuse anything but ``(N, C)`` logits with ``(N,)`` class indices, naming the fix."""
    if prediction.ndim != 2 or target.shape != prediction.shape[:1]:
        raise ContractError(
            f"{name} expects (N, C) logits and (N,) class indices, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if target.is_floating_point():
        raise ContractError(
            f"{name} expects integer class indices, got {target.dtype} targets: this is a "
            "regression term; use a loss for `value` or `unit` terms (SmoothL1, Huber, ...)"
        )
