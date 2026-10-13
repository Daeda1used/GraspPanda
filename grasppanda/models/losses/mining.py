"""Wrappers that change *which* elements count, or with what weight, around a primitive.

A wrapper is a primitive that holds another one (``loss: {type: ...}``) and fits the
loss terms its inner primitive fits::

    objectness_loss:
      type: TopK                  # keep the hardest quarter of the points
      fraction: 0.25
      loss: {type: CrossEntropy, weight: 1.0}

The weight of the term is the weight of the inner primitive; a wrapper has none.
``UncertaintyWeighted`` is the wrapper that learns a factor on top of that weight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from grasppanda.models.base import Loss
from grasppanda.registry import LOSSES, Cfg
from grasppanda.structures import ContractError

from .detection import SetLoss
from .elementwise import ElementwiseLoss, check_class_term
from .terms import accepted_terms

__all__ = ["ClassBalanced", "TopK", "UncertaintyWeighted"]


def _inner_terms(options: Mapping[str, Any], limit: tuple[str, ...]) -> tuple[str, ...]:
    found = accepted_terms(options.get("loss"))
    if found is None:
        return limit
    return tuple(kind for kind in found[1] if kind in limit)


@LOSSES.register()
class TopK(Loss):
    """Online hard example mining: only the hardest fraction of the elements counts.

    The inner primitive computes its penalty for every supervised element; the
    ``fraction`` with the largest penalties is kept and averaged (weighted, when the
    stage passes weights), the rest gets no gradient. The selection runs over the
    elements of the whole call, i.e. of the batch, since a primitive does not see frame
    boundaries.

    Args:
        loss: The primitive whose penalties are mined. It must give one penalty per
            element (not ``GHMC``, ``GHMR``, ``DiceLoss`` or ``LovaszSoftmaxLoss``, which
            score the supervised set as a whole).
        fraction: Share of the supervised elements that is kept, in ``(0, 1]``.
        min_kept: Keep at least this many elements (all, if there are fewer).
    """

    source = (
        "Online Hard Example Mining (Shrivastava et al., CVPR 2016); top-k of the penalties "
        "as in Contact-GraspNet (Sundermeyer et al., ICRA 2021), written here"
    )
    status = "tested"
    accepts = ("class", "unit", "value", "bins")

    @classmethod
    def accepted_terms(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        return _inner_terms(options, cls.accepts)

    def __init__(self, loss: Cfg[Loss], fraction: float = 0.25, min_kept: int = 1) -> None:
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
        if min_kept < 1:
            raise ValueError(f"min_kept must be positive, got {min_kept}")
        self.loss = LOSSES.build(loss)
        if not isinstance(self.loss, ElementwiseLoss) or isinstance(self.loss, SetLoss):
            raise ValueError(
                f"TopK mines per-element penalties, but {type(self.loss).__name__} scores the "
                "supervised set as a whole. Wrap an element-wise primitive (CrossEntropy, "
                "FocalLoss, SmoothL1, QualityFocal, ...)."
            )
        self.fraction = fraction
        self.min_kept = min_kept

    def as_value(self, output: Tensor) -> Tensor:
        return self.loss.as_value(output)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        inner = self.loss
        inner._check(prediction, target)
        values = inner.penalty(prediction, target)
        for name, extra in (("mask", mask), ("weights", weights)):
            if extra is not None and extra.shape != values.shape:
                raise ContractError(
                    f"TopK: {name} {tuple(extra.shape)} does not match the penalty "
                    f"{tuple(values.shape)}"
                )
        if mask is not None:
            values = values[mask]
            weights = None if weights is None else weights[mask]
        values = values.reshape(-1)
        if values.numel() == 0:
            return prediction.sum() * 0.0
        keep = min(values.numel(), max(self.min_kept, math.ceil(self.fraction * values.numel())))
        hardest = values.detach().topk(keep).indices
        values = values[hardest]
        if weights is None:
            return inner.weight * values.mean()
        weights = weights.reshape(-1)[hardest]
        total = weights.sum()
        if not bool(total > 0):
            return prediction.sum() * 0.0
        return inner.weight * (values * weights).sum() / total

    def extra_repr(self) -> str:
        return f"fraction={self.fraction}, min_kept={self.min_kept}"


@LOSSES.register()
class ClassBalanced(Loss):
    """Class-balanced weighting: every class counts by the inverse of its effective number.

    The elements of class ``c`` are weighted by ``(1 - beta) / (1 - beta^n_c)``, where
    ``n_c`` is the number of elements of the class; the factors are scaled to sum to the
    number of classes present. ``beta = 0`` weights nothing, ``beta → 1`` approaches the
    inverse class frequency. The factors multiply into the weights the stage passes, and
    the inner primitive takes the weighted mean.

    Args:
        loss: A primitive for ``class`` terms that applies weights (``CrossEntropy``,
            ``FocalLoss``, ``SoftmaxFocalLoss``, ``Poly1CrossEntropy``, ``GHMC``).
        beta: The hyper-parameter of the effective number, in ``[0, 1)``.
        class_counts: Elements per class in the training set, one number per class.
            ``None`` counts the supervised elements of every call instead: the weights
            then follow the batch, which needs no statistic of the label set (the label
            set a stage sees depends on its proposal and assigner).
    """

    source = (
        "Class-Balanced Loss Based on Effective Number of Samples (Cui et al., CVPR 2019); "
        "the weight formula of the paper, written here"
    )
    status = "tested"
    accepts = ("class",)

    @classmethod
    def accepted_terms(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        return _inner_terms(options, cls.accepts)

    def __init__(
        self, loss: Cfg[Loss], beta: float = 0.999, class_counts: list[float] | None = None
    ) -> None:
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must lie in [0, 1), got {beta}")
        self.loss = LOSSES.build(loss)
        if "class" not in type(self.loss).accepts or not type(self.loss).applies_weights:
            raise ValueError(
                f"ClassBalanced weights the elements of a `class` term, but "
                f"{type(self.loss).__name__} "
                + (
                    "cannot weight single elements."
                    if "class" in type(self.loss).accepts
                    else "is no loss for class terms."
                )
                + " Wrap CrossEntropy, FocalLoss, SoftmaxFocalLoss, Poly1CrossEntropy or GHMC."
            )
        self.beta = beta
        self.class_counts = None if class_counts is None else [float(v) for v in class_counts]

    def as_value(self, output: Tensor) -> Tensor:
        return self.loss.as_value(output)

    def class_factors(self, counts: Tensor) -> Tensor:
        """``(C,)`` factor per class from ``(C,)`` counts; 0 for a class without elements."""
        present = counts > 0
        if self.beta == 0.0:
            factors = present.to(counts.dtype)
        else:
            effective = (1.0 - torch.pow(self.beta, counts)) / (1.0 - self.beta)
            factors = torch.where(present, 1.0 / effective.clamp(min=1e-12), 0.0)
        return factors / factors.sum().clamp(min=1e-12) * present.sum()

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        check_class_term(type(self).__name__, prediction, target)
        classes = prediction.shape[1]
        if self.class_counts is None:
            selected = target if mask is None else target[mask]
            counts = torch.bincount(selected.long().reshape(-1), minlength=classes)
        elif len(self.class_counts) != classes:
            raise ContractError(
                f"ClassBalanced: class_counts lists {len(self.class_counts)} value(s), but the "
                f"term has {classes} classes"
            )
        else:
            counts = prediction.new_tensor(self.class_counts)
        factors = self.class_factors(counts.to(prediction.dtype))[target.long()]
        weights = factors if weights is None else weights * factors
        return self.loss(prediction, target, mask, weights)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, class_counts={self.class_counts}"


@LOSSES.register()
class UncertaintyWeighted(Loss):
    """Uncertainty weighting: a learned factor on the term, which pays for being small.

    The wrapped term ``L`` becomes ``L / (2 exp(s)) + s / 2`` with one learned ``s`` (the
    log-variance of the task) per wrapped term. A term that is hard to fit raises its
    ``s`` and counts less; the ``s / 2`` keeps every ``s`` from growing without bound.
    Wrap every loss term that should take part in the balance; a term that is not
    wrapped keeps its fixed weight. ``s`` is a parameter of the model: it is optimised
    with the network, saved in the checkpoint, and decayed by a weight decay that covers
    all parameters. The value of a wrapped term can be negative.

    Args:
        loss: The primitive of the term, with its weight (the factor multiplies it).
        initial_log_variance: ``s`` at the start. The default ``-ln 2`` makes the factor
            1, so training starts from the weights of the config.
    """

    source = (
        "Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., CVPR 2018); "
        "the weighting formula as in LibMTL (UW), one log-variance per wrapped term, written "
        "here"
    )
    status = "tested"
    accepts = ("class", "unit", "value", "bins")

    @classmethod
    def accepted_terms(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        return _inner_terms(options, cls.accepts)

    def __init__(self, loss: Cfg[Loss], initial_log_variance: float = -math.log(2.0)) -> None:
        super().__init__()
        self.loss = LOSSES.build(loss)
        self.initial_log_variance = float(initial_log_variance)
        self.log_variance = nn.Parameter(torch.tensor(self.initial_log_variance))

    def as_value(self, output: Tensor) -> Tensor:
        return self.loss.as_value(output)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        value = self.loss(prediction, target, mask, weights)
        if target.numel() == 0 or (mask is not None and not bool(mask.any())):
            return value  # nothing supervised: no data, and no pull on the log-variance
        return value / (2.0 * self.log_variance.exp()) + self.log_variance / 2.0

    def extra_repr(self) -> str:
        return f"initial_log_variance={self.initial_log_variance:.4g}"
