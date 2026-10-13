"""Classification primitives of PyTorch functions: cross-entropy with label smoothing
and class weights, Poly-1, and sigmoid cross-entropy on soft targets. See
:mod:`.elementwise` for what a primitive is.

``CrossEntropy`` itself keeps its single option: the options of a component a shipped
config uses are part of the hash of that config.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from grasppanda.registry import LOSSES
from grasppanda.structures import ContractError

from .detection import check_unit_target
from .elementwise import ElementwiseLoss, check_class_term

__all__ = ["BinaryCrossEntropy", "LabelSmoothingCrossEntropy", "Poly1CrossEntropy"]


@LOSSES.register()
class LabelSmoothingCrossEntropy(ElementwiseLoss):
    """Softmax cross-entropy with label smoothing and, optionally, a factor per class.

    Args:
        weight: Weight of the term.
        label_smoothing: Mass taken from the target class and spread evenly over all
            ``C`` classes; 0 is the plain cross-entropy. Every output takes part, also
            one that is never a target (the extra class of ``EconomicHead`` with
            ``unmatched_class: true``).
        class_weights: One factor per class on the penalty of the elements of that class
            (``C`` values); ``None`` weights all classes alike. The result stays a mean
            over the selected elements, not PyTorch's weighted mean.
    """

    source = (
        "Rethinking the Inception Architecture (Szegedy et al., CVPR 2016), via PyTorch "
        "cross_entropy"
    )
    status = "tested"
    accepts = ("class",)

    def __init__(
        self,
        weight: float = 1.0,
        label_smoothing: float = 0.1,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__(weight)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must lie in [0, 1), got {label_smoothing}")
        self.label_smoothing = label_smoothing
        self.class_weights = None if class_weights is None else [float(v) for v in class_weights]

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        check_class_term(type(self).__name__, prediction, target)
        if self.class_weights is not None and len(self.class_weights) != prediction.shape[1]:
            raise ContractError(
                f"{type(self).__name__}: class_weights lists {len(self.class_weights)} value(s), "
                f"but the term has {prediction.shape[1]} classes"
            )

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        factors = None
        if self.class_weights is not None:
            factors = prediction.new_tensor(self.class_weights)
        return F.cross_entropy(
            prediction,
            target.long(),
            weight=factors,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )


@LOSSES.register()
class Poly1CrossEntropy(ElementwiseLoss):
    """Poly-1 loss: cross-entropy plus ``epsilon x (1 - p_t)``.

    The cross-entropy is a series in ``(1 - p_t)^j``; Poly-1 changes the coefficient of
    its first term by ``epsilon`` and leaves the rest. A positive ``epsilon`` keeps a
    gradient on elements that are already classified well, a negative one (above ``-1``)
    removes some of it.

    Args:
        weight: Weight of the term.
        epsilon: Change of the first polynomial coefficient; 0 is the plain cross-entropy.
        label_smoothing: As in ``LabelSmoothingCrossEntropy``; it applies to the
            cross-entropy part.
    """

    source = (
        "PolyLoss (Leng et al., ICLR; arXiv 2204.12511); the Poly-1 formula of the paper on "
        "PyTorch cross_entropy"
    )
    status = "tested"
    accepts = ("class",)

    def __init__(
        self, weight: float = 1.0, epsilon: float = 1.0, label_smoothing: float = 0.0
    ) -> None:
        super().__init__(weight)
        if epsilon < -1.0:
            raise ValueError(f"epsilon must be at least -1, got {epsilon}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must lie in [0, 1), got {label_smoothing}")
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        check_class_term(type(self).__name__, prediction, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = target.long()
        entropy = F.cross_entropy(
            prediction, target, reduction="none", label_smoothing=self.label_smoothing
        )
        hit = prediction.softmax(dim=-1).gather(1, target[:, None]).squeeze(1)
        return entropy + self.epsilon * (1.0 - hit)


@LOSSES.register()
class BinaryCrossEntropy(ElementwiseLoss):
    """Sigmoid cross-entropy between a logit and a target in ``[0, 1]``, soft or binary.

    The minimum lies at ``sigmoid(x) = y`` for every target value, so a graded quality
    is learned as a probability-like number.

    Args:
        weight: Weight of the term.
        positive_weight: Factor on the ``y x log(sigmoid(x))`` part (PyTorch
            ``pos_weight``): above 1 it trades precision for recall of high targets.
    """

    source = "PyTorch binary_cross_entropy_with_logits"
    status = "tested"
    accepts = ("unit",)
    prediction = "logit"

    def __init__(self, weight: float = 1.0, positive_weight: float = 1.0) -> None:
        super().__init__(weight)
        if positive_weight <= 0:
            raise ValueError(f"positive_weight must be positive, got {positive_weight}")
        self.positive_weight = positive_weight

    def _check(self, prediction: Tensor, target: Tensor) -> None:
        super()._check(prediction, target)
        check_unit_target(type(self).__name__, target)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            prediction,
            target.type_as(prediction),
            reduction="none",
            pos_weight=prediction.new_tensor(self.positive_weight),
        )
