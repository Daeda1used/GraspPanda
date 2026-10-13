"""Squared-error primitive; see :mod:`.elementwise` for what a primitive is."""

from __future__ import annotations

from torch import Tensor

from grasppanda.registry import LOSSES

from .elementwise import ElementwiseLoss

__all__ = ["MSE"]


@LOSSES.register()
class MSE(ElementwiseLoss):
    """Squared error. Unlike ``SmoothL1`` it does not flatten out for large errors."""

    source = "PyTorch mse_loss"

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    def penalty(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).square()
