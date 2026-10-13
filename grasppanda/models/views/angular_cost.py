# Copyright (c) 2022 Gou Minghao
# Adapted for GraspPanda from https://github.com/GouMinghao/rgb_matters @ 2e176728f6 (MIT)
"""Angular-distance-aware loss over view classes (the ``Class_Loss`` of RGB Matters).

A plain cross-entropy treats every wrong view alike. This loss charges a wrong view by
how far it is from the right one: the cost of predicting class ``j`` when ``i`` is
right is ``(1 - cos∠(i, j)) / 2`` (0 for the same direction, 1 for the opposite one),
and the loss of a sample is the expected cost under the softmax of its logits.

Adapted from ``rgbd_graspnet/net/loss.py`` of the upstream repository: the cost matrix
is built from the directions the caller passes instead of a fixed 60-view template, it
is a buffer of a module instead of a tensor moved by hand, and the reduction takes the
mask and the per-seed weights that GraspPanda losses take.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["AngularCost"]


class AngularCost(nn.Module):
    """Expected angular cost of a predicted distribution over directions.

    Args:
        directions: ``(V, 3)`` unit vectors, one per class.
    """

    cost: Tensor

    def __init__(self, directions: Tensor) -> None:
        super().__init__()
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"directions must be (V, 3), got {tuple(directions.shape)}")
        cosine = directions @ directions.T
        self.register_buffer("cost", ((1.0 - cosine) / 2.0).clamp(0.0, 1.0), persistent=False)

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Mean expected cost.

        Args:
            logits: ``(N, V)`` class scores.
            target: ``(N,)`` index of the right class.
            mask: ``(N,)`` samples that count; all when omitted.
            weights: ``(N,)`` non-negative sample weights; a plain mean when omitted.
        """
        expected = (self.cost[target] * logits.softmax(dim=-1)).sum(dim=-1)
        if mask is not None:
            expected = expected[mask]
            weights = None if weights is None else weights[mask]
        if expected.numel() == 0:
            return logits.sum() * 0.0
        if weights is None:
            return expected.mean()
        total = weights.sum()
        return (expected * weights).sum() / total if bool(total > 0) else logits.sum() * 0.0

    @torch.no_grad()
    def mean_error(self, logits: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        """Mean angle in radians between the arg-max class and the right class."""
        cost = self.cost[target, logits.argmax(dim=-1)]
        angle = torch.acos((1.0 - 2.0 * cost).clamp(-1.0, 1.0))
        if mask is not None:
            angle = angle[mask]
        return angle.mean() if angle.numel() else logits.sum() * 0.0
