"""Feature interpolation between point sets.

PointNet++ (Qi et al., NeurIPS 2017, Eq. 2) propagates features from a sparse set to a
denser one with inverse-distance weights over the ``k`` nearest known points:

    f(x) = Σᵢ wᵢ(x) fᵢ / Σᵢ wᵢ(x),    wᵢ(x) = 1 / d(x, xᵢ)ᵖ

The paper states ``p = 2, k = 3`` and the authors' TensorFlow code implements that. The
PyTorch ports that most later work builds on (Pointnet2_PyTorch, VoteNet,
GraspNet-baseline) take the square root of the squared distance first and therefore
use ``p = 1``. Checkpoints only reproduce with the exponent they were trained with, so
it is an argument here rather than a hidden choice.
"""

from __future__ import annotations

from torch import Tensor

from .query import group_points

__all__ = ["inverse_distance_weights", "three_interpolate"]


def inverse_distance_weights(distance: Tensor, *, power: float = 1.0, eps: float = 1e-8) -> Tensor:
    """Normalised weights ``w ∝ 1 / (distance ** power + eps)`` over the last axis.

    Args:
        distance: ``(..., K)`` Euclidean distances to the ``K`` neighbours, as returned
            by :func:`grasppanda.ops.knn`.
        power: Exponent ``p``. ``1.0`` is the rule of the PyTorch PointNet++ ports;
            ``2.0`` is the rule of the paper.
        eps: Added to the denominator so that a coincident neighbour (distance 0)
            receives almost all the weight instead of dividing by zero.

    Returns:
        ``(..., K)`` weights that sum to one.
    """
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    reciprocal = 1.0 / (distance.pow(power) + eps)
    return reciprocal / reciprocal.sum(dim=-1, keepdim=True)


def three_interpolate(feats: Tensor, index: Tensor, weight: Tensor) -> Tensor:
    """Weighted sum of neighbour features.

    ``out[b, n] = Σₖ weight[b, n, k] · feats[b, index[b, n, k]]``. The name is the
    PointNet++ one; any number of neighbours ``K`` is accepted.

    Args:
        feats: ``(B, M, C)`` features of the known points.
        index: ``(B, N, K)`` indices into the known points.
        weight: ``(B, N, K)`` weights, usually from :func:`inverse_distance_weights`.

    Returns:
        ``(B, N, C)`` interpolated features.
    """
    if index.shape != weight.shape:
        raise ValueError(
            f"index and weight must have the same shape, got {tuple(index.shape)} and "
            f"{tuple(weight.shape)}"
        )
    return (group_points(feats, index) * weight[..., None]).sum(dim=2)
