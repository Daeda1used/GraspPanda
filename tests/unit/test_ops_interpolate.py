"""Inverse-distance interpolation, checked against a loop written out in NumPy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from grasppanda.ops import inverse_distance_weights, knn, three_interpolate


def cloud(batch: int, count: int, seed: int) -> torch.Tensor:
    return torch.rand(batch, count, 3, generator=torch.Generator().manual_seed(seed))


def interpolate_oracle(
    unknown: np.ndarray, known: np.ndarray, feats: np.ndarray, k: int, power: float, eps: float
) -> np.ndarray:
    """The feature-propagation loop of PointNet++, one query point at a time."""
    out = np.zeros((len(unknown), feats.shape[1]))
    for row, point in enumerate(unknown):
        distance = np.sqrt(((known - point) ** 2).sum(-1))
        nearest = np.argsort(distance, kind="stable")[:k]
        weight = 1.0 / (distance[nearest] ** power + eps)
        out[row] = (weight[:, None] * feats[nearest]).sum(0) / weight.sum()
    return out


@pytest.mark.parametrize(("k", "power"), [(3, 1.0), (3, 2.0), (5, 1.0)])
def test_interpolation_matches_the_definition(k, power):
    unknown, known = cloud(2, 40, 0).double(), cloud(2, 12, 1).double()
    feats = torch.rand(2, 12, 7, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    distance, index = knn(unknown, known, k)
    weight = inverse_distance_weights(distance, power=power)
    out = three_interpolate(feats, index, weight)
    assert out.shape == (2, 40, 7)
    for b in range(2):
        expected = interpolate_oracle(
            unknown[b].numpy(), known[b].numpy(), feats[b].numpy(), k, power, 1e-8
        )
        assert np.allclose(out[b].numpy(), expected, atol=1e-10)


def test_weights_sum_to_one_and_a_coincident_point_takes_its_neighbour():
    known = cloud(1, 9, 3)
    feats = torch.rand(1, 9, 4, generator=torch.Generator().manual_seed(4))
    distance, index = knn(known[:, [5]], known, 3)
    weight = inverse_distance_weights(distance)
    assert torch.allclose(weight.sum(-1), torch.ones(1, 1))
    assert index[0, 0, 0] == 5 and weight[0, 0, 0] > 1 - 1e-5
    assert torch.allclose(three_interpolate(feats, index, weight)[0, 0], feats[0, 5], atol=1e-5)


def test_interpolation_reproduces_a_constant_field_and_is_differentiable():
    unknown, known = cloud(1, 30, 5), cloud(1, 10, 6)
    feats = torch.full((1, 10, 2), 3.5, requires_grad=True)
    distance, index = knn(unknown, known, 3)
    out = three_interpolate(feats, index, inverse_distance_weights(distance, power=2.0))
    assert torch.allclose(out, torch.full_like(out, 3.5), atol=1e-5)
    out.sum().backward()
    assert feats.grad.sum() == pytest.approx(60.0, rel=1e-5)  # every weight row sums to one


def test_interpolation_rejects_bad_arguments():
    with pytest.raises(ValueError, match="power must be positive"):
        inverse_distance_weights(torch.rand(2, 3), power=0.0)
    with pytest.raises(ValueError, match="same shape"):
        three_interpolate(
            torch.rand(1, 4, 2), torch.zeros(1, 5, 3, dtype=torch.long), torch.rand(1, 5, 2)
        )
