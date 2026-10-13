"""Primitive losses: values against hand computation, masking, and misuse."""

from __future__ import annotations

import math

import pytest
import torch

from grasppanda.models.losses.elementwise import L1, CrossEntropy, SmoothL1
from grasppanda.registry import LOSSES
from grasppanda.structures import ContractError


def test_smooth_l1_is_quadratic_then_linear_and_weighted():
    prediction = torch.tensor([0.0, 0.5, 3.0, -2.0])
    target = torch.zeros(4)
    # |x| < 1: x² / 2; otherwise |x| - 1/2
    expected = (0.0 + 0.125 + 2.5 + 1.5) / 4
    assert SmoothL1()(prediction, target).item() == pytest.approx(expected)
    assert SmoothL1(weight=10.0)(prediction, target).item() == pytest.approx(10 * expected)
    # beta moves the switch-over point: |x| < beta → x² / (2 beta)
    assert SmoothL1(beta=4.0)(prediction, target).item() == pytest.approx(
        (0 + 0.25 / 8 + 9 / 8 + 4 / 8) / 4
    )


def test_l1_and_cross_entropy_match_their_definitions():
    prediction, target = (
        torch.tensor([[1.0, -2.0], [0.5, 0.5]]),
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert L1(weight=2.0)(prediction, target).item() == pytest.approx(2 * (1 + 2 + 0.5 + 0.5) / 4)
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0], [-1.0, 3.0]])
    labels = torch.tensor([0, 1, 1])
    expected = (math.log(1 + math.exp(-2)) + math.log(2) + math.log(1 + math.exp(-4))) / 3
    assert CrossEntropy()(logits, labels).item() == pytest.approx(expected, rel=1e-6)
    assert CrossEntropy(weight=0.5)(logits, labels.bool()).item() == pytest.approx(
        expected / 2, rel=1e-6
    )


def test_mask_selects_the_supervised_elements():
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.zeros(2, 2)
    mask = torch.tensor([[True, False], [False, True]])
    loss = L1()(prediction, target, mask)
    assert loss.item() == pytest.approx((1 + 4) / 2)
    loss.backward()
    assert prediction.grad.tolist() == [[0.5, 0.0], [0.0, 0.5]]
    rows = torch.tensor([True, False, True])
    logits = torch.tensor([[2.0, 0.0], [0.0, 9.0], [0.0, 0.0]])
    kept = CrossEntropy()(logits, torch.tensor([0, 0, 1]), rows).item()
    assert kept == pytest.approx((math.log(1 + math.exp(-2)) + math.log(2)) / 2, rel=1e-6)


@pytest.mark.parametrize("loss", [SmoothL1(), L1(), CrossEntropy()])
def test_an_empty_selection_is_zero_and_keeps_the_graph(loss):
    if isinstance(loss, CrossEntropy):
        prediction, target = torch.randn(3, 2, requires_grad=True), torch.zeros(3, dtype=torch.long)
        mask = torch.zeros(3, dtype=torch.bool)
    else:
        prediction, target = torch.randn(3, 2, requires_grad=True), torch.zeros(3, 2)
        mask = torch.zeros(3, 2, dtype=torch.bool)
    value = loss(prediction, target, mask)
    assert value.item() == 0.0 and torch.isfinite(value)
    value.backward()  # a NaN-free zero that still belongs to the graph
    assert prediction.grad is not None and prediction.grad.abs().sum() == 0


def test_shapes_are_never_broadcast():
    with pytest.raises(ContractError, match="losses never broadcast"):
        SmoothL1()(torch.zeros(4, 3), torch.zeros(4, 1))
    with pytest.raises(ContractError, match="mask .* does not match"):
        L1()(torch.zeros(4, 3), torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))
    with pytest.raises(ContractError, match=r"\(N, C\) logits and \(N,\) class indices"):
        CrossEntropy()(torch.zeros(4, 2), torch.zeros(4, 2))


def test_losses_are_registered_and_built_from_config():
    assert {"SmoothL1", "L1", "CrossEntropy"} <= set(LOSSES.names())
    loss = LOSSES.build({"type": "SmoothL1", "weight": 15.0})
    assert isinstance(loss, SmoothL1) and loss.weight == 15.0 and loss.beta == 1.0
    assert "weight=15.0" in repr(loss)
