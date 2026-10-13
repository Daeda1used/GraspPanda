"""The loss primitives: what each computes, which loss term it fits, and that the config
check refuses every other pairing instead of letting it fail at the first loss.

Tiny synthetic tensors and the tiny methods of ``test_swap_matrix``; CPU, no dataset.
Losses that run an optional library are skipped where the library is absent.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems
from grasppanda.models.losses.terms import TERM_KINDS, accepted_terms, fitting_losses
from grasppanda.registry import LOSSES
from grasppanda.structures import ContractError
from grasppanda.utils.imports import MissingDependencyError

from .test_swap_matrix import METHODS, STAGES, encoder, method, refusal, registered, run

HAS = {name: importlib.util.find_spec(name) is not None for name in ("torchvision", "kornia")}


def needs(*libraries: str) -> tuple[Any, ...]:
    return tuple(
        pytest.mark.skipif(not HAS[name], reason=f"{name} is not installed") for name in libraries
    )


#: Every registered primitive in at least one configuration: label -> (node, marks).
CASES: dict[str, tuple[dict[str, Any], tuple[Any, ...]]] = {
    "SmoothL1": ({"type": "SmoothL1", "beta": 0.5}, ()),
    "L1": ({"type": "L1"}, ()),
    "MSE": ({"type": "MSE"}, ()),
    "Huber": ({"type": "Huber", "delta": 0.5}, ()),
    "BalancedL1": ({"type": "BalancedL1"}, ()),
    "GHMR": ({"type": "GHMR", "momentum": 0.5}, ()),
    "CrossEntropy": ({"type": "CrossEntropy"}, ()),
    "LabelSmoothingCrossEntropy": (
        {"type": "LabelSmoothingCrossEntropy", "label_smoothing": 0.1},
        (),
    ),
    "UncertaintyWeighted(SmoothL1)": (
        {"type": "UncertaintyWeighted", "loss": {"type": "SmoothL1"}},
        (),
    ),
    "UncertaintyWeighted(CrossEntropy)": (
        {"type": "UncertaintyWeighted", "loss": {"type": "CrossEntropy"}},
        (),
    ),
    "Poly1CrossEntropy": ({"type": "Poly1CrossEntropy", "epsilon": 1.0}, ()),
    "BinaryCrossEntropy": ({"type": "BinaryCrossEntropy", "positive_weight": 2.0}, ()),
    "QualityFocal": ({"type": "QualityFocal"}, ()),
    "Varifocal": ({"type": "Varifocal"}, ()),
    "GHMC": ({"type": "GHMC"}, ()),
    "DistributionFocal": ({"type": "DistributionFocal"}, ()),
    "TopK(SmoothL1)": ({"type": "TopK", "fraction": 0.5, "loss": {"type": "SmoothL1"}}, ()),
    "TopK(CrossEntropy)": ({"type": "TopK", "loss": {"type": "CrossEntropy"}}, ()),
    "TopK(QualityFocal)": ({"type": "TopK", "loss": {"type": "QualityFocal"}}, ()),
    "ClassBalanced(CrossEntropy)": (
        {"type": "ClassBalanced", "beta": 0.99, "loss": {"type": "CrossEntropy"}},
        (),
    ),
    "FocalLoss": ({"type": "FocalLoss"}, needs("torchvision")),
    "ClassBalanced(FocalLoss)": (
        {"type": "ClassBalanced", "loss": {"type": "FocalLoss", "alpha": -1}},
        needs("torchvision"),
    ),
    "SoftmaxFocalLoss": ({"type": "SoftmaxFocalLoss"}, needs("kornia")),
    "DiceLoss": ({"type": "DiceLoss"}, needs("kornia")),
    "LovaszSoftmaxLoss": ({"type": "LovaszSoftmaxLoss"}, needs("kornia")),
    "RobustLoss": ({"type": "RobustLoss", "kind": "cauchy"}, needs("kornia")),
}


def cases() -> list[Any]:
    return [pytest.param(node, id=label, marks=marks) for label, (node, marks) in CASES.items()]


def sample(kind: str, rows: int = 24, classes: int = 5):
    """A prediction that needs gradients and a target of one kind of loss term."""
    generator = torch.Generator().manual_seed(0)
    if kind == "class":
        logits = torch.randn(rows, classes, generator=generator, requires_grad=True)
        return logits, torch.randint(classes, (rows,), generator=generator)
    if kind == "bins":
        logits = torch.randn(rows, classes, generator=generator, requires_grad=True)
        return logits, torch.rand(rows, generator=generator) * (classes - 1)
    prediction = torch.randn(rows, 3, generator=generator, requires_grad=True)
    target = torch.rand(rows, 3, generator=generator)
    if kind == "unit":
        target[::4] = 0.0  # empty cells, as in a score grid
        return prediction, target
    return prediction, target * 2.3


def test_every_registered_loss_has_a_case_and_states_what_it_fits():
    assert {node["type"] for node, _ in CASES.values()} == set(registered("LOSSES"))
    for name in registered("LOSSES"):
        cls = LOSSES.get(name)
        assert cls.accepts and set(cls.accepts) <= set(TERM_KINDS), name
        assert cls.prediction in ("value", "logit"), name
    # A loss that reads a logit never fits an unbounded target.
    for name in registered("LOSSES"):
        cls = LOSSES.get(name)
        if cls.prediction == "logit" and "loss" not in cls.__init__.__annotations__:
            assert "value" not in cls.accepts, name
    assert set(fitting_losses("value")) >= {"SmoothL1", "L1", "MSE", "Huber", "BalancedL1", "GHMR"}
    assert "DistributionFocal" in fitting_losses("bins")


@pytest.mark.parametrize("node", cases())
def test_contract_of_a_primitive_in_every_term_it_fits(node):
    """Finite scalar, gradient, mask, empty selection and weights, per accepted kind."""
    cls, kinds = accepted_terms(node)
    assert kinds, node
    for kind in kinds:
        loss = LOSSES.build(node)
        prediction, target = sample(kind)
        value = loss(prediction, target)
        assert value.ndim == 0 and torch.isfinite(value), (kind, value)
        value.backward()
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        assert bool(prediction.grad.any()), "no gradient reaches the prediction"
        rows = target.shape
        # Masked elements get no gradient.
        mask = torch.zeros(rows, dtype=torch.bool)
        mask[: rows[0] // 2] = True
        prediction.grad = None
        loss(prediction, target, mask).backward()
        assert not bool(prediction.grad[rows[0] // 2 :].any())
        # Nothing selected: a zero that still depends on the prediction.
        empty = loss(prediction, target, torch.zeros(rows, dtype=torch.bool))
        assert float(empty) == 0.0 and empty.requires_grad
        # Weights: an element of weight zero gets no gradient.
        if cls.applies_weights:
            weights = torch.ones(rows)
            weights[0] = 0.0
            prediction.grad = None
            weighted = loss(prediction, target, None, weights)
            weighted.backward()
            assert torch.isfinite(weighted) and not bool(prediction.grad[0].any())
        else:
            with pytest.raises(ContractError, match="cannot weight single elements"):
                loss(prediction, target, None, torch.ones(rows))
    # A term of another kind is refused with a sentence, not a shape error from PyTorch.
    for kind in set(TERM_KINDS) - set(kinds) - {"bins"}:
        other_shape = (kind == "class") != ("class" in kinds or "bins" in kinds)
        if not other_shape and not (kind == "value" and cls.prediction == "logit"):
            continue  # the same tensors would be accepted: only the declared kind differs
        prediction, target = sample(kind)
        with pytest.raises(ContractError):
            LOSSES.build(node)(prediction, target)


# ------------------------------------------------------------------------- formulas


def test_quality_focal_and_varifocal_reduce_to_sigmoid_cross_entropy():
    prediction, target = sample("unit")
    plain = F.binary_cross_entropy_with_logits(prediction, target)
    assert torch.allclose(
        LOSSES.build({"type": "QualityFocal", "beta": 0.0})(prediction, target), plain
    )
    assert torch.allclose(LOSSES.build({"type": "BinaryCrossEntropy"})(prediction, target), plain)
    # Quality focal: |y - p|^beta x BCE; a perfect prediction of a soft target costs nothing
    # more than the entropy weighted by zero.
    logit = torch.logit(torch.tensor([0.3, 0.7]))
    assert float(LOSSES.build({"type": "QualityFocal"})(logit, torch.tensor([0.3, 0.7]))) < 1e-12
    # Varifocal: positives weighted by their target, empty cells by alpha x p^gamma.
    loss = LOSSES.build({"type": "Varifocal", "alpha": 0.75, "gamma": 2.0})
    x, y = torch.tensor([0.5, -1.0]), torch.tensor([0.8, 0.0])
    bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    expected = (bce[0] * 0.8 + bce[1] * 0.75 * torch.sigmoid(x[1]) ** 2) / 2
    assert torch.allclose(loss(x, y), expected)


def test_a_logit_loss_refuses_targets_outside_the_unit_interval_and_names_the_fix():
    prediction, target = sample("value")
    for name in ("QualityFocal", "Varifocal", "BinaryCrossEntropy", "GHMC"):
        with pytest.raises(ContractError, match=r"needs targets in \[0, 1\]") as error:
            LOSSES.build({"type": name})(prediction, target)
        assert "SmoothL1" in str(error.value) and "`value` term" in str(error.value)
    assert LOSSES.build({"type": "QualityFocal"}).as_value(torch.zeros(1)).item() == 0.5
    assert LOSSES.build({"type": "SmoothL1"}).as_value(torch.zeros(1)).item() == 0.0
    wrapped = LOSSES.build({"type": "TopK", "loss": {"type": "QualityFocal"}})
    assert wrapped.as_value(torch.zeros(1)).item() == 0.5


def test_distribution_focal_pulls_the_expectation_to_a_fractional_bin():
    loss = LOSSES.build({"type": "DistributionFocal"})
    logits, _ = sample("bins")
    whole = torch.randint(5, (24,))
    assert torch.allclose(loss(logits, whole.float()), F.cross_entropy(logits, whole))
    # Between bins 1 and 2 at 1.25: three quarters of the mass belong on bin 1.
    free = torch.zeros(1, 5, requires_grad=True)
    optimizer = torch.optim.SGD([free], lr=1.0)
    for _ in range(300):
        optimizer.zero_grad()
        loss(free, torch.tensor([1.25])).backward()
        optimizer.step()
    probability = free.softmax(dim=-1)[0]
    assert abs(float((probability * torch.arange(5)).sum()) - 1.25) < 0.02
    assert abs(float(probability[1]) - 0.75) < 0.02
    with pytest.raises(ContractError, match=r"bin positions must lie in \[0, 4\]"):
        loss(logits, torch.full((24,), 4.5))
    with pytest.raises(ContractError, match=r"\(N, C\) logits over bins"):
        loss(torch.zeros(4, 3), torch.zeros(4, 3))


def test_regression_penalties_against_their_definitions():
    prediction, target = sample("value")
    huber = LOSSES.build({"type": "Huber", "delta": 0.5})(prediction, target)
    smooth = LOSSES.build({"type": "SmoothL1", "beta": 0.5})(prediction, target)
    assert torch.allclose(huber, 0.5 * smooth)
    # Balanced L1 is continuous at beta and linear with slope gamma beyond it.
    balanced = LOSSES.build({"type": "BalancedL1", "beta": 1.0, "alpha": 0.5, "gamma": 1.5})
    zero = torch.zeros(1)
    below, above = (
        balanced(torch.tensor([1.0 - 1e-4]), zero),
        balanced(torch.tensor([1.0 + 1e-4]), zero),
    )
    assert abs(float(below) - float(above)) < 1e-3
    slope = balanced(torch.tensor([3.0]), zero) - balanced(torch.tensor([2.0]), zero)
    assert abs(float(slope) - 1.5) < 1e-5
    # GHM-R: with all errors in one bin the weights are 1 and the penalty is the smooth L1.
    ghm = LOSSES.build({"type": "GHMR", "mu": 0.02, "bins": 10})
    error = torch.full((8,), 5.0)
    assert torch.allclose(ghm(error, torch.zeros(8)), torch.tensor(math.sqrt(25 + 0.0004) - 0.02))
    # Two bins with 6 and 2 elements: each bin contributes half of the total.
    mixed = torch.cat([torch.full((6,), 5.0), torch.full((2,), 0.001)])
    values = (torch.sqrt(mixed**2 + 0.0004) - 0.02) * torch.tensor(
        [8 / 6 / 2] * 6 + [8 / 2 / 2] * 2
    )
    assert torch.allclose(ghm(mixed, torch.zeros(8)), values.mean())


def test_ghm_counts_supervised_elements_only_and_keeps_its_running_mean():
    ghm = LOSSES.build({"type": "GHMC", "bins": 10, "momentum": 0.9})
    assert "acc_sum" in ghm.state_dict()
    prediction, target = sample("unit")
    mask = torch.zeros(target.shape, dtype=torch.bool)
    mask[:6] = True
    ghm(prediction, target, mask)
    assert float(ghm.acc_sum.sum()) == pytest.approx(0.1 * 18)  # 6 rows x 3 cells, once
    fresh = LOSSES.build({"type": "GHMC", "bins": 10})
    # Without momentum the result of a masked call equals the call on the selection.
    assert torch.allclose(fresh(prediction, target, mask), fresh(prediction[:6], target[:6]))


def test_cross_entropy_options_and_poly1():
    logits, classes = sample("class")
    smoothed = LOSSES.build({"type": "LabelSmoothingCrossEntropy"})(logits, classes)
    assert torch.allclose(smoothed, F.cross_entropy(logits, classes, label_smoothing=0.1))
    factors = [1.0, 2.0, 0.5, 1.0, 3.0]
    weighted = LOSSES.build(
        {"type": "LabelSmoothingCrossEntropy", "label_smoothing": 0.0, "class_weights": factors}
    )(logits, classes)
    expected = (
        F.cross_entropy(logits, classes, reduction="none") * torch.tensor(factors)[classes]
    ).mean()
    assert torch.allclose(weighted, expected)
    with pytest.raises(ContractError, match="class_weights lists 2 value"):
        LOSSES.build({"type": "LabelSmoothingCrossEntropy", "class_weights": [1.0, 2.0]})(
            logits, classes
        )
    with pytest.raises(ContractError, match="integer class indices") as error:
        LOSSES.build({"type": "CrossEntropy"})(logits, classes.float())
    assert "regression term" in str(error.value)
    plain = F.cross_entropy(logits, classes)
    assert torch.allclose(
        LOSSES.build({"type": "Poly1CrossEntropy", "epsilon": 0.0})(logits, classes), plain
    )
    hit = logits.softmax(-1).gather(1, classes[:, None]).squeeze(1)
    poly = LOSSES.build({"type": "Poly1CrossEntropy", "epsilon": 2.0})(logits, classes)
    assert torch.allclose(poly, plain + 2.0 * (1 - hit).mean())


def test_top_k_keeps_the_hardest_elements():
    prediction = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 10.0], requires_grad=True)
    target = torch.zeros(6)
    mask = torch.tensor([True, True, True, True, True, False])
    mined = LOSSES.build({"type": "TopK", "fraction": 0.4, "loss": {"type": "L1", "weight": 2.0}})
    value = mined(prediction, target, mask)  # 2 of the 5 selected: errors 10 and 3
    assert float(value) == pytest.approx(2.0 * (10.0 + 3.0) / 2)
    value.backward()
    assert prediction.grad.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    everything = LOSSES.build({"type": "TopK", "fraction": 1.0, "loss": {"type": "L1"}})
    assert torch.allclose(
        everything(prediction, target), LOSSES.build({"type": "L1"})(prediction, target)
    )
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0, 1.0])
    assert float(mined(prediction, target, mask, weights)) == pytest.approx(2.0 * (10.0 + 9.0) / 4)
    with pytest.raises(ValueError, match="scores the supervised set as a whole"):
        LOSSES.build({"type": "TopK", "loss": {"type": "GHMR"}})
    assert accepted_terms({"type": "TopK", "loss": {"type": "CrossEntropy"}})[1] == ("class",)
    assert set(accepted_terms(mined)[1]) == {"unit", "value"}  # a built wrapper, too


def test_uncertainty_weighting_learns_one_log_variance_per_wrapped_term():
    wrapped = LOSSES.build({"type": "UncertaintyWeighted", "loss": {"type": "L1", "weight": 3.0}})
    prediction, target = sample("value")
    inner = LOSSES.build({"type": "L1", "weight": 3.0})(prediction, target)
    # The factor starts at 1: L / (2 exp(-ln 2)) + (-ln 2) / 2.
    assert torch.allclose(wrapped(prediction, target), inner - math.log(2.0) / 2)
    assert [name for name, _ in wrapped.named_parameters()] == ["log_variance"]
    wrapped(prediction, target).backward()
    # d/ds [L / (2 e^s) + s / 2] = 1/2 - L at the start: a large term raises its s.
    assert wrapped.log_variance.grad.item() == pytest.approx(0.5 - float(inner), rel=1e-5)
    slow = LOSSES.build(
        {"type": "UncertaintyWeighted", "initial_log_variance": 0.0, "loss": {"type": "L1"}}
    )
    plain = LOSSES.build({"type": "L1"})(prediction, target)
    assert torch.allclose(slow(prediction, target), plain / 2)
    # Nothing supervised: no data and no pull on the log-variance.
    nothing = wrapped(prediction, target, torch.zeros(target.shape, dtype=torch.bool))
    assert float(nothing) == 0.0
    assert accepted_terms({"type": "UncertaintyWeighted", "loss": {"type": "QualityFocal"}})[1] == (
        "unit",
    )
    assert wrapped.as_value(torch.zeros(1)).item() == 0.0
    # A wrapper cannot weight seeds when the loss it wraps cannot.
    from grasppanda.models.losses.terms import applies_weights

    assert applies_weights({"type": "UncertaintyWeighted", "loss": {"type": "CrossEntropy"}})
    if HAS["kornia"]:
        assert not applies_weights({"type": "UncertaintyWeighted", "loss": {"type": "DiceLoss"}})


def test_class_balanced_weights_follow_the_effective_number():
    balanced = LOSSES.build(
        {"type": "ClassBalanced", "beta": 0.9, "loss": {"type": "CrossEntropy"}}
    )
    counts = torch.tensor([10.0, 1.0, 0.0])
    factors = balanced.class_factors(counts)
    effective = (1 - 0.9 ** counts[:2]) / (1 - 0.9)
    expected = 1 / effective
    assert torch.allclose(factors[:2], expected / expected.sum() * 2) and float(factors[2]) == 0.0
    logits, classes = sample("class")
    plain = LOSSES.build({"type": "CrossEntropy"})
    unweighted = LOSSES.build(
        {"type": "ClassBalanced", "beta": 0.0, "loss": {"type": "CrossEntropy"}}
    )
    assert torch.allclose(unweighted(logits, classes), plain(logits, classes))
    per_class = balanced.class_factors(torch.bincount(classes, minlength=5).float())[classes]
    assert torch.allclose(balanced(logits, classes), plain(logits, classes, None, per_class))
    fixed = LOSSES.build(
        {"type": "ClassBalanced", "class_counts": [5, 5, 5, 5, 5], "loss": {"type": "CrossEntropy"}}
    )
    assert torch.allclose(fixed(logits, classes), plain(logits, classes))
    with pytest.raises(ContractError, match="class_counts lists 2 value"):
        LOSSES.build(
            {"type": "ClassBalanced", "class_counts": [1, 2], "loss": {"type": "CrossEntropy"}}
        )(logits, classes)
    with pytest.raises(ValueError, match="is no loss for class terms"):
        LOSSES.build({"type": "ClassBalanced", "loss": {"type": "SmoothL1"}})


# -------------------------------------------------------------------------- bridges


def test_focal_loss_is_the_torchvision_function_in_both_forms():
    ops = pytest.importorskip("torchvision.ops")
    logits, classes = sample("class")
    focal = LOSSES.build({"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0})
    one_hot = F.one_hot(classes, 5).float()
    expected = ops.sigmoid_focal_loss(logits, one_hot, 0.25, 2.0, "none").sum(dim=1).mean()
    assert torch.allclose(focal(logits, classes), expected)
    plain = LOSSES.build({"type": "FocalLoss", "alpha": -1, "gamma": 0.0})
    bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none").sum(dim=1).mean()
    assert torch.allclose(plain(logits, classes), bce)
    prediction, target = sample("unit")
    assert torch.allclose(
        focal(prediction, target), ops.sigmoid_focal_loss(prediction, target, 0.25, 2.0, "mean")
    )
    assert focal.extras == ("torchvision",) and focal.status == "bridge"


def test_kornia_losses_run_the_library_functions():
    losses = pytest.importorskip("kornia.losses")
    logits, classes = sample("class")
    plain = F.cross_entropy(logits, classes)
    assert torch.allclose(
        LOSSES.build({"type": "SoftmaxFocalLoss", "gamma": 0.0})(logits, classes), plain, atol=1e-6
    )
    hit = logits.softmax(-1).gather(1, classes[:, None]).squeeze(1)
    focal = ((1 - hit) ** 2 * F.cross_entropy(logits, classes, reduction="none")).mean()
    assert torch.allclose(
        LOSSES.build({"type": "SoftmaxFocalLoss", "gamma": 2.0})(logits, classes), focal, atol=1e-6
    )
    dice = LOSSES.build({"type": "DiceLoss"})
    image, labels = logits.t()[None, :, :, None], classes[None, :, None]
    assert torch.allclose(dice(logits, classes), losses.dice_loss(image, labels))
    # The mask selects the set whose overlap is measured.
    mask = torch.arange(24) < 10
    assert torch.allclose(dice(logits, classes, mask), dice(logits[:10], classes[:10]))
    lovasz = LOSSES.build({"type": "LovaszSoftmaxLoss"})
    assert torch.allclose(lovasz(logits, classes), losses.lovasz_softmax_loss(image, labels))
    prediction, target = sample("value")
    for kind in ("charbonnier", "welsch", "cauchy", "geman_mcclure"):
        robust = LOSSES.build({"type": "RobustLoss", "kind": kind})
        expected = getattr(losses, f"{kind}_loss")(prediction, target, reduction="mean")
        assert torch.allclose(robust(prediction, target), expected)
    with pytest.raises(ValueError, match="unknown kind"):
        LOSSES.build({"type": "RobustLoss", "kind": "tukey"})


@pytest.mark.parametrize(
    ("name", "library"),
    [("FocalLoss", "torchvision"), ("SoftmaxFocalLoss", "kornia"), ("DiceLoss", "kornia")],
)
def test_a_missing_library_is_reported_with_the_extra_to_install(name, library):
    if HAS[library]:
        pytest.skip(f"{library} is installed")
    with pytest.raises(MissingDependencyError, match=rf"\.\[{library}\]"):
        LOSSES.build({"type": name})


# ----------------------------------------------- every loss in every term of every method


def loss_terms() -> list[tuple[str, str, str, str]]:
    """(method, slot, option, kind) of every loss term of the tiny reference models."""
    found = []
    for detector, chosen in METHODS.items():
        for slot in ("proposal", "view", "head"):
            cls = STAGE_REGISTRIES[slot].get(STAGES[slot][chosen[slot]]["type"])
            found += [(detector, slot, option, kind) for option, kind in cls.loss_terms.items()]
    return found


def with_loss(detector: str, slot: str, option: str, loss: dict[str, Any]) -> dict[str, Any]:
    dense = detector in ("GSNet", "EconomicGrasp")
    node = method(detector, encoder("PointNet2Dense" if dense else "PointNet2"))
    node[slot] = {**node[slot], option: loss}
    return node


def test_every_loss_option_of_a_registered_stage_declares_its_kind():
    for slot in ("proposal", "view", "head"):
        for name in registered(STAGE_REGISTRIES[slot].name):
            cls = STAGE_REGISTRIES[slot].get(name)
            options = {
                option
                for option, annotation in cls.__init__.__annotations__.items()
                if "Cfg[Loss]" in str(annotation)
            }
            reader = getattr(cls, "loss_terms_for", None)
            declared = set(reader({}) if reader else cls.loss_terms)
            assert declared == options, f"{name}: loss options {options}, declared {declared}"
    kinds = {kind for _, _, _, kind in loss_terms()}
    assert kinds == {"class", "unit", "value"}
    assert len(loss_terms()) == 24


def test_the_view_loss_of_a_lattice_stage_follows_its_target():
    """Scores per view are a `unit` term, the best view a `class` term."""
    for name in ("LatticeViewStage", "IcosphereViewStage", "HealpixViewStage"):
        cls = STAGE_REGISTRIES["view"].get(name)
        assert cls.loss_terms_for({}) == {"loss": "unit"}
        assert cls.loss_terms_for({"target": "best_view"}) == {"loss": "class"}
    scored = {"type": "LatticeViewStage", "loss": {"type": "CrossEntropy"}}
    messages = stage_contract_problems(view=scored)
    assert any(
        "LatticeViewStage.loss is a `unit` term" in m and "QualityFocal" in m for m in messages
    )
    classified = {"type": "LatticeViewStage", "target": "best_view", "loss": {"type": "SmoothL1"}}
    messages = stage_contract_problems(view=classified)
    assert any("LatticeViewStage.loss is a `class` term" in m for m in messages)
    fine = {"type": "LatticeViewStage", "target": "best_view", "loss": {"type": "CrossEntropy"}}
    assert not [m for m in stage_contract_problems(view=fine) if "term" in m]


@pytest.mark.parametrize("node", cases())
def test_a_loss_runs_in_every_term_it_fits_and_is_refused_in_every_other(node):
    """No pairing passes the config check and fails at the first loss."""
    cls, accepted = accepted_terms(node)
    ran = 0
    for detector, slot, option, kind in loss_terms():
        model = with_loss(detector, slot, option, node)
        messages = refusal(model)
        weighted = detector == "ScaleBalancedGrasp" and (slot, option) != (
            "view",
            "objectness_loss",
        )
        if kind in accepted and (cls.applies_weights or not weighted):
            assert messages == [], (detector, slot, option, messages)
            built = run(model)
            assert type(getattr(built, slot)).loss_terms[option] == kind
            ran += 1
            continue
        (message,) = messages
        stage = STAGES[slot][METHODS[detector][slot]]["type"]
        if kind in accepted:  # the right kind, but it cannot apply the seed weights
            assert "cannot weight single elements" in message and "CrossEntropy" in message
            continue
        assert message.startswith(f"{stage}.{option} is a `{kind}` term"), message
        assert f"but {cls.__name__} fits" in message
        listed = message.split(f"Losses for a `{kind}` term: ")[1].rstrip(".").split(", ")
        assert listed == fitting_losses(kind)
        # A wrapper is listed for every kind: it fits around a primitive that does.
        assert cls.__name__ not in listed or "loss" in node
        # The Python API refuses the same pairing when the detector is built.
        assert stage_contract_problems(**{slot: model[slot]}) == [message]
    assert ran or accepted == ("bins",), "the loss fits no term of any method"


def test_a_stage_built_in_python_is_checked_like_a_config():
    from grasppanda.registry import DETECTORS

    node = with_loss("GSNet", "view", "loss", {"type": "CrossEntropy"})
    with pytest.raises(ContractError, match="TemplateViewStage.loss is a `unit` term"):
        DETECTORS.build(node)
    node = with_loss("GraspNetBaseline", "head", "score_loss", {"type": "QualityFocal"})
    with pytest.raises(ContractError, match="OperationHead.score_loss is a `value` term"):
        DETECTORS.build(node)


def test_a_logit_loss_is_decoded_and_thresholded_through_its_sigmoid():
    """SWAD's decoded score and the graspness threshold see a quality, not a logit."""
    node = with_loss("GSNet", "head", "score_loss", {"type": "QualityFocal"})
    node["proposal"] = {**node["proposal"], "graspness_loss": {"type": "BinaryCrossEntropy"}}
    model = run(node)
    from .test_swap_matrix import batch

    observation, _ = batch()
    out = model.run_stages(observation)
    grasps = model.head.decode(out.raw, out.seeds, out.approach)
    assert 0.0 <= float(grasps.score.min()) <= float(grasps.score.max()) <= 1.0
    best = model.head.codec.flatten(out.raw.score).max(dim=-1).values
    assert torch.allclose(grasps.score, best.sigmoid())
    graspness = out.seeds.scores["graspness"]
    assert 0.0 <= float(graspness.min()) <= float(graspness.max()) <= 1.0
    assert bool((graspness[out.seeds.valid] > model.proposal.graspness_threshold).all())
