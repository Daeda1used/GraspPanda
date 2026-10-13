"""Kinds of loss term, and which primitive fits which.

A stage declares the kind of every loss option it has (``loss_terms`` on the class), a
primitive the kinds it accepts (``accepts``); see :class:`grasppanda.models.base.Loss`.
The functions here answer the two questions the config checker and the documentation
ask: what does this loss node accept, and which registered primitives fit this kind.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from grasppanda.models.base import Loss
from grasppanda.registry import LOSSES, component_parameters

__all__ = ["TERM_KINDS", "accepted_terms", "applies_weights", "fitting_losses"]

#: Kind of loss term → what a stage hands to the primitive in it.
TERM_KINDS: dict[str, str] = {
    "class": "(N, C) logits against (N,) class indices",
    "unit": "a prediction and a target of equal shape, the target a quality in [0, 1]",
    "value": "a prediction and a target of equal shape, the target an unbounded quantity",
    "bins": "(N, C) logits over bins against (N,) fractional bin positions",
}


def accepted_terms(loss: Any) -> tuple[type, tuple[str, ...]] | None:
    """Class and accepted term kinds of a loss given as a config node or built.

    ``None`` for anything else (an option left at ``None``, a method of the same name).
    """
    if isinstance(loss, Mapping) and isinstance(loss.get("type"), str):
        cls = LOSSES.get(loss["type"])
        return cls, tuple(cls.accepted_terms(loss))
    if isinstance(loss, Loss):
        params, _ = component_parameters(type(loss))
        options = {name: getattr(loss, name) for name in params if hasattr(loss, name)}
        return type(loss), tuple(type(loss).accepted_terms(options))
    return None


def applies_weights(loss: Any) -> bool:
    """Whether a loss, given as a config node or built, can weight single elements; a
    wrapper can when the primitive it wraps can."""
    found = accepted_terms(loss)
    if found is None:
        return True
    inner = loss.get("loss") if isinstance(loss, Mapping) else getattr(loss, "loss", None)
    return bool(getattr(found[0], "applies_weights", True)) and applies_weights(inner)


def fitting_losses(kind: str) -> list[str]:
    """Names of the registered primitives that fit a kind of loss term."""
    return [name for name in LOSSES.names() if kind in getattr(LOSSES.get(name), "accepts", ())]
