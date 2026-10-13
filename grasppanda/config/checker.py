"""Cross-field consistency rules.

Rules never edit a config: a value is either stated once and injected where needed,
or stated twice and required to agree. Checks that concern the slots of one component
live on that component (``check_config``); rules here span top-level sections.

A project adds its own with::

    from grasppanda.config import rule

    @rule
    def my_rule(data, command):
        if ...:
            yield ("train", "epochs"), "explain the problem and the fix"
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from .loader import KeyPath

__all__ = ["COMMAND_REQUIREMENTS", "RULES", "rule", "run_rules"]

Finding = tuple[KeyPath, str]
Rule = Callable[[Mapping[str, Any], "str | None"], Iterable[Finding]]

RULES: list[Rule] = []

#: Sections each command cannot run without.
COMMAND_REQUIREMENTS: dict[str, tuple[KeyPath, ...]] = {
    "train": (("model",), ("data", "train"), ("optimizer",)),
    "test": (("model",), ("data", "test"), ("evaluation", "evaluator")),
    "infer": (("model",),),
    "demo": (("model",),),
    "analyze": (("model",),),
}


def rule(function: Rule) -> Rule:
    """Register a cross-field rule."""
    RULES.append(function)
    return function


def run_rules(data: Mapping[str, Any], command: str | None) -> Iterator[Finding]:
    for check in RULES:
        yield from check(data, command) or ()


def _select(data: Mapping[str, Any], path: KeyPath) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, Mapping) or node.get(key) is None:
            return None
        node = node[key]
    return node


@rule
def command_requirements(data: Mapping[str, Any], command: str | None) -> Iterator[Finding]:
    """A command fails early, by name, when a section it needs is absent."""
    for path in COMMAND_REQUIREMENTS.get(command or "", ()):
        if _select(data, path) is None:
            yield path, f"`grasppanda {command}` needs this section, but it is not set."


@rule
def save_best_names_a_variant(data: Mapping[str, Any], command: str | None) -> Iterator[Finding]:
    """``evaluation.save_best`` must monitor a metric of a declared variant."""
    evaluation = data.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return
    target = evaluation.get("save_best")
    if not isinstance(target, str):
        return
    variants = evaluation.get("variants")
    names = list(variants) if isinstance(variants, Mapping) else ["raw"]
    variant, separator, metric = target.partition("/")
    if not separator or not metric or variant not in names:
        yield (
            ("evaluation", "save_best"),
            f"must look like '<variant>/<metric>' with a variant from evaluation.variants "
            f"({', '.join(names)}), e.g. '{names[0]}/AP'; got '{target}'.",
        )
