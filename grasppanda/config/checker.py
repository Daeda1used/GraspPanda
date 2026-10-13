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


def _spelled(field: str) -> str:
    """A supervision field as it is written in code: ``point_labels.x`` → ``point_labels['x']``."""
    head, _, name = field.partition(".")
    if head in ("point_labels", "extras") and name:
        return f"supervision.{head}['{name}']"
    return f"supervision.{field}"


@rule
def training_data_serves_what_the_model_reads(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """The training dataset must put into its batches every label the model reads.

    Both sides declare it: a detector what it reads (``supervision_needs``, collected from
    the ``needs_supervision`` of its stages), a dataset reader what a config makes it
    serve (``served_supervision``). A side that declares nothing is not compared. Only
    ``train`` reads the training data, so other commands skip the rule.
    """
    from grasppanda.registry import DATASETS, DETECTORS

    if command not in (None, "train"):
        return
    model, dataset = _select(data, ("model",)), _select(data, ("data", "train", "dataset"))
    if not (isinstance(model, Mapping) and isinstance(dataset, Mapping)):
        return
    if model.get("type") not in DETECTORS or dataset.get("type") not in DATASETS:
        return  # an unknown component: the signature check reports it
    detector, reader = DETECTORS.get(model["type"]), DATASETS.get(dataset["type"])
    try:
        needs = getattr(detector, "supervision_needs", lambda node: None)(model)
        served = getattr(reader, "served_supervision", lambda node: None)(dataset)
    except (LookupError, TypeError, ValueError, AttributeError):
        return  # a malformed node: the signature check reports it
    if needs is None or served is None:
        return
    readers: dict[str, list[str]] = {}
    for field, who in needs:
        if field not in served and who not in readers.setdefault(field, []):
            readers[field].append(who)
    for field, names in readers.items():
        if any(field.startswith(f"{other}.") for other in readers):
            continue  # a part of something that is itself not served: said once, there
        who = " and ".join(names)
        verb = "reads" if len(names) == 1 else "read"
        hint = reader.supervision_hint(field, dataset)
        yield (
            ("data", "train", "dataset"),
            f"{who} {verb} `{_spelled(field)}`, which this training dataset does not serve."
            + (f" {hint}" if hint else ""),
        )


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
