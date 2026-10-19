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

A rule says why it refuses with ``@rule(kind=...)``, one of
:data:`~grasppanda.config.errors.REFUSAL_KINDS`; a message that is a
:class:`~grasppanda.config.errors.Refusal` states its own kind instead. Every rule shipped
here states one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, overload

from .errors import REFUSAL_KINDS, Refusal
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


@overload
def rule(function: Rule) -> Rule: ...
@overload
def rule(*, kind: str) -> Callable[[Rule], Rule]: ...
def rule(function: Rule | None = None, *, kind: str | None = None) -> Any:
    """Register a cross-field rule, as ``@rule`` or ``@rule(kind="structural")``.

    ``kind`` is what the rule's refusals are (:data:`REFUSAL_KINDS`). It is kept as the
    attribute ``kind`` of the function and given to every message that does not state
    its own.
    """
    if kind is not None and kind not in REFUSAL_KINDS:
        raise ValueError(f"unknown refusal kind '{kind}'; kinds: {', '.join(REFUSAL_KINDS)}")

    def register(check: Rule) -> Rule:
        check.kind = kind  # type: ignore[attr-defined]
        RULES.append(check)
        return check

    return register if function is None else register(function)


def run_rules(data: Mapping[str, Any], command: str | None) -> Iterator[Finding]:
    for check in RULES:
        kind = getattr(check, "kind", None)
        for path, message in check(data, command) or ():
            if kind is not None and not isinstance(message, Refusal):
                message = Refusal(message, kind, getattr(check, "__name__", ""))
            yield path, message


def _select(data: Mapping[str, Any], path: KeyPath) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, Mapping) or node.get(key) is None:
            return None
        node = node[key]
    return node


@rule(kind="not-a-combination")
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


@rule(kind="fixed-by-compose")
def training_data_serves_what_the_model_reads(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """The training dataset must put into its batches every label the model reads.

    Both sides declare it: a detector what it reads (``supervision_needs``, collected from
    the ``needs_supervision`` of its stages), a dataset reader what a config makes it
    serve (``served_supervision``). A side that declares nothing is not compared. Only
    ``train`` reads the training data, so other commands skip the rule.

    What it refuses is a data recipe that does not serve a label set or a derived asset a
    stage reads, which `grasppanda compose` writes into the recipe (``--set labels=``, the
    dataset section of a preset): ``fixed-by-compose``. A dataset configured to serve no
    labels at all is ``not-a-combination``; a reader with no option that serves the field is
    ``structural``.
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
        message = f"{who} {verb} `{_spelled(field)}`, which this training dataset does not serve."
        kind = "fixed-by-compose"
        if dataset.get("load_labels", True) is False or (
            field.startswith("objects") and dataset.get("object_labels", True) is False
        ):
            kind = "not-a-combination"
        elif not hint or hint.endswith("has no option that serves it."):
            kind = "structural"
        yield (
            ("data", "train", "dataset"),
            Refusal(
                message + (f" {hint}" if hint else ""),
                kind,
                "training_data_serves_what_the_model_reads",
            ),
        )


@rule(kind="not-a-combination")
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


def _dataset_sections(data: Mapping[str, Any], names: Iterable[str]) -> Iterator[tuple[str, Any]]:
    for name in names:
        node = _select(data, ("data", name, "dataset"))
        if isinstance(node, Mapping):
            yield name, node


def _component_class(registry: Any, node: Any) -> Any:
    if isinstance(node, Mapping) and node.get("type") in registry:
        return registry.get(node["type"])
    return None


@rule(kind="not-implemented")
def image_data_is_not_moved_in_point_space(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """A dataset section that serves images, or labels keyed to the image, must not
    list a point transform that moves the points: the images would stay behind.

    A transform declares it with ``moves_points = True``; a section serves image-space
    data when it sets ``images`` or ``view_grasps``. A step that uses an image up
    before the mover names it in ``released_images(node)``.
    """
    from grasppanda.registry import TRANSFORMS

    for name, dataset in _dataset_sections(data, ("train", "val", "test")):
        images = dataset.get("images")
        carried = [str(item) for item in images] if isinstance(images, (list, tuple)) else []
        if dataset.get("view_grasps"):
            carried.append("view_grasps")
        steps = dataset.get("transforms")
        if not carried or not isinstance(steps, list):
            continue
        movers: list[str] = []
        left: list[str] = []
        for node in steps:
            cls = _component_class(TRANSFORMS, node)
            released = getattr(cls, "released_images", None)
            if callable(released):  # a step that uses an image up, e.g. DepthNormals
                carried = [item for item in carried if item not in released(node)]
            if getattr(cls, "moves_points", False) and carried:
                movers.append(str(node["type"]))
                left = left or list(carried)
        if movers:
            verb = "moves" if len(movers) == 1 else "move"
            yield (
                ("data", name, "dataset", "transforms"),
                f"{' and '.join(dict.fromkeys(movers))} {verb} the points, but this dataset "
                f"section also serves image-space data (`{'`, `'.join(left)}`), which would "
                "stay where it was. Remove the point-space augmentation (with `grasppanda "
                "compose`: leave it out of `--set augment=`); a detector that reads images "
                "needs an augmentation that moves image and cloud together, and none is "
                "shipped.",
            )


@rule(kind="fixed-by-compose")
def post_processors_find_the_cloud_they_read(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """A post-processor that needs the cloud before sub-sampling (``reads_full_points``,
    with ``cloud: full_points``) must be evaluated on a dataset section that keeps it.

    ``train`` evaluates on ``data.val`` (as does ``grasppanda probe``), ``test`` on
    ``data.test``; without a command both are compared.
    """
    from grasppanda.registry import POSTPROCESSORS

    variants = _select(data, ("evaluation", "variants"))
    if not isinstance(variants, Mapping):
        return
    sections = {"train": ("val",), "test": ("test",), None: ("val", "test")}.get(command, ())
    for variant, chain in variants.items():
        for node in chain if isinstance(chain, list) else ():
            cls = _component_class(POSTPROCESSORS, node)
            if not getattr(cls, "reads_full_points", False):
                continue
            if node.get("cloud", "full_points") != "full_points":
                continue
            for name, dataset in _dataset_sections(data, sections):
                if dataset.get("keep_full_points", True) is False:
                    yield (
                        ("evaluation", "variants", str(variant)),
                        f"{node['type']} reads the cloud before sub-sampling "
                        "(`observation.full_points`), but `data."
                        f"{name}.dataset.keep_full_points` is false, so the frames of "
                        f"`data.{name}` do not carry it. Set `data.{name}.dataset."
                        f"keep_full_points: true` (`--opts data.{name}.dataset."
                        "keep_full_points=true`), or let the step test the cloud the network "
                        "saw with `cloud: points`.",
                    )


@rule(kind="not-implemented")
def mixed_precision_needs_a_detector_that_is_ready(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """``train.amp: true`` is refused unless the detector declares ``amp_ready = True``.

    Autocast wraps the whole loss computation, label matching and geometry included; a
    detector that was not written for it stops at a half/float mismatch or returns a
    loss that is not finite. No shipped detector declares it yet.
    """
    from grasppanda.registry import DETECTORS

    if command not in (None, "train") or _select(data, ("train", "amp")) is not True:
        return
    cls = _component_class(DETECTORS, data.get("model"))
    if cls is None or getattr(cls, "amp_ready", False):
        return
    yield (
        ("train", "amp"),
        f"mixed precision is not supported by {cls.__name__} yet: its loss is not written "
        "for half-precision tensors (the detector does not declare `amp_ready = True`). "
        "Set `train.amp: false` (with `grasppanda compose`: leave `amp` out of "
        "`--set hooks=`).",
    )


def _stage_option(registry: Any, node: Any, name: str) -> Any:
    """An option of a component node, or its constructor default; ``None`` when the
    component is unknown or has no such option."""
    import inspect

    from grasppanda.registry import component_parameters

    cls = _component_class(registry, node)
    if cls is None:
        return None
    if name in node:
        return node[name]
    param = component_parameters(cls)[0].get(name)
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return param.default


@rule(kind="not-a-combination")
def stages_use_the_grid_of_the_labels(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """The head bundle and the view stage must be laid out on the grid of the grasp
    labels the training dataset serves: its reader's ``label_grid`` (views × in-plane
    angles × depths).

    The head states its grid in ``num_angle`` and ``num_depth``; a view stage states the
    lattice its labels are indexed by in ``label_views``, or in ``num_views`` when it
    scores the label lattice itself.
    """
    from grasppanda.registry import DATASETS, HEADS, VIEWS

    if command not in (None, "train"):
        return
    model = _select(data, ("model",))
    reader = _component_class(DATASETS, _select(data, ("data", "train", "dataset")))
    grid = getattr(reader, "label_grid", None)
    if not isinstance(model, Mapping) or grid is None:
        return
    head, view = model.get("head"), model.get("view")
    for name, served in (("num_angle", grid.num_angle), ("num_depth", grid.num_depth)):
        mine = _stage_option(HEADS, head, name)
        if isinstance(mine, int) and not isinstance(mine, bool) and mine != served:
            yield (
                ("model", "head", name),
                f"{head['type']} has {name} = {mine}, but the grasp labels of "
                f"{reader.__name__} are laid out on {grid.num_angle} in-plane angles × "
                f"{grid.num_depth} depths. The head bundle reads the labels cell by cell, so "
                f"set `{name}: {served}`.",
            )
    option = "label_views" if _stage_option(VIEWS, view, "label_views") is not None else "num_views"
    mine = _stage_option(VIEWS, view, option)
    if isinstance(mine, int) and not isinstance(mine, bool) and mine != grid.num_view:
        other = (
            ""
            if option == "label_views"
            else " To score another number of views, use a stage that keeps its own lattice "
            "apart from the labels' (`--set view=lattice-view --opts model.view.num_views=...`)."
        )
        yield (
            ("model", "view", option),
            f"{view['type']} has {option} = {mine}, but the grasp labels of {reader.__name__} "
            f"are indexed by a lattice of {grid.num_view} views. Set `{option}: "
            f"{grid.num_view}`.{other}",
        )


def _model_parts(model: Mapping[str, Any]) -> list[str]:
    """The parts a detector config names: its keys whose value is a component node."""
    return sorted(key for key, value in model.items() if isinstance(value, Mapping))


@rule(kind="structural")
def training_recipe_names_parts_the_model_has(
    data: Mapping[str, Any], command: str | None
) -> Iterator[Finding]:
    """A learning-rate pattern and the layer-decay module of ``ParamGroups``, and a module of
    ``FreezeHook``, must name a part of the model.

    Both are written against the parts of a detector (``backbone`` for the scene encoder of
    the seed-based skeleton). A pattern that names a part the model does not have matches
    no parameter, which the optimizer reports only when training starts; HGGD, for one,
    has no ``backbone``. A pattern written as ``re:<regular expression>`` is not checked, and
    neither is a detector whose config names no component nodes (it builds its parts itself).
    """
    model = _select(data, ("model",))
    if not isinstance(model, Mapping) or not isinstance(model.get("type"), str):
        return
    parts = _model_parts(model)
    if not parts:
        return  # a detector that builds its parts itself: nothing to compare with
    named = ", ".join(f"`{part}`" for part in parts)

    def missing(pattern: Any) -> bool:
        text = str(pattern)
        return not text.startswith("re:") and text.split(".")[0] not in parts

    optimizer = data.get("optimizer")
    if isinstance(optimizer, Mapping) and optimizer.get("type") == "ParamGroups":
        module = optimizer.get("layer_decay_module")
        if module is not None and optimizer.get("layer_decay") is not None and missing(module):
            yield (
                ("optimizer", "layer_decay_module"),
                f"layer decay is set on `{module}`, a part {model['type']} does not have. "
                f"Its parts: {named}.",
            )
        multipliers = optimizer.get("lr_multipliers")
        for pattern in multipliers if isinstance(multipliers, Mapping) else ():
            if missing(pattern):
                yield (
                    ("optimizer", "lr_multipliers"),
                    f"`{pattern}` names a part {model['type']} does not have, so it would "
                    f"match no parameter. Its parts: {named}.",
                )
    hooks = data.get("hooks")
    for index, hook in enumerate(hooks if isinstance(hooks, list) else ()):
        if not (isinstance(hook, Mapping) and hook.get("type") == "FreezeHook"):
            continue
        modules = hook.get("modules")
        for module in modules if isinstance(modules, list) else ():
            if missing(module):
                yield (
                    ("hooks", index, "modules"),
                    f"FreezeHook freezes `{module}`, a part {model['type']} does not have. "
                    f"Its parts: {named}.",
                )
