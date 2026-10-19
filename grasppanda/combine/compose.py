"""Compose a config: a base config, one component per ``slot=choice``, checked.

    composition = compose(
        "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e",
        ["backbone=sparse-unet14"],
    )
    composition.accepted          # the config check passes
    composition.text(path)        # the child config, with a header that says how it was made

What happens, in this order, and what is reported for each step:

1. every ``slot=choice`` is resolved to a preset (:mod:`grasppanda.combine.presets`) or,
   failing that, to a registered component with its constructor defaults;
2. a new encoder is fitted to what the base method feeds and expects: its
   ``in_channels`` when the method feeds real point features, and its ``out_channels``,
   where it has that option, to the seed width the grouper of the composed model fixes
   or, when it fixes none, to the width of the encoder it replaces;
3. when the encoder rule of the stage contracts refuses the pair of encoder and proposal
   and exactly one registered adapter bridges it, the adapter is inserted, with the seed
   count of the proposal and the sampling rules of the replaced encoder;
4. the label set the head bundle reads (or ``--set labels=``) is written into the
   assigner and the data recipe, after the most alike shipped config that serves it;
5. what a stage declares for the other stages is set: a proposal leaves a seed score to
   the view stage that owns it, a head decodes without a score no stage predicts, the
   assigner returns the fields a stage reads, and an axis adapter joins a grouper and a
   head whose feature axes differ (``--no-adapter`` skips 3 to 5, but for ``labels=``);
6. a dataset section a preset's stage reads is added when the base config has none;
7. the child config is written to a scratch file and loaded, so the contract verdicts,
   the warnings and the config check are those of the file the user gets.

A loss is set inside a stage: ``head.score_loss=quality-focal`` puts a preset of
``grasppanda/presets/loss/`` (or a registered loss, by its type) into that loss option of
the head bundle and keeps the weight of the loss it replaces.
A config chain is two levels deep, so the base must be a config that inherits fragments
only (a method config). A base that was itself composed is composed again from *its*
base, with the two requests merged.

No rule about what fits what lives here: every verdict is
:func:`~grasppanda.models.detectors.seed_based.stage_contract_problems` or the config
checker, quoted, and what compose adapts it reads from the contracts the components
declare (the width an encoder returns, the width a grouper needs, the labels a stage
reads and a dataset serves). One fact the config cannot show is added: a recipe preset
written for the scene encoder of the seed-based skeleton is refused under a detector
without one (HGGD), also where its pattern is a regular expression the check cannot read.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import inspect
import os
import re
import shlex
import tempfile
import textwrap
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from grasppanda.config import (
    BASE_KEY,
    MAX_BASE_HOPS,
    Config,
    ConfigError,
    Issue,
    available_configs,
    configs_root,
    parse_yaml,
    resolve_config_path,
)
from grasppanda.config.loader import LoadedTree, apply_overrides, resolve_base
from grasppanda.models.base import configured_width
from grasppanda.registry import (
    BACKBONES,
    DATASETS,
    DETECTORS,
    GROUPERS,
    LOSSES,
    Registry,
    RegistryError,
    component_parameters,
    slot_registry,
)

from . import recipes as recipe_presets
from . import yamltext
from .post import POST_SLOT, POST_VARIANT, PostSpecError, chain_nodes
from .presets import LOSS_SLOT, Preset, PresetError, find, presets

__all__ = [
    "CONTRACT_GROUPS",
    "Choice",
    "ComposeError",
    "Composition",
    "ContractVerdict",
    "command_parser",
    "component_slots",
    "compose",
    "is_adapter",
    "recompose",
]

#: Groups of slots whose contracts the rules compare, in the order data flows. A stage
#: set is reported group by group; a problem no group shows is listed for the whole set.
CONTRACT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("backbone", "proposal"),
    ("backbone", "grouper"),
    ("proposal", "view"),
    ("view", "assigner"),
    ("assigner", "head"),
    ("grouper", "head"),
    ("view", "head"),
    ("proposal", "view", "head"),
)
_WIDTH = 90
_COMMAND = "grasppanda compose"


class ComposeError(ValueError):
    """The request cannot be composed at all: an unknown slot, preset or component."""


@dataclass
class Choice:
    """One ``slot=choice`` of a request, resolved.

    Attributes:
        slot: The slot it fills.
        request: What was asked for (a preset name or a component type).
        node: The component node that goes into the config. Compose edits it in place.
        preset: The preset it came from; ``None`` for a bare component type.
        comments: Inline comment per key path inside ``node``.
    """

    slot: str
    request: str
    node: dict[str, Any]
    preset: Preset | None = None
    comments: dict[tuple[str, ...], str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractVerdict:
    """What the contract rules say about one group of slots; no problem means they fit."""

    slots: tuple[str, ...]
    problems: tuple[str, ...]


@dataclass
class Composition:
    """The outcome of :func:`compose`.

    Attributes:
        base: The base config as it is named in the command: a shipped config name where
            there is one, else the path.
        base_path: Its file.
        detector: Detector type of the base config.
        seed_based: Whether the detector is built on the seed-based skeleton, the family
            the stage contracts and the adapters belong to.
        choices: The resolved ``slot=choice`` pairs.
        overrides: Extra ``key=value`` overrides of the request.
        adapters: Whether an adapter may be inserted.
        adapter: The adapter that was inserted, if any.
        actions: What compose did, one sentence each.
        hints: For a refused request: what to type instead, where compose can tell.
        merged: The model and the training dataset of the composed config as the checks
            saw them (``model``, ``dataset``); what the hints are computed from.
        body: The child config without its ``_base_``.
        comments: Inline comment per key path of ``body``.
        stages: Slot → final component, for seed-based detectors (``PointNet2 in
            FeaturePropagation``).
        origins: Slot → where that component comes from.
        verdicts: The contract rules, group by group.
        problems: Contract problems of the whole stage set.
        issues: What the config check reports beyond those problems.
        config_hash: Hash of the composed config; ``None`` when the check refuses it.
        post: The post-processing chain of ``--set post=...`` as it was typed; it
            becomes the evaluation variant ``post``.
        recipes: Recipe presets of the request (optimizer, schedule, hooks, augment,
            recipe), in the order given: ``(slot, names as requested, presets)``.
        labels: The label set of ``--set labels=...``; ``None`` when the head decides.
        unsets: The paths of ``--unset``.
        warnings: What the stage set trains differently from what its names promise,
            without being refused (``stage_contract_warnings``).
    """

    base: str
    base_path: Path
    detector: str
    seed_based: bool
    choices: list[Choice]
    overrides: tuple[str, ...]
    adapters: bool = True
    adapter: str | None = None
    actions: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    merged: dict[str, Any] = field(default_factory=dict, repr=False)
    body: dict[str, Any] = field(default_factory=dict)
    comments: dict[tuple[str, ...], str] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    verdicts: list[ContractVerdict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    config_hash: str | None = None
    post: tuple[str, ...] = ()
    recipes: list[tuple[str, str, list[Any]]] = field(default_factory=list)
    labels: str | None = None
    unsets: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Whether the composed config passes the contract rules and the config check."""
        return not self.problems and not self.issues

    @property
    def command(self) -> str:
        """The command that composes this config again."""
        parts = [_COMMAND, self.base]
        parts += [f"--set {choice.slot}={choice.request}" for choice in self.choices]
        parts += ["--set " + shlex.quote(f"{POST_SLOT}={text}") for text in self.post]
        parts += [f"--set {slot}={request}" for slot, request, _ in self.recipes]
        if self.labels:
            parts.append(f"--set {LABELS_SLOT}={self.labels}")
        parts += [f"--unset {path}" for path in self.unsets]
        if self.overrides:
            parts.append("--opts " + " ".join(shlex.quote(item) for item in self.overrides))
        if not self.adapters:
            parts.append("--no-adapter")
        return " ".join(parts)

    def refusal(self) -> list[str]:
        """Every reason the composed config is refused, contracts first."""
        return [*self.problems, *(f"{issue.path}: {issue.message}" for issue in self.issues)]

    def refusal_kinds(self) -> list[str]:
        """The kind of every reason of :meth:`refusal`, in its order
        (:data:`~grasppanda.config.errors.REFUSAL_KINDS`; ``unclassified`` for a rule of a
        project that does not say)."""
        kinds = [getattr(problem, "kind", None) for problem in self.problems]
        kinds += [issue.kind for issue in self.issues]
        return [kind or "unclassified" for kind in kinds]

    def text(self, output: str | Path | None = None) -> str:
        """The child config as a file: header, ``_base_`` and the composed fragment.

        ``output`` is where the file will live. Inside the shipped config tree the base is
        referred to by a relative path, as shipped configs do; elsewhere by its absolute
        path, so the file can be moved on this machine but names a path of this
        installation. The command in its header composes it again anywhere.
        """
        target = None if output is None else Path(output).resolve()
        root = configs_root().resolve()
        base = self.base_path.resolve()
        if target is not None and root in target.parents and root in base.parents:
            reference = Path(os.path.relpath(base, target.parent)).as_posix()
            name = target.relative_to(root).with_suffix("").as_posix()
        else:
            reference, name = base.as_posix(), "<this file>" if target is None else str(output)
        header = yamltext.comment_block(self._header(name))
        body = yamltext.dump(self.body, self.comments) if self.body else ""
        return f"{header}_base_:\n  - {reference}\n" + (f"\n{body}" if body else "")

    def _header(self, name: str) -> list[str]:
        def wrap(text: str, first: str = "", rest: str = "") -> list[str]:
            return textwrap.wrap(
                text,
                _WIDTH,
                initial_indent=first,
                subsequent_indent=rest,
                break_long_words=False,
                break_on_hyphens=False,
            )

        changed = ", ".join(
            [f"{choice.slot} `{choice.request}`" for choice in self.choices]
            + [f"{slot} `{request}`" for slot, request, _ in self.recipes]
            + ([f"{LABELS_SLOT} `{self.labels}`"] if self.labels else [])
        )
        lines = [
            "EXPERIMENTAL. No AP is claimed. Composed by `grasppanda compose`: this config passes",
            "the config check; it has not been trained to convergence or evaluated.",
            "",
            f"  base  {self.base} ({_method_label(self.base_path, self.detector)})",
            f"  set   {changed or 'nothing; only overrides'}",
        ]
        lines += ["", *_command_lines(self.command)]
        lines += [f"  grasppanda probe {name}", f"  grasppanda train {name}", ""]
        lines += wrap(
            "Everything not set below is the base config's: the other stages, the data and "
            "the schedule. `probe` trains a few iterations on real frames and predicts, in "
            "about a minute. What a swap changes besides these lines is in "
            "docs/advanced_guides/swapping.md."
        )
        if self.actions:
            lines += ["", "What compose did, and why:"]
            for action in self.actions:
                lines += wrap(action, "* ", "  ")
        if self.warnings:
            lines += ["", "Warnings (accepted, but read them):"]
            for warning in self.warnings:
                lines += wrap(warning, "* ", "  ")
        shown = [choice.preset for choice in self.choices if choice.preset is not None]
        shown += [preset for preset in self._adapter_presets() if preset not in shown]
        lines += ["", "What must be re-tuned, and why"]
        lines[-1] += " (from grasppanda/presets/<slot>/<preset>.yaml):" if shown else ":"
        if self.recipes:
            lines.append("(recipe presets: grasppanda/recipes/<slot>/<preset>.yaml)")
            shown = shown + [preset for _, _, members in self.recipes for preset in members]
        for preset in shown:
            lines.append(f"{preset.slot} `{preset.name}`")
            for bullet in preset.retune:
                first, *rest = bullet.split("\n")
                lines += [f"* {first}", *(f"  {line}" for line in rest)]
        for choice in self.choices:
            if choice.preset is None:
                kind = choice.node["type"]
                lines.append(f"{choice.slot} `{choice.request}`")
                lines += wrap(
                    f"{kind} has no preset here, so its options are the constructor "
                    f"defaults; `grasppanda components {kind}` lists them and says what the "
                    "component was written for.",
                    "* ",
                    "  ",
                )
        lines.append("the base config")
        lines += wrap(
            "the recipe: optimizer, schedule, batch size and point budget are the base "
            "config's, tuned by its authors for its own stages. A starting point, not a claim.",
            "* ",
            "  ",
        )
        return lines

    def _adapter_presets(self) -> list[Preset]:
        if self.adapter is None:
            return []
        return [p for p in presets().get("backbone", {}).values() if p.type == self.adapter][:1]


@lru_cache(maxsize=32)
def _metafile_method(folder: Path) -> str:
    """The method a config folder holds, as its ``metafile.yaml`` names it; ``""`` if none."""
    path = folder / "metafile.yaml"
    if not path.is_file():
        return ""
    data = parse_yaml(path.read_text(encoding="utf-8"))
    return str(data.get("method") or "") if isinstance(data, Mapping) else ""


def _method_label(base_path: Path, detector: str) -> str:
    """The method of a base config by its name, with the detector class where that name
    is another (``FineGrasp; detector EconomicGrasp``); the class alone outside a method
    folder."""
    method = _metafile_method(base_path.resolve().parent)
    if not method:
        return detector
    same = re.sub(r"[^a-z0-9]", "", method.lower()) == re.sub(r"[^a-z0-9]", "", detector.lower())
    return method if same else f"{method}; detector {detector}"


def _spell(
    base: str,
    sets: Sequence[str],
    overrides: Sequence[str],
    adapters: bool,
    unsets: Sequence[str] = (),
) -> str:
    """A compose request as the command a user types."""
    parts = [_COMMAND, base, *(f"--set {item}" for item in sets)]
    parts += [f"--unset {path}" for path in unsets]
    if overrides:
        parts.append("--opts " + " ".join(shlex.quote(item) for item in overrides))
    if not adapters:
        parts.append("--no-adapter")
    return " ".join(parts)


def _command_lines(command: str) -> list[str]:
    """A command as indented lines of a header, continued with a backslash."""
    lines: list[str] = []
    current = " "
    for word in command.split(" "):
        if current.strip() and len(current) + len(word) + 3 > _WIDTH:
            lines.append(f"{current} \\")
            current = "      " + word
        else:
            current = f"{current} {word}"
    return [*lines, current]


# --------------------------------------------------------------------- slots and choices


def component_slots(detector: type) -> dict[str, Registry]:
    """Slot → registry for the component slots of a registered class.

    A slot is a constructor parameter annotated ``Cfg[Base]`` or, for a slot with a
    built-in default, ``Cfg[Base] | None``. A detector whose parts are plain options has
    none, and nothing can be composed into it.
    """
    try:
        hints = typing.get_type_hints(detector.__init__)
    except Exception:
        return {}
    params, _ = component_parameters(detector)
    slots = {}
    for name in params:
        hint = hints.get(name)
        if typing.get_origin(hint) in (typing.Union, types.UnionType):
            members = [arg for arg in typing.get_args(hint) if arg is not type(None)]
            hint = members[0] if len(members) == 1 else hint
        try:
            registry = slot_registry(hint)
        except RegistryError:
            registry = None
        if registry is not None:
            slots[name] = registry
    return slots


def is_adapter(backbone: type) -> bool:
    """Whether a backbone class wraps another backbone (it has a backbone slot itself)."""
    return BACKBONES in component_slots(backbone).values()


def _inner_slot(backbone: type) -> str:
    return next(name for name, reg in component_slots(backbone).items() if reg is BACKBONES)


def _choice(slot: str, registry: Registry, request: str) -> Choice:
    members = presets().get(slot, {})
    if request in members:
        preset = members[request]
        comments = {
            path[2:]: text for path, text in preset.comments.items() if path[:2] == ("model", slot)
        }
        return Choice(slot, request, copy.deepcopy(preset.node), preset, comments)
    if request in registry:
        return Choice(slot, request, {"type": request})
    try:
        find(slot, request)
    except PresetError as error:
        close = difflib.get_close_matches(request, registry.names(), n=3, cutoff=0.75)
        hint = (
            f" Did you mean the component {' or '.join(repr(c) for c in close)}?" if close else ""
        )
        raise ComposeError(
            f"{error}{hint} Components of {registry.name}: {', '.join(registry.names())}."
        ) from None
    raise AssertionError("unreachable")  # pragma: no cover


def _parse_sets(sets: Sequence[str], slots: Mapping[str, Registry], detector: str) -> list[Choice]:
    choices: list[Choice] = []
    for item in sets:
        slot, separator, request = item.partition("=")
        slot, request = slot.strip(), request.strip()
        if not separator or not slot or not request:
            raise ComposeError(f"--set {item!r} must look like `slot=preset` or `slot=Component`")
        if not slots:
            raise ComposeError(
                f"{detector} has no component slots, so `--set {item}` has nothing to replace. "
                f"{detector} is a detector family of its own: its parts are options of the "
                f"detector (`grasppanda components {detector}`), not registered components, so "
                "there is no backbone, proposal, view stage, grouper or head bundle to swap "
                "yet. Change its options with `--opts model.<part>.<option>=<value>`, or "
                "compose from a config of a seed-based method."
            )
        if slot not in slots:
            close = difflib.get_close_matches(slot, list(slots), n=1, cutoff=0.5)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            painting = (
                " A seed-based method reads image features through point painting (the "
                "`ImagePainted` wrapper around its scene encoder, written into a config by "
                "hand): docs/slots/image_encoders.md#point-painting."
                if slot == "image_encoder" and "backbone" in slots
                else ""
            )
            family = (
                painting
                if "backbone" in slots
                else f" {detector} is a detector family of its own: the stages of the "
                "seed-based methods (backbone, proposal, view, grouper, head) do not exist in "
                "it. Its other parts are options of the detector (`grasppanda components "
                f"{detector}`); change them with `--opts model.<part>.<option>=<value>`."
            )
            raise ComposeError(
                f"{detector} has no slot '{slot}'.{hint} Its slots: {', '.join(slots)}.{family}"
            )
        if any(choice.slot == slot for choice in choices):
            raise ComposeError(f"slot '{slot}' is set twice")
        choices.append(_choice(slot, slots[slot], request))
    return choices


def _loss_options(stage: type) -> list[str]:
    """Names of the loss options of a stage class."""
    return [name for name, registry in component_slots(stage).items() if registry is LOSSES]


def _parse_terms(
    sets: Sequence[str], slots: Mapping[str, Registry], stages: Mapping[str, Any], detector: str
) -> list[Choice]:
    """Resolve every ``slot.option=choice``: a loss for one loss option of a stage.

    ``stages`` holds the node of every slot as it will be in the composed model. The
    returned choices carry ``slot.option`` as their slot.
    """
    found: list[Choice] = []
    for item in sets:
        target, separator, request = item.partition("=")
        target, request = target.strip(), request.strip()
        slot, _, option = target.partition(".")
        if not separator or not slot or not option or not request:
            raise ComposeError(
                f"--set {item!r} must look like `slot.loss_option=preset` or "
                "`slot.loss_option=Loss`, e.g. `head.score_loss=quality-focal`"
            )
        node = stages.get(slot)
        if slot not in slots or not isinstance(node, Mapping) or "type" not in node:
            listed = ", ".join(slots) or "none"
            raise ComposeError(
                f"`{item}` sets a loss of the stage '{slot}', but {detector} has no such "
                f"slot. Its slots: {listed}."
            )
        stage = slots[slot].get(node["type"])
        options = _loss_options(stage)
        if option not in options:
            close = difflib.get_close_matches(option, options, n=1, cutoff=0.4)
            hint = f" Did you mean '{slot}.{close[0]}'?" if close else ""
            raise ComposeError(
                f"{stage.__name__} has no loss option '{option}'.{hint} Its loss options: "
                f"{', '.join(options) or 'none'}."
            )
        if any(choice.slot == target for choice in found):
            raise ComposeError(f"'{target}' is set twice")
        choice = _choice(LOSS_SLOT, LOSSES, request)
        choice.slot = target
        found.append(choice)
    return found


def _weighted_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """The node of a loss that carries the weight of the term: the loss itself, or the
    innermost loss of a wrapper (``TopK``, ``ClassBalanced``)."""
    while isinstance(node.get("loss"), dict) and "type" in node["loss"]:
        node = node["loss"]
    try:
        params, _ = component_parameters(LOSSES.get(node["type"]))
    except RegistryError:
        return None
    return node if "weight" in params else None


def _set_terms(
    result: Composition,
    terms: Sequence[Choice],
    addressed: Mapping[str, list[str]],
    stages: Mapping[str, Any],
    nodes: dict[str, dict[str, Any]],
    notes: dict[str, dict[tuple[str, ...], str]],
) -> dict[str, dict[str, Any]]:
    """Put every loss of ``terms`` into its stage; returns the options of the stages that
    are not replaced as a whole, to be written as overrides of the base config's node."""
    partial: dict[str, dict[str, Any]] = {}
    for term in terms:
        slot, option = term.slot.split(".", 1)
        mine = addressed.get(term.slot, [])
        if mine:
            apply_overrides(LoadedTree(data=term.node), mine)
            for item in mine:
                term.comments.pop(tuple(item.partition("=")[0].strip().split(".")), None)
        replaced = stages[slot].get(option)
        was = replaced.get("type") if isinstance(replaced, Mapping) else None
        said = f"preset `{term.request}` ({term.node['type']})" if term.preset else term.request
        action = f"{term.slot}: {said} in place of {was or 'the loss the stage builds itself'}."
        if term.preset is not None:
            action += " " + term.preset.summary
        try:
            takes_inner = "loss" in component_parameters(LOSSES.get(term.node["type"]))[0]
        except RegistryError:
            takes_inner = False
        if takes_inner and "loss" not in term.node and was:
            # A wrapper without a loss of its own wraps the loss that was there.
            term.node["loss"] = copy.deepcopy(dict(replaced))
            action += f" It wraps the {was} that was there, with its weight."
        carrier = _weighted_node(term.node)
        kept = _weighted_node(copy.deepcopy(dict(replaced))) if was else None
        if carrier is not None and "weight" not in carrier and kept and "weight" in kept:
            carrier["weight"] = kept["weight"]
            path = ("loss",) * _depth(term.node, carrier)
            term.comments[(*path, "weight")] = (
                f"kept from the {kept['type']} it replaces; the penalties differ in scale: re-tune"
            )
            action += f" The weight {kept['weight']} of the replaced loss is kept."
        if mine:
            action += f" Overrides as given: {' '.join(mine)}."
        result.actions.append(action)
        node = dict(term.node)
        comments = {(option, *path): text for path, text in term.comments.items()}
        if slot in nodes:  # the stage is replaced as a whole: the loss goes into its node
            nodes[slot][option] = node
            notes[slot].update(comments)
            continue
        if was == node["type"]:
            node = {"_delete_": True, **node}
            comments[(option, "_delete_")] = (
                f"replace the base config's {was}: none of its options is inherited"
            )
        partial.setdefault(slot, {})[option] = node
        notes.setdefault(slot, {}).update(comments)
    return partial


def _depth(node: Mapping[str, Any], inner: Mapping[str, Any]) -> int:
    depth = 0
    while node is not inner:
        node, depth = node["loss"], depth + 1
    return depth


# ------------------------------------------------------------------- fitting an encoder


def _encoder(node: Mapping[str, Any]) -> Mapping[str, Any]:
    """The innermost encoder of a backbone node: what an adapter wraps, or the node."""
    while True:
        cls = BACKBONES.get(node["type"])
        inner = node.get(_inner_slot(cls)) if is_adapter(cls) else None
        if not isinstance(inner, Mapping) or "type" not in inner:
            return node
        node = inner


def _seed_width(grouper: Mapping[str, Any]) -> int | None:
    """The seed width a grouper node fixes (``Grouper.required_seed_channels``); ``None`` for any."""
    try:
        cls = GROUPERS.get(grouper["type"])
        params, _ = component_parameters(cls)
        defaults = {name: param.default for name, param in params.items()}
        return cls.required_seed_channels({**defaults, **grouper})
    except (LookupError, TypeError, AttributeError):
        return None


def _describe(node: Mapping[str, Any]) -> str:
    inner = _encoder(node)
    return str(node["type"]) if inner is node else f"{inner['type']} in {node['type']}"


def _option(registry: Registry, node: Mapping[str, Any], name: str) -> Any:
    """An option of a component node, falling back to the constructor default."""
    if name in node:
        return node[name]
    params, _ = component_parameters(registry.get(node["type"]))
    default = params[name].default if name in params else None
    return None if default is inspect.Parameter.empty else default


def _fit_encoder(
    result: Composition,
    choice: Choice,
    model: Mapping[str, Any],
    keep: set[str],
    grouper: Mapping[str, Any],
) -> None:
    """Fit a new encoder to what the base method feeds and to the width the model needs.

    ``grouper`` is the grouper of the composed model. Options named in ``keep`` were set
    by the user and are left alone. Both widths are read from what the components
    declare (``Backbone.output_channels``, ``Grouper.required_seed_channels``); nothing is built.
    """
    node = choice.node
    params, _ = component_parameters(BACKBONES.get(node["type"]))
    replaced = model["backbone"]
    fed = _encoder(replaced).get("in_channels")
    if (
        model.get("point_features") == "observation"
        and "in_channels" in params
        and "in_channels" not in keep
        and isinstance(fed, int)
        and node.get("in_channels") != fed
    ):
        node["in_channels"] = fed
        choice.comments[("in_channels",)] = "the point features the base config feeds"
        result.actions.append(
            f"in_channels: {fed}. The base config feeds real point features "
            f"(`point_features: observation`, {fed} per point), so the encoder reads those "
            "instead of the constant ones of its preset."
        )
    if "out_channels" not in params or "out_channels" in keep:
        return
    needed, kept = _seed_width(grouper), configured_width(replaced)
    if needed is not None:
        width = needed
        comment = f"the seed width {grouper['type']} needs as configured"
        why = (
            f"out_channels: {width}, the width of the seed features {grouper['type']} needs "
            "as it is configured here (it adds them to its crop features)."
        )
    elif kept is not None:
        width = kept
        comment = f"the width of the encoder it replaces ({_describe(replaced)})"
        why = (
            f"out_channels: {width}, the output width of the encoder it replaces "
            f"({_describe(replaced)}), so that the other stages keep the width they have in "
            "the base model and the encoder is the only stage that differs."
        )
    else:
        result.actions.append(
            f"out_channels of {node['type']} was left as it is: {_describe(replaced)}, the "
            "encoder it replaces, does not declare the width it returns."
        )
        return
    if node.get("out_channels") != width:
        node["out_channels"] = width
        choice.comments[("out_channels",)] = comment
        result.actions.append(why)


_PROJECTION_NOTE = "the same crop with a linear layer in front of its seed residual"


def _project_seed_features(
    result: Composition,
    nodes: dict[str, dict[str, Any]],
    notes: dict[str, dict[tuple[str, ...], str]],
    written: Mapping[str, Any],
    model: Mapping[str, Any],
) -> bool:
    """Replace the grouper by its projected variant when the seed-width rule asks for it.

    A grouper that adds the seed features to its crops names the variant that takes an
    encoder of any width (``projected_variant``). The variant gets the options of the
    grouper it replaces, as far as it has them. Returns whether it was put in.
    """
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems

    registry = STAGE_REGISTRIES["grouper"]
    backbone = nodes.get("backbone", written.get("backbone"))
    grouper = nodes.get("grouper", written.get("grouper"))
    if not all(isinstance(node, Mapping) and "type" in node for node in (backbone, grouper)):
        return False
    try:
        refused = stage_contract_problems(backbone=backbone, grouper=grouper)
        variant = getattr(registry.get(grouper["type"]), "projected_variant", None)
        if not refused or not variant:
            return False
        params, _ = component_parameters(registry.get(variant))
        fixed = {"type": variant, **{k: v for k, v in grouper.items() if k in params}}
        if stage_contract_problems(backbone=backbone, grouper=fixed):
            return False
    except RegistryError:
        return False
    nodes["grouper"] = copy.deepcopy(fixed)
    notes["grouper"] = {("type",): _PROJECTION_NOTE}
    result.actions.append(
        f"grouper: {variant} in place of {grouper['type']}. {refused[0].split('. ')[0]}. "
        f"{variant} is the same crop with a linear layer in front of its seed residual; it "
        "changes the model (the base method has none). `--no-adapter` leaves the pair refused."
    )
    result.origins["grouper"] = f"{result.origins.get('grouper', 'base config')}, projected"
    return True


def _adapters_for(backbone: Mapping[str, Any], proposal: Mapping[str, Any]) -> list[str]:
    """Registered adapters that make the contract rules accept this encoder and proposal."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems

    fitting = []
    for name in BACKBONES.names():
        cls = BACKBONES.get(name)
        if not is_adapter(cls):
            continue
        wrapped = {"type": name, _inner_slot(cls): backbone}
        # An adapter that sub-samples is written with the seed count of the proposal
        # (`_wrap`); it is judged as it will be written.
        seeds = _option(STAGE_REGISTRIES["proposal"], proposal, "num_seeds")
        if isinstance(seeds, int) and "npoint" in component_parameters(cls)[0]:
            wrapped["npoint"] = seeds
        if stage_contract_problems(backbone=wrapped, proposal=proposal):
            continue
        if list(getattr(cls, "check_config", lambda node: ())(wrapped) or ()):
            continue
        fitting.append(name)
    return fitting


def _wrap(
    result: Composition,
    adapter: str,
    backbone: dict[str, Any],
    comments: Mapping[tuple[str, ...], str],
    *,
    reason: str,
    seeds: Any,
    replaced: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[tuple[str, ...], str]]:
    """Put ``backbone`` inside ``adapter``; returns the adapter node and its comments.

    ``seeds`` is the seed count of the proposal; ``replaced`` the encoder of the base
    config when the request replaces it, whose sampling rules the adapter then takes.
    """
    cls = BACKBONES.get(adapter)
    inner = _inner_slot(cls)
    preset = next((p for p in presets().get("backbone", {}).values() if p.type == adapter), None)
    node: dict[str, Any] = copy.deepcopy(preset.node) if preset else {"type": adapter}
    notes = {
        path[2:]: text
        for path, text in (preset.comments if preset else {}).items()
        if path[:2] == ("model", "backbone")
    }
    params, _ = component_parameters(cls)
    with_preset = f" (preset `{preset.name}`)" if preset else ""
    result.actions.append(
        f"wrapped {_describe(backbone)} in the adapter {adapter}{with_preset}: {reason} "
        "An adapter changes the model; its notes below say how."
    )
    if "npoint" in params and isinstance(seeds, int):
        node["npoint"] = seeds
        notes[("npoint",)] = "= model.proposal.num_seeds"
        result.actions.append(
            f"{adapter}.npoint: {seeds}, the number of seeds of the proposal "
            "(`model.proposal.num_seeds`): the points the adapter keeps are the seeds."
        )
    rules = [name for name in params if name.startswith("fps_")]
    kept = {name: replaced[name] for name in rules if replaced is not None and name in replaced}
    if kept:
        node.update(kept)
        for name in kept:
            notes[(name,)] = "the rule of the encoder it replaces: same seeds"
        result.actions.append(
            f"{adapter}: "
            + ", ".join(f"{name}: {yamltext.flow(value)}" for name, value in kept.items())
            + f", the sampling rules of the encoder it replaces ({replaced['type']}), so that "
            "the adapter keeps the very points that encoder ends on."
        )
    node[inner] = backbone
    notes.update({(inner, *path): text for path, text in comments.items()})
    result.adapter = adapter
    return node, notes


# ------------------------------------------------- what an option or an adapter lifts


class _Stages:
    """The stage nodes of the composed model while compose adapts them.

    A stage compose replaces as a whole lives in ``nodes`` (with its comments in
    ``notes``); a stage of the base config that only gets options set lives in
    ``partial`` (written as an override of the base node). ``asked`` is the model as it
    was asked for (base config, choices, overrides); ``user`` the keys the overrides of
    the request set, which compose never changes.
    """

    def __init__(
        self,
        nodes: dict[str, dict[str, Any]],
        notes: dict[str, dict[tuple[str, ...], str]],
        partial: dict[str, dict[str, Any]],
        asked: Mapping[str, Any],
        user: set[str],
    ) -> None:
        self.nodes, self.notes, self.partial, self.asked, self.user = (
            nodes,
            notes,
            partial,
            asked,
            user,
        )

    def node(self, slot: str) -> dict[str, Any]:
        """The node of a stage as it stands now."""
        if slot in self.nodes:
            return copy.deepcopy(self.nodes[slot])
        node = copy.deepcopy(dict(self.asked.get(slot) or {}))
        for key, value in self.partial.get(slot, {}).items():
            node[key] = copy.deepcopy(value)
        return node

    def options(self, slot: str) -> Mapping[str, Any]:
        """Constructor defaults overlaid with the node: what the contract rules read."""
        from grasppanda.models.detectors.seed_based import _Stage

        return _Stage(slot, self.node(slot)).options

    def fixed_by_user(self, slot: str, key: str) -> bool:
        return f"model.{slot}.{key}" in self.user

    def set(self, slot: str, values: Mapping[str, Any], comment: str) -> None:
        """Set options of a stage, keeping what else it states."""
        target = self.nodes[slot] if slot in self.nodes else self.partial.setdefault(slot, {})
        for key, value in values.items():
            target[key] = copy.deepcopy(value)
            self.notes.setdefault(slot, {})[(key,)] = comment

    def replace(self, slot: str, node: dict[str, Any], comment: str) -> None:
        """Put another node in place of a stage."""
        self.nodes[slot] = copy.deepcopy(node)
        self.notes[slot] = {("type",): comment}
        self.partial.pop(slot, None)


def _lift_stages(result: Composition, stages: _Stages) -> bool:
    """Set the options and insert the adapters that make the contract rules accept the
    stage set, where a stage declares them, and say so. ``--no-adapter`` skips this.
    Returns whether the grouper was wrapped in an axis adapter.

    In this order, each step on the stages as the steps before left them:

    1. two owners of a seed score: the proposal that has an option to leave it to the view
       stage gets it (``without_seed_score``: a graspness proposal predicts graspness only);
    2. a score the decoder reads and no stage predicts: the head gets the decode option
       that does not read it;
    3. a field of the matched labels a stage reads behind an assigner option: the option;
    4. a grouper whose axes differ from the head's: the registered axis adapter that
       bridges them (repeat the crop over the depth bins, or pool over them).
    """
    from grasppanda.models.detectors.seed_based import (
        STAGE_REGISTRIES,
        _seed_scores,
        _Stage,
        axis_adapters,
    )

    if not all(stages.node(slot).get("type") for slot in ("proposal", "view", "head")):
        return False
    registries = STAGE_REGISTRIES
    # 1. one owner per seed score
    proposal, view = (
        _Stage("proposal", stages.node("proposal")),
        _Stage("view", stages.node("view")),
    )
    for score in [s for s in _seed_scores(proposal) if s in _seed_scores(view)]:
        leave = proposal.cls.without_seed_score(proposal.options, score)
        if not isinstance(leave, Mapping):
            continue
        before = stages.node("proposal")
        if leave.get("type", before["type"]) != before["type"]:
            params, _ = component_parameters(registries["proposal"].get(leave["type"]))
            node = {key: value for key, value in leave.items() if key == "type" or key in params}
            stages.replace(
                "proposal",
                node,
                f"in place of {before['type']}: {view.name} owns the '{score}' of the seeds",
            )
            summary = (registries["proposal"].get(node["type"]).__doc__ or "").strip()
            change = (
                f"{node['type']} with "
                + ", ".join(
                    f"{key}: {yamltext.flow(value)}" for key, value in node.items() if key != "type"
                )
                + f", in place of {before['type']}, which has no option to leave the "
                f"'{score}' to another stage ({node['type']}: {summary.splitlines()[0]})"
            )
        else:
            changed = {k: v for k, v in leave.items() if k != "type" and before.get(k) != v}
            changed = {k: v for k, v in changed.items() if proposal.options.get(k) != v}
            if any(stages.fixed_by_user("proposal", key) for key in changed):
                continue
            dropped = [key for key in before if key != "type" and key not in leave]
            if dropped:
                node = {k: v for k, v in before.items() if k not in dropped}
                node.update(changed)
                stages.replace("proposal", node, f"{view.name} owns the '{score}' of the seeds")
            else:
                stages.set("proposal", changed, f"{view.name} owns the '{score}' of the seeds")
            change = ", ".join(f"{key}: {yamltext.flow(value)}" for key, value in changed.items())
            if dropped:
                change += f" (without {', '.join(dropped)})"
        result.origins["proposal"] = (
            f"{result.origins.get('proposal', 'base config')}, {score} left to the view stage"
        )
        result.actions.append(
            f"proposal: {change}. {before['type']} and {view.name} both predict the '{score}' "
            f"of a seed, and a detector needs one owner for it: {view.name} owns it, and the "
            "proposal neither predicts nor supervises it. This changes the proposal; "
            "`--no-adapter` leaves the pair refused."
        )
        proposal = _Stage("proposal", stages.node("proposal"))
    # 2. a score the decoder reads that no stage predicts
    head = _Stage("head", stages.node("head"))
    predicted = set(_seed_scores(proposal)) | set(_seed_scores(view))
    reader = getattr(head.cls, "required_seed_scores", None)
    for score in [s for s in (reader(head.options) if reader else ()) if s not in predicted]:
        decode = head.cls.without_seed_score(head.options, score)
        if not isinstance(decode, Mapping) or decode.get("type", head.name) != head.name:
            continue
        changed = {k: v for k, v in decode.items() if k != "type" and head.options.get(k) != v}
        if not changed or any(stages.fixed_by_user("head", key) for key in changed):
            continue
        stages.set("head", changed, f"no stage predicts the '{score}' this would read")
        result.origins["head"] = f"{result.origins.get('head', 'base config')}, decode option set"
        said = ", ".join(f"{key}: {yamltext.flow(value)}" for key, value in changed.items())
        result.actions.append(
            f"head: {said}. {head.name} as configured multiplies the '{score}' of the seeds "
            f"into the decoded score, but neither {proposal.name} nor {view.name} predicts "
            "it: this is the decode option of the head that does not read it. The emitted "
            "scores mean something else than the base method's (`score_kind`)."
        )
        head = _Stage("head", stages.node("head"))
    # 3. fields of the matched labels behind an assigner option
    assigner = _Stage("assigner", stages.node("assigner"))
    outputs = assigner.contract("optional_outputs", {})
    produced = assigner.contract("label_format", "dense")
    switches: dict[str, Any] = {}
    readers: list[str] = []
    for slot in ("view", "head"):
        stage = _Stage(slot, stages.node(slot))
        readable = stage.contract("label_formats") or (stage.contract("label_format", "dense"),)
        if produced not in readable:
            continue
        for field_name in stage.contract("needs_matched", ()):
            switch = outputs.get(field_name)
            if switch is None or assigner.option(switch[0]) == switch[1]:
                continue
            if stages.fixed_by_user("assigner", switch[0]):
                continue
            switches[switch[0]] = switch[1]
            readers.append(f"'{field_name}' ({stage.name})")
    if switches:
        stages.set("assigner", switches, "a stage reads what this option returns")
        result.origins["assigner"] = f"{result.origins.get('assigner', 'base config')}, options set"
        said = " and ".join(f"`{key}: {_yaml_value(value)}`" for key, value in switches.items())
        result.actions.append(
            f"assigner: {said}, the options with which {assigner.name} returns the field(s) "
            f"{', '.join(dict.fromkeys(readers))} of the matched labels."
        )
    # 4. feature axes between grouper and head
    grouper, head = _Stage("grouper", stages.node("grouper")), _Stage("head", stages.node("head"))
    emitted, expected = grouper.contract("out_axes"), head.contract("in_axes")
    if emitted == expected or grouper.contract("inner_axes") is not None:
        return False
    fitting = axis_adapters(emitted, expected, grouper.contract("support"))
    if len(fitting) != 1:
        return False
    adapter = fitting[0]
    cls = GROUPERS.get(adapter)
    preset = next((p for p in presets().get("grouper", {}).values() if p.type == adapter), None)
    node: dict[str, Any] = {"type": adapter}
    notes: dict[tuple[str, ...], str] = {("type",): "the axis adapter compose inserted"}
    params, _ = component_parameters(cls)
    if preset is not None:
        node.update(
            {k: copy.deepcopy(v) for k, v in preset.node.items() if k not in ("type", "grouper")}
        )
        notes.update(
            {
                path[2:]: text
                for path, text in preset.comments.items()
                if path[:2] == ("model", "grouper") and len(path) > 2 and path[2] != "grouper"
            }
        )
    bins = head.option("num_depth")
    if "num_depth" in params and isinstance(bins, int):
        node["num_depth"] = bins
        notes[("num_depth",)] = "= model.head.num_depth"
    inner = stages.node("grouper")
    inner_notes = stages.notes.get("grouper", {}) if "grouper" in stages.nodes else {}
    node["grouper"] = inner
    notes.update({("grouper", *path): text for path, text in inner_notes.items()})
    stages.nodes["grouper"], stages.notes["grouper"] = node, notes
    stages.partial.pop("grouper", None)
    with_preset = f" (preset `{preset.name}`)" if preset is not None else ""
    what = (
        f"the crop is computed once per seed and repeated over the {node.get('num_depth')} "
        "depth bins of the head with a learned embedding of the bin"
        if "depth" in expected
        else f"the crops of all depth bins are reduced by a {node.get('pool', 'max')} over the bins"
    )
    result.origins["grouper"] = f"{result.origins.get('grouper', 'base config')}, in {adapter}"
    result.actions.append(
        f"grouper: wrapped {inner['type']} in the axis adapter {adapter}{with_preset}: "
        f"{inner['type']} emits features with axes {emitted}, {head.name} reads {expected}, "
        f"and an axis is never added or dropped implicitly. {what[0].upper()}{what[1:]}. An "
        "adapter changes the model; `--no-adapter` leaves the pair refused."
    )
    return True


def _yaml_value(value: Any) -> str:
    return {True: "true", False: "false", None: "null"}.get(value, str(value))


# ------------------------------------------------------------------ the label set

#: The pseudo-slot of ``--set labels=dense|compact``: the label set of the training data.
LABELS_SLOT = "labels"
#: Label sets, by the supervision field that carries them.
LABEL_FIELDS = {"dense": "objects", "compact": f"extras.{'compact_labels'}"}


def _label_set(dataset: Mapping[str, Any]) -> str | None:
    """The label set a training dataset node serves, read from what its reader declares."""
    served = _served(dataset) if dataset.get("type") in DATASETS else None
    if served is None:
        return None
    for name, field_name in LABEL_FIELDS.items():
        if field_name in served:
            return name
    return None


def _moves_points(node: Any) -> bool:
    from grasppanda.registry import TRANSFORMS

    if not (isinstance(node, Mapping) and node.get("type") in TRANSFORMS):
        return False
    return bool(getattr(TRANSFORMS.get(node["type"]), "moves_points", False))


def _is_label_sampler(node: Any) -> bool:
    from grasppanda.registry import TRANSFORMS
    from grasppanda.transforms.points import SampleGraspLabels

    if not (isinstance(node, Mapping) and node.get("type") in TRANSFORMS):
        return False
    return issubclass(TRANSFORMS.get(node["type"]), SampleGraspLabels)


@lru_cache(maxsize=1)
def _shipped_models() -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """The model node of every shipped method config of :func:`_method_recipes`."""
    found = []
    for name, _ in _method_recipes():
        try:
            found.append((name, Config.load(name, validate=False).to_dict()["model"]))
        except (ConfigError, LookupError, TypeError):
            continue
    return tuple(found)


def _like(head: Mapping[str, Any], dataset: Mapping[str, Any]) -> list[str]:
    """Shipped method configs, the most alike first: the same head bundle, then a head
    that reads the same axes, then the same camera."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    heads = STAGE_REGISTRIES["head"]

    def axes(node: Mapping[str, Any]) -> Any:
        return heads.get(node["type"]).in_axes if node.get("type") in heads else None

    recipes = dict(_method_recipes())
    ranked = []
    for name, model in _shipped_models():
        theirs = model.get("head") or {}
        camera = recipes.get(name, {}).get("camera")
        ranked.append(
            (
                theirs.get("type") != head.get("type"),
                axes(theirs) != axes(head),
                camera != dataset.get("camera"),
                name,
            )
        )
    return [item[-1] for item in sorted(ranked)]


def _assigner_for(
    label_set: str, head: Mapping[str, Any], dataset: Mapping[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    """The assigner of the most alike shipped method config (:func:`_like`) that returns
    ``label_set`` labels without a weight per seed: its name and its assigner node."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    assigners = STAGE_REGISTRIES["assigner"]
    models = dict(_shipped_models())
    for name in _like(head, dataset):
        node = models[name].get("assigner") or {}
        if node.get("type") not in assigners:
            continue
        cls = assigners.get(node["type"])
        if getattr(cls, "label_format", "dense") == label_set and "weight" not in getattr(
            cls, "optional_outputs", {}
        ):
            return name, copy.deepcopy(dict(node))
    return None


def _label_rewrite(
    dataset: Mapping[str, Any], wanted: str, head: Mapping[str, Any], view_friction: Any
) -> tuple[dict[str, Any], dict[tuple[str, ...], str], str, str] | None:
    """The keys of a training dataset node that make it serve ``wanted`` labels, after the
    most alike shipped method config with the same reader that serves them: the keys, their
    comments, a sentence saying how, and the config they come from. ``None`` when no
    shipped config serves them with this reader."""
    recipes = dict(_method_recipes())
    source = next(
        (
            name
            for name in _like(head, dataset)
            if recipes.get(name, {}).get("type") == dataset.get("type")
            and _label_set(recipes[name]) == wanted
        ),
        None,
    )
    if source is None:
        return None
    theirs = recipes[source]
    steps = [copy.deepcopy(step) for step in dataset.get("transforms") or []]
    edits: dict[str, Any] = {}
    comments: dict[tuple[str, ...], str] = {}
    if wanted == "compact":
        removed = [str(step["type"]) for step in steps if _is_label_sampler(step)]
        edits["transforms"] = [step for step in steps if not _is_label_sampler(step)]
        edits["compact_labels"] = copy.deepcopy(dict(theirs["compact_labels"]))
        if isinstance(view_friction, (int, float)) and not isinstance(view_friction, bool):
            edits["compact_labels"]["view_friction_threshold"] = view_friction
        comments[("compact_labels",)] = f"the compact labels of {source}"
        how = (
            f"the `compact_labels` section of {source}"
            + (f" in place of the {', '.join(removed)} step" if removed else "")
            + ". The asset has to be built: `grasppanda prepare <dataset> --compact-labels "
            "--config <this config>` (from the dense label store of `grasppanda prepare "
            "<dataset> --labels`)"
        )
    else:
        sampler = next(step for step in theirs.get("transforms") or [] if _is_label_sampler(step))
        at = next((i for i, step in enumerate(steps) if _moves_points(step)), len(steps))
        edits["compact_labels"] = None
        edits["transforms"] = [*steps[:at], copy.deepcopy(dict(sampler)), *steps[at:]]
        comments[("compact_labels",)] = "unset: the dense label grids are served instead"
        before = f" before {steps[at]['type']}" if at < len(steps) else ""
        how = (
            f"`compact_labels` unset and the {sampler['type']} step of {source} inserted"
            f"{before}. The dense grids are read from the label store of `grasppanda prepare "
            "<dataset> --labels`, which a probe or a run needs (the config check does not look "
            "for it)"
        )
    comments[("transforms",)] = f"the base config's steps, for {wanted} labels"
    return edits, comments, how, source


def _fit_labels(
    result: Composition,
    stages: _Stages,
    dataset: Mapping[str, Any],
    explicit: str | None,
    chosen: set[str],
    automatic: bool,
) -> tuple[dict[str, Any], dict[tuple[str, ...], str]]:
    """Make the training data serve the label set the head bundle reads.

    The head bundle decides the label set (``label_format``); ``--set labels=`` states it.
    When the base config's training data serves the other one, the dataset section is
    rewritten (the ``compact_labels`` section, the ``SampleGraspLabels`` step) after the
    most alike shipped method config with the same reader that serves it, and an assigner
    of the base config that returns the other set is replaced by the assigner of the most
    alike shipped config that returns this one; an assigner set in the request is left as
    it is. Without ``automatic`` (``--no-adapter``) only ``--set labels=`` does this.
    Returns the dataset keys to write and their comments.
    """
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    head = stages.node("head")
    heads = STAGE_REGISTRIES["head"]
    if head.get("type") not in heads or dataset.get("type") not in DATASETS:
        return {}, {}
    if not (explicit or automatic):
        return {}, {}
    wanted = explicit or getattr(heads.get(head["type"]), "label_format", "dense")
    served = _label_set(dataset)
    edits: dict[str, Any] = {}
    comments: dict[tuple[str, ...], str] = {}
    if served is not None and served != wanted:
        friction = stages.options("view").get("max_friction")
        rewrite = _label_rewrite(dataset, wanted, head, friction)
        if rewrite is None:
            result.actions.append(
                f"labels: {head['type']} reads {wanted} labels and the training data of the base "
                f"config serves {served} ones, but no shipped config with {dataset['type']} "
                "serves them to copy the recipe from; the data recipe is left as it is."
            )
            return {}, {}
        edits, comments, how, _ = rewrite
        why = (
            f"`--set labels={wanted}`"
            if explicit
            else f"{head['type']} is supervised with {wanted} labels"
        )
        result.actions.append(
            f"labels: {wanted}. {why}, and the training data of the base config serves "
            f"{served} labels: data.train.dataset now serves {wanted} ones, {how}. The rest of "
            "the data recipe is the base config's."
        )
    # The assigner returns the label set the head reads, unless the request set it.
    assigner = stages.node("assigner")
    assigners = STAGE_REGISTRIES["assigner"]
    if (
        "assigner" not in chosen
        and assigner.get("type") in assigners
        and getattr(assigners.get(assigner["type"]), "label_format", "dense") != wanted
    ):
        found = _assigner_for(wanted, head, dataset)
        if found is not None:
            name, node = found
            stages.replace("assigner", node, f"the assigner of {name}: it returns {wanted} labels")
            result.origins["assigner"] = f"{name}, for {wanted} labels"
            result.actions.append(
                f"assigner: {node['type']}, the assigner of {name}. {assigner['type']} returns "
                f"{'dense' if wanted == 'compact' else 'compact'} matched labels and "
                f"{head['type']} is supervised with {wanted} ones."
            )
    return edits, comments


# ------------------------------------------------------------------------------ compose


def _shipped_name(path: Path) -> str | None:
    root = configs_root().resolve()
    resolved = path.resolve()
    if root in resolved.parents:
        return resolved.relative_to(root).with_suffix("").as_posix()
    return None


def _split_overrides(
    overrides: Sequence[str], choices: Sequence[Choice]
) -> tuple[dict[str, list[str]], list[str]]:
    """Overrides that address a component of this request, by slot, and the others."""
    addressed: dict[str, list[str]] = {}
    rest = []
    for item in overrides:
        key = item.partition("=")[0].strip()
        tokens = key.split(".")
        numeric = next((i for i, token in enumerate(tokens) if token.lstrip("-").isdigit()), None)
        if numeric is not None:
            whole = ".".join(tokens[:numeric])
            raise ComposeError(
                f"`{item}` addresses an element of a list. A composed config is a child "
                "config, and a list replaces the inherited one wholesale: give the whole list "
                f"(`{whole}=[...]`) or edit the composed file."
            )
        owner = next((c for c in choices if key.startswith(f"model.{c.slot}.")), None)
        if owner is None:
            rest.append(item)
            continue
        tail = item.strip()[len(f"model.{owner.slot}.") :]
        if tail.partition("=")[0].strip() == "type":
            raise ComposeError(f"`{item}` changes a component: use `--set {owner.slot}=...`")
        addressed.setdefault(owner.slot, []).append(tail)
    return addressed, rest


def _announce(result: Composition, choice: Choice) -> None:
    result.origins[choice.slot] = (
        f"preset `{choice.request}`" if choice.preset else "constructor defaults"
    )
    if choice.preset is not None:
        result.actions.append(
            f"{choice.slot}: preset `{choice.preset.name}` ({choice.preset.type}). "
            + choice.preset.summary
        )
        return
    members = presets().get(choice.slot, {}).values()
    same = [preset.name for preset in members if preset.type == choice.node["type"]]
    same.sort(key=lambda name: (len(name), name))  # the plain preset before its variants
    hint = ""
    if same:
        hint = (
            f" Its options are the component's own defaults, not a recipe for table-top "
            f"clouds; that is what its preset holds: `--set {choice.slot}={same[0]}`"
            + (f" (also: {', '.join(same[1:])})" if same[1:] else "")
            + "."
        )
    result.actions.append(
        f"{choice.slot}: {choice.node['type']} with its constructor defaults; no preset was "
        f"named.{hint}"
    )


# ------------------------------------------------------------------- the base of a chain


def _bases(path: Path) -> list[Path]:
    """The files a config file inherits from directly."""
    try:
        return list(_bases_of(path, path.stat().st_mtime_ns))
    except OSError:  # unreadable: loading it reports that
        return []


@lru_cache(maxsize=256)
def _bases_of(path: Path, stamp: int) -> tuple[Path, ...]:
    """:func:`_bases` of the file as it was at modification time ``stamp``."""
    try:
        raw = parse_yaml(path.read_text(encoding="utf-8"))
    except Exception:  # not YAML: loading it reports that
        return ()
    listed = raw.get(BASE_KEY, []) if isinstance(raw, Mapping) else []
    listed = [listed] if isinstance(listed, str) else listed
    found = []
    for item in listed:
        if not isinstance(item, str):
            continue
        try:  # next to the file, else a shipped config of that name (as loading does)
            found.append(resolve_base(path, item))
        except ConfigError:  # missing: loading it reports that
            continue
    return tuple(found)


def _levels(path: Path, seen: tuple[Path, ...] = ()) -> int:
    """How many levels of inheritance lie below a config file; 0 for a fragment."""
    if path in seen:
        return 0  # a cycle: loading it reports that
    return max((1 + _levels(base, (*seen, path)) for base in _bases(path)), default=0)


def _name(path: Path) -> str:
    return _shipped_name(path) or str(path)


def _merged_request(
    path: Path,
    sets: Sequence[str],
    overrides: Sequence[str],
    adapters: bool,
    unsets: Sequence[str] = (),
) -> tuple[argparse.Namespace, list[str], list[str], bool, list[str]] | None:
    """The request that composes ``path`` plus a new one, as one request on its base.

    ``None`` when ``path`` is not a composed file, or was edited after it was composed
    (its content, comments aside, is not what the command in its header writes): then
    that command no longer says what the file is.
    """
    try:
        earlier = _request(path)
        again = compose(
            earlier.base,
            earlier.sets,
            earlier.opts,
            adapters=earlier.adapters,
            unsets=earlier.unsets,
        )
        if parse_yaml(again.text(path)) != parse_yaml(path.read_text(encoding="utf-8")):
            return None
    except (ComposeError, ConfigError, OSError):
        return None
    slots = {item.partition("=")[0].strip() for item in sets}
    merged = [item for item in earlier.sets if item.partition("=")[0].strip() not in slots]
    keys = {item.partition("=")[0].strip() for item in overrides}
    kept = [item for item in earlier.opts if item.partition("=")[0].strip() not in keys]
    unset = [*dict.fromkeys([*earlier.unsets, *unsets])]
    return earlier, [*merged, *sets], [*kept, *overrides], earlier.adapters and adapters, unset


def _too_deep(base: str | Path, path: Path) -> ComposeError:
    """Why ``path`` cannot be inherited from once more, and what to compose from instead."""
    parents = [_name(parent) for parent in _bases(path)]
    message = (
        f"{_name(path)} cannot be the base of a composed config: it inherits from "
        f"{' and '.join(parents)}, and a config chain is limited to {MAX_BASE_HOPS} levels, so "
        "a composed config (one level more) cannot inherit from it."
    )
    try:
        earlier = _request(path)
    except ComposeError:
        earlier = None
    if earlier is not None:
        again = _spell(earlier.base, earlier.sets, earlier.opts, earlier.adapters, earlier.unsets)
        return ComposeError(
            f"{message} It was composed with `{again}` and edited afterwards, so that command "
            f"no longer says what the file is. Compose from `{earlier.base}` in one command, "
            "with those options and yours, and carry the edits over with `--opts "
            "key.path=value`."
        )
    return ComposeError(
        f"{message} Compose from `{parents[0] if parents else base}` instead; what this "
        "config changes on top of it is not carried over, so state it with `--opts "
        "key.path=value` where you need it."
    )


# ------------------------------------------------------------------------------ compose


def _asked_stages(
    model: Mapping[str, Any], choices: Sequence[Choice], rest: Sequence[str]
) -> dict[str, Any]:
    """The stages of the composed model as they were asked for, before compose adapts the
    encoder: the base config's, the choices, and the overrides addressed to other stages."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    stages = {
        slot: copy.deepcopy(dict(model[slot]))
        for slot in STAGE_REGISTRIES
        if isinstance(model.get(slot), Mapping)
    }
    stages.update({choice.slot: choice.node for choice in choices if choice.slot in stages})
    mine = [
        item
        for item in rest
        if (parts := item.partition("=")[0].strip().split("."))[0] == "model"
        and len(parts) > 2
        and parts[1] in stages
    ]
    tree = LoadedTree(data={"model": stages})
    try:
        apply_overrides(tree, mine)
    except ConfigError:
        return stages  # an override that cannot be applied: the config check says so
    return tree.data["model"]


def compose(
    base: str | Path,
    sets: Sequence[str] = (),
    overrides: Sequence[str] = (),
    *,
    adapters: bool = True,
    unsets: Sequence[str] = (),
) -> Composition:
    """Compose a child config of ``base``.

    Args:
        base: A config file or the name of a shipped config. It must inherit fragments
            only, like a method config: a chain is two levels deep. A config that was
            itself composed is composed again from its own base, both requests merged.
        sets: ``slot=choice`` pairs; a choice is a preset name or a component type.
        overrides: Further ``key.path=value`` overrides, written into the child config.
            A path below a slot of ``sets`` (``model.backbone.grid_size=0.01``) addresses
            the component that was set, wherever an adapter puts it.
        adapters: Insert the adapter the contract rules ask for, when there is exactly one,
            and set the options of a stage that lift a refusal (a decode option, the options
            of an assigner, graspness only, the label set the head reads). ``False`` leaves
            a refused stage set refused.
        unsets: ``key.path`` of options the base config sets that go back to the default of
            their component (``data.train.dataset.compact_labels``).

    Returns:
        The :class:`Composition`; check ``accepted`` before writing ``text()``.

    Raises:
        ComposeError: The request names an unknown slot, preset or component, or a base
            that cannot be inherited from once more.
        ConfigError: The base config itself is invalid, or an override cannot be applied.
    """
    from grasppanda.models.detectors.seed_based import (
        STAGE_REGISTRIES,
        SeedGraspDetector,
        stage_contract_problems,
    )

    base_path = resolve_config_path(base)
    if base_path.is_file() and _levels(base_path) >= MAX_BASE_HOPS:
        merged = _merged_request(base_path, sets, overrides, adapters, unsets)
        if merged is None:
            raise _too_deep(base, base_path)
        earlier, all_sets, all_overrides, both, all_unsets = merged
        result = compose(earlier.base, all_sets, all_overrides, adapters=both, unsets=all_unsets)
        result.actions.insert(
            0,
            f"{_name(base_path)} is itself a composed config, and a config chain is limited "
            f"to {MAX_BASE_HOPS} levels: this config is composed from its base, "
            f"{result.base}, with its request and yours merged (a `--set` of yours replaces "
            "the one of the same slot).",
        )
        return result
    config = Config.load(base_path)
    try:
        sets, recipe_sets = recipe_presets.split(sets)
        chosen = [
            (slot, request, recipe_presets.resolve(slot, request)) for slot, request in recipe_sets
        ]
    except PresetError as error:
        raise ComposeError(str(error)) from None
    written, model = config.to_dict().get("model"), config.resolved().get("model")
    if not isinstance(model, Mapping):
        raise ComposeError(f"{base} has no `model` section to compose into")
    detector = DETECTORS.get(model["type"])
    slots = component_slots(detector)
    # Post-processing is no part of the model: every detector has this slot.
    post_sets = [item for item in sets if item.partition("=")[0].strip() == POST_SLOT]
    sets = [item for item in sets if item not in post_sets]
    # The label set of the training data: `--set labels=dense|compact`.
    label_sets = [item for item in sets if item.partition("=")[0].strip() == LABELS_SLOT]
    sets = [item for item in sets if item not in label_sets]
    labels = _labels_request(label_sets, detector)
    # A loss sits inside a stage: `head.score_loss=quality-focal`.
    term_sets = [item for item in sets if "." in item.partition("=")[0]]
    sets = [item for item in sets if item not in term_sets]
    seed_based = issubclass(detector, SeedGraspDetector)
    result = Composition(
        base=_shipped_name(base_path) or str(base),
        base_path=base_path,
        detector=model["type"],
        seed_based=seed_based,
        choices=_parse_sets(sets, slots, model["type"]),
        overrides=tuple(overrides),
        adapters=adapters,
        recipes=chosen,
        labels=labels,
        unsets=tuple(unsets),
    )
    by_slot = {choice.slot: choice for choice in result.choices}
    stages = {slot: model.get(slot) for slot in slots}
    stages.update({slot: choice.node for slot, choice in by_slot.items()})
    terms = _parse_terms(term_sets, slots, stages, model["type"])
    # The more specific owner first: `model.head.score_loss.beta` belongs to the loss.
    addressed, rest = _split_overrides(overrides, [*terms, *result.choices])
    for choice in result.choices:
        cls = slots[choice.slot].get(choice.node["type"])
        if slots[choice.slot] is BACKBONES and is_adapter(cls):
            how = (
                "Compose does not write this wrapper: it joins an image encoder to a scene "
                "encoder, and the YAML fragment to copy is in "
                "docs/slots/image_encoders.md#point-painting."
                if "Painted" in choice.node["type"]
                else "Set the encoder (`--set backbone=<encoder>`); compose inserts the "
                "adapter when the contracts ask for it."
            )
            raise ComposeError(
                f"'{choice.request}' is an adapter: {choice.node['type']} wraps another "
                f"encoder and is not one itself. {how}"
            )
        if choice.slot != "backbone" and addressed.get(choice.slot):
            apply_overrides(LoadedTree(data=choice.node), addressed[choice.slot])
    # What the other stages will be, so that the encoder is fitted to the model that was
    # asked for: a grouper or a seed count set by an override counts like one set by --set.
    asked = _asked_stages(model, result.choices, rest) if seed_based else {}
    for choice in result.choices:
        _announce(result, choice)
        mine = addressed.get(choice.slot, [])
        if seed_based and choice.slot == "backbone":
            keep = {item.partition("=")[0].strip().split(".")[0] for item in mine}
            _fit_encoder(result, choice, model, keep, asked.get("grouper", model["grouper"]))
            if mine:
                apply_overrides(LoadedTree(data=choice.node), mine)
        if mine:
            for item in mine:
                choice.comments.pop(tuple(item.partition("=")[0].strip().split(".")), None)
            result.actions.append(f"{choice.slot}: overrides as given, {' '.join(mine)}.")

    nodes = {choice.slot: choice.node for choice in result.choices}
    notes = {choice.slot: dict(choice.comments) for choice in result.choices}
    if seed_based and adapters:
        proposal = asked.get("proposal", model["proposal"])
        encoder = nodes.get("backbone", written["backbone"])
        refused = stage_contract_problems(backbone=encoder, proposal=proposal)
        fitting = _adapters_for(encoder, proposal) if refused else []
        if len(fitting) == 1:
            nodes["backbone"], notes["backbone"] = _wrap(
                result,
                fitting[0],
                copy.deepcopy(dict(encoder)),
                notes.get("backbone", {}),
                reason=refused[0].split(". ")[0].rstrip(".") + ".",
                seeds=_option(slots["proposal"], proposal, "num_seeds"),
                replaced=model["backbone"] if "backbone" in by_slot else None,
            )
            origin = result.origins.get("backbone", "base config")
            result.origins["backbone"] = f"{origin}, in the adapter {fitting[0]}"

    partial = _set_terms(result, terms, addressed, stages, nodes, notes)
    result.choices += terms

    # The width of the seed features: an encoder that returns another width than a grouper
    # adds to its crops gets the grouper's own projection, where the grouper has one.
    if seed_based and adapters:
        _project_seed_features(result, nodes, notes, written, model)

    # The label set the head reads, then what an option or an adapter of a stage lifts.
    dataset = ((config.resolved().get("data") or {}).get("train") or {}).get("dataset")
    label_edits: dict[str, Any] = {}
    label_notes: dict[tuple[str, ...], str] = {}
    if seed_based:
        user = {item.partition("=")[0].strip() for item in overrides}
        # Stages are copied as the base config writes them, not with every default filled.
        lifting = _Stages(nodes, notes, partial, _asked_stages(written, result.choices, rest), user)
        if isinstance(dataset, Mapping):
            label_edits, label_notes = _fit_labels(
                result, lifting, dataset, labels, set(by_slot), adapters
            )
        if adapters and _lift_stages(result, lifting) and "grouper" not in by_slot:
            # An override of the base grouper addresses it inside the axis adapter.
            rest = [
                "model.grouper.grouper." + item.removeprefix("model.grouper.")
                if item.startswith("model.grouper.")
                else item
                for item in rest
            ]

    # A node of the same type as the base config's would be merged with it by the loader,
    # and options of the base would leak into what was asked for. `--set` replaces.
    for slot, node in list(nodes.items()):
        inherited = written.get(slot)
        if isinstance(inherited, Mapping) and inherited.get("type") == node["type"]:
            nodes[slot] = {"_delete_": True, **node}
            notes[slot][("_delete_",)] = (
                f"replace the base config's {node['type']}: none of its options is inherited"
            )

    nodes.update({slot: options for slot, options in partial.items() if slot not in nodes})
    order = [slot for slot in (STAGE_REGISTRIES if seed_based else slots) if slot in nodes]
    body: dict[str, Any] = {"model": {slot: nodes[slot] for slot in order}} if order else {}
    comments = {
        ("model", slot, *path): text for slot in order for path, text in notes[slot].items()
    }

    # A dataset section a stage reads is added where the base config has none. The label
    # set is the head's: the section of the other label set is not added.
    added: dict[str, Any] = dict(label_edits)
    comments.update({("data", "train", "dataset", *path): t for path, t in label_notes.items()})
    serves = _label_set({**dataset, **label_edits}) if isinstance(dataset, Mapping) else None
    for choice in result.choices:
        preset = choice.preset
        for section, value in (preset.data if preset else {}).items():
            if not isinstance(dataset, Mapping) or dataset.get(section) is not None:
                continue
            if section in added or (section == "compact_labels" and serves != "compact"):
                continue
            added[section] = copy.deepcopy(value)
            prefix = ("data", "train", "dataset", section)
            comments.update(
                {p: t for p, t in preset.comments.items() if p[: len(prefix)] == prefix}
            )
            result.actions.append(
                f"added `data.train.dataset.{section}`: {preset.type} reads it and the training "
                "dataset of the base config does not serve it. The asset has to be built; "
                f"{preset.location} says how."
            )
    if added:
        body["data"] = {"train": {"dataset": added}}
    if post_sets:
        _set_post(result, body, post_sets)
        _keep_full_cloud(result, body, comments, config.resolved().get("data") or {})
    if chosen:
        members = [preset for _, _, group in chosen for preset in group]
        try:
            result.actions += recipe_presets.apply(members, config.to_dict(), body, comments)
        except PresetError as error:
            raise ComposeError(str(error)) from None
    if rest:
        tree = LoadedTree(data=body)
        apply_overrides(tree, rest)
        body = tree.data
        result.actions.append("overrides as given: " + " ".join(rest) + ".")
    if unsets:
        _unset(result, config.resolved(), body, comments, unsets)
    result.body, result.comments = body, comments
    _check(result)
    _encoder_recipes(result)
    if not result.accepted:
        result.hints = _hints(result)
    return result


def _labels_request(items: Sequence[str], detector: type) -> str | None:
    """The label set of ``--set labels=...``, checked; ``None`` when not asked for."""
    from grasppanda.models.detectors.seed_based import SeedGraspDetector

    if not items:
        return None
    if len(items) > 1:
        raise ComposeError(f"slot '{LABELS_SLOT}' is set twice")
    value = items[0].partition("=")[2].strip()
    if value not in LABEL_FIELDS:
        raise ComposeError(
            f"`--set {items[0]}`: the label set is one of {', '.join(LABEL_FIELDS)} (dense grids "
            "of every grasp per point and view, or compact labels of one grasp per point and "
            "view)."
        )
    if not issubclass(detector, SeedGraspDetector):
        raise ComposeError(
            f"`--set {LABELS_SLOT}=` chooses the label set of a seed-based method; "
            f"{detector.__name__} reads its own labels."
        )
    return value


def _unset(
    result: Composition,
    resolved: Mapping[str, Any],
    body: dict[str, Any],
    comments: dict[tuple[str, ...], str],
    paths: Sequence[str],
) -> None:
    """``--unset key.path``: an option of a component back to its constructor default,
    written into the child config (a child config cannot delete what it inherits)."""
    for path in paths:
        keys = [key for key in path.strip().split(".") if key]
        node: Any = resolved
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, Mapping) else None
        if not keys or not isinstance(node, Mapping) or keys[-1] not in node:
            raise ComposeError(
                f"`--unset {path}`: the base config sets nothing there. --unset takes the path "
                "of an option of a component that the base config sets, e.g. "
                "`data.train.dataset.compact_labels`."
            )
        cls = _component_class_at(resolved, keys[:-1])
        params = component_parameters(cls)[0] if cls is not None else {}
        param = params.get(keys[-1])
        if param is None or param.default is inspect.Parameter.empty:
            raise ComposeError(
                f"`--unset {path}`: {'.'.join(keys[:-1])} is not a component with a default "
                f"for '{keys[-1]}', so there is nothing to go back to. Give a value with "
                f"`--opts {path}=...` instead."
            )
        default = param.default
        target = body
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = copy.deepcopy(default)
        comments[tuple(keys)] = f"unset: back to the default of {cls.__name__}"
        result.actions.append(
            f"unset `{path}`: {cls.__name__}'s default, {_yaml_value(default)}, in place of "
            "the base config's value."
        )


def _component_class_at(resolved: Mapping[str, Any], keys: Sequence[str]) -> type | None:
    """The registered class of the component node at ``keys`` of a config, if it is one."""
    from grasppanda.config.validate import iter_components

    for path, registry, name in iter_components(resolved):
        if list(path) == list(keys):
            try:
                return registry.get(name)
            except RegistryError:
                return None
    return None


def _set_post(result: Composition, body: dict[str, Any], post_sets: Sequence[str]) -> None:
    """Write the chain of ``--set post=a+b`` as the evaluation variant ``post``."""
    if len(post_sets) > 1:
        raise ComposeError(
            f"slot '{POST_SLOT}' is set twice; a chain is one `--set {POST_SLOT}=...` "
            "with its steps joined by `+`, e.g. `--set post=collision+nms+top-k=50`"
        )
    text = post_sets[0].partition("=")[2].strip()
    try:
        chain = chain_nodes(text)
    except PostSpecError as error:
        raise ComposeError(str(error)) from None
    body.setdefault("evaluation", {}).setdefault("variants", {})[POST_VARIANT] = chain
    result.post = (text,)
    steps = " → ".join(str(node["type"]) for node in chain)
    result.actions.append(
        f"post: evaluation variant `{POST_VARIANT}` = {steps}, applied in this order to the "
        "raw predictions of every frame and scored beside the variants of the base config "
        f"(`grasppanda demo --variant {POST_VARIANT}` applies it to one frame)."
    )


def _keep_full_cloud(
    result: Composition,
    body: dict[str, Any],
    comments: dict[tuple[str, ...], str],
    data: Mapping[str, Any],
) -> None:
    """Switch ``keep_full_points`` on in the evaluation sections that drop the cloud
    before sub-sampling, when a step of the chain cannot run without it."""
    from grasppanda.registry import POSTPROCESSORS

    chain = body["evaluation"]["variants"][POST_VARIANT]
    needy = [
        str(node["type"])
        for node in chain
        if getattr(POSTPROCESSORS.get(node["type"]), "reads_full_points", False)
        and node.get("cloud", "full_points") == "full_points"
    ]
    changed = []
    for section in ("val", "test"):
        dataset = (data.get(section) or {}).get("dataset")
        if needy and isinstance(dataset, Mapping) and dataset.get("keep_full_points") is False:
            target = body.setdefault("data", {}).setdefault(section, {}).setdefault("dataset", {})
            target["keep_full_points"] = True
            comments[("data", section, "dataset", "keep_full_points")] = (
                f"{needy[0]} reads the cloud before sub-sampling"
            )
            changed.append(f"data.{section}.dataset.keep_full_points")
    if changed:
        result.actions.append(
            f"set `{'` and `'.join(changed)}` to true: {' and '.join(dict.fromkeys(needy))} "
            "reads the cloud before sub-sampling (`observation.full_points`), which the base "
            "config drops there. To test the cloud the network saw instead, write the step "
            "as `<step>:cloud=points`."
        )


# -------------------------------------------------------------- what to type instead


@lru_cache(maxsize=1)
def _method_recipes() -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Shipped configs a composed config can inherit from, as shipped (not composed), each
    with the node of its training dataset. Read once per process."""
    root = configs_root()
    found = []
    for name in available_configs():
        path = root / f"{name}.yaml"
        if name.startswith(("_base_/", "experimental/")) or _levels(path) != 1:
            continue
        try:
            dataset = Config.load(path, validate=False).to_dict()["data"]["train"]["dataset"]
        except (ConfigError, LookupError, TypeError):
            continue
        found.append((name, dataset))
    return tuple(found)


def _served(dataset: Mapping[str, Any]) -> frozenset[str] | None:
    reader = DATASETS.get(dataset["type"]) if dataset.get("type") in DATASETS else None
    declare = getattr(reader, "served_supervision", None)
    return declare(dataset) if callable(declare) else None


def _hints(result: Composition) -> list[str]:
    """What to type instead, where compose can tell: for a head bundle and an assigner of
    two label sets, and for a data recipe that does not serve what the model reads (the
    exact ``--set``, ``--unset`` and ``--opts`` that rewrite it, and the shipped configs
    whose recipe serves it).

    Read from what stages and datasets declare (``supervision_needs``,
    ``served_supervision``); a request refused for another reason gets no hint.
    """
    hints = _label_set_hints(result)
    if any(issue.path == "data.train.dataset" for issue in result.issues):
        hints += _data_hints(result)
    return hints


def _opts_line(edits: Mapping[str, Any], prefix: str = "data.train.dataset") -> str:
    """``--unset`` and ``--opts`` that write ``edits`` (a ``None`` is an unset)."""
    unsets = [f"--unset {prefix}.{key}" for key, value in edits.items() if value is None]
    values = [
        shlex.quote(f"{prefix}.{key}={yamltext.flow(value)}")
        for key, value in edits.items()
        if value is not None
    ]
    return " ".join([*unsets, *(["--opts", *values] if values else [])])


def _data_hints(result: Composition) -> list[str]:
    """For a data recipe that does not serve what the composed model reads: per missing
    label set or derived asset, the exact command-line words that make it serve it."""
    model, dataset = result.merged.get("model") or {}, result.merged.get("dataset") or {}
    try:
        needs = DETECTORS.get(model["type"]).supervision_needs(model)
        served = _served(dataset)
    except (LookupError, TypeError):
        return []
    if needs is None or served is None:
        return []
    needed = {field_name for field_name, _ in needs}
    missing = needed - served
    hints = []
    view = model.get("view") or {}
    head = model.get("head") or {}
    friction = view.get("max_friction")
    for wanted, field_name in LABEL_FIELDS.items():
        if field_name not in missing:
            continue
        readers = " and ".join(dict.fromkeys(who for name, who in needs if name == field_name))
        rewrite = _label_rewrite(dataset, wanted, head, friction)
        if rewrite is None:
            continue
        edits, _, _, source = rewrite
        hints.append(
            f"To make the training data serve the {wanted} labels {readers} reads: "
            f"`--set {LABELS_SLOT}={wanted}`, which writes `{_opts_line(edits)}` (the recipe of "
            f"{source})."
        )
    # A derived asset: the dataset section that serves it, as a preset writes it.
    for field_name in sorted(missing - set(LABEL_FIELDS.values())):
        if any(field_name.startswith(f"{other}.") for other in missing):
            continue
        for members in presets().values():
            found = next(
                (
                    (section, value, preset)
                    for preset in members.values()
                    for section, value in preset.data.items()
                    if field_name
                    in ((_served({**dataset, section: value}) or frozenset()) - served)
                ),
                None,
            )
            if found is not None:
                section, value, preset = found
                hints.append(
                    f"To serve `{field_name}`: `{_opts_line({section: value})}` (the section of "
                    f"{preset.location}, which says how to build the asset)."
                )
                break
    # A component named by its type whose preset carries the dataset section it reads.
    for choice in result.choices:
        if choice.preset is not None:
            continue
        bringing = [
            preset
            for preset in presets().get(choice.slot, {}).values()
            if preset.type == choice.node["type"]
            and ((_served({**dataset, **preset.data}) or frozenset()) - served) & missing
        ]
        bringing.sort(key=lambda preset: (len(preset.name), preset.name))
        if bringing:
            brought = dict.fromkeys(section for preset in bringing for section in preset.data)
            ways = " or ".join(f"`--set {choice.slot}={preset.name}`" for preset in bringing)
            hints.append(
                f"{choice.node['type']} was named by its type, which brings no dataset "
                f"section. A preset of it does (`{'`, `'.join(brought)}`): {ways}."
            )

    def serves(recipe: Mapping[str, Any]) -> bool:
        reader = DATASETS.get(recipe["type"]) if recipe.get("type") in DATASETS else None
        clean = not list(getattr(reader, "check_config", lambda node: ())(recipe) or ())
        return clean and needed <= (_served(recipe) or frozenset())

    fitting = [name for name, theirs in _method_recipes() if name != result.base and serves(theirs)]
    if fitting:
        hints.append(
            "Or compose from a shipped config whose training data serves all of it: "
            + ", ".join(fitting)
            + "."
        )
    return hints


def _label_set_hints(result: Composition) -> list[str]:
    """For a head bundle and an assigner of two label sets: compose writes the label set the
    head reads unless the request set the assigner (or ``--set labels=``) otherwise."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    if not any("must agree on the label set" in problem for problem in result.problems):
        return []
    model = result.merged.get("model") or {}
    head, assigner = model.get("head") or {}, model.get("assigner") or {}
    heads, assigners = STAGE_REGISTRIES["head"], STAGE_REGISTRIES["assigner"]
    if head.get("type") not in heads or assigner.get("type") not in assigners:
        return []
    wanted = getattr(heads.get(head["type"]), "label_format", "dense")
    theirs = getattr(assigners.get(assigner["type"]), "label_format", "dense")
    able = [
        name
        for name in heads.names()
        if getattr(heads.get(name), "label_format", "dense") == theirs
    ]
    if result.labels and result.labels != wanted:
        asked = f"the request asked for that with `--set {LABELS_SLOT}={result.labels}`"
    elif any(choice.slot == "assigner" for choice in result.choices):
        asked = "the request set the assigner (`--set assigner=...`)"
    else:
        asked = "`--no-adapter` keeps the assigner and the data of the base config"
    return [
        f"{head['type']} reads {wanted} labels and {assigner['type']} returns {theirs} ones; "
        f"{asked}. Without it compose gives {head['type']} an assigner and training data of "
        f"{wanted} labels; or keep {assigner['type']} with a head bundle that reads {theirs} "
        f"labels ({', '.join(able)})."
    ]


def _check(result: Composition) -> None:
    """Load the composed file as a user would and record what the checks say."""
    from grasppanda.models.detectors.seed_based import (
        STAGE_REGISTRIES,
        stage_contract_problems,
        stage_contract_warnings,
    )

    with tempfile.TemporaryDirectory(prefix="grasppanda-compose-") as scratch:
        path = Path(scratch) / "composed.yaml"
        path.write_text(result.text(), encoding="utf-8")
        merged = Config.load(path, validate=False)
        model = merged.to_dict().get("model") or {}
        train = (merged.to_dict().get("data") or {}).get("train") or {}
        result.merged = {"model": model, "dataset": train.get("dataset") or {}}
        if result.seed_based and all(
            isinstance(model.get(slot), Mapping) and "type" in model[slot]
            for slot in STAGE_REGISTRIES
        ):
            stages = {slot: model[slot] for slot in STAGE_REGISTRIES}
            try:
                result.problems = list(dict.fromkeys(stage_contract_problems(**stages)))
                result.warnings = list(dict.fromkeys(stage_contract_warnings(**stages)))
                for group in CONTRACT_GROUPS:
                    found = stage_contract_problems(**{slot: stages[slot] for slot in group})
                    result.verdicts.append(ContractVerdict(group, tuple(dict.fromkeys(found))))
            except RegistryError:
                pass  # an unknown component: the config check below says so
            for slot in STAGE_REGISTRIES:
                result.stages[slot] = (
                    _describe(model[slot]) if slot == "backbone" else str(model[slot]["type"])
                )
                result.origins.setdefault(slot, "base config")
        try:
            merged.validate()
        except ConfigError as error:
            known = set(result.problems)
            result.issues = [issue for issue in error.issues if issue.message not in known]
        else:
            result.config_hash = merged.digest()


def _encoder_recipes(result: Composition) -> None:
    """Refuse a recipe preset written for the scene encoder of the seed-based skeleton
    (:data:`grasppanda.combine.grid.RECIPE_ENCODER_PRESETS`) under a detector without that
    skeleton, where the config check cannot: it refuses a pattern that names `backbone`,
    but a pattern written as a regular expression (``ptv3-adamw-onecycle``'s
    ``re:\\.block``) meets the parameters only when training starts. Structural."""
    if result.seed_based or not result.recipes:
        return
    from .grid import RECIPE_ENCODER_PRESETS

    told = {issue.path.split(".")[0] for issue in result.issues}
    for _, _, members in result.recipes:
        for member in members:
            if f"{member.slot}/{member.name}" not in RECIPE_ENCODER_PRESETS:
                continue
            if told & set(member.body):
                continue  # the config check refused what it names
            optimizer = member.body.get("optimizer") or {}
            patterns = [
                str(pattern)
                for pattern in optimizer.get("lr_multipliers") or {}
                if str(pattern).startswith("re:")
            ]
            why = (
                f": no parameter of {result.detector} matches its pattern `{patterns[0]}`, so "
                "training would stop before its first step"
                if patterns
                else ""
            )
            result.issues.append(
                Issue(
                    "optimizer.lr_multipliers" if patterns else member.slot,
                    f"`{member.slot}={member.name}` is written for the scene encoder of the "
                    f"seed-based skeleton (`model.backbone`), which {result.detector} does not "
                    f"have{why}. It applies to the seed-based methods only.",
                    kind="structural",
                )
            )
            result.config_hash = None


# ------------------------------------------------------------------- the command, as text


def command_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """The arguments of a compose request; shared by the command line and :func:`recompose`."""
    parser = parser or argparse.ArgumentParser(prog=_COMMAND, add_help=False)
    parser.add_argument("base", nargs="?", help="config file or name of a shipped config")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="SLOT=CHOICE",
        help="put a preset or a component into a slot, e.g. backbone=ptv3 (repeatable); "
        "post=collision+nms+top-k=50 defines a post-processing variant; "
        "the recipe has slots too: optimizer, schedule, hooks, augment, recipe",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="further overrides written into the composed config, e.g. "
        "model.backbone.grid_size=0.01 train.epochs=20",
    )
    parser.add_argument(
        "--unset",
        dest="unsets",
        action="append",
        default=[],
        metavar="KEY",
        help="an option the base config sets goes back to the default of its component, "
        "e.g. data.train.dataset.compact_labels (repeatable)",
    )
    parser.add_argument(
        "--no-adapter",
        dest="adapters",
        action="store_false",
        help="do not insert an adapter or set an option that lifts a refusal; a refused "
        "pair then stays refused",
    )
    return parser


def _request(path: str | Path) -> argparse.Namespace:
    """The compose request in the header of a composed file (``base``, ``sets``, ``opts``,
    ``adapters``); :class:`ComposeError` when the file has none."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"#   {_COMMAND} ")), None)
    if start is None:
        raise ComposeError(f"{path} has no `{_COMMAND} ...` line in its header")
    command, index = "", start
    while True:
        piece = lines[index][1:].strip()
        command += " " + piece.removesuffix("\\").strip()
        if not piece.endswith("\\") or index + 1 >= len(lines):
            break
        index += 1
    return command_parser().parse_args(shlex.split(command)[2:])


def recompose(path: str | Path) -> Composition:
    """Compose again what the header of a composed file says it was composed from."""
    args = _request(path)
    return compose(args.base, args.sets, args.opts, adapters=args.adapters, unsets=args.unsets)
