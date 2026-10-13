"""Compose a config: a base config, one component per ``slot=choice``, checked.

    composition = compose("gsnet/gsnet_sunet14_g1b-realsense_10e", ["backbone=pointnet2"])
    composition.accepted          # the config check passes
    composition.text(path)        # the child config, with a header that says how it was made

What happens, in this order, and what is reported for each step:

1. every ``slot=choice`` is resolved to a preset (:mod:`grasppanda.combine.presets`) or,
   failing that, to a registered component with its constructor defaults;
2. a new encoder is fitted to what the base method feeds and expects: its
   ``in_channels`` when the method feeds real point features, and its ``out_channels``,
   where it has that option, to the width of the encoder it replaces;
3. when the encoder rule of the stage contracts refuses the pair of encoder and proposal
   and exactly one registered adapter bridges it, the adapter is inserted, with the seed
   count of the proposal and the sampling rules of the replaced encoder;
4. a dataset section a preset's stage reads is added when the base config has none;
5. the child config is written to a scratch file and loaded, so the contract verdicts
   and the config check are those of the file the user gets.

A loss is set inside a stage: ``head.score_loss=quality-focal`` puts a preset of
``grasppanda/presets/loss/`` (or a registered loss, by its type) into that loss option of
the head bundle and keeps the weight of the loss it replaces.

No rule about what fits what lives here: every verdict is
:func:`~grasppanda.models.detectors.seed_based.stage_contract_problems` or the config
checker, quoted.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import inspect
import os
import shlex
import tempfile
import textwrap
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grasppanda.config import Config, ConfigError, Issue, configs_root, resolve_config_path
from grasppanda.config.loader import LoadedTree, apply_overrides
from grasppanda.registry import (
    BACKBONES,
    DETECTORS,
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
        if self.overrides:
            parts.append("--opts " + " ".join(shlex.quote(item) for item in self.overrides))
        if not self.adapters:
            parts.append("--no-adapter")
        return " ".join(parts)

    def refusal(self) -> list[str]:
        """Every reason the composed config is refused, contracts first."""
        return [*self.problems, *(f"{issue.path}: {issue.message}" for issue in self.issues)]

    def text(self, output: str | Path | None = None) -> str:
        """The child config as a file: header, ``_base_`` and the composed fragment.

        ``output`` is where the file will live. Inside the shipped config tree the base is
        referred to by a relative path, as shipped configs do; elsewhere by its absolute
        path, so the file works from any directory.
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
        )
        lines = [
            "EXPERIMENTAL. No AP is claimed. Composed by `grasppanda compose`: this config passes",
            "the config check; it has not been trained to convergence or evaluated.",
            "",
            f"  base  {self.base} ({self.detector})",
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
        close = difflib.get_close_matches(request, registry.names(), n=3, cutoff=0.6)
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
            family = (
                ""
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


def _output_width(node: Mapping[str, Any]) -> int | None:
    """Channels a backbone node returns: its ``out_channels`` option, else by building it."""
    if isinstance(node.get("out_channels"), int):
        return int(node["out_channels"])
    try:
        return int(BACKBONES.build(copy.deepcopy(dict(node))).out_channels)
    except Exception:  # an optional dependency is missing, or the node cannot be built alone
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
    result: Composition, choice: Choice, model: Mapping[str, Any], keep: set[str]
) -> None:
    """Fit a new encoder to what the base method feeds and to the width it replaces.

    Options named in ``keep`` were set by the user and are left alone.
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
    width = _output_width(replaced)
    fits = "out_channels" in params and "out_channels" not in keep and width is not None
    if fits and node.get("out_channels") != width:
        node["out_channels"] = width
        choice.comments[("out_channels",)] = (
            f"the width of the encoder it replaces ({_describe(replaced)})"
        )
        result.actions.append(
            f"out_channels: {width}, the output width of the encoder it replaces "
            f"({_describe(replaced)}), so that the other stages keep the width they have in "
            "the base model and the encoder is the only stage that differs."
        )


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
    from grasppanda.models.detectors.seed_based import stage_contract_problems

    fitting = []
    for name in BACKBONES.names():
        cls = BACKBONES.get(name)
        if not is_adapter(cls):
            continue
        wrapped = {"type": name, _inner_slot(cls): backbone}
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
        if any(token.lstrip("-").isdigit() for token in key.split(".")):
            raise ComposeError(
                f"`{item}` addresses an element of a list. A composed config is a child "
                "config, and a list replaces the inherited one wholesale: give the whole list "
                f"(`{key.rsplit('.', 1)[0]}=[...]`) or edit the composed file."
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
    hint = f" Presets for it: {', '.join(same)}." if same else ""
    result.actions.append(
        f"{choice.slot}: {choice.node['type']} with its constructor defaults; no preset was "
        f"named.{hint}"
    )


def compose(
    base: str | Path,
    sets: Sequence[str] = (),
    overrides: Sequence[str] = (),
    *,
    adapters: bool = True,
) -> Composition:
    """Compose a child config of ``base``.

    Args:
        base: A config file or the name of a shipped config.
        sets: ``slot=choice`` pairs; a choice is a preset name or a component type.
        overrides: Further ``key.path=value`` overrides, written into the child config.
            A path below a slot of ``sets`` (``model.backbone.grid_size=0.01``) addresses
            the component that was set, wherever an adapter puts it.
        adapters: Insert the adapter the contract rules ask for, when there is exactly one.

    Returns:
        The :class:`Composition`; check ``accepted`` before writing ``text()``.

    Raises:
        ComposeError: The request names an unknown slot, preset or component.
        ConfigError: The base config itself is invalid, or an override cannot be applied.
    """
    from grasppanda.models.detectors.seed_based import (
        STAGE_REGISTRIES,
        SeedGraspDetector,
        stage_contract_problems,
    )

    base_path = resolve_config_path(base)
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
            raise ComposeError(
                f"'{choice.request}' is an adapter: {choice.node['type']} wraps another "
                "encoder and is not one itself. Set the encoder (`--set backbone=<encoder>`); "
                "compose inserts the adapter when the contracts ask for it."
            )
        _announce(result, choice)
        mine = addressed.get(choice.slot, [])
        if seed_based and choice.slot == "backbone":
            keep = {item.partition("=")[0].strip().split(".")[0] for item in mine}
            _fit_encoder(result, choice, model, keep)
        if mine:
            apply_overrides(LoadedTree(data=choice.node), mine)
            for item in mine:
                choice.comments.pop(tuple(item.partition("=")[0].strip().split(".")), None)
            result.actions.append(f"{choice.slot}: overrides as given, {' '.join(mine)}.")

    nodes = {choice.slot: choice.node for choice in result.choices}
    notes = {choice.slot: dict(choice.comments) for choice in result.choices}
    if seed_based and adapters:
        proposal = nodes.get("proposal", model["proposal"])
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

    # A dataset section a stage reads is added where the base config has none.
    dataset = ((config.resolved().get("data") or {}).get("train") or {}).get("dataset")
    added: dict[str, Any] = {}
    for choice in result.choices:
        preset = choice.preset
        for section, value in (preset.data if preset else {}).items():
            if not isinstance(dataset, Mapping) or dataset.get(section) is not None:
                continue
            if section in added:
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
    result.body, result.comments = body, comments
    _check(result)
    return result


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


def _check(result: Composition) -> None:
    """Load the composed file as a user would and record what the checks say."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems

    with tempfile.TemporaryDirectory(prefix="grasppanda-compose-") as scratch:
        path = Path(scratch) / "composed.yaml"
        path.write_text(result.text(), encoding="utf-8")
        merged = Config.load(path, validate=False)
        model = merged.to_dict().get("model") or {}
        if result.seed_based and all(
            isinstance(model.get(slot), Mapping) and "type" in model[slot]
            for slot in STAGE_REGISTRIES
        ):
            stages = {slot: model[slot] for slot in STAGE_REGISTRIES}
            try:
                result.problems = list(dict.fromkeys(stage_contract_problems(**stages)))
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
        "--no-adapter",
        dest="adapters",
        action="store_false",
        help="do not insert an encoder adapter; a refused pair then stays refused",
    )
    return parser


def recompose(path: str | Path) -> Composition:
    """Compose again what the header of a composed file says it was composed from."""
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
    args = command_parser().parse_args(shlex.split(command)[2:])
    return compose(args.base, args.sets, args.opts, adapters=args.adapters)
