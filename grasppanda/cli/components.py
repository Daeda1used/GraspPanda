"""``grasppanda components``: the catalogue of what a config can name.

Everything shown is read from the registries and from the classes registered in them:
the contract attributes the config checker compares (``output``, ``requires``,
``support``, ``out_axes``, ``in_axes``, …), what each class says about itself (``status``,
``source``, ``extras``, the first sentence of its docstring), and the shipped configs
that use it. Nothing is maintained by hand, so the catalogue cannot disagree with the
code; ``docs/components.md`` is the same data rendered by ``tools/make_component_docs.py``.

    grasppanda components                    # every slot, one table each
    grasppanda components --slot backbones   # one slot
    grasppanda components PTv3               # one component: options, configs, what it fits
    grasppanda components --json             # the same data for scripts
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from grasppanda.config import Config, available_configs
from grasppanda.registry import INJECTED, Registry, all_registries, component_parameters
from grasppanda.utils.console import style, table

__all__ = [
    "CONTRACT_FIELDS",
    "FIELD_HELP",
    "SLOTS",
    "Entry",
    "collect",
    "compatibility",
    "parameters",
    "register",
    "render_value",
    "resolve_slot",
]

#: Registry -> (where its components go in a config, what they do). The order is the
#: order of presentation: a detector, its six stages in the order data flows, the rest.
SLOTS: dict[str, tuple[str, str]] = {
    "DETECTORS": ("model", "A complete method. Seed-based detectors fill the six slots below."),
    "BACKBONES": (
        "model.backbone",
        "Scene encoder: points to features, on every point or on a subsample.",
    ),
    "IMAGE_ENCODERS": (
        "model.image_encoder",
        "Image encoder: an RGB or RGB-D image to feature maps (HGGD, point painting).",
    ),
    "PROPOSALS": ("model.proposal", "Scene features to seeds."),
    "SAMPLERS": (
        "model.proposal.sampler",
        "Part of a proposal: which of the candidate points become seeds.",
    ),
    "VIEWS": ("model.view", "One approach direction per seed."),
    "ASSIGNERS": (
        "model.assigner",
        "Training only: seeds and chosen views to matched grasp labels.",
    ),
    "GROUPERS": ("model.grouper", "A local crop per seed, aligned with its approach."),
    "HEADS": (
        "model.head",
        "Head bundle: prediction layers, target encoder, loss and decoder as one unit.",
    ),
    "LOSSES": ("<stage>.<name>_loss", "Primitive loss inside a stage."),
    "POSTPROCESSORS": (
        "evaluation.variants.<name>[i]",
        "Predictions to predictions at inference time.",
    ),
    "DATASETS": ("data.<split>.dataset", "Frames with optional supervision."),
    "TRANSFORMS": (
        "data.<split>.dataset.preprocess[i] / .transforms[i]",
        "Preprocessing and augmentation steps of a dataset.",
    ),
    "EVALUATORS": ("evaluation.evaluator", "Benchmark protocol: prediction dumps to scores."),
    "HOOKS": ("hooks[i]", "Extra behaviour around the training loop."),
    "OPTIMIZERS": ("optimizer", "Every optimizer of torch.optim, by its class name."),
    "SCHEDULERS": ("scheduler", "Every scheduler of torch.optim.lr_scheduler, by its class name."),
}

#: Registry -> class attributes that make up the contract of its components.
CONTRACT_FIELDS: dict[str, tuple[str, ...]] = {
    "DETECTORS": ("required_inputs", "needs_supervision"),
    "BACKBONES": ("output",),
    "PROPOSALS": ("requires", "seed_scores", "needs_supervision"),
    "SAMPLERS": ("needs",),
    "VIEWS": ("seed_scores", "label_formats", "needs_matched", "applies_seed_weights"),
    "ASSIGNERS": ("label_format", "optional_outputs", "needs_supervision"),
    "GROUPERS": ("support", "out_axes"),
    "HEADS": (
        "in_axes",
        "label_format",
        "score_kind",
        "applies_seed_weights",
        "needs_supervision",
    ),
    "LOSSES": ("accepts", "prediction"),
}

#: Contract field -> what it states, in one sentence.
FIELD_HELP: dict[str, str] = {
    "required_inputs": "Observation fields the detector reads; the dataset must provide them.",
    "output": (
        "Where the returned features live: `dense` = on every input point, `subsampled` = on "
        "a furthest-point subsample that records its parent. A proposal accepts one kind."
    ),
    "requires": "The kind of scene features the proposal works on (`dense`, `subsampled`, `any`).",
    "seed_scores": (
        "Per-seed scores the stage predicts, supervises and stores on the seeds. A score has "
        "one owner per detector."
    ),
    "needs": (
        "What the sampler reads besides coordinates: `scores` (one value per candidate) and/or "
        "`feats` (the scene features). The proposal that holds it must provide them."
    ),
    "label_formats": "Kinds of matched labels the view stage can compute its targets from.",
    "needs_matched": (
        "Optional fields of the matched labels the stage cannot work without; the assigner "
        "must return them."
    ),
    "applies_seed_weights": (
        "Whether the losses honour a per-seed weight from the assigner. An assigner that "
        "returns weights is refused next to a stage that would ignore them."
    ),
    "label_format": (
        "Kind of matched labels: `dense` (angle × depth grids per view) or `compact` (one "
        "grasp per point and view). Assigner and head bundle must agree."
    ),
    "optional_outputs": (
        "Optional fields of the matched labels the assigner can return, each with the option "
        "that switches it on."
    ),
    "support": (
        "What a crop gathers: `seeds` (the seeds and their learned features) or `scene` (raw "
        "scene geometry). Not the same crop at another resolution."
    ),
    "out_axes": "Named axes of the crop features. Must equal the head bundle's `in_axes`.",
    "in_axes": "Named axes of the crop features the head bundle expects.",
    "score_kind": "Meaning of the decoded grasp score. Scores of different kinds are not comparable.",
    "accepts": (
        "Kinds of loss term the primitive fits: `class` (logits against class indices), `unit` "
        "(a target quality in [0, 1]), `value` (an unbounded target) or `bins` (logits against "
        "a fractional bin position). A stage names the kind of each of its loss options, and "
        "the config check refuses any other pairing."
    ),
    "prediction": (
        "How the primitive reads the output of a `unit` term: as the `value` itself or as a "
        "`logit`. A stage hands a logit on through its sigmoid."
    ),
    "needs_supervision": (
        "Fields of the training supervision the component reads (`objects` are the dense "
        "label grids), each with the option that makes it read the field. The training "
        "dataset must serve every one; the config check compares."
    ),
}

#: Pairs of slots whose contracts are compared with each other, per registry.
_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "BACKBONES": ("PROPOSALS",),
    "PROPOSALS": ("BACKBONES", "VIEWS"),
    "VIEWS": ("PROPOSALS", "ASSIGNERS", "HEADS"),
    "ASSIGNERS": ("VIEWS", "HEADS"),
    "GROUPERS": ("HEADS",),
    "HEADS": ("GROUPERS", "ASSIGNERS", "VIEWS"),
}


@dataclass(frozen=True)
class Entry:
    """One registered component as the catalogue describes it.

    Attributes:
        registry: Name of the registry that owns it (``BACKBONES``).
        name: The name a config uses (``type: <name>``).
        module: Where it is defined.
        summary: First sentence of the class docstring.
        source: The paper, method or library it implements; empty when undeclared.
        status: What has been proven about it; empty when undeclared.
        contract: Contract field -> value, in the order of :data:`CONTRACT_FIELDS`.
        extras: Optional dependencies it needs.
        used_by: Shipped configs that use it.
        in_tree: Whether it is part of the ``grasppanda`` package.
    """

    registry: str
    name: str
    module: str
    summary: str
    source: str
    status: str
    contract: dict[str, Any]
    extras: tuple[str, ...]
    used_by: tuple[str, ...]
    in_tree: bool


# ------------------------------------------------------------------------ collecting


def _summary(obj: Any) -> str:
    """First sentence of the docstring, as plain text."""
    doc = inspect.getdoc(obj) or ""
    paragraph = " ".join(doc.split("\n\n")[0].split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", paragraph)
    sentence = match.group(1) if match else paragraph
    sentence = re.sub(r":\w+:`~?(?:[\w.]*\.)?(\w+)`", r"\1", sentence)  # :class:`a.b.C` -> C
    return sentence.replace("``", "`").replace("**", "")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@lru_cache(maxsize=1)
def _usage() -> dict[tuple[str, str], tuple[str, ...]]:
    """(registry, component) -> shipped configs whose resolved tree names it."""
    from grasppanda.config.validate import iter_components

    found: dict[tuple[str, str], set[str]] = {}
    for name in available_configs():
        try:
            tree = Config.load(name, validate=False).to_dict()
        except Exception:  # a broken shipped config is reported by the tests, not here
            continue
        for _, registry, component in iter_components(tree):
            found.setdefault((registry.name, component), set()).add(name)
    return {key: tuple(sorted(configs)) for key, configs in found.items()}


def _entry(registry: Registry, name: str) -> Entry:
    obj = registry.get(name)
    own = vars(obj) if inspect.isclass(obj) else {}
    contract = {
        field: _jsonable(getattr(obj, field))
        for field in CONTRACT_FIELDS.get(registry.name, ())
        if hasattr(obj, field)
    }
    # A registry filled from a library can also hold components of this package.
    third_party = registry.third_party and obj.__module__.split(".")[0] != "grasppanda"
    return Entry(
        registry=registry.name,
        name=name,
        module=obj.__module__,
        summary="" if third_party else _summary(obj),
        # Declared on the class itself: a subclass does not inherit a citation or a proof.
        source=obj.__module__.split(".")[0] if third_party else str(own.get("source", "")),
        status="" if third_party else str(own.get("status", "")),
        contract=contract,
        extras=tuple(getattr(obj, "extras", ()) or ()),
        used_by=_usage().get((registry.name, name), ()),
        in_tree=obj.__module__.split(".")[0] == "grasppanda",
    )


def resolve_slot(text: str) -> str | None:
    """Registry name for what a user typed: any case, singular or plural, or the slot."""
    wanted = text.strip().lower()
    for name, (where, _) in SLOTS.items():
        plural = name.lower()
        singular = plural[:-1]
        if wanted in (plural, singular, where) or where.endswith(f".{wanted}"):
            return name
    return None


def collect(slots: Iterable[str] | None = None, *, in_tree_only: bool = False) -> list[Entry]:
    """Describe every registered component.

    Args:
        slots: Registry names to restrict to; ``None`` is all of them.
        in_tree_only: Leave out components defined outside the ``grasppanda`` package
            (plugins, ``custom_imports``, third-party optimizers and schedulers).
    """
    registries = all_registries()
    order = [name for name in SLOTS if name in registries] + sorted(set(registries) - set(SLOTS))
    if slots is not None:
        order = [name for name in order if name in set(slots)]
    entries = []
    for registry_name in order:
        registry = registries[registry_name]
        for name in registry.names():
            entry = _entry(registry, name)
            if entry.in_tree or not in_tree_only:
                entries.append(entry)
    return entries


def parameters(entry: Entry) -> list[tuple[str, str, str]]:
    """``(name, type, default)`` of every constructor option of a component."""
    obj = all_registries()[entry.registry].get(entry.name)
    params, _ = component_parameters(obj)
    provided = all_registries()[entry.registry].provided
    rows = []
    for name, param in params.items():
        if name in provided:
            continue
        annotation = param.annotation
        kind = "" if annotation is param.empty else str(annotation).replace("typing.", "")
        if param.default is param.empty:
            default = "required"
        elif param.default is INJECTED:
            default = "injected by the parent"
        else:
            default = _yaml(param.default)
        rows.append((name, kind, default))
    return rows


def _yaml(value: Any) -> str:
    """A default as it would be written in a config (flow style)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}: {_yaml(item)}" for key, item in value.items()) + "}"
    return str(value)


def compatibility(entry: Entry) -> dict[str, dict[str, list[str]]]:
    """How a stage of the seed-based skeleton combines with the stages next to it.

    Returns ``{neighbouring registry: {component: problems}}`` where ``problems`` is
    what the contract rules say about the pair with default options (empty = they fit).
    Empty for components that are not stages of the skeleton.
    """
    if entry.registry not in _NEIGHBOURS:
        return {}
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems

    slot_of = {registry.name: slot for slot, registry in STAGE_REGISTRIES.items()}
    registries = all_registries()
    own = registries[entry.registry].get(entry.name)
    verdicts: dict[str, dict[str, list[str]]] = {}
    for other_registry in _NEIGHBOURS[entry.registry]:
        verdicts[other_registry] = {
            other: stage_contract_problems(
                **{
                    slot_of[entry.registry]: own,
                    slot_of[other_registry]: registries[other_registry].get(other),
                }
            )
            for other in registries[other_registry].names()
        }
    return verdicts


# ------------------------------------------------------------------------- rendering


def render_value(value: Any) -> str:
    """A contract value or default as short text (the YAML spelling where one exists)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        parts = []
        for key, switch in value.items():
            if switch is None:
                parts.append(f"{key} (always)")
            else:
                option, wanted = switch
                spelled = {True: "true", False: "false"}.get(wanted, wanted)
                parts.append(f"{key} (with {option}: {spelled})")
        return ", ".join(parts) if parts else "—"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        return "(" + ", ".join(str(item) for item in items) + ")" if items else "—"
    return str(value)


def _where_used(entry: Entry) -> str:
    """Shipped configs of an entry, grouped by method folder: ``gsnet ×1, hggd ×2``."""
    folders: dict[str, int] = {}
    for config in entry.used_by:
        folder = config.split("/")[0]
        folders[folder] = folders.get(folder, 0) + 1
    return ", ".join(f"{folder} ×{count}" for folder, count in folders.items()) or "—"


def _print_slot(registry: str, entries: list[Entry], color: bool, legend: bool = False) -> None:
    where, role = SLOTS.get(registry, ("", "Registered by a plugin."))
    head = style(registry, "bold", "cyan", enabled=color)
    print(f"{head}  {style(where, 'bold', enabled=color)}  {role}")
    if not entries:
        print("  (no registered components yet)\n")
        return
    library = [entry for entry in entries if not entry.in_tree and not entry.summary]
    if library:  # torch classes: names only
        print("  " + ", ".join(entry.name for entry in library) + "\n")
        entries = [entry for entry in entries if entry not in library]
        if not entries:
            return
    fields = CONTRACT_FIELDS.get(registry, ())
    rows = [
        [
            entry.name,
            *(render_value(entry.contract[f]) if f in entry.contract else "—" for f in fields),
            entry.status or "—",
            entry.source or ("—" if entry.in_tree else f"plugin: {entry.module}"),
            _where_used(entry),
        ]
        for entry in entries
    ]
    print(table(["name", *fields, "status", "source", "shipped configs"], rows, enabled=color))
    if legend:
        for name in fields:
            text = FIELD_HELP[name].replace("`", "")
            print(style(f"  {name}: {text}", "gray", enabled=color))
    print()


def _preset_lines(entry: Entry) -> list[str]:
    """The presets that configure a component for `grasppanda compose`, as lines of the
    detail view: a title, then one indented line per preset. Empty for a component that
    is not a stage of the seed-based skeleton."""
    from grasppanda.combine.presets import presets
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    slot = next((s for s, reg in STAGE_REGISTRIES.items() if reg.name == entry.registry), None)
    if slot is None:
        return []
    mine = [preset for preset in presets().get(slot, {}).values() if preset.type == entry.name]
    lines = ["presets  (grasppanda compose <config> --set <slot>=<preset>)"]
    lines += [
        f"  {slot}={preset.name}    grasppanda compose --show {slot}={preset.name}"
        for preset in mine
    ]
    if not mine:
        lines.append(
            f"  (none)    --set {slot}={entry.name} composes it with the defaults listed above"
        )
    return lines


def _print_detail(entry: Entry, color: bool) -> None:
    where, _ = SLOTS.get(entry.registry, ("", ""))
    print(f"{style(entry.name, 'bold', 'green', enabled=color)}  {entry.registry}  ({where})")
    if entry.summary:
        print(f"  {entry.summary}")
    rows = [
        ["source", entry.source or "—"],
        ["status", entry.status or "—"],
        ["module", entry.module],
        ["needs", ", ".join(entry.extras) or "nothing beyond the core install"],
    ]
    rows += [[field, render_value(value)] for field, value in entry.contract.items()]
    width = max(len(key) for key, _ in rows)
    print()
    for key, value in rows:
        print(f"  {style(key.ljust(width), 'gray', enabled=color)}  {value}")
    options = parameters(entry)
    if options:
        print()
        print(table(["option", "type", "default"], [list(row) for row in options], enabled=color))
    print()
    print(style("shipped configs", "bold", enabled=color))
    for config in entry.used_by or ("(none)",):
        print(f"  {config}")
    for line in _preset_lines(entry):
        print(line if line.startswith(" ") else "\n" + style(line, "bold", enabled=color))
    for registry, verdicts in compatibility(entry).items():
        print()
        print(style(f"with {registry.lower()}", "bold", enabled=color))
        for other, problems in verdicts.items():
            if problems:
                print(f"  {style('✗', 'red', enabled=color)} {other}: {problems[0]}")
            else:
                print(f"  {style('✓', 'green', enabled=color)} {other}")


# ----------------------------------------------------------------------- the command


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "components",
        help="list what a config can name: components, contracts, status, source",
        description=(
            "The catalogue of registered components, generated from the registries: one "
            "table per slot with each component's contract, status, the paper or method it "
            "implements and the shipped configs that use it. Name a component to see its "
            "options and what it can be combined with."
        ),
    )
    parser.add_argument("name", nargs="?", help="describe one component in detail")
    parser.add_argument(
        "--slot",
        action="append",
        metavar="SLOT",
        help="restrict to a slot, e.g. backbones, head or model.grouper (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--config",
        metavar="CONFIG",
        help="also load the out-of-tree components this config imports (custom_imports)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    if args.config:  # importing is a side effect of validation
        Config.load(args.config)
    slots = None
    if args.slot:
        slots = []
        for text in args.slot:
            registry = resolve_slot(text)
            if registry is None:
                known = ", ".join(name.lower() for name in SLOTS)
                print(f"unknown slot '{text}'. Slots: {known}", file=sys.stderr)
                return 2
            slots.append(registry)
    entries = collect(slots)
    if args.name:
        matches = [entry for entry in entries if entry.name == args.name]
        if not matches:
            close = difflib.get_close_matches(args.name, [e.name for e in entries], n=3, cutoff=0.5)
            hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
            print(
                f"'{args.name}' is not a registered component.{hint} "
                "Run `grasppanda components` to list them.",
                file=sys.stderr,
            )
            return 2
        entries = matches
    if args.json:
        print(json.dumps([asdict(entry) for entry in entries], indent=2, ensure_ascii=False))
        return 0
    color = sys.stdout.isatty()
    if args.name:
        for index, entry in enumerate(entries):
            if index:
                print()
            _print_detail(entry, color)
        return 0
    shown = list(dict.fromkeys(entry.registry for entry in entries))
    # A slot asked for by name is shown even when nothing is registered in it.
    for registry in [name for name in SLOTS if name in (slots or shown)] + [
        name for name in shown if name not in SLOTS
    ]:
        members = [entry for entry in entries if entry.registry == registry]
        _print_slot(registry, members, color, legend=bool(slots))
    print(
        "Details and compatibility of one component: grasppanda components <name>. "
        "What can replace what: docs/advanced_guides/swapping.md."
    )
    return 0
