"""The options of every slot of a method, with what was measured about each.

``grasppanda compose <method config> --list [<slot>]``, the per-method pages of the
documentation (``tools/make_method_pages.py``) and the lanes of the combination explorer
(``tools/make_explorer.py``) all list the options of a method from here, so that the three
cannot disagree (``tests/unit/test_method_options.py`` checks the explorer against it).

The options of a slot under a method are the cells of the measured grid
(:mod:`grasppanda.combine.grid`) that replace that one slot of that method, plus the presets
of the slot the grid has no cell for, each with the reason. Each option carries its record
from the measured file (``docs/advanced_guides/measured_combinations.json``, packaged in
the wheel as ``grasppanda/_measured/``): the verdict, the levels of
:mod:`grasppanda.combine.levels`, the class of the option at each level across the methods
it applies to, the grasps per frame and the fall of every loss term; for a refusal the
messages, their kind and the lift. An option without a record is *not probed*. Where an
option works across the methods is said as a fraction with the methods it leaves out
(:mod:`grasppanda.combine.reach`, the words of the pages and the explorer). Nothing is
probed or composed here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .levels import (
    CAVEAT,
    LEVELS,
    NO,
    NOT_PROBED,
    YES,
    CellLevels,
    OptionClass,
    cell_levels,
    option_class,
    refusal_kind,
    refusal_kinds,
)

__all__ = [
    "SLOT_ORDER",
    "Entry",
    "Measured",
    "Option",
    "SlotListing",
    "listing",
    "measured",
    "method_of",
    "method_slots",
    "methods",
    "options",
    "records_file",
    "render",
    "report",
    "scope",
    "slot_methods",
]

#: What the listing, the pages and the explorer call level 2 (``tools/docs_chips.py``
#: ``LEVEL2_NAME``; a test holds the two together).
LEVEL2_NAME = "trains like shipped"
#: The slots in the order data flows, then the training recipe.
SLOT_ORDER = (
    "image_encoder",
    "backbone",
    "proposal",
    "view",
    "assigner",
    "grouper",
    "head",
    "loss",
    "post",
    "optimizer",
    "schedule",
    "hooks",
    "augment",
    "recipe",
)
#: Where a slot is written in a config.
SLOT_KEYS = {
    "image_encoder": "model.image_encoder",
    "backbone": "model.backbone",
    "proposal": "model.proposal",
    "view": "model.view",
    "assigner": "model.assigner",
    "grouper": "model.grouper",
    "head": "model.head",
    "loss": "model.<stage>.<term>_loss",
    "post": "evaluation.variants.post",
    "optimizer": "optimizer",
    "schedule": "scheduler",
    "hooks": "hooks",
    "augment": "data.train.dataset.transforms",
    "recipe": "optimizer, scheduler, hooks, transforms",
}
#: The groups of the grid whose cells replace one slot of a method.
SINGLE_GROUPS = ("encoder", "stage", "family-slot", "post", "loss", "recipe")
#: Why an encoder adapter or wrapper among the presets of the scene encoder has no cell.
ADAPTER_REASON = (
    "an encoder adapter: compose inserts it where the contract between encoder and proposal "
    "asks for it, so it is measured inside the cells it was inserted in, not as an option"
)
WRAPPER_REASON = (
    "a wrapper around the method's encoder (point painting) that a config names on purpose; "
    "the grid has no cell for it"
)
ENCODER_RECIPE_REASON = (
    "it is written for the scene encoder of the seed-based skeleton (`model.backbone`), which "
    "{method} does not have; compose refuses it"
)
PENDING_REASON = (
    "a cell of the grid without a record yet: `python -m tools.probe_grid run --only '{cell}'`"
)


# --------------------------------------------------------------------------- records


def records_file() -> Path:
    """The measured file: packaged in a wheel, ``docs/advanced_guides/`` in a checkout."""
    package = Path(__file__).resolve().parents[1]
    packaged = package / "_measured" / "measured_combinations.json"
    if packaged.is_file():
        return packaged
    return package.parent / "docs" / "advanced_guides" / "measured_combinations.json"


class Measured:
    """The records of the measured file, by id, with the levels of every cell."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.about: dict[str, Any] = dict(payload.get("about") or {})
        self.by_id: dict[str, dict[str, Any]] = {
            record["id"]: record for record in payload.get("cells") or []
        }
        self._levels: dict[str, CellLevels] = {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> Measured:
        file = Path(path) if path is not None else records_file()
        if not file.is_file():
            return cls({})
        return cls(json.loads(file.read_text(encoding="utf-8")))

    def get(self, cell_id: str) -> dict[str, Any] | None:
        return self.by_id.get(cell_id)

    def numbers(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """The record that holds the numbers of ``record``: itself, or the cell it names."""
        return self.by_id.get(record.get("probed_as", ""), dict(record))

    def levels(self, record: Mapping[str, Any]) -> CellLevels:
        """The levels of a cell, held against the reference of its method
        (:meth:`CellLevels.gated`): a level the method as shipped does not reach either
        is undecided under that method."""
        found = self._levels.get(record["id"])
        if found is None:
            found = cell_levels(record, self.numbers(record))
            reference = self.reference(str(record.get("method", "")))
            if reference is not None and reference["id"] != record["id"]:
                found = found.gated(self.levels(reference))
            self._levels[record["id"]] = found
        return found

    def reference(self, method: str) -> dict[str, Any] | None:
        return self.by_id.get(f"reference/{method}") or self.by_id.get(f"family/{method}")


@lru_cache(maxsize=4)
def _measured(path: str) -> Measured:
    return Measured.load(path)


def measured(path: str | Path | None = None) -> Measured:
    """The records of ``path`` (the measured file by default), read once per process."""
    return _measured(str(Path(path) if path is not None else records_file()))


# --------------------------------------------------------------------------- options


@dataclass(frozen=True)
class Option:
    """One option of one slot under one method.

    Attributes:
        slot: The slot (a key of :data:`SLOT_KEYS`).
        name: What the explorer calls it: the preset or component after ``--set
            <slot>=``; for a loss, ``<presets>@<terms>`` (one cell of the grid).
        sets: The compose request; empty for an option no request expresses.
        component: The component type it puts there.
        preset: Whether it is a preset (else a component with its constructor defaults).
        cell: The id of its cell in the grid; ``""`` when the grid has none.
        reason: Why the grid has no cell for it; ``""`` when it has one.
        presets: The loss presets a loss option is made of.
    """

    slot: str
    name: str
    sets: tuple[str, ...]
    component: str
    preset: bool = True
    cell: str = ""
    reason: str = ""
    presets: tuple[str, ...] = ()

    @property
    def request(self) -> str:
        return " ".join(f"--set {item}" for item in self.sets)


def methods() -> dict[str, str]:
    """Method → its config: the five seed-based methods, then the detector families."""
    from .grid import FAMILIES, METHODS

    return {**METHODS, **FAMILIES}


@lru_cache(maxsize=1)
def _grid_options() -> tuple[tuple[str, Option], ...]:
    """``(method, option)`` of every single-slot cell of the grid, in the grid's order."""
    from .grid import cells
    from .presets import LOSS_SLOT, presets

    members = presets()
    found = []
    for cell in cells(SINGLE_GROUPS):
        if cell.group == "loss":
            names = tuple(dict.fromkeys(item.partition("=")[2] for item in cell.sets))
            name = cell.id.split("/", 2)[2]
            found.append(
                (
                    cell.method,
                    Option(LOSS_SLOT, name, cell.sets, cell.component, True, cell.id, "", names),
                )
            )
            continue
        slot = cell.slot
        known = members.get(slot, {})
        preset = cell.group == "recipe" or cell.variant in known  # a recipe is always a preset
        found.append(
            (
                cell.method,
                Option(slot, cell.variant, cell.sets, cell.component, preset, cell.id),
            )
        )
    return tuple(found)


@lru_cache(maxsize=1)
def _extra_options() -> tuple[tuple[str, Option], ...]:
    """``(method, option)`` of the presets the grid has no cell for, with the reason."""
    from grasppanda.registry import BACKBONES

    from .compose import is_adapter
    from .grid import FAMILIES, METHODS, POST_METHODS, POST_NOT_PROBED, RECIPE_ENCODER_PRESETS
    from .presets import POST_SLOT, presets
    from .recipes import recipes

    found = []
    for preset in presets().get("backbone", {}).values():
        cls = BACKBONES.get(preset.type)
        if not is_adapter(cls):
            continue
        reason = ADAPTER_REASON if getattr(cls, "compose_inserts", True) else WRAPPER_REASON
        for method in METHODS:
            sets = (f"backbone={preset.name}",)
            found.append(
                (method, Option("backbone", preset.name, sets, preset.type, True, "", reason))
            )
    for name, reason in POST_NOT_PROBED.items():
        preset = presets().get(POST_SLOT, {}).get(name)
        for method in POST_METHODS:
            component = preset.type if preset else ""
            sets = (f"{POST_SLOT}={name}",)
            found.append((method, Option(POST_SLOT, name, sets, component, True, "", reason)))
    for slot, members in recipes().items():
        for name, recipe in members.items():
            under = RECIPE_ENCODER_PRESETS.get(f"{slot}/{name}")
            if under is None:
                continue
            for method in [*METHODS, *FAMILIES]:
                if method in under:
                    continue
                reason = ENCODER_RECIPE_REASON.format(method=method)
                sets = (f"{slot}={name}",)
                found.append((method, Option(slot, name, sets, recipe.type, True, "", reason)))
    return tuple(found)


def options(method: str, slot: str | None = None) -> list[Option]:
    """The options of ``method`` (of one slot), in the order of :data:`SLOT_ORDER` and,
    within a slot, of the grid; the presets without a cell come last."""
    found = [
        option
        for owner, option in (*_grid_options(), *_extra_options())
        if owner == method and (slot is None or option.slot == slot)
    ]
    order = {name: index for index, name in enumerate(SLOT_ORDER)}
    return sorted(found, key=lambda option: order.get(option.slot, len(order)))


def method_slots(method: str) -> list[str]:
    """The slots of ``method`` that have options, in the order of :data:`SLOT_ORDER`."""
    return list(dict.fromkeys(option.slot for option in options(method)))


def slot_methods(slot: str) -> list[str]:
    """The methods that have ``slot``, in page order (the seed-based ones by year, then
    the detector families)."""
    return [method for method in methods() if slot in method_slots(method)]


def scope(slot: str) -> str:
    """The methods ``slot`` applies to, in the words of the pages: ``All 6 methods``,
    ``Seed-based · 5 methods`` or ``HGGD only`` (``tools/docs_chips.py`` prints the same
    words; ``tests/unit/test_combine_compose.py`` holds the two together)."""
    from .grid import METHODS

    under = slot_methods(slot)
    if not under:
        return "no method"
    if len(under) == len(methods()):
        return f"All {len(under)} methods"
    if set(under) == set(METHODS):
        return f"Seed-based · {len(under)} methods"
    if len(under) == 1:
        return f"{under[0]} only"
    return ", ".join(under)


def notes(method: str, slot: str) -> list[str]:
    """What a listing says below the options of a slot: components that are no option."""
    from .grid import LOSS_NOT_PROBED, METHODS, STAGE_SLOTS, needs_an_option, stage_variants

    if method not in METHODS:
        return []
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    found = []
    if slot in STAGE_SLOTS:
        registry = STAGE_REGISTRIES[slot]
        bare = [
            component
            for choice, component in stage_variants(slot)
            if choice == component
            and registry.get(component).__module__.split(".")[0] == "grasppanda"
            and needs_an_option(slot, choice, component)
        ]
        if bare:
            found.append(
                "Not an option by its type name: "
                + ", ".join(f"`{name}`" for name in bare)
                + (" needs" if len(bare) == 1 else " need")
                + " an option only a preset supplies (refused before any contract is "
                "compared); the presets are listed."
            )
    if slot == "loss":
        for name, reason in LOSS_NOT_PROBED.items():
            found.append(f"`{name}` has no cell of its own: {reason}.")
    return found


# ------------------------------------------------------------------------- the class


def _state(levels: CellLevels | None, level: str) -> str:
    return NOT_PROBED if levels is None else levels.states.get(level, NOT_PROBED)


def _best(states: Iterable[str]) -> str:
    """The state of a method with several cells for one option (a loss in several terms):
    reached when it was reached in one of them, undecided (the first reason) while one is
    undecided, not reached when it was reached in none."""
    found = list(states)
    if YES in found:
        return YES
    undecided = [state for state in found if state != NO]
    if undecided:
        return undecided[0]
    return NO if found else NOT_PROBED


@lru_cache(maxsize=1)
def _class_index() -> dict[tuple[str, str], dict[str, list[str]]]:
    """``(slot, option name or loss preset)`` → method → the cell ids of the option."""
    index: dict[tuple[str, str], dict[str, list[str]]] = {}
    for method, option in _grid_options():
        keys = [(option.slot, name) for name in option.presets] or [(option.slot, option.name)]
        for key in keys:
            index.setdefault(key, {}).setdefault(method, []).append(option.cell)
    # A preset without any cell (an adapter, a post-processor no dataset can feed) is "not
    # probed" under the methods that list it; one that has cells under other methods (a
    # recipe that names the scene encoder) is classified over those methods only.
    graded = set(index)
    for method, option in _extra_options():
        key = (option.slot, option.name)
        if key not in graded:
            index.setdefault(key, {}).setdefault(method, [])
    return index


def classes(slot: str, name: str, records: Measured | None = None) -> dict[str, OptionClass]:
    """Level → the class of an option (of a loss preset, for ``slot == "loss"``)."""
    from .grid import METHODS

    records = records or measured()
    by_method = _class_index().get((slot, name), {})
    found: dict[str, OptionClass] = {}
    previous: dict[str, str] = {}
    for level in LEVELS:
        states = {}
        for method, cells in by_method.items():
            recorded = [records.get(cell) for cell in cells]
            present = [records.levels(record) for record in recorded if record is not None]
            if present:
                states[method] = _best(_state(levels, level) for levels in present)
        found[level] = option_class(level, states, by_method, tuple(METHODS))
        if level == next(iter(LEVELS)):
            checked = found[level].decided
        else:
            checked = sum(
                previous.get(method) == YES and states.get(method) in (YES, NO)
                for method in by_method
            )
        found[level] = found[level].with_checked(checked)
        # A level that was decided only where the level before it failed says nothing of
        # its own: it was not checked where it could have been reached.
        below = [method for method, state in previous.items() if state != NO]
        if below and all(states.get(method) not in (YES, NO) for method in below):
            only = option_class(level, {m: states.get(m, NOT_PROBED) for m in below}, below, ())
            found[level] = OptionClass(level, only.name, 0, 0, len(by_method), only.undecided, 0)
        previous = states
    return found


# ------------------------------------------------------------------------- a listing


@dataclass
class Entry:
    """One option of a method's slot with what was measured about it.

    ``status`` is ``fits`` (the config check accepts it), ``does not fit`` (refused) or
    ``not probed`` (no record); the other fields are empty where they do not apply.
    :meth:`reach` and :meth:`trains_like_shipped` say where the option works across the
    methods (read on demand from ``records``).
    """

    option: Option
    shipped: bool = False
    status: str = NOT_PROBED
    record: str = ""
    verdict: str = ""
    levels: CellLevels | None = None
    classes: dict[str, OptionClass] = field(default_factory=dict)
    grasps: list[int] = field(default_factory=list)
    fall: dict[str, float | None] = field(default_factory=dict)
    judge: int = 0
    reason: str = ""
    refusal: list[dict[str, str]] = field(default_factory=list)
    kind: str = ""
    command: str = ""
    adapter: str = ""
    records: Measured | None = field(default=None, repr=False, compare=False)
    _reach: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def counted_as(self) -> str | None:
        """The name the class and the reach of the option are counted under: the option,
        or the loss preset of a loss option; ``None`` for a loss option of several presets."""
        keys = self.option.presets or (self.option.name,)
        return keys[0] if len(keys) == 1 else None

    def reach(self, level: str = "runs") -> Any:
        """Where the option reached ``level``: a :class:`grasppanda.combine.reach.Reach`
        over the methods it is a swap under (``fraction``, ``tail``); ``None`` for a loss
        option of several presets."""
        from .reach import reach

        name = self.counted_as
        if name is None:
            return None
        if level not in self._reach:
            self._reach[level] = reach(self.option.slot, name, level, self.records)
        return self._reach[level]

    def trains_like_shipped(self) -> Any:
        """Level 2, *trains like shipped*: how the option trains beside the component each
        method ships (:class:`grasppanda.combine.reach.Level2`); ``None`` as for
        :meth:`reach`."""
        from .reach import level2

        name = self.counted_as
        if name is None:
            return None
        if "level 2" not in self._reach:
            self._reach["level 2"] = level2(self.option.slot, name, self.records)
        return self._reach["level 2"]

    def to_dict(self) -> dict[str, Any]:
        option = self.option
        out: dict[str, Any] = {
            "option": option.name,
            "slot": option.slot,
            "component": option.component,
            "preset": option.preset,
            "sets": list(option.sets),
            "shipped": self.shipped,
            "status": self.status,
            "record": self.record or None,
            "verdict": self.verdict or None,
            "judge": self.judge or None,
            "levels": self.levels.to_dict() if self.levels else None,
            "classes": {level: found.to_dict() for level, found in self.classes.items()},
            # Where it works across the methods it is a swap under: the fraction, the
            # methods it leaves out and why (grasppanda.combine.reach), level by level.
            "reach": (
                {level: self.reach(level).to_dict() for level in LEVELS}
                if self.counted_as is not None
                else None
            ),
            # Level 2, trains like shipped (grasppanda.combine.reach.level2).
            "trains_like_shipped": (
                self.trains_like_shipped().to_dict() if self.counted_as is not None else None
            ),
            "grasps_per_frame": self.grasps,
            "fall": self.fall,
            "reason": self.reason,
            "command": self.command,
        }
        if option.presets:
            out["presets"] = list(option.presets)
        if self.adapter:
            out["adapter"] = self.adapter
        if self.refusal:
            out["refusal"] = self.refusal
        if self.refusal or self.kind:
            out["kind"] = self.kind or None
        return out


@dataclass
class SlotListing:
    """The options of one slot of one method."""

    method: str
    config: str
    slot: str
    shipped: str
    entries: list[Entry]
    notes: list[str] = field(default_factory=list)

    def group(self, status: str) -> list[Entry]:
        return [entry for entry in self.entries if entry.status == status]

    @property
    def shipped_option(self) -> str:
        """The option that composes to the method as shipped (the same config), or ``""``."""
        return next((entry.option.name for entry in self.entries if entry.shipped), "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config,
            "slot": self.slot,
            "scope": scope(self.slot),
            "written_as": SLOT_KEYS.get(self.slot, self.slot),
            "shipped": self.shipped,
            "shipped_option": self.shipped_option or None,
            "counts": {
                status: len(self.group(status)) for status in ("fits", "does not fit", NOT_PROBED)
            },
            "options": [entry.to_dict() for entry in self.entries],
            "notes": self.notes,
        }


def _first_sentence(message: str) -> str:
    match = re.match(r"(.+?[.;])(?:\s|$)", message)
    return (match.group(1) if match else message).rstrip(";").rstrip(".") + "."


def reason_text(record: Mapping[str, Any]) -> str:
    """Why a cell did not reach a level, in one line: the refusal, the failure or what was
    unstable."""
    verdict = record.get("verdict")
    if verdict == "refused":
        return " ".join(_first_sentence(message) for message in record.get("refusal") or [])
    if verdict == "fails":
        failure = record.get("failure") or {}
        where = failure.get("phase", "?")
        if failure.get("component"):
            where += f", in the {failure['component']}"
        message = _first_sentence(failure.get("message", ""))
        return f"at {where}: {failure.get('error', '')}: {message}"
    if verdict == "runs-unstable":
        return str(record.get("unstable", ""))
    return ""


def _slug(text: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", str(text).lower()))


def output_name(method: str, sets: Sequence[str]) -> str:
    """The file a listing's command writes; ``explorer_logic.js`` names it the same way."""
    parts = [_slug(item.split("=", 1)[-1]) for item in sets]
    name = "_".join([_slug(method), *parts])
    if len(name) > 72:
        name = f"{_slug(method)}_{len(sets)}-swaps"
    return name + ".yaml"


def compose_command(config: str, method: str, sets: Sequence[str]) -> str:
    request = " ".join(f"--set {item}" for item in sets)
    return f"grasppanda compose {config} {request} -o {output_name(method, sets)}"


def _shipped_component(method: str, slot: str) -> str:
    """What the method's config has in a slot, as a short text."""
    from grasppanda.config import Config

    config = Config.load(methods()[method]).resolved()
    model = config.get("model") or {}
    if slot in ("loss",):
        return "the losses of the method's stages"
    if slot == "post":
        names = ", ".join(sorted((config.get("evaluation") or {}).get("variants") or {}))
        return f"no variant `post` (the config has {names})"
    if slot == "optimizer":
        return str((config.get("optimizer") or {}).get("type") or "")
    if slot == "schedule":
        return str((config.get("scheduler") or {}).get("type") or "none")
    if slot == "hooks":
        names = [str(node.get("type")) for node in config.get("hooks") or []]
        return ", ".join(names) or "none"
    if slot in ("augment", "recipe"):
        return "the method's own"
    node = model.get(slot)
    if isinstance(node, Mapping):
        return str(node.get("type") or "")
    if slot == "image_encoder" and isinstance(model.get("encoder"), Mapping):
        return "the reference encoder of the detector (built in)"
    return ""


def entry(method: str, option: Option, records: Measured, shipped_hash: str | None = None) -> Entry:
    """One option with its record, its levels and its class."""
    config = methods()[method]
    found = Entry(option, records=records)
    if option.sets:
        found.command = compose_command(config, method, option.sets)
    keys = option.presets or (option.name,)
    if len(keys) == 1:
        found.classes = classes(option.slot, keys[0], records)
    record = records.get(option.cell) if option.cell else None
    if record is None:
        found.reason = option.reason or (
            PENDING_REASON.format(cell=option.cell) if option.cell else ""
        )
        if option.reason and option.reason == ENCODER_RECIPE_REASON.format(method=method):
            # names a part the method does not have: compose refuses it (structural)
            found.status, found.kind = "does not fit", "structural"
        return found
    numbers = records.numbers(record)
    found.record = record["id"]
    found.verdict = record["verdict"]
    found.levels = records.levels(record)
    found.adapter = record.get("adapter", "")
    found.judge = int(numbers.get("judge", 1) or 1) if numbers.get("probe") else 0
    found.shipped = bool(shipped_hash) and record.get("config_hash") == shipped_hash
    found.grasps = list(numbers.get("grasps_per_frame") or [])
    trend = numbers.get("trend") or {}
    found.fall = {
        term.removeprefix("loss/"): values[2]
        for term, values in trend.items()
        if term.startswith("loss/")
    }
    found.reason = reason_text(record)
    if record["verdict"] == "refused":
        found.status = "does not fit"
        found.refusal = refusal_kinds(record)
        found.kind = refusal_kind(record)
    else:
        found.status = "fits"
    return found


def slot_listing(method: str, slot: str, records: Measured | None = None) -> SlotListing:
    records = records or measured()
    reference = records.reference(method) or {}
    entries = [
        entry(method, option, records, reference.get("config_hash"))
        for option in options(method, slot)
    ]
    entries.sort(key=lambda found: not found.shipped)
    return SlotListing(
        method,
        methods()[method],
        slot,
        _shipped_component(method, slot),
        entries,
        notes(method, slot),
    )


def listing(
    method: str, slot: str | None = None, records: Measured | None = None
) -> list[SlotListing]:
    """The options of every slot of ``method`` (or of one slot), with what was measured."""
    slots = method_slots(method)
    if slot is not None:
        if slot not in slots:
            raise KeyError(f"{method} has no slot '{slot}'. Its slots: {', '.join(slots)}.")
        slots = [slot]
    return [slot_listing(method, name, records) for name in slots]


# ------------------------------------------------------------------- the method of a config


def method_of(config: str | Path) -> tuple[str, str]:
    """``(method, note)`` for a config: one of the method configs of the grid, or a config
    composed from one (the note then says so). :class:`KeyError` otherwise."""
    from grasppanda.config import ConfigError, resolve_config_path

    from .compose import ComposeError, _request

    configs = methods()

    def path(name: str | Path) -> Path | None:
        try:
            return resolve_config_path(name).resolve()
        except ConfigError:
            return None

    wanted = path(config)
    if wanted is None:
        raise KeyError(f"'{config}' is neither a file nor a shipped config name")
    for method, name in configs.items():
        if path(name) == wanted:
            return method, ""
    try:
        request = _request(wanted)
    except (ComposeError, OSError, UnicodeDecodeError):
        request = None
    if request is not None and request.base:
        base = path(request.base)
        for method, name in configs.items():
            if path(name) == base:
                sets = " ".join(f"--set {item}" for item in request.sets)
                return method, (
                    f"{config} was composed from {name}"
                    + (f" with {sets}" if sets else "")
                    + f"; the options below were measured on {name} with one slot replaced."
                )
    raise KeyError(
        f"the options of a slot were measured per method config; {config} is none of them "
        f"and was not composed from one. Method configs: {', '.join(configs.values())}."
    )


# ---------------------------------------------------------------------------- terminal

#: Marks of a level in a terminal listing.
MARKS = {YES: "✓", NO: "✗"}


def _mark(levels: CellLevels | None, level: str) -> str:
    if levels is None:
        return "·"
    return MARKS.get(levels.states.get(level, ""), "?")


def _fraction(entry: Entry) -> str:
    """Where the option ran, as a fraction of the methods it is a swap under."""
    found = entry.reach("runs")
    return "-" if found is None else found.fraction


def _level2(entry: Entry) -> str:
    found = entry.trains_like_shipped()
    return "-" if found is None else found.fraction


def _left_out(entry: Entry) -> str:
    """The methods the runs fraction leaves out and why; ``""`` when the option ran under
    every method it is a swap under (the JSON also names those that did not train like
    shipped)."""
    found = entry.reach("runs")
    if found is None or found.tone == "ok" or not found.measured or not found.swaps:
        return ""
    tail = "; ".join(part for part in found.tail.split("; ") if not part.startswith("own in"))
    return f"runs {found.fraction}: {tail}" if tail else ""


def _fall(fall: Mapping[str, float | None]) -> str:
    values = [value for value in fall.values() if value is not None]
    if not values:
        return "-"
    low, high = min(values), max(values)
    return f"{low * 100:.0f}..{high * 100:.0f} %"


def _shown(option: Option) -> str:
    """An option as a terminal listing names it: a loss by the terms it replaces."""
    if option.slot == "loss" and option.sets:
        return " ".join(option.sets)
    return option.name


def report(method: str, slot: str | None = None, note: str = "") -> dict[str, Any]:
    """What ``grasppanda compose <config> --list [<slot>] --json`` prints."""
    return {
        "method": method,
        "config": methods()[method],
        "note": note,
        "caveat": CAVEAT,
        "levels": dict(LEVELS),
        "slots": [found.to_dict() for found in listing(method, slot)],
    }


def render(listings: Sequence[SlotListing], *, color: bool = False, note: str = "") -> str:
    """A listing as text for a terminal."""
    from grasppanda.utils.console import style, table

    if not listings:
        return ""
    lines = []
    first = listings[0]
    lines.append(
        style(f"{first.method}: options of every slot", "bold", enabled=color)
        if len(listings) > 1
        else style(
            f"{first.method} · {first.slot}  ({SLOT_KEYS.get(first.slot, first.slot)})",
            "bold",
            enabled=color,
        )
    )
    lines.append(f"measured on {first.config} ({CAVEAT})")
    if note:
        lines.append(note)
    lines.append(
        "levels C R L P: composes, runs, learns, predicts under this method "
        "(✓ reached, ✗ not, ? undecided, · not probed)"
    )
    lines.append(
        "runs under: of the methods the option is a swap under (not shipped there), those it "
        "ran under"
    )
    lines.append(
        f"{LEVEL2_NAME}: trained beside the shipped component; on a par where its losses fall alike"
    )
    for found in listings:
        lines.append("")
        head = f"{found.slot}  --set {found.slot if found.slot != 'loss' else '<stage>.<term>_loss'}=..."
        shipped = found.shipped + (
            f" (option {found.shipped_option})" if found.shipped_option else ""
        )
        lines.append(style(head, "bold", "cyan", enabled=color) + f"   shipped: {shipped}")
        fits = found.group("fits")
        if fits:
            rows = []
            for item in fits:
                name = _shown(item.option) + (" (shipped)" if item.shipped else "")
                rows.append(
                    [
                        name,
                        item.verdict + (f" in {item.adapter}" if item.adapter else ""),
                        " ".join(_mark(item.levels, level) for level in LEVELS),
                        ", ".join(map(str, item.grasps)) or "-",
                        _fall(item.fall) if item.judge >= 2 else "not checked",
                        _fraction(item),
                        _level2(item),
                    ]
                )
            lines.append(f"  fit: {len(fits)} (the config check accepts them)")
            lines.append(
                table(
                    [
                        "option",
                        "verdict",
                        "C R L P",
                        "grasps/frame",
                        "term fall",
                        "runs under",
                        LEVEL2_NAME,
                    ],
                    rows,
                    enabled=color,
                )
            )
            for item in fits:
                parts = (
                    []
                    if item.levels is None
                    else [
                        f"{level}: {', '.join(items)}" for level, items in item.levels.why.items()
                    ]
                )
                if item.verdict in ("fails", "runs-unstable") and item.reason:
                    parts.insert(0, item.reason)
                left_out = _left_out(item)
                if left_out:
                    parts.append(left_out)
                if parts:
                    lines.append(f"    {_shown(item.option)}: " + "; ".join(parts))
        refused = found.group("does not fit")
        if refused:
            lines.append(f"  do not fit: {len(refused)} (refused by compose)")
            for item in refused:
                kind = f" [{item.kind}]" if item.kind else ""
                lines.append(f"    {_shown(item.option)}{kind}: {item.reason}")
                for part in item.refusal:
                    if part.get("lift"):
                        lines.append(f"      lift: {part['lift']}")
        missing = found.group(NOT_PROBED)
        if missing:
            lines.append(f"  not probed: {len(missing)}")
            for item in missing:
                lines.append(f"    {_shown(item.option)}: {item.reason}")
        for text in found.notes:
            lines.append(f"  {text}")
    lines.append("")
    lines.append(
        f"compose one: grasppanda compose {first.config} --set <slot>=<option> -o <file>; "
        "the JSON (--json) holds every message, level and fraction"
    )
    return "\n".join(lines)
