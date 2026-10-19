"""Generate what follows from the presets and from the measured grid.

* ``docs/advanced_guides/measured_combinations.md`` (*Does it run?*) is written whole from
  ``docs/advanced_guides/measured_combinations.json``, the compact record of
  ``tools/probe_grid.py``: the answer as a slot × method matrix at each level, how a cell
  is judged, and per slot every option with where it reached each level and who is left
  out and why, in the fractions and tails of :mod:`grasppanda.combine.reach` (the words of
  the explorer and of ``grasppanda compose --list``); the per-cell numbers stay in the
  records and the explorer. Nothing on the page is inferred: a method without a record is
  named as undecided.
* ``docs/evidence/index.md`` (*How we measure*): the answer per slot (the options of
  ``docs_chips.option_count``, how many run and train like shipped everywhere), the slot ×
  method scope (folded), the words of the evidence pages and what was measured on, from the
  same records; the card of the encoders counts the table of ``docs/model_zoo_backbones.md``.
* ``docs/slots/index.md``: the first page of the slot documentation.
* ``configs/experimental/*.yaml`` are written by ``grasppanda compose`` from the command
  in their own header, so their numbers are those of the presets.

Run it after changing a preset, ``grasppanda compose`` or the measured file::

    python -m tools.make_combination_docs           # rewrite
    python -m tools.make_combination_docs --check   # exit 1 and name what is stale

``tests/unit/test_combine_grid.py`` runs the check, so a stale page or config fails CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from grasppanda.combine import options as option_lists
from grasppanda.combine.compose import recompose
from grasppanda.combine.grid import (
    FAMILIES,
    LOSS_NOT_PROBED,
    METHODS,
    POST_NOT_PROBED,
    STAGE_SLOTS,
    Cell,
    cells,
    loss_terms,
    needs_an_option,
    stage_variants,
)
from grasppanda.combine.levels import (
    CAVEAT,
    CLASS_WORDS,
    LEVELS,
    NO,
    UNDECIDED,
    YES,
    CellLevels,
    OptionClass,
    refusal_kind,
)
from grasppanda.combine.presets import LOSS_SLOT, POST_SLOT, presets
from grasppanda.combine.probe import JUDGE, SCHEDULES, TREND_FLAGS, VERDICTS
from grasppanda.combine.recipes import recipes

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
PAGE = REPO / "docs" / "advanced_guides" / "measured_combinations.md"
EXPERIMENTAL = REPO / "configs" / "experimental"
SLOTS_INDEX = REPO / "docs" / "slots" / "index.md"

#: The class words of a stage decided under all five seed-based methods (levels.py),
#: then the columns of a relative class and of an undecided one.
CLASSES = tuple(CLASS_WORDS)
RELATIVE = "N of M probed"
UNDECIDED_COLUMN = "undecided"
#: Verdicts of a probe that ran and has numbers: built, trained and predicted.
WORKS = ("runs", "runs-empty", "runs-unstable")
VERDICT_ORDER = ("runs", "runs-empty", "runs-unstable", "refused", "fails")
#: The sentence every place that shows a class carries next to it.
CAVEAT_SENTENCE = f"*Class: {CAVEAT}.*"
SLOT_TITLES = {
    "backbone": "Scene encoders",
    "proposal": "Seed proposals",
    "view": "View stages",
    "assigner": "Assigners",
    "grouper": "Groupers",
    "head": "Head bundles",
}
GROUP_TITLES = {
    "reference": "the methods as shipped",
    "encoder": "one scene encoder replaced",
    "stage": "one other stage replaced",
    "stage-set": "sets of stages",
    "family": "other detector families, as shipped",
    "family-slot": "one component of another detector family replaced",
    "post": "one post-processing step as the variant `post`",
    "loss": "one loss of one stage replaced",
    "recipe": "one recipe preset applied (optimizer, schedule, hooks, augmentation)",
    "cross": "options of several slots replaced together",
}


@lru_cache(maxsize=1)
def _grid() -> tuple[Cell, ...]:
    return tuple(cells())


def _runner() -> Any:
    """``tools/probe_grid.py``: the fields of a record are defined there, once."""
    name = "probe_grid"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def grid(group: str | None = None) -> list[Cell]:
    """The cells of the grid (of one group), enumerated once per process."""
    return [cell for cell in _grid() if group is None or cell.group == group]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return lines


def _indent(lines: list[str]) -> list[str]:
    return [f"    {line}".rstrip() for line in lines]


def _first_sentence(message: str) -> str:
    match = re.match(r"(.+?[.;])(?:\s|$)", message)
    return (match.group(1) if match else message).rstrip(";").rstrip(".") + "."


def measured() -> dict[str, Any]:
    return json.loads(DATA.read_text(encoding="utf-8"))


class Records:
    """The cells of the measured file, by id; a shared probe is looked up at its owner."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.by_id = {record["id"]: record for record in records}
        self.measured = option_lists.Measured({"cells": list(self.by_id.values())})

    def get(self, cell_id: str) -> dict[str, Any] | None:
        return self.by_id.get(cell_id)

    def probed(self, record: dict[str, Any]) -> dict[str, Any]:
        """The record that holds the numbers of ``record``: itself, or the cell it names."""
        return self.by_id.get(record.get("probed_as", ""), record)

    def levels(self, record: dict[str, Any]) -> CellLevels:
        """The four levels of a cell (grasppanda.combine.levels)."""
        return self.measured.levels(record)

    def classes(self, slot: str, name: str) -> dict[str, OptionClass]:
        """Level → the class of an option (a loss preset for ``slot == "loss"``)."""
        return option_lists.classes(slot, name, self.measured)

    def own(self, record: dict[str, Any] | None) -> bool:
        """Whether a cell is the method's own configuration: the config as shipped."""
        if record is None or not record.get("config_hash"):
            return False
        reference = self.by_id.get(f"reference/{record['method']}")
        return reference is not None and record["config_hash"] == reference.get("config_hash")


# ------------------------------------------------------------------------ cell texts


def reason(record: dict[str, Any]) -> str:
    """Why a cell was refused or failed, as one line."""
    if record["verdict"] == "refused":
        return " ".join(_first_sentence(message) for message in record.get("refusal", []))
    failure = record.get("failure") or {}
    where = failure.get("phase", "?")
    if failure.get("component"):
        where += f", in the {failure['component']}"
    return f"at {where}: {failure.get('error', '')}: {_first_sentence(failure.get('message', ''))}"


class Notes:
    """Numbered reasons below a table; the same reason gets the same number."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def number(self, text: str) -> int:
        if text not in self.items:
            self.items.append(text)
        return self.items.index(text) + 1

    def lines(self) -> list[str]:
        return [f"{index}. {text}" for index, text in enumerate(self.items, start=1)]


#: The mark of a verdict in a table cell.
MARKS = {"runs": "✓", "runs-empty": "∅", "runs-unstable": "≈"}


def no_grasp(record: dict[str, Any]) -> bool:
    """Whether a probe predicted no grasp on any of its frames."""
    grasps = record.get("grasps_per_frame") or []
    return bool(grasps) and not any(grasps)


def grey(text: str) -> str:
    """A cell of a probe that predicted no grasp: greyed on the page (extra.css)."""
    return f'<span class="gp-nograsp">{text}</span>'


def fall_text(record: dict[str, Any]) -> str:
    """The fall of every loss term of a record of judge 2, in a few words; empty for one
    of the first judge, which has no trend."""
    trend = record.get("trend") or {}
    falls = [
        values[2]
        for term, values in trend.items()
        if term.startswith("loss/") and values[2] is not None
    ]
    if record.get("judge", 1) < JUDGE or not falls:
        return ""
    low, high = min(falls), max(falls)
    return (
        f"fall {low * 100:.0f} to {high * 100:.0f} %" if low != high else f"fall {low * 100:.0f} %"
    )


def _level_notes(levels: CellLevels, notes: Notes) -> list[str]:
    """``learns ✗ (n)`` for every level a probed cell did not reach, with the reason below."""
    return [
        f"{level} ✗ ({notes.number(f'{level}: ' + '; '.join(levels.why[level]))})"
        for level in ("learns", "predicts")
        if levels.states.get(level) == NO and levels.why.get(level)
    ]


def verdict_text(
    record: dict[str, Any] | None, own: bool, notes: Notes, records: Records | None = None
) -> str:
    """One table cell: what was measured for this component under this method; for a
    probe, the grasps per frame and (judge 2) the fall of the loss terms below the mark.
    A probe that predicted no grasp is greyed."""
    if record is None:
        return "not probed"
    verdict = record["verdict"]
    adapter = f" in `{record['adapter']}`" if record.get("adapter") else ""
    fitted = "†" if any("_channels" in item for item in record.get("fitted", [])) else ""
    if verdict in MARKS:
        text = MARKS[verdict] + (" own" if own else f"{adapter}{fitted}")
        if verdict == "runs-unstable":
            text += f" ({notes.number(record.get('unstable', ''))})"
        if records is None:
            return text
        numbers = records.probed(record)
        grasps = numbers.get("grasps_per_frame") or []
        lines = [text]
        if grasps:
            lines.append(" · ".join(map(str, grasps)))
        lines += [part for part in [fall_text(numbers)] if part]
        lines += _level_notes(records.levels(record), notes)
        text = "<br>".join(lines)
        return grey(text) if no_grasp(numbers) else text
    if verdict == "refused":
        kind = refusal_kind(record)
        return f"✗ ({notes.number(reason(record))})" + (f"<br>{kind}" if kind else "")
    if verdict == "fails":
        return f"**fails** ({notes.number(reason(record))})"
    return verdict


def class_text(found: dict[str, OptionClass]) -> str:
    """The classes of an option at the four levels, levels of the same class said once:
    ``**universal** (composes, runs) · not checked (learns, predicts)``."""
    groups: list[tuple[str, list[str]]] = []
    for level in LEVELS:
        text = found[level].text
        if groups and groups[-1][0] == text:
            groups[-1][1].append(level)
        else:
            groups.append((text, [level]))
    return " · ".join(
        (f"**{text}**" if text in CLASS_WORDS else text) + f" ({', '.join(levels)})"
        for text, levels in groups
    )


def column_of(found: OptionClass) -> str:
    """The column of a class table an option goes into at one level."""
    if found.name in CLASS_WORDS:
        return found.name
    return RELATIVE if found.decided else UNDECIDED_COLUMN


# ---------------------------------------------------------------------------- tables


def _rows(slot: str, group: str) -> dict[tuple[str, str], list[Cell]]:
    """The cells of one slot, by ``(component, variant)`` in the order of the grid."""
    found: dict[tuple[str, str], list[Cell]] = {}
    for cell in grid(group):
        if cell.slot == slot:
            found.setdefault((cell.component, cell.variant), []).append(cell)
    return found


def bare_components() -> list[tuple[str, str]]:
    """``(slot, component)`` of every stage variant that is a component named by its type
    and lacks an option only a preset supplies: it has no cell (``needs_an_option``)."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    return [
        (slot, component)
        for slot in STAGE_SLOTS
        for choice, component in stage_variants(slot)
        if _in_tree_module(STAGE_REGISTRIES[slot].get(component))
        and needs_an_option(slot, choice, component)
    ]


def _bare_sentence() -> str:
    bare = bare_components()
    if not bare:
        return ""
    named = ", ".join(f"`{component}`" for _, component in bare)
    return (
        f"A component that needs an option only a preset supplies ({named}) has no cell "
        "under its type name: that request is refused for the missing option before any "
        "contract is compared, which says nothing about a combination. Its presets have cells."
    )


def _schedule_sentence(records: list[dict[str, Any]]) -> str:
    """How many probes of the page ran which schedule."""
    probed = [record for record in records if "probe" in record and "refusal" not in record]
    held = sum(record.get("schedule") == "constant" for record in probed)
    recipe = sum(record.get("schedule") == "recipe" for record in probed)
    before = len(probed) - held - recipe
    parts = [
        f"{held} held the peak rate (`schedule: constant` in the record)" if held else "",
        f"{recipe} ran the recipe's schedule (`schedule: recipe`)" if recipe else "",
        f"{before} were made before the probe had the option and ran the recipe's schedule "
        "over the probe's epochs (no `schedule` in the record)"
        if before
        else "",
    ]
    if before == len(probed):
        return (
            f"All {before} probes on this page were made before the probe had the option and "
            "ran the recipe's schedule over the probe's epochs (no `schedule` in their records)."
        )
    return f"Of the {len(probed)} probes on this page, " + "; ".join(p for p in parts if p) + "."


def slot_classes(
    slot: str, records: Records, group: str
) -> list[tuple[str, str, dict[str, OptionClass]]]:
    """``(component, variant, level → class)`` of every row of a slot. A cell without a
    record is ``not probed``: it is named after the class, never counted as a failure."""
    return [
        (component, variant, records.classes(slot, variant))
        for component, variant in _rows(slot, group)
    ]


def _drop_empty_columns(
    headers: list[str], rows: list[list[str]], keep: int
) -> tuple[list[str], list[list[str]]]:
    """A class table without the class columns no row uses (the first ``keep`` stay)."""
    used = [
        index
        for index in range(len(headers))
        if index < keep or any(row[index] != "—" for row in rows)
    ]
    return [headers[i] for i in used], [[row[i] for i in used] for row in rows]


def _undecided_sentence(level: str, options: int, reasons: Counter[str]) -> str:
    """A level no option of a table is decided at, in one sentence with the reasons."""
    return (
        f"No option of the {options} below is decided at the level **{level}**. Of their "
        "cells where the level before it was reached or undecided, "
        + ", ".join(
            f"{count} {'is' if count == 1 else 'are'} *{reason}*"
            for reason, count in reasons.most_common()
        )
        + ". "
        + " ".join(
            f"*{reason[0].upper() + reason[1:]}*: {UNDECIDED[reason]}."
            for reason in reasons
            if reason in UNDECIDED
        )
    )


def class_tables(
    rows_by_slot: list[tuple[list[str], list[tuple[str, dict[str, OptionClass]]]]],
    first: list[str],
) -> list[str]:
    """One table per level: every option of every slot under its class at that level.

    ``rows_by_slot`` holds ``(the first cells of a row, [(option label, level → class)])``;
    ``first`` the headers of those cells. A level no option is decided at is one sentence.
    """
    lines: list[str] = []
    columns = [*CLASSES, RELATIVE, UNDECIDED_COLUMN]
    for level in LEVELS:
        rows = []
        undecided: Counter[str] = Counter()
        options = 0
        decided = False
        for head, members in rows_by_slot:
            row = list(head)
            for column in columns:
                found = []
                for label, classes in members:
                    item = classes[level]
                    if column_of(item) != column:
                        continue
                    decided = decided or bool(item.checked)
                    if column == UNDECIDED_COLUMN:
                        options += 1
                        undecided.update(dict(item.undecided))
                    shown = label if column in CLASSES else f"{label} ({item.text})"
                    found.append(shown)
                row.append("<br>".join(found) or "—")
            rows.append(row)
        if not decided:
            lines += [_undecided_sentence(level, options, undecided), ""]
            continue
        headers, rows = _drop_empty_columns([*first, *columns], rows, len(first))
        lines += [f"**{level}**", "", *_table(headers, rows), ""]
    return lines


#: What a record of the first judge says at `learns` and `predicts`: it recorded neither.
NOT_CHECKED = "not checked"


#: What a cell of the grid without a record says.
NOT_YET = "not probed yet"


# ------------------------------------------------------------ where an option works
#
# The measured page and the evidence index say where an option works with the fractions
# and tails of grasppanda.combine.reach, the words of the explorer and of `grasppanda
# compose --list`; the methods, the scopes and the chips come from tools/docs_chips.py.

EVIDENCE_INDEX = REPO / "docs" / "evidence" / "index.md"
#: The caveat every evidence page carries once.
CAVEAT_SNIPPET = '--8<-- "docs/includes/caveat_probe.md"'
#: The levels a probe decides after `composes`, in the order of the columns.
SHOWN_LEVELS = ("runs", "learns", "predicts")
#: The slots of the training recipe, one section of the measured page.
RECIPE_SLOTS = ("optimizer", "schedule", "hooks", "augment", "recipe")
#: Section of the measured page → (its anchor, the slots it holds). The slot pages link the
#: anchors of the scene encoders, the losses and the training recipe.
SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Image encoders": ("image-encoders", ("image_encoder",)),
    SLOT_TITLES["backbone"]: ("scene-encoders", ("backbone",)),
    SLOT_TITLES["proposal"]: ("seed-proposals", ("proposal",)),
    SLOT_TITLES["view"]: ("view-stages", ("view",)),
    SLOT_TITLES["assigner"]: ("assigners", ("assigner",)),
    SLOT_TITLES["grouper"]: ("groupers", ("grouper",)),
    SLOT_TITLES["head"]: ("head-bundles", ("head",)),
    "Losses": ("losses", (LOSS_SLOT,)),
    "Post-processing": ("post-processing", (POST_SLOT,)),
    "Training recipe": ("the-training-recipe", RECIPE_SLOTS),
}
#: Slot → the anchor of its section on the measured page.
SECTION_OF = {slot: anchor for anchor, slots in SECTIONS.values() for slot in slots}
#: The columns of the table of a section; the last is level 2 (``docs_chips.LEVEL2_HEADER``).
OPTION_HEADERS = ["Option", "Runs", "Learns", "Predicts", "Trains like shipped"]
#: The words that introduce the list under a table: who each option leaves out.
LEFT_OUT = "Who is left out, and why:"


def _tool(name: str) -> Any:
    """A module of ``tools/``, loaded by path (``tools`` is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _chips() -> Any:
    """``tools/docs_chips.py``: the methods, scopes, chips and words every page shares."""
    return _tool("docs_chips")


def explorer_link(slot: str, depth: str = "../../") -> str:
    """A link that opens the explorer's comparison of ``slot`` under every method. Raw HTML,
    with the path as the site serves it (a page is a folder): the hash is the explorer's
    state, not a heading the link check could find."""
    return f'<a href="{depth}explorer/#grid={slot}">Compare in the explorer</a>'


class Reached(NamedTuple):
    """Where one option of one slot works.

    Attributes:
        slot: The slot (``loss`` for a loss preset).
        name: The option.
        levels: Level → its :class:`grasppanda.combine.reach.Reach`, for :data:`SHOWN_LEVELS`.
        level2: Its :class:`grasppanda.combine.reach.Level2`.
    """

    slot: str
    name: str
    levels: dict[str, Any]
    level2: Any


def _reached(records: Records, slot: str, name: str) -> Reached:
    from grasppanda.combine import reach as reaching

    levels = {level: reaching.reach(slot, name, level, records.measured) for level in SHOWN_LEVELS}
    return Reached(slot, name, levels, reaching.level2(slot, name, records.measured))


def where_options_work(records: Records) -> dict[str, list[Reached]]:
    """Slot → every option of the slot a user can ``--set`` (``docs_chips.options_of``: what
    every page counts) with where it reached each level and how it trained beside each
    method's own component; slots in page order, options in the order of the records."""
    chips = _chips()
    found = {
        slot: [_reached(records, slot, name) for name in chips.options_of(slot)]
        for slot in chips.SLOT_ORDER
    }
    return {slot: rows for slot, rows in found.items() if rows}


def not_options(records: Records) -> dict[str, list[Reached]]:
    """Slot → what the grid lists in the slot that is not an option a user can ``--set``: the
    adapters compose inserts and the point-painting wrappers. Each says, in its reach note,
    where it was measured (an adapter: in how many cells compose inserted it)."""
    chips = _chips()
    found: dict[str, list[Reached]] = {}
    for slot, name in option_lists._class_index():
        if slot in chips.SLOT_ORDER and name not in chips.options_of(slot):
            found.setdefault(slot, []).append(_reached(records, slot, name))
    return found


def chip_of(found: Any) -> str:
    """The chip of a reach or a level-2 result, its tail as the tooltip."""
    return _chips().chip(found.fraction, found.tone, found.tail or getattr(found, "none", ""))


def _parts(tail: str) -> list[str]:
    """The parts of a tail: split at ``; `` outside brackets."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(tail):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == ";" and depth == 0 and tail[index + 1 : index + 2] == " ":
            parts.append(tail[start:index])
            start = index + 2
    parts.append(tail[start:])
    return [part for part in parts if part]


def named_parts(found: Any) -> list[str]:
    """The parts of the tail of a reach, each naming the methods it is about: a tail leaves
    them out where the fraction says it (every swap missed for one reason, or the one method
    an option applies to), and a table cell in words names them all the same. The note of an
    option measured inside another one is not a part: the row says it once."""
    tail = found.tail
    if found.note and tail.startswith(found.note):
        tail = tail[len(found.note) :].removeprefix("; ")
    parts = _parts(tail)
    reasons = set(found.missed.values())
    if len(reasons) == 1 and not found.reached and not found.undecided:
        reason = next(iter(reasons))
        missed = ", ".join(found.missed)
        parts = [f"not {missed} ({reason})" if part == reason else part for part in parts]
    if len(found.applies) == 1 and found.undecided:
        method = found.applies[0]
        parts = [f"{method} {part}" if part.startswith("undecided (") else part for part in parts]
    return parts


def left_out(row: Reached) -> str:
    """Who the chips of a row leave out, and why, in words: what each level adds to the
    level before it, then the level-2 outcomes that are not on a par. An entry without a
    record says why; an option measured inside another one says so first."""
    runs = row.levels["runs"]
    if runs.note and not runs.applies:
        return runs.note
    seen: set[str] = set()
    shown = [runs.note] if runs.note else []
    for level in SHOWN_LEVELS:
        new = [part for part in named_parts(row.levels[level]) if part not in seen]
        seen.update(new)
        if new:
            shown.append(f"*{level}*: " + "; ".join(new))
    if not row.level2.none and row.level2.tail:
        shown.append(f"*{_chips().LEVEL2_NAME}*: " + row.level2.tail)
    return " · ".join(shown) or "—"


def option_label(slot: str, name: str) -> str:
    """An option as a reader writes it after ``--set``; a stage component without a preset
    is built with its constructor defaults."""
    if slot in RECIPE_SLOTS:
        return f"`{slot}={name}`"
    if slot in ("backbone", *STAGE_SLOTS) and name not in presets().get(slot, {}):
        return f"`{name}` (defaults)"
    return f"`{name}`"


def row_label(slot: str, name: str) -> str:
    """An option as the first cell of a table row: plain text, which wraps at its hyphens in
    a narrow column where code would break anywhere."""
    return option_label(slot, name).replace("`", "")


def option_rows(rows: list[Reached]) -> list[list[str]]:
    return [
        [
            row_label(row.slot, row.name),
            *(chip_of(row.levels[level]) for level in SHOWN_LEVELS),
            chip_of(row.level2),
        ]
        for row in rows
    ]


def left_out_list(rows: list[Reached]) -> list[str]:
    """Under a table, one item per option that leaves a method out: who, and why. A list,
    so the table stays narrow enough to read without scrolling."""
    items = [(row, left_out(row)) for row in rows]
    shown = [f"- {option_label(row.slot, row.name)}: {text}" for row, text in items if text != "—"]
    return [LEFT_OUT, "", *shown] if shown else []


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def counted(rows: list[Reached], options: int) -> str:
    """One line of chips: how many of the ``options`` of a section (``docs_chips.
    option_count``) run under every method they were tried under, under some, under none;
    how many train like shipped under every method they were compared under; an option no
    method has as a swap (it writes out a method's own component) counts as shipped."""
    chips = _chips()
    tones = Counter(row.levels["runs"].tone for row in rows)
    no_record = sum(row.levels["runs"].fraction == "no record" for row in rows)
    own = sum(row.levels["runs"].fraction == "own only" for row in rows)
    undecided = tones["muted"] - no_record - own
    shown = [
        f"**{_plural(options, 'option')}**",
        chips.chip(f"{tones['ok']} run everywhere", "ok" if tones["ok"] else "muted"),
    ]
    shown += [chips.chip(f"{tones['warn']} under some", "warn")] if tones["warn"] else []
    shown += [chips.chip(f"{tones['no']} under none", "no")] if tones["no"] else []
    shown += [chips.chip(f"{undecided} undecided", "muted")] if undecided else []
    shipped = chips.chip(f"{own} as shipped", "muted", "the method's own component, not a swap")
    shown += [shipped] if own else []
    shown += [chips.chip(f"{no_record} without a record", "muted")] if no_record else []
    compared = [row for row in rows if row.level2.compared]
    if compared:
        par = sum(len(row.level2.on_a_par) == len(row.level2.compared) for row in compared)
        title = f"{chips.LEVEL2_NAME}: on a par under every method compared"
        shown.append(chips.chip(f"{par} of {len(compared)} train like shipped", "", title))
    return " ".join(shown)


def answer_matrix(found: dict[str, list[Reached]], level: str) -> list[str]:
    """Slot × method: the options that reached ``level``, out of those tried as a swap."""
    chips = _chips()
    names = [info.name for info in chips.methods()]
    rows = []
    for slot, members in found.items():
        under = chips.slot_methods(slot)
        cells = []
        for method in names:
            if method not in under:
                cells.append("—")
                continue
            tried = [row for row in members if method in row.levels[level].swaps]
            missed = [row.name for row in tried if method not in row.levels[level].reached]
            if not tried:
                cells.append(chips.chip("own only", "muted"))
                continue
            reached = len(tried) - len(missed)
            tone = "ok" if not missed else "warn" if reached else "no"
            title = ""
            if missed:
                title = "left out: " + ", ".join(missed[:6])
                title += f" and {len(missed) - 6} more" if len(missed) > 6 else ""
            cells.append(chips.chip(f"{reached}/{len(tried)}", tone, title))
        title = chips.SLOT_TITLES[slot].replace(" ", "\u00a0")  # one line per slot
        label = f"[{title}](#{SECTION_OF[slot]})"
        rows.append([label, *cells])
    return _table(["Slot", *names], rows)


def matrix(lines: list[str]) -> list[str]:
    """A slot × method table in the wrapper that lets its columns narrow (extra.css)."""
    return ['<div class="gp-matrix" markdown>', "", *lines, "", "</div>"]


def answer_tabs(found: dict[str, list[Reached]]) -> list[str]:
    """The answer matrix at each level, one content tab per level."""
    lines: list[str] = []
    for level in SHOWN_LEVELS:
        lines += [f'=== "{level}"', "", *_indent(matrix(answer_matrix(found, level))), ""]
    return lines


def _section_notes(slots: tuple[str, ...]) -> list[str]:
    """What a section's table does not show: options without a cell, and why."""
    from grasppanda.combine.grid import RECIPE_ENCODER_PRESETS

    notes = []
    bare = [component for slot, component in bare_components() if slot in slots]
    if bare:
        notes.append(
            "No row for " + ", ".join(f"`{name}`" for name in bare) + " named by its type: it "
            "needs an option only a preset supplies, so the bare request is refused before any "
            "contract is compared. Its presets have rows."
        )
    if LOSS_SLOT in slots:
        shown = _chips().options_of(LOSS_SLOT)  # a row of its own says where it was measured
        notes += [
            f"`{name}` has no cell of its own: {why}."
            for name, why in LOSS_NOT_PROBED.items()
            if name not in shown
        ]
    encoder = [name for name in RECIPE_ENCODER_PRESETS if name.split("/")[0] in slots]
    if encoder:
        notes.append(
            "A preset that names the scene encoder ("
            + ", ".join(f"`{name.replace('/', '=')}`" for name in encoder)
            + ") applies to the seed-based methods only: HGGD has no scene encoder."
        )
    return notes


def loss_terms_table() -> list[str]:
    """The term a loss cell replaces: one per kind of term and method, the last of that kind
    in the order of the pipeline."""
    from grasppanda.models.losses.terms import TERM_KINDS

    terms = {method: loss_terms(method) for method in METHODS}
    kinds = [kind for kind in TERM_KINDS if any(kind in found for found in terms.values())]
    rows = [
        [method, *(f"`{terms[method][kind]}`" if kind in terms[method] else "—" for kind in kinds)]
        for method in METHODS
    ]
    return [
        "A loss cell replaces **one term per kind and method: the last term of that kind in "
        "the pipeline** (proposal, view, head), the one nearest the grasp. Those terms:",
        "",
        *_table(["Method", *(f"`{kind}` term" for kind in kinds)], rows),
        "",
    ]


#: The words that introduce what a section lists that is not an option.
NOT_OPTIONS = "Not options, so not counted:"


def not_options_list(extra: list[Reached]) -> list[str]:
    """What a section lists that is not an option (adapters, point-painting wrappers), each
    with its reach note: where it was measured, or why it has no record."""
    if not extra:
        return []
    items = [f"- `{row.name}`: {row.levels['runs'].note or left_out(row)}." for row in extra]
    return [NOT_OPTIONS, "", *items]


def option_section(
    title: str,
    anchor: str,
    slots: tuple[str, ...],
    rows: list[Reached],
    extra: list[Reached] | None = None,
) -> list[str]:
    """One section of the measured page: a line that counts, then every option and the
    explorer in a fold; what the section lists that is not an option comes after the
    options, each with where it was measured."""
    options = sum(_chips().option_count(slot) for slot in slots)
    fold = [f"{explorer_link(slots[0])}: the same options, cell by cell.", ""]
    fold += ["<div data-search-exclude markdown>", "", *_table(OPTION_HEADERS, option_rows(rows))]
    fold += ["", *left_out_list(rows), "", "</div>"]
    before = loss_terms_table() if LOSS_SLOT in slots else []
    notes = _section_notes(slots)
    after = not_options_list(extra or [])
    return [
        f"### {title} {{#{anchor}}}",
        "",
        counted(rows, options),
        "",
        f'??? note "{title}: {_plural(options, "option")}, one row each"',
        "",
        *_indent(
            [
                *before,
                *fold,
                *(["", *after] if after else []),
                *(["", *notes] if notes else []),
            ]
        ),
        "",
    ]


def level_cell(record: dict[str, Any], levels: CellLevels, level: str) -> str:
    """Whether a probe reached ``level``: a chip, with what kept it from the level beside it;
    :data:`NOT_CHECKED` for a record of the first judge."""
    if record.get("judge", 1) < JUDGE:
        return NOT_CHECKED
    state = levels.states.get(level, "")
    if state == YES:
        return _chips().chip("✓", "ok")
    if state == NO:
        why = "; ".join(levels.why.get(level, []))
        return _chips().chip("✗", "no", why) + (f" {why}" if why else "")
    return state or "—"


def references_section(records: Records) -> list[str]:
    """Every method as shipped: its verdict, its grasps and the levels it reached."""
    rows = []
    for info in _chips().methods():
        record = records.measured.reference(info.name)
        if record is None:
            rows.append([info.name, NOT_YET, "—", "—", "—"])
            continue
        numbers, levels = records.probed(record), records.levels(record)
        grasps = "\u00a0·\u00a0".join(str(count) for count in numbers.get("grasps_per_frame") or [])
        rows.append(
            [
                f"[{info.name}](../{info.page})",
                f"`{record['verdict']}`",
                grasps or "—",
                level_cell(numbers, levels, "learns"),
                level_cell(numbers, levels, "predicts"),
            ]
        )
    return [
        "Each method's own config, probed the same way: the base every swap is held against. A "
        "level the method as shipped does not reach in a short run is undecided for its swaps.",
        "",
        *_table(["Method", "Verdict", "Grasps", "Learns", "Predicts"], rows),
        "",
    ]


#: Every set of registered stage classes the contract rules accept, written by
#: ``tools/make_component_docs.py`` for *Swap rules*.
STAGE_SETS = REPO / "docs" / "advanced_guides" / "stage_sets.json"


def registered_sets() -> int:
    """How many sets of registered stage classes the contract rules accept: the count of
    *Swap rules* (``stage_sets.json``). The grid counts stage options per kind of encoder
    output instead, so its number differs."""
    return int(json.loads(STAGE_SETS.read_text(encoding="utf-8"))["count"])


def stage_sets_section(records: Records) -> list[str]:
    """The sets of stages: how many the rules accept, how many were probed, their verdicts."""
    wanted = grid("stage-set")
    recorded = [cell for cell in wanted if records.get(cell.id) is not None]
    if not recorded:
        return [
            f"None of the {len(wanted)} sets of stage options (per kind of encoder output) the "
            "contract rules accept was probed yet. "
            "`python -m tools.probe_grid run --groups stage-set` probes them.",
            "",
        ]

    def variants(members: list[Cell]) -> set[tuple[str, str]]:
        """``(kind of encoder, slot=choice)`` of every stage variant in some of the sets."""
        return {(cell.variant, item) for cell in members for item in cell.sets[1:]}

    listed, seen = variants(wanted), variants(recorded)
    verdict = {cell.id: records.get(cell.id)["verdict"] for cell in recorded}
    used = [name for name in VERDICT_ORDER if name in verdict.values()]
    rows = []
    for method in METHODS:
        members = [cell for cell in recorded if cell.method == method]
        if not members:
            continue
        counts = Counter(verdict[cell.id] for cell in members)
        rows.append(
            [
                method,
                str(len(members)),
                *(str(counts[name]) if counts[name] else "—" for name in used),
            ]
        )
    counts = Counter(verdict.values())
    return [
        "A set replaces the encoder and every stage at once. The rules accept "
        f"{len(wanted)} sets of stage options (per kind of encoder output); "
        f"**{len(recorded)} were probed** ("
        + ", ".join(f"{counts[name]} `{name}`" for name in used)
        + f"), covering {len(seen)} of the {len(listed)} pairs of a stage variant and a kind of "
        "encoder. A set that runs says nothing about another set with the same parts.",
        "",
        '??? note "The probed sets, by the method whose data recipe serves their labels"',
        "",
        *_indent(
            [
                *_table(["Data recipe of", "Sets probed", *(f"`{name}`" for name in used)], rows),
                "",
                "Every probed set, with its request and its verdict, is in the "
                "[records](measured_combinations.json) (group `stage-set`). "
                "[Swap rules](swapping.md#every-legal-set-of-stages) counts the same rules over "
                f"registered classes: {registered_sets()} sets.",
            ]
        ),
        "",
    ]


def cross_section(records: Records) -> list[str]:
    """Requests that replace options of several slots at once."""
    wanted = grid("cross")
    recorded = [cell for cell in wanted if records.get(cell.id) is not None]
    if not recorded:
        return ["No request that replaces several slots at once was probed yet.", ""]
    verdict = {cell.id: records.get(cell.id)["verdict"] for cell in recorded}
    used = [name for name in VERDICT_ORDER if name in verdict.values()]
    rows, requests = [], []
    for info in _chips().methods():
        members = [cell for cell in recorded if cell.method == info.name]
        if not members:
            continue
        counts = Counter(verdict[cell.id] for cell in members)
        rows.append(
            [
                info.name,
                str(len(members)),
                *(str(counts[name]) if counts[name] else "—" for name in used),
            ]
        )
        requests += [
            [info.name, " ".join(f"`--set {item}`" for item in cell.sets), f"`{verdict[cell.id]}`"]
            for cell in members
        ]
    counts = Counter(verdict.values())
    return [
        "These cells replace an encoder and options of other slots in one request: a "
        f"hand-picked sample of {len(recorded)}, at least two per method ("
        + ", ".join(f"{counts[name]} `{name}`" for name in used)
        + "). A combination not listed was not probed; its verdict does not follow from its "
        "parts.",
        "",
        f'??? note "The {len(requests)} requests, method by method"',
        "",
        *_indent(
            [
                *_table(["Method", "Requests probed", *(f"`{name}`" for name in used)], rows),
                "",
                *_table(["Method", "Request", "Verdict"], requests),
            ]
        ),
        "",
    ]


def findings_section(records: list[dict[str, Any]]) -> list[str]:
    """Cells that passed the config check and failed afterwards, grouped by reason."""
    failed = [record for record in records if record["verdict"] == "fails"]
    if not failed:
        return ["No combination that passed the config check failed afterwards.", ""]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in failed:
        groups.setdefault(re.sub(r"\b\d+\b", "N", reason(record)), []).append(record)
    lines = [
        f"{len(failed)} cell(s) passed the config check and failed afterwards: each group is a "
        "gap between what the config check knows and what the components need (*N* stands for "
        "a number that differs from cell to cell).",
        "",
    ]
    for key, members in groups.items():
        listed = ", ".join(f"`{record['id']}`" for record in members)
        lines += [f"- **{key}** {len(members)} cell(s): {listed}."]
    return [*lines, ""]


def counts_section(records: list[dict[str, Any]]) -> list[str]:
    """Cells and verdicts per group of the grid."""
    rows = []
    every = grid()
    for group, title in GROUP_TITLES.items():
        members = [record for record in records if record["group"] == group]
        counts = Counter(record["verdict"] for record in members)
        total = sum(cell.group == group for cell in every)
        rows.append(
            [
                f"`{group}`: {title}",
                f"{len(members)} of {total}",
                *(str(counts[verdict]) if counts[verdict] else "—" for verdict in VERDICT_ORDER),
            ]
        )
    counts = Counter(record["verdict"] for record in records)
    rows.append(
        [
            "**all**",
            f"**{len(records)} of {len(every)}**",
            *(f"**{counts[verdict]}**" for verdict in VERDICT_ORDER),
        ]
    )
    return _table(["Group", "Cells recorded", *(f"`{v}`" for v in VERDICT_ORDER)], rows)


def _runs(about: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    """One sentence per run that measured cells of the file."""
    sentences = []
    counts = Counter(record["run"] for record in records)
    for name, run in about["runs"].items():
        commit = f" at commit `{run['commit']}`" if run.get("commit") else ""
        sentences.append(
            f"{counts[name]} cells on {run['date']} with GraspPanda{commit}, PyTorch "
            f"{run['torch']}, Python {run['python']}, on one {run['gpu']} shared with other "
            f"jobs, {run['probes_at_a_time']} probes at a time."
        )
    return sentences


def judges_text(records: list[dict[str, Any]]) -> list[str]:
    """Which judge gave the verdicts, and what the flags of judge 2 mean."""
    second = sum(record.get("judge", 1) >= JUDGE for record in records)
    side = DATA.with_suffix("") / _runner().SERIES_NAME
    series = f"measured_combinations/{side.name}"
    where = f"[{series}]({series})" if side.is_file() else f"`{series}`"
    return [
        f"**{second} of the {len(records)} records** on this page were judged by judge "
        f"{JUDGE} (`judge: {JUDGE}` in the records), which also records whether the parts of "
        "a model learned: the trend of every loss term, the units no gradient reached, the "
        "parts that did not change and the spread of the scores. A record of the first judge "
        f"has none of it, so its `learns` and `predicts` are undecided ({NOT_CHECKED}). The "
        f"loss of every iteration of every probe of judge 2 is in {where}.",
        "",
        "The flags of a loss term, named in the tails (*active* iterations: the term was "
        "reported, finite and not exactly 0):",
        "",
        *[f"- `{name}`: {meaning}." for name, meaning in TREND_FLAGS.items()],
        "",
        "The schedules of a probe (`schedule` in a record):",
        "",
        *[f"- `{name}`: {meaning}." for name, meaning in SCHEDULES.items()],
    ]


def _but(without: dict[str, str]) -> str:
    """`` (but `target-mask`: …)``: the presets a group of the grid has no cell for."""
    if not without:
        return ""
    return " (" + "; ".join(f"but `{name}`: {why}" for name, why in without.items()) + ")"


def what_was_run(about: dict[str, Any], cells_: list[dict[str, Any]]) -> list[str]:
    """The fold of the measured page that says what ran, where, and how it was judged."""
    counts = Counter(record["verdict"] for record in cells_)
    shared = sum("probed_as" in record for record in cells_)
    iterations, frames = about["probe"]["iterations"], about["probe"]["frames"]
    body = [
        f"**{len(cells_)} recorded cells** ("
        + ", ".join(f"{counts[v]} `{v}`" for v in VERDICT_ORDER if counts[v])
        + "): every shipped method config as it is, every registered scene encoder under every "
        "seed-based method, every variant of every other stage with one stage replaced at a "
        "time, every image-encoder preset under HGGD, every post-processing preset under every "
        f"method{_but(POST_NOT_PROBED)}, every recipe preset under every method it applies to, "
        f"every loss preset in one term of each kind it accepts{_but(LOSS_NOT_PROBED)}, a "
        "sample of the sets of stages and a few requests that replace several slots at once. "
        + _bare_sentence(),
        "",
        *counts_section(cells_),
        "",
        f"- **A cell**: `grasppanda compose <method config> --set <slot>=<choice>` (so an encoder "
        "is fitted and adapters are inserted exactly as a user gets them), then `grasppanda "
        f"probe <composed config> --iters {iterations} --frames {frames} --schedule constant`: "
        "the real trainer with the method's own recipe on the split `sample` of "
        "GraspNet-1Billion (RealSense; scene 0, frames 0, 32, …, 224), then prediction on two "
        "of those frames. Both commands of every cell are in the "
        "[records](measured_combinations.json).",
        "- **The learning rate**: the recipe's peak rate at every step (`--schedule constant`), "
        "scaled by the ratio of the batch sizes when the probe's batches are smaller (FineGrasp's "
        "0.008 for 32 frames trains at 0.002 on 8); the optimizer, weight decay, clipping and "
        "hooks as the recipe has them. Over the few epochs of a probe the schedules would decay "
        "some methods far more than others, and the trends would not compare. A request that "
        "sets the schedule (a schedule preset, a schedule-free optimizer, a recipe preset) runs "
        "that schedule. " + _schedule_sentence(cells_),
        '- **One exception to "as a user gets them"**: configs that read the width histogram of '
        "Scale-Balanced-Grasp were probed with the reference's own file "
        "(`scale_statistic.source: legacy_import`): the native histogram is a statistic of the "
        "whole training split, and a probe reads eight frames.",
        "- **Two cells with one config are one experiment**: a cell that composes to the config "
        f"of another cell (the same config hash) shares its probe; {shared} of the "
        f"{len(cells_)} cells do. A cell that composes to its method's shipped config is the "
        "method's own component and is not counted as a swap.",
        "- **When and where**: " + " ".join(_runs(about, cells_)),
        "",
        *judges_text(cells_),
        "",
        "**Passed the check, failed afterwards.**",
        "",
        *findings_section(cells_),
    ]
    while body and not body[-1]:
        body.pop()
    return ['??? note "What was run, and how it was judged in detail"', "", *_indent(body), ""]


def levels_section() -> list[str]:
    """The levels a cell reaches and the classes of an option, with their definitions and
    thresholds as :mod:`grasppanda.combine.levels` holds them. The measured page, the slot
    index and the method pages print it."""
    return [
        f"**{CAVEAT[0].upper() + CAVEAT[1:]}.** A cell (one option under one method) is "
        "decided at four levels, each from its record alone; every level needs the one "
        "before it:",
        "",
        *[f"- **{level}**: {meaning}." for level, meaning in LEVELS.items()],
        "",
        "A level that cannot be decided from a record says why:",
        "",
        *[f"- *{reason}*: {meaning}." for reason, meaning in UNDECIDED.items()],
        "",
        "The **class** of an option at a level counts the methods under which it reached "
        "that level. For a stage of the seed-based skeleton decided under all five "
        "seed-based methods (" + ", ".join(METHODS) + "):",
        "",
        *[f"- **{name}**: {meaning}." for name, (_, meaning) in CLASS_WORDS.items()],
        "",
        "An option of another slot (post-processing, loss, recipe, image encoder) is "
        "**universal** only when it reached the level under every method it applies to and "
        "those include the five seed-based ones. Every other class reads **N of M probed**: "
        "it reached the level under N of the M methods where the level was decided, and the "
        "methods where it was not are named after it (*+2 not checked*). A level that no "
        "method decided is named by the reason. A level decided only where the level before "
        "it failed is not checked: the class says so instead of counting the failures again.",
        "",
    ]


# ------------------------------------------------------------------------------ page

#: What the reach note of an entry without a record of its own starts with → how a sentence
#: counts such entries (one, several).
NO_RECORD_KINDS = {
    "an adapter": ("adapter compose inserts", "adapters compose inserts"),
    "point painting": ("point-painting wrapper", "point-painting wrappers"),
}


def without_record(found: dict[str, list[Reached]], extra: dict[str, list[Reached]]) -> str:
    """The entries of the grid without a record of their own, in words: ``two adapters
    compose inserts, two point-painting wrappers, `post=target-mask` and
    `loss=distribution-focal`, measured inside `decoupled-grid-bins```, counted from the
    reach notes (the adapters and wrappers are not options; an option without a cell is
    named, and so is a loss measured inside a head bundle, last)."""
    from grasppanda.combine.reach import LOSS_INSIDE

    kinds: dict[str, int] = {}
    named: list[str] = []
    for rows in extra.values():
        for row in rows:
            kind = row.levels["runs"].note.split(":")[0]
            if kind in NO_RECORD_KINDS:
                kinds[kind] = kinds.get(kind, 0) + 1
            else:
                named.append(f"`{row.slot}={row.name}`")
    for rows in found.values():
        named += [
            f"`{row.slot}={row.name}`" for row in rows if row.levels["runs"].fraction == "no record"
        ]
    named += [
        f"`{LOSS_SLOT}={row.name}`, measured inside `{LOSS_INSIDE[row.name][0]}`"
        for row in found.get(LOSS_SLOT, [])
        if row.name in LOSS_INSIDE
    ]
    words = [
        f"{_number(count)} {NO_RECORD_KINDS[kind][count != 1]}" for kind, count in kinds.items()
    ]
    return _join_words([*words, *named])


def coverage(found: dict[str, list[Reached]], extra: dict[str, list[Reached]]) -> str:
    """What the records cover, in two sentences: never more than they hold."""
    return (
        "Every option was composed under each method it fits; those accepted were probed on "
        f"real frames. The few without a record of their own ({without_record(found, extra)}), "
        "the refusals and the options that need your consent are named per option."
    )


def page() -> str:
    """``docs/advanced_guides/measured_combinations.md``: *Does it run?*"""
    data = measured()
    about, cells_ = data["about"], data["cells"]
    records = Records(cells_)
    found = where_options_work(records)
    extra = not_options(records)
    chips = _chips()
    lines = [
        "<!-- Generated by tools/make_combination_docs.py from measured_combinations.json.",
        "     Do not edit by hand: probe (python -m tools.probe_grid), export, regenerate. -->",
        "",
        "# Does it run?",
        "",
        "Which option trains on real frames under which method? " + coverage(found, extra),
        "",
        CAVEAT_SNIPPET,
        "",
        "## The answer, slot by method {#the-answer}",
        "",
        *answer_tabs(found),
        "A cell counts the options that reached the level, out of those tried there as a swap "
        "(a method's own component is not counted); — : no such slot. Hover a cell for who is "
        "left out; [below](#measured-classes), why.",
        "",
        "## How it was judged {#how-it-was-judged}",
        "",
        "- **One cell, one option under one method**: `grasppanda compose <method config> --set "
        "<slot>=<option>`, then `grasppanda probe` on the composed config. Measured on: "
        f"{chips.measured_on(1)}.",
        "- **Four levels, each needing the one before**: `composes` (config accepted), `runs` "
        "(trains on real frames without an error), `learns` (its losses fall), `predicts` (it "
        "ranks grasps). [Exact definitions](../evidence/index.md#levels).",
        "- **A refusal is the config check's answer**: nothing was built, and the kind says why "
        "([refusal words](../evidence/index.md#refusals)).",
        "- **Undecided counts against**: a method where a level could not be decided stays in the "
        "count, named with the reason.",
        "- **One run per cell**: the config hash and the first losses repeat exactly; later "
        "numbers may differ in the next run.",
        "",
        *what_was_run(about, cells_),
        "## Where each option works {#measured-classes}",
        "",
        f"Each section counts the options a user can `--set` ({chips.option_count()} in all). "
        "In its table, a chip counts the methods under which the option reached the level, out "
        f"of those it was tried under as a swap; *{chips.LEVEL2_NAME}* is "
        "[level 2](learns_like_reference.md). The list under it says who is left out, and why.",
        "",
    ]
    for title, (anchor, slots) in SECTIONS.items():
        rows = [row for slot in slots for row in found.get(slot, [])]
        if rows:
            more = [row for slot in slots for row in extra.get(slot, [])]
            lines += option_section(title, anchor, slots, rows, more)
    lines += ["## The methods as shipped {#the-methods-as-shipped}", ""]
    lines += references_section(records)
    lines += ["## Sets of stages {#sets-of-stages}", "", *stage_sets_section(records)]
    lines += ["## Several slots at once {#several-slots-at-once}", "", *cross_section(records)]
    lines += [
        "## The records {#the-records}",
        "",
        "[measured_combinations.json](measured_combinations.json) holds every cell: its two "
        "commands, its verdict, why it was refused or failed, and what the probe saw; "
        "[losses_per_iteration.json](measured_combinations/losses_per_iteration.json) every loss "
        "term at every iteration. The [explorer](../explorer/index.md) answers any selection "
        "from them, with the command.",
        "",
        '??? note "Add a component to the grid"',
        "",
        "    A new encoder or stage enters the grid by being registered. Give it a preset if its "
        "options cannot be guessed ([Compose and probe](../user_guides/combinations.md)), probe "
        "its cells and regenerate. `tests/unit/test_combine_grid.py` fails until every cell of "
        "a registered component has a record.",
        "",
        "    ```bash",
        "    python -m tools.probe_grid run --only 'encoder/*/MyEncoder' 'stage/*/*/my-preset'",
        "    python -m tools.probe_grid export",
        "    python -m tools.make_combination_docs",
        "    ```",
    ]
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------ the evidence index

#: The levels in one line each; the exact definitions are :data:`LEVELS`.
LEVEL_LINES = {
    "composes": "the config check accepts the composed config",
    "runs": "it trains on real frames without an error",
    "learns": "its losses fall",
    "predicts": "it ranks grasps: every frame has one, and the scores differ",
}
#: The verdicts of a probe in one line each; the exact definitions are :data:`VERDICTS`.
VERDICT_LINES = {
    "runs": "built, trained on real frames with finite losses, and predicted grasps",
    "runs-empty": "as `runs`, but no probed frame has a grasp",
    "runs-unstable": "a loss or a weight became non-finite, or a part got no seed or no gradient",
    "fails": "an exception after the config check had passed",
    "refused": "the config check said no, with the fix; nothing was built",
    "wired": "a synthetic scene only (no dataset): shows the stages connect, nothing else",
}
#: The level-2 outcomes in one line each, worst last; the exact definitions are the
#: ``outcomes`` of ``learns_like_reference.json``.
LEVEL2_LINES = {
    "on a par": "every comparable loss ends within the method's tolerance of its own component's",
    "worse": "a comparable loss ends above that tolerance, or a part got no gradient",
    "not comparable": "the option redefines every loss term",
    "diverging": "a loss rises at the end, a value became non-finite, or no grasp was predicted",
    "did not train": "the run did not finish (record value `fails`)",
}
#: The pages of the evidence section, with what each answers. ``{iterations}`` is the length
#: of a level-2 run; ``{encoders}`` counts the table of the encoders' page
#: (:func:`encoder_card`).
EVIDENCE_PAGES = (
    (
        "Model zoo",
        "../model_zoo.md",
        "AP on whole test splits: the authors' weights scored here, "
        "and the one model trained here.",
    ),
    (
        "Does it run?",
        "../advanced_guides/measured_combinations.md",
        "Each option under each method it fits, trained briefly on real frames.",
    ),
    (
        "Does it train like the shipped component?",
        "../advanced_guides/learns_like_reference.md",
        "Each swap that ran, trained {iterations} iterations beside the shipped component.",
    ),
    (
        "Swap rules",
        "../advanced_guides/swapping.md",
        "Which stages fit together, and why a request is refused.",
    ),
    (
        "Encoders checked against their authors' code",
        "../model_zoo_backbones.md",
        "{encoders}.",
    ),
)
#: The page of the scene encoders checked against their authors' code (written by hand), and
#: the heading of its table.
BACKBONES_PAGE = REPO / "docs" / "model_zoo_backbones.md"
BACKBONES_TABLE = "## The encoders and how each was checked"
#: The status of an encoder in that table → (what the page's summary calls it, what the
#: evidence card says of that many encoders).
ENCODER_CHECKS = {
    "`weights-verified`": ("Weights verified", "with released weights loaded and outputs compared"),
    "`tested` + parity": ("Compared on synthetic weights", "compared on synthetic weights"),
    "`tested`": ("Unit-tested only", "unit-tested only"),
}


def encoder_checks(text: str | None = None) -> dict[str, list[str]]:
    """Status → the encoders of that status, in the order of the table of the encoders' page
    (``docs/model_zoo_backbones.md``, under :data:`BACKBONES_TABLE`).

    Raises:
        ValueError: For a status :data:`ENCODER_CHECKS` does not know.
    """
    text = BACKBONES_PAGE.read_text(encoding="utf-8") if text is None else text
    lines = [line.strip() for line in text.split(BACKBONES_TABLE, 1)[1].splitlines()]
    start = next(index for index, line in enumerate(lines) if line.startswith("|"))
    found: dict[str, list[str]] = {status: [] for status in ENCODER_CHECKS}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        name, status = cells[0].strip("`"), cells[3]
        if status not in found:
            raise ValueError(f"{name}: status {status!r} is none of {list(ENCODER_CHECKS)}")
        found[status].append(name)
    return found


def encoder_card() -> str:
    """``5 encoders with released weights loaded and outputs compared; 3 compared on
    synthetic weights; 4 unit-tested only``: the table of the encoders' page, counted."""
    parts = [
        f"{len(names)} {ENCODER_CHECKS[status][1]}"
        for status, names in encoder_checks().items()
        if names
    ]
    first, _, rest = parts[0].partition(" ")
    return "; ".join([f"{first} encoders {rest}", *parts[1:]])


#: The headers of the table that answers per slot (the scope of a slot is in the folded
#: slot × method table below it).
PER_SLOT_HEADERS = [
    "Slot",
    "Options",
    "Run under every method tried",
    "Train like shipped under every method compared",
]


#: Why a slot of the per-slot table has nothing compared at level 2, in a few words: a slot
#: of ``reach.LEVEL2_NONE``, or the reason ``reach.LEVEL2_EXCLUDED`` gives for every option
#: of the slot. The long words are in the fold of level 2.
NOTHING_COMPARED = {
    "post": "trains nothing",
    "schedule": "held fixed",
    "excluded:schedule": "own schedules",
    "excluded:runs": "did not run",
}


def nothing_compared(slot: str, members: list[Reached]) -> str:
    """``— (why)``: the level-2 cell of a slot with nothing compared."""
    from grasppanda.combine.reach import LEVEL2_EXCLUDED, LEVEL2_NONE

    if slot in LEVEL2_NONE:
        return f"— ({NOTHING_COMPARED[slot]})"
    kinds = {text: kind for kind, text in LEVEL2_EXCLUDED.items()}
    shared = set(LEVEL2_EXCLUDED)
    for row in members:
        shared &= {kinds[text] for text in row.level2.excluded.values() if text in kinds}
    reasons = [NOTHING_COMPARED[f"excluded:{kind}"] for kind in LEVEL2_EXCLUDED if kind in shared]
    return f"— ({'; '.join(reasons)})" if reasons else "— (nothing compared)"


def per_slot_summary(found: dict[str, list[Reached]]) -> list[str]:
    """Slot → its options (``docs_chips.option_count``), how many run under every method
    they were tried under, how many train like shipped under every method they were compared
    under (``grasppanda.combine.reach``); ``— (why)`` where nothing was compared."""
    chips = _chips()
    rows = []
    for slot, members in found.items():
        tried = [row for row in members if row.levels["runs"].swaps]
        everywhere = sum(row.levels["runs"].tone == "ok" for row in tried)
        compared = [row for row in members if row.level2.compared]
        par = sum(len(row.level2.on_a_par) == len(row.level2.compared) for row in compared)
        title = chips.SLOT_TITLES[slot].replace(" ", "\u00a0")  # one line per slot
        rows.append(
            [
                f"[{title}](../advanced_guides/measured_combinations.md#{SECTION_OF[slot]})",
                str(chips.option_count(slot)),
                f"{everywhere} of {len(tried)}",
                f"{par} of {len(compared)}" if compared else nothing_compared(slot, members),
            ]
        )
    return _table(PER_SLOT_HEADERS, rows)


def as_shipped(text: str) -> str:
    """Level-2 words with *the method as shipped* for its *reference*: the level-2 data and
    :mod:`grasppanda.combine.reach` call the run of the method as shipped the reference,
    which the evidence page, where "reference" also means the authors' code, spells out."""
    for old, new in (
        ("its reference's", "the shipped method's"),
        ("the reference's", "the shipped method's"),
        ("the reference", "the method as shipped"),
    ):
        text = text.replace(old, new)
    return text


def level2_left_out() -> list[str]:
    """What level 2 does not compare, on every page, in the words of
    :mod:`grasppanda.combine.reach`."""
    from grasppanda.combine.reach import LEVEL2_EXCLUDED, LEVEL2_NONE

    return [
        f"- **Post-processing**: {as_shipped(LEVEL2_NONE['post'])}.",
        f"- **Schedules**: {as_shipped(LEVEL2_NONE['schedule'])}.",
        "- **A cell on a schedule of its own** (a recipe preset with a scheduler, a "
        f"schedule-free optimizer): {as_shipped(LEVEL2_EXCLUDED['schedule'])}.",
        f"- **A swap that did not run**: {as_shipped(LEVEL2_EXCLUDED['runs'])}; it stays named.",
    ]


def level1_iterations(records: Records) -> int:
    """The training iterations of a level-1 probe (those of the first method as shipped)."""
    reference = records.measured.reference(_chips().methods()[0].name) or {}
    return int(reference.get("iterations", 20))


def evidence_index() -> str:
    """``docs/evidence/index.md``: *How we measure*."""
    from grasppanda.combine.reach import REFUSAL_WORDS
    from grasppanda.config import REFUSAL_KINDS

    data = measured()
    records = Records(data["cells"])
    found = where_options_work(records)
    chips = _chips()
    level2 = json.loads(chips.LEVEL2.read_text(encoding="utf-8")) if chips.LEVEL2.is_file() else {}
    about = level2.get("about") or {}
    outcomes = about.get("outcomes") or {}
    iterations = (about.get("settings") or {}).get("iterations", 250)
    filled = {"iterations": iterations, "encoders": encoder_card()}
    cards = [
        f"-   **[{title}]({link})**\n\n    {text.format(**filled)}"
        for title, link, text in EVIDENCE_PAGES
    ]
    lines = [
        "<!-- Generated by tools/make_combination_docs.py from the measured records and the",
        "     words of grasppanda.combine. Do not edit by hand. -->",
        "",
        "# How we measure",
        "",
        coverage(found, not_options(records)),
        "",
        '<div class="grid cards" markdown>',
        "",
        *[line for card in cards for line in (card, "")],
        "</div>",
        "",
        CAVEAT_SNIPPET,
        "",
        "## Slot by slot {#the-answer}",
        "",
        f"The {chips.option_count()} options a user can `--set`, slot by slot:",
        "",
        *matrix(per_slot_summary(found)),
        "",
        '??? note "Which slot applies to which method"',
        "",
        *_indent(matrix(scope_table())),
        "",
        "## The four levels {#levels}",
        "",
        "A cell, one option under one method, is judged at four levels; each needs the one "
        "before it.",
        "",
        *[f"- **`{level}`**: {text}." for level, text in LEVEL_LINES.items()],
        "",
        '??? note "Exact definitions, and why a level can stay undecided"',
        "",
        *_indent([f"- **{level}**: {meaning}." for level, meaning in LEVELS.items()]),
        "",
        *_indent([f"- *{reason}*: {meaning}." for reason, meaning in UNDECIDED.items()]),
        "",
        "## Probe verdicts {#verdicts}",
        "",
        *[f"- `{name}`: {text}." for name, text in VERDICT_LINES.items()],
        "",
        '??? note "Exact definitions"',
        "",
        *_indent([f"- `{name}`: {meaning}" for name, meaning in VERDICTS.items()]),
        "",
        f"## {chips.LEVEL2_HEADER} {{#level-2}}",
        "",
        f"Level 2: a swap that ran trains {iterations} iterations beside the shipped "
        "component, and their losses are compared term by term:",
        "",
        *[f"- **{name}**: {text}." for name, text in LEVEL2_LINES.items()],
        "",
        "Level 2 needs `runs`, not `learns`: a term flat after "
        f"{level1_iterations(records)} iterations can fall by iteration {iterations}.",
        "",
        '??? note "Exact definitions, and what level 2 leaves out"',
        "",
        *_indent([f"- `{name}`: {as_shipped(meaning)}." for name, meaning in outcomes.items()]),
        "",
        *_indent(["Not compared, on every page:", "", *level2_left_out()]),
        "",
        "## Why a request is refused {#refusals}",
        "",
        *[
            f"- **{words}** (`{kind}`): {REFUSAL_KINDS[kind].split(':')[0].split(';')[0]}."
            for kind, words in REFUSAL_WORDS.items()
        ],
        "",
        "## What it was measured on {#measured-on}",
        "",
        f"- **Does it run?** {chips.measured_on(1)[0].upper() + chips.measured_on(1)[1:]}.",
        f"- **Does it train like the shipped component?** "
        f"{chips.measured_on(2)[0].upper() + chips.measured_on(2)[1:]}.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------ the slot index
#
# The first page of the components, docs/slots/index.md, and the option lists that
# tools/make_method_pages.py writes into the slot pages. Their text is written here; nothing
# measured is: an option is a preset (or a component a method lists without one), the scope
# of a slot and where an option works come from tools/docs_chips.py, which reads
# grasppanda.combine.reach, so the index, the slot pages, the explorer and
# `grasppanda compose --list` cannot disagree.


def _chips() -> Any:
    """``tools/docs_chips.py``: methods, slot scopes and chips, the vocabulary of every page."""
    name = "docs_chips"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The registry whose components a slot holds (a recipe slot may name several).
SLOT_REGISTRIES = {
    "image_encoder": ("IMAGE_ENCODERS",),
    "backbone": ("BACKBONES",),
    "proposal": ("PROPOSALS",),
    "view": ("VIEWS",),
    "assigner": ("ASSIGNERS",),
    "grouper": ("GROUPERS",),
    "head": ("HEADS",),
    LOSS_SLOT: ("LOSSES",),
    POST_SLOT: ("POSTPROCESSORS",),
    "optimizer": ("OPTIMIZERS",),
    "schedule": ("SCHEDULERS",),
    "hooks": ("HOOKS",),
    "augment": ("TRANSFORMS",),
    "recipe": ("OPTIMIZERS", "SCHEDULERS", "HOOKS", "TRANSFORMS"),
}
#: What follows ``--set`` for a slot whose key is not its name.
CLI_KEYS = {LOSS_SLOT: "<stage>.<term>_loss"}
#: What an option of a slot is called in a count ("38 options", "17 optimizers").
OPTION_WORDS = {
    "head": ("head bundle", "head bundles"),
    "assigner": ("assigner", "assigners"),
    "optimizer": ("optimizer", "optimizers"),
    "schedule": ("schedule", "schedules"),
    "hooks": ("hook", "hooks"),
    "augment": ("augmentation", "augmentations"),
    "recipe": ("whole recipe", "whole recipes"),
}


@dataclass(frozen=True)
class SlotOption:
    """One option of a slot as the pages list it.

    Attributes:
        slot: The slot (``grasppanda.combine.options.SLOT_KEYS``).
        name: What is written after ``--set <slot>=`` (a loss preset for ``loss``).
        component: The component type(s) it puts there.
        summary: The first paragraph of the preset's header, or the first sentence of the
            component's docstring for a component without a preset.
        source: Where it comes from: the paragraph ``Source: …`` of the preset's header
            when it has one (a preset that is one work's setting of a generic component,
            such as Pointcept's factor written as a ``ParamGroups`` pattern), else the
            catalogue's ``source`` of its component.
        extras: The optional libraries it needs.
        preset: Whether it is a preset (else a component with its constructor defaults).
    """

    slot: str
    name: str
    component: str
    summary: str
    source: str
    extras: tuple[str, ...]
    preset: bool


@lru_cache(maxsize=1)
def _catalogue() -> dict[tuple[str, str], Any]:
    from grasppanda.cli.components import collect

    return {(entry.registry, entry.name): entry for entry in collect()}


def _entry(slot: str, component: str) -> Any:
    """The catalogue entry of a component type, in the registries of ``slot`` first."""
    catalogue = _catalogue()
    for registry in SLOT_REGISTRIES.get(slot, ()):
        if (registry, component) in catalogue:
            return catalogue[registry, component]
    return next((entry for (_, name), entry in catalogue.items() if name == component), None)


#: How a preset header names its own source: a paragraph that starts with these words.
SOURCE_PARAGRAPH = "Source: "


def declared_source(preset: Any) -> str:
    """The paragraph ``Source: …`` of a preset's header, without the label and the final
    full stop; ``""`` when the header has none."""
    for paragraph in getattr(preset, "about", ()):
        if paragraph.startswith(SOURCE_PARAGRAPH):
            return paragraph[len(SOURCE_PARAGRAPH) :].strip().removesuffix(".")
    return ""


@lru_cache(maxsize=32)
def slot_options(slot: str) -> tuple[SlotOption, ...]:
    """Every option of ``slot``, sorted by name: its presets (a recipe preset for a recipe
    slot, a loss preset for ``loss``) and every component a method lists without a preset."""
    found: dict[str, SlotOption] = {}
    members = {**presets().get(slot, {}), **recipes().get(slot, {})}
    for name, preset in members.items():
        parts = preset.type.split(", ")
        entries = [entry for entry in (_entry(slot, part) for part in parts) if entry]
        extras = sorted({extra for entry in entries for extra in entry.extras})
        found[name] = SlotOption(
            slot,
            name,
            preset.type,
            preset.summary,
            declared_source(preset) or (entries[0].source if entries else ""),
            tuple(extras),
            True,
        )
    if slot != LOSS_SLOT:
        for method in option_lists.methods():
            for option in option_lists.options(method, slot):
                if option.name in found:
                    continue
                entry = _entry(slot, option.component)
                found[option.name] = SlotOption(
                    slot,
                    option.name,
                    option.component,
                    entry.summary if entry else "",
                    entry.source if entry else "",
                    tuple(entry.extras) if entry else (),
                    option.preset,
                )
    return tuple(sorted(found.values(), key=lambda option: option.name.lower()))


def option_count(slot: str) -> str:
    """``38 options``, ``17 optimizers``: how many options a slot has, in words. The count
    is ``docs_chips.option_count``, the one every page and the explorer use: what a user
    can ``--set`` (the adapters compose inserts and the point-painting wrappers are not
    counted, though the overview of the scene encoders lists them)."""
    one, many = OPTION_WORDS.get(slot, ("option", "options"))
    count = _chips().option_count(slot)
    return f"{count} {one if count == 1 else many}"


class Card(NamedTuple):
    """One card of the slot index: a slot page and the slots it documents."""

    title: str
    page: str
    slots: tuple[str, ...]
    icon: str
    task: str
    more: tuple[tuple[str, str], ...] = ()


#: The cards of the slot index, in the order data flows, then the training recipe. The task
#: of a card is its one sentence; everything else on the card is computed.
SLOTS: dict[str, Card] = {
    "image_encoder": Card(
        "Image encoder",
        "image_encoders.md",
        ("image_encoder",),
        "image-outline",
        "Turns the colour image into feature maps; HGGD predicts its grasp heat-map from them.",
    ),
    "backbone": Card(
        "Scene encoder",
        "scene_encoders.md",
        ("backbone",),
        "cube-scan",
        "One feature vector per point of the depth cloud: what a seed knows before any crop.",
        (
            ("point-based", "scene_encoders_point.md"),
            ("sparse-voxel", "scene_encoders_sparse.md"),
            ("transformers", "scene_encoders_transformer.md"),
        ),
    ),
    "proposal": Card(
        "Seed proposal",
        "proposal.md",
        ("proposal",),
        "target",
        "Picks the points of the scene at which grasps are predicted.",
    ),
    "view": Card(
        "View",
        "view.md",
        ("view",),
        "compass-outline",
        "Chooses the approach direction of every seed.",
    ),
    "grouper": Card(
        "Grouper",
        "grouper.md",
        ("grouper",),
        "select-group",
        "Gathers the points around a seed into the feature the head reads.",
    ),
    "head": Card(
        "Heads and assigners",
        "head_loss.md",
        ("head", "assigner"),
        "crosshairs-gps",
        "The head predicts the grasp at a seed; the assigner matches seeds with labels in "
        "training.",
    ),
    LOSS_SLOT: Card(
        "Losses",
        "losses.md",
        (LOSS_SLOT,),
        "chart-bell-curve",
        "One term of the loss of a stage, replaced on its own.",
    ),
    POST_SLOT: Card(
        "Post-processing",
        "post.md",
        (POST_SLOT,),
        "filter-outline",
        "Filters, selects and corrects predicted grasps before a robot or an evaluator.",
    ),
    "recipe": Card(
        "Training recipe",
        "recipe.md",
        ("optimizer", "schedule", "hooks", "augment", "recipe"),
        "tune-variant",
        "How weights are updated, how the rate moves, what runs around the loop, how a "
        "sample is perturbed.",
    ),
}


def method_page(method: str) -> str:
    """The stem of the page of a method under ``docs/methods/``: its config directory."""
    return {**METHODS, **FAMILIES}[method].split("/")[0]


def _in_tree_module(cls: Any) -> bool:
    return cls.__module__.split(".")[0] == "grasppanda"


def seeds_per_frame() -> str:
    """How many seeds the seed-based methods pick per frame, read from their configs."""
    from grasppanda.config import Config

    found = {
        Config.load(config)["model"]["proposal"].get("num_seeds") for config in METHODS.values()
    }
    counts = sorted(str(count) for count in found if count)
    if len(counts) == 1:
        return f"{counts[0]} per frame in each of the {_number(len(METHODS))} seed-based methods"
    return f"{' to '.join([counts[0], counts[-1]])} per frame in the seed-based methods"


def _card(card: Card) -> list[str]:
    chips = _chips()
    scopes = list(dict.fromkeys(chips.scope_chip(slot) for slot in card.slots))
    keys = " ".join(f"`--set {CLI_KEYS.get(slot, slot)}=`" for slot in card.slots[:2])
    counts = " · ".join(option_count(slot) for slot in card.slots)
    more = " · ".join(f"[{label}]({page})" for label, page in card.more)
    return [
        f"-   :material-{card.icon}:{{ .lg .middle }} **[{card.title}]({card.page})**",
        "",
        "    ---",
        "",
        f"    {card.task}" + (f" By family: {more}." if more else ""),
        "",
        f"    {' '.join(scopes)} {counts}",
        "",
        f"    {keys}" + (" …" if len(card.slots) > 2 else ""),
        "",
    ]


def slots_index(records: Records | None = None) -> str:
    """docs/slots/index.md. It shows no measured value, so ``records`` changes nothing."""
    del records
    chips = _chips()
    infos = chips.methods()
    seed = [info.name for info in infos if info.family == "seed-based"]
    own = [info.name for info in infos if info.family != "seed-based"]
    total = chips.option_count()
    scopes = {key: {chips.scope(slot) for slot in card.slots} for key, card in SLOTS.items()}
    for_all = [card.title for key, card in SLOTS.items() if scopes[key] == {"all"}]
    for_all = [for_all[0], *[f"the {title.lower()}" for title in for_all[1:]]]
    hggd = [card.title.lower() for key, card in SLOTS.items() if scopes[key] == {"hggd"}]
    links = " · ".join(f"[{info.name}](../{info.page}#the-slots)" for info in infos)
    lines = [
        "---",
        "hide: [toc]",
        "---",
        "",
        # The title first: mkdocs takes a page's title from a heading only when it leads.
        "# Components",
        "",
        "<!-- Generated by tools/make_combination_docs.py from the presets, the option lists of",
        "     the methods and tools/docs_chips.py. Do not edit by hand. -->",
        "",
        "GraspPanda builds a grasp detector from stages, and every stage is a **slot**: a place "
        "in the config where one component can be swapped for another with one line, "
        f"`--set <slot>=<option>`. {_number(len(seed)).capitalize()} methods share the seed-based "
        "skeleton and its "
        f"slots; {' and '.join(own)} reads images and has one slot of its own, the "
        f"{' and the '.join(hggd)}. {_join_words(for_all)} apply to all "
        f"{_number(len(infos))} methods. In all, {total} options you can `--set`.",
        "",
        *pipeline_figure(),
        "",
        "## Which method has which slot",
        "",
        *scope_table(),
        "",
        "## The slots",
        "",
        '<div class="grid cards" markdown>',
        "",
        *[line for card in SLOTS.values() for line in _card(card)],
        "</div>",
        "",
        "## Find your way",
        "",
        "- **Will an option work under my method?** The [explorer](../explorer/index.md) "
        "shows what was measured for every option under every method, with the command.",
        f"- **What do *runs* and *{chips.LEVEL2_NAME}* mean?** "
        "[How we measure](../evidence/index.md) defines every word of the tables.",
        f"- **Every slot of one method:** {links}.",
        "",
        "## Words used on these pages",
        "",
        '??? note "Seed, preset, contract, adapter, label set"',
        "",
        "    - **Seed**: a point of the scene at which a grasp is predicted; the seed proposal "
        f"picks them ({seeds_per_frame()}).",
        "    - **Preset**: a small YAML file with a component and settings for table-top depth "
        "clouds; its name is what follows `--set <slot>=`.",
        "    - **Contract**: what a component declares about its input and output. `grasppanda "
        "compose` compares contracts before anything is built and refuses a mismatch with the "
        "fix.",
        "    - **Adapter**: a wrapper `compose` puts around a scene encoder so that it fits the "
        "next stage.",
        "    - **Label set**: the grasp labels a head is trained on, *dense* (every view, angle "
        "and depth of a labelled point) or *compact* (the best grasp per view). A head, its "
        "assigner and the dataset section use the same one.",
    ]
    return "\n".join(lines).rstrip() + "\n"


#: The pipeline figure of ``tools/make_brand_assets.py``, one file per colour scheme.
PIPELINE = ("../assets/pipeline.png", "../assets/pipeline-dark.png")
PIPELINE_ALT = (
    "Two detector families fed by GraspNet-1Billion and GraspClutter6D: the slots of the "
    "seed-based skeleton, HGGD's image encoder and fixed stages, and the post-processing and "
    "training recipe all methods share"
)


def pipeline_figure() -> list[str]:
    """The pipeline in the reader's colour scheme, each image a link to itself at full size
    (on a phone the figure is too small to read)."""
    light, dark = PIPELINE
    return [
        f"[![{PIPELINE_ALT}]({light}#only-light)]({light})",
        f"[![{PIPELINE_ALT}]({dark}#only-dark)]({dark})",
        "",
        f"<small>Open the figure at full size: [light]({light}) · [dark]({dark}).</small>",
    ]


def _number(count: int) -> str:
    """A small count in words, as a sentence writes it."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return words[count] if 0 <= count < len(words) else str(count)


def scope_table() -> list[str]:
    """``docs_chips.scope_matrix`` without its column "Applies to": the marks under the six
    methods say it, every card names the scope as a chip, and eight columns do not fit the
    width of a page."""

    def without_scope(line: str) -> str:
        cells = line.split("|")
        return "|".join([*cells[:2], *cells[3:]])

    return [without_scope(line) for line in _chips().scope_matrix("../")]


def _join_words(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


# ------------------------------------------------------------------------- staleness


def experimental() -> dict[Path, str]:
    """Every composed config under ``configs/experimental`` → what compose writes for it."""
    return {path: recompose(path).text(path) for path in sorted(EXPERIMENTAL.glob("*.yaml"))}


def expected() -> dict[Path, str]:
    pages = experimental()
    if DATA.is_file():
        pages[PAGE] = page()
        pages[EVIDENCE_INDEX] = evidence_index()
        pages[SLOTS_INDEX] = slots_index(Records(measured()["cells"]))
    return pages


def stale() -> list[Path]:
    return [
        path
        for path, content in expected().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    args = parser.parse_args(argv)
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"stale: {path.relative_to(REPO)} (run python -m tools.make_combination_docs)")
        return 1 if outdated else 0
    for path, content in expected().items():
        changed = not path.is_file() or path.read_text(encoding="utf-8") != content
        path.write_text(content, encoding="utf-8")
        print(f"{'wrote' if changed else 'unchanged'} {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
