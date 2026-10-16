"""Generate what follows from the presets and from the measured grid.

* ``docs/advanced_guides/measured_combinations.md`` is written whole from
  ``docs/advanced_guides/measured_combinations.json``, the compact record of
  ``tools/probe_grid.py``: for every slot a table component by method with the measured
  verdict, the measured class of every component, the reasons of every refusal and
  failure, and the numbers the probes saw. Nothing on the page is inferred: a cell without
  a record is shown as not probed.
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
from functools import lru_cache
from pathlib import Path
from typing import Any

from grasppanda.combine import options as option_lists
from grasppanda.combine.compose import recompose
from grasppanda.combine.grid import (
    FAMILIES,
    LOSS_CELLS,
    LOSS_NOT_PROBED,
    METHODS,
    POST_METHODS,
    POST_NOT_PROBED,
    STAGE_SLOTS,
    Cell,
    cells,
    encoder_presets,
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
    cell_levels,
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


def _label(component: str, variant: str, slot: str, *, short: bool = False) -> str:
    preset = variant in presets().get(slot, {})
    if short:
        return f"`{component}` (`{variant}`)" if preset else f"`{component}` (defaults)"
    return f"`{component}`<br>" + (f"preset `{variant}`" if preset else "constructor defaults")


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


def glance_section(records: Records) -> list[str]:
    """Every component of every slot under its class, one table per level."""
    by_slot = []
    for slot in SLOT_TITLES:
        group = "encoder" if slot == "backbone" else "stage"
        classes = slot_classes(slot, records, group)
        several = Counter(component for component, *_ in classes)

        def name_of(component: str, variant: str, slot: str = slot, several: Any = several) -> str:
            if several[component] > 1:
                return _label(component, variant, slot, short=True)
            return f"`{component}`"

        by_slot.append(([SLOT_TITLES[slot]], [(name_of(c, v), found) for c, v, found in classes]))
    return [CAVEAT_SENTENCE, "", *class_tables(by_slot, ["Slot"])]


def slot_section(slot: str, records: Records, group: str) -> list[str]:
    """The table, the classes and the numbers of one slot."""
    notes = Notes()
    rows, numbers = [], []
    classes = {(c, v): found for c, v, found in slot_classes(slot, records, group)}
    for (component, variant), members in _rows(slot, group).items():
        label = _label(component, variant, slot)
        row = [label]
        per_method = {cell.method: records.get(cell.id) for cell in members}
        for method in METHODS:
            record = per_method.get(method)
            own = records.own(record)
            row.append(verdict_text(record, own, notes, records))
            if record is not None and record["verdict"] in WORKS and not own:
                name = f"{label.replace('<br>', ', ')} under {method}"
                numbers.append(_numbers_row(name, records.probed(record), records.levels(record)))
        rows.append([*row, class_text(classes[component, variant])])
    lines = [f"### {SLOT_TITLES[slot]}", ""]
    lines += _table(["Component", *METHODS, "Class"], rows)
    lines += ["", CAVEAT_SENTENCE]
    bare = [component for where, component in bare_components() if where == slot]
    if bare:
        lines += [
            "",
            "No row for " + ", ".join(f"`{name}`" for name in bare) + " named by its type: "
            "it needs an option only a preset supplies, so the bare request is refused for the "
            "missing option and is not a combination. Its presets have rows.",
        ]
    if notes.items:
        lines += ["", *notes.lines()]
    if numbers:
        lines += ["", f'??? note "What the probes of the {SLOT_TITLES[slot].lower()} saw"', ""]
        lines += _indent(_table(NUMBER_HEADERS, numbers))
    return [*lines, ""]


NUMBER_HEADERS = [
    "Combination",
    "Parameters",
    "Total loss, first → last",
    "Real seeds, min",
    "Supervised seeds, mean",
    "Grasps per frame",
    "Fall per loss term §",
    "Learns, predicts §",
    "s / iteration ‡",
    "GPU GiB ‡",
]
#: The judge-2 columns of a record of the first judge, which recorded none of it.
NOT_CHECKED = "not checked"


def term_falls(record: dict[str, Any]) -> str:
    """The fall of every loss term of a record of judge 2 (`loss/` left out), with the
    number of iterations it was active in where that is fewer than ran; :data:`NOT_CHECKED`
    for a record of the first judge."""
    if record.get("judge", 1) < JUDGE:
        return NOT_CHECKED
    active = record.get("active") or {}
    found = []
    for term, values in (record.get("trend") or {}).items():
        if not term.startswith("loss/"):
            continue
        name = term.removeprefix("loss/")
        fall = values[2]
        text = f"{name} " + ("never active" if fall is None else f"{fall * 100:.0f} %")
        if term in active:
            text += f" (active {active[term]})"
        found.append(text)
    return ", ".join(found) or "—"


def learning_text(record: dict[str, Any], levels: CellLevels | None = None) -> str:
    """Whether a probe learns and predicts (grasppanda.combine.levels), with what kept it
    from either; :data:`NOT_CHECKED` for a record of the first judge."""
    if record.get("judge", 1) < JUDGE:
        return NOT_CHECKED
    levels = levels or cell_levels(record)
    parts = []
    for level in ("learns", "predicts"):
        state = levels.states.get(level, "")
        if state == YES:
            parts.append(f"{level} ✓")
        elif state == NO:
            parts.append(f"{level} ✗: " + "; ".join(levels.why.get(level, [])))
        else:
            parts.append(f"{level}: {state}")
    return "; ".join(parts + levels.notes)


def _numbers_row(label: str, record: dict[str, Any], levels: CellLevels | None = None) -> list[str]:
    loss = record.get("losses", {}).get("loss")
    real, supervised = record.get("real_seeds"), record.get("supervised_seeds")
    shared = record.get("shared_machine", {})
    seconds = shared.get("seconds_per_iteration")
    memory = shared.get("peak_gpu_mib")
    grasps = ", ".join(str(count) for count in record.get("grasps_per_frame", [])) or "—"
    return [
        label,
        f"{record['parameters']:,}".replace(",", " ") if record.get("parameters") else "—",
        f"{loss[0]:.4g} → {loss[1]:.4g}" if loss else "—",
        f"{real['min']:.2f}" if real else "—",
        f"{supervised['mean']:.2f}" if supervised else "—",
        grey(grasps) if no_grasp(record) else grasps,
        term_falls(record),
        learning_text(record, levels),
        f"{seconds:.2f}" if seconds is not None else "—",
        f"{memory / 1024:.1f}" if memory is not None else "—",
    ]


def adapters_section(records: list[dict[str, Any]]) -> list[str]:
    """What was measured with every encoder adapter compose inserted."""
    from grasppanda.combine.compose import is_adapter
    from grasppanda.registry import BACKBONES

    rows = []
    members = presets().get("backbone", {}).values()
    for name in BACKBONES.names():
        cls = BACKBONES.get(name)
        if cls.__module__.split(".")[0] != "grasppanda" or not is_adapter(cls):
            continue
        if not getattr(cls, "compose_inserts", True):  # a wrapper a user writes down
            continue
        with_it = [record for record in records if record.get("adapter") == name]
        counts = Counter(record["verdict"] for record in with_it)
        under = [m for m in METHODS if any(record["method"] == m for record in with_it)]
        preset = next((p.name for p in members if p.type == name), None)
        rows.append(
            [
                f"`{name}`" + (f"<br>preset `{preset}`" if preset else ""),
                str(len(with_it)),
                *(str(counts[verdict]) if counts[verdict] else "—" for verdict in VERDICT_ORDER),
                ", ".join(under) or "—",
            ]
        )
    headers = ["Adapter", "Cells it was inserted in", *(f"`{v}`" for v in VERDICT_ORDER)]
    return [
        "### Encoder adapters",
        "",
        "An adapter wraps an encoder and is not one itself, so it has no row above: compose "
        "inserts it where the contract between encoder and proposal asks for it (`✓ in "
        "Adapter`). What was measured with each, over all groups of the grid:",
        "",
        *_table([*headers, "Under"], rows),
        "",
    ]


def references_section(records: Records) -> list[str]:
    notes = Notes()
    rows = []
    for method, config in [*METHODS.items(), *FAMILIES.items()]:
        group = "reference" if method in METHODS else "family"
        record = records.get(f"{group}/{method}")
        verdict = verdict_text(record, False, notes, records)
        works = record is not None and record["verdict"] in WORKS
        numbers = _numbers_row("", record, records.levels(record))[1:] if works else []
        rows.append(
            [method, f"`{config}`", verdict, *(numbers or ["—"] * (len(NUMBER_HEADERS) - 1))]
        )
    lines = _table(["Method", "Config", "Verdict", *NUMBER_HEADERS[1:]], rows)
    return [*lines, *(["", *notes.lines()] if notes.items else []), ""]


def stage_sets_section(records: Records) -> list[str]:
    wanted = grid("stage-set")
    recorded = [cell for cell in wanted if records.get(cell.id) is not None]
    if not recorded:
        return [
            f"Not probed: the {len(wanted)} sets of stage variants the contract rules accept "
            "were left out of this run. `python -m tools.probe_grid run --groups stage-set` "
            "probes them.",
            "",
        ]
    notes = Notes()
    rows, numbers = [], []
    for cell in recorded:
        record = records.get(cell.id)
        chosen = dict(item.split("=", 1) for item in cell.sets)
        own = records.own(record)
        rows.append(
            [
                cell.variant,
                *(f"`{chosen[slot]}`" for slot in ("backbone", *STAGE_SLOTS)),
                cell.method,
                verdict_text(record, own, notes, records),
            ]
        )
        if record["verdict"] in WORKS and not own:
            name = " + ".join(f"`{chosen[slot]}`" for slot in ("backbone", *STAGE_SLOTS))
            numbers.append(_numbers_row(name, records.probed(record), records.levels(record)))
    counts = Counter(records.get(cell.id)["verdict"] for cell in recorded)
    summary = ", ".join(f"{counts[v]} `{v}`" for v in VERDICT_ORDER if counts[v])

    def variants(members: list[Cell]) -> set[tuple[str, str]]:
        """``(kind of encoder, slot=choice)`` of every stage variant in some of the sets."""
        return {(cell.variant, item) for cell in members for item in cell.sets[1:]}

    listed, seen = variants(wanted), variants(recorded)
    headers = ["Encoder output", "Encoder", "Proposal", "View stage", "Assigner", "Grouper"]
    lines = [
        f"The contract rules accept {len(wanted)} sets of stage variants, too many to probe "
        f"each: **{len(recorded)} were probed** ({summary}) and "
        f"**{len(wanted) - len(recorded)} were not**. The probed sets were picked so that a "
        "stage variant that occurs in a legal set occurs in a probed one, under a dense and "
        "under a sub-sampling encoder: "
        f"{len(seen)} of the {len(listed)} pairs of variant and kind of encoder are in a "
        "probed set. That a set runs says nothing about another set with the same parts. The "
        'sets that were not probed are the sets of `grasppanda.combine.grid.cells(("stage-set",))` '
        "without a record (`about.stage_sets_not_probed` of the "
        "[measured file](measured_combinations.json) counts them); "
        "`python -m tools.probe_grid run --only '<id>'` probes one.",
        "",
        "A set states all five stages and the encoder: the preset named, under which the set "
        "needs no adapter. "
        "It was composed onto the method config whose data recipe serves its labels (last but "
        "one column), so the optimizer, the schedule and the data pipeline are that method's.",
        "",
        *_table([*headers, "Head bundle", "Data recipe of", "Verdict"], rows),
    ]
    if notes.items:
        lines += ["", *notes.lines()]
    if numbers:
        lines += ["", '??? note "What the probes of the stage sets saw"', ""]
        lines += _indent(_table(NUMBER_HEADERS, numbers))
    return [*lines, ""]


#: Commentary on a known gap of the config check, printed under the failures whose reason
#: contains the key. It explains a measurement; it is not one.
KNOWN_GAPS: dict[str, str] = {}
"""Fragment of a failure message -> why the config check did not see it. Empty: the two
gaps probing found (the width a grouper fixes, the labels a data recipe serves) are
contract rules now, and a cell that meets one is `refused`."""


#: What a cell of the grid without a record says in the tables of post, loss and recipe.
NOT_YET = "not probed yet"


def _waiting(group: str, records: Records) -> list[str]:
    """One sentence on the cells of a group that have no record yet; none when all have."""
    wanted = grid(group)
    missing = sum(records.get(cell.id) is None for cell in wanted)
    if not missing:
        return []
    return [
        f"{missing} of the {len(wanted)} cells of this group are {NOT_YET}: the grid gained "
        "them after the run that measured the others. They are listed in `about.not_probed` "
        "of the [measured file](measured_combinations.json); `python -m tools.probe_grid run "
        f"--groups {group}` probes them.",
        "",
    ]


def post_section(records: Records) -> list[str]:
    """Post-processing presets as the variant ``post``: verdict and grasps left, per method."""
    members = grid("post")
    if not any(records.get(cell.id) is not None for cell in members):
        return []
    rows: dict[str, list[str]] = {}
    for cell in members:
        row = rows.setdefault(cell.variant, [f"`{cell.variant}` ({cell.component})"])
        record = records.get(cell.id)
        if record is None:
            row.append(NOT_YET)
            continue
        counts = (records.probed(record).get("postprocessing") or {}).get("grasps") or []
        left = "; ".join(f"{frame[0]} → {frame[-1]}" for frame in counts)
        text = record["verdict"] + (f": {left}" if left else "")
        if record["verdict"] in ("fails", "refused"):
            text += f". {reason(record)}"
        row.append(text)
    return [
        "## Post-processing",
        "",
        "Every post-processing preset as the evaluation variant `post` of every method "
        "(`grasppanda compose <config> --set post=<preset>`), applied by the probe to the "
        "predictions of its two frames: grasps predicted → grasps left. A post-processor "
        "reads grasps and the observation, never a network, so it needs no training of its "
        "own: a cell is the method's prediction plus the chain. The grasps are those of a "
        "detector trained for the iterations of the probe; "
        "[the page of the slot](../slots/post.md) has the same steps on the predictions of "
        "released weights.",
        "",
        *_waiting("post", records),
        *_table(["preset", *POST_METHODS], list(rows.values())),
        "",
    ]


def loss_section(records: Records) -> list[str]:
    """Loss presets in the loss terms they were probed in: one term of every kind a preset
    accepts under every method, then the hand-picked cells with the value of every term."""
    from grasppanda.models.losses.terms import TERM_KINDS

    members = grid("loss")
    if not any(records.get(cell.id) is not None for cell in members):
        return []
    notes = Notes()
    terms = {method: loss_terms(method) for method in METHODS}
    kinds = [kind for kind in TERM_KINDS if any(kind in found for found in terms.values())]
    listed = {cell.id for cell in members}
    shown: set[str] = set()
    one_term = []
    for preset in presets().get(LOSS_SLOT, {}):
        if preset in LOSS_NOT_PROBED:
            continue
        row = [f"`{preset}`"]
        for method in METHODS:
            found = []
            for kind in kinds:
                cell_id = f"loss/{method}/{preset}@{terms[method].get(kind)}"
                if kind in terms[method] and cell_id in listed:
                    record = records.get(cell_id)
                    text = verdict_text(record, False, notes, records) if record else NOT_YET
                    found.append(f"{kind} {text}")
                    shown.add(cell_id)
            row.append("<br>".join(found) or "—")
        one_term.append(row)
    picked = {(method, tuple(sets)) for method, sets in LOSS_CELLS}
    rows = []
    for cell in members:
        record = records.get(cell.id)
        if (cell.method, cell.sets) not in picked or cell.id in shown or record is None:
            continue
        losses = records.probed(record).get("losses") or {}
        verdict = verdict_text(record, False, notes, records)
        for item in cell.sets:
            target, _, preset = item.partition("=")
            stage, _, option = target.partition(".")
            term = "loss/" + (option.removesuffix("_loss") if option != "loss" else stage)
            values = losses.get(term)
            rows.append(
                [
                    f"`{preset}`",
                    cell.method,
                    f"`{target}`",
                    verdict,
                    f"{values[0]:.4g} → {values[1]:.4g}" if values else "—",
                ]
            )
        rows.append(["", "", "", "", ""])
    rule = [
        [method, *(f"`{terms[method][kind]}`" if kind in terms[method] else "—" for kind in kinds)]
        for method in METHODS
    ]
    lines = [
        "## Losses",
        "",
        "Every loss preset in every kind of loss term it accepts, under every seed-based method "
        "that has a term of that kind (`grasppanda compose <config> --set <stage>.<loss "
        "option>=<preset>`). Which loss fits which kind of term is a contract "
        "([Heads and losses](../slots/head_loss.md)); the unit tests run every loss in every "
        "term of its kind on tiny models. A method has several terms of some kinds, so a cell "
        "replaces **one representative term per kind and method: the last term of that kind "
        "in the order of the pipeline** (proposal, view, head; within a stage, the order the "
        "stage declares its terms in). That is the term nearest the grasp, where the methods "
        "differ; GSNet, EconomicGrasp and FineGrasp share their proposal. The representative "
        "terms:",
        "",
        *_table(["Method", *(f"`{kind}` term" for kind in kinds)], rule),
        "",
        *_waiting("loss", records),
        *_table(["preset", *METHODS], one_term),
        "",
        "The hand-picked cells below replace several terms of a method at once, or pair a "
        "loss with a term the config check refuses (the last rows): the rows between two empty "
        "lines are one probed config and share its verdict. The last column is the weighted "
        "value of the replaced term at the first and at the last of the 20 iterations: it "
        "shows that the term is finite and moves, nothing about accuracy, and values of "
        "different losses are not comparable.",
        "",
        *_table(["preset", "method", "term", "verdict", "term value, first → last"], rows[:-1]),
        "",
    ]
    return [*lines, *notes.lines(), *([""] if notes.items else [])]


def cross_section(records: Records) -> list[str]:
    """Requests that replace options of several slots at once: the request and its verdict."""
    wanted = grid("cross")
    if not any(records.get(cell.id) is not None for cell in wanted):
        return []
    notes = Notes()
    rows, numbers = [], []
    for cell in wanted:
        record = records.get(cell.id)
        request = " ".join(f"`{item}`" for item in cell.sets)
        rows.append([cell.method, request, verdict_text(record, False, notes, records)])
        if record is not None and record["verdict"] in WORKS:
            levels = records.levels(record)
            numbers.append(
                _numbers_row(f"{cell.method}: {request}", records.probed(record), levels)
            )
    lines = [
        "## Several slots at once",
        "",
        "The tables above replace one thing at a time, and the stage sets keep the encoder to "
        "two. These cells replace an encoder together with options of other slots (a grouper, "
        "a proposal, a view stage, a recipe preset, a post-processing preset), each "
        "as one request: `grasppanda compose <method config> --set <slot>=<option> --set …`. "
        "They are a hand-picked sample, at least two per method, of what a visitor composes; "
        "a pair or a triple that is not listed here was not probed, and its verdict does "
        "not follow from the cells of its parts. The [explorer](../explorer/index.md) shows "
        "what the config check answers for any two options.",
        "",
        *_table(["Method", "Request", "Verdict"], rows),
    ]
    if notes.items:
        lines += ["", *notes.lines()]
    if numbers:
        lines += ["", '??? note "What the probes of these cells saw"', ""]
        lines += _indent(_table(NUMBER_HEADERS, numbers))
    return [*lines, ""]


def findings_section(records: list[dict[str, Any]]) -> list[str]:
    """Cells that passed the config check and failed afterwards, grouped by reason."""
    failed = [record for record in records if record["verdict"] == "fails"]
    if not failed:
        return ["No combination that passed the config check failed afterwards.", ""]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in failed:
        groups.setdefault(re.sub(r"\d+", "N", reason(record)), []).append(record)
    lines = [
        f"{len(failed)} cell(s) passed the config check and failed afterwards. A combination "
        "should be refused with the conflict and the fix, or run; each group below is a gap "
        "between what the config check knows and what the components need. (*N* stands for "
        "a number that differs from cell to cell.)",
        "",
    ]
    for key, members in groups.items():
        message = (members[0].get("failure") or {}).get("message", "")
        listed = ", ".join(f"`{record['id']}`" for record in members)
        lines += [f"- **{key}** {len(members)} cell(s): {listed}.", ""]
        lines += [f"    The message of the first, whole: *{message}*", ""]
        for fragment, comment in KNOWN_GAPS.items():
            if fragment in key:
                lines += [f"    {comment}", ""]
    return lines


def family_slots_section(records: Records) -> list[str]:
    """Every preset of every component slot of a detector family of its own."""
    members = grid("family-slot")
    if not members:
        return []
    notes = Notes()
    lines = [
        "## Other detector families: one component replaced",
        "",
        "A detector that is not built on the seed-based skeleton has slots of its own; HGGD has "
        "one, the encoder of its image network ([Image encoders](../slots/image_encoders.md)). "
        "Every preset of such a slot was composed into the family's shipped config and probed:",
        "",
    ]
    for method, slot in dict.fromkeys((cell.method, cell.slot) for cell in members):
        rows = []
        for cell in members:
            if (cell.method, cell.slot) != (method, slot):
                continue
            record = records.get(cell.id)
            verdict = verdict_text(record, False, notes, records)
            works = record is not None and record["verdict"] in WORKS
            numbers = (
                _numbers_row("", records.probed(record), records.levels(record))[1:]
                if works
                else ["—"] * (len(NUMBER_HEADERS) - 1)
            )
            rows.append([f"`{cell.variant}`", f"`{cell.component}`", verdict, *numbers])
        headers = [f"`{slot}=`", "Component", method, *NUMBER_HEADERS[1:]]
        lines += [f"### {method}: `{slot}`", "", *_table(headers, rows), ""]
    return [*lines, *([*notes.lines(), ""] if notes.items else [])]


def recipes_section(records: Records) -> list[str]:
    """Recipe presets: one row per preset, one column per method it is probed under."""
    by_id = records.by_id
    members = grid("recipe")
    if not members:
        return []
    methods = list(dict.fromkeys(cell.method for cell in members))
    rows: dict[tuple[str, str], dict[str, str]] = {}
    sets: dict[tuple[str, str], str] = {}
    for cell in members:
        record = by_id.get(cell.id)
        rows.setdefault((cell.slot, cell.variant), {})[cell.method] = (
            f"`{record['verdict']}`" if record else NOT_YET
        )
        sets[(cell.slot, cell.variant)] = " ".join(f"`--set {item}`" for item in cell.sets)
    lines = [
        "A recipe preset changes what is around the model: the optimizer, the schedule, the "
        "hooks of the training loop, the point augmentation. Each is applied alone to every "
        "method as shipped and probed like a stage; an augmentation a method's data contract "
        "does not accept is refused there. `—`: the preset names the scene encoder of the "
        "seed-based skeleton, which HGGD does not have. A preset that sets the schedule (the "
        "schedule presets, the schedule-free optimizers, the recipes) is probed with that "
        "schedule, `schedule: recipe` in its record, since the schedule is what its cell "
        "measures; the others with the schedule of the grid (*A cell* above). What each "
        "preset is and how to use it: [the recipe slot](../slots/recipe.md).",
        "",
        *_waiting("recipe", records),
    ]
    table = [
        [slot, sets[(slot, name)], cell_.component, *(found.get(m, "—") for m in methods)]
        for (slot, name), found in rows.items()
        for cell_ in [next(c for c in members if (c.slot, c.variant) == (slot, name))]
    ]
    lines += _table(["Slot", "Request", "Sets", *methods], table)
    failed = [
        by_id[cell.id]
        for cell in members
        if cell.id in by_id and by_id[cell.id]["verdict"] not in ("runs", "runs-empty", "wired")
    ]
    if failed:
        lines += ["", "What did not run, in the words of the tool:", ""]
        for record in failed:
            text = _one_line(record.get("unstable") or reason(record) or "no message recorded")
            lines.append(f"- `{record['id']}` (`{record['verdict']}`): {text}")
    return [*lines, ""]


def _one_line(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


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
    return _table(["Group", "Cells probed", *(f"`{v}`" for v in VERDICT_ORDER)], rows)


def _refused_by(records: list[dict[str, Any]]) -> str:
    """Which command refused the refused cells of the file."""
    refused = [record for record in records if record["verdict"] == "refused"]
    by_probe = sum("probe" in record for record in refused)
    if not refused:
        return ""
    if not by_probe:
        return (
            f" On this page all {len(refused)} refusals come from `grasppanda compose`, which "
            "runs that check on the composed config and writes nothing when it fails."
        )
    return (
        f" On this page {len(refused) - by_probe} refusals come from `grasppanda compose` and "
        f"{by_probe} from the probe's check of what `train` needs."
    )


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


#: The fields judge 2 adds to a record, in the order the page lists them, and how the
#: learning rate of the probe was set.
JUDGE_RECORD_FIELDS = (
    "judge",
    "trend",
    "active",
    "flags",
    "no_gradient",
    "unchanged",
    "checked",
    "score_quantiles",
    "learning_rate, lr_factor",
    "schedule, rate",
    "losses_per_iteration",
)


def judges_section(records: list[dict[str, Any]]) -> list[str]:
    """Which judge gave the verdicts of the page, and what judge 2 records beyond them."""
    fields = _runner().FIELDS
    second = [record for record in records if record.get("judge", 1) >= JUDGE]
    first = [record for record in records if record.get("judge", 1) < JUDGE]
    empty = sum(
        record["verdict"] == "runs"
        and bool(record.get("grasps_per_frame"))
        and not any(record["grasps_per_frame"])
        for record in first
    )
    checked = sum("no_gradient" in record for record in first)
    side = DATA.with_suffix("") / _runner().SERIES_NAME
    series = f"measured_combinations/{side.name}"
    where = f"[{series}]({series})" if side.is_file() else f"`{series}`"
    meanings = {
        **{name: fields[name] for name in JUDGE_RECORD_FIELDS},
        "flags": "per loss term with a flag: the flags of the list below that hold",
        "losses_per_iteration": (
            f"not in the record: the side file {where} holds, for every record of judge 2 "
            "with a probe of its own, every loss term and the total at every iteration, to "
            "four significant digits (null where the term was not reported). It is written "
            "with the records and holds what the trends are computed from"
        ),
    }
    lines = [
        "## Judges",
        "",
        "A probe is judged by `grasppanda.combine.probe`. The verdicts above and the fields "
        f"below are those of **judge {JUDGE}**, from 2026-10-15; **{len(second)} of the "
        f"{len(records)} records** on this page were judged by it (`judge: {JUDGE}` in the "
        "[measured file](measured_combinations.json)). The other "
        f"{len(first)} were judged by the first judge. It had no `runs-empty`, so a run that "
        f"predicted no grasp on any probed frame is `runs` ({empty} probes of this page); it "
        "recorded none of the fields below; and from 2026-10-13 on it also checked a run for "
        "a top-level part that no gradient reached, which only a record with `no_gradient` "
        f"shows ({checked} of them). A record keeps its judge until its cell is probed again.",
        "",
        "Judge 2 also records whether the parts of a model *learned*, not only whether they "
        "ran. None of it changes a verdict, except that a top-level part no gradient reached "
        "makes a run `runs-unstable` (as before) and an empty prediction makes it "
        "`runs-empty`. In a record of judge 2:",
        "",
        *_table(["Field", "Meaning"], [[f"`{name}`", text] for name, text in meanings.items()]),
        "",
        "The flags of a loss term; *active* iterations are those in which the term was "
        "reported, finite and not exactly 0:",
        "",
        *[f"- `{name}`: {meaning}." for name, meaning in TREND_FLAGS.items()],
        "",
        "The schedules of a probe (`schedule` in a record):",
        "",
        *[f"- `{name}`: {meaning}." for name, meaning in SCHEDULES.items()],
        "",
        "*Fall per loss term §*, a column of the tables of what the probes saw, gives the "
        "fall of every term (the term without `loss/`; *active n* where it was active in "
        "fewer iterations than ran), and *Learns, predicts §* whether the run reached those "
        "levels ([Measured classes](#measured-classes)) and what kept it from them: the "
        "flags, the units no gradient reached, the parts that did not change, frames without "
        f"a grasp or with one score for all; `{NOT_CHECKED}` for a record of the first judge. "
        "A flag is a statement about one short run, a reason to look, not a verdict. Under "
        "`schedule: constant` every step of that run used the recipe's peak "
        "rate, so flags compare across methods; a record without `schedule` ran the method's "
        "own schedule over the probe's epochs, which may have lowered the rate within the "
        "probe (`lr_factor`) by very different amounts from one method to the next.",
        "",
    ]
    return lines


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


def page() -> str:
    data = measured()
    about, cells_ = data["about"], data["cells"]
    records = Records(cells_)
    counts = Counter(record["verdict"] for record in cells_)
    iterations, frames = about["probe"]["iterations"], about["probe"]["frames"]
    shared = sum("probed_as" in record for record in cells_)
    lines = [
        "<!-- Generated by tools/make_combination_docs.py from measured_combinations.json.",
        "     Do not edit by hand: probe (python -m tools.probe_grid), export, regenerate. -->",
        "",
        "# Measured combinations",
        "",
        "Which component works under which method, **measured**: every cell of this page is a",
        "recorded run of [`grasppanda compose` and `grasppanda probe`](../user_guides/combinations.md),",
        "and nothing is inferred from a contract or from another cell. The contract tables of",
        "[Swapping components](swapping.md) say what the config check accepts; this page says",
        "what happened when the accepted combinations were built, trained and asked to predict.",
        "A `refused` cell is the verdict of the config check on that config: nothing was built.",
        "The [combination explorer](../explorer/index.md) answers a selection of method and",
        "options from these records; the [slot pages](../slots/index.md) say what each option is.",
        "",
        "**A cell is one run, and a run is not repeatable.** The two recorded commands repeat",
        "the experiment, not its numbers: the config hash and the parameter count repeat",
        "exactly and the losses of the first iteration to the digits given; later losses, the",
        "share of real and of supervised seeds, the grasps per frame and, for a combination",
        "near the edge, the split between `runs` and `runs-unstable` are those of the recorded",
        "run and differ in the next one. A record is bound to the commit of its run: it is",
        "replaced when its config changes, not when the code of a component does.",
        "",
        "**No AP appears on this page, and none follows from it.** A probe trains "
        f"{iterations} iterations",
        f"on {frames} frames. It shows whether a combination *works*: it builds, its losses are "
        "finite,",
        "seeds reach its head, it predicts. It says nothing about how good the combination is;",
        "that takes a full training run and an evaluation on a benchmark split.",
        "Whether a cell that runs also learns like the stage it replaces (250 iterations, term "
        "by term): [Learns like the reference?](learns_like_reference.md).",
        "",
        "## What was run",
        "",
        f"- **Grid**: {len(cells_)} recorded cells ("
        + ", ".join(f"{counts[v]} `{v}`" for v in VERDICT_ORDER if counts[v])
        + "): every shipped method config as it is, every registered scene encoder under every "
        "seed-based method, every variant of every other stage with one stage replaced at a "
        "time, a sample of the legal sets of stage variants, HGGD as it is, every image-encoder "
        "preset under HGGD, every post-processing preset under all six methods, every recipe "
        "preset under every method it applies to, every loss preset in one term of every kind "
        "it accepts under every seed-based method with such a term, and a few requests that "
        "replace several slots at once. The variants of a stage are its presets plus every "
        "component whose constructor defaults no preset states. " + _bare_sentence(),
        "- **A cell**: `grasppanda compose <method config> --set <slot>=<choice>`, so that an "
        "encoder is fitted and adapters are inserted exactly as a user gets them, then "
        f"`grasppanda probe <composed config> --iters {iterations} --frames {frames} --schedule "
        "constant`: the real trainer with the method's own recipe on the split `sample` of "
        "GraspNet-1Billion (RealSense; scene 0, frames 0, 32, …, 224), then prediction on two "
        "of those frames. Both commands of every cell are in the "
        "[measured file](measured_combinations.json).",
        "- **The learning rate**: grid probes train at the recipe's peak rate without its "
        "schedule (`--schedule constant`: the optimizer's rate, or `max_lr` of a one-cycle "
        "schedule, at every step, scaled by the ratio of the batch sizes when the probe's "
        "batches are smaller than the recipe's, so FineGrasp's 0.008 for 32 frames trains at "
        "0.002 on 8; optimizer, weight decay, clipping and hooks as the recipe has them, "
        "except that a batch-norm momentum the recipe decays per epoch is held at its "
        "first-epoch value, as the rate is held at its peak: a probe's epochs are a few "
        "iterations long). Run "
        "over the few epochs of a probe, the schedules would decay GSNet to "
        "0.63 and EconomicGrasp to 2.4 % of its rate and keep FineGrasp in its warm-up, and "
        "the trends of different methods would not compare. A cell whose request sets the "
        "schedule (a schedule preset, a schedule-free optimizer, a recipe preset) runs that "
        "schedule. `grasppanda probe --schedule recipe` runs a recipe as shipped. "
        + _schedule_sentence(cells_),
        '- **One exception to "as a user gets them"**: configs that read the width histogram '
        "of Scale-Balanced-Grasp were probed with the reference's own file "
        "(`scale_statistic.source: legacy_import`), because the native histogram is a statistic "
        "of the whole training split and a probe reads eight frames. The override is part of "
        "the recorded command.",
        "- **When and where**: " + " ".join(_runs(about, cells_)),
        "- **Two cells with one config are one experiment.** A cell that composes to the very "
        "config of another cell (the same config hash) shares its probe; one that composes "
        f"to its method's shipped config is marked *own*. {shared} of the {len(cells_)} cells "
        "share a probe.",
        "",
        *counts_section(cells_),
        "",
        "## Verdicts",
        "",
        *_table(
            ["Verdict", "In a table", "Meaning"],
            [
                ["`runs`", "✓ · ✓ own · ✓ in `Adapter`", VERDICTS["runs"]],
                ["`runs-empty`", "∅ · ∅ in `Adapter`", VERDICTS["runs-empty"]],
                ["`runs-unstable`", "≈ (n)", VERDICTS["runs-unstable"]],
                ["`refused`", "✗ (n)", VERDICTS["refused"] + _refused_by(cells_)],
                ["`fails`", "**fails** (n)", VERDICTS["fails"]],
                ["`wired`", "—", VERDICTS["wired"] + " Not used on this page."],
            ],
        ),
        "",
        "*✓ in `Adapter`*: the bare swap is refused by the contract between encoder and "
        "proposal; compose wrapped the encoder in the one adapter that bridges it, said so, "
        "and the adapted config ran. *†*: compose also fitted the encoder to the method, "
        "setting `out_channels` to the width of the encoder it replaces or `in_channels` to "
        "the point features the method feeds. *(n)* points to the reason below the table: "
        "for a refusal the first sentence of each message of the config check (the whole "
        "messages, with the fix they propose, are in the measured file), for a failure the "
        "phase, the component and the exception, for an unstable run what was unstable. *‡* "
        "marks numbers that depend on the load of a shared machine. *§* marks what judge 2 "
        "flagged in a run that did not stop it from running ([Judges](#judges)). Below the "
        "mark of a probe: the grasps it predicted on each probed frame and, for a record of "
        "judge 2, the range of the fall of its loss terms and the levels it did not reach "
        "(*learns ✗ (n)*); a probe that predicted no grasp on any frame is greyed. Below a "
        "refusal, its kind when the record has one (structural, not-implemented, policy, "
        "not-a-combination).",
        "",
        *judges_section(cells_),
        "## Measured classes",
        "",
        *levels_section(),
        "A class is a statement about this grid: these presets, the methods' own recipes, "
        "this dataset, one stage replaced at a time. A stage that is refused alone may work "
        "together with the stage it depends on (a head bundle with its assigner and its "
        "labels); those are the [sets of stages](#sets-of-stages). The classes of the scene "
        "encoders and of the other stages, level by level:",
        "",
        *glance_section(records),
        "",
        "## The methods as shipped",
        "",
        *references_section(records),
        "## One component replaced",
        "",
        *slot_section("backbone", records, "encoder"),
        *adapters_section(cells_),
    ]
    for slot in STAGE_SLOTS:
        lines += slot_section(slot, records, "stage")
    recipe = recipes_section(records)
    if recipe:
        lines += ["## The training recipe", "", *recipe]
    lines += ["## Sets of stages", "", *stage_sets_section(records)]
    lines += family_slots_section(records)
    lines += post_section(records)
    lines += loss_section(records)
    lines += cross_section(records)
    lines += ["## Passed the check, failed afterwards", "", *findings_section(cells_)]
    missing = about.get("not_probed") or []
    lines += [
        "## Not on this page",
        "",
        "- Accuracy of any kind. See the method pages of the [model zoo](../model_zoo.md).",
        "- Other datasets, cameras and recipes than the ones named above.",
        "- Combinations of two or more replaced options beyond the stage sets that were "
        "probed and the cells of [Several slots at once](#several-slots-at-once).",
        "- Inner parts of HGGD: it has no registered slots yet, so it is probed as it is.",
        *(
            [f"- {len(missing)} cell(s) of the grid that were not probed in this run."]
            if missing
            else []
        ),
        "",
        "## Adding a component",
        "",
        "A new encoder or stage appears in the grid by being registered. Give it a preset if "
        "its options cannot be guessed ([Try a combination](../user_guides/combinations.md)), "
        "probe its cells and regenerate:",
        "",
        "```bash",
        "python -m tools.probe_grid run --only 'encoder/*/MyEncoder' 'stage/*/*/my-preset'",
        "python -m tools.probe_grid export",
        "python -m tools.make_combination_docs",
        "```",
        "",
        "`tests/unit/test_combine_grid.py` fails until every cell of a registered component "
        "has a record in the measured file.",
    ]
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------ the slot index
#
# The first page of the slot documentation, docs/slots/index.md. Its text is written here;
# its numbers are not: how many options a slot has is counted from the presets and the
# registries, and the class of every option from the measured records with the rules of
# the measured page (grasppanda.combine.levels), so the two pages cannot disagree.

#: Slot → (title, where it is written in a config, its page, task, input → output).
SLOTS: dict[str, tuple[str, str, str, str, str]] = {
    "image_encoder": (
        "Image encoder",
        "`model.image_encoder`",
        "image_encoders.md",
        "Turns the colour (or RGB-D) image into feature maps. HGGD predicts its grasp "
        "heat-map from them; a seed-based method can read them at the pixel of every point "
        "(point painting).",
        "image `(B, C, H, W)` → a list of feature maps, finest first",
    ),
    "backbone": (
        "Scene encoder",
        "`model.backbone`",
        "scene_encoders.md",
        "The first learned stage of a seed-based method: one feature vector per point of "
        "the depth cloud. Its receptive field is what a seed knows before any crop is taken.",
        "points `(N, 3)` in metres, camera frame, with per-point features → features on "
        "every point (`dense`) or on a sub-sample (`subsampled`)",
    ),
    "proposal": (
        "Seed proposal",
        "`model.proposal`",
        "proposal.md",
        "Decides where grasps are predicted: picks a fixed number of seeds per frame "
        "(usually 1024) among the scene points, by a predicted score, a sampler, or both.",
        "encoder features → `SeedSet`: seed points, their features, named scores",
    ),
    "view": (
        "View",
        "`model.view`",
        "view.md",
        "Chooses the approach direction of every seed: two of the three degrees of freedom "
        "of the grasp orientation, and the frame the crop is taken in.",
        "`SeedSet` → `ApproachSet`: a unit direction and a rotation per seed, and the index "
        "of the label view",
    ),
    "assigner": (
        "Assigner",
        "`model.assigner`",
        "",
        "Training only: matches every seed with the grasp labels of the scene point it "
        "sits on, in the label set the head is supervised with.",
        "seeds, approach, supervision → matched labels (`dense` grids or `compact` best "
        "grasps), optional seed weights",
    ),
    "grouper": (
        "Grouper",
        "`model.grouper`",
        "grouper.md",
        "Gathers the neighbourhood of every seed in the frame of its approach direction "
        "and turns it into the feature the head reads.",
        "points with features, crop centres and frames → one feature per seed, or one per "
        "seed and depth bin",
    ),
    "head": (
        "Head bundle",
        "`model.head`",
        "head_loss.md",
        "Predicts the grasp at a seed: in-plane rotation, depth, width and score. Network "
        "head, target encoder, loss and decoder share one parameterisation, so they are "
        "swapped as a whole.",
        "crop features → loss terms in training; `GraspGroup` (one grasp per seed) at prediction",
    ),
    LOSS_SLOT: (
        "Loss",
        "`model.<stage>.<term>_loss`",
        "head_loss.md",
        "One term of the loss of a stage (objectness, graspness, view score, grasp score, "
        "angle, depth, width). A stage declares the kind of each term, a loss the kinds it "
        "fits.",
        "prediction and target of one term → a weighted scalar",
    ),
    POST_SLOT: (
        "Post-processing",
        "`evaluation.variants.<name>`",
        "post.md",
        "What happens to predicted grasps before an evaluator or a robot gets them: "
        "filters, selection, corrections of pose and width, a change of frame. Steps are "
        "chained.",
        "`GraspGroup` of one frame and the observation → `GraspGroup`",
    ),
    "recipe": (
        "Training recipe",
        "`optimizer`, `scheduler`, `hooks`, `data.train.dataset.transforms`",
        "recipe.md",
        "Everything around the model: how weights are updated, how the learning rate "
        "moves, what runs around the loop, how a training sample is perturbed.",
        "parameters and gradients → updates; a training sample → a perturbed sample",
    ),
}
#: Seed-based slots whose class is the count over the five methods (the measured page's).
SEED_SLOTS = ("backbone", *STAGE_SLOTS)


def method_page(method: str) -> str:
    """The stem of the page of a method under ``docs/methods/``: its config directory."""
    return {**METHODS, **FAMILIES}[method].split("/")[0]


def option_counts() -> dict[str, tuple[int, int]]:
    """Slot → ``(options, component types)``: what can be written after ``--set <slot>=``."""
    members = presets()
    counts: dict[str, tuple[int, int]] = {}
    # Components of the package only: a test or a project may register its own.
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES
    from grasppanda.registry import BACKBONES

    encoders = {
        name: found
        for name, found in encoder_presets().items()
        if _in_tree_module(BACKBONES.get(name))
    }
    counts["backbone"] = (sum(len(found) or 1 for found in encoders.values()), len(encoders))
    for slot in STAGE_SLOTS:
        variants = [
            (choice, component)
            for choice, component in stage_variants(slot)
            if _in_tree_module(STAGE_REGISTRIES[slot].get(component))
        ]
        counts[slot] = (len(variants), len({component for _, component in variants}))
    for slot in ("image_encoder", LOSS_SLOT, POST_SLOT):
        found = members.get(slot, {})
        counts[slot] = (len(found), len({preset.type for preset in found.values()}))
    found = [preset for group in recipes().values() for preset in group.values()]
    counts["recipe"] = (len(found), len({preset.type for preset in found}))
    return counts


def _pipeline() -> list[str]:
    """The pipeline as text; the right-hand rail is aligned by construction."""
    stages = (
        "  │    scene encoder ─► seed proposal ─► view ─► (assigner) ─► grouper ─► head bundle ─"
    )
    hggd = "  └─ HGGD: image encoder ─► heat-map and region stages (no slots of their own yet) "
    rail = len(stages)

    def to_rail(text: str, end: str, fill: str = " ") -> str:
        return text + fill * (rail - len(text)) + end

    return [
        "```text",
        "observation: a depth image as a point cloud (HGGD also reads the colour image)",
        "  │",
        "  ├─ the seed-based methods",
        to_rail(stages, "┐"),
        to_rail("  │    features         seeds            approach  labels,       crops", "│"),
        to_rail("  │                                                training", "│"),
        to_rail(hggd, "┤", "─"),
        to_rail(
            "  grasps for the evaluator or a robot ◄─ post-processing ◄─ GraspGroup ◄", "┘", "─"
        ),
        "",
        "around the model: the loss of every stage · the training recipe",
        "```",
    ]


def _in_tree_module(cls: Any) -> bool:
    return cls.__module__.split(".")[0] == "grasppanda"


def _slot_table(counts: dict[str, tuple[int, int]]) -> list[str]:
    rows = []
    for slot, (title, key, link, task, flow) in SLOTS.items():
        options, types = counts[slot]
        name = f"[{title}]({link})" if link else title
        extra = ""
        if slot == "proposal":
            from grasppanda.registry import SAMPLERS

            samplers = [n for n in SAMPLERS.names() if _in_tree_module(SAMPLERS.get(n))]
            extra = f"; {len(samplers)} samplers as `model.proposal.sampler`"
        rows.append(
            [
                f"**{name}**<br>{key}",
                f"{task}<br>*{flow}*",
                f"{options} ({types} component {'type' if types == 1 else 'types'}{extra})",
            ]
        )
    return _table(["Slot", "Task, and input → output", "Options"], rows)


def _class_rows(
    records: Records,
) -> tuple[list[tuple[list[str], list[tuple[str, dict[str, OptionClass]]]]], list[str]]:
    """The rows of the class tables of the slot index and the notes under them."""
    cells = list(records.by_id.values())
    rows: list[tuple[list[str], list[tuple[str, dict[str, OptionClass]]]]] = []
    notes: list[str] = []
    five = "the five seed-based methods"
    for slot in SEED_SLOTS:
        group = "encoder" if slot == "backbone" else "stage"
        members = []
        for _component, variant, found in slot_classes(slot, records, group):
            preset = variant in presets().get(slot, {})
            members.append((f"`{variant}`" + ("" if preset else " (defaults)"), found))
        rows.append(([SLOTS[slot][0], five], members))

    def under(group: str) -> str:
        """The methods a group has cells under in the grid, in the order of the grid."""
        found = {cell.method for cell in grid(group)}
        return ", ".join(method for method in (*METHODS, *FAMILIES) if method in found)

    def relative(slot: str, names: Iterable[str], title: str, where: str) -> None:
        members = [(f"`{name}`", records.classes(slot, name)) for name in names]
        members = [(label, found) for label, found in members if found["composes"].methods]
        rows.append(([title, where], members))

    members = presets()
    relative(
        "image_encoder",
        members.get("image_encoder", {}),
        f"{SLOTS['image_encoder'][0]} †",
        "HGGD, the one method with this slot",
    )
    relative(
        LOSS_SLOT,
        members.get(LOSS_SLOT, {}),
        f"{SLOTS[LOSS_SLOT][0]} †",
        f"loss terms of {under('loss')}",
    )
    relative(POST_SLOT, members.get(POST_SLOT, {}), f"{SLOTS[POST_SLOT][0]} †", under("post"))
    recipe_members = []
    for slot, group in recipes().items():
        for name in group:
            found = records.classes(slot, name)
            if found["composes"].methods:
                recipe_members.append((f"`{slot}={name}`", found))
    encoder = "; HGGD not for the presets that name the scene encoder"
    rows.append(([f"{SLOTS['recipe'][0]} †", under("recipe") + encoder], recipe_members))
    for name, reason in {**POST_NOT_PROBED, **LOSS_NOT_PROBED}.items():
        notes.append(f"`{name}` has no cell of its own: {reason}.")
    for slot, component in bare_components():
        notes.append(
            f"`{component}` ({SLOTS[slot][0].lower()}) has no cell under its type name: it "
            "needs an option only a preset supplies; its presets are classified."
        )
    waiting = sum(
        cell.id not in records.by_id for group in ("post", "loss", "recipe") for cell in grid(group)
    )
    if waiting:
        notes.append(
            f"{waiting} cells of these slots (†) are {NOT_YET}: the grid now has every "
            "post-processing preset under all six methods, every recipe preset under every "
            "method it applies to and every loss preset in one term of each kind it accepts; "
            "a class names them after the methods it counts (*+n not probed*)."
        )
    del cells
    return rows, notes


def slots_index(records: Records) -> str:
    counts = option_counts()
    cells = list(records.by_id.values())
    verdicts = Counter(record["verdict"] for record in cells)
    rows, notes = _class_rows(records)
    total = sum(options for options, _ in counts.values())
    methods = ", ".join(METHODS)
    lines = [
        "<!-- Generated by tools/make_combination_docs.py from the presets, the registries and",
        "     measured_combinations.json. Do not edit by hand: the text is in the generator,",
        "     the counts and the classes are computed. -->",
        "",
        "# Slots: the stages of the pipeline and what can go into each",
        "",
        "GraspPanda builds a grasp detector from stages, and every stage is a **slot** of the "
        "config: a place where one component can be replaced by another that keeps the same "
        f"contract. This page is the map: the pipeline, the task of every slot, its {total} "
        "options in all, and which option was measured to work under which method. Every "
        "slot has a page of its own with one row and one command per option.",
        "",
        "**No accuracy is claimed for any option that is not one of the six ported methods.** "
        "An option is listed because it was built, trained for 20 iterations on 8 real "
        "frames of GraspNet-1Billion and asked to predict (`grasppanda probe`). That shows it "
        "is usable, not that it is good; the model zoo is still empty.",
        "",
        "To try options without reading further, open the "
        "[combination explorer](../explorer/index.md): pick a method, change a slot, read "
        "the measured verdict and the command.",
        "",
        "## Words used on these pages",
        "",
        "- **seed**: a point of the scene at which a grasp is predicted; the proposal picks "
        "the seeds (1024 per frame in the ported methods).",
        "- **contract**: what a component declares about its input and output (where its "
        "features live, which axes a tensor has, which labels it reads). The config check "
        "compares contracts before anything is built and refuses a mismatch with the fix.",
        "- **preset**: a small YAML file with a component and settings for table-top depth "
        "clouds; the name after `--set <slot>=`.",
        "- **adapter**: a wrapper `compose` puts around a scene encoder so that it fits the "
        "next stage (features propagated back to every point, or kept on a subsample).",
        "- **dense / compact label set**: the grasp labels a head is trained on. *Dense* is "
        "the full grid of GraspNet-1Billion (300 views × 12 in-plane angles × 4 depths per "
        "labelled point); *compact* keeps the best grasp per view (EconomicGrasp, FineGrasp). "
        "A head, its assigner and the dataset section must use the same one.",
        "- **point painting**: giving every point the image feature at its pixel, which is "
        "how a seed-based method reads an image encoder.",
        "",
        "## The pipeline",
        "",
        *_pipeline(),
        "",
        "Six methods are ported. Five share the seed-based skeleton and therefore its six "
        f"stage slots ({methods}); HGGD is a detector family of its own with one slot so "
        "far, the image encoder. Losses, post-processing and the training recipe are around "
        "the stages and apply to all six.",
        "",
        "## The slots",
        "",
        "An option is a name you can write after `--set <slot>=`: a preset (a component with "
        "settings for table-top depth clouds) or a component with its defaults.",
        "",
        *_slot_table(counts),
        "",
        "The assigner has no page of its own: it goes with the head bundle and the label set "
        "([Heads and losses](head_loss.md), [Data contracts](../advanced_guides/data_flow.md)). "
        "The scene encoders have an [overview](scene_encoders.md) and three pages: "
        "[point-based networks](scene_encoders_point.md), "
        "[sparse-voxel networks and released weights](scene_encoders_sparse.md), "
        "[transformers and pretrained 3-D encoders](scene_encoders_transformer.md). The "
        "encoders ported before these pages are in "
        "[Scene encoders of the model zoo](../model_zoo_backbones.md).",
        "",
        "## Which option fits which method",
        "",
        "From the [measured records](../advanced_guides/measured_combinations.md) "
        f"({len(cells)} recorded cells: "
        + ", ".join(f"{verdicts[name]} `{name}`" for name in VERDICT_ORDER if verdicts[name])
        + "), with **one slot replaced at a time**. Every method has a page that lists the "
        "options of each of its slots with what was measured for each, and so does "
        "`grasppanda compose <method config> --list <slot>`: "
        + ", ".join(
            f"[{name}](../methods/{method_page(name)}.md)" for name in (*METHODS, *FAMILIES)
        )
        + ".",
        "",
        *levels_section(),
        "† The slot was not probed under the five seed-based methods but under the methods "
        "of the second column, so its classes count those. *(defaults)*: the component named "
        "by its type and built with its constructor defaults, not a preset; a component that "
        "needs an option only a preset supplies does not work that way. Classes, level by "
        "level:",
        "",
        *class_tables(rows, ["Slot", "Probed under"]),
        *[f"- {note}" for note in notes],
        *([""] if notes else []),
        "How to read it:",
        "",
        "- **A class is not a ranking.** *universal* at a level says that a swap reached that "
        "level under every method without anything else changed; it says nothing about "
        "accuracy.",
        "- **An option that is not universal is usually bound to its neighbours, not to a "
        "paper.** A head bundle goes with the assigner and the label set it is supervised "
        "with, a grouper with the axes its head reads, a scoring proposal with the stage "
        "that owns objectness. Replaced together with what they depend on they run under "
        "other methods too: see "
        "[Sets of stages](../advanced_guides/measured_combinations.md#sets-of-stages) and "
        "[Several slots at once](../advanced_guides/measured_combinations.md#several-slots-at-once).",
        "- **A refusal comes with the fix.** `grasppanda compose` prints the contract that "
        "is violated and the option or preset that satisfies it, before anything is built.",
        "- **A loss or a recipe preset with a class *N of M probed* below M did not run "
        "everywhere it was put**: the [loss](head_loss.md) and [recipe](recipe.md) pages say "
        "where and why (a term of another kind, a detector that reads images, mixed "
        "precision).",
        "- **What was not probed is not classified.** Most combinations of two or more "
        "options have no record; the explorer shows for any two options what the config "
        "check answers, which is a weaker statement than a probe.",
        "",
        "## How to try one",
        "",
        "```bash",
        "grasppanda compose --list grouper                                # the presets of a slot",
        "grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --list grouper",
        "                                                                 # ... under GSNet, measured",
        "grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \\",
        "    --set grouper=roi-grid -o gsnet_roi-grid.yaml                 # a checked config",
        "grasppanda probe gsnet_roi-grid.yaml                             # 20 iterations on",
        "                                                                 # real frames: runs?",
        "grasppanda train gsnet_roi-grid.yaml                             # the real run",
        "```",
        "",
        "- Several `--set` go into one command (`--set backbone=point-transformer-v2 --set "
        "grouper=cylinder-vector-attention`, a recorded combination); "
        "`compose` inserts the adapter the contracts ask for and says so, or refuses with the "
        "conflict and the fix.",
        "- `grasppanda compose --show grouper=roi-grid` prints one preset with what must be "
        "re-tuned; `grasppanda components --slot groupers` prints the contracts, sources and "
        "optional libraries of every component.",
        "- `grasppanda explore` serves the explorer locally and checks any selection with "
        "the `compose` of your installation.",
        "- `probe` needs GraspNet-1Billion and the label assets of the method "
        "([Try a combination](../user_guides/combinations.md) lists the `prepare` commands). "
        "An option that is a bridge to a library names the extra to install on its slot page. "
        "Until the first release on PyPI, install an extra from the repository root with "
        '`pip install -e ".[<extra>]"`: the form `pip install "grasppanda[<extra>]"` resolves '
        "against the metadata of the copy that is already installed and may install nothing.",
        "- Under **Scale-Balanced-Grasp** a probe ends as `wired` (a synthetic scene, exit "
        "status 3) unless it is given the reference's width histogram: `grasppanda probe "
        "<config> --opts data.train.dataset.scale_statistic.source=legacy_import "
        "data.train.dataset.scale_statistic.reference_sha256=<SHA-256>`. The histogram is a "
        "statistic of a whole training split and cannot be built from the eight frames of a "
        "probe (`configs/scale_balanced_grasp/README.md` says how to import the file); the "
        "recorded `runs` "
        "of that method were measured this way.",
        "- `grasppanda compose --list image_encoder` lists the image encoders (the slot of "
        "HGGD); the seed-based methods read one through point painting.",
    ]
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------------- staleness


def experimental() -> dict[Path, str]:
    """Every composed config under ``configs/experimental`` → what compose writes for it."""
    return {path: recompose(path).text(path) for path in sorted(EXPERIMENTAL.glob("*.yaml"))}


def expected() -> dict[Path, str]:
    pages = experimental()
    if DATA.is_file():
        pages[PAGE] = page()
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
