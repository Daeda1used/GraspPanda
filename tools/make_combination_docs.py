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
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from grasppanda.combine.compose import recompose
from grasppanda.combine.grid import (
    FAMILIES,
    LOSS_NOT_PROBED,
    METHODS,
    POST_METHODS,
    POST_NOT_PROBED,
    STAGE_SLOTS,
    Cell,
    cells,
    encoder_presets,
    stage_variants,
)
from grasppanda.combine.presets import LOSS_SLOT, POST_SLOT, presets
from grasppanda.combine.recipes import recipes

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
PAGE = REPO / "docs" / "advanced_guides" / "measured_combinations.md"
EXPERIMENTAL = REPO / "configs" / "experimental"
SLOTS_INDEX = REPO / "docs" / "slots" / "index.md"

#: The measured classes: name → (number of methods a component works under, definition).
CLASSES: dict[str, tuple[range, str]] = {
    "universal": (range(5, 6), "it works under all five seed-based methods"),
    "most": (range(3, 5), "it works under three or four of them"),
    "some": (range(2, 3), "it works under two"),
    "specific": (range(1, 2), "it works under exactly one"),
    "none": (range(0, 1), "it works under none of them as it was probed"),
}
#: Verdicts that count as "works" for a class.
WORKS = ("runs", "runs-unstable")
VERDICT_ORDER = ("runs", "runs-unstable", "refused", "fails")
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

    def get(self, cell_id: str) -> dict[str, Any] | None:
        return self.by_id.get(cell_id)

    def probed(self, record: dict[str, Any]) -> dict[str, Any]:
        """The record that holds the numbers of ``record``: itself, or the cell it names."""
        return self.by_id.get(record.get("probed_as", ""), record)

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


def verdict_text(record: dict[str, Any] | None, own: bool, notes: Notes) -> str:
    """One table cell: what was measured for this component under this method."""
    if record is None:
        return "not probed"
    verdict = record["verdict"]
    adapter = f" in `{record['adapter']}`" if record.get("adapter") else ""
    fitted = "†" if any("_channels" in item for item in record.get("fitted", [])) else ""
    if verdict == "runs":
        return "✓ own" if own else f"✓{adapter}{fitted}"
    if verdict == "runs-unstable":
        return f"≈{adapter}{fitted} ({notes.number(record.get('unstable', ''))})"
    if verdict == "refused":
        return f"✗ ({notes.number(reason(record))})"
    if verdict == "fails":
        return f"**fails** ({notes.number(reason(record))})"
    return verdict


def classify(records: Iterable[dict[str, Any] | None]) -> tuple[str, int, int]:
    """``(class, methods it works under, of which unstable)`` of one component."""
    found = [record for record in records if record is not None]
    works = sum(record["verdict"] in WORKS for record in found)
    unstable = sum(record["verdict"] == "runs-unstable" for record in found)
    name = next(label for label, (count, _) in CLASSES.items() if works in count)
    return name, works, unstable


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


def _class_text(name: str, unstable: int) -> str:
    if name not in CLASSES:
        return name
    return f"**{name}**" + (f" ≈{unstable}" if unstable else "")


#: Shown instead of a class for a component that has a cell without a record.
INCOMPLETE = "not probed in full"


def slot_classes(slot: str, records: Records, group: str) -> list[tuple[str, str, str, int, int]]:
    """``(component, variant, class, works, unstable)`` of every row of a slot.

    A component has a class only when all of its cells were probed: a class is a count
    over the methods, and a missing cell would be counted as "does not work".
    """
    found = []
    for (component, variant), members in _rows(slot, group).items():
        measured_ = [records.get(cell.id) for cell in members]
        name, works, unstable = classify(measured_)
        if any(record is None for record in measured_):
            name = INCOMPLETE
        found.append((component, variant, name, works, unstable))
    return found


def glance_section(records: Records) -> list[str]:
    """Every component of every slot under its class, in one table."""
    rows, unclassed = [], []
    for slot in SLOT_TITLES:
        group = "encoder" if slot == "backbone" else "stage"
        classes = slot_classes(slot, records, group)
        several = Counter(component for component, *_ in classes)

        def name_of(component: str, variant: str, slot: str = slot, several: Any = several) -> str:
            if several[component] > 1:
                return _label(component, variant, slot, short=True)
            return f"`{component}`"

        row = [SLOT_TITLES[slot]]
        for wanted in CLASSES:
            members = [
                name_of(component, variant) + (f" ≈{unstable}" if unstable else "")
                for component, variant, name, _, unstable in classes
                if name == wanted
            ]
            row.append("<br>".join(members) or "—")
        rows.append(row)
        unclassed += [name_of(c, v) for c, v, name, *_ in classes if name == INCOMPLETE]
    lines = _table(["Slot", *CLASSES], rows)
    if unclassed:
        lines += ["", f"Without a class, because a cell is {INCOMPLETE}: {', '.join(unclassed)}."]
    return lines


def slot_section(slot: str, records: Records, group: str) -> list[str]:
    """The table, the classes and the numbers of one slot."""
    notes = Notes()
    rows, numbers = [], []
    classes = {
        (c, v): (name, unstable) for c, v, name, _, unstable in slot_classes(slot, records, group)
    }
    for (component, variant), members in _rows(slot, group).items():
        label = _label(component, variant, slot)
        row = [label]
        per_method = {cell.method: records.get(cell.id) for cell in members}
        for method in METHODS:
            record = per_method.get(method)
            own = records.own(record)
            row.append(verdict_text(record, own, notes))
            if record is not None and record["verdict"] in WORKS and not own:
                name = f"{label.replace('<br>', ', ')} under {method}"
                numbers.append(_numbers_row(name, records.probed(record)))
        rows.append([*row, _class_text(*classes[component, variant])])
    lines = [f"### {SLOT_TITLES[slot]}", ""]
    lines += _table(["Component", *METHODS, "Class"], rows)
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
    "s / iteration ‡",
    "GPU GiB ‡",
]


def _numbers_row(label: str, record: dict[str, Any]) -> list[str]:
    loss = record.get("losses", {}).get("loss")
    real, supervised = record.get("real_seeds"), record.get("supervised_seeds")
    shared = record.get("shared_machine", {})
    seconds = shared.get("seconds_per_iteration")
    memory = shared.get("peak_gpu_mib")
    return [
        label,
        f"{record['parameters']:,}".replace(",", " ") if record.get("parameters") else "—",
        f"{loss[0]:.4g} → {loss[1]:.4g}" if loss else "—",
        f"{real['min']:.2f}" if real else "—",
        f"{supervised['mean']:.2f}" if supervised else "—",
        ", ".join(str(count) for count in record.get("grasps_per_frame", [])) or "—",
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
        verdict = verdict_text(record, False, notes)
        numbers = _numbers_row("", record)[1:] if record and record["verdict"] in WORKS else []
        rows.append([method, f"`{config}`", verdict, *(numbers or ["—"] * 7)])
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
                verdict_text(record, own, notes),
            ]
        )
        if record["verdict"] in WORKS and not own:
            name = " + ".join(f"`{chosen[slot]}`" for slot in ("backbone", *STAGE_SLOTS))
            numbers.append(_numbers_row(name, records.probed(record)))
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
        "sets that were not probed are named in `about.not_probed` of the "
        "[measured file](measured_combinations.json); "
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


def post_section(records: list[dict[str, Any]]) -> list[str]:
    """Post-processing presets as the variant ``post``: verdict and grasps left, per method."""
    members = [record for record in records if record["group"] == "post"]
    if not members:
        return []
    methods = list(dict.fromkeys(record["method"] for record in members))
    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for record in members:
        by_variant.setdefault(record["variant"], {})[record["method"]] = record
    rows = []
    for variant, found in by_variant.items():
        component = next(iter(found.values()))["component"]
        row = [f"`{variant}` ({component})"]
        for method in methods:
            record = found.get(method)
            if record is None:
                row.append("—")
                continue
            counts = (record.get("postprocessing") or {}).get("grasps") or []
            left = "; ".join(f"{frame[0]} → {frame[-1]}" for frame in counts)
            text = record["verdict"] + (f": {left}" if left else "")
            if record["verdict"] in ("fails", "refused"):
                text += f". {reason(record)}"
            row.append(text)
        rows.append(row)
    return [
        "## Post-processing",
        "",
        "Every post-processing preset as the evaluation variant `post` of a method "
        "(`grasppanda compose <config> --set post=<preset>`), applied by the probe to the "
        "predictions of its two frames: grasps predicted → grasps left. A post-processor "
        "reads grasps and the observation, never a network, so three methods stand for the "
        "three kinds of input: seeds on a dense cloud with the full cloud kept, seeds of a "
        "sub-sampling encoder, and HGGD. The grasps are those of a detector trained for the "
        "iterations of the probe; "
        "[the page of the slot](../slots/post.md) has the same steps on the predictions of "
        "released weights.",
        "",
        *_table(["preset", *methods], rows),
        "",
    ]


def loss_section(records: list[dict[str, Any]]) -> list[str]:
    """Loss presets in the loss terms they were probed in: the verdict of the cell and the
    first and last value of every replaced term."""
    members = [record for record in records if record["group"] == "loss"]
    if not members:
        return []
    by_id = {record["id"]: record for record in records}
    notes = Notes()
    rows = []
    for record in members:
        losses = by_id.get(record.get("probed_as", ""), record).get("losses") or {}
        verdict = verdict_text(record, False, notes)
        for item in record["sets"]:
            target, _, preset = item.partition("=")
            stage, _, option = target.partition(".")
            term = "loss/" + (option.removesuffix("_loss") if option != "loss" else stage)
            values = losses.get(term)
            rows.append(
                [
                    f"`{preset}`",
                    record["method"],
                    f"`{target}`",
                    verdict,
                    f"{values[0]:.4g} → {values[1]:.4g}" if values else "—",
                ]
            )
        rows.append(["", "", "", "", ""])
    return [
        "## Losses",
        "",
        "Every loss preset in a loss term it is meant for, under GSNet and under one more "
        "method (`grasppanda compose <config> --set <stage>.<loss option>=<preset>`). A probe "
        "replaces several loss terms of a method at once: the rows between two empty lines "
        "are one probed config and share its verdict. The last column is the weighted value "
        "of the replaced term at the first and at the last of the 20 iterations: it shows "
        "that the term is finite and moves, nothing about accuracy, and values of different "
        "losses are not comparable. The last rows are pairings the config check refuses. "
        "Which loss fits which term is a contract "
        "([Heads and losses](../slots/head_loss.md)); the unit tests run every loss in every "
        "term of its kind on tiny models.",
        "",
        *_table(["preset", "method", "term", "verdict", "term value, first → last"], rows[:-1]),
        "",
        *notes.lines(),
        *([""] if notes.items else []),
    ]


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
        rows.append([cell.method, request, verdict_text(record, False, notes)])
        if record is not None and record["verdict"] in WORKS:
            numbers.append(_numbers_row(f"{cell.method}: {request}", records.probed(record)))
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
            verdict = verdict_text(record, False, notes)
            works = record is not None and record["verdict"] in WORKS
            numbers = _numbers_row("", record)[1:] if works else ["—"] * 7
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
            f"`{record['verdict']}`" if record else "not probed"
        )
        sets[(cell.slot, cell.variant)] = " ".join(f"`--set {item}`" for item in cell.sets)
    lines = [
        "A recipe preset changes what is around the model: the optimizer, the schedule, the "
        "hooks of the training loop, the point augmentation. Each is applied alone to a "
        "method as shipped and probed like a stage. `—`: the preset names the scene encoder "
        "of the seed-based skeleton and is probed under a second seed-based method instead "
        "of HGGD. What each preset is and how to use it: [the recipe slot](../slots/recipe.md).",
        "",
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
        if cell.id in by_id and by_id[cell.id]["verdict"] not in ("runs", "wired")
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
        "",
        "## What was run",
        "",
        f"- **Grid**: {len(cells_)} cells ("
        + ", ".join(f"{counts[v]} `{v}`" for v in VERDICT_ORDER if counts[v])
        + "): every shipped method config as it is, every registered scene encoder under every "
        "seed-based method, every variant of every other stage with one stage replaced at a "
        "time, a sample of the legal sets of stage variants, HGGD as it is, every image-encoder "
        "preset under HGGD, every post-processing, loss and recipe preset under two or three "
        "methods, and a few requests that replace several slots at once. The variants of a stage "
        "are its presets plus every component whose constructor defaults no preset states.",
        "- **A cell**: `grasppanda compose <method config> --set <slot>=<choice>`, so that an "
        "encoder is fitted and adapters are inserted exactly as a user gets them, then "
        f"`grasppanda probe <composed config> --iters {iterations} --frames {frames}`: the real "
        "trainer with the method's own recipe on the split `sample` of GraspNet-1Billion "
        "(RealSense; scene 0, frames 0, 32, …, 224), then prediction on two of those frames. "
        "Both commands of every cell are in the [measured file](measured_combinations.json).",
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
                ["`runs`", "✓ · ✓ own · ✓ in `Adapter`", about["verdicts"]["runs"]],
                ["`runs-unstable`", "≈ (n)", about["verdicts"]["runs-unstable"]],
                ["`refused`", "✗ (n)", about["verdicts"]["refused"] + _refused_by(cells_)],
                ["`fails`", "**fails** (n)", about["verdicts"]["fails"]],
                ["`wired`", "—", about["verdicts"]["wired"] + " Not used on this page."],
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
        "marks numbers that depend on the load of a shared machine.",
        "",
        "## Measured classes",
        "",
        "A component **works** under a method when its cell is `runs` or `runs-unstable`: the "
        "combination was built, trained for the probe's iterations and predicted, directly or "
        "in the adapter compose inserted. The class of a component counts the seed-based "
        "methods (" + ", ".join(METHODS) + ") it works under:",
        "",
        *[f"- **{name}**: {meaning}." for name, (_, meaning) in CLASSES.items()],
        "",
        "*≈n* after a class: under n of those methods the run was `runs-unstable`; the note "
        "under the table says what was unstable. A class is a statement about this grid: "
        "these presets, the methods' own recipes, this dataset, one stage replaced at a time. "
        "A stage that is refused alone may work together with the stage it depends on (a head "
        "bundle with its assigner and its labels); those are the "
        "[sets of stages](#sets-of-stages).",
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
    lines += post_section(cells_)
    lines += loss_section(cells_)
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
# the measured page (`classify`), so the two pages cannot disagree.

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
CLASS_COLUMNS = tuple(CLASSES)


def _works(records: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """``(methods it works under, methods it has a record under)`` of one option."""
    by_method: dict[str, bool] = {}
    for record in records:
        works = record["verdict"] in WORKS
        by_method[record["method"]] = by_method.get(record["method"], False) or works
    return sum(by_method.values()), len(by_method)


def _relative_class(works: int, probed: int) -> str:
    """The class of an option that was probed under fewer than five methods.

    The words are those of the measured page, counted over the methods the option was
    probed under: all of them, none of them, exactly one of several, or the rest.
    """
    if works == probed:
        return "universal"
    if works == 0:
        return "none"
    return "specific" if works == 1 and probed > 2 else "some" if probed == 2 else "most"


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


def _class_rows(records: Records) -> tuple[list[list[str]], list[str]]:
    """The rows of the class table and the notes under it."""
    cells = list(records.by_id.values())
    rows: list[list[str]] = []
    notes: list[str] = []

    def row(title: str, under: str, members: dict[str, list[str]]) -> None:
        rows.append(
            [title, under, *("<br>".join(members.get(name, [])) or "—" for name in CLASS_COLUMNS)]
        )

    five = "the five seed-based methods"
    for slot in SEED_SLOTS:
        group = "encoder" if slot == "backbone" else "stage"
        found: dict[str, list[str]] = {}
        for component, variant, name, _, unstable in slot_classes(slot, records, group):
            preset = variant in presets().get(slot, {})
            label = f"`{variant}`" + ("" if preset else " (defaults)")
            label += f" ≈{unstable}" if unstable else ""
            if name == INCOMPLETE:
                notes.append(f"`{variant}` ({SLOTS[slot][0].lower()}): {name}.")
                continue
            found.setdefault(name, []).append(label)
            del component
        row(SLOTS[slot][0], five, found)

    def relative(slot: str, group: str, names: Iterable[str], under: str, dagger: str) -> None:
        found: dict[str, list[str]] = {}
        for name in names:
            members = [
                record
                for record in cells
                if record["group"] == group
                and any(item.partition("=")[2] == name for item in record.get("sets", []))
            ]
            if not members:
                continue
            works, probed = _works(members)
            found.setdefault(_relative_class(works, probed), []).append(f"`{name}`")
        row(f"{SLOTS[slot][0]} {dagger}", under, found)

    members = presets()
    relative(
        "image_encoder",
        "family-slot",
        members.get("image_encoder", {}),
        "HGGD, the one method with this slot",
        "†",
    )
    relative(
        LOSS_SLOT,
        "loss",
        members.get(LOSS_SLOT, {}),
        "loss terms under GSNet and under one or two other seed-based methods",
        "†",
    )
    relative(POST_SLOT, "post", members.get(POST_SLOT, {}), ", ".join(POST_METHODS), "†")
    found: dict[str, list[str]] = {}
    for slot, group in recipes().items():
        for name in group:
            hits = [r for r in cells if r["group"] == "recipe" and f"{slot}={name}" in r["sets"]]
            if hits:
                works, probed = _works(hits)
                found.setdefault(_relative_class(works, probed), []).append(f"`{slot}={name}`")
    row(f"{SLOTS['recipe'][0]} †", "GSNet and HGGD (GSNet and EconomicGrasp for the presets "
        "that name the scene encoder)", found)  # fmt: skip
    for name, reason in {**POST_NOT_PROBED, **LOSS_NOT_PROBED}.items():
        notes.append(f"`{name}` has no cell of its own: {reason}.")
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
        "From the [measured records](../advanced_guides/measured_combinations.md), in the "
        f"words of that page ({len(cells)} recorded cells: "
        + ", ".join(f"{verdicts[name]} `{name}`" for name in VERDICT_ORDER if verdicts[name])
        + "). An option **works** under a method when the combination was built, trained for "
        "the probe's iterations and predicted, directly or in the adapter compose inserted. "
        "Its class counts the methods it works under, with **one slot replaced at a time**:",
        "",
        *[f"- **{name}**: {meaning}." for name, (_, meaning) in CLASSES.items() if name != "none"],
        "- **none**: it works under none of them as it was probed.",
        "",
        "*≈n*: under n of those methods the run was `runs-unstable`. † The slot was not "
        "probed under the five seed-based methods but under the methods of the second "
        "column, so its classes count those: *universal* is all of them, *some* one of two, "
        "*most* all but one of three, *none* none. *(defaults)*: the component named by its "
        "type and built with its constructor defaults, not a preset; a component that needs "
        "an option only a preset supplies does not work that way.",
        "",
        *_table(["Slot", "Probed under", *CLASS_COLUMNS], rows),
        "",
        *[f"- {note}" for note in notes],
        *([""] if notes else []),
        "How to read it:",
        "",
        "- **A class is not a ranking.** *universal* says that a swap needs nothing else; it "
        "says nothing about accuracy.",
        "- **An option that is not universal is usually bound to its neighbours, not to a "
        "paper.** A head bundle goes with the assigner and the label set it is supervised "
        "with, a grouper with the axes its head reads, a scoring proposal with the stage "
        "that owns objectness. Replaced together with what they depend on they run under "
        "other methods too: see "
        "[Sets of stages](../advanced_guides/measured_combinations.md#sets-of-stages) and "
        "[Several slots at once](../advanced_guides/measured_combinations.md#several-slots-at-once).",
        "- **A refusal comes with the fix.** `grasppanda compose` prints the contract that "
        "is violated and the option or preset that satisfies it, before anything is built.",
        "- **A loss or a recipe preset in the column *most*, *some* or *none* did not run "
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
        "grasppanda compose --list grouper                                # the options of a slot",
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
