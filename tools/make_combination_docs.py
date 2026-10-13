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
from grasppanda.combine.grid import FAMILIES, METHODS, STAGE_SLOTS, Cell, cells
from grasppanda.combine.presets import presets

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
PAGE = REPO / "docs" / "advanced_guides" / "measured_combinations.md"
EXPERIMENTAL = REPO / "configs" / "experimental"

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
    for cell in wanted:
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
        if record is not None and record["verdict"] in WORKS and not own:
            name = " + ".join(f"`{chosen[slot]}`" for slot in ("backbone", *STAGE_SLOTS))
            numbers.append(_numbers_row(name, records.probed(record)))
    counts = Counter(records.get(cell.id)["verdict"] for cell in recorded)
    summary = ", ".join(f"{counts[v]} `{v}`" for v in VERDICT_ORDER if counts[v])
    headers = ["Encoder output", "Encoder", "Proposal", "View stage", "Assigner", "Grouper"]
    lines = [
        f"{len(recorded)} of the {len(wanted)} sets were probed: {summary}. A set states all "
        "five stages and the encoder: the preset named, under which the set needs no adapter. "
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
KNOWN_GAPS: dict[str, str] = {
    "gated seed residual adds N-channel seed features": (
        "Why the check does not see it: the contract rules compare what components "
        "declare, and no contract states the width an encoder returns; it is known once "
        "the encoder is built. An encoder with an `out_channels` option does not meet the "
        "gap, because compose sets that option to the width of the encoder it replaces. "
        "The encoders above have no such option and return their own width, which this "
        "grouper cannot add to its 256-channel crops. Until a width contract closes the "
        "gap, `grasppanda probe` is what catches it, and the ways out are the three of the "
        "message."
    ),
}


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
        "time, the legal sets of stage variants, HGGD as it is, and every image-encoder preset "
        "under HGGD. The variants of a stage "
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
    lines += ["## Passed the check, failed afterwards", "", *findings_section(cells_)]
    missing = about.get("not_probed") or []
    lines += [
        "## Not on this page",
        "",
        "- Accuracy of any kind. See the method pages of the [model zoo](../model_zoo.md).",
        "- Other datasets, cameras and recipes than the ones named above.",
        "- Combinations of two or more replaced stages beyond the stage sets.",
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


# ------------------------------------------------------------------------- staleness


def experimental() -> dict[Path, str]:
    """Every composed config under ``configs/experimental`` → what compose writes for it."""
    return {path: recompose(path).text(path) for path in sorted(EXPERIMENTAL.glob("*.yaml"))}


def expected() -> dict[Path, str]:
    pages = experimental()
    if DATA.is_file():
        pages[PAGE] = page()
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
