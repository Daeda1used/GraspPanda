"""Where an option was measured to work: its reach at one level, over the methods it applies to.

The pages, the explorer and ``grasppanda compose --list`` say where an option works with
the words of this module, so the three cannot disagree. The rules:

* the denominator is every method the option is a swap under: the methods it applies to
  (has a cell under), less the methods that ship it (its cell there *is* the method as
  shipped, told apart by the config hash of the method's reference record);
* a method under which the level is undecided stays in the denominator and is named with
  the reason, so a fraction is never better than the records;
* a method under which the option was refused or failed is named with why, in plain words
  (:data:`REFUSAL_WORDS`, :data:`VERDICT_WORDS`, the level's own reasons, and the known
  failures of :data:`grasppanda.combine.grid.KNOWN_FAILURES`).

Level 2 (``learns_like_reference.json``: 250 iterations beside the method's own component)
is read by :func:`level2`; post-processing has no level 2, and a schedule is not compared
(the reference runs at a constant rate).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .levels import (
    BELOW_REFERENCE,
    CONSENT,
    NO,
    NOT_A_COMBINATION,
    NOT_CHECKED,
    NOT_PROBED,
    NOT_REPEATED,
    YES,
    refusal_kind,
)

__all__ = [
    "LEVEL2_TONE",
    "REFUSAL_WORDS",
    "UNDECIDED_WORDS",
    "VERDICT_WORDS",
    "Reach",
    "level2",
    "level2_file",
    "method_order",
    "reach",
]

#: Refusal kind (``grasppanda.config.REFUSAL_KINDS``) → the words a page uses for it.
REFUSAL_WORDS = {
    "structural": "incompatible by design",
    "not-implemented": "not supported yet",
    "fixed-by-compose": "needs a setting compose adds",
    "not-a-combination": "needs a preset's settings",
    "policy": "needs your consent",
}
#: A level-1 verdict that did not run → its words.
VERDICT_WORDS = {
    "fails": "fails",
    "runs-unstable": "unstable twice in the short probe",
}
#: Why a level is undecided (``levels.UNDECIDED``) → the words a page uses for it.
UNDECIDED_WORDS = {
    NOT_CHECKED: "not checked by the current judge",
    NOT_REPEATED: "unstable once, not repeated",
    CONSENT: "needs your consent (remote code)",
    NOT_A_COMBINATION: "needs a preset's settings",
    NOT_PROBED: "not probed yet",
    BELOW_REFERENCE: "the method as shipped does not reach it in a short run either",
}
#: Level-2 outcome → its tone; the worst outcome of an option sets the tone of its chip.
LEVEL2_TONE = {
    "on a par": "ok",
    "worse": "warn",
    "not comparable": "muted",
    "diverging": "no",
    "fails": "no",
}
#: Level-2 outcomes from worst to best.
LEVEL2_ORDER = ("fails", "diverging", "worse", "not comparable", "on a par")
#: The words a page uses for a level-2 outcome.
LEVEL2_WORDS = {"fails": "did not train"}
#: Slots without a level-2 comparison, and why.
LEVEL2_NONE = {
    "post": "post-processing trains nothing: it changes predictions, not the training",
    "schedule": (
        "level 2 holds the schedule of the method as shipped fixed (a constant rate; FineGrasp: "
        "its own warm-up), so a schedule swap is not compared"
    ),
}


def method_order() -> tuple[str, ...]:
    """Every method in page order: the seed-based skeleton by year, then HGGD."""
    from .grid import FAMILIES, METHODS

    return (*METHODS, *FAMILIES)


def _sorted(names: Sequence[str]) -> list[str]:
    order = method_order()
    return sorted(dict.fromkeys(names), key=lambda name: order.index(name))


def _join(names: Sequence[str]) -> str:
    return ", ".join(names)


@dataclass(frozen=True)
class Reach:
    """Where one option reached one level.

    Attributes:
        level: ``composes``, ``runs``, ``learns`` or ``predicts``.
        applies: The methods the option has a cell under, in page order.
        own: The methods that ship it (not counted as swaps).
        reached: The methods under which the swap reached the level.
        missed: Method → why the swap did not reach the level there (decided ``no``).
        undecided: Method → why the level is undecided there.
        note: For an option the grid has no cell for (an adapter compose inserts, the
            point-painting wrapper, a post-processor no dataset can feed): why.
    """

    level: str
    applies: tuple[str, ...]
    own: tuple[str, ...] = ()
    reached: tuple[str, ...] = ()
    missed: dict[str, str] = field(default_factory=dict)
    undecided: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def swaps(self) -> tuple[str, ...]:
        return tuple(name for name in self.applies if name not in self.own)

    @property
    def measured(self) -> bool:
        return bool(self.reached or self.missed)

    @property
    def fraction(self) -> str:
        """``3/5``; ``HGGD ✓`` / ``HGGD ✗`` / ``HGGD ?`` for an option of one method;
        ``own only`` when every method it applies to ships it; ``not measured``."""
        swaps = self.swaps
        if self.note and not self.applies:
            return "no record"
        if not self.applies:
            return "not measured"
        if not swaps:
            return "own only"
        if len(swaps) == 1 and len(self.applies) == 1:
            name = swaps[0]
            mark = "✓" if self.reached else "✗" if self.missed else "?"
            return f"{name} {mark}"
        if not self.measured:
            return "not measured"
        return f"{len(self.reached)}/{len(swaps)}"

    @property
    def tone(self) -> str:
        """``ok`` (every swap reached), ``warn`` (some), ``no`` (none), ``muted`` (nothing
        decided)."""
        swaps = self.swaps
        if not swaps or not self.measured:
            return "muted"
        if len(self.reached) == len(swaps):
            return "ok"
        return "warn" if self.reached else "no"

    @property
    def tail(self) -> str:
        """The methods the fraction leaves out and why: ``own in EconomicGrasp, FineGrasp;
        not GraspNet-baseline, Scale-Balanced-Grasp (incompatible by design)``."""
        if self.note and not self.applies:
            return self.note
        parts = [self.note] if self.note else []
        if self.own:
            parts.append(f"own in {_join(self.own)}")
        swaps = self.swaps
        groups: dict[str, list[str]] = {}
        for name, words in self.missed.items():
            groups.setdefault(words, []).append(name)
        if swaps and not self.reached and len(groups) == 1 and not self.undecided:
            parts.append(next(iter(groups)))
        else:
            parts += [f"not {_join(names)} ({words})" for words, names in groups.items()]
        undecided: dict[str, list[str]] = {}
        for name, words in self.undecided.items():
            undecided.setdefault(words, []).append(name)
        parts += [f"{_join(names)} undecided ({words})" for words, names in undecided.items()]
        if len(swaps) == 1 and len(self.applies) == 1:
            # one method: the mark says it, the reason follows without repeating the name
            name = swaps[0]
            parts = [
                part.replace(f"not {name} (", "(", 1).replace(
                    f"{name} undecided (", "undecided (", 1
                )
                for part in parts
            ]
        return "; ".join(parts)

    @property
    def text(self) -> str:
        tail = self.tail
        return f"{self.fraction} · {tail}" if tail else self.fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "fraction": self.fraction,
            "tone": self.tone,
            "tail": self.tail,
            "applies": list(self.applies),
            "own": list(self.own),
            "reached": list(self.reached),
            "missed": dict(self.missed),
            "undecided": dict(self.undecided),
            "note": self.note,
        }


#: Why the grid has no cell for an option (``options.Option.reason``) → the words of a page.
NO_CELL_WORDS = {
    "an encoder adapter": "an adapter: compose inserts it where the next stage needs it, so "
    "it is measured inside the cells it was inserted in",
    "a wrapper around the method's encoder (point painting)": "point painting: a wrapper "
    "written into a config by hand (compose does not write it); no probe record",
}


#: Loss presets that fit only a term one head bundle declares, so they are measured inside
#: that bundle's cells: preset → (the head preset, the term).
LOSS_INSIDE = {"distribution-focal": ("decoupled-grid-bins", "width")}


def _no_cell(slot: str, name: str) -> str:
    """Why the grid has no cell for option ``name`` of ``slot``; ``""`` when it has one."""
    from .options import _extra_options

    for _, option in _extra_options():
        if (option.slot, option.name) == (slot, name) and option.reason:
            for start, words in NO_CELL_WORDS.items():
                if option.reason.startswith(start):
                    if start == "an encoder adapter":
                        words += _inserted(option.component)
                    return words
            return option.reason.split(":")[0]
    return "no cell in the grid"


def _inserted(component: str) -> str:
    """``: inserted in 65 cells (43 runs, 18 runs-empty, 4 refused)`` for an adapter."""
    from .options import measured

    verdicts: dict[str, int] = {}
    for record in measured().by_id.values():
        if record.get("adapter") == component:
            verdict = str(record.get("verdict", ""))
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    if not verdicts:
        return ""
    told = ", ".join(f"{count} {verdict}" for verdict, count in verdicts.items())
    return f"; inserted in {sum(verdicts.values())} cells ({told})"


def _shipped(records: Any, method: str, record: Mapping[str, Any]) -> bool:
    """Whether ``record`` is the method as shipped (its config hash is the reference's)."""
    from .grid import SHIPPED_EQUIVALENTS

    if record.get("id") in SHIPPED_EQUIVALENTS:
        return True
    reference = records.reference(method)
    if reference is None or reference["id"] == record["id"]:
        return False
    found = record.get("config_hash")
    return bool(found) and found == reference.get("config_hash")


def _why_missed(records: Any, record: Mapping[str, Any], level: str) -> str:
    """Why a cell did not reach ``level``, in plain words."""
    from .grid import KNOWN_FAILURES

    known = KNOWN_FAILURES.get(str(record.get("id", "")))
    if known is not None:
        return known[0]
    levels = records.levels(record)
    verdict = str(record.get("verdict", ""))
    if verdict == "refused":
        return REFUSAL_WORDS.get(refusal_kind(record), "refused")
    if levels.states.get("composes") != YES:
        return "compose failed"
    if levels.states.get("runs") != YES:
        return VERDICT_WORDS.get(verdict, verdict or "did not run")
    reasons = levels.why.get(level) or []
    if level == "learns":
        return "does not learn: " + "; ".join(reasons[:2]) if reasons else "does not learn"
    if level == "predicts":
        return "does not predict: " + "; ".join(reasons[:2]) if reasons else "does not predict"
    return "; ".join(reasons[:2]) or f"does not reach {level}"


def reach(slot: str, name: str, level: str = "runs", records: Any = None) -> Reach:
    """Where option ``name`` of ``slot`` (a loss preset for ``slot == "loss"``) reached
    ``level``, from the level-1 records."""
    from .options import _class_index, _state, measured

    records = records or measured()
    if slot == "loss" and name in LOSS_INSIDE:
        head, term = LOSS_INSIDE[name]
        inside = reach("head", head, level, records)
        note = f"fits only the {term} term of the head bundle `{head}`; measured with it"
        return Reach(
            level, inside.applies, inside.own, inside.reached, inside.missed, inside.undecided, note
        )
    by_method = _class_index().get((slot, name), {})
    if not any(by_method.values()):
        return Reach(level, (), note=_no_cell(slot, name) if by_method else "")
    applies = _sorted(list(by_method))
    own, reached = [], []
    missed: dict[str, str] = {}
    undecided: dict[str, str] = {}
    for method in applies:
        recorded = [records.get(cell) for cell in by_method[method]]
        recorded = [record for record in recorded if record is not None]
        if not recorded:
            undecided[method] = UNDECIDED_WORDS[NOT_PROBED]
            continue
        if all(_shipped(records, method, record) for record in recorded):
            own.append(method)
            continue
        states = [_state(records.levels(record), level) for record in recorded]
        if YES in states:
            reached.append(method)
        elif NO in states:
            record = recorded[states.index(NO)]
            missed[method] = _why_missed(records, record, level)
        else:
            undecided[method] = UNDECIDED_WORDS.get(states[0], states[0])
    return Reach(level, tuple(applies), tuple(own), tuple(reached), missed, undecided)


# ---------------------------------------------------------------------------- level 2


def level2_file() -> Path:
    """The level-2 summary: packaged in a wheel, ``docs/advanced_guides/`` in a checkout."""
    package = Path(__file__).resolve().parents[1]
    packaged = package / "_measured" / "learns_like_reference.json"
    if packaged.is_file():
        return packaged
    return package.parent / "docs" / "advanced_guides" / "learns_like_reference.json"


@lru_cache(maxsize=2)
def _level2_cells(path: str) -> dict[str, dict[str, Any]]:
    file = Path(path)
    if not file.is_file():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    return {cell["id"]: cell for cell in payload.get("cells") or []}


@dataclass(frozen=True)
class Level2:
    """How option ``name`` trains beside each method's own component (250 iterations).

    Attributes:
        outcomes: Method → ``on a par``, ``worse``, ``diverging``, ``not comparable`` or
            ``fails``, for the methods it was compared under (swaps only).
        excluded: Method → why a level-2 cell there is not compared (the swap did not run
            at level 1; it trained on a schedule of its own while the method's reference
            ran at a constant rate).
        none: Why the slot has no level 2 (``""`` when it has).
    """

    outcomes: dict[str, str]
    none: str = ""
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def compared(self) -> list[str]:
        return [m for m, outcome in self.outcomes.items() if outcome != "not comparable"]

    @property
    def on_a_par(self) -> list[str]:
        return [m for m, outcome in self.outcomes.items() if outcome == "on a par"]

    @property
    def fraction(self) -> str:
        """``on a par 5/5``; ``4/5 on a par · 1 diverging`` when not all are, so the words
        say what the colour says; ``HGGD: worse`` for an option of one method;
        ``not compared``; ``—`` for a slot without level 2."""
        if self.none:
            return "—"
        compared = self.compared
        if not compared:
            return "not compared"
        if len(self.outcomes) == 1 and len(compared) == 1:
            method, outcome = next(iter(self.outcomes.items()))
            return f"{method}: {LEVEL2_WORDS.get(outcome, outcome)}"
        par = len(self.on_a_par)
        if par == len(compared):
            return f"{par}/{len(compared)} on a par"
        counts: dict[str, int] = {}
        for outcome in self.outcomes.values():
            if outcome not in ("on a par", "not comparable"):
                words = LEVEL2_WORDS.get(outcome, outcome)
                counts[words] = counts.get(words, 0) + 1
        order = [LEVEL2_WORDS.get(o, o) for o in LEVEL2_ORDER]
        rest = " · ".join(
            f"{n} {words}" for words, n in sorted(counts.items(), key=lambda i: order.index(i[0]))
        )
        return f"{par}/{len(compared)} on a par · {rest}"

    @property
    def tone(self) -> str:
        if self.none or not self.compared:
            return "muted"
        worst = min(
            (o for o in self.outcomes.values() if o != "not comparable"), key=LEVEL2_ORDER.index
        )
        return LEVEL2_TONE[worst]

    @property
    def tail(self) -> str:
        if self.none:
            return self.none
        groups: dict[str, list[str]] = {}
        for method, outcome in self.outcomes.items():
            if outcome != "on a par":
                groups.setdefault(LEVEL2_WORDS.get(outcome, outcome), []).append(method)
        parts = [f"{words} under {_join(names)}" for words, names in groups.items()]
        excluded: dict[str, list[str]] = {}
        for method, words in self.excluded.items():
            excluded.setdefault(words, []).append(method)
        parts += [
            f"not compared under {_join(names)}: {words}" for words, names in excluded.items()
        ]
        if len(self.outcomes) == 1 and len(self.compared) == 1:
            parts = [part for part in parts if not part.endswith(f"under {self.compared[0]}")]
        return "; ".join(parts)

    @property
    def text(self) -> str:
        tail = self.tail
        return f"{self.fraction} · {tail}" if tail and not self.none else self.fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "tone": self.tone,
            "tail": self.tail,
            "outcomes": dict(self.outcomes),
            "excluded": dict(self.excluded),
        }


#: Why a level-2 cell is left out of the comparison.
LEVEL2_EXCLUDED = {
    "runs": "the swap did not run at level 1",
    "schedule": "it trained on a schedule of its own, not on that of the method as shipped",
}


@lru_cache(maxsize=2)
def _level2_schedules(path: str) -> dict[str, str]:
    """Method → the schedule its level-2 reference ran on (``constant`` or ``recipe``)."""
    file = Path(path)
    if not file.is_file():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    return {
        str(entry["method"]): str(entry.get("schedule", "constant"))
        for entry in payload.get("references") or []
    }


def level2(slot: str, name: str, records: Any = None, path: str | Path | None = None) -> Level2:
    """The level-2 outcomes of option ``name`` of ``slot``: one per method it is a swap
    under. A method with several cells (a loss preset in several terms) is read from its
    single-term cells only; a method whose cells disagree is given the worst outcome. A
    method under which the swap did not run at level 1, or under which it trained on a
    schedule other than its reference's, is left out and named (:attr:`Level2.excluded`)."""
    from .options import _class_index, measured

    if slot in LEVEL2_NONE:
        return Level2({}, LEVEL2_NONE[slot])
    records = records or measured()
    source = str(path or level2_file())
    cells = _level2_cells(source)
    schedules = _level2_schedules(source)
    ran = set(reach(slot, name, "runs", records).reached)
    found: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for method, ids in _class_index().get((slot, name), {}).items():
        single = [cell for cell in ids if not _joint(records, cell)] or list(ids)
        recorded = [records.get(cell) for cell in single]
        if any(r is not None and _shipped(records, method, r) for r in recorded):
            continue
        entries = [cells[cell] for cell in single if cell in cells]
        if not entries:
            continue
        if method not in ran:
            excluded[method] = LEVEL2_EXCLUDED["runs"]
            continue
        reference = schedules.get(method, "constant")
        own = _sets_scheduler(slot, name)
        if own or any(str(entry.get("schedule", "constant")) != reference for entry in entries):
            excluded[method] = LEVEL2_EXCLUDED["schedule"]
            continue
        found[method] = min((entry["outcome"] for entry in entries), key=LEVEL2_ORDER.index)
    return Level2(
        {m: found[m] for m in _sorted(list(found))},
        excluded={m: excluded[m] for m in _sorted(list(excluded))},
    )


def _sets_scheduler(slot: str, name: str) -> bool:
    """Whether a recipe option writes a scheduler of its own (a schedule-free optimizer, a
    whole recipe): its cells did not train on their reference's schedule."""
    from .recipes import recipes

    found = recipes().get(slot, {}).get(name)
    return found is not None and "scheduler" in found.body


def _joint(records: Any, cell: str) -> bool:
    """Whether a loss cell replaces several terms at once (a hand-picked joint cell)."""
    record = records.get(cell)
    sets = list(record.get("sets") or []) if record is not None else []
    return len(sets) > 1
