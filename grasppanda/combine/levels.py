"""What a measured combination reached, level by level, and the class of an option.

A cell of the measured grid (a record of ``docs/advanced_guides/measured_combinations.json``,
written by ``tools/probe_grid.py``) is decided at four levels, from the record alone:

``composes``
    the config check accepts the composed config;
``runs``
    the probe ran clean on real frames;
``learns``
    every loss term fell, every unit received a gradient, every part changed;
``predicts``
    every predicted frame has grasps, and their scores are not all the same.

:data:`LEVELS` holds the definitions, with the thresholds of this module and of
:mod:`grasppanda.combine.probe` spelled out; the documentation pages print them from here.
A level is ``yes``, ``no`` or undecided, and an undecided level says why
(:data:`UNDECIDED`): a record of the first judge has no field to decide ``learns`` and
``predicts`` from (``not checked``), an unstable run that was not probed again decides
nothing yet (``unstable, not repeated``), a refusal by policy waits for the user's consent.

The class of an option at a level counts the methods under which it reached that level
(:func:`option_class`): ``universal`` only when it reached the level under every method it
applies to and those include all five seed-based methods; ``most``, ``some``, ``specific``
and ``none`` for a stage of the seed-based skeleton decided under all five; otherwise
``N of M probed``, counted over the methods where the level was decided, with the others
named. A class counts verdicts of one short run per cell, not accuracy (:data:`CAVEAT`).

Nothing here probes or composes anything.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from grasppanda.config.errors import REFUSAL_KINDS

from .probe import FLAT_BELOW, JUDGE

__all__ = [
    "ACTIVE_SHARE",
    "BELOW_REFERENCE",
    "CAVEAT",
    "CLASS_WORDS",
    "FELL_BY",
    "LEVELS",
    "REFUSAL_KINDS",
    "REPEAT",
    "UNDECIDED",
    "CellLevels",
    "OptionClass",
    "cell_levels",
    "option_class",
    "refusal_kinds",
]

#: What every place that shows a class says next to it.
CAVEAT = (
    "one run, seed 0, 20 iterations, 8 frames of one scene; a class counts verdicts, not accuracy"
)
#: A loss term fell when its fall, (first - last quarter) / |first| over the iterations in
#: which it was active, is at least this: the margin below which the probe calls it `flat`.
FELL_BY = FLAT_BELOW
#: A loss term counts for ``learns`` when it was active (reported, finite, not exactly 0)
#: in at least this share of the iterations that ran. A term that is 0 in an iteration has
#: nothing to supervise there (the frame term of GraspNet-baseline when no seed has a
#: label it applies to, 6 of 20 iterations in the example of the first judge-2 probes), so
#: a term that is 0 at the last iteration (`zero_at_end`) is judged on its active ones.
ACTIVE_SHARE = 0.5
#: The field of a record that holds a second probe of the same config, same seed and same
#: schedule: written by ``tools/probe_grid.py`` for a cell whose first probe was
#: ``runs-unstable``. It holds the compact fields of that probe (``verdict``, ``unstable``,
#: ``judge``, ``iterations``, ``grasps_per_frame``, ``trend``, ``active``, ``flags``,
#: ``no_gradient``, ``unchanged``, ``score_quantiles``, ``run``); a bare verdict string is
#: read as ``{"verdict": ...}``.
REPEAT = "repeat"

YES, NO = "yes", "no"
NOT_CHECKED = "not checked"
NOT_REPEATED = "unstable, not repeated"
CONSENT = "needs consent"
NOT_A_COMBINATION = "not a combination"
NOT_PROBED = "not probed"
BELOW_REFERENCE = "not reached by the method as shipped"
#: The levels a cell is held against its method's reference for (:meth:`CellLevels.gated`).
GATED = ("learns", "predicts")
#: Why a level can be undecided in a cell, and what that means.
UNDECIDED: dict[str, str] = {
    NOT_CHECKED: (
        "the record was judged by the first judge, which recorded neither the trend of the "
        "loss terms nor the scores of the predicted grasps"
    ),
    NOT_REPEATED: (
        "the probe was `runs-unstable` and has not been repeated: an unstable run counts as "
        "not running only when a second probe of the same config is unstable too"
    ),
    CONSENT: (
        "refused by a policy that waits for the user's consent (`trust_remote_code` needs "
        "`accept_remote_code: true`): a rule about this machine, not about the combination"
    ),
    NOT_A_COMBINATION: (
        "the request names a component that lacks an option only a preset supplies: it is "
        "refused before any contract is compared, and is no combination"
    ),
    NOT_PROBED: "the grid has the cell, but no record yet",
    BELOW_REFERENCE: (
        "the method as shipped, probed the same way, does not reach this level either (for "
        "example FineGrasp at the constant rate, whose fusion of crop scales gives every seed "
        "one feature in a short run, so every grasp gets one score): under that method the "
        "level says nothing about the option at this budget"
    ),
}

#: The levels in order, each with its definition. Every level needs the one before it.
LEVELS: dict[str, str] = {
    "composes": (
        "the contract rules and the config check accept the composed config: `grasppanda "
        "compose` writes it (any verdict but `refused`). A refusal by policy is undecided "
        f"({CONSENT}), and so is a request that is {NOT_A_COMBINATION}"
    ),
    "runs": (
        "the probe ran clean on real frames: the verdict `runs` or `runs-empty`. A "
        "`runs-unstable` probe counts as not running only when its repeat (`repeat` in the "
        f"record: the same config, seed and schedule) was unstable too; without a repeat the "
        f"level is undecided ({NOT_REPEATED}); when the repeat ran clean the cell runs, and "
        "`learns` and `predicts` are read from the repeat. `fails` does not run"
    ),
    "learns": (
        "judge 2 only. Every loss term was active (reported, finite, not exactly 0) in at "
        f"least {ACTIVE_SHARE * 100:g} % of the iterations and fell by at least "
        f"{FELL_BY * 100:g} % (fall = (first - mean of the last quarter) / |first| over its "
        "active iterations), and no term is flagged `flat`, `rising` or `constant`; a term "
        "that is 0 at the last iteration (`zero_at_end`) is judged on its active iterations. "
        "Every unit with trainable parameters received a gradient (`no_gradient` is empty) "
        "and every part changed (`unchanged` is empty)"
    ),
    "predicts": (
        "judge 2 only. Every predicted frame has at least one grasp, and on every frame the "
        "smallest and the largest score differ (`score_quantiles`): a model whose scores "
        "collapsed to one value ranks nothing"
    ),
}


#: The class words for a stage decided under all five seed-based methods: the number of
#: methods under which the option reached the level.
CLASS_WORDS: dict[str, tuple[range, str]] = {
    "universal": (range(5, 6), "it reached the level under all five seed-based methods"),
    "most": (range(3, 5), "under three or four of them"),
    "some": (range(2, 3), "under two"),
    "specific": (range(1, 2), "under exactly one"),
    "none": (range(0, 1), "under none of them"),
}


# ------------------------------------------------------------------------ refusals


def _kind_of(entry: Any) -> tuple[str, str]:
    if isinstance(entry, Mapping):
        return str(entry.get("kind") or ""), str(entry.get("lift") or "")
    return (str(entry) if entry else ""), ""


def refusal_kinds(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """``[{message, kind, lift}]`` for every refusal message of a record.

    The kinds are those the record carries (``refusal_kinds``: a list parallel to
    ``refusal`` of kind names or of ``{"kind", "lift"}``, or a mapping from message to
    either; ``refusal_lifts``, a list parallel to ``refusal``, may hold the lifts). A
    record exported before compose reported kinds has none: its policy refusal is still
    told by its message (``accept_remote_code``), every other kind is left empty.
    """
    messages = list(record.get("refusal") or [])
    given = record.get("refusal_kinds")
    lifts = list(record.get("refusal_lifts") or [])
    found = []
    for index, message in enumerate(messages):
        entry: Any = None
        if isinstance(given, Mapping):
            entry = given.get(message)
        elif isinstance(given, Sequence) and not isinstance(given, str) and index < len(given):
            entry = given[index]
        kind, lift = _kind_of(entry)
        if not lift and index < len(lifts):
            lift = str(lifts[index] or "")
        if not kind and "accept_remote_code" in message:
            kind = "policy"
        found.append({"message": message, "kind": kind, "lift": lift})
    return found


def refusal_kind(record: Mapping[str, Any]) -> str:
    """The kind of a refused cell as a whole: ``policy`` or ``not-a-combination`` only when
    every message is of that kind; else ``structural`` when one is, else
    ``not-implemented`` when one is; ``""`` when the record does not say."""
    kinds = [entry["kind"] for entry in refusal_kinds(record)]
    if not kinds:
        return ""
    for whole in ("policy", "not-a-combination"):
        if all(kind == whole for kind in kinds):
            return whole
    for some in ("structural", "not-implemented"):
        if some in kinds:
            return some
    return next((kind for kind in kinds if kind), "")


# ---------------------------------------------------------------------- the levels


@dataclass(frozen=True)
class CellLevels:
    """The four levels of one cell.

    Attributes:
        states: Level → ``yes``, ``no`` or a key of :data:`UNDECIDED`.
        why: Level → what kept it from ``yes`` (short phrases), for a level that is ``no``.
        notes: What a reader should know although it decides nothing (a term that is 0 at
            the end, a run that was unstable once and ran when repeated).
        repeated: The verdict of the repeat, when the record has one.
    """

    states: dict[str, str]
    why: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    repeated: str = ""

    def reached(self, level: str) -> bool:
        return self.states.get(level) == YES

    def highest(self) -> str:
        """The last level reached in a row from ``composes``; ``""`` for none."""
        found = ""
        for level in LEVELS:
            if self.states.get(level) != YES:
                break
            found = level
        return found

    def gated(self, reference: CellLevels) -> CellLevels:
        """These levels held against the reference cell of the method: a level of
        :data:`GATED` that neither reaches is :data:`BELOW_REFERENCE`, not ``no``. A level
        the cell reaches stays reached; one the reference reaches stays as it is.

        Only a cell that trained is held against its reference: a cell that was refused,
        failed or did not run stays ``no`` at every later level, since its own failure, not
        the method's, kept it from them."""
        states, why, notes = dict(self.states), dict(self.why), list(self.notes)
        if states.get("runs") != YES:
            return CellLevels(states, why, notes, self.repeated)
        for level in GATED:
            if states.get(level) == NO and reference.states.get(level) == NO:
                states[level] = BELOW_REFERENCE
                own = why.pop(level, [])
                shipped = "; ".join(reference.why.get(level, [])) or "no"
                notes.append(
                    f"{level}: not reached, as the method as shipped does not reach it "
                    f"({shipped})" + (f"; here: {'; '.join(own)}" if own else "")
                )
        return CellLevels(states, why, notes, self.repeated)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"states": dict(self.states)}
        if self.why:
            out["why"] = {level: list(items) for level, items in self.why.items()}
        if self.notes:
            out["notes"] = list(self.notes)
        if self.repeated:
            out["repeated"] = self.repeated
        return out


def _percent(value: float) -> str:
    return f"{abs(value) * 100:.0f} %"


def _learning(fields: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """What kept a probe of judge 2 from ``learns``, and the notes (zero at the end)."""
    trend = fields.get("trend") or {}
    ran = int(fields.get("iterations") or 0)
    active = fields.get("active") or {}
    flags = fields.get("flags") or {}
    why, notes = [], []
    terms = [term for term in trend if term.startswith("loss/")]
    if not terms:
        why.append("no loss term recorded")
    for term in terms:
        name = term.removeprefix("loss/")
        values = list(trend[term]) + [None] * 3
        fall = values[2]
        count = int(active.get(term, ran))
        held = set(flags.get(term, []))
        if fall is None or count == 0:
            why.append(f"{name} never active")
            continue
        if ran and count < ACTIVE_SHARE * ran:
            why.append(f"{name} active in {count} of {ran} iterations")
            continue
        if "zero_at_end" in held:
            notes.append(f"{name} is 0 at the last iteration (active in {count} of {ran})")
        if "constant" in held:
            why.append(f"{name} constant")
        elif "rising" in held:
            why.append(f"{name} rising")
        elif "flat" in held:
            why.append(f"{name} flat")
        elif fall < FELL_BY:
            why.append(f"{name} {'rose' if fall < 0 else 'fell only'} {_percent(fall)}")
    if fields.get("no_gradient"):
        why.append("no gradient: " + ", ".join(fields["no_gradient"]))
    if fields.get("unchanged"):
        why.append("unchanged: " + ", ".join(fields["unchanged"]))
    return why, notes


def _prediction(fields: Mapping[str, Any]) -> list[str]:
    """What kept a probe of judge 2 from ``predicts``."""
    grasps = list(fields.get("grasps_per_frame") or [])
    quantiles = list(fields.get("score_quantiles") or [])
    if not grasps:
        return ["no frame was predicted"]
    why = []
    empty = [index + 1 for index, count in enumerate(grasps) if not count]
    if empty and len(empty) == len(grasps):
        why.append("no grasp on any frame")
    elif empty:
        why.append("no grasp on frame " + ", ".join(map(str, empty)))
    for index, found in enumerate(quantiles):
        if found and found[0] == found[-1] and grasps[index : index + 1] != [0]:
            why.append(f"every score of frame {index + 1} is {found[0]:g}")
    if len(quantiles) < len(grasps):
        why.append("the scores were not recorded for every frame")
    return why


def _repeat(record: Mapping[str, Any]) -> dict[str, Any]:
    found = record.get(REPEAT)
    if isinstance(found, str):
        return {"verdict": found}
    return dict(found) if isinstance(found, Mapping) else {}


def cell_levels(record: Mapping[str, Any], numbers: Mapping[str, Any] | None = None) -> CellLevels:
    """The levels of one cell.

    Args:
        record: The cell's record (its verdict, refusal, failure).
        numbers: The record that holds its probe's numbers: the cell named by
            ``probed_as`` when it shares a probe; the record itself by default.
    """
    numbers = numbers if numbers is not None else record
    verdict = record.get("verdict", "")
    states: dict[str, str] = {}
    why: dict[str, list[str]] = {}
    notes: list[str] = []

    def rest(state: str, *, start: str) -> None:
        for level in list(LEVELS)[list(LEVELS).index(start) :]:
            states[level] = state

    failure = record.get("failure") or {}
    if verdict == "refused":
        kind = refusal_kind(record)
        undecided = {"policy": CONSENT, "not-a-combination": NOT_A_COMBINATION}.get(kind)
        if undecided:
            rest(undecided, start="composes")
        else:
            rest(NO, start="composes")
            why["composes"] = ["refused" + (f" ({kind})" if kind else "")]
        return CellLevels(states, why)
    if failure.get("phase") == "compose":
        rest(NO, start="composes")
        why["composes"] = ["compose itself failed"]
        return CellLevels(states, why)
    states["composes"] = YES
    repeat = _repeat(numbers) or _repeat(record)
    fields: Mapping[str, Any] = numbers
    if verdict in ("runs", "runs-empty"):
        states["runs"] = YES
    elif verdict == "runs-unstable":
        again = repeat.get("verdict", "")
        if not again:
            rest(NOT_REPEATED, start="runs")
            return CellLevels(states, why, notes)
        if again in ("runs", "runs-empty"):
            states["runs"] = YES
            notes.append(f"unstable in the first probe; the repeat was `{again}`")
            fields = repeat
        else:
            rest(NO, start="runs")
            why["runs"] = [f"unstable, and the repeat was `{again}`"]
            return CellLevels(states, why, notes, again)
    else:
        rest(NO, start="runs")
        why["runs"] = [verdict or "no verdict"]
        return CellLevels(states, why, notes, repeat.get("verdict", ""))
    again = repeat.get("verdict", "")
    if int(fields.get("judge", 1) or 1) < JUDGE:
        rest(NOT_CHECKED, start="learns")
        return CellLevels(states, why, notes, again)
    learned, zero = _learning(fields)
    notes += zero
    states["learns"] = NO if learned else YES
    if learned:
        why["learns"] = learned
    predicted = _prediction(fields)
    states["predicts"] = NO if predicted else YES
    if predicted:
        why["predicts"] = predicted
    return CellLevels(states, why, notes, again)


# ---------------------------------------------------------------------- the classes


@dataclass(frozen=True)
class OptionClass:
    """The class of one option at one level.

    Attributes:
        level: A key of :data:`LEVELS`.
        name: ``universal``, ``most``, ``some``, ``specific``, ``none``, ``N of M probed``,
            or, when no method decided the level, the reason (``not checked``, ...).
        reached: Methods under which the option reached the level.
        decided: Methods under which the level was decided (``yes`` or ``no``).
        methods: Methods the option applies to.
        undecided: Reason → number of methods, for the others.
        checked: Methods under which the level was decided although the level before it
            was reached: a decision of this level, not one inherited from a failure below.
            ``None`` when it was not counted.
    """

    level: str
    name: str
    reached: int
    decided: int
    methods: int
    undecided: tuple[tuple[str, int], ...] = ()
    checked: int | None = None

    @property
    def text(self) -> str:
        """The name with the undecided methods named after it: ``3 of 3 probed, +2 not
        checked``; for a level no method decided, the reasons with their counts when they
        differ."""
        rest = ", ".join(f"{count} {reason}" for reason, count in self.undecided)
        if not self.decided:
            return rest if len(self.undecided) > 1 else self.name
        return f"{self.name}, +{rest}" if self.undecided else self.name

    @property
    def relative(self) -> bool:
        return self.name.endswith(" probed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "reached": self.reached,
            "decided": self.decided,
            "methods": self.methods,
            "undecided": dict(self.undecided),
        }

    def with_checked(self, checked: int) -> OptionClass:
        return OptionClass(
            self.level, self.name, self.reached, self.decided, self.methods, self.undecided, checked
        )


def option_class(
    level: str,
    states: Mapping[str, str],
    methods: Iterable[str],
    seed_based: Sequence[str],
) -> OptionClass:
    """The class of an option at ``level``.

    Args:
        level: A key of :data:`LEVELS`.
        states: Method → the state of the level in the option's cell under that method;
            a method without an entry has no record (``not probed``).
        methods: Every method the option applies to (has a cell for, recorded or not).
        seed_based: The five seed-based methods.
    """
    applies = list(dict.fromkeys(methods))
    found = {method: states.get(method, NOT_PROBED) for method in applies}
    reached = sum(state == YES for state in found.values())
    decided = sum(state in (YES, NO) for state in found.values())
    undecided = Counter(state for state in found.values() if state not in (YES, NO))
    order = tuple(sorted(undecided.items(), key=lambda item: (-item[1], item[0])))
    if not decided:
        name = order[0][0] if len(order) == 1 else "undecided"
        if not applies:
            name = NOT_PROBED
        return OptionClass(level, name, 0, 0, len(applies), order)
    every_seed = set(seed_based) <= set(applies)
    if not undecided and every_seed and reached == decided:
        name = "universal"
    elif not undecided and set(applies) == set(seed_based):
        name = next(word for word, (count, _) in CLASS_WORDS.items() if reached in count)
    else:
        name = f"{reached} of {decided} probed"
    return OptionClass(level, name, reached, decided, len(applies), order)
