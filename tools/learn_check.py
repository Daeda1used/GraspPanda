"""Level 2 of the measured grid: does an option train like the shipped component it replaces?

Level 1 (``tools/probe_grid.py``, 20 iterations on 8 frames) says that a cell composes,
builds, trains without blowing up and predicts. It does not say whether the option learns
like the part of the method it replaces. Level 2 trains the same cell longer, under the
same rule as level 1 so that methods compare, and puts its losses beside those of the
method as shipped (its *reference*) trained under exactly the same settings
(:data:`SETTINGS`): 250 iterations on the same 8 frames, seed 0, prediction on 4 frames,
``--schedule constant``. Per loss term it takes the mean over the last 25 iterations as a
ratio to the reference's, and judges the cell with the rule of :data:`OUTCOMES`; the
tolerance and the rising threshold were calibrated from the run-to-run spread of the
references themselves (:data:`CALIBRATION`).

A cell is one run. The outcome is a statement about training dynamics on 8 frames of one
scene, not an accuracy measurement.

The runs are those of ``tools/probe_grid.py``: one run directory per kind of cell,
``$GRASPPANDA_CACHE/probe_grid/L2-<kind>``, and for the references ``L2-reference`` (seed
0) and ``L2-reference-seed<N>`` (the noise), so an interrupted run continues where it
stopped and a cell that died of GPU memory is run again by the next ``run``. ``export``
compares, writes one file per cell under ``$GRASPPANDA_CACHE/verification/L2/`` and the
committed summary ``docs/advanced_guides/learns_like_reference.json`` (losses in a side
file beside it, as level 1 does); ``page`` writes the page from the summary, counting what
:func:`grasppanda.combine.reach.level2` counts (the slot pages and the explorer read it)::

    python -m tools.learn_check references --seeds 0 1 --scale-statistic-sha <sha>
    python -m tools.learn_check run --kind encoder --scale-statistic-sha <sha>
    python -m tools.learn_check run --kind grouper --only 'stage/GSNet/*' --scale-statistic-sha <sha>
    python -m tools.learn_check status          # what is selected, run, pending; time left
    python -m tools.learn_check export          # compare -> cache + committed summary
    python -m tools.learn_check page            # --check: exit 1 when the page is stale

The cells of a kind are the level-1 cells of that kind whose verdict is ``runs``,
``runs-empty`` or ``runs-unstable`` (read from the committed measured file, or from a
probe_grid run directory with ``--level1-run``), minus the cells whose config is the
reference's own. Post-processing (a slot that trains nothing) is left out
(:data:`LEFT_OUT`).
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

REPO = Path(__file__).resolve().parents[1]
LEVEL1 = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
SUMMARY = REPO / "docs" / "advanced_guides" / "learns_like_reference.json"
PAGE = REPO / "docs" / "advanced_guides" / "learns_like_reference.md"
#: The side file of the losses, beside the summary (see :func:`side_file`).
SERIES_NAME = "losses_per_iteration.json"

#: The settings every level-2 run shares, cells and references alike.
SETTINGS: dict[str, Any] = {
    "iterations": 250,
    "frames": 8,
    "predict_frames": 4,
    "split": "sample",
    "seed": 0,
    "schedule": "constant",
    "window": 25,
}
ITERATIONS, WINDOW = SETTINGS["iterations"], SETTINGS["window"]
#: Methods whose level-2 runs (reference and cells alike) keep the recipe's schedule instead
#: of the constant rate, and why. Their cells compare with their reference all the same.
METHOD_SCHEDULES: dict[str, tuple[str, str]] = {
    "FineGrasp": (
        "recipe",
        "at the constant 0.002 without the recipe's 800-iteration warm-up, the two post-norm "
        "transformer layers that fuse FineGrasp's four crop scales give every seed one feature "
        "(the reference implementation's fusion does the same), so the head learns the label "
        "distribution and every grasp gets one score until about iteration 530; under the "
        "warm-up the fusion recovers by iteration 80",
    ),
}


def method_of(cell_id: str) -> str:
    """The method of a cell id (``<group>/<method>/...``, ``reference/<method>``)."""
    parts = cell_id.split("/")
    return parts[1] if len(parts) > 1 else ""


def schedule_of(cell_id: str) -> str:
    """The schedule a level-2 run of ``cell_id`` uses (:data:`METHOD_SCHEDULES`)."""
    return METHOD_SCHEDULES.get(method_of(cell_id), (SETTINGS["schedule"], ""))[0]


#: The least tolerance: a cell is ``worse`` when a term's mean over the last
#: :data:`WINDOW` iterations is above this many times the reference's. The tolerance of a
#: method is this, or more when its references differ more between two seeds
#: (:func:`calibrate`, :data:`CALIBRATION`).
TOLERANCE = 1.25
#: The least rising threshold: a term is rising when the mean of the second half of the
#: last half of the run exceeds the mean of its first half by more than this fraction, and
#: by more than this fraction beyond the reference's own rise of the term. Calibrated per
#: method as :data:`TOLERANCE`.
RISING_ABOVE = 0.10
#: A calibrated tolerance or threshold is the first multiple of this above the spread.
STEP = 0.05
#: The side file keeps the mean of every this many iterations.
BLOCK = 5
#: Seconds a level-2 probe may take before it counts as failed.
TIMEOUT = 3600.0
#: Level-1 verdicts whose cells get a level-2 run.
WORKS = ("runs", "runs-empty", "runs-unstable")
#: Run directories (under ``$GRASPPANDA_CACHE/probe_grid``).
REFERENCE_RUN = "L2-reference"

#: Kind of cell -> (group of the grid, slot or None, title on the page).
KINDS: dict[str, tuple[str, str | None, str]] = {
    "encoder": ("encoder", None, "Scene encoders"),
    "proposal": ("stage", "proposal", "Seed proposals"),
    "view": ("stage", "view", "View stages"),
    "assigner": ("stage", "assigner", "Assigners"),
    "grouper": ("stage", "grouper", "Groupers"),
    "head": ("stage", "head", "Head bundles"),
    "loss": ("loss", None, "Losses"),
    "recipe": ("recipe", None, "Training recipe"),
    "image-encoder": ("family-slot", "image_encoder", "HGGD image encoders"),
}
#: Groups of the grid without a level-2 check, each with the reason.
LEFT_OUT: dict[str, str] = {
    "post": (
        "post-processing trains nothing, so a training run says nothing about it; its "
        "check is to compare the grasps in and out of the chain (and the evaluator) on 20 "
        "frames of a trained method, which is not part of this page yet"
    ),
    "stage-set, cross": (
        "several slots replaced at once: no single stage of the reference to compare with"
    ),
}

#: The outcomes of a cell, in order of precedence, and what each one means.
OUTCOMES: dict[str, str] = {
    "fails": (
        "the run did not finish: an exception, a crash, a timeout, or no real frame read. "
        "A run that died of GPU memory is not recorded (it says nothing about the config) "
        "and is run again"
    ),
    "no reference": (
        "the method as shipped has no finished run under the same settings: nothing to compare with"
    ),
    "diverging": (
        "a loss term is rising over the last half of the run (see `rising`), a loss or a "
        f"weight became non-finite, or no grasp was predicted on any of the "
        f"{SETTINGS['predict_frames']} frames while the reference predicts some"
    ),
    "worse": (
        "not diverging, and a comparable loss term ends above the tolerance of the method "
        f"times the reference's (mean of the last {WINDOW} iterations), or a part of the "
        "model that holds trainable parameters received no gradient in any iteration"
    ),
    "not comparable": (
        "not diverging, nothing worse, but the cell changes the definition of every term "
        "the reference has: there is no term to compare"
    ),
    "on a par": (
        "every comparable loss term ends within the tolerance of the method, at least one "
        "term is comparable, nothing rises, every value is finite and a frame has a grasp. "
        "A term far below the reference's is reported (`lower`), not judged"
    ),
}
#: How a term is measured, as the page and the summary state it.
MEASURES: dict[str, str] = {
    "mean": (
        f"the mean of a loss term over the last {WINDOW} of the {ITERATIONS} iterations "
        "(reported, finite values; the total `loss` is not judged, it sums terms whose "
        "definitions may differ)"
    ),
    "ratio": "the cell's mean divided by the reference's mean of the same term",
    "rise": (
        "over the last half of the run (iterations 126 to 250), the mean of its second "
        "half divided by the mean of its first half, minus 1 (the middle iteration of an "
        "odd count in neither half): +0.30 means the term ended 30 % higher than it was "
        "half-way"
    ),
    "tolerance": (
        f"per method: {TOLERANCE:g}, or the first multiple of {STEP:g} above the largest "
        "ratio between two seeds of its reference (any term but the total, either way "
        "round), when that is larger"
    ),
    "rising": (
        "a rise above the rising threshold of the method that also exceeds the "
        "reference's rise of the term by more than it (the rise alone for a term the "
        f"reference does not report); the threshold is {RISING_ABOVE:g}, or the first "
        f"multiple of {STEP:g} above the largest difference between the rises of one term "
        "under two seeds of the reference, when that is larger"
    ),
    "lower": "a comparable term below 1/tolerance of the reference's: reported, not judged",
}
#: Which terms are not comparable, and why (applied by :func:`not_comparable`).
NOT_COMPARABLE: dict[str, str] = {
    "loss": "a loss preset replaces the function of the term it is put in",
    "head": (
        "a head bundle defines the targets and the loss of every head term "
        "(score, width, angle, depth, tolerance)"
    ),
    "assigner": (
        "an assigner changes the labels the view stage and the head see (a view stage "
        "sees the labels through the assigner only): every view and head term"
    ),
    "stage code": (
        "a proposal or view stage owns its targets and its loss: one whose class computes "
        "them with other code (its own `loss` or target methods, not inherited from the "
        "reference's stage) defines every term of that stage anew; one that computes them "
        "as the reference's does, with the same loss options, keeps them (a proposal that "
        "changes only how it samples)"
    ),
    "presence": "a term only one of the two runs reports",
    "sign": "a term whose mean is not positive (a weighted loss can go below zero)",
}
#: Why the tolerance and the rising threshold are what they are. Written when they were
#: calibrated from the references (W3.0); the numbers are in the summary's ``noise``.
CALIBRATION = (
    "A cell and its reference are two runs, so a ratio away from 1 means something only "
    "beyond what two runs of the very same config show. Every reference was therefore "
    "trained twice, under seed 0 and seed 1 (*The noise floor*). Under the five seed-based "
    "methods, a term that has settled (it rose or fell by less than 10 % over the last half) "
    "agrees within 7 % between the seeds; a term still falling steeply at the end differs "
    "more (EconomicGrasp's graspness by 1.32x), and HGGD at its peak rate differs by up to "
    "1.55x (angle, offsets). One tolerance for all would be blind under the stable methods "
    f"or trigger-happy under HGGD, so each method gets its own: {TOLERANCE:g} (the plan's "
    "value, more than three times the spread of a settled term) unless its references "
    f"spread more, then the first multiple of {STEP:g} above their largest ratio. The rising "
    "threshold follows the same rule from the rises. The spread comes from one pair of runs "
    "per method: a third seed would firm it up"
)

STAGES_WITH_TERMS = ("proposal", "view", "head")


def _runner() -> Any:
    """``tools/probe_grid.py``, which runs every probe of level 2."""
    name = "probe_grid"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_name(kind: str) -> str:
    """The probe_grid run directory of a kind of cell."""
    return f"L2-{kind}"


def reference_run(seed: int) -> str:
    return REFERENCE_RUN if seed == SETTINGS["seed"] else f"{REFERENCE_RUN}-seed{seed}"


def reference_id(method: str) -> str:
    """The cell of the method as shipped: ``reference/<m>``, ``family/HGGD``."""
    from grasppanda.combine.grid import FAMILIES

    return f"family/{method}" if method in FAMILIES else f"reference/{method}"


def kind_of(record: Mapping[str, Any]) -> str | None:
    """The kind of a level-1 record (:data:`KINDS`), or None for a group left out."""
    for kind, (group, slot, _) in KINDS.items():
        if record.get("group") == group and (slot is None or record.get("slot") == slot):
            return kind
    return None


# ------------------------------------------------------------------- the measures


def _finite(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def window_mean(series: Sequence[float | None], window: int = WINDOW) -> float | None:
    """The mean of the reported, finite values among the last ``window`` iterations."""
    values = _finite(series[-window:])
    return sum(values) / len(values) if values else None


def rise(series: Sequence[float | None]) -> float | None:
    """How much a term rose over the last half of the run (:data:`MEASURES`, ``rise``)."""
    late = _finite(series[len(series) - len(series) // 2 :])
    half = len(late) // 2
    if not half:
        return None
    early, last = sum(late[:half]) / half, sum(late[-half:]) / half
    if early <= 0:
        return None
    return last / early - 1


def _digits(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else float(f"{value:.{digits}g}")


def machine_failure(probe: Mapping[str, Any] | None) -> bool:
    """Whether a probe failed for a reason of the machine, not of its config: its process
    ended without a report (a crash or a timeout), or the GPU reported a launch failure or
    ran out of memory. ``run --retry-failed`` probes such cells again."""
    if not probe or probe.get("verdict") != "fails":
        return False
    failure = probe.get("failure") or {}
    text = json.dumps([failure, probe.get("detail") or ""]).lower()
    signs = (
        "unspecified launch failure",
        "cuda error: an illegal",
        "out of memory",
        "alloc_failed",
    )
    return failure.get("error") == "ProcessError" or any(sign in text for sign in signs)


def _failure(probe: Mapping[str, Any] | None, iterations: int = ITERATIONS) -> str | None:
    """Why a probe report is no finished level-2 run; None when it is one."""
    if not probe:
        return "no run"
    verdict = probe.get("verdict")
    if verdict == "fails":
        failure = probe.get("failure") or {}
        where = failure.get("phase") or "run"
        what = " ".join(str(failure.get("message") or probe.get("detail") or "").split())
        machine = (
            " (a failure of the machine, not of the config: `run --retry-failed` probes it again)"
            if machine_failure(probe)
            else ""
        )
        return f"failed at {where}: {failure.get('error', 'error')}: {what[-300:]}{machine}"
    if verdict == "refused":
        return "the config check refused it: " + "; ".join(probe.get("refusal") or [])[:300]
    if verdict == "wired" or probe.get("fallback"):
        return "no real frame was read: " + str(probe.get("fallback") or "")[:300]
    asked = probe.get("iterations_requested") or iterations
    if asked != iterations:
        return f"a run of {asked} iterations, not {iterations}"
    ran = probe.get("iterations") or 0
    if ran < asked and not probe.get("nonfinite"):
        return f"only {ran} of {asked} iterations ran"
    return None


#: The note of a run whose every predicted grasp has the same score.
CONSTANT_SCORES = "every predicted score is the same on every frame (a constant score head)"


def constant_scores(probe: Mapping[str, Any]) -> bool:
    """Whether every predicted frame with a grasp has one score for all of them (its
    smallest and largest score agree): the score head ignores its input."""
    quantiles = [q for q in probe.get("score_quantiles") or [] if q]
    return bool(quantiles) and all(q[0] == q[2] for q in quantiles)


def _step_above(value: float) -> float:
    """The first multiple of :data:`STEP` strictly above ``value``."""
    return round((math.floor(round(value / STEP, 9)) + 1) * STEP, 2)


def calibrate(noise: Mapping[str, Sequence[float | None]] | None) -> dict[str, float]:
    """The tolerance and the rising threshold of a method from the noise table of its
    references (:func:`noise_table`: per term the means under two seeds, their ratio and
    the two rises). Without a table, the least values :data:`TOLERANCE` and
    :data:`RISING_ABOVE`."""
    ratios, rises = [], []
    for name, values in (noise or {}).items():
        if name == "loss":
            continue
        _, _, ratio, first, second = values
        if ratio:
            ratios.append(max(ratio, 1 / ratio))
        if first is not None and second is not None:
            rises.append(abs(second - first))
    tolerance = max(TOLERANCE, _step_above(max(ratios))) if ratios else TOLERANCE
    rising = max(RISING_ABOVE, _step_above(max(rises))) if rises else RISING_ABOVE
    return {"tolerance": tolerance, "rising_above": rising}


def compare(
    cell: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    not_comparable: Mapping[str, str] | None = None,
    *,
    tolerance: float = TOLERANCE,
    rising_above: float = RISING_ABOVE,
    iterations: int = ITERATIONS,
) -> dict[str, Any]:
    """Judge one cell against its reference, from their probe reports.

    ``not_comparable`` maps a loss term to the reason its definition differs between the
    two (:func:`not_comparable`). Returns ``outcome`` (a key of :data:`OUTCOMES`),
    ``reasons`` (what decided it), ``notes`` (what is reported, not judged) and ``terms``:
    per loss term the cell's and the reference's mean over the last window, their rises,
    and either the ``ratio`` or why it is ``not_comparable``.
    """
    result: dict[str, Any] = {"outcome": "", "reasons": [], "notes": [], "terms": {}}
    failed = _failure(cell, iterations)
    if failed or cell is None:
        result.update(outcome="fails", reasons=[failed or "no run"])
        return result
    missing = _failure(reference, iterations)
    if missing or reference is None:
        result.update(outcome="no reference", reasons=[f"the reference: {missing}"])
        return result
    excluded = dict(not_comparable or {})
    mine = cell.get("losses_per_iteration") or {}
    theirs = reference.get("losses_per_iteration") or {}
    diverging: list[str] = []
    worse: list[str] = []
    for name in dict.fromkeys([*theirs, *mine]):
        if name == "loss":
            continue
        own, ref = mine.get(name), theirs.get(name)
        a = window_mean(own) if own is not None else None
        b = window_mean(ref) if ref is not None else None
        up = rise(own) if own is not None else None
        ref_up = rise(ref) if ref is not None else None
        entry: dict[str, Any] = {"cell": a, "reference": b, "rise": up, "reference_rise": ref_up}
        why = excluded.get(name)
        if why is None and own is None:
            why = "the cell does not report it"
        if why is None and ref is None:
            why = "only the cell reports it"
        if why is None and (a is None or b is None or a < 0 or b <= 0):
            why = "a mean that is not positive"
        if why is None and a is not None and b is not None:
            entry["ratio"] = a / b
        else:
            entry["not_comparable"] = why
        result["terms"][name] = entry
        if up is not None and up > rising_above and (ref_up is None or up - ref_up > rising_above):
            against = "" if ref_up is None else f"; the reference {ref_up * 100:+.0f} %"
            # A term that blows up rises by orders of magnitude: say how many times over.
            amount = f"{up * 100:+.0f} %" if up < 9 else f"{up + 1:.3g}-fold"
            diverging.append(f"{name} rises {amount} over the last half{against}")
    for item in cell.get("nonfinite") or []:
        diverging.append(f"non-finite: {item}")
    grasps = list(cell.get("grasps_per_frame") or [])
    if grasps and not any(grasps):
        if any(reference.get("grasps_per_frame") or []):
            diverging.append(f"no grasp on any of the {len(grasps)} frames")
        else:
            result["notes"].append("no grasp on any frame, as for the reference")
    from grasppanda.combine.probe import dead_parts

    for part in dead_parts(cell.get("no_gradient") or []):
        worse.append(f"no gradient reached {part}")
    comparable = {name: e for name, e in result["terms"].items() if "ratio" in e}
    lower = []
    for name, entry in comparable.items():
        if entry["ratio"] > tolerance:
            worse.append(f"{name} {entry['ratio']:.3f}x the reference (tolerance {tolerance:g})")
        elif entry["ratio"] < 1 / tolerance:
            lower.append(f"{name} {entry['ratio']:.2f}x")
    if lower:
        result["notes"].append("lower than the reference: " + ", ".join(lower))
    if constant_scores(cell):
        result["notes"].append(CONSTANT_SCORES)
    if cell.get("verdict") == "runs-unstable" and not cell.get("nonfinite"):
        detail = str(cell.get("detail") or "")
        result["notes"].append("unstable: " + detail.split(", but ", 1)[-1])
    if diverging:
        outcome = "diverging"
    elif worse:
        outcome = "worse"
    elif not comparable:
        outcome = "not comparable"
    else:
        outcome = "on a par"
    result["outcome"] = outcome
    result["reasons"] = diverging + worse
    return result


# ------------------------------------------------------------ comparable terms


def _loss_code(cls: type) -> tuple[str, ...]:
    """Where a stage computes its loss terms: the qualified names of its ``loss`` method
    and of every method about targets. A subclass that inherits them computes the same."""
    names = sorted(name for name in dir(cls) if name == "loss" or "target" in name)
    return tuple(
        getattr(getattr(cls, name), "__qualname__", name)
        for name in names
        if callable(getattr(cls, name, None))
    )


def _canonical(module: Any) -> Any:
    """A built loss as comparable data: its type, its plain settings and its children."""
    import torch

    if not isinstance(module, torch.nn.Module):
        return repr(module)
    plain = (int, float, str, bool, type(None))
    settings = sorted(
        (key, value)
        for key, value in vars(module).items()
        if not key.startswith("_") and key != "training" and isinstance(value, plain)
    )
    children = sorted((name, _canonical(child)) for name, child in module.named_children())
    return (type(module).__name__, tuple(settings), tuple(children))


_BUILT: dict[str, Any] = {}


def _stage_default(slot: str, cls: type, options: Mapping[str, Any], option: str) -> Any:
    """The loss a stage builds for a loss option left empty (its own default), read from
    the stage built alone with placeholder sizes; None when it cannot be built here (a
    missing optional library, a part only the detector supplies)."""
    from grasppanda.registry import INJECTED, component_parameters

    key = json.dumps([slot, cls.__name__, options], sort_keys=True, default=str)
    if key not in _BUILT:
        params, _ = component_parameters(cls)
        given = dict(options)
        for name, param in params.items():
            if param.default is INJECTED and name not in given:
                given[name] = 64
        try:
            _BUILT[key] = cls(**given)
        except Exception:
            _BUILT[key] = None
    built = _BUILT[key]
    if built is None:
        return None
    for name in (f"{option}_fn", option):  # a view stage keeps `loss` as `loss_fn`
        loss = getattr(built, name, None)
        if loss is not None and not callable(getattr(loss, "__func__", None)):
            return _canonical(loss)
    return None


def effective_loss(slot: str, cls: type, options: Mapping[str, Any], option: str) -> Any:
    """The loss a term is computed with, as comparable data: the configured node built, or
    the stage's own default when the option is empty (``("default of", type)`` when that
    default cannot be read)."""
    from grasppanda.registry import LOSSES

    node = options.get(option)
    if isinstance(node, Mapping):
        try:
            return _canonical(LOSSES.build(dict(node)))
        except Exception:
            return json.dumps(node, sort_keys=True, default=str)
    if node is None:
        found = _stage_default(slot, cls, options, option)
        return found if found is not None else ("default of", cls.__name__)
    return repr(node)


def term_owners(model: Mapping[str, Any]) -> dict[str, tuple[str, str, tuple[str, ...], Any]]:
    """Loss term -> (stage, its type, where it computes its loss (:func:`_loss_code`), the
    loss it is computed with (:func:`effective_loss`)) of a seed-based model node; empty
    for another detector. A stage names the kind of every loss option it has
    (``loss_terms``); option ``x_loss`` reports ``loss/x`` and option ``loss`` reports
    ``loss/<stage>``."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    owners: dict[str, tuple[str, str, tuple[str, ...], Any]] = {}
    for slot in STAGES_WITH_TERMS:
        node = model.get(slot)
        if not isinstance(node, Mapping) or "type" not in node or slot not in STAGE_REGISTRIES:
            continue
        try:
            cls = STAGE_REGISTRIES[slot].get(node["type"])
        except Exception:
            continue
        options = {key: value for key, value in node.items() if key != "type"}
        reader = getattr(cls, "loss_terms_for", None)
        terms = reader(options) if reader else getattr(cls, "loss_terms", {})
        code = _loss_code(cls)
        for option in terms or {}:
            name = slot if option == "loss" else option.removesuffix("_loss")
            loss = effective_loss(slot, cls, options, option)
            owners[f"loss/{name}"] = (slot, str(node["type"]), code, loss)
    return owners


def not_comparable(
    kind: str, cell_model: Mapping[str, Any], reference_model: Mapping[str, Any]
) -> dict[str, str]:
    """Loss term -> why the cell changes its definition (:data:`NOT_COMPARABLE`).

    Terms only one run reports are left to :func:`compare`, which sees the losses."""
    mine, theirs = term_owners(cell_model), term_owners(reference_model)
    stages = {
        slot: (_stage_code(cell_model, slot), _stage_code(reference_model, slot))
        for slot in STAGES_WITH_TERMS
    }
    found: dict[str, str] = {}
    for name in dict.fromkeys([*theirs, *mine]):
        own, ref = mine.get(name), theirs.get(name)
        slot = (ref or own)[0]  # type: ignore[index]
        (own_type, own_code), (ref_type, ref_code) = stages[slot]
        if kind == "head" and slot == "head":
            found[name] = "the head bundle defines its targets and its loss"
        elif kind == "assigner" and slot in ("view", "head"):
            found[name] = f"the assigner changes the labels the {slot} sees"
        elif own_code != ref_code:
            # A stage that does not declare the term still reports it (a regression view
            # stage reports `loss/view`): the stage of the slot decides, declared or not.
            found[name] = f"another {slot} stage computes it ({own_type} for {ref_type})"
        elif own is not None and ref is not None and own[3] != ref[3]:
            loss = own[3][0] if isinstance(own[3], tuple) else own[3]
            found[name] = f"its loss is replaced ({loss})"
    return found


def _stage_code(model: Mapping[str, Any], slot: str) -> tuple[str | None, tuple[str, ...]]:
    """The type of the stage of ``slot`` and where it computes its loss (:func:`_loss_code`);
    ``(None, ())`` without one."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    node = model.get(slot)
    if not isinstance(node, Mapping) or "type" not in node or slot not in STAGE_REGISTRIES:
        return None, ()
    try:
        return str(node["type"]), _loss_code(STAGE_REGISTRIES[slot].get(node["type"]))
    except Exception:
        return str(node["type"]), (str(node["type"]),)


def _model(config: str) -> dict[str, Any]:
    from grasppanda.config import Config

    return dict(Config.load(config).resolved()["model"])


# ------------------------------------------------------------------- selection


def level1_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The level-1 records cells are selected from: a probe_grid run directory
    (``--level1-run``) or a measured file (``--level1``, the committed one by default)."""
    if getattr(args, "level1_run", None):
        return list(_runner().load(args.level1_run).values())
    path = Path(getattr(args, "level1", None) or LEVEL1)
    return list(json.loads(path.read_text(encoding="utf-8"))["cells"])


def select(
    records: Iterable[Mapping[str, Any]], kind: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """The level-1 records of ``kind`` that get a level-2 run, and the ids of those left
    out because they compose to the reference's own config (the same hash)."""
    records = list(records)
    by_id = {record["id"]: record for record in records}
    chosen, own = [], []
    for record in records:
        if kind_of(record) != kind or record.get("verdict") not in WORKS:
            continue
        reference = by_id.get(reference_id(record["method"])) or {}
        if record.get("config_hash") and record.get("config_hash") == reference.get("config_hash"):
            own.append(record["id"])
            continue
        chosen.append(dict(record))
    return chosen, own


def _wanted(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    patterns = getattr(args, "only", None) or []
    methods = getattr(args, "methods", None) or []
    return [
        record
        for record in records
        if (not patterns or any(fnmatch.fnmatch(record["id"], p) for p in patterns))
        and (not methods or record["method"] in methods)
    ]


# ------------------------------------------------------------------------- run


def _probe(
    name: str,
    ids: list[str],
    args: argparse.Namespace,
    seed: int | None = None,
    *,
    redo: bool = False,
) -> int:
    """Run (or resume) the probes of ``ids`` in the probe_grid run directory ``name``;
    ``redo`` composes them again and probes again those whose earlier probe says nothing
    about the config (probe_grid's ``--redo``). The cells of a method of
    :data:`METHOD_SCHEDULES` run apart, with their schedule and always with ``--redo``: a
    probe of that schedule made before is reused, one of another schedule is not."""
    if not ids:
        return 0
    apart = [cell_id for cell_id in ids if schedule_of(cell_id) != SETTINGS["schedule"]]
    if apart:
        rest = [cell_id for cell_id in ids if cell_id not in apart]
        status = _probe(name, rest, args, seed, redo=redo) if rest else 0
        for schedule in sorted({schedule_of(cell_id) for cell_id in apart}):
            chosen = [cell_id for cell_id in apart if schedule_of(cell_id) == schedule]
            status |= _probe_with(name, chosen, args, seed, redo=True, schedule=schedule)
        return status
    return _probe_with(name, ids, args, seed, redo=redo, schedule=SETTINGS["schedule"])


def _probe_with(
    name: str,
    ids: list[str],
    args: argparse.Namespace,
    seed: int | None,
    *,
    redo: bool,
    schedule: str,
) -> int:
    argv = ["run", "--name", name, "--only", *(glob.escape(cell_id) for cell_id in ids)]
    if redo:
        argv.append("--redo")
    argv += ["--iters", str(ITERATIONS), "--frames", str(SETTINGS["frames"])]
    argv += ["--predict-frames", str(SETTINGS["predict_frames"])]
    argv += ["--schedule", schedule, "--jobs", str(args.jobs)]
    argv += ["--timeout", str(args.timeout)]
    if args.scale_statistic_sha:
        argv += ["--scale-statistic-sha", args.scale_statistic_sha]
    if seed is not None and seed != SETTINGS["seed"]:
        argv += ["--seed", str(seed)]
    if getattr(args, "allow_dirty", False):
        argv.append("--allow-dirty")
    if getattr(args, "commit", None):
        argv += ["--commit", args.commit]
    print(f"== {name}: {len(ids)} cell(s)")
    return int(_runner().main(argv))


def references(args: argparse.Namespace) -> int:
    from grasppanda.combine.grid import FAMILIES, METHODS

    methods = args.methods or [*METHODS, *FAMILIES]
    status = 0
    for seed in args.seeds:
        ids = [reference_id(method) for method in methods]
        status |= _probe(reference_run(seed), ids, args, seed)
    return status


def run(args: argparse.Namespace) -> int:
    records = level1_records(args)
    plan: dict[str, list[dict[str, Any]]] = {}
    for kind in args.kind:
        chosen, _ = select(records, kind)
        plan[kind] = _wanted(chosen, args)
    methods = sorted({record["method"] for members in plan.values() for record in members})
    status = _probe(REFERENCE_RUN, [reference_id(method) for method in methods], args)
    for kind, members in plan.items():
        ids = [record["id"] for record in members]
        if args.retry_failed:
            done = _runner().load(run_name(kind))
            again = [i for i in ids if i in done and machine_failure(done[i].get("probe"))]
            status |= _probe(run_name(kind), again, args, redo=True)
        status |= _probe(run_name(kind), ids, args)
    return status


# ---------------------------------------------------------------------- status


def _seconds(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Median wall time of a finished level-2 probe, per method."""
    found: dict[str, list[float]] = {}
    for record in records:
        probe = record.get("probe") or {}
        if (
            probe.get("iterations") == ITERATIONS
            and probe.get("seconds")
            and not record.get("probed_as")
        ):
            found.setdefault(record["method"], []).append(float(probe["seconds"]))
    return {method: statistics.median(values) for method, values in found.items()}


def status(args: argparse.Namespace) -> int:
    runner = _runner()
    records = level1_records(args)
    every = [*runner.load(REFERENCE_RUN).values()]
    done_by_kind = {kind: runner.load(run_name(kind)) for kind in KINDS}
    for done in done_by_kind.values():
        every += done.values()
    seconds = _seconds(every)
    fallback = statistics.median(seconds.values()) if seconds else 120.0
    total_left = 0.0
    print(f"references (seed {SETTINGS['seed']}): {len(runner.load(REFERENCE_RUN))} recorded")
    print(f"{'kind':<14}{'selected':>9}{'recorded':>9}{'pending':>8}   pending by method")
    for kind in KINDS:
        chosen, _ = select(records, kind)
        done = done_by_kind[kind]
        pending = [record for record in chosen if record["id"] not in done]
        per_method = Counter(record["method"] for record in pending)
        left = sum(seconds.get(method, fallback) * n for method, n in per_method.items())
        total_left += left
        listed = ", ".join(f"{method} {n}" for method, n in sorted(per_method.items()))
        print(f"{kind:<14}{len(chosen):>9}{len(done):>9}{len(pending):>8}   {listed}")
    times = ", ".join(f"{method} {value:.0f} s" for method, value in sorted(seconds.items()))
    print(f"median seconds per probe: {times or 'none measured yet'}")
    print(f"pending: about {total_left / 3600:.1f} h at one probe at a time")
    return 0


# ---------------------------------------------------------------------- export


def dump(payload: Mapping[str, Any]) -> str:
    """A summary as text: the ``about`` block indented, then every other list one entry
    per line (as the measured file of level 1)."""
    parts = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines = ",\n".join(
                "  " + json.dumps(entry, ensure_ascii=False, separators=(", ", ": "))
                for entry in value
            )
            parts.append(f' "{key}": [\n{lines}\n ]' if value else f' "{key}": []')
        else:
            text = json.dumps(value, indent=1, ensure_ascii=False).replace("\n", "\n ")
            parts.append(f' "{key}": {text}')
    return "{\n" + ",\n".join(parts) + "\n}\n"


def side_file(target: Path) -> Path:
    """Where the losses of the summary ``target`` are written."""
    return target.with_suffix("") / SERIES_NAME


def blocks(series: Sequence[float | None], size: int = BLOCK) -> list[float | None]:
    """The mean of every ``size`` iterations (reported, finite values); None for none."""
    out: list[float | None] = []
    for start in range(0, len(series), size):
        values = _finite(series[start : start + size])
        out.append(_digits(sum(values) / len(values)) if values else None)
    return out


def _label(record: Mapping[str, Any]) -> str:
    return ", ".join(record.get("sets") or []) or record["id"]


def _reference_line(record: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    probe = record.get("probe") or {}
    losses = probe.get("losses_per_iteration") or {}
    line: dict[str, Any] = {"id": cell_id, "method": record["method"], "verdict": record["verdict"]}
    failed = _failure(probe)
    if failed:
        line["failure"] = failed
        return line
    line["terms"] = {name: _digits(window_mean(values)) for name, values in losses.items()}
    line["rise"] = {name: _digits(rise(values), 3) for name, values in losses.items()}
    line.update(_shared(record))
    return line


def _shared(record: Mapping[str, Any]) -> dict[str, Any]:
    """What a cell and a reference line both carry from their probe."""
    probe = record.get("probe") or {}
    line: dict[str, Any] = {}
    for key in ("grasps_per_frame", "no_gradient", "unchanged"):
        if probe.get(key) is not None:
            line[key] = probe[key]
    quantiles = probe.get("score_quantiles") or []
    if quantiles:
        line["score_median"] = [None if q is None else q[1] for q in quantiles]
    if constant_scores(probe):
        line["constant_scores"] = True
    line["checked"] = [probe.get("units_checked", 0), probe.get("parts_checked", 0)]
    line["schedule"] = probe.get("schedule")
    if probe.get("rate") is not None:
        line["rate"] = probe["rate"]
    line["probe"] = _runner()._as_typed(record.get("probe_command", ""))
    if record.get("compose_command"):
        file = record["id"].replace("/", "__") + ".yaml"
        line["compose"] = f"{record['compose_command']} -o {file}"
    line["config_hash"] = (record.get("config_hash") or "")[:16] or None
    measured = record.get("measured") or {}
    if measured:
        line["run"] = _runner()._run_id(measured)
    line["seconds"] = probe.get("seconds")
    return line


def _cell_line(
    record: Mapping[str, Any], kind: str, level1: Mapping[str, Any], judged: Mapping[str, Any]
) -> dict[str, Any]:
    line: dict[str, Any] = {
        "id": record["id"],
        "kind": kind,
        "method": record["method"],
        "option": _label(record),
        "level1": level1.get("verdict"),
        "outcome": judged["outcome"],
        "reasons": judged["reasons"],
    }
    if judged["notes"]:
        line["notes"] = judged["notes"]
    terms: dict[str, Any] = {}
    for name, entry in judged["terms"].items():
        if "ratio" in entry:
            terms[name] = [
                _digits(entry["cell"]),
                _digits(entry["reference"]),
                round(entry["ratio"], 3),
            ]
        else:
            terms[name] = entry["not_comparable"]
    if terms:
        line["terms"] = terms
        line["rise"] = {
            name: round(entry["rise"], 3)
            for name, entry in judged["terms"].items()
            if entry["rise"] is not None
        }
    if judged["outcome"] != "fails":
        line.update(_shared(record))
    return line


def _cache_dir() -> Path:
    from grasppanda.utils.paths import cache_root

    return cache_root() / "verification" / "L2"


def _write_cache(kind: str, cell_id: str, payload: Mapping[str, Any]) -> None:
    target = _cache_dir() / kind / (cell_id.replace("/", "__") + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def noise_table(
    first: Mapping[str, Any] | None, second: Mapping[str, Any] | None
) -> dict[str, list[float | None]]:
    """Per loss term: the mean over the last window under the two seeds and their ratio."""
    out: dict[str, list[float | None]] = {}
    if _failure(first) or _failure(second):
        return out
    assert first is not None and second is not None
    a, b = first.get("losses_per_iteration") or {}, second.get("losses_per_iteration") or {}
    for name in a:
        if name not in b:
            continue
        x, y = window_mean(a[name]), window_mean(b[name])
        ratio = y / x if x and y is not None and x > 0 else None
        up_a, up_b = rise(a[name]), rise(b[name])
        out[name] = [
            _digits(x),
            _digits(y),
            None if ratio is None else round(ratio, 3),
            None if up_a is None else round(up_a, 3),
            None if up_b is None else round(up_b, 3),
        ]
    return out


def _level1_name(args: argparse.Namespace) -> str:
    """The level-1 records the cells were selected from, as the summary names them."""
    if getattr(args, "level1_run", None):
        return f"probe_grid run {args.level1_run}"
    path = Path(getattr(args, "level1", None) or LEVEL1).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name


def export(args: argparse.Namespace) -> int:
    from grasppanda.combine.grid import FAMILIES, METHODS

    runner = _runner()
    level1 = level1_records(args)
    level1_by_id = {record["id"]: record for record in level1}
    references0 = runner.load(REFERENCE_RUN)
    if not references0:
        raise SystemExit(f"{runner.directory(REFERENCE_RUN)} holds no record: run the references")
    runs: dict[str, Any] = {}
    reference_lines, series = [], []
    noise: dict[str, Any] = {}
    models: dict[str, dict[str, Any]] = {}
    for method in [*METHODS, *FAMILIES]:
        ident = reference_id(method)
        record = references0.get(ident)
        if record is None:
            continue
        reference_lines.append(_reference_line(record, ident))
        if record.get("measured"):
            runs[runner._run_id(record["measured"])] = record["measured"]
        losses = (record.get("probe") or {}).get("losses_per_iteration") or {}
        series.append({"id": ident, "losses": {n: blocks(v) for n, v in losses.items()}})
        _write_cache("reference", ident, {"reference": record})
        for seed in args.seeds:
            if seed == SETTINGS["seed"]:
                continue
            other = runner.load(reference_run(seed)).get(ident)
            if other is None:
                continue
            noise.setdefault(method, {})[f"seed {seed}"] = noise_table(
                record.get("probe"), other.get("probe")
            )
            line = _reference_line(other, f"{ident}@seed{seed}")
            reference_lines.append(line)
            if other.get("measured"):
                runs[runner._run_id(other["measured"])] = other["measured"]
            other_losses = (other.get("probe") or {}).get("losses_per_iteration") or {}
            series.append(
                {
                    "id": f"{ident}@seed{seed}",
                    "losses": {n: blocks(v) for n, v in other_losses.items()},
                }
            )
            _write_cache("reference", f"{ident}@seed{seed}", {"reference": other})
        if method in METHODS:
            models[method] = _model(METHODS[method])
    calibrated = {
        method: calibrate(
            {
                f"{seed} {name}": values
                for seed, terms in noise.get(method, {}).items()
                for name, values in terms.items()
                if name != "loss"
            }
        )
        for method in [*METHODS, *FAMILIES]
    }
    cells_out: list[dict[str, Any]] = []
    selected: dict[str, dict[str, int]] = {}
    for kind in KINDS:
        chosen, own = select(level1, kind)
        done = runner.load(run_name(kind))
        chosen_ids = {record["id"] for record in chosen}
        measured_here = [done[cell_id] for cell_id in sorted(done) if cell_id in chosen_ids]
        selected[kind] = {
            "selected": len(chosen),
            "measured": len(measured_here),
            "the reference itself": len(own),
        }
        order = [record["id"] for record in chosen]
        for record in sorted(measured_here, key=lambda r: order.index(r["id"])):
            method = record["method"]
            reference = references0.get(reference_id(method))
            excluded: dict[str, str] = {}
            if method in models and record.get("config"):
                try:
                    excluded = not_comparable(kind, _model(record["config"]), models[method])
                except Exception as error:  # a config that no longer loads
                    excluded = {}
                    print(f"  {record['id']}: config not read ({error})")
            probe = record.get("probe")
            if probe is None and record.get("verdict") in ("refused", "fails"):
                # Composed again and refused, or compose itself failed: no probe ran.
                probe = {
                    "verdict": record["verdict"],
                    "refusal": record.get("refusal") or [],
                    "failure": record.get("failure") or {},
                }
            judged = compare(
                probe,
                (reference or {}).get("probe"),
                excluded,
                tolerance=calibrated[method]["tolerance"],
                rising_above=calibrated[method]["rising_above"],
            )
            line = _cell_line(record, kind, level1_by_id.get(record["id"], {}), judged)
            cells_out.append(line)
            if record.get("measured"):
                runs[runner._run_id(record["measured"])] = record["measured"]
            losses = (record.get("probe") or {}).get("losses_per_iteration") or {}
            if losses and not record.get("probed_as"):
                series.append(
                    {"id": record["id"], "losses": {n: blocks(v) for n, v in losses.items()}}
                )
            _write_cache(
                kind,
                record["id"],
                {
                    "comparison": judged,
                    "not_comparable": excluded,
                    "cell": record,
                    "reference": reference_id(method),
                },
            )
    summary = {
        "about": {
            "what": (
                "Level 2 of the measured grid (tools/learn_check.py): every cell trained for "
                f"{ITERATIONS} iterations under the settings below and compared with the method "
                "as shipped under the same settings. One run per cell; a statement about "
                "training dynamics on 8 frames of one scene, not an accuracy measurement. "
                "See docs/advanced_guides/learns_like_reference.md."
            ),
            "settings": SETTINGS,
            "tolerance": TOLERANCE,
            "rising_above": RISING_ABOVE,
            "step": STEP,
            "calibrated": calibrated,
            "outcomes": OUTCOMES,
            "measures": MEASURES,
            "not_comparable": NOT_COMPARABLE,
            "calibration": CALIBRATION,
            "fields": FIELDS,
            "noise": noise,
            "kinds": selected,
            "left_out": LEFT_OUT,
            "level1": _level1_name(args),
            "runs": {key: runs[key] for key in sorted(runs)},
        },
        "references": reference_lines,
        "cells": cells_out,
    }
    target = Path(args.to)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump(summary), encoding="utf-8")
    side = side_file(target)
    side.parent.mkdir(parents=True, exist_ok=True)
    side_payload = {
        "about": {
            "what": (
                f"The losses of every level-2 run of {target.name}: per run, per loss term "
                f"and for the total `loss`, the mean of every {BLOCK} iterations of the "
                f"{ITERATIONS} (reported, finite values; null for none), four significant "
                "digits. Every iteration is in the run's file under "
                "$GRASPPANDA_CACHE/verification/L2/. No number here is an accuracy."
            ),
            "block": BLOCK,
            "records": target.name,
        },
        "cells": series,
    }
    side.write_text(dump(side_payload), encoding="utf-8")
    counts = Counter(line["outcome"] for line in cells_out)
    print(
        f"wrote {target} ({target.stat().st_size / 1024:.0f} KiB): {len(reference_lines)} "
        f"reference runs, {len(cells_out)} cells {dict(counts)}; {side} "
        f"({side.stat().st_size / 1024:.0f} KiB); per-cell files in {_cache_dir()}"
    )
    return 0


#: What a cell of the summary holds, field by field.
FIELDS: dict[str, str] = {
    "id, kind, method": "the cell of the grid (tools/probe_grid.py), its kind (KINDS) and method",
    "option": "the slot=choice pairs of its compose request",
    "level1": "its level-1 verdict (20 iterations)",
    "outcome, reasons": "the outcome (about.outcomes) and what decided it",
    "notes": "what is reported, not judged: lower terms, a level-2 run that is runs-unstable "
    "for another reason than a non-finite value, no grasp as for the reference",
    "terms": "per loss term: [cell's mean, reference's mean, ratio] over the last window, or "
    "the reason it is not comparable (about.not_comparable)",
    "rise": "per loss term: its rise over the last half of the run (about.measures)",
    "grasps_per_frame, score_median": "grasps predicted on each of the predicted frames, and "
    "the median score of each (null for a frame without a grasp)",
    "constant_scores": "true when every predicted grasp of a frame has the same score, on "
    "every frame with a grasp: the score head ignores its input",
    "no_gradient, unchanged, checked": "units without a gradient, parts whose weights did not "
    "change, and [units, parts] checked (grasppanda.combine.probe.JUDGE_FIELDS)",
    "schedule, rate": "how the probe set the learning rate: `constant` at `rate` (the recipe's "
    "peak, scaled to the batch), or `recipe` for a cell whose request sets the schedule",
    "probe, compose, config_hash, run": "the commands, the config hash (with the seed for a "
    "seeded run) and the run that measured it (about.runs)",
    "seconds": "wall time of the probe, measured on a shared machine",
}


# ------------------------------------------------------------------------ page


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def cell(text: str) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(item) for item in row) + " |" for row in rows]
    return lines


def _grasps(line: Mapping[str, Any]) -> str:
    grasps = line.get("grasps_per_frame") or []
    return ", ".join(str(count) for count in grasps) or "—"


def _terms_text(line: Mapping[str, Any]) -> str:
    """The comparable terms as `name ratio`, the worst first."""
    terms = line.get("terms") or {}
    pairs = [(name, value[2]) for name, value in terms.items() if isinstance(value, list)]
    pairs.sort(key=lambda pair: -pair[1])
    return ", ".join(f"{name.removeprefix('loss/')} {ratio:.2f}" for name, ratio in pairs) or "—"


def _excluded_text(line: Mapping[str, Any]) -> str:
    terms = line.get("terms") or {}
    names = [name.removeprefix("loss/") for name, value in terms.items() if isinstance(value, str)]
    return ", ".join(names) or "—"


def _why(line: Mapping[str, Any]) -> str:
    items = list(line.get("reasons") or [])
    items += list(line.get("notes") or [])
    return "; ".join(items) or "—"


def _chips() -> Any:
    """``tools/docs_chips.py``: the chips and the words every page shares."""
    name = "docs_chips"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The kinds in the order of the page: the slots as data flows, then the recipe.
PAGE_KINDS = (
    "image-encoder",
    "encoder",
    "proposal",
    "view",
    "assigner",
    "grouper",
    "head",
    "loss",
    "recipe",
)
#: Outcome → (the mark of a cell, the tone of its chip). The worst outcomes come last.
MARKS: dict[str, tuple[str, str]] = {
    "on a par": ("✓", "ok"),
    "not comparable": ("n/c", "muted"),
    "no reference": ("no reference", "muted"),
    "worse": ("worse", "warn"),
    "diverging": ("diverging", "no"),
    "fails": ("did not train", "no"),
}
#: The outcomes the answer counts, each with its tone and its words.
COUNTED = (("on a par", "ok", "on a par"), ("worse", "warn", "worse"))


def _anchor(title: str) -> str:
    return title.lower().replace(" ", "-")


def _methods(cells_: Iterable[Mapping[str, Any]]) -> list[str]:
    """The methods of ``cells_`` in page order: the seed-based skeleton by year, then HGGD."""
    from grasppanda.combine.reach import method_order

    found = {line["method"] for line in cells_}
    return [name for name in method_order() if name in found]


def _ordered(names: Iterable[str]) -> list[str]:
    """``names`` in page order; a name the page does not know comes last."""
    from grasppanda.combine.reach import method_order

    order = method_order()
    return sorted(names, key=lambda name: order.index(name) if name in order else len(order))


def _row_key(line: Mapping[str, Any]) -> str:
    """The option of a cell as a row of its table: the choice after ``=``, the slot too for
    the recipe, the loss presets (one cell per term they were put in) for a loss."""
    if line["kind"] == "loss":
        return " + ".join(line["id"].split("/", 2)[2].split("@")[0].split("+"))
    if line["kind"] == "recipe":
        return line["option"]
    return line["option"].split("=", 1)[-1]


def _short_reason(line: Mapping[str, Any]) -> str:
    """The first reason of an outcome, in a few words: ``score 1.42x``."""
    reasons = list(line.get("reasons") or [])
    if not reasons:
        return line["outcome"]
    first = reasons[0].replace("loss/", "")
    match = re.match(r"(\S+) ([\d.]+)x the reference", first)
    if match:
        return f"{match.group(1)} {float(match.group(2)):.2f}x"
    return first.split(";")[0]


def _cell_chip(line: Mapping[str, Any]) -> str:
    """One cell as a chip; its ratios and reasons are the tooltip."""
    mark, tone = MARKS.get(line["outcome"], (line["outcome"], ""))
    term = ""
    if line["kind"] == "loss":
        term = ", ".join(item.split("=")[0] for item in line["option"].split(", ")) + ": "
    tip = f"{term}{line['outcome']}. Ratios to the reference: {_terms_text(line)}. {_why(line)}"
    return _chips().chip(mark, tone, tip)


class Compared(NamedTuple):
    """One option as the page shows it: its level-2 result, by the rules of
    :func:`grasppanda.combine.reach.level2`, and the cells that result was read from.

    Attributes:
        slot: The slot of the option (``loss`` for a loss preset).
        name: The option.
        level2: Its :class:`grasppanda.combine.reach.Level2`.
        lines: Method → the summary lines its outcome there was read from (one per term a
            loss preset was put in; one otherwise).
    """

    slot: str
    name: str
    level2: Any
    lines: dict[str, list[Mapping[str, Any]]]


#: Why a cell of the summary is not counted, beyond the reasons of
#: :mod:`grasppanda.combine.reach` (``LEVEL2_NONE``, ``LEVEL2_EXCLUDED``).
JOINT = "a loss preset is read from its cells of one term"
#: Why a cell whose preset writes out the method's own component
#: (``grasppanda.combine.grid.SHIPPED_EQUIVALENTS``) is not counted.
OWN = "it is the method as shipped, not a swap"
NOT_READ = "level 2 does not compare it (no option of the grid)"
#: The groups of :func:`compared_options` that are not counted, in the order of the page,
#: each with its short title.
NOT_COUNTED_TITLES = {
    "schedule-slot": "Schedules",
    "schedule": "On a schedule of its own",
    "runs": "Did not run at level 1",
    "joint": "Several loss terms at once",
    "own": "The method's own component",
    "other": "Not read",
}
#: The groups of the grid without a level-2 check (:data:`LEFT_OUT`) → their titles.
LEFT_OUT_TITLES = {
    "post": "Post-processing",
    "stage-set, cross": "Sets of stages, and requests that replace several slots",
}


def compared_options(
    summary: Mapping[str, Any], path: Path | None = None
) -> tuple[dict[str, list[Compared]], dict[str, list[tuple[Mapping[str, Any], str]]]]:
    """Kind → its options, each with its level-2 result as
    :func:`grasppanda.combine.reach.level2` gives it (the summary at ``path``); and the cells
    of the summary that result does not count, grouped (:data:`NOT_COUNTED_TITLES`), each with
    why. The page counts and shows only what ``level2`` counts, so it cannot disagree with the
    slot pages, the method pages and the explorer."""
    from grasppanda.combine import reach as reaching
    from grasppanda.combine.grid import SHIPPED_EQUIVALENTS
    from grasppanda.combine.options import _class_index, measured

    records = measured()
    source = path or SUMMARY
    lines = {line["id"]: line for line in summary["cells"]}
    shown: dict[str, list[Compared]] = {}
    left: dict[str, list[tuple[Mapping[str, Any], str]]] = {}
    seen: set[str] = set()
    for (slot, name), by_method in _class_index().items():
        if not any(cell in lines for ids in by_method.values() for cell in ids):
            continue
        found = reaching.level2(slot, name, records, source)
        used: dict[str, list[Mapping[str, Any]]] = {}
        for method, ids in by_method.items():
            single = [cell for cell in ids if not reaching._joint(records, cell)] or list(ids)
            mine = [lines[cell] for cell in single if cell in lines]
            if not mine:
                continue
            detail = ""
            if found.none:
                group = "schedule-slot"
            elif method in found.excluded:
                group = next(
                    key
                    for key, words in reaching.LEVEL2_EXCLUDED.items()
                    if words == found.excluded[method]
                )
                if group == "runs":
                    level1 = reaching.reach(slot, name, "runs", records)
                    detail = level1.missed.get(method) or level1.undecided.get(method) or ""
            elif method in found.outcomes:
                used[method] = mine
                seen.update(line["id"] for line in mine)
                continue
            elif method in reaching.reach(slot, name, "runs", records).own:
                group = "own"
                why = [SHIPPED_EQUIVALENTS.get(line["id"], "") for line in mine]
                detail = "; ".join(dict.fromkeys(text for text in why if text))
            else:
                continue
            for line in mine:
                if line["id"] not in seen:
                    left.setdefault(group, []).append((line, detail))
                    seen.add(line["id"])
        if used:
            kind = next(iter(used.values()))[0]["kind"]
            ordered = {method: used[method] for method in found.outcomes if method in used}
            shown.setdefault(kind, []).append(Compared(slot, name, found, ordered))
    for cell_id, line in lines.items():
        if cell_id not in seen:
            group = "joint" if reaching._joint(records, cell_id) else "other"
            left.setdefault(group, []).append((line, ""))
    return shown, {group: left[group] for group in NOT_COUNTED_TITLES if group in left}


def _option_chip(outcome: str, lines: list[Mapping[str, Any]]) -> str:
    """The outcome of an option under one method as a chip: the worst of its cells (a loss
    preset in several terms has a cell per term); the ratios of every cell as the tooltip."""
    if len(lines) == 1:
        return _cell_chip(lines[0])
    mark, tone = MARKS.get(outcome, (outcome, ""))
    tips = []
    for line in lines:
        term = ", ".join(item.split("=")[0] for item in line["option"].split(", "))
        tips.append(f"{term}: {line['outcome']}, ratios {_terms_text(line)}")
    return _chips().chip(mark, tone, f"{outcome} (the worst of its terms). " + "; ".join(tips))


def _answer(
    shown: Mapping[str, list[Compared]], cells_: list[Mapping[str, Any]], methods: list[str]
) -> list[str]:
    """Kind × method: how many options are on a par, worse, diverging (or did not train),
    not comparable, by :func:`grasppanda.combine.reach.level2`."""
    chips = _chips()
    rows = []
    for kind in PAGE_KINDS:
        title = KINDS[kind][2]
        row = [f"[{title}](#{_anchor(title)})"]
        for method in methods:
            counts = Counter(
                option.level2.outcomes[method]
                for option in shown.get(kind, [])
                if method in option.level2.outcomes
            )
            if not counts:
                ran = any(line["kind"] == kind and line["method"] == method for line in cells_)
                row.append(chips.chip("not counted", "muted", NOT_COUNTED_TIP) if ran else "—")
                continue
            parts = [
                chips.chip(str(counts[name]), tone, words)
                for name, tone, words in COUNTED
                if counts[name]
            ]
            bad = counts["diverging"] + counts["fails"]
            if bad:
                parts.append(chips.chip(str(bad), "no", "diverging or did not train"))
            if counts["not comparable"]:
                parts.append(chips.chip(str(counts["not comparable"]), "muted", "not comparable"))
            row.append(" ".join(parts))
        rows.append(row)
    return _table(["Kind", *methods], rows)


#: The tooltip of a kind under a method whose every cell is listed under *Not on this page*.
NOT_COUNTED_TIP = "every cell of this kind is listed under Not on this page"


def _kind_section(
    kind: str,
    title: str,
    about: Mapping[str, Any],
    cells_: list[Mapping[str, Any]],
    options: list[Compared],
    left: int,
) -> list[str]:
    """One kind: a sentence that counts, and every option it compares in a fold."""
    members = [line for line in cells_ if line["kind"] == kind]
    numbers = (about.get("kinds") or {}).get(kind, {})
    selected = numbers.get("selected", 0)
    lines = [f"### {title} {{#{_anchor(title)}}}", ""]
    if not members:
        return [*lines, f"Not run yet ({selected} cells selected).", ""]
    methods = _ordered({method for option in options for method in option.lines})
    rows, reasons, par = [], [], 0
    for option in options:
        compared = option.level2.compared
        par += bool(compared) and len(option.level2.on_a_par) == len(compared)
        key = _row_key(next(iter(option.lines.values()))[0])
        why = [
            f"{method}: {line['outcome']} ({_short_reason(line)})"
            for method in methods
            for line in option.lines.get(method, [])
            if line["outcome"] not in ("on a par", "not comparable")
        ]
        cells = [
            _option_chip(option.level2.outcomes[method], option.lines[method])
            if method in option.lines
            else "—"
            for method in methods
        ]
        rows.append([key, *cells])  # plain text wraps at its hyphens in a narrow column
        if why:
            reasons.append(f"- `{key}`: " + "; ".join(why))
    pending = selected - len(members)
    counted = f"{len(members)} of {selected} selected cells run" + (
        f"; {pending} not yet." if pending > 0 else "."
    )
    if left:
        counted += f" {left} not counted ([why](#not-on-this-page))."
    noun = "option" if len(rows) == 1 else "options"
    return [
        *lines,
        f"**{len(rows)} {noun} compared** "
        + _chips().chip(f"{par} on a par under every method compared", "ok" if par else "muted")
        + f" · {counted}",
        "",
        f'??? note "{title}: one row per option"',
        "",
        "    <div data-search-exclude markdown>",
        "",
        *[f"    {row}" for row in _table(["Option", *methods], rows)],
        "",
        *[
            f"    {item}".rstrip()
            for item in (["Not on a par, and why:", "", *reasons] if reasons else [])
        ],
        *([""] if reasons else []),
        "    </div>",
        "",
    ]


def _not_counted_label(line: Mapping[str, Any]) -> str:
    """A cell not counted, as the option a reader asks for."""
    if line["kind"] == "loss":
        return line["option"]
    return line["option"] if line["kind"] == "recipe" else line["option"].split("=", 1)[-1]


def _not_counted_section(
    about: Mapping[str, Any], left: Mapping[str, list[tuple[Mapping[str, Any], str]]]
) -> list[str]:
    """What the page leaves out: the groups of the grid without level 2, and the cells of
    the summary :func:`grasppanda.combine.reach.level2` does not count, each with why; the
    cells one row per option in a fold."""
    from grasppanda.combine.reach import LEVEL2_EXCLUDED, LEVEL2_NONE

    words = {
        "schedule-slot": LEVEL2_NONE["schedule"],
        "schedule": LEVEL2_EXCLUDED["schedule"] + " (a schedule-free optimizer, a recipe "
        "preset with a scheduler)",
        "runs": LEVEL2_EXCLUDED["runs"] + "; it stays named on every page",
        "joint": JOINT,
        "own": OWN,
        "other": NOT_READ,
    }
    lines = [
        "## Not on this page {#not-on-this-page}",
        "",
        *[
            f"- **{LEFT_OUT_TITLES.get(group, group)}**: {reason}."
            for group, reason in about["left_out"].items()
        ],
        *[
            f"- **{NOT_COUNTED_TITLES[group]}** ({len(found)} "
            f"{'cell' if len(found) == 1 else 'cells'}): {words[group]}."
            for group, found in left.items()
        ],
        "",
    ]
    rows = []
    for group, found in left.items():
        # one row per option and reason: the methods it names, the group and the detail
        by_option: dict[tuple[str, str], list[str]] = {}
        for line, detail in found:
            by_option.setdefault((_not_counted_label(line), detail), []).append(line["method"])
        for (option, detail), methods in by_option.items():
            why = NOT_COUNTED_TITLES[group] + (f": {detail}" if detail else "")
            rows.append([f"`{option}`", ", ".join(_ordered(methods)), why])
    if rows:
        lines += [
            '??? note "The cells not counted, option by option"',
            "",
            "    <div data-search-exclude markdown>",
            "",
            *[f"    {row}" for row in _table(["Option", "Under", "Why"], rows)],
            "",
            "    </div>",
            "",
        ]
    lines += [
        "Accuracy of any kind is on the [model zoo](../model_zoo.md). The noise floor above is "
        "the only repeat: every cell is one run.",
        "",
    ]
    return lines


def page(summary: Mapping[str, Any], path: Path | None = None) -> str:
    """The page of ``summary`` (read from ``path``, :data:`SUMMARY` by default): what it
    counts is what :func:`grasppanda.combine.reach.level2` counts."""
    about = summary["about"]
    settings = about["settings"]
    cells_ = summary["cells"]
    references_ = summary["references"]
    methods = _methods([*cells_, *references_])
    chips = _chips()
    shown, left = compared_options(summary, path)
    left_by_kind = Counter(line["kind"] for found in left.values() for line, _ in found)
    lines = [
        "<!-- Generated by tools/learn_check.py from learns_like_reference.json.",
        "     Do not edit by hand: run (python -m tools.learn_check), export, regenerate. -->",
        "",
        "# Does it train like the shipped component?",
        "",
        "An option that runs ([Does it run?](measured_combinations.md)) may still train badly. "
        f"Here each swap that ran trains {settings['iterations']} iterations beside the method "
        "as shipped (the *reference* below, not the authors' code), and the losses are "
        f"compared term by term. Other pages call this check *{chips.LEVEL2_NAME}*.",
        "",
        '--8<-- "docs/includes/caveat_probe.md"',
        "",
        f"**One run per cell, {settings['iterations']} iterations on {settings['frames']} "
        "frames of one scene.** An outcome is about training dynamics, **not an accuracy "
        "measurement**: a cell on a par may still detect worse grasps, and one that is worse "
        "here may catch up over a full schedule.",
        "",
        "## The answer, kind by method {#the-answer}",
        "",
        '<div class="gp-matrix" markdown>',
        "",
        *_answer(shown, cells_, methods),
        "",
        "</div>",
        "",
        "A cell counts the options of a kind under that method: "
        + " ".join(chips.chip(words, tone) for _, tone, words in COUNTED)
        + f" {chips.chip('diverging or did not train', 'no')} "
        f"{chips.chip('not comparable', 'muted')}. — : the method has no such kind. Every "
        "option is [below](#option-by-option); what is not counted is "
        "[at the end](#not-on-this-page).",
        "",
        "## How a cell is judged {#how-a-cell-is-judged}",
        "",
        f"- **The same settings for a cell and its reference**: {chips.measured_on(2)}.",
        "- **Per loss term**, the mean of the last "
        f"{settings['window']} iterations of the cell is divided by the reference's.",
        "- **On a par**: every comparable term ends within the method's tolerance, nothing rises, "
        "every value is finite and a frame has a grasp. **Worse**: a term ends above the "
        "tolerance, or a part got no gradient. **Diverging**: a term rises at the end, a value "
        "became non-finite, or no grasp was predicted.",
        "- **A term the option redefines is not compared** (a loss preset, a head bundle, an "
        "assigner, a stage with its own loss code); the cell is judged on the other terms.",
        "- **Counted only where the comparison holds**: a swap that did not run at level 1, and a "
        "cell trained on a schedule other than its reference's, are not counted; "
        "[Not on this page](#not-on-this-page) names them.",
        f"- **The tolerance of a method** is {about.get('tolerance', TOLERANCE):g} unless its two "
        "reference runs (seed 0 and 1) "
        "differ by more; then it is the next step of "
        f"{about.get('step', STEP):g} above that spread.",
        *[
            f"- **{method}** runs under `--schedule {schedule}`, its reference and its cells "
            f"alike: {why}."
            for method, (schedule, why) in METHOD_SCHEDULES.items()
        ],
        "",
        '??? note "Thresholds, measures and outcomes in full"',
        "",
        *_indent(
            [
                *[f"- **{name}**: {text}." for name, text in about["measures"].items()],
                "",
                "Outcomes, in order of precedence:",
                "",
                *_table(
                    ["Outcome", "Meaning"], [[f"`{k}`", v] for k, v in about["outcomes"].items()]
                ),
                "",
                "A term is **not comparable** when the cell changes its definition; the cell is "
                "judged on the others:",
                "",
                *[f"- {name}: {text}." for name, text in about["not_comparable"].items()],
                "",
                f"**The tolerance and the rising threshold of a method.** {about['calibration']}.",
                "",
                *_calibration_table(about),
            ]
        ),
        "",
        "## Option by option {#option-by-option}",
        "",
        "One chip per option and method: ✓ on a par, *worse*, *diverging*, *did not train*, "
        "*n/c* not comparable. Hover a chip for the ratios to the reference; the list under a "
        "table says why a cell is not on a par. A loss preset put in several terms shows the "
        "worst of them.",
        "",
    ]
    for kind in PAGE_KINDS:
        lines += _kind_section(
            kind, KINDS[kind][2], about, cells_, shown.get(kind, []), left_by_kind[kind]
        )
    lines += _noise_section(about, references_)
    lines += _not_counted_section(about, left)
    lines += [
        "## The records {#the-records}",
        "",
        "- [learns_like_reference.json](learns_like_reference.json): every cell with its "
        "outcome, its reasons, the ratio of every term and its two commands.",
        "- [learns_like_reference/losses_per_iteration.json]"
        "(learns_like_reference/losses_per_iteration.json): the losses of every run, the mean "
        f"of every {BLOCK} iterations.",
        "",
        '??? note "Run it again"',
        "",
        "    ```bash",
        "    python -m tools.learn_check references --seeds 0 1 --scale-statistic-sha <SHA-256>",
        "    python -m tools.learn_check run --kind encoder --scale-statistic-sha <SHA-256>",
        "    python -m tools.learn_check status    # selected, run and pending cells; time left",
        "    python -m tools.learn_check export    # compare -> learns_like_reference.json",
        "    python -m tools.learn_check page      # this page",
        "    ```",
        "",
        "    The runs are those of `tools/probe_grid.py` (`--iters "
        f"{settings['iterations']} --predict-frames {settings['predict_frames']}`), one run "
        "directory per kind, so an interrupted run continues where it stopped. The SHA-256 is "
        "that of the imported width histogram of Scale-Balanced-Grasp.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _indent(lines: list[str]) -> list[str]:
    return [f"    {line}".rstrip() for line in lines]


def _calibration_table(about: Mapping[str, Any]) -> list[str]:
    """Per method: the largest seed spread of its references, and what it sets."""
    rows = []
    calibrated = about.get("calibrated") or {}
    for method in _ordered(calibrated):
        values = calibrated[method]
        terms = {
            f"{seed}: {name}": numbers
            for seed, by_term in (about.get("noise") or {}).get(method, {}).items()
            for name, numbers in by_term.items()
            if name != "loss"
        }
        ratios = [max(v[2], 1 / v[2]) for v in terms.values() if v[2]]
        rises = [abs(v[4] - v[3]) for v in terms.values() if v[3] is not None and v[4] is not None]
        rows.append(
            [
                method,
                f"{max(ratios):.3f}" if ratios else "not measured",
                f"**{values['tolerance']:g}**",
                f"{max(rises):.3f}" if rises else "not measured",
                f"**{values['rising_above']:g}**",
            ]
        )
    return _table(
        [
            "Method",
            "Largest ratio between two seeds",
            "Tolerance",
            "Largest difference of rise",
            "Rising threshold",
        ],
        rows,
    )


def _noise_section(about: Mapping[str, Any], references_: list[Mapping[str, Any]]) -> list[str]:
    """The references, and the noise floor: each reference trained under two seeds."""
    noise = about.get("noise") or {}
    lines = ["## The noise floor: the references twice {#the-noise-floor}", ""]
    if not noise:
        return [*lines, "The references have not been run under a second seed yet.", ""]
    rows = []
    for method in _ordered(noise):
        for seed, terms in noise[method].items():
            for name, (a, b, ratio, up_a, up_b) in terms.items():
                rows.append(
                    [
                        method,
                        seed,
                        f"`{name}`",
                        _fmt(a, 4),
                        _fmt(b, 4),
                        _fmt(ratio, 3),
                        f"{_fmt(up_a, 2)} / {_fmt(up_b, 2)}",
                    ]
                )
    ratios = [
        abs(math.log(values[2]))
        for by_seed in noise.values()
        for terms in by_seed.values()
        for name, values in terms.items()
        if name != "loss" and values[2]
    ]
    spread = ""
    if ratios:
        largest = math.exp(max(ratios))
        typical = math.exp(math.sqrt(sum(r * r for r in ratios) / len(ratios)))
        spread = (
            f" Over {len(ratios)} terms (the total left out), the largest ratio away from 1 is "
            f"{largest:.2f}x and the root mean square {typical:.3f}x."
        )
    lines += [
        "Each reference trained under seed 0 and again under seed 1, nothing else changed: the "
        "spread two runs of one config show, which sets each method's tolerance." + spread,
        "",
        '??? note "The references, and every term under both seeds"',
        "",
        *_indent(
            [
                *_references_table(references_),
                "",
                "*ratio* is seed 1 over seed 0 (mean of the last window); *rise* is each run's "
                "rise over the last half.",
                "",
                *_table(
                    [
                        "Method",
                        "Again with",
                        "Term",
                        "Seed 0",
                        "Other seed",
                        "Ratio",
                        "Rise (seed 0 / other)",
                    ],
                    rows,
                ),
            ]
        ),
        "",
    ]
    return lines


def _references_table(references_: list[Mapping[str, Any]]) -> list[str]:
    """The references of seed 0: their terms, grasps and scores."""
    if not references_:
        return ["No reference has been run yet."]
    rows = []
    for line in references_:
        if "@" in line["id"]:
            continue
        if line.get("failure"):
            rows.append([line["method"], "—", "—", "—", line["failure"]])
            continue
        terms = ", ".join(
            f"{name.removeprefix('loss/')} {_fmt(value, 3)}"
            for name, value in (line.get("terms") or {}).items()
            if name != "loss"
        )
        medians = ", ".join(_fmt(value, 3) for value in line.get("score_median") or [])
        if line.get("constant_scores"):
            medians += " (every score the same)"
        rows.append(
            [
                line["method"],
                terms,
                _grasps(line),
                medians or "—",
                f"{line.get('seconds') or 0:.0f} s",
            ]
        )
    order = _ordered(row[0] for row in rows)
    rows.sort(key=lambda row: order.index(row[0]))
    return [
        *_table(
            ["Reference", "Loss terms (mean of the last window)", "Grasps", "Median score", "Run"],
            rows,
        ),
        "",
        "*Run* is the wall time of the probe on a shared machine: an order of magnitude.",
    ]


def expected() -> str:
    return page(json.loads(SUMMARY.read_text(encoding="utf-8")), SUMMARY)


def write_page(args: argparse.Namespace) -> int:
    content = expected()
    if args.check:
        if not PAGE.is_file() or PAGE.read_text(encoding="utf-8") != content:
            print(f"stale: {PAGE.relative_to(REPO)} (run python -m tools.learn_check page)")
            return 1
        return 0
    changed = not PAGE.is_file() or PAGE.read_text(encoding="utf-8") != content
    PAGE.write_text(content, encoding="utf-8")
    print(f"{'wrote' if changed else 'unchanged'} {PAGE.relative_to(REPO)}")
    return 0


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def level1_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--level1",
            help="measured file the cells are selected from (default: the committed one)",
        )
        sub.add_argument(
            "--level1-run", help="select from a probe_grid run directory instead (e.g. verify-L1-a)"
        )

    def probe_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--jobs", type=int, default=1, help="probes at a time (default 1)")
        sub.add_argument("--timeout", type=float, default=TIMEOUT, help="seconds per probe")
        sub.add_argument(
            "--scale-statistic-sha",
            help="SHA-256 of the imported width histogram of Scale-Balanced-Grasp",
        )
        sub.add_argument("--allow-dirty", action="store_true", help="as for probe_grid run")
        sub.add_argument(
            "--commit",
            help="commit of the code that runs, from an exported copy of the repository "
            "(as for probe_grid run): the probes then run from that copy while the "
            "checkout changes",
        )
        sub.add_argument("--methods", nargs="+", help="only these methods")

    sub = commands.add_parser("references", help="run the references (seed 0, other seeds)")
    sub.add_argument("--seeds", type=int, nargs="+", default=[SETTINGS["seed"]])
    probe_options(sub)
    sub.set_defaults(handler=references)

    sub = commands.add_parser("run", help="run (or resume) the cells of one or more kinds")
    sub.add_argument("--kind", nargs="+", required=True, choices=tuple(KINDS))
    sub.add_argument("--only", nargs="+", help="glob patterns of cell ids")
    sub.add_argument(
        "--retry-failed",
        action="store_true",
        help="probe again the selected cells whose run failed for a reason of the machine "
        "(a crash or timeout without a report, a GPU launch failure)",
    )
    level1_options(sub)
    probe_options(sub)
    sub.set_defaults(handler=run)

    sub = commands.add_parser("status", help="selected, recorded and pending cells per kind")
    level1_options(sub)
    sub.set_defaults(handler=status)

    sub = commands.add_parser("export", help="compare and write the summary")
    sub.add_argument("--to", default=str(SUMMARY), help="summary to write")
    sub.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1], help="reference seeds (default 0 1)"
    )
    level1_options(sub)
    sub.set_defaults(handler=export)

    sub = commands.add_parser("page", help="write the page from the summary")
    sub.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    sub.set_defaults(handler=write_page)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
