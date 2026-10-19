"""Probe a config: does this combination really build, train and predict?

A probe is a miniature of ``grasppanda train`` followed by a few predictions. It runs the
real trainer on a handful of real frames (the split ``sample`` of a dataset exists for
this), with the config's own recipe: its optimizer, schedule, gradient accumulation and
clipping, and the hooks it lists. Nothing is written: no run directory, no checkpoint.

The outcome is one verdict of a fixed vocabulary, :data:`VERDICTS`, and the numbers it
rests on. The probe never reports more than it ran: without a dataset tree, or without
the label assets of the real frames, it says which and falls back to a wiring check on a
synthetic scene, whose best verdict is ``wired``, not ``runs``.

Beyond the verdict the report says whether the parts *learned* anything (judge 2,
:data:`JUDGE`): every loss term at every iteration and its trend over the iterations in
which it was active (:func:`loss_trend`, flags :data:`TREND_FLAGS`), which sub-modules
received a gradient, which parts changed, the scores of the predicted grasps and how far
the learning rate decayed. :data:`JUDGE_FIELDS` defines each field. They are reported,
not judged; only an empty prediction changes the verdict (``runs-empty``).

A probe is one run, and training on a GPU is not repeatable here. The config hash and
the parameter count are the same every time and the losses of the first iteration agree
to about six digits; everything after the first optimizer step (later losses, seed
fractions, the number of predicted grasps) differs from run to run. A combination near
the edge can therefore be ``runs`` in one probe and ``runs-unstable`` in the next.

The learning rate follows the recipe's schedule by default, run over the few epochs of
the probe. Schedules then treat methods unequally (one decays to 2 % within 20
iterations, another is still in its warm-up), so ``schedule="constant"`` trains every
step at the recipe's peak rate instead (:data:`SCHEDULES`, :func:`peak_rates`).

    report = probe("experimental/economicgrasp_ptv3_g1b-realsense_10e", iters=20, frames=8)
    report.verdict, report.detail
"""

from __future__ import annotations

import math
import statistics
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from grasppanda.config import Config, ConfigError
from grasppanda.registry import HOOKS, all_registries

__all__ = [
    "EXIT_STATUS",
    "FLAT_BELOW",
    "GRADIENT_THRESHOLD",
    "JUDGE",
    "JUDGE_FIELDS",
    "PARAMETER_TOLERANCE",
    "RISING_ABOVE",
    "SCHEDULES",
    "TREND_FLAGS",
    "VERDICTS",
    "ProbeReport",
    "ProbeRequestError",
    "dead_parts",
    "loss_trend",
    "out_of_memory",
    "peak_rates",
    "probe",
]

#: Version of the judgement. A report (and a record of the measured grid) without a
#: ``judge`` field was judged by the first judge, which had no ``runs-empty`` and reported
#: none of :data:`JUDGE_FIELDS` except ``no_gradient`` at the granularity of top-level parts.
JUDGE = 2
#: A loss term is ``flat`` when the mean of its last quarter differs from its first active
#: value by less than this fraction of it.
FLAT_BELOW = 0.05
#: A loss term is ``rising`` when the mean of the last half of its active iterations exceeds
#: the mean of the first half by more than this fraction of it.
RISING_ABOVE = 0.10
#: A sub-module is reached when the L2 norm of its gradient exceeds this in an iteration.
#: Adam's default epsilon: a gradient far below it moves no weight.
GRADIENT_THRESHOLD = 1e-8
#: A part changed when the L2 norm of the change of its trainable parameters exceeds this
#: times the larger of 1 and the L2 norm of the parameters before the first step.
PARAMETER_TOLERANCE = 1e-6
#: Significant digits of the loss values, scores and learning rates a report keeps.
DIGITS = 4
#: How a probe sets the learning rate (``grasppanda probe --schedule``), and what each means.
SCHEDULES: dict[str, str] = {
    "recipe": (
        "the config's own scheduler, run over the probe. The probe sets `train.epochs` to "
        "ceil(iterations / iterations per epoch), so a schedule that steps per epoch is "
        "compressed into those few epochs (GSNet's 0.95 per epoch ends at 0.95^9 = 0.63 "
        "after 20 iterations of 2 per epoch, EconomicGrasp's cosine at 2.4 % of its start) "
        "and one that steps per iteration runs its first steps only (FineGrasp's warm-up "
        "of 800 iterations keeps a probe of 20 at no more than 2.5 % of its rate)"
    ),
    "constant": (
        "no scheduler: every parameter group of every optimizer trains at the recipe's "
        "peak rate for the whole probe. The peak is the rate the optimizer gives the group "
        "(the configured `lr`, with the multipliers of ParamGroups), except for a schedule "
        "that sets a peak of its own: `max_lr` of OneCycleLR and CyclicLR, the larger "
        "factor of LinearLR. Every other schedule of the toolbox starts at that rate or "
        "reaches it at the end of its warm-up. When the probe's batches hold fewer frames "
        "than the recipe's, the peak is scaled by their ratio (the linear scaling rule: "
        "FineGrasp's 0.008 for 32 frames trains at 0.002 on batches of 8). A batch-norm "
        "momentum that the recipe decays per epoch (BNMomentumHook: GraspNet-baseline, "
        "Scale-Balanced-Grasp, HGGD) is held at its first epoch's value likewise "
        "(`hooks.<i>.decay=1`): decayed over epochs of a few iterations it can reach its "
        "floor of 0.001 within the probe (Scale-Balanced-Grasp's at iteration 73), and the "
        "running statistics, which prediction normalises with, then stop following the "
        "weights; training itself does not read them. The optimizer (its weight decay and "
        "its momentum as configured), clipping, accumulation and every other hook are the "
        "recipe's"
    ),
}
#: Length of the run a scheduler is built with to read its peak rate (:func:`peak_rates`).
#: The peak does not depend on it; a run of normal length avoids the adjustments a
#: scheduler makes for a run of a few steps.
_NOMINAL_STEPS = 10_000

#: The verdicts of a probe and what each one means. Exactly one is reported.
VERDICTS: dict[str, str] = {
    "runs": (
        "On real frames: the detector was built, every requested training iteration ran "
        "with finite losses and at least one real seed, every top-level part of the model "
        "that holds trainable parameters received a gradient in at least one iteration, the "
        "weights stayed finite, and prediction returned one valid grasp group for every "
        "probed frame, at least one of them with a grasp."
    ),
    "runs-empty": (
        "As `runs`, except that no probed frame has a grasp: prediction returned one valid "
        "grasp group for every probed frame and every group is empty. Every stage ran; a "
        "model trained for a few iterations often keeps no seed. A run that is also unstable "
        "is `runs-unstable`."
    ),
    "runs-unstable": (
        "On real frames: the detector was built, trained and predicted, but a loss or a "
        "weight became non-finite (a non-finite loss ends the training there), or an "
        "iteration had no real seed (no scene point passed the proposal, so only the "
        "proposal was trained), or no iteration supervised a seed, or a top-level part with "
        "trainable parameters received no gradient in any iteration, or fewer iterations ran "
        "or fewer frames were predicted than were asked for."
    ),
    "fails": (
        "An exception was raised after the config check had passed. The report names the "
        "phase (build, data, optimizer, loss, backward, step, hook, predict), the component "
        "it came from and the message."
    ),
    "refused": (
        "The config check refused the config (the same check as `grasppanda print-config "
        "--check --for train`). Nothing was built; the report carries the messages."
    ),
    "wired": (
        "No real frame was read. On a synthetic table-top scene with random labels the "
        "detector was built, back-propagated a loss through every stage and predicted. "
        "This shows that the stages connect; it says nothing about real data."
    ),
}
#: Exit status of ``grasppanda probe`` per verdict. ``wired`` is 0 only when the synthetic
#: check was asked for; as a fallback for missing real frames it is 3.
EXIT_STATUS = {
    "runs": 0,
    "runs-empty": 0,
    "wired": 0,
    "runs-unstable": 1,
    "fails": 1,
    "refused": 2,
}


def _power(value: float) -> str:
    """``1e-08`` as ``1e-8``: a threshold as the definitions print it."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


#: The flags of the trend of a loss term (:func:`loss_trend`), in the order they are listed.
#: "Active" iterations are those in which the term was reported, finite and not exactly 0.
TREND_FLAGS: dict[str, str] = {
    "flat": (
        "the term was active in at least two iterations, and the mean of its last quarter "
        f"differs from its first active value by less than {FLAT_BELOW * 100:g} % of that value"
    ),
    "rising": (
        "the term was active in at least two iterations, and the mean of the last half of "
        "its active iterations exceeds the mean of the first half by more than "
        f"{RISING_ABOVE * 100:g} % of the first half's mean (with an odd count the middle "
        "iteration is in neither half)"
    ),
    "zero_at_end": "the term is exactly 0.0, or not reported, at the last completed iteration",
    "constant": (
        "the term was reported in at least two iterations with the very same value in every "
        "one of them, zeros included"
    ),
}
#: What judge 2 adds to a report, field by field. Reported, not judged: none of these
#: changes the verdict, except that a dead top-level part (an entry of ``no_gradient``
#: without a dot) makes a run ``runs-unstable``, as it did for the first judge.
JUDGE_FIELDS: dict[str, str] = {
    "judge": f"the version of the judgement, {JUDGE}; absent from a report of the first judge",
    "losses_per_iteration": (
        "every loss term and the total `loss` at every completed iteration, to "
        f"{DIGITS} significant digits; null where the detector did not report the term"
    ),
    "loss_trends": (
        "per loss term and for the total: `active`, the number of iterations in which the "
        "term was reported, finite and not exactly 0.0; `first`, its value at the first of "
        "them; `last_quarter`, the mean of the last ceil(active / 4) of them; `fall`, "
        "(first - last_quarter) / |first|, positive when the term fell (0.7: by 70 %); "
        "null when the term was never active; `flags`, those of TREND_FLAGS that hold"
    ),
    "no_gradient": (
        "the units of the model whose gradient never had an L2 norm above "
        f"{_power(GRADIENT_THRESHOLD)} in an iteration, over its trainable parameters. A unit is a "
        "registered sub-module at depth 2 (`backbone.encoder`, `head.score_net`) or a "
        "parameter a top-level part holds itself (`head.bias`); when no unit of a top-level "
        "part was reached, the part is listed instead of its units (`backbone`: a dead part). "
        "Gradients are read before every optimizer step, unscaled under mixed precision"
    ),
    "unchanged": (
        "the top-level parts whose trainable parameters did not change: the L2 norm of the "
        "difference between before the first optimizer step and after the last one is at "
        f"most {_power(PARAMETER_TOLERANCE)} times the larger of 1 and the L2 norm before"
    ),
    "units_checked": "the number of units with trainable parameters (see no_gradient)",
    "parts_checked": "the number of top-level parts with trainable parameters",
    "score_quantiles": (
        "per predicted frame, the smallest, the median and the largest score of its "
        f"predicted grasps, to {DIGITS} significant digits; null for a frame without a "
        "grasp. Scores are those the detector returns (their meaning is its `score_kind`)"
    ),
    "learning_rate": (
        "the learning rate of the first parameter group of the first optimizer at the first "
        "and at the last optimizer step, as those steps used it"
    ),
    "lr_factor": (
        "the last of these divided by the first: the decay the schedule applied within the "
        "probe. A schedule that steps per epoch decays with the probe's epochs, which follow "
        "from the batch size and the number of frames (GSNet's 0.95 per epoch: 0.95^9 after "
        "20 iterations of 2 per epoch); 1 under the schedule `constant`; null when the "
        "first is 0"
    ),
}
#: Functions of the engine whose frame on a traceback names the phase that failed.
_PHASES = {
    "build_model": "build",
    "build_dataset": "data",
    "build_train_loader": "data",
    "build_optimizers": "optimizer",
    "build_schedulers": "optimizer",
    "build_hooks": "hook",
}
_SLOT_NAMES = {
    "BACKBONES": "backbone",
    "PROPOSALS": "proposal",
    "VIEWS": "view stage",
    "ASSIGNERS": "assigner",
    "GROUPERS": "grouper",
    "HEADS": "head bundle",
    "LOSSES": "loss",
    "DATASETS": "dataset",
    "TRANSFORMS": "transform",
    "HOOKS": "hook",
    "POSTPROCESSORS": "post-processor",
    "DETECTORS": "detector",
}


@dataclass
class ProbeReport:
    """What a probe ran and what it saw.

    Attributes:
        verdict: One key of :data:`VERDICTS`.
        detail: One sentence that justifies it.
        config: The probed config, as it was named.
        overrides: Overrides given by the caller.
        probe_overrides: Overrides the probe added to select frames and to keep the run
            short; with ``overrides`` they reproduce it through ``grasppanda train``
            (under the schedule ``constant``: ``scheduler=null``, ``hooks.<i>.decay=1`` for
            a batch-norm momentum the recipe decays, and a note says when the recipe's
            peak is not the optimizer's rate).
        config_hash: Hash of the config as given (without the probe's overrides).
        schedule: How the learning rate was set: a key of :data:`SCHEDULES`.
        rate: Under the schedule ``constant``, the rate every step used in the first
            parameter group of the first optimizer (the recipe's peak, scaled for batches
            smaller than the recipe's); ``None`` otherwise and before an optimizer was
            built.
        data: ``real``, ``synthetic`` or ``none``.
        fallback: Why real frames were not read, when they were not.
        detector: Detector type.
        parameters: Number of parameters.
        device: Device the probe ran on.
        frames: The frames trained on, by key.
        batch_size: Frames per iteration.
        iterations_requested, iterations: Asked for and completed.
        losses: Loss term → ``[first, last]`` over the completed iterations; ``loss`` is
            their total.
        metrics: Logged metric → ``[first, last]``.
        real_seeds: Fraction of seed slots that held a real seed, per iteration; empty
            when the detector reports none (every seed is real, or it has no seeds).
        supervised_seeds: Fraction of seeds that took part in the grasp losses, per
            iteration; empty when the detector reports none.
        seconds_per_iteration: Median time of an iteration without data loading, first
            iteration excluded. Measured on this machine at this moment.
        seconds_data: Median time spent waiting for a batch.
        peak_gpu_mib: Peak GPU memory allocated by PyTorch in this process.
        grasps_per_frame: Number of grasps predicted for each probed frame.
        postprocessing: For the evaluation variant ``post``, when the config has one:
            ``steps`` (component types, in order), ``grasps`` (per predicted frame:
            the number predicted, then the number left after every step) and
            ``seconds`` (per frame, per step; measured on this machine).
        predicted_frames: Their keys.
        no_gradient: Units of the model that never received a gradient (see
            :data:`JUDGE_FIELDS`); a top-level part when none of its units did (a dead
            stage). Empty when every unit was reached, and when no iteration ran.
        judge, losses_per_iteration, loss_trends, unchanged, units_checked,
            parts_checked, score_quantiles, learning_rate, lr_factor: What judge 2
            adds, defined in :data:`JUDGE_FIELDS`.
        nonfinite: Non-finite values that were met, as sentences.
        failure: Phase, component, exception type and message of a failure.
        refusal: Messages of the config check.
        notes: Things the probe changed or noticed that a reader should know.
        seconds: Wall time of the probe.
    """

    verdict: str = "fails"
    detail: str = ""
    config: str = ""
    overrides: list[str] = field(default_factory=list)
    probe_overrides: list[str] = field(default_factory=list)
    config_hash: str | None = None
    schedule: str = "recipe"
    rate: float | None = None
    data: str = "none"
    fallback: str | None = None
    detector: str | None = None
    parameters: int | None = None
    device: str | None = None
    frames: list[str] = field(default_factory=list)
    batch_size: int | None = None
    iterations_requested: int = 0
    iterations: int = 0
    losses: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    real_seeds: list[float] = field(default_factory=list)
    supervised_seeds: list[float] = field(default_factory=list)
    seconds_per_iteration: float | None = None
    seconds_data: float | None = None
    peak_gpu_mib: float | None = None
    grasps_per_frame: list[int] = field(default_factory=list)
    postprocessing: dict[str, Any] = field(default_factory=dict)
    predicted_frames: list[str] = field(default_factory=list)
    no_gradient: list[str] = field(default_factory=list)
    judge: int = JUDGE
    losses_per_iteration: dict[str, list[float | None]] = field(default_factory=dict)
    loss_trends: dict[str, dict[str, Any]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    units_checked: int = 0
    parts_checked: int = 0
    score_quantiles: list[list[float] | None] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    lr_factor: float | None = None
    nonfinite: list[str] = field(default_factory=list)
    failure: dict[str, str] | None = None
    refusal: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def exit_status(self, *, synthetic_requested: bool = False) -> int:
        if self.verdict == "wired" and not synthetic_requested:
            return 3
        return EXIT_STATUS[self.verdict]


class _Enough(Exception):
    """Raised by the recorder when the requested iterations are done."""


class _NoRealFrames(Exception):
    """Real frames cannot be read on this machine; the message says what is missing."""


class ProbeRequestError(ValueError):
    """The probe was asked for something that cannot be run: a split the dataset does not
    have, or a count below one. A problem of the request, not a verdict about the config."""


# -------------------------------------------------------------------------- failures


def _component(error: BaseException) -> str:
    """The registered component nearest to where ``error`` was raised, e.g. ``grouper
    DepthCylinderGroup``; empty when no frame belongs to one."""
    registered = {
        obj: (_SLOT_NAMES.get(name, name.lower()), key)
        for name, registry in all_registries().items()
        if not registry.third_party
        for key in registry.names()
        if isinstance(obj := registry.get(key), type)
    }
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    for frame in reversed(frames):
        owner = frame.f_locals.get("self")
        for cls in type(owner).__mro__ if owner is not None else ():
            if cls in registered and registered[cls][0] != "detector":
                slot, name = registered[cls]
                return f"{slot} {name}"
    return ""


def _phase(error: BaseException, default: str) -> str:
    """The phase a traceback belongs to, from the engine functions on it."""
    tb, found = error.__traceback__, default
    while tb is not None:
        code = tb.tb_frame.f_code
        name, file = code.co_name, code.co_filename.replace("\\", "/")
        if name in _PHASES:
            return _PHASES[name]
        if "/torch/utils/data/" in file:
            return "data"
        if name == "backward" and file.endswith("torch/_tensor.py"):
            return "backward"
        if file.endswith("engine/trainer.py") and name == "step":
            return "step"
        if file.endswith("engine/trainer.py") and name == "_call":
            return "hook"
        tb = tb.tb_next
    return found


def _fail(report: ProbeReport, error: BaseException, default: str, when: str = "") -> None:
    phase, component = _phase(error, default), _component(error)
    message = " ".join(str(error).split())
    report.failure = {
        "phase": phase,
        "when": when,
        "component": component,
        "error": type(error).__name__,
        "message": message,
    }
    where = phase + (f" ({when})" if when else "") + (f", in the {component}" if component else "")
    report.verdict = "fails"
    report.detail = f"at {where}: {type(error).__name__}: {message}"
    if device_fault(report):
        report.notes.append(
            "the GPU reported a launch failure. On a shared or nearly full device that is "
            "usually a statement about this machine at this moment, not about the "
            "combination: probe again; if it repeats on a free device, it is a defect."
        )
    if out_of_memory(report):
        report.notes.append(
            "the probe ran out of GPU memory. That is a statement about this machine at this "
            "moment (other processes may hold memory on the device), not about the "
            "combination: probe again when the device is free, or with a smaller "
            "`--opts data.train.batch_size=...`."
        )


def out_of_memory(report: ProbeReport) -> bool:
    """Whether a probe failed for lack of GPU memory rather than because of its config."""
    failure = report.failure or {}
    text = f"{failure.get('error', '')} {failure.get('message', '')}".lower()
    # The allocator's own error, or a CUDA library that could not allocate its handle.
    return "out of memory" in text or "alloc_failed" in text


def device_fault(report: ProbeReport) -> bool:
    """Whether a probe failed with a CUDA launch failure, which a busy device produces."""
    failure = report.failure or {}
    text = str(failure.get("message", "")).lower()
    return "unspecified launch failure" in text or "cuda error: an illegal" in text


# ------------------------------------------------------------------------- recording


def _digits(value: float, digits: int = DIGITS) -> float:
    """``value`` to ``digits`` significant digits."""
    return float(f"{value:.{digits}g}")


def _maybe(value: float | None) -> float | None:
    return None if value is None else _digits(value)


def _unit(name: str) -> str:
    """The unit of gradient reach a parameter belongs to: the first two components of its
    dotted name, a sub-module at depth 2 or a parameter a top-level part holds itself."""
    return ".".join(name.split(".")[:2])


def dead_parts(no_gradient: Sequence[str]) -> list[str]:
    """The top-level parts of ``no_gradient`` (entries without a dot): parts none of whose
    units received a gradient. The first judge listed parts only, so its entries all are."""
    return [name for name in no_gradient if "." not in name]


def loss_trend(series: Sequence[float | None]) -> dict[str, Any]:
    """The trend of one loss term over the completed iterations (``None``: not reported).

    Returns ``active``, ``first``, ``last_quarter``, ``fall`` and ``flags`` as
    :data:`JUDGE_FIELDS` defines them (``loss_trends``), with the flags of
    :data:`TREND_FLAGS` that hold. Only the iterations in which the term was active count
    for the trend: a grasp term that is 0.0 until a seed is supervised starts there.
    """
    reported = [value for value in series if value is not None]
    active = [value for value in reported if value != 0.0 and math.isfinite(value)]
    count = len(active)
    first = last_quarter = fall = None
    held = set()
    if count:
        first = active[0]
        tail = active[-math.ceil(count / 4) :]
        last_quarter = sum(tail) / len(tail)
        fall = (first - last_quarter) / abs(first)
        if count >= 2 and abs(last_quarter - first) < FLAT_BELOW * abs(first):
            held.add("flat")
        half = count // 2
        if half:
            early, late = sum(active[:half]) / half, sum(active[-half:]) / half
            if late - early > RISING_ABOVE * abs(early):
                held.add("rising")
    if reported and (series[-1] is None or series[-1] == 0.0):
        held.add("zero_at_end")
    if len(reported) >= 2 and all(value == reported[0] for value in reported):
        held.add("constant")
    return {
        "active": count,
        "first": _maybe(first),
        "last_quarter": _maybe(last_quarter),
        "fall": None if fall is None else round(fall, 3),
        "flags": [name for name in TREND_FLAGS if name in held],
    }


class _Recorder:
    """A training hook that keeps every iteration's numbers and stops after ``limit``.

    Before every optimizer step it notes which units of the model received a gradient and
    the learning rate the step uses; before the first step it keeps a copy of the trainable
    parameters, to tell afterwards which parts changed.
    """

    priority = 95

    def __init__(self, limit: int, device: torch.device) -> None:
        self.limit, self.device = limit, device
        self.rows: list[dict[str, float]] = []
        self.compute: list[float] = []
        self.waiting: list[float] = []
        self.keys: list[list[str]] = []
        self.units: dict[str, str] = {}  # unit -> its top-level part, trainable ones only
        self.reached: set[str] = set()
        self.before: dict[str, torch.Tensor] = {}  # trainable parameters before the first step
        self.model: Any = None
        self.learning_rates: list[float] = []
        self._mark = self._started = time.perf_counter()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def begin(self, observation: Any) -> None:
        """Called just before the loss of an iteration is computed."""
        self._started = time.perf_counter()
        self.waiting.append(self._started - self._mark)
        self.keys.append([str(key) for key in observation.keys])

    def before_run(self, state: Any) -> None:
        self._mark = time.perf_counter()

    def before_step(self, state: Any) -> None:
        """Called with the gradients of the iteration in place, before the optimizer step."""
        scaler = getattr(state, "scaler", None)
        enabled = scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)())
        self.note_gradients(state.model, float(scaler.get_scale()) if enabled else 1.0)
        self.note_learning_rate(getattr(state, "optimizers", {}).values())

    def note_gradients(self, model: Any, scale: float = 1.0) -> None:
        """Note the units whose gradient norm exceeds :data:`GRADIENT_THRESHOLD`; ``scale``
        is the factor of a mixed-precision scaler the gradients still carry."""
        if self.model is None:
            self.model = model
            self.before = {
                name: parameter.detach().to("cpu", copy=True)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        pending: list[str] = []
        norms: list[torch.Tensor] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            unit = _unit(name)
            self.units.setdefault(unit, name.split(".")[0])
            if unit in self.reached or parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            grad = grad.coalesce().values() if grad.is_sparse else grad
            pending.append(unit)
            norms.append(torch.linalg.vector_norm(grad.float()))
        if not norms:
            return
        squares: dict[str, float] = {}
        values = torch.stack([norm.to(norms[0].device) for norm in norms]).tolist()
        for unit, value in zip(pending, values):
            squares[unit] = squares.get(unit, 0.0) + value * value
        for unit, total in squares.items():
            if math.sqrt(total) / scale > GRADIENT_THRESHOLD:
                self.reached.add(unit)

    def note_learning_rate(self, optimizers: Any) -> None:
        """The learning rate of the first parameter group of the first optimizer."""
        for optimizer in optimizers:
            groups = getattr(optimizer, "param_groups", None) or []
            try:
                self.learning_rates.append(float(groups[0]["lr"]))
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            return

    def no_gradient(self) -> list[str]:
        """Units without a gradient; a top-level part when none of its units had one."""
        parts: dict[str, list[str]] = {}
        for unit, part in self.units.items():
            parts.setdefault(part, []).append(unit)
        found: list[str] = []
        for part, members in parts.items():
            missing = [unit for unit in members if unit not in self.reached]
            found += [part] if len(missing) == len(members) else missing
        return sorted(found)

    def unchanged(self) -> list[str]:
        """Top-level parts whose trainable parameters did not change since the first step."""
        if self.model is None:
            return []
        current = dict(self.model.named_parameters())
        change: dict[str, float] = {}
        size: dict[str, float] = {}
        for name, before in self.before.items():
            after = current.get(name)
            if after is None:
                continue
            part = name.split(".")[0]
            start = before.to(torch.float64)
            moved = after.detach().to("cpu", torch.float64) - start
            change[part] = change.get(part, 0.0) + float(moved.pow(2).sum())
            size[part] = size.get(part, 0.0) + float(start.pow(2).sum())
        return sorted(
            part
            for part in change
            if math.sqrt(change[part]) <= PARAMETER_TOLERANCE * max(math.sqrt(size[part]), 1.0)
        )

    def before_epoch(self, state: Any) -> None:
        pass

    def after_iter(self, state: Any) -> None:
        self.rows.append({name: float(value) for name, value in state.outputs.items()})
        self._sync()
        self._mark = time.perf_counter()
        self.compute.append(self._mark - self._started)
        if len(self.rows) >= self.limit:
            raise _Enough

    def after_epoch(self, state: Any) -> None:
        pass

    def after_run(self, state: Any) -> None:
        pass


def _summarise(report: ProbeReport, recorder: _Recorder) -> None:
    rows = recorder.rows
    report.iterations = len(rows)
    report.frames = list(dict.fromkeys(key for keys in recorder.keys for key in keys))
    report.batch_size = max((len(keys) for keys in recorder.keys), default=None)
    if not rows:
        report.no_gradient = []
        return
    report.no_gradient = recorder.no_gradient()
    report.unchanged = recorder.unchanged()
    report.units_checked = len(recorder.units)
    report.parts_checked = len(set(recorder.units.values()))
    rates = recorder.learning_rates
    if rates:
        report.learning_rate = [_digits(rates[0]), _digits(rates[-1])]
        report.lr_factor = _digits(rates[-1] / rates[0]) if rates[0] else None
    # Every name any iteration reported, in the order they first appear: a term may switch
    # on later (a grasp term once a seed is supervised).
    for name in dict.fromkeys(name for row in rows for name in row):
        series = [row.get(name) for row in rows]
        present = [value for value in series if value is not None]
        pair = [present[0], present[-1]]
        if name == "loss" or name.startswith("loss/"):
            report.losses[name] = pair
            report.losses_per_iteration[name] = [_maybe(value) for value in series]
            report.loss_trends[name] = loss_trend(series)
        elif name.startswith("metric/"):
            report.metrics[name] = pair
    report.real_seeds = [row["metric/real_seeds"] for row in rows if "metric/real_seeds" in row]
    report.supervised_seeds = [
        row["metric/supervised_seeds"] for row in rows if "metric/supervised_seeds" in row
    ]
    steady = recorder.compute[1:] or recorder.compute
    report.seconds_per_iteration = statistics.median(steady)
    report.seconds_data = statistics.median(recorder.waiting[1:] or recorder.waiting)


# ------------------------------------------------------------------- the learning rate


class _Groups(torch.optim.Optimizer):
    """The parameter groups of an optimizer with a placeholder for its parameters: what a
    scheduler reads and writes, so that a scheduler can be built on it without touching
    the optimizer (OneCycleLR sets the rate and the momentum of every group it is given)."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        groups = []
        for group in optimizer.param_groups:
            settings = {key: value for key, value in group.items() if key != "params"}
            groups.append({**settings, "params": [torch.zeros(1, requires_grad=True)]})
        super().__init__(groups, dict(optimizer.defaults))

    def step(self, closure: Any = None) -> None:  # never stepped
        return None


def peak_rates(
    scheduler: Mapping[str, Any] | None, optimizer: torch.optim.Optimizer
) -> list[float]:
    """The peak learning rate of every parameter group of ``optimizer`` under the scheduler
    node ``scheduler`` of a config (``None``: no scheduler), as :data:`SCHEDULES` defines it.

    ``optimizer`` must not have been given to a scheduler yet: its rates are those it was
    configured with. The scheduler is built on a copy of its groups, never on it.
    """
    configured = [float(group["lr"]) for group in optimizer.param_groups]
    if scheduler is None:
        return configured
    from grasppanda.engine.optim import build_schedulers

    groups = _Groups(optimizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_schedulers(scheduler, {"peak": groups}, total_steps=_NOMINAL_STEPS)["peak"]
    kind = type(built).__name__
    if kind == "OneCycleLR":
        return [float(group["max_lr"]) for group in groups.param_groups]
    if kind == "CyclicLR":
        return [float(rate) for rate in built.max_lrs]
    if kind == "LinearLR":
        factor = max(float(built.start_factor), float(built.end_factor))
        return [rate * factor for rate in configured]
    return configured


def _hold_peak(
    report: ProbeReport,
    state: Any,
    scheduler: Mapping[str, Any] | None,
    batches: tuple[int, int] | None = None,
) -> None:
    """Set every parameter group of every optimizer of a trainer that runs no scheduler to
    the peak rate of the recipe's ``scheduler``; record the rate of the first group.

    ``batches`` is (frames per batch of the probe, of the recipe). When the probe's batches
    are smaller, every peak is scaled by their ratio (the linear scaling rule)."""
    scale = 1.0
    if batches is not None and 0 < batches[0] < batches[1]:
        scale = batches[0] / batches[1]
    raised = []
    for optimizer in state.optimizers.values():
        for group, peak in zip(optimizer.param_groups, peak_rates(scheduler, optimizer)):
            if not math.isclose(peak, float(group["lr"]), rel_tol=1e-9):
                raised.append((float(group["lr"]), peak))
            group["lr"] = peak * scale
            if report.rate is None:
                report.rate = _digits(peak * scale)
    if scale != 1.0 and batches is not None and report.rate is not None:
        report.notes.append(
            f"schedule constant: batches hold {batches[0]} frames where the recipe's hold "
            f"{batches[1]}, so every peak rate was scaled by {batches[0]}/{batches[1]} (the "
            f"linear scaling rule): the probe trained at {report.rate:g}."
        )
    if raised:
        kind = (scheduler or {}).get("type", "the scheduler")
        before, after = raised[0]
        report.notes.append(
            f"schedule constant: {kind} of the recipe peaks at {after:g}, not at the rate of "
            f"the optimizer ({before:g}): the probe trained at {after * scale:g} ({len(raised)} "
            "parameter group(s) raised). With `scheduler=null`, `grasppanda train` would use "
            "the optimizer's rate."
        )


def _decaying_momentum(given: Config) -> list[tuple[int, Any, int]]:
    """The batch-norm momentum schedules of the recipe's hooks that decay: ``(position in
    `hooks`, the built hook, the first epoch at its floor)`` for every BNMomentumHook whose
    momentum is not the same in every epoch.

    The schedule ``constant`` holds them at the momentum of the first epoch, as it holds
    the learning rate at its peak: the recipe decays the momentum per epoch, an epoch of a
    probe is a few iterations long, and a momentum at its floor of 0.001 leaves the running
    statistics, which prediction normalises with, where they were while the weights train on.
    """
    from grasppanda.engine.hooks.bn_momentum import BNMomentumHook

    far = 10_000  # epochs: further than any recipe's decay
    found = []
    for position, node in enumerate(given.get("hooks") or []):
        kind = node.get("type") if isinstance(node, Mapping) else None
        if kind not in HOOKS or HOOKS.get(kind) is not BNMomentumHook:
            continue
        hook = HOOKS.build(dict(node))
        last = hook.momentum(far)
        if last != hook.momentum(0):
            floor = next(epoch for epoch in range(far + 1) if hook.momentum(epoch) == last)
            found.append((position, hook, floor))
    return found


# ------------------------------------------------------------------------ real frames


def _dataset_options(config: Config, section: str) -> set[str]:
    """Constructor options of the dataset of ``data.<section>``; empty without one."""
    from grasppanda.registry import DATASETS, component_parameters

    node = config.select(f"data.{section}.dataset")
    if not isinstance(node, dict) or node.get("type") not in DATASETS:
        return set()
    return set(component_parameters(DATASETS.get(node["type"]))[0])


def _plan(
    given: Config,
    config: str | Path,
    overrides: Sequence[str],
    *,
    split: str,
    frames: int,
    predict_frames: int,
    iters: int,
    workers: int,
    schedule: str = "recipe",
) -> tuple[Config, list[str], str | None, list[str], tuple[int, int]]:
    """The config of the miniature run: the given one plus the probe's overrides.

    Returns the config, the overrides the probe added, the ``data`` section to predict
    on, notes about what had to change and the frames per batch (of the probe, of the
    recipe). Raises :class:`_NoRealFrames` when the frames or their label assets are not
    on this machine.

    Frames are selected through the options of the dataset reader: ``split`` names the
    few frames a reader ships for this purpose and ``frame_step`` thins them. A reader
    without those options is probed on the frames the config names. Under the schedule
    ``constant`` the scheduler is removed (``scheduler=null``); the trainer then sets the
    peak rate (:func:`_hold_peak`). A batch-norm momentum that the recipe decays per epoch
    is held at its first epoch's value (``hooks.<i>.decay=1``, :func:`_decaying_momentum`).
    """
    from grasppanda.datasets.assets import AssetError
    from grasppanda.engine.builders import build_dataset

    options = _dataset_options(given, "train")
    notes: list[str] = []
    added = [f"data.train.dataset.split={split}"] if "split" in options else []
    if "split" not in options:
        notes.append(
            "the training dataset has no `split` option, so the probe trains on the frames "
            "the config names."
        )
    added += [f"data.train.num_workers={workers}", "evaluation.interval=0"]
    added.append("evaluation.save_best=null")
    if schedule == "constant" and given.get("scheduler") is not None:
        added.append("scheduler=null")
    held = _decaying_momentum(given) if schedule == "constant" else []
    added += [f"hooks.{position}.decay=1" for position, _, _ in held]
    cfg = Config.load(config, [*overrides, *added], command="train").resolved()
    try:
        dataset = build_dataset(cfg, "train")
    except FileNotFoundError as error:
        raise _NoRealFrames(f"the dataset is not on this machine. {error}") from None
    except AssetError as error:
        if "scale_statistic asset" in str(error):
            # A histogram over a whole split: the few frames of a probe leave bins empty,
            # and the generator refuses that. The reference's own file does not depend on
            # the split.
            name = cfg["data"]["train"]["dataset"].get("type", "<dataset>").lower()
            section = "data.train.dataset.scale_statistic"
            values = f"{section}.source=legacy_import {section}.reference_sha256=<SHA-256>"
            raise _NoRealFrames(
                "the width histogram of Scale-Balanced-Grasp (`scale_statistic`) does not "
                "exist for the frames of the probe, and cannot be built from them: it is a "
                "statistic of a whole training split. Use the reference's own file, which "
                "does not depend on the split. If it is imported on this machine, probe with "
                f"`--opts {values}`. To import it once: write those two values into a config "
                f"(`grasppanda compose <config> --opts {values} -o <file>`) and run `grasppanda "
                f"prepare {name} --scale-statistic --import-legacy --scale-statistic-file "
                "<statistic/scale_distribution.npy of the reference repository> --config "
                "<file>`; the SHA-256 is that of the file "
                "(configs/scale_balanced_grasp/README.md)."
            ) from None
        hint = f" For the frames of the probe add `--split {split}` to that command."
        raise _NoRealFrames(
            f"a label asset of the frames is missing. {error}"
            + (hint if "split" in options else "")
        ) from None
    except ValueError as error:
        if "is not a split of" not in str(error):
            raise
        raise ProbeRequestError(f"{error} Choose the frames with `--split`.") from None
    count = used = len(dataset)
    loader = cfg["data"]["train"]
    step = max(1, math.ceil(count / max(frames, 1)))
    if step > 1 and "frame_step" in options:
        base = int(loader["dataset"].get("frame_step") or 1)
        added.append(f"data.train.dataset.frame_step={base * step}")
        used = len(range(0, count, step))
    if loader["drop_last"] and used < loader["batch_size"]:
        added.append("data.train.drop_last=false")
        notes.append(
            f"data.train.drop_last was switched off: the probe has {used} frames and the "
            f"config drops batches smaller than {loader['batch_size']}."
        )
        per_epoch = 1
    elif loader["drop_last"]:
        per_epoch = used // loader["batch_size"]
    else:
        per_epoch = math.ceil(used / loader["batch_size"])
    if used < loader["batch_size"]:
        notes.append(
            f"batches hold {used} frames, not the {loader['batch_size']} of the recipe: the "
            "probe has no more frames."
        )
    added.append(f"train.epochs={math.ceil(iters / per_epoch)}")
    for position, hook, floor in held:
        notes.append(
            f"schedule constant: the batch-norm momentum was held at {hook.momentum(0):g}, its "
            f"value in the recipe's first epoch (hooks.{position}.decay=1). The recipe "
            f"(BNMomentumHook) decays it per epoch to {hook.momentum(floor):g}, which an epoch "
            f"of {per_epoch} iteration(s) reaches at iteration {floor * per_epoch + 1}; from "
            "there the running statistics that prediction normalises with would stay where "
            "they were while the weights train on."
        )
    section = next((name for name in ("val", "test") if (cfg["data"] or {}).get(name)), None)
    if section is not None:
        other = _dataset_options(given, section)
        if "split" in other and "split" in options:
            added.append(f"data.{section}.dataset.split={split}")
            if "frame_step" in other:
                keep = max(1, math.ceil(count / max(predict_frames, 1)))
                added.append(f"data.{section}.dataset.frame_step={keep}")
        added.append(f"data.{section}.num_workers=0")
    batches = (min(used, loader["batch_size"]), loader["batch_size"])
    return (
        Config.load(config, [*overrides, *added], command="train"),
        added,
        section,
        notes,
        batches,
    )


def _real(
    report: ProbeReport,
    planned: Config,
    section: str | None,
    *,
    iters: int,
    predict_frames: int,
    device: str | None,
    hold: tuple[Mapping[str, Any] | None, tuple[int, int] | None] | None = None,
) -> None:
    """Train ``iters`` iterations with the real trainer, then predict. ``hold`` holds the
    recipe's scheduler node and the frames per batch (of the probe, of the recipe) under
    the schedule ``constant`` (``None``: the recipe's schedule)."""
    from grasppanda.engine.run import Run, resolve_device
    from grasppanda.engine.seeding import determinism_restored
    from grasppanda.ops import backends

    cfg = planned.resolved()
    target = resolve_device(device or cfg["runtime"]["device"])
    report.device, report.data = str(target), "real"
    backends.configure(cfg["runtime"]["ops"])
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    with tempfile.TemporaryDirectory(prefix="grasppanda-probe-") as scratch, determinism_restored():
        run = Run(Path(scratch), "probe", "train", planned, cfg, target)
        try:
            finished = _train_and_predict(
                report, run, section, iters=iters, predict_frames=predict_frames, hold=hold
            )
        finally:
            if target.type == "cuda":
                report.peak_gpu_mib = round(torch.cuda.max_memory_allocated(target) / 2**20, 1)
    if finished:
        _judge(report, "runs", f"{len(report.frames)} frames of the dataset", predict_frames)


def _summarised(report: ProbeReport, recorder: Any) -> Exception | None:
    """:func:`_summarise`, or the error that stopped it. The summary reads the model's
    weights and gradients; after a device fault (a CUDA launch failure) that read fails
    too, and the report must still be written."""
    try:
        _summarise(report, recorder)
    except Exception as error:
        return error
    return None


def _train_and_predict(
    report: ProbeReport,
    run: Any,
    section: str | None,
    *,
    iters: int,
    predict_frames: int,
    hold: tuple[Mapping[str, Any] | None, tuple[int, int] | None] | None = None,
) -> bool:
    """The miniature run; ``False`` when it failed (the report then says where)."""
    from grasppanda.engine.builders import build_dataset
    from grasppanda.engine.errors import NonFiniteLossError
    from grasppanda.engine.trainer import Trainer

    cfg, device = run.cfg, run.device
    recorder = _Recorder(iters, device)
    try:
        trainer = Trainer(run)
    except Exception as error:
        _fail(report, error, "build")
        return False
    if hold is not None:
        try:
            _hold_peak(report, trainer.state, *hold)
        except Exception as error:
            _fail(report, error, "optimizer", "the peak rate of the recipe's schedule")
            return False
    model = trainer.state.model
    report.parameters = sum(parameter.numel() for parameter in model.parameters())
    # The hooks the config lists are part of its recipe; loggers, checkpoints and
    # in-training evaluation are not what is probed.
    listed = tuple(HOOKS.get(node["type"]) for node in cfg["hooks"])
    trainer.hooks = [hook for hook in trainer.hooks if isinstance(hook, listed)] + [recorder]
    compute = trainer.compute_loss

    def timed(observation: Any, supervision: Any) -> Any:
        recorder.begin(observation)
        return compute(observation, supervision)

    trainer.compute_loss = timed
    try:
        trainer.train()
    except _Enough:
        pass
    except NonFiniteLossError as error:
        if not recorder.rows:  # the very first loss: nothing was trained, so nothing "ran"
            _summarised(report, recorder)
            _fail(report, error, "loss", "iteration 1")
            return False
        report.nonfinite.append(" ".join(str(error).split()))
    except Exception as error:
        _summarised(report, recorder)
        _fail(report, error, "loss", f"iteration {len(recorder.rows) + 1}")
        return False
    failed = _summarised(report, recorder)
    if failed is not None:  # a device fault surfaces late, often in the first read after it
        _fail(report, failed, "summary", "reading the model after training")
        return False
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        report.nonfinite.append(f"a weight is not finite after {report.iterations} iterations")
    try:
        model.eval()
        dataset = build_dataset(cfg, section or "train")
        predicted: list[Any] = []
        for index in range(min(predict_frames, len(dataset))):
            batch = dataset.collate([dataset[index]]).to(device)
            with torch.no_grad():
                groups = model.predict(batch.observation)
            _count_grasps(report, batch.observation.keys, groups)
            predicted.append((batch.observation, list(groups)))
    except Exception as error:
        _fail(report, error, "predict")
        return False
    try:
        _post_process(report, cfg, predicted, device)
    except Exception as error:
        _fail(report, error, "predict", "post-processing the predictions")
        return False
    return True


def _post_process(report: ProbeReport, cfg: Any, predicted: list[Any], device: Any) -> None:
    """Apply the evaluation variant ``post`` of the config, step by step, to the
    predictions of every probed frame, and record what each step left and took.

    The grasps of a probe come from a detector trained for a few iterations: the
    record shows that every step runs on real frames and how many grasps it passes
    there, not what it does to a trained detector's grasps."""
    from grasppanda.registry import POSTPROCESSORS

    from .post import POST_VARIANT

    nodes = ((cfg.get("evaluation") or {}).get("variants") or {}).get(POST_VARIANT)
    if not nodes:
        return
    steps = [POSTPROCESSORS.build(node) for node in nodes]
    record: dict[str, Any] = {
        "steps": [type(step).__name__ for step in steps],
        "grasps": [],
        "seconds": [],
    }
    for observation, groups in predicted:
        for key, grasps in zip(observation.keys, groups):
            grasps.key = grasps.key or key
            sizes, times = [len(grasps)], []
            for step in steps:
                started = time.perf_counter()
                grasps = step(grasps, observation)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(round(time.perf_counter() - started, 4))
                sizes.append(len(grasps))
            record["grasps"].append(sizes)
            record["seconds"].append(times)
    report.postprocessing = record


def _count_grasps(report: ProbeReport, keys: Sequence[Any], groups: Any) -> None:
    """Record the predictions of a batch: one grasp group per frame, finite.

    Whether a rotation is orthonormal enough is the detector's own contract (it validates
    its output with the tolerance its parameterisation needs); the probe checks what any
    consumer needs, the type and finiteness.
    """
    from grasppanda.structures import ContractError, GraspGroup

    groups = list(groups)
    if len(groups) != len(keys) or not all(isinstance(group, GraspGroup) for group in groups):
        kinds = sorted({type(group).__name__ for group in groups})
        raise ContractError(
            f"predict() returned {len(groups)} item(s) of type {kinds} for {len(keys)} frame(s); "
            "it must return one GraspGroup per frame"
        )
    for key, group in zip(keys, groups):
        report.predicted_frames.append(str(key))
        report.grasps_per_frame.append(len(group))
        for name in ("translation", "rotation", "width", "depth", "score"):
            if not bool(torch.isfinite(getattr(group, name)).all()):
                report.nonfinite.append(f"predicted {name} of {key} is not finite")
        scores = group.score.detach().reshape(-1).float()
        scores = scores[torch.isfinite(scores)].cpu()
        report.score_quantiles.append(
            [
                _digits(float(scores.min())),
                _digits(float(torch.quantile(scores, 0.5))),
                _digits(float(scores.max())),
            ]
            if scores.numel()
            else None
        )


def _judge(report: ProbeReport, good: str, what: str, predict_frames: int = 1) -> None:
    """Turn the recorded run into a verdict; ``good`` is the verdict of a clean run.

    ``predict_frames`` is how many frames were to be predicted: a run that trained or
    predicted less than it was asked to is not a clean one. A clean run on real frames
    that predicted no grasp on any frame is ``runs-empty``. What judge 2 adds beyond that
    (trends, units without a gradient, unchanged parts, scores) is reported, not judged.
    """
    unstable = list(report.nonfinite)
    empty = [index + 1 for index, value in enumerate(report.real_seeds) if value == 0.0]
    if empty:
        listed = ", ".join(map(str, empty[:6])) + (" ..." if len(empty) > 6 else "")
        unstable.append(
            f"{len(empty)} of {report.iterations} iterations had no real seed (iteration {listed})"
        )
    if report.supervised_seeds and max(report.supervised_seeds) == 0.0:
        unstable.append("no iteration supervised a seed")
    dead = dead_parts(report.no_gradient)
    if dead:
        unstable.append(
            f"no gradient reached {', '.join(dead)} in any iteration (a part "
            "before it produced nothing to learn from, or its output is constant)"
        )
    short = report.iterations < max(report.iterations_requested, 1)
    if short and not report.nonfinite:
        unstable.append(f"only {report.iterations} of {report.iterations_requested} iterations ran")
    if len(report.grasps_per_frame) < max(predict_frames, 1):
        unstable.append(
            f"only {len(report.grasps_per_frame)} of {predict_frames} frame(s) could be "
            "predicted (the dataset has no more)"
        )
    done = (
        f"trained {report.iterations} iterations on {what} and predicted on "
        f"{len(report.grasps_per_frame)} frame(s)"
    )
    nothing = bool(report.grasps_per_frame) and not any(report.grasps_per_frame)
    seeds = " and real seeds in every iteration" if report.real_seeds else ""
    if unstable:
        report.verdict = "runs-unstable" if good == "runs" else "fails"
        report.detail = f"{done}, but " + "; ".join(unstable) + "."
    elif nothing and good == "runs":
        report.verdict = "runs-empty"
        report.detail = (
            f"{done}, with finite losses{seeds}, but no grasp was predicted on any probed "
            "frame: every stage ran, and the model kept no seed."
        )
        return
    else:
        report.verdict = good
        report.detail = f"{done}, with finite losses{seeds}."
    if nothing:
        report.detail += " No grasp was predicted on any probed frame."


# ---------------------------------------------------------------------------- the probe


def probe(
    config: str | Path,
    overrides: Sequence[str] = (),
    *,
    iters: int = 20,
    frames: int = 8,
    predict_frames: int = 2,
    split: str = "sample",
    device: str | None = None,
    workers: int = 0,
    synthetic: bool = False,
    schedule: str = "recipe",
) -> ProbeReport:
    """Probe ``config``: build it, train it briefly on real frames, predict.

    Args:
        config: A config file or the name of a shipped config.
        overrides: ``key.path=value`` overrides, as for ``grasppanda train``.
        iters: Training iterations to run.
        frames: At most this many frames of ``split`` are trained on (every k-th frame).
        predict_frames: Frames to predict on afterwards.
        split: The split the frames are taken from; ``sample`` is the few frames every
            dataset reader ships for this purpose.
        device: Overrides ``runtime.device`` of the config.
        workers: Loader processes; 0 loads the frames in the probing process.
        synthetic: Skip real frames and run the synthetic wiring check.
        schedule: ``recipe`` (the config's scheduler over the probe's epochs) or
            ``constant`` (the recipe's peak rate at every step); see :data:`SCHEDULES`.
            The config hash is that of the config as given either way.

    Returns:
        The :class:`ProbeReport`. It never raises for a problem of the config: a refused
        config, a missing asset and a crash are all verdicts.

    Raises:
        ProbeRequestError: The request cannot be run: a count below one, a split the
            dataset does not have, or a schedule that is not one of :data:`SCHEDULES`.
    """
    for name, value in (("iters", iters), ("frames", frames), ("predict_frames", predict_frames)):
        if value < 1:
            raise ProbeRequestError(f"{name} must be at least 1, got {value}")
    if workers < 0:
        raise ProbeRequestError(f"workers must be 0 or more, got {workers}")
    if schedule not in SCHEDULES:
        raise ProbeRequestError(f"schedule must be one of {', '.join(SCHEDULES)}, got '{schedule}'")
    started = time.perf_counter()
    report = ProbeReport(
        config=str(config),
        overrides=list(overrides),
        iterations_requested=iters,
        schedule=schedule,
    )
    try:
        # The wiring check needs a model only; a real probe everything `train` needs.
        given = Config.load(config, overrides, command=None if synthetic else "train")
    except ConfigError as error:
        report.verdict = "refused"
        report.refusal = [f"{issue.path or '<file>'}: {issue.message}" for issue in error.issues]
        report.detail = f"the config check reports {len(error.issues)} problem(s)."
        report.seconds = round(time.perf_counter() - started, 1)
        return report
    report.config_hash = given.digest()
    report.detector = str(given["model"]["type"])
    if not synthetic:
        try:
            planned, added, section, notes, batches = _plan(
                given,
                config,
                overrides,
                split=split,
                frames=frames,
                predict_frames=predict_frames,
                iters=iters,
                workers=workers,
                schedule=schedule,
            )
        except _NoRealFrames as reason:
            report.fallback = str(reason)
        except ProbeRequestError:
            raise
        except Exception as error:
            _fail(report, error, "data")
            report.seconds = round(time.perf_counter() - started, 1)
            return report
        else:
            report.probe_overrides, report.notes = added, notes
            _real(
                report,
                planned,
                section,
                iters=iters,
                predict_frames=predict_frames,
                device=device,
                hold=(given.get("scheduler"), batches) if schedule == "constant" else None,
            )
            report.seconds = round(time.perf_counter() - started, 1)
            return report
    from .synthetic import wiring_check

    wiring_check(report, given, iters=min(iters, 2), device=device)
    report.seconds = round(time.perf_counter() - started, 1)
    return report
