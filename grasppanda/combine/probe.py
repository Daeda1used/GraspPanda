"""Probe a config: does this combination really build, train and predict?

A probe is a miniature of ``grasppanda train`` followed by a few predictions. It runs the
real trainer on a handful of real frames (the split ``sample`` of a dataset exists for
this), with the config's own recipe: its optimizer, schedule, gradient accumulation and
clipping, and the hooks it lists. Nothing is written: no run directory, no checkpoint.

The outcome is one verdict of a fixed vocabulary, :data:`VERDICTS`, and the numbers it
rests on. The probe never reports more than it ran: without a dataset tree, or without
the label assets of the real frames, it says which and falls back to a wiring check on a
synthetic scene, whose best verdict is ``wired``, not ``runs``.

    report = probe("experimental/gsnet_ptv3_g1b-realsense_10e", iters=20, frames=8)
    report.verdict, report.detail
"""

from __future__ import annotations

import math
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from grasppanda.config import Config, ConfigError
from grasppanda.registry import HOOKS, all_registries

__all__ = ["EXIT_STATUS", "VERDICTS", "ProbeReport", "probe"]

#: The verdicts of a probe and what each one means. Exactly one is reported.
VERDICTS: dict[str, str] = {
    "runs": (
        "On real frames: the detector was built, every requested training iteration ran "
        "with finite losses and at least one real seed, the weights stayed finite, and "
        "prediction returned valid grasp groups for every probed frame."
    ),
    "runs-unstable": (
        "On real frames: training and prediction ran to the end, but a loss or a weight "
        "became non-finite, or an iteration had no real seed (no scene point passed the "
        "proposal, so only the proposal was trained), or no iteration supervised a seed."
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
EXIT_STATUS = {"runs": 0, "wired": 0, "runs-unstable": 1, "fails": 1, "refused": 2}
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
            short; with ``overrides`` they reproduce it through ``grasppanda train``.
        config_hash: Hash of the config as given (without the probe's overrides).
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


# ------------------------------------------------------------------------- recording


class _Recorder:
    """A training hook that keeps every iteration's numbers and stops after ``limit``."""

    priority = 95

    def __init__(self, limit: int, device: torch.device) -> None:
        self.limit, self.device = limit, device
        self.rows: list[dict[str, float]] = []
        self.compute: list[float] = []
        self.waiting: list[float] = []
        self.keys: list[list[str]] = []
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
        return
    for name in rows[0]:
        series = [row[name] for row in rows if name in row]
        pair = [series[0], series[-1]]
        if name == "loss" or name.startswith("loss/"):
            report.losses[name] = pair
        elif name.startswith("metric/"):
            report.metrics[name] = pair
    report.real_seeds = [row["metric/real_seeds"] for row in rows if "metric/real_seeds" in row]
    report.supervised_seeds = [
        row["metric/supervised_seeds"] for row in rows if "metric/supervised_seeds" in row
    ]
    steady = recorder.compute[1:] or recorder.compute
    report.seconds_per_iteration = statistics.median(steady)
    report.seconds_data = statistics.median(recorder.waiting[1:] or recorder.waiting)


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
) -> tuple[Config, list[str], str | None, list[str]]:
    """The config of the miniature run: the given one plus the probe's overrides.

    Returns the config, the overrides the probe added, the ``data`` section to predict
    on and notes about what had to change. Raises :class:`_NoRealFrames` when the frames
    or their label assets are not on this machine.

    Frames are selected through the options of the dataset reader: ``split`` names the
    few frames a reader ships for this purpose and ``frame_step`` thins them. A reader
    without those options is probed on the frames the config names.
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
    cfg = Config.load(config, [*overrides, *added], command="train").resolved()
    try:
        dataset = build_dataset(cfg, "train")
    except FileNotFoundError as error:
        raise _NoRealFrames(f"the dataset is not on this machine. {error}") from None
    except AssetError as error:
        hint = f" For the frames of the probe add `--split {split}` to that command."
        if "scale_statistic asset" in str(error):
            # A histogram over a whole split: the few frames of a probe leave bins empty,
            # and the generator refuses that. The reference's own file does not depend on
            # the split.
            hint = (
                " This width histogram is a statistic of a whole split and cannot be built "
                "from the few frames of a probe. Import the reference's file "
                "(configs/scale_balanced_grasp/README.md) and probe with `--opts "
                "data.train.dataset.scale_statistic.source=legacy_import "
                "data.train.dataset.scale_statistic.reference_sha256=<its SHA-256>`."
            )
        raise _NoRealFrames(
            f"a label asset of the frames is missing. {error}"
            + (hint if "split" in options else "")
        ) from None
    except ValueError as error:
        if "is not a split of" not in str(error):
            raise
        raise _NoRealFrames(f"{error} Choose the frames with `--split`.") from None
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
    section = next((name for name in ("val", "test") if (cfg["data"] or {}).get(name)), None)
    if section is not None:
        other = _dataset_options(given, section)
        if "split" in other and "split" in options:
            added.append(f"data.{section}.dataset.split={split}")
            if "frame_step" in other:
                keep = max(1, math.ceil(count / max(predict_frames, 1)))
                added.append(f"data.{section}.dataset.frame_step={keep}")
        added.append(f"data.{section}.num_workers=0")
    return Config.load(config, [*overrides, *added], command="train"), added, section, notes


def _real(report: ProbeReport, planned: Config, section: str | None, **options: Any) -> None:
    """Train ``iters`` iterations with the real trainer, then predict."""
    from grasppanda.engine.run import Run, resolve_device
    from grasppanda.engine.seeding import determinism_restored
    from grasppanda.ops import backends

    cfg = planned.resolved()
    device = resolve_device(options["device"] or cfg["runtime"]["device"])
    report.device, report.data = str(device), "real"
    backends.configure(cfg["runtime"]["ops"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(prefix="grasppanda-probe-") as scratch, determinism_restored():
        run = Run(Path(scratch), "probe", "train", planned, cfg, device)
        try:
            finished = _train_and_predict(report, run, section, **options)
        finally:
            if device.type == "cuda":
                report.peak_gpu_mib = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    if finished:
        _judge(report, "runs", f"{len(report.frames)} frames of the dataset")


def _train_and_predict(report: ProbeReport, run: Any, section: str | None, **options: Any) -> bool:
    """The miniature run; ``False`` when it failed (the report then says where)."""
    from grasppanda.engine.builders import build_dataset
    from grasppanda.engine.errors import NonFiniteLossError
    from grasppanda.engine.trainer import Trainer

    cfg, device = run.cfg, run.device
    recorder = _Recorder(options["iters"], device)
    try:
        trainer = Trainer(run)
    except Exception as error:
        _fail(report, error, "build")
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
            _summarise(report, recorder)
            _fail(report, error, "loss", "iteration 1")
            return False
        report.nonfinite.append(" ".join(str(error).split()))
    except Exception as error:
        _summarise(report, recorder)
        _fail(report, error, "loss", f"iteration {len(recorder.rows) + 1}")
        return False
    _summarise(report, recorder)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        report.nonfinite.append(f"a weight is not finite after {report.iterations} iterations")
    try:
        model.eval()
        dataset = build_dataset(cfg, section or "train")
        predicted: list[Any] = []
        for index in range(min(options["predict_frames"], len(dataset))):
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


def _judge(report: ProbeReport, good: str, what: str) -> None:
    """Turn the recorded run into a verdict; ``good`` is the verdict of a clean run."""
    unstable = list(report.nonfinite)
    empty = [index + 1 for index, value in enumerate(report.real_seeds) if value == 0.0]
    if empty:
        listed = ", ".join(map(str, empty[:6])) + (" ..." if len(empty) > 6 else "")
        unstable.append(
            f"{len(empty)} of {report.iterations} iterations had no real seed (iteration {listed})"
        )
    if report.supervised_seeds and max(report.supervised_seeds) == 0.0:
        unstable.append("no iteration supervised a seed")
    if report.iterations < report.iterations_requested and not report.nonfinite:
        unstable.append(f"only {report.iterations} of {report.iterations_requested} iterations ran")
    done = (
        f"trained {report.iterations} iterations on {what} and predicted on "
        f"{len(report.grasps_per_frame)} frame(s)"
    )
    if unstable:
        report.verdict = "runs-unstable" if good == "runs" else "fails"
        report.detail = f"{done}, but " + "; ".join(unstable) + "."
    else:
        report.verdict = good
        seeds = " and real seeds in every iteration" if report.real_seeds else ""
        report.detail = f"{done}, with finite losses{seeds}."


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

    Returns:
        The :class:`ProbeReport`. It never raises for a problem of the config: a refused
        config, a missing asset and a crash are all verdicts.
    """
    started = time.perf_counter()
    report = ProbeReport(config=str(config), overrides=list(overrides), iterations_requested=iters)
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
    options = {"iters": iters, "predict_frames": predict_frames, "device": device}
    if not synthetic:
        try:
            planned, added, section, notes = _plan(
                given,
                config,
                overrides,
                split=split,
                frames=frames,
                predict_frames=predict_frames,
                iters=iters,
                workers=workers,
            )
        except _NoRealFrames as reason:
            report.fallback = str(reason)
        except Exception as error:
            _fail(report, error, "data")
            report.seconds = round(time.perf_counter() - started, 1)
            return report
        else:
            report.probe_overrides, report.notes = added, notes
            _real(report, planned, section, **options)
            report.seconds = round(time.perf_counter() - started, 1)
            return report
    from .synthetic import wiring_check

    wiring_check(report, given, iters=min(iters, 2), device=device)
    report.seconds = round(time.perf_counter() - started, 1)
    return report
