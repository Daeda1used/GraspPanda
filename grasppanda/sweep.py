"""Sweeps: a grid of config overrides, validated as a whole before anything runs.

A sweep file names a base config and a few axes; each axis offers named alternatives,
each alternative being a list of overrides. The plan is the product of the axes times
the seeds. Every cell is a complete, validated config, so an incompatible combination
is reported at planning time rather than hours into the sweep.

    base: finegrasp/finegrasp_sunet14_g1b-realsense_10e
    seeds: [0, 1, 2]
    axes:
      backbone:
        sunet14: []
        ptv3: ["model.backbone={type: PTv3}"]
      seeds_per_frame:
        "1024": []
        "2048": ["model.proposal.num_seeds=2048"]

An alternative may splice in others by YAML anchor, a list inside its list
(``both: [*ptv3, *more_seeds]``): the way to write "these changes together" once.

``base`` may also name several configs, each under a short name. They become the first
axis, ``config``: the way to compare configs that ``grasppanda compose`` wrote, with the
same seeds, side by side.

    base:
      shipped: scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e
      ptv3: sbg_ptv3.yaml          # written by: grasppanda compose <the shipped config>
                                   #     --set backbone=ptv3 -o sbg_ptv3.yaml
    seeds: [0, 1, 2]
    overrides: ["runtime.gpu_memory_fraction=0.45"]   # for every cell, before the axes

The cells of a plan are independent runs. :func:`run_cells` runs them one after the other
in this process, or, given GPUs, one process per cell with one cell per GPU at a time.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grasppanda.config import Config, ConfigError, Issue, parse_yaml, resolve_config_path

__all__ = [
    "Cell",
    "Sweep",
    "aggregate",
    "cell_record",
    "load_sweep",
    "render_table",
    "run_cell",
    "run_cells",
]

#: Written into a cell's work directory when it has finished; makes a sweep resumable.
CELL_RECORD = "sweep_cell.json"
#: The console output of a cell that ran in a process of its own, in its work directory.
CELL_LOG = "sweep_cell.log"


@dataclass
class Cell:
    """One run of a sweep."""

    name: str
    choices: dict[str, str]
    seed: int
    overrides: list[str]
    config: Config | None = None
    error: ConfigError | None = None
    #: The config this cell overrides: the sweep's base, or one of its bases.
    base: Path | None = None

    @property
    def valid(self) -> bool:
        return self.config is not None


#: The axis that several base configs form.
CONFIG_AXIS = "config"


@dataclass
class Sweep:
    name: str
    base: Path
    axes: dict[str, dict[str, list[str]]]
    seeds: list[int]
    path: Path | None = None
    #: Several base configs by name, the axis ``config``; empty for a sweep with one base.
    bases: dict[str, Path] = field(default_factory=dict)
    #: Overrides of every cell, applied before those of its alternatives.
    overrides: list[str] = field(default_factory=list)
    #: Those of ``overrides`` that came from the command line (``--opts``), not the file:
    #: a cell that runs in a process of its own is started with them again.
    extra: list[str] = field(default_factory=list)

    def variants(self) -> list[dict[str, str]]:
        """Every combination of one alternative per axis, in file order; with several base
        configs, their names lead as the axis ``config``."""
        names = list(self.axes)
        products = [
            dict(zip(names, combo))
            for combo in itertools.product(*(self.axes[name] for name in names))
        ]
        if not self.bases:
            return products
        return [{CONFIG_AXIS: base, **choices} for base in self.bases for choices in products]

    def plan(self, command: str | None = None) -> list[Cell]:
        """Build and validate the config of every cell.

        ``command`` additionally requires the sections that command needs (``"train"``).
        Invalid cells are returned with their error instead of raising, so the caller
        can show every problem of the grid at once.
        """
        cells = []
        for choices in self.variants():
            label = "_".join(choices.values()) or "base"
            for seed in self.seeds:
                overrides = [
                    *self.overrides,
                    *(
                        item
                        for axis, pick in choices.items()
                        if axis in self.axes
                        for item in self.axes[axis][pick]
                    ),
                ]
                overrides += [
                    f"runtime.seed={seed}",
                    f"runtime.work_dir=work_dirs/{self.name}/{label}/seed{seed}",
                ]
                base = self.bases[choices[CONFIG_AXIS]] if self.bases else self.base
                cell = Cell(f"{label}/seed{seed}", choices, seed, overrides, base=base)
                try:
                    cell.config = Config.load(base, overrides, command=command)
                except ConfigError as error:
                    cell.error = error
                cells.append(cell)
        return cells


def _fail(path: Path, where: str, message: str) -> ConfigError:
    return ConfigError([Issue(where, message)], str(path))


def load_sweep(path: str | Path, extra: Sequence[str] = ()) -> Sweep:
    """Read a sweep file. Paths in ``base`` are relative to the sweep file.

    ``extra`` overrides are added to those of the file for every cell (``--opts`` of the
    command).
    """
    path = Path(path)
    raw: Any = parse_yaml(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise _fail(path, "", "a sweep file must be a mapping")
    unknown = set(raw) - {"base", "seeds", "axes", "overrides"}
    if unknown:
        raise _fail(
            path,
            sorted(unknown)[0],
            "unknown key; a sweep file has `base`, `seeds`, `axes` and `overrides`",
        )
    shared = raw.get("overrides") or []
    if not isinstance(shared, list) or not all(isinstance(item, str) for item in shared):
        raise _fail(path, "overrides", 'must be a list of "key.path=value" strings')

    def resolve(name: str) -> Path:
        local = path.parent / name
        return local if local.is_file() else resolve_config_path(name)

    given = raw.get("base")
    bases: dict[str, Path] = {}
    if isinstance(given, dict) and given:
        for label, name in given.items():
            if not isinstance(name, str):
                raise _fail(path, f"base.{label}", "must be a config file or a shipped config")
            bases[str(label)] = resolve(name)
        base = next(iter(bases.values()))
    elif isinstance(given, str):
        base = resolve(given)
    else:
        raise _fail(
            path,
            "base",
            "is required: a config file or the name of a shipped config, or several of them "
            "by name (`base: {a: one.yaml, b: two.yaml}`)",
        )
    axes = raw.get("axes") or {}
    if not isinstance(axes, dict):
        raise _fail(path, "axes", "must map axis names to alternatives")
    if bases and CONFIG_AXIS in axes:
        raise _fail(
            path, f"axes.{CONFIG_AXIS}", "is the axis the several `base` configs form: rename it"
        )
    for axis, alternatives in axes.items():
        if not isinstance(alternatives, dict) or not alternatives:
            raise _fail(path, f"axes.{axis}", "must map alternative names to lists of overrides")
        for name, overrides in list(alternatives.items()):
            if overrides is None:
                overrides = []
            if isinstance(overrides, list):  # a list in the list: another alternative, by anchor
                overrides = [
                    item
                    for part in overrides
                    for item in (part if isinstance(part, list) else [part])
                ]
            alternatives[name] = overrides
            if not isinstance(overrides, list) or not all(
                isinstance(item, str) for item in overrides
            ):
                raise _fail(
                    path, f"axes.{axis}.{name}", 'must be a list of "key.path=value" strings'
                )
        axes[axis] = {str(name): overrides for name, overrides in alternatives.items()}
    seeds = raw.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        raise _fail(path, "seeds", "must be a non-empty list of integers")
    return Sweep(
        name=path.stem,
        base=base,
        axes=axes,
        seeds=seeds,
        path=path,
        bases=bases,
        overrides=[*shared, *extra],
        extra=list(extra),
    )


# ------------------------------------------------------------------------ execution


def cell_record(cell: Cell) -> dict[str, Any] | None:
    """The record of a finished cell, or ``None`` when the cell has to run.

    A record written for another config hash (the sweep file changed) does not count, nor
    does a file that cannot be read. The choices come back in the order of the plan, not in
    the sorted order of the file, so that records read and records just made tabulate alike.
    """
    if cell.config is None:
        raise ValueError(f"cell {cell.name} has no valid config")
    path = Path(cell.config.select("runtime.work_dir")) / CELL_RECORD
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("config_hash") != cell.config.digest():
        return None
    record["choices"] = dict(cell.choices)
    return record


def run_cell(sweep: Sweep, cell: Cell) -> dict[str, Any]:
    """Train one cell, score it on the test split when one is configured, record the outcome.

    A cell whose record exists for the same config hash is not run again.
    """
    from grasppanda.engine import commands
    from grasppanda.evaluation.records import in_percent

    finished = cell_record(cell)
    if finished is not None:
        return finished
    assert cell.config is not None
    work_dir = Path(cell.config.select("runtime.work_dir"))
    record_path = work_dir / CELL_RECORD
    digest = cell.config.digest()
    state = commands.train(cell.config)
    record: dict[str, Any] = {
        "cell": cell.name,
        "choices": cell.choices,
        "seed": cell.seed,
        "config_hash": digest,
        "run": str(state.run.dir),
        "validation": dict(state.metrics),
        "test": None,
    }
    resolved = cell.config.resolved()
    checkpoints = sorted((state.run.dir / "checkpoints").glob("epoch_*.pth"))
    has_test = (resolved.get("data") or {}).get("test") and resolved["evaluation"]["evaluator"]
    if has_test and checkpoints:
        overrides = [
            *cell.overrides,
            f"runtime.load_from={json.dumps(str(checkpoints[-1]))}",
            "evaluation.split=test",
        ]
        evaluation = commands.test(Config.load(cell.base or sweep.base, overrides, command="test"))
        record["test"] = dict(evaluation.metrics)
        record["partial"] = list(evaluation.partial)
        # the evaluator states its metrics as fractions: tables print them in percent
        scored = list(evaluation.records.values())
        record["percent"] = bool(scored) and all(in_percent(entry) for entry in scored)
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary = record_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(record_path)
    return record


def _device_mask(gpu: str) -> str:
    """What a cell's ``CUDA_VISIBLE_DEVICES`` has to say for the GPU this process calls ``gpu``.

    A process that is itself restricted to some GPUs numbers them from 0: its GPU 1 is the
    second entry of its own mask, not the machine's GPU 1.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ or not gpu.isdigit():
        return gpu
    mask = [part.strip() for part in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    mask = [part for part in mask if part]
    if int(gpu) >= len(mask):
        raise ValueError(
            f"GPU {gpu} is not one of the {len(mask)} that CUDA_VISIBLE_DEVICES leaves to "
            "this process"
        )
    return mask[int(gpu)]


def _failed(cell: Cell, why: str) -> dict[str, Any]:
    return {"cell": cell.name, "choices": dict(cell.choices), "seed": cell.seed, "failed": why}


#: The variable that hands a cell the module search path of the sweep that started it.
CELL_PATH = "GRASPPANDA_SWEEP_PATH"
#: What a cell's interpreter runs. ``python -m grasppanda.cli.main`` would search the
#: working directory and the interpreter's own packages first, and so could run another
#: GraspPanda than the sweep's: a checkout the sweep was started in, a release installed
#: beside the source tree this process runs from.
_CELL = (
    f"import json, os, sys; sys.path[:] = json.loads(os.environ.pop({CELL_PATH!r})); "
    "from grasppanda.cli.main import main; sys.exit(main(sys.argv[1:]))"
)


def _cell_environment(gpu: str) -> dict[str, str]:
    """The environment of a cell's process: one GPU, and the modules of this process.

    ``gpu`` is nvidia-smi's number where this process has no mask of its own. CUDA numbers
    the fastest GPU first unless told otherwise, which differs from nvidia-smi on a machine
    with two kinds of GPU; a process that set its own mask or order keeps its numbering.
    """
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": _device_mask(gpu)}
    if not {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"} & set(os.environ):
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment[CELL_PATH] = json.dumps(sys.path)
    return environment


class _Cells:
    """The processes of a running sweep: started, and stopped together.

    Starting a process and noting it happen under one lock, as does stopping, so no cell
    starts unnoticed while the sweep is being stopped.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: set[subprocess.Popen[bytes]] = set()
        self.stopped = False

    def start(self, command: Sequence[str], **options: Any) -> subprocess.Popen[bytes] | None:
        """Start a process, or return ``None`` when the sweep was stopped."""
        with self._lock:
            if self.stopped:
                return None
            process = subprocess.Popen(command, **options)
            self._running.add(process)
            return process

    def finished(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._running.discard(process)

    def stop(self) -> None:
        """Terminate every running process; none starts afterwards."""
        with self._lock:
            self.stopped = True
            for process in self._running:
                process.terminate()


NOT_STARTED = "not started: the sweep was stopped"


def _run_in_process(
    sweep: Sweep, cell: Cell, gpu: str, cells: _Cells | None = None
) -> dict[str, Any]:
    """Run one cell in a process of its own that sees only ``gpu``; its output goes to a log.

    ``cells`` starts the process for a caller that may have to stop it.
    """
    assert cell.config is not None and sweep.path is not None
    cells = cells if cells is not None else _Cells()
    work_dir = Path(cell.config.select("runtime.work_dir"))
    work_dir.mkdir(parents=True, exist_ok=True)
    log = work_dir / CELL_LOG
    command = [sys.executable, "-c", _CELL, "sweep", str(sweep.path), "--cell", cell.name]
    if sweep.extra:
        command += ["--opts", *sweep.extra]
    with open(log, "w", encoding="utf-8") as handle:
        process = cells.start(
            command, env=_cell_environment(gpu), stdout=handle, stderr=subprocess.STDOUT
        )
        if process is None:
            return _failed(cell, NOT_STARTED)
        try:
            status = process.wait()
        finally:
            cells.finished(process)
    record = cell_record(cell) if status == 0 else None
    return record if record is not None else _failed(cell, str(log))


def run_cells(
    sweep: Sweep,
    cells: Sequence[Cell],
    *,
    gpus: Sequence[str] = (),
    report: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run every cell of a plan and return their records, in plan order.

    Without ``gpus`` the cells run one after the other in this process. With ``gpus``
    every cell runs in a process of its own that sees one GPU (``CUDA_VISIBLE_DEVICES``),
    and as many cells run at a time as GPUs were named: four runs on four GPUs take the
    time of the longest, not the sum. A GPU named twice takes two cells at a time. The ids
    are those of this process: ``nvidia-smi``'s, or positions in its own
    ``CUDA_VISIBLE_DEVICES`` when that is set. The console output of such a cell is in
    ``<its work directory>/sweep_cell.log``.

    A cell that fails does not stop the others, with or without ``gpus``. Its record is
    ``{"cell", "choices", "seed", "failed": <the log, or the error>}`` and carries no
    metric, so its variant reports none; running the sweep again runs the failed cells
    only. When this process is interrupted or terminated, the cells it started are
    terminated with it.

    Raises:
        ValueError: ``gpus`` names a GPU outside this process's own
            ``CUDA_VISIBLE_DEVICES``, or the sweep has no file to start cells from.
    """
    total = len(cells)
    if not gpus:
        records = []
        for index, cell in enumerate(cells, start=1):
            report(f"\n[{index}/{total}] {cell.name}")
            try:
                records.append(run_cell(sweep, cell))
            except Exception as error:  # the next cell may well run
                traceback.print_exc()
                report(f"[{index}/{total}] {cell.name}: FAILED, {type(error).__name__}: {error}")
                records.append(_failed(cell, f"{type(error).__name__}: {error}"))
        return records
    if sweep.path is None:
        raise ValueError("cells run in processes of their own from the sweep's file: it has none")
    for gpu in gpus:
        _device_mask(gpu)  # refused before anything starts

    free: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        free.put(gpu)
    running = _Cells()

    def work(item: tuple[int, Cell]) -> dict[str, Any]:
        index, cell = item
        try:
            finished = cell_record(cell)
            if finished is not None:
                report(f"[{index}/{total}] {cell.name}: already recorded")
                return finished
            gpu = free.get()
            try:
                if running.stopped:
                    return _failed(cell, NOT_STARTED)
                report(f"[{index}/{total}] {cell.name}: started on GPU {gpu}")
                started = time.perf_counter()
                record = _run_in_process(sweep, cell, gpu, running)
                if record.get("failed") == NOT_STARTED:
                    return record
                took = time.perf_counter() - started
                outcome = f"FAILED, see {record['failed']}" if "failed" in record else "done"
                report(f"[{index}/{total}] {cell.name}: {outcome} on GPU {gpu} in {took:.0f} s")
                return record
            finally:
                free.put(gpu)
        except Exception as error:  # one cell's trouble (its log, its record) is its own
            report(f"[{index}/{total}] {cell.name}: FAILED, {type(error).__name__}: {error}")
            return _failed(cell, f"{type(error).__name__}: {error}")

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    # A terminated parent would leave its cells training on the GPUs: stop them with it.
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, terminated) if main else None
    try:
        with ThreadPoolExecutor(len(gpus)) as pool:
            try:
                return list(pool.map(work, enumerate(cells, start=1)))
            except BaseException:
                running.stop()
                raise
    finally:
        if main:
            signal.signal(signal.SIGTERM, previous)


def aggregate(records: Sequence[dict[str, Any]], *, source: str = "test") -> list[dict[str, Any]]:
    """Mean and sample standard deviation over seeds, per variant of the grid.

    ``source`` is ``"test"`` or ``"validation"``. A variant whose runs miss a metric
    (for example because a run failed or was only partially scored) does not report it: a
    mean over fewer runs than seeds would not be comparable with its neighbours. A row is
    marked ``percent`` when every run of it that has metrics states them as fractions.
    """
    groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    shown: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for record in records:
        key = tuple(sorted(record["choices"].items()))  # the order of the keys is no choice
        groups.setdefault(key, []).append(record)
        shown.setdefault(key, dict(record["choices"]))
    rows = []
    for key, members in groups.items():
        measured = [member.get(source) or {} for member in members]
        names = sorted(set.intersection(*(set(m) for m in measured))) if measured else []
        metrics = {}
        for name in names:
            values = [float(m[name]) for m in measured]
            spread = statistics.stdev(values) if len(values) > 1 else None
            metrics[name] = {"mean": statistics.fmean(values), "std": spread, "n": len(values)}
        with_metrics = [member for member in members if member.get(source)]
        rows.append(
            {
                "choices": shown[key],
                "seeds": len(members),
                "metrics": metrics,
                "percent": bool(with_metrics) and all(m.get("percent") for m in with_metrics),
            }
        )
    return rows


def render_table(rows: Sequence[dict[str, Any]], *, fmt: str = "markdown") -> str:
    """Render aggregated rows as a Markdown table or as CSV.

    Metrics that every row states as fractions are printed in percent (68.75, as
    ``grasppanda test`` prints them), others as they were reported.
    """
    if not rows:
        return ""
    axes = list(rows[0]["choices"])
    names = sorted({name for row in rows for name in row["metrics"]})
    measured = [row for row in rows if row["metrics"]]
    scale = 100.0 if measured and all(row.get("percent") for row in measured) else 1.0

    def cell(row: dict[str, Any], name: str) -> str:
        stat = row["metrics"].get(name)
        if stat is None:
            return "—" if fmt == "markdown" else ""
        if stat["std"] is None or fmt != "markdown":
            return f"{scale * stat['mean']:.2f}"
        return f"{scale * stat['mean']:.2f} ± {scale * stat['std']:.2f}"

    header = [*axes, "seeds", *names]
    body = [
        [*(row["choices"][a] for a in axes), str(row["seeds"]), *(cell(row, n) for n in names)]
        for row in rows
    ]
    if fmt == "csv":
        extra = [f"{name}_std" for name in names]
        lines = [",".join([*header, *extra])]
        for row, line in zip(rows, body):
            spreads = [
                ""
                if (s := row["metrics"].get(n)) is None or s["std"] is None
                else f"{scale * s['std']:.4f}"
                for n in names
            ]
            lines.append(",".join([*line, *spreads]))
        return "\n".join(lines) + "\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(line) + " |" for line in body]
    return "\n".join(lines) + "\n"
