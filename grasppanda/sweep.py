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

The cells of a plan are independent runs. :func:`run_cells` runs them one after the other
in this process, or, given GPUs, one process per cell with one cell per GPU at a time.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

    @property
    def valid(self) -> bool:
        return self.config is not None


@dataclass
class Sweep:
    name: str
    base: Path
    axes: dict[str, dict[str, list[str]]]
    seeds: list[int]
    path: Path | None = None

    def variants(self) -> list[dict[str, str]]:
        """Every combination of one alternative per axis, in file order."""
        names = list(self.axes)
        return [
            dict(zip(names, combo))
            for combo in itertools.product(*(self.axes[name] for name in names))
        ]

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
                    item for axis, pick in choices.items() for item in self.axes[axis][pick]
                ]
                overrides += [
                    f"runtime.seed={seed}",
                    f"runtime.work_dir=work_dirs/{self.name}/{label}/seed{seed}",
                ]
                cell = Cell(f"{label}/seed{seed}", choices, seed, overrides)
                try:
                    cell.config = Config.load(self.base, overrides, command=command)
                except ConfigError as error:
                    cell.error = error
                cells.append(cell)
        return cells


def _fail(path: Path, where: str, message: str) -> ConfigError:
    return ConfigError([Issue(where, message)], str(path))


def load_sweep(path: str | Path) -> Sweep:
    """Read a sweep file. Paths in ``base`` are relative to the sweep file."""
    path = Path(path)
    raw: Any = parse_yaml(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise _fail(path, "", "a sweep file must be a mapping")
    unknown = set(raw) - {"base", "seeds", "axes"}
    if unknown:
        raise _fail(
            path, sorted(unknown)[0], "unknown key; a sweep file has `base`, `seeds` and `axes`"
        )
    if not isinstance(raw.get("base"), str):
        raise _fail(path, "base", "is required: a config file or the name of a shipped config")
    local = path.parent / raw["base"]
    base = local if local.is_file() else resolve_config_path(raw["base"])
    axes = raw.get("axes") or {}
    if not isinstance(axes, dict):
        raise _fail(path, "axes", "must map axis names to alternatives")
    for axis, alternatives in axes.items():
        if not isinstance(alternatives, dict) or not alternatives:
            raise _fail(path, f"axes.{axis}", "must map alternative names to lists of overrides")
        for name, overrides in list(alternatives.items()):
            if overrides is None:
                alternatives[name] = overrides = []
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
    return Sweep(name=path.stem, base=base, axes=axes, seeds=seeds, path=path)


# ------------------------------------------------------------------------ execution


def cell_record(cell: Cell) -> dict[str, Any] | None:
    """The record of a finished cell, or ``None`` when the cell has to run.

    A record written for another config hash (the sweep file changed) does not count.
    """
    if cell.config is None:
        raise ValueError(f"cell {cell.name} has no valid config")
    path = Path(cell.config.select("runtime.work_dir")) / CELL_RECORD
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record if record.get("config_hash") == cell.config.digest() else None


def run_cell(sweep: Sweep, cell: Cell) -> dict[str, Any]:
    """Train one cell, score it on the test split when one is configured, record the outcome.

    A cell whose record exists for the same config hash is not run again.
    """
    from grasppanda.engine import commands

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
        evaluation = commands.test(Config.load(sweep.base, overrides, command="test"))
        record["test"] = dict(evaluation.metrics)
        record["partial"] = list(evaluation.partial)
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary = record_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(record_path)
    return record


def _run_in_process(sweep: Sweep, cell: Cell, gpu: str) -> dict[str, Any]:
    """Run one cell in a process of its own that sees only ``gpu``; its output goes to a log."""
    assert cell.config is not None and sweep.path is not None
    work_dir = Path(cell.config.select("runtime.work_dir"))
    work_dir.mkdir(parents=True, exist_ok=True)
    log = work_dir / CELL_LOG
    command = [sys.executable, "-m", "grasppanda.cli.main", "sweep", str(sweep.path)]
    command += ["--cell", cell.name]
    with open(log, "w", encoding="utf-8") as handle:
        status = subprocess.run(
            command,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    record = cell_record(cell) if status == 0 else None
    if record is None:
        return {"cell": cell.name, "choices": cell.choices, "seed": cell.seed, "failed": str(log)}
    return record


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
    time of the longest, not the sum. A GPU named twice takes two cells at a time. The
    console output of such a cell is in ``<its work directory>/sweep_cell.log``.

    A cell that fails does not stop the others. Its record is
    ``{"cell", "choices", "seed", "failed": <log>}`` and carries no metric, so its variant
    reports none; running the sweep again runs the failed cells only.
    """
    total = len(cells)
    if not gpus:
        records = []
        for index, cell in enumerate(cells, start=1):
            report(f"\n[{index}/{total}] {cell.name}")
            records.append(run_cell(sweep, cell))
        return records

    free: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def work(item: tuple[int, Cell]) -> dict[str, Any]:
        index, cell = item
        finished = cell_record(cell)
        if finished is not None:
            report(f"[{index}/{total}] {cell.name}: already recorded")
            return finished
        gpu = free.get()
        try:
            report(f"[{index}/{total}] {cell.name}: started on GPU {gpu}")
            started = time.perf_counter()
            record = _run_in_process(sweep, cell, gpu)
            took = time.perf_counter() - started
            outcome = f"FAILED, see {record['failed']}" if "failed" in record else "done"
            report(f"[{index}/{total}] {cell.name}: {outcome} on GPU {gpu} in {took:.0f} s")
            return record
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(len(gpus)) as pool:
        return list(pool.map(work, enumerate(cells, start=1)))


def aggregate(records: Sequence[dict[str, Any]], *, source: str = "test") -> list[dict[str, Any]]:
    """Mean and sample standard deviation over seeds, per variant of the grid.

    ``source`` is ``"test"`` or ``"validation"``. A variant whose runs miss a metric
    (for example because a run was only partially scored) does not report it: a mean
    over fewer runs than seeds would not be comparable with its neighbours.
    """
    groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record["choices"].items()), []).append(record)
    rows = []
    for choices, members in groups.items():
        measured = [member.get(source) or {} for member in members]
        names = sorted(set.intersection(*(set(m) for m in measured))) if measured else []
        metrics = {}
        for name in names:
            values = [float(m[name]) for m in measured]
            spread = statistics.stdev(values) if len(values) > 1 else None
            metrics[name] = {"mean": statistics.fmean(values), "std": spread, "n": len(values)}
        rows.append({"choices": dict(choices), "seeds": len(members), "metrics": metrics})
    return rows


def render_table(rows: Sequence[dict[str, Any]], *, fmt: str = "markdown") -> str:
    """Render aggregated rows as a Markdown table or as CSV."""
    if not rows:
        return ""
    axes = list(rows[0]["choices"])
    names = sorted({name for row in rows for name in row["metrics"]})

    def cell(row: dict[str, Any], name: str) -> str:
        stat = row["metrics"].get(name)
        if stat is None:
            return "—" if fmt == "markdown" else ""
        if stat["std"] is None or fmt != "markdown":
            return f"{stat['mean']:.2f}"
        return f"{stat['mean']:.2f} ± {stat['std']:.2f}"

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
                else f"{s['std']:.4f}"
                for n in names
            ]
            lines.append(",".join([*line, *spreads]))
        return "\n".join(lines) + "\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(line) + " |" for line in body]
    return "\n".join(lines) + "\n"
