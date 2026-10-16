"""Probe the grid of combinations and record what happened, cell by cell.

Every cell of :mod:`grasppanda.combine.grid` is composed with ``grasppanda compose`` and,
when the config check accepts it, probed with ``grasppanda probe``: each in a process of
its own, the commands recorded with the outcome. Results are appended to
``$GRASPPANDA_CACHE/probe_grid/<name>/results.jsonl``, so an interrupted run continues
where it stopped; the composed configs and the output of every probe lie beside it.

    python -m tools.probe_grid run --groups reference encoder stage family
    python -m tools.probe_grid run --groups stage-set          # the stage sets, later
    python -m tools.probe_grid status
    python -m tools.probe_grid export       # -> docs/advanced_guides/measured_combinations.json
    python -m tools.probe_grid refresh      # the file for a changed grid, without probing

Configs that read the width histogram of Scale-Balanced-Grasp need the imported
reference file (a probe cannot build a statistic of a whole split from eight frames):
pass its SHA-256 as ``--scale-statistic-sha <sha>`` to ``run``, or those cells end as
``wired`` and are not accepted as measurements.

The probes of a run train at the recipe's peak learning rate without its schedule
(``grasppanda probe --schedule constant``, the default of ``run``): run over the few
epochs of a probe, the schedules of the methods would decay or warm up the rate by very
different amounts, and the trends judge 2 reports would not compare across methods.
``--schedule recipe`` probes the recipes as shipped. A cell whose request sets the
schedule (a schedule preset, or an optimizer or recipe preset that sets one) is always
probed with it (:func:`cell_schedule`): there the schedule is what is measured. Every
record says which schedule it ran.

Two cells that compose to the same config (the same config hash) are one experiment: it
is probed once and both cells record it, naming the command that ran. At most ``--jobs``
probes run at a time. A probe that ran out of GPU memory says nothing about its config
on a shared machine: it is run again alone, ``--retries`` times with a pause, and if
memory is still short the cell is **not recorded**; the run then ends with status 1 and
a later ``run`` picks the cell up.

The export is the compact file the documentation page is generated from
(``python -m tools.make_combination_docs``): one line per cell with its verdict, its two
commands, the messages of a refusal or a failure and the numbers that do not depend on
machine load, plus timings and memory labelled as measured on a shared machine. A cell
that shares the probe of another one names it instead of repeating its numbers, and the
composed config is named as a file of the working directory, not by its place in the
cache of the machine that ran the grid.

The export merges: cells this run recorded replace or join those of the file as it is.
Every record carries the date, the commit and the environment of the probe that measured
it, and the file lists those runs; a record that reuses an earlier probe of the same
config keeps the run of that probe. A record of judge 2 (``judge: 2``) also carries the
trend of every loss term, the units without a gradient, the unchanged parts, the score
quantiles and the learning-rate decay (:data:`FIELDS`); the loss of every iteration goes
into a side file, ``measured_combinations/losses_per_iteration.json``, merged the same
way, so that the records stay one short line per cell. A probe of an older judge never
stands for a new one. Adding a component therefore takes its own cells, on any machine
with the dataset::

    python -m tools.probe_grid run --name mine --only 'encoder/*/MyEncoder' \\
        --scale-statistic-sha <SHA-256 of the imported reference histogram>
    python -m tools.probe_grid export --name mine
    python -m tools.make_combination_docs

``--redo`` composes the selected cells again; a config whose hash this run directory has
already probed (with the same schedule) is not probed twice, so after a change to a preset
only the cells whose config changed cost a probe. After a change to the code of a
component use a new ``--name``: its records then start empty.

``--predict-frames`` and ``--seed`` serve longer runs of the same cells
(``tools/learn_check.py``, 250 iterations): the frames every probe predicts on, and
``runtime.seed`` for every probe. A run with ``--seed`` records the hash of the config with
that seed, so probes of two seeds are two experiments. A run directory holds one setting:
give each its own ``--name``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from grasppanda.combine.grid import LOSS_NOT_PROBED, POST_NOT_PROBED, Cell, cells
from grasppanda.combine.probe import JUDGE, JUDGE_FIELDS, SCHEDULES, TREND_FLAGS, VERDICTS
from grasppanda.config import Config, git_info
from grasppanda.utils.paths import cache_root

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
#: The side file of the per-iteration losses, beside the measured file (see `side_file`).
SERIES_NAME = "losses_per_iteration.json"
#: What the measured file keeps of the second probe of a runs-unstable config (``repeat``).
REPEAT_FIELDS = (
    "verdict",
    "unstable",
    "failure",
    "judge",
    "iterations",
    "grasps_per_frame",
    "losses",
    "nonfinite",
    "trend",
    "active",
    "flags",
    "no_gradient",
    "unchanged",
    "checked",
    "score_quantiles",
)
CLI = [sys.executable, "-m", "grasppanda.cli.main"]
_LOCK = threading.Lock()


def directory(name: str) -> Path:
    return cache_root() / "probe_grid" / name


def load(name: str) -> dict[str, dict[str, Any]]:
    """The latest record of every finished cell."""
    file = directory(name) / "results.jsonl"
    records: dict[str, dict[str, Any]] = {}
    if file.is_file():
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


def _append(name: str, record: dict[str, Any]) -> None:
    with _LOCK, (directory(name) / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _spell(command: list[str]) -> str:
    """A command as a user would type it."""
    words = ["grasppanda", *command[len(CLI) :]] if command[: len(CLI)] == CLI else command
    return " ".join(shlex.quote(word) for word in words)


def _run(command: list[str], log: Path, timeout: float) -> tuple[int | None, str]:
    """Run a command of the toolbox from the repository root; ``None`` means timed out."""
    environment = {**os.environ, "PYTHONPATH": str(REPO)}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as errors:
        try:
            done = subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, ""
    return done.returncode, done.stdout


def _json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except ValueError:
        return None


# ----------------------------------------------------------------------------- compose


def compose_cell(cell: Cell, name: str) -> dict[str, Any]:
    """Compose one cell; returns the start of its record."""
    safe = cell.id.replace("/", "__")
    record: dict[str, Any] = {
        "id": cell.id,
        "group": cell.group,
        "method": cell.method,
        "base": cell.base,
        "sets": list(cell.sets),
        "slot": cell.slot,
        "component": cell.component,
        "variant": cell.variant,
    }
    if not cell.sets:  # a config as shipped: nothing to compose
        record.update(config=cell.base, config_hash=Config.load(cell.base).digest())
        return record
    target = directory(name) / "configs" / f"{safe}.yaml"
    command = [*CLI, "compose", cell.base]
    for item in cell.sets:
        command += ["--set", item]
    record["compose_command"] = _spell(command)
    log = directory(name) / "logs" / f"{safe}.compose.log"
    # --force: a cell composed again replaces the config of its earlier run.
    status, output = _run([*command, "-o", str(target), "--force", "--json"], log, 600)
    report = _json(output)
    if status not in (0, 1) or report is None:
        record.update(
            verdict="fails",
            failure={
                "phase": "compose",
                "component": "",
                "error": "ComposeError",
                "message": " ".join(log.read_text(encoding="utf-8").split())[-600:],
            },
        )
        return record
    record.update(
        adapter=report["adapter"],
        stages=report["stages"],
        actions=report["actions"],
        config_hash=report["config_hash"],
    )
    if not report["accepted"]:
        messages = list(report["problems"])
        messages += [f"{issue['path']}: {issue['message']}" for issue in report["issues"]]
        record.update(verdict="refused", refusal=messages, refused_by="compose")
        kinds = report.get("refusal_kinds")  # parallel to `refusal` (compose from 2026-10-15)
        if kinds and len(kinds) == len(messages):
            record["refusal_kinds"] = list(kinds)
    else:
        record["config"] = str(target)
    return record


# ------------------------------------------------------------------------------- probe


def _probe_options(config: str, args: argparse.Namespace) -> list[str]:
    """Overrides a probe of this config needs on this machine, and the seed of a run that
    sets one (``--seed``), as ``--opts`` words."""
    words: list[str] = []
    statistic = Config.load(config).select("data.train.dataset.scale_statistic")
    if args.scale_statistic_sha and statistic is not None:
        section = "data.train.dataset.scale_statistic"
        words += [
            f"{section}.source=legacy_import",
            f"{section}.reference_sha256={args.scale_statistic_sha}",
        ]
    if getattr(args, "seed", None) is not None:
        words.append(f"runtime.seed={args.seed}")
    return ["--opts", *words] if words else []


def seeded_hash(config: str, seed: int | None) -> str | None:
    """The hash of ``config`` trained with ``runtime.seed=<seed>``: the seed is part of the
    experiment, so a run with ``--seed`` records (and shares probes by) this hash, never
    the hash of the config as composed. ``None`` for no seed."""
    if seed is None:
        return None
    return Config.load(config, [f"runtime.seed={seed}"]).digest()


def sets_a_schedule(sets: list[str] | tuple[str, ...]) -> bool:
    """Whether a compose request sets the learning-rate schedule: a recipe preset among
    its ``slot=name`` pairs (``hooks=a+b``: each name) whose file sets ``scheduler``."""
    from grasppanda.combine.recipes import recipes

    members = recipes()
    for item in sets:
        slot, _, names = item.partition("=")
        for name in names.split("+"):
            preset = members.get(slot, {}).get(name)
            if preset is not None and "scheduler" in preset.body:
                return True
    return False


def cell_schedule(sets: list[str] | tuple[str, ...], args: argparse.Namespace) -> str:
    """The schedule a cell is probed with: the run's, except for a cell whose request sets
    the schedule (a schedule preset, or an optimizer or recipe preset that sets one). Its
    schedule is part of what it measures, so it runs as the recipe has it."""
    return "recipe" if sets_a_schedule(sets) else args.schedule


def probe_config(
    config: str, options: list[str], tag: str, name: str, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Probe one config in a process of its own; returns the command and the report.
    ``options`` are words of ``grasppanda probe`` after ``--frames`` (the schedule, the
    overrides of this machine)."""
    command = [*CLI, "probe", config, "--iters", str(args.iters), "--frames", str(args.frames)]
    if getattr(args, "predict_frames", None):
        command += ["--predict-frames", str(args.predict_frames)]
    command += options
    log = directory(name) / "logs" / f"{tag}.probe.log"
    started = time.perf_counter()
    status, output = _run([*command, "--json"], log, args.timeout)
    report = _json(output)
    if report is None:
        tail = " ".join(log.read_text(encoding="utf-8").split())[-600:]
        what = f"timeout after {args.timeout:.0f} s" if status is None else "crash"
        report = {
            "verdict": "fails",
            "detail": f"the probe process ended without a report ({what}): {tail}",
            "failure": {
                "phase": what,
                "component": "",
                "error": "ProcessError",
                "message": tail,
            },
            "seconds": round(time.perf_counter() - started, 1),
        }
    return _spell(command), report


def _probe_first(
    members: list[dict[str, Any]], name: str, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    first = members[0]
    tag = first["id"].replace("/", "__")
    return probe_config(first["config"], first["probe_options"], tag, name, args)


def _repeat(
    members: list[dict[str, Any]], report: dict[str, Any], name: str, args: argparse.Namespace
) -> dict[str, Any] | None:
    """The second probe of a config whose probe was ``runs-unstable`` (the same command,
    seed and schedule); ``None`` for any other verdict, with ``--no-repeat``, or when the
    second probe ran out of GPU memory (it says nothing then)."""
    if report.get("verdict") != "runs-unstable" or not getattr(args, "repeat", True):
        return None
    first = members[0]
    tag = first["id"].replace("/", "__") + ".repeat"
    _command, again = probe_config(first["config"], first["probe_options"], tag, name, args)
    return None if _out_of_memory(again) else again


def _out_of_memory(report: dict[str, Any]) -> bool:
    """Whether a probe died for lack of GPU memory: the allocator's own error, or a CUDA
    library that could not allocate its handle (``CUBLAS_STATUS_ALLOC_FAILED``)."""
    text = json.dumps([report.get("failure") or {}, report.get("detail") or ""]).lower()
    return "out of memory" in text or "alloc_failed" in text


def _device_fault(report: dict[str, Any]) -> bool:
    """Whether a probe failed with a CUDA launch failure, which a busy device produces
    (``grasppanda.combine.probe.device_fault`` for a report as a dict)."""
    text = str((report.get("failure") or {}).get("message", "")).lower()
    return "unspecified launch failure" in text or "cuda error: an illegal" in text


def _conclusive(record: dict[str, Any]) -> bool:
    """Whether an earlier probe is a statement about its config and can stand for a new
    one: it read real frames (or failed before reading any) and ended with a report. A
    probe that ran out of GPU memory, or whose GPU reported a launch failure, is a
    statement about the machine at that moment (the GPU is shared), so it never stands for
    a new one."""
    probe = record.get("probe") or {}
    if not probe or probe.get("verdict") == "wired" or probe.get("fallback"):
        return False
    if _out_of_memory(probe) or _device_fault(probe):
        return False
    if probe.get("judge", 1) < JUDGE:  # judged by an older judge: it lacks what is judged now
        return False
    return (probe.get("failure") or {}).get("error") != "ProcessError"


def _schedule(record: dict[str, Any]) -> str:
    """The schedule a record was probed with; a probe that does not say ran the recipe's."""
    return (record.get("probe") or {}).get("schedule", "recipe")


def run(args: argparse.Namespace) -> int:
    name = args.name
    root = directory(name)
    root.mkdir(parents=True, exist_ok=True)
    wanted = [
        cell
        for cell in cells(tuple(args.groups) if args.groups else None)
        if not args.only or any(fnmatch.fnmatch(cell.id, pattern) for pattern in args.only)
    ]
    recorded = load(name)
    done = {} if args.redo else recorded
    pending = [cell for cell in wanted if cell.id not in done]
    print(f"{root}: {len(wanted)} cells selected, {len(wanted) - len(pending)} already recorded")
    if not pending:
        return 0
    environment = _environment(args)
    (root / "run.json").write_text(
        json.dumps(environment, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    measured = _measured(environment)
    # Configs are loaded on this thread only (the registries import lazily and not
    # thread-safely); the pool runs nothing but the compose processes.
    shipped = {cell.id: compose_cell(cell, name) for cell in pending if not cell.sets}
    with ThreadPoolExecutor(4) as pool:
        made = pool.map(lambda cell: compose_cell(cell, name), [c for c in pending if c.sets])
        composed = [*shipped.values(), *made]
    for record in composed:
        if "verdict" not in record:
            record["schedule"] = cell_schedule(record["sets"], args)
            record["probe_options"] = ["--schedule", record["schedule"]]
            record["probe_options"] += _probe_options(record["config"], args)
            if getattr(args, "seed", None) is not None:
                record["config_hash"] = seeded_hash(record["config"], args.seed)
                record["seed"] = args.seed
    refused = [record for record in composed if "verdict" in record]
    for record in refused:
        _append(name, {**record, "finished": _now(), "measured": measured})
    print(f"composed {len(composed)} cells: {len(refused)} refused, probing the others")

    # One probe per distinct config and schedule: cells that compose to the same config
    # share it. With --redo the cells are composed again; a config that was probed before
    # is not, unless that probe said nothing about the config (it read no real frame, or its
    # process died). The hash is that of the config, not of the probe's schedule, so a probe
    # stands only for the same schedule.
    probed = {
        (record["config_hash"], _schedule(record)): record
        for record in recorded.values()
        if _conclusive(record) and record.get("config_hash") and not record.get("probed_as")
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in composed:
        if "verdict" not in record:
            groups.setdefault((record["config_hash"], record["schedule"]), []).append(record)

    def finish(
        members: list[dict[str, Any]],
        command: str,
        report: dict[str, Any],
        owner: str,
        when: dict[str, Any],
        repeat: dict[str, Any] | None = None,
    ) -> None:
        """Record the probe of a config for every cell that composes to it. ``when`` is
        the run that probed it: this one, or the earlier one whose probe is reused;
        ``repeat`` the second probe of a runs-unstable config, kept by the owner."""
        for record in members:
            extra: dict[str, Any] = {} if record["id"] == owner else {"probed_as": owner}
            if repeat is not None and record["id"] == owner:
                extra["repeat"] = repeat
            _append(
                name,
                {
                    **record,
                    **extra,
                    "probe_command": command,
                    "probe": report,
                    "verdict": report["verdict"],
                    "finished": _now(),
                    "measured": when,
                },
            )
        print(
            f"  {report['verdict']:<14}{owner}"
            + (f"  (+{len(members) - 1})" if members[1:] else "")
        )

    Item = tuple[tuple[str, str], list[dict[str, Any]]]

    def work(item: Item) -> Item | None:
        key, members = item
        if key in probed:
            earlier = probed[key]
            # The cell that owns the probe is named by the id it has in this grid.
            owner = next((r["id"] for r in members if r["id"] == earlier["id"]), members[0]["id"])
            when = earlier.get("measured") or measured
            repeat = earlier.get("repeat") or _repeat(members, earlier["probe"], name, args)
            finish(members, earlier["probe_command"], earlier["probe"], owner, when, repeat)
            return None
        first = members[0]
        command, report = _probe_first(members, name, args)
        if _out_of_memory(report):
            return item  # run again alone, below
        repeat = _repeat(members, report, name, args)
        finish(members, command, report, first["id"], measured, repeat)
        return None

    with ThreadPoolExecutor(max(args.jobs, 1)) as pool:
        again = [item for item in pool.map(work, groups.items()) if item is not None]
    starved = []
    for _key, members in again:
        first = members[0]
        for attempt in range(max(args.retries, 1)):
            if attempt:
                time.sleep(args.wait)
            command, report = _probe_first(members, name, args)
            if not _out_of_memory(report):
                break
        if _out_of_memory(report):
            # Not a statement about the config: the cell stays without a record.
            starved.append(first["id"])
            print(f"  {'no GPU memory':<14}{first['id']}  (not recorded)")
            continue
        report.setdefault("notes", []).append("probed alone after running out of GPU memory")
        finish(
            members, command, report, first["id"], measured, _repeat(members, report, name, args)
        )
    if starved:
        print(
            f"{len(starved)} probe(s) ran out of GPU memory {max(args.retries, 1)} time(s) alone "
            "and were not recorded: the device is shared. Run the same command again."
        )
        return 1
    return 0


def repeat(args: argparse.Namespace) -> int:
    """Probe once more every config of a run whose probe was runs-unstable and that has no
    repeat yet (a run made before `run` repeated them), and record the repeat."""
    name = args.name
    records = load(name)
    owners = [
        record
        for record in records.values()
        if record.get("verdict") == "runs-unstable"
        and not record.get("probed_as")
        and not record.get("repeat")
        and record.get("config")
        and (not args.only or any(fnmatch.fnmatch(record["id"], p) for p in args.only))
    ]
    print(f"{directory(name)}: {len(owners)} runs-unstable probe(s) without a repeat")
    args.repeat = True

    def work(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return record, _repeat([record], record["probe"], name, args)

    missing = 0
    with ThreadPoolExecutor(max(args.jobs, 1)) as pool:
        for record, again in pool.map(work, owners):
            if again is None:
                missing += 1
                print(f"  {'no GPU memory':<14}{record['id']}  (no repeat recorded)")
                continue
            _append(name, {**record, "repeat": again, "repeated": _now()})
            print(f"  {again['verdict']:<14}{record['id']}  (repeat)")
    return 1 if missing else 0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    """Where and with what a run probes. The code is named by its commit, from git or,
    for an exported copy of the repository, from ``--commit``; the version string of the
    package is not recorded, because it is whatever the last build wrote."""
    import torch

    git = git_info()
    commit = args.commit or (git.get("commit") if git.get("available") else None) or ""
    if not args.commit and git.get("dirty"):
        if not getattr(args, "allow_dirty", False):
            raise SystemExit(
                "the working tree has uncommitted changes, so no commit would name the code "
                "these probes run. Commit first; or pass --allow-dirty to record the cells "
                "as measured at `<commit>+dirty`."
            )
        commit += "+dirty"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "git": git,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": gpu,
        "host": platform.node(),
        "started": _now(),
        "iters": args.iters,
        "frames": args.frames,
        "schedule": getattr(args, "schedule", "recipe"),
        "jobs": args.jobs,
        "scale_statistic_sha": args.scale_statistic_sha,
        "commit": commit,
        "libraries": _optional_libraries(),
        **{
            key: getattr(args, key)
            for key in ("predict_frames", "seed")
            if getattr(args, key, None) is not None
        },
    }


#: Optional libraries a probed config may run on; their versions are part of a run.
OPTIONAL_LIBRARIES = (
    "spconv",
    "timm",
    "transformers",
    "torchvision",
    "torch_geometric",
    "torch_cluster",
    "fpsample",
    "kornia",
    "roma",
    "trimesh",
    "pytorch_optimizer",
    "schedulefree",
    "ocnn",
    "mamba_ssm",
    "healpix",
    "open_clip_torch",
)


def _optional_libraries() -> dict[str, str]:
    """Installed versions of the optional libraries, by distribution name."""
    from importlib import metadata

    found = {}
    for name in OPTIONAL_LIBRARIES:
        for spelling in (name, name.replace("_", "-"), f"{name}-cu118", f"{name}-cu121"):
            try:
                found[name] = metadata.version(spelling)
                break
            except metadata.PackageNotFoundError:
                continue
    return found


def _measured(environment: dict[str, Any], date: str | None = None) -> dict[str, Any]:
    """What a record keeps of the run that measured it."""
    commit = environment.get("commit") or (environment.get("git") or {}).get("commit") or ""
    short = commit[:7] + ("+dirty" if commit.endswith("+dirty") else "")
    return {
        "date": date or environment.get("started", _now())[:10],
        "commit": short or None,
        "torch": environment["torch"],
        "python": environment["python"],
        "gpu": environment["gpu"],
        "probes_at_a_time": environment["jobs"],
        "scale_statistic_sha256": environment.get("scale_statistic_sha"),
        **({"libraries": environment["libraries"]} if environment.get("libraries") else {}),
    }


# ------------------------------------------------------------------- status and export


def status(args: argparse.Namespace) -> int:
    records = load(args.name)
    every = cells()
    print(f"{directory(args.name)}: {len(records)} of {len(every)} cells recorded")
    for group in dict.fromkeys(cell.group for cell in every):
        members = [records[cell.id] for cell in every if cell.group == group and cell.id in records]
        total = sum(cell.group == group for cell in every)
        counts = Counter(record["verdict"] for record in members)
        listed = ", ".join(
            f"{counts[verdict]} {verdict}" for verdict in VERDICTS if counts[verdict]
        )
        print(f"  {group:<10} {len(members):>3}/{total:<3} {listed}")
    for record in records.values():
        if record["verdict"] in ("fails", "runs-unstable", "wired"):
            detail = (record.get("probe") or {}).get("detail") or record.get("failure", {})
            print(f"  {record['verdict']:<14}{record['id']}: {str(detail)[:230]}")
    return 0


def _rounded(pair: list[float]) -> list[float]:
    return [float(f"{value:.4g}") for value in pair]


#: What a cell of the measured file holds, field by field; exported with the cells.
FIELDS: dict[str, str] = {
    "id": "the cell: <group>/<method>/..., unique",
    "group": "reference, encoder, stage, stage-set, family, family-slot, post, loss, recipe "
    "or cross "
    "(grasppanda.combine.grid)",
    "method": "the method whose shipped config the cell starts from",
    "sets": "the slot=choice pairs of the compose request; absent for a config as shipped",
    "slot, component, variant": "for encoder and stage cells: the replaced slot, the "
    "component type put there, and its preset (or the type again, for constructor defaults)",
    "verdict": "runs, runs-unstable, fails or refused (about.verdicts)",
    "config_hash": "first 16 hex digits of the hash of the composed config; null when the "
    "config check refused it",
    "compose": "the command that wrote the config; absent for a config as shipped",
    "probe": "the command that probed it, to be run after `compose` in the same directory. "
    "It repeats the experiment, not its numbers: see about.one_run",
    "schedule, rate": "how the probe set the learning rate (about.schedules): `constant`, the "
    "recipe's peak rate at every step, or `recipe`, the recipe's schedule run over the "
    "probe's epochs (always for a cell whose request sets the schedule); `rate` is the rate "
    "every step used under `constant`, in the first parameter group of the first optimizer. "
    "A record without `schedule` was probed before the probe had the option, with the "
    "recipe's schedule",
    "probed_as": "the cell whose probe this cell shares: both composed to the same config "
    "(the same hash), which was probed once. The numbers are in that cell",
    "adapter": "the encoder adapter compose inserted, when it did",
    "fitted": "options compose set to fit the method, beyond copying a preset",
    "refusal": "the messages of the contract rules and of the config check, whole",
    "refusal_kinds": "parallel to `refusal`, the kind of each message (about.refusal_kinds: "
    "structural, not-implemented, fixed-by-compose, not-a-combination, policy); absent in a "
    "record composed before compose reported kinds",
    "repeat": "for a cell whose probe was runs-unstable: the same probe run once more (same "
    "command, seed and schedule), with the fields of a probe from `verdict` to `run`. A "
    "class counts a run as unstable only when the repeat was unstable too",
    "failure": "phase, component, exception and message of a failure after the check passed",
    "unstable": "why a run is runs-unstable",
    "parameters, iterations, batch_size, frames, grasps_per_frame": "what the probe built, "
    "trained on and predicted",
    "losses": "every loss term at the first and the last iteration, four significant digits",
    "real_seeds, supervised_seeds": "minimum and mean over the iterations of the fraction of "
    "seed slots with a real seed and of seeds that took part in the grasp losses, and the "
    "number of iterations at zero",
    "judge": "2 for a record judged by judge 2, which holds the fields from `trend` to "
    "`lr_factor`; absent for a record of the first judge, which holds none of them (about."
    "judges)",
    "trend": "per loss term and for the total `loss`, over the iterations in which the term "
    "was active (reported, finite and not exactly 0.0): [first, last_quarter, fall], its value "
    "at the first of them, the mean of the last ceil(active / 4) of them, and (first - "
    "last_quarter) / |first|, positive when the term fell (0.7: by 70 %), to four significant "
    "digits and three decimals; three nulls when the term was never active",
    "active": "per loss term active in fewer iterations than ran: the number it was active in",
    "flags": "per loss term with a flag: the flags that hold. "
    + "; ".join(f"`{name}`: {meaning}" for name, meaning in TREND_FLAGS.items()),
    "no_gradient": JUDGE_FIELDS["no_gradient"] + ". In a record of the first judge (only "
    "when its probe had the check): the top-level parts without a non-zero gradient",
    "unchanged": JUDGE_FIELDS["unchanged"],
    "checked": "[units, parts] with trainable parameters, over which `no_gradient` and "
    "`unchanged` were checked",
    "score_quantiles": JUDGE_FIELDS["score_quantiles"],
    "learning_rate, lr_factor": JUDGE_FIELDS["learning_rate"] + "; " + JUDGE_FIELDS["lr_factor"],
    "losses_per_iteration": f"not in this file: the side file measured_combinations/{SERIES_NAME} "
    "holds, for every cell of judge 2 with a probe of its own, "
    + JUDGE_FIELDS["losses_per_iteration"],
    "shared_machine": "seconds per iteration, peak GPU memory (MiB) and wall time of the "
    "probe (s): numbers that depend on the load of the machine",
    "postprocessing": "for post cells: the steps of the variant `post`, and per predicted "
    "frame the number of grasps predicted and left after every step. The grasps are those "
    "of a detector trained for the iterations of the probe",
    "notes": "what the probe changed or noticed, in its own words: for example a batch "
    "smaller than the recipe's because the split has fewer frames",
    "run": "the run that measured the cell (about.runs): date, commit and environment. A "
    "record is bound to that commit: it is replaced when its config changes, not when the "
    "code of a component does",
}

_FITTED = re.compile(r"(in_channels|out_channels): (\d+)|added `(data[\w.]+)`")


def _fitted(actions: list[str]) -> list[str]:
    """Options compose set to fit a method, from its own sentences."""
    found = []
    for action in actions:
        match = _FITTED.match(action)
        if match is not None:
            found.append(f"{match[1]}: {match[2]}" if match[1] else match[3])
    return found


def _as_typed(command: str) -> str:
    """A recorded command with the composed config named as a file of the working
    directory instead of its place in the cache of the machine that ran it."""
    return re.sub(r"\S*/configs/(\S+\.yaml)", r"\1", command)


_PATH = re.compile(
    r"(?:/[^\s\"'/;,()]+)+/((?:grasppanda|tools|tests|configs)/[^\s\"';,()]*)"
    r"|(?:/[^\s\"'/;,()]+){2,}"
)


def _not_probed(order: list[str], recorded: Any) -> dict[str, Any]:
    """``not_probed``: the cells of the grid without a record, but for the sets of stages,
    which the grid enumerates by the thousand (``grasppanda.combine.grid.cells(
    ("stage-set",))``): of those only the number without a record, ``stage_sets_not_probed``."""
    missing = [cell_id for cell_id in order if cell_id not in recorded]
    sets = sum(cell_id.startswith("stage-set/") for cell_id in missing)
    return {
        "not_probed": [cell_id for cell_id in missing if not cell_id.startswith("stage-set/")],
        "stage_sets_not_probed": sets,
    }


def _no_paths(value: Any) -> Any:
    """A message without the paths of the machine that ran it: a file of the repository by
    its place in the repository, any other absolute path as ``<path>``."""
    if not isinstance(value, str):
        return value
    return _PATH.sub(lambda match: match.group(1) or "<path>", value)


def _trends(probe: dict[str, Any]) -> dict[str, Any]:
    """``trend``, ``active`` and ``flags`` of a record from the probe's ``loss_trends``:
    an entry of ``active`` or ``flags`` only where it says something."""
    trends = probe.get("loss_trends") or {}
    ran = probe.get("iterations") or 0
    out: dict[str, Any] = {
        "trend": {
            name: [trend["first"], trend["last_quarter"], trend["fall"]]
            for name, trend in trends.items()
        }
    }
    active = {name: trend["active"] for name, trend in trends.items() if trend["active"] < ran}
    flags = {name: trend["flags"] for name, trend in trends.items() if trend["flags"]}
    if active:
        out["active"] = active
    if flags:
        out["flags"] = flags
    return out


def compact(record: dict[str, Any]) -> dict[str, Any]:
    """What the documentation keeps of one record (the fields of :data:`FIELDS`)."""
    probe = record.get("probe") or {}
    out: dict[str, Any] = {key: record[key] for key in ("id", "group", "method")}
    out.update(
        {key: record[key] for key in ("sets", "slot", "component", "variant") if record[key]}
    )
    out["verdict"] = record["verdict"]
    if probe.get("judge", 1) >= 2:
        out["judge"] = probe["judge"]
    out["config_hash"] = (record.get("config_hash") or "")[:16] or None
    if record.get("compose_command"):
        file = record["id"].replace("/", "__") + ".yaml"
        out["compose"] = f"{record['compose_command']} -o {file}"
    if record.get("adapter"):
        out["adapter"] = record["adapter"]
    fitted = _fitted(record.get("actions", []))
    if fitted:
        out["fitted"] = fitted
    if record["verdict"] == "refused":
        out["refusal"] = record.get("refusal") or probe.get("refusal") or []
        if record.get("refusal_kinds"):
            out["refusal_kinds"] = record["refusal_kinds"]
        if not record.get("refused_by"):  # the probe's own config check, for `train`
            out["probe"] = _as_typed(record["probe_command"])
        return out
    failure = record.get("failure") or probe.get("failure")
    if failure:
        out["failure"] = {key: _no_paths(value) for key, value in failure.items() if value}
    if record["verdict"] == "runs-unstable":
        detail = probe.get("detail", "")
        out["unstable"] = detail.split(", but ", 1)[1] if ", but " in detail else detail
    if record.get("probed_as"):  # the numbers are the other cell's
        out["probed_as"] = record["probed_as"]
        return out
    if not record.get("probe_command"):  # compose itself failed
        return out
    out["probe"] = _as_typed(record["probe_command"])
    if record.get("repeat"):  # the second probe of a runs-unstable config
        again = compact(
            {
                **record,
                "probe": record["repeat"],
                "verdict": record["repeat"]["verdict"],
                "repeat": None,
            }
        )
        out["repeat"] = {key: again[key] for key in REPEAT_FIELDS if key in again}
    if "schedule" in probe:
        out["schedule"] = probe["schedule"]
        if probe.get("rate") is not None:
            out["rate"] = probe["rate"]
    if probe.get("fallback"):
        out["fallback"] = probe["fallback"]
    for key in ("parameters", "iterations", "batch_size", "grasps_per_frame", "notes", "nonfinite"):
        if probe.get(key):
            out[key] = probe[key]
    if probe.get("frames"):
        out["frames"] = len(probe["frames"])
    post = probe.get("postprocessing") or {}
    if post:
        out["postprocessing"] = {"steps": post["steps"], "grasps": post["grasps"]}
    if probe.get("losses"):
        out["losses"] = {name: _rounded(pair) for name, pair in probe["losses"].items()}
    judged = probe.get("judge", 1) >= 2
    if judged:
        out.update(_trends(probe))
    for key in ("real_seeds", "supervised_seeds"):
        series = probe.get(key) or []
        if series:
            out[key] = {
                "min": round(min(series), 3),
                "mean": round(sum(series) / len(series), 3),
                "zero_iterations": sum(value == 0.0 for value in series),
            }
    if judged or "no_gradient" in probe:  # the first judge had the check from 2026-10-13 on
        out["no_gradient"] = probe.get("no_gradient") or []
    if judged:
        out["unchanged"] = probe.get("unchanged") or []
        out["checked"] = [probe.get("units_checked", 0), probe.get("parts_checked", 0)]
        if probe.get("score_quantiles"):
            out["score_quantiles"] = probe["score_quantiles"]
        if probe.get("learning_rate"):
            out["learning_rate"] = probe["learning_rate"]
            out["lr_factor"] = probe.get("lr_factor")
    # Machine-dependent numbers, kept apart and labelled as such.
    shared = {
        "seconds_per_iteration": probe.get("seconds_per_iteration"),
        "peak_gpu_mib": probe.get("peak_gpu_mib"),
        "seconds": probe.get("seconds"),
    }
    if post:  # seconds of the whole chain on the first predicted frame
        shared["post_seconds"] = sum(post["seconds"][0])
    shared = {key: value for key, value in shared.items() if value is not None}
    if shared:
        out["shared_machine"] = {
            key: round(value, 3 if key in ("seconds_per_iteration", "post_seconds") else 1)
            for key, value in shared.items()
        }
    return out


#: Which judge a record was judged by, as the measured file says it (``about.judges``).
JUDGES = (
    "A record with `judge: 2` was judged by judge 2 (grasppanda.combine.probe, from "
    "2026-10-15): `runs-empty` is a verdict, a dead top-level part makes a run "
    "`runs-unstable`, and the record holds the trend and flags of every loss term, the units "
    "without a gradient, the unchanged parts, the score quantiles and the learning-rate decay; "
    "the loss of every iteration is in the side file. A record without `judge` was judged by "
    "the first judge: it had no `runs-empty` (a run that predicted no grasp on any frame is "
    "`runs`), it holds none of those fields, and it was checked for dead top-level parts only "
    "if its probe had that check (then `no_gradient` is in the record)."
)


def side_file(target: Path) -> Path:
    """Where the per-iteration losses of the measured file ``target`` are written."""
    return target.with_suffix("") / SERIES_NAME


def merge_series(
    previous: dict[str, Any] | None, records: dict[str, dict[str, Any]], payload: dict[str, Any]
) -> dict[str, Any]:
    """The side file of per-iteration losses after a run, merged like the measured file.

    One entry per cell of ``payload`` (the merged measured file) that holds the numbers of
    a probe of judge 2 (a probe of its own, not ``probed_as``): from this run's record for
    the cells this run recorded, else the entry of the side file as it was, when it belongs
    to the same run as the record that is kept. Cells that no longer have such a record are
    dropped, so the side file never holds losses the measured file does not account for.
    """
    kept = {entry["id"]: entry for entry in (previous or {}).get("cells", [])}
    merged = []
    for cell in payload["cells"]:
        if "losses" not in cell or cell.get("judge", 1) < 2:  # no numbers of its own
            continue
        if cell["id"] in records:
            series = (records[cell["id"]].get("probe") or {}).get("losses_per_iteration")
            if series:
                merged.append({"id": cell["id"], "run": cell["run"], "losses": series})
        elif cell["id"] in kept and kept[cell["id"]].get("run") == cell["run"]:
            merged.append(kept[cell["id"]])
    return {
        "about": {
            "what": (
                "The loss of every iteration of every probe of judge 2 in "
                "measured_combinations.json: per cell, per loss term and for the total `loss`, "
                f"{JUDGE_FIELDS['losses_per_iteration']}. Written by `python -m "
                "tools.probe_grid export`; the record of a cell holds the trend computed from "
                "these values. No number here is an accuracy."
            ),
            "records": "measured_combinations.json",
        },
        "cells": merged,
    }


def dump(payload: dict[str, Any]) -> str:
    """The measured file as text: the ``about`` block indented, one cell per line."""
    about = json.dumps(payload["about"], indent=1, ensure_ascii=False)
    cells_ = ",\n".join(
        "  " + json.dumps(cell, ensure_ascii=False, separators=(", ", ": "))
        for cell in payload["cells"]
    )
    indented = about.replace("\n", "\n ")
    return f'{{\n "about": {indented},\n "cells": [\n{cells_}\n ]\n}}\n'


def _follow(cell: dict[str, Any], owner: dict[str, Any]) -> dict[str, Any]:
    """``cell``, which shares the probe of ``owner``, with the verdict, the judge, the
    reasons and the run of the owner's record; its own fields stay, in their order."""
    out: dict[str, Any] = {}
    for key, value in cell.items():
        if key in ("judge", "unstable", "failure", "run"):
            continue
        if key == "probed_as":
            out.update({name: owner[name] for name in ("failure", "unstable") if name in owner})
        out[key] = owner["verdict"] if key == "verdict" else value
        if key == "verdict" and "judge" in owner:
            out["judge"] = owner["judge"]
    out["run"] = owner["run"]
    return out


def _run_id(measured: dict[str, Any]) -> str:
    return f"{measured['date']}@{measured.get('commit') or 'unknown'}"


def merge(
    previous: dict[str, Any] | None,
    records: dict[str, dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """The measured file after a run: the cells this run recorded replace or join those of
    the file as it was, and every cell says which run measured it.

    So a component that is added later needs only its own cells probed, on any machine:
    the other cells of the committed file stay what they are, with the run that measured
    them. Cells that are no longer in the grid are dropped. A record carries the run that
    measured it (``measured``); one written before records did is given the environment
    of the run directory and the date it finished.
    """
    probe = {"iterations": environment["iters"], "frames": environment["frames"]}
    about = (previous or {}).get("about", {})
    if previous and about.get("probe") != probe:
        raise SystemExit(
            f"the measured file holds probes of {about.get('probe')} and this run is {probe}: "
            "one file, one probe. Run with the same --iters and --frames, or export with "
            "--fresh to start the file again from this run alone."
        )
    # The version string of the package is not part of a run (see `_environment`).
    runs = {
        key: {field: value for field, value in run.items() if field != "grasppanda"}
        for key, run in about.get("runs", {}).items()
    }
    kept = {cell["id"]: cell for cell in (previous or {}).get("cells", [])}
    merged = []
    order = [cell.id for cell in cells()]
    for cell_id in order:
        if cell_id in records:
            record = records[cell_id]
            when = record.get("measured") or _measured(environment, record["finished"][:10])
            runs[_run_id(when)] = when
            merged.append({**compact(record), "run": _run_id(when)})
        elif cell_id in kept:
            merged.append(kept[cell_id])
    # A kept cell that shares the probe of a cell this run probed again shares the new
    # probe: the same config, so the same experiment, with its verdict and its run.
    fresh = {cell["id"]: cell for cell in merged if cell["id"] in records}
    for index, cell in enumerate(merged):
        owner = fresh.get(cell.get("probed_as", ""))
        same = owner is not None and owner.get("config_hash") == cell.get("config_hash")
        if same and cell["id"] not in records:
            merged[index] = _follow(cell, owner)
    used = {cell["run"] for cell in merged}
    recorded = {cell["id"] for cell in merged}
    return {
        "about": {
            "what": (
                "Measured outcomes of `grasppanda compose` + `grasppanda probe` for the grid "
                "of grasppanda.combine.grid. No number here is an accuracy; see "
                "docs/advanced_guides/measured_combinations.md."
            ),
            "verdicts": VERDICTS,
            "fields": FIELDS,
            "probe": probe,
            "runs": {key: runs[key] for key in sorted(runs) if key in used},
            "one_run": (
                "A cell is one run, and training on a GPU is not repeatable here. The config "
                "hash and the parameter count repeat exactly and the losses of the first "
                "iteration to the digits given; later losses, `real_seeds`, "
                "`supervised_seeds`, `grasps_per_frame` and, for a combination near the edge, "
                "the split between `runs` and `runs-unstable` are those of the recorded run "
                "and differ in the next one."
            ),
            "scale_statistic": (
                "Configs that read the width histogram of Scale-Balanced-Grasp were probed with "
                "the reference's own file (`scale_statistic.source: legacy_import`, the SHA-256 "
                "in the probe command): the native histogram is a statistic of a whole training "
                "split, and a probe reads eight frames."
            ),
            "shared_machine": (
                "`shared_machine` holds seconds per iteration, peak GPU memory and wall time. "
                "They were measured while other jobs used the same GPU and CPUs, with as many "
                "probes at a time as the run says: an order of magnitude, not a benchmark."
            ),
            "judges": JUDGES,
            "schedules": SCHEDULES,
            **_not_probed(order, recorded),
            "presets_without_a_cell": {
                **{f"post/{name}": reason for name, reason in POST_NOT_PROBED.items()},
                **{f"loss/{name}": reason for name, reason in LOSS_NOT_PROBED.items()},
            },
        },
        "cells": merged,
    }


def export(args: argparse.Namespace) -> int:
    records = load(args.name)
    if not records:
        raise SystemExit(f"{directory(args.name)} holds no record: nothing to export")
    environment = json.loads((directory(args.name) / "run.json").read_text(encoding="utf-8"))
    target = Path(args.to)
    previous = None
    if target.is_file() and not args.fresh:
        previous = json.loads(target.read_text(encoding="utf-8"))
    payload = merge(previous, records, environment)
    target.write_text(dump(payload), encoding="utf-8")
    counts = Counter(cell["verdict"] for cell in payload["cells"])
    mine = sum(cell["id"] in records for cell in payload["cells"])
    size = target.stat().st_size / 1024
    print(
        f"wrote {target} ({size:.0f} KiB): {len(payload['cells'])} cells, {mine} of them from "
        f"this run, {dict(counts)}, {len(payload['about']['not_probed'])} not probed and "
        f"{payload['about']['stage_sets_not_probed']} sets of stages not probed"
    )
    side = side_file(target)
    earlier = None
    if side.is_file() and previous is not None:
        earlier = json.loads(side.read_text(encoding="utf-8"))
    series = merge_series(earlier, records, payload)
    if series["cells"] or side.is_file():
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(dump(series), encoding="utf-8")
        print(f"wrote {side} ({side.stat().st_size / 1024:.0f} KiB): {len(series['cells'])} cells")
    return 0


def refresh(args: argparse.Namespace) -> int:
    """Rewrite the measured file for today's grid without probing anything: drop the
    records of cells that left the grid, list the cells without a record. Everything else
    (the records that stay, their runs, the definitions of the file) is kept as it is."""
    target = Path(args.to)
    payload = json.loads(target.read_text(encoding="utf-8"))
    order = [cell.id for cell in cells()]
    by_id = {cell["id"]: cell for cell in payload["cells"]}
    kept = [by_id[cell_id] for cell_id in order if cell_id in by_id]
    dropped = sorted(set(by_id) - set(order))
    orphans = [cell["id"] for cell in kept if cell.get("probed_as") in dropped]
    if orphans:
        raise SystemExit(
            f"{len(orphans)} kept record(s) share the probe of a dropped cell, e.g. "
            f"{orphans[:3]}: probe them again instead of refreshing"
        )
    about = payload["about"]
    used = {cell["run"] for cell in kept}
    about["runs"] = {key: run for key, run in about["runs"].items() if key in used}
    about.update(_not_probed(order, by_id))
    about["presets_without_a_cell"] = {
        **{f"post/{name}": reason for name, reason in POST_NOT_PROBED.items()},
        **{f"loss/{name}": reason for name, reason in LOSS_NOT_PROBED.items()},
    }
    payload["cells"] = kept
    target.write_text(dump(payload), encoding="utf-8")
    side = side_file(target)
    if side.is_file():
        series = json.loads(side.read_text(encoding="utf-8"))
        series["cells"] = [entry for entry in series["cells"] if entry["id"] not in dropped]
        side.write_text(dump(series), encoding="utf-8")
    print(
        f"{target}: {len(kept)} records kept, {len(dropped)} dropped (no longer in the grid), "
        f"{len(about['not_probed']) + about['stage_sets_not_probed']} cells of the grid without a record"
    )
    for cell_id in dropped:
        print(f"  dropped {cell_id}")
    return 0


def _hashes(cell: Cell, scratch: Path) -> tuple[str | None, str | None]:
    """The hash of the config a cell composes to today, and the hash the same config had
    before the digest left out the defaults of library classes."""
    if not cell.sets:
        config = Config.load(cell.base)
    else:
        from grasppanda.combine.compose import compose

        result = compose(cell.base, cell.sets)
        if result.config_hash is None:
            return None, None
        written = scratch / (cell.id.replace("/", "__") + ".yaml")
        written.write_text(result.text(written), encoding="utf-8")
        config = Config.load(written)
    return config.digest()[:16], config._digest(library=True)[:16]


def rehash(args: argparse.Namespace) -> int:
    """Rewrite the hashes of records made before the config hash stopped depending on the
    installed PyTorch (it used to include the defaults of the optimizer and scheduler
    classes). A record is rewritten only when the config its cell composes to today had
    exactly the recorded hash under the old scheme, which can be checked only with the
    PyTorch version the record was made with; any other record is left alone and listed."""
    import tempfile

    target = Path(args.to)
    payload = json.loads(target.read_text(encoding="utf-8"))
    by_id = {cell.id: cell for cell in cells()}
    rewritten, current, other = [], 0, []
    with tempfile.TemporaryDirectory() as scratch:
        for record in payload["cells"]:
            cell = by_id.get(record["id"])
            if cell is None or not record.get("config_hash"):
                continue
            new, old = _hashes(cell, Path(scratch))
            if record["config_hash"] == new:
                current += 1
            elif old is not None and record["config_hash"] == old:
                record["config_hash"] = new
                rewritten.append(record["id"])
            else:
                other.append(record["id"])
    if rewritten:
        target.write_text(dump(payload), encoding="utf-8")
    print(f"{target}: {current} hashes current, {len(rewritten)} rewritten, {len(other)} neither")
    if other:
        print(
            "neither the hash of today's config nor its old-scheme hash (a changed preset or "
            "config, or records made with another PyTorch); probe these again:\n  "
            + "\n  ".join(other)
        )
    return 1 if other else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = (
        ("run", run),
        ("status", status),
        ("export", export),
        ("rehash", rehash),
        ("refresh", refresh),
        ("repeat", repeat),
    )
    for label, handler in handlers:
        sub = commands.add_parser(label, help=(handler.__doc__ or "").split(".")[0] or None)
        if label not in ("rehash", "refresh"):
            sub.add_argument("--name", default="current", help="name of the run directory")
        else:
            sub.add_argument("--to", default=str(EXPORT), help="the measured file to rewrite")
        sub.set_defaults(handler=handler)
        if label == "repeat":
            sub.add_argument("--only", nargs="+", help="glob patterns of cell ids")
            sub.add_argument("--jobs", type=int, default=2, help="probes at a time (default 2)")
            sub.add_argument("--iters", type=int, default=20, help="as the run was made")
            sub.add_argument("--frames", type=int, default=8, help="as the run was made")
            sub.add_argument("--timeout", type=float, default=1200.0, help="seconds per probe")
        if label == "run":
            sub.add_argument(
                "--no-repeat",
                dest="repeat",
                action="store_false",
                help="do not probe a runs-unstable config a second time (by default it is, "
                "and the record keeps the repeat: a class counts a run as unstable only when "
                "the repeat was unstable too)",
            )
            sub.add_argument("--groups", nargs="+", help="cell groups to run (default: all)")
            sub.add_argument("--only", nargs="+", help="glob patterns of cell ids")
            sub.add_argument("--jobs", type=int, default=2, help="probes at a time (default 2)")
            sub.add_argument("--iters", type=int, default=20)
            sub.add_argument("--frames", type=int, default=8)
            sub.add_argument(
                "--predict-frames",
                type=int,
                help="frames every probe predicts on (default: that of `grasppanda probe`, 2)",
            )
            sub.add_argument(
                "--seed",
                type=int,
                help="train every probe with `runtime.seed=<seed>`; the records then carry the "
                "hash of the config with that seed, so a probe of another seed never stands "
                "for it (default: the seed of the config)",
            )
            sub.add_argument(
                "--schedule",
                choices=tuple(SCHEDULES),
                default="constant",
                help="how the probes set the learning rate (default constant: the recipe's "
                "peak rate at every step, so that trends compare across methods; recipe: the "
                "recipe's schedule over the probe's epochs, as `grasppanda probe` does). A "
                "cell whose request sets the schedule is probed with it either way",
            )
            sub.add_argument("--timeout", type=float, default=1200.0, help="seconds per probe")
            sub.add_argument(
                "--retries",
                type=int,
                default=3,
                help="probes of a config that ran out of GPU memory, alone, before it is left "
                "without a record (default 3)",
            )
            sub.add_argument(
                "--wait", type=float, default=30.0, help="seconds between those (default 30)"
            )
            sub.add_argument("--redo", action="store_true", help="ignore earlier records")
            sub.add_argument(
                "--allow-dirty",
                action="store_true",
                help="probe from a working tree with uncommitted changes; the cells are then "
                "recorded as measured at `<commit>+dirty`",
            )
            sub.add_argument(
                "--commit",
                help="commit of the code that is probed, when it runs from an exported copy "
                "of the repository and git cannot say",
            )
            sub.add_argument(
                "--scale-statistic-sha",
                help="probe configs that read a width histogram with the imported reference "
                "file of this SHA-256 (the native one is a statistic of the whole training split)",
            )
        if label == "export":
            sub.add_argument("--to", default=str(EXPORT), help="file to write")
            sub.add_argument(
                "--fresh",
                action="store_true",
                help="write the cells of this run only, instead of merging them into the file",
            )
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
