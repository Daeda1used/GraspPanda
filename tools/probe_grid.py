"""Probe the grid of combinations and record what happened, cell by cell.

Every cell of :mod:`grasppanda.combine.grid` is composed with ``grasppanda compose`` and,
when the config check accepts it, probed with ``grasppanda probe``: each in a process of
its own, the commands recorded with the outcome. Results are appended to
``$GRASPPANDA_CACHE/probe_grid/<name>/results.jsonl``, so an interrupted run continues
where it stopped; the composed configs and the output of every probe lie beside it.

    python -m tools.probe_grid run --groups reference encoder stage family
    python -m tools.probe_grid run --groups stage-set          # the legal stage sets, later
    python -m tools.probe_grid status
    python -m tools.probe_grid export       # -> docs/advanced_guides/measured_combinations.json

Two cells that compose to the same config (the same config hash) are one experiment: it
is probed once and both cells record it, naming the command that ran. At most ``--jobs``
probes run at a time; a probe that ran out of GPU memory next to another one is run
again alone before it is recorded.

The export is the compact file the documentation page is generated from
(``python -m tools.make_combination_docs``): one line per cell with its verdict, its two
commands, the messages of a refusal or a failure and the numbers that do not depend on
machine load, plus timings and memory labelled as measured on a shared machine. A cell
that shares the probe of another one names it instead of repeating its numbers, and the
composed config is named as a file of the working directory, not by its place in the
cache of the machine that ran the grid.

The export merges: cells this run recorded replace or join those of the file as it is,
and every cell names the run that measured it (date, commit, environment). Adding a
component therefore takes its own cells, on any machine with the dataset::

    python -m tools.probe_grid run --name mine --only 'encoder/*/MyEncoder'
    python -m tools.probe_grid export --name mine
    python -m tools.make_combination_docs

``--redo`` composes the selected cells again; a config whose hash this run directory has
already probed is not probed twice, so after a change to a preset only the cells whose
config changed cost a probe. After a change to the code of a component use a new
``--name``: its records then start empty.
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

from grasppanda.combine.grid import Cell, cells
from grasppanda.combine.probe import VERDICTS
from grasppanda.config import Config, git_info
from grasppanda.utils.paths import cache_root

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
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
    status, output = _run([*command, "-o", str(target), "--json"], log, 600)
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
    else:
        record["config"] = str(target)
    return record


# ------------------------------------------------------------------------------- probe


def _probe_options(config: str, args: argparse.Namespace) -> list[str]:
    """Overrides a probe of this config needs on this machine, as ``--opts`` words."""
    statistic = Config.load(config).select("data.train.dataset.scale_statistic")
    if args.scale_statistic_sha and statistic is not None:
        section = "data.train.dataset.scale_statistic"
        return [
            "--opts",
            f"{section}.source=legacy_import",
            f"{section}.reference_sha256={args.scale_statistic_sha}",
        ]
    return []


def probe_config(
    config: str, options: list[str], tag: str, name: str, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Probe one config in a process of its own; returns the command and the report."""
    command = [*CLI, "probe", config, "--iters", str(args.iters), "--frames", str(args.frames)]
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


def _out_of_memory(report: dict[str, Any]) -> bool:
    """Whether a probe died for lack of GPU memory: the allocator's own error, or a CUDA
    library that could not allocate its handle (``CUBLAS_STATUS_ALLOC_FAILED``)."""
    text = json.dumps([report.get("failure") or {}, report.get("detail") or ""]).lower()
    return "out of memory" in text or "alloc_failed" in text


def _conclusive(record: dict[str, Any]) -> bool:
    """Whether an earlier probe is a statement about its config and can stand for a new
    one: it read real frames (or failed before reading any) and ended with a report. A
    probe that ran out of GPU memory is a statement about the machine at that moment (the
    GPU is shared), so it never stands for a new one."""
    probe = record.get("probe") or {}
    if not probe or probe.get("verdict") == "wired" or probe.get("fallback"):
        return False
    if _out_of_memory(probe):
        return False
    return (probe.get("failure") or {}).get("error") != "ProcessError"


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
    (root / "run.json").write_text(
        json.dumps(_environment(args), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Configs are loaded on this thread only (the registries import lazily and not
    # thread-safely); the pool runs nothing but the compose processes.
    shipped = {cell.id: compose_cell(cell, name) for cell in pending if not cell.sets}
    with ThreadPoolExecutor(4) as pool:
        made = pool.map(lambda cell: compose_cell(cell, name), [c for c in pending if c.sets])
        composed = [*shipped.values(), *made]
    for record in composed:
        if "verdict" not in record:
            record["probe_options"] = _probe_options(record["config"], args)
    refused = [record for record in composed if "verdict" in record]
    for record in refused:
        _append(name, {**record, "finished": _now()})
    print(f"composed {len(composed)} cells: {len(refused)} refused, probing the others")

    # One probe per distinct config: cells that compose to the same config share it. With
    # --redo the cells are composed again; a config that was probed before is not, unless
    # that probe said nothing about the config (it read no real frame, or its process died).
    probed = {
        record["config_hash"]: record
        for record in recorded.values()
        if _conclusive(record) and record.get("config_hash") and not record.get("probed_as")
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in composed:
        if "verdict" not in record:
            groups.setdefault(record["config_hash"], []).append(record)

    def finish(members: list[dict[str, Any]], command: str, report: dict[str, Any], owner: str):
        for record in members:
            extra = {} if record["id"] == owner else {"probed_as": owner}
            _append(
                name,
                {
                    **record,
                    **extra,
                    "probe_command": command,
                    "probe": report,
                    "verdict": report["verdict"],
                    "finished": _now(),
                },
            )
        print(
            f"  {report['verdict']:<14}{owner}"
            + (f"  (+{len(members) - 1})" if members[1:] else "")
        )

    def work(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]] | None:
        digest, members = item
        if digest in probed:
            earlier = probed[digest]
            finish(members, earlier["probe_command"], earlier["probe"], earlier["id"])
            return None
        first = members[0]
        command, report = _probe_first(members, name, args)
        if _out_of_memory(report):
            return item  # run again alone, below
        finish(members, command, report, first["id"])
        return None

    with ThreadPoolExecutor(max(args.jobs, 1)) as pool:
        again = [item for item in pool.map(work, groups.items()) if item is not None]
    for _digest, members in again:
        first = members[0]
        command, report = _probe_first(members, name, args)
        report.setdefault("notes", []).append("probed alone after running out of GPU memory")
        finish(members, command, report, first["id"])
    return 0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from grasppanda import __version__

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "grasppanda": __version__,
        "git": git_info(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": gpu,
        "host": platform.node(),
        "started": _now(),
        "iters": args.iters,
        "frames": args.frames,
        "jobs": args.jobs,
        "scale_statistic_sha": args.scale_statistic_sha,
        "commit": args.commit,
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
    "group": "reference, encoder, stage, stage-set, family, family-slot or post "
    "(grasppanda.combine.grid)",
    "method": "the method whose shipped config the cell starts from",
    "sets": "the slot=choice pairs of the compose request; absent for a config as shipped",
    "slot, component, variant": "for encoder and stage cells: the replaced slot, the "
    "component type put there, and its preset (or the type again, for constructor defaults)",
    "verdict": "runs, runs-unstable, fails or refused (about.verdicts)",
    "config_hash": "first 16 hex digits of the hash of the composed config; null when the "
    "config check refused it",
    "compose": "the command that wrote the config; absent for a config as shipped",
    "probe": "the command that probed it. Run after `compose` in the same directory it "
    "repeats the cell",
    "probed_as": "the cell whose probe this cell shares: both composed to the same config "
    "(the same hash), which was probed once. The numbers are in that cell",
    "adapter": "the encoder adapter compose inserted, when it did",
    "fitted": "options compose set to fit the method, beyond copying a preset",
    "refusal": "the messages of the contract rules and of the config check, whole",
    "failure": "phase, component, exception and message of a failure after the check passed",
    "unstable": "why a run is runs-unstable",
    "parameters, iterations, batch_size, frames, grasps_per_frame": "what the probe built, "
    "trained on and predicted",
    "losses": "every loss term at the first and the last iteration, four significant digits",
    "real_seeds, supervised_seeds": "minimum and mean over the iterations of the fraction of "
    "seed slots with a real seed and of seeds that took part in the grasp losses, and the "
    "number of iterations at zero",
    "shared_machine": "seconds per iteration, peak GPU memory (MiB) and wall time of the "
    "probe (s): numbers that depend on the load of the machine",
    "postprocessing": "for post cells: the steps of the variant `post`, and per predicted "
    "frame the number of grasps predicted and left after every step. The grasps are those "
    "of a detector trained for the iterations of the probe",
    "run": "the run that measured the cell (about.runs): date, commit and environment",
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


def compact(record: dict[str, Any]) -> dict[str, Any]:
    """What the documentation keeps of one record (the fields of :data:`FIELDS`)."""
    probe = record.get("probe") or {}
    out: dict[str, Any] = {key: record[key] for key in ("id", "group", "method")}
    out.update(
        {key: record[key] for key in ("sets", "slot", "component", "variant") if record[key]}
    )
    out["verdict"] = record["verdict"]
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
        if not record.get("refused_by"):  # the probe's own config check, for `train`
            out["probe"] = _as_typed(record["probe_command"])
        return out
    failure = record.get("failure") or probe.get("failure")
    if failure:
        out["failure"] = {key: value for key, value in failure.items() if value}
    if record["verdict"] == "runs-unstable":
        detail = probe.get("detail", "")
        out["unstable"] = detail.split(", but ", 1)[1] if ", but " in detail else detail
    if record.get("probed_as"):  # the numbers are the other cell's
        out["probed_as"] = record["probed_as"]
        return out
    if not record.get("probe_command"):  # compose itself failed
        return out
    out["probe"] = _as_typed(record["probe_command"])
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
    for key in ("real_seeds", "supervised_seeds"):
        series = probe.get(key) or []
        if series:
            out[key] = {
                "min": round(min(series), 3),
                "mean": round(sum(series) / len(series), 3),
                "zero_iterations": sum(value == 0.0 for value in series),
            }
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


def dump(payload: dict[str, Any]) -> str:
    """The measured file as text: the ``about`` block indented, one cell per line."""
    about = json.dumps(payload["about"], indent=1, ensure_ascii=False)
    cells_ = ",\n".join(
        "  " + json.dumps(cell, ensure_ascii=False, separators=(", ", ": "))
        for cell in payload["cells"]
    )
    indented = about.replace("\n", "\n ")
    return f'{{\n "about": {indented},\n "cells": [\n{cells_}\n ]\n}}\n'


def merge(
    previous: dict[str, Any] | None,
    records: dict[str, dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """The measured file after a run: the cells this run recorded replace or join those of
    the file as it was, and every cell says which run measured it.

    So a component that is added later needs only its own cells probed, on any machine:
    the other cells of the committed file stay what they are, with the run that measured
    them. Cells that are no longer in the grid are dropped.
    """
    probe = {"iterations": environment["iters"], "frames": environment["frames"]}
    about = (previous or {}).get("about", {})
    if previous and about.get("probe") != probe:
        raise SystemExit(
            f"the measured file holds probes of {about.get('probe')} and this run is {probe}: "
            "one file, one probe. Run with the same --iters and --frames, or export with "
            "--fresh to start the file again from this run alone."
        )
    commit = environment.get("commit") or (environment.get("git") or {}).get("commit") or ""
    date = max(record["finished"] for record in records.values())[:10]
    run_id = f"{date}@{commit[:7] or 'unknown'}"
    runs = dict(about.get("runs", {}))
    runs[run_id] = {key: environment[key] for key in ("grasppanda", "torch", "python", "gpu")} | {
        "commit": commit[:7] or None,
        "date": date,
        "probes_at_a_time": environment["jobs"],
        "scale_statistic_sha256": environment.get("scale_statistic_sha"),
    }
    kept = {cell["id"]: cell for cell in (previous or {}).get("cells", [])}
    merged = []
    order = [cell.id for cell in cells()]
    for cell_id in order:
        if cell_id in records:
            merged.append({**compact(records[cell_id]), "run": run_id})
        elif cell_id in kept:
            merged.append(kept[cell_id])
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
            "runs": {key: value for key, value in runs.items() if key in used},
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
            "not_probed": [cell_id for cell_id in order if cell_id not in recorded],
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
        f"this run, {dict(counts)}, {len(payload['about']['not_probed'])} not probed"
    )
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
    handlers = (("run", run), ("status", status), ("export", export), ("rehash", rehash))
    for label, handler in handlers:
        sub = commands.add_parser(label, help=(handler.__doc__ or "").split(".")[0] or None)
        if label != "rehash":
            sub.add_argument("--name", default="current", help="name of the run directory")
        else:
            sub.add_argument("--to", default=str(EXPORT), help="the measured file to rewrite")
        sub.set_defaults(handler=handler)
        if label == "run":
            sub.add_argument("--groups", nargs="+", help="cell groups to run (default: all)")
            sub.add_argument("--only", nargs="+", help="glob patterns of cell ids")
            sub.add_argument("--jobs", type=int, default=2, help="probes at a time (default 2)")
            sub.add_argument("--iters", type=int, default=20)
            sub.add_argument("--frames", type=int, default=8)
            sub.add_argument("--timeout", type=float, default=1200.0, help="seconds per probe")
            sub.add_argument("--redo", action="store_true", help="ignore earlier records")
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
