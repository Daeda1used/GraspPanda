"""``grasppanda sweep``: run a grid of overrides and tabulate it."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from grasppanda.sweep import aggregate, load_sweep, render_table, run_cell, run_cells
from grasppanda.utils.console import style, table

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "sweep",
        help="run a grid of config overrides and tabulate the results",
        description=(
            "Expand a sweep file into runs (axes x seeds), validate every config before "
            "anything starts, train and test each cell, and write a summary table. "
            "Finished cells are skipped, so a sweep can be resumed. With --gpus the runs "
            "are spread over GPUs, one run per GPU at a time."
        ),
    )
    parser.add_argument(
        "sweep",
        help="sweep file (YAML with `base`: one config, or several by name; `axes`; `seeds`)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    parser.add_argument(
        "--gpus",
        metavar="IDS",
        help="run the cells in parallel, one process per cell and one cell per GPU at a "
        "time, e.g. 0,1,2,3 (name a GPU twice for two cells on it); each cell logs to "
        "sweep_cell.log in its work directory",
    )
    parser.add_argument(
        "--cell",
        metavar="NAME",
        help="run only this cell of the plan, in this process (what --gpus starts per "
        "cell; also for a job scheduler), e.g. ptv3/seed0",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override config values in every cell, e.g. runtime.gpu_memory_fraction=0.45 "
        "evaluation.num_workers=8 (after the file's `overrides`, before its axes)",
    )
    parser.set_defaults(handler=run)


def _unknown_gpus(gpus: list[str]) -> str | None:
    """What is wrong with the ids of ``--gpus``, or ``None``.

    A cell started with an id this machine does not have would see no GPU at all, and a
    config with ``device: auto`` would then train on the CPU without a word.
    """
    if not gpus:
        return None
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        return "--gpus spreads cells over GPUs, and PyTorch sees none here (grasppanda env)."
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        scope = f"of the {count} that CUDA_VISIBLE_DEVICES leaves visible"
    else:
        scope = f"of this machine's {count}"
    wrong = [gpu for gpu in gpus if not gpu.isdigit() or int(gpu) >= count]
    if not wrong:
        return None
    return (
        f"--gpus names {', '.join(wrong)}: not a GPU {scope} "
        f"(ids 0 to {count - 1}, as nvidia-smi numbers them)."
    )


def run(args: argparse.Namespace) -> int:
    sweep = load_sweep(args.sweep, args.opts)
    cells = sweep.plan(command="train")
    rows = [
        [
            cell.name,
            style("valid", "green") if cell.valid else style("invalid", "red"),
            cell.config.digest()[:12] if cell.config else "—",
        ]
        for cell in cells
    ]
    source = f"{len(sweep.bases)} configs" if sweep.bases else str(sweep.base)
    print(f"sweep {sweep.name}: {len(cells)} runs from {source}")
    print(table(["run", "config", "hash"], rows))
    invalid = [cell for cell in cells if not cell.valid]
    if invalid:
        for cell in invalid:
            print(f"\n{cell.name}\n{cell.error}", file=sys.stderr)
        print(
            f"\n{len(invalid)} of {len(cells)} runs have an invalid config; nothing was run.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return 0
    if args.cell:
        if args.gpus is not None:
            print(
                "--cell runs one cell in this process: choose its GPU with "
                "CUDA_VISIBLE_DEVICES, not with --gpus.",
                file=sys.stderr,
            )
            return 2
        chosen = [cell for cell in cells if cell.name == args.cell]
        if not chosen:
            names = ", ".join(cell.name for cell in cells)
            print(f"no cell '{args.cell}' in this sweep. Cells: {names}", file=sys.stderr)
            return 2
        run_cell(sweep, chosen[0])
        return 0
    gpus = [part.strip() for part in (args.gpus or "").split(",") if part.strip()]
    if args.gpus is not None and not gpus:
        print("--gpus needs at least one GPU id, e.g. --gpus 0,1", file=sys.stderr)
        return 2
    unknown = _unknown_gpus(gpus)
    if unknown:
        print(unknown, file=sys.stderr)
        return 2
    if gpus:
        print(f"{len(cells)} runs on GPUs {', '.join(gpus)}, {len(gpus)} at a time")
    records = run_cells(
        sweep,
        cells,
        gpus=gpus,
        report=lambda line: print(style(line, "bold") if not gpus else line, flush=True),
    )
    failed = [record for record in records if "failed" in record]
    out = Path("work_dirs") / sweep.name
    out.mkdir(parents=True, exist_ok=True)
    for source in ("test", "validation"):
        rows = aggregate(records, source=source)
        if not any(row["metrics"] for row in rows):
            continue
        (out / f"summary_{source}.md").write_text(render_table(rows), encoding="utf-8")
        (out / f"summary_{source}.csv").write_text(render_table(rows, fmt="csv"), encoding="utf-8")
        print(style(f"\n{source} metrics, mean ± std over seeds", "bold"))
        print(render_table(rows))
    print(f"summaries written to {out}/")
    if failed:
        for record in failed:
            print(f"✗ {record['cell']} failed: {record['failed']}", file=sys.stderr)
        print(
            f"{len(failed)} of {len(records)} runs failed and are not in the summaries; "
            "run the sweep again to run them only.",
            file=sys.stderr,
        )
        return 1
    return 0
