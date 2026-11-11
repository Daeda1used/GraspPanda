"""``grasppanda sweep``: run a grid of overrides and tabulate it."""

from __future__ import annotations

import argparse
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
    parser.add_argument("sweep", help="sweep file (YAML with `base`, `axes`, `seeds`)")
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
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    sweep = load_sweep(args.sweep)
    cells = sweep.plan(command="train")
    rows = [
        [
            cell.name,
            style("valid", "green") if cell.valid else style("invalid", "red"),
            cell.config.digest()[:12] if cell.config else "—",
        ]
        for cell in cells
    ]
    print(f"sweep {sweep.name}: {len(cells)} runs from {sweep.base}")
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
