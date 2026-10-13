"""``grasppanda sweep``: run a grid of overrides and tabulate it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from grasppanda.sweep import aggregate, load_sweep, render_table, run_cell
from grasppanda.utils.console import style, table

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "sweep",
        help="run a grid of config overrides and tabulate the results",
        description=(
            "Expand a sweep file into runs (axes x seeds), validate every config before "
            "anything starts, train and test each cell, and write a summary table. "
            "Finished cells are skipped, so a sweep can be resumed."
        ),
    )
    parser.add_argument("sweep", help="sweep file (YAML with `base`, `axes`, `seeds`)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
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
    records = []
    for index, cell in enumerate(cells, start=1):
        print(style(f"\n[{index}/{len(cells)}] {cell.name}", "bold"))
        records.append(run_cell(sweep, cell))
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
    return 0
