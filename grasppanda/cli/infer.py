"""``grasppanda infer``: predict on a split or a directory and write the grasps.

Nothing is scored and no evaluator is needed. The flags are shorthand for overrides:

* ``--weights PATH``  → ``runtime.load_from=PATH``
* ``--input NAME``    → ``evaluation.split=NAME`` for ``train``, ``val`` or ``test``
* ``--input DIR``     → ``data.<evaluation.split>.dataset.root=DIR``: the reader of that
  section is pointed at another directory, so any dataset with a ``root`` parameter works
* ``--output DIR``    → ``runtime.work_dir`` and ``runtime.run_id``, so that ``DIR`` *is*
  the run directory: it receives ``predictions/<variant>/`` and the usual snapshot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grasppanda.config import Config, ConfigError, Issue

from .test import SECTIONS, require_section

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "infer",
        help="run on a dataset split or a directory and dump grasps",
        description=(
            "Predict and write one GraspNet (G, 17) array per frame, for the raw predictions "
            "and for every post-processing variant of evaluation.variants."
        ),
    )
    parser.add_argument("config", help="config file or name of a shipped config")
    parser.add_argument("--weights", metavar="PATH", help="shorthand for runtime.load_from=PATH")
    parser.add_argument(
        "--input",
        metavar="SPLIT|DIR",
        help="a data section (train, val, test) or a directory for that section's reader",
    )
    parser.add_argument("--output", metavar="DIR", help="run directory to create and write into")
    parser.add_argument(
        "--opts", nargs="+", default=[], metavar="KEY=VALUE", help="override config values"
    )
    parser.set_defaults(handler=run)


def overrides(args: argparse.Namespace) -> list[str]:
    out = list(args.opts)
    if args.weights:
        out.append(f"runtime.load_from={json.dumps(args.weights)}")
    if args.input in SECTIONS:
        out.append(f"evaluation.split={args.input}")
    elif args.input:
        if not Path(args.input).is_dir():
            message = (
                f"'{args.input}' is neither a data section ({', '.join(SECTIONS)}) nor a directory."
            )
            raise ConfigError([Issue("--input", message)])
        section = Config.load(args.config, out, validate=False).select("evaluation.split", "test")
        out.append(f"data.{section}.dataset.root={json.dumps(args.input)}")
    if args.output:
        target = Path(args.output).expanduser().resolve()
        out.append(f"runtime.work_dir={json.dumps(str(target.parent))}")
        out.append(f"runtime.run_id={json.dumps(target.name)}")
    return out


def run(args: argparse.Namespace) -> int:
    config = Config.load(args.config, overrides(args), command="infer")
    require_section(config, "infer")
    from grasppanda.engine import commands

    return commands.launch(commands.infer, config)
