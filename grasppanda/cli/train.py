"""``grasppanda train``: train a detector from a config.

The command has no settings of its own. ``--work-dir`` and ``--resume`` are shorthand
for the overrides ``runtime.work_dir=...`` and ``runtime.resume=...``; they go through
the same validation and end up in the run snapshot like any other override.

Distributed training needs no flag either::

    torchrun --nproc_per_node=4 -m grasppanda.cli.main train <config>
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from grasppanda.config import Config

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "train",
        help="train a model from a config",
        description="Train a detector. Every setting lives in the config; see `--opts`.",
    )
    parser.add_argument("config", help="config file or name of a shipped config")
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. train.epochs=20 optimizer.lr=3e-4",
    )
    parser.add_argument("--work-dir", metavar="DIR", help="shorthand for runtime.work_dir=DIR")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        metavar="CHECKPOINT",
        help="shorthand for runtime.resume: the latest checkpoint in the work dir, or a file",
    )
    parser.set_defaults(handler=run)


def overrides(args: argparse.Namespace) -> list[str]:
    """``--opts`` followed by the convenience flags, written as overrides."""
    out = list(args.opts)
    if args.work_dir:
        out.append(f"runtime.work_dir={json.dumps(args.work_dir)}")
    if args.resume is True:
        out.append("runtime.resume=true")
    elif args.resume:
        out.append(f"runtime.resume={json.dumps(args.resume)}")
    return out


def run(args: argparse.Namespace) -> int:
    config = Config.load(args.config, overrides(args), command="train")
    from grasppanda.engine import commands  # deferred: keeps `grasppanda --help` instant

    return commands.launch(commands.train, config)
