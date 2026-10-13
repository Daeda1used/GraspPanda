"""``grasppanda test``: score weights on a split and write result records.

``--weights``, ``--split`` and ``--work-dir`` are shorthand for the overrides
``runtime.load_from=...``, ``evaluation.split=...`` and ``runtime.work_dir=...``. Which
frames a split holds is the dataset's business: change it with
``--opts data.test.dataset.<parameter>=...``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from grasppanda.config import Config, ConfigError, Issue

__all__ = ["register"]

SECTIONS = ("train", "val", "test")


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "test",
        help="evaluate weights under the configured protocol",
        description=(
            "Predict on a split, apply every post-processing variant of evaluation.variants "
            "and score each with the configured evaluator. Writes one result record per variant."
        ),
    )
    parser.add_argument("config", help="config file or name of a shipped config")
    parser.add_argument(
        "--weights", metavar="PATH", help="shorthand for runtime.load_from=PATH (required once)"
    )
    parser.add_argument(
        "--split",
        choices=SECTIONS,
        help="shorthand for evaluation.split: the data section to score (default: test)",
    )
    parser.add_argument("--work-dir", metavar="DIR", help="shorthand for runtime.work_dir=DIR")
    parser.add_argument(
        "--opts", nargs="+", default=[], metavar="KEY=VALUE", help="override config values"
    )
    parser.set_defaults(handler=run)


def overrides(args: argparse.Namespace) -> list[str]:
    out = list(args.opts)
    if args.weights:
        out.append(f"runtime.load_from={json.dumps(args.weights)}")
    if args.split:
        out.append(f"evaluation.split={args.split}")
    if args.work_dir:
        out.append(f"runtime.work_dir={json.dumps(args.work_dir)}")
    return out


def require_section(config: Config, command: str) -> None:
    """The split a command reads must have a ``data`` section."""
    section = config.select("evaluation.split")
    if config.select(f"data.{section}") is None:
        message = (
            f"`grasppanda {command}` reads this section (evaluation.split is '{section}'), "
            "but it is not set."
        )
        raise ConfigError([Issue(f"data.{section}", message)], str(config.path))


def run(args: argparse.Namespace) -> int:
    config = Config.load(args.config, overrides(args), command="test")
    require_section(config, "test")
    from grasppanda.engine import commands

    return commands.launch(commands.test, config)
