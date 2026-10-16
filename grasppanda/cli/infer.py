"""``grasppanda infer``: predict on a split or a directory and write the grasps.

Nothing is scored and no evaluator is needed. The flags are shorthand for overrides:

* ``--weights PATH``  → ``runtime.load_from=PATH``
* ``--input NAME``    → ``evaluation.split=NAME`` for ``train``, ``val`` or ``test``
* ``--input DIR``     → ``data.<evaluation.split>.dataset.root=DIR``: the reader of that
  section is pointed at another directory, so any dataset with a ``root`` parameter works.
  A directory the reader cannot read (a plain folder of frames) is refused before anything
  runs, with what to do instead
* ``--output DIR``    → ``runtime.work_dir`` and ``runtime.run_id``, so that ``DIR`` *is*
  the run directory: it receives ``predictions/<variant>/`` and the usual snapshot
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from grasppanda.config import Config, ConfigError, Issue
from grasppanda.utils.console import style

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
        "--post",
        nargs="+",
        metavar="STEP",
        help="also write the variant `post`: this chain of post-processing steps, e.g. "
        "collision nms top-k=50 (presets of `grasppanda compose --list post`, or component "
        "names; name=value sets the main option, name:option=value,... any option)",
    )
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
    if getattr(args, "post", None):
        from grasppanda.combine.post import PostSpecError, chain_override

        try:
            out.append(chain_override(args.post))
        except PostSpecError as error:
            raise ConfigError([Issue("--post", str(error))]) from None
    if args.output:
        target = Path(args.output).expanduser().resolve()
        out.append(f"runtime.work_dir={json.dumps(str(target.parent))}")
        out.append(f"runtime.run_id={json.dumps(target.name)}")
    return out


def check_directory(args: argparse.Namespace) -> None:
    """Raise :class:`~grasppanda.apis.InputError` unless the reader of the inferred section
    can read ``--input DIR``: it needs a ``root`` parameter, and the directory has to be laid
    out like its dataset (its first frame is read here, before anything else runs)."""
    from grasppanda.apis.inferencer import InputError, plain_folder_advice
    from grasppanda.registry import DATASETS

    directory = args.input
    advice = plain_folder_advice(args.config, args.weights or "<file>")
    try:  # the config without the directory; its own problems are reported by `run`
        config = Config.load(
            args.config, overrides(argparse.Namespace(**{**vars(args), "input": None}))
        )
        section = config.select("evaluation.split", "test")
        node = dict(config.resolved()["data"][section]["dataset"])
    except (ConfigError, KeyError, TypeError):
        return
    reader = DATASETS.get(node["type"])
    if "root" not in inspect.signature(reader).parameters:
        raise InputError(
            f"--input {directory}: the reader of data.{section}, {node['type']}, has no `root` "
            "parameter, so it cannot be pointed at a directory. --input DIR hands a directory "
            "to a reader with `root` as the root of its dataset tree. " + advice
        )
    try:
        dataset = DATASETS.build({**node, "root": directory})
        if len(dataset):
            dataset[0]
    except (OSError, ValueError) as error:
        title = getattr(reader, "title", node["type"])
        raise InputError(
            f"--input {directory}: the reader of data.{section}, {node['type']}, reads a "
            f"directory as the root of a {title} tree and cannot read this one: {error}. " + advice
        ) from None


def run(args: argparse.Namespace) -> int:
    if args.input and args.input not in SECTIONS and Path(args.input).is_dir():
        from grasppanda.apis.inferencer import InputError

        try:
            check_directory(args)
        except InputError as error:
            mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
            print(f"{mark} InputError: {error}", file=sys.stderr)
            return 2
    config = Config.load(args.config, overrides(args), command="infer")
    require_section(config, "infer")
    from grasppanda.engine import commands

    return commands.launch(commands.infer, config)
