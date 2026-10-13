"""Entry point of the ``grasppanda`` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from grasppanda import __version__
from grasppanda.config import ConfigError
from grasppanda.utils.console import banner, style

from . import analyze, components, demo, env, infer, prepare, print_config, sweep, test, train

__all__ = ["main"]

#: Commands of the design that are not implemented yet, with the milestone that brings them.
PLANNED = (("profile", "parameters, FLOPs and per-stage latency", "M3"),)


def _epilog() -> str:
    lines = ["planned commands (not available yet):"]
    lines += [f"  {name:<14}{summary} ({milestone})" for name, summary, milestone in PLANNED]
    lines += ["", "docs: https://daeda1used.github.io/GraspPanda"]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grasppanda",
        description="GraspPanda: a toolbox for 6-DoF grasp detection in cluttered scenes.",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=banner(__version__))
    commands = parser.add_subparsers(dest="command", metavar="<command>", title="commands")
    train.register(commands)
    test.register(commands)
    infer.register(commands)
    demo.register(commands)
    analyze.register(commands)
    sweep.register(commands)
    print_config.register(commands)
    components.register(commands)
    prepare.register(commands)
    env.register(commands)
    return parser


def _short(source: str) -> str:
    """Show paths relative to the working directory when they are below it."""
    try:
        return str(Path(source).resolve().relative_to(Path.cwd()))
    except (ValueError, OSError):
        return source


def report_config_error(error: ConfigError) -> None:
    """Print a :class:`ConfigError` to stderr, one block per problem."""
    color = sys.stderr.isatty()
    count = len(error.issues)
    head = f"✗ {count} config problem{'s' if count != 1 else ''}"
    if error.title:
        head += f" in {error.title}"
    print(style(head, "bold", "red", enabled=color), file=sys.stderr)
    for issue in error.issues:
        print(file=sys.stderr)
        print("  " + style(issue.path or "<file>", "bold", enabled=color), file=sys.stderr)
        for line in issue.message.splitlines():
            print(f"    {line}", file=sys.stderr)
        if issue.source:
            print(
                style(f"    set in: {_short(issue.source)}", "gray", enabled=color), file=sys.stderr
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return args.handler(args)
    except ConfigError as error:
        report_config_error(error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
