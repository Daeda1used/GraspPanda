"""``grasppanda print-config``: show, validate and explore configs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from grasppanda.config import Config
from grasppanda.registry import all_registries
from grasppanda.utils.console import highlight_yaml, style, table

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "print-config",
        help="print the fully resolved config, validate it, or list components",
        description=(
            "Print a config after inheritance, overrides and interpolation. "
            "Validation always runs unless --no-check is given."
        ),
    )
    parser.add_argument("config", nargs="?", help="config file or name of a shipped config")
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. model.backbone.type=PTv3 train.epochs=20",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate only; print a summary instead of the config"
    )
    parser.add_argument(
        "--no-check", action="store_true", help="print the merged config without validating it"
    )
    parser.add_argument(
        "--defaults", action="store_true", help="also write out every default value"
    )
    parser.add_argument(
        "--for",
        dest="for_command",
        metavar="COMMAND",
        help="additionally require the sections a command needs (train, test, …)",
    )
    parser.add_argument(
        "--list",
        dest="list_registry",
        nargs="?",
        const="all",
        metavar="REGISTRY",
        help="list registered components (of one registry, or all)",
    )
    parser.set_defaults(handler=run)


def _components(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, Mapping):
        if "type" in node and path:
            yield path, str(node["type"])
        for key, value in node.items():
            yield from _components(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _components(value, f"{path}.{index}")


def _origin(obj: Any) -> str:
    """Module of a component; the file path for modules loaded from a ``.py`` path."""
    module = sys.modules.get(obj.__module__)
    file = getattr(module, "__file__", None)
    if obj.__module__.startswith("grasppanda_custom_") and file:
        try:
            return str(Path(file).relative_to(Path.cwd()))
        except ValueError:
            return file
    return obj.__module__


def _list(which: str) -> int:
    registries = all_registries()
    selected = (
        registries
        if which == "all"
        else {k: v for k, v in registries.items() if k.lower() == which.lower()}
    )
    if not selected:
        print(
            f"unknown registry '{which}'. Registries: {', '.join(n.lower() for n in registries)}",
            file=sys.stderr,
        )
        return 2
    rows = []
    for name, registry in selected.items():
        for component in registry.names():
            obj = registry.get(component)
            rows.append([name, component, getattr(obj, "status", "—"), _origin(obj)])
        if which != "all" and not registry.names():
            print(f"{name} has no registered components yet.")
            return 0
    print(table(["registry", "component", "status", "module"], rows))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.list_registry:
        if args.config:  # make out-of-tree components of that config visible too
            Config.load(args.config, args.opts, validate=False).validate()
        return _list(args.list_registry)
    if not args.config:
        print("print-config needs a config file or name (or --list).", file=sys.stderr)
        return 2

    config = Config.load(
        args.config, args.opts, validate=not args.no_check, command=args.for_command
    )
    color = sys.stderr.isatty()
    for drop in config.dropped:
        print(
            style("note: ", "yellow", enabled=color)
            + f"{drop.path} changed type {drop.old_type} → {drop.new_type}; "
            f"dropped inherited keys: {', '.join(drop.keys)}",
            file=sys.stderr,
        )
    if args.check:
        components = list(_components(config.to_dict()))
        print(style("✓ ", "bold", "green") + style(str(config.path), "bold") + " is valid")
        if len(config.sources) > 1:
            print(f"  inherits    {', '.join(p.name for p in config.sources[1:])}")
        if config.overrides:
            print(f"  overrides   {' '.join(config.overrides)}")
        if components:
            print(f"  components  {' · '.join(f'{p}: {t}' for p, t in components)}")
        print(f"  hash        {config.digest()[:12]}")
        return 0
    sys.stdout.write(highlight_yaml(config.dump(defaults=args.defaults)))
    return 0
