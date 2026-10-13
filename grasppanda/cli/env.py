"""``grasppanda env``: environment report to paste into bug reports."""

from __future__ import annotations

import argparse
import json
from typing import Any

from grasppanda import __version__
from grasppanda.utils.console import banner, table
from grasppanda.utils.env import collect_env

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser("env", help="print the environment report used in bug reports")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    env = collect_env()
    if args.json:
        print(json.dumps(env, indent=2))
        return 0
    rows = [
        ["python", env["python"]],
        ["platform", env["platform"]],
        [
            "torch",
            f"{env['torch']} (CUDA build {env['cuda_build'] or 'none'}, cuDNN {env['cudnn'] or 'none'})",
        ],
        ["numpy", env["numpy"]],
        ["GPUs", ", ".join(env["gpus"]) or ("none visible" if not env["cuda_available"] else "?")],
    ]
    rows += [[name, version or "not installed"] for name, version in env["optional"].items()]
    library = env["ops"].pop("library", {})
    for name, backend in env["ops"].items():
        used = ", ".join(
            f"{device} {backend[device]}" for device in ("cpu", "cuda") if device in backend
        )
        rows.append([f"ops.{name}", f"{used} (requested {backend['requested']})"])
    rows.append(["ops kernels", library.get("detail") or library.get("reason") or "not built"])
    print(banner(__version__))
    print()
    print(table(["component", "value"], rows))
    return 0
