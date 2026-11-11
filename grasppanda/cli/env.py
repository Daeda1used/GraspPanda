"""``grasppanda env``: environment report to paste into bug reports.

Below the report it says what on this machine keeps a shipped method from running, with
the command that fixes it: the first thing to read after an install.
"""

from __future__ import annotations

import argparse
import json
import shutil
from typing import Any

from grasppanda import __version__
from grasppanda.utils.console import banner, style, table
from grasppanda.utils.env import collect_env
from grasppanda.utils.imports import install_hint

__all__ = ["notes", "register"]


def notes(env: dict[str, Any]) -> list[str]:
    """What this environment lacks for the shipped methods, each with its fix.

    Two things go wrong after an install that follows no index: a PyTorch built for a
    CUDA version the driver does not run sees no GPU, and sparse convolution has no build
    for the newest CUDA versions.
    """
    found = []
    build = env["cuda_build"]
    if build and not env["cuda_available"]:
        present = (
            " (`nvidia-smi` is installed, so there is one)" if shutil.which("nvidia-smi") else ""
        )
        found.append(
            f"PyTorch is built for CUDA {build} and sees no GPU{present}. A driver older than "
            f"CUDA {build} does not run it: the header of `nvidia-smi` names the newest CUDA "
            "version yours does. Install a PyTorch built for that version or an older one, "
            "for example `pip install torch --index-url https://download.pytorch.org/whl/cu126`."
        )
    if not env["optional"].get("spconv"):
        found.append(
            "spconv is not installed: GSNet, EconomicGrasp, FineGrasp and the scene encoders "
            "built on sparse convolution need it (GraspNet-baseline, Scale-Balanced-Grasp and "
            f"HGGD do not). {install_hint('spconv')}"
        )
    return found


def register(commands: Any) -> None:
    parser = commands.add_parser("env", help="print the environment report used in bug reports")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    env = collect_env()
    found = notes(env)
    if args.json:
        print(json.dumps({**env, "notes": found}, indent=2))
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
    print()
    if not found:
        ready = "a GPU is visible" if env["cuda_available"] else "no GPU: the CPU runs everything"
        print(style("✓ ", "bold", "green") + f"{ready}, and sparse convolution is installed")
    for note in found:
        print(style("! ", "bold", "yellow") + note)
    return 0
