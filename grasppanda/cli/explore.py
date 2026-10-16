"""``grasppanda explore``: the explorer in a browser, with the real compose check and a Run panel.

    grasppanda explore                                # http://127.0.0.1:8765/, in the browser
    grasppanda explore --port 9000 --no-browser
    grasppanda explore --weights-dir ~/weights        # the Run panel lists the files there
    grasppanda explore --weights gsnet=runs/gsnet/epoch_0010.pth --device cpu

The page is the explorer of the documentation: pick a method, replace the option of a
slot, read what was measured. Served by this command it also composes every selection with
this installation, so a combination nobody measured gets the verdict of the contract rules,
and an accepted one can be saved as a config into the working directory.

Its Run panel runs a ported method, or the composed selection, on the built-in example
scene or on an uploaded depth image, with a weights file the command line named, and shows
the picture, the best grasps and the counts through the post-processing that ``grasppanda
demo`` prints. Nothing is trained; the page prints the ``grasppanda probe`` command. The
server listens on the loopback address only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

__all__ = ["register"]

_EPILOG = """\
The Run panel reads weights only from files named here: every .pth, .pt, .tar or .ckpt
file under --weights-dir (a few folders deep), and each --weights NAME=PATH. The page
picks one of them; it cannot name another path. A run takes one frame (the example scene,
or a depth image of at most 20 MB with its intrinsics fx, fy, cx, cy, an optional colour
image and an optional workspace box), predicts, and draws the result as `grasppanda demo`
does; one run at a time, in a temporary directory that is removed afterwards.

examples:
  grasppanda explore --weights-dir ~/grasppanda_weights
  grasppanda explore --weights gsnet=work_dirs/gsnet/run/checkpoints/epoch_0010.pth \\
      --weights hggd=hggd_realsense.pth --device cpu
"""


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "explore",
        help="try combinations and run ported methods in a browser",
        description=(
            "Serve the explorer on localhost. Every selection is composed and checked by\n"
            "this installation; an accepted combination can be saved as a config into the\n"
            "working directory. With weights (--weights-dir or --weights), its Run panel runs\n"
            "a ported method, or the composed selection, on the example scene or on your\n"
            "depth image, and shows the picture and the best grasps. Nothing is trained."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (0: any free)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--save-dir", metavar="DIR", help="where composed configs are saved (default: here)"
    )
    panel = parser.add_argument_group("the Run panel")
    panel.add_argument(
        "--weights-dir",
        metavar="DIR",
        help="list the weights files under DIR in the Run panel (default: none)",
    )
    panel.add_argument(
        "--weights",
        metavar="NAME=PATH",
        action="append",
        default=[],
        help="offer this weights file under NAME (repeatable; a bare PATH: named after the file)",
    )
    panel.add_argument("--device", metavar="NAME", help="auto, cpu, cuda or cuda:N for the runs")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from grasppanda.combine.explore import serve
    from grasppanda.combine.explore_run import Runner, WeightsSource

    directory = Path(args.save_dir) if args.save_dir else None
    if directory is not None and not directory.is_dir():
        print(f"✗ {directory} is not a directory", file=sys.stderr)
        return 2
    try:
        weights = WeightsSource.from_arguments(args.weights_dir, args.weights)
    except ValueError as error:
        print(f"✗ {error}", file=sys.stderr)
        return 2
    runner = Runner(weights, device=args.device)
    try:
        return serve(args.port, directory, open_browser=not args.no_browser, runner=runner)
    except (FileNotFoundError, OSError) as error:
        print(f"✗ {error}", file=sys.stderr)
        return 2
