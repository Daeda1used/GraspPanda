"""``grasppanda explore``: the combination explorer in a browser, with the real compose check.

    grasppanda explore                 # http://127.0.0.1:8765/, opened in the browser
    grasppanda explore --port 9000 --no-browser

The page is the explorer of the documentation: pick a method, replace the option of a
slot, read what was measured. Served by this command it also composes every selection with
this installation, so a combination nobody measured gets the verdict of the contract rules,
and an accepted one can be saved as a config into the working directory. Nothing is
trained; the page prints the ``grasppanda probe`` command. The server listens on the
loopback address only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

__all__ = ["register"]


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "explore",
        help="try combinations in a browser, checked by the real compose",
        description=(
            "Serve the combination explorer on localhost. Every selection is composed and "
            "checked by this installation; an accepted combination can be saved as a config "
            "into the working directory. Nothing is trained."
        ),
    )
    parser.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (0: any free)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--save-dir", metavar="DIR", help="where composed configs are saved (default: here)"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from grasppanda.combine.explore import serve

    directory = Path(args.save_dir) if args.save_dir else None
    if directory is not None and not directory.is_dir():
        print(f"✗ {directory} is not a directory", file=sys.stderr)
        return 2
    try:
        return serve(args.port, directory, open_browser=not args.no_browser)
    except (FileNotFoundError, OSError) as error:
        print(f"✗ {error}", file=sys.stderr)
        return 2
