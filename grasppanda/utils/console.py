"""Terminal output helpers: colour, tables and YAML highlighting without dependencies.

Colour is used only on a TTY and honours ``NO_COLOR`` / ``FORCE_COLOR``.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from typing import TextIO

__all__ = ["banner", "highlight_yaml", "style", "supports_color", "table"]

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "gray": "90",
}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def supports_color(stream: TextIO | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


def style(text: str, *styles: str, enabled: bool | None = None) -> str:
    """Wrap ``text`` in ANSI styles when colour is enabled."""
    if enabled is None:
        enabled = supports_color()
    if not enabled or not styles:
        return text
    return f"\x1b[{';'.join(_CODES[s] for s in styles)}m{text}\x1b[0m"


def _width(text: str) -> int:
    return len(_ANSI.sub("", text))


def table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, enabled: bool | None = None
) -> str:
    """Render a plain aligned table with a dim rule under the header."""
    columns = [list(column) for column in zip(headers, *rows)] if rows else [[h] for h in headers]
    widths = [max(_width(cell) for cell in column) for column in columns]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            cell + " " * (width - _width(cell)) for cell, width in zip(cells, widths)
        ).rstrip()

    out = [line([style(h, "bold", enabled=enabled) for h in headers])]
    out.append(style("  ".join("─" * width for width in widths), "gray", enabled=enabled))
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def banner(version: str, *, enabled: bool | None = None) -> str:
    """One-line product header for ``--version`` and command output."""
    name = style("GraspPanda", "bold", enabled=enabled)
    tagline = style("6-DoF grasp detection in clutter", "gray", enabled=enabled)
    return f"ʕ•ᴥ•ʔ  {name} {version}  ·  {tagline}"


_YAML_LINE = re.compile(r"^(?P<indent>\s*(?:- )*)(?P<key>[^\s:#][^:#]*)(?P<colon>:)(?P<rest>.*)$")


def highlight_yaml(text: str, *, enabled: bool | None = None) -> str:
    """Colour YAML keys, and component names after ``type:``, for terminal display."""
    if enabled is None:
        enabled = supports_color()
    if not enabled:
        return text
    out = []
    for raw in text.splitlines():
        match = _YAML_LINE.match(raw)
        if not match:
            out.append(raw)
            continue
        key, rest = match["key"], match["rest"]
        if key == "type":
            rest = style(rest, "bold", "green", enabled=True)
        out.append(f"{match['indent']}{style(key, 'cyan', enabled=True)}{match['colon']}{rest}")
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")
