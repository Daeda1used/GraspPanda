"""Write a config fragment as YAML a person would have written: block mappings, flow
lists, a comment beside the lines that have something to say.

PyYAML drops comments, and the comments are the point of a composed config. This emitter
covers what a config fragment holds (mappings of scalars, lists and further mappings) and
:func:`dump` checks its own output: the text must parse back to the data it was given.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

from grasppanda.config import parse_yaml

__all__ = ["COMMENT_COLUMN", "comment_block", "dump", "flow"]

#: Column at which inline comments start when the line is short enough.
COMMENT_COLUMN = 34
KeyPath = tuple[str, ...]


def flow(value: Any) -> str:
    """A scalar, list or empty mapping on one line, as a config file spells it."""
    text = yaml.safe_dump(value, default_flow_style=True, sort_keys=False, width=10**6).strip()
    if text.endswith("\n..."):  # the document end marker PyYAML adds after a bare scalar
        text = text[:-4].rstrip()
    return text


def _line(code: str, comment: str | None) -> str:
    if not comment:
        return code
    gap = COMMENT_COLUMN - len(code)
    return f"{code}{' ' * gap if gap >= 2 else '   '}# {comment}"


def _block(
    data: Mapping[str, Any], comments: Mapping[KeyPath, str], path: KeyPath, indent: int
) -> list[str]:
    lines = []
    for key, value in data.items():
        here = (*path, str(key))
        pad = " " * indent
        if isinstance(value, Mapping) and value:
            lines.append(_line(f"{pad}{key}:", comments.get(here)))
            lines += _block(value, comments, here, indent + 2)
        else:
            lines.append(_line(f"{pad}{key}: {flow(value)}", comments.get(here)))
    return lines


def dump(data: Mapping[str, Any], comments: Mapping[KeyPath, str] | None = None) -> str:
    """``data`` as block-style YAML with ``comments`` beside the keys they belong to.

    Args:
        data: A mapping of scalars, lists and mappings. Lists are written in flow style.
        comments: Key path → one-line comment for that key.

    Returns:
        The text, with a blank line between top-level sections. It parses back to
        ``data`` with the config loader's YAML rules; anything else raises.
    """
    sections = [_block({key: value}, comments or {}, (), 0) for key, value in data.items()]
    text = "\n\n".join("\n".join(lines) for lines in sections) + "\n"
    if parse_yaml(text) != _plain(data):
        raise ValueError(f"cannot write this fragment as YAML faithfully: {data!r}")
    return text


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def comment_block(lines: list[str]) -> str:
    """Lines as a YAML comment block; an empty line becomes a bare ``#``."""
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)
