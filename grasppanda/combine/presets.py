"""Presets: the part of a config a newcomer cannot guess, one file per component variant.

A preset is a config fragment under ``grasppanda/presets/<slot>/<name>.yaml``: the node
``model.<slot>`` with the settings that make sense for table-top depth clouds of
GraspNet-1Billion and, where the stage reads a label asset, the section of the training
dataset that serves it (``data.train.dataset.<section>``). Its leading comment says what
the preset is and what must be re-tuned, and why::

    # One sentence that says what this is.
    #
    # Free text.
    #
    # What must be re-tuned, and why:
    # * option: value, reason ...
    model:
      backbone:
        type: PTv3                    # a comment that travels with the line
        grid_size: 0.005

Presets are not configs and are not inherited: the config loader refuses two sibling
bases that set the same value, and a preset by definition sets what the method's own
model already sets. They are applied by ``grasppanda compose``, which writes their values
into the child config, where an override belongs.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from grasppanda.config import TYPE_KEY, parse_yaml

__all__ = [
    "LOSS_SLOT",
    "PRESET_ROOT",
    "RETUNE_HEADING",
    "Preset",
    "PresetError",
    "find",
    "presets",
    "read",
]

#: Directory of the shipped presets (package data).
PRESET_ROOT = Path(__file__).resolve().parents[1] / "presets"
#: The line of a preset header after which its re-tuning notes are listed.
RETUNE_HEADING = "What must be re-tuned, and why:"
#: The slot of post-processing presets. Their files hold a config fragment,
#: ``evaluation.variants.post: [<one node>]``, instead of ``model.<slot>``.
POST_SLOT = "post"
#: The slot of loss presets. Their files hold one loss node, ``model.loss``, which
#: ``grasppanda compose --set <stage>.<loss option>=<name>`` puts into a loss option of a
#: stage (``head.score_loss``, ``proposal.objectness_loss``).
LOSS_SLOT = "loss"

_KEY = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][\w-]*):(?P<rest>.*)$")


class PresetError(ValueError):
    """A preset file is malformed, or a preset was asked for that does not exist."""


@dataclass(frozen=True)
class Preset:
    """One preset file.

    Attributes:
        slot: The slot of the detector it fills (``backbone``, ``grouper``, …).
        name: Its name, the file name without extension (``ptv3``).
        path: The file.
        summary: First paragraph of the header: what the preset is.
        about: The header paragraphs between the summary and the re-tuning notes.
        retune: The re-tuning notes, one entry per bullet, lines as written.
        node: The component node, ``model.<slot>``.
        data: Sections of ``data.train.dataset`` the stage reads; empty for most.
        comments: Inline comment of every commented line, by key path from the file
            root (``("model", "backbone", "type")``).
    """

    slot: str
    name: str
    path: Path
    summary: str
    about: tuple[str, ...]
    retune: tuple[str, ...]
    node: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)
    comments: dict[tuple[str, ...], str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """Registered name of the component the preset configures."""
        return str(self.node[TYPE_KEY])

    @property
    def location(self) -> str:
        """Where the file lives inside the package, for messages."""
        return f"grasppanda/presets/{self.slot}/{self.path.name}"


def _header(text: str) -> tuple[list[str], str]:
    """Split a file into the lines of its leading comment (without ``#``) and the rest."""
    lines = text.splitlines()
    count = 0
    while count < len(lines) and lines[count].startswith("#"):
        count += 1
    header = [line[1:].removeprefix(" ").rstrip() for line in lines[:count]]
    return header, "\n".join(lines[count:])


def _paragraphs(lines: list[str]) -> list[str]:
    out: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            out[-1].append(line)
        elif out[-1]:
            out.append([])
    return ["\n".join(block) for block in out if block]


def _bullets(lines: list[str]) -> tuple[str, ...]:
    items: list[list[str]] = []
    for line in lines:
        if line.startswith("* "):
            items.append([line[2:]])
        elif line.strip() and items:
            items[-1].append(line.strip())
    return tuple("\n".join(item) for item in items)


def _inline_comments(body: str) -> dict[tuple[str, ...], str]:
    """Inline comments of a block-style YAML mapping, by key path."""
    comments: dict[tuple[str, ...], str] = {}
    stack: list[tuple[int, str]] = []
    for line in body.splitlines():
        match = _KEY.match(line)
        if match is None:
            continue
        indent = len(match["indent"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, match["key"]))
        _, mark, comment = match["rest"].partition(" #")
        if mark and comment.strip():
            comments[tuple(key for _, key in stack)] = comment.strip()
    return comments


def _post_as_model(content: Any, where: str) -> dict[str, Any]:
    """A post-processing preset, ``evaluation.variants.post: [node]``, in the shape of
    a stage preset (``model.post: node``), so that one reader serves both."""
    try:
        (section,), (variants,) = list(content), list(content.values())
        (group,), (chains,) = list(variants), list(variants.values())
        (variant,), (chain,) = list(chains), list(chains.values())
        (node,) = chain
    except (TypeError, ValueError, AttributeError):
        section = group = variant = node = None
    if (section, group, variant) != ("evaluation", "variants", POST_SLOT):
        raise PresetError(
            f"{where} must hold exactly `evaluation.variants.{POST_SLOT}` with one "
            "post-processor node in its list"
        )
    return {"model": {POST_SLOT: node}}


def read(path: Path, slot: str | None = None) -> Preset:
    """Read and check one preset file; ``slot`` defaults to the name of its directory."""
    slot = slot or path.parent.name
    text = path.read_text(encoding="utf-8")
    header, body = _header(text)
    where = f"preset {path}"
    try:
        content = parse_yaml(body)
    except Exception as error:
        raise PresetError(f"{where}: invalid YAML: {error}") from None
    if slot == POST_SLOT:
        content = _post_as_model(content, where)
    if not isinstance(content, dict) or set(content) - {"model", "data"}:
        raise PresetError(f"{where} must hold `model:` and optionally `data:`, nothing else")
    model = content.get("model")
    if not isinstance(model, dict) or list(model) != [slot]:
        raise PresetError(f"{where} must set exactly `model.{slot}`, got {model!r}")
    node = model[slot]
    if not isinstance(node, dict) or not isinstance(node.get(TYPE_KEY), str):
        raise PresetError(f"{where}: `model.{slot}` must be a component node with a `type`")
    data: dict[str, Any] = {}
    if "data" in content:
        train = content["data"].get("train") if isinstance(content["data"], dict) else None
        dataset = train.get("dataset") if isinstance(train, dict) else None
        only = list(content["data"] or {}) == ["train"] and list(train or {}) == ["dataset"]
        if not only or not isinstance(dataset, dict) or not dataset:
            raise PresetError(f"{where}: `data` may only set sections of `data.train.dataset`")
        data = dataset
    paragraphs = _paragraphs(header)
    notes = [index for index, line in enumerate(header) if line.strip() == RETUNE_HEADING]
    if not paragraphs or len(notes) != 1:
        raise PresetError(
            f"{where} needs a leading comment: a summary paragraph and one line "
            f"'{RETUNE_HEADING}' followed by '* ' bullets"
        )
    retune = _bullets(header[notes[0] + 1 :])
    if not retune:
        raise PresetError(f"{where}: no '* ' bullet follows '{RETUNE_HEADING}'")
    return Preset(
        slot=slot,
        name=path.stem,
        path=path,
        summary=" ".join(paragraphs[0].split()),
        about=tuple(_paragraphs(header[: notes[0]])[1:]),
        retune=retune,
        node=node,
        data=data,
        comments=_inline_comments(body),
    )


@lru_cache(maxsize=1)
def presets() -> dict[str, dict[str, Preset]]:
    """Every shipped preset: slot → name → :class:`Preset`, both sorted."""
    found: dict[str, dict[str, Preset]] = {}
    if not PRESET_ROOT.is_dir():
        return found
    for directory in sorted(path for path in PRESET_ROOT.iterdir() if path.is_dir()):
        members = {path.stem: read(path) for path in sorted(directory.glob("*.yaml"))}
        if members:
            found[directory.name] = members
    return found


def find(slot: str, name: str) -> Preset:
    """The preset ``name`` of ``slot``; :class:`PresetError` with candidates otherwise."""
    members = presets().get(slot, {})
    if name in members:
        return members[name]
    close = difflib.get_close_matches(name, list(members), n=3, cutoff=0.5)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    listed = ", ".join(members) or "none"
    raise PresetError(f"no preset '{name}' for slot '{slot}'.{hint} Presets: {listed}.")
