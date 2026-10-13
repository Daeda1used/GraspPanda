"""Recipe presets: the part of a config around the model, one file per choice.

A stage preset (:mod:`grasppanda.combine.presets`) fills a slot of the detector. A recipe
preset fills a slot of the *training recipe*, which any detector has, HGGD included::

    grasppanda compose <config> --set optimizer=timm-lamb --set schedule=onecycle \\
        --set hooks=ema+freeze-encoder --set augment=jitter

==========  =====================================================================
slot        what a file under ``grasppanda/recipes/<slot>/`` may set
==========  =====================================================================
optimizer   ``optimizer`` and what it needs with it (``scheduler``, ``hooks``, ``train``)
schedule    ``scheduler`` and ``train.scheduler_unit``
hooks       ``hooks`` (appended to those of the base config), ``train``, ``evaluation``
augment     ``transforms``: steps put before (``prepend``) or after (``append``) the
            point transforms of the training dataset
recipe      any of the above: a whole recipe under one name
==========  =====================================================================

``hooks`` and ``augment`` take several names joined by ``+``. A file has the header of a
stage preset: a summary paragraph, free text, then ``What must be re-tuned, and why:``
with ``*`` bullets. Compose writes the values into the child config; nothing here decides
whether a recipe suits a method: the config check and ``grasppanda probe`` do.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from grasppanda.config import parse_yaml

from .presets import RETUNE_HEADING, PresetError, _bullets, _header, _inline_comments, _paragraphs

__all__ = ["RECIPE_ROOT", "RECIPE_SLOTS", "Recipe", "apply", "find", "recipes", "split"]

#: Directory of the shipped recipe presets (package data).
RECIPE_ROOT = Path(__file__).resolve().parents[1] / "recipes"
#: Slot → the top-level keys a file of that slot may set.
RECIPE_SLOTS: dict[str, frozenset[str]] = {
    "optimizer": frozenset({"optimizer", "scheduler", "hooks", "train"}),
    "schedule": frozenset({"scheduler", "train"}),
    "hooks": frozenset({"hooks", "train", "evaluation"}),
    "augment": frozenset({"transforms"}),
    "recipe": frozenset({"optimizer", "scheduler", "hooks", "train", "evaluation", "transforms"}),
}
#: Slots that take several names joined by ``+``.
_SEVERAL = ("hooks", "augment")


@dataclass(frozen=True)
class Recipe:
    """One recipe preset file; the attributes a stage preset has, plus ``body``."""

    slot: str
    name: str
    path: Path
    summary: str
    about: tuple[str, ...]
    retune: tuple[str, ...]
    body: dict[str, Any]
    comments: dict[tuple[str, ...], str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """What the file configures, for messages: the component types it names."""
        found = []
        for key in ("optimizer", "scheduler"):
            node = self.body.get(key)
            if isinstance(node, Mapping):
                found.append(str(node.get("type")))
        found += [str(node.get("type")) for node in self.body.get("hooks") or []]
        steps = self.body.get("transforms") or {}
        found += [str(n.get("type")) for key in ("prepend", "append") for n in steps.get(key, [])]
        return ", ".join(found) or "train options"

    @property
    def location(self) -> str:
        return f"grasppanda/recipes/{self.slot}/{self.path.name}"


def read(path: Path, slot: str | None = None) -> Recipe:
    """Read and check one recipe preset; ``slot`` defaults to the name of its directory."""
    slot = slot or path.parent.name
    header, text = _header(path.read_text(encoding="utf-8"))
    where = f"recipe preset {path}"
    try:
        body = parse_yaml(text)
    except Exception as error:
        raise PresetError(f"{where}: invalid YAML: {error}") from None
    allowed = RECIPE_SLOTS[slot]
    if not isinstance(body, dict) or not body or set(body) - allowed:
        raise PresetError(f"{where} may set {', '.join(sorted(allowed))} and nothing else")
    for key in ("optimizer", "scheduler"):
        node = body.get(key, {})
        if node is not None and not (
            isinstance(node, dict) and (key not in body or "type" in node)
        ):
            raise PresetError(f"{where}: `{key}` must be a component node with a `type`, or null")
    hooks = body.get("hooks", [])
    if not isinstance(hooks, list) or not all(isinstance(n, dict) and "type" in n for n in hooks):
        raise PresetError(f"{where}: `hooks` must be a list of component nodes")
    steps = body.get("transforms", {})
    if not isinstance(steps, dict) or set(steps) - {"prepend", "append"}:
        raise PresetError(f"{where}: `transforms` may hold `prepend` and `append` lists")
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
    return Recipe(
        slot=slot,
        name=path.stem,
        path=path,
        summary=" ".join(paragraphs[0].split()),
        about=tuple(_paragraphs(header[: notes[0]])[1:]),
        retune=retune,
        body=body,
        comments=_inline_comments(text),
    )


@lru_cache(maxsize=1)
def recipes() -> dict[str, dict[str, Recipe]]:
    """Every shipped recipe preset: slot → name → :class:`Recipe`."""
    found: dict[str, dict[str, Recipe]] = {}
    for slot in RECIPE_SLOTS:
        directory = RECIPE_ROOT / slot
        if directory.is_dir():
            found[slot] = {path.stem: read(path) for path in sorted(directory.glob("*.yaml"))}
    return found


def find(slot: str, name: str) -> Recipe:
    """The recipe preset ``name`` of ``slot``; :class:`PresetError` with candidates otherwise."""
    members = recipes().get(slot, {})
    if name in members:
        return members[name]
    close = difflib.get_close_matches(name, list(members), n=3, cutoff=0.5)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    listed = ", ".join(members) or "none"
    raise PresetError(f"no recipe preset '{name}' for slot '{slot}'.{hint} Presets: {listed}.")


def split(sets: Sequence[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """``--set`` items that address a stage of the detector, and ``(slot, names)`` of the recipe."""
    stages: list[str] = []
    recipe: list[tuple[str, str]] = []
    for item in sets:
        slot, separator, request = item.partition("=")
        if separator and slot.strip() in RECIPE_SLOTS:
            if any(slot.strip() == other for other, _ in recipe):
                raise PresetError(
                    f"slot '{slot.strip()}' is set twice"
                    + (": join the names with `+`" if slot.strip() in _SEVERAL else "")
                )
            recipe.append((slot.strip(), request.strip()))
        else:
            stages.append(item)
    return stages, recipe


def resolve(slot: str, request: str) -> list[Recipe]:
    names = request.split("+") if slot in _SEVERAL else [request]
    return [find(slot, name.strip()) for name in names]


def _transforms(written: Mapping[str, Any]) -> list[Any] | None:
    dataset = ((written.get("data") or {}).get("train") or {}).get("dataset") or {}
    steps = dataset.get("transforms")
    return list(steps) if isinstance(steps, list) else None


def apply(
    chosen: Sequence[Recipe],
    base: Mapping[str, Any],
    body: dict[str, Any],
    comments: dict[tuple[str, ...], str],
) -> list[str]:
    """Write recipe presets into the child config ``body``; returns what was done.

    ``base`` is the base config as merged from its files (not resolved): the hooks and
    the point transforms of the base are lists, which a child config replaces wholesale,
    so they are carried over and the preset's entries are added.
    """
    actions: list[str] = []
    for recipe in chosen:
        actions.append(f"{recipe.slot}: recipe preset `{recipe.name}`. {recipe.summary}")
        for key in ("optimizer", "scheduler"):
            if key not in recipe.body:
                continue
            node = copy.deepcopy(recipe.body[key])
            inherited = base.get(key)
            if (
                isinstance(node, dict)
                and isinstance(inherited, Mapping)
                and inherited.get("type") == node.get("type")
            ):
                # Same type as the base: the loader would merge the two nodes.
                node = {"_delete_": True, **node}
                comments[(key, "_delete_")] = f"replace the base config's {node['type']}"
            body[key] = node
            if node is None:
                actions.append(f"`{key}: null`: the base config's {key} is removed.")
        for key in ("train", "evaluation"):
            if key in recipe.body:
                body.setdefault(key, {}).update(copy.deepcopy(recipe.body[key]))
        if recipe.body.get("hooks"):
            new = copy.deepcopy(recipe.body["hooks"])
            kinds = {node["type"] for node in new}
            kept = body["hooks"] if "hooks" in body else copy.deepcopy(base.get("hooks") or [])
            body["hooks"] = [node for node in kept if node.get("type") not in kinds] + new
            if base.get("hooks"):
                actions.append(
                    "hooks: a list replaces the inherited one, so the hooks of the base config "
                    "are written out again, followed by those of the preset."
                )
        steps = recipe.body.get("transforms")
        if steps:
            dataset = body.setdefault("data", {}).setdefault("train", {}).setdefault("dataset", {})
            current = dataset.get("transforms") or _transforms(base)
            if current is None:
                raise PresetError(
                    f"recipe preset '{recipe.name}' adds point transforms, but the training "
                    "dataset of the base config has no `transforms` list."
                )
            dataset["transforms"] = [
                *copy.deepcopy(steps.get("prepend", [])),
                *copy.deepcopy(current),
                *copy.deepcopy(steps.get("append", [])),
            ]
            actions.append(
                "data.train.dataset.transforms: the list of the base config with the "
                "preset's steps added (a list replaces the inherited one)."
            )
        for path, text in recipe.comments.items():
            if path[0] in ("optimizer", "scheduler", "train", "evaluation"):
                comments[path] = text
    return list(dict.fromkeys(actions))
