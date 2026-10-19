"""Shared vocabulary of the documentation generators.

Every generator that prints a method list, the scope of a slot or where an option works
takes it from here, so the pages say the same thing in the same words as the explorer and
``grasppanda compose --list`` (which read :mod:`grasppanda.combine.reach` directly). Nothing
in this module is typed by hand from a record.

* methods: :func:`methods` (page order: the seed-based skeleton by year, then HGGD; venue and
  title from ``configs/<method>/metafile.yaml``);
* the scope of a slot: :func:`scope` / :func:`scope_chip` (``All 6 methods``,
  ``Seed-based · 5 methods``, ``HGGD only``) and :func:`scope_matrix` (slot × method);
* where an option works: :func:`reach_chip` (level 1) and :func:`level2_chip` (level 2), with
  their plain-text tails as tooltips;
* where things are: :data:`SLOT_TITLES`, :data:`SLOT_PAGES`;
* what was measured on: :func:`measured_on`.

The API of this module is frozen while the pages are rebuilt: add, do not rename.
"""

from __future__ import annotations

import functools
import html
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from grasppanda.combine import reach as reaching

ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / "docs" / "advanced_guides" / "measured_combinations.json"
LEVEL2 = ROOT / "docs" / "advanced_guides" / "learns_like_reference.json"

#: Family → the words a page uses for it.
FAMILY_NAMES = {"seed-based": "Seed-based skeleton", "own": "Own family"}
#: Scope key → chip text.
SCOPE_TEXT = {
    "all": "All 6 methods",
    "seed-based": "Seed-based · 5 methods",
    "hggd": "HGGD only",
}
#: Slot → what the pages call it.
SLOT_TITLES = {
    "image_encoder": "Image encoder",
    "backbone": "Scene encoder",
    "proposal": "Seed proposal",
    "view": "View",
    "assigner": "Assigner",
    "grouper": "Grouper",
    "head": "Head bundle",
    "loss": "Loss",
    "post": "Post-processing",
    "optimizer": "Optimizer",
    "schedule": "Schedule",
    "hooks": "Hooks",
    "augment": "Augmentation",
    "recipe": "Recipe",
}
#: Slot → the page that documents its options (relative to ``docs/``).
SLOT_PAGES = {
    "image_encoder": "slots/image_encoders.md",
    "backbone": "slots/scene_encoders.md",
    "proposal": "slots/proposal.md",
    "view": "slots/view.md",
    "assigner": "slots/head_loss.md",
    "grouper": "slots/grouper.md",
    "head": "slots/head_loss.md",
    "loss": "slots/losses.md",
    "post": "slots/post.md",
    "optimizer": "slots/recipe.md",
    "schedule": "slots/recipe.md",
    "hooks": "slots/recipe.md",
    "augment": "slots/recipe.md",
    "recipe": "slots/recipe.md",
}
#: The slots in the order data flows, then the training recipe.
SLOT_ORDER = tuple(SLOT_TITLES)


@dataclass(frozen=True)
class MethodInfo:
    """One ported method as the pages present it.

    Attributes:
        name: The name used in tables (``GraspNet-baseline``).
        config: The shipped config, relative to ``configs/``, without ``.yaml``.
        folder: The folder under ``configs/`` and the stem of ``docs/methods/<stem>.md``.
        family: ``seed-based`` or ``own``.
        venue: As the metafile states it (``CVPR 2020``).
        year: The year of the venue.
        title: The paper title.
        inputs: ``depth`` or ``RGB-D``.
    """

    name: str
    config: str
    folder: str
    family: str
    venue: str
    year: int
    title: str
    inputs: str

    @property
    def page(self) -> str:
        """The method page, relative to ``docs/``."""
        return f"methods/{self.folder}.md"


@functools.lru_cache(maxsize=1)
def methods() -> tuple[MethodInfo, ...]:
    """The six methods in page order: the seed-based skeleton by year, then HGGD."""
    from grasppanda.combine.grid import FAMILIES, METHODS

    found = []
    for name, config in [*METHODS.items(), *FAMILIES.items()]:
        folder = config.split("/")[0]
        meta = yaml.safe_load((ROOT / "configs" / folder / "metafile.yaml").read_text())
        venue = str(meta["venue"])
        found.append(
            MethodInfo(
                name=name,
                config=config,
                folder=folder,
                family="seed-based" if name in METHODS else "own",
                venue=venue,
                year=int(venue.split()[-1]),
                title=str(meta.get("title", name)),
                inputs="RGB-D" if name in FAMILIES else "depth",
            )
        )
    return tuple(found)


def method(name: str) -> MethodInfo:
    return next(info for info in methods() if info.name == name)


@functools.cache
def slot_methods(slot: str) -> tuple[str, ...]:
    """The methods that have ``slot``, in page order."""
    from grasppanda.combine.options import method_slots

    return tuple(info.name for info in methods() if slot in method_slots(info.name))


def scope(slot: str) -> str:
    """``all``, ``seed-based`` or ``hggd``: the exact set of methods ``slot`` applies to.

    Raises:
        ValueError: For a slot whose methods are none of these sets.
    """
    under = set(slot_methods(slot))
    every = {info.name for info in methods()}
    seed = {info.name for info in methods() if info.family == "seed-based"}
    for key, members in (("all", every), ("seed-based", seed), ("hggd", every - seed)):
        if under == members:
            return key
    raise ValueError(f"slot {slot!r} applies to {sorted(under)}, which is no scope of the pages")


def chip(text: str, tone: str = "", title: str = "") -> str:
    """A chip: ``<span class="gp-chip gp-chip--ok">5/5</span>`` (tone ``ok``, ``warn``,
    ``no``, ``muted`` or none); ``title`` becomes its tooltip."""
    kind = f"gp-chip gp-chip--{tone}" if tone else "gp-chip"
    tip = f' title="{html.escape(title, quote=True)}"' if title else ""
    return f'<span class="{kind}"{tip}>{html.escape(text)}</span>'


def scope_chip(slot: str) -> str:
    return chip(SCOPE_TEXT[scope(slot)])


def reach(slot: str, name: str, level: str = "runs") -> reaching.Reach:
    """Where option ``name`` of ``slot`` reached ``level`` (:mod:`grasppanda.combine.reach`)."""
    return reaching.reach(slot, name, level)


def reach_chip(slot: str, name: str, level: str = "runs") -> str:
    """The fraction as a chip, with the tail (who is left out and why) as its tooltip."""
    found = reach(slot, name, level)
    return chip(found.fraction, found.tone, found.tail)


def level2(slot: str, name: str) -> reaching.Level2:
    return reaching.level2(slot, name)


def level2_chip(slot: str, name: str) -> str:
    found = level2(slot, name)
    return chip(found.fraction, found.tone, found.tail)


def scope_matrix(link: str = "") -> list[str]:
    """Markdown lines of the slot × method table: which method has which slot.

    Args:
        link: The prefix from the page being written to ``docs/`` (``../`` for a page one
            level down), for the links to the slot pages; ``""`` for no links.
    """
    names = [info.name for info in methods()]
    header = ["Slot", "Applies to", *names]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for slot in SLOT_ORDER:
        if not slot_methods(slot):
            continue
        title = SLOT_TITLES[slot]
        label = f"[{title}]({link}{SLOT_PAGES[slot]})" if link else title
        marks = ["✓" if name in slot_methods(slot) else "—" for name in names]
        lines.append("| " + " | ".join([label, SCOPE_TEXT[scope(slot)], *marks]) + " |")
    return lines


@functools.lru_cache(maxsize=1)
def _level2_settings() -> dict:
    if not LEVEL2.is_file():
        return {}
    return dict(json.loads(LEVEL2.read_text()).get("about", {}).get("settings", {}))


def measured_on(level: int = 1) -> str:
    """One sentence that says what the records were measured on.

    Level 1: every probe of the grid (the iterations and frames of the reference cells);
    level 2: the settings of ``learns_like_reference.json``.
    """
    if level == 2:
        settings = _level2_settings()
        return (
            f"{settings.get('iterations', 250)} iterations on {settings.get('frames', 8)} "
            "RealSense frames of GraspNet-1Billion (one scene), seed "
            f"{settings.get('seed', 0)}, at a constant learning rate (FineGrasp: its own "
            "warm-up), beside the method as shipped trained the same way twice"
        )
    from grasppanda.combine.options import measured

    reference = measured().reference(methods()[0].name) or {}
    return (
        f"{reference.get('iterations', 20)} training iterations on "
        f"{reference.get('frames', 8)} RealSense frames of GraspNet-1Billion (one scene), "
        "seed 0, one run per cell; nothing was measured on GraspClutter6D or on Kinect frames"
    )


#: What the pages call level 2, and its one-line meaning.
LEVEL2_NAME = "trains like shipped"
LEVEL2_HEADER = "Trains like shipped"
LEVEL2_MEANING = (
    "trained 250 iterations beside the component the method ships, its losses fall as theirs do"
)
#: The option names that are not swap options: adapters compose inserts and the
#: point-painting wrappers (a config names them by hand).
_NOT_OPTIONS = ("an encoder adapter", "a wrapper around the method's encoder")


@functools.cache
def options_of(slot: str) -> tuple[str, ...]:
    """Every option of ``slot`` a user can ``--set``, in the order of the records: what
    every page counts. Adapters and point-painting wrappers are not counted (they are not
    swap options); a loss preset measured inside a head bundle is."""
    from grasppanda.combine.options import _class_index, _extra_options

    skipped = {
        (option.slot, option.name)
        for _, option in _extra_options()
        if option.reason.startswith(_NOT_OPTIONS)
    }
    names = [name for key_slot, name in _class_index() if key_slot == slot]
    names = [name for name in names if (slot, name) not in skipped]
    if slot == "loss":
        names += [name for name in reaching.LOSS_INSIDE if name not in names]
    return tuple(dict.fromkeys(names))


def option_count(slot: str | None = None) -> int:
    """How many options ``slot`` has (:func:`options_of`); all slots for ``None``."""
    if slot is not None:
        return len(options_of(slot))
    return sum(len(options_of(name)) for name in SLOT_ORDER)
