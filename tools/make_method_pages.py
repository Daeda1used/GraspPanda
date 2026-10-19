"""Generate the method pages: one per method, the methods overview, and the chooser.

``docs/methods/<method>.md`` is a short page in four parts:

* the head: the family, the venue, what the method reads, that it equals its reference
  code (the parity test of :data:`PARITY`) and the status of its numbers (from the
  provenance of the metafile rows: trained here, released weights scored here, or
  reported only), then one sentence of what it predicts from what. Who wrote the
  reference code and released the checkpoint is read from ``upstream`` of the metafile
  (:func:`upstream_words`): the authors, or a third-party implementation (GSNet's);
* *Run <method> in five minutes*: the weights, the demo on the example scene, the same
  from Python, in the browser and on a camera image, and a first number on the benchmark.
  The facts come from the metafile (``configs/<method>/metafile.yaml``: weights, licence,
  ``convert``, the whole-split rows), from the config (the pipeline and the
  post-processing) and from ``docs/methods/five_minutes.json`` (the measured runs);
  what was printed, the machine, the dates and the checksums are folded;
* *Numbers*: at most four whole-split rows of the shipped config, with their provenance;
* *The options of every slot*: one summary table (per slot: what the config check accepts,
  what it refuses, and how many of the options tried as a swap ran under this method,
  counted as *Does it run?* counts them), then one fold per slot with every option, where
  it runs (:func:`tools.docs_chips.reach_chip`),
  whether it trains like the component each method ships
  (:func:`tools.docs_chips.level2_chip`) and what happened under this method, a refusal
  with the reason of its record. The lists come from :mod:`grasppanda.combine.options`, the
  module behind ``grasppanda compose <method config> --list``, so the page, the command and
  the explorer list the same options; the options counted are those of
  :func:`tools.docs_chips.options_of`.

How the port differs from its reference is read from the fold of the same name in
``configs/<method>/README.md``, so it is written once. ``docs/methods/index.md`` puts the
six methods side by side (cards by family, a method × capability table computed from the
configs and the metafiles), and ``docs/includes/choose_method.md`` is the table that helps
a reader choose, with the measured demo times of ``five_minutes.json`` (included by the
overview and by Get started).
Nothing measured is typed into this generator.

The slot pages (``docs/slots/``) are written by hand around generated regions, written here
too: the chips under the title, "Use it" (the slot under every method that has it, with an
option that ran there) and "Options" (one row per option: where it runs and how it learns,
from :mod:`grasppanda.combine.reach` through ``tools/docs_chips.py``).

    python -m tools.make_method_pages           # rewrite
    python -m tools.make_method_pages --check   # exit 1 and name what is stale

``tests/unit/test_method_options.py`` runs the check, so a stale page fails CI;
``tests/unit/test_method_quickstart.py`` parses every command of the five-minute sections
against the metafile and the command line, runs the demo, its camera variant and the
Python block on a toy detector of the method's family, and holds the measured runs to the
metafiles (:data:`TESTED`).
"""

from __future__ import annotations

import argparse
import functools
import html
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from grasppanda.combine import options as option_lists
from grasppanda.combine import reach as reaching
from grasppanda.combine.levels import BELOW_REFERENCE, NO, YES
from grasppanda.config import Config
from grasppanda.hub import Metafile, ModelEntry, Result, Weights, load_metafiles

REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "methods"
SLOT_PAGES = REPO / "docs" / "slots"
MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
#: The measured runs of the five-minute sections.
FIVE_MINUTES = PAGES / "five_minutes.json"
#: The overview of the methods and the chooser it shares with Get started.
INDEX = PAGES / "index.md"
CHOOSER = REPO / "docs" / "includes" / "choose_method.md"
#: Where a page links a file of the repository.
GITHUB = "https://github.com/Daeda1used/GraspPanda/blob/main"
#: The explorer, from a page one level below ``docs/`` (raw HTML: the site uses directory
#: URLs, and a fragment of the app is no anchor MkDocs could check).
EXPLORER = "../../explorer/"
#: The probe caveat, one per page (``docs/includes/caveat_probe.md``).
CAVEAT_SNIPPET = '--8<-- "docs/includes/caveat_probe.md"'
#: The level-2 page, whose title is the text of a link to it.
LEVEL2_PAGE = REPO / "docs" / "advanced_guides" / "learns_like_reference.md"

#: The parity test that holds each method to its reference code, stage by stage, with the
#: same weights loaded into both: the evidence behind "equal to the reference".
PARITY = {
    "GraspNet-baseline": "tests/integration/test_baseline_parity.py",
    "GSNet": "tests/integration/test_gsnet_parity.py",
    "Scale-Balanced-Grasp": "tests/integration/test_sbg_parity.py",
    "EconomicGrasp": "tests/integration/test_economicgrasp_parity.py",
    "FineGrasp": "tests/integration/test_finegrasp_parity.py",
    "HGGD": "tests/integration/test_hggd_parity.py",
}
#: What each method predicts from what: one sentence for its page, one line for its card.
WHAT = {
    "GraspNet-baseline": (
        "From the point cloud of a depth image, it keeps 1024 PointNet++ seeds, scores 300 "
        "approach views per seed and predicts score, in-plane angle, width and tolerance at "
        "four depths along the best one.",
        "The baseline released with GraspNet-1Billion: PointNet++ seeds, 300 views, four depths.",
    ),
    "GSNet": (
        "From the point cloud of a depth image, it scores the graspness of every point, keeps "
        "1024 seeds where it is high, picks one of 300 approach views per seed and regresses "
        "score and width over a grid of angles and depths.",
        "Graspness-guided seeds on a sparse U-Net, one of 300 views, a grid of angles and depths.",
    ),
    "Scale-Balanced-Grasp": (
        "From the point cloud of a depth image, it predicts grasps as GraspNet-baseline does, "
        "with point-transformer layers, crops at four radii and loss weights that favour "
        "rare grasp widths.",
        "Small grasps not drowned out: four crop radii and width-balanced losses.",
    ),
    "EconomicGrasp": (
        "From the point cloud of a depth image, it picks graspness-guided seeds and one of 300 "
        "views, then classifies angle, depth and quality with interacting task tokens, "
        "trained on one compact label per point and view.",
        "One compact label per point and view; a head that classifies angle, depth and quality.",
    ),
    "FineGrasp": (
        "From the point cloud of a depth image and its surface normals, it predicts grasps as "
        "EconomicGrasp does, with graspness normalised within every object and crops at four "
        "radii fused by attention.",
        "Surface normals as input, graspness per object, four crop radii fused by attention.",
    ),
    "HGGD": (
        "From an RGB-D image, it predicts a heat-map of grasp centres in 2-D, lifts each peak "
        "to 3-D and completes the grasp with a PointNet over a ball of scene points.",
        "A 2-D heat-map of grasp centres, completed in 3-D by a PointNet over local regions.",
    ),
}
#: The host of a checkpoint, by the network location of its link.
HOSTS = {
    "drive.google.com": "Google Drive",
    "github.com": "GitHub",
    "huggingface.co": "Hugging Face",
    "cloud.tsinghua.edu.cn": "Tsinghua Cloud",
}
#: Camera and dataset names as the configs spell them → as the pages do.
CAMERAS = {"realsense": "RealSense", "kinect": "Kinect", "realsense-d435": "RealSense D435"}
DATASETS = {"GraspNet1B": "GraspNet-1Billion", "GraspClutter6D": "GraspClutter6D"}
#: The licence of the GraspNet-1Billion data and labels, which every checkpoint trained on.
G1B_LICENCE = "CC BY-NC-SA 4.0"
#: The README fold whose text a method page shows as "How this port differs from the reference".
DIFFERENCES = "How this port differs from the reference"
#: Provenance of a metafile row → the words of a method page, and the order of its rows
#: (``author-ckpt``: the released weights, named per method by :func:`upstream_words`).
PROVENANCE_WORDS = {
    "reported": "reported",
    "author-ckpt": "{short}, scored here",
    "reproduced": "trained here",
}
FILTER_WORDS = {False: "off", True: "on", None: "not stated"}
#: The same, short, for the table of a method page.
SHORT_PROVENANCE = {
    "reported": "reported",
    "author-ckpt": "{short}",
    "reproduced": "trained here",
}
#: The words of ``upstream.license`` in a metafile that mark a reference implementation not
#: written by the paper's authors (GSNet's graspness_unofficial: "a third-party
#: implementation, there is no official training code").
THIRD_PARTY = "third-party"
#: What ``tests/unit/test_method_quickstart.py`` does with the commands of a five-minute
#: section, as a page says it.
TESTED = (
    f"A [test]({GITHUB}/tests/unit/test_method_quickstart.py) parses each command and runs the "
    "demo and the Python code on a toy detector of the method's family."
)
#: Slot → its title on the page.
TITLES = {
    "image_encoder": "Image encoder",
    "backbone": "Scene encoder",
    "proposal": "Seed proposal",
    "view": "View",
    "assigner": "Assigner",
    "grouper": "Grouper",
    "head": "Head bundle",
    "loss": "Losses",
    "post": "Post-processing",
    "optimizer": "Optimizer",
    "schedule": "Schedule",
    "hooks": "Hooks",
    "augment": "Augmentation",
    "recipe": "Whole recipe",
}
#: Slot → the slot page that explains its options.
SLOT_PAGE = {
    "image_encoder": "image_encoders",
    "backbone": "scene_encoders",
    "proposal": "proposal",
    "view": "view",
    "assigner": "head_loss",
    "grouper": "grouper",
    "head": "head_loss",
    "loss": "losses",
    "post": "post",
    "optimizer": "recipe",
    "schedule": "recipe",
    "hooks": "recipe",
    "augment": "recipe",
    "recipe": "recipe",
}
#: The slot pages that get the generated regions, with the slots they explain.
SLOT_PAGE_SLOTS = {
    "image_encoders": ("image_encoder",),
    "scene_encoders": ("backbone",),
    "scene_encoders_point": ("backbone",),
    "scene_encoders_sparse": ("backbone",),
    "scene_encoders_transformer": ("backbone",),
    "proposal": ("proposal",),
    "view": ("view",),
    "grouper": ("grouper",),
    "head_loss": ("head", "assigner"),
    "losses": ("loss",),
    "post": ("post",),
    "recipe": ("optimizer", "schedule", "hooks", "augment", "recipe"),
}
BEGIN = "<!-- method-options:begin (generated by tools/make_method_pages.py; do not edit) -->"
END = "<!-- method-options:end -->"
#: Marks of a level in a table.
MARKS = {YES: "✓", NO: "✗"}
VERDICT_MARKS = {"runs": "✓", "runs-empty": "∅", "runs-unstable": "≈", "fails": "!"}


def _docs_tool() -> Any:
    """``tools/make_combination_docs.py``: the table helpers and the level definitions."""
    name = "make_combination_docs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def chips() -> Any:
    """``tools/docs_chips.py``, the shared vocabulary of the pages (loaded by its file:
    ``tools/`` is no package)."""
    name = "docs_chips"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def page_of(method: str) -> str:
    """The stem of a method's page: the directory of its config."""
    return option_lists.methods()[method].split("/")[0]


def anchor(slot: str) -> str:
    return slot.replace("_", "-")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table; a pipe inside a cell is escaped."""

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    return [line(headers), "|" + "---|" * len(headers), *(line(row) for row in rows)]


def _indent(lines: list[str], by: str = "    ") -> list[str]:
    return [by + line if line else "" for line in lines]


def _fold(title: str, body: list[str]) -> list[str]:
    """A closed fold (``??? note``) around ``body``."""
    return [f'??? note "{title}"', "", *_indent(body), ""]


def _raw(lines: list[str]) -> list[str]:
    """Search leaves a raw table out (it lands on the prose instead)."""
    return ["<div data-search-exclude markdown>", "", *lines, "", "</div>"]


def _name(text: str) -> str:
    """A name in code that does not break inside a narrow table column."""
    return f'<code class="gp-nowrap">{html.escape(text)}</code>'


def _readme(method: str) -> str:
    return f"{GITHUB}/configs/{page_of(method)}/README.md"


def _top_level(text: str, separator: str = ";") -> str:
    """``text`` up to its first ``separator`` outside parentheses."""
    depth = 0
    for index, char in enumerate(text):
        depth += char == "("
        depth -= char == ")"
        if char == separator and depth == 0:
            return text[:index].strip()
    return text.strip()


def licence_short(text: str) -> str:
    """A licence in a word or three, for a narrow table; the full text goes in its tooltip."""
    lower = text.lower()
    restricted = "non-commercial" in lower
    if lower.startswith("mit") or "mit file" in lower:
        return "MIT" + (", with non-commercial parts" if restricted else "")
    if lower.startswith("apache"):
        return "Apache-2.0" + (", with non-commercial parts" if restricted else "")
    return "non-commercial" if restricted else _top_level(text)


def host_of(url: str) -> str:
    """Where a checkpoint is downloaded from, in words (``Google Drive``)."""
    parts = urlsplit(url)
    host = HOSTS.get(parts.netloc, parts.netloc)
    return host + (" release" if host == "GitHub" and "/releases/" in parts.path else "")


@functools.cache
def metafile_of(method: str) -> Metafile:
    return next(meta for meta in load_metafiles() if meta.method == method)


@functools.cache
def upstream_words(method: str) -> dict[str, str]:
    """Who wrote a method's reference code and released the checkpoint a user downloads,
    in the words of a page, from ``upstream`` of the metafile: the authors, or a
    third-party implementation (:data:`THIRD_PARTY` in its licence).

    Keys: ``code`` (the reference code), ``weights`` (the released checkpoint, said in
    full), ``checkpoint`` and ``file`` (the same, short, once the page has said whose it
    is), ``short`` (for a chip or a table label), ``repository`` (``owner/name``) and
    ``third_party`` (``"yes"`` or ``""``).
    """
    upstream = metafile_of(method).upstream
    repository = upstream.get("repository", "")
    if THIRD_PARTY in upstream.get("license", "").lower():
        name = repository.split("/")[-1] or "the reference repository"
        where = f"the reference implementation ({name}, third-party)"
        return {
            "code": where,
            "weights": f"the checkpoint of {where}",
            "checkpoint": "the released checkpoint",
            "file": "the released file",
            "short": "released weights",
            "repository": repository,
            "third_party": "yes",
        }
    return {
        "code": "the authors' code",
        "weights": "the authors' checkpoint",
        "checkpoint": "the authors' checkpoint",
        "file": "the authors' released file",
        "short": "authors' weights",
        "repository": repository,
        "third_party": "",
    }


def provenance_words(method: str, provenance: str, short: bool = False) -> str:
    """A row's provenance in the words of a method page (the released weights per method)."""
    text = (SHORT_PROVENANCE if short else PROVENANCE_WORDS)[provenance]
    return text.format(short=upstream_words(method)["short"])


def numbers_status(method: str) -> tuple[str, str, str]:
    """The status of a method's numbers, from the provenance of its metafile rows: the
    chip text, its tone and what it means."""
    found = {result.provenance for model in metafile_of(method).models for result in model.results}
    if "reproduced" in found:
        return (
            "trained here",
            "ok",
            "a model trained by GraspPanda, scored on whole test splits",
        )
    if "author-ckpt" in found:
        words = upstream_words(method)
        return (
            f"{words['short']} scored here",
            "ok",
            f"{words['weights']}, converted and scored on whole test splits; nothing trained "
            "here yet",
        )
    return ("reported only", "muted", "only the numbers of the paper; nothing scored here")


def equal_chip(method: str) -> str:
    """The chip *equal to the reference ✓*, backed by the method's parity test."""
    test = PARITY[method]
    if not (REPO / test).is_file():
        raise FileNotFoundError(f"{method}: the parity test {test} is missing")
    return chips().chip(
        "equal to the reference ✓",
        "ok",
        f"stage by stage against {upstream_words(method)['code']}, the same weights loaded "
        f"into both: {test}",
    )


def head_chips(method: str) -> str:
    """The chips under a method's title: family, venue, input, parity, numbers."""
    dc = chips()
    info = dc.method(method)
    if info.inputs != ("RGB-D" if pipeline_facts(info.config)["rgb"] else "depth"):
        raise ValueError(f"{method}: docs_chips says {info.inputs}, the config says otherwise")
    status, tone, meaning = numbers_status(method)
    family = dc.FAMILY_NAMES[info.family]
    items = [
        dc.chip(family, "", f"{family}: {', '.join(_family_members(info.family))}"),
        dc.chip(info.venue, "", info.title),
        dc.chip(f"{info.inputs} input"),
        equal_chip(method),
        dc.chip(status, tone, meaning),
    ]
    return '<div class="gp-chips">' + " ".join(items) + "</div>"


def _family_members(family: str) -> list[str]:
    return [info.name for info in chips().methods() if info.family == family]


def paper_link(method: str) -> str:
    """The paper: the metafile's link, else the first link of the method's README."""
    paper = metafile_of(method).paper
    if not paper:
        readme = (REPO / "configs" / page_of(method) / "README.md").read_text(encoding="utf-8")
        found = re.search(r"^> \[[^\]]+\]\((https?://[^)]+)\)", readme, re.M)
        paper = found.group(1) if found else ""
    return paper


def links_line(method: str) -> str:
    upstream = metafile_of(method).upstream
    repository = upstream.get("repository", "")
    commit = upstream.get("commit", "")
    parts = []
    paper = paper_link(method)
    if paper:
        parts.append(f"[Paper]({paper})")
    if repository:
        tree = f"https://github.com/{repository}" + (f"/tree/{commit}" if commit else "")
        third = " (third-party)" if upstream_words(method)["third_party"] else ""
        parts.append(f"[Reference code{third}]({tree})")
    parts.append(f"[Configs, results and notes]({_readme(method)})")
    return " · ".join(parts)


# ----------------------------------------------------------------- the slots of a method


def _slot_key(slot: str) -> str:
    """What follows ``--set`` for a slot."""
    return "<stage>.<term>_loss" if slot == "loss" else slot


def _shipped_short(found: option_lists.SlotListing) -> str:
    """What the method ships in a slot, in a few words."""
    if found.shipped_option:
        return f"`{found.shipped_option}`"
    if found.slot == "loss":
        return "its own losses"
    if found.slot == "post":
        names = re.findall(r"\(the config has (.*)\)", found.shipped)
        return f"variants {names[0]}" if names else "—"
    if found.slot in ("augment", "recipe"):
        return "its own"
    if found.slot == "image_encoder" and "built in" in found.shipped:
        return "its built-in encoder"
    if found.shipped in ("", "none"):
        return found.shipped or "—"
    return ", ".join(f"`{name}`" for name in found.shipped.split(", "))


#: A tail (who is left out, and why) at most this long stands under its chip in an options
#: table of a method page; a longer one is listed below the table (a tooltip is not seen on a
#: touch screen, and a long tail would squeeze the column that says what happened here).
TAIL_IN_CELL = 60


def _tailed(chip: str, tail: str) -> tuple[str, str]:
    """A chip with its tail under it when the tail is short; else the chip and the tail."""
    if not tail:
        return chip, ""
    if len(tail) <= TAIL_IN_CELL:
        return f"{chip}<br><small>{tail}</small>", ""
    return chip, tail


@functools.cache
def _reach_cell(slot: str, name: str) -> tuple[str, str]:
    """Where an option runs: the chip (its tail as the tooltip, a short tail under it), and
    a tail too long for the cell (``""`` when there is none)."""
    dc = chips()
    return _tailed(dc.reach_chip(slot, name), dc.reach(slot, name).tail)


@functools.cache
def _level2_cell(slot: str, name: str) -> tuple[str, str]:
    """Whether an option trains like the component each method ships, as
    :func:`_reach_cell` says where it runs."""
    dc = chips()
    found = dc.level2(slot, name)
    if found.none:
        return "—", ""
    return _tailed(dc.level2_chip(slot, name), found.tail)


def _chip_cells(slot: str, name: str, long: list[str]) -> list[str]:
    """The *Runs* and *Trains like shipped* cells of option ``name``; a tail too long for
    its cell goes to ``long``, the list below the table."""
    runs, runs_tail = _reach_cell(slot, name)
    second, second_tail = _level2_cell(slot, name)
    if runs_tail:
        long.append(f"`{name}`, runs: {runs_tail}.")
    if second_tail:
        long.append(f"`{name}`, {chips().LEVEL2_NAME}: {second_tail}.")
    return [runs, second]


def _sentence(text: str) -> str:
    """The first sentence of a message: up to a full stop or a semicolon before a space."""
    found = re.match(r"(.+?[.;])(?:\s|$)", text.strip(), re.S)
    return " ".join((found.group(1) if found else text).split()).rstrip(";.") + "."


def _requires(method: str, option: option_lists.Option) -> list[str]:
    """What a recipe preset needs in the same request that ``method`` has no slot for
    (``grid.RECIPE_REQUIRES``: ``ptv3-adamw-onecycle`` needs ``backbone=ptv3``)."""
    from grasppanda.combine.grid import RECIPE_REQUIRES

    needed = RECIPE_REQUIRES.get(f"{option.slot}/{option.name}", ())
    slots = option_lists.method_slots(method)
    return [item for item in needed if item.partition("=")[0] not in slots]


def refusal_text(method: str, entry: option_lists.Entry) -> str:
    """Why an option does not fit a method: the kind of the refusal, then the first sentence
    of the record's first message (all of it when the option waits for consent: it says how
    to give it; for an option the grid has no cell for, the reason of the listing), and the
    lift when the record names one."""
    words = reaching.REFUSAL_WORDS.get(entry.kind, "")
    lead = words if entry.kind == "policy" else (f"refused ({words})" if words else "refused")
    messages = [part["message"] for part in entry.refusal if part.get("message")]
    missing = _requires(method, entry.option)
    if messages and entry.kind == "policy":  # the whole message: it says how to consent
        why = " ".join(messages[0].split())
    elif messages:
        why = _sentence(messages[0])
    elif missing:
        needed = ", ".join(f"`--set {item}`" for item in missing)
        slots = ", ".join(f"`{item.partition('=')[0]}`" for item in missing)
        why = f"it needs {needed} in the same request, and {method} has no slot {slots}."
    else:
        why = _sentence(entry.reason) if entry.reason else ""
    text = f"{lead}: {why}" if why else lead
    lifts = [part["lift"] for part in entry.refusal if part.get("lift")]
    if lifts:
        text += " Lift: " + " ".join(_sentence(lift) for lift in lifts)
    return text


#: What *Under <method>* says of a level the method as shipped does not reach either
#: (``levels.BELOW_REFERENCE``), as the explorer says "reference misses it too"; why stands
#: once under the table (:func:`as_shipped_notes`).
AS_SHIPPED = "{level} ? (as shipped)"


def as_shipped_notes(
    method: str, rows: list[list[str]], records: option_lists.Measured
) -> list[str]:
    """Under a table whose last column says :data:`AS_SHIPPED` of a level: why the method as
    shipped does not reach it (its reference record), once, not in every row."""
    reference = records.reference(method)
    notes = []
    for level in ("learns", "predicts"):
        mark = AS_SHIPPED.format(level=level)
        if reference is None or not any(mark in row[-1] for row in rows):
            continue
        why = "; ".join(records.levels(reference).why.get(level, [])[:2])
        notes.append(
            f"*{mark}*: {method} as shipped does not reach *{level}* in a short run either"
            + (f" ({why})" if why else "")
            + f", so under {method} that level says nothing about an option."
        )
    return notes


def _here(method: str, entry: option_lists.Entry, records: option_lists.Measured) -> str:
    """What happened under this method: ``✓`` when every level was reached, else why; a
    level the method as shipped does not reach either is undecided, :data:`AS_SHIPPED`
    (``predicts ? (as shipped)``, as the explorer says "reference misses it too"; why is
    said once under the table); for a refusal, the reason (:func:`refusal_text`)."""
    option = entry.option
    if entry.status == "does not fit":
        return refusal_text(method, entry)
    if entry.status == option_lists.NOT_PROBED:
        key = option.presets[0] if option.presets else option.name
        if reaching.reach(option.slot, key, records=records).note:
            return "no record"  # no cell anywhere: the tail of *Runs* says why
        return "no record: " + (entry.reason or "not probed yet")
    parts = ["shipped"] if entry.shipped else []
    levels = entry.levels
    if levels is not None:
        runs = levels.states.get("runs")
        if runs == NO:
            key = option.presets[0] if len(option.presets) == 1 else option.name
            found = reaching.reach(option.slot, key, records=records)
            words = found.missed.get(method) if len(option.presets) <= 1 else ""
            parts.append(words or reaching.VERDICT_WORDS.get(entry.verdict, entry.verdict))
        elif runs != YES:
            parts.append(reaching.UNDECIDED_WORDS.get(runs, runs))
        else:
            missed = False
            for level in ("learns", "predicts"):
                state = levels.states.get(level)
                if state == NO:
                    why = "; ".join(levels.why.get(level, [])[:2])
                    parts.append(f"{level} {MARKS[NO]}" + (f": {why}" if why else ""))
                    missed = True
                elif state == BELOW_REFERENCE and not (entry.shipped or missed):
                    # the option shipped is the method as shipped; after a level missed,
                    # the row already says why it has no ✓
                    parts.append(AS_SHIPPED.format(level=level))
                elif state not in (YES, BELOW_REFERENCE):
                    parts.append(f"{level} ? ({reaching.UNDECIDED_WORDS.get(state, state)})")
    return "; ".join(parts) or MARKS[YES]


def counted(found: option_lists.SlotListing) -> list[option_lists.Entry]:
    """The entries of a listing that are options (:func:`tools.docs_chips.options_of`): the
    adapters compose inserts and the point-painting wrappers are not. A loss entry is one
    preset (or a set of them) on one term (or several): every one is counted."""
    if found.slot == "loss":
        return list(found.entries)
    names = set(chips().options_of(found.slot))
    return [entry for entry in found.entries if entry.option.name in names]


def tally(entries: list[option_lists.Entry]) -> dict[str, list[option_lists.Entry]]:
    """The entries of a slot by what the config check said under the method: ``composes``
    (accepted), ``refused``, ``consent`` (held back until the user accepts code it
    downloads: neither accepted nor refused), ``none`` (no record). What ran is counted by
    :func:`runs_here`, as *Does it run?* counts it."""
    found: dict[str, list[option_lists.Entry]] = {
        key: [] for key in ("composes", "refused", "consent", "none")
    }
    for entry in entries:
        if entry.status == "fits":
            found["composes"].append(entry)
        elif entry.status == "does not fit":
            found["consent" if entry.kind == "policy" else "refused"].append(entry)
        else:
            found["none"].append(entry)
    return found


def _option_rows(
    method: str,
    entries: list[option_lists.Entry],
    records: option_lists.Measured,
    long: list[str],
) -> list[list[str]]:
    rows = []
    for entry in entries:
        option = entry.option
        rows.append(
            [
                _name(option.name),
                *_chip_cells(option.slot, option.name, long),
                _here(method, entry, records),
            ]
        )
    return rows


def _loss_rows(
    method: str, found: option_lists.SlotListing, records: option_lists.Measured, long: list[str]
) -> tuple[list[list[str]], list[str], list[str]]:
    """One row per loss preset, with the terms it replaced under this method; the cells
    that replace several terms at once, summed up; and the presets listed."""
    terms: dict[str, dict[str, list[str]]] = {}
    joint = []
    for entry in found.entries:
        option = entry.option
        if len(option.sets) != 1:
            joint.append(entry)
            for preset in option.presets:
                terms.setdefault(preset, {})
            continue
        term = option.sets[0].partition("=")[0]
        preset = option.presets[0]
        terms.setdefault(preset, {}).setdefault(_here(method, entry, records), []).append(
            f"`{term}`"
        )
    rows = []
    for preset, by_note in terms.items():
        ordered = sorted(by_note.items(), key=lambda item: item[0] != MARKS[YES])  # ✓ first
        here = "; ".join(f"{', '.join(names)}: {note}" for note, names in ordered)
        rows.append(
            [
                _name(preset),
                *_chip_cells("loss", preset, long),
                here or "only in a set of several losses",
            ]
        )
    notes = []
    if joint:
        ran = sum(
            1 for entry in joint if entry.levels is not None and entry.levels.states["runs"] == YES
        )
        names = ", ".join("`" + entry.option.name.split("@")[0] + "`" for entry in joint)
        notes.append(
            f"Also probed under {method}: {len(joint)} cells that replace several terms at once "
            f"({names}); {ran} of them run."
        )
    return rows, notes, list(terms)


def _loss_not_tried(method: str, listed: list[str]) -> list[str]:
    """Why the loss presets not listed under ``method`` were not tried there: the kinds of
    term they fit, of which the method has none (``grid.loss_terms``). A preset measured
    elsewhere (``grid.LOSS_NOT_PROBED``) is explained by the notes of the listing."""
    from grasppanda.combine.grid import LOSS_NOT_PROBED, loss_terms
    from grasppanda.combine.presets import LOSS_SLOT, presets
    from grasppanda.models.losses.terms import accepted_terms

    kinds = loss_terms(method)
    groups: dict[tuple[str, ...], list[str]] = {}
    for name in chips().options_of("loss"):
        if name in listed or name in LOSS_NOT_PROBED:
            continue
        found = accepted_terms(dict(presets()[LOSS_SLOT][name].node))
        accepted = tuple(found[1]) if found else ()
        if set(accepted) & set(kinds):
            raise ValueError(f"{method}: the loss preset {name} fits a term but has no cell")
        groups.setdefault(accepted, []).append(name)
    notes = []
    for accepted, names in groups.items():
        fits = " or ".join(f"`{kind}`" for kind in accepted) or "no"
        notes.append(
            f"Not tried under {method}: {', '.join(f'`{name}`' for name in names)} "
            f"{'fits' if len(names) == 1 else 'fit'} {fits} terms only, and {method} has "
            f"none (its terms are of kind {', '.join(f'`{kind}`' for kind in kinds)})."
        )
    return notes


def _not_counted(found: option_lists.SlotListing) -> list[str]:
    """The entries of a listing that are no options (adapters, point painting), with why."""
    listed = {id(entry) for entry in counted(found)}
    others = [entry for entry in found.entries if id(entry) not in listed]
    if not others:
        return []
    dc = chips()
    lines = ["Not counted as options (no probe of their own):", ""]
    for entry in others:
        tail = dc.reach(found.slot, entry.option.name).tail
        lines.append(f"- `{entry.option.name}`: {tail or entry.reason}.")
    return ["\n".join(lines)]  # one note of several lines (slot_body splits it)


def _options_word(count: int, total: int, word: str = "options") -> str:
    """``47 options``, or ``19 of 20 presets`` when the method lists fewer than there are."""
    return f"{count} {word}" if count == total else f"{count} of {total} {word}"


def slot_body(
    method: str,
    found: option_lists.SlotListing,
    records: option_lists.Measured,
    titled: bool = False,
) -> tuple[list[str], str]:
    """The options of one slot: every option, where it runs, how it trains, and what
    happened under this method; and how many there are (``docs_chips.option_count``). The
    anchor ``#<slot>`` is the paragraph above the table (a fold opens when a link points
    into it)."""
    dc = chips()
    slot = found.slot
    title = dc.SLOT_TITLES[slot]
    notes = list(found.notes)
    total = dc.option_count(slot)
    long: list[str] = []
    if slot == "loss":
        rows, joint, listed = _loss_rows(method, found, records, long)
        notes = joint + _loss_not_tried(method, listed) + notes
        what = _options_word(len(rows), total, "presets")
    else:
        rows = _option_rows(method, counted(found), records, long)
        notes = _not_counted(found) + notes
        what = _options_word(len(rows), total)
    if long:
        notes.insert(0, "\n".join(["Who is left out, and why:", "", *(f"- {t}" for t in long)]))
    notes = as_shipped_notes(method, rows, records) + notes
    none = dc.level2(slot, "").none
    if none:
        notes.append(f"*{dc.LEVEL2_HEADER}*: — ({none}).")
    head = ["Option", "Runs", dc.LEVEL2_HEADER, f"Under {method}"]
    body = [
        (f"**{title}**, {what}: " if titled else "")
        + f"`--set {_slot_key(slot)}=<option>`. What each option is: the "
        f"[{title.lower()} page](../{dc.SLOT_PAGES[slot]}).",
        f"{{: #{anchor(slot)} }}",
        "",
        *_matrix(_table(head, rows)),  # narrow padding: the table fits in its fold
    ]
    for note in notes:
        body += ["", *note.split("\n")]
    return body, what


def _matrix(lines: list[str], kind: str = "") -> list[str]:
    """A wide table that may scroll on its own (``gp-matrix`` in ``extra.css``); ``kind``
    adds a class (``gp-slots``: the slot summary, whose columns 3-4 give way on a phone)."""
    classes = " ".join(["gp-matrix", *([kind] if kind else [])])
    return [f'<div class="{classes}" markdown>', "", *lines, "", "</div>"]


def runs_here(method: str, slot: str, records: option_lists.Measured) -> str:
    """The options of ``slot`` that ran under ``method``, out of those tried there as a swap,
    counted as *Does it run?* counts them (:func:`grasppanda.combine.reach.reach`): the
    option the method ships is its own and not counted, a refusal stays in the denominator,
    a loss preset is one option."""
    tried = reached = 0
    for name in chips().options_of(slot):
        found = reaching.reach(slot, name, "runs", records=records)
        if method in found.swaps:
            tried += 1
            reached += method in found.reached
    return f"{reached}/{tried}" if tried else "own only"


def refused_unprobed(
    method: str, found: option_lists.SlotListing, records: option_lists.Measured
) -> list[str]:
    """The options of a listing that the config check refuses under ``method`` with no
    record there (the grid has no cell for them under it): *Runs here* does not count them,
    as *Does it run?* does not."""
    names = []
    for entry in counted(found):
        if entry.status != "does not fit":
            continue
        option = entry.option
        key = option.presets[0] if found.slot == "loss" and option.presets else option.name
        if method not in reaching.reach(found.slot, key, "runs", records=records).applies:
            names.append(f"`{key}`")
    return names


def _names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def level2_needs(records: option_lists.Measured) -> str:
    """Why a swap can miss *learns* and still train like shipped, with the iterations of
    the probes (level 1: the reference cells; level 2: ``learns_like_reference.json``)."""
    first = records.reference(chips().methods()[0].name)["iterations"]
    about = json.loads(reaching.level2_file().read_text(encoding="utf-8"))["about"]
    second = about["settings"]["iterations"]
    return (
        f"Level 2 needs *runs*, not *learns*: a term flat after {first} iterations can fall "
        f"by {second}."
    )


def level2_title() -> str:
    """The title of the level-2 page (its first heading): the text of every link to it."""
    text = LEVEL2_PAGE.read_text(encoding="utf-8")
    return next(line[2:].strip() for line in text.splitlines() if line.startswith("# "))


def slots_section(
    method: str, listings: list[option_lists.SlotListing], records: option_lists.Measured
) -> list[str]:
    dc = chips()
    config = option_lists.methods()[method]
    rows = []
    unrecorded: list[str] = []
    unprobed: list[str] = []
    for found in listings:
        counts = tally(counted(found))
        unprobed += refused_unprobed(method, found, records)
        unrecorded += [
            f"`{(entry.option.presets or (entry.option.name,))[0]}`" for entry in counts["none"]
        ]
        composes = str(len(counts["composes"])) + "".join(
            f" · {len(counts[key])} {words}"
            for key, words in (("consent", "needs consent"), ("none", "no record"))
            if counts[key]
        )
        label = f"[{dc.SLOT_TITLES[found.slot]}](#{anchor(found.slot)})"
        explorer = f"{EXPLORER}#m={quote(method)}&amp;grid={found.slot}"
        rows.append(
            [
                label,
                _shipped_short(found),
                composes,
                str(len(counts["refused"])),
                runs_here(method, found.slot, records),
                f'<a href="{explorer}">compare</a>',
            ]
        )
    unrecorded = list(dict.fromkeys(unrecorded))
    fits = f"Every option that fits {method}"
    every = (
        f"{fits}, except {_names(unrecorded)}, was composed under it"
        if unrecorded
        else f"{fits} was composed under it"
    )
    unprobed = list(dict.fromkeys(unprobed))
    losses = any(found.slot == "loss" for found in listings)
    not_counted = (
        f" Not counted either: {_names(unprobed)}, refused under {method} with no probe record."
        if unprobed
        else ""
    )
    measured = [
        "Swap one option in with a single line, or list the options of a slot:",
        "",
        "```bash",
        *_wrapped(f"grasppanda compose {config} --set <slot>=<option> -o swap.yaml", ("--set",)),
        f"grasppanda compose {config} --list <slot>",
        "```",
        "",
        "The summary table:",
        "",
        f"- **Composes**: the options the config check accepts under {method}, the one it "
        "ships included. *needs consent*: an option it holds back until you accept the code "
        "it downloads (the row says how), neither accepted nor refused. *no record*: never "
        "composed; the row says why.",
        f"- **Refused**: the options the config check refuses under {method}; each row gives "
        "the first sentence of the reason."
        + (
            " For losses, both columns count the terms (or sets of terms) a preset was tried on."
            if losses
            else ""
        ),
        f"- **Runs here**: the swaps whose probe trained on real frames under {method} "
        "without an error, out of every option tried there as a swap: a refusal counts, the "
        "option the method ships does not"
        + (", and a loss preset is one option." if losses else ".")
        + f"{not_counted} "
        "[Does it run?](../advanced_guides/measured_combinations.md#the-answer) counts the "
        "same way.",
        "",
        "The lists of each slot:",
        "",
        "- **Runs**: the methods under which the swap trained on real frames without an "
        "error, out of every method it is a swap under; the methods that ship the option are "
        "*own* and not counted. Who is left out, and why, stands under the chip, or below the "
        f"table when it is long. Measured: {dc.measured_on(1)}.",
        f"- **{dc.LEVEL2_HEADER}**: {dc.LEVEL2_MEANING}, out of the methods it was compared "
        "under; a method under which the swap did not run, or where it trained on a schedule "
        f"of its own, is named and not counted. {level2_needs(records)} Measured: "
        f"{dc.measured_on(2)}.",
        f"- **Under {method}**: ✓ when it reached every level here; "
        f"*{AS_SHIPPED.format(level='level')}* when {method} as shipped does not reach that "
        "level either (a note below the table says why), so the level says nothing about "
        "the option; else the level it missed and why (*composes*, *runs*, *learns*, "
        "*predicts*), or why it was refused.",
        "",
        "The levels, the refusal kinds and the thresholds: "
        "[How we measure](../evidence/index.md); every record: "
        "[Does it run?](../advanced_guides/measured_combinations.md#measured-classes); every "
        f"comparison at level 2: [{level2_title()}](../advanced_guides/learns_like_reference.md).",
    ]
    lines = [
        "## The options of every slot {#the-slots}",
        "",
        f"{every}; those accepted were probed on real frames; the others say why.",
        "",
        CAVEAT_SNIPPET,
        "",
        *_matrix(
            _table(["Slot", "Shipped", "Composes", "Refused", "Runs here", "Explorer"], rows),
            "gp-slots",  # on a phone, Composes and Refused (columns 3-4) give way
        ),
        "",
    ]
    # The slots of the training recipe share one page and one fold.
    recipe = [found for found in listings if dc.SLOT_PAGES[found.slot] == dc.SLOT_PAGES["recipe"]]
    for found in listings:
        if found not in recipe:
            body, what = slot_body(method, found, records)
            lines += _fold(f"{dc.SLOT_TITLES[found.slot]}: {what}", _raw(body))
    if recipe:
        body: list[str] = []
        for found in recipe:
            part, _ = slot_body(method, found, records, titled=True)
            for line in part:  # a note of the method as shipped, once per fold
                if line.startswith("*") and line in body and "? (as shipped)*:" in line:
                    if body[-1] == "":
                        body.pop()  # the blank line before the note
                    continue
                body.append(line)
            body.append("")
        listed = sum(len(counted(found)) for found in recipe)
        what = _options_word(listed, sum(dc.option_count(found.slot) for found in recipe))
        slots = ", ".join(dc.SLOT_TITLES[found.slot].lower() for found in recipe)
        lines += _fold(f"Training recipe: {what} ({slots})", _raw(body[:-1]))
    lines += _fold("How to swap, and how to read these lists", measured)
    return lines


# --------------------------------------------------------- run it in five minutes

#: The anchor of the first section of a method page.
RUN_ANCHOR = "run-in-five-minutes"
#: The box of the camera frame the camera example keeps (metres): the example scene's objects.
CAMERA_BOX = "-0.25 -0.2 0.35 0.25 0.2 0.8"
#: What a row's provenance means, as the pages say it (``{weights}``: the released
#: checkpoint, named per method by :func:`upstream_words`).
PROVENANCE_TEXT = {
    "author-ckpt": "{weights}, evaluated by GraspPanda",
    "reproduced": "trained and evaluated by GraspPanda",
}


def five_minutes() -> dict[str, Any]:
    """The measured runs of the five-minute sections (``docs/methods/five_minutes.json``)."""
    return json.loads(FIVE_MINUTES.read_text(encoding="utf-8"))


def zoo_entry(method: str) -> tuple[Metafile, ModelEntry]:
    """The metafile and the model entry of a method's config."""
    config = option_lists.methods()[method] + ".yaml"
    for metafile in load_metafiles():
        for model in metafile.models:
            if model.config == config:
                return metafile, model
    raise LookupError(f"no metafile lists {config}")


def weights_of(model: ModelEntry, entry: str) -> Weights:
    """The weights entry of a model a measured run names (``weights`` or ``author_weights``)."""
    found = getattr(model, entry)
    if found is None:
        raise LookupError(f"the model {model.name} has no {entry} entry")
    return found


def downloadable(model: ModelEntry) -> Weights:
    """The weights a user can get for the model: the released checkpoint, to be converted."""
    if model.weights is not None and model.weights.source == "author":
        return model.weights
    if model.author_weights is not None:
        return model.author_weights
    raise LookupError(f"the model {model.name} lists no checkpoint a user can download")


def provenance_of(weights: Weights) -> str:
    return "author-ckpt" if weights.source == "author" else "reproduced"


@functools.cache
def pipeline_facts(config: str) -> dict[str, Any]:
    """What the test pipeline of a config asks of an input, and its collision-filter chain."""
    from grasppanda.registry import TRANSFORMS

    resolved = Config.load(config).resolved()
    node = resolved["data"][resolved["evaluation"]["split"]]["dataset"]
    providers = [TRANSFORMS.get(step["type"]) for step in node.get("preprocess") or ()]
    chain = next(
        (
            [step["type"] for step in steps]
            for steps in resolved["evaluation"]["variants"].values()
            if any(step["type"] == "ModelFreeCollisionFilter" for step in steps)
        ),
        [],
    )
    return {
        # the workspace is cropped around a segmentation: the example scene's mask stands in
        "mask": any(getattr(cls, "uses_ground_truth", False) for cls in providers),
        "rgb": "rgb" in (node.get("images") or ()),
        "chain": chain,
        "variant": next(
            name
            for name, steps in resolved["evaluation"]["variants"].items()
            if [step["type"] for step in steps] == chain
        ),
        "out": f"work_dirs/{Path(config).name}/demo",
    }


#: The longest line of a command block, in characters: wider lines scroll on a page.
WIDTH = 84


def _wrapped(command: str, breaks: tuple[str, ...]) -> list[str]:
    """A command as shell lines: a new continuation line before each word of ``breaks``,
    and before a word that would make a line wider than :data:`WIDTH` (never between an
    option and its value)."""
    lines: list[list[str]] = [[]]
    for word in command.split(" "):
        width = len(" ".join([*lines[-1], word])) + (4 if len(lines) > 1 else 0) + 2
        after_flag = bool(lines[-1]) and lines[-1][-1].startswith("--")
        if lines[-1] and (word in breaks or (width > WIDTH and not after_flag)):
            lines.append([])
        lines[-1].append(word)
    text = [("    " if index else "") + " ".join(words) for index, words in enumerate(lines)]
    return [line + " \\" for line in text[:-1]] + [text[-1]]


def demo_command(config: str, weights: str) -> str:
    return f"grasppanda demo {config} --weights {weights} --example --collision-filter"


def camera_command(config: str, weights: str, facts: dict[str, Any]) -> str:
    """The demo on the files the example run wrote, as if a camera had taken them."""
    out = f"{facts['out']}/example"
    command = f"grasppanda demo {config} --weights {weights} --depth {out}/depth.png"
    command += f" --intrinsics {out}/intrinsics.txt"
    if facts["rgb"]:
        command += f" --rgb {out}/rgb.png"
    if facts["mask"]:
        command += f" --workspace {CAMERA_BOX}"
    return command + " --collision-filter --out work_dirs/camera"


def subset_command(config: str, weights: str, test: dict[str, Any]) -> str:
    """The test command on the frames of a measured subset."""
    opts = [
        f"data.test.dataset.split={test['split']}",
        f"data.test.dataset.frame_step={test['frame_step']}",
        *test["opts"],
    ]
    return f"grasppanda test {config} --weights {weights} --opts {' '.join(opts)}"


def explore_command() -> str:
    return "grasppanda explore --weights-dir ~/grasppanda_weights"


def python_snippet(config: str, weights: str, facts: dict[str, Any]) -> list[str]:
    """The Python route: the demo's pipeline on the demo's scene, in six statements."""
    inputs = ["depth=scene.depth", "intrinsics=scene.intrinsics", "depth_scale=scene.depth_scale"]
    if facts["rgb"]:
        inputs.append("rgb=scene.rgb")
    if facts["mask"]:  # the box around the mask is built in the table frame
        inputs += ["mask=scene.mask", "table_from_camera=scene.table_from_camera"]
    inputs.append("collision_filter=True")
    call, line = [], "   "
    for item in inputs:
        if len(line) + len(item) + 2 > 80:
            call.append(line.rstrip())
            line = "   "
        line += f" {item},"
    call.append(line)
    return [
        "from grasppanda import GraspInferencer",
        "from grasppanda.apis import example_scene",
        "scene = example_scene()  # the scene of --example",
        *(
            [f'inferencer = GraspInferencer("{config}", "{weights}")']
            if len(config) + len(weights) <= 46
            else ["inferencer = GraspInferencer(", f'    "{config}", "{weights}"', ")"]
            if len(config) + len(weights) + 10 <= WIDTH
            else ["inferencer = GraspInferencer(", f'    "{config}",', f'    "{weights}")']
        ),
        "grasps = inferencer(",
        *call,
        ")",
        "print(len(grasps), grasps.sort_by_score().to_graspnet_array()[:3])",
    ]


def _when(measured: dict[str, Any], machines: dict[str, Any]) -> str:
    return (
        f"{measured['date']}, commit `{measured['commit']}`, {machines[measured['machine']]['gpu']}"
    )


def _counts(demo: dict[str, Any]) -> str:
    """``1024 predicted, 725 left after ModelFreeCollisionFilter`` (each step of the chain)."""
    steps = demo["after"]
    text = f"{demo['predicted']} grasps predicted, "
    if len(steps) == 1:
        return text + f"{steps[0][1]} left after {steps[0][0]}"
    return text + "then " + " → ".join(f"{count} after {name}" for name, count in steps)


def _kept(demo: dict[str, Any]) -> int:
    return demo["after"][-1][1] if demo["after"] else demo["predicted"]


def _cloud(demo: dict[str, Any]) -> str:
    text = f"{demo['points']:,} points to the model"
    if demo["workspace_points"]:
        text += f", sampled from {demo['workspace_points']:,} in the workspace"
    return text


def _whole_split(model: ModelEntry, provenance: str, collision: bool) -> Result | None:
    return next(
        (
            result
            for result in model.results
            if result.provenance == provenance
            and result.collision_filter is collision
            and "seen" in result.ap
        ),
        None,
    )


def _weights_label(method: str, weights: Weights, entry: str) -> str:
    """The weights a measured run used, in the words of a page."""
    if weights.source == "grasppanda":
        return "the weights trained here (not published)"
    words = upstream_words(method)
    return f"{words['file' if entry == 'author_weights' else 'checkpoint']}, converted"


def _shell(lines: list[str]) -> list[str]:
    return ["```bash", *lines, "```", ""]


def _weights_step(method: str, model: ModelEntry, get: Weights) -> list[str]:
    trained = model.weights if model.weights is not None and model.weights is not get else None
    unpublished = trained is not None and trained.source == "grasppanda"
    released = upstream_words(method)["weights"]
    if unpublished:
        lead = (
            f"GraspPanda trained {method} once, but those weights are not published yet: the "
            f"file to download is {released}."
        )
    else:
        lead = (
            f"GraspPanda publishes no weights for {method}: the file to download is "
            f"{released}, under its licence."
        )
    lines = [
        "### 1. Get the weights {#run-weights}",
        "",
        f"{lead} Download [`{get.downloaded_name}`]({get.url}) ({host_of(get.url)}) and "
        "convert it once, from the root of a GraspPanda checkout:",
        "",
        *_shell(_wrapped(get.convert, ("--source-format",))),
    ]
    if get.note:
        lines += [f"**Before you use it:** {get.note}", ""]
    details = [
        f"- **Licence**: {get.license}",
        f"- **SHA-256** of the download: `{get.sha256}`",
        f"- **What the converter writes**: `{get.converted_name}`, every tensor of the downloaded "
        "file under the name of the module that plays its role here, no value changed, with a "
        "note of the file it came from. A tensor left over or missing stops the conversion "
        "with a report; nothing is ever loaded partially. The converter is a tool of the "
        "repository; it is not installed with the package.",
    ]
    if unpublished:
        why = re.sub(r"(?i)^not published yet:\s*", "", trained.note)
        details.append(
            f"- **The weights trained here** (the `reproduced` rows of "
            f"[the README]({_readme(method)})): {why}"
        )
    return [*lines, *_fold("Licence, checksum and what the converter writes", details)]


def _demo_step(
    method: str,
    config: str,
    get: Weights,
    facts: dict[str, Any],
    runs: list[dict[str, Any]],
    machines: dict[str, Any],
    model: ModelEntry,
) -> list[str]:
    file = get.converted_name
    variant = facts["variant"]
    chain = " then ".join(f"`{name}`" for name in facts["chain"])
    first = runs[0]["demo"][0]
    lines = [
        "### 2. Grasps on the example scene {#run-demo}",
        "",
        *_shell(_wrapped(demo_command(config, file), ("--weights",))),
        "`--example` is a synthetic table-top scene shipped with GraspPanda: no dataset, no "
        f"download. **{first['wall_s']:.1f} s** for the whole command, {_kept(first)} grasps "
        f"kept ({machines[first['machine']]['gpu']}).",
        "",
    ]
    details = [
        "The command prints the config and its hash, how the weights loaded (every tensor "
        "matched, none missing), the test pipeline, the cloud, how many grasps the model "
        "predicted and how many the collision filter left (the config's variant "
        f"`{variant}`: {chain}), and the ten best grasps. It writes into `{facts['out']}/`:",
        "",
        "- `grasps.npy`: the grasps that are left, `(G, 17)` float32 in the GraspNet layout, "
        "best first, camera frame, metres;",
        "- `grasps.png` and `grasps.gif`: the cloud with the 30 best grasps, still and swinging "
        "around the scene;",
        "- `example/`: the scene as files (`depth.png`, `rgb.png`, `mask.png`, `intrinsics.txt`).",
        "",
    ]
    for run in runs:
        weights = weights_of(model, run["weights"])
        first, *again = run["demo"]
        who = _weights_label(method, weights, run["weights"])
        text = (
            f"Measured with {who}: **{first['wall_s']:.1f} s** for the whole command; "
            f"{_cloud(first)}; {_counts(first)} ({_when(first, machines)})."
        )
        for later in again:
            same = later["predicted"] == first["predicted"] and later["after"] == first["after"]
            text += (
                f" Run again at commit `{later['commit']}` ({later['date']}): "
                + ("the same counts" if same else _counts(later))
                + f", {later['wall_s']:.1f} s."
            )
        details += [text, ""]
    for run in runs:
        for entry in run["python"]:
            details += [
                f"From Python (step 3): {entry['grasps']} grasps, {entry['wall_s']:.1f} s for the "
                f"script ({_when(entry, machines)}).",
                "",
            ]
    return [*lines, *_fold("What it prints and writes, and every measured run", details)]


def _anywhere_step(
    method: str,
    config: str,
    get: Weights,
    facts: dict[str, Any],
    runs: list[dict[str, Any]],
    machines: dict[str, Any],
) -> list[str]:
    """Step 3: the same from Python, in the browser and on a camera image, as content tabs."""
    file = get.converted_name
    demo = runs[0]["demo"][0]
    text = f"It runs the demo's pipeline on the demo's scene and returns its {_kept(demo)} grasps."
    if facts["mask"]:
        text += (
            " `mask=` stands in for the segmentation the config crops the workspace with; on "
            "your camera, pass a mask or `workspace=(xmin, ymin, zmin, xmax, ymax, zmax)` in "
            "metres."
        )
    else:
        text += f" {method} reads the colour image (`rgb=`) and takes no mask."
    python = [
        "```python",
        *python_snippet(config, file, facts),
        "```",
        "",
        text + " More: [From Python](../user_guides/inference.md#from-python).",
    ]
    browser = [
        *_shell([f"# ~/grasppanda_weights: the folder that holds {file}", explore_command()]),
        f"Open <http://127.0.0.1:8765/> and choose {method} in the **Run** tab: `{file}` is "
        "chosen for you by its name. Keep the example scene or upload a depth image with its "
        "intrinsics, and press *Run*: the page shows the picture, the best ten grasps, the "
        "counts through the post-processing and the command that does the same in a terminal. "
        "The server listens on your machine only and downloads nothing: "
        "[Run in the browser](../user_guides/inference.md#run-in-the-browser).",
    ]
    if facts["rgb"]:
        crop = (
            f"{method} reads the colour image registered to the depth image (`--rgb`) and crops "
            "no workspace."
        )
    else:
        crop = (
            f"{method} was trained on clouds cropped around the objects; on a camera image, "
            "`--workspace` (a box in the camera frame, metres) or `--mask` does that crop."
        )
    camera = [
        *_shell(
            _wrapped(
                camera_command(config, file, facts),
                (
                    "--weights",
                    "--depth",
                    "--intrinsics",
                    "--rgb",
                    "--workspace",
                    "--collision-filter",
                ),
            )
        ),
        f"The scene files of step 2 stand in for a camera. {crop} Units, frames, the depth "
        "scale and the intrinsics: [Your own camera](../user_guides/inference.md#your-own-camera).",
    ]
    return [
        "### 3. From Python, a browser or your camera {#run-anywhere}",
        "",
        '=== "Python"',
        "",
        *_indent(python),
        "",
        '=== "Browser"',
        "",
        *_indent(browser),
        "",
        '=== "Your camera"',
        "",
        *_indent(camera),
        "",
    ]


def _test_step(
    method: str,
    config: str,
    get: Weights,
    facts: dict[str, Any],
    model: ModelEntry,
    runs: list[dict[str, Any]],
    machines: dict[str, Any],
) -> list[str]:
    first = runs[0]["test"][0]
    command = subset_command(config, get.converted_name, first)
    variant = facts["variant"]
    rows, gaps = [], []
    lead_gaps: list[float] = []
    for index, run in enumerate(runs):
        weights = weights_of(model, run["weights"])
        provenance = provenance_of(weights)
        test = run["test"][0]
        for name, collision in (("raw", False), (variant, True)):
            subset = test["AP"][name]
            whole = _whole_split(model, provenance, collision)
            if whole is None:
                whole_text, gap = "not scored on the whole split", "—"
            else:
                value = whole.ap["seen"]
                record = f"{GITHUB}/configs/{page_of(method)}/{whole.record}"
                whole_text = f"{value:.2f} ([record]({record}))"
                gaps.append(abs(subset - value))
                if index == 0:
                    lead_gaps.append(abs(subset - value))
                gap = f"{subset - value:+.2f}".replace("-", "−")
            label = f"{_weights_label(method, weights, run['weights'])}, `{provenance}`"
            rows.append(
                [
                    label if name == "raw" else "",
                    "on" if collision else "off",
                    f"{subset:.2f}",
                    whole_text,
                    gap,
                    f"{test['wall_s']:g} s, `{test['commit']}`" if name == "raw" else "",
                ]
            )
    lead = runs[0]["test"][0]["AP"]
    used = _weights_label(method, weights_of(model, runs[0]["weights"]), runs[0]["weights"])
    measured = (
        f"Measured with {used}: "
        f"AP {lead['raw']:.2f} without the collision filter and {lead[variant]:.2f} with it"
    )
    measured += (
        f", within {max(lead_gaps):.2f} AP of the whole split."
        if lead_gaps
        else " (this file was not scored on the whole split)."
    )
    assisted = first["gt_assisted"]
    inputs = (
        "the workspace is cropped around the ground-truth segmentation "
        f"(`gt_assisted: {', '.join(assisted)}`), as in the reference"
        if assisted
        else "no input is derived from ground truth (the config crops no workspace)"
    )
    shown = {provenance_of(weights_of(model, run["weights"])) for run in runs}
    released = upstream_words(method)["weights"]
    meanings = [
        f"`{key}`, {text.format(weights=released)}"
        for key, text in PROVENANCE_TEXT.items()
        if key in shown
    ]
    text = f"Provenance: {'; '.join(meanings)}. Inputs: {inputs}. "
    if gaps:
        text += (
            f"On these {first['frames']} frames the AP is within {max(gaps):.2f} AP of the "
            "whole split, so the subset tells you quickly whether your installation gives "
            "what the records say; it is not a benchmark number. "
        )
    when = sorted(
        {
            f"{run['test'][0]['date']} on {machines[run['test'][0]['machine']]['gpu']}"
            for run in runs
        }
    )
    text += f"Measured on {'; '.join(when)}."
    details = sorted({machines[m["machine"]]["details"] for run in runs for m in run["demo"]})
    body = [
        "It prints AP, AP0.8 and AP0.4 for every variant of the config, `raw` (the "
        f"predictions as they are) and `{variant}` (with the collision filter), and writes a "
        "result record per variant into the run directory under `work_dirs/`. Testing needs "
        "no derived asset; the evaluator extracts a cache from the dataset's object models the "
        "first time it runs, which the times below do not include.",
        "",
        *_matrix(
            _table(
                [
                    "Weights, provenance",
                    "Collision filter",
                    f"AP, {first['frames']} frames of {first['split']}",
                    "AP, whole seen split",
                    "Difference",
                    "Wall time, commit",
                ],
                rows,
            )
        ),
        "",
        text,
        "",
        f"The machine: {'; '.join(details)}.",
    ]
    return [
        "### 4. A first number on the benchmark {#run-test}",
        "",
        "With GraspNet-1Billion on disk ([Datasets](../user_guides/datasets.md)), score every "
        f"{first['frame_step']}th frame of the seen test scenes, {first['frames']} frames:",
        "",
        *_shell(_wrapped(command, ("--weights", "--opts"))),
        measured,
        "",
        *_fold("What it prints, and every measured run against the whole split", body),
    ]


def run_section(method: str, data: dict[str, Any] | None = None) -> list[str]:
    """The section *Run <method> in five minutes* of a method page."""
    data = data or five_minutes()
    entry = data["methods"][method]
    config = option_lists.methods()[method]
    if entry["config"] != config:
        raise ValueError(f"{FIVE_MINUTES.name} measured {entry['config']}, not {config}")
    _, model = zoo_entry(method)
    get = downloadable(model)
    facts = pipeline_facts(config)
    machines = data["machines"]
    runs = entry["runs"]
    return [
        f"## Run {method} in five minutes {{#{RUN_ANCHOR}}}",
        "",
        f"Four steps from {upstream_words(method)['checkpoint']} to grasps. {TESTED}",
        "",
        *_weights_step(method, model, get),
        *_demo_step(method, config, get, facts, runs, machines, model),
        *_anywhere_step(method, config, get, facts, runs, machines),
        *_test_step(method, config, get, facts, model, runs, machines),
    ]


# ------------------------------------------------------------------------ the numbers


def _short_source(source: str) -> str:
    """``paper`` of ``paper (arXiv 2212.05275), Table 2, "Ours"``."""
    first = _top_level(_top_level(_top_level(source, ";"), ","), ".")
    return re.sub(r"\s*\([^)]*\)", "", first).strip()


def numbers_section(method: str) -> list[str]:
    """At most four whole-split rows of the shipped config, and where they come from."""
    _, model = zoo_entry(method)
    order = list(PROVENANCE_WORDS)
    filters = [False, True, None]
    results = sorted(
        (r for r in model.results if "seen" in r.ap),
        key=lambda r: (order.index(r.provenance), filters.index(r.collision_filter)),
    )
    if len(results) > 4:  # the measured rows first, then the reported ones
        keep = [r for r in results if r.provenance != "reported"][:4]
        keep += [r for r in results if r.provenance == "reported"][: 4 - len(keep)]
        results = [r for r in results if r in keep]
    rows = []
    for result in results:
        label = provenance_words(method, result.provenance, short=True)
        if result.provenance == "reported" and result.source:
            label += f" ({_short_source(result.source)})"
        record = f"{GITHUB}/configs/{page_of(method)}/{result.record}"
        rows.append(
            [
                label,
                FILTER_WORDS[result.collision_filter],
                *(f"{result.ap[split]:.2f}" for split in ("seen", "similar", "novel")),
                f"[record]({record})" if result.record else "—",
            ]
        )
    sources = list(
        dict.fromkeys(r.source for r in results if r.provenance == "reported" and r.source)
    )
    inputs = sorted({r.inputs for r in results if r.provenance != "reported" and r.inputs})
    status, _, meaning = numbers_status(method)
    body = [f"- **The status of these numbers**: {status}, {meaning}."]
    if sources:
        body.append("- **reported**: " + "; ".join(sources) + ".")
    if any(r.provenance == "author-ckpt" for r in results):
        body.append(
            f"- **{provenance_words(method, 'author-ckpt')}**: "
            f"{upstream_words(method)['weights']}, converted and run through `grasppanda test` "
            "on every frame of the test split; the record holds the per-scene values, the "
            "config hash and the hashes of code and weights."
        )
    if any(r.provenance == "reproduced" for r in results):
        body.append(
            f"- **{provenance_words(method, 'reproduced')}**: a model trained from scratch by "
            "GraspPanda with the config's recipe, scored the same way."
        )
    if inputs:
        body.append("- **Inputs**: " + "; ".join(inputs) + ".")
    body.append(
        "- Other cameras, GraspClutter6D and what is known about each gap: "
        f"[the README]({_readme(method)}#results-on-graspnet-1billion); every method side by "
        "side: [Model zoo](../model_zoo.md)."
    )
    return [
        "## Numbers",
        "",
        f"GraspNet-1Billion, {CAMERAS.get(model.camera, model.camera)}, the whole test split: AP "
        f"on seen / similar / novel, from the [metafile]({GITHUB}/configs/{page_of(method)}/"
        "metafile.yaml).",
        "",
        *_matrix(
            _table(["Weights", "Collision filter", "Seen", "Similar", "Novel", "Record"], rows)
        ),
        "",
        *_fold("Where these numbers come from", body),
    ]


# -------------------------------------------------------------- differences and next


def differences(method: str) -> list[str]:
    """The fold *How this port differs from the reference* of the method's README, with its
    relative links made absolute."""
    readme = REPO / "configs" / page_of(method) / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = (
        r'<details markdown="1"><summary>' + re.escape(DIFFERENCES) + r"</summary>\n(.*?)</details>"
    )
    found = re.search(pattern, text, re.S)
    if found is None:
        raise ValueError(f"{readme.relative_to(REPO)} has no fold '{DIFFERENCES}'")
    base = f"{GITHUB}/configs/{page_of(method)}/"

    def absolute(match: re.Match[str]) -> str:
        target = match.group(1)
        if re.match(r"[a-z]+://", target):
            return match.group(0)
        if target.startswith("#"):
            return f"]({_readme(method)}{target})"
        parts: list[str] = []
        for part in (base.split("/blob/main/")[1] + target).split("/"):
            if part == "..":
                parts.pop()
            elif part and part != ".":
                parts.append(part)
        return f"]({GITHUB}/{'/'.join(parts)})"

    body = re.sub(r"\]\(([^)]+)\)", absolute, found.group(1)).strip().splitlines()
    body += [
        "",
        f"How it was checked, stage by stage: `{PARITY[method]}` and "
        f"[the README]({_readme(method)}).",
    ]
    return _fold(DIFFERENCES, body)


def next_section(method: str) -> list[str]:
    return [
        "## Next",
        "",
        f"- [Swap a component](../get_started/swap.md): change one slot of {method} with a "
        "single line.",
        f'- <a href="{EXPLORER}#m={quote(method)}">Open {method} in the explorer</a>: every '
        "option of every slot, with what was measured.",
        f"- [All {_words(len(chips().methods()))} methods](index.md) side by side, and the "
        "[model zoo](../model_zoo.md).",
    ]


# ------------------------------------------------------------------- the overview

#: The slots of a detector model, for the table of what each method ships.
MODEL_SLOTS = ("image_encoder", "backbone", "proposal", "view", "assigner", "grouper", "head")


@functools.cache
def config_facts(folder: str) -> tuple[dict[str, Any], ...]:
    """Every config of a method folder: the dataset and camera it trains and tests on."""
    found = []
    for path in sorted((REPO / "configs" / folder).glob("*.yaml")):
        if path.name == "metafile.yaml":
            continue
        name = f"{folder}/{path.stem}"
        data = Config.load(name).resolved()["data"]
        train, test = data["train"]["dataset"], data["test"]["dataset"]
        found.append(
            {
                "name": name,
                "train": (train["type"], train.get("camera") or ""),
                "test": (test["type"], test.get("camera") or ""),
            }
        )
    return tuple(found)


def _words(count: int) -> str:
    """A small count in words (``six``)."""
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    return words[count] if count < len(words) else str(count)


def _cameras(names: Any) -> str:
    return ", ".join(dict.fromkeys(CAMERAS.get(name, name) for name in names)) or "—"


def capabilities(method: str) -> list[str]:
    """A row of the method × capability table, from the metafile and the configs (the
    measured demo time is the chooser's *Demo* column, included below the table)."""
    info = chips().method(method)
    meta = metafile_of(method)
    trains = [m for m in meta.models if m.protocol.startswith("graspnet1b")]
    author = [
        m.camera
        for m in trains
        if any(w is not None and w.source == "author" for w in (m.weights, m.author_weights))
    ]
    trained = [m for m in trains if any(r.provenance == "reproduced" for r in m.results)]
    trained_text = "—"
    if trained:
        published = all(m.weights is not None and m.weights.published for m in trained)
        trained_text = _cameras(m.camera for m in trained)
        trained_text += "" if published else ", not published"
    configs = config_facts(info.folder)
    g1b, gc6d = "GraspNet1B", "GraspClutter6D"
    on_g1b = [c["train"][1] for c in configs if c["train"][0] == g1b and c["test"][0] == g1b]
    transfer = [c["train"][1] for c in configs if c["train"][0] == g1b and c["test"][0] == gc6d]
    on_gc6d = [c["train"][1] for c in configs if c["train"][0] == gc6d]
    gc6d_text = [f"transfer from {_cameras(transfer)}"] if transfer else []
    gc6d_text += ["training"] if on_gc6d else []
    return [
        f"[{method}]({info.folder}.md)",
        _cameras(author),
        trained_text,
        _cameras(on_g1b),
        "; ".join(gc6d_text) or "—",
    ]


def ships(records: option_lists.Measured) -> list[str]:
    """The component each method's config puts in every model slot."""
    dc = chips()
    names = [info.name for info in dc.methods()]
    rows = []
    for slot in MODEL_SLOTS:
        cells = []
        for name in names:
            if slot not in option_lists.method_slots(name):
                cells.append("—")
                continue
            shipped = option_lists.slot_listing(name, slot, records).shipped
            if "built in" in shipped:
                shipped = "built in"
            cells.append(f"`{shipped}`" if shipped and " " not in shipped else shipped or "—")
        rows.append([dc.SLOT_TITLES[slot], *cells])
    body = [
        "The component each method's config puts in every slot of its model; — where the "
        "method has no such slot.",
        "{: #what-each-method-ships }",
        "",
        *_matrix(_table(["Slot", *names], rows)),
    ]
    return _fold("What each method ships in each slot", _raw(body))


def config_extras(config: str) -> list[str]:
    """The extras of ``pyproject.toml`` the components of a config need (by library)."""
    from grasppanda.config.validate import iter_components

    found: list[str] = []
    for _, registry, name in iter_components(Config.load(config).to_dict()):
        for extra in getattr(registry.get(name), "extras", ()) or ():
            if extra not in found:
                found.append(extra)
    return found


def chooser() -> str:
    """``docs/includes/choose_method.md``: what each method reads and needs before its
    five-minute run. Its links work from ``docs/methods/`` and ``docs/get_started/``."""
    from grasppanda.utils.imports import EXTRA_BUILDS

    dc = chips()
    data = five_minutes()
    rows: list[list[str]] = []
    machines: list[str] = []
    extras: list[str] = []
    datasets: set[str] = set()
    plain: list[str] = []  # the methods that need no extra install
    for info in dc.methods():
        _, model = zoo_entry(info.name)
        get = downloadable(model)
        needs = config_extras(info.config)
        extras += [extra for extra in needs if extra not in extras]
        plain += [] if needs else [info.name]
        demo = data["methods"][info.name]["runs"][0]["demo"][0]
        machine = data["machines"][demo["machine"]]["gpu"]
        machines += [machine] if machine not in machines else []
        datasets |= {
            facts["train"][0] for facts in config_facts(info.folder) if facts["name"] == info.config
        }
        rows.append(
            [
                f"[{info.name}](../methods/{info.folder}.md)",
                info.inputs,
                ", ".join(f"`{extra}`" for extra in needs) or "none",
                host_of(get.url),
                f'<span title="{html.escape(get.license or "", quote=True)}">'
                f"{licence_short(get.license or '—')}</span>",
                f"{demo['wall_s']:.1f} s",  # a no-break space: the column is narrow
            ]
        )
    colour = [info.name for info in dc.methods() if info.inputs == "RGB-D"]
    first = f"{_names(plain)} {'needs' if len(plain) == 1 else 'need'} no extra install."
    if colour:
        first += (
            f" {_names(colour)} {'reads' if len(colour) == 1 else 'read'} a colour image "
            "registered to the depth image as well; the others read depth alone."
        )
    notes = []
    for extra in extras:
        builds = EXTRA_BUILDS.get(extra)
        if builds:
            notes.append(
                f"*Extra*: `{extra}` is installed as the extra that matches your PyTorch: "
                + ", ".join(f"`{build}`" for build in builds)
                + "."
            )
    notes.append("*Licence*: hover for the full text; each method page states it in full.")
    notes.append(
        f"*Demo* is the whole `grasppanda demo … --example` command, measured on "
        f"{' or '.join(machines)}; no run was timed on a CPU."
    )
    if datasets == {"GraspNet1B"}:
        notes.append(
            "Every checkpoint was trained on GraspNet-1Billion, whose data and labels are under "
            f"{G1B_LICENCE}."
        )
    lines = [
        "<!-- Generated by tools/make_method_pages.py from the configs, the metafiles and "
        "five_minutes.json; do not edit. -->",
        "",
        first,
        "",
        *_matrix(_table(["Method", "Reads", "Extra", "Weights<br>from", "Licence", "Demo"], rows)),
        "",
        *(f"- {note}" for note in notes),
    ]
    return "\n".join(lines) + "\n"


def index_page(records: option_lists.Measured) -> str:
    """``docs/methods/index.md``: the methods by family, what each comes with, and the
    chooser."""
    dc = chips()
    every = dc.methods()
    seed = [info for info in every if info.family == "seed-based"]
    own = [info for info in every if info.family != "seed-based"]
    third = [info.name for info in every if upstream_words(info.name)["third_party"]]
    others = ""
    if third:
        repositories = [upstream_words(name)["repository"] for name in third]
        links = ", ".join(f"[{repo}](https://github.com/{repo})" for repo in repositories)
        others = (
            f" For {_names(third)}, the reference code is a third-party implementation ({links})."
        )
    lines = [
        "---",
        "title: Methods",  # MkDocs takes no title from a heading below a comment
        "hide:",
        "  - toc",  # the wide tables of this page take the room of the table of contents
        "---",
        "",
        "<!-- Generated by tools/make_method_pages.py from the metafiles, the configs and",
        "     five_minutes.json. Do not edit by hand. -->",
        "",
        "# Methods",
        "",
        f"{_words(len(every)).capitalize()} published methods, each ported stage by stage and "
        "checked against its reference code, the same weights loaded into both."
        f"{others} {_words(len(seed)).capitalize()} share the seed-based skeleton and its "
        f"slots; {', '.join(info.name for info in own)} reads RGB-D images and is a family of "
        "its own.",
        "",
    ]
    for family, members in (("seed-based", seed), ("own", own)):
        title = dc.FAMILY_NAMES[family]
        if len(members) > 1:
            title += f" · {len(members)} methods"
        lines += [f"## {title}", "", '<div class="grid cards" markdown>', ""]
        for info in members:
            status, tone, meaning = numbers_status(info.name)
            lines += [
                f"-   **[{info.name}]({info.folder}.md)** · {info.venue} · {info.inputs}",
                "",
                "    ---",
                "",
                f"    {WHAT[info.name][1]}",
                "",
                f"    {equal_chip(info.name)} {dc.chip(status, tone, meaning)}",
                "",
            ]
        lines += ["</div>", ""]
    lines += [
        "## What each method comes with",
        "",
        "Computed from the configs and the metafiles.",
        "",
        *_matrix(
            _table(
                [
                    "Method",
                    "Released<br>weights",
                    "Trained<br>here",
                    "GraspNet-1Billion<br>configs",
                    "GraspClutter6D<br>configs",
                ],
                [capabilities(info.name) for info in every],
            )
        ),
        "",
        *ships(records),
        "## Which one to run first",
        "",
        "Every method page starts with a five-minute run from released weights.",
        # no blank line: the include then starts with no indentation in every tool that reads it
        '--8<-- "docs/includes/choose_method.md"',
    ]
    return "\n".join(lines).rstrip() + "\n"


def page(method: str, records: option_lists.Measured) -> str:
    listings = option_lists.listing(method, records=records)
    lines = [
        "<!-- Generated by tools/make_method_pages.py from the metafile, the config, the README,",
        "     five_minutes.json, the presets, the grid, measured_combinations.json and",
        "     learns_like_reference.json. Do not edit by hand. -->",
        "",
        f"# {method}",
        "",
        head_chips(method),
        "",
        WHAT[method][0],
        "",
        links_line(method),
        "",
        *run_section(method),
        *numbers_section(method),
        *slots_section(method, listings, records),
        *differences(method),
        *next_section(method),
    ]
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------- the slot pages
#
# A slot page (docs/slots/<stem>.md) is written by hand around generated regions:
#
# * HEAD_BEGIN … HEAD_END, under the title: the methods the slot applies to, how a command
#   names it, how many options the page lists;
# * BEGIN … END, under "## Use it": the slot under every method that has it, one linked
#   content tab per method with an option that ran there as a swap, and the link to each
#   method's own list of the slot;
# * OPTIONS_BEGIN … OPTIONS_END, under "## Options": one row per option with where it runs
#   and how it learns (tools/docs_chips.py, which reads grasppanda.combine.reach), the probe
#   caveat, and the link to the explorer;
# * REACH_BEGIN … REACH_END, on the page of the image encoders: where point painting stands.
#
# The options themselves are enumerated by make_combination_docs.slot_options, which the slot
# index counts too. Nothing measured is typed into a slot page by hand.

#: The marks of the chips under the title of a slot page.
HEAD_BEGIN = "<!-- slot-head:begin (generated by tools/make_method_pages.py; do not edit) -->"
HEAD_END = "<!-- slot-head:end -->"
#: The marks of the "Options" region of a slot page.
OPTIONS_BEGIN = "<!-- slot-options:begin (generated by tools/make_method_pages.py; do not edit) -->"
OPTIONS_END = "<!-- slot-options:end -->"
#: The marks of a line that says where options of another slot stand (point painting on the
#: page of the image encoders), and those options per page.
REACH_BEGIN = "<!-- slot-reach:begin (generated by tools/make_method_pages.py; do not edit) -->"
REACH_END = "<!-- slot-reach:end -->"
REACH_LINES = {
    "image_encoders": (("backbone", "image-painted"), ("backbone", "image-painted-subsampled"))
}
#: The probe caveat: once per slot page, in its "Options" region.
CAVEAT_SNIPPET = '--8<-- "docs/includes/caveat_probe.md"'
#: The scene-encoder pages and the encoder types each documents. The overview
#: (``scene_encoders``) lists the rest: the adapters compose inserts and point painting.
ENCODER_PAGES = {
    "scene_encoders_point": (
        "ASSANet",
        "DGCNN",
        "DeLA",
        "PointMLP",
        "PointMetaBase",
        "PointNeXt",
        "PointNet2",
        "PointNet2Dense",
        "PointNet2MSG",
        "PointNet2MSGDense",
        "PointNet2Transformer",
        "PointVector",
        "PyGPointEncoder",
        "RandLANet",
    ),
    "scene_encoders_sparse": ("MinkUNet", "OACNNs", "SpUNet", "SparseUNet14"),
    "scene_encoders_transformer": (
        "LitePT",
        "OctFormer",
        "PTv3",
        "PTv3Sonata",
        "PTv3Utonia",
        "PointMamba",
        "PointTransformerV1",
        "PointTransformerV2",
        "PointViT",
        "Uni3D",
        "Volt",
    ),
}
#: Slot → (title of a content tab, the component types of that kind). A slot with more than
#: one kind on its page gets one tab per kind; a type no kind names goes to the tab "Other".
KINDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "image_encoder": (
        ("timm", ("TimmImageEncoder",)),
        ("Hugging Face transformers", ("HFImageEncoder",)),
        ("torchvision", ("TorchvisionImageEncoder",)),
        ("HGGD's own", ("HGGDResidualEncoder",)),
    ),
    "proposal": (
        ("Score the points, then sample", ("ScoredSeeds", "GraspnessFPS")),
        ("Sample without a score", ("SampledSeeds", "SubsampledSeeds")),
    ),
    "view": (
        (
            "Score a lattice of views",
            (
                "TemplateViewStage",
                "LatticeViewStage",
                "IcosphereViewStage",
                "HealpixViewStage",
                "ObjectnessViewStage",
            ),
        ),
        (
            "Regress a direction or a frame",
            ("DirectionViewStage", "TwoVectorViewStage", "RomaRotationViewStage"),
        ),
    ),
}
#: The kinds of a grouper follow the axes of its crop feature (``out_axes``): they decide
#: which heads can read it.
GROUPER_KINDS = {
    ("seed", "channel"): "One feature per seed",
    ("seed", "depth", "channel"): "One feature per depth bin",
}
#: The kinds of a loss follow the kinds of term it accepts
#: (``grasppanda.models.losses.terms``): (title of its tab, test on those kinds).
LOSS_KINDS = (
    ("For class terms", lambda kinds: "class" in kinds and "value" not in kinds),
    ("For quality and value terms", lambda kinds: bool(kinds) and kinds <= {"unit", "value"}),
    ("For bins, or any term", lambda kinds: True),
)
#: Slot pages whose slots stand one under the other (else: one content tab per slot).
STACKED = ("head_loss",)
#: Stem → the options the "Use it" example prefers. A method's example is the first of them
#: that ran under it as a swap, else the first option of the page that did.
EXAMPLES = {
    "image_encoders": ("resnet34",),
    "scene_encoders": ("ptv3", "pointnext"),
    "scene_encoders_point": ("pointnext", "pointnet2-dense"),
    "scene_encoders_sparse": ("minkunet34c", "spunet-gaussiancross"),
    "scene_encoders_transformer": ("ptv3", "point-transformer-v2"),
    "proposal": ("graspness-sfps", "dense-fps"),
    "view": ("lattice-view", "healpix-768"),
    "grouper": ("roi-grid", "depth-box"),
    "head_loss": ("swad-focal", "decoupled-grid"),
    "losses": ("focal", "huber", "charbonnier"),
    "post": ("nms", "top-k"),
    "recipe": ("adamw", "timm-lamb"),
}
#: A tail (who is left out, and why) at most this long stands in the table under its chip; a
#: longer one is listed in the fold under the table (a tooltip is not seen on a touch screen).
TAIL_INLINE = 60
#: The attribute list that keeps a code span on one line (``attr_list``).
NOWRAP = "{ .gp-nowrap }"
#: Slots whose options table says under each name the kinds of term the option fits.
TERM_KINDS_UNDER_NAME = ("loss",)
#: The wrapper of every options table (``docs/stylesheets/extra.css``: on a phone the
#: column "What it is", the second, gives way), and of a fold that holds a raw table
#: (search lands on the prose instead).
OPTIONS_DIV = ('<div class="gp-options" markdown>', "</div>")
#: Slot pages that say by hand where every option comes from, so no generated fold "Sources
#: and extras" repeats it: the fold "Sources and licences" of the losses (each by name) and of
#: the view (each stage by its class, which the option's description names); the image
#: encoders (each library in the label of its tab, `hggd-residual` as HGGD's own).
SOURCES_BY_HAND = ("image_encoders", "view", "losses")
SEARCH_EXCLUDED_DIV = ("<div data-search-exclude markdown>", "</div>")
#: The register of adapted third-party code: per upstream repository its URL, commit,
#: licence and the files (classes) of this repository that adapt it.
NOTICES = REPO / "THIRD_PARTY_NOTICES.md"
#: What the Code column says of an in-tree class the register does not list.
WRITTEN_HERE = "written here"


def _chips() -> Any:
    """``tools/docs_chips.py``, loaded once by ``make_combination_docs``."""
    return _docs_tool()._chips()


def option_page(slot: str, name: str) -> str:
    """The stem of the slot page that lists option ``name`` of ``slot``: for a scene encoder
    the page of its kind, else the page of the slot."""
    if slot != "backbone":
        return SLOT_PAGE[slot]
    found = next((o for o in _docs_tool().slot_options(slot) if o.name == name), None)
    for stem, types in ENCODER_PAGES.items():
        if found is not None and found.component in types:
            return stem
    return SLOT_PAGE[slot]


def _page_options(stem: str, slot: str) -> list[Any]:
    """The options of ``slot`` that page ``stem`` lists."""
    return [o for o in _docs_tool().slot_options(slot) if option_page(slot, o.name) == stem]


def _kind(slot: str, option: Any) -> str:
    """The content tab of an option on its page."""
    if slot == "grouper":
        from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

        cls = STAGE_REGISTRIES["grouper"].get(option.component)
        return GROUPER_KINDS.get(tuple(getattr(cls, "out_axes", ())), "Other")
    if slot == "loss":
        kinds = set(_term_kinds(option))
        for title, test in LOSS_KINDS:
            if test(kinds):
                return title
    for title, types in KINDS.get(slot, ()):
        if option.component in types:
            return title
    return "Other"


def _plain(text: str) -> str:
    """Text for a table cell: one line, no pipe, no tag."""
    return " ".join(text.replace("|", "\\|").replace("<", "&lt;").split())


def _top_level_split(text: str, separator: str) -> list[str]:
    """``text`` split at ``separator`` where it stands outside parentheses and code."""
    parts, depth, start, index, code = [], 0, 0, 0, False
    while index < len(text):
        depth += (text[index] == "(") - (text[index] == ")")
        code ^= text[index] == "`"
        if depth == 0 and not code and text.startswith(separator, index):
            parts.append(text[start:index])
            index += len(separator)
            start = index
        else:
            index += 1
    return [*parts, text[start:]]


#: The authors of a citation, ``(Qi et al., `` or ``(Ma and Huang, ``: a table keeps the venue.
_AUTHORS = re.compile(r"\((?:[A-Z][\w'-]+)(?: [A-Z][\w'-]+)*(?: et al\.| and [A-Z][\w'-]+)?, ")


def short_source(source: str) -> str:
    """The catalogue's ``source`` of a component, short enough for a table cell: the first
    work it names, with its venue (``PointNeXt (NeurIPS 2022)``)."""
    if not source or source == "?":
        return "—"
    parts = [part.strip() for part in _top_level_split(source, "; ") if part.strip()]
    first = parts[0]
    if first.lower() == "generic" and len(parts) > 1:
        first = parts[1]
    for separator in (", adapted from ", ", layout of ", ", via ", ": "):
        first = _top_level_split(first, separator)[0]
    first = _AUTHORS.sub("(", first)
    # A parenthesis without a number (a year, an arXiv id) is a gloss, not a citation.
    first = re.sub(r"\s*\([^()\d]*\)", "", first).strip().rstrip(",")
    if len(first.split()) > 10 and ")" in first:
        first = first[: first.index(")") + 1] + " …"
    if len(first.split()) > 12 and " (" in first:
        first = first[: first.index(" (")] + " …"
    # A sentence starts with a capital; a bare module name (``torch``) stays as written.
    return first[:1].upper() + first[1:] if " " in first else first


def _term_kinds(option: Any) -> tuple[str, ...]:
    """The kinds of loss term a loss preset fits (``grasppanda.models.losses.terms``)."""
    from grasppanda.combine.presets import presets
    from grasppanda.models.losses.terms import accepted_terms

    preset = presets().get(option.slot, {}).get(option.name)
    found = accepted_terms(preset.node) if preset is not None else None
    return tuple(found[1]) if found else ()


def _fits(option: Any) -> str:
    return ", ".join(f"`{kind}`" for kind in _term_kinds(option)) or "—"


def _with_tail(chip: str, tail: str, lead: str = "") -> str:
    """A chip, with ``lead`` under it (always) and its tail (when short enough)."""
    notes = [lead] if lead else []
    if tail and len(tail) <= TAIL_INLINE:
        notes.append(_plain(tail))
    return chip + "".join(f"<br><small>{note}</small>" for note in notes)


def _source_cell(option: Any, *, code: bool = True, full: bool = False) -> str:
    """Where an option comes from, short (``PointNeXt (NeurIPS 2022) · extra `spconv```), or
    as the catalogue says it (``full``, for a fold); ``code=False`` for a content tab's label,
    which shows no code span and names only the extras the source does not already name
    (``Hugging Face Transformers``)."""
    if full and option.source not in ("", "?"):
        # A sentence starts with a capital; a bare module name (``torch``) stays as written.
        whole = option.source.strip()
        source = _plain(whole[:1].upper() + whole[1:] if " " in whole else whole)
    else:
        source = _plain(short_source(option.source))
    if not code:
        named = [extra for extra in option.extras if extra.lower() not in source.lower()]
        return f"{source} (extra {', '.join(named)})" if named else source
    if option.extras:
        # A short name never breaks; a long one (`torch_geometric`) is kept on one line.
        names = [f"`{e}`" + (NOWRAP if len(e) > 8 else "") for e in option.extras]
        source += " · extra " + ", ".join(names)
    return source


def _shared_source(options: list[Any]) -> str:
    """The source every one of several options shares (for the label of their tab, so the
    table does not repeat it on every row); ``""`` when they differ or are one."""
    found = {_source_cell(option, code=False) for option in options}
    return found.pop() if len(options) > 1 and len(found) == 1 else ""


def _only_under(slot: str, name: str) -> tuple[str, str]:
    """``(short, why)`` for a recipe preset that applies to fewer methods than its slot
    (``grasppanda.combine.grid.RECIPE_ENCODER_PRESETS``: the presets that name the scene
    encoder, which compose refuses under a method without one); ``("", "")`` otherwise."""
    from grasppanda.combine.grid import RECIPE_ENCODER_PRESETS, RECIPE_REQUIRES

    key = f"{slot}/{name}"
    under = RECIPE_ENCODER_PRESETS.get(key)
    if under is None:
        return "", ""
    chips = _chips()
    left = [method for method in chips.slot_methods(slot) if method not in under]
    if not left:
        return "", ""
    seed = [info.name for info in chips.methods() if info.family == "seed-based"]
    short = "seed-based only" if sorted(under) == sorted(seed) else f"not {', '.join(left)}"
    if key in RECIPE_REQUIRES:
        needs = " and ".join(f"`--set {request}`" for request in RECIPE_REQUIRES[key])
        why = f"it needs {needs}, a slot {' and '.join(left)} does not have"
    else:
        why = "; ".join(option_lists.ENCODER_RECIPE_REASON.format(method=m) for m in left)
    return short, why


def _inside(slot: str, name: str) -> str:
    """For a loss preset measured inside a head bundle (``reach.LOSS_INSIDE``): where."""
    if slot != "loss" or name not in reaching.LOSS_INSIDE:
        return ""
    head, _ = reaching.LOSS_INSIDE[name]
    return f"measured in `{head}`"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def _has_level2(slot: str, options: list[Any]) -> bool:
    """Whether a table of ``options`` has a level-2 column: not for a slot without level 2
    (``grasppanda.combine.reach.LEVEL2_NONE``), nor for options never compared."""
    from grasppanda.combine.reach import LEVEL2_NONE

    chips = _chips()
    return slot not in LEVEL2_NONE and any(chips.level2(slot, o.name).outcomes for o in options)


def options_table(slot: str, options: list[Any], long_tails: list[str]) -> list[str]:
    """The options table of ``slot`` in its wrapper: option, what it is, where it runs,
    whether it trains like shipped. The order of the columns is the one the stylesheet thins
    on a phone; where an option comes from is in a fold (:func:`_origins`, :func:`_sources`).
    A tail too long for its cell is appended to ``long_tails``."""
    chips = _chips()
    learns = _has_level2(slot, options)
    headers = ["Option", "What it is", "Runs", *([chips.LEVEL2_HEADER] if learns else [])]
    rows = []
    for option in options:
        # The name is what a reader copies: it never breaks across lines.
        name = f"`{option.name}`{NOWRAP}"
        if slot in TERM_KINDS_UNDER_NAME and _term_kinds(option):
            name += f"<br><small>fits {_fits(option)}</small>"
        cells = [name, _plain(option.summary) or "—"]
        found = chips.reach(slot, option.name)
        short, why = _only_under(slot, option.name)
        lead = short or _inside(slot, option.name)
        cells.append(
            _with_tail(chips.chip(found.fraction, found.tone, found.tail), found.tail, lead)
        )
        if why:
            long_tails.append(f"`{option.name}`, runs: {short}: {_plain(why)}.")
        if len(found.tail) > TAIL_INLINE:
            long_tails.append(f"`{option.name}`, runs: {_plain(found.tail)}.")
        if learns:
            second = chips.level2(slot, option.name)
            chip = chips.chip(second.fraction, second.tone, second.tail)
            cells.append(_with_tail(chip, second.tail))
            if len(second.tail) > TAIL_INLINE:
                long_tails.append(f"`{option.name}`, {chips.LEVEL2_NAME}: {_plain(second.tail)}.")
        rows.append(cells)
    begin, end = OPTIONS_DIV
    return [begin, "", *_table(headers, rows), "", end]


def _origins(options: list[Any]) -> list[str]:
    """A fold with where each of ``options`` comes from: the work it follows and the extra
    it needs, the options of one source on one row; nothing for no option."""
    by_source: dict[str, list[str]] = {}
    for option in options:
        source = _source_cell(option, full=True)
        if source != "—":  # no source given: nothing to say
            by_source.setdefault(source, []).append(f"`{option.name}`")
    if not by_source:
        return []
    rows = [[source, ", ".join(names)] for source, names in by_source.items()]
    begin, end = SEARCH_EXCLUDED_DIV
    return [
        '??? note "Sources and extras"',
        "",
        "    The work each option follows; *extra*: the optional install it needs "
        "([every extra](../get_started/installation.md)).",
        "",
        f"    {begin}",
        "",
        *[f"    {line}" for line in _table(["Source", "Options"], rows)],
        "",
        f"    {end}",
        "",
    ]


def _tabbed(groups: list[tuple[str, list[str]]]) -> list[str]:
    """One content tab per ``(title, lines)``; a single group is written without a tab."""
    if len(groups) == 1:
        return [*groups[0][1], ""]
    lines = []
    for title, body in groups:
        lines += [f'=== "{title}"', "", *[f"    {line}" if line else "" for line in body], ""]
    return lines


def _plural(slot: str) -> str:
    return _docs_tool().OPTION_WORDS.get(slot, ("option", "options"))[1]


def _page_title(stem: str) -> str:
    """The title of a slot page: its first heading of level 1 (after the front matter)."""
    lines = (SLOT_PAGES / f"{stem}.md").read_text(encoding="utf-8").splitlines()
    return next(line[2:].strip() for line in lines if line.startswith("# "))


def _encoder_kinds() -> list[str]:
    """The overview of the scene encoders: which page lists which encoder."""
    rows = []
    for stem in ENCODER_PAGES:
        names = [option.name for option in _page_options(stem, "backbone")]
        rows.append(
            [
                f"[{_page_title(stem)}]({stem}.md)",
                str(len(names)),
                ", ".join(f"`{name}`" for name in names),
            ]
        )
    return [*_table(["Page", "Options", "Encoders"], rows), ""]


def _download(url: str) -> str:
    """The link of a released file: ``hub @ `5e25cfa``` for a revision-pinned Hugging Face
    file, else the host."""
    parts = urlsplit(url)
    pinned = re.match(r"/[^/]+/[^/]+/resolve/([0-9a-f]{7,40})/", parts.path)
    # A file name in a link is no prose: the spell check skips the line (an invisible comment).
    quiet = "<!-- codespell:ignore -->"
    if parts.netloc == "huggingface.co" and pinned:
        return f"[hub @ `{pinned.group(1)[:7]}`]({url}){quiet}"
    return f"[{parts.netloc.removeprefix('www.')}]({url}){quiet}"


def _released(stem: str) -> list[str]:
    """A fold with the released checkpoints that load by name into the encoders of a page
    (``grasppanda.models.backbones._pretrained.RELEASED``): what they are and where they
    are fetched from, pinned to a hub revision where the host has one."""
    from grasppanda.models.backbones._pretrained import RELEASED

    rows = []
    for name, found in RELEASED.items():
        if found.component not in ENCODER_PAGES[stem]:
            continue
        options = ", ".join(f"`{key}: {json.dumps(value)}`" for key, value in found.options.items())
        rows.append(
            [
                f"`{name}`",
                f"`{found.component}`",
                _plain(found.paper),
                _plain(found.trained_on),
                _plain(found.licence),
                str(found.megabytes),
                _plain(options) or "defaults",
                _download(found.url),
            ]
        )
    if not rows:
        return []
    headers = ["`pretrained:`", "Class", "Work", "Trained on", "Licence", "MB", "Strict load with"]
    begin, end = SEARCH_EXCLUDED_DIV
    return [
        '??? note "Released weights that load by name"',
        "",
        "    Each file was downloaded and loaded strictly into its class (every tensor of the "
        "class from the file, with its shape). Write the name as `pretrained:` of the encoder; "
        "building the model without the file prints the command that fetches it. A file on a "
        "hub is fetched at the revision it was loaded from (the last column). No file was "
        "evaluated for grasping.",
        "",
        f"    {begin}",
        "",
        *[f"    {line}" for line in _table([*headers, "File, revision"], rows)],
        "",
        f"    {end}",
        "",
    ]


@functools.lru_cache(maxsize=1)
def third_party_code() -> tuple[tuple[str, tuple[str, ...], str], ...]:
    """``(file, classes, code)`` for every file of this repository that
    ``THIRD_PARTY_NOTICES.md`` lists: ``code`` is the upstream repository (linked), its
    commit and its licence; ``classes`` are the classes the entry names (none: every class
    of the file)."""
    found = []
    for section in re.split(r"^## ", NOTICES.read_text(encoding="utf-8"), flags=re.M)[1:]:
        repository = re.search(r"^- Repository: (\S+)", section, re.M)
        commit = re.search(r"^- Commit: `([0-9a-f]+)`", section, re.M)
        licence = re.search(r"^- Licence: (\S+)", section, re.M)
        if not (repository and commit and licence):
            continue
        url = repository.group(1)
        name = url.removeprefix("https://github.com/")
        code = f"[{name}]({url}) @ `{commit.group(1)[:7]}`, {licence.group(1).rstrip('.')}"
        for match in re.finditer(r"`(grasppanda/[\w/]+\.py)`(\s*\(class(?:es)? [^;)]*)?", section):
            classes = tuple(re.findall(r"`([A-Za-z_]\w*)`", match.group(2) or ""))
            found.append((match.group(1), classes, code))
    return tuple(found)


def code_of(entry: Any) -> str:
    """The code base of a registered class (a catalogue entry): what
    ``THIRD_PARTY_NOTICES.md`` lists for its file, else the library a bridge calls, else
    :data:`WRITTEN_HERE`."""
    path = entry.module.replace(".", "/") + ".py"
    codes = [
        code
        for file, classes, code in third_party_code()
        if file == path and (not classes or entry.name in classes)
    ]
    if codes:
        return "; ".join(dict.fromkeys(codes))
    if entry.status == "bridge" and entry.extras:
        return "bridge to " + ", ".join(f"`{extra}`" for extra in entry.extras)
    return WRITTEN_HERE


def _work(source: str) -> str:
    """The work a catalogue ``source`` names, without the code it was taken from."""
    first = _top_level_split(source, "; ")[0]
    for separator in (", adapted from ", ", layout of "):
        first = _top_level_split(first, separator)[0]
    return first.strip() or "—"


def _output(options: list[Any]) -> str:
    """Where the features of an encoder's presets live and how wide they are, read from
    the class without building it (``configured_width``, ``configured_points``); a free
    ``out_channels`` is set by compose."""
    from grasppanda.combine.compose import is_adapter
    from grasppanda.combine.presets import presets
    from grasppanda.models.base import configured_points, configured_width
    from grasppanda.registry import BACKBONES, component_parameters

    places: dict[str, None] = {}
    widths: dict[str, list[str]] = {}
    for option in options:
        node = presets()["backbone"][option.name].node
        cls = BACKBONES.get(node["type"])
        params, _ = component_parameters(cls)
        kind, points = str(getattr(cls, "output", "")), configured_points(node)
        places[f"{kind}, {points} points" if kind == "subsampled" and points else kind] = None
        width = configured_width(node)
        if "out_channels" in params:
            text = "width set by compose"
        elif width:
            text = str(width)
        else:
            text = "the width of the encoder it wraps" if is_adapter(cls) else "width not declared"
        widths.setdefault(text, []).append(option.name)
    place = " / ".join(places)
    if len(widths) == 1:
        return f"{place} · {next(iter(widths))}"
    ranked = sorted(widths.items(), key=lambda item: -len(item[1]))
    if len(ranked[0][1]) > len(ranked[1][1]):
        rest = [f"`{n}`: {w}" for w, names in ranked[1:] for n in names]
        return f"{place} · {ranked[0][0]}; " + ", ".join(rest)
    named = [f"`{n}` {w}" for w, names in widths.items() for n in names]
    return f"{place} · " + ", ".join(named)


def _sources(stem: str) -> list[str]:
    """A fold with the classes of a scene-encoder page: the work each implements, its code
    base and licence, the extra it needs, and the features its presets return (the catalogue
    of ``grasppanda components`` and ``THIRD_PARTY_NOTICES.md``)."""
    catalogue = _docs_tool()._catalogue()
    by_class: dict[str, list[Any]] = {}
    for option in _page_options(stem, "backbone"):
        by_class.setdefault(option.component, []).append(option)
    rows = []
    for name, options in by_class.items():
        entry = catalogue.get(("BACKBONES", name))
        if entry is None:
            continue
        extras = dict.fromkeys(extra for option in options for extra in option.extras)
        rows.append(
            [
                f"`{name}`",
                ", ".join(f"`{option.name}`" for option in options),
                _plain(_work(entry.source)),
                code_of(entry),
                ", ".join(f"`{extra}`" for extra in extras) or "—",
                _output(options),
            ]
        )
    begin, end = SEARCH_EXCLUDED_DIV
    return [
        '??? note "Sources and licences"',
        "",
        "    *Code base*: the upstream repository, commit and licence that "
        "`THIRD_PARTY_NOTICES.md` lists for the class's file (code adapted from it, or a "
        "module tree that follows it so that its weights load); *written here*: no "
        "third-party code. *Extra*: the optional install the class needs "
        "([every extra](../get_started/installation.md)). *Output*: where the features live "
        "and how many channels the preset returns; compose sets a free `out_channels` to the "
        "width the method's other stages need. Released weights carry licences of their own.",
        "",
        f"    {begin}",
        "",
        *[
            f"    {line}"
            for line in _table(["Class", "Options", "Work", "Code base", "Extra", "Output"], rows)
        ],
        "",
        f"    {end}",
        "",
    ]


def _loss_terms() -> list[str]:
    """A fold with the kinds of loss term and the terms of every method that has the slot,
    read from the stages of the shipped configs."""
    from grasppanda.combine.grid import LOSS_STAGES
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES
    from grasppanda.models.losses.terms import TERM_KINDS

    rows = []
    for method in _chips().slot_methods("loss"):
        model = Config.load(option_lists.methods()[method]).resolved()["model"]
        terms = []
        for slot in LOSS_STAGES:
            node = model[slot]
            cls = STAGE_REGISTRIES[slot].get(node["type"])
            options = {name: value for name, value in node.items() if name != "type"}
            reader = getattr(cls, "loss_terms_for", None)
            found = reader(options) if reader else getattr(cls, "loss_terms", {})
            terms += [f"`{slot}.{option}` ({kind})" for option, kind in found.items()]
        rows.append([method, ", ".join(terms)])
    return [
        '??? note "The loss terms of every method, and their kinds"',
        "",
        *[f"    - `{kind}`: {meaning}." for kind, meaning in TERM_KINDS.items()],
        "",
        *[f"    {line}" for line in _table(["Method", "Term and its kind"], rows)],
        "",
    ]


def options_block(stem: str) -> list[str]:
    """The "Options" region of a slot page: the table(s), the probe caveat, what the columns
    say and who is left out, and the links to the explorer."""
    chips = _chips()
    slots = SLOT_PAGE_SLOTS[stem]
    long_tails: list[str] = []
    # (title, slot, options) of every table; a page with several becomes content tabs.
    tables: list[tuple[str, str, list[Any]]] = []
    for slot in slots:
        options = _page_options(stem, slot)
        if len(slots) > 1:
            tables.append((_plural(slot).capitalize(), slot, options))
            continue
        kinds: dict[str, list[Any]] = {}
        for option in options:
            kinds.setdefault(_kind(slot, option), []).append(option)
        order = [title for title, _ in (*KINDS.get(slot, ()), *LOSS_KINDS)]
        order += list(GROUPER_KINDS.values())
        for kind in sorted(kinds, key=lambda t: order.index(t) if t in order else len(order)):
            tables.append((kind, slot, kinds[kind]))
    groups: list[tuple[str, list[str]]] = []
    # The options whose source no label of a tab says: the fold "Sources and extras" lists them.
    unlabelled: list[Any] = []
    for title, slot, options in tables:
        # A tab whose rows share one source says it in its label.
        shared = _shared_source(options) if len(tables) > 1 and stem not in STACKED else ""
        unlabelled += [] if shared else options
        body = options_table(slot, options, long_tails)
        groups.append((f"{shared or title} · {len(options)}", body))
    lines = [OPTIONS_BEGIN, "", CAVEAT_SNIPPET, ""]
    if stem == "scene_encoders":
        lines += _encoder_kinds()
        lines += ["The adapters and point painting: options with no probe of their own.", ""]
        lines += [*groups[0][1], ""]
    elif stem in STACKED:
        for title, body in groups:
            lines += [f"**{title.split(' · ')[0]}**", "", *body, ""]
    else:
        lines += _tabbed(groups)
    from grasppanda.combine.reach import LEVEL2_NONE

    learns = any(_has_level2(slot, _page_options(stem, slot)) for slot in slots)
    lines += [
        '??? note "What the columns say, and who is left out"',
        "",
        "    - **Runs**: out of the methods the option is a swap under, those under which its "
        "probe trained on real frames without error. A method that ships the option is named "
        "*own* and not counted; a method still undecided is counted and named. Measured on "
        f"{chips.measured_on(1)}.",
    ]
    if learns:
        lines.append(
            f"    - **{chips.LEVEL2_HEADER}**: out of the methods under which the swap ran, "
            "those under which every comparable loss term ended within the method's tolerance "
            "of the component it ships, with nothing rising; a method left out is named with "
            f"why. Measured on {chips.measured_on(2)}."
        )
    for slot in slots:
        if slot in LEVEL2_NONE:
            why = LEVEL2_NONE[slot]
            lines.append(f"    - **{_plural(slot).capitalize()}** have no level 2: {why}.")
    lines += [
        "    - [How we measure](../evidence/index.md) defines every word of these columns.",
        "",
    ]
    if long_tails:
        lines += ["    Who is left out, and why:", "", *[f"    - {t}" for t in long_tails], ""]
    if "backbone" not in slots and stem not in SOURCES_BY_HAND:
        lines += _origins(unlabelled)
    if "loss" in slots:
        lines += _loss_terms()
    if "backbone" in slots:
        lines += _sources(stem)
    if stem in ENCODER_PAGES:
        lines += _released(stem)
    links = [f'<a href="../../explorer/#grid={slot}">{_plural(slot)}</a>' for slot in slots]
    if len(slots) == 1:
        links = [f'<a href="../../explorer/#grid={slots[0]}">Compare under every method</a>']
    else:
        links[0] = "Compare under every method: " + links[0]
    lines += [" · ".join(links), "", OPTIONS_END]
    return lines


def reach_block(stem: str) -> list[str]:
    """Where the options of :data:`REACH_LINES` stand: a chip and its tail, from
    :mod:`grasppanda.combine.reach` (the same words as the explorer and compose)."""
    chips = _chips()
    groups: dict[tuple[str, str, str], list[str]] = {}
    for slot, name in REACH_LINES.get(stem, ()):
        found = chips.reach(slot, name)
        groups.setdefault((found.fraction, found.tone, found.tail), []).append(f"`{name}`")
    lines = [REACH_BEGIN, ""]
    for (fraction, tone, tail), names in groups.items():
        chip = chips.chip(fraction, tone, tail)
        lines += [f"{chip} {', '.join(names)} · {_plain(tail)}.", ""]
    return [*lines, REACH_END]


def head_block(stem: str) -> list[str]:
    """The chips under the title of a slot page: the methods its slots apply to, how a
    command names them (as code), and how many options the page lists: on the page of a
    slot every option of it (``docs_chips.option_count``), on a page of one family of
    scene encoders the options it lists."""
    chips = _chips()
    docs = _docs_tool()
    slots = SLOT_PAGE_SLOTS[stem]
    scopes = list(dict.fromkeys(chips.scope_chip(slot) for slot in slots))
    keys = [docs.CLI_KEYS.get(slot, slot) for slot in slots]
    # One chip per key: the row wraps on a phone, a single long chip would not.
    command = [f"--set {key}=" for key in keys]
    if stem == SLOT_PAGE[slots[0]]:
        count = sum(chips.option_count(slot) for slot in slots)
    else:
        counted = {slot: set(chips.options_of(slot)) for slot in slots}
        count = sum(
            sum(option.name in counted[slot] for option in _page_options(stem, slot))
            for slot in slots
        )
    codes = [f'<code class="gp-chip">{html.escape(text)}</code>' for text in command]
    found = [*scopes, *codes, chips.chip(f"{count} options")]
    return [HEAD_BEGIN, f'<div class="gp-chips">{"".join(found)}</div>', HEAD_END]


def _example(stem: str, method: str, slot: str) -> option_lists.Option | None:
    """The option of page ``stem`` a "Use it" tab shows under ``method``: the first preferred
    one that ran there as a swap, else the first of the page that did."""
    from grasppanda.combine.presets import LOSS_SLOT

    records = option_lists.measured()
    chips = _chips()
    # The overview of the scene encoders lists only the adapters; its example is any encoder.
    pool = _docs_tool().slot_options(slot) if stem == "scene_encoders" else None
    listed = {option.name for option in pool or _page_options(stem, slot)}

    def name(option: option_lists.Option) -> str:
        return option.presets[0] if slot == LOSS_SLOT else option.name

    def ran(option: option_lists.Option) -> bool:
        if slot != LOSS_SLOT:
            return method in chips.reach(slot, option.name).reached
        record = records.get(option.cell)
        return record is not None and records.levels(record).states.get("runs") == YES

    candidates = [
        option
        for option in option_lists.options(method, slot)
        if option.cell and len(option.sets) == 1 and name(option) in listed
    ]
    prefer = EXAMPLES.get(stem, ())
    candidates.sort(key=lambda o: prefer.index(name(o)) if name(o) in prefer else len(prefer))
    return next((option for option in candidates if ran(option)), None)


def slot_block(stem: str) -> list[str]:
    """The "Use it" region of a slot page: a command under every method that has the slot
    (one linked content tab each), and the link to each method's list of the slot."""
    slot = SLOT_PAGE_SLOTS[stem][0]
    methods = _chips().slot_methods(slot)
    groups = []
    for method in methods:
        config = option_lists.methods()[method]
        body = ["```bash"]
        option = _example(stem, method, slot)
        if option is not None:
            request = " ".join(f"--set {item}" for item in option.sets)
            output = option_lists.output_name(method, option.sets)
            body += [f"grasppanda compose {config} \\", f"    {request} -o {output}"]
        body += [f"grasppanda compose {config} --list {slot}", "```"]
        groups.append((method, body))
    links = " · ".join(
        f"[{method}](../methods/{page_of(method)}.md#{anchor(slot)})" for method in methods
    )
    one = f"{methods[0]} is the one method with this slot. " if len(methods) == 1 else ""
    return [
        BEGIN,
        "",
        *_tabbed(groups),
        f"{one}The first command writes a checked config; the second lists every option of "
        "the slot under that method, with what was measured. Then `grasppanda probe <file>` "
        "tries the config on real frames and `grasppanda train <file>` trains it "
        f"([Swap a component](../get_started/swap.md)). Each method page lists the same: {links}.",
        "",
        END,
    ]


def with_region(text: str, begin: str, end: str, block: list[str], page: str) -> str:
    """A slot page with the region between ``begin`` and ``end`` replaced by ``block``."""
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"docs/slots/{page}: the marks {begin!r} … {end!r} are missing")
    return pattern.sub(lambda _: "\n".join(block), text, count=1)


def slot_page(stem: str) -> str:
    """A slot page with every one of its generated regions current."""
    text = (SLOT_PAGES / f"{stem}.md").read_text(encoding="utf-8")
    text = with_region(text, HEAD_BEGIN, HEAD_END, head_block(stem), f"{stem}.md")
    text = with_region(text, BEGIN, END, slot_block(stem), f"{stem}.md")
    if stem in REACH_LINES:
        text = with_region(text, REACH_BEGIN, REACH_END, reach_block(stem), f"{stem}.md")
    return with_region(text, OPTIONS_BEGIN, OPTIONS_END, options_block(stem), f"{stem}.md")


def expected() -> dict[Path, str]:
    records = option_lists.Measured.load(MEASURED)
    found = {
        PAGES / f"{page_of(method)}.md": page(method, records) for method in option_lists.methods()
    }
    found.update({INDEX: index_page(records), CHOOSER: chooser()})
    for stem in SLOT_PAGE_SLOTS:
        path = SLOT_PAGES / f"{stem}.md"
        if path.is_file():
            found[path] = slot_page(stem)
    return found


def stale() -> list[Path]:
    return [
        path
        for path, content in expected().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    args = parser.parse_args(argv)
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"stale: {path.relative_to(REPO)} (run python -m tools.make_method_pages)")
        return 1 if outdated else 0
    PAGES.mkdir(parents=True, exist_ok=True)
    for path, content in expected().items():
        changed = not path.is_file() or path.read_text(encoding="utf-8") != content
        path.write_text(content, encoding="utf-8")
        print(f"{'wrote' if changed else 'unchanged'} {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
