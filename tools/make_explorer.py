"""Generate the data of the combination explorer (``docs/explorer/``).

The explorer is a page of the documentation a visitor plays with: pick a method, replace
the option of a slot, and the page answers from the measured records. What it shows is in
two files written here:

* ``docs/explorer/explorer.json``: the methods, the slots, the options and every level-1
  record, with where each option works (:mod:`grasppanda.combine.reach`);
* ``docs/explorer/explorer_learns.json``: level 2, how each recorded swap trains beside the
  method's own component (``learns_like_reference.json``), and why a record has no level 2.

They are written from

* ``docs/advanced_guides/measured_combinations.json``: the verdict, the reasons and the
  numbers of every probed combination;
* ``docs/advanced_guides/learns_like_reference.json``: the level-2 outcome of a record;
* the presets (``grasppanda/presets/<slot>/``) and the recipe presets
  (``grasppanda/recipes/<slot>/``): the options of a slot, what each is and the YAML
  fragment ``grasppanda compose`` starts from;
* the catalogue (``grasppanda components --json``): source and extras of the component
  behind an option;
* ``tools/docs_chips.py``: the order, venue and input of the methods (from their
  metafiles), the title, page and scope of a slot, and the sentences of what was measured.

Nothing measured is typed in: names, counts, verdicts and fractions are those of the
records at the time the tool runs. The page logic (``docs/explorer/explorer_logic.js``)
only looks things up.

    python -m tools.make_explorer           # rewrite
    python -m tools.make_explorer --check   # exit 1 and name what is stale

``tests/unit/test_explorer.py`` runs the check, so stale data fails CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from grasppanda.cli.components import SLOTS as REGISTRY_SLOTS
from grasppanda.cli.components import collect
from grasppanda.combine import options as option_lists
from grasppanda.combine import reach as reaching
from grasppanda.combine.compose import component_slots, compose
from grasppanda.combine.grid import FAMILIES, METHODS, SHIPPED_EQUIVALENTS, cells
from grasppanda.combine.levels import LEVELS, NO, UNDECIDED, YES
from grasppanda.combine.presets import LOSS_SLOT, POST_SLOT, presets
from grasppanda.combine.probe import VERDICTS
from grasppanda.combine.recipes import RECIPE_SLOTS, recipes
from grasppanda.config import Config, resolve_config_path
from grasppanda.registry import DETECTORS

REPO = Path(__file__).resolve().parents[1]
MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
LEVEL2 = REPO / "docs" / "advanced_guides" / "learns_like_reference.json"
CAVEAT_SNIPPET = REPO / "docs" / "includes" / "caveat_probe.md"
OUT = REPO / "docs" / "explorer" / "explorer.json"
LEARNS_OUT = REPO / "docs" / "explorer" / "explorer_learns.json"

#: The slots of the training recipe: around the model, not a stage of it, and a slot of
#: every method. Slot → where it is written in a config.
RECIPE_LANES = {
    "optimizer": "optimizer",
    "schedule": "scheduler",
    "hooks": "hooks",
    "augment": "data.train.dataset.transforms",
    "recipe": "optimizer, scheduler, hooks, transforms",
}
assert set(RECIPE_LANES) == set(RECIPE_SLOTS), "a recipe slot without a lane"
#: The lanes of the pipeline in the order data flows; a slot that is not listed here (a
#: new directory of presets) is appended, so it shows up without anyone editing this file.
LANES = (
    "image_encoder",
    "backbone",
    "proposal",
    "view",
    "assigner",
    "grouper",
    "head",
    LOSS_SLOT,
    POST_SLOT,
    *RECIPE_LANES,
)
#: Slot → the registry that holds its components.
SLOT_REGISTRY = {
    "image_encoder": "IMAGE_ENCODERS",
    "backbone": "BACKBONES",
    "proposal": "PROPOSALS",
    "view": "VIEWS",
    "assigner": "ASSIGNERS",
    "grouper": "GROUPERS",
    "head": "HEADS",
    LOSS_SLOT: "LOSSES",
    POST_SLOT: "POSTPROCESSORS",
}
#: What a slot does, in one plain sentence (the page shows it under the slot's title).
TASKS = {
    "image_encoder": "Turns the RGB-D image into feature maps.",
    "backbone": "Turns the point cloud into features, on every point or on a subsample.",
    "proposal": "Picks the seed points where grasps are predicted.",
    "view": "Picks one approach direction per seed.",
    "assigner": "Training only: matches each seed with its grasp labels.",
    "grouper": "Gathers the points around each seed, along its approach, into one feature.",
    "head": "Prediction layers, training targets, loss and decoding, as one unit.",
    LOSS_SLOT: "The loss of one predicted quantity, inside a stage.",
    POST_SLOT: "Filters or reorders the predicted grasps at inference.",
    "optimizer": "The optimizer, and what it needs with it.",
    "schedule": "How the learning rate changes, and the unit it steps in.",
    "hooks": "Extras around the training loop. On the command line, join several with `+`.",
    "augment": "Random changes to the training data. On the command line, join several with `+`.",
    "recipe": "A whole training recipe under one name: optimizer, schedule, hooks, augmentation.",
}
#: The other encoder slot: a method without one of the two has the other.
COUNTERPART = {"backbone": "image_encoder", "image_encoder": "backbone"}
#: What a method does, in one line (the cards of the method gallery). Taken from the first
#: paragraph of ``configs/<method>/README.md``.
METHOD_LINES = {
    "GraspNet-baseline": "The benchmark's baseline: PointNet++ seeds, 300 views, cylinder crops.",
    "GSNet": "Learns where grasps are likely (graspness) and puts seeds only there.",
    "Scale-Balanced-Grasp": "GraspNet-baseline for small grasps: four crop radii, "
    "scale-balanced losses.",
    "EconomicGrasp": "Graspness seeds, one grasp label per seed, an interactive head.",
    "FineGrasp": "EconomicGrasp for delicate objects: per-object graspness, multi-radius crops.",
    "HGGD": "From an RGB-D image: a heat map of grasp centres, then 6-DoF grasps.",
}
#: The four levels in a few words (the page links to the definitions).
LEVEL_WORDS = {
    "composes": "config accepted",
    "runs": "trains on real frames without error",
    "learns": "its losses fall",
    "predicts": "it ranks grasps",
}
#: Keys of a preset node that name released weights to load.
WEIGHT_KEYS = ("pretrained", "weights", "checkpoint")
#: One letter per state of a level in a cell (grasppanda.combine.levels): a record holds
#: its four levels as a word of four letters, ``YYCC``.
STATE_CODES = {
    "yes": "Y",
    "no": "N",
    "not checked": "C",
    "unstable, not repeated": "R",
    "needs consent": "P",
    "not a combination": "X",
    "not probed": "-",
    "not reached by the method as shipped": "S",
}
#: Level-2 outcomes, best first (the order of :mod:`grasppanda.combine.reach`, which reads
#: ``fails`` as "did not train"): the page shows the worst of several.
LEVEL2_OUTCOMES = tuple(reversed(reaching.LEVEL2_ORDER))
#: Why a level-1 record has no level-2 outcome on the page. The first is the method as
#: shipped; :func:`learns` appends why a swap that did not run at level 1 is left out. The
#: rules are those of :func:`grasppanda.combine.reach.level2`: a slot without level 2, a
#: swap that did not run at level 1, a run on a schedule other than the reference's.
LEVEL2_NONE = (
    "the method as shipped: the reference the swaps are compared with",
    "compose refused it: nothing was trained",
    "it failed the short probe, so it was not trained longer",
    reaching.LEVEL2_NONE["post"],
    "several slots replaced at once: no single component of the method to compare with",
    "not measured at level 2",
    reaching.LEVEL2_NONE["schedule"],
    reaching.LEVEL2_EXCLUDED["schedule"],
)


def _tool(name: str) -> Any:
    """A module of ``tools/``, loaded by path (the directory is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _docs_tool() -> Any:
    """``tools/make_combination_docs.py``: the order of the verdicts is defined there, once."""
    return _tool("make_combination_docs")


def _chips() -> Any:
    """``tools/docs_chips.py``: the shared vocabulary of the pages."""
    return _tool("docs_chips")


def no_accuracy() -> str:
    """The caveat the page carries: the snippet every page of the documentation includes."""
    return " ".join(re.sub(r"<[^>]+>", " ", CAVEAT_SNIPPET.read_text(encoding="utf-8")).split())


# ----------------------------------------------------------------------------- options


def _weights(node: Any) -> str:
    """What a preset node says about released weights, as it is written there."""
    if isinstance(node, Mapping):
        for key in WEIGHT_KEYS:
            value = node.get(key)
            if value not in (None, False, "", "none"):
                return f"{key}: {str(value).lower() if isinstance(value, bool) else value}"
        for value in node.values():
            found = _weights(value)
            if found:
                return found
    return ""


def _body(path: Path) -> str:
    """A preset file without its leading comment."""
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip() + "\n"


def options(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Slot → its options, sorted by name (a base before its variants, as on the slot
    pages): every preset, and every component the records name without one. The source of
    a preset is the one its header declares (``declared_source``, what the slot pages
    print), else that of its component."""
    catalogue = {(entry.registry, entry.name): entry for entry in collect()}
    declared = _docs_tool().declared_source
    found: dict[str, list[dict[str, Any]]] = {}

    def add(slot: str, name: str, component: str, preset: Any) -> None:
        entry = catalogue.get((SLOT_REGISTRY.get(slot, ""), component))
        found.setdefault(slot, []).append(
            {
                "name": name,
                "component": component,
                "preset": preset is not None,
                "summary": preset.summary if preset else (entry.summary if entry else ""),
                "source": (declared(preset) if preset else "") or (entry.source if entry else ""),
                "extras": list(entry.extras) if entry else [],
                "weights": _weights(preset.node) if preset else "",
                "yaml": _body(preset.path)
                if preset
                else f"model:\n  {slot}:\n    type: {component}   # constructor defaults\n",
            }
        )

    for slot, members in presets().items():
        for name, preset in members.items():
            add(slot, name, preset.type, preset)
    # A recipe preset names one or several components (an optimizer with its hook, a whole
    # recipe): the source is the one it declares, else that of the first component; the
    # extras are those of all of them.
    by_name = {entry.name: entry for entry in catalogue.values()}
    for slot, recipe_members in recipes().items():
        for name, recipe in recipe_members.items():
            entries = [by_name[part] for part in recipe.type.split(", ") if part in by_name]
            found.setdefault(slot, []).append(
                {
                    "name": name,
                    "component": recipe.type,
                    "preset": True,
                    "summary": recipe.summary,
                    "source": declared(recipe) or (entries[0].source if entries else ""),
                    "extras": sorted({extra for entry in entries for extra in entry.extras}),
                    "weights": "",
                    "yaml": _body(recipe.path),
                }
            )
    for record in records:
        slot, variant = record.get("slot"), record.get("variant")
        if not slot or slot == LOSS_SLOT or len(record.get("sets", [])) != 1:
            continue
        if all(option["name"] != variant for option in found.get(slot, [])):
            add(slot, variant, record["component"], None)
    # Every option a method lists (grasppanda.combine.options), with a record or not.
    for method in option_lists.methods():
        for option in option_lists.options(method):
            if option.slot == LOSS_SLOT:
                continue
            if all(entry["name"] != option.name for entry in found.get(option.slot, [])):
                add(option.slot, option.name, option.component, None)
    for members in found.values():
        members.sort(key=lambda entry: entry["name"].lower())
    return found


# ----------------------------------------------------------------------------- methods


def methods(
    by_id: Mapping[str, dict[str, Any]], measured: option_lists.Measured | None = None
) -> list[dict[str, Any]]:
    """Every method of the grid, in page order: what the gallery says of it (venue and
    input from its metafile), its config, its slots and the option each slot ships with."""
    measured = measured or option_lists.measured()
    about = {info.name: info for info in _chips().methods()}
    found = []
    for name, config in {**METHODS, **FAMILIES}.items():
        group = "reference" if name in METHODS else "family"
        reference = by_id.get(f"{group}/{name}")
        resolved = Config.load(resolve_config_path(config)).resolved()
        model = resolved["model"]
        detector = DETECTORS.get(model["type"])
        slots = list(component_slots(detector))
        stages = compose(config).stages if name in METHODS else {}
        ships: dict[str, dict[str, str]] = {}
        for slot in slots:
            node = model.get(slot)
            component = stages.get(slot) or (node.get("type") if isinstance(node, Mapping) else "")
            ships[slot] = {"component": component or "", "option": ""}
            # A slot the shipped config leaves out is filled by the detector itself ("built
            # in", as the method pages and `compose --list` say); its sizes for the tooltip.
            built_in = model.get("encoder") if slot == "image_encoder" and not component else None
            if isinstance(built_in, Mapping) and built_in.get("blocks"):
                ships[slot]["label"] = "built in"
                ships[slot]["about"] = (
                    "the reference encoder of the detector, blocks "
                    + "-".join(str(count) for count in built_in["blocks"])
                    + f", width {built_in.get('base_width')}"
                )
                ships[slot]["plain"] = True
        # The slots and their options are those of grasppanda.combine.options, the lists
        # of `grasppanda compose <config> --list` (a test holds the two together):
        # post-processing and the training recipe for every method, the losses of the
        # stages for the seed-based ones.
        slots = option_lists.method_slots(name)
        listed: dict[str, list[Any]] = {}
        not_probed: dict[str, dict[str, str]] = {}
        # Sorted by name, a base before its variants, as on the slot pages.
        by_name = sorted(option_lists.options(name), key=lambda option: option.name.lower())
        for option in by_name:
            plain = option.sets == (f"{option.slot}={option.name}",)
            listed.setdefault(option.slot, []).append(
                option.name if plain else {"name": option.name, "sets": list(option.sets)}
            )
            if option.reason:
                not_probed.setdefault(option.slot, {})[option.name] = option.reason
        for slot, key in (("optimizer", "optimizer"), ("schedule", "scheduler")):
            ships[slot] = {
                "component": str((resolved.get(key) or {}).get("type") or ""),
                "option": "",
            }
        ships["hooks"] = {
            "component": ", ".join(str(node.get("type")) for node in resolved.get("hooks") or []),
            "option": "",
        }
        # The option of a slot that composes to the method as shipped (its reference's
        # config hash), as `grasppanda compose <config> --list` marks it: HGGD's
        # `accumulate-2` is what it ships, not a change.
        for slot in ships:
            if slot in slots:
                found_option = option_lists.slot_listing(name, slot, measured).shipped_option
                ships[slot]["option"] = found_option
        if POST_SLOT in slots:
            ships[POST_SLOT] = shipped_post(resolved)
        slots.sort(key=lambda slot: LANES.index(slot) if slot in LANES else len(LANES))
        info = about[name]
        found.append(
            {
                "name": name,
                "config": config,
                "folder": info.folder,
                "page": info.page.removesuffix(".md"),
                "family": info.family,
                "venue": info.venue,
                "year": info.year,
                "title": info.title,
                "inputs": info.inputs,
                "line": METHOD_LINES.get(name, info.title),
                "detector": model["type"],
                "seed_based": name in METHODS,
                "slots": slots,
                "ships": ships,
                "options": listed,
                "not_probed": not_probed,
                "reference": reference["id"] if reference else "",
            }
        )
    order = [info.name for info in about.values()]
    return sorted(found, key=lambda method: order.index(method["name"]))


def shipped_post(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """What a method ships in the post-processing slot, from its config: the variants it is
    evaluated with (``evaluation.variants``, in the order of the config). ``--set post=``
    adds a variant ``post`` beside them; nothing is removed."""
    variants = (resolved.get("evaluation") or {}).get("variants") or {}
    chains = [
        f"{name}: " + (" → ".join(str(step["type"]) for step in steps) or "as the model emits them")
        for name, steps in variants.items()
    ]
    return {
        "component": "",
        "option": "",
        "label": f"as shipped (variants {', '.join(variants)})" if variants else "as shipped",
        "about": "; ".join(chains),
        "plain": True,
    }


# ------------------------------------------------------------------------------- cells


def key_of(method: Mapping[str, Any], sets: list[str]) -> str:
    """What identifies a combination: the method and the sets that change it.

    A ``slot=option`` that names what the method ships with composes to the shipped
    config (the same hash, see :func:`methods`), so it is left out; the rest is sorted.
    ``explorer_logic.js`` computes the same key for a selection.
    """
    own = {f"{slot}={ships['option']}" for slot, ships in method["ships"].items()}
    return method["name"] + "|" + "&".join(sorted(item for item in sets if item not in own))


def index_of(found: list[Mapping[str, Any]], keys: list[str]) -> dict[str, int]:
    """Key → the position of the cell that answers it.

    Two records with one key are one experiment (a test asserts it); the first wins,
    unless it shares the probe of another record and a later one with the same verdict
    has a probe of its own. ``prepare`` in explorer_logic.js builds the same index at load,
    so the file does not carry it.
    """
    index: dict[str, int] = {}
    for position, (item, key) in enumerate(zip(found, keys)):
        if key not in index:
            index[key] = position
            continue
        kept = found[index[key]]
        if "shares" in kept and "shares" not in item and kept["verdict"] == item["verdict"]:
            index[key] = position
    return index


def shipped(record: Mapping[str, Any], reference: Mapping[str, Any] | None) -> bool:
    """Whether ``record`` is the method as shipped under another name: the config hash of
    the method's reference record, or a preset that writes out what the method builds in
    (``grid.SHIPPED_EQUIVALENTS``): the rule of :mod:`grasppanda.combine.reach`."""
    if record.get("id") in SHIPPED_EQUIVALENTS:
        return True
    if not reference or reference["id"] == record["id"]:
        return False
    found = record.get("config_hash")
    return bool(found) and found == reference.get("config_hash")


def _probe_opts(record: Mapping[str, Any]) -> str:
    probe = record.get("probe", "")
    return probe.split(" --opts ", 1)[1] if " --opts " in probe else ""


def cell(
    record: Mapping[str, Any],
    by_id: Mapping[str, dict[str, Any]],
    method: Mapping[str, Any],
    measured: option_lists.Measured,
) -> dict[str, Any]:
    """One record as the page needs it: verdict, reasons, the levels it reached
    (grasppanda.combine.levels) and the numbers honest to show. Its key (:func:`key_of`)
    is not written: the page computes it (``prepare`` in explorer_logic.js)."""
    numbers = by_id.get(record.get("probed_as", ""), record)
    out: dict[str, Any] = {
        "id": record["id"],
        "group": record["group"],
        "method": record["method"],
        "sets": list(record.get("sets", [])),
        "verdict": record["verdict"],
    }
    if record["id"] == method.get("reference") or shipped(record, by_id.get(method["reference"])):
        # The method as shipped: its reference, or a set that composes to the same config
        # (an option the method ships). No swap: reach counts it "own".
        out["own"] = True
    # The variant of a single swap is the option after `slot=`: the page reads it there.
    for name in ("slot", "component", "adapter", "fitted", "refusal", "unstable"):
        if record.get(name):
            out[name] = record[name]
    if record.get("failure"):
        out["failure"] = {
            name: record["failure"][name]
            for name in ("phase", "when", "component", "error", "message")
            if record["failure"].get(name)
        }
    if record.get("probed_as"):
        out["shares"] = record["probed_as"]
    if numbers.get("parameters") is not None:
        out["parameters"] = numbers["parameters"]
        out["grasps"] = numbers.get("grasps_per_frame", [])
        total = (numbers.get("losses") or {}).get("loss")
        if total:
            out["loss"] = total
    if numbers.get("judge", 1) >= 2 and numbers.get("trend"):
        # What judge 2 adds, as the page needs it: the fall of every term and what was
        # flagged. A record of the first judge has none of it (and no `judge`).
        out["judge"] = numbers["judge"]
        # Per loss term (without `loss/`; the total is not shown), to the percent shown.
        out["fall"] = {
            term[len("loss/") :]: None if values[2] is None else round(values[2], 2)
            for term, values in numbers["trend"].items()
            if term.startswith("loss/")
        }
        for name in ("flags", "no_gradient", "unchanged"):
            if numbers.get(name) not in (None, [], {}):
                out[name] = numbers[name]
    levels = measured.levels(record)
    out["levels"] = "".join(STATE_CODES[levels.states[level]] for level in LEVELS)
    why = [f"{level}: {'; '.join(items)}" for level, items in levels.why.items()]
    if why or levels.notes:
        out["why"] = why + levels.notes
    kind = option_lists.refusal_kind(record) if record["verdict"] == "refused" else ""
    if kind:
        out["kind"] = kind
    lifts = [part["lift"] for part in option_lists.refusal_kinds(record) if part.get("lift")]
    if lifts:
        out["lifts"] = lifts
    if numbers.get("postprocessing"):
        out["post"] = numbers["postprocessing"]
    if _probe_opts(numbers):
        out["probe_opts"] = _probe_opts(numbers)  # replaced by its index in `probe_opts`
    out["run"] = record.get("run", "")  # replaced by its index in `runs` (data())
    return out


# ------------------------------------------------------------------------------ reach


class _Rows:
    """Distinct ``[fraction, tone, tail]`` rows, each stored once (the file refers to a row
    by its position): most options share the same few rows."""

    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self._at: dict[tuple[str, str, str], int] = {}

    def add(self, found: Any) -> int:
        row = (found.fraction, found.tone, found.tail)
        if row not in self._at:
            self._at[row] = len(self.rows)
            self.rows.append(list(row))
        return self._at[row]


def reach_rows(
    slot_options: Mapping[str, list[dict[str, Any]]],
    found: list[dict[str, Any]],
    measured: option_lists.Measured,
) -> list[list[str]]:
    """Add to every option where it works (:mod:`grasppanda.combine.reach`): ``reach``, its
    row at each of the four levels, and ``learns``, its row at level 2; for the losses, also
    the cells it sits in. Returns the rows."""
    rows = _Rows()
    for slot, members in slot_options.items():
        for option in members:
            if slot == LOSS_SLOT:
                option["cells"] = [
                    index
                    for index, item in enumerate(found)
                    if item["group"] == LOSS_SLOT
                    and any(pair.endswith(f"={option['name']}") for pair in item["sets"])
                ]
            option["reach"] = [
                rows.add(reaching.reach(slot, option["name"], level, measured)) for level in LEVELS
            ]
            option["learns"] = rows.add(
                reaching.level2(slot, option["name"], measured, path=LEVEL2)
            )
    return rows.rows


# -------------------------------------------------------------------------------- data


def lanes(slot_options: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Every slot as the page draws it: its title, where it is written, what it does, its
    page of the documentation and which methods have it (``docs_chips``)."""
    chips = _chips()
    order = [slot for slot in LANES if slot in slot_options]
    order += sorted(set(slot_options) - set(order))
    found = []
    for slot in order:
        where, task = REGISTRY_SLOTS.get(SLOT_REGISTRY.get(slot, ""), (slot, ""))
        if slot in RECIPE_LANES:
            where = RECIPE_LANES[slot]
        page = chips.SLOT_PAGES.get(slot, "")
        entry = {
            "key": slot,
            "title": chips.SLOT_TITLES.get(slot, slot.replace("_", " ").capitalize()),
            "recipe": slot in RECIPE_LANES,
            "config": where,
            "task": TASKS.get(slot, task),
            "page": page.removesuffix(".md"),
            "scope": chips.scope(slot),
            # How many options a user can `--set` (docs_chips.option_count, what every page
            # counts): adapters and point-painting wrappers are listed, not counted.
            "count": chips.option_count(slot),
        }
        if slot in COUNTERPART:
            entry["counterpart"] = COUNTERPART[slot]
        found.append(entry)
    return found


def data() -> dict[str, Any]:
    """Everything the explorer shows but level 2 (:func:`learns`)."""
    chips = _chips()
    measured = json.loads(MEASURED.read_text(encoding="utf-8"))
    records = measured["cells"]
    by_id = {record["id"]: record for record in records}
    found_measured = option_lists.Measured(measured)
    method_list = methods(by_id, found_measured)
    by_name = {method["name"]: method for method in method_list}
    found = [cell(record, by_id, by_name[record["method"]], found_measured) for record in records]
    keys = [key_of(by_name[record["method"]], list(record.get("sets", []))) for record in records]
    # The run of a cell, by its index in `runs`: a dozen runs measured more than a thousand
    # cells (explorer_logic.prepare puts the name back); the options of a probe likewise.
    runs = sorted({item["run"] for item in found})
    probe_opts = sorted({item["probe_opts"] for item in found if "probe_opts" in item})
    for item in found:
        item["run"] = runs.index(item["run"])
        if "probe_opts" in item:
            item["probe_opts"] = probe_opts.index(item["probe_opts"])
    slot_options = options(records)
    # What is listed but is no option a user can `--set` (an adapter compose inserts, a
    # point-painting wrapper): not counted, as on every page (docs_chips.options_of).
    for slot, members in slot_options.items():
        counted = set(chips.options_of(slot))
        for option in members:
            if option["name"] not in counted:
                option["not_option"] = True
    rows = reach_rows(slot_options, found, found_measured)
    # Two records with one key must be one experiment (a test asserts the count is zero).
    index = index_of(found, keys)
    conflicts = 0
    for item, key in zip(found, keys):
        kept = found[index[key]]
        if kept is item:
            continue
        hashes = {by_id[kept["id"]]["config_hash"], by_id[item["id"]]["config_hash"]}
        conflicts += len(hashes) > 1 or kept["verdict"] != item["verdict"]
    grid = list(cells())
    grid_ids = {entry.id for entry in grid}
    listed_sets = {entry.id for entry in grid if entry.group == "stage-set"}
    about = measured["about"]
    return {
        "generated_by": "tools/make_explorer.py; do not edit by hand",
        "runs": runs,
        "probe_opts": probe_opts,
        "no_accuracy": no_accuracy(),
        "measured_on": chips.measured_on(1),
        "levels": [
            {"key": level, "short": LEVEL_WORDS.get(level, level), "meaning": meaning}
            for level, meaning in LEVELS.items()
        ],
        "level_states": {code: state for state, code in STATE_CODES.items()},
        "undecided": UNDECIDED,
        "undecided_words": dict(reaching.UNDECIDED_WORDS),
        "refusal_words": dict(reaching.REFUSAL_WORDS),
        "probe": about["probe"],
        "verdict_order": docs_verdicts(found),
        # The meaning the file states for a verdict, else today's (a file of the first
        # judge does not define `runs-empty`).
        "verdicts": {
            name: about["verdicts"].get(name) or VERDICTS[name] for name in docs_verdicts(found)
        },
        "counts": {
            "records": len(found),
            "by_verdict": dict(Counter(item["verdict"] for item in found)),
            "grid": len(grid_ids),
            "grid_without_record": len(grid_ids - set(by_id)),
            "key_conflicts": conflicts,
            # What stands behind a record: a probe of its own on real frames, the probe of
            # another record that composes to the same config, or a refusal by compose
            # (nothing was built, nothing trained).
            "probed": sum("probe" in record for record in records),
            "shared_probe": sum("probed_as" in record for record in records),
            "stage_sets": {
                "listed": len(listed_sets),
                "recorded": len(listed_sets & set(by_id)),
            },
            "runs": sorted({run["date"] for run in about.get("runs", {}).values()}),
        },
        "scale_statistic": about.get("scale_statistic", ""),
        "families": dict(chips.FAMILY_NAMES),
        "scopes": dict(chips.SCOPE_TEXT),
        "reach": rows,
        "lanes": lanes(slot_options),
        "methods": method_list,
        "options": slot_options,
        "cells": found,
    }


def docs_verdicts(found: list[dict[str, Any]]) -> list[str]:
    order = _docs_tool().VERDICT_ORDER
    used = {item["verdict"] for item in found}
    return [name for name in order if name in used] + sorted(used - set(order))


def level2_none(
    record: Mapping[str, Any],
    entry: Mapping[str, Any] | None,
    records: option_lists.Measured,
    reference: Mapping[str, Any] | None,
    schedule: str,
    excluded: Mapping[str, str] | None = None,
) -> str:
    """Why ``record`` shows no level-2 outcome (its words), or ``""`` when it shows the
    outcome of ``entry`` (its cell of ``learns_like_reference.json``). The rules of
    :func:`grasppanda.combine.reach.level2`, for one record: the method as shipped is the
    reference; a slot without level 2, a swap that did not run at level 1, and a run on a
    schedule other than the one the method's reference ran on are not compared.
    ``excluded`` is :attr:`~grasppanda.combine.reach.Level2.excluded` of the record's
    option: a method it names is not compared, as on every page (an option that writes a
    scheduler of its own, a swap that did not run there)."""
    if record["group"] in ("reference", "family") or shipped(record, reference):
        return LEVEL2_NONE[0]
    if record["verdict"] == "refused":
        return LEVEL2_NONE[1]
    if record["verdict"] == "fails":
        return LEVEL2_NONE[2]
    if record.get("slot") in reaching.LEVEL2_NONE:
        return reaching.LEVEL2_NONE[str(record["slot"])]
    state = records.levels(record).states.get("runs")
    if state != YES:
        detail = (
            reaching.VERDICT_WORDS.get(str(record["verdict"]))
            if state == NO
            else reaching.UNDECIDED_WORDS.get(str(state))
        ) or str(state)
        return f"{reaching.LEVEL2_EXCLUDED['runs']} ({detail})"
    if entry is None:
        multiple = record["group"] in ("stage-set", "cross") or len(record.get("sets") or []) > 1
        return LEVEL2_NONE[4] if multiple else LEVEL2_NONE[5]
    if str(entry.get("schedule", "constant")) != schedule:
        return reaching.LEVEL2_EXCLUDED["schedule"]
    return (excluded or {}).get(str(record["method"]), "")


def learns() -> dict[str, Any]:
    """Level 2 as the page needs it: the outcome of every record compared beside its
    method's own component, with the first reasons of a ``worse`` or ``diverging``, and why
    every other record has none (:func:`level2_none`)."""
    chips = _chips()
    measured = json.loads(MEASURED.read_text(encoding="utf-8"))
    records = option_lists.Measured(measured)
    payload = json.loads(LEVEL2.read_text(encoding="utf-8")) if LEVEL2.is_file() else {}
    references = {
        record["method"]: record
        for record in measured["cells"]
        if record["group"] in ("reference", "family")
    }
    schedules = {
        str(entry["method"]): str(entry.get("schedule", "constant"))
        for entry in payload.get("references") or []
    }
    entries = {
        entry["id"]: entry
        for entry in payload.get("cells") or []
        if str(entry["outcome"]) in LEVEL2_OUTCOMES
    }
    words = list(LEVEL2_NONE)
    found: dict[str, list[Any]] = {}
    none: dict[str, int] = {}
    # A cell → its option (slot, name), as reach.level2 counts it: a loss cell by its one
    # preset; a cell that replaces several presets at once is no single option.
    option_of: dict[str, tuple[str, str]] = {}
    for name in option_lists.methods():
        for option in option_lists.options(name):
            if option.cell and len(option.presets) <= 1:
                option_of[option.cell] = (option.slot, (*option.presets, option.name)[0])
    # (slot, option) → the methods reach.level2 leaves out of its comparison, and why.
    excluded: dict[tuple[str, str], dict[str, str]] = {}
    for record in measured["cells"]:
        method = record["method"]
        entry = entries.get(record["id"])
        found_option = option_of.get(record["id"])
        if found_option is not None and found_option not in excluded:
            level2 = reaching.level2(*found_option, records, path=LEVEL2)
            excluded[found_option] = dict(level2.excluded)
        why = level2_none(
            record,
            entry,
            records,
            references.get(method),
            schedules.get(method, "constant"),
            excluded.get(found_option) if found_option else None,
        )
        if why:
            if why not in words:
                words.append(why)
            none[record["id"]] = words.index(why)
        elif entry is not None:
            outcome = LEVEL2_OUTCOMES.index(str(entry["outcome"]))
            found[record["id"]] = [outcome, *list(entry.get("reasons") or [])[:2]]
    return {
        "generated_by": "tools/make_explorer.py; do not edit by hand",
        "name": chips.LEVEL2_NAME,
        "header": chips.LEVEL2_HEADER,
        "meaning": chips.LEVEL2_MEANING,
        "measured_on": chips.measured_on(2),
        "outcomes": [reaching.LEVEL2_WORDS.get(name, name) for name in LEVEL2_OUTCOMES],
        "tones": [reaching.LEVEL2_TONE[name] for name in LEVEL2_OUTCOMES],
        "none_words": words,
        "cells": found,
        "none": none,
    }


def _encode(payload: dict[str, Any]) -> str:
    """Compact JSON on one line, safe inside a ``<script>`` element."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    encoded = re.sub(r"</", r"<\\/", encoded).replace("<!--", "<\\u0021--")
    return encoded + "\n"


def text() -> str:
    """The file ``explorer.json``."""
    return _encode(data())


def learns_text() -> str:
    """The file ``explorer_learns.json``."""
    return _encode(learns())


def _files() -> dict[Path, str]:
    return {OUT: text(), LEARNS_OUT: learns_text()}


def stale() -> list[Path]:
    return [
        path
        for path, content in _files().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    args = parser.parse_args(argv)
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"stale: {path.relative_to(REPO)} (run python -m tools.make_explorer)")
        return 1 if outdated else 0
    for path, content in _files().items():
        changed = not path.is_file() or path.read_text(encoding="utf-8") != content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        state = "wrote" if changed else "unchanged"
        print(f"{state} {path.relative_to(REPO)} ({len(content.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
