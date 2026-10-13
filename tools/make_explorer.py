"""Generate the data of the combination explorer (``docs/explorer/``).

The explorer is a page of the documentation a visitor plays with: pick a method, replace
the option of a slot, and the page answers from the measured records. Everything it shows
is in one file, ``docs/explorer/explorer.json``, written here from

* ``docs/advanced_guides/measured_combinations.json``: the verdict, the reasons and the
  numbers of every probed combination;
* the presets (``grasppanda/presets/<slot>/``): the options of a slot, what each is and
  the YAML fragment ``grasppanda compose`` starts from;
* the recipe presets (``grasppanda/recipes/<slot>/``): the options of the training recipe
  (optimizer, schedule, hooks, augmentation), which every method has;
* the catalogue (``grasppanda components --json``): source, status and extras of the
  component behind an option;
* the slot pages (``docs/slots/``): where an option is explained.

Nothing is typed in: names, counts and verdicts are those of the records at the time the
tool runs. The page logic (``docs/explorer/explorer_logic.js``) only looks things up.

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
from grasppanda.combine.compose import component_slots, compose
from grasppanda.combine.grid import FAMILIES, METHODS, cells
from grasppanda.combine.presets import LOSS_SLOT, POST_SLOT, presets
from grasppanda.combine.recipes import RECIPE_SLOTS, recipes
from grasppanda.config import Config, resolve_config_path
from grasppanda.registry import DETECTORS

REPO = Path(__file__).resolve().parents[1]
MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
SLOT_PAGES = REPO / "docs" / "slots"
OUT = REPO / "docs" / "explorer" / "explorer.json"

#: The slots of the training recipe: around the model, not a stage of it, and a slot of
#: every method. Slot → (title, where it is written in a config, what it is).
RECIPE_LANES = {
    "optimizer": ("Optimizer", "optimizer", "The optimizer, and what it needs with it."),
    "schedule": ("Schedule", "scheduler", "The learning-rate schedule and the unit it steps in."),
    "hooks": (
        "Hooks",
        "hooks",
        "Hooks around the training loop, added to those of the method. On the command line "
        "several names are joined with `+`.",
    ),
    "augment": (
        "Augmentation",
        "data.train.dataset.transforms",
        "Point transforms put before or after those of the training dataset. On the command "
        "line several names are joined with `+`.",
    ),
    "recipe": (
        "Whole recipe",
        "optimizer, scheduler, hooks, transforms",
        "A whole training recipe under one name: optimizer, schedule, hooks, augmentation.",
    ),
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
#: Short lane titles; the sentence under each title is the catalogue's own.
LANE_TITLES = {
    "image_encoder": "Image encoder",
    "backbone": "Scene encoder",
    "proposal": "Seed proposal",
    "view": "View",
    "assigner": "Assigner",
    "grouper": "Grouper",
    "head": "Head bundle",
    LOSS_SLOT: "Losses",
    POST_SLOT: "Post-processing",
}
#: Keys of a preset node that name released weights to load.
WEIGHT_KEYS = ("pretrained", "weights", "checkpoint")
#: A sentence the page must carry; the test looks for it.
NO_ACCURACY = (
    "No accuracy appears here, and none follows from a verdict: a probe trains 20 "
    "iterations on 8 frames and shows whether a combination works, not how good it is."
)


def _docs_tool() -> Any:
    """``tools/make_combination_docs.py``: the measured classes are defined there, once."""
    name = "make_combination_docs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _slot_pages() -> dict[str, str]:
    # The index and the overview of the scene encoders name every option once; an option
    # is explained on the page of its slot.
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(SLOT_PAGES.glob("*.md"))
        if path.stem not in ("index", "scene_encoders")
    }


def _page_of(names: tuple[str, ...], pages: Mapping[str, str]) -> str:
    """The slot page that mentions an option most often (by any of ``names``), or ''."""
    best, count = "", 0
    for stem, text in pages.items():
        found = sum(text.count(f"`{name}`") for name in names)
        if found > count:
            best, count = stem, found
    return best


def options(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Slot → its options: every preset, and every component the records name without one."""
    catalogue = {(entry.registry, entry.name): entry for entry in collect()}
    pages = _slot_pages()
    found: dict[str, list[dict[str, Any]]] = {}

    def add(slot: str, name: str, component: str, preset: Any) -> None:
        entry = catalogue.get((SLOT_REGISTRY.get(slot, ""), component))
        found.setdefault(slot, []).append(
            {
                "name": name,
                "component": component,
                "preset": preset is not None,
                "summary": preset.summary if preset else (entry.summary if entry else ""),
                "source": entry.source if entry else "",
                "status": entry.status if entry else "",
                "extras": list(entry.extras) if entry else [],
                "weights": _weights(preset.node) if preset else "",
                "page": _page_of((name, component) if preset else (name,), pages),
                "yaml": _body(preset.path)
                if preset
                else f"model:\n  {slot}:\n    type: {component}   # constructor defaults\n",
            }
        )

    for slot, members in presets().items():
        for name, preset in members.items():
            add(slot, name, preset.type, preset)
    # A recipe preset names one or several components (an optimizer with its hook, a whole
    # recipe): the source is that of the first, the extras are those of all of them.
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
                    "source": entries[0].source if entries else "",
                    "status": entries[0].status if entries else "",
                    "extras": sorted({extra for entry in entries for extra in entry.extras}),
                    "weights": "",
                    # The recipe page writes a preset as `slot=name`.
                    "page": _page_of((name, f"{slot}={name}"), pages),
                    "yaml": _body(recipe.path),
                }
            )
    for record in records:
        slot, variant = record.get("slot"), record.get("variant")
        if not slot or slot == LOSS_SLOT or len(record.get("sets", [])) != 1:
            continue
        if all(option["name"] != variant for option in found.get(slot, [])):
            add(slot, variant, record["component"], None)
    return found


# ----------------------------------------------------------------------------- methods


def methods(by_id: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every method of the grid: its config, its slots and the option each slot ships with."""
    catalogue = {entry.name: entry for entry in collect(["DETECTORS"])}
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
            # A slot the shipped config leaves out is filled by the detector itself; what
            # it puts there is described by the options of the config, and named from them.
            built_in = model.get("encoder") if slot == "image_encoder" and not component else None
            if isinstance(built_in, Mapping) and built_in.get("blocks"):
                ships[slot]["label"] = (
                    "reference encoder · blocks "
                    + "-".join(str(count) for count in built_in["blocks"])
                    + f" · width {built_in.get('base_width')}"
                )
        # The option of a slot that composes to the config as shipped: the same config hash.
        if reference and reference.get("config_hash"):
            for record in by_id.values():
                same = record["config_hash"] == reference["config_hash"]
                if record["method"] == name and same and len(record.get("sets", [])) == 1:
                    slot = record.get("slot", "")
                    if slot in ships and not ships[slot]["option"]:
                        ships[slot]["option"] = record["variant"]
        # Post-processing is no part of the model: every detector has that slot. Losses
        # sit inside the stages; the lane is offered where a loss set was recorded.
        slots.append(POST_SLOT)
        if any(r["method"] == name and r["group"] == LOSS_SLOT for r in by_id.values()):
            slots.append(LOSS_SLOT)
        # The training recipe is around the model: every method has its slots. What a
        # method ships with there is what its config names.
        slots.extend(RECIPE_LANES)
        for slot, key in (("optimizer", "optimizer"), ("schedule", "scheduler")):
            ships[slot] = {
                "component": str((resolved.get(key) or {}).get("type") or ""),
                "option": "",
            }
        ships["hooks"] = {
            "component": ", ".join(str(node.get("type")) for node in resolved.get("hooks") or []),
            "option": "",
        }
        slots.sort(key=lambda slot: LANES.index(slot) if slot in LANES else len(LANES))
        entry = catalogue.get(model["type"])
        found.append(
            {
                "name": name,
                "config": config,
                "detector": model["type"],
                "seed_based": name in METHODS,
                "source": entry.source if entry else "",
                "slots": slots,
                "ships": ships,
                "reference": reference["id"] if reference else "",
            }
        )
    return found


# ------------------------------------------------------------------------------- cells


def key_of(method: Mapping[str, Any], sets: list[str]) -> str:
    """What identifies a combination: the method and the sets that change it.

    A ``slot=option`` that names what the method ships with composes to the shipped
    config (the same hash, see :func:`methods`), so it is left out; the rest is sorted.
    ``explorer_logic.js`` computes the same key for a selection.
    """
    own = {f"{slot}={ships['option']}" for slot, ships in method["ships"].items()}
    return method["name"] + "|" + "&".join(sorted(item for item in sets if item not in own))


def _probe_opts(record: Mapping[str, Any]) -> str:
    probe = record.get("probe", "")
    return probe.split(" --opts ", 1)[1] if " --opts " in probe else ""


def cell(
    record: Mapping[str, Any], by_id: Mapping[str, dict[str, Any]], method: Mapping[str, Any]
) -> dict[str, Any]:
    """One record as the page needs it: verdict, reasons, and the numbers honest to show."""
    numbers = by_id.get(record.get("probed_as", ""), record)
    out: dict[str, Any] = {
        "id": record["id"],
        "group": record["group"],
        "method": record["method"],
        "sets": list(record.get("sets", [])),
        "key": key_of(method, list(record.get("sets", []))),
        "verdict": record["verdict"],
    }
    for name in ("slot", "variant", "component", "adapter", "fitted", "refusal", "unstable"):
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
    if numbers.get("postprocessing"):
        out["post"] = numbers["postprocessing"]
    if _probe_opts(numbers):
        out["probe_opts"] = _probe_opts(numbers)
    out["run"] = record.get("run", "")
    return out


def grid_rows(
    slot_options: Mapping[str, list[dict[str, Any]]],
    found: list[dict[str, Any]],
    method_list: list[dict[str, Any]],
) -> None:
    """Add to every option the measured class and, for the losses, the cells it sits in."""
    docs = _docs_tool()
    seed_based = [method["name"] for method in method_list if method["seed_based"]]
    for slot, members in slot_options.items():
        for option in members:
            wanted = f"{slot}={option['name']}"
            if slot == LOSS_SLOT:
                hits = [
                    index
                    for index, item in enumerate(found)
                    if item["group"] == LOSS_SLOT
                    and any(pair.endswith(f"={option['name']}") for pair in item["sets"])
                ]
                option["cells"] = hits
                continue
            singles = {
                item["method"]: item
                for item in found
                if item["sets"] == [wanted] and item["method"] in seed_based
            }
            if len(singles) == len(seed_based):
                name, works, unstable = docs.classify(singles.values())
                option.update({"class": name, "works": works, "unstable": unstable})


# -------------------------------------------------------------------------------- data


def lanes(slot_options: Mapping[str, list[Any]], pages: Mapping[str, str]) -> list[dict[str, Any]]:
    order = [slot for slot in LANES if slot in slot_options]
    order += sorted(set(slot_options) - set(order))
    found = []
    for slot in order:
        where, task = REGISTRY_SLOTS.get(SLOT_REGISTRY.get(slot, ""), (slot, ""))
        if slot in RECIPE_LANES:
            _, where, task = RECIPE_LANES[slot]
        stems = sorted(
            {option["page"] for option in slot_options[slot] if option["page"]},
            key=lambda stem: (-sum(o["page"] == stem for o in slot_options[slot]), stem),
        )
        found.append(
            {
                "key": slot,
                "title": RECIPE_LANES[slot][0]
                if slot in RECIPE_LANES
                else LANE_TITLES.get(slot, slot.replace("_", " ").capitalize()),
                "recipe": slot in RECIPE_LANES,
                "config": where,
                "task": task,
                "pages": [
                    {"stem": stem, "title": pages[stem].splitlines()[0].lstrip("# ").strip()}
                    for stem in stems
                ],
            }
        )
    return found


def data() -> dict[str, Any]:
    """Everything the explorer shows."""
    measured = json.loads(MEASURED.read_text(encoding="utf-8"))
    records = measured["cells"]
    by_id = {record["id"]: record for record in records}
    method_list = methods(by_id)
    by_name = {method["name"]: method for method in method_list}
    found = [cell(record, by_id, by_name[record["method"]]) for record in records]
    slot_options = options(records)
    grid_rows(slot_options, found, method_list)
    # Two records with one key must be one experiment; where they are not, the single
    # swap wins and the count says so (a test asserts it is zero).
    index: dict[str, int] = {}
    conflicts = 0
    for position, item in enumerate(found):
        if item["key"] not in index:
            index[item["key"]] = position
            continue
        kept = found[index[item["key"]]]
        hashes = {by_id[kept["id"]]["config_hash"], by_id[item["id"]]["config_hash"]}
        conflicts += len(hashes) > 1 or kept["verdict"] != item["verdict"]
        if "shares" in kept and "shares" not in item and kept["verdict"] == item["verdict"]:
            index[item["key"]] = position
    grid = list(cells())
    grid_ids = {entry.id for entry in grid}
    listed_sets = {entry.id for entry in grid if entry.group == "stage-set"}
    about = measured["about"]
    return {
        "generated_by": "tools/make_explorer.py; do not edit by hand",
        "no_accuracy": NO_ACCURACY,
        "probe": about["probe"],
        "verdict_order": docs_verdicts(found),
        "verdicts": {name: about["verdicts"][name] for name in docs_verdicts(found)},
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
        "lanes": lanes(slot_options, _slot_pages()),
        "methods": method_list,
        "options": slot_options,
        "cells": found,
        "index": index,
    }


def docs_verdicts(found: list[dict[str, Any]]) -> list[str]:
    order = _docs_tool().VERDICT_ORDER
    used = {item["verdict"] for item in found}
    return [name for name in order if name in used] + sorted(used - set(order))


def text() -> str:
    """The file: compact JSON on one line, safe inside a ``<script>`` element."""
    encoded = json.dumps(data(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    encoded = re.sub(r"</", r"<\\/", encoded).replace("<!--", "<\\u0021--")
    return encoded + "\n"


def stale() -> list[Path]:
    content = text()
    return [] if OUT.is_file() and OUT.read_text(encoding="utf-8") == content else [OUT]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    args = parser.parse_args(argv)
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"stale: {path.relative_to(REPO)} (run python -m tools.make_explorer)")
        return 1 if outdated else 0
    content = text()
    changed = not OUT.is_file() or OUT.read_text(encoding="utf-8") != content
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(f"{'wrote' if changed else 'unchanged'} {OUT.relative_to(REPO)} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
