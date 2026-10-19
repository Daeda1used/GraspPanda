"""Generate the component catalogue and the compatibility tables of the documentation.

Both are read from the registries, the contract rules and the shipped configs, so they
cannot drift from the code:

* ``docs/components.md`` (*Registered classes*) is written whole: per registry, the
  scope of the slot it fills and, folded, every registered class with its contract,
  status, source and the shipped configs that use it (the page form of ``grasppanda
  components``).
* ``docs/advanced_guides/swapping.md`` (*Swap rules*) is prose with generated regions
  between ``<!-- generated:NAME:begin -->`` and ``<!-- generated:NAME:end -->`` markers:
  which encoder fits which seed-based method as it is or in an adapter, which pairs of
  stages the contracts allow, and how many sets of stages they accept.
* ``docs/advanced_guides/stage_sets.json`` holds every legal set of stages, one per line,
  beside the page.

Run it after adding or changing a component, a contract attribute or a shipped config::

    python -m tools.make_component_docs           # rewrite
    python -m tools.make_component_docs --check   # exit 1 and name what is stale

``tests/unit/test_components_catalogue.py`` runs the check, so a stale page fails CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from grasppanda.cli.components import (
    CONTRACT_FIELDS,
    FIELD_HELP,
    SLOTS,
    Entry,
    collect,
    render_value,
)
from grasppanda.config import Config, ConfigError
from grasppanda.models.base import configured_width
from grasppanda.models.detectors.seed_based import (
    STAGE_REGISTRIES,
    SeedGraspDetector,
    stage_contract_problems,
)
from grasppanda.registry import BACKBONES, DETECTORS, GROUPERS, INJECTED, component_parameters

REPO = Path(__file__).resolve().parents[1]
COMPONENTS_PAGE = REPO / "docs" / "components.md"
SWAPPING_PAGE = REPO / "docs" / "advanced_guides" / "swapping.md"

TITLES = {
    "DETECTORS": "Detectors",
    "BACKBONES": "Scene encoders",
    "IMAGE_ENCODERS": "Image encoders",
    "PROPOSALS": "Seed proposals",
    "SAMPLERS": "Seed samplers",
    "VIEWS": "View stages",
    "ASSIGNERS": "Assigners",
    "GROUPERS": "Groupers",
    "HEADS": "Head bundles",
    "LOSSES": "Primitive losses",
    "POSTPROCESSORS": "Post-processors",
    "DATASETS": "Datasets",
    "TRANSFORMS": "Transforms",
    "EVALUATORS": "Evaluators",
    "HOOKS": "Hooks",
    "OPTIMIZERS": "Optimizers",
    "SCHEDULERS": "Schedulers",
}
YES, NO = "✓", "✗"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _needs(extras: tuple[str, ...]) -> str:
    """Extras as they are named in `pip install -e ".[<extra>]"`; a library with one
    extra per build shows the first and says that there are others."""
    from grasppanda.utils.imports import EXTRA_BUILDS

    cells = []
    for extra in extras:
        builds = EXTRA_BUILDS.get(extra)
        cells.append(f"`{builds[0]}` (or another `{extra}-*` build)" if builds else f"`{extra}`")
    return ", ".join(cells) or "—"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return lines


def _code(value: Any) -> str:
    text = render_value(value)
    return text if text == "—" else f"`{text}`"


# ------------------------------------------------------------------ components page


def _chips() -> Any:
    """``tools/docs_chips.py``: the scopes and the chips every page shares."""
    name = "docs_chips"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: Registry → the slot whose options it holds. A registry of the toolbox itself (detectors,
#: datasets, transforms, evaluators) has none, and so no scope chip.
REGISTRY_SLOTS = {
    "BACKBONES": "backbone",
    "IMAGE_ENCODERS": "image_encoder",
    "PROPOSALS": "proposal",
    "SAMPLERS": "proposal",
    "VIEWS": "view",
    "ASSIGNERS": "assigner",
    "GROUPERS": "grouper",
    "HEADS": "head",
    "LOSSES": "loss",
    "POSTPROCESSORS": "post",
    "HOOKS": "hooks",
    "OPTIMIZERS": "optimizer",
    "SCHEDULERS": "schedule",
}


def _indent(lines: list[str]) -> list[str]:
    return [f"    {line}".rstrip() for line in lines]


def _registry_head(registry: str) -> str:
    """The line under a registry's heading: where it is written, what it does, the methods
    its slot applies to and the page of its options."""
    where, role = SLOTS[registry]
    text = f"`{where}` — {role}"
    slot = REGISTRY_SLOTS.get(registry)
    if slot is None:
        return text
    chips = _chips()
    return f"{text} {chips.scope_chip(slot)} [Options]({chips.SLOT_PAGES[slot]})"


def components_page() -> str:
    entries = collect(in_tree_only=False)
    by_registry: dict[str, list[Entry]] = {}
    for entry in entries:
        if entry.in_tree or entry.registry in ("OPTIMIZERS", "SCHEDULERS"):
            by_registry.setdefault(entry.registry, []).append(entry)
    lines = [
        "<!-- Generated by tools/make_component_docs.py from the registries. Do not edit by",
        "     hand: run `python -m tools.make_component_docs` after changing a component. -->",
        "",
        "# Registered classes",
        "",
        "Everything a config can name with `type:`, registry by registry, generated from the "
        "registries, from what each class declares about itself and from the shipped configs. "
        "A chip says which methods the slot applies to; the toolbox's own registries have none. "
        "The same tables in a terminal:",
        "",
        "```bash",
        "grasppanda components                    # every registry",
        "grasppanda components --slot backbones   # one registry, its columns explained",
        "grasppanda components PTv3               # one class: options, configs, fits",
        "grasppanda components --json             # for scripts",
        "```",
        "",
        "Which class can replace which: [Swap rules](advanced_guides/swapping.md). What was "
        "measured for every option under every method: "
        "[Does it run?](advanced_guides/measured_combinations.md).",
        "",
        '??? note "How to read the tables"',
        "",
        *_indent(
            [
                "- **Contract columns** (between the name and the status) are the class "
                "attributes the config checker compares before anything is built. Each registry "
                "explains its own.",
                "- **Status** is what has been proven. `tested`: unit tests on CPU cover the "
                "class's contract and behaviour; for the stages of a ported method, the method's "
                "README says how it was compared with the reference implementation. "
                "`weights-verified`: additionally, weights released by the authors load strictly "
                "and the outputs match the official implementation; the measured difference is "
                "on [Encoders checked against their authors' code](model_zoo_backbones.md). "
                "`bridge`: an adapter around a library installed as an optional extra; the "
                "library's own code runs, and the tests here cover the adapter (contract, shapes, "
                "gradients), not the library. `—`: the class declares no status. No status is a "
                "claim about grasp-detection accuracy.",
                "- **Source** is the paper, method or library the class implements, and where the "
                "code comes from: written for GraspPanda from the paper and the behaviour of the "
                "reference code (the ported methods; licences and commits of the references are "
                "in the method READMEs), a bridge to a library, or adapted from permissively "
                "licensed code (listed in `THIRD_PARTY_NOTICES.md`).",
                '- **Needs** lists optional dependencies (`pip install -e ".[...]"`).',
                "- **Shipped configs** are the runnable configs under `configs/` whose resolved "
                "tree names the class.",
            ]
        ),
        "",
    ]
    for registry in [name for name in SLOTS if name in by_registry]:
        members = by_registry[registry]
        lines += [f"## {TITLES[registry]}", "", _registry_head(registry), ""]
        if registry in ("OPTIMIZERS", "SCHEDULERS"):  # library classes by name, ours in a table
            # Which classes exist depends on the installed PyTorch, so the page does not
            # list them: it would differ from one environment to the next.
            lines += [
                "Named by class, whatever the installed PyTorch provides; "
                f"`grasppanda components --slot {registry.lower()}` lists them.",
                "",
            ]
            members = [entry for entry in members if entry.in_tree]
            if not members:
                continue
        fields = CONTRACT_FIELDS.get(registry, ())
        with_needs = any(entry.extras for entry in members)
        headers = ["Component", *(f"`{name}`" for name in fields), "Status", "Source"]
        headers += ["What it is"] + (["Needs"] if with_needs else []) + ["Shipped configs"]
        rows = []
        for entry in members:
            row = [f"`{entry.name}`"]
            row += [_code(entry.contract[f]) if f in entry.contract else "—" for f in fields]
            row += [f"`{entry.status}`" if entry.status else "—", entry.source or "—"]
            row += [entry.summary]
            if with_needs:
                row += [_needs(entry.extras)]
            row += ["<br>".join(f"`{config}`" for config in entry.used_by) or "—"]
            rows.append(row)
        body = [f"- `{name}`: {FIELD_HELP[name]}" for name in fields] + ([""] if fields else [])
        body += ["<div data-search-exclude markdown>", "", *_table(headers, rows), "", "</div>"]
        noun = "class" if len(members) == 1 else "classes"
        lines += [f'??? note "{len(members)} registered {noun}"', "", *_indent(body), ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------- compatibility data


def in_tree(registry_name: str) -> list[str]:
    return [entry.name for entry in collect([registry_name], in_tree_only=True)]


def reference_models() -> dict[str, dict[str, Any]]:
    """Detector type -> model node of its shipped reference model, seed-based only."""
    models = {}
    for path in sorted((REPO / "configs" / "_base_" / "models").glob("*.yaml")):
        model = Config.load(path, validate=False).to_dict().get("model") or {}
        if model.get("type") in DETECTORS and issubclass(
            DETECTORS.get(model["type"]), SeedGraspDetector
        ):
            models[model["type"]] = model
    return models


def _wraps(name: str) -> bool:
    """Whether a registered backbone wraps another backbone."""
    params, _ = component_parameters(BACKBONES.get(name))
    return "backbone" in params


def adapters() -> dict[str, str]:
    """Output kind an adapter produces -> its name. An adapter wraps another backbone.

    Wrappers that no contract asks for (point painting, ``compose_inserts = False``) are
    not adapters in this sense: nothing inserts them.
    """
    found = {}
    for name in in_tree("BACKBONES"):
        if _wraps(name) and getattr(BACKBONES.get(name), "compose_inserts", True):
            found[BACKBONES.get(name).output] = name
    return found


def _refusal(model: dict[str, Any]) -> list[str]:
    """Messages of the config check for a model, the width of the seed features aside.

    The tables are about kinds of features and labels. Whether an encoder's width fits a
    grouper that adds the seed features to its crops depends on options, and `compose`
    settles it (an `out_channels` option, or the grouper's `seed_projection`).
    """
    try:
        Config.from_dict({"model": model})
    except ConfigError as error:
        width = set()
        if all(isinstance(model.get(slot), dict) for slot in ("backbone", "grouper")):
            width = set(
                stage_contract_problems(backbone=model["backbone"], grouper=model["grouper"])
            )
        return [issue.message for issue in error.issues if issue.message not in width]
    return []


def encoder_verdicts() -> dict[str, dict[str, str]]:
    """Encoder -> detector type -> "" (its kind fits as is), an adapter name, or "refused"."""
    wrap, models = adapters(), reference_models()
    verdicts: dict[str, dict[str, str]] = {}
    for name in in_tree("BACKBONES"):
        if _wraps(name):
            continue
        verdicts[name] = {}
        for detector, model in models.items():
            if not _refusal({**model, "backbone": {"type": name}}):
                verdicts[name][detector] = ""
                continue
            verdicts[name][detector] = "refused"
            for adapter in wrap.values():
                wrapped = {"type": adapter, "backbone": {"type": name}}
                if not _refusal({**model, "backbone": wrapped}):
                    verdicts[name][detector] = adapter
    return verdicts


def assigner_variants() -> dict[str, dict[str, Any]]:
    """Label -> assigner node: every assigner as is, and with its optional fields on."""
    variants: dict[str, dict[str, Any]] = {}
    registry = STAGE_REGISTRIES["assigner"]
    for name in in_tree("ASSIGNERS"):
        variants[name] = {"type": name}
        params, _ = component_parameters(registry.get(name))
        switches = {
            option: value
            for switch in getattr(registry.get(name), "optional_outputs", {}).values()
            if switch is not None
            for option, value in [switch]
            if option in params and params[option].default != value
        }
        if switches:
            spelled = ", ".join(
                f"{option}: {str(value).lower() if isinstance(value, bool) else value}"
                for option, value in switches.items()
            )
            variants[f"{name} ({spelled})"] = {"type": name, **switches}
    return variants


def proposal_variants() -> dict[str, dict[str, Any]]:
    """Label -> proposal node: every proposal as is, and with the option that leaves a
    seed score to another stage where it has one (``without_seed_score``)."""
    variants: dict[str, dict[str, Any]] = {}
    registry = STAGE_REGISTRIES["proposal"]
    for name in in_tree("PROPOSALS"):
        variants[name] = {"type": name}
        cls = registry.get(name)
        params, _ = component_parameters(cls)
        defaults = {
            key: param.default
            for key, param in params.items()
            if param.default is not param.empty and param.default is not INJECTED
        }
        for score in cls.seed_scores:
            leave = cls.without_seed_score(defaults, score)
            if not isinstance(leave, dict) or leave.get("type", name) != name:
                continue
            switches = {k: v for k, v in leave.items() if k != "type" and defaults.get(k) != v}
            spelled = ", ".join(f"{option}: {value}" for option, value in switches.items())
            variants[f"{name} ({spelled})"] = {"type": name, **switches}
    return variants


class _Kind:
    """Stand-in for "any encoder of this output kind" in a contract check."""

    def __init__(self, output: str) -> None:
        self.cls = type(f"a {output} encoder", (), {"output": output})


def stage_variants() -> dict[str, dict[str, Any]]:
    """Slot -> label -> what to hand to the contract rules."""
    variants: dict[str, dict[str, Any]] = {
        "backbone": {kind: _Kind(kind).cls for kind in ("dense", "subsampled")},
        "assigner": assigner_variants(),
        "proposal": proposal_variants(),
    }
    for slot, registry in (
        ("view", "VIEWS"),
        ("grouper", "GROUPERS"),
        ("head", "HEADS"),
    ):
        variants[slot] = {name: {"type": name} for name in in_tree(registry)}
    return variants


def legal_stage_sets() -> list[dict[str, str]]:
    """Every combination of stage variants the contract rules accept, by label."""
    variants = stage_variants()
    slots = ["backbone", "proposal", "view", "assigner", "grouper", "head"]
    legal = []
    for labels in itertools.product(*(variants[slot] for slot in slots)):
        chosen = {slot: variants[slot][label] for slot, label in zip(slots, labels)}
        if not stage_contract_problems(**chosen):
            legal.append(dict(zip(slots, labels)))
    return legal


def _first_sentence(message: str) -> str:
    match = re.match(r"(.+?\.)(?:\s|$)", message)
    return match.group(1) if match else message


def pair_table(row_slot: str, column_slot: str) -> Iterator[str]:
    """A Markdown table of pairwise verdicts with the reasons numbered below it."""
    variants = stage_variants()
    reasons: list[str] = []
    rows = []
    for row_label, row in variants[row_slot].items():
        cells = [f"`{row_label}`"]
        for column in variants[column_slot].values():
            problems = stage_contract_problems(**{row_slot: row, column_slot: column})
            if not problems:
                cells.append(YES)
                continue
            # Each reason is the first sentence of the error, after the kind of refusal.
            reason = " ".join(
                f"[{getattr(problem, 'kind', 'unclassified')}] {_first_sentence(problem)}"
                for problem in problems
            )
            if reason not in reasons:
                reasons.append(reason)
            cells.append(f"{NO} ({reasons.index(reason) + 1})")
        rows.append(cells)
    head = f"{row_slot} ↓ · {column_slot} →"
    yield from _table([head, *(f"`{label}`" for label in variants[column_slot])], rows)
    if reasons:
        yield ""
        yield from (f"{index}. {reason}" for index, reason in enumerate(reasons, start=1))


# ------------------------------------------------------------------ swapping regions


def method_models() -> dict[str, dict[str, Any]]:
    """Seed-based method → the model node of its shipped config, in page order."""
    from grasppanda.combine.grid import METHODS

    return {
        name: Config.load(REPO / "configs" / f"{config}.yaml", validate=False).to_dict()["model"]
        for name, config in METHODS.items()
    }


def method_encoder_verdicts() -> dict[str, dict[str, str]]:
    """Encoder → method → "" (its kind fits as is), an adapter name, or "refused": the
    verdicts of :func:`encoder_verdicts` on the shipped model of each method."""
    wrap, models = adapters(), method_models()
    verdicts: dict[str, dict[str, str]] = {}
    for name in in_tree("BACKBONES"):
        if _wraps(name):
            continue
        verdicts[name] = {}
        for method, model in models.items():
            if not _refusal({**model, "backbone": {"type": name}}):
                verdicts[name][method] = ""
                continue
            verdicts[name][method] = "refused"
            for adapter in wrap.values():
                wrapped = {"type": adapter, "backbone": {"type": name}}
                if not _refusal({**model, "backbone": wrapped}):
                    verdicts[name][method] = adapter
    return verdicts


def _encoders_region() -> str:
    verdicts = method_encoder_verdicts()
    methods = list(method_models())
    wrap = adapters()
    entries = {entry.name: entry for entry in collect(["BACKBONES"], in_tree_only=True)}
    headers = ["Encoder", "`output`", "Width", "Status", "Needs", *methods]
    rows = []
    for name, per_method in verdicts.items():
        entry = entries[name]
        settable = "out_channels" in component_parameters(BACKBONES.get(name))[0]
        declared = configured_width({"type": name})
        width = ("once built" if declared is None else f"{declared}") + (
            " (`out_channels`)" if settable else ""
        )
        row = [f"`{name}`", f"`{entry.contract['output']}`", width, f"`{entry.status}`"]
        row += [_needs(entry.extras)]
        for method in methods:
            verdict = per_method[method]
            row.append(YES if not verdict else NO if verdict == "refused" else f"in `{verdict}`")
        rows.append(row)
    kinds = Counter(entries[name].contract["output"] for name in verdicts)
    refused = sorted(
        {name for name, per_method in verdicts.items() if "refused" in per_method.values()}
    )
    summary = (
        f"{len(verdicts)} registered scene encoders: "
        + ", ".join(f"{count} return `{kind}` features" for kind, count in sorted(kinds.items()))
        + ". "
    )
    if refused:
        summary += (
            "Refused under at least one method even with an adapter: "
            + ", ".join(f"`{name}`" for name in refused)
            + "."
        )
    else:
        summary += (
            "Every one passes the contract under every seed-based method: as it is where its "
            "output is what the method's proposal needs, otherwise inside the adapter that "
            "`grasppanda compose` inserts ("
            + " or ".join(f"`{name}`" for name in wrap.values())
            + ")."
        )
    legend = [
        f"{YES}: the features of `model.backbone: {{type: <Encoder>}}` live where the proposal "
        "of the method needs them. *in `Adapter`*: the bare swap is refused with a message "
        "that names the adapter; `model.backbone: {type: <Adapter>, backbone: {type: "
        "<Encoder>}}` passes that rule.",
        "",
        "Adapters: "
        + "; ".join(
            f"`{name}` turns any encoder into a `{kind}` one" for kind, name in wrap.items()
        )
        + ".",
        "",
        "*Width* is the number of channels the encoder returns with its constructor "
        "defaults, as it declares it (`Backbone.output_channels`); `(out_channels)` marks an "
        "encoder that has an option of that name to set it. The other stages take whatever "
        "width arrives, with these exceptions among the methods as shipped: "
        + (
            "; ".join(
                f"{method} needs **{width}** channels (`{grouper}` adds the seed features "
                "to its crop features)"
                for method, (width, grouper) in method_seed_widths().items()
            )
            or "none"
        )
        + ". An encoder of another width is refused there by the config check, with both "
        "numbers and the ways out; `grasppanda compose` takes one of them by itself (the "
        "encoder's `out_channels`, or the projected variant of the grouper). *once built*: "
        "the width follows from a library model or a layout that only the constructor "
        "resolves, so the check builds the encoder once, without released weights.",
    ]
    return "\n".join(
        [
            summary,
            "",
            '??? note "Every encoder under every method"',
            "",
            *_indent(
                [
                    "<div data-search-exclude markdown>",
                    "",
                    *_table(headers, rows),
                    "",
                    "</div>",
                    "",
                    *legend,
                ]
            ),
        ]
    )


def method_seed_widths() -> dict[str, tuple[int, str]]:
    """Method → (seed width its shipped grouper fixes, the grouper), for the methods whose
    grouper fixes one (``Grouper.required_seed_channels``)."""
    found = {}
    for method, model in method_models().items():
        cls = GROUPERS.get(model["grouper"]["type"])
        params, _ = component_parameters(cls)
        defaults = {name: param.default for name, param in params.items()}
        width = cls.required_seed_channels({**defaults, **model["grouper"]})
        if width is not None:
            found[method] = (width, cls.__name__)
    return found


#: The pair tables of the swapping page: (row slot, column slot, title).
PAIRS = (
    ("backbone", "proposal", "Encoder output and proposal"),
    ("proposal", "view", "Proposal and view stage"),
    ("view", "assigner", "View stage and assigner"),
    ("assigner", "head", "Assigner and head bundle"),
    ("grouper", "head", "Grouper and head bundle"),
    ("view", "head", "View stage and head bundle"),
)


def _pairs_region() -> str:
    lines: list[str] = []
    for row_slot, column_slot, title in PAIRS:
        table = list(pair_table(row_slot, column_slot))
        rows = [line for line in table[2:] if line.startswith("|")]
        refused = sum(line.count(f"{NO} (") for line in rows)
        total = refused + sum(line.count(f"| {YES} ") for line in rows)
        title = f"{title}: {refused} of {total} pairs refused"
        lines += [f'??? note "{title}"', "", *_indent(table), ""]
    return "\n".join(lines).rstrip()


SLOT_ORDER = ("backbone", "proposal", "view", "assigner", "grouper", "head")
#: Where the list of every legal set of stages is published, beside the swapping page.
STAGE_SETS = REPO / "docs" / "advanced_guides" / "stage_sets.json"


def _labels(model: dict[str, Any]) -> tuple[str, ...]:
    """A shipped model as a set of stage variants (labels in :data:`SLOT_ORDER`)."""
    registry = STAGE_REGISTRIES["assigner"]
    labels = {slot: model[slot]["type"] for slot in SLOT_ORDER}
    labels["backbone"] = BACKBONES.get(model["backbone"]["type"]).output
    # An assigner with its optional fields switched on is a variant of its own.
    params, _ = component_parameters(registry.get(model["assigner"]["type"]))
    resolved = {name: param.default for name, param in params.items()} | model["assigner"]
    for label, variant in assigner_variants().items():
        if len(variant) > 1 and all(resolved.get(k) == v for k, v in variant.items()):
            labels["assigner"] = label
    return tuple(labels[slot] for slot in SLOT_ORDER)


def method_stage_sets() -> dict[str, tuple[str, ...]]:
    """Method → the set of stage variants of its shipped model, in page order."""
    return {method: _labels(model) for method, model in method_models().items()}


def _stage_sets_region() -> str:
    legal = legal_stage_sets()
    variants = stage_variants()
    total = 1
    for slot in variants:
        total *= len(variants[slot])
    keys = {tuple(chosen[slot] for slot in SLOT_ORDER) for chosen in legal}
    shipped = method_stage_sets()
    outside = {method: key for method, key in shipped.items() if key not in keys}
    if outside:  # a model the contracts would refuse cannot be shipped
        raise SystemExit(f"methods outside the legal stage sets: {outside}")
    kinds = list(variants["backbone"])
    counts = Counter((chosen["head"], chosen["backbone"]) for chosen in legal)
    rows = [
        [
            f"`{head}`",
            *(str(counts[head, kind]) for kind in kinds),
            ", ".join(method for method, key in shipped.items() if key[-1] == head) or "—",
        ]
        for head in variants["head"]
    ]
    return "\n".join(
        [
            f"The contracts accept **{len(legal)} of {total}** combinations of the registered "
            "stage classes (an assigner counts twice when it has options that switch optional "
            "fields on, a proposal when it has an option that leaves a seed score to the view "
            "stage). By head bundle and kind of encoder output:",
            "",
            *_table(["Head bundle", *(f"`{kind}` encoder" for kind in kinds), "Shipped by"], rows),
            "",
            f"Every set, one per line: [stage_sets.json]({STAGE_SETS.name}) (the stage classes "
            "by index into `variants`; `methods` holds the set of each method as shipped). A "
            "sample of them was trained on real frames: "
            "[Sets of stages](measured_combinations.md#sets-of-stages).",
        ]
    )


def stage_sets_json() -> str:
    """Every legal set of stages as a file: the variants of each slot once, and every set as
    the indices of its variants, one set per line."""
    legal = legal_stage_sets()
    variants = {slot: list(stage_variants()[slot]) for slot in SLOT_ORDER}
    index = {
        slot: {label: i for i, label in enumerate(labels)} for slot, labels in variants.items()
    }
    head = {
        "about": (
            "Every combination of stage variants the contract rules of the seed-based skeleton "
            "accept (tools/make_component_docs.py; docs/advanced_guides/swapping.md). A set is "
            "the index of its variant in `variants`, slot by slot in the order of `slots`. "
            "Accepted by the contracts is not measured: a sample was probed on real frames "
            "(docs/advanced_guides/measured_combinations.md)."
        ),
        "slots": list(SLOT_ORDER),
        "variants": variants,
        "methods": {
            method: [index[slot][label] for slot, label in zip(SLOT_ORDER, key, strict=True)]
            for method, key in method_stage_sets().items()
        },
        "count": len(legal),
    }
    body = json.dumps(head, indent=2, ensure_ascii=False)
    sets = [json.dumps([index[slot][chosen[slot]] for slot in SLOT_ORDER]) for chosen in legal]
    return (
        body[:-2] + ',\n  "sets": [\n' + ",\n".join(f"    {item}" for item in sets) + "\n  ]\n}\n"
    )


def swapping_regions() -> dict[str, str]:
    return {
        "encoders": _encoders_region(),
        "pairs": _pairs_region(),
        "stage-sets": _stage_sets_region(),
    }


def apply_regions(text: str, regions: dict[str, str]) -> str:
    """Replace every generated region of a page; a missing marker is an error."""
    for name, body in regions.items():
        begin, end = f"<!-- generated:{name}:begin -->", f"<!-- generated:{name}:end -->"
        if begin not in text or end not in text:
            raise SystemExit(f"{SWAPPING_PAGE}: markers of region '{name}' not found")
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text


def expected() -> dict[Path, str]:
    """Path -> content each generated page should have."""
    pages = {COMPONENTS_PAGE: components_page(), STAGE_SETS: stage_sets_json()}
    if SWAPPING_PAGE.is_file():
        current = SWAPPING_PAGE.read_text(encoding="utf-8")
        pages[SWAPPING_PAGE] = apply_regions(current, swapping_regions())
    return pages


def stale() -> list[Path]:
    """Generated pages whose content on disk is not what the code produces."""
    return [
        path
        for path, content in expected().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    args = parser.parse_args(argv)
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"stale: {path.relative_to(REPO)} (run python -m tools.make_component_docs)")
        return 1 if outdated else 0
    for path, content in expected().items():
        changed = not path.is_file() or path.read_text(encoding="utf-8") != content
        path.write_text(content, encoding="utf-8")
        print(f"{'wrote' if changed else 'unchanged'} {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
