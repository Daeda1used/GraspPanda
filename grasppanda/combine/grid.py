"""The grid of combinations that is probed for the measured compatibility page.

A cell is a compose request (a base config and ``slot=choice`` pairs) plus where it sits
in the tables. The grid is derived from the registries and the presets, so a component
added to either gets its cells without anyone listing them:

``reference``
    every method config as it is shipped;
``encoder``
    every registered scene encoder (through its preset) under every seed-based method;
``stage``
    every variant of every other stage, one stage replaced at a time, under every
    seed-based method. The variants of a slot are its presets plus every component whose
    constructor defaults no preset already states;
``stage-set``
    every set of stage variants the contract rules accept, once under a dense and once
    under a sub-sampling encoder;
``family``
    detectors that are not built on the seed-based skeleton, as they are;
``family-slot``
    every preset of every component slot such a detector has (HGGD: ``image_encoder``).
``post``
    every post-processing preset as the evaluation variant ``post`` of three methods
    (:data:`POST_METHODS`), seed-based or not: a post-processor reads grasps, not a network.

``tools/probe_grid.py`` runs the cells; ``tools/make_combination_docs.py`` renders what
was recorded. Nothing here probes anything.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from grasppanda.registry import BACKBONES, component_parameters

from .compose import is_adapter
from .presets import POST_SLOT, Preset, presets

__all__ = [
    "FAMILIES",
    "METHODS",
    "POST_METHODS",
    "POST_NOT_PROBED",
    "STAGE_SET_ENCODERS",
    "STAGE_SLOTS",
    "Cell",
    "cells",
    "encoder_presets",
    "family_slots",
    "stage_variants",
]

#: The seed-based methods of the grid: the name shown in the tables → its config.
METHODS: dict[str, str] = {
    "GSNet": "gsnet/gsnet_sunet14_g1b-realsense_10e",
    "EconomicGrasp": "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e",
    "FineGrasp": "finegrasp/finegrasp_sunet14_g1b-realsense_10e",
    "GraspNet-baseline": "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e",
    "Scale-Balanced-Grasp": "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e",
}
#: Detector families of their own, probed as they are.
FAMILIES: dict[str, str] = {"HGGD": "hggd/hggd_r34w8_g1b-realsense_15e"}
#: The methods post-processing is probed under. A post-processor reads grasps and the
#: observation, never a network, so three kinds of input cover the six methods: seeds on a
#: dense cloud with the full cloud kept (GSNet, as EconomicGrasp and FineGrasp), seeds of a
#: sub-sampling encoder (GraspNet-baseline, as Scale-Balanced-Grasp), and HGGD, whose grasp
#: centres are computed and whose observation carries images.
POST_METHODS = ("GSNet", "GraspNet-baseline", "HGGD")
#: Post-processing presets without a cell, each with the reason.
POST_NOT_PROBED: dict[str, str] = {
    "target-mask": "reads observation.masks, which no shipped dataset fills",
}
#: Slots of the seed-based skeleton besides the encoder, in the order data flows.
STAGE_SLOTS = ("proposal", "view", "assigner", "grouper", "head")
#: Encoder preset used for the stage sets, by the kind of features it returns.
STAGE_SET_ENCODERS = {"dense": "sparse-unet14", "subsampled": "pointnet2"}


@dataclass(frozen=True)
class Cell:
    """One combination of the grid.

    Attributes:
        id: Unique name, also the name of its files (``encoder/GSNet/PTv3``).
        group: ``reference``, ``encoder``, ``stage``, ``stage-set``, ``family`` or
            ``family-slot``.
        method: The method (a key of :data:`METHODS` or :data:`FAMILIES`).
        base: The config it is composed from.
        sets: ``slot=choice`` pairs of the compose request; empty for a config as shipped.
        slot: The slot that is replaced, for ``encoder`` and ``stage`` cells.
        component: The component type that is put there.
        variant: How the cell is labelled in a table: the preset name, or the component
            type for constructor defaults.
    """

    id: str
    group: str
    method: str
    base: str
    sets: tuple[str, ...] = ()
    slot: str = ""
    component: str = ""
    variant: str = ""


def _states_defaults(preset: Preset, registry: Any) -> bool:
    """Whether a preset sets nothing but constructor defaults of its component.

    A component node among its options (a loss) is not compared: the default of such an
    option is ``None``, for "the loss this stage builds itself", and a stage preset writes
    that loss out.
    """
    params, _ = component_parameters(registry.get(preset.type))
    for name, value in preset.node.items():
        default = params[name].default if name in params else object()
        if name == "type" or (default is None and isinstance(value, dict) and "type" in value):
            continue
        if isinstance(default, tuple):
            default = list(default)
        if default != value:
            return False
    return True


def encoder_presets() -> dict[str, list[Preset]]:
    """Registered scene encoder (adapters excluded) → its presets."""
    members = presets().get("backbone", {}).values()
    return {
        name: [preset for preset in members if preset.type == name]
        for name in BACKBONES.names()
        if not is_adapter(BACKBONES.get(name))
    }


def stage_variants(slot: str) -> list[tuple[str, str]]:
    """``(choice, component type)`` of every variant of a stage slot.

    The presets of the slot, then every registered component whose constructor defaults
    are not what one of its presets already states.
    """
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    registry = STAGE_REGISTRIES[slot]
    members = list(presets().get(slot, {}).values())
    variants = [(preset.name, preset.type) for preset in members]
    for name in registry.names():
        stated = any(p.type == name and _states_defaults(p, registry) for p in members)
        if not stated:
            variants.append((name, name))
    return variants


def family_slots(method: str) -> dict[str, Any]:
    """Slot → registry of the component slots of a detector family of :data:`FAMILIES`."""
    from grasppanda.config import Config
    from grasppanda.registry import DETECTORS

    from .compose import component_slots

    detector = Config.load(FAMILIES[method]).resolved()["model"]["type"]
    return component_slots(DETECTORS.get(detector))


def _in_tree(registry: Any, name: str) -> bool:
    return registry.get(name).__module__.split(".")[0] == "grasppanda"


def _stage_sets() -> Iterator[tuple[str, dict[str, tuple[str, str]]]]:
    """Every combination of stage variants the contract rules accept, with the kind of
    encoder it needs."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems

    members = presets()

    def node(slot: str, choice: str) -> dict[str, Any]:
        preset = members.get(slot, {}).get(choice)
        return dict(preset.node) if preset is not None else {"type": choice}

    def complete(slot: str, choice: str, component: str) -> bool:
        """A bare component that lacks a required option is no variant of a stage set."""
        if choice in members.get(slot, {}):
            return True
        params, _ = component_parameters(STAGE_REGISTRIES[slot].get(component))
        return all(param.default is not param.empty for param in params.values())

    options = {
        slot: [
            (choice, component)
            for choice, component in stage_variants(slot)
            if _in_tree(STAGE_REGISTRIES[slot], component) and complete(slot, choice, component)
        ]
        for slot in STAGE_SLOTS
    }
    for kind, encoder in STAGE_SET_ENCODERS.items():
        backbone = dict(members["backbone"][encoder].node)
        for chosen in itertools.product(*(options[slot] for slot in STAGE_SLOTS)):
            picked = dict(zip(STAGE_SLOTS, chosen))
            nodes = {slot: node(slot, choice) for slot, (choice, _) in picked.items()}
            try:
                problems = stage_contract_problems(backbone=backbone, **nodes)
            except Exception:  # a variant that cannot be checked alone (a missing option)
                continue
            if not problems:
                yield kind, picked


def _stage_set_base(picked: dict[str, tuple[str, str]]) -> str:
    """The method config whose data recipe serves the labels a stage set reads."""
    from grasppanda.registry import ASSIGNERS, HEADS

    assigner = ASSIGNERS.get(picked["assigner"][1])
    if getattr(assigner, "label_format", "dense") == "compact":
        return "EconomicGrasp"
    if "weight" in getattr(assigner, "optional_outputs", {}):
        return "Scale-Balanced-Grasp"
    if "depth" in getattr(HEADS.get(picked["head"][1]), "in_axes", ()):
        return "GraspNet-baseline"
    return "GSNet"


def cells(groups: tuple[str, ...] | None = None) -> list[Cell]:
    """Every cell of the grid, in a fixed order; ``groups`` restricts it."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    found: list[Cell] = []
    for method, base in METHODS.items():
        found.append(Cell(f"reference/{method}", "reference", method, base))
    for encoder, members in encoder_presets().items():
        if not _in_tree(BACKBONES, encoder):
            continue
        for method, base in METHODS.items():
            for preset in members or [None]:
                choice = preset.name if preset is not None else encoder
                suffix = "" if len(members) <= 1 else f"@{choice}"
                found.append(
                    Cell(
                        f"encoder/{method}/{encoder}{suffix}",
                        "encoder",
                        method,
                        base,
                        (f"backbone={choice}",),
                        "backbone",
                        encoder,
                        choice,
                    )
                )
    for slot in STAGE_SLOTS:
        for choice, component in stage_variants(slot):
            if not _in_tree(STAGE_REGISTRIES[slot], component):
                continue
            for method, base in METHODS.items():
                found.append(
                    Cell(
                        f"stage/{method}/{slot}/{choice}",
                        "stage",
                        method,
                        base,
                        (f"{slot}={choice}",),
                        slot,
                        component,
                        choice,
                    )
                )
    for kind, picked in _stage_sets():
        method = _stage_set_base(picked)
        sets = [f"backbone={STAGE_SET_ENCODERS[kind]}"]
        sets += [f"{slot}={choice}" for slot, (choice, _) in picked.items()]
        name = "+".join(choice for choice, _ in picked.values())
        found.append(
            Cell(
                f"stage-set/{kind}/{name}",
                "stage-set",
                method,
                METHODS[method],
                tuple(sets),
                variant=kind,
            )
        )
    for method, base in FAMILIES.items():
        found.append(Cell(f"family/{method}", "family", method, base))
    for method, base in FAMILIES.items():
        for slot in family_slots(method):
            for preset in presets().get(slot, {}).values():
                found.append(
                    Cell(
                        f"family-slot/{method}/{slot}/{preset.name}",
                        "family-slot",
                        method,
                        base,
                        (f"{slot}={preset.name}",),
                        slot,
                        preset.type,
                        preset.name,
                    )
                )
    for preset in presets().get(POST_SLOT, {}).values():
        if preset.name in POST_NOT_PROBED:
            continue
        for method in POST_METHODS:
            base = {**METHODS, **FAMILIES}[method]
            found.append(
                Cell(
                    f"post/{method}/{preset.name}",
                    "post",
                    method,
                    base,
                    (f"{POST_SLOT}={preset.name}",),
                    POST_SLOT,
                    preset.type,
                    preset.name,
                )
            )
    if groups is not None:
        found = [cell for cell in found if cell.group in groups]
    return found
