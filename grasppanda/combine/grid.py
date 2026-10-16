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
    constructor defaults no preset already states; a component named by its type that
    lacks an option only a preset supplies is no combination and has no cell (its presets
    have theirs);
``stage-set``
    every set of stage variants the contract rules accept, once under a dense and once
    under a sub-sampling encoder;
``family``
    detectors that are not built on the seed-based skeleton, as they are;
``family-slot``
    every preset of every component slot such a detector has (HGGD: ``image_encoder``).
``post``
    every post-processing preset as the evaluation variant ``post`` of all six methods
    (:data:`POST_METHODS`): a post-processor reads grasps, not a network, so a cell is
    the method's prediction plus the chain.
``loss``
    every loss preset in every kind of loss term it accepts, under every seed-based
    method that has a term of that kind, in one representative term per kind and method
    (:func:`loss_terms`); and the hand-picked cells of :data:`LOSS_CELLS`, which replace
    several terms of a method at once or pair a loss with a term the contract rules
    refuse. That a loss runs in *every* term of its kind is a unit test on tiny models
    (``tests/unit/test_losses_zoo.py``); the grid shows it on real frames.
``recipe``
    every recipe preset (optimizer, schedule, hooks, augment, recipe) under every method
    it can apply to: all six, except the presets that name the scene encoder of the
    seed-based skeleton (:data:`RECIPE_ENCODER_PRESETS`), which HGGD does not have. An
    augmentation HGGD's contract does not accept is refused there, and that refusal is
    the cell's outcome.
``cross``
    options of several slots replaced together (:data:`CROSS_CELLS`): an encoder with a
    grouper, a proposal, a view stage, a recipe or a post-processing preset. The
    other groups replace one thing at a time (or five stages under two fixed encoders);
    these cells are a hand-picked sample of what a user composes, not a covering of it.

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
from .presets import LOSS_SLOT, POST_SLOT, Preset, presets
from .recipes import recipes

__all__ = [
    "CROSS_CELLS",
    "FAMILIES",
    "LOSS_CELLS",
    "LOSS_NOT_PROBED",
    "METHODS",
    "POST_METHODS",
    "POST_NOT_PROBED",
    "RECIPE_ENCODER_PRESETS",
    "RECIPE_METHODS",
    "RECIPE_REQUIRES",
    "STAGE_SET_ENCODERS",
    "STAGE_SET_SKIPPED",
    "STAGE_SLOTS",
    "Cell",
    "cells",
    "encoder_presets",
    "family_slots",
    "loss_terms",
    "needs_an_option",
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
#: The methods post-processing is probed under: all six. A post-processor reads grasps and
#: the observation, never a network, so it needs no training of its own; a cell is the
#: method's prediction plus the chain, on the grasps each method predicts.
POST_METHODS = (*METHODS, *FAMILIES)
#: Post-processing presets without a cell, each with the reason.
POST_NOT_PROBED: dict[str, str] = {
    "target-mask": "reads observation.masks, which no shipped dataset fills",
}
#: The hand-picked loss cells: a method and the loss terms that are replaced together, as
#: ``slot.option=preset``. Each replaces several terms of one method at once (every loss
#: preset sits in one under GSNet and in one under another method) or pairs a loss with a
#: term the config check refuses (the last three). Beside them every loss preset has a cell
#: of one term per kind it accepts and per seed-based method (:func:`loss_terms`).
LOSS_CELLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "GSNet",
        (
            "proposal.objectness_loss=focal",
            "head.score_loss=quality-focal",
            "head.width_loss=balanced-l1",
        ),
    ),
    (
        "GSNet",
        (
            "proposal.objectness_loss=softmax-focal",
            "head.score_loss=varifocal",
            "head.width_loss=huber",
        ),
    ),
    (
        "GSNet",
        (
            "proposal.objectness_loss=label-smoothing",
            "proposal.graspness_loss=bce",
            "head.width_loss=ghm-r",
        ),
    ),
    (
        "GSNet",
        (
            "proposal.objectness_loss=poly1",
            "head.score_loss=ghm-c",
            "head.width_loss=charbonnier",
        ),
    ),
    (
        "GSNet",
        (
            "proposal.objectness_loss=ohem",
            "head.score_loss=ohem-smooth-l1",
            "head.width_loss=cauchy",
        ),
    ),
    ("GSNet", ("proposal.objectness_loss=class-balanced",)),
    ("GSNet", ("proposal.objectness_loss=dice",)),
    ("GSNet", ("proposal.objectness_loss=lovasz-softmax",)),
    (
        "EconomicGrasp",
        (
            "proposal.objectness_loss=dice",
            "proposal.graspness_loss=quality-focal",
            "head.angle_loss=softmax-focal",
            "head.depth_loss=label-smoothing",
            "head.score_loss=class-balanced",
            "head.width_loss=cauchy",
        ),
    ),
    (
        "FineGrasp",
        (
            "proposal.objectness_loss=lovasz-softmax",
            "proposal.graspness_loss=bce",
            "view.loss=varifocal",
            "head.angle_loss=poly1",
            "head.depth_loss=ohem",
            "head.score_loss=focal",
        ),
    ),
    (
        "GraspNet-baseline",
        (
            "view.view_loss=ohem-smooth-l1",
            "head.score_loss=ghm-r",
            "head.angle_loss=ghm-c",
            "head.width_loss=balanced-l1",
            "head.tolerance_loss=charbonnier",
        ),
    ),
    (
        "Scale-Balanced-Grasp",
        (
            "view.view_loss=huber",
            "head.angle_loss=class-balanced",
            "head.width_loss=huber",
        ),
    ),
    (
        "GSNet",
        (
            "proposal.objectness_loss=uncertainty-weighting",
            "proposal.graspness_loss=uncertainty-weighting",
            "view.loss=uncertainty-weighting",
            "head.score_loss=uncertainty-weighting",
            "head.width_loss=uncertainty-weighting",
        ),
    ),
    (
        "GraspNet-baseline",
        (
            "view.objectness_loss=uncertainty-weighting",
            "view.view_loss=uncertainty-weighting",
            "head.score_loss=uncertainty-weighting",
            "head.angle_loss=uncertainty-weighting",
            "head.width_loss=uncertainty-weighting",
            "head.tolerance_loss=uncertainty-weighting",
        ),
    ),
    ("GSNet", ("head.width_loss=focal",)),  # refused: a width is no class and no quality
    ("GraspNet-baseline", ("head.score_loss=quality-focal",)),  # refused: targets up to ln 10
    ("Scale-Balanced-Grasp", ("head.angle_loss=dice",)),  # refused: it cannot weight seeds
)
#: Loss presets without a cell of their own, each with the reason.
LOSS_NOT_PROBED: dict[str, str] = {
    "distribution-focal": (
        "fits `bins` terms only, which no shipped method has; probed as the width loss of "
        "the head preset `decoupled-grid-bins` (stage cells)"
    ),
}
#: The stages that own loss terms, in the order of the pipeline.
LOSS_STAGES = ("proposal", "view", "head")
#: The cross cells: a method and options of several slots replaced together, as
#: ``slot=choice``. Each pairs a scene encoder (or, under HGGD, an image encoder) with
#: options of other slots, so that every method and every slot appears at least once. One
#: of them is a request the config check refuses. The pairs of a request are in the order
#: of the pipeline (stages, post-processing, recipe), the order the explorer writes them in.
CROSS_CELLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GSNet", ("backbone=point-transformer-v2", "grouper=cylinder-vector-attention")),
    ("GSNet", ("backbone=pointvector", "proposal=graspness-nms", "grouper=roi-grid")),
    ("GSNet", ("backbone=minkunet34c", "view=healpix-192", "grouper=multi-scale-linear")),
    (
        "GSNet",
        ("backbone=volt", "proposal=graspness-fpsample", "view=icosphere-162", "grouper=knn"),
    ),
    ("GSNet", ("backbone=ptv3", "proposal=dense-fps", "view=direction-vmf", "grouper=ball")),
    (
        "GSNet",
        (
            "backbone=uni3d",
            "view=rotation-6d",
            "grouper=multi-scale",
            "optimizer=schedulefree-adamw",
        ),
    ),
    ("EconomicGrasp", ("backbone=dela", "grouper=pyg-edge-conv", "optimizer=adamw")),
    (
        "EconomicGrasp",
        (
            "backbone=spunet-gaussiancross",
            "proposal=graspness-voxel",
            "grouper=box",
            "schedule=warmup-cosine",
        ),
    ),
    ("EconomicGrasp", ("backbone=pointmlp", "view=lattice-topk", "grouper=multi-scale")),
    ("FineGrasp", ("backbone=randlanet", "view=lattice-soft-argmax", "hooks=ema")),
    ("FineGrasp", ("backbone=point-transformer-v1", "grouper=cylinder-invres")),
    (
        "FineGrasp",  # refused: the head reads crops without a depth axis
        ("backbone=point-transformer-v1", "view=lattice-best-view", "grouper=depth-ball"),
    ),
    ("GraspNet-baseline", ("backbone=pointnet2-msg", "grouper=depth-ball", "post=nms")),
    (
        "GraspNet-baseline",
        (
            "backbone=assanet",
            "view=two-vector-baseline",
            "grouper=depth-knn",
            "optimizer=timm-lamb",
        ),
    ),
    ("Scale-Balanced-Grasp", ("backbone=pointmetabase", "grouper=depth-cylinder-attentive")),
    ("Scale-Balanced-Grasp", ("backbone=dgcnn", "grouper=depth-box", "post=gripper-limits")),
    (
        "HGGD",
        ("image_encoder=convnextv2-atto", "post=nms", "optimizer=adamw", "schedule=cosine"),
    ),
    ("HGGD", ("image_encoder=tv-mobilenetv3-large", "post=top-k", "hooks=ema")),
)
#: Stage variants that are left out of the stage sets, each with the reason. A variant
#: that shares every contract with another one would only repeat that one's sets; it is
#: probed alone, one stage replaced under every method (group ``stage``).
STAGE_SET_SKIPPED: dict[str, str] = {
    "swad-focal": "preset `swad` with another score loss: the same contracts",
    "economic-focal": "preset `economic` with other class losses: the same contracts",
    "decoupled-grid": "the contracts of `swad` (axes, dense labels); probed alone",
    "decoupled-grid-bins": "as `decoupled-grid`",
    "decoupled-grid-class": "as `decoupled-grid`",
    "decoupled-grid-regression": "as `decoupled-grid`",
}
#: The methods a recipe preset is probed under: every method, seed-based or not. An
#: optimizer, a schedule or a hook applies to any detector; an augmentation a detector's
#: data contract does not accept is refused by the config check, which is a measurement.
RECIPE_METHODS = (*METHODS, *FAMILIES)
#: Recipe presets that name the `backbone` of the seed-based skeleton: probed under the
#: seed-based methods, not under HGGD, which has no such module.
RECIPE_ENCODER_PRESETS = dict.fromkeys(
    (
        "optimizer/encoder-lr-0.1",
        "optimizer/layer-decay",
        "hooks/freeze-encoder",
        "recipe/finetune-encoder",
        "recipe/ptv3-adamw-onecycle",
    ),
    tuple(METHODS),
)
#: What else a recipe preset needs in the same request.
RECIPE_REQUIRES = {"recipe/ptv3-adamw-onecycle": ("backbone=ptv3",)}
#: Slots of the seed-based skeleton besides the encoder, in the order data flows.
STAGE_SLOTS = ("proposal", "view", "assigner", "grouper", "head")
#: Encoder preset used for the stage sets, by the kind of features it returns.
STAGE_SET_ENCODERS = {"dense": "sparse-unet14", "subsampled": "pointnet2"}


@dataclass(frozen=True)
class Cell:
    """One combination of the grid.

    Attributes:
        id: Unique name, also the name of its files (``encoder/GSNet/PTv3``).
        group: ``reference``, ``encoder``, ``stage``, ``stage-set``, ``family``,
            ``family-slot``, ``post``, ``loss``, ``recipe`` or ``cross``.
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


def needs_an_option(slot: str, choice: str, component: str) -> bool:
    """Whether a stage variant is a component named by its type that lacks a required
    option, one only a preset supplies (``MultiScaleGroup`` without its ``branch``).

    Such a request is refused for the missing option before any contract is compared:
    it is not a combination, so it has no cell; the presets of the component have theirs.
    """
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    if choice in presets().get(slot, {}):
        return False
    params, _ = component_parameters(STAGE_REGISTRIES[slot].get(component))
    return any(param.default is param.empty for param in params.values())


def loss_terms(method: str) -> dict[str, str]:
    """Kind of loss term → the representative term of that kind of a seed-based method,
    as ``<stage>.<loss option>``.

    The representative is the **last** term of the kind in the order of the pipeline
    (:data:`LOSS_STAGES`: proposal, view, head; within a stage, the order the stage
    declares its terms in): the term nearest the grasp. That is where the methods differ
    (three head bundles), while GSNet, EconomicGrasp and FineGrasp share their proposal.
    """
    from grasppanda.config import Config
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    model = Config.load(METHODS[method]).resolved()["model"]
    found: dict[str, str] = {}
    for slot in LOSS_STAGES:
        node = model[slot]
        cls = STAGE_REGISTRIES[slot].get(node["type"])
        reader = getattr(cls, "loss_terms_for", None)
        options = {name: value for name, value in node.items() if name != "type"}
        terms = reader(options) if reader else getattr(cls, "loss_terms", {})
        for option, kind in terms.items():
            found[kind] = f"{slot}.{option}"  # a later term of the kind replaces an earlier
    return found


def _loss_cells() -> list[tuple[str, tuple[str, ...]]]:
    """``(method, sets)`` of every loss cell: the hand-picked ones, then one per loss
    preset, kind of term it accepts and seed-based method with a term of that kind."""
    from grasppanda.models.losses.terms import TERM_KINDS, accepted_terms

    found = list(LOSS_CELLS)
    seen = {(method, tuple(sets)) for method, sets in found}
    terms = {method: loss_terms(method) for method in METHODS}
    for preset in presets().get(LOSS_SLOT, {}).values():
        if preset.name in LOSS_NOT_PROBED:
            continue
        found_terms = accepted_terms(dict(preset.node))
        accepted = found_terms[1] if found_terms is not None else ()
        for method in METHODS:
            for kind in TERM_KINDS:
                if kind in accepted and kind in terms[method]:
                    request = (method, (f"{terms[method][kind]}={preset.name}",))
                    if request not in seen:
                        seen.add(request)
                        found.append(request)
    return found


def family_slots(method: str) -> dict[str, Any]:
    """Slot → registry of the component slots of a detector family of :data:`FAMILIES`."""
    from grasppanda.config import Config
    from grasppanda.registry import DETECTORS

    from .compose import component_slots

    detector = Config.load(FAMILIES[method]).resolved()["model"]["type"]
    return component_slots(DETECTORS.get(detector))


def _replaces_an_inner_part(slot: str, choice: str) -> bool:
    """Whether a variant only replaces a part *inside* its stage (the sampler of a proposal,
    the score that selects) and leaves every contract with the neighbours as it is.

    Such a variant is measured under every method in the ``stage`` group. In the stage
    sets it would multiply the sets without measuring a new contract, so the sets take
    the variant that states the constructor defaults of the component and no other.
    """
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    preset = presets().get(slot, {}).get(choice)
    if preset is None or slot != "proposal":
        return False
    registry = STAGE_REGISTRIES[slot]
    if not hasattr(registry.get(preset.type), "sampler_inputs"):
        return False
    # A component node among the options is not compared by `_states_defaults`: the
    # default sampler is named here.
    sampler = preset.node.get("sampler") or {}
    return sampler.get("type", "FPS") != "FPS" or not _states_defaults(preset, registry)


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

    options = {
        slot: [
            (choice, component)
            for choice, component in stage_variants(slot)
            if _in_tree(STAGE_REGISTRIES[slot], component)
            and not needs_an_option(slot, choice, component)
            and choice not in STAGE_SET_SKIPPED
            and not _replaces_an_inner_part(slot, choice)
        ]
        for slot in STAGE_SLOTS
    }
    for kind in STAGE_SET_ENCODERS:
        # "Any encoder of this kind": a set is a statement about the five stages. The
        # width of the encoder is fitted by compose where a stage fixes one.
        backbone = type(f"a {kind} encoder", (), {"output": kind})
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
            if needs_an_option(slot, choice, component):  # not a combination
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
    # The stage sets take seconds to enumerate (every product of the variants is checked
    # against the contract rules): only when they are asked for.
    stage_sets = _stage_sets() if groups is None or "stage-set" in groups else iter(())
    for kind, picked in stage_sets:
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
    for method, sets in _loss_cells():
        names = [item.partition("=")[2] for item in sets]
        terms = [item.partition("=")[0] for item in sets]
        found.append(
            Cell(
                f"loss/{method}/{'+'.join(dict.fromkeys(names))}@{'+'.join(terms)}",
                "loss",
                method,
                METHODS[method],
                sets,
                LOSS_SLOT,
                "+".join(presets()[LOSS_SLOT][name].type for name in names),
                "+".join(names),
            )
        )
    bases = {**METHODS, **FAMILIES}
    for slot, members in recipes().items():
        for name, preset in members.items():
            key = f"{slot}/{name}"
            for method in RECIPE_ENCODER_PRESETS.get(key, RECIPE_METHODS):
                found.append(
                    Cell(
                        f"recipe/{method}/{key}",
                        "recipe",
                        method,
                        bases[method],
                        (*RECIPE_REQUIRES.get(key, ()), f"{slot}={name}"),
                        slot,
                        preset.type,
                        name,
                    )
                )
    for method, sets in CROSS_CELLS:
        name = "+".join(item.partition("=")[2] for item in sets)
        found.append(Cell(f"cross/{method}/{name}", "cross", method, bases[method], sets))
    if groups is not None:
        found = [cell for cell in found if cell.group in groups]
    return found
