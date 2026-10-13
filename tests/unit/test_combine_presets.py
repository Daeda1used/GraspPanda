"""Presets: every file is well formed, and every component that needs one has one.

A preset is what a newcomer cannot guess about a component. The tripwire at the end makes
that a property of the registries: whoever registers a scene encoder or a stage of the
seed-based skeleton must either write a preset or state why the constructor defaults are
enough.
"""

from __future__ import annotations

import pytest

from grasppanda.cli.components import collect
from grasppanda.combine.compose import is_adapter
from grasppanda.combine.grid import (
    FAMILIES,
    METHODS,
    encoder_presets,
    family_slots,
    stage_variants,
)
from grasppanda.combine.presets import PRESET_ROOT, RETUNE_HEADING, PresetError, find, presets, read
from grasppanda.config import Config, parse_yaml
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES
from grasppanda.registry import BACKBONES, LOSSES, POSTPROCESSORS, component_parameters

#: Slot → registry of every slot that has presets: the stages, post-processing, whose
#: presets are fragments of `evaluation.variants` (grasppanda.combine.post), and the
#: losses, whose presets go into a loss option of a stage (`head.score_loss=<preset>`).
SLOT_REGISTRIES = {**STAGE_REGISTRIES, "post": POSTPROCESSORS, "loss": LOSSES}

#: Stage presets that are an option and not a stage of a shipped reference model, each
#: with what it is. Every other stage preset is, option for option, a stage of a method.
OPTION_PRESETS = {
    "head/decoupled-grid": "DecoupledGridHead, a head no ported method uses",
    "head/decoupled-grid-bins": "DecoupledGridHead with the width as a distribution over bins",
    "head/decoupled-grid-class": "DecoupledGridHead with depth and width classified",
    "head/decoupled-grid-regression": "DecoupledGridHead with depth and width regressed",
    "head/swad-focal": "SWADHead with a quality focal loss and fixed-range score targets",
    "head/economic-focal": "EconomicHead with softmax focal losses on its class terms",
}

#: Stages that need no preset, each with the reason. The constructor defaults of every
#: one are the reference values of the method named, and the stage reads no label asset
#: beyond what its own method's dataset config already serves.
DEFAULTS_ARE_ENOUGH = {
    "SubsampledSeeds": "one option, the seed count, which the encoder's subsample fixes",
    "CylinderGroup": "defaults are GSNet's crop; no loss, no label asset",
    "CylinderInteractionGroup": "defaults are EconomicGrasp's crop; no loss, no label asset",
}

#: Stages that no ported method uses: options a visitor may swap in. Their presets state
#: settings for table-top clouds of GraspNet-1Billion, not a stage of a reference model.
BEYOND_THE_REFERENCES = {
    "LatticeViewStage",
    "IcosphereViewStage",
    "HealpixViewStage",
    "DirectionViewStage",
    "TwoVectorViewStage",
    "RomaRotationViewStage",
    "ScoredSeeds",
    "SampledSeeds",
}


def slot_owners(slot: str) -> tuple[object, list[str]]:
    """The registry of a preset's slot and the configs it can be composed into: a stage of
    the seed-based skeleton under the five methods, a slot of another detector family
    (HGGD's ``image_encoder``) under that family. Post-processing presets, fragments of
    ``evaluation.variants``, go with the five methods as well."""
    if slot in SLOT_REGISTRIES:
        return SLOT_REGISTRIES[slot], list(METHODS.values())
    owners = {name: family_slots(name) for name in FAMILIES}
    registry = next(slots[slot] for slots in owners.values() if slot in slots)
    return registry, [FAMILIES[name] for name, slots in owners.items() if slot in slots]


#: Grouper presets that are alternatives to the crops of the ported methods, not a stage
#: of a shipped reference model (docs/slots/grouper.md lists them with their sources).
GROUPER_ALTERNATIVES = {
    "ball",
    "knn",
    "box",
    "cylinder-edgeconv",
    "cylinder-attentive",
    "cylinder-vector-attention",
    "cylinder-invres",
    "depth-ball",
    "depth-box",
    "depth-knn",
    "depth-cylinder-attentive",
    "roi-grid",
    "roi-aware",
    "voxel-set-abstraction",
    "multi-scale-linear",
    "depth-cylinder-projection",
    "depth-embed",
    "depth-pool",
    "pyg-pointnet-conv",
    "pyg-point-transformer-conv",
    "pyg-edge-conv",
    "pyg-gatv2-conv",
    "pyg-transformer-conv",
}


def every_preset():
    return [preset for members in presets().values() for preset in members.values()]


def in_tree(registry: str) -> list[str]:
    return [entry.name for entry in collect([registry], in_tree_only=True)]


@pytest.mark.parametrize("preset", every_preset(), ids=lambda p: f"{p.slot}/{p.name}")
def test_a_preset_says_what_it_is_and_what_to_retune(preset):
    assert preset.path.parent.parent == PRESET_ROOT and preset.path.suffix == ".yaml"
    assert preset.name == preset.name.lower() and " " not in preset.name
    assert preset.summary.endswith(".") and len(preset.summary) > 30
    assert preset.retune and all(len(bullet) > 20 for bullet in preset.retune)
    text = preset.path.read_text(encoding="utf-8")
    assert f"# {RETUNE_HEADING}" in text
    assert max(len(line) for line in text.splitlines()) <= 100
    # The component exists in the registry of the slot and takes every option that is set.
    registry, _ = slot_owners(preset.slot)
    params, _ = component_parameters(registry.get(preset.type))
    assert set(preset.node) - {"type"} <= set(params), preset.node
    # Every inline comment belongs to a line that exists.
    content = parse_yaml(text)
    for path in preset.comments:
        node = content
        for key in path:
            assert isinstance(node, dict) and key in node, (preset.name, path)
            node = node[key]
    assert set(preset.data) <= {"graspness", "compact_labels", "tolerance", "scale_statistic"}


def test_presets_are_found_by_slot_and_name_and_a_miss_names_candidates():
    assert find("backbone", "ptv3").type == "PTv3"
    assert find("backbone", "ptv3").location == "grasppanda/presets/backbone/ptv3.yaml"
    with pytest.raises(PresetError, match=r"Did you mean 'ptv3'"):
        find("backbone", "ptv33")
    with pytest.raises(PresetError, match=r"no preset 'ptv3' for slot 'head'"):
        find("head", "ptv3")


def test_a_malformed_preset_is_refused_with_the_reason(tmp_path):
    slot = tmp_path / "backbone"
    slot.mkdir()
    cases = {
        "no header": "model:\n  backbone: {type: PTv3}\n",
        "no notes": "# A preset.\nmodel:\n  backbone: {type: PTv3}\n",
        "two slots": (
            f"# A preset.\n#\n# {RETUNE_HEADING}\n# * x: because it matters here\n"
            "model:\n  backbone: {type: PTv3}\n  head: {type: SWADHead}\n"
        ),
        "other data": (
            f"# A preset.\n#\n# {RETUNE_HEADING}\n# * x: because it matters here\n"
            "model:\n  backbone: {type: PTv3}\ndata:\n  val: {batch_size: 1}\n"
        ),
        "not a component": (
            f"# A preset.\n#\n# {RETUNE_HEADING}\n# * x: because it matters here\n"
            "model:\n  backbone: {grid_size: 0.005}\n"
        ),
    }
    for name, text in cases.items():
        file = slot / "broken.yaml"
        file.write_text(text, encoding="utf-8")
        with pytest.raises(PresetError):
            read(file)
        assert name


def test_a_stage_preset_is_exactly_a_stage_of_a_shipped_reference_model():
    """Option for option, losses included. Composed under the method it comes from, a
    stage preset therefore gives that method's own config, with the same hash: the
    measured page recognises such a cell as the method's own component."""
    from .conftest import REPO

    models = {
        path.name: Config.load(path)["model"]
        for path in sorted((REPO / "configs" / "_base_" / "models").glob("*.yaml"))
    }
    owners: dict[str, list[str]] = {}
    for slot in STAGE_REGISTRIES:
        if slot == "backbone":
            continue
        for preset in presets().get(slot, {}).values():
            if preset.type in BEYOND_THE_REFERENCES:
                continue
            same = [name for name, model in models.items() if model.get(slot) == preset.node]
            if slot == "grouper" and preset.name in GROUPER_ALTERNATIVES:
                assert not same, f"{slot}/{preset.name} is a reference stage, not an alternative"
                continue
            if f"{slot}/{preset.name}" in OPTION_PRESETS:
                assert not same, f"{slot}/{preset.name} is a stage of {same}: no option"
                continue
            assert same, (
                f"{slot}/{preset.name} is no stage of a shipped reference model; list it in "
                "OPTION_PRESETS with what it is"
            )
            owners[f"{slot}/{preset.name}"] = same
    assert owners["head/swad"] == ["gsnet.yaml"]
    assert owners["view/template-view"] == ["economicgrasp.yaml", "finegrasp.yaml", "gsnet.yaml"]
    assert owners["assigner/scale-balanced"] == ["scale_balanced_grasp.yaml"]
    from grasppanda.combine.compose import compose

    for method, choice in (("GSNet", "head=swad"), ("FineGrasp", "grouper=multi-scale")):
        assert (
            compose(METHODS[method], [choice]).config_hash == Config.load(METHODS[method]).digest()
        )


def test_every_preset_passes_the_signature_check_under_a_method_that_fits():
    """Composing a preset never trips over an option name or a value type: whatever is
    refused is refused by a contract between stages or by the dataset, not by the preset."""
    from grasppanda.combine.compose import compose

    for preset in every_preset():
        registry, bases = slot_owners(preset.slot)
        if preset.slot == "backbone" and is_adapter(registry.get(preset.type)):
            continue
        if preset.slot == "loss":
            continue  # a loss goes into a term of a stage: the test below
        accepted = 0
        for base in bases:
            result = compose(base, [f"{preset.slot}={preset.name}"])
            accepted += result.accepted
            for issue in result.issues:
                assert not issue.path.startswith(f"model.{preset.slot}."), (preset.name, issue)
        assert accepted, f"{preset.slot}/{preset.name} is accepted under no method"


def test_every_loss_preset_is_composed_into_a_term_it_fits_or_says_why_not():
    """A loss preset is set as `<stage>.<loss option>=<preset>`; the cells of the grid
    name a term for every preset, and compose accepts it there."""
    from grasppanda.combine.compose import ComposeError, compose
    from grasppanda.combine.grid import LOSS_CELLS, LOSS_NOT_PROBED

    used: dict[str, set[str]] = {}
    for method, sets in LOSS_CELLS:
        for item in sets:
            used.setdefault(item.partition("=")[2], set()).add(method)
    names = set(presets()["loss"])
    assert set(used) | set(LOSS_NOT_PROBED) == names and not set(used) & set(LOSS_NOT_PROBED)
    for name, methods in used.items():  # under GSNet and under one more method
        assert "GSNet" in methods and len(methods) >= 2, (name, methods)
    refused = 0
    for method, sets in LOSS_CELLS:
        result = compose(METHODS[method], list(sets))
        assert not result.issues, (method, sets, result.issues)
        refused += not result.accepted
        if result.accepted:
            for item in sets:  # the loss sits where it was put, with a weight
                slot, option = item.partition("=")[0].split(".")
                node = result.body["model"][slot][option]
                assert node["type"] == presets()["loss"][item.partition("=")[2]].type
                assert "weight" in node or "weight" in node["loss"], (item, node)
    assert refused == 3
    # The weight of the replaced loss is kept, also inside a wrapper, and said so.
    result = compose(
        METHODS["GSNet"], ["head.score_loss=quality-focal", "view.loss=ohem-smooth-l1"]
    )
    assert result.body["model"]["head"] == {
        "score_loss": {"type": "QualityFocal", "beta": 2.0, "weight": 15.0}
    }
    assert result.body["model"]["view"]["loss"]["loss"]["weight"] == 100.0
    assert "kept from the SmoothL1 it replaces" in result.text()
    assert "--set head.score_loss=quality-focal --set view.loss=ohem-smooth-l1" in result.command
    # A loss of a stage that is set in the same request goes into that stage's node.
    result = compose(METHODS["GSNet"], ["head=decoupled-grid-bins", "head.angle_loss=poly1"])
    assert result.accepted and result.body["model"]["head"]["angle_loss"] == {
        "type": "Poly1CrossEntropy",
        "epsilon": 1.0,
        "weight": 1.0,
    }
    # A registered loss by its type, and an option of it by `--opts`.
    result = compose(
        METHODS["GSNet"], ["head.width_loss=Huber"], ["model.head.width_loss.delta=0.1"]
    )
    assert result.accepted
    assert result.body["model"]["head"]["width_loss"] == {
        "type": "Huber",
        "delta": 0.1,
        "weight": 10.0,
    }
    # The same type as the base config's: replaced, not merged.
    result = compose(METHODS["GSNet"], ["head.width_loss=SmoothL1"])
    assert result.body["model"]["head"]["width_loss"] == {
        "_delete_": True,
        "type": "SmoothL1",
        "weight": 10.0,
    }
    # A wrapper without a loss of its own wraps the loss that was there, with its weight.
    result = compose(METHODS["GSNet"], ["head.score_loss=uncertainty-weighting"])
    assert result.accepted and result.body["model"]["head"]["score_loss"]["loss"] == {
        "type": "SmoothL1",
        "weight": 15.0,
        "beta": 1.0,
    }
    # What cannot be set says what can.
    with pytest.raises(
        ComposeError, match=r"SWADHead has no loss option 'scor_loss'. Did you mean"
    ):
        compose(METHODS["GSNet"], ["head.scor_loss=quality-focal"])
    with pytest.raises(ComposeError, match=r"no preset 'qualityfocal' for slot 'loss'"):
        compose(METHODS["GSNet"], ["head.score_loss=qualityfocal"])
    with pytest.raises(ComposeError, match="HGGD has no such slot"):
        compose(FAMILIES["HGGD"], ["head.score_loss=quality-focal"])
    # A refusal names the term, its kind and the losses that fit.
    result = compose(METHODS["GraspNet-baseline"], ["head.score_loss=quality-focal"])
    (problem,) = result.problems
    assert problem.startswith("OperationHead.score_loss is a `value` term")
    assert "but QualityFocal fits `unit` terms" in problem and "SmoothL1" in problem


# ------------------------------------------------------------------------- the tripwire


def test_every_encoder_and_every_stage_has_a_preset_or_a_stated_reason():
    """Add a component and this fails until it has a preset (or a reason not to)."""
    members = presets()
    for name, found in encoder_presets().items():
        if name in in_tree("BACKBONES"):
            assert found, (
                f"scene encoder {name} has no preset: add grasppanda/presets/backbone/<name>.yaml "
                "with the settings for table-top depth clouds and what must be re-tuned"
            )
    adapters = [name for name in in_tree("BACKBONES") if is_adapter(BACKBONES.get(name))]
    assert adapters and all(
        any(preset.type == name for preset in members["backbone"].values()) for name in adapters
    )
    for slot, registry in STAGE_REGISTRIES.items():
        if slot == "backbone":
            continue
        for name in in_tree(registry.name):
            has_preset = any(preset.type == name for preset in members.get(slot, {}).values())
            assert has_preset or name in DEFAULTS_ARE_ENOUGH, (
                f"{registry.name}.{name} has neither a preset under grasppanda/presets/{slot}/ "
                "nor an entry in DEFAULTS_ARE_ENOUGH that says why its defaults suffice"
            )
            if not has_preset:  # "defaults are enough" must at least be constructible
                params, _ = component_parameters(registry.get(name))
                assert all(param.default is not param.empty for param in params.values()), name
    assert set(DEFAULTS_ARE_ENOUGH) <= {
        name
        for slot, registry in STAGE_REGISTRIES.items()
        for name in in_tree(registry.name)
        if slot != "backbone"
    }, "DEFAULTS_ARE_ENOUGH names a component that is no longer registered"


def test_the_variants_of_a_slot_are_its_presets_plus_unstated_defaults():
    proposals = stage_variants("proposal")
    assert ("graspness-fps", "GraspnessFPS") in proposals
    assert proposals[-1] == ("SubsampledSeeds", "SubsampledSeeds")
    # The proposals with a sampler slot: one preset per sampler and selection rule, and
    # one that states the defaults, so the bare component is no variant of its own.
    kinds = {kind for _, kind in proposals}
    assert kinds == {"GraspnessFPS", "SubsampledSeeds", "ScoredSeeds", "SampledSeeds"}
    assert {"scored-fps", "graspness-sfps", "graspness-ffps", "dense-fps", "dense-ffps"} <= {
        choice for choice, _ in proposals
    }
    assert not {"ScoredSeeds", "SampledSeeds"} & {choice for choice, _ in proposals}
    assigners = dict(stage_variants("assigner"))
    # A preset that states the constructor defaults stands for the bare component ...
    assert "CompactLabelAssigner" not in assigners and "ScaleBalancedAssigner" not in assigners
    # ... one that changes options leaves the defaults a variant of their own.
    assert assigners["NearestLabelAssigner"] == assigners["nearest-label-object-seeds"]
    # A loss written out is the loss the stage builds itself: still the defaults.
    reference_views = [
        variant for variant in stage_variants("view") if variant[1] not in BEYOND_THE_REFERENCES
    ]
    assert reference_views == [
        ("objectness-view-graspable", "ObjectnessViewStage"),
        ("objectness-view", "ObjectnessViewStage"),
        ("template-view", "TemplateViewStage"),
    ]
    # Every option beyond the references has a preset that states its defaults, so the
    # bare component is no variant of its own.
    assert not {choice for choice, _ in stage_variants("view")} & BEYOND_THE_REFERENCES
    heads = in_tree("HEADS")  # other test modules register heads of their own
    assert [choice for choice, kind in stage_variants("head") if kind in heads] == [
        "decoupled-grid-bins",
        "decoupled-grid-class",
        "decoupled-grid-regression",
        "decoupled-grid",
        "economic-focal",
        "economic",
        "operation-objectness",
        "operation",
        "swad-focal",
        "swad",
    ]
    # A component that cannot be built from its defaults stays a variant: it is refused,
    # and the refusal is the measurement.
    groupers = [choice for choice, _ in stage_variants("grouper")]
    assert {"multi-scale", "MultiScaleGroup", "DepthCylinderGroup"} <= set(groupers)


def test_a_composed_preset_cannot_smuggle_in_an_unknown_option(tmp_path, monkeypatch):
    """The config check sees a preset's options like any others."""
    from grasppanda.combine import presets as module

    root = tmp_path / "presets"
    (root / "backbone").mkdir(parents=True)
    (root / "backbone" / "typo.yaml").write_text(
        f"# A preset with a misspelt option.\n#\n# {RETUNE_HEADING}\n"
        "# * grid_sise: it should have been grid_size\n"
        "model:\n  backbone:\n    type: PTv3\n    grid_sise: 0.005\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PRESET_ROOT", root)
    module.presets.cache_clear()
    try:
        from grasppanda.combine.compose import compose

        result = compose(METHODS["GSNet"], ["backbone=typo"])
        assert not result.accepted
        (issue,) = result.issues
        assert (
            issue.path == "model.backbone.grid_sise" and "Did you mean 'grid_size'" in issue.message
        )
    finally:
        monkeypatch.undo()
        module.presets.cache_clear()
