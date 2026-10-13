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
from grasppanda.combine.grid import METHODS, encoder_presets, stage_variants
from grasppanda.combine.presets import PRESET_ROOT, RETUNE_HEADING, PresetError, find, presets, read
from grasppanda.config import Config, parse_yaml
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES
from grasppanda.registry import BACKBONES, component_parameters

#: Stages that need no preset, each with the reason. The constructor defaults of every
#: one are the reference values of the method named, and the stage reads no label asset
#: beyond what its own method's dataset config already serves.
DEFAULTS_ARE_ENOUGH = {
    "SubsampledSeeds": "one option, the seed count, which the encoder's subsample fixes",
    "CylinderGroup": "defaults are GSNet's crop; no loss, no label asset",
    "CylinderInteractionGroup": "defaults are EconomicGrasp's crop; no loss, no label asset",
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
    registry = STAGE_REGISTRIES[preset.slot]
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
            same = [name for name, model in models.items() if model.get(slot) == preset.node]
            assert same, f"{slot}/{preset.name} is no stage of a shipped reference model"
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
        if is_adapter(STAGE_REGISTRIES[preset.slot].get(preset.type)) and preset.slot == "backbone":
            continue
        accepted = 0
        for base in METHODS.values():
            result = compose(base, [f"{preset.slot}={preset.name}"])
            accepted += result.accepted
            for issue in result.issues:
                assert not issue.path.startswith(f"model.{preset.slot}."), (preset.name, issue)
        assert accepted, f"{preset.slot}/{preset.name} is accepted under no method"


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
    assert stage_variants("proposal") == [
        ("graspness-fps", "GraspnessFPS"),
        ("SubsampledSeeds", "SubsampledSeeds"),
    ]
    assigners = dict(stage_variants("assigner"))
    # A preset that states the constructor defaults stands for the bare component ...
    assert "CompactLabelAssigner" not in assigners and "ScaleBalancedAssigner" not in assigners
    # ... one that changes options leaves the defaults a variant of their own.
    assert assigners["NearestLabelAssigner"] == assigners["nearest-label-object-seeds"]
    # A loss written out is the loss the stage builds itself: still the defaults.
    assert stage_variants("view") == [
        ("objectness-view-graspable", "ObjectnessViewStage"),
        ("objectness-view", "ObjectnessViewStage"),
        ("template-view", "TemplateViewStage"),
    ]
    heads = in_tree("HEADS")  # other test modules register heads of their own
    assert [choice for choice, kind in stage_variants("head") if kind in heads] == [
        "economic",
        "operation-objectness",
        "operation",
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
