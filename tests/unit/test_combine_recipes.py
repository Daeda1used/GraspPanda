"""Recipe presets and ``grasppanda compose --set optimizer|schedule|hooks|augment|recipe=...``.

Every shipped file is well formed and names registered components with options they take;
compose writes them into a child config of any detector, HGGD included, keeping the hooks
and the point transforms of the base config.
"""

from __future__ import annotations

import pytest

from grasppanda.cli.main import main
from grasppanda.combine.compose import ComposeError, compose, recompose
from grasppanda.combine.grid import RECIPE_ENCODER_PRESETS, RECIPE_REQUIRES, cells
from grasppanda.combine.presets import RETUNE_HEADING, PresetError
from grasppanda.combine.recipes import RECIPE_ROOT, RECIPE_SLOTS, find, read, recipes, split
from grasppanda.config import Config
from grasppanda.registry import HOOKS, OPTIMIZERS, SCHEDULERS, TRANSFORMS, component_parameters

GSNET = "gsnet/gsnet_sunet14_g1b-realsense_10e"
HGGD = "hggd/hggd_r34w8_g1b-realsense_15e"


def every_recipe():
    return [preset for members in recipes().values() for preset in members.values()]


def loaded(result, tmp_path) -> Config:
    path = tmp_path / "composed.yaml"
    path.write_text(result.text(path), encoding="utf-8")
    return Config.load(path)


def _takes(registry, node):
    params, _ = component_parameters(registry.get(node["type"]))
    assert set(node) - {"type"} <= set(params), node


@pytest.mark.parametrize("preset", every_recipe(), ids=lambda p: f"{p.slot}/{p.name}")
def test_a_recipe_preset_is_well_formed_and_names_what_exists(preset):
    assert preset.path.parent.parent == RECIPE_ROOT and preset.slot in RECIPE_SLOTS
    assert preset.name == preset.name.lower() and " " not in preset.name
    assert preset.summary.endswith(".") and len(preset.summary) > 30
    assert preset.retune and all(len(bullet) > 20 for bullet in preset.retune)
    text = preset.path.read_text(encoding="utf-8")
    assert f"# {RETUNE_HEADING}" in text
    assert max(len(line) for line in text.splitlines()) <= 100
    assert set(preset.body) <= RECIPE_SLOTS[preset.slot]
    if isinstance(preset.body.get("optimizer"), dict):
        _takes(OPTIMIZERS, preset.body["optimizer"])
        inner = preset.body["optimizer"].get("optimizer")
        if inner is not None:
            assert inner["type"] in OPTIMIZERS and "lr" in inner
    if isinstance(preset.body.get("scheduler"), dict):
        _takes(SCHEDULERS, preset.body["scheduler"])
    for node in preset.body.get("hooks", []):
        _takes(HOOKS, node)
    for nodes in preset.body.get("transforms", {}).values():
        for node in nodes:
            _takes(TRANSFORMS, node)
    assert preset.type


def test_recipe_presets_cover_every_slot_and_are_found_by_name():
    found = recipes()
    assert set(found) == set(RECIPE_SLOTS) and all(found.values())
    assert find("optimizer", "timm-lamb").body["optimizer"]["name"] == "lamb"
    assert find("hooks", "ema").location == "grasppanda/recipes/hooks/ema.yaml"
    with pytest.raises(PresetError, match=r"Did you mean 'onecycle'"):
        find("schedule", "onecycl")
    assert split(["backbone=ptv3", "hooks=ema+amp", "optimizer=adamw"]) == (
        ["backbone=ptv3"],
        [("hooks", "ema+amp"), ("optimizer", "adamw")],
    )
    with pytest.raises(PresetError, match="set twice: join the names with `\\+`"):
        split(["hooks=ema", "hooks=amp"])


def test_a_malformed_recipe_preset_is_refused(tmp_path):
    slot = tmp_path / "schedule"
    slot.mkdir()
    head = f"# A schedule preset for tests.\n#\n# {RETUNE_HEADING}\n# * x: because it matters\n"
    cases = {
        "no header": "scheduler: {type: StepLR, step_size: 1}\n",
        "foreign key": head + "optimizer: {type: Adam, lr: 1.0}\n",
        "no type": head + "scheduler: {step_size: 1}\n",
        "empty": head,
    }
    for name, text in cases.items():
        file = slot / "broken.yaml"
        file.write_text(text, encoding="utf-8")
        with pytest.raises(PresetError):
            read(file)
        assert name
    good = slot / "good.yaml"
    good.write_text(head + "scheduler: null\n", encoding="utf-8")
    assert read(good).body == {"scheduler": None}


def test_optimizer_and_schedule_replace_those_of_the_base(tmp_path):
    result = compose(GSNET, ["optimizer=timm-lamb", "schedule=onecycle"])
    assert result.accepted, result.refusal()
    assert result.command == (
        f"grasppanda compose {GSNET} --set optimizer=timm-lamb --set schedule=onecycle"
    )
    config = loaded(result, tmp_path).to_dict()
    assert config["optimizer"] == {
        "type": "TimmOptimizer",
        "name": "lamb",
        "lr": 2.0e-3,
        "weight_decay": 0.02,
    }
    assert config["scheduler"]["type"] == "OneCycleLR" and "gamma" not in config["scheduler"]
    assert config["train"]["scheduler_unit"] == "iteration"
    assert config["train"]["epochs"] == 10  # everything else is the base config's
    text = result.text()
    assert "optimizer `timm-lamb`, schedule `onecycle`" in text
    assert "(recipe presets: grasppanda/recipes/<slot>/<preset>.yaml)" in text
    assert "# the peak; replaces optimizer.lr" in text
    # The same optimizer type as the base: its options are not inherited.
    same = compose(HGGD, ["optimizer=adamw"])
    assert loaded(same, tmp_path).to_dict()["optimizer"] == {
        "type": "AdamW",
        "lr": 1.0e-3,
        "weight_decay": 0.05,
    }


def test_a_schedule_free_optimizer_removes_the_schedule_and_brings_its_hook(tmp_path):
    result = compose(GSNET, ["optimizer=schedulefree-adamw"])
    assert result.accepted, result.refusal()
    config = loaded(result, tmp_path).to_dict()
    assert config["scheduler"] is None and config["optimizer"]["type"] == "ScheduleFree"
    assert config["hooks"] == [{"type": "ScheduleFreeHook"}]
    assert any("the base config's scheduler is removed" in action for action in result.actions)


def test_hooks_are_added_to_those_of_the_base_also_for_a_detector_without_slots(tmp_path):
    base = Config.load(HGGD).to_dict()["hooks"]
    result = compose(HGGD, ["hooks=ema+agc"])
    assert result.accepted, result.refusal()
    config = loaded(result, tmp_path).to_dict()
    assert config["hooks"][: len(base)] == base
    assert [node["type"] for node in config["hooks"][len(base) :]] == [
        "EMAHook",
        "AdaptiveGradClipHook",
    ]
    assert config["evaluation"]["weights"] == "ema" and config["train"]["grad_clip"] is None
    assert any("hooks of the base config are written out again" in a for a in result.actions)
    again = recompose_from(result, tmp_path)
    assert again.config_hash == result.config_hash


def recompose_from(result, tmp_path):
    path = tmp_path / "again.yaml"
    path.write_text(result.text(path), encoding="utf-8")
    return recompose(path)


def test_augmentation_goes_around_the_transforms_of_the_base(tmp_path):
    base = Config.load(GSNET).to_dict()["data"]["train"]["dataset"]["transforms"]
    result = compose(GSNET, ["augment=dropout+jitter+shift"])
    assert result.accepted, result.refusal()
    steps = loaded(result, tmp_path).to_dict()["data"]["train"]["dataset"]["transforms"]
    assert [step["type"] for step in steps] == [
        "RandomDropout",
        *(step["type"] for step in base),
        "RandomJitter",
        "RandomShift",
    ]
    assert steps[1 : 1 + len(base)] == base


def test_a_whole_recipe_and_stage_presets_combine_and_overrides_win(tmp_path):
    result = compose(
        GSNET,
        ["backbone=ptv3", "recipe=ptv3-adamw-onecycle", "hooks=amp"],
        ["optimizer.optimizer.lr=2e-3", "scheduler.max_lr=2e-3"],
    )
    assert result.accepted, result.refusal()
    config = loaded(result, tmp_path).to_dict()
    assert config["model"]["backbone"]["type"] == "PTv3"
    assert config["optimizer"]["optimizer"]["lr"] == 2e-3
    assert config["optimizer"]["lr_multipliers"] == {"re:\\.block": 0.1}
    assert config["scheduler"]["max_lr"] == 2e-3 and config["train"]["amp"] is True


def test_requests_that_cannot_be_composed_say_why():
    with pytest.raises(ComposeError, match="no recipe preset 'lamb' for slot 'optimizer'"):
        compose(GSNET, ["optimizer=lamb"])
    with pytest.raises(ComposeError, match="slot 'optimizer' is set twice"):
        compose(GSNET, ["optimizer=adamw", "optimizer=timm-lamb"])


def test_the_command_lists_and_shows_recipe_presets(capsys):
    assert main(["compose", "--list", "schedule"]) == 0
    out = capsys.readouterr().out
    assert "--set schedule=<preset>" in out and "warmup-cosine" in out and "TimmSchedule" in out
    assert main(["compose", "--list"]) == 0
    out = capsys.readouterr().out
    assert "--set hooks=<preset> (several: a+b)" in out and "--set backbone=" in out
    assert main(["compose", "--show", "hooks=swa"]) == 0
    assert "# grasppanda/recipes/hooks/swa.yaml" in capsys.readouterr().out


def test_every_recipe_preset_composes_under_the_methods_it_is_probed_under():
    members = cells(("recipe",))
    assert {cell.id.split("/", 2)[2] for cell in members} == {
        f"{preset.slot}/{preset.name}" for preset in every_recipe()
    }
    assert set(RECIPE_ENCODER_PRESETS) | set(RECIPE_REQUIRES) <= {
        f"{preset.slot}/{preset.name}" for preset in every_recipe()
    }
    for cell in members:
        result = compose(cell.base, list(cell.sets))
        assert result.accepted, (cell.id, result.refusal())


@pytest.mark.parametrize(
    "name", ["adamw_onecycle_10e", "adamw_warmup-cosine_10e", "schedulefree-adamw_10e"]
)
def test_the_method_neutral_schedule_bases_are_the_recipe_presets(name):
    from .conftest import REPO

    base = Config.load(REPO / "configs" / "_base_" / "schedules" / f"{name}.yaml").to_dict()
    assert base["train"]["epochs"] == 10
    preset = {
        "adamw_onecycle_10e": find("recipe", "adamw-onecycle").body,
        "adamw_warmup-cosine_10e": {
            **find("optimizer", "adamw").body,
            **find("schedule", "warmup-cosine").body,
        },
        "schedulefree-adamw_10e": find("optimizer", "schedulefree-adamw").body,
    }[name]
    assert base["optimizer"] == preset["optimizer"]
    assert base.get("scheduler") == preset.get("scheduler")
    assert base.get("hooks", []) == preset.get("hooks", [])
