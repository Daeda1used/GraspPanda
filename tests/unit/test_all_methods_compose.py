"""Compose under every method: what the toolbox promises for all six, checked for all six.

The toolbox starts from GraspNet-1Billion; GSNet is one of its six methods, not the
default. Whatever ``grasppanda compose`` does to a config (write a file that loads to the
config that was checked, add a post-processing chain, wrap a training recipe around the
base, list the options of a slot) is checked here under every method of the grid: the
five seed-based methods of :data:`~grasppanda.combine.grid.METHODS` and the detector
families of :data:`~grasppanda.combine.grid.FAMILIES` (HGGD). Methods, slots, options and
outcomes are read from the grid, the registries and the measured records, never typed in.
Compose only: nothing is built or trained.
"""

from __future__ import annotations

import pytest

from grasppanda.cli.components import collect
from grasppanda.cli.main import main
from grasppanda.combine import options as option_lists
from grasppanda.combine.compose import ComposeError, component_slots, compose, recompose
from grasppanda.combine.grid import FAMILIES, KNOWN_FAILURES, METHODS, encoder_presets, family_slots
from grasppanda.combine.post import chain_nodes
from grasppanda.combine.presets import presets
from grasppanda.combine.recipes import recipes
from grasppanda.config import Config
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, SeedGraspDetector
from grasppanda.registry import DETECTORS

#: Every method of the grid → its shipped config, in page order.
ALL = {**METHODS, **FAMILIES}
#: A post-processing chain of three steps, one of them with options.
CHAIN = "nms+top-k=20+workspace:z=[0.2,0.9]"


def detector_of(method: str) -> str:
    return Config.load(ALL[method])["model"]["type"]


def written(result, tmp_path, name: str = "composed.yaml"):
    """The composed config as the file a user gets: its path and the loaded config."""
    path = tmp_path / name
    path.write_text(result.text(path), encoding="utf-8")
    return path, Config.load(path)


def a_recorded_swap(method: str) -> option_lists.Option:
    """An option of the method's first component slot (the scene encoder of a seed-based
    method, HGGD's image encoder) whose measured record says ``runs``."""
    first = next(iter(component_slots(DETECTORS.get(detector_of(method)))))
    records = option_lists.measured().by_id
    for option in option_lists.options(method, first):
        if option.cell and records.get(option.cell, {}).get("verdict") == "runs":
            return option
    raise AssertionError(f"no option of {method}'s {first} slot has a record that runs")


# ------------------------------------------------------------------- the written file


@pytest.mark.parametrize("method", ALL)
def test_the_written_file_is_the_config_that_was_checked_and_probed(method, tmp_path):
    """One accepted swap per method: the file compose writes loads to the config that was
    checked, composes again to itself, and is the config whose record the pages show."""
    option = a_recorded_swap(method)
    result = compose(ALL[method], list(option.sets))
    assert result.accepted, (method, option.sets, result.refusal())
    path, config = written(result, tmp_path)
    assert config.digest() == result.config_hash == recompose(path).config_hash
    record = option_lists.measured().by_id[option.cell]
    assert record["config_hash"] == result.config_hash[:16], (
        f"{option.cell}: compose writes another config than the one that was probed"
    )
    node = config["model"][option.slot]
    inner = node.get("backbone") if isinstance(node.get("backbone"), dict) else {}
    assert option.component in (node["type"], inner.get("type")), node  # or in its adapter


@pytest.mark.parametrize("method", ALL)
def test_the_method_as_shipped_composes_to_itself(method):
    result = compose(ALL[method])
    assert result.accepted and result.config_hash == Config.load(ALL[method]).digest()
    assert option_lists.method_of(ALL[method]) == (method, "")


def registered_detectors() -> list[str]:
    return [entry.name for entry in collect(["DETECTORS"], in_tree_only=True)]


@pytest.mark.parametrize("detector", registered_detectors())
def test_every_detector_type_has_the_slots_of_its_family(detector):
    """A seed-based type has the six slots of the skeleton; a family of its own has its
    own slots, each with presets, and none of the skeleton's, which compose refuses with
    the list of the slots it does have."""
    cls = DETECTORS.get(detector)
    slots = list(component_slots(cls))
    types = {detector_of(method): method for method in ALL}
    assert detector in types, f"{detector} is no method of the grid"
    if issubclass(cls, SeedGraspDetector):
        assert set(slots) == set(STAGE_REGISTRIES) and types[detector] in METHODS
        return
    method = types[detector]
    assert method in FAMILIES and slots == list(family_slots(method))
    assert slots and not set(slots) & set(STAGE_REGISTRIES)
    assert all(presets().get(slot) for slot in slots), slots
    for slot in STAGE_REGISTRIES:
        with pytest.raises(ComposeError) as error:
            compose(ALL[method], [f"{slot}=anything"])
        assert f"no slot '{slot}'" in str(error.value)
        assert all(own in str(error.value) for own in slots), str(error.value)


# ------------------------------------------------------------------- post-processing


@pytest.mark.parametrize("method", ALL)
def test_a_post_processing_chain_is_the_variant_post_of_every_method(method, tmp_path):
    base_variants = list(Config.load(ALL[method])["evaluation"]["variants"])
    result = compose(ALL[method], [f"post={CHAIN}"])
    assert result.accepted, result.refusal()
    assert result.body["evaluation"]["variants"]["post"] == chain_nodes(CHAIN)
    assert f"--set 'post={CHAIN}'" in result.command
    path, config = written(result, tmp_path)
    assert list(config["evaluation"]["variants"]) == list(dict.fromkeys([*base_variants, "post"]))
    assert recompose(path).config_hash == result.config_hash


@pytest.mark.parametrize("method", ALL)
def test_a_step_that_reads_the_full_cloud_gets_it_under_every_method(method, tmp_path):
    """``collision`` reads the cloud before sub-sampling. Compose switches it on for
    validation and test where the base config drops it, and says so; nowhere else."""
    before = Config.load(ALL[method]).to_dict()["data"]
    kept = all(before[split]["dataset"].get("keep_full_points") for split in ("val", "test"))
    result = compose(ALL[method], ["post=collision+nms"])
    assert result.accepted, result.refusal()
    _, config = written(result, tmp_path)
    data = config.to_dict()["data"]
    assert data["val"]["dataset"]["keep_full_points"] is True
    assert data["test"]["dataset"]["keep_full_points"] is True
    switched = any("keep_full_points" in action for action in result.actions)
    assert switched == (not kept), (method, result.actions)


# ------------------------------------------------------------------- training recipe


@pytest.mark.parametrize("method", ALL)
def test_a_schedule_free_optimizer_drops_the_schedule_and_brings_its_hook(method, tmp_path):
    base = Config.load(ALL[method]).to_dict()
    result = compose(ALL[method], ["optimizer=schedulefree-adamw"])
    assert result.accepted, result.refusal()
    config = written(result, tmp_path)[1].to_dict()
    assert config["scheduler"] is None and config["optimizer"]["type"] == "ScheduleFree"
    assert config["hooks"] == [*(base.get("hooks") or []), {"type": "ScheduleFreeHook"}]
    removed = any("the base config's scheduler is removed" in a for a in result.actions)
    assert removed == (base.get("scheduler") is not None)


@pytest.mark.parametrize("method", ALL)
def test_hooks_are_added_after_those_of_the_base(method, tmp_path):
    base = Config.load(ALL[method]).to_dict().get("hooks") or []
    result = compose(ALL[method], ["hooks=ema+agc"])
    assert result.accepted, result.refusal()
    config = written(result, tmp_path)[1].to_dict()
    added = [node for name in ("ema", "agc") for node in recipes()["hooks"][name].body["hooks"]]
    assert config["hooks"] == [*base, *added]
    assert config["evaluation"]["weights"] == "ema" and config["train"]["grad_clip"] is None
    rewritten = any("hooks of the base config are written out again" in a for a in result.actions)
    assert rewritten == bool(base)
    assert recompose(written(result, tmp_path, "again.yaml")[0]).config_hash == result.config_hash


def accepted_augmentations(method: str) -> list[str]:
    return [
        name for name in recipes()["augment"] if compose(ALL[method], [f"augment={name}"]).accepted
    ]


@pytest.mark.parametrize("method", ALL)
def test_augmentation_goes_around_the_transforms_of_the_base(method, tmp_path):
    """Every augmentation preset is accepted under the seed-based methods; HGGD, which
    also reads images, takes only `dropout` (it moves no point). A preset's steps go
    before (`prepend`) or after (`append`) the base transforms, which are kept as they are."""
    names = accepted_augmentations(method)
    assert set(names) == (set(recipes()["augment"]) if method in METHODS else {"dropout"})
    bodies = [recipes()["augment"][name].body["transforms"] for name in names]
    prepend = [node for body in bodies for node in body.get("prepend", [])]
    append = [node for body in bodies for node in body.get("append", [])]
    base = Config.load(ALL[method]).to_dict()["data"]["train"]["dataset"]["transforms"]
    result = compose(ALL[method], [f"augment={'+'.join(names)}"])
    assert result.accepted, result.refusal()
    steps = written(result, tmp_path)[1].to_dict()["data"]["train"]["dataset"]["transforms"]
    failing = [f"recipe/{method}/augment/{name}" for name in names]
    failing = [cell for cell in failing if cell in KNOWN_FAILURES]
    if not failing:
        assert steps == [*prepend, *base, *append]
        return
    # A recorded failure of this order (FineGrasp computes its normals first): the steps are
    # all there, and the record still says why the order fails.
    assert sorted(map(str, steps)) == sorted(map(str, [*prepend, *base, *append]))
    assert all("transform order" in KNOWN_FAILURES[cell][0] for cell in failing), failing


# ------------------------------------------------------------------- scene encoders


@pytest.mark.parametrize("method", METHODS)
def test_every_scene_encoder_preset_composes_under_every_seed_based_method(method):
    """Compose fits the width and inserts the adapter an encoder needs: every preset of
    every registered scene encoder is accepted under every seed-based method."""
    refused = []
    for members in encoder_presets().values():
        for preset in members:
            result = compose(METHODS[method], [f"backbone={preset.name}"])
            if not result.accepted:
                refused.append((preset.name, result.refusal()[:1]))
    assert not refused, f"{method} refuses {len(refused)} encoder preset(s): {refused[:5]}"
    assert sum(len(members) for members in encoder_presets().values()) >= 40


# ------------------------------------------------------------------- the option lists


def listing_cases() -> list:
    return [
        pytest.param(method, slot, id=f"{method}-{slot}")
        for method in ALL
        for slot in option_lists.method_slots(method)
    ]


@pytest.mark.parametrize(("method", "slot"), listing_cases())
def test_compose_lists_the_options_of_every_slot_of_every_method(method, slot, capsys):
    assert main(["compose", ALL[method], "--list", slot]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert method in first and slot in first, first


def test_every_method_lists_options_for_the_slots_its_family_has():
    """The slots with options: the component slots of the detector, the losses of the
    seed-based skeleton, and post-processing and the training recipe for all six."""
    for method in ALL:
        slots = set(option_lists.method_slots(method))
        own = set(component_slots(DETECTORS.get(detector_of(method))))
        assert own <= slots, (method, own - slots)
        assert {"post", *recipes()} <= slots, method
        assert ("loss" in slots) == (method in METHODS), method
