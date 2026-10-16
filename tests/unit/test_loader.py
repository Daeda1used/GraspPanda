"""Inheritance, merge rules, overrides and interpolation."""

from __future__ import annotations

import pytest

from grasppanda.config import Config, ConfigError, configs_root
from grasppanda.config.loader import CLI_SOURCE, apply_overrides, load_tree, parse_yaml


def load(path, *opts):
    return Config.load(path, opts, validate=False)


def test_scientific_notation_is_a_float():
    assert parse_yaml("lr: 1e-3")["lr"] == pytest.approx(0.001)
    assert isinstance(parse_yaml("lr: 1e-3")["lr"], float)
    assert parse_yaml("n: 10")["n"] == 10 and isinstance(parse_yaml("n: 10")["n"], int)


def test_duplicate_keys_are_rejected(write):
    path = write("a.yaml", "train: {epochs: 1}\ntrain: {epochs: 2}\n")
    with pytest.raises(ConfigError, match="duplicate key 'train'"):
        load(path)


def test_child_overrides_base_and_lists_replace(write):
    write("base.yaml", "train: {epochs: 10, amp: false}\nhooks: [a, b]\n")
    path = write("child.yaml", "_base_: base.yaml\ntrain: {epochs: 20}\nhooks: [c]\n")
    config = load(path)
    assert config["train"] == {"epochs": 20, "amp": False}
    assert config["hooks"] == ["c"]
    assert config.source_of("train.epochs").endswith("child.yaml")
    assert config.source_of("train.amp").endswith("base.yaml")


def test_changed_type_replaces_the_node(write):
    write("base.yaml", "optimizer: {type: Adam, lr: 0.001, betas: [0.9, 0.99]}\n")
    path = write("child.yaml", "_base_: base.yaml\noptimizer: {type: SGD, momentum: 0.9}\n")
    config = load(path)
    assert config["optimizer"] == {"type": "SGD", "momentum": 0.9}
    (drop,) = config.dropped
    assert (drop.path, drop.old_type, drop.new_type, drop.keys) == (
        "optimizer",
        "Adam",
        "SGD",
        ("betas", "lr"),
    )


def test_same_type_merges(write):
    write("base.yaml", "optimizer: {type: Adam, lr: 0.001, eps: 1.0e-8}\n")
    path = write("child.yaml", "_base_: base.yaml\noptimizer: {type: Adam, lr: 0.01}\n")
    config = load(path)
    assert config["optimizer"] == {"type": "Adam", "lr": 0.01, "eps": 1e-8}
    assert not config.dropped


def test_delete_replaces_an_untyped_dict(write):
    write("base.yaml", "runtime: {seed: 1, deterministic: true}\n")
    path = write("child.yaml", "_base_: base.yaml\nruntime: {_delete_: true, seed: 2}\n")
    assert load(path)["runtime"] == {"seed": 2}


def test_inheritance_depth_is_limited_to_two(write):
    write("c.yaml", "train: {epochs: 1}\n")
    write("b.yaml", "_base_: c.yaml\n")
    write("a.yaml", "_base_: b.yaml\n")
    assert load(write("ok.yaml", "_base_: b.yaml\n"))["train"] == {"epochs": 1}
    with pytest.raises(ConfigError, match="limited to 2 levels"):
        load(write("deep.yaml", "_base_: a.yaml\n"))


def test_sibling_bases_may_not_define_the_same_value(write):
    write("one.yaml", "train: {epochs: 1}\nruntime: {seed: 0}\n")
    write("two.yaml", "train: {epochs: 2}\nruntime: {deterministic: true}\n")
    path = write("child.yaml", "_base_: [one.yaml, two.yaml]\n")
    with pytest.raises(ConfigError) as error:
        load(path)
    (issue,) = error.value.issues
    assert issue.path == "train.epochs"
    assert "one.yaml" in issue.message and "two.yaml" in issue.message


def test_sibling_bases_combine_disjoint_values(write):
    write("one.yaml", "runtime: {seed: 0}\n")
    write("two.yaml", "runtime: {deterministic: true}\n")
    config = load(write("child.yaml", "_base_: [one.yaml, two.yaml]\n"))
    assert config["runtime"] == {"seed": 0, "deterministic": True}


GSNET = "gsnet/gsnet_sunet14_g1b-realsense_10e"


def test_a_base_may_name_a_shipped_building_block_from_any_directory(write, monkeypatch):
    """Outside the checkout (or from a wheel) ``_base_/...`` falls back to the shipped tree."""
    monkeypatch.chdir(write("elsewhere/keep.yaml", "").parent)
    path = write("mine/child.yaml", "_base_: [_base_/default_runtime.yaml]\ntrain: {epochs: 3}\n")
    config = load(path)
    shipped = (configs_root() / "_base_" / "default_runtime.yaml").resolve()
    assert config.sources == [path.resolve(), shipped]
    assert config.source_of("runtime.seed") == str(shipped)
    assert config["train"]["epochs"] == 3


@pytest.mark.parametrize("name", [f"{GSNET}.yaml", GSNET])
def test_a_base_may_name_a_shipped_config(write, monkeypatch, name):
    """A child of a shipped method config is the same experiment when it changes nothing."""
    monkeypatch.chdir(write("elsewhere/keep.yaml", "").parent)
    child = Config.load(write("mine/child.yaml", f"_base_: [{name}]\n"))
    shipped = Config.load(configs_root() / f"{GSNET}.yaml")
    assert child.digest() == shipped.digest()
    assert child.sources[1] == (configs_root() / f"{GSNET}.yaml").resolve()
    assert len(child.sources) == 1 + len(shipped.sources)


def test_a_file_next_to_the_child_comes_before_a_shipped_one(write):
    local = write("mine/_base_/default_runtime.yaml", "runtime: {seed: 41}\n")
    config = load(write("mine/child.yaml", "_base_: [_base_/default_runtime.yaml]\n"))
    assert config.sources[1] == local.resolve() and config["runtime"] == {"seed": 41}


def test_a_missing_base_is_reported_at_the_file_that_names_it(write):
    write("mine/middle.yaml", "_base_: [no/such/base.yaml]\n")
    child = write("mine/child.yaml", "_base_: [middle.yaml]\n")
    with pytest.raises(ConfigError) as error:
        load(child)
    (issue,) = error.value.issues
    assert issue.path == "_base_" and issue.source == str((child.parent / "middle.yaml").resolve())
    assert "cannot find the base 'no/such/base.yaml'" in issue.message
    assert str(configs_root()) in issue.message and "_base_/default_runtime.yaml" in issue.message


def test_circular_inheritance_is_reported(write):
    write("a.yaml", "_base_: b.yaml\n")
    path = write("b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ConfigError, match="circular inheritance"):
        load(path)


def test_overrides_use_yaml_values_and_record_their_source(write):
    path = write("a.yaml", "train: {epochs: 10}\nhooks: [{type: A, interval: 1}]\n")
    config = load(
        path,
        "train.epochs=20",
        "train.amp=true",
        "runtime.seed=3",
        "hooks.0.interval=5",
        "tags=[x, y]",
    )
    assert config["train"] == {"epochs": 20, "amp": True}
    assert config["runtime"] == {"seed": 3}
    assert config["hooks"][0]["interval"] == 5
    assert config["tags"] == ["x", "y"]
    assert config.source_of("train.epochs") == CLI_SOURCE


def test_overriding_type_drops_stale_keys(write):
    path = write("a.yaml", "optimizer: {type: Adam, lr: 0.001}\n")
    config = load(path, "optimizer.type=SGD")
    assert config["optimizer"] == {"type": "SGD"}
    assert config.dropped[0].keys == ("lr",)


def test_override_with_a_mapping_follows_the_merge_rule(write):
    path = write("a.yaml", "optimizer: {type: Adam, lr: 0.001}\n")
    assert load(path, "optimizer={type: Adam, eps: 1e-6}")["optimizer"] == {
        "type": "Adam",
        "lr": 0.001,
        "eps": 1e-6,
    }
    assert load(path, "optimizer={type: SGD}")["optimizer"] == {"type": "SGD"}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("train.epochs", "must look like"),
        ("hooks.3.x=1", "not a valid index"),
        ("train.epochs.x=1", "not a mapping or list"),
    ],
)
def test_malformed_overrides(write, override, message):
    tree = load_tree(write("a.yaml", "train: {epochs: 10}\nhooks: [1]\n"))
    with pytest.raises(ConfigError, match=message):
        apply_overrides(tree, [override])


def test_interpolation_keeps_types_and_follows_overrides(write):
    path = write(
        "exp.yaml",
        """
        model: {head: {num_angle: 12}, decoder: {num_angle: "${model.head.num_angle}"}}
        runtime: {work_dir: "work_dirs/${config_name}/a${model.head.num_angle}"}
        """,
    )
    config = load(path, "model.head.num_angle=6")
    assert config["model"]["decoder"]["num_angle"] == 6
    assert config["runtime"]["work_dir"] == "work_dirs/exp/a6"


def test_interpolation_errors(write):
    with pytest.raises(ConfigError, match=r"Did you mean '\$\{train.epochs\}'"):
        load(write("a.yaml", "train: {epochs: 1}\nx: ${train.epoch}\n"))
    with pytest.raises(ConfigError, match="circular interpolation"):
        load(write("b.yaml", "a: ${b}\nb: ${a}\n"))
