"""What the command line says about the methods is computed or checked, and one method is
never the implied default.

The help texts and module examples of the commands rotate over the six methods; the help
of ``grasppanda prepare`` names exactly the methods whose configs read each label asset;
the messages of ``grasppanda explore`` and the frame keys of ``grasppanda demo`` are those
of no method and of both datasets.
"""

from __future__ import annotations

import argparse
import importlib
import re
from collections import Counter

import pytest

from grasppanda.apis.inferencer import parse_frame_key
from grasppanda.cli.main import main
from grasppanda.combine.grid import FAMILIES, METHODS
from grasppanda.config import Config

ALL = {**METHODS, **FAMILIES}
#: The commands of ``grasppanda``.
COMMANDS = (
    "train",
    "test",
    "infer",
    "demo",
    "analyze",
    "sweep",
    "print-config",
    "components",
    "compose",
    "probe",
    "explore",
    "prepare",
    "env",
)
#: Modules whose docstring examples a user reads (``help()``, the API pages).
MODULES = (
    *(f"grasppanda.cli.{name.replace('-', '_')}" for name in COMMANDS),
    "grasppanda.sweep",
    "grasppanda.apis.inferencer",
    "grasppanda.combine.compose",
    "grasppanda.combine.probe",
)
#: ``grasppanda prepare`` option → the section of a training dataset that reads its asset.
ASSETS = {
    "--graspness": "graspness",
    "--compact-labels": "compact_labels",
    "--tolerance": "tolerance",
    "--scale-statistic": "scale_statistic",
    "--view-grasps": "view_grasps",
}


def _help(command: str, capsys) -> str:
    with pytest.raises(SystemExit) as stop:
        main([command, "--help"])
    assert stop.value.code == 0
    return capsys.readouterr().out


def _named(text: str) -> Counter:
    """How often a text names the config folder of each method (``economicgrasp/...``)."""
    found = Counter()
    for method, config in ALL.items():
        folder = config.split("/")[0]
        found[method] += len(re.findall(rf"(?<![\w-]){re.escape(folder)}/", text))
        found[method] += len(
            re.findall(rf"experimental/{re.escape(folder.replace('_', '-'))}_", text)
        )
    return found


def test_the_examples_rotate_over_the_six_methods_and_none_is_the_default(capsys):
    """Every method is named by some example; a help that names a method config names
    more than one; GSNet is named no more often than any other method."""
    total = Counter()
    for command in COMMANDS:
        named = _named(_help(command, capsys))
        assert len(+named) != 1, (command, +named)
        total += named
    for module in MODULES:
        total += _named(importlib.import_module(module).__doc__ or "")
    assert set(+total) == set(ALL), total
    assert total["GSNet"] <= min(total.values()), total


def _prepare_options() -> dict[str, str]:
    from grasppanda.cli import prepare

    commands = argparse.ArgumentParser().add_subparsers()
    prepare.register(commands)
    parser = commands.choices["prepare"]
    return {
        action.option_strings[0]: action.help or ""
        for action in parser._actions
        if action.option_strings
    }


def test_prepare_names_its_default_dataset_config_and_the_methods_that_train_with_it():
    """Without --config, prepare reads ``_base_/datasets/<dataset>_<camera>.yaml``; the help
    and the module say so, and name exactly the methods whose config inherits that file."""
    from grasppanda.cli import prepare
    from grasppanda.config import configs_root, parse_yaml

    default = prepare._dataset_class("graspnet1b").default_config(None)
    assert default == "_base_/datasets/graspnet1b_realsense.yaml"
    inherits = {
        method
        for method, config in ALL.items()
        if f"../{default}" in parse_yaml((configs_root() / f"{config}.yaml").read_text())["_base_"]
    }
    assert inherits == {"GSNet", "EconomicGrasp", "FineGrasp"}
    for text in (_prepare_options()["--config"], " ".join(prepare.__doc__.split())):
        assert "_base_/datasets/<dataset>_<camera>.yaml" in text.replace("``", "")
        assert {method for method in ALL if method in text} == inherits, text
    assert "graspnet1b_realsense.yaml" in prepare.__doc__


def test_prepare_names_exactly_the_methods_whose_configs_read_each_asset():
    helps = _prepare_options()
    for option, section in ASSETS.items():
        readers = {
            method
            for method, config in ALL.items()
            if Config.load(config).resolved()["data"]["train"]["dataset"].get(section) is not None
        }
        named = {method for method in ALL if method in helps[option]}
        assert readers and named == readers, (option, named, readers)
    assert "graspnet1b or graspclutter6d" in _prepare_dataset_help()


def _prepare_dataset_help() -> str:
    from grasppanda.cli import prepare

    commands = argparse.ArgumentParser().add_subparsers()
    prepare.register(commands)
    parser = commands.choices["prepare"]
    (dataset,) = [action for action in parser._actions if action.dest == "dataset"]
    return dataset.help


def test_the_frame_keys_of_demo_name_both_datasets_and_parse(capsys):
    text = " ".join(_help("demo", capsys).split())
    keys = re.findall(r"(?:graspnet1b|graspclutter6d):scene_[\w./-]+", text)
    assert {key.split(":")[0] for key in keys} == {"graspnet1b", "graspclutter6d"}
    for key in keys:
        parsed = parse_frame_key(key.rstrip("."))
        assert parsed.dataset in key and parsed.scene >= 0


def test_the_weights_messages_of_explore_name_no_method_as_the_default():
    from grasppanda.combine.explore_run import WeightsSource

    for wrong in (["not a name=w.pth"], ["=w.pth"]):
        with pytest.raises(ValueError) as error:
            WeightsSource.from_arguments(None, wrong)
        message = str(error.value)
        assert "for example" in message and "gsnet" not in message.lower(), message


def test_the_weights_hints_of_explore_are_the_names_the_convert_commands_write():
    """The words that suggest a method for a weights file are those of the files the
    ``convert`` commands of the metafiles write: each such file is suggested for its own
    method and for no other, and no method has a word that is only a guess."""
    from grasppanda.combine.explore_run import _HINTS
    from grasppanda.config import configs_root

    assert list(_HINTS) == list(ALL)
    for method, config in ALL.items():
        metafile = (configs_root() / config.split("/")[0] / "metafile.yaml").read_text()
        written = re.findall(r"convert_checkpoint \S+ (\S+\.pth)", metafile)
        assert written, method
        for name in written:
            suggested = {other for other, words in _HINTS.items() if any(w in name for w in words)}
            assert suggested == {method}, (name, suggested)
        for word in _HINTS[method]:
            assert any(word in name for name in written) or word.replace("_", "-") in method.lower()
