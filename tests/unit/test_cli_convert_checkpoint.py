"""``python -m tools.convert_checkpoint``: the source format is named, never assumed.

Two methods release a file of the same name (``checkpoint-rs.tar``), so a default format
would convert one method's file with another's mapping. Every command of the model zoo
names its format, and every format names one method and the model it is converted for.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from grasppanda.combine.grid import FAMILIES, METHODS
from grasppanda.config import Config

from .conftest import REPO

ALL = {**METHODS, **FAMILIES}


@pytest.fixture(scope="module")
def converter():
    name = "convert_checkpoint"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _zoo_commands() -> dict[str, list[list[str]]]:
    """Method → the ``convert`` commands of its metafile, as words."""
    found: dict[str, list[list[str]]] = {}
    for method, config in ALL.items():
        meta = yaml.safe_load(
            (REPO / "configs" / config.split("/")[0] / "metafile.yaml").read_text()
        )

        def walk(node, into):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "convert" and isinstance(value, str):
                        into.append(shlex.split(value))
                    else:
                        walk(value, into)
            elif isinstance(node, list):
                for value in node:
                    walk(value, into)

        found[method] = []
        walk(meta, found[method])
    return found


def test_without_a_source_format_nothing_is_converted(converter, capsys):
    with pytest.raises(SystemExit) as stop:
        converter.build_parser().parse_args(["checkpoint-rs.tar", "out.pth"])
    assert stop.value.code == 2
    error = capsys.readouterr().err
    assert "--source-format" in error and "required" in error
    with pytest.raises(TypeError, match="source_format"):
        converter.convert_file(Path("checkpoint-rs.tar"))  # keyword-only, no default
    # Its help names every format with the method whose released file it reads.
    (option,) = [a for a in converter.build_parser()._actions if a.dest == "source_format"]
    assert option.required and option.default is None
    for name, method in converter.FORMAT_METHODS.items():
        assert f"{name} ({method})" in option.help


def test_every_format_names_one_method_and_the_model_of_its_config(converter):
    assert list(converter.FORMAT_METHODS) == list(converter.SOURCE_FORMATS)
    assert sorted(converter.FORMAT_METHODS.values()) == sorted(ALL)
    for name, method in converter.FORMAT_METHODS.items():
        model = Config.load(converter.SOURCE_FORMATS[name][1]).resolved()["model"]
        assert model == Config.load(ALL[method]).resolved()["model"], name


def test_every_command_of_the_model_zoo_and_of_the_docstring_names_its_format(converter):
    zoo = _zoo_commands()
    by_format = {}
    for method, commands in zoo.items():
        assert commands, f"{method}: no convert command in its metafile"
        for words in commands:
            assert words[:3] == ["python", "-m", "tools.convert_checkpoint"]
            args = converter.build_parser().parse_args(words[3:])
            assert converter.FORMAT_METHODS[args.source_format] == method, words
            by_format.setdefault(args.source_format, []).append(" ".join(words))
    # The docstring's commands are commands of the model zoo, one per format.
    text = converter.__doc__.replace("\\\n", " ")
    shown = [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip().startswith("python -m tools.convert_checkpoint")
    ]
    assert len(shown) == len(converter.SOURCE_FORMATS)
    for line in shown:
        args = converter.build_parser().parse_args(shlex.split(line)[3:])
        assert line in by_format[args.source_format], line
