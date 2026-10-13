"""The animated picture of the README shows real output, and is what its tool writes."""

from __future__ import annotations

import html
import importlib.util
import json
import re
import sys

import pytest

from .conftest import REPO

RESOURCES = REPO / "resources"
PICTURES = ("showcase-light.svg", "showcase-dark.svg")


@pytest.fixture(scope="module")
def tool():
    name = "make_showcase"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_pictures_are_what_the_tool_writes(tool):
    assert not tool.stale(), "stale: run `python -m tools.make_showcase`."
    assert tool.main(["--check"]) == 0


def test_every_line_of_the_terminal_is_a_line_of_the_transcript(tool):
    record = json.loads((RESOURCES / "showcase_transcript.json").read_text(encoding="utf-8"))
    assert set(record["commands"]) == set(tool.SCENES) == set(tool.COMMANDS)
    picture = (RESOURCES / PICTURES[0]).read_text(encoding="utf-8")
    drawn = html.unescape(picture).replace(" ", " ")
    for name, scene in tool.SCENES.items():
        command = record["commands"][name]
        assert command["command"] == "grasppanda " + " ".join(tool.COMMANDS[name])
        lines = tool.shown(name, command)
        assert len(lines) == len(scene["lines"])
        for line in lines:
            assert line in command["output"], f"{name}: not in the transcript: {line!r}"
        # What is drawn is those lines, whole: a long one is wrapped at a word, so the
        # words of the picture and their order are the words of the transcript.
        groups = tool.paragraphs(name, command)
        assert [line for group in groups for line in group] == lines
        for group in groups:
            wrapped = tool.wrap(group)
            assert " ".join(" ".join(wrapped).split()) == " ".join(" ".join(group).split())
            assert all(len(row) <= tool.COLUMNS for row in wrapped)
            for row in wrapped:
                assert f">{row}</text>" in drawn, f"{name}: not drawn: {row!r}"
        typed = tool._typed(command["command"])
        assert " ".join(" ".join(typed).replace("\\", " ").split()) == command["command"]
        assert all(len(row) + (4 if at else 2) <= tool.COLUMNS for at, row in enumerate(typed))
    # The outcome each scene stands for is the outcome of the command.
    assert record["commands"]["compose"]["exit_status"] == 0
    assert record["commands"]["refuse"]["exit_status"] == 1
    assert record["commands"]["probe"]["exit_status"] == 0
    assert any(line.startswith("✓ runs") for line in record["commands"]["probe"]["output"])
    with pytest.raises(ValueError, match="no line of the transcript"):
        tool.SCENES["probe"]["lines"].append(r"^a line nobody printed$")
        try:
            tool.shown("probe", record["commands"]["probe"])
        finally:
            tool.SCENES["probe"]["lines"].pop()


def test_the_pictures_are_plain_animated_svg(tool):
    for name in PICTURES:
        path = RESOURCES / name
        text = path.read_text(encoding="utf-8")
        assert path.stat().st_size < 900_000
        assert "@keyframes" in text and "infinite" in text
        assert "<script" not in text and "href=" not in text and "url(" not in text
        assert "@font-face" not in text and "@import" not in text
        width, height = map(int, re.search(r'viewBox="0 0 (\d+) (\d+)"', text).groups())
        assert width == 1280 and height < 800
    total = sum(scene["seconds"] for scene in tool.SCENES.values())
    assert 20 <= total <= 26


def test_the_picture_reads_at_the_pace_of_a_reader_and_stands_still_complete(tool):
    """Lines arrive apart, the result of a scene stays, and the still picture is whole."""
    assert tool.STEP >= 0.6 and tool.HOLD >= 3.0
    for name, scene in tool.SCENES.items():
        record = json.loads((RESOURCES / "showcase_transcript.json").read_text(encoding="utf-8"))
        groups = tool.paragraphs(name, record["commands"][name])
        assert 1.4 + tool.STEP * (len(groups) - 1) <= scene["seconds"] - tool.HOLD, name
    for name in PICTURES:
        text = (RESOURCES / name).read_text(encoding="utf-8")
        assert "prefers-reduced-motion:reduce" in text and "animation:none" in text
        # Without the animation an element shows with its own opacity: the last scene
        # (the probe and its verdict) is 1, everything else 0.
        still = dict(re.findall(r"\.(k\d+)\{opacity:([01]);", text))
        verdict = re.search(r'<text class="(k\d+)"[^>]*>✓\xa0runs:', html.unescape(text))
        assert verdict and still[verdict.group(1)] == "1"
        refused = re.search(r'<text class="(k\d+)"[^>]*>✗\xa0refused:', html.unescape(text))
        assert refused and still[refused.group(1)] == "0"
        # No development version, which a reader could not look up, and the project's name.
        assert "dev" not in text and "+g" not in text and ">GraspPanda<" in text
