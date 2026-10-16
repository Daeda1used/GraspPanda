"""The five-minute sections of the method pages: every command runs, every number has a source.

``tools/make_method_pages.py`` starts ``docs/methods/<method>.md`` with *Run <method> in five
minutes*, written from the metafile, the config and ``docs/methods/five_minutes.json``. The
commands are taken from the pages themselves, so what a reader copies is what is checked:

* every shell command parses with the parser of the program it calls (``grasppanda``, or the
  converter of the repository) and names what exists: the shipped config, a source format
  whose model is the model of that config, the overrides of the test command;
* the demo command and its camera variant run end to end on the CPU, on the method's config
  with a toy detector of its family in place of the model (the real pipeline and
  post-processing), and the camera variant reads the files where the page says the demo
  wrote them;
* the Python block runs against the same toy and returns the array the demo wrote;
* the browser route finds the converted file in a folder and picks it for its method;
* the measured runs agree with the metafile (weights, provenance, inputs, whole-split rows)
  and with the config (the collision-filter chain), and every conversion command of a
  metafile is printed in the method's README.

Commands that need the authors' checkpoint or the dataset were also run for real; what they
printed is in ``five_minutes.json`` with the date, the commit and the machine.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
import pytest

import grasppanda
from grasppanda.apis import GraspInferencer
from grasppanda.cli.main import build_parser, main
from grasppanda.combine import options as option_lists
from grasppanda.combine.explore_run import WeightsSource, suggest
from grasppanda.config import Config
from grasppanda.hub import load_metafiles

from .conftest import REPO
from .inference_fixtures import PROJECT, save_weights
from .test_engine_tiny import single_thread  # noqa: F401  (autouse fixture)
from .test_method_options import tool

PAGES = tool("make_method_pages")
CONVERTER = tool("convert_checkpoint")
METHODS = list(option_lists.methods())
#: Smaller pictures than the page's commands draw: the pictures are not what is tested here.
SMALL = ["--size", "160", "120", "--gif-frames", "2"]
#: The methods whose toy runs are in the quick selection; the others are marked slow.
QUICK = ("GraspNet-baseline", "HGGD")
#: Toy detectors in place of a method's model, one per family: point clouds, or images too.
TOY_POINTS = {
    "type": "ToyDetector",
    "num_seeds": 32,
    "backbone": {"type": "ToyBackbone", "out_channels": 16},
    "grouper": {"type": "ToyGrouper", "radius": 0.05, "nsample": 8},
    "head": {"type": "ToyHead", "num_angle": 12, "num_depth": 4},
}
TOY_IMAGE = {"type": "ImageToyDetector", "num_grasps": 64}


def section(method: str) -> str:
    page = (REPO / "docs" / "methods" / f"{PAGES.page_of(method)}.md").read_text(encoding="utf-8")
    start = page.index(f"## Run {method} in five minutes {{#{PAGES.RUN_ANCHOR}}}")
    return page[start : page.index("## The options of every slot", start)]


def blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def shell_commands(text: str) -> list[list[str]]:
    """Every command of the shell blocks, as words (continuations joined, comments dropped)."""
    found = []
    for block in blocks(text, "bash"):
        for line in block.replace("\\\n", " ").splitlines():
            if words := shlex.split(line, comments=True):
                found.append(words)
    return found


def facts(method: str):
    """The config of a method, the weights a user downloads for it, and its zoo entry."""
    _, model = PAGES.zoo_entry(method)
    return option_lists.methods()[method], PAGES.downloadable(model), model


def flat(text: str) -> str:
    """Text with shell continuations joined and runs of white space made one space."""
    return re.sub(r"\s+", " ", text.replace("\\\n", " "))


# ------------------------------------------------------------------- the commands


@pytest.mark.parametrize("method", METHODS)
def test_every_shell_command_parses_and_names_what_exists(method):
    config, weights, _ = facts(method)
    converted = weights.converted_name
    seen = []
    for words in shell_commands(section(method)):
        if words[:3] == ["python", "-m", "tools.convert_checkpoint"]:
            args = CONVERTER.build_parser().parse_args(words[3:])
            assert " ".join(words) == weights.convert  # the command of the metafile
            assert str(args.destination) == converted
            # the converter checks the file against this model: the model of the config
            default = CONVERTER.SOURCE_FORMATS[args.source_format][1]
            assert (
                Config.load(default).resolved()["model"] == Config.load(config).resolved()["model"]
            )
            seen.append("convert")
            continue
        assert words[0] == "grasppanda", words
        args = build_parser().parse_args(words[1:])
        seen.append(args.command)
        if args.command in ("demo", "test"):
            assert args.config == config and args.weights == converted, words
            resolved = Config.load(args.config, args.opts).resolved()
        if args.command == "demo":
            assert args.collision_filter
        elif args.command == "test":
            node = resolved["data"]["test"]["dataset"]
            assert (node["split"], node["frame_step"]) == ("test_seen", 64)
            # the page says that testing needs no derived asset
            assets = ("graspness", "compact_labels", "view_grasps", "tolerance", "scale_statistic")
            assert not node["load_labels"] and all(node[name] is None for name in assets)
        elif args.command == "explore":
            assert args.weights_dir
    assert sorted(seen) == ["convert", "demo", "demo", "explore", "test"]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A method's config with a toy detector of its family as the model, and its weights.

    The data and evaluation sections stay those of the method, so the demo and the Python
    block run its pipeline and its post-processing. The file has the name of the method's
    config, so the demo writes where the page says.
    """
    work = tmp_path_factory.mktemp("five_minutes")

    def make(method: str) -> tuple[Path, Path]:
        config = option_lists.methods()[method]
        data = Config.load(config).resolved()
        data["model"] = dict(TOY_IMAGE if PAGES.pipeline_facts(config)["rgb"] else TOY_POINTS)
        data["custom_imports"] = [str(PROJECT / "toy_components.py")]
        data["runtime"]["device"] = "cpu"
        path = work / PAGES.page_of(method) / f"{Path(config).name}.yaml"
        path.parent.mkdir()
        path.write_text(Config.from_dict(data).dump(), encoding="utf-8")
        return path, save_weights(path, path.with_suffix(".pth"))

    return make


@pytest.mark.parametrize(
    "method", [pytest.param(m, marks=() if m in QUICK else pytest.mark.slow) for m in METHODS]
)
def test_the_demo_the_camera_files_and_python_give_the_same_grasps(
    method, toy, tmp_path, monkeypatch, capsys
):
    config, weights, _ = facts(method)
    toy_config, toy_weights = toy(method)
    swap = {config: str(toy_config), weights.converted_name: str(toy_weights)}
    text = section(method)
    demo, camera = (words for words in shell_commands(text) if words[:2] == ["grasppanda", "demo"])
    monkeypatch.chdir(tmp_path)

    assert main([swap.get(word, word) for word in demo[1:]] + SMALL) == 0
    out = tmp_path / PAGES.pipeline_facts(config)["out"]
    assert f"`{PAGES.pipeline_facts(config)['out']}/`" in text
    written = np.load(out / "grasps.npy")
    for name in ("grasps.png", "grasps.gif", "example/depth.png", "example/intrinsics.txt"):
        assert (out / name).is_file(), name
    # the camera variant reads the scene files where the page says the demo wrote them
    assert main([swap.get(word, word) for word in camera[1:]] + SMALL) == 0
    assert (tmp_path / "work_dirs/camera/grasps.npy").is_file()
    capsys.readouterr()

    def inferencer(name, file, **options):  # the page's names; the toy is built instead
        assert (name, file) == (config, weights.converted_name)
        return GraspInferencer(toy_config, toy_weights, **options)

    monkeypatch.setattr(grasppanda, "GraspInferencer", inferencer, raising=False)
    (code,) = blocks(text, "python")
    statements = [line for line in code.splitlines() if line and not line[0].isspace()]
    assert len([line for line in statements if line != ")"]) <= 6, "the block is meant to be short"
    namespace: dict = {}
    exec(compile(code, f"docs/methods/{PAGES.page_of(method)}.md", "exec"), namespace)
    np.testing.assert_array_equal(namespace["grasps"].sort_by_score().to_graspnet_array(), written)
    assert capsys.readouterr().out.split()[0] == str(len(written))


@pytest.mark.parametrize("method", METHODS)
def test_the_browser_route_offers_the_converted_file_for_its_method(method, tmp_path):
    config, weights, _ = facts(method)
    folder = tmp_path / "grasppanda_weights"
    folder.mkdir()
    for other in METHODS:  # everyone's converted file in one folder, as a user would keep them
        (folder / facts(other)[1].converted_name).write_bytes(b"weights")
    source = WeightsSource.from_arguments(str(folder), [])
    entries, _ = source.entries()
    chosen = suggest(method, config, entries)
    assert chosen and source.resolve(chosen).name == weights.converted_name
    assert f"`{weights.converted_name}` is chosen for you by its name" in section(method)


# --------------------------------------------------------------- the measured runs


@pytest.mark.parametrize("method", METHODS)
def test_the_measured_runs_agree_with_the_metafile_and_the_config(method):
    data = PAGES.five_minutes()
    entry = data["methods"][method]
    config, weights, model = facts(method)
    pipeline = PAGES.pipeline_facts(config)
    assert entry["config"] == config
    # the first run, whose demo and test the page's commands repeat, is of the file a user gets
    assert PAGES.weights_of(model, entry["runs"][0]["weights"]) is weights
    for run in entry["runs"]:
        used = PAGES.weights_of(model, run["weights"])
        provenance = PAGES.provenance_of(used)
        assert run["demo"] and run["test"], "a run measures the demo and the test"
        for kind in ("demo", "python", "test"):
            for measured in run[kind]:
                assert measured["machine"] in data["machines"]
                assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", measured["date"])
                assert re.fullmatch(r"[0-9a-f]{7,40}", measured["commit"])
                assert re.fullmatch(r"[0-9a-f]{64}", measured["weights_sha256"])
                assert measured["evidence"].startswith("$GRASPPANDA_CACHE/verification/")
                if used.source == "grasppanda":  # weights trained here: the file itself
                    assert measured["weights_sha256"] == used.sha256
        for demo in run["demo"]:
            assert [name for name, _ in demo["after"]] == pipeline["chain"]
            counts = [demo["predicted"], *(count for _, count in demo["after"])]
            assert counts == sorted(counts, reverse=True), "a filter only removes grasps"
            assert (demo["workspace_points"] is not None) == pipeline["mask"]
        for python in run["python"]:
            same = [demo for demo in run["demo"] if demo["commit"] == python["commit"]]
            assert same and python["grasps"] == PAGES._kept(same[0])
        for test in run["test"]:
            assert set(test["AP"]) == {"raw", pipeline["variant"]}
            assert ("workspace" in test["gt_assisted"]) == pipeline["mask"]
            for variant, collision in (("raw", False), (pipeline["variant"], True)):
                whole = PAGES._whole_split(model, provenance, collision)
                if whole is None:
                    continue
                inputs = "gt_assisted: workspace" in whole.inputs
                assert inputs == ("workspace" in test["gt_assisted"])
                # a subset far from the whole split is a typo or a broken run
                assert abs(test["AP"][variant] - whole.ap["seen"]) < 1.5, (variant, whole)


def test_every_conversion_command_of_a_metafile_is_in_the_method_readme():
    for metafile in load_metafiles():
        readme = flat((metafile.path.parent / "README.md").read_text(encoding="utf-8"))
        for model in metafile.models:
            for weights in (model.weights, model.author_weights):
                if weights is not None and weights.convert:
                    assert weights.convert in readme, (metafile.method, weights.convert)


def test_the_sections_are_linked_from_the_quick_starts_and_the_method_readmes():
    linked = {
        "README.md": "docs/methods/{stem}.md",
        "README_zh-CN.md": "docs/methods/{stem}.md",
        "docs/get_started/quickstart.md": "../methods/{stem}.md#run-in-five-minutes",
        "docs/user_guides/inference.md": "../methods/{stem}.md#run-in-five-minutes",
    }
    for method in METHODS:
        stem = PAGES.page_of(method)
        for name, link in linked.items():
            assert link.format(stem=stem) in (REPO / name).read_text(encoding="utf-8"), name
        readme = (REPO / "configs" / stem / "README.md").read_text(encoding="utf-8")
        assert f"**Run it in five minutes:** [docs/methods/{stem}.md]" in readme
