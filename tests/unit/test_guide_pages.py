"""The hand-written guides say for which methods something holds, and the code agrees.

The pages under ``docs/user_guides/`` and the reference pages beside them give commands per
method (in content tabs labelled with the method names) and name the methods a statement
applies to. Each of those statements is computed here from the shipped configs, the
registry or the records, so a page cannot drift from what the toolbox does.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex

import pytest
import yaml

from grasppanda.cli import prepare
from grasppanda.combine.grid import FAMILIES, METHODS
from grasppanda.config import Config
from grasppanda.registry import BACKBONES, IMAGE_ENCODERS

from .conftest import REPO

GUIDES = REPO / "docs" / "user_guides"
#: Every method in page order, and its shipped GraspNet-1Billion RealSense config.
CONFIGS = {**METHODS, **FAMILIES}
NAMES = list(CONFIGS)
#: The tab label of a method is its name; the folder under configs/ holds its configs.
FOLDERS = {name: config.split("/")[0] for name, config in CONFIGS.items()}
#: Asset flag of `grasppanda prepare` -> the section of a training dataset that reads it.
ASSET_SECTIONS = {
    "--graspness": "graspness",
    "--compact-labels": "compact_labels",
    "--tolerance": "tolerance",
    "--scale-statistic": "scale_statistic",
    "--view-grasps": "view_grasps",
}


def page(name: str) -> str:
    return (GUIDES / name).read_text(encoding="utf-8")


def method_tabs(text: str) -> dict[str, str]:
    """Label -> body of the first set of content tabs whose labels are method names."""
    tabs: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        match = re.match(r'^=== "([^"]+)"\s*$', line)
        if match:
            current = match.group(1)
            if current in tabs:  # a second set of tabs: the first one is the answer
                break
            tabs[current] = []
            continue
        if current is None:
            continue
        if line.strip() and not line.startswith("    "):
            if tabs and len(tabs) == len(NAMES):
                break
            current = None
            continue
        tabs[current].append(line[4:])
    return {label: "\n".join(lines) for label, lines in tabs.items() if label in CONFIGS}


def commands(body: str, program: str) -> list[list[str]]:
    """The shell commands of ``body`` that start with ``program``, continuation lines joined."""
    joined = re.sub(r"\\\n\s*", " ", body)
    found = []
    for line in joined.splitlines():
        line = line.strip()
        if line.startswith(program + " "):
            found.append(shlex.split(line.split("#", 1)[0]))
    return found


def train_dataset(config: str) -> dict:
    return dict(Config.load(config).select("data.train.dataset"))


def joined(names: list[str]) -> str:
    """``A``, ``A and B``, ``A, B and C``: names as the pages list them."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@pytest.mark.parametrize(
    "guide",
    ["datasets.md", "train_test.md", "cross_dataset.md", "inference.md", "combinations.md"],
)
def test_a_guide_with_method_tabs_has_one_tab_per_method_in_page_order(guide):
    assert list(method_tabs(page(guide))) == NAMES, guide


@pytest.mark.parametrize("method", NAMES)
def test_the_prepare_commands_of_a_method_build_what_its_config_reads(method):
    """Datasets › Build what your method reads: every asset of the method's training
    dataset, built for exactly the settings the config asks for, and nothing else."""
    body = method_tabs(page("datasets.md"))[method]
    reads = train_dataset(CONFIGS[method])
    wanted = {"--labels"} | {
        flag for flag, section in ASSET_SECTIONS.items() if reads.get(section) is not None
    }
    built: set[str] = set()
    cls = prepare._dataset_class("graspnet1b")
    for argv in commands(body, "grasppanda prepare"):
        assert argv[2] == "graspnet1b", argv
        flags = {word for word in argv if word.startswith("--")}
        built |= flags & ({"--labels"} | set(ASSET_SECTIONS))
        option = dict(zip(argv, [*argv[1:], None], strict=True))
        args = argparse.Namespace(
            config=option.get("--config"), loader="train", root=None, split=None
        )
        node = prepare._node(args, cls, option.get("--camera"))
        for flag in flags & set(ASSET_SECTIONS):
            section = ASSET_SECTIONS[flag]
            assert node.get(section) == reads[section], (method, flag)
            assert node.get("camera") == reads.get("camera"), (method, flag)
    assert built == wanted, method


def test_the_methods_named_as_dense_label_readers_are_those_with_dense_grids():
    """Datasets and Data loading name the methods that read the label grids in training."""
    dense = [
        name
        for name in NAMES
        if any(
            step.get("type") == "SampleGraspLabels"
            for step in train_dataset(CONFIGS[name])["transforms"]
        )
    ]
    datasets = " ".join(page("datasets.md").split())
    assert f"methods on dense labels ({', '.join(dense)})" in datasets
    loading = (REPO / "docs/advanced_guides/data_loading.md").read_text(encoding="utf-8")
    scope = loading.split("\n\n")[1].replace("\n", " ")
    assert scope.startswith(
        f"**Applies to the methods that train on dense label grids: {joined(dense)}**"
    )
    others = [name for name in NAMES if name not in dense]
    assert all(name in scope for name in others), scope


@pytest.mark.parametrize("method", NAMES)
def test_the_train_and_demo_tabs_name_the_shipped_config_of_their_method(method):
    for guide, program in (
        ("train_test.md", "grasppanda train"),
        ("inference.md", "grasppanda demo"),
    ):
        found = commands(method_tabs(page(guide))[method], program)
        assert [argv[2] for argv in found] == [CONFIGS[method]], (guide, method)


@pytest.mark.parametrize("method", NAMES)
def test_the_transfer_tabs_list_every_shipped_transfer_config_of_their_method(method):
    body = method_tabs(page("cross_dataset.md"))[method]
    shown = sorted(argv[2] for argv in commands(body, "grasppanda test"))
    folder = REPO / "configs" / FOLDERS[method]
    shipped = sorted(f"{FOLDERS[method]}/{path.stem}" for path in folder.glob("*-to-gc6d-*.yaml"))
    assert shown == shipped, method
    if not shipped:
        assert "No transfer config ships for" in body, method
    for config in shipped:  # a transfer config is a child of a GraspNet-1Billion config
        text = (REPO / "configs" / f"{config}.yaml").read_text(encoding="utf-8")
        parent = re.search(r"_base_:\s*\n\s*-\s*(\S+)", text).group(1)
        assert (folder / parent).is_file() and "g1b-" in parent, (config, parent)
        assert f"--weights <weights of {parent.removesuffix('.yaml')}>" in body, config


def test_the_compose_examples_were_recorded_as_runs_under_their_method():
    """Compose and probe: each tab composes one option that the probe grid recorded as
    ``runs`` under the method of the tab."""
    records = json.loads(
        (REPO / "docs/advanced_guides/measured_combinations.json").read_text(encoding="utf-8")
    )["cells"]
    recorded = {}
    for cell in records:
        if "compose" not in cell:
            continue
        argv = shlex.split(cell["compose"])
        sets = tuple(argv[i + 1] for i, word in enumerate(argv) if word == "--set")
        recorded[(argv[2], sets)] = cell
    for method, body in method_tabs(page("combinations.md")).items():
        found = commands(body, "grasppanda compose")
        assert len(found) == 1, method
        argv = found[0]
        sets = tuple(argv[i + 1] for i, word in enumerate(argv) if word == "--set")
        assert argv[2] == CONFIGS[method], method
        cell = recorded.get((argv[2], sets))
        assert cell is not None, f"{method}: no record of {sets}"
        assert cell["method"] == method and cell["verdict"] == "runs", (method, cell["verdict"])


def test_the_methods_named_for_sparse_voxels_points_and_the_extra_are_those_of_the_configs():
    """Cross-dataset evaluation and Train and test name methods by their model and recipe."""
    voxel = []
    points: dict[int, list[str]] = {}
    sparse = []
    for name in NAMES:
        config = Config.load(CONFIGS[name])
        model = config.select("model")
        backbone = model.get("backbone") or {}
        if backbone.get("voxel_size") == 0.005:
            voxel.append(name)
        sample = [
            step["num_points"]
            for step in config.select("data.test.dataset")["transforms"]
            if step.get("type") == "RandomSample"
        ]
        points.setdefault(sample[0], []).append(name)
        if needs_spconv(name):
            sparse.append(name)
    cross = page("cross_dataset.md").replace("\n    ", " ")
    assert f"the 5 mm voxels of the sparse U-Net ({', '.join(voxel)})" in cross
    budgets = list(points)  # in the order of the first method that samples each budget
    phrase = f"{budgets[0]:,}".replace(",", " ") + f" points for {joined(points[budgets[0]])}"
    for budget in budgets[1:]:
        phrase += ", " + f"{budget:,}".replace(",", " ") + f" for {joined(points[budget])}"
    assert phrase in cross, phrase
    train = page("train_test.md").replace("\n", " ")
    assert f"for {joined(sparse)} the sparse-convolution extra" in train
    assert sparse == voxel


def test_the_structures_the_data_contracts_page_scopes_to_the_seed_based_skeleton():
    """HGGD exchanges none of the seed-based structures; the skeleton exchanges all of them."""
    skeleton = ("PointSet", "PointFeatures", "SeedSet", "ApproachSet", "GroupFeatures")
    hggd = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO / "grasppanda/models/detectors/hggd").glob("*.py")
    )
    seed_based = "\n".join(  # the skeleton and the stage contracts it is built from
        (REPO / path).read_text(encoding="utf-8")
        for path in ("grasppanda/models/detectors/seed_based.py", "grasppanda/models/base.py")
    )
    assert not [name for name in skeleton if name in hggd]
    assert all(name in seed_based for name in skeleton)
    text = (REPO / "docs/advanced_guides/data_flow.md").read_text(encoding="utf-8")
    seed_names = [name for name in NAMES if name in METHODS]
    assert f"| the seed-based skeleton: {', '.join(seed_names)} |" in text


def test_every_method_has_the_stage_parity_test_the_operators_page_names():
    """Operators and backends: the parity gate covers all six methods."""
    stems = {
        "GraspNet-baseline": "baseline",
        "GSNet": "gsnet",
        "Scale-Balanced-Grasp": "sbg",
        "EconomicGrasp": "economicgrasp",
        "FineGrasp": "finegrasp",
        "HGGD": "hggd",
    }
    assert set(stems) == set(NAMES)
    for stem in stems.values():
        assert (REPO / f"tests/integration/test_{stem}_parity.py").is_file(), stem
    text = (REPO / "docs/advanced_guides/operators.md").read_text(encoding="utf-8")
    assert "the stage-parity tests of all six ported methods" in text.replace("\n    ", " ")


def needs_spconv(name: str) -> bool:
    """Whether the encoder of a method's shipped config needs the sparse-convolution extra."""
    model = Config.load(CONFIGS[name]).select("model")
    classes = []
    if model.get("backbone"):
        classes.append(BACKBONES.get(model["backbone"]["type"]))
    if model.get("image_encoder"):
        classes.append(IMAGE_ENCODERS.get(model["image_encoder"]["type"]))
    return any("spconv" in getattr(cls, "extras", ()) for cls in classes)


def test_the_demo_tabs_name_the_extra_for_exactly_the_methods_that_need_it():
    tabs = method_tabs(page("inference.md"))
    named = [name for name, body in tabs.items() if "sparse-convolution extra" in body]
    assert named == [name for name in NAMES if needs_spconv(name)]


def test_the_slot_scopes_named_by_compose_and_probe_are_those_of_the_methods():
    from grasppanda.combine.options import method_slots

    def having(slot: str) -> list[str]:
        return [name for name in NAMES if slot in method_slots(name)]

    seed = [name for name in NAMES if name in METHODS]
    for slot in ("backbone", "proposal", "view", "assigner", "grouper", "head"):
        assert having(slot) == seed, slot
    assert having("image_encoder") == [name for name in NAMES if name in FAMILIES]
    for slot in ("post", "optimizer", "schedule", "hooks", "augment", "recipe"):
        assert having(slot) == NAMES, slot
    assert (len(seed), having("image_encoder"), len(NAMES)) == (5, ["HGGD"], 6)
    text = " ".join(page("combinations.md").split())
    assert (
        "the scene-encoder and stage slots exist in the five seed-based methods, the image "
        "encoder in HGGD, post-processing and the training recipe in all six"
    ) in text


def inherited(path) -> set[str]:
    """Every file a config inherits, as a path below ``configs/``, through the whole chain."""
    node = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    found: set[str] = set()
    for base in node.get("_base_", []):
        target = (path.parent / base).resolve()
        found.add(target.relative_to(REPO / "configs").as_posix())
        found |= inherited(target)
    return found


def test_the_methods_named_for_the_unsuffixed_dataset_base_are_those_that_inherit_it():
    """Datasets: the base `prepare` reads without --config is the graspness recipe of
    exactly the methods whose shipped config inherits it."""
    base = "_base_/datasets/graspnet1b_realsense.yaml"
    assert prepare._dataset_class("graspnet1b").default_config("realsense") == base
    users = [
        name for name in NAMES if base in inherited(REPO / "configs" / f"{CONFIGS[name]}.yaml")
    ]
    phrase = f"the graspness recipe of {joined(users)}"
    assert " ".join(page("datasets.md").split()).count(phrase) == 2, phrase


def test_the_evaluator_evidence_of_the_datasets_guide_is_that_of_the_parity_suite():
    """Datasets: the GraspNet-1Billion parity sentence describes the recorded suites of
    ``test_eval_parity.py``. The frames live in the cache (6 x 256 + 6 x 8 + 10, of which
    2 have tied scores); the test prints the tie-free frames it compared per suite."""
    source = (REPO / "tests/integration/test_eval_parity.py").read_text(encoding="utf-8")
    dumps = re.findall(r'\("dump", (\d+), "(\w+)"\)', source)
    labels = re.findall(r'\("label", (\d+), "(\w+)"\)', source)
    edges = re.findall(r'\("edge", (\d+), "(\w+)"\)', source)
    assert len(dumps) == 6 and all(100 <= int(scene) < 190 for scene, _ in dumps)
    assert {camera for _, camera in labels} == {"realsense", "kinect"} and edges
    text = " ".join(page("datasets.md").split())
    assert (
        "every stage equals `graspnetAPI` value for value on 1,592 recorded frames: six whole "
        "test scenes of a detector's dump, grasps drawn from the labels on both cameras and "
        "edge cases (`tests/integration/test_eval_parity.py`;"
    ) in text
    assert "{: #evaluator-parity }" in text  # the anchor other pages link


def test_the_methods_named_as_cropped_with_ground_truth_are_those_whose_test_pipeline_does():
    """Your camera and Python: the methods whose test pipeline crops to the ground-truth
    segmentation are named, and the one that crops nothing is said to."""
    cropped, uncropped = [], []
    for name in NAMES:
        steps = Config.load(CONFIGS[name]).select("data.test.dataset").get("preprocess") or []
        types = [step.get("type") for step in steps]
        (cropped if "SegmentationBoxWorkspace" in types else uncropped).append(name)
    text = " ".join(page("inference.md").split())
    assert f"{joined(cropped)} were trained and tested on clouds cropped to the objects" in text
    assert uncropped == ["HGGD"] and "HGGD crops nothing." in text


def test_the_probe_transcript_of_compose_and_probe_is_of_a_file_the_page_composes():
    """Compose and probe: a probe shown on the page probes a file a command of the page
    writes, and the transcript of a tab's file names that method's detector."""
    text = page("combinations.md")
    flat = re.sub(r"\\\n\s*", " ", text)
    written = set(re.findall(r"grasppanda compose \S+[^\n]*?-o (\S+)", flat))
    probed = re.findall(r"^\s*\$ grasppanda probe (\S+)", text, re.M)
    assert probed and set(probed) <= written, (probed, written)
    shown = []
    for method, body in method_tabs(text).items():
        for argv in commands(body, "grasppanda compose"):
            out = argv[argv.index("-o") + 1]
            if out in probed:
                detector = Config.load(CONFIGS[method]).select("model")["type"]
                block = text[text.index(f"$ grasppanda probe {out}") :].split("```")[0]
                assert re.search(rf"\n\s+model\s+{detector} · ", block), (method, detector)
                shown.append(method)
    assert shown, "no transcript probes the file of a method tab"
    # The fold that says what a probe needs has an anchor, and the page links it.
    assert "{: #what-a-probe-needs }" in text and "(#what-a-probe-needs)" in text
