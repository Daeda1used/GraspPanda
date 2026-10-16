"""The options of every slot of a method: one list for the command line, the method pages
and the explorer (``grasppanda.combine.options``).

``grasppanda compose <method config> --list [<slot>]`` prints it, ``tools/make_method_pages.py``
writes it into ``docs/methods/``, and the explorer offers it in the lanes of a method; the
tests below hold the three to the grid and to each other, on the committed records (the
first judge, mostly) and on records of judge 2.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from grasppanda.cli.main import main
from grasppanda.combine import options as option_lists
from grasppanda.combine.grid import FAMILIES, METHODS, cells
from grasppanda.combine.levels import CAVEAT, LEVELS
from grasppanda.config import Config

from .conftest import REPO
from .test_combine_levels import BALL, FINEGRASP_COLLISION, GSNET, POINTNEXT, TWO_VECTOR

MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
EXPLORER = REPO / "docs" / "explorer" / "explorer.json"
SINGLE = ("encoder", "stage", "family-slot", "post", "loss", "recipe")


def tool(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def measured() -> dict:
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def judged(payload: dict) -> dict:
    """The committed records with the real probes of judge 2 in their cells (the cells of
    the grid the examples of ``test_combine_levels`` measured)."""
    found = copy.deepcopy(payload)
    by_id = {record["id"]: record for record in found["cells"]}
    for example in (GSNET, POINTNEXT, TWO_VECTOR, BALL):
        record = by_id[example["id"]]
        for key in ("judge", "iterations", "grasps_per_frame", "trend", "active", "flags"):
            record.pop(key, None)
        record.update(example)
        record.pop("probed_as", None)
    # A runs-unstable probe without a repeat, in place of the committed record of its cell
    # (or added, as an export would, when there is none).
    collision = {
        **FINEGRASP_COLLISION,
        "group": "post",
        "method": "FineGrasp",
        "sets": ["post=collision"],
        "slot": "post",
        "component": "ModelFreeCollisionFilter",
        "variant": "collision",
        "config_hash": "82615de6ee313879",
        "probe": "grasppanda probe post__FineGrasp__collision.yaml --iters 20 --frames 8",
        "run": next(iter(found["about"]["runs"])),
    }
    found["cells"] = [record for record in found["cells"] if record["id"] != collision["id"]]
    found["cells"].append(collision)
    return found


# --------------------------------------------------------------------- the module


def test_the_options_of_a_method_are_its_single_slot_cells_of_the_grid():
    grid = {(cell.method, cell.id) for cell in cells(SINGLE)}
    for method in (*METHODS, *FAMILIES):
        found = option_lists.options(method)
        listed = {(method, option.cell) for option in found if option.cell}
        assert listed == {item for item in grid if item[0] == method}, method
        # The presets without a cell are listed with the reason; nothing else lacks a cell.
        for option in found:
            assert bool(option.cell) != bool(option.reason), option
            assert option.sets, option
        names = [(option.slot, option.name) for option in found]
        assert len(names) == len(set(names)), f"{method}: an option listed twice"
    assert option_lists.method_slots("HGGD") == [
        "image_encoder",
        "post",
        "optimizer",
        "schedule",
        "hooks",
        "augment",
        "recipe",
    ]
    assert option_lists.method_slots("GSNet")[:7] == [
        "backbone",
        "proposal",
        "view",
        "assigner",
        "grouper",
        "head",
        "loss",
    ]


def test_a_listing_says_what_fits_what_does_not_and_why():
    view = option_lists.listing("GSNet", "view")[0]
    assert view.entries[0].shipped and view.entries[0].option.name == "template-view"
    assert view.shipped == "TemplateViewStage"
    # Since compose lifts the score owner (W4), every view stage fits GSNet; under
    # EconomicGrasp the dense-label view stage stays a structural refusal.
    assert not view.group("does not fit")
    compact = option_lists.listing("EconomicGrasp", "view")[0]
    refused = {entry.option.name: entry for entry in compact.group("does not fit")}
    assert "objectness-view" in refused
    assert "'dense' matched labels" in refused["objectness-view"].reason
    for entry in view.group("fits"):
        assert entry.verdict in ("runs", "runs-empty", "runs-unstable", "fails")
        assert set(entry.classes) == set(LEVELS) and entry.levels is not None
        assert entry.command.startswith(
            "grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set view="
        )
    # HGGD's image encoders: a refusal by policy is no failure, and it says its kind.
    images = option_lists.listing("HGGD", "image_encoder")[0]
    radio = next(e for e in images.entries if e.option.name == "hf-radio-v3-base")
    assert radio.status == "does not fit" and radio.kind == "policy"
    assert radio.classes["composes"].name == "needs consent"
    # A slot the method does not have.
    with pytest.raises(KeyError, match="HGGD has no slot 'backbone'"):
        option_lists.listing("HGGD", "backbone")
    # The encoder presets of a recipe are no option of HGGD's, for a reason that is said.
    hooks = option_lists.listing("HGGD", "hooks")[0]
    freeze = next(e for e in hooks.entries if e.option.name == "freeze-encoder")
    assert freeze.status == "not probed" and "model.backbone" in freeze.reason


def test_the_method_of_a_config():
    assert option_lists.method_of("gsnet/gsnet_sunet14_g1b-realsense_10e") == ("GSNet", "")
    method, note = option_lists.method_of(
        REPO / "configs" / "experimental" / "gsnet_ptv3_g1b-realsense_10e.yaml"
    )
    assert method == "GSNet" and "--set backbone=ptv3" in note
    with pytest.raises(KeyError, match="measured per method config"):
        option_lists.method_of("gsnet/gsnet_sunet14_gc6d-d435_10e")
    with pytest.raises(KeyError, match="neither a file nor a shipped config"):
        option_lists.method_of("no/such-config")


def test_the_measured_file_is_found_in_a_checkout_and_in_a_wheel():
    assert option_lists.records_file() == MEASURED
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    packaged = '"docs/advanced_guides/measured_combinations.json" = "grasppanda/_measured/measured_combinations.json"'
    assert packaged in pyproject


# ------------------------------------------------------------------ the command line


def test_compose_lists_the_options_of_a_method(capsys):
    assert main(["compose", "gsnet/gsnet_sunet14_g1b-realsense_10e", "--list", "view"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("GSNet · view  (model.view)")
    assert CAVEAT in out and "template-view (shipped)" in out and "do not fit" not in out
    assert (
        main(["compose", "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e", "--list", "view"])
        == 0
    )
    assert "do not fit: 2" in capsys.readouterr().out
    assert (
        main(["compose", "hggd/hggd_r34w8_g1b-realsense_15e", "--list", "image_encoder", "--json"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["method"] == "HGGD" and report["caveat"] == CAVEAT
    assert set(report["levels"]) == set(LEVELS)
    (slot,) = report["slots"]
    assert slot["slot"] == "image_encoder" and slot["counts"]["fits"] >= 30
    radio = next(item for item in slot["options"] if item["option"] == "hf-radio-v3-base")
    assert radio["kind"] == "policy" and radio["status"] == "does not fit"
    assert radio["refusal"][0]["message"].startswith("model.image_encoder")
    # A plural registry name is a slot too; a wrong slot, method or config exits 2.
    assert main(["compose", "gsnet/gsnet_sunet14_g1b-realsense_10e", "--list", "groupers"]) == 0
    assert "GSNet · grouper" in capsys.readouterr().out
    assert main(["compose", "hggd/hggd_r34w8_g1b-realsense_15e", "--list", "backbone"]) == 2
    assert "Its slots: image_encoder" in capsys.readouterr().err
    assert main(["compose", "gsnet/gsnet_sunet14_gc6d-d435_10e", "--list"]) == 2
    # Without a config, --list is the list of presets it always was.
    assert main(["compose", "--list", "view"]) == 0
    assert "preset" in capsys.readouterr().out


# ----------------------------------------------------------- the explorer (W6.3)


def test_the_explorer_lanes_list_exactly_the_options_compose_lists():
    data = json.loads(EXPLORER.read_text(encoding="utf-8"))
    lanes = {}
    for method in data["methods"]:
        assert method["slots"] == option_lists.method_slots(method["name"]), method["name"]
        for slot in method["slots"]:
            listed = [
                item if isinstance(item, str) else item["name"] for item in method["options"][slot]
            ]
            wanted = [option.name for option in option_lists.options(method["name"], slot)]
            assert listed == wanted, (method["name"], slot)
            lanes[(method["name"], slot)] = sorted(wanted)
    assert {name for name, _ in lanes} == {*METHODS, *FAMILIES}
    # The page logic offers exactly these in a lane (where node can run it).
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the lists of the data are compared above")
    script = (
        "const L = require(process.argv[1]); const data = require(process.argv[2]);"
        "const out = {}; for (const m of data.methods) for (const s of m.slots)"
        " out[m.name + '|' + s] = L.choices(data, m.name, s).map((c) => c.name).sort();"
        "console.log(JSON.stringify(out));"
    )
    done = subprocess.run(
        [node, "-e", script, str(REPO / "docs/explorer/explorer_logic.js"), str(EXPLORER)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    offered = json.loads(done.stdout)
    assert offered == {f"{name}|{slot}": names for (name, slot), names in lanes.items()}


# ------------------------------------------------------------------- the pages (W6.2)


def test_the_method_pages_are_current_and_linked():
    generator = tool("make_method_pages")
    assert not generator.stale(), "stale: run `python -m tools.make_method_pages`."
    assert generator.main(["--check"]) == 0
    nav = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    for method, config in {**METHODS, **FAMILIES}.items():
        stem = config.split("/")[0]
        page = (REPO / "docs" / "methods" / f"{stem}.md").read_text(encoding="utf-8")
        assert page.startswith("<!-- Generated by tools/make_method_pages.py")
        assert f"\n# {method}\n" in page and CAVEAT[1:] in page
        assert f"## Run {method} in five minutes {{#{generator.RUN_ANCHOR}}}" in page
        assert "## The options of every slot {#the-slots}" in page
        assert f"methods/{stem}.md" in nav, f"{stem} is not in the nav of mkdocs.yml"
        readme = (REPO / "configs" / stem / "README.md").read_text(encoding="utf-8")
        assert f"docs/methods/{stem}.md" in readme, (
            f"configs/{stem}/README.md does not link its page"
        )
        assert f"grasppanda compose {config} --list" in readme
        for slot in option_lists.method_slots(method):
            assert f"{{#{generator.anchor(slot)}}}" in page, (method, slot)
    for stem in generator.SLOT_PAGE_SLOTS:
        text = (REPO / "docs" / "slots" / f"{stem}.md").read_text(encoding="utf-8")
        assert generator.BEGIN in text and f"*Class: {CAVEAT}.*" in text, stem


def test_every_method_config_names_the_command_that_lists_each_slot():
    """A comment block above each slot (comments leave the config hash alone)."""
    for method, config in {**METHODS, **FAMILIES}.items():
        path = REPO / "configs" / f"{config}.yaml"
        text = path.read_text(encoding="utf-8")
        for slot in option_lists.method_slots(method):
            assert f"grasppanda compose {config} --list {slot}" in text, (config, slot)
        assert Config.load(config).digest()  # still a config


# --------------------------------------------- the generators on records of judge 2


def test_the_generators_read_records_of_judge_two(measured, tmp_path, monkeypatch):
    payload = judged(measured)
    target = tmp_path / "measured_combinations.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    records = option_lists.Measured.load(target)
    # The command line's listing.
    view = option_lists.listing("GraspNet-baseline", "view", records)[0]
    two = next(e for e in view.entries if e.option.name == "two-vector")
    assert [two.levels.states[level] for level in LEVELS] == ["yes", "yes", "no", "yes"]
    assert two.fall["view_frame"] == -0.19 and two.judge == 2
    text = option_lists.render([view])
    assert "two-vector" in text and "learns: view_frame rose 19 %, score rose 74 %" in text
    post = option_lists.listing("FineGrasp", "post", records)[0]
    collision = next(e for e in post.entries if e.option.name == "collision")
    assert (
        collision.status == "fits" and collision.levels.states["runs"] == "unstable, not repeated"
    )
    # The method page.
    pages = tool("make_method_pages")
    page = pages.page("GraspNet-baseline", records)
    row = next(line for line in page.splitlines() if line.startswith("| `two-vector`<br>"))
    assert "| ✓ ✓ ✗ ✓ |" in row and "view_frame -19 %" in row
    assert "view_frame is 0 at the last iteration (active in 14 of 20)" in row
    gsnet = pages.page("GSNet", records)
    row = next(
        line
        for line in gsnet.splitlines()
        if line.startswith('| <span class="gp-nograsp">`pointnext`')
    )
    assert "| ✓ ✓ ✓ ✗ |" in row and "predicts: no grasp on any frame" in row
    # The measured page and the explorer.
    docs, explorer = tool("make_combination_docs"), tool("make_explorer")
    monkeypatch.setattr(docs, "DATA", target)
    measured_page = docs.page()
    assert "learns ✗: view_frame rose 19 %; score rose 74 %" in measured_page
    assert "predicts ✗: no grasp on any frame" in measured_page
    monkeypatch.setattr(explorer, "MEASURED", target)
    data = explorer.data()
    cells_ = {cell["id"]: cell for cell in data["cells"]}
    assert cells_[TWO_VECTOR["id"]]["levels"] == "YYNY"
    assert cells_[POINTNEXT["id"]]["levels"] == "YYYN"
    assert cells_[FINEGRASP_COLLISION["id"]]["levels"] == "YRRR"


def test_the_classes_of_a_grid_judged_by_judge_two_everywhere(measured, tmp_path, monkeypatch):
    """Every probe with numbers as judge 2 would record it (every term fell, scores
    spread): the classes at `learns` and `predicts` are decided, and the tables say so."""
    payload = copy.deepcopy(measured)
    for record in payload["cells"]:
        losses = record.get("losses")
        if not losses:
            continue
        record["judge"] = 2
        record["trend"] = {
            name: [
                pair[0],
                pair[1],
                round((pair[0] - pair[1]) / abs(pair[0]), 3) if pair[0] else None,
            ]
            for name, pair in losses.items()
        }
        record.update(no_gradient=[], unchanged=[])
        record["score_quantiles"] = [
            [0.1, 0.2, 0.3] if count else None for count in record.get("grasps_per_frame", [])
        ]
    target = tmp_path / "measured_combinations.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    docs = tool("make_combination_docs")
    monkeypatch.setattr(docs, "DATA", target)
    page = docs.page()
    glance = page.split("## Measured classes")[1].split("## The methods as shipped")[0]
    assert "**learns**\n\n| Slot |" in glance and "**predicts**\n\n| Slot |" in glance
    records = docs.Records(payload["cells"])
    index = docs.slots_index(records)
    assert "**learns**\n\n| Slot | Probed under |" in index
    found = {v: c for _, v, c in docs.slot_classes("proposal", records, "stage")}
    assert found["dense-fps"]["learns"].decided == 5
    pages = tool("make_method_pages")
    page = pages.page("GSNet", option_lists.Measured(payload))
    assert "not checked" not in page.split("## Seed proposal")[1].split("## View")[0]
