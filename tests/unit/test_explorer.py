"""The combination explorer: its data is generated, complete and honest.

The page logic is JavaScript (``docs/explorer/explorer_logic.js``); it is exercised with
node against every measured record where node is installed, and its lookup is mirrored
here in Python so that the suite says something without it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys

import pytest

from .conftest import REPO

EXPLORER = REPO / "docs" / "explorer"
DATA = EXPLORER / "explorer.json"
CONTRACTS = EXPLORER / "explorer_contracts.json"
MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"


def tool(name: str):
    """A module of ``tools/``, loaded by path (the directory is not a package on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured() -> dict:
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def test_the_committed_data_is_what_the_generator_writes():
    generator = tool("make_explorer")
    assert not generator.stale(), "stale: run `python -m tools.make_explorer`."
    assert generator.main(["--check"]) == 0


def test_every_record_is_in_the_data_with_its_verdict(data, measured):
    cells = {cell["id"]: cell for cell in data["cells"]}
    assert set(cells) == {record["id"] for record in measured["cells"]}
    assert data["counts"]["records"] == len(measured["cells"])
    for record in measured["cells"]:
        cell = cells[record["id"]]
        assert cell["verdict"] == record["verdict"]
        assert cell.get("refusal", []) == record.get("refusal", [])
        # The lookup of the page, mirrored: the key of a record leads to its verdict.
        found = data["cells"][data["index"][cell["key"]]]
        assert found["verdict"] == record["verdict"], record["id"]
        assert found["method"] == record["method"]
    counted = data["counts"]["by_verdict"]
    assert sum(counted.values()) == len(measured["cells"])
    assert data["counts"]["key_conflicts"] == 0, (
        "two records with one key (a method and the sets that change it) disagree"
    )


def test_a_set_that_names_what_a_method_ships_with_is_no_change(data, measured):
    """What the key leaves out is measured to be a no-op: the same config hash."""
    by_id = {record["id"]: record for record in measured["cells"]}
    for method in data["methods"]:
        reference = by_id[method["reference"]]
        for slot, ships in method["ships"].items():
            if not ships["option"]:
                continue
            same = [
                record
                for record in measured["cells"]
                if record["method"] == method["name"]
                and record.get("sets") == [f"{slot}={ships['option']}"]
            ]
            assert same and same[0]["config_hash"] == reference["config_hash"]
    keyed: dict[str, set] = {}
    for cell in data["cells"]:
        keyed.setdefault(cell["key"], set()).add(by_id[cell["id"]]["config_hash"])
    assert all(len(hashes) == 1 for hashes in keyed.values())


def test_options_come_from_the_presets_and_the_catalogue(data):
    from grasppanda.cli.components import collect
    from grasppanda.combine.presets import presets

    registered = {entry.name for entry in collect()}
    for slot, members in presets().items():
        names = {option["name"] for option in data["options"][slot]}
        assert set(members) <= names, f"{slot}: a preset is missing"
    # The training recipe: its presets are options too, and every method has its slots.
    from grasppanda.combine.recipes import recipes

    for slot, members in recipes().items():
        assert set(members) == {option["name"] for option in data["options"][slot]}, slot
        assert all(slot in method["slots"] for method in data["methods"]), slot
    around = {lane["key"] for lane in data["lanes"] if lane["recipe"]}
    assert around == set(recipes())
    for slot, options in data["options"].items():
        assert slot in {lane["key"] for lane in data["lanes"]}
        for option in options:
            # A recipe preset may name several components, or only options of the trainer.
            parts = option["component"].split(", ") if slot in around else [option["component"]]
            for part in parts:
                assert part in registered or (slot in around and part == "train options"), option
            assert option["summary"], option["name"]
            assert option["yaml"].strip()
            if option["page"]:
                assert (REPO / "docs" / "slots" / f"{option['page']}.md").is_file()
    for lane in data["lanes"]:
        for page in lane["pages"]:
            assert (REPO / "docs" / "slots" / f"{page['stem']}.md").is_file()


def test_the_page_claims_no_accuracy_and_says_what_is_measured(data):
    generator = tool("make_explorer")
    page = (EXPLORER / "index.md").read_text(encoding="utf-8")
    assert data["no_accuracy"] == generator.NO_ACCURACY
    assert "**No accuracy.**" in page and "*not measured*" in page
    assert '--8<-- "docs/explorer/explorer.json"' in page
    script = (EXPLORER / "explorer.js").read_text(encoding="utf-8")
    assert "data.no_accuracy" in script and "this page does not guess" in script
    # No number of the data is an accuracy: the records hold none, and none is derived.
    for cell in data["cells"]:
        assert not {"ap", "AP", "accuracy", "score"} & set(cell)


def test_the_counts_say_what_stands_behind_a_record(data, measured):
    """A refusal is no probe: the page must not call every record "probed on real frames"."""
    records = measured["cells"]
    counts = data["counts"]
    assert counts["probed"] == sum("probe" in record for record in records)
    assert counts["shared_probe"] == sum("probed_as" in record for record in records)
    refused = counts["by_verdict"].get("refused", 0)
    assert counts["probed"] + counts["shared_probe"] + refused == counts["records"]
    from grasppanda.combine.grid import cells

    listed = {cell.id for cell in cells() if cell.group == "stage-set"}
    assert counts["stage_sets"]["listed"] == len(listed)
    assert counts["stage_sets"]["recorded"] == len(listed & {r["id"] for r in records})
    script = (EXPLORER / "explorer.js").read_text(encoding="utf-8")
    assert "probed on real frames" not in script
    assert "counts.probed" in script and "counts.stage_sets" in script


def test_the_contract_answers_are_those_of_compose(data):
    """The stored answers of compose: complete in shape, and a sample composed again."""
    generator = tool("make_explorer_contracts")
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert set(contracts["methods"]) <= {method["name"] for method in data["methods"]}
    assert contracts["table"][0] == {"ok": True}
    counted = 0
    for entry in contracts["methods"].values():
        for key, row in entry["answers"].items():
            size = 1
            for slot in key.split("|"):
                size *= len(entry["options"][slot])
            assert len(row) == size, key
            assert all(0 <= index < len(contracts["table"]) for index in row)
            counted += size
    assert counted == contracts["counts"]["answers"]
    for found in contracts["table"]:
        assert found["ok"] or found["messages"], "a refusal without its reason"
        assert not ({"ap", "accuracy", "score", "loss"} & set(found))
    # A refused record is refused by compose today, and nothing else is (the node test
    # checks the same through the page logic, for every single swap).
    gsnet = next(method for method in data["methods"] if method["name"] == "GSNet")
    refused = generator.lookup(contracts, "GSNet", ["head=economic"])
    assert refused and not refused["ok"] and "EconomicHead" in refused["messages"][0]
    assert generator.lookup(contracts, gsnet["name"], ["backbone=ptv3"]) == {"ok": True}
    assert generator.lookup(contracts, "GSNet", ["backbone=no-such-option"]) is None
    assert generator.lookup(contracts, "GSNet", ["head=economic", "backbone=ptv3"]) == (
        generator.lookup(contracts, "GSNet", ["backbone=ptv3", "head=economic"])
    )
    # In an interpreter of its own: the messages of compose list the registered
    # components, and other tests of this suite register toy ones.
    done = subprocess.run(
        [sys.executable, "-m", "tools.make_explorer_contracts", "--sample", "16"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]


def test_the_page_is_self_contained_and_small():
    for path in EXPLORER.iterdir():
        assert path.stat().st_size < 1_000_000, f"{path.name} is over 1 MB"
    for path in (DATA, CONTRACTS):
        text = path.read_text(encoding="utf-8")
        assert "</" not in text and text.count("\n") == 1, "not safe inside a <script> element"
    page = (EXPLORER / "index.md").read_text(encoding="utf-8")
    assert '--8<-- "docs/explorer/explorer_contracts.json"' in page
    for name in ("index.md", "explorer.js", "explorer_logic.js", "explorer.css"):
        source = (EXPLORER / name).read_text(encoding="utf-8")
        for remote in re.findall(
            r"""(?:src|href)=["'](https?:)?//[^"']+|url\(\s*["']?https?:""", source
        ):
            raise AssertionError(f"{name} loads something from the network: {remote}")
        assert "@import" not in source


def test_the_page_logic_with_node(data):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the lookup is mirrored in Python above")
    done = subprocess.run(
        [node, str(REPO / "tests" / "js" / "explorer_logic.test.js"), str(REPO)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    assert re.search(r"^ok \d+ checks$", done.stdout.strip().splitlines()[-1])
