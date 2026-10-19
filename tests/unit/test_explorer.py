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

from grasppanda.combine.grid import SHIPPED_EQUIVALENTS

from .conftest import REPO

EXPLORER = REPO / "docs" / "explorer"
DATA = EXPLORER / "explorer.json"
CONTRACTS = EXPLORER / "explorer_contracts.json"
LEARNS = EXPLORER / "explorer_learns.json"
MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
LEVEL2 = REPO / "docs" / "advanced_guides" / "learns_like_reference.json"


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
    """The data file as the page prepares it at load: the key of every cell, the names of
    its run and probe options (stored as indexes) and the index from a key to its cell."""
    loaded = json.loads(DATA.read_text(encoding="utf-8"))
    generator = tool("make_explorer")
    by_name = {method["name"]: method for method in loaded["methods"]}
    assert "index" not in loaded, "the index is built by the page, not stored"
    for cell in loaded["cells"]:
        assert "key" not in cell, "the key is computed by the page, not stored"
        cell["key"] = generator.key_of(by_name[cell["method"]], cell["sets"])
        cell["run"] = loaded["runs"][cell["run"]]  # stored as an index
        if "probe_opts" in cell:
            cell["probe_opts"] = loaded["probe_opts"][cell["probe_opts"]]
    loaded["index"] = generator.index_of(loaded["cells"], [cell["key"] for cell in loaded["cells"]])
    return loaded


@pytest.fixture(scope="module")
def chips():
    return tool("docs_chips")


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
        # By name, a base before its variants, and with the source the slot page prints.
        names = [option["name"] for option in options]
        assert names == sorted(names, key=str.lower), slot
        page = {
            found.name: found.source for found in tool("make_combination_docs").slot_options(slot)
        }
        for option in options:
            if option["name"] in page:
                assert option["source"] == page[option["name"]], (slot, option["name"])
    for method in data["methods"]:
        for slot, listed in method["options"].items():
            names = [item if isinstance(item, str) else item["name"] for item in listed]
            assert names == sorted(names, key=str.lower), (method["name"], slot)


def test_a_slot_is_named_placed_and_scoped_as_the_pages_do(data, chips):
    """Lane titles, doc pages and the methods a slot applies to come from docs_chips, the
    vocabulary of every page; nothing is counted from the text of a page."""
    for lane in data["lanes"]:
        slot = lane["key"]
        assert lane["title"] == chips.SLOT_TITLES[slot]
        assert lane["page"] + ".md" == chips.SLOT_PAGES[slot]
        assert (REPO / "docs" / chips.SLOT_PAGES[slot]).is_file(), slot
        assert lane["scope"] == chips.scope(slot)
        assert lane["count"] == chips.option_count(slot), f"{slot}: counted unlike the pages"
        listed = [o["name"] for o in data["options"][slot] if not o.get("not_option")]
        assert sorted(listed) == sorted(chips.options_of(slot)), slot
        having = [m["name"] for m in data["methods"] if slot in m["slots"]]
        assert having == list(chips.slot_methods(slot)), slot
        if "counterpart" in lane:
            other = next(entry for entry in data["lanes"] if entry["key"] == lane["counterpart"])
            assert other["counterpart"] == slot and other["scope"] != lane["scope"]
    assert data["scopes"] == chips.SCOPE_TEXT and data["families"] == chips.FAMILY_NAMES
    for option in (o for members in data["options"].values() for o in members):
        assert "page" not in option and "classes" not in option


def test_the_methods_are_in_page_order_with_their_own_venue(data, chips):
    """No method comes first by accident: the gallery shows them by family, then year, each
    with the venue of its own metafile (FineGrasp is not EconomicGrasp's ECCV 2024)."""
    import yaml

    assert [m["name"] for m in data["methods"]] == [info.name for info in chips.methods()]
    for method, info in zip(data["methods"], chips.methods()):
        folder = method["config"].split("/")[0]
        meta = yaml.safe_load((REPO / "configs" / folder / "metafile.yaml").read_text())
        assert method["venue"] == str(meta["venue"]) == info.venue
        assert (method["family"], method["inputs"], method["page"]) == (
            info.family,
            info.inputs,
            info.page.removesuffix(".md"),
        )
        assert (REPO / "docs" / info.page).is_file() and method["line"]
        assert "source" not in method, "a citation of the detector type, not of the method"
    finegrasp = next(m for m in data["methods"] if m["name"] == "FineGrasp")
    assert finegrasp["venue"] == "arXiv 2025"
    for name in ("explorer.js", "explorer_logic.js"):
        source = (EXPLORER / name).read_text(encoding="utf-8")
        assert "data.methods[0]" not in source, f"{name}: the page chooses no method for anyone"


def test_where_an_option_works_is_the_reach_of_the_records(data):
    """Every fraction, tone and tail of the page is grasppanda.combine.reach's, at the four
    levels and at level 2; no class word is shown."""
    from grasppanda.combine import options as option_lists
    from grasppanda.combine import reach as reaching
    from grasppanda.combine.levels import LEVELS

    records = option_lists.Measured(json.loads(MEASURED.read_text(encoding="utf-8")))
    rows = data["reach"]
    for slot, members in data["options"].items():
        for option in members:
            for level, at in zip(LEVELS, option["reach"], strict=True):
                found = reaching.reach(slot, option["name"], level, records)
                assert rows[at] == [found.fraction, found.tone, found.tail], (slot, option, level)
            found = reaching.level2(slot, option["name"], records, path=LEVEL2)
            assert rows[option["learns"]] == [found.fraction, found.tone, found.tail]
    words = re.compile(r"\b(universal|most|some|specific)\b", re.I)
    assert not any(words.search(" ".join(row)) for row in rows)
    for name in ("explorer.js", "explorer_logic.js"):
        source = (EXPLORER / name).read_text(encoding="utf-8")
        assert not re.search(r"\b[Uu]niversal\b|CLASS_NAMES", source), name
    point_painting = next(o for o in data["options"]["backbone"] if o["name"] == "image-painted")
    assert rows[point_painting["reach"][0]][0] == "no record"


def test_level_two_is_in_a_file_of_its_own(data, chips):
    """Every record compared beside its method's own component says how it went, with the
    first reasons of a worse or diverging run; every other record says why it was not, by
    the rules of grasppanda.combine.reach.level2: no outcome for a swap that did not run at
    level 1 (unstable, failed, refused), for a slot without level 2, or for a run on a
    schedule other than the one its method's reference ran on."""
    from grasppanda.combine import options as option_lists
    from grasppanda.combine import reach as reaching

    learns = json.loads(LEARNS.read_text(encoding="utf-8"))
    level2 = json.loads(LEVEL2.read_text(encoding="utf-8"))
    records = option_lists.Measured(json.loads(MEASURED.read_text(encoding="utf-8")))
    outcomes = {cell["id"]: cell for cell in level2["cells"]}
    schedules = {ref["method"]: ref.get("schedule", "constant") for ref in level2["references"]}
    assert set(learns["cells"]) <= set(outcomes)
    assert set(learns["cells"]) | set(learns["none"]) == {cell["id"] for cell in data["cells"]}
    assert not set(learns["cells"]) & set(learns["none"])
    by_id = {cell["id"]: cell for cell in data["cells"]}
    for cell_id, found in learns["cells"].items():
        words = learns["outcomes"][found[0]]
        expected = outcomes[cell_id]["outcome"]
        assert words == ("did not train" if expected == "fails" else expected)
        assert found[1:] == outcomes[cell_id]["reasons"][:2]
        record = records.get(cell_id)
        assert records.levels(record).states["runs"] == "yes", f"{cell_id}: did not run"
        assert record.get("slot") not in reaching.LEVEL2_NONE, cell_id
        assert outcomes[cell_id].get("schedule", "constant") == schedules[record["method"]]
    # A method reach.level2 leaves out for an option (a scheduler of its own, a swap that
    # did not run there) shows no outcome for it: what every page says.
    option_of = {
        option.cell: (option.slot, (*option.presets, option.name)[0])
        for name in option_lists.methods()
        for option in option_lists.options(name)
        if option.cell and len(option.presets) <= 1
    }
    for cell_id in learns["cells"]:
        if cell_id in option_of:
            excluded = reaching.level2(*option_of[cell_id], records, path=LEVEL2).excluded
            assert records.get(cell_id)["method"] not in excluded, cell_id
    words = learns["none_words"][learns["none"]["recipe/FineGrasp/optimizer/schedulefree-adamw"]]
    assert words == reaching.LEVEL2_EXCLUDED["schedule"]
    left_out = set(outcomes) & set(learns["none"])
    assert left_out, "the level-2 file holds swaps that are not compared"
    for cell_id in left_out:
        words = learns["none_words"][learns["none"][cell_id]]
        if cell_id in SHIPPED_EQUIVALENTS:
            assert learns["none"][cell_id] == 0, "a written-out own component is no swap"
            continue
        assert words.startswith(
            (
                reaching.LEVEL2_EXCLUDED["runs"],
                reaching.LEVEL2_EXCLUDED["schedule"],
                *reaching.LEVEL2_NONE.values(),
            )
        ), (cell_id, words)
    for cell_id, why in learns["none"].items():
        cell = by_id[cell_id]
        assert 0 <= why < len(learns["none_words"])
        if cell["verdict"] == "refused" and why != 0:
            assert why == 1, cell_id
        if cell["verdict"] == "runs-unstable":
            assert learns["none_words"][why].startswith(
                (reaching.LEVEL2_EXCLUDED["runs"], *reaching.LEVEL2_NONE.values())
            ), cell_id
    for method in data["methods"]:
        assert learns["none"][method["reference"]] == 0, "the method as shipped is the reference"
    assert learns["measured_on"].startswith(str(level2["about"]["settings"]["iterations"]))
    # One name for level 2, that of every page.
    assert (learns["name"], learns["header"], learns["meaning"]) == (
        chips.LEVEL2_NAME,
        chips.LEVEL2_HEADER,
        chips.LEVEL2_MEANING,
    )
    for name in ("explorer.js", "explorer_logic.js", "index.md"):
        source = (EXPLORER / name).read_text(encoding="utf-8")
        assert not re.search(r"[Ll]earns like own|\"Like own\"", source), name
    assert "*Trains like shipped*" in (EXPLORER / "index.md").read_text(encoding="utf-8")


def test_what_a_method_ships_is_marked_as_its_own(data, measured):
    """A record that composes to the method as shipped (its reference's config hash), or a
    preset that writes out what the method builds in (``grid.SHIPPED_EQUIVALENTS``), is the
    method's own option: drawn as such in Compare at every level, never counted as a swap."""
    by_id = {record["id"]: record for record in measured["cells"]}
    methods = {method["name"]: method for method in data["methods"]}
    for cell in data["cells"]:
        record = by_id[cell["id"]]
        reference = by_id[methods[cell["method"]]["reference"]]
        own = (
            record["id"] == reference["id"]
            or record["id"] in SHIPPED_EQUIVALENTS
            or (
                bool(record.get("config_hash"))
                and record["config_hash"] == reference["config_hash"]
            )
        )
        assert bool(cell.get("own")) == own, cell["id"]
    assert sum(bool(cell.get("own")) for cell in data["cells"]) > len(data["methods"])


def test_as_shipped_is_what_compose_lists(data):
    """The option a slot ships is the one `grasppanda compose <config> --list` marks
    shipped (HGGD's `accumulate-2` is no change); a part the detector builds in is "built
    in", as on the method pages, not the name of the preset that writes it out."""
    from grasppanda.combine import options as option_lists

    records = option_lists.measured()
    for method in data["methods"]:
        for slot, ships in method["ships"].items():
            if slot != "post":
                listing = option_lists.slot_listing(method["name"], slot, records)
                assert ships["option"] == listing.shipped_option, (method["name"], slot)
    hggd = next(method for method in data["methods"] if method["name"] == "HGGD")
    assert hggd["ships"]["hooks"]["option"] == "accumulate-2"
    assert hggd["ships"]["image_encoder"]["label"] == "built in"
    assert hggd["ships"]["image_encoder"]["option"] == ""


def test_build_names_what_a_method_ships_in_every_slot(data):
    """No "None" for post-processing: a method is evaluated with the variants of its config."""
    from grasppanda.config import Config, resolve_config_path

    for method in data["methods"]:
        resolved = Config.load(resolve_config_path(method["config"])).resolved()
        variants = list(resolved["evaluation"]["variants"])
        assert method["ships"]["post"]["label"] == f"as shipped (variants {', '.join(variants)})"
        assert method["ships"]["post"]["option"] == ""
        for name in variants:
            assert f"{name}: " in method["ships"]["post"]["about"]
    script = (EXPLORER / "explorer.js").read_text(encoding="utf-8")
    assert "ships.plain" in script


def test_links_to_the_documentation_are_internal_on_the_site():
    """The pages the explorer links to are pages of the same site: no external-link icon
    there (only `grasppanda explore`, which links to the hosted site, draws one)."""
    script = (EXPLORER / "explorer.js").read_text(encoding="utf-8")
    assert "external(" not in script
    assert 'docLink("user_guides/inference#run-in-the-browser", "How it works")' in script
    assert "/^https?:/.test(docsBase)" in script


def test_the_page_claims_no_accuracy_and_says_what_is_measured(data, chips):
    page = (EXPLORER / "index.md").read_text(encoding="utf-8")
    caveat = (REPO / "docs" / "includes" / "caveat_probe.md").read_text(encoding="utf-8")
    assert data["no_accuracy"] == " ".join(re.sub(r"<[^>]+>", " ", caveat).split())
    assert data["measured_on"] == chips.measured_on(1)
    assert '--8<-- "docs/includes/caveat_probe.md"' in page, "the caveat of every page"
    assert "**No accuracy**" in page and "*not measured*" in page
    assert '--8<-- "docs/explorer/explorer.json"' in page
    assert "../evidence/index.md" in page, "the questions lead to how it is measured"
    assert page.count('??? note "') <= 4, "at most four questions below the tool"
    script = (EXPLORER / "explorer.js").read_text(encoding="utf-8")
    assert "data.no_accuracy" in script and "data.measured_on" in script
    assert "Not tested together yet" in script
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
    assert set(contracts["methods"]) == {method["name"] for method in data["methods"]}
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
    refused = generator.lookup(contracts, "GSNet", ["grouper=depth-embed"])
    assert refused and not refused["ok"] and "DepthEmbedGroup" in refused["messages"][0]
    assert refused["kind"] == "structural", "a refusal says its kind, as a record does"
    # The scene-encoder recipes name parts HGGD lacks: compose refuses them there.
    hggd = generator.lookup(contracts, "HGGD", ["optimizer=encoder-lr-0.1"])
    assert hggd is not None and not hggd["ok"], hggd
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
    assert DATA.stat().st_size <= 900_000, "explorer.json is over its 900 KB budget"
    for path in (DATA, CONTRACTS, LEARNS):
        text = path.read_text(encoding="utf-8")
        assert "</" not in text and text.count("\n") == 1, "not safe inside a <script> element"
    page = (EXPLORER / "index.md").read_text(encoding="utf-8")
    assert '--8<-- "docs/explorer/explorer_contracts.json"' in page
    assert '--8<-- "docs/explorer/explorer_learns.json"' in page
    from grasppanda.combine.explore import STATIC

    data_files = {path.name for path in EXPLORER.glob("*.json")}
    assert data_files <= set(STATIC), "a data file `grasppanda explore` does not serve"
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
