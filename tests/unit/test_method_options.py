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
import re
import shutil
import subprocess
import sys

import pytest

from grasppanda.cli.main import main
from grasppanda.combine import options as option_lists
from grasppanda.combine.grid import FAMILIES, METHODS, cells
from grasppanda.combine.levels import BELOW_REFERENCE, CAVEAT, LEVELS
from grasppanda.config import Config
from grasppanda.utils.imports import EXTRA_BUILDS

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
    # The encoder presets of a recipe do not fit HGGD, for a reason that is said.
    hooks = option_lists.listing("HGGD", "hooks")[0]
    freeze = next(e for e in hooks.entries if e.option.name == "freeze-encoder")
    assert freeze.status == "does not fit" and freeze.kind == "structural"
    assert "model.backbone" in freeze.reason and not freeze.record


def test_the_seed_based_recipes_are_refused_under_hggd(capsys):
    """HGGD's page counts the recipe presets that name the scene encoder as refused: compose
    refuses each one under HGGD as incompatible by design, and the one that needs a scene
    encoder in the same request (``RECIPE_REQUIRES``) needs a slot HGGD does not have."""
    from grasppanda.combine.grid import RECIPE_ENCODER_PRESETS, RECIPE_REQUIRES

    config = FAMILIES["HGGD"]
    for key in RECIPE_ENCODER_PRESETS:
        slot, name = key.split("/")
        listed = option_lists.listing("HGGD", slot)[0]
        entry = next(e for e in listed.entries if e.option.name == name)
        assert entry.status == "does not fit" and entry.kind == "structural", key
        needed = RECIPE_REQUIRES.get(key, ())
        if needed:
            slots = {item.partition("=")[0] for item in needed}
            assert not slots & set(option_lists.method_slots("HGGD")), key
            request = [arg for item in (key.replace("/", "="), *needed) for arg in ("--set", item)]
            assert main(["compose", config, *request]) == 2, key  # no such slot under HGGD
            continue
        assert main(["compose", config, "--set", f"{slot}={name}", "--json"]) == 1, key
        report = json.loads(capsys.readouterr().out)
        assert not report["accepted"] and set(report["refusal_kinds"]) == {"structural"}, key
        assert all("`backbone`" in message for message in report["refusal"]), key
    capsys.readouterr()


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
            # by name, a base before its variants, as the slot pages sort them
            assert listed == sorted(wanted, key=str.lower), (method["name"], slot)
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
        assert f"\n# {method}\n" in page
        assert page.count(generator.CAVEAT_SNIPPET) == 1, "one probe caveat per page"
        # the template: head, run, numbers, the slots, how it differs, next, in this order
        sections = [
            f"## Run {method} in five minutes {{#{generator.RUN_ANCHOR}}}",
            "## Numbers",
            "## The options of every slot {#the-slots}",
            f'??? note "{generator.DIFFERENCES}"',
            "## Next",
        ]
        where = [page.index(heading) for heading in sections]
        assert where == sorted(where), method
        assert f"methods/{stem}.md" in nav, f"{stem} is not in the nav of mkdocs.yml"
        readme = (REPO / "configs" / stem / "README.md").read_text(encoding="utf-8")
        # read on GitHub, a README links the built page (GitHub shows the MkDocs source)
        site = f"https://daeda1used.github.io/GraspPanda/methods/{stem}/"
        assert f"{site}#run-in-five-minutes" in readme and f"{site}#the-slots" in readme, (
            f"configs/{stem}/README.md does not link its page"
        )
        assert f"grasppanda compose {config} --list" in readme
        # every slot keeps its anchor (the slot pages link it), inside the fold of its options
        for slot in option_lists.method_slots(method):
            assert f"{{: #{generator.anchor(slot)} }}" in page, (method, slot)
            assert f"](#{generator.anchor(slot)})" in page, (method, slot)
        # the explorer opens on this method (raw HTML: directory URLs, no MkDocs anchor)
        assert f'href="../../explorer/#m={method}"' in page
        assert "equal to the reference ✓" in page and generator.PARITY[method] in page
    for stem in generator.SLOT_PAGE_SLOTS:
        text = (REPO / "docs" / "slots" / f"{stem}.md").read_text(encoding="utf-8")
        for mark in (generator.HEAD_BEGIN, generator.BEGIN, generator.OPTIONS_BEGIN):
            assert mark in text, (stem, mark)
        assert text.count(generator.CAVEAT_SNIPPET) == 1, stem


def test_a_method_page_says_where_each_option_runs_and_what_happened_under_it():
    """The chips are those of docs_chips (the fraction over every method the option is a swap
    under, its tail as text too), and the last column is the method's own record."""
    generator, chips = tool("make_method_pages"), tool("docs_chips")
    economic = (REPO / "docs" / "methods" / "economicgrasp.md").read_text(encoding="utf-8")
    row = next(line for line in economic.splitlines() if ">objectness-view</code> |" in line)
    assert chips.reach_chip("view", "objectness-view") in row
    assert chips.level2_chip("view", "objectness-view") in row
    # who is left out: under the chip when short, else in the list below the table
    tail = chips.reach("view", "objectness-view").tail
    assert len(tail) > generator.TAIL_IN_CELL and f"<small>{tail}</small>" not in row
    assert f"- `objectness-view`, runs: {tail}." in economic
    short = chips.reach("backbone", "sparse-unet14").tail
    assert len(short) <= generator.TAIL_IN_CELL
    # a refusal says its kind and the first sentence of the record's message
    view = option_lists.listing("EconomicGrasp", "view")[0]
    refused = next(e for e in view.entries if e.option.name == "objectness-view")
    first = refused.refusal[0]["message"].split(". ")[0].rstrip(".")
    assert row.rstrip(" |").endswith(f"refused (incompatible by design): {first}.")
    assert "ObjectnessViewStage computes its targets from 'dense' matched labels" in row
    # an option the method ships is named so; post-processing has no level 2
    gsnet = (REPO / "docs" / "methods" / "gsnet.md").read_text(encoding="utf-8")
    shipped = next(line for line in gsnet.splitlines() if ">sparse-unet14</code> |" in line)
    assert shipped.rstrip(" |").endswith("shipped") and f"<small>{short}</small>" in shipped
    assert f"*{chips.LEVEL2_HEADER}*: — (post-processing trains nothing" in gsnet
    assert f"| {chips.LEVEL2_HEADER} |" in gsnet and "Learns like the reference" not in gsnet
    assert chips.LEVEL2_MEANING in gsnet
    # a loss preset is one row, with the terms it was tried on under this method
    focal = next(line for line in gsnet.splitlines() if ">focal</code> |" in line)
    assert "`proposal.objectness_loss`" in focal
    assert "refused (incompatible by design): SWADHead.width_loss is a `value` term" in focal
    # HGGD: a refusal by consent is said so, and the seed-based recipes say why they are refused
    hggd = (REPO / "docs" / "methods" / "hggd.md").read_text(encoding="utf-8")
    radio = next(line for line in hggd.splitlines() if ">hf-radio-v3-base</code> |" in line)
    assert "| needs your consent: model.image_encoder: HFImageEncoder:" in radio
    assert "`--opts model.image_encoder.accept_remote_code=true`" in radio  # how to consent
    for name in ("encoder-lr-0.1", "layer-decay", "freeze-encoder", "finetune-encoder"):
        row = next(line for line in hggd.splitlines() if f">{name}</code> |" in line)
        assert row.rstrip(" |").endswith(
            "refused (incompatible by design): it is written for the scene encoder of the "
            "seed-based skeleton (`model.backbone`), which HGGD does not have."
        ), name
    onecycle = next(line for line in hggd.splitlines() if ">ptv3-adamw-onecycle</code> |" in line)
    assert onecycle.rstrip(" |").endswith(
        "it needs `--set backbone=ptv3` in the same request, and HGGD has no slot `backbone`."
    )
    # the numbers: at most four rows, from the metafile
    for method in (*METHODS, *FAMILIES):
        page = (REPO / "docs" / "methods" / f"{generator.page_of(method)}.md").read_text(
            encoding="utf-8"
        )
        numbers = page.split("## Numbers")[1].split("??? note")[0]
        rows = [line for line in numbers.splitlines() if line.startswith("| ")][1:]
        assert 1 <= len(rows) <= 4, method
        _, model = generator.zoo_entry(method)
        shown = {line.split(" | ")[2] for line in rows}
        assert shown <= {f"{r.ap['seen']:.2f}" for r in model.results if "seen" in r.ap}


def does_it_run() -> dict[tuple[str, str], str]:
    """(slot title, method) → the fraction of *Does it run?* at ``runs``, from the generator
    of that page (``make_combination_docs.answer_matrix``), not from its committed copy."""
    docs = tool("make_combination_docs")
    found = docs.where_options_work(docs.Records(docs.measured()["cells"]))
    lines = docs.answer_matrix(found, "runs")
    names = [cell.strip() for cell in lines[0].strip("|").split(" | ")][1:]
    shown = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split(" | ")]
        title = re.match(r"\[([^\]]+)\]", cells[0]).group(1).replace(" ", " ")
        for method, cell in zip(names, cells[1:], strict=True):
            shown[(title, method)] = re.sub(r"<[^>]+>", "", cell)
    return shown


def test_the_summary_of_the_slots_counts_what_composes_what_is_refused_and_what_ran():
    """Composes / Refused come from the method's own records: an option held back for
    consent is neither, an option without a record is said so, and the options counted are
    those of ``docs_chips.options_of`` (no adapter, no point painting). *Runs here* counts as
    *Does it run?* does (``reach``: swaps only, refusals in the denominator), so the two
    pages show the same fraction for a method and a slot."""
    chips = tool("docs_chips")
    records = option_lists.measured()
    answer = does_it_run()
    for method, config in {**METHODS, **FAMILIES}.items():
        page = (REPO / "docs" / "methods" / f"{config.split('/')[0]}.md").read_text(
            encoding="utf-8"
        )
        section = page.split("## The options of every slot {#the-slots}")[1]
        summary = section.split("??? note")[0]
        # on a phone, Composes and Refused (columns 3-4) give way (gp-slots in extra.css)
        assert '<div class="gp-matrix gp-slots" markdown>' in summary
        assert "| Slot | Shipped | Composes | Refused | Runs here | Explorer |" in summary
        assert f"Every option that fits {method}" in summary
        assert "was composed under it; those accepted were probed on real frames" in summary
        assert "Every option but" not in page
        assert "under every method that has the slot" not in page
        shown = {}
        for line in summary.splitlines():
            found = re.match(r"\| \[[^\]]+\]\(#([a-z-]+)\)", line)
            if found:
                shown[found.group(1)] = [cell.strip() for cell in line.strip("|").split(" | ")]
        refused_rows = 0
        for slot in option_lists.method_slots(method):
            listed = option_lists.listing(method, slot, records)[0]
            names = set(chips.options_of(slot))
            entries = [e for e in listed.entries if slot == "loss" or e.option.name in names]
            composes = [e for e in entries if e.status == "fits"]
            refused = [e for e in entries if e.status == "does not fit" and e.kind != "policy"]
            consent = [e for e in entries if e.status == "does not fit" and e.kind == "policy"]
            none = [e for e in entries if e.status == "not probed"]
            cells = shown[slot.replace("_", "-")]
            assert cells[2].split(" · ")[0] == str(len(composes)), (method, slot)
            assert ("needs consent" in cells[2]) == bool(consent), (method, slot)
            assert ("no record" in cells[2]) == bool(none), (method, slot)
            assert cells[3] == str(len(refused)), (method, slot)
            assert cells[4] == answer[(chips.SLOT_TITLES[slot], method)], (method, slot)
            if slot == "loss":
                continue
            refused_rows += len(refused)
            if chips.SLOT_PAGES[slot] != chips.SLOT_PAGES["recipe"]:  # a fold of its own
                total = chips.option_count(slot)
                count = (
                    f"{len(entries)} options"
                    if len(entries) == total
                    else f"{len(entries)} of {total} options"
                )
                assert f'??? note "{chips.SLOT_TITLES[slot]}: {count}"' in section, (method, slot)
        outside = section  # every row but those of the loss presets (one row, several terms)
        if '??? note "Loss:' in section:
            before, after = section.split('??? note "Loss:', 1)
            outside = before + after.split('??? note "', 1)[1]
        assert outside.count("| refused (") == refused_rows, method
    hggd = (REPO / "docs" / "methods" / "hggd.md").read_text(encoding="utf-8")
    assert "| 37 · 1 needs consent | 0 |" in hggd
    # the examples of the review: swaps only, a refusal in the denominator
    baseline = (REPO / "docs" / "methods" / "graspnet_baseline.md").read_text(encoding="utf-8")
    assert "| [Seed proposal](#proposal) | `SubsampledSeeds` | 22 | 2 | 21/23 |" in baseline
    assert "| [Head bundle](#head) | `operation` | 8 | 2 | 7/9 |" in baseline
    # the recipes of the seed-based skeleton are refused under HGGD with no record there:
    # not counted, and the page says so
    uncounted = hggd.split("Not counted either: ")[1].split(", refused under HGGD")[0]
    for name in ("encoder-lr-0.1", "layer-decay", "freeze-encoder", "finetune-encoder"):
        assert f"`{name}`" in uncounted, name


def test_a_level_the_method_as_shipped_misses_too_is_undecided_under_it():
    """*Under <method>*: a level the method as shipped does not reach either
    (``levels.BELOW_REFERENCE``) reads ``<level> ? (as shipped)``, as the explorer says
    "reference misses it too", never ✓; why stands once under the table, the legend says
    so, and the fold says what level 2 needs and links the level-2 page by its title."""
    generator = tool("make_method_pages")
    records = option_lists.measured()
    title = generator.level2_title()
    assert title == "Does it train like the shipped component?"
    undecided = set()
    for method, config in {**METHODS, **FAMILIES}.items():
        for slot in option_lists.method_slots(method):
            for entry in option_lists.listing(method, slot, records)[0].entries:
                if entry.levels is None or entry.levels.states.get("runs") != "yes":
                    continue
                said = generator._here(method, entry, records)
                states = entry.levels.states
                for level in ("learns", "predicts"):
                    if states.get(level) != BELOW_REFERENCE or entry.shipped:
                        continue
                    if level == "predicts" and states.get("learns") == "no":
                        assert "learns ✗" in said  # the row says why it has no ✓
                        continue
                    assert generator.AS_SHIPPED.format(level=level) in said, (method, entry)
                    undecided.add((method, level))
                if any(states.get(level) != "yes" for level in LEVELS):
                    assert said != "✓", (method, entry.option.name)
        page = (REPO / "docs" / "methods" / f"{config.split('/')[0]}.md").read_text(
            encoding="utf-8"
        )
        # why, once per table: the reasons of the method's own record
        reference = records.levels(records.reference(method))
        for level in ("learns", "predicts"):
            note = f"*{level} ? (as shipped)*: {method} as shipped does not reach *{level}*"
            assert (note in page) == ((method, level) in undecided), (method, level)
            if note in page:
                assert f"either ({'; '.join(reference.why[level][:2])})" in page
        assert f"*level ? (as shipped)* when {method} as shipped does not reach" in page
        assert generator.level2_needs(records) in page
        assert generator.level2_needs(records).startswith("Level 2 needs *runs*, not *learns*")
        assert f"[{title}](../advanced_guides/learns_like_reference.md)" in page
    # FineGrasp and Scale-Balanced-Grasp as shipped predict nothing useful; HGGD as shipped
    # has a flat term
    assert undecided == {
        ("Scale-Balanced-Grasp", "predicts"),
        ("FineGrasp", "predicts"),
        ("HGGD", "learns"),
    }
    # the rows of the review: no grasp (or one score) under the method as shipped either
    finegrasp = (REPO / "docs" / "methods" / "finegrasp.md").read_text(encoding="utf-8")
    row = next(line for line in finegrasp.splitlines() if ">dgcnn-static</code> |" in line)
    assert row.rstrip(" |").endswith("| predicts ? (as shipped)")
    sbg = (REPO / "docs" / "methods" / "scale_balanced_grasp.md").read_text(encoding="utf-8")
    row = next(line for line in sbg.splitlines() if ">pointnet2</code> |" in line)
    assert row.rstrip(" |").endswith("| predicts ? (as shipped)")
    assert "*predicts ? (as shipped)*: Scale-Balanced-Grasp as shipped does not reach " in sbg
    # every option that fits was composed: a loss preset with no term of its kind is said so
    baseline = (REPO / "docs" / "methods" / "graspnet_baseline.md").read_text(encoding="utf-8")
    assert "Every option that fits GraspNet-baseline, except `target-mask`, was composed" in (
        baseline
    )
    assert "Not tried under GraspNet-baseline: `varifocal`, `bce` fit `unit` terms only" in (
        baseline
    )


def test_the_pages_say_whose_reference_code_and_weights_they_are():
    """A reference implementation the metafile calls third-party (GSNet's) is never called
    the authors': its page, its card and the chooser say whose code and weights they are."""
    generator = tool("make_method_pages")
    index = (REPO / "docs" / "methods" / "index.md").read_text(encoding="utf-8")
    third = []
    for method, config in {**METHODS, **FAMILIES}.items():
        upstream = generator.metafile_of(method).upstream
        page = (REPO / "docs" / "methods" / f"{config.split('/')[0]}.md").read_text(
            encoding="utf-8"
        )
        run = page.split("## Numbers")[0]
        if "third-party" in upstream["license"]:
            third.append(method)
            assert "authors'" not in page and "authors&#x27;" not in page, method
            assert upstream["repository"].split("/")[-1] in run, method
            assert "(third-party)" in run, method
            card = index.split(f"**[{method}](")[1].split("\n-   ")[0]
            assert "authors" not in card, method
        else:
            assert "the authors' checkpoint" in run, method
    assert third == ["GSNet"]
    assert "authors" not in index.split("## Seed-based")[0]
    assert "authors" not in (REPO / "docs" / "includes" / "choose_method.md").read_text("utf-8")
    assert "released weights" in index


def test_the_methods_overview_and_the_chooser_are_computed():
    generator, chips = tool("make_method_pages"), tool("docs_chips")
    converter = tool("convert_checkpoint")
    index = (REPO / "docs" / "methods" / "index.md").read_text(encoding="utf-8")
    assert index.startswith("---\ntitle: Methods\n")
    # the demo time is the chooser's Demo column (included below), not shown twice
    capability = next(line for line in index.splitlines() if line.startswith("| Method |"))
    assert "demo" not in capability.lower() and "Five-minute" not in index
    assert "<!-- Generated by tools/make_method_pages.py" in index
    assert "## Seed-based skeleton · 5 methods" in index and "## Own family" in index
    assert '--8<-- "docs/includes/choose_method.md"' in index
    assert "{: #what-each-method-ships }" in index  # the method READMEs link it
    # the cards in the order of the papers, by family
    order = [index.index(f"**[{info.name}]({info.folder}.md)**") for info in chips.methods()]
    assert order == sorted(order)
    # the capability table agrees with the names of the config files
    for info in chips.methods():
        row = next(line for line in index.splitlines() if line.startswith(f"| [{info.name}]("))
        names = [path.stem for path in (REPO / "configs" / info.folder).glob("*.yaml")]
        gc6d = row.split(" | ")[4]
        assert ("transfer" in gc6d) == any("-to-gc6d" in name for name in names), info.name
        assert ("training" in gc6d) == any("_gc6d-" in name for name in names), info.name
        assert ("Kinect" in row.split(" | ")[3]) == any("g1b-kinect_" in name for name in names)
    chooser = (REPO / "docs" / "includes" / "choose_method.md").read_text(encoding="utf-8")
    # its links work from docs/methods/ and docs/get_started/ alike
    targets = re.findall(r"\]\(([^)]+)\)", chooser)
    assert targets and all(re.fullmatch(r"\.\./methods/[a-z_]+\.md", t) for t in targets)
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    data = generator.five_minutes()
    assert '<div class="gp-matrix" markdown>' in chooser
    header = next(line for line in chooser.splitlines() if line.startswith("| Method |"))
    assert header == "| Method | Reads | Extra | Weights<br>from | Licence | Demo |"
    assert "Format" not in chooser and "--source-format" not in chooser
    plain, colour = [], []
    for info in chips.methods():
        row = next(line for line in chooser.splitlines() if f"[{info.name}](" in line)
        cells = [cell.strip() for cell in row.strip("|").split(" | ")]
        assert cells[1] == info.inputs
        needs = generator.config_extras(info.config)
        assert cells[2] == (", ".join(f"`{extra}`" for extra in needs) or "none")
        for extra in needs:  # an extra of pyproject.toml, or a library with one per build
            builds = EXTRA_BUILDS.get(extra, (extra,))
            assert all(re.search(rf"^{build} = ", pyproject, re.M) for build in builds), extra
        assert ("spconv" in needs) == (info.name in ("GSNet", "EconomicGrasp", "FineGrasp"))
        plain += [] if needs else [info.name]
        colour += [info.name] if info.inputs == "RGB-D" else []
        # the source format is shown where it is typed: step 1 of the method page
        _, model = generator.zoo_entry(info.name)
        source = re.search(r"--source-format (\S+)", generator.downloadable(model).convert)
        assert source and source.group(1) in converter.SOURCE_FORMATS, info.name
        page = (REPO / "docs" / "methods" / f"{info.folder}.md").read_text(encoding="utf-8")
        step = page.split("### 1. Get the weights")[1].split("### 2.")[0]
        assert f"--source-format {source.group(1)}" in step, info.name
        demo = data["methods"][info.name]["runs"][0]["demo"][0]
        assert cells[5] == f"{demo['wall_s']:.1f} s"
    # which first: the methods that need no extra install, and the one that reads colour too
    first = next(line for line in chooser.splitlines() if "no extra install" in line)
    assert first.startswith(generator._names(plain) + " need no extra install.")
    assert plain == ["GraspNet-baseline", "Scale-Balanced-Grasp", "HGGD"]
    assert colour == ["HGGD"] and "HGGD reads a colour image registered to the depth" in first
    # no claim about a CPU without a record of one: every five-minute run had a GPU
    machines = [
        data["machines"][demo["machine"]]
        for entry in data["methods"].values()
        for run in entry["runs"]
        for demo in run["demo"]
    ]
    assert all(machine.get("gpu") for machine in machines)
    assert "no run was timed on a CPU" in chooser and chooser.count("CPU") == 1


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
    # The method page: the last column says what happened under the method, from its record.
    pages = tool("make_method_pages")
    page = pages.page("GraspNet-baseline", records)
    row = next(line for line in page.splitlines() if ">two-vector</code> |" in line)
    assert row.rstrip(" |").endswith("learns ✗: view_frame rose 19 %; score rose 74 %")
    gsnet = pages.page("GSNet", records)
    row = next(line for line in gsnet.splitlines() if ">pointnext</code> |" in line)
    assert row.rstrip(" |").endswith("predicts ✗: no grasp on any frame")
    fine = pages.page("FineGrasp", records)
    row = next(line for line in fine.splitlines() if ">collision</code> |" in line)
    assert row.rstrip(" |").endswith("unstable once, not repeated")
    # The measured page and the explorer.
    docs, explorer = tool("make_combination_docs"), tool("make_explorer")
    monkeypatch.setattr(docs, "DATA", target)
    measured_page = docs.page()
    assert "does not learn: view_frame rose 19 %; score rose 74 %" in measured_page
    assert "does not predict: no grasp on any frame" in measured_page
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
    answer = page.split("## The answer, slot by method")[1].split("## How it was judged")[0]
    assert '=== "learns"' in answer and '=== "predicts"' in answer
    assert "not checked by the current judge" not in page
    records = docs.Records(payload["cells"])
    # The slot index shows nothing measured, so records of judge 2 leave it as it is.
    index = docs.slots_index(records)
    assert index == (REPO / "docs" / "slots" / "index.md").read_text(encoding="utf-8")
    found = {v: c for _, v, c in docs.slot_classes("proposal", records, "stage")}
    assert found["dense-fps"]["learns"].decided == 5
    pages = tool("make_method_pages")
    page = pages.page("GSNet", option_lists.Measured(payload))
    proposal = page.split('??? note "Seed proposal:')[1].split('??? note "View:')[0]
    rows = [line for line in proposal.splitlines() if "</code> |" in line]
    assert len(rows) == len(option_lists.options("GSNet", "proposal"))
    # every term fell and every frame has scores: no level is missed under GSNet
    assert all(row.rstrip(" |").endswith(("✓", "shipped")) for row in rows), rows
