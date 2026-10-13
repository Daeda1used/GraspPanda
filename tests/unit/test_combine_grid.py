"""The measured grid: what is probed, what is committed, and the page generated from it.

Nothing is probed here (that needs the dataset and a GPU, see
``tests/integration/test_combine_grid.py``). These tests are the tripwire between the
registries and the committed measurements: a registered scene encoder or stage whose
cells have no record makes the suite fail and says which commands produce one, and so
does a record that no longer is what ``grasppanda compose`` writes today.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter

import pytest

from grasppanda.cli.components import collect
from grasppanda.combine.compose import compose, is_adapter
from grasppanda.combine.grid import (
    FAMILIES,
    METHODS,
    STAGE_SLOTS,
    cells,
    family_slots,
    stage_variants,
)
from grasppanda.combine.presets import presets
from grasppanda.combine.probe import VERDICTS
from grasppanda.config import Config, available_configs
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, SeedGraspDetector
from grasppanda.registry import BACKBONES, DETECTORS

from .conftest import REPO

DATA = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
PAGE = REPO / "docs" / "advanced_guides" / "measured_combinations.md"

#: Cells of the grid that need no record, each with the reason. Empty: every group below
#: was probed. A cell goes here only with a reason a reader of the page can check.
NOT_PROBED: dict[str, str] = {}
#: Groups every registered component must have records in. The sets of stages are probed
#: too, but a new stage does not have to wait for its sets: the page counts what is missing.
REQUIRED_GROUPS = ("reference", "encoder", "stage", "family", "family-slot", "loss", "cross")
HOW_TO_PROBE = (
    "Probe them on a machine with GraspNet-1Billion and a GPU, then export and regenerate:\n"
    "  python -m tools.probe_grid run --name mine --only '<cell id>' ... \\\n"
    "      --scale-statistic-sha <SHA-256 of the imported reference histogram>\n"
    "(Cells under Scale-Balanced-Grasp read its width histogram; without the SHA they end as\n"
    "`wired`, which is no measurement. A cell the config check refuses needs no GPU.)\n"
    "  python -m tools.probe_grid export --name mine\n"
    "  python -m tools.make_combination_docs\n"
)


def tool(name: str):
    """A module of ``tools/``, loaded by path (the directory is not a package on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def docs_tool():
    return tool("make_combination_docs")


@pytest.fixture(scope="module")
def measured() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def by_id(measured) -> dict:
    return {record["id"]: record for record in measured["cells"]}


def in_tree(registry: str) -> set[str]:
    return {entry.name for entry in collect([registry], in_tree_only=True)}


# ------------------------------------------------------------------------- the grid


def test_the_grid_follows_the_registries_and_the_shipped_configs():
    grid = cells()
    assert len({cell.id for cell in grid}) == len(grid)
    by_group = Counter(cell.group for cell in grid)
    assert set(by_group) == {
        "reference",
        "encoder",
        "stage",
        "stage-set",
        "family",
        "family-slot",
        "post",
        "loss",
        "recipe",
        "cross",
    }
    # The methods are the seed-based detectors' own shipped configs, HGGD is the other family.
    shipped = set(available_configs())
    assert set(METHODS.values()) | set(FAMILIES.values()) <= shipped
    seed_based = {
        name for name in in_tree("DETECTORS") if issubclass(DETECTORS.get(name), SeedGraspDetector)
    }
    assert {Config.load(base)["model"]["type"] for base in METHODS.values()} == seed_based
    others = in_tree("DETECTORS") - seed_based
    assert {Config.load(base)["model"]["type"] for base in FAMILIES.values()} == others
    # Every registered encoder under every method; adapters are not encoders.
    encoders = {name for name in in_tree("BACKBONES") if not is_adapter(BACKBONES.get(name))}
    assert {(cell.component, cell.method) for cell in grid if cell.group == "encoder"} == {
        (encoder, method) for encoder in encoders for method in METHODS
    }
    # Every registered stage, in every variant, under every method. (Other test modules
    # register stages of their own; the grid is about the components of the package.)
    for slot in STAGE_SLOTS:
        members = [cell for cell in grid if cell.group == "stage" and cell.slot == slot]
        shipped_stages = in_tree(STAGE_REGISTRIES[slot].name)
        assert {cell.component for cell in members} == shipped_stages
        assert {(cell.variant, cell.method) for cell in members} == {
            (choice, method)
            for choice, component in stage_variants(slot)
            if component in shipped_stages
            for method in METHODS
        }
    # A stage set states all five stages and an encoder that needs no adapter.
    for cell in grid:
        if cell.group == "stage-set":
            assert [item.split("=")[0] for item in cell.sets] == ["backbone", *STAGE_SLOTS]
    # A cross cell replaces an encoder together with options of other slots.
    for cell in grid:
        if cell.group == "cross":
            slots = [item.split("=")[0] for item in cell.sets]
            assert slots[0] in ("backbone", "image_encoder") and len(set(slots)) >= 2, cell.id
    assert {cell.method for cell in grid if cell.group == "cross"} == {*METHODS, *FAMILIES}
    assert cells(("family",)) == [cell for cell in grid if cell.group == "family"]
    # Every preset of every slot of another detector family, under that family.
    assert {(cell.method, cell.slot, cell.variant) for cell in cells(("family-slot",))} == {
        (method, slot, name)
        for method in FAMILIES
        for slot in family_slots(method)
        for name in presets().get(slot, {})
    }


# --------------------------------------------------------------------- the tripwire


def test_every_cell_of_a_registered_component_has_a_measured_record(measured, by_id):
    """Register an encoder or a stage and this fails until it has been probed."""
    wanted = [cell for cell in cells(REQUIRED_GROUPS) if cell.id not in NOT_PROBED]
    missing = [cell.id for cell in wanted if cell.id not in by_id]
    assert not missing, (
        f"{len(missing)} cell(s) of the grid have no measured record, e.g. {missing[:5]}. "
        f"{HOW_TO_PROBE}or list the cell in NOT_PROBED of this file with the reason."
    )
    grid = {cell.id for cell in cells()}
    stale = set(by_id) - grid
    assert not stale, f"records of cells that are no longer in the grid: {sorted(stale)[:5]}"
    assert set(NOT_PROBED) <= grid
    assert set(measured["about"]["not_probed"]) == grid - set(by_id)
    # An adapter has no cells of its own: it is measured where compose inserted it.
    # (Wrappers that compose never inserts, point painting, are probed by hand.)
    adapters = {
        name
        for name in in_tree("BACKBONES")
        if is_adapter(BACKBONES.get(name)) and getattr(BACKBONES.get(name), "compose_inserts", True)
    }
    working = {record.get("adapter") for record in measured["cells"] if record["verdict"] == "runs"}
    assert adapters <= working, (
        f"no measured cell runs through the adapter(s) {sorted(adapters - working)}: probe an "
        "encoder that needs it"
    )


@pytest.mark.slow
def test_every_recorded_cell_is_what_compose_writes_today(by_id):
    """A record belongs to a config. Change a preset, a shipped config or a contract rule
    and the cells whose composed config (or refusal) changed must be probed again.

    A refusal is compared by the first sentence of each message, which states the
    conflict and is what the page prints; the sentences after it list the registered
    components that would fit, and those lists grow with the registries."""
    first = docs_tool()._first_sentence
    outdated = []
    for cell in cells():
        record = by_id.get(cell.id)
        if record is None:
            continue
        if not cell.sets:
            current = Config.load(cell.base).digest()
        else:
            result = compose(cell.base, cell.sets)
            current = result.config_hash
            if record["verdict"] == "refused" and "probe" not in record:
                said = [first(message) for message in result.refusal()]
                if result.accepted or said != [first(message) for message in record["refusal"]]:
                    outdated.append(cell.id)
                continue
        if current is None or current[:16] != record["config_hash"]:
            outdated.append(cell.id)
    assert not outdated, (
        f"{len(outdated)} measured cell(s) are not what compose writes today, e.g. "
        f"{outdated[:5]}. {HOW_TO_PROBE}(`--only` takes several ids and glob patterns.)\n"
        "Records made before the config hash stopped depending on the installed PyTorch are "
        "rewritten, without probing, by `python -m tools.probe_grid rehash` on the machine "
        "that measured them."
    )


def test_a_record_is_a_run_with_its_commands_and_a_verdict_of_the_vocabulary(measured, by_id):
    about = measured["about"]
    assert about["verdicts"] == VERDICTS
    assert about["probe"] == {"iterations": 20, "frames": 8}
    assert "shared" in about["shared_machine"]
    assert about["runs"] and {record["run"] for record in measured["cells"]} == set(about["runs"])
    for name, run in about["runs"].items():
        assert re.fullmatch(r"\d{4}-\d\d-\d\d", run["date"]) and name.startswith(run["date"])
        assert run["torch"] and run["gpu"] and run["probes_at_a_time"] <= 2
    for cell in cells():
        record = by_id.get(cell.id)
        if record is None:
            continue
        assert record["verdict"] in VERDICTS and record["verdict"] != "wired", cell.id
        assert (record["method"], record.get("sets", [])) == (cell.method, list(cell.sets))
        file = cell.id.replace("/", "__") + ".yaml"
        if cell.sets:  # composed: the command that made the config is recorded
            assert record["compose"].startswith(f"grasppanda compose {cell.base} --set ")
            assert record["compose"].endswith(f" -o {file}"), cell.id
        if record["verdict"] == "refused":
            assert record["refusal"] and "probe" not in record, cell.id
            assert record["config_hash"] is None
            continue
        assert len(record["config_hash"]) == 16
        if "probed_as" in record:  # the same config as another cell: one experiment
            owner = by_id[record["probed_as"]]
            assert "probe" in owner and "probed_as" not in owner, cell.id
            assert owner["config_hash"] == record["config_hash"], cell.id
            assert owner["verdict"] == record["verdict"] and "losses" not in record, cell.id
            continue
        # Everything else was probed, by a recorded command, on real frames.
        probe = record["probe"]
        assert probe.startswith(f"grasppanda probe {file if cell.sets else cell.base} "), cell.id
        assert "--iters 20 --frames 8" in probe and "--synthetic" not in probe
        assert "fallback" not in record, cell.id  # never a synthetic stand-in
        if record["verdict"] == "fails":
            assert record["failure"]["phase"] and record["failure"]["message"], cell.id
        else:
            assert record["iterations"] >= 1 and record["frames"] == 8, cell.id
            assert record["parameters"] > 0 and "loss" in record["losses"], cell.id
            assert len(record["grasps_per_frame"]) == 2, cell.id
            assert (record["verdict"] == "runs-unstable") == ("unstable" in record), cell.id
        # Machine-dependent numbers are kept apart from the others.
        assert not {"seconds_per_iteration", "peak_gpu_mib", "seconds"} & set(record), cell.id
    text = DATA.read_text(encoding="utf-8")
    # No accuracy of any kind is recorded, and no path of the machine that ran the grid.
    assert '"AP' not in text and "average precision" not in text.lower()
    assert "/home/" not in text and "/tmp/" not in text and "probe_grid/" not in text
    # The file is the runner's own serialisation: one cell per line, nothing edited by hand.
    assert tool("probe_grid").dump(measured) == text
    assert sum(line.startswith('  {"id": ') for line in text.splitlines()) == len(by_id)
    # Compact: about a kilobyte per probed cell. The list of the stage sets that were not
    # probed is no record; it grows with the product of the stage variants.
    unprobed = len(json.dumps(measured["about"]["not_probed"]))
    assert len(text) - unprobed < 1_300 * len(by_id)


def test_a_probe_that_ran_out_of_gpu_memory_never_stands_for_its_config():
    """The GPU is shared: an out-of-memory failure says something about the machine at that
    moment, so the runner probes the config again instead of repeating the old report."""
    grid = tool("probe_grid")

    def failed(phase, error, message):
        failure = {"phase": phase, "component": "", "error": error, "message": message}
        return {
            "probe": {"verdict": "fails", "detail": f"at {phase}: {message}", "failure": failure}
        }

    allocator = failed(
        "loss", "OutOfMemoryError", "CUDA out of memory. Tried to allocate 1006.00 MiB."
    )
    library = failed(
        "backward", "RuntimeError", "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling"
    )
    contract = failed(
        "build", "ValueError", "the gated seed residual adds 72-channel seed features"
    )
    assert not grid._conclusive(allocator) and not grid._conclusive(library)
    assert grid._conclusive(contract) and grid._conclusive({"probe": {"verdict": "runs"}})


def test_a_probe_that_runs_out_of_memory_is_not_recorded(tmp_path, monkeypatch, capsys):
    """On a shared GPU "out of memory" says nothing about a config: the runner probes the
    config again alone and, if memory stays short, leaves the cell without a record."""
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    starved = {
        "verdict": "fails",
        "failure": {"phase": "loss", "error": "OutOfMemoryError", "message": "CUDA out of memory."},
        "seconds": 1.0,
    }
    fine = {"verdict": "runs", "iterations": 2, "seconds": 1.0}
    outcomes: list[dict] = []
    calls: list[str] = []

    def probe_config(config, options, tag, name, args):
        calls.append(tag)
        return "grasppanda probe x --iters 2 --frames 8", dict(outcomes.pop(0))

    monkeypatch.setattr(runner, "probe_config", probe_config)
    arguments = ["run", "--name", "shared", "--only", "reference/GSNet", "--iters", "2"]
    arguments += ["--retries", "2", "--wait", "0", "--commit", "abcdef0123"]
    # Out of memory next to another probe, and twice more alone: no record, status 1.
    outcomes[:] = [starved, starved, starved]
    assert runner.main(arguments) == 1
    assert len(calls) == 3 and runner.load("shared") == {}
    assert "not recorded" in capsys.readouterr().out
    # The next run picks the cell up. Out of memory once, then memory: recorded, with a note.
    outcomes[:] = [starved, fine]
    assert runner.main(arguments) == 0
    (record,) = runner.load("shared").values()
    assert record["verdict"] == "runs" and record["id"] == "reference/GSNet"
    assert record["probe"]["notes"] == ["probed alone after running out of GPU memory"]
    # Every record names the run that measured it; the commit is the one that was given.
    assert record["measured"]["commit"] == "abcdef0" and record["measured"]["probes_at_a_time"] == 2
    assert re.fullmatch(r"\d{4}-\d\d-\d\d", record["measured"]["date"])
    # Any other failure is a verdict about the config and is recorded as it is.
    broken = {
        "verdict": "fails",
        "failure": {"phase": "build", "error": "ValueError", "message": "x"},
    }
    outcomes[:] = [{**broken, "seconds": 1.0}]
    assert runner.main(["run", "--name", "other", *arguments[3:]]) == 0
    assert runner.load("other")["reference/GSNet"]["verdict"] == "fails"
    # Uncommitted changes: no commit names the code, so nothing is probed unless asked.
    import argparse

    dirty = {"available": True, "commit": "a" * 40, "dirty": True}
    monkeypatch.setattr(runner, "git_info", lambda: dirty)
    asked = argparse.Namespace(
        commit=None, allow_dirty=False, iters=2, frames=8, jobs=2, scale_statistic_sha=None
    )
    with pytest.raises(SystemExit, match="uncommitted changes"):
        runner._environment(asked)
    asked.allow_dirty = True
    environment = runner._environment(asked)
    assert environment["commit"] == "a" * 40 + "+dirty"
    assert runner._measured(environment)["commit"] == "aaaaaaa+dirty"
    assert isinstance(environment["libraries"], dict)


def test_the_methods_as_shipped_run(by_id):
    for method in [*METHODS, *FAMILIES]:
        group = "reference" if method in METHODS else "family"
        assert by_id[f"{group}/{method}"]["verdict"] == "runs", method
    # A component under its own method is that method's shipped config, probed once.
    own = by_id["encoder/GSNet/SparseUNet14"]
    assert own["config_hash"] == by_id["reference/GSNet"]["config_hash"]
    assert own["probed_as"] == "reference/GSNet" and own["verdict"] == "runs"
    # Every method's own stage, in the preset or the defaults that state it, is recognised.
    for cell_id in (
        "stage/GSNet/head/swad",
        "stage/EconomicGrasp/assigner/compact-label",
        "stage/FineGrasp/grouper/multi-scale",
        "stage/GraspNet-baseline/view/objectness-view",
        "stage/Scale-Balanced-Grasp/grouper/depth-cylinder-multi-scale",
    ):
        method = cell_id.split("/")[1]
        assert by_id[cell_id]["probed_as"] == f"reference/{method}", cell_id


def test_a_run_merges_into_the_file_and_names_itself():
    """Probing the cells of one new component keeps every other cell of the file."""
    runner = tool("probe_grid")
    first, second = (cell.id for cell in cells(("reference",))[:2])
    environment = {
        "torch": "0",
        "python": "3",
        "gpu": "none",
        "iters": 20,
        "frames": 8,
        "jobs": 1,
        "commit": "abcdef0123",
    }

    def record(cell_id: str, day: str) -> dict:
        return {
            "id": cell_id,
            "group": "reference",
            "method": cell_id.split("/")[1],
            "sets": [],
            "slot": "",
            "component": "",
            "variant": "",
            "verdict": "runs",
            "config_hash": "0" * 64,
            "probe_command": "grasppanda probe x --iters 20 --frames 8",
            "probe": {"verdict": "runs", "iterations": 20, "seconds": 1.0},
            "finished": f"{day}T00:00:00Z",
        }

    one = runner.merge(None, {first: record(first, "2026-01-01")}, environment)
    assert [cell["id"] for cell in one["cells"]] == [first]
    assert (
        one["cells"][0]["run"] == "2026-01-01@abcdef0" and first not in one["about"]["not_probed"]
    )
    later = {**environment, "commit": "1234567890"}
    two = runner.merge(one, {second: record(second, "2026-02-02")}, later)
    assert [cell["run"] for cell in two["cells"]] == ["2026-01-01@abcdef0", "2026-02-02@1234567"]
    assert set(two["about"]["runs"]) == {"2026-01-01@abcdef0", "2026-02-02@1234567"}
    # A cell probed again replaces its record; a run nothing refers to any more is dropped.
    again = runner.merge(two, {first: record(first, "2026-03-03")}, later)
    assert set(again["about"]["runs"]) == {"2026-02-02@1234567", "2026-03-03@1234567"}
    assert json.loads(runner.dump(again)) == again
    # A record that carries the run that measured it keeps it, whatever the directory's
    # last run was: a probe reused for the same config is not dated again.
    earlier = {"date": "2025-12-12", "commit": "0123456", "torch": "0", "python": "3"}
    earlier |= {"gpu": "none", "probes_at_a_time": 2, "scale_statistic_sha256": None}
    reused = runner.merge(
        None,
        {
            first: {**record(first, "2026-03-03"), "measured": earlier},
            second: record(second, "2026-03-03"),
        },
        later,
    )
    assert [cell["run"] for cell in reused["cells"]] == ["2025-12-12@0123456", "2026-03-03@1234567"]
    assert reused["about"]["runs"]["2025-12-12@0123456"] == earlier
    assert "grasppanda" not in reused["about"]["runs"]["2026-03-03@1234567"]
    assert "not repeatable" in reused["about"]["one_run"]
    with pytest.raises(SystemExit, match="one file, one probe"):
        runner.merge(two, {first: record(first, "2026-03-03")}, {**later, "iters": 5})
    # Commands name the composed config as a file, not by its place in a cache.
    typed = runner._as_typed(
        "grasppanda probe /data/cache/probe_grid/x/configs/a__b.yaml --iters 2"
    )
    assert typed == "grasppanda probe a__b.yaml --iters 2"
    assert runner._fitted(["out_channels: 256, the width", "in_channels: 6. The base", "x"]) == [
        "out_channels: 256",
        "in_channels: 6",
    ]


# ------------------------------------------------------------------------- the page


def test_the_generated_page_and_the_composed_configs_are_current():
    generator = docs_tool()
    outdated = generator.stale()
    assert not outdated, (
        f"stale: {[str(p.relative_to(REPO)) for p in outdated]}. "
        "Run `python -m tools.make_combination_docs`."
    )
    assert generator.main(["--check"]) == 0


def test_the_page_states_what_it_measures_and_what_it_does_not(measured):
    generator = docs_tool()
    page = PAGE.read_text(encoding="utf-8")
    assert page.startswith("<!-- Generated by tools/make_combination_docs.py")
    assert "**No AP appears on this page, and none follows from it.**" in page
    assert "shared with other jobs" in page and "‡" in page
    # The classes are defined on the page, in terms of the grid.
    for name, (_, meaning) in generator.CLASSES.items():
        assert f"- **{name}**: {meaning}." in page
    assert "A component **works** under a method when its cell is `runs` or `runs-unstable`" in page
    assert "| Slot | universal | most | some | specific | none |" in page
    # One table per slot, one row per component variant, one column per method.
    for slot in ("backbone", *STAGE_SLOTS):
        assert f"### {generator.SLOT_TITLES[slot]}" in page
    header = "| Component | " + " | ".join(METHODS) + " | Class |"
    assert page.count(header) == 1 + len(STAGE_SLOTS)
    for encoder in sorted(n for n in in_tree("BACKBONES") if not is_adapter(BACKBONES.get(n))):
        assert f"| `{encoder}`<br>preset `" in page, encoder
    # Every verdict of the measured file is on the page, nothing else.
    counts = Counter(record["verdict"] for record in measured["cells"])
    for verdict, count in counts.items():
        assert f"{count} `{verdict}`" in page
    if not NOT_PROBED:  # every cell of the required groups has a verdict on the page
        assert "not probed" not in page.split("## Sets of stages")[0]


def test_classes_are_counted_from_the_cells(measured, by_id):
    generator = docs_tool()
    runs, unstable, refused = (
        {"verdict": "runs"},
        {"verdict": "runs-unstable"},
        {"verdict": "refused"},
    )
    assert generator.classify([runs] * 5) == ("universal", 5, 0)
    assert generator.classify([runs, runs, unstable, refused, {"verdict": "fails"}]) == (
        "most",
        3,
        1,
    )
    assert generator.classify([runs, runs, refused, refused, refused]) == ("some", 2, 0)
    assert generator.classify([runs, refused, refused, refused, None]) == ("specific", 1, 0)
    assert generator.classify([refused] * 5) == ("none", 0, 0)
    assert list(generator.CLASSES) == ["universal", "most", "some", "specific", "none"]
    # On the committed measurements: the class printed in a row is the count of its cells.
    page = PAGE.read_text(encoding="utf-8")
    records = generator.Records(measured["cells"])
    for slot in ("backbone", *STAGE_SLOTS):
        group = "encoder" if slot == "backbone" else "stage"
        for component, variant, name, works, unsteady in generator.slot_classes(
            slot, records, group
        ):
            wanted = [
                cell.id
                for cell in generator.grid(group)
                if (cell.slot, cell.component, cell.variant) == (slot, component, variant)
            ]
            assert len(wanted) == len(METHODS)
            if any(cell_id not in by_id for cell_id in wanted):
                # A class is a count over all methods: without every cell there is none.
                assert name == generator.INCOMPLETE and set(wanted) & set(NOT_PROBED)
                continue
            members = [by_id[cell_id] for cell_id in wanted]
            assert works == sum(record["verdict"] in generator.WORKS for record in members)
            assert works in generator.CLASSES[name][0]
            label = generator._label(component, variant, slot)
            row = next(line for line in page.splitlines() if line.startswith(f"| {label} |"))
            assert row.endswith(f"| {generator._class_text(name, unsteady)} |")
    # Every method's own encoder works under it.
    for cell in cells(("encoder",)):
        if cell.id in by_id and by_id[cell.id].get("probed_as") == f"reference/{cell.method}":
            assert by_id[cell.id]["verdict"] == "runs"


# -------------------------------------------------------------------- the slot index


def test_the_slot_index_counts_what_exists_and_classifies_from_the_records(measured):
    generator = docs_tool()
    slots = REPO / "docs" / "slots"
    page = (slots / "index.md").read_text(encoding="utf-8")
    assert page == generator.slots_index(generator.Records(measured["cells"]))
    assert "**No accuracy is claimed" in page and "the model zoo is still empty" in page
    # Every slot page is linked from the index, and every link of the slot table exists.
    for path in slots.glob("*.md"):
        assert path.name == "index.md" or f"]({path.name})" in page, path.name
    for _, _, link, _, _ in generator.SLOTS.values():
        assert not link or (slots / link).is_file(), link
    # The numbers are counted: what can be written after `--set <slot>=`.
    counts = generator.option_counts()
    for slot in STAGE_SLOTS:
        shipped = in_tree(STAGE_REGISTRIES[slot].name)
        assert counts[slot][0] == sum(c in shipped for _, c in stage_variants(slot))
    assert counts["image_encoder"][0] == len(presets()["image_encoder"])
    for options, types in counts.values():
        assert f"| {options} ({types} component type" in page
    # Every option of a seed-based slot stands under exactly one class.
    table = page.split("| Slot | Probed under |")[1].split("\n\n")[0]
    for slot in STAGE_SLOTS:
        shipped = in_tree(STAGE_REGISTRIES[slot].name)
        for choice, component in stage_variants(slot):
            assert component not in shipped or f"`{choice}`" in table, choice
    # Fewer than five methods probed: the class counts those, and the page says so.
    assert generator._relative_class(3, 3) == "universal"
    assert generator._relative_class(2, 3) == "most"
    assert generator._relative_class(1, 2) == "some"
    assert generator._relative_class(0, 2) == "none"
    assert "† The slot was not probed under the five seed-based methods" in page
