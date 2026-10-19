"""The measured grid: what is probed, what is committed, and the page generated from it.

Nothing is probed here (that needs the dataset and a GPU, see
``tests/integration/test_combine_grid.py``). These tests are the tripwire between the
registries and the committed measurements: a registered scene encoder or stage whose
cells have no record makes the suite fail and says which commands produce one, and so
does a record that no longer is what ``grasppanda compose`` writes today.
"""

from __future__ import annotations

import ast
import html
import importlib.util
import json
import re
import sys
from collections import Counter

import pytest

from grasppanda.cli.components import collect
from grasppanda.combine import reach
from grasppanda.combine.compose import compose, is_adapter
from grasppanda.combine.grid import (
    FAMILIES,
    LOSS_CELLS,
    LOSS_NOT_PROBED,
    METHODS,
    POST_NOT_PROBED,
    RECIPE_ENCODER_PRESETS,
    STAGE_SLOTS,
    Cell,
    cells,
    family_slots,
    loss_terms,
    needs_an_option,
    stage_variants,
)
from grasppanda.combine.levels import CLASS_WORDS, LEVELS, option_class
from grasppanda.combine.presets import presets
from grasppanda.combine.probe import TREND_FLAGS, VERDICTS
from grasppanda.config import Config, available_configs
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, SeedGraspDetector
from grasppanda.registry import BACKBONES, DETECTORS

from .conftest import REPO

DATA = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
PAGE = REPO / "docs" / "advanced_guides" / "measured_combinations.md"

#: Cells of the grid that need no record, each with the reason. Empty: every group below
#: was probed. A cell goes here only with a reason a reader of the page can check.
NOT_PROBED: dict[str, str] = {}
#: Cells that were added to the grid and wait for their first probe, one id per line in
#: the file: the post-processing presets under the methods they were not probed under,
#: the recipe presets under the methods beyond GSNet and HGGD, the one-term loss cells.
#: Temporary and visible: a listed cell that has a record fails the suite, so the list
#: shrinks with every export and is empty once every cell has a record.
PENDING_FILE = REPO / "tests" / "unit" / "combine_grid_pending.txt"
PENDING = frozenset(
    line.strip()
    for line in PENDING_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
)
#: Refused cells whose record predates the refusals compose lifts since October 15 2026
#: (an axis adapter, the options a stage declares, the label set of the head): compose now
#: accepts them, or refuses them with other messages. Their records stand as measured
#: until the next probe of the whole grid replaces them; an id leaves this set then.
AWAITING_PROBE: frozenset[str] = frozenset(
    json.loads(
        (REPO / "tests" / "unit" / "fixtures" / "awaiting_probe.json").read_text(encoding="utf-8")
    )
)
#: Groups every registered component must have records in. The sets of stages are probed
#: too, but a new stage does not have to wait for its sets: the page counts what is missing.
REQUIRED_GROUPS = (
    "reference",
    "encoder",
    "stage",
    "family",
    "family-slot",
    "post",
    "loss",
    "recipe",
    "cross",
)
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
        variants = [(c, t) for c, t in stage_variants(slot) if t in shipped_stages]
        assert {(cell.variant, cell.method) for cell in members} == {
            (choice, method)
            for choice, component in variants
            if not needs_an_option(slot, choice, component)
            for method in METHODS
        }
        # A component named by its type without an option only a preset supplies is no
        # combination: it has no cell, and the presets that supply the option have theirs.
        for choice, component in variants:
            if needs_an_option(slot, choice, component):
                assert choice == component and choice not in presets()[slot], choice
                assert any(p.type == component for p in presets()[slot].values()), choice
    # Every post-processing preset under all six methods: prediction plus the chain.
    assert {(cell.variant, cell.method) for cell in grid if cell.group == "post"} == {
        (name, method)
        for name in presets()["post"]
        if name not in POST_NOT_PROBED
        for method in (*METHODS, *FAMILIES)
    }
    # Every recipe preset under all six methods, except those that name the scene encoder
    # of the seed-based skeleton: not under HGGD.
    recipe = [cell for cell in grid if cell.group == "recipe"]
    for key in {cell.id.split("/", 2)[2] for cell in recipe}:
        under = {cell.method for cell in recipe if cell.id.endswith(f"/{key}")}
        wanted = set(METHODS) if key in RECIPE_ENCODER_PRESETS else {*METHODS, *FAMILIES}
        assert under == wanted, key
    # Every loss preset in one term of every kind it accepts, under every seed-based method
    # with a term of that kind: the last of the kind in the pipeline. The hand-picked cells
    # are kept, and a one-term cell that is also hand-picked is one cell.
    from grasppanda.models.losses.terms import accepted_terms

    loss = [cell for cell in grid if cell.group == "loss"]
    requests = {(cell.method, cell.sets) for cell in loss}
    picked = {(method, tuple(sets)) for method, sets in LOSS_CELLS}
    assert picked <= requests and len(requests) == len(loss)
    for preset in presets()["loss"].values():
        if preset.name in LOSS_NOT_PROBED:
            assert not any(cell.variant == preset.name for cell in loss)
            continue
        kinds = accepted_terms(dict(preset.node))[1]
        for method in METHODS:
            for kind, term in loss_terms(method).items():
                request = (method, (f"{term}={preset.name}",))
                assert (request in requests) == (kind in kinds or request in picked), request
    for method, sets in requests - picked:  # one representative term each, nothing else
        assert len(sets) == 1 and sets[0].partition("=")[0] in loss_terms(method).values()
    assert loss_terms("GSNet") == {
        "class": "proposal.objectness_loss",  # the only class term of GSNet
        "unit": "head.score_loss",  # after proposal.graspness_loss and view.loss
        "value": "head.width_loss",
    }
    assert loss_terms("GraspNet-baseline") == {
        "class": "head.angle_loss",  # after view.objectness_loss
        "value": "head.tolerance_loss",  # the last of view_loss, score, width, tolerance
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
    wanted = [cell for cell in cells(REQUIRED_GROUPS) if cell.id not in {*NOT_PROBED, *PENDING}]
    missing = [cell.id for cell in wanted if cell.id not in by_id]
    assert not missing, (
        f"{len(missing)} cell(s) of the grid have no measured record, e.g. {missing[:5]}. "
        f"{HOW_TO_PROBE}or list the cell in NOT_PROBED of this file with the reason."
    )
    grid = {cell.id for cell in cells()}
    stale = set(by_id) - grid
    assert not stale, f"records of cells that are no longer in the grid: {sorted(stale)[:5]}"
    assert set(NOT_PROBED) <= grid
    # The cells without a record are named, but for the sets of stages: those are counted.
    missing = grid - set(by_id)
    named = {cell_id for cell_id in missing if not cell_id.startswith("stage-set/")}
    assert set(measured["about"]["not_probed"]) == named
    assert measured["about"]["stage_sets_not_probed"] == len(missing - named)
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


def test_the_cells_waiting_for_their_first_probe_leave_the_list_with_their_record(by_id):
    """The pending list is a promise with an end: every listed cell is a cell of the grid
    in a required group without a record, and a record makes its cell leave the list."""
    grid = {cell.id for cell in cells(REQUIRED_GROUPS)}
    assert not PENDING - grid, f"not cells of a required group: {sorted(PENDING - grid)[:5]}"
    assert not PENDING & set(NOT_PROBED)
    probed = sorted(PENDING & set(by_id))
    assert not probed, (
        f"{len(probed)} pending cell(s) have a record now, e.g. {probed[:5]}: remove them "
        f"from {PENDING_FILE.relative_to(REPO)}. The list is empty once every cell has a record."
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
    outdated = [cell_id for cell_id in outdated if cell_id not in AWAITING_PROBE]
    assert not outdated, (
        f"{len(outdated)} measured cell(s) are not what compose writes today, e.g. "
        f"{outdated[:5]}. {HOW_TO_PROBE}(`--only` takes several ids and glob patterns.)\n"
        "Records made before the config hash stopped depending on the installed PyTorch are "
        "rewritten, without probing, by `python -m tools.probe_grid rehash` on the machine "
        "that measured them."
    )


def test_a_record_is_a_run_with_its_commands_and_a_verdict_of_the_vocabulary(measured, by_id):
    about = measured["about"]
    # The file defines the verdicts of the judge that wrote it. A file of the first judge
    # (no record carries `judge`) has every verdict but `runs-empty`; an export of judge 2
    # writes today's vocabulary, which the records of both judges are read with.
    judges = {record.get("judge", 1) for record in measured["cells"] if "probe" in record}
    if judges == {1}:
        assert set(about["verdicts"]) == set(VERDICTS) - {"runs-empty"}
    else:
        assert about["verdicts"] == VERDICTS and "judges" in about
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
            if record.get("judge", 1) >= 2:  # what judge 2 adds, term by term
                assert set(record["trend"]) == set(record["losses"]), cell.id
                assert set(record.get("flags", {})) <= set(record["losses"]), cell.id
                assert {"no_gradient", "unchanged", "checked"} <= set(record), cell.id
                grasps = record["grasps_per_frame"]
                assert len(record["score_quantiles"]) == len(grasps), cell.id
                empty = record["verdict"] == "runs-empty"
                assert not empty or not any(grasps), cell.id
                assert empty or record["verdict"] != "runs" or any(grasps), cell.id
        # Machine-dependent numbers are kept apart from the others.
        assert not {"seconds_per_iteration", "peak_gpu_mib", "seconds"} & set(record), cell.id
    text = DATA.read_text(encoding="utf-8")
    # No accuracy of any kind is recorded, and no path of the machine that ran the grid.
    assert '"AP' not in text and "average precision" not in text.lower()
    assert "/home/" not in text and "/tmp/" not in text and "probe_grid/" not in text
    # The file is the runner's own serialisation: one cell per line, nothing edited by hand.
    assert tool("probe_grid").dump(measured) == text
    assert sum(line.startswith('  {"id": ') for line in text.splitlines()) == len(by_id)
    # Compact: about a kilobyte per probed cell, and up to 700 bytes more for a probe of
    # judge 2 (the trend of every loss term, the units and parts checked, the scores and the
    # learning rate: 400 to 550 bytes on four probes of 6 to 8 terms; the notes of the
    # held rate and batch-norm momentum, about 150 more). The loss of every
    # iteration, another kilobyte or more per probe, is in the side file. The list of the
    # stage sets that were not probed is no record; it grows with the product of the stage
    # variants.
    unprobed = len(json.dumps(measured["about"]["not_probed"]))
    judged = sum(record.get("judge", 1) >= 2 and "losses" in record for record in by_id.values())
    assert len(text) - unprobed < 1_300 * len(by_id) + 700 * judged


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
    for report in (allocator, library, contract):
        report["probe"]["judge"] = 2
    assert not grid._conclusive(allocator) and not grid._conclusive(library)
    assert grid._conclusive(contract)
    assert grid._conclusive({"probe": {"verdict": "runs", "judge": 2}})
    # A probe of the first judge lacks what is judged now: it is probed again.
    assert not grid._conclusive({"probe": {"verdict": "runs"}})


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
    fine = {"verdict": "runs", "iterations": 2, "seconds": 1.0, "judge": 2}
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


def test_grid_probes_hold_the_peak_rate_and_a_probe_of_another_schedule_is_not_reused(
    tmp_path, monkeypatch
):
    """The grid probes with `--schedule constant` unless told otherwise. The config hash
    does not depend on the schedule, so a probe of one schedule never stands for the other."""
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    commands: list[list[str]] = []

    def run(command, log, timeout):
        commands.append(command)
        schedule = command[command.index("--schedule") + 1]
        report = {"verdict": "runs", "judge": 2, "iterations": 2, "seconds": 1.0}
        report |= {"schedule": schedule, "rate": 0.001 if schedule == "constant" else None}
        return 0, json.dumps(report)

    monkeypatch.setattr(runner, "_run", run)
    # One cell is enough, and the whole grid takes seconds to enumerate.
    reference = Cell("reference/GSNet", "reference", "GSNet", METHODS["GSNet"])
    monkeypatch.setattr(runner, "cells", lambda groups=None: [reference])
    arguments = ["run", "--name", "held", "--iters", "2", "--commit", "abcdef0123"]
    assert runner.main(arguments) == 0
    (record,) = runner.load("held").values()
    assert record["probe_command"].endswith("--iters 2 --frames 8 --schedule constant")
    assert record["probe"]["schedule"] == "constant" and len(commands) == 1
    assert runner.main([*arguments, "--redo"]) == 0  # the same schedule: the probe stands
    assert len(commands) == 1
    assert runner.main([*arguments, "--redo", "--schedule", "recipe"]) == 0  # another one
    assert len(commands) == 2 and runner.load("held")["reference/GSNet"]["probe"]["rate"] is None
    assert runner.main([*arguments, "--redo"]) == 0  # back to constant: probed again
    assert len(commands) == 3
    # The record says how the rate was set; a record of a probe before the option says nothing.
    held = runner.compact(runner.load("held")["reference/GSNet"])
    assert held["schedule"] == "constant" and held["rate"] == 0.001
    assert list(held)[list(held).index("probe") + 1 : list(held).index("probe") + 3] == [
        "schedule",
        "rate",
    ]
    # A cell whose request sets the schedule is probed with it: it is what the cell measures.
    import argparse

    held_args = argparse.Namespace(schedule="constant")
    for sets in (["schedule=cosine"], ["recipe=adamw-onecycle"], ["optimizer=schedulefree-adamw"]):
        assert runner.cell_schedule(sets, held_args) == "recipe", sets
    for sets in ([], ["optimizer=adamw"], ["backbone=ptv3", "hooks=ema+swa"], ["post=nms"]):
        assert runner.cell_schedule(sets, held_args) == "constant", sets
    recipe = judge_two_record("reference/GSNet", schedule="recipe")
    assert runner.compact(recipe)["schedule"] == "recipe" and "rate" not in runner.compact(recipe)
    older = judge_two_record("reference/GSNet")
    for name in ("schedule", "rate"):
        older["probe"].pop(name)
    assert not {"schedule", "rate"} & set(runner.compact(older))
    assert "schedule, rate" in runner.FIELDS


def test_the_methods_as_shipped_run(by_id):
    for method in [*METHODS, *FAMILIES]:
        group = "reference" if method in METHODS else "family"
        # Judge 2 calls a clean run that predicted no grasp in 20 iterations `runs-empty`
        # (Scale-Balanced-Grasp's objectness decision is still marginal there): it runs.
        assert by_id[f"{group}/{method}"]["verdict"] in ("runs", "runs-empty"), method
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
        reference = by_id[f"reference/{method}"]
        # The shipped config again (its hash), so it shares a probe of that config: the
        # reference's, or that of another cell of the same config probed in the same run.
        assert by_id[cell_id]["config_hash"] == reference["config_hash"], cell_id
        owner = by_id[by_id[cell_id]["probed_as"]]
        assert owner["config_hash"] == reference["config_hash"], cell_id


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
    assert "judge: 2" in again["about"]["judges"]
    typed = runner._as_typed(
        "grasppanda probe /data/cache/probe_grid/x/configs/a__b.yaml --iters 2"
    )
    assert typed == "grasppanda probe a__b.yaml --iters 2"
    assert runner._fitted(["out_channels: 256, the width", "in_channels: 6. The base", "x"]) == [
        "out_channels: 256",
        "in_channels: 6",
    ]


def judge_two_record(cell_id: str, day: str = "2026-10-15", **changes) -> dict:
    """A record of the runner as a probe of judge 2 leaves it: the report is a real
    ``ProbeReport`` and the trends are computed by the probe's own function."""
    from grasppanda.combine.probe import ProbeReport, loss_trend

    series = {
        "loss/view": [1.2, 0.6, 0.0, 0.0],
        "loss/score": [0.0, 0.5, 0.49, 0.49],
        "loss": [2.0, 1.4, 0.8, 0.8],
    }
    report = ProbeReport(
        verdict="runs-empty",
        detail="trained 4 iterations ..., but no grasp was predicted on any probed frame",
        parameters=1000,
        iterations_requested=4,
        iterations=4,
        batch_size=4,
        frames=[f"f{index}" for index in range(8)],
        losses={name: [values[0], values[-1]] for name, values in series.items()},
        losses_per_iteration=series,
        loss_trends={name: loss_trend(values) for name, values in series.items()},
        grasps_per_frame=[0, 0],
        predicted_frames=["f0", "f4"],
        no_gradient=["head.score_net"],
        unchanged=[],
        units_checked=12,
        parts_checked=5,
        score_quantiles=[None, None],
        learning_rate=[0.001, 0.0006302],
        lr_factor=0.6302,
        seconds=30.0,
    ).to_dict()
    report.update(changes)
    return {
        "id": cell_id,
        "group": cell_id.split("/")[0],
        "method": cell_id.split("/")[1],
        "sets": [],
        "slot": "",
        "component": "",
        "variant": "",
        "verdict": report["verdict"],
        "config_hash": "0" * 64,
        "probe_command": "grasppanda probe x --iters 20 --frames 8",
        "probe": report,
        "finished": f"{day}T00:00:00Z",
    }


def test_a_record_of_judge_two_carries_its_fields_and_old_records_stay_readable():
    runner = tool("probe_grid")
    first, second = (cell.id for cell in cells(("reference",))[:2])
    record = runner.compact(judge_two_record(second))
    assert record["verdict"] == "runs-empty" and record["judge"] == 2
    # Every loss term with [first active, mean of the last quarter, fall]; the number of
    # active iterations and the flags only where they say something.
    assert record["trend"] == {
        "loss/view": [1.2, 0.6, 0.5],
        "loss/score": [0.5, 0.49, 0.02],
        "loss": [2.0, 0.8, 0.6],
    }
    assert record["active"] == {"loss/view": 2, "loss/score": 3}
    assert record["flags"] == {"loss/view": ["zero_at_end"], "loss/score": ["flat"]}
    assert record["no_gradient"] == ["head.score_net"] and record["unchanged"] == []
    assert record["checked"] == [12, 5] and record["score_quantiles"] == [None, None]
    assert record["learning_rate"] == [0.001, 0.0006302] and record["lr_factor"] == 0.6302
    assert "losses_per_iteration" not in record and "loss_trends" not in record
    # A probe of the first judge gives a record of the first judge: none of these fields,
    # and `no_gradient` only when its probe had that check.
    old = judge_two_record(first)
    for name in ("judge", "loss_trends", "unchanged", "score_quantiles", "learning_rate"):
        old["probe"].pop(name)
    old["probe"].pop("no_gradient")
    old_record = runner.compact(old)
    assert not {"judge", "trend", "flags", "no_gradient", "unchanged", "checked"} & set(old_record)
    old["probe"]["no_gradient"] = []
    assert runner.compact(old)["no_gradient"] == []
    # Merged: the cell of the first judge stays as it was, the new one joins with its run.
    environment = {"torch": "0", "python": "3", "gpu": "none", "iters": 20, "frames": 8}
    environment |= {"jobs": 1, "commit": "abcdef0123"}
    before = runner.merge(None, {first: old}, environment)
    after = runner.merge(before, {second: judge_two_record(second)}, environment)
    assert after["cells"][0] == before["cells"][0] and "judge" not in after["cells"][0]
    assert after["cells"][1]["judge"] == 2 and after["about"]["verdicts"] == VERDICTS
    assert json.loads(runner.dump(after)) == after
    # The loss of every iteration goes into the side file, one entry per probe of judge 2.
    series = runner.merge_series(None, {second: judge_two_record(second)}, after)
    assert series["cells"] == [
        {
            "id": second,
            "run": after["cells"][1]["run"],
            "losses": judge_two_record(second)["probe"]["losses_per_iteration"],
        }
    ]
    assert json.loads(runner.dump(series)) == series
    # Kept while its record is kept; dropped when the cell is probed again without it.
    assert runner.merge_series(series, {}, after)["cells"] == series["cells"]
    composed = {k: v for k, v in judge_two_record(second).items() if "probe" not in k}
    composed |= {"verdict": "refused", "refused_by": "compose", "refusal": ["model: refused"]}
    refused = runner.merge(after, {second: composed}, environment)
    assert runner.merge_series(series, {second: composed}, refused)["cells"] == []
    assert runner.merge_series(series, {}, refused)["cells"] == []
    assert runner.side_file(DATA) == DATA.parent / "measured_combinations" / runner.SERIES_NAME
    # A kept cell that shares the probe of a cell probed again shares the new probe.
    owner, sharer = "reference/GSNet", "encoder/GSNet/SparseUNet14"
    shared = {
        "id": sharer,
        "group": "encoder",
        "method": "GSNet",
        "sets": ["backbone=sparse-unet14"],
        "verdict": "runs",
        "config_hash": "0" * 16,
        "compose": "grasppanda compose ...",
        "probed_as": owner,
        "run": "2026-10-13@0000000",
    }
    old_owner = {**old, "id": owner, "group": "reference", "method": "GSNet"}
    earlier = runner.merge(None, {owner: old_owner}, environment)
    earlier["cells"].append(shared)
    again = runner.merge(earlier, {owner: judge_two_record(owner)}, environment)
    follows = next(cell for cell in again["cells"] if cell["id"] == sharer)
    assert follows["verdict"] == "runs-empty" and follows["judge"] == 2
    assert follows["run"] == again["cells"][0]["run"] and follows["probed_as"] == owner
    assert list(follows) == [*list(shared)[:5], "judge", *list(shared)[5:]]


def first_judge(payload: dict) -> dict:
    """The records as the first judge would have left them: without the fields of judge 2,
    and `runs` for a run that predicted nothing (the first judge had no `runs-empty`)."""
    found = json.loads(json.dumps(payload))
    added = ("judge", "trend", "active", "flags", "unchanged", "checked", "score_quantiles")
    for record in found["cells"]:
        for key in (*added, "learning_rate", "lr_factor", "repeat"):
            record.pop(key, None)
        if record["verdict"] == "runs-empty":
            record["verdict"] = "runs"
    return found


def test_the_page_and_the_explorer_read_records_of_both_judges(measured, tmp_path, monkeypatch):
    """Records of the first judge with one record of judge 2 merged in: the page says which
    judge gave which verdicts and what judge 2 flagged, and the explorer passes the new
    fields on; neither breaks on the records without them."""
    generator, explorer, runner = docs_tool(), tool("make_explorer"), tool("probe_grid")
    measured = first_judge(measured)
    cell_id = "encoder/GSNet/PointNeXt@pointnext"
    record = judge_two_record(cell_id)
    record.update(sets=["backbone=pointnext"], slot="backbone", component="PointNeXt")
    record["variant"] = "pointnext"
    environment = {"torch": "0", "python": "3", "gpu": "none", "iters": 20, "frames": 8}
    environment |= {"jobs": 1, "commit": "abcdef0123"}
    payload = runner.merge(measured, {cell_id: record}, environment)
    target = tmp_path / "measured_combinations.json"
    target.write_text(runner.dump(payload), encoding="utf-8")
    monkeypatch.setattr(generator, "DATA", target)
    page = generator.page()
    total = len(payload["cells"])
    assert f"**1 of the {total} records** on this page were judged by judge 2" in page
    # Under its table, `pointnext` says who is left out at each level: the one record of
    # judge 2 ran but did not learn and predicted nothing; the records of the first judge
    # recorded neither, so the other methods are undecided there, named with the reason.
    row = next(line for line in page.splitlines() if line.lstrip().startswith("- `pointnext`: "))
    assert "not GSNet (does not learn: score flat; no gradient: head.score_net)" in row
    assert "does not predict: no grasp on any frame" in row
    assert "undecided (not checked by the current judge)" in row
    for flag, meaning in TREND_FLAGS.items():
        assert f"- `{flag}`: {meaning}." in page
    # It ran: the class at the level `runs` counts it, as it counted the first judge's
    # empty `runs`; at `learns` it is the one method decided, and it did not learn.
    records = generator.Records(payload["cells"])
    classes = {v: found for _, v, found in generator.slot_classes("backbone", records, "encoder")}
    assert classes["pointnext"]["runs"].name == "universal"
    assert classes["pointnext"]["learns"].text == "0 of 1 probed, +4 not checked"
    monkeypatch.setattr(explorer, "MEASURED", target)
    data = explorer.data()
    shown = next(cell for cell in data["cells"] if cell["id"] == cell_id)
    assert shown["verdict"] == "runs-empty" and shown["judge"] == 2
    # The fall of every term as the page shows it (the total is not shown).
    assert shown["fall"] == {"view": 0.5, "score": 0.02}
    assert shown["flags"] == {"loss/view": ["zero_at_end"], "loss/score": ["flat"]}
    assert shown["no_gradient"] == ["head.score_net"] and "lr_factor" not in shown
    assert shown["levels"] == "YYNN" and "predicts: no grasp on any frame" in shown["why"]
    assert data["verdicts"]["runs-empty"] == VERDICTS["runs-empty"]
    plain = next(cell for cell in data["cells"] if cell["id"] == "reference/GSNet")
    assert "judge" not in plain and "fall" not in plain and plain["levels"] == "YYCC"


def test_the_losses_of_every_iteration_are_in_a_side_file_beside_the_records(measured):
    runner = tool("probe_grid")
    side = runner.side_file(DATA)
    own = {
        record["id"]: record
        for record in measured["cells"]
        if record.get("judge", 1) >= 2 and "losses" in record  # numbers of its own
    }
    if not side.is_file():  # no record of judge 2 has been exported yet
        assert not own
        return
    text = side.read_text(encoding="utf-8")
    series = json.loads(text)
    assert runner.dump(series) == text
    assert {entry["id"] for entry in series["cells"]} == set(own)
    for entry in series["cells"]:
        record = own[entry["id"]]
        assert entry["run"] == record["run"] and set(entry["losses"]) == set(record["losses"])
        assert all(len(values) == record["iterations"] for values in entry["losses"].values())


# ------------------------------------------------------------------------- the page


def test_the_generated_page_and_the_composed_configs_are_current():
    generator = docs_tool()
    outdated = generator.stale()
    assert not outdated, (
        f"stale: {[str(p.relative_to(REPO)) for p in outdated]}. "
        "Run `python -m tools.make_combination_docs`."
    )
    assert generator.main(["--check"]) == 0


def chips_tool():
    return tool("docs_chips")


def test_the_page_states_what_it_measures_and_what_it_does_not(measured):
    generator, chips = docs_tool(), chips_tool()
    page = PAGE.read_text(encoding="utf-8")
    assert page.startswith("<!-- Generated by tools/make_combination_docs.py")
    assert "\n# Does it run?\n" in page
    # The caveat once, what was measured on, and the machine of the runs.
    assert page.count(generator.CAVEAT_SNIPPET) == 1
    assert chips.measured_on(1) in page
    assert "shared with other jobs" in page
    # Fractions with their tails, never the class words of the slot index.
    for name in CLASS_WORDS:
        assert f"**{name}**" not in page, name
    # The anchors other pages link: the frozen ones and those of the slot pages.
    for anchor in (
        "measured-classes",
        "sets-of-stages",
        "several-slots-at-once",
        "scene-encoders",
        "losses",
        "the-training-recipe",
    ):
        assert f"{{#{anchor}}}" in page, anchor
    # One answer matrix per level, its columns the six methods in page order.
    header = "| Slot | " + " | ".join(info.name for info in chips.methods()) + " |"
    assert page.count(header) == len(generator.SHOWN_LEVELS)
    # It claims no more than the records hold: every option was composed under the methods it
    # fits, and the entries without a record of their own are named.
    assert "Every option was composed under each method it fits" in page
    assert "every method that has its slot" not in page
    for name in (
        "two adapters compose inserts",
        "two point-painting wrappers",
        "post=target-mask",
        "loss=distribution-focal`, measured inside `decoupled-grid-bins",
    ):
        assert name in page.split("\n## ")[0], name
    # The sets of stages: the grid counts stage options per kind of encoder output, Swap
    # rules counts registered classes; each count says what it counts.
    sets = page.split("{#sets-of-stages}")[1].split("\n## ")[0]
    wanted = len(generator.grid("stage-set"))
    assert f"accept {wanted} sets of stage options (per kind of encoder output)" in sets
    swapping = (REPO / "docs" / "advanced_guides" / "swapping.md").read_text(encoding="utf-8")
    assert "\n## Every legal set of stages\n" in swapping
    assert f"**{generator.registered_sets()} of " in swapping
    assert (
        "[Swap rules](swapping.md#every-legal-set-of-stages) counts the same rules over "
        f"registered classes: {generator.registered_sets()} sets."
    ) in sets
    # One section per group of slots, counting the options a user can --set
    # (docs_chips.option_count), one row per option; what is not an option (an adapter, a
    # point-painting wrapper) is named after the options with where it was measured.
    for title, (anchor, slots) in generator.SECTIONS.items():
        assert f"### {title} {{#{anchor}}}" in page
        count = sum(chips.option_count(slot) for slot in slots)
        section = page.split(f"### {title} {{#{anchor}}}")[1]
        assert section.lstrip().startswith(f"**{count} option"), title
    assert chips.option_count("loss") == 20 and chips.option_count() == 244
    assert f"({chips.option_count()} in all)" in page
    rows = {line.strip().split(" | ")[0] + " |" for line in page.splitlines()}
    records = generator.Records(measured["cells"])
    for slot in chips.SLOT_ORDER:
        for name in chips.options_of(slot):
            assert f"| {generator.row_label(slot, name)} |" in rows, (slot, name)
    shown = 0
    for slot, extra in generator.not_options(records).items():
        for row in extra:
            note = row.levels["runs"].note
            assert note and f"- `{row.name}`: {note}." in page, (slot, row.name)
            assert f"| {generator.row_label(slot, row.name)} |" not in rows, row.name
            shown += 1
    assert shown == 4 and "inserted in 65 cells" in page
    # A count of zero is not green.
    for line in page.splitlines():
        if line.startswith("**") and ">0 run everywhere<" in line:
            assert 'gp-chip--muted">0 run everywhere<' in line, line
    # Every verdict count of the measured file.
    counts = Counter(record["verdict"] for record in measured["cells"])
    for verdict, count in counts.items():
        assert f"{count} `{verdict}`" in page
    # Cells that wait for their first probe are named where their option is shown.
    for group in ("post", "loss", "recipe"):
        waiting = sum(cell.group == group for cell in cells((group,)) if cell.id in PENDING)
        assert not waiting or "not probed yet" in page, group


def test_the_answer_counts_every_option_tried_under_each_method(measured):
    """A cell of the answer: the options of the slot that reached the level under the
    method, out of those tried there as a swap, counted from grasppanda.combine.reach."""
    generator, chips = docs_tool(), chips_tool()
    page = PAGE.read_text(encoding="utf-8")
    records = generator.Records(measured["cells"])
    names = [info.name for info in chips.methods()]
    by_slot = {slot: chips.options_of(slot) for slot in chips.SLOT_ORDER if chips.options_of(slot)}
    for level in generator.SHOWN_LEVELS:
        tab = page.split(f'=== "{level}"')[1].split("\n\n\n")[0]
        for slot, options in by_slot.items():
            title = chips.SLOT_TITLES[slot].replace(" ", "\u00a0")
            row = next(line for line in tab.splitlines() if f"[{title}](#" in line)
            cells_ = [cell.strip() for cell in row.strip().strip("|").split("|")][1:]
            for method, cell in zip(names, cells_, strict=True):
                if method not in chips.slot_methods(slot):
                    assert cell == "—", (slot, method)
                    continue
                found = [reach.reach(slot, name, level, records.measured) for name in options]
                tried = [item for item in found if method in item.swaps]
                if not tried:
                    assert ">own only<" in cell, (slot, method)
                    continue
                reached = sum(method in item.reached for item in tried)
                assert f">{reached}/{len(tried)}<" in cell, (level, slot, method, cell)


def test_every_row_names_who_is_left_out_and_why(measured):
    """The chips of a row are the reach of its option at each level and its level-2
    outcome; the list under the table names every method a level left out, and why."""
    generator, chips = docs_tool(), chips_tool()
    page = PAGE.read_text(encoding="utf-8")
    records = generator.Records(measured["cells"])
    lines = [line.strip() for line in page.splitlines()]
    options = [(slot, name) for slot in chips.SLOT_ORDER for name in chips.options_of(slot)]
    for slot, name in options:
        label = generator.option_label(slot, name)
        row = next(
            line for line in lines if line.startswith(f"| {generator.row_label(slot, name)} |")
        )
        last = next((line for line in lines if line.startswith(f"- {label}: ")), "")
        for level in generator.SHOWN_LEVELS:
            found = reach.reach(slot, name, level, records.measured)
            assert f">{found.fraction}</span>" in row, (slot, name, level)
            if found.note:
                assert found.note in last
                if not found.applies:  # no record of its own: the note says why
                    continue
            for method in found.swaps:
                if method not in found.reached:
                    assert method in last, (slot, name, level, method)
        level2 = reach.level2(slot, name, records.measured)
        assert f">{level2.fraction}</span>" in row, (slot, name)
        for method, outcome in level2.outcomes.items():
            if outcome != "on a par" and not level2.none:
                # an option of one method names it in the chip itself ("HGGD: worse")
                named = level2.fraction.startswith(f"{method}: ")
                assert named or method in last, (slot, name, method)
        for method in level2.excluded:  # not compared there, and named
            assert method in last, (slot, name, method)


def test_classes_are_counted_from_the_cells(measured, by_id):
    generator = docs_tool()
    notes = generator.Notes()
    empty = {"id": "x", "verdict": "runs-empty", "adapter": "FurthestPointSubsample"}
    assert generator.verdict_text(empty, False, notes) == "∅ in `FurthestPointSubsample`"
    records = generator.Records([{**empty, "grasps_per_frame": [0, 0]}])
    shown = generator.verdict_text(records.get("x"), False, notes, records)
    assert shown == '<span class="gp-nograsp">∅ in `FurthestPointSubsample`<br>0 · 0</span>'
    assert list(generator.CLASSES) == ["universal", "most", "some", "specific", "none"]
    # On the committed measurements: the class of an option (the slot index prints it) is
    # the count, level by level, of the levels of its cells (grasppanda.combine.levels).
    records = generator.Records(measured["cells"])
    for slot in ("backbone", *STAGE_SLOTS):
        group = "encoder" if slot == "backbone" else "stage"
        for component, variant, found in generator.slot_classes(slot, records, group):
            wanted = [
                cell
                for cell in generator.grid(group)
                if (cell.slot, cell.component, cell.variant) == (slot, component, variant)
            ]
            assert len(wanted) == len(METHODS)
            states = {
                cell.method: records.levels(by_id[cell.id]).states["runs"]
                for cell in wanted
                if cell.id in by_id
            }
            expected = option_class("runs", states, METHODS, tuple(METHODS))
            assert found["runs"] == expected.with_checked(found["runs"].checked)
    # Every method's own encoder works under it.
    for cell in cells(("encoder",)):
        if cell.id in by_id and by_id[cell.id].get("probed_as") == f"reference/{cell.method}":
            assert by_id[cell.id]["verdict"] in ("runs", "runs-empty")


def test_the_evidence_index_defines_the_words_and_counts_from_the_records(measured):
    from grasppanda.config import REFUSAL_KINDS

    generator, chips = docs_tool(), chips_tool()
    page = (REPO / "docs" / "evidence" / "index.md").read_text(encoding="utf-8")
    assert page.startswith("<!-- Generated by tools/make_combination_docs.py")
    assert "\n# How we measure\n" in page and page.count(generator.CAVEAT_SNIPPET) == 1
    # One line per level, verdict, level-2 outcome and refusal kind, the exact definitions
    # beside them.
    assert set(generator.LEVEL_LINES) == set(LEVELS)
    for level, meaning in LEVELS.items():
        assert f"- **{level}**: {meaning}." in page
    assert set(generator.VERDICT_LINES) == set(VERDICTS)
    for name, meaning in VERDICTS.items():
        assert f"- `{name}`: {generator.VERDICT_LINES[name]}." in page
        assert f"- `{name}`: {meaning}" in page
    words = {reach.LEVEL2_WORDS.get(name, name) for name in reach.LEVEL2_ORDER}
    assert set(generator.LEVEL2_LINES) == words
    assert set(reach.REFUSAL_WORDS) == set(REFUSAL_KINDS)
    for kind, text in reach.REFUSAL_WORDS.items():
        assert f"- **{text}** (`{kind}`): " in page
    # What was measured on; level 2 by its one name; no claim beyond the records.
    for level in (1, 2):
        sentence = chips.measured_on(level)
        assert sentence[0].upper() + sentence[1:] in page
    assert f"## {chips.LEVEL2_HEADER} {{#level-2}}" in page
    assert "learn like the reference" not in page.lower()
    assert "Every option was composed under each method it fits" in page
    assert "every method that has its slot" not in page and "Every option of every" not in page
    # The options without a record of their own are named, a loss measured inside a head
    # bundle among them.
    coverage = page.split("<div")[0]
    for name, (head, _) in reach.LOSS_INSIDE.items():
        assert f"`loss={name}`, measured inside `{head}`" in coverage
    # Level 2 reads swaps that ran, not those that learned in the short probe; "the
    # reference" (the method as shipped) is spelled out on this page.
    assert "Level 2 needs `runs`, not `learns`: a term flat after " in page
    level2_part = page.split("{#level-2}")[1].split("\n## ")[0]
    assert "the reference" not in level2_part.replace("`no reference`", "")
    # The answer per slot, visible (not in a fold): the options a user can --set
    # (docs_chips.option_count) and how many run, and train like shipped, everywhere they
    # were tried, counted from grasppanda.combine.reach.
    records = generator.Records(measured["cells"])
    found_rows = generator.where_options_work(records)
    answer = page.split("## Slot by slot {#the-answer}")[1].split("\n## ")[0]
    assert f"The {chips.option_count()} options a user can `--set`" in answer
    for slot in chips.SLOT_ORDER:
        options = chips.options_of(slot)
        if not options:
            continue
        found = [reach.reach(slot, name, "runs", records.measured) for name in options]
        tried = [item for item in found if item.swaps]
        everywhere = sum(item.tone == "ok" for item in tried)
        compared = [reach.level2(slot, name, records.measured) for name in options]
        compared = [item for item in compared if item.compared]
        par = sum(len(item.on_a_par) == len(item.compared) for item in compared)
        assert not (compared and slot in reach.LEVEL2_NONE), slot
        title = chips.SLOT_TITLES[slot].replace(" ", "\u00a0")
        row = next(line for line in answer.splitlines() if f"[{title}](" in line)
        assert row.startswith("| "), slot  # a row of the visible table, not of a fold
        cells = [cell.strip() for cell in row.strip().strip("|").split(" | ")]
        assert cells[1:3] == [str(len(options)), f"{everywhere} of {len(tried)}"], slot
        if compared:
            assert cells[3] == f"{par} of {len(compared)}", slot
        else:  # nothing compared: a dash and why, never "0 of 0"
            assert cells[3] == generator.nothing_compared(slot, found_rows[slot]), slot
            assert cells[3].startswith("— (") and cells[3].endswith(")"), slot
    assert " 0 of 0 " not in answer
    # The slot × method scope, folded, without the column the marks already say.
    for line in generator.scope_table():
        assert f"    {line}" in page
    assert "| Applies to |" not in page
    # Every evidence page is a card; the encoders' card counts the table of its page.
    for _, link, _ in generator.EVIDENCE_PAGES:
        assert f"]({link})" in page and (REPO / "docs" / "evidence" / link).resolve().is_file()
    checks = generator.encoder_checks()
    counts = [len(names) for names in checks.values()]
    assert generator.encoder_card() in page and all(counts)


# -------------------------------------------------------------------- the slot index


def test_the_slot_index_maps_the_slots_from_the_code_and_claims_nothing_measured(measured):
    generator = docs_tool()
    chips = tool("docs_chips")
    slots = REPO / "docs" / "slots"
    page = (slots / "index.md").read_text(encoding="utf-8")
    # Generated, and the same whatever the records say: the index shows nothing measured.
    assert page == generator.slots_index(generator.Records(measured["cells"]))
    assert page == generator.slots_index(generator.Records([]))
    # Every slot page is linked from the index, and every card names a page that exists.
    for path in slots.glob("*.md"):
        assert path.name == "index.md" or f"]({path.name})" in page, path.name
    for card in generator.SLOTS.values():
        assert (slots / card.page).is_file(), card.page
    # Every slot a method has stands on exactly one card, with its scope and its options.
    carded = [slot for card in generator.SLOTS.values() for slot in card.slots]
    assert sorted(carded) == sorted(s for s in chips.SLOT_ORDER if chips.slot_methods(s))
    assert len(carded) == len(set(carded))
    for card in generator.SLOTS.values():
        text = page.split(f"**[{card.title}]({card.page})**")[1].split("\n-   ")[0]
        for slot in card.slots:
            assert chips.scope_chip(slot) in text, slot
            assert generator.option_count(slot) in text, slot
            assert generator.option_count(slot).startswith(f"{chips.option_count(slot)} ")
    # One count everywhere: what a user can --set (docs_chips), the losses included.
    assert chips.option_count("loss") == 20 and chips.option_count() == 244
    assert f"In all, {chips.option_count()} options you can `--set`." in page
    # The pipeline in both colour schemes, each image a link to itself at full size.
    for path, scheme in zip(generator.PIPELINE, ("light", "dark"), strict=True):
        assert (slots / path).resolve().is_file(), path
        assert f"]({path}#only-{scheme})]({path})" in page, scheme
    assert f"*{chips.LEVEL2_NAME}*" in page and "learns like the reference" not in page
    # Which method has which slot is the matrix of tools/docs_chips.py, read from the
    # configs: the losses are not HGGD's, the image encoder is HGGD's alone.
    assert "\n".join(generator.scope_table()) in page
    for line, short in zip(chips.scope_matrix("../"), generator.scope_table(), strict=True):
        cells, kept = line.split("|"), short.split("|")
        assert kept == [*cells[:2], *cells[3:]]  # the matrix less its column "Applies to"
    assert chips.scope("loss") == "seed-based" and chips.scope("image_encoder") == "hggd"
    loss_row = next(line for line in chips.scope_matrix() if line.startswith("| Loss |"))
    assert loss_row.endswith("| — |")  # the HGGD column
    # No measured class, no stale claim; the seed count is read from the configs.
    assert "model zoo is still empty" not in page
    assert not re.search(r"\b(universal|specific)\b", page)
    assert f"({generator.seeds_per_frame()})" in page
    assert generator.seeds_per_frame().endswith(" per frame in each of the five seed-based methods")
    for info in chips.methods():
        assert f"[{info.name}](../{info.page}#the-slots)" in page, info.name


def _slot_page_regions() -> dict[str, tuple[str, str]]:
    """Slot page → its "Use it" region and its "Options" region (``make_method_pages``)."""
    pages = tool("make_method_pages")
    found = {}
    for stem in pages.SLOT_PAGE_SLOTS:
        text = (REPO / "docs" / "slots" / f"{stem}.md").read_text(encoding="utf-8")
        use = text.split(pages.BEGIN)[1].split(pages.END)[0]
        options = text.split(pages.OPTIONS_BEGIN)[1].split(pages.OPTIONS_END)[0]
        found[stem] = (use, options)
    return found


def test_every_option_stands_on_one_slot_page_with_where_it_runs():
    """The options table of a slot page lists every option of its slot once, with the
    fraction of grasppanda.combine.reach; nothing measured is typed by hand."""
    pages = tool("make_method_pages")
    docs = docs_tool()
    chips = tool("docs_chips")
    regions = _slot_page_regions()
    for stem in pages.SLOT_PAGE_SLOTS:
        text = (REPO / "docs" / "slots" / f"{stem}.md").read_text(encoding="utf-8")
        assert text.count(pages.CAVEAT_SNIPPET) == 1, stem  # one caveat per page
        assert "*Class:" not in text and "Class at *runs*" not in text, stem
    by_slot: dict[str, list[str]] = {}
    for stem, slots in pages.SLOT_PAGE_SLOTS.items():
        for slot in slots:
            by_slot.setdefault(slot, []).append(stem)
    for slot, stems in by_slot.items():
        for option in docs.slot_options(slot):
            home = pages.option_page(slot, option.name)
            assert home in stems, (slot, option.name)
            if slot == "backbone" and not is_adapter(BACKBONES.get(option.component)):
                # Every encoder is on the page of its family, not on the overview.
                assert home in pages.ENCODER_PAGES, option.name
            # The name, then the next cell (or, for a loss, the kinds of term it fits).
            cells = (f"| `{option.name}`{pages.NOWRAP} |", f"| `{option.name}`{pages.NOWRAP}<br>")
            rows = [
                line.strip()
                for stem in stems
                for line in regions[stem][1].splitlines()
                if line.strip().startswith(cells)
            ]
            assert len(rows) == 1, (slot, option.name, len(rows))
            assert rows[0] in regions[home][1]
            found = chips.reach(slot, option.name)
            assert f">{html.escape(found.fraction)}</span>" in rows[0], (slot, option.name)


def test_a_slot_page_shows_its_slot_under_the_methods_that_have_it():
    """The "Use it" commands of a slot page name exactly the methods that have the slot,
    each with an option that ran there as a swap (the image encoders: HGGD alone)."""
    pages = tool("make_method_pages")
    chips = tool("docs_chips")
    configs = {info.name: info.config for info in chips.methods()}
    for stem, (use, _) in _slot_page_regions().items():
        slot = pages.SLOT_PAGE_SLOTS[stem][0]
        named = set(re.findall(r"grasppanda compose (\S+)", use))
        assert named == {configs[name] for name in chips.slot_methods(slot)}, stem
        for name in chips.slot_methods(slot):
            chosen = re.search(rf"compose {re.escape(configs[name])} \\\n\s+--set (\S+)=(\S+)", use)
            assert chosen, (stem, name)
            if slot != "loss":
                assert chosen.group(1) == slot
                assert name in chips.reach(slot, chosen.group(2)).reached, (stem, name)
    use = _slot_page_regions()["image_encoders"][0]
    assert "hggd/" in use and "gsnet/" not in use
    # Point painting has a section of its own (compose's refusal points there), labelled
    # with the words of grasppanda.combine.reach.
    page = (REPO / "docs" / "slots" / "image_encoders.md").read_text(encoding="utf-8")
    assert "## Point painting {#point-painting}" in page
    painting = page.split(pages.REACH_BEGIN)[1].split(pages.REACH_END)[0]
    found = chips.reach("backbone", "image-painted")
    assert found.fraction == "no record" and html.escape(found.tail) in html.escape(painting)
    # The point-painting fragment: a tab per kind of wrapper, in the order of the methods.
    fragment = page.split('??? note "The complete fragment')[1].split("\n??? ")[0]
    tabs = re.findall(r'^    === "([^"]+)"', fragment, re.M)
    assert tabs == ["GraspNet-baseline", "GSNet"]
    assert fragment.index("type: ImagePaintedSubsampled") < fragment.index("type: ImagePainted ")
    assert "../graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e.yaml" in fragment


def test_every_options_table_stands_in_its_wrapper_with_the_columns_in_order():
    """Every generated options table is wrapped for the stylesheet, which hides the second
    column on a phone: option, what it is, runs, trains like shipped; where an option comes
    from is in a fold. The chips under a title name the key of a command as code and count
    what docs_chips counts."""
    pages = tool("make_method_pages")
    chips = tool("docs_chips")
    begin, end = pages.OPTIONS_DIV
    allowed = {("Option", "What it is", "Runs", *level2) for level2 in ((), (chips.LEVEL2_HEADER,))}
    for stem, (_, options) in _slot_page_regions().items():
        lines = [line.strip() for line in options.splitlines()]
        at = [index for index, line in enumerate(lines) if line.startswith("| Option |")]
        headers = [tuple(cell.strip() for cell in lines[i].strip("|").split("|")) for i in at]
        assert headers and all(header in allowed for header in headers), (stem, headers)
        assert options.count(begin) == len(headers), stem
        for index in at:
            assert lines[index - 2] == begin, (stem, lines[index])
            assert end in lines[index:], stem
        text = (REPO / "docs" / "slots" / f"{stem}.md").read_text(encoding="utf-8")
        assert "Learns like the reference" not in text and "learns like the reference" not in text
        head = text.split(pages.HEAD_BEGIN)[1].split(pages.HEAD_END)[0]
        assert '<code class="gp-chip">--set ' in head, stem
        slots = pages.SLOT_PAGE_SLOTS[stem]
        if stem == pages.SLOT_PAGE[slots[0]]:
            count = sum(chips.option_count(slot) for slot in slots)
            assert f">{count} options</span>" in head, stem
    # A tab whose rows share one source says it in its label instead of on every row.
    image = _slot_page_regions()["image_encoders"][1]
    assert '=== "PyTorch Image Models (extra timm) · 24"' in image
    assert image.count("PyTorch Image Models") == 1


def test_every_slot_page_folds_where_each_option_comes_from():
    """The options tables have no source column: a page without a fold of its own lists, in
    the generated fold "Sources and extras", the work every option follows (as the catalogue
    says it) and the extra it needs, unless the label of its tab says it. The source of a
    preset that is one work's setting of a generic component is the work's, from the preset's
    header. The scene-encoder pages give the work and the extra of each class."""
    pages = tool("make_method_pages")
    for stem, (_, options) in _slot_page_regions().items():
        labels = re.findall(r'^=== "(.+) · \d+"$', options, re.M)
        slots = pages.SLOT_PAGE_SLOTS[stem]
        generated = '??? note "Sources and extras"' in options
        assert generated == ("backbone" not in slots and stem not in pages.SOURCES_BY_HAND), stem
        if not generated:
            continue
        fold = options.split('??? note "Sources and extras"')[1].split("\n??? ")[0]
        rows = [line.strip() for line in fold.splitlines() if line.strip().startswith("| ")]
        for slot in slots:
            for option in pages._page_options(stem, slot):
                if option.source in ("", "?") or pages._source_cell(option, code=False) in labels:
                    continue
                row = next(r for r in rows[1:] if f"`{option.name}`" in r.split(" | ")[-1])
                assert pages._plain(option.source)[1:40] in row, (stem, option.name)
                assert all(f"`{extra}`" in row for extra in option.extras), (stem, option.name)
    recipe = _slot_page_regions()["recipe"][1].split('??? note "Sources and extras"')[1]
    for name in ("encoder-lr-0.1", "ptv3-adamw-onecycle"):
        row = next(line for line in recipe.splitlines() if f"`{name}`" in line.split(" | ")[-1])
        assert "Pointcept" in row and "BEiT" not in row, name
    point = _slot_page_regions()["scene_encoders_point"][1]
    fold = point.split('??? note "Sources and licences"')[1].split("\n??? ")[0]
    row = next(line for line in fold.splitlines() if "| `PyGPointEncoder` |" in line)
    assert "EdgeConv" in row and "PointGNNConv" in row and "| `torch_geometric` |" in row


def test_the_option_tables_say_what_the_records_and_the_grid_say_beside_the_fraction():
    """A loss measured inside a head bundle shows that bundle; a recipe preset that names the
    scene encoder says it is for the seed-based methods (grid.RECIPE_ENCODER_PRESETS, the
    presets compose refuses under a method without one)."""
    regions = _slot_page_regions()
    losses = regions["losses"][1]
    row = next(line for line in losses.splitlines() if "`distribution-focal`" in line)
    found = reach.reach("loss", "distribution-focal")
    assert found.reached and f">{found.fraction}</span>" in row
    assert "measured in `decoupled-grid-bins`" in row
    recipe = regions["recipe"][1]
    for key in RECIPE_ENCODER_PRESETS:
        name = key.split("/")[1]
        row = next(line for line in recipe.splitlines() if line.strip().startswith(f"| `{name}`"))
        assert "<small>seed-based only</small>" in row, name


def test_the_scene_encoder_pages_say_where_each_class_comes_from():
    """Every scene-encoder page folds the work, the code base (THIRD_PARTY_NOTICES.md) and
    the output of its classes; the released-weights folds give the revision of a hub file,
    and the checksums typed on the point page are those the integration test checks."""
    pages = tool("make_method_pages")
    regions = _slot_page_regions()
    for stem in ("scene_encoders", *pages.ENCODER_PAGES):
        options = regions[stem][1]
        fold = options.split('??? note "Sources and licences"')[1].split("\n??? ")[0]
        for option in pages._page_options(stem, "backbone"):
            assert f"| `{option.component}` |" in fold, (stem, option.component)
    transformer = regions["scene_encoders_transformer"][1]
    assert "| `PTv3` |" in transformer and "[Pointcept/Pointcept](" in transformer
    assert "[hub @ `5e25cfa`](https://huggingface.co/Pointcept/PointTransformerV3/resolve/" in (
        transformer
    )
    sparse = regions["scene_encoders_sparse"][1]
    assert "| `SparseUNet14` |" in sparse and pages.WRITTEN_HERE in sparse
    module = ast.parse(
        (REPO / "tests" / "integration" / "test_backbone_released_weights.py").read_text()
    )
    cases = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CASES"
    )
    page = (REPO / "docs" / "slots" / "scene_encoders_point.md").read_text(encoding="utf-8")
    typed = set(re.findall(r"`([0-9a-f]{16})`", page))
    assert typed == {sha256[:16] for *_, sha256 in cases}


def test_a_runs_unstable_probe_is_repeated_and_refusals_keep_their_kinds(tmp_path, monkeypatch):
    """A class counts a run as unstable only after a repeat: the runner probes a
    runs-unstable config a second time and the owner's record keeps the repeat, in the
    file as the probe's fields. A refusal keeps compose's kind of every message."""
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    shaky = {
        "verdict": "runs-unstable",
        "detail": "trained 2 iterations, but 1 of 2 iterations had no real seed",
        "judge": 2,
        "iterations": 2,
        "grasps_per_frame": [3, 0],
        "seconds": 1.0,
    }
    calm = {"verdict": "runs", "judge": 2, "iterations": 2, "grasps_per_frame": [5, 4]}
    outcomes: list[dict] = []
    tags: list[str] = []

    def probe_config(config, options, tag, name, args):
        tags.append(tag)
        return "grasppanda probe x --iters 2 --frames 8", dict(outcomes.pop(0))

    monkeypatch.setattr(runner, "probe_config", probe_config)
    arguments = ["run", "--name", "shaky", "--only", "reference/GSNet", "--iters", "2"]
    arguments += ["--commit", "abcdef0123"]
    outcomes[:] = [shaky, calm]
    assert runner.main(arguments) == 0
    assert tags == ["reference__GSNet", "reference__GSNet.repeat"]
    record = runner.load("shaky")["reference/GSNet"]
    assert record["verdict"] == "runs-unstable" and record["repeat"]["verdict"] == "runs"
    kept = runner.compact(record)
    assert kept["repeat"] == {
        "verdict": "runs",
        "judge": 2,
        "iterations": 2,
        "grasps_per_frame": [5, 4],
        "trend": {},
        "no_gradient": [],
        "unchanged": [],
        "checked": [0, 0],
    }
    assert "repeat" in runner.FIELDS
    # --no-repeat probes once; a clean run is never repeated.
    outcomes[:] = [shaky]
    assert runner.main(["run", "--name", "once", *arguments[3:], "--no-repeat"]) == 0
    assert "repeat" not in runner.load("once")["reference/GSNet"] and not outcomes
    outcomes[:] = [calm]
    assert runner.main(["run", "--name", "calm", *arguments[3:]]) == 0 and not outcomes
    # `repeat` adds the second probe to a run made without one.
    outcomes[:] = [shaky]
    assert runner.main(["repeat", "--name", "once"]) == 0
    assert runner.load("once")["reference/GSNet"]["repeat"]["verdict"] == "runs-unstable"
    # A refusal: the kinds compose reported, parallel to the messages, in the record and the file.
    refused = {
        "id": "stage/GSNet/head/x",
        "group": "stage",
        "method": "GSNet",
        "sets": ["head=x"],
        "slot": "head",
        "component": "X",
        "variant": "x",
        "verdict": "refused",
        "refused_by": "compose",
        "refusal": ["a: b", "c: d"],
        "refusal_kinds": ["structural", "policy"],
    }
    assert runner.compact(refused)["refusal_kinds"] == ["structural", "policy"]
    assert "refusal_kinds" in runner.FIELDS


def test_a_failure_message_keeps_no_path_of_the_machine():
    runner = tool("probe_grid")
    message = (
        'File "/' + 'home/me/GraspPanda/.claude/worktrees/x/grasppanda/combine/probe.py", '
        "line 650, in unchanged; cannot read /tmp/run/a.yaml; a ratio 3/4"
    )
    assert runner._no_paths(message) == (
        'File "grasppanda/combine/probe.py", line 650, in unchanged; cannot read <path>; '
        "a ratio 3/4"
    )
