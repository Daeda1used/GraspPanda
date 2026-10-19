"""Level 2 of the measured grid: does an option learn like the stage it replaces?

Nothing is trained here. The comparison of ``tools/learn_check.py`` is tested on synthetic
records, the options it adds to ``tools/probe_grid.py`` with a fake probe, and the
committed summary and page against the generator.
"""

from __future__ import annotations

import importlib.util
import json
import sys

from grasppanda.combine.grid import METHODS, Cell

from .conftest import REPO


def tool(name: str):
    """A module of ``tools/``, loaded by path (the directory is not a package on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- probe_grid options


def test_a_seeded_run_records_the_hash_of_the_seeded_config_and_predicts_more_frames(
    tmp_path, monkeypatch
):
    """`--seed` is part of the experiment: the record carries the hash of the config with
    that seed, so a probe of seed 0 never stands for seed 1. `--predict-frames` reaches the
    probe. Without either option, the commands are those of level 1."""
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    commands: list[list[str]] = []

    def run(command, log, timeout):
        commands.append(command)
        report = {"verdict": "runs", "judge": 2, "iterations": 2, "seconds": 1.0}
        return 0, json.dumps(report | {"schedule": "constant", "rate": 0.001})

    monkeypatch.setattr(runner, "_run", run)
    reference = Cell("reference/GSNet", "reference", "GSNet", METHODS["GSNet"])
    monkeypatch.setattr(runner, "cells", lambda groups=None: [reference])
    base = ["run", "--iters", "2", "--commit", "abcdef0123"]

    assert runner.main([*base, "--name", "plain"]) == 0
    plain = runner.load("plain")["reference/GSNet"]
    assert plain["probe_command"].endswith("--iters 2 --frames 8 --schedule constant")
    assert "seed" not in plain and "predict_frames" not in json.loads(
        (runner.directory("plain") / "run.json").read_text()
    )

    seeded = [*base, "--name", "seeded", "--seed", "1", "--predict-frames", "4"]
    assert runner.main(seeded) == 0
    record = runner.load("seeded")["reference/GSNet"]
    assert record["probe_command"].endswith(
        "--iters 2 --frames 8 --predict-frames 4 --schedule constant --opts runtime.seed=1"
    )
    assert record["seed"] == 1
    assert record["config_hash"] == runner.seeded_hash(METHODS["GSNet"], 1)
    assert record["config_hash"] != plain["config_hash"]
    environment = json.loads((runner.directory("seeded") / "run.json").read_text())
    assert environment["seed"] == 1 and environment["predict_frames"] == 4
    # Seed 0 is the seed of every shipped config: the same experiment, the same hash.
    assert runner.seeded_hash(METHODS["GSNet"], 0) == plain["config_hash"]
    assert runner.seeded_hash(METHODS["GSNet"], None) is None


def test_a_probe_the_gpu_failed_never_stands_for_its_config():
    runner = tool("probe_grid")
    fault = {
        "verdict": "fails",
        "judge": 2,
        "failure": {
            "error": "AcceleratorError",
            "message": "CUDA error: unspecified launch failure",
        },
    }
    assert not runner._conclusive({"probe": fault})
    real = fault | {"failure": {"error": "ValueError", "message": "shapes differ"}}
    assert runner._conclusive({"probe": real})


def test_run_retry_failed_probes_again_only_what_the_machine_failed(tmp_path, monkeypatch):
    learn = checker()
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    crash = {"verdict": "fails", "failure": {"error": "ProcessError", "message": "crash"}}
    fault = {"verdict": "fails", "failure": {"error": "X", "message": "unspecified launch failure"}}
    broken = {"verdict": "fails", "failure": {"error": "ValueError", "message": "shapes differ"}}
    assert learn.machine_failure(crash) and learn.machine_failure(fault)
    assert not learn.machine_failure(broken) and not learn.machine_failure(report())
    assert "run --retry-failed" in learn._failure(crash)
    folder = runner.directory("L2-encoder")
    folder.mkdir(parents=True)
    rows = [
        {"id": "encoder/GSNet/A", "probe": crash},
        {"id": "encoder/GSNet/B", "probe": broken},
        {"id": "encoder/GSNet/C", "probe": report()},
    ]
    (folder / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    level1 = {
        "cells": [
            level1_record("reference/GSNet", "reference", "runs", "h0"),
            *(level1_record(f"encoder/GSNet/{n}", "encoder", "runs", f"h{n}") for n in "ABCD"),
        ]
    }
    (tmp_path / "level1.json").write_text(json.dumps(level1))
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "main", lambda argv: calls.append(argv) or 0)
    arguments = ["run", "--kind", "encoder", "--level1", str(tmp_path / "level1.json")]
    assert learn.main([*arguments, "--retry-failed"]) == 0

    def only(argv: list[str]) -> list[str]:
        rest = argv[argv.index("--only") + 1 :]
        return rest[: next(i for i, word in enumerate(rest) if word.startswith("--"))]

    names = [argv[argv.index("--name") + 1] for argv in calls]
    assert names == ["L2-reference", "L2-encoder", "L2-encoder"]
    assert only(calls[0]) == ["reference/GSNet"]
    assert "--redo" in calls[1] and only(calls[1]) == ["encoder/GSNet/A"]
    assert "--redo" not in calls[2] and only(calls[2]) == [f"encoder/GSNet/{n}" for n in "ABCD"]
    assert calls[2][calls[2].index("--iters") + 1] == "250"
    assert calls[2][calls[2].index("--predict-frames") + 1] == "4"
    calls.clear()
    assert learn.main(arguments) == 0  # without the flag nothing is probed again
    assert [("--redo" in argv) for argv in calls] == [False, False]


# ------------------------------------------------------------- the comparison

ITERATIONS = 250


def checker():
    return tool("learn_check")


def report(
    terms: dict[str, list[float | None]] | None = None,
    *,
    verdict: str = "runs",
    grasps: tuple[int, ...] = (100, 120, 90, 80),
    iterations: int = ITERATIONS,
    **extra,
):
    """A probe report of a level-2 run with these per-iteration losses (a single value is
    held for every iteration)."""
    terms = terms if terms is not None else {"loss/a": [1.0], "loss/b": [2.0]}
    losses = {
        name: (values * iterations if len(values) == 1 else values)[:iterations]
        for name, values in terms.items()
    }
    losses["loss"] = [
        sum(series[i] or 0.0 for series in losses.values()) for i in range(iterations)
    ]
    out = {
        "verdict": verdict,
        "detail": "trained",
        "iterations_requested": ITERATIONS,
        "iterations": iterations,
        "losses_per_iteration": losses,
        "grasps_per_frame": list(grasps),
        "nonfinite": [],
        "no_gradient": [],
        "unchanged": [],
    }
    return out | extra


def ramp(start: float, end: float, count: int = ITERATIONS) -> list[float]:
    return [start + (end - start) * i / (count - 1) for i in range(count)]


def test_a_cell_within_the_tolerance_is_on_a_par_and_lower_terms_are_only_reported():
    learn = checker()
    reference = report({"loss/a": [1.0], "loss/b": [2.0]})
    judged = learn.compare(report({"loss/a": [1.2], "loss/b": [1.0]}), reference)
    assert judged["outcome"] == "on a par", judged
    assert judged["reasons"] == []
    assert abs(judged["terms"]["loss/a"]["ratio"] - 1.2) < 1e-9
    assert "loss" not in judged["terms"]  # the total is not judged
    assert judged["notes"] == ["lower than the reference: loss/b 0.50x"]
    # A score head that returns one score for every grasp is reported, not judged.
    flat = [[-0.31, -0.31, -0.31], None, [-0.31, -0.31, -0.31], [-0.31, -0.31, -0.31]]
    judged = learn.compare(report(score_quantiles=flat), reference)
    assert judged["outcome"] == "on a par" and learn.CONSTANT_SCORES in judged["notes"]
    spread = [[-1.0, -0.3, 0.5], [-0.31, -0.31, -0.31]]
    assert (
        learn.CONSTANT_SCORES
        not in learn.compare(report(score_quantiles=spread), reference)["notes"]
    )


def test_a_term_above_the_tolerance_makes_a_cell_worse():
    learn = checker()
    reference = report({"loss/a": [1.0], "loss/b": [2.0]})
    above = 2.0 * (learn.TOLERANCE + 0.05)
    judged = learn.compare(report({"loss/a": [1.0], "loss/b": [above]}), reference)
    assert judged["outcome"] == "worse"
    assert judged["reasons"] == [
        f"loss/b {learn.TOLERANCE + 0.05:.3f}x the reference (tolerance {learn.TOLERANCE:g})"
    ]
    # Only the last window counts: a term that was high early and ends level is on a par.
    early = [5.0] * (ITERATIONS - learn.WINDOW) + [2.0] * learn.WINDOW
    later = learn.compare(report({"loss/a": [1.0], "loss/b": early}), reference)
    assert later["outcome"] == "on a par"
    # A part that no gradient reached is worse, whatever its terms.
    dead = report({"loss/a": [1.0], "loss/b": [2.0]}, no_gradient=["proposal", "head.net"])
    judged = learn.compare(dead, reference)
    assert judged["outcome"] == "worse" and judged["reasons"] == ["no gradient reached proposal"]


def test_a_method_whose_references_spread_more_gets_a_wider_tolerance():
    learn = checker()
    # [seed 0, other seed, ratio, rise under seed 0, rise under the other]
    stable = {"loss/a": [1.0, 1.05, 1.05, -0.1, -0.12], "loss": [9.0, 1.0, 0.11, 0.0, 0.9]}
    assert learn.calibrate(stable) == {"tolerance": 1.25, "rising_above": 0.1}
    noisy = {
        "loss/a": [0.067, 0.051, 0.759, -0.352, -0.461],  # 1/0.759 = 1.317
        "loss/b": [0.66, 1.02, 1.552, -0.134, -0.139],
        "loss/c": [0.77, 0.79, 1.03, -0.063, -0.429],  # rises 0.366 apart
    }
    assert learn.calibrate(noisy) == {"tolerance": 1.6, "rising_above": 0.4}
    assert learn.calibrate({"loss/a": [1.0, 0.759, 0.759, None, None]})["tolerance"] == 1.35
    # A spread on a multiple of the step gets the next one: the tolerance is above it.
    assert learn.calibrate({"loss/a": [1.0, 1.3, 1.3, 0.0, 0.0]})["tolerance"] == 1.35
    assert learn.calibrate(None) == {"tolerance": learn.TOLERANCE, "rising_above": 0.1}
    # The tolerance a method gets is the one its cells are judged with.
    reference = report({"loss/a": [1.0]})
    cell = report({"loss/a": [1.5]})
    assert learn.compare(cell, reference)["outcome"] == "worse"
    assert learn.compare(cell, reference, tolerance=1.6)["outcome"] == "on a par"


def test_a_rising_term_a_non_finite_value_or_no_grasp_is_diverging():
    learn = checker()
    reference = report({"loss/a": [1.0], "loss/b": [2.0]})
    rising = [1.0] * 125 + ramp(1.0, 1.0 + 6 * learn.RISING_ABOVE, 125)
    judged = learn.compare(report({"loss/a": rising, "loss/b": [2.0]}), reference)
    assert judged["outcome"] == "diverging"
    assert judged["reasons"][0].startswith("loss/a rises +")
    # A rise the reference shows too is the method's, not the option's.
    alike = report({"loss/a": rising, "loss/b": [2.0]})
    assert learn.compare(alike, alike)["outcome"] == "on a par"
    # Diverging comes before worse: both reasons are kept, the rise first.
    both = learn.compare(report({"loss/a": rising, "loss/b": [4.0]}), reference)
    assert both["outcome"] == "diverging" and both["reasons"][0].startswith("loss/a rises")
    assert "loss/b 2.000x the reference (tolerance 1.25)" in both["reasons"]
    # A non-finite loss ends the training early: diverging, not a failed run.
    nan = report(
        {"loss/a": [1.0], "loss/b": [2.0]},
        verdict="runs-unstable",
        iterations=120,
        nonfinite=["loss/b is nan at iteration 121"],
    )
    judged = learn.compare(nan, reference)
    assert judged["outcome"] == "diverging"
    assert judged["reasons"] == ["non-finite: loss/b is nan at iteration 121"]
    empty = report(verdict="runs-empty", grasps=(0, 0, 0, 0))
    judged = learn.compare(empty, reference)
    assert judged["outcome"] == "diverging"
    assert judged["reasons"] == ["no grasp on any of the 4 frames"]
    # ... unless the reference predicts none either: then it is a note.
    judged = learn.compare(empty, report(grasps=(0, 0, 0, 0)))
    assert judged["outcome"] == "on a par" and judged["notes"]


def test_terms_the_cell_redefines_are_not_comparable_and_the_cell_is_judged_on_the_others():
    learn = checker()
    reference = report({"loss/a": [1.0], "loss/b": [2.0]})
    cell = report({"loss/a": [1.0], "loss/b": [9.0], "loss/c": [4.0]})
    judged = learn.compare(cell, reference, {"loss/b": "its loss is replaced (Focal)"})
    assert judged["outcome"] == "on a par"
    assert judged["terms"]["loss/b"]["not_comparable"] == "its loss is replaced (Focal)"
    assert judged["terms"]["loss/c"]["not_comparable"] == "only the cell reports it"
    gone = learn.compare(report({"loss/a": [1.0]}), reference)
    assert gone["terms"]["loss/b"]["not_comparable"] == "the cell does not report it"
    assert gone["outcome"] == "on a par"
    # No comparable term at all: not comparable.
    everything = {"loss/a": "replaced", "loss/b": "replaced"}
    assert learn.compare(cell, reference, everything)["outcome"] == "not comparable"
    negative = report({"loss/a": [-0.5], "loss/b": [2.0]})
    judged = learn.compare(negative, reference)
    assert judged["terms"]["loss/a"]["not_comparable"] == "a mean that is not positive"


def test_a_run_that_did_not_finish_fails_and_a_missing_reference_is_said():
    learn = checker()
    reference = report()
    crashed = {
        "verdict": "fails",
        "failure": {"phase": "loss", "error": "RuntimeError", "message": "shapes differ"},
        "iterations_requested": ITERATIONS,
        "iterations": 12,
    }
    judged = learn.compare(crashed, reference)
    assert judged["outcome"] == "fails"
    assert judged["reasons"] == ["failed at loss: RuntimeError: shapes differ"]
    assert learn.compare(None, reference)["outcome"] == "fails"
    short = report(iterations=200)
    assert learn.compare(short, reference)["reasons"] == ["only 200 of 250 iterations ran"]
    level1 = report(iterations=20) | {"iterations_requested": 20}
    assert learn.compare(level1, reference)["reasons"] == ["a run of 20 iterations, not 250"]
    wired = report(verdict="wired", fallback="the dataset is not on this machine")
    assert learn.compare(wired, reference)["outcome"] == "fails"
    judged = learn.compare(report(), None)
    assert judged["outcome"] == "no reference" and judged["reasons"] == ["the reference: no run"]
    assert learn.compare(report(), crashed)["outcome"] == "no reference"


def test_the_measures_of_a_term():
    learn = checker()
    assert learn.window_mean([None] * 10 + [2.0] * 25) == 2.0
    assert learn.window_mean([1.0, float("nan"), 3.0], window=3) == 2.0
    assert learn.window_mean([None, None]) is None
    # The last half of 250 iterations is 125 of them: 62 against 62, the middle in neither.
    series = [9.0] * 125 + [1.0] * 62 + [5.0] + [1.5] * 62
    assert abs(learn.rise(series) - 0.5) < 1e-12
    assert learn.rise([1.0]) is None
    assert learn.blocks([1.0, 3.0, None, 5.0, None, None], size=2) == [2.0, 5.0, None]


def gsnet_model() -> dict:
    from grasppanda.config import Config

    return json.loads(json.dumps(Config.load(METHODS["GSNet"]).resolved()["model"]))


def test_which_terms_a_swap_redefines():
    learn = checker()
    reference = gsnet_model()
    assert set(learn.term_owners(reference)) == {
        "loss/objectness",
        "loss/graspness",
        "loss/view",
        "loss/score",
        "loss/width",
    }
    # An encoder, a grouper or a recipe leaves every term as it is.
    encoder = gsnet_model() | {"backbone": {"type": "PointNeXt"}}
    assert learn.not_comparable("encoder", encoder, reference) == {}
    # A loss preset: only the term it is put in.
    loss = gsnet_model()
    loss["head"]["score_loss"] = {"type": "QualityFocal", "weight": 15.0}
    assert set(learn.not_comparable("loss", loss, reference)) == {"loss/score"}
    # A head bundle: every head term, the new head's included.
    head = gsnet_model()
    head["head"] = {"type": "EconomicHead"}
    assert set(learn.not_comparable("head", head, reference)) == {
        "loss/score",
        "loss/width",
        "loss/angle",
        "loss/depth",
    }
    # An assigner: the labels of the view stage and of the head.
    assigner = gsnet_model()
    assigner["assigner"] = {"type": "CompactLabelAssigner"}
    assert set(learn.not_comparable("assigner", assigner, reference)) == {
        "loss/view",
        "loss/score",
        "loss/width",
    }
    # A proposal that inherits the loss of the reference's (only its sampling differs)
    # keeps its terms; a view stage with other code redefines its term.
    scored = gsnet_model()
    scored["proposal"] = scored["proposal"] | {"type": "ScoredSeeds"}
    assert learn.not_comparable("proposal", scored, reference) == {}
    # ... also when it leaves its losses to the stage, whose defaults are the reference's.
    left = gsnet_model()
    left["proposal"] = {"type": "ScoredSeeds", "objectness_loss": None, "graspness_loss": None}
    assert learn.not_comparable("proposal", left, reference) == {}
    heavier = gsnet_model()
    heavier["proposal"]["graspness_loss"] = {"type": "SmoothL1", "weight": 20.0}
    assert learn.not_comparable("proposal", heavier, reference) == {
        "loss/graspness": "its loss is replaced (SmoothL1)"
    }
    lattice = gsnet_model()
    lattice["view"] = {"type": "LatticeViewStage", "loss": reference["view"]["loss"]}
    assert set(learn.not_comparable("view", lattice, reference)) == {"loss/view"}
    # A regression view stage declares no loss option and still reports `loss/view`.
    regression = gsnet_model()
    regression["view"] = {"type": "TwoVectorViewStage"}
    assert learn.not_comparable("view", regression, reference) == {
        "loss/view": "another view stage computes it (TwoVectorViewStage for TemplateViewStage)"
    }
    # HGGD has no stages of the skeleton: every term is comparable.
    assert learn.term_owners({"type": "HGGD", "image_encoder": {"type": "X"}}) == {}


def level1_record(cell_id: str, group: str, verdict: str, config_hash: str | None, **extra) -> dict:
    method = cell_id.split("/")[1]
    return {
        "id": cell_id,
        "group": group,
        "method": method,
        "verdict": verdict,
        "config_hash": config_hash,
    } | extra


def test_cells_are_selected_from_the_level_one_records():
    learn = checker()
    records = [
        level1_record("reference/GSNet", "reference", "runs", "h0"),
        level1_record(
            "encoder/GSNet/SparseUNet14", "encoder", "runs", "h0", probed_as="reference/GSNet"
        ),
        level1_record("encoder/GSNet/A", "encoder", "runs", "h1"),
        level1_record("encoder/GSNet/B", "encoder", "runs-empty", "h2"),
        level1_record("encoder/GSNet/C", "encoder", "refused", None),
        level1_record("stage/GSNet/grouper/ball", "stage", "runs-unstable", "h3", slot="grouper"),
        level1_record(
            "family-slot/HGGD/image_encoder/x", "family-slot", "runs", "h4", slot="image_encoder"
        ),
        level1_record("post/GSNet/nms", "post", "runs", "h5", slot="post"),
    ]
    chosen, own = learn.select(records, "encoder")
    assert [record["id"] for record in chosen] == ["encoder/GSNet/A", "encoder/GSNet/B"]
    assert own == ["encoder/GSNet/SparseUNet14"]
    assert [r["id"] for r in learn.select(records, "grouper")[0]] == ["stage/GSNet/grouper/ball"]
    assert [r["id"] for r in learn.select(records, "image-encoder")[0]] == [
        "family-slot/HGGD/image_encoder/x"
    ]
    assert learn.kind_of(records[-1]) is None  # post-processing is left out
    assert learn.reference_id("HGGD") == "family/HGGD"
    assert learn.reference_id("GSNet") == "reference/GSNet"


# ----------------------------------------------------------- summary and page


def test_export_compares_the_recorded_runs_and_the_page_renders_them(tmp_path, monkeypatch):
    """Run directories as probe_grid writes them, a level-1 file, and an export: the
    summary judges every recorded cell against its reference, holds the noise of the
    references, keeps the losses in the side file, and the page renders it."""
    learn = checker()
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    measured = {"date": "2026-10-15", "commit": "abcdef0"}

    def write(run: str, *records: dict) -> None:
        folder = runner.directory(run)
        folder.mkdir(parents=True)
        lines = [json.dumps(record) for record in records]
        (folder / "results.jsonl").write_text("\n".join(lines) + "\n")

    def record(cell_id: str, probe: dict, **extra) -> dict:
        return {
            "id": cell_id,
            "method": cell_id.split("/")[1],
            "verdict": probe["verdict"],
            "probe": probe,
            "probe_command": f"grasppanda probe {cell_id} --iters 250",
            "config_hash": "0123456789abcdef0123",
            "measured": measured,
        } | extra

    terms = {"loss/objectness": [0.2], "loss/score": [0.1]}
    write("L2-reference", record("reference/GSNet", report(terms)))
    seed1 = report({"loss/objectness": [0.21], "loss/score": [0.09]})
    write("L2-reference-seed1", record("reference/GSNet", seed1))
    adamw = record(
        "recipe/GSNet/optimizer/adamw",
        report({"loss/objectness": [0.4], "loss/score": [0.1]}),
        config=METHODS["GSNet"],
        sets=["optimizer=adamw"],
        compose_command="grasppanda compose gsnet --set optimizer=adamw",
    )
    lion = record("recipe/GSNet/optimizer/timm-lion", report(terms), config=METHODS["GSNet"])
    write("L2-recipe", adamw, lion)
    level1 = {
        "cells": [
            level1_record("reference/GSNet", "reference", "runs", "h0"),
            level1_record("recipe/GSNet/optimizer/adamw", "recipe", "runs", "h1"),
            level1_record("recipe/GSNet/optimizer/timm-lion", "recipe", "runs-empty", "h2"),
            level1_record("recipe/GSNet/optimizer/sgd", "recipe", "runs", "h3"),
        ]
    }
    (tmp_path / "level1.json").write_text(json.dumps(level1))
    target = tmp_path / "out" / "learns.json"
    arguments = ["export", "--to", str(target), "--level1", str(tmp_path / "level1.json")]
    assert learn.main(arguments) == 0
    summary = json.loads(target.read_text())
    by_id = {line["id"]: line for line in summary["cells"]}
    assert by_id["recipe/GSNet/optimizer/adamw"]["outcome"] == "worse"
    assert by_id["recipe/GSNet/optimizer/adamw"]["terms"]["loss/objectness"] == [0.4, 0.2, 2.0]
    assert by_id["recipe/GSNet/optimizer/timm-lion"]["outcome"] == "on a par"
    assert by_id["recipe/GSNet/optimizer/timm-lion"]["level1"] == "runs-empty"
    assert summary["about"]["kinds"]["recipe"] == {
        "selected": 3,
        "measured": 2,
        "the reference itself": 0,
    }
    noise = summary["about"]["noise"]["GSNet"]["seed 1"]
    assert noise["loss/objectness"][:3] == [0.2, 0.21, 1.05]
    assert [line["id"] for line in summary["references"]] == [
        "reference/GSNet",
        "reference/GSNet@seed1",
    ]
    assert summary["about"]["runs"] == {"2026-10-15@abcdef0": measured}
    side = json.loads(learn.side_file(target).read_text())
    assert {entry["id"] for entry in side["cells"]} == {
        "reference/GSNet",
        "reference/GSNet@seed1",
        "recipe/GSNet/optimizer/adamw",
        "recipe/GSNet/optimizer/timm-lion",
    }
    cached = tmp_path / "cache" / "verification" / "L2" / "recipe"
    assert (cached / "recipe__GSNet__optimizer__adamw.json").is_file()
    page = learn.page(summary, target)
    # One row per option: a chip per method (its ratios as the tooltip); under the table, in
    # words, why a cell is not on a par, and nothing for a cell that is.
    row = next(line for line in page.splitlines() if "| optimizer=adamw |" in line)
    assert '<span class="gp-chip gp-chip--warn"' in row and ">worse</span>" in row
    assert "Ratios to the reference: objectness 2.00, score 1.00." in row
    assert any(">✓</span>" in line for line in page.splitlines())
    recipe = page.split("### Training recipe")[1].split("\n## ")[0]
    reasons = [line.strip() for line in recipe.splitlines() if line.strip().startswith("- `")]
    assert reasons == ["- `optimizer=adamw`: GSNet: worse (objectness 2.00x)"]
    assert "2 of 3 selected cells run; 1 not yet." in page
    assert "Not run yet (0 cells selected)." in page  # a kind without a cell
    # The answer counts the options of a kind under each method.
    answer = page.split("## The answer, kind by method")[1].split("## How a cell")[0]
    recipe = next(line for line in answer.splitlines() if "[Training recipe](#" in line)
    assert 'title="on a par">1</span>' in recipe and 'title="worse">1</span>' in recipe


def test_the_page_is_generated_from_the_committed_summary():
    learn, chips = checker(), tool("docs_chips")
    assert learn.main(["page", "--check"]) == 0, "run `python -m tools.learn_check page`"
    page = learn.PAGE.read_text(encoding="utf-8")
    assert page.startswith("<!-- Generated by tools/learn_check.py")
    for sentence in (
        "# Does it train like the shipped component?",
        f"*{chips.LEVEL2_NAME}*",
        "One run per cell, 250 iterations on 8 frames",
        "not an accuracy measurement",
        "## The noise floor",
        "## Not on this page {#not-on-this-page}",
        chips.measured_on(2),
    ):
        assert sentence in page, sentence
    assert "learn like the reference" not in page.lower()
    assert page.count('--8<-- "docs/includes/caveat_probe.md"') == 1
    assert "advanced_guides/learns_like_reference.md" in (REPO / "mkdocs.yml").read_text()
    # The answer: one row per kind, the methods in page order; a section per kind.
    methods = [info.name for info in chips.methods()]
    assert "| Kind | " + " | ".join(methods) + " |" in page
    for kind in learn.PAGE_KINDS:
        title = learn.KINDS[kind][2]
        assert f"### {title} {{#{learn._anchor(title)}}}" in page


def _level2_by_kind(summary: dict) -> dict:
    """Kind → (slot, option) → its grasppanda.combine.reach.level2, for every option of the
    grid with a cell in ``summary``: computed apart from the page's own code."""
    from grasppanda.combine import reach
    from grasppanda.combine.options import _class_index

    kinds = {line["id"]: line["kind"] for line in summary["cells"]}
    found: dict = {}
    for key, by_method in _class_index().items():
        mine = [kinds[cell] for ids in by_method.values() for cell in ids if cell in kinds]
        if mine:
            found.setdefault(mine[0], {})[key] = reach.level2(*key)
    return found


def test_the_page_counts_what_reach_level2_counts():
    """The level-2 page counts what grasppanda.combine.reach.level2 counts, as the slot
    pages, the method pages and the explorer do: a cell of the answer is the outcomes of the
    kind's options under the method; a swap that did not run at level 1, a cell trained on a
    schedule other than its reference's and the schedules are not counted, and the page
    names each of them under *Not on this page*."""
    from collections import Counter

    learn, chips = checker(), tool("docs_chips")
    page = learn.PAGE.read_text(encoding="utf-8")
    summary = json.loads(learn.SUMMARY.read_text(encoding="utf-8"))
    by_kind = _level2_by_kind(summary)
    methods = [info.name for info in chips.methods()]
    answer = page.split("## The answer, kind by method")[1].split("## How a cell")[0]
    for kind in learn.PAGE_KINDS:
        title = learn.KINDS[kind][2]
        row = next(line for line in answer.splitlines() if f"[{title}](#" in line)
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")][1:]
        for method, cell in zip(methods, cells, strict=True):
            counts = Counter(
                found.outcomes[method]
                for found in by_kind.get(kind, {}).values()
                if method in found.outcomes
            )
            shown = {
                "on a par": counts["on a par"],
                "worse": counts["worse"],
                "diverging or did not train": counts["diverging"] + counts["fails"],
                "not comparable": counts["not comparable"],
            }
            for words, count in shown.items():
                chip = f'title="{words}">{count}</span>'
                assert (chip in cell) == bool(count), (kind, method, words, cell)
            if not any(shown.values()):
                assert ">not counted<" in cell or cell == "—", (kind, method, cell)
    # Every option level 2 compares has its row, a chip per method it was compared under.
    rows = {line.strip().split(" | ")[0] + " |" for line in page.splitlines()}
    for options in by_kind.values():
        for (slot, name), found in options.items():
            if found.outcomes:  # the row is the option, or `slot=option` for the recipe
                assert any(key == f"| {name} |" or key.endswith(f"={name} |") for key in rows), (
                    slot,
                    name,
                )
    # Not counted: named under "Not on this page", with the methods.
    rest = page.split("## Not on this page")[1].split("## The records")[0]
    lines = [line for line in rest.splitlines() if line.strip().startswith("| `")]
    for options in by_kind.values():
        for (slot, name), found in options.items():
            for method, why in found.excluded.items():
                group = "Did not run at level 1" if "did not run" in why else "schedule of its own"
                assert any(name in line and method in line and group in line for line in lines), (
                    slot,
                    name,
                    method,
                )
    assert "**Schedules** (" in rest and "| `schedule=cosine` |" in rest
    recipe = page.split("### Training recipe")[1].split("\n## ")[0]
    assert "| schedule=cosine |" not in recipe
    # Nothing is counted twice or lost: the cells shown plus those not counted are the summary.
    shown, left = learn.compared_options(summary)
    counted = {
        line["id"]
        for options in shown.values()
        for option in options
        for cells in option.lines.values()
        for line in cells
    }
    named = {line["id"] for found in left.values() for line, _ in found}
    assert not counted & named
    assert counted | named == {line["id"] for line in summary["cells"]}


def test_the_summary_states_its_rule_and_holds_one_line_per_cell():
    learn = checker()
    summary = json.loads(learn.SUMMARY.read_text(encoding="utf-8"))
    about = summary["about"]
    assert about["tolerance"] == learn.TOLERANCE
    assert about["rising_above"] == learn.RISING_ABOVE
    # Every method's tolerance and threshold follow from its noise table by the rule.
    for method, values in about["calibrated"].items():
        noise = {
            f"{seed} {name}": numbers
            for seed, terms in about["noise"].get(method, {}).items()
            for name, numbers in terms.items()
            if name != "loss"
        }
        assert values == learn.calibrate(noise), method
    assert about["settings"] == learn.SETTINGS
    assert about["outcomes"] == learn.OUTCOMES
    ids = [line["id"] for line in summary["cells"]]
    assert len(ids) == len(set(ids))
    for line in summary["cells"]:
        assert line["outcome"] in learn.OUTCOMES and line["kind"] in learn.KINDS, line["id"]
        assert "losses_per_iteration" not in line
    # The losses of every run are in the side file, in blocks.
    side = json.loads(learn.side_file(learn.SUMMARY).read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in side["cells"]}
    for line in summary["references"]:
        if "failure" not in line:
            assert line["id"] in entries
    # A run that ended early (a loss became non-finite and training stopped) has fewer
    # blocks; such a run is diverging.
    outcome = {line["id"]: line["outcome"] for line in summary["cells"]}
    for entry in entries.values():
        for values in entry["losses"].values():
            full = learn.ITERATIONS // side["about"]["block"]
            assert 0 < len(values) <= full, entry["id"]
            assert len(values) == full or outcome.get(entry["id"]) == "diverging", entry["id"]


def test_finegrasp_runs_under_its_recipe_and_apart(tmp_path, monkeypatch):
    """FineGrasp's fusion collapses at the constant rate without its warm-up: its reference
    and its cells run under the recipe's schedule, in a call of their own with --redo (an
    earlier probe of that schedule is reused, one of the constant rate is not)."""
    learn = checker()
    runner = tool("probe_grid")
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "main", lambda argv: calls.append(argv) or 0)
    args = learn.argparse.Namespace(jobs=1, timeout=60.0, scale_statistic_sha=None)
    ids = ["reference/GSNet", "reference/FineGrasp", "encoder/FineGrasp/A", "encoder/GSNet/B"]
    assert learn._probe("L2-encoder", ids, args) == 0

    def only(argv: list[str]) -> list[str]:
        rest = argv[argv.index("--only") + 1 :]
        return rest[: next(i for i, word in enumerate(rest) if word.startswith("--"))]

    plain, apart = calls
    assert only(plain) == ["reference/GSNet", "encoder/GSNet/B"] and "--redo" not in plain
    assert plain[plain.index("--schedule") + 1] == "constant"
    assert only(apart) == ["reference/FineGrasp", "encoder/FineGrasp/A"] and "--redo" in apart
    assert apart[apart.index("--schedule") + 1] == "recipe"
    assert learn.schedule_of("loss/FineGrasp/x@head.score_loss") == "recipe"
    # The page says so.
    page = (REPO / "docs" / "advanced_guides" / "learns_like_reference.md").read_text()
    assert "**FineGrasp** runs under `--schedule recipe`" in page
