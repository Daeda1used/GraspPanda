"""``grasppanda analyze``: the record fields behind it, reports, comparisons and charts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grasppanda.cli.analyze import compare, load_records, report
from grasppanda.cli.main import main
from grasppanda.evaluation import EvaluationResult
from grasppanda.evaluation.breakdown import Tally, frame_tallies
from grasppanda.evaluation.graspnet1b.frame import precision_table
from grasppanda.evaluation.records import ResultRecord, scene_label
from grasppanda.structures import SampleKey

from .inference_fixtures import TOY_CONFIG, save_weights
from .test_engine_tiny import single_thread  # noqa: F401  (autouse fixture)
from .test_eval_breakdown import FRICTIONS, PROTOCOL, random_frame

# ------------------------------------------------------------------ synthetic records


def synthetic_record(
    seed: int, variant: str, *, scenes=(100, 101, 102), keep: float = 1.0
) -> ResultRecord:
    """A record as ``grasppanda test`` writes it, from random evaluator decisions."""
    generator = np.random.default_rng(seed)
    keys = [
        SampleKey("graspnet1b", scene, "realsense", frame) for scene in scenes for frame in (0, 128)
    ]
    precision, per_scene, per_object, counts = {}, {}, {}, {}
    for sample in keys:
        frame = random_frame(generator, 60)
        whole, objects = frame_tallies(
            **frame, object_ids=[5, 9, 14], top_k=50, frictions=FRICTIONS
        )
        precision[sample] = precision_table(frame["friction"][frame["ranking"]], PROTOCOL)
        per_scene.setdefault(scene_label(sample), Tally()).add(whole)
        for object_id in (5, 9, 14):
            per_object.setdefault(object_id, Tally()).frames += 1
        for object_id, part in objects.items():
            per_object[object_id].add(part)
        row = counts.setdefault(scene_label(sample), [0, 0])
        row[0] += 400
        row[1] += int(400 * keep)
    stack = np.stack(list(precision.values()))
    metrics = {
        "AP": float(stack.mean()),
        "AP0.8": float(stack[:, :, 3].mean()),
        "AP0.4": float(stack[:, :, 1].mean()),
    }
    breakdown = {
        "top_k": 50,
        "frictions": FRICTIONS,
        "per_scene": {
            name: tally.to_dict(6, with_unfilled=True) for name, tally in per_scene.items()
        },
        "per_object": {
            str(i): tally.to_dict(6, with_unfilled=False) for i, tally in per_object.items()
        },
    }
    info = {
        "constants": PROTOCOL.as_dict(),
        "grasps_read": sum(row[-1] for row in counts.values()),
        "empty_predictions": 0,
    }
    result = EvaluationResult(precision, metrics, len(keys), [], info, breakdown)
    chain = [] if keep == 1.0 else [{"type": "ModelFreeCollisionFilter", "voxel_size": 0.01}]
    steps = ["predicted", *(step["type"] for step in chain)]
    return ResultRecord.from_result(
        result,
        variant=variant,
        keys=keys,
        protocol=PROTOCOL.name,
        dataset="graspnet1b",
        split="test_seen",
        postprocessing=chain,
        gt_assisted=["workspace"],
        evaluator={"type": "GraspNet1BEvaluator"},
        config_hash="c0ffee",
        code={"commit": "abcdef0123456789", "dirty": False, "version": "0.1"},
        weights={"sha256": "f" * 64},
        seed=0,
        grasp_counts={
            "steps": steps,
            "per_scene": {n: row[: len(steps)] for n, row in counts.items()},
        },
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    """Two run directories with a raw and a filtered record each."""
    root = tmp_path_factory.mktemp("runs")
    for name, seed in (("first", 1), ("second", 2)):
        synthetic_record(seed, "raw").write(root / name / "results/raw.json")
        synthetic_record(seed + 10, "cd", keep=0.4).write(root / name / "results/cd.json")
    return {"root": root, "first": root / "first", "second": root / "second"}


def test_new_fields_round_trip_and_older_records_still_read(runs, tmp_path):
    file = runs["first"] / "results/cd.json"
    record = ResultRecord.read(file)
    assert record.grasp_counts["steps"] == ["predicted", "ModelFreeCollisionFilter"]
    assert set(record.breakdown) == {"top_k", "frictions", "per_scene", "per_object"}
    assert ResultRecord.read(record.write(tmp_path / "again.json")) == record

    old = json.loads(file.read_text())
    del old["grasp_counts"], old["breakdown"]  # what a run before these fields wrote
    (tmp_path / "old.json").write_text(json.dumps(old))
    older = ResultRecord.read(tmp_path / "old.json")
    assert older.grasp_counts == {} and older.breakdown == {}
    assert older.metrics == record.metrics and older.per_scene == record.per_scene


def test_report_says_what_was_measured_and_where_the_precision_comes_from(runs):
    (loaded,), note = load_records(str(runs["first"]), "cd")
    record = loaded.record
    text = report(loaded, scenes=1, objects=2)
    assert note == "" and text.startswith(f"{runs['first']} · cd")
    assert (
        "protocol        graspnet1b-sv-v1 · dataset graspnet1b · split test_seen · camera realsense"
        in text
    )
    assert "6 of 6 frames scored (100.0 %), 0 missing" in text
    assert (
        "a SUBSET of split 'test_seen': 6 of its 7,680 frames (2 of 256 frames per scene, 3 of 30 scenes)"
        in text
    )
    assert "Not a benchmark number." in text
    assert "ground-truth-assisted: workspace" in text and "ModelFreeCollisionFilter" in text
    assert "weights sha256 ffffffffffff · code abcdef0123 · seed 0" in text
    for name, value in record.metrics.items():
        assert f"{name} {value:.4f}" in text

    line = next(row for row in text.splitlines() if row.startswith("mean precision"))
    columns = [float(cell) for cell in line.split()[2:]]
    assert np.isclose(columns[-1], record.metrics["AP"], atol=5e-5)  # the mean of the columns is AP
    assert np.isclose(columns[3], record.metrics["AP0.8"], atol=5e-5)
    assert np.isclose(np.mean(columns[:6]), columns[-1], atol=1e-4)
    earned = next(row for row in text.splitlines() if row.startswith("earned"))
    assert earned.split()[1:] == line.split()[2:]  # the split's "earned" row is that precision
    total = next(row for row in text.splitlines() if row.startswith("sum"))
    assert total.split()[1:] == ["1.0000"] * 7
    for cause in (
        "lost: no force closure",
        "lost: collision",
        "lost: empty",
        "lost: unfilled ranks",
    ):
        assert cause in text

    assert "By scene (the 1 lowest and the 1 highest of 3; --scenes 0 lists all)" in text
    lowest = min(record.per_scene, key=lambda name: record.per_scene[name]["mean_precision"])
    scene_rows = text.split("By scene")[1].split("By object")[0].splitlines()
    assert scene_rows[3].startswith(lowest) and scene_rows[3].split()[-1] == "160.0"  # grasps/frame
    assert "By object (the 2 that lose most of 3;" in text
    assert "test_seen  3/30" in text  # the split table says how much of the split was seen
    grasps = text.split("Grasps")[1]
    assert "predicted                       2,400   400.0" in grasps
    assert (
        "after ModelFreeCollisionFilter  960     160.0      60.0 %                40.0 %" in grasps
    )
    assert "frames with an empty prediction: 0" in grasps


def test_records_written_before_the_new_fields_are_reported_without_them(runs, tmp_path):
    for variant in ("raw", "cd"):
        old = json.loads((runs["first"] / f"results/{variant}.json").read_text())
        del old["grasp_counts"], old["breakdown"]
        target = tmp_path / "old/results"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{variant}.json").write_text(json.dumps(old))
    (loaded,), _ = load_records(f"{tmp_path / 'old'}:cd")
    text = report(loaded)
    assert "This record has no breakdown by cause and object" in text and "By object" not in text
    assert "scored: 960 grasps, 160.0 per frame" in text
    assert (
        "the 'raw' record of the same run read 2,400: the chain (ModelFreeCollisionFilter) removed 60.0 % in total"
        in text
    )
    assert "per-step counts are not in this record" in text
    assert "By scene (all 3 scenes, lowest first)" in text  # few scenes: all are listed


def test_a_partial_record_reports_no_headline(runs, tmp_path):
    record = ResultRecord.read(runs["first"] / "results/raw.json")
    record.requested, record.missing = 8, ["graspnet1b/scene_0103/realsense/0000", "x"]
    record.write(tmp_path / "partial.json")
    (loaded,), _ = load_records(str(tmp_path / "partial.json"))
    text = report(loaded)
    assert "6 of 8 frames scored (75.0 %), 2 missing" in text and "PARTIAL" in text
    assert "metrics " not in text and "graspnet1b/scene_0103/realsense/0000" in text


def test_compare_pairs_scenes_and_sorts_by_size(runs):
    (first,), _ = load_records(str(runs["first"]), "raw")
    (second,), _ = load_records(f"{runs['second']}:raw")
    text, numbers = compare(first, second, scenes=2)
    a, b = first.record.per_scene, second.record.per_scene
    expected = {name: b[name]["mean_precision"] - a[name]["mean_precision"] for name in a}
    order = sorted(expected, key=lambda name: -abs(expected[name]))
    assert numbers["names"] == order
    assert np.allclose(numbers["differences"], [expected[name] for name in order])
    assert np.isclose(numbers["mean"], np.mean(list(expected.values())))
    assert numbers["wins"] + numbers["losses"] + numbers["ties"] == numbers["scenes"] == 3
    low, high = numbers["interval"]
    assert low <= numbers["mean"] <= high
    assert (
        "3 paired scenes" in text
        and "bootstrap over the 3 scenes, 10 000 resamples, seed 0" in text
    )
    assert "only 3 scenes: read the interval as a rough guide" in text
    assert "Per scene, largest difference first (the 2 largest of 3; --scenes 0 lists all)" in text
    assert f"{order[0]}  " in text and order[2] not in text.split("Per scene")[1]
    assert "differs:" not in text  # same split, cameras, inputs, chain, weights and code

    (filtered,), _ = load_records(str(runs["first"]), "cd")
    text, numbers = compare(first, filtered)
    assert "differs: post-processing: A none · B ['ModelFreeCollisionFilter']" in text
    same, zero = compare(first, first)
    assert zero["mean"] == 0.0 and zero["ties"] == 3 and zero["p_sign"] == 1.0
    assert "B higher in 0 scenes, A higher in 0, equal in 3" in same


def test_sign_test_is_exact():
    from grasppanda.cli.analyze import _sign_test

    assert _sign_test(0, 0) == 1.0 and _sign_test(5, 5) == 1.0
    assert np.isclose(_sign_test(10, 0), 2 / 2**10)
    assert np.isclose(_sign_test(8, 2), 2 * (1 + 10 + 45) / 2**10)


def test_command_line(runs, tmp_path, capsys):
    assert main(["analyze", str(runs["first"])]) == 0
    printed = capsys.readouterr().out
    assert printed.count("Where the precision comes from") == 2  # both variants of the run
    assert "All records" in printed and "first · raw" in printed and "first · cd" in printed

    plot = tmp_path / "plots/one.png"
    assert (
        main(
            ["analyze", str(runs["first"]), "--variant", "cd", "--plot", str(plot), "--scenes", "0"]
        )
        == 0
    )
    assert f"wrote {plot}" in capsys.readouterr().out
    with Image.open(plot) as image:
        assert image.size[0] == 1100 and image.size[1] > 400

    difference = tmp_path / "diff.png"
    pair = ["--compare", f"{runs['first']}:raw", f"{runs['second']}:raw"]
    assert main(["analyze", *pair, "--plot", str(difference)]) == 0
    assert "Paired summary over scenes" in capsys.readouterr().out
    with Image.open(difference) as image:
        assert image.size[0] == 1100
    assert (
        main(["analyze", "--compare", str(runs["first"]), str(runs["second"]), "--variant", "cd"])
        == 0
    )
    capsys.readouterr()


def test_command_line_problems_are_explained(runs, tmp_path, capsys):
    assert main(["analyze"]) == 1
    assert "name at least one run directory" in capsys.readouterr().err
    assert main(["analyze", str(tmp_path / "nowhere")]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert main(["analyze", str(runs["root"])]) == 1  # a work directory with two runs
    error = capsys.readouterr().err
    assert "holds 2 runs; name one of them" in error and str(runs["first"]) in error
    assert main(["analyze", str(runs["first"]), "--variant", "nms"]) == 1
    assert "no record of variant 'nms'; it holds cd, raw" in capsys.readouterr().err
    assert main(["analyze", str(runs["first"]), "--plot", str(tmp_path / "x.png")]) == 1
    assert "--plot draws one record, but 2 were read" in capsys.readouterr().err
    assert main(["analyze", "--compare", str(runs["first"]), str(runs["second"])]) == 1
    assert "pick one with" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert main(["analyze", str(tmp_path / "empty")]) == 1
    assert "is not a run directory" in capsys.readouterr().err
    other = ResultRecord.read(runs["first"] / "results/raw.json")
    other.protocol = "graspclutter6d-sv-v1"
    other.write(tmp_path / "other.json")
    assert main(["analyze", "--compare", f"{runs['first']}:raw", str(tmp_path / "other.json")]) == 1
    assert "not comparable" in capsys.readouterr().err


def test_a_run_of_grasppanda_test_is_analyzed_as_it_is(tmp_path, capsys):
    """The engine records how many grasps each step of a chain left; analyze reads it back."""
    weights = save_weights(TOY_CONFIG, tmp_path / "toy.pth")
    arguments = [
        "test",
        TOY_CONFIG,
        "--weights",
        str(weights),
        "--work-dir",
        str(tmp_path / "work"),
    ]
    assert main([*arguments, "--opts", "runtime.run_id=scored"]) == 0
    capsys.readouterr()
    record = ResultRecord.read(tmp_path / "work/scored/results/centre.json")
    counts = record.grasp_counts
    assert counts["steps"] == ["predicted", "ToyCentreFilter"]
    assert set(counts["per_scene"]) == set(record.per_scene) == {"toy/scene_0200", "toy/scene_0201"}
    totals = np.sum(list(counts["per_scene"].values()), axis=0)
    assert totals[0] == 16 * 64 and totals[1] < totals[0]  # 16 frames of 64 grasps, then the filter
    dumped = sum(
        len(np.load(file)) for file in (tmp_path / "work/scored/predictions/centre").rglob("*.npy")
    )
    assert totals[1] == dumped
    raw = ResultRecord.read(tmp_path / "work/scored/results/raw.json")
    assert raw.grasp_counts["steps"] == ["predicted"] and raw.breakdown == {}

    assert (
        main(["analyze", str(tmp_path / "work"), "--variant", "centre"]) == 0
    )  # the one run below
    printed = capsys.readouterr().out
    assert "protocol        toy-synthetic-v0 · dataset toy · split test" in printed
    assert "after ToyCentreFilter" in printed and "16 of 16 frames scored" in printed
    assert "This record has no breakdown by cause and object" in printed
    assert "whether that is the whole split 'test' is not recorded" in printed


def test_a_training_run_is_read_at_its_last_evaluation(runs, tmp_path, capsys):
    for epoch, seed in ((1, 5), (2, 6)):
        synthetic_record(seed, "raw").write(
            tmp_path / f"trained/results/epoch_{epoch:04d}/raw.json"
        )
    (loaded,), note = load_records(str(tmp_path / "trained"))
    assert "last evaluation (epoch_0002)" in note
    assert loaded.record.metrics == synthetic_record(6, "raw").metrics
    assert loaded.label == f"{tmp_path / 'trained'} · raw"
    assert main(["analyze", str(tmp_path / "trained")]) == 0
    assert "a training run: the records of its last evaluation" in capsys.readouterr().out
    (tmp_path / "unscored/results").mkdir(parents=True)
    assert main(["analyze", str(tmp_path / "unscored")]) == 1
    assert "holds no result record; was the run scored" in capsys.readouterr().err
