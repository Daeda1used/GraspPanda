"""Result records: what they hold, how a partial result is marked, and the console summary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from grasppanda.evaluation import EvaluationResult
from grasppanda.evaluation.records import RECORD_SCHEMA, ResultRecord, per_scene_metrics, summarize
from grasppanda.structures import SampleKey

KEYS = [SampleKey("d", scene, "cam", frame) for scene in (1, 2) for frame in (0, 1)]


def result(scored: int = 4) -> EvaluationResult:
    """Four requested frames, the first ``scored`` of them scored with precision 0.5."""
    precision = {key: np.full((2, 3), 0.5) for key in KEYS[:scored]}
    return EvaluationResult(precision, {"AP": 50.0}, len(KEYS), KEYS[scored:], {"version": "t1"})


def record(scored: int = 4, variant: str = "raw") -> ResultRecord:
    return ResultRecord.from_result(
        result(scored),
        variant=variant,
        keys=KEYS,
        protocol="proto-v1",
        dataset="d",
        split="test_seen",
        postprocessing=[{"type": "Filter", "voxel": 0.01}],
        gt_assisted=["workspace"],
        evaluator={"type": "E", "config": {"type": "E"}},
        config_hash="abc",
        code={"commit": "deadbeef", "dirty": False, "version": "0.1"},
        weights={"path": "w.pth", "sha256": "f" * 64},
        seed=3,
    )


def test_a_complete_record_carries_metrics_and_their_conditions(tmp_path):
    complete = record()
    assert not complete.partial and complete.coverage == 1.0
    assert complete.headline() == {"AP": 50.0}
    written = json.loads(complete.write(tmp_path / "results/raw.json").read_text())
    assert written["schema"] == RECORD_SCHEMA and written["metrics"] == {"AP": 50.0}
    assert written["partial"] is False and written["partial_metrics"] is None
    for field, value in {
        "protocol": "proto-v1",
        "dataset": "d",
        "split": "test_seen",
        "cameras": ["cam"],
        "requested": 4,
        "scored": 4,
        "missing": [],
        "postprocessing": [{"type": "Filter", "voxel": 0.01}],
        "gt_assisted": ["workspace"],
        "evaluator": {"type": "E", "config": {"type": "E"}, "info": {"version": "t1"}},
        "config_hash": "abc",
        "code": {"commit": "deadbeef", "dirty": False, "version": "0.1"},
        "weights": {"path": "w.pth", "sha256": "f" * 64},
        "seed": 3,
    }.items():
        assert written[field] == value, field
    assert set(written["per_scene"]) == {"cam/scene_0001", "cam/scene_0002"}
    assert ResultRecord.read(tmp_path / "results/raw.json") == complete
    assert not list((tmp_path / "results").glob("*.tmp"))


def test_a_partial_record_is_written_but_holds_no_headline_metrics(tmp_path):
    partial = record(scored=3)
    assert partial.partial and partial.coverage == 0.75 and partial.headline() is None
    assert partial.missing == ["d/scene_0002/cam/0001"]
    written = json.loads(partial.write(tmp_path / "raw.json").read_text())
    assert written["partial"] is True and written["metrics"] is None
    assert written["partial_metrics"] == {"AP": 50.0} and written["coverage"] == 0.75
    again = ResultRecord.read(tmp_path / "raw.json")
    assert again == partial and again.headline() is None


def test_summary_prints_headlines_of_complete_variants_only():
    text = summarize({"raw": record(), "cd": record(scored=3, variant="cd")})
    lines = text.splitlines()
    assert "protocol proto-v1" in lines[0] and "split test_seen" in lines[0]
    assert "ground-truth-assisted inputs: workspace" in lines[0]
    raw = next(line for line in lines if line.startswith("raw "))
    cd = next(line for line in lines if line.startswith("cd "))
    assert "4/4" in raw and "50.00" in raw
    assert "3/4" in cd and "withheld (partial)" in cd and "50.00" not in cd
    assert "cd: PARTIAL, 3 of 4 frames scored (75.0 %), 1 missing" in text
    assert "cam/scene_0002  frames 1  0.5000" in text  # per-scene numbers instead
    assert text.count("50.00") == 1
    assert summarize({}) == "no variant was scored"


def test_per_scene_metrics_group_frames_by_scene_and_camera():
    precision = {
        SampleKey("d", 1, "a", 0): np.full((2, 3), 1.0),
        SampleKey("d", 1, "a", 1): np.zeros((2, 3)),
        SampleKey("d", 1, "b", 0): np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        SampleKey("d", 2, "a", 0): np.full((2, 3), 0.25),
    }
    scenes = per_scene_metrics(precision)
    assert list(scenes) == ["a/scene_0001", "a/scene_0002", "b/scene_0001"]
    assert scenes["a/scene_0001"] == {
        "frames": 2,
        "mean_precision": 0.5,
        "per_column": [0.5, 0.5, 0.5],
    }
    assert scenes["b/scene_0001"]["per_column"] == [1.0, 0.5, 0.0]
    assert per_scene_metrics({}) == {}


def test_a_record_of_nothing_is_partial():
    empty = EvaluationResult(precision={}, metrics={}, requested=0)
    nothing = ResultRecord.from_result(
        empty, variant="raw", keys=[], protocol="p", dataset="d", split="s"
    )
    assert nothing.partial and nothing.coverage == 0.0 and nothing.headline() is None
    assert "withheld (partial)" in summarize({"raw": nothing})


def test_an_unknown_record_layout_is_refused(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema": 0, "metrics": {"AP": 1.0}}))
    with pytest.raises(ValueError, match="unsupported result record schema 0"):
        ResultRecord.read(path)
