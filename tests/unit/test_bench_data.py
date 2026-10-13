"""``tools/bench_data.py`` on the synthetic tree: the loader measure, the switch between
the two label storages, and the bookkeeping around the numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tools import bench_data

from .test_data_graspnet1b import PREPROCESS, build, tree  # noqa: F401

TRANSFORMS = [
    {"type": "RandomSample", "num_points": 64},
    {"type": "FilterObjects", "min_visible_points": 2},
    {"type": "SampleGraspLabels", "point_fraction": 1.0, "min_points": 4, "collision": "ignore"},
]


@pytest.fixture
def config(tree, tmp_path):  # noqa: F811
    build(tree)  # builds the label store of the synthetic tree
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "all",
        "root": str(tree["root"]),
        "preprocess": PREPROCESS,
        "transforms": TRANSFORMS,
        "seed": 0,
    }
    document = {
        "custom_imports": ["tests.unit.test_data_graspnet1b"],
        "data": {"train": {"dataset": node, "batch_size": 2, "num_workers": 0}},
    }
    file = tmp_path / "tiny_bench.yaml"
    file.write_text(yaml.safe_dump(document))
    return str(file)


def test_payload_counts_arrays_tensors_and_strings_however_nested():
    class Reader:
        def __init__(self) -> None:
            self.rows, self.file = np.zeros(10, dtype=np.int64), "abcd"

    value = {"a": torch.zeros(3, 4), "b": [np.zeros(5, dtype=np.float32), None, 7, Reader()]}
    assert bench_data.payload_bytes(value) == 48 + 20 + 80 + 4


def test_summary_and_table_report_best_and_median():
    trials = [{"loader frames/s": 4.0, "iteration s": 1.5}, {"loader frames/s": 6.0}]
    summary = bench_data.summarise(trials)
    assert summary["loader frames/s"] == {"min": 4.0, "median": 5.0, "max": 6.0}
    assert summary["iteration s"]["median"] == 1.5
    fast = bench_data.summarise([{"loader frames/s": 20.0, "iteration s": 0.5}])
    text = bench_data.render({"dense": summary, "deferred": fast}, markdown=True)
    lines = text.splitlines()
    assert lines[0] == "| measure (best / median) | dense | deferred | ratio of medians |"
    assert "| loader frames/s | 6.00 / 5.00 | 20.0 / 20.0 | 0.25 |" in lines  # a rate: best is max
    assert "| iteration s | 1.50 / 1.50 | 0.50 / 0.50 | 3.00 |" in lines  # a time: best is min


def test_the_loader_is_measured_for_both_storages_on_the_same_frames(config, tmp_path, capsys):
    out = tmp_path / "trials.json"
    arguments = "--measure loader --grids dense deferred --scenes 2 --iterations 2 --warmup 1"
    arguments += " --repeats 2 --device cpu --json"
    assert bench_data.main([config, *arguments.split(), str(out)]) == 0
    printed = capsys.readouterr().out
    assert "batch 2, 0 workers, 2 batches after 1 of warm-up, 2 repeats, first 2 scenes" in printed
    assert "loader frames/s" in printed and "ratio of medians" in printed
    trials = json.loads(out.read_text())["trials"]
    assert set(trials) == {"dense", "deferred"} and all(len(runs) == 2 for runs in trials.values())
    dense, deferred = trials["dense"][0], trials["deferred"][0]
    assert dense["loader frames/s"] > 0 and deferred["loader frames/s"] > 0
    # the grids of two frames against two small readers
    assert dense["batch payload MiB"] > deferred["batch payload MiB"] > 0


def test_switching_storage_needs_a_dense_label_transform(config, tmp_path):
    document = yaml.safe_load(Path(config).read_text())
    document["data"]["train"]["dataset"]["transforms"] = TRANSFORMS[:1]
    document["data"]["train"]["dataset"]["load_labels"] = False
    other = tmp_path / "no_labels.yaml"
    other.write_text(yaml.safe_dump(document))
    with pytest.raises(SystemExit, match="no SampleGraspLabels transform"):
        bench_data.main([str(other), "--measure", "loader", "--grids", "deferred"])
    with pytest.raises(SystemExit):
        bench_data.main([config, "--iterations", "0"])
