"""Sweep planning: the whole grid is validated before anything runs."""

from __future__ import annotations

import pytest

from grasppanda.config import ConfigError
from grasppanda.sweep import load_sweep

from .conftest import REPO

TOY = REPO / "projects/example_project/configs/toy_detector.yaml"


def sweep_file(write, axes: str, seeds: str = "[0, 1]"):
    return write("sweeps/ablation.yaml", f"base: {TOY}\nseeds: {seeds}\naxes:\n{axes}")


def test_plan_is_the_product_of_axes_and_seeds(write):
    sweep = load_sweep(
        sweep_file(
            write,
            "  seeds_per_frame:\n    s64: []\n    s128: ['model.num_seeds=128']\n"
            "  radius:\n    r5: []\n    r8: ['model.grouper.radius=0.08']\n",
        )
    )
    cells = sweep.plan()
    assert len(cells) == 2 * 2 * 2 and all(cell.valid for cell in cells)
    assert [cell.name for cell in cells[:3]] == ["s64_r5/seed0", "s64_r5/seed1", "s64_r8/seed0"]
    last = cells[-1]
    assert last.choices == {"seeds_per_frame": "s128", "radius": "r8"} and last.seed == 1
    assert last.config.select("model.num_seeds") == 128
    assert last.config.select("model.grouper.radius") == 0.08
    assert last.config.select("runtime.work_dir") == "work_dirs/ablation/s128_r8/seed1"
    assert len({cell.config.digest() for cell in cells}) == 8  # the seed is part of the experiment


def test_invalid_cells_are_reported_not_raised(write):
    sweep = load_sweep(
        sweep_file(
            write,
            "  grouper:\n    flat: []\n    depth: ['model.grouper.type=ToyDepthGrouper']\n",
            seeds="[0]",
        )
    )
    flat, depth = sweep.plan()
    assert flat.valid and not depth.valid
    assert "ToyHead expects ('seed', 'channel')" in str(depth.error)


def test_command_requirements_apply_to_every_cell(write):
    cells = load_sweep(sweep_file(write, "  a:\n    x: []\n", seeds="[0]")).plan(command="train")
    assert not cells[0].valid and "data.train" in str(cells[0].error)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("seeds: [0]\n", "is required"),
        ("base: x.yaml\nextra: 1\n", "unknown key"),
        (f"base: {TOY}\nseeds: []\n", "non-empty list of integers"),
        (f"base: {TOY}\naxes:\n  a:\n    x: 'model.num_seeds=1'\n", "must be a list"),
    ],
)
def test_malformed_sweep_files(write, text, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_sweep(write("sweeps/bad.yaml", text))


def test_aggregate_means_over_seeds_and_withholds_incomplete_metrics():
    from grasppanda.sweep import aggregate, render_table

    records = [
        {"choices": {"backbone": "a"}, "seed": 0, "test": {"raw/AP": 60.0, "cd/AP": 62.0}},
        {"choices": {"backbone": "a"}, "seed": 1, "test": {"raw/AP": 62.0, "cd/AP": 64.0}},
        {"choices": {"backbone": "b"}, "seed": 0, "test": {"raw/AP": 50.0}},
        {"choices": {"backbone": "b"}, "seed": 1, "test": {"raw/AP": 54.0, "cd/AP": 55.0}},
    ]
    first, second = aggregate(records)
    assert first["metrics"]["raw/AP"] == {"mean": 61.0, "std": pytest.approx(2**0.5), "n": 2}
    assert set(second["metrics"]) == {"raw/AP"}  # cd/AP is missing for one seed: not reported
    markdown = render_table([first, second])
    assert (
        "| a | 2 | 63.00 ± 1.41 | 61.00 ± 1.41 |" in markdown
        and "| b | 2 | — | 52.00 ± 2.83 |" in markdown
    )
    assert (
        render_table([first, second], fmt="csv").splitlines()[0]
        == "backbone,seeds,cd/AP,raw/AP,cd/AP_std,raw/AP_std"
    )


@pytest.mark.slow
def test_sweep_command_trains_tests_and_tabulates(write, tmp_path, monkeypatch, capsys):
    from grasppanda.cli.main import main

    train = REPO / "projects/example_project/configs/toy_train.yaml"
    path = write(
        "sweeps/tiny.yaml",
        f"base: {train}\nseeds: [0]\naxes:\n  seeds_per_frame:\n    s32: ['model.num_seeds=32']\n    s64: []\n",
    )
    monkeypatch.chdir(tmp_path)
    shrink = "train.epochs=1"
    text = (
        path.read_text()
        .replace("['model.num_seeds=32']", f"['model.num_seeds=32', '{shrink}']")
        .replace("s64: []", f"s64: ['{shrink}']")
    )
    path.write_text(text)
    assert main(["sweep", str(path), "--dry-run"]) == 0
    assert "2 runs" in capsys.readouterr().out
    assert main(["sweep", str(path)]) == 0
    summary = (tmp_path / "work_dirs/tiny/summary_test.md").read_text()
    assert "| s32 | 1 |" in summary and "| s64 | 1 |" in summary and "raw/AP" in summary
    assert main(["sweep", str(path)]) == 0  # finished cells are not run again
    assert (
        len(list((tmp_path / "work_dirs/tiny").glob("*/seed0/2*"))) == 4
    )  # 2 cells x (train + test)
