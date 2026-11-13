"""Sweep planning: the whole grid is validated before anything runs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


@pytest.mark.slow
def test_sweep_runs_cells_in_parallel_one_per_gpu(write, tmp_path, monkeypatch, capsys):
    """``--gpus``: a process per cell that sees one GPU, as many at a time as GPUs named;
    the records, the summaries and the resume are those of a sequential sweep."""
    import json

    from grasppanda.cli.main import main
    from grasppanda.sweep import CELL_LOG, CELL_RECORD

    train = REPO / "projects/example_project/configs/toy_train.yaml"
    shrink = "train.epochs=1"
    path = write(
        "sweeps/wide.yaml",
        f"base: {train}\nseeds: [0]\naxes:\n  seeds_per_frame:\n"
        f"    s32: ['model.num_seeds=32', '{shrink}']\n    s64: ['{shrink}']\n"
        f"    s96: ['model.num_seeds=96', '{shrink}']\n",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["sweep", str(path), "--gpus", ","]) == 2  # no id named
    assert "--gpus needs at least one GPU id" in capsys.readouterr().err
    # an id this machine does not have would start a cell that sees no GPU: refused
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with monkeypatch.context() as machine:
        machine.setattr(torch.cuda, "device_count", lambda: 0)
        assert main(["sweep", str(path), "--gpus", "0"]) == 2
        assert "PyTorch sees none here" in capsys.readouterr().err
        machine.setattr(torch.cuda, "device_count", lambda: 9)
        assert main(["sweep", str(path), "--gpus", "7,9,a"]) == 2
        refused = capsys.readouterr().err
        assert "--gpus names 9, a: not a GPU of this machine's 9 (ids 0 to 8" in refused
    assert not (tmp_path / "work_dirs").exists()  # nothing ran
    assert main(["sweep", str(path), "--cell", "s128/seed0"]) == 2
    assert "no cell 's128/seed0'" in capsys.readouterr().err

    # one cell alone, in this process: what the parallel runner starts for each cell. It
    # trains on the devices this machine has, so the machine is itself again here
    assert main(["sweep", str(path), "--cell", "s64/seed0"]) == 0
    capsys.readouterr()
    cells = tmp_path / "work_dirs/wide"
    assert (cells / "s64/seed0" / CELL_RECORD).is_file()
    assert main(["sweep", str(path), "--cell", "s64/seed0", "--gpus", "7"]) == 2
    assert "choose its GPU with CUDA_VISIBLE_DEVICES" in capsys.readouterr().err

    # the parallel runner: this process only starts the cells, each in a process of its own
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 9)
    # another `grasppanda` in the working directory is not what a cell runs
    (tmp_path / "grasppanda").mkdir()
    (tmp_path / "grasppanda/__init__.py").write_text("raise SystemExit('another GraspPanda')\n")
    assert main(["sweep", str(path), "--gpus", "7,8"]) == 0
    out = capsys.readouterr().out
    assert "3 runs on GPUs 7, 8, 2 at a time" in out
    assert "s64/seed0: already recorded" in out
    started = {
        line.split(": started on GPU ")[0].split("] ")[1]: line.rsplit(" ", 1)[1]
        for line in out.splitlines()
        if ": started on GPU " in line
    }
    assert set(started) == {"s32/seed0", "s96/seed0"} and set(started.values()) <= {"7", "8"}
    for name in started:
        log = (cells / name / CELL_LOG).read_text()
        assert "✓ trained epochs 1-1" in log  # the cell's own console output
        record = json.loads((cells / name / CELL_RECORD).read_text())
        assert record["cell"] == name and "raw/AP" in record["test"]
    summary = (cells / "summary_test.md").read_text()
    assert all(f"| {name} | 1 |" in summary for name in ("s32", "s64", "s96"))

    assert main(["sweep", str(path), "--gpus", "7,8"]) == 0  # everything is recorded
    assert capsys.readouterr().out.count("already recorded") == 3

    # --opts reach every cell, also one that runs in a process of its own: another
    # experiment (another hash), so the cells run again
    assert main(["sweep", str(path), "--gpus", "7", "--opts", "model.grouper.radius=0.07"]) == 0
    assert capsys.readouterr().out.count("done on GPU 7") == 3
    record = json.loads((cells / "s32/seed0" / CELL_RECORD).read_text())
    from grasppanda.sweep import load_sweep as load

    planned = load(path, ["model.grouper.radius=0.07"]).plan(command="train")[0]
    assert planned.config.select("model.grouper.radius") == 0.07
    assert record["config_hash"] == planned.config.digest()


def test_a_cell_gets_one_gpu_and_the_modules_of_the_sweep_that_started_it(monkeypatch):
    """The cell's process searches modules where this process does, whatever its working
    directory or interpreter would find first, and its GPU is the one meant: nvidia-smi's
    number, or a position in this process's own mask."""
    import json
    import sys

    from grasppanda import sweep

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    environment = sweep._cell_environment("3")
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"  # as nvidia-smi numbers them
    assert json.loads(environment[sweep.CELL_PATH]) == sys.path
    assert sweep.CELL_PATH in sweep._CELL and "sys.path[:]" in sweep._CELL

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")  # a mask of its own: its numbering
    environment = sweep._cell_environment("1")
    assert environment["CUDA_VISIBLE_DEVICES"] == "5" and "CUDA_DEVICE_ORDER" not in environment
    with pytest.raises(ValueError, match="GPU 2 is not one of the 2 that CUDA_VISIBLE_DEVICES"):
        sweep._cell_environment("2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # every GPU hidden from this process
    with pytest.raises(ValueError, match="GPU 0 is not one of the 0"):
        sweep._cell_environment("0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")  # an order of its own
    assert sweep._cell_environment("3")["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_no_cell_starts_once_the_sweep_is_stopped(monkeypatch):
    """Stopping terminates what runs and refuses what would start, under one lock."""
    from grasppanda import sweep

    class Child:
        def __init__(self, command, **options):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(sweep.subprocess, "Popen", Child)
    cells = sweep._Cells()
    first, second = cells.start(["a"]), cells.start(["b"])
    cells.finished(second)
    cells.stop()
    assert first.terminated and not second.terminated
    assert cells.start(["c"]) is None


def test_a_failed_cell_does_not_stop_the_others(write, tmp_path, monkeypatch):
    """A cell whose process fails is reported with its log and carries no metric."""
    from grasppanda import sweep as module
    from grasppanda.sweep import aggregate, run_cells

    train = REPO / "projects/example_project/configs/toy_train.yaml"
    path = write(
        "sweeps/two.yaml", f"base: {train}\nseeds: [0]\naxes:\n  a:\n    x: []\n    y: []\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # GPU 0 is the machine's
    sweep = module.load_sweep(path)
    cells = sweep.plan(command="train")
    done = {"cell": "y/seed0", "choices": {"a": "y"}, "seed": 0, "test": {"raw/AP": 0.5}}

    def fake(sweep, cell, gpu, cells=None):  # no process: x fails, y succeeds
        if cell.name == "x/seed0":
            return {"cell": cell.name, "choices": cell.choices, "seed": 0, "failed": "x.log"}
        return done

    monkeypatch.setattr(module, "_run_in_process", fake)
    lines: list[str] = []
    records = run_cells(sweep, cells, gpus=["0"], report=lines.append)
    assert [record["cell"] for record in records] == ["x/seed0", "y/seed0"]  # plan order
    assert any("x/seed0: FAILED, see x.log on GPU 0" in line for line in lines)
    assert any("y/seed0: done on GPU 0" in line for line in lines)
    rows = {row["choices"]["a"]: row["metrics"] for row in aggregate(records)}
    assert rows["x"] == {} and rows["y"]["raw/AP"]["mean"] == 0.5

    def broken(sweep, cell, gpu, cells=None):  # the runner itself fails for x
        if cell.name == "x/seed0":
            raise OSError("no space left on device")
        return done

    monkeypatch.setattr(module, "_run_in_process", broken)
    lines.clear()
    records = run_cells(sweep, cells, gpus=["0"], report=lines.append)
    assert records[0]["failed"] == "OSError: no space left on device" and records[1] == done
    assert any("x/seed0: FAILED, OSError: no space left on device" in line for line in lines)

    # one after the other in this process: the same
    def train(sweep, cell):
        if cell.name == "x/seed0":
            raise RuntimeError("CUDA out of memory")
        return done

    monkeypatch.setattr(module, "run_cell", train)
    lines.clear()
    records = run_cells(sweep, cells, report=lines.append)
    assert records[0]["failed"] == "RuntimeError: CUDA out of memory" and records[1] == done
    assert any("x/seed0: FAILED, RuntimeError: CUDA out of memory" in line for line in lines)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")  # GPU 1 is not this process's
    with pytest.raises(ValueError, match="GPU 1 is not one of the 1"):
        run_cells(sweep, cells, gpus=["0", "1"], report=lines.append)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")

    sweep.path = None  # built in code, not loaded: there is no file to start a cell from
    with pytest.raises(ValueError, match="from the sweep's file"):
        run_cells(sweep, cells, gpus=["0"], report=lines.append)


def test_a_terminated_sweep_takes_its_cells_with_it(write, tmp_path, monkeypatch):
    """SIGTERM to the sweep stops the cells it started, and starts no further one."""
    import os
    import signal
    import threading

    from grasppanda import sweep as module

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals are delivered to the main thread")
    train = REPO / "projects/example_project/configs/toy_train.yaml"
    path = write(
        "sweeps/three.yaml",
        f"base: {train}\nseeds: [0]\naxes:\n  a:\n    x: []\n    y: []\n    z: []\n",
    )
    monkeypatch.chdir(tmp_path)
    sweep = module.load_sweep(path)
    cells = sweep.plan(command="train")
    started: list[Child] = []
    running = threading.Event()

    class Child:  # a cell that trains until it is told to stop
        def __init__(self, command, **options):
            self.stopped = threading.Event()
            self.mask = options["env"]["CUDA_VISIBLE_DEVICES"]
            started.append(self)
            running.set()

        def wait(self):
            assert self.stopped.wait(30)
            return -signal.SIGTERM

        def terminate(self):
            self.stopped.set()

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")  # this process's GPU 1 is the machine's 5
    before = signal.getsignal(signal.SIGTERM)

    def terminate():
        assert running.wait(30)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=terminate, daemon=True).start()
    with pytest.raises(SystemExit) as stopped:
        module.run_cells(sweep, cells, gpus=["1"], report=lambda line: None)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert len(started) == 1 and started[0].stopped.is_set()  # y and z were never started
    assert started[0].mask == "5"
    assert signal.getsignal(signal.SIGTERM) is before


def test_records_tabulate_alike_whatever_their_key_order_and_in_percent(
    write, tmp_path, monkeypatch
):
    """A record read back from disk has sorted keys; it is the same variant, in plan order.
    Metrics that the evaluator states as fractions are tabulated in percent."""
    import json

    from grasppanda import sweep as module
    from grasppanda.sweep import CELL_RECORD, aggregate, cell_record, render_table

    train = REPO / "projects/example_project/configs/toy_train.yaml"
    path = write(
        "sweeps/order.yaml",
        f"base: {train}\nseeds: [0, 1]\naxes:\n  width:\n    w: []\n  attention:\n    a: []\n",
    )
    monkeypatch.chdir(tmp_path)
    first, second = module.load_sweep(path).plan(command="train")
    assert list(first.choices) == ["width", "attention"]
    on_disk = {
        "cell": first.name,
        "choices": {"attention": "a", "width": "w"},  # as json.dumps(sort_keys=True) wrote it
        "seed": 0,
        "config_hash": first.config.digest(),
        "test": {"raw/AP": 0.6875},
        "percent": True,
    }
    work_dir = Path(first.config.select("runtime.work_dir"))
    work_dir.mkdir(parents=True)
    (work_dir / CELL_RECORD).write_text(json.dumps(on_disk, sort_keys=True))
    read = cell_record(first)
    assert list(read["choices"]) == ["width", "attention"]
    fresh = {**on_disk, "cell": second.name, "seed": 1, "test": {"raw/AP": 0.6925}}
    for records in ([read, fresh], [fresh, read], [on_disk, fresh]):
        rows = aggregate(records)
        assert len(rows) == 1 and rows[0]["seeds"] == 2 and rows[0]["percent"]
    rows = aggregate([read, fresh])
    assert "| w | a | 2 | 69.00 ± 0.35 |" in render_table(rows)
    assert render_table(rows, fmt="csv").splitlines()[1] == "w,a,2,69.00,0.3536"
    # a variant in another unit beside it: nothing is rescaled
    other = {"cell": "b", "choices": {"width": "v", "attention": "a"}, "test": {"raw/AP": 61.0}}
    assert "| w | a | 2 | 0.69 ± 0.00 |" in render_table(aggregate([read, fresh, other]))


def test_several_base_configs_are_the_axis_config(write):
    """Configs that compose wrote are compared with the same seeds: `base` names them."""
    other = write("sweeps/wider.yaml", f"_base_: ['{TOY}']\nmodel: {{num_seeds: 128}}\n")
    path = write(
        "sweeps/compare.yaml",
        f"base:\n  toy: {TOY}\n  wider: {other.name}\nseeds: [0, 1]\n"
        "axes:\n  radius:\n    r5: []\n    r8: ['model.grouper.radius=0.08']\n",
    )
    sweep = load_sweep(path)
    assert list(sweep.bases) == ["toy", "wider"] and sweep.bases["wider"] == other
    cells = sweep.plan()
    assert len(cells) == 2 * 2 * 2 and all(cell.valid for cell in cells)
    assert [cell.name for cell in cells[::2]] == [
        "toy_r5/seed0",
        "toy_r8/seed0",
        "wider_r5/seed0",
        "wider_r8/seed0",
    ]
    last = cells[-1]
    assert last.choices == {"config": "wider", "radius": "r8"} and last.base == other
    assert last.config.select("model.num_seeds") == 128
    assert last.config.select("model.grouper.radius") == 0.08
    assert last.config.select("runtime.work_dir") == "work_dirs/compare/wider_r8/seed1"

    from grasppanda.sweep import aggregate

    records = [
        {"choices": cell.choices, "seed": cell.seed, "test": {"raw/AP": 0.5 + 0.1 * index}}
        for index, cell in enumerate(cells)
    ]
    rows = aggregate(records)
    assert [row["choices"]["config"] for row in rows] == ["toy", "toy", "wider", "wider"]
    assert all(row["seeds"] == 2 for row in rows)

    clash = write(
        "sweeps/clash.yaml",
        f"base:\n  toy: {TOY}\nseeds: [0]\naxes:\n  config:\n    a: []\n",
    )
    with pytest.raises(ConfigError, match="the several `base` configs form"):
        load_sweep(clash)


def test_overrides_of_the_sweep_reach_every_cell_and_an_alternative_wins(write):
    path = write(
        "sweeps/shared.yaml",
        f"base: {TOY}\nseeds: [0]\noverrides: ['model.num_seeds=96', 'runtime.gpu_memory_fraction=0.4']\n"
        "axes:\n  seeds_per_frame:\n    same: []\n    more: ['model.num_seeds=128']\n",
    )
    same, more = load_sweep(path).plan()
    assert (
        same.config.select("model.num_seeds") == 96 and more.config.select("model.num_seeds") == 128
    )
    assert same.config.select("runtime.gpu_memory_fraction") == 0.4
    plain = load_sweep(sweep_file(write, "  a:\n    x: ['model.num_seeds=96']\n", seeds="[0]"))
    # a memory limit says how a run lives on a machine: it is not part of the experiment
    assert same.config.digest() == plain.plan()[0].config.digest()
    bad = write("sweeps/bad.yaml", f"base: {TOY}\noverrides: model.num_seeds=96\n")
    with pytest.raises(ConfigError, match="must be a list"):
        load_sweep(bad)


def test_an_alternative_splices_in_other_alternatives_by_anchor(write):
    """``both: [*more, *wide]``: a list inside the list of an alternative is spliced in."""
    sweep = load_sweep(
        sweep_file(
            write,
            "  change:\n    more: &more ['model.num_seeds=128']\n"
            "    wide: &wide ['model.grouper.radius=0.08']\n"
            "    both: [*more, *wide, 'model.grouper.nsample=8']\n",
            seeds="[0]",
        )
    )
    assert sweep.axes["change"]["both"] == [
        "model.num_seeds=128",
        "model.grouper.radius=0.08",
        "model.grouper.nsample=8",
    ]
    more, wide, both = sweep.plan()
    assert more.config.select("model.grouper.radius") != 0.08
    assert wide.config.select("model.num_seeds") != 128
    assert both.config.select("model.num_seeds") == 128
    assert both.config.select("model.grouper.radius") == 0.08
    deeper = sweep_file(write, "  a:\n    x: [[['model.num_seeds=1']]]\n", seeds="[0]")
    with pytest.raises(ConfigError, match="must be a list"):  # one level, not a tree
        load_sweep(deeper)


#: The sweeps of ``projects/``: file → the runs it plans.
PROJECT_SWEEPS = {
    "encoder_comparison/gsnet_encoders.yaml": ["sunet14", "ptv3", "spunet", "litept"],
    "encoder_comparison/gsnet_recipe.yaml": [
        "sunet14_adamw-onecycle",
        "ptv3_adamw-onecycle",
        "litept_adamw-onecycle",
    ],
    "finegrasp_ablation/changes.yaml": [
        "graspness",
        "normals",
        "multi-scale",
        "graspness+multi-scale",
        "normals+multi-scale",
        "all",
    ],
    "hggd_labels/labels.yaml": ["1cm-30deg", "unsuppressed"],
}


@pytest.mark.parametrize("name", PROJECT_SWEEPS)
def test_the_sweeps_of_the_projects_plan_valid_runs(name):
    cells = load_sweep(REPO / "projects" / name).plan(command="train")
    assert [cell.name for cell in cells] == [f"{run}/seed0" for run in PROJECT_SWEEPS[name]]
    assert all(cell.valid for cell in cells), [str(cell.error) for cell in cells if not cell.valid]
    assert len({cell.config.digest() for cell in cells}) == len(cells)


def test_the_recipe_of_the_encoder_comparison_is_the_recipe_preset():
    """Its overrides are what ``grasppanda compose --set recipe=adamw-onecycle`` writes, for
    the shipped encoder and for each swap: the sweep and the preset cannot drift apart."""
    from grasppanda.combine.compose import compose

    gsnet = "gsnet/gsnet_sunet14_g1b-realsense_10e"
    cells = {
        cell.choices["config"]: cell.config.digest()
        for cell in load_sweep(REPO / "projects/encoder_comparison/gsnet_recipe.yaml").plan()
    }
    assert cells == {
        "sunet14": compose(gsnet, ["recipe=adamw-onecycle"]).config_hash,
        "ptv3": compose(gsnet, ["backbone=ptv3", "recipe=adamw-onecycle"]).config_hash,
        "litept": compose(gsnet, ["backbone=litept", "recipe=adamw-onecycle"]).config_hash,
    }


def test_the_ablation_adds_up_to_finegrasp_under_the_recipe_of_economicgrasp():
    """Its cell ``all`` is FineGrasp's model and data, trained as EconomicGrasp is: every
    other cell differs from EconomicGrasp in the entries of its one change."""
    from grasppanda.config import Config, resolve_config_path

    def shipped(name: str) -> Config:
        return Config.load(resolve_config_path(f"{name}_sunet14_g1b-realsense_10e"))

    economic, fine = shipped("economicgrasp/economicgrasp"), shipped("finegrasp/finegrasp")
    cells = {
        cell.choices["change"]: cell.config
        for cell in load_sweep(REPO / "projects/finegrasp_ablation/changes.yaml").plan()
    }
    together = cells["all"]
    assert together.select("model") == fine.select("model")
    assert together.select("data.train.dataset") == fine.select("data.train.dataset")
    assert together.select("data.test.dataset") == fine.select("data.test.dataset")
    for key in ("optimizer", "scheduler", "train", "data.train.batch_size", "evaluation"):
        assert together.select(key) == economic.select(key), key
    # one change each: the label asset, the input features, the crop
    assert cells["graspness"].select("model") == economic.select("model")
    assert cells["graspness"].select("data.train.dataset.graspness.normalization") == "instance"
    assert cells["normals"].select("model.grouper") == economic.select("model.grouper")
    assert cells["normals"].select("model.backbone.in_channels") == 6
    assert cells["multi-scale"].select("model.grouper") == fine.select("model.grouper")
    assert cells["multi-scale"].select("data") == economic.select("data")
    # the crop with one of the two other changes: FineGrasp without the third
    for other, without in (("graspness", "normals"), ("normals", "graspness")):
        pair = cells[f"{other}+multi-scale"]
        assert pair.select("model.grouper") == fine.select("model.grouper")
        assert pair.select("data") == cells[other].select("data")
        assert pair.select("model.backbone") == cells[other].select("model.backbone")
        assert pair.select("data") != cells[without].select("data"), "the third change is off"
