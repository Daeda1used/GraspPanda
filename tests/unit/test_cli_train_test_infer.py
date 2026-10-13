"""``grasppanda train``, ``test`` and ``infer`` end to end on the example project, on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grasppanda.cli.main import PLANNED, main
from grasppanda.config import parse_yaml
from grasppanda.engine.checkpoint import file_sha256, read_checkpoint

from .conftest import REPO
from .test_engine_tiny import single_thread  # noqa: F401  (autouse fixture)

TOY = str(REPO / "projects/example_project/configs/toy_train.yaml")
MODEL_ONLY = str(REPO / "projects/example_project/configs/toy_detector.yaml")
SMALL = ["train.epochs=2", "data.train.dataset.num_scenes=1", "runtime.log.interval=1"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """One short training run shared by the tests of this module; returns its run directory."""
    work = tmp_path_factory.mktemp("cli") / "work"
    assert (
        main(["train", TOY, "--work-dir", str(work), "--opts", *SMALL, "runtime.run_id=toy"]) == 0
    )
    return work / "toy"


def test_commands_are_registered_and_no_longer_planned(capsys):
    assert main([]) == 0
    usage = capsys.readouterr().out
    assert all(f"  {name} " in usage for name in ("train", "test", "infer"))
    assert not {"train", "test", "infer"} & {name for name, _, _ in PLANNED}
    for command in ("train", "test", "infer"):
        with pytest.raises(SystemExit) as stop:
            main([command, "--help"])
        assert stop.value.code == 0 and "--opts" in capsys.readouterr().out


def test_train_writes_a_complete_run_directory(trained, capsys):
    names = {path.name for path in trained.iterdir()}
    assert {
        "config.resolved.yaml", "config.sources", "run.json", "git.json", "env.json",
        "log.txt", "checkpoints", "predictions", "results",
    } <= names  # fmt: skip
    assert sorted(p.name for p in (trained / "checkpoints").iterdir()) == [
        "best.pth",
        "epoch_0001.pth",
        "epoch_0002.pth",
    ]
    assert sorted(p.name for p in (trained / "predictions").iterdir()) == ["centre", "raw"]
    assert sorted(p.name for p in (trained / "results/epoch_0002").iterdir()) == [
        "centre.json",
        "raw.json",
    ]
    resolved = parse_yaml((trained / "config.resolved.yaml").read_text())
    assert resolved["train"]["epochs"] == 2 and resolved["train"]["scheduler_unit"] == "epoch"
    overrides = (trained / "config.sources/overrides.txt").read_text().splitlines()
    assert "train.epochs=2" in overrides and any(
        o.startswith("runtime.work_dir=") for o in overrides
    )
    run = json.loads((trained / "run.json").read_text())
    assert run["command_name"] == "train" and run["protocol"] == "toy-synthetic-v0"
    assert run["gt_assisted"] == [] and run["world_size"] == 1
    log = (trained / "log.txt").read_text()
    assert "ToyDetector" in log and "epoch 2/2" in log and "✓ trained epochs 1-2" in log
    assert (
        read_checkpoint(trained / "checkpoints/epoch_0002.pth")["config_hash"] == run["config_hash"]
    )


def test_test_scores_every_variant_and_writes_records(trained, tmp_path, capsys):
    weights = trained / "checkpoints/best.pth"
    arguments = [
        "test",
        TOY,
        "--weights",
        str(weights),
        "--work-dir",
        str(tmp_path),
        "--opts",
        "runtime.run_id=scored",
    ]
    assert main(arguments) == 0
    out = capsys.readouterr().out
    assert "8 matched, 0 missing, 0 unexpected, 0 with a different shape" in out
    assert "protocol toy-synthetic-v0" in out and "split test" in out and "16/16" in out
    for variant in ("raw", "centre"):
        record = json.loads((tmp_path / f"scored/results/{variant}.json").read_text())
        assert record["partial"] is False and record["requested"] == record["scored"] == 16
        assert (
            record["weights"]["sha256"] == file_sha256(weights)
            and record["weights"]["kind"] == "model"
        )
        assert record["weights"]["config_hash"] == read_checkpoint(weights)["config_hash"]
        assert set(record["metrics"]) == {"AP", "AP_tight"} and len(record["per_scene"]) == 2
    centre = json.loads((tmp_path / "scored/results/centre.json").read_text())
    assert centre["postprocessing"] == [{"type": "ToyCentreFilter", "radius": 0.15}]
    assert len(list((tmp_path / "scored/predictions/raw").rglob("*.npy"))) == 16

    assert main([*arguments[:-1], "runtime.run_id=on-val", "--split", "val"]) == 0  # another split
    assert "split val" in capsys.readouterr().out
    assert json.loads((tmp_path / "on-val/results/raw.json").read_text())["requested"] == 8


def test_infer_dumps_grasps_without_an_evaluator(trained, tmp_path, capsys):
    weights, output = trained / "checkpoints/epoch_0002.pth", tmp_path / "grasps"
    arguments = ["infer", TOY, "--weights", str(weights), "--input", "val", "--output", str(output)]
    assert main([*arguments, "--opts", "evaluation.evaluator=null"]) == 0
    assert "✓ 8 frames" in capsys.readouterr().out
    arrays = sorted((output / "predictions/raw/scene_0100/toy").glob("*.npy"))
    assert [p.name for p in arrays] == [f"{frame:04d}.npy" for frame in range(8)]
    assert np.load(arrays[0]).shape == (64, 17)
    assert len(list((output / "predictions/centre").rglob("*.npy"))) == 8
    assert not (output / "results").exists() and (output / "config.resolved.yaml").is_file()
    run = json.loads((output / "run.json").read_text())
    assert run["frames"] == 8 and run["weights"]["sha256"] == file_sha256(weights)
    assert main(arguments) == 1  # the output directory now holds a run and is not overwritten
    assert "already holds a run" in capsys.readouterr().err


def test_failures_return_non_zero_with_a_reason(trained, tmp_path, capsys):
    work = ["--work-dir", str(tmp_path)]
    weights = str(trained / "checkpoints/best.pth")

    assert main(["train", MODEL_ONLY, *work]) == 2  # config problems: 2
    assert "`grasppanda train` needs this section" in capsys.readouterr().err
    assert main(["train", TOY, *work, "--opts", "train.epoch=3"]) == 2
    assert "Did you mean 'epochs'?" in capsys.readouterr().err
    assert main(["test", TOY, *work, "--weights", weights, "--opts", "data.test=null"]) == 2
    assert "`grasppanda test` needs this section" in capsys.readouterr().err
    assert (
        main(["infer", TOY, "--weights", weights, "--input", "val", "--opts", "data.val=null"]) == 2
    )
    assert "`grasppanda infer` reads this section" in capsys.readouterr().err
    assert main(["infer", TOY, "--weights", weights, "--input", "no/such/dir"]) == 2
    assert "neither a data section" in capsys.readouterr().err

    assert main(["test", TOY, *work]) == 1  # the run itself failed: 1
    assert "needs weights" in capsys.readouterr().err
    assert (
        main(["test", TOY, *work, "--weights", weights, "--opts", "model.backbone.out_channels=32"])
        == 1
    )
    err = capsys.readouterr().err
    assert "WeightMismatchError" in err and "different shape" in err and "Nothing was loaded" in err
    assert main(["train", TOY, *work, "--resume"]) == 1
    assert "no checkpoint below" in capsys.readouterr().err
    assert main(["train", TOY, *work, "--opts", "runtime.device=gpu"]) == 1
    assert "runtime.device: 'gpu' is not a device" in capsys.readouterr().err


def test_resume_flag_continues_the_run(tmp_path, capsys):
    work = ["--work-dir", str(tmp_path)]
    opts = [
        "--opts",
        "data.train.dataset.num_scenes=1",
        "evaluation.interval=0",
        "evaluation.save_best=null",
    ]
    assert main(["train", TOY, *work, *opts, "train.epochs=1"]) == 0
    capsys.readouterr()
    assert main(["train", TOY, *work, "--resume", *opts, "train.epochs=3"]) == 1  # another config
    assert "different experiment" in capsys.readouterr().err
    extended = [*opts, "train.epochs=3", "runtime.allow_config_mismatch=true"]
    assert main(["train", TOY, *work, "--resume", *extended]) == 0
    out = capsys.readouterr().out
    assert "resumed" in out and "✓ trained epochs 2-3" in out
    (run,) = tmp_path.iterdir()
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == [
        f"epoch_000{n}.pth" for n in (1, 2, 3)
    ]


def test_infer_points_the_reader_at_a_directory(tmp_path):
    """``--input DIR`` becomes ``data.<split>.dataset.root=DIR``; a reader without ``root`` says so."""
    from grasppanda.cli import infer

    parser_args = type("Args", (), {})()
    parser_args.opts, parser_args.weights, parser_args.output = [], None, None
    parser_args.config, parser_args.input = TOY, str(tmp_path)
    assert infer.overrides(parser_args) == [f'data.test.dataset.root="{tmp_path}"']
    assert main(["infer", TOY, "--input", str(tmp_path)]) == 2  # ToyScenes has no `root`
