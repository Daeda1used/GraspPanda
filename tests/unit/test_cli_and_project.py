"""The command line, run snapshots and the example project, end to end."""

from __future__ import annotations

import json
import tarfile

import pytest
import torch

from grasppanda import __version__
from grasppanda.cli.main import main
from grasppanda.config import Config, available_configs, configs_root, parse_yaml, write_snapshot
from grasppanda.registry import DETECTORS
from grasppanda.structures import Observation, PointSet, SampleKey

from .conftest import REPO

TOY = str(REPO / "projects/example_project/configs/toy_detector.yaml")


def test_version_and_help(capsys):
    assert __version__ and __version__ != "0.0.0+unknown"
    assert main([]) == 0
    assert "print-config" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0 and "GraspPanda" in capsys.readouterr().out


def test_print_config_check(capsys):
    assert main(["print-config", TOY, "--check"]) == 0
    out = capsys.readouterr().out
    assert "is valid" in out and "model.head: ToyHead" in out and "adam_10e.yaml" in out


def test_print_config_outputs_loadable_yaml(capsys):
    assert main(["print-config", TOY, "--defaults", "--opts", "train.epochs=3"]) == 0
    printed = parse_yaml(capsys.readouterr().out)
    assert printed["train"]["epochs"] == 3 and printed["optimizer"]["lr"] == pytest.approx(1e-3)
    assert printed["runtime"]["work_dir"] == "work_dirs/toy_detector"


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("model.head.num_angles=6", "Did you mean 'num_angle'?"),
        ("model.grouper.type=ToyDepthGrouper", "ToyHead expects ('seed', 'channel')"),
        ("runtim.seed=1", "Did you mean 'runtime'?"),
    ],
)
def test_print_config_reports_problems_and_fails(capsys, override, expected):
    assert main(["print-config", TOY, "--opts", override]) == 2
    captured = capsys.readouterr()
    assert expected in captured.err and captured.out == ""


def test_changed_type_is_announced(capsys):
    assert main(["print-config", TOY, "--check", "--opts", "optimizer.type=SGD"]) == 0
    assert "dropped inherited keys: lr, weight_decay" in capsys.readouterr().err


def test_list_and_env(capsys):
    assert main(["print-config", TOY, "--list", "detectors"]) == 0
    assert "ToyDetector" in capsys.readouterr().out
    assert main(["print-config", "--list", "nonsense"]) == 2
    capsys.readouterr()
    assert main(["env", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["torch"] == torch.__version__


def test_missing_config_name_suggests_shipped_ones(capsys):
    assert main(["print-config", "no/such_config"]) == 2
    assert "neither a file nor a shipped config name" in capsys.readouterr().err
    assert configs_root().is_dir() and isinstance(available_configs(), list)


def test_example_project_runs_end_to_end():
    config = Config.load(TOY)
    model = DETECTORS.build(config["model"]).eval()
    observation = Observation(
        keys=[SampleKey("demo", 0, "none", frame) for frame in range(2)],
        points=PointSet.from_dense(torch.rand(2, 1024, 3)),
    )
    grasps = model.predict(observation)
    assert len(grasps) == 2 and all(len(group) == 64 for group in grasps)
    assert grasps[0].to_graspnet_array().shape == (64, 17)
    assert (grasps[0].width >= 0).all() and (grasps[0].width <= 0.1).all()
    assert set((grasps[0].depth * 100).round().int().tolist()) <= {1, 2, 3, 4}


def test_snapshot_records_config_code_and_environment(tmp_path):
    config = Config.load(TOY, ["train.epochs=2"])
    run = write_snapshot(config, tmp_path / "run", command=["grasppanda", "train", TOY])
    run_dir = tmp_path / "run"
    assert run["config_hash"] == config.digest() and run["seed"] == 0
    assert parse_yaml((run_dir / "config.resolved.yaml").read_text())["train"]["epochs"] == 2
    names = sorted(p.name for p in (run_dir / "config.sources").iterdir())
    assert names == [
        "00_toy_detector.yaml",
        "01_default_runtime.yaml",
        "02_adam_10e.yaml",
        "overrides.txt",
    ]
    assert (run_dir / "config.sources/overrides.txt").read_text() == "train.epochs=2\n"
    env = json.loads((run_dir / "env.json").read_text())
    assert env["torch"] == torch.__version__ and any(
        p.startswith("numpy==") for p in env["packages"]
    )
    git = json.loads((run_dir / "git.json").read_text())
    if git["available"] and (run_dir / "code.tar.gz").exists():
        with tarfile.open(run_dir / "code.tar.gz") as archive:
            assert "grasppanda/registry.py" in archive.getnames()
