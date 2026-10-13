"""``grasppanda prepare`` end to end on the synthetic tree."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from grasppanda.cli.main import main

from .test_data_graspness import NATIVE, write_legacy
from .test_data_graspnet1b import PREPROCESS, TRAIN, TinyGraspNet, build, tree  # noqa: F401


@pytest.fixture
def config(tree, tmp_path):  # noqa: F811
    """A dataset config for the synthetic tree, as a user would write it."""
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "root": str(tree["root"]),
        "preprocess": PREPROCESS,
        "graspness": NATIVE,
        "transforms": TRAIN,
    }
    file = tmp_path / "tiny.yaml"
    document = {
        "custom_imports": ["tests.unit.test_data_graspnet1b"],
        "data": {"train": {"dataset": node}},
    }
    file.write_text(yaml.safe_dump(document))
    return str(file)


def test_prepare_is_a_registered_command(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "prepare" in out.split("planned commands")[0]
    assert "prepare" not in out.split("planned commands")[1]


def test_check_labels_graspness_then_train(tree, config, capsys):  # noqa: F811
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--check"]) == 1
    out = capsys.readouterr().out
    assert "tree is complete for realsense, kinect" in out
    assert "grasp labels 0/5 shards" in out and "graspness 0/2 scenes" in out

    assert main([*base, "--graspness", "--workers", "1"]) == 1  # labels come first
    assert "grasppanda prepare tinygraspnet --labels" in capsys.readouterr().err

    assert main([*base, "--labels"]) == 0
    assert "5 shards built and verified against their sources, 0 already" in capsys.readouterr().out
    assert main([*base, "--labels"]) == 0
    assert "0 shards built and verified against their sources, 5 already" in capsys.readouterr().out

    assert main([*base, "--graspness", "--workers", "1", "--scenes", "0"]) == 0
    assert "1 scenes written" in capsys.readouterr().out
    assert main([*base, "--graspness", "--workers", "1"]) == 0  # resumes: scene 1 only
    out = capsys.readouterr().out
    assert "1 of 2 scenes already present" in out and "1 scenes written" in out
    assert main([*base, "--graspness", "--workers", "1"]) == 0
    assert "nothing to do" in capsys.readouterr().out

    assert main([*base, "--check", "--camera", "realsense"]) == 0
    out = capsys.readouterr().out
    assert "grasp labels 5/5 shards" in out and "graspness 2/2 scenes" in out and "native" in out

    dataset = build(tree, graspness=NATIVE)  # what the config describes can now be read
    batch = dataset[0]
    assert set(batch.supervision.point_labels) == {"objectness", "instance", "graspness"}
    manifest = json.loads((dataset.graspness.asset.path / "manifest.json").read_text())
    assert manifest["uses_ground_truth"] is True
    assert manifest["preprocessing_fingerprint"] == dataset.fingerprint
    assert sorted(manifest["shards"]) == ["scene_000000", "scene_000001"]


def test_check_reports_a_broken_tree(tree, config, capsys):  # noqa: F811
    (tree["root"] / "scenes/scene_0000/realsense/meta/0001.mat").unlink()
    assert main(["prepare", "tinygraspnet", "--config", config, "--check"]) == 1
    out = capsys.readouterr().out
    assert "the tree lacks 1 item(s)" in out
    assert "scenes/scene_0000/realsense/meta/: 1 of 3 .mat files missing, e.g. 0001.mat" in out


def test_import_legacy_builds_a_marked_asset(tree, config, capsys):  # noqa: F811
    probe = build(tree, transforms=[])
    write_legacy(probe, tree, [0, 1])
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--graspness", "--import-legacy", "--workers", "1"]) == 0
    assert "legacy-import" in capsys.readouterr().out
    imported = build(tree, graspness={"source": "legacy_import"})
    direct = build(tree, graspness={"source": "legacy"})
    assert imported.asset_records()[1]["provenance"] == "legacy-import"
    key = imported.keys[3]
    assert np.allclose(
        imported.sample(key).point_labels["graspness"],
        direct.sample(key).point_labels["graspness"],
        atol=2.5e-4,  # float16 storage
    )
    write_legacy(probe, tree, [0], drop_rows=2)
    other = ["--split", "sample", "--camera", "kinect"]
    write_legacy(build(tree, transforms=[], camera="kinect"), tree, [0], drop_rows=2)
    assert main([*base, "--graspness", "--import-legacy", "--workers", "1", *other]) == 1
    assert "does not align with this dataset's preprocessing" in capsys.readouterr().err


@pytest.mark.slow
def test_graspness_in_worker_processes_matches_single_process(tree, config, capsys):  # noqa: F811
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--labels", "--split", "all"]) == 0
    assert main([*base, "--graspness", "--split", "all", "--workers", "2"]) == 0
    assert "3 scenes written" in capsys.readouterr().out
    parallel = build(tree, graspness=NATIVE, split="all")
    serial = build(tree, graspness={**NATIVE, "dtype": "float32"}, split="all", check_assets=False)
    from .test_data_graspness import generate

    generate(serial)
    for key in parallel.keys:
        one = parallel.graspness.read_map(key).astype(np.float32)
        two = serial.graspness.read_map(key)
        assert np.array_equal(np.isnan(one), np.isnan(two))
        assert np.allclose(one[~np.isnan(one)], two[~np.isnan(two)], atol=2.5e-4)
    assert sorted(parallel.graspness.asset.shards()) == [f"scene_{s:06d}" for s in range(3)]


def test_usage_errors(tree, config, capsys):  # noqa: F811
    assert main(["prepare", "tinygraspnet", "--config", config]) == 2
    assert "nothing to do: pass --check or one of --labels, --graspness" in capsys.readouterr().err
    assert main(["prepare", "tinygraspnet", "--config", config, "--import-legacy", "--check"]) == 2
    with pytest.raises(SystemExit, match="unknown dataset 'graspnet2b'. Datasets: .*graspnet1b"):
        main(["prepare", "graspnet2b", "--check"])
    assert main(["prepare", "graspnet1b", "--check", "--root", str(tree["root"] / "none")]) == 1
    assert "does not exist" in capsys.readouterr().out
    assert TinyGraspNet.default_config("kinect") == "_base_/datasets/tinygraspnet_kinect.yaml"
