"""Looking a config up in the model zoo: what loads as it is, and what has to be converted."""

from __future__ import annotations

import subprocess
import sys

from grasppanda.config import configs_root
from grasppanda.hub import find_model, sha256_of, zoo_weights

from .conftest import REPO

ENTRY = """
method: Demo
title: A demo method
venue: Nowhere 2026
upstream: {{repository: someone/demo, commit: abc1234, license: MIT}}
implementation: clean-room
models:
  - name: demo_model
    config: _base_/default_runtime.yaml
    protocol: graspnet1b-sv-v1
    camera: realsense
{weights}
"""
CONFIG = configs_root() / "_base_/default_runtime.yaml"


def test_shipped_configs_are_found_and_none_has_ready_weights_yet():
    name = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e.yaml"
    metafile, model = find_model(configs_root() / name)
    assert metafile.method == "GraspNet-baseline" and model.camera == "realsense"
    assert model.weights.source == "author"
    path, note = zoo_weights(configs_root() / name)
    assert path is None  # the authors' file is in the authors' format and under their licence
    assert "not downloaded for you" in note and "tools.convert_checkpoint" in note
    assert "non-commercial" in note and str(metafile.path.parent / "README.md") in note
    # the exact command of the metafile, and the file to pass
    assert f"`{model.weights.convert}`" in note and "--weights baseline_realsense.pth" in note
    for config in configs_root().glob("*/*.yaml"):  # nothing can be downloaded yet
        if config.name != "metafile.yaml":
            assert zoo_weights(config)[0] is None, config
    assert find_model(configs_root() / "_base_/default_runtime.yaml") is None
    assert zoo_weights(REPO / "pyproject.toml") == (
        None,
        "The model zoo does not list this config.",
    )


def test_weights_released_by_the_project_are_fetched_and_verified(write, tmp_path):
    released = tmp_path / "released.pth"
    released.write_bytes(b"weights trained here")
    block = (
        f"    weights: {{url: 'file://{released}', sha256: {sha256_of(released)}, "
        "license: Apache-2.0, source: grasppanda}"
    )
    root = write("zoo/demo/metafile.yaml", ENTRY.format(weights=block)).parent.parent
    path, note = zoo_weights(CONFIG, root, cache_dir=tmp_path / "cache")
    assert note == "" and path.read_bytes() == b"weights trained here"
    assert path.parent == tmp_path / "cache"


def test_author_weights_and_missing_weights_are_explained_not_fetched(write, tmp_path):
    block = (
        "    weights: {url: 'https://example.org/authors.tar', sha256: " + "a" * 64 + ", "
        "license: non-commercial, source: author}"
    )
    root = write("author/demo/metafile.yaml", ENTRY.format(weights=block)).parent.parent
    path, note = zoo_weights(CONFIG, root, cache_dir=tmp_path / "cache")
    assert path is None and not (tmp_path / "cache").exists()  # nothing was downloaded
    assert "https://example.org/authors.tar (licence: non-commercial)" in note

    root = write("none/demo/metafile.yaml", ENTRY.format(weights="")).parent.parent
    path, note = zoo_weights(CONFIG, root)
    assert path is None and "has no weights for 'demo_model' yet" in note


def test_weights_trained_here_but_not_published_are_explained_not_fetched(write, tmp_path):
    block = (
        "    weights: {url: null, sha256: " + "b" * 64 + ", license: null, source: grasppanda, "
        "note: 'Not published yet: the owner decides where.'}"
    )
    root = write("unpublished/demo/metafile.yaml", ENTRY.format(weights=block)).parent.parent
    path, note = zoo_weights(CONFIG, root, cache_dir=tmp_path / "cache")
    assert path is None and not (tmp_path / "cache").exists()  # nothing was downloaded
    assert "'demo_model' were trained by GraspPanda but are not published yet" in note
    assert "cannot be downloaded (sha256 " + "b" * 64 + ")" in note
    assert "Not published yet: the owner decides where." in note


def test_a_command_given_the_trained_gsnet_says_that_its_weights_are_not_published(capsys):
    from grasppanda.cli.main import main

    name = "gsnet/gsnet_sunet14_g1b-realsense_10e"
    _, model = find_model(configs_root() / f"{name}.yaml")
    assert model.weights.source == "grasppanda" and not model.weights.published
    path, note = zoo_weights(configs_root() / f"{name}.yaml")
    assert path is None and "are not published yet" in note and model.weights.sha256 in note
    # what can be downloaded instead: the authors' released file, with what it gives
    assert model.author_weights.url in note and model.author_weights.convert in note
    assert model.author_weights.note in note
    # `grasppanda demo <zoo name>` without --weights stops before it builds the model
    assert main(["demo", name, "--example", "--device", "cpu"]) == 1
    error = capsys.readouterr().err
    assert "needs weights" in error and "trained by GraspPanda but are not published yet" in error


def test_the_inferencer_is_importable_from_the_package_without_loading_it_early():
    code = (
        "import sys, grasppanda\n"
        "assert 'grasppanda.apis' not in sys.modules and 'torch' not in sys.modules\n"
        "from grasppanda import GraspInferencer\n"
        "assert GraspInferencer.__module__ == 'grasppanda.apis.inferencer'\n"
        "assert 'open3d' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
