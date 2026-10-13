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
    for config in configs_root().glob("*/*.yaml"):  # the zoo is empty: nothing loads as it is
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
