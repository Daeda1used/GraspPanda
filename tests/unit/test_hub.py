"""Model zoo metadata and verified weight fetching."""

from __future__ import annotations

import pytest

from grasppanda.config import ConfigError
from grasppanda.hub import ChecksumError, fetch, load_metafile, load_metafiles, sha256_of

VALID = """
method: Demo
title: A demo method
venue: Nowhere 2026
upstream: {repository: someone/demo, commit: abc1234, license: MIT}
implementation: clean-room
models:
  - name: demo_model
    config: _base_/default_runtime.yaml
    protocol: graspnet1b-sv-v1
    camera: realsense
    results:
      - {provenance: reported, collision_filter: false, source: paper, AP: {seen: 65.7, similar: 53.75, novel: 23.98}}
"""


def test_valid_metafile(write):
    meta = load_metafile(write("demo/metafile.yaml", VALID))
    (model,) = meta.models
    assert meta.method == "Demo" and model.weights is None
    assert model.results[0].ap == {"seen": 65.7, "similar": 53.75, "novel": 23.98}
    assert model.results[0].collision_filter is False


@pytest.mark.parametrize(
    ("change", "path", "fragment"),
    [
        ("collision_filter: false, ", "models.0.results.0.collision_filter", "is required"),
        ("provenance: reported", "models.0.results.0.provenance", "must be one of"),
        ("source: paper, ", "models.0.results.0.source", "must say where it is reported"),
        ("config: _base_/default_runtime.yaml", "models.0.config", "no such shipped config"),
    ],
)
def test_reporting_policy_is_enforced(write, change, path, fragment):
    replacement = {
        "provenance: reported": "provenance: guessed",
        "config: _base_/default_runtime.yaml": "config: nope/missing.yaml",
    }.get(change, "")
    with pytest.raises(ConfigError) as error:
        load_metafile(write("demo/metafile.yaml", VALID.replace(change, replacement)))
    assert fragment in {issue.path: issue.message for issue in error.value.issues}[path]


def test_measured_results_need_evidence(write):
    text = VALID.replace(
        "provenance: reported, collision_filter: false, source: paper",
        "provenance: reproduced, collision_filter: null",
    )
    with pytest.raises(ConfigError) as error:
        load_metafile(write("demo/metafile.yaml", text))
    found = {issue.path: issue.message for issue in error.value.issues}
    assert "cannot have an unknown collision filter" in found["models.0.results.0.collision_filter"]
    assert "it is the evidence" in found["models.0.results.0.record"]
    assert "needs the model's weights entry" in found["models.0.results.0"]


def test_shipped_metafiles_are_valid():
    assert isinstance(load_metafiles(), list)


def test_fetch_verifies_the_checksum(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"panda" * 1000)
    digest = sha256_of(source)
    cached = fetch(str(source), digest, cache_dir=tmp_path / "cache")
    assert cached.read_bytes() == source.read_bytes()
    assert fetch(f"file://{source}", digest, cache_dir=tmp_path / "cache") == cached
    with pytest.raises(ChecksumError, match="sha256 mismatch"):
        fetch(str(source), "0" * 64, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.part"))
