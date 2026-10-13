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


def test_a_result_states_its_numbers_for_the_splits_of_its_protocol(write):
    transfer = VALID.replace("graspnet1b-sv-v1", "graspclutter6d-sv-v1")
    numbers = "AP: {test: 21.67}, AP0.8: {test: 26.62}, AP0.4: {test: 15.9}"
    text = transfer.replace("AP: {seen: 65.7, similar: 53.75, novel: 23.98}", numbers)
    (model,) = load_metafile(write("demo/metafile.yaml", text)).models
    assert model.results[0].ap == {"test": 21.67}
    assert model.results[0].numbers() == {
        "AP": {"test": 21.67},
        "AP0.8": {"test": 26.62},
        "AP0.4": {"test": 15.9},
    }
    (plain,) = load_metafile(write("plain/metafile.yaml", VALID)).models
    assert plain.results[0].ap_at == {} and list(plain.results[0].numbers()) == ["AP"]

    cases = [
        # the splits of another protocol
        (transfer, "models.0.results.0.AP", "must map splits (test) to numbers"),
        # a friction level given for other splits than AP
        (
            VALID.replace("AP: {seen", "AP0.4: {seen: 50.0}, AP: {seen"),
            "models.0.results.0.AP0.4",
            "must be given for the same splits as AP",
        ),
        (
            VALID.replace("graspnet1b-sv-v1", "demo-v0"),
            "models.0.protocol",
            "unknown protocol 'demo-v0'; known: graspnet1b-sv-v1, graspclutter6d-sv-v1",
        ),
    ]
    for index, (broken, path, fragment) in enumerate(cases):
        with pytest.raises(ConfigError) as error:
            load_metafile(write(f"broken{index}/metafile.yaml", broken))
        assert fragment in {issue.path: issue.message for issue in error.value.issues}[path]


def test_a_measured_result_names_a_record_that_exists(write):
    weights = "    weights: {url: 'https://example.org/w.tar', sha256: " + "a" * 64 + ", "
    weights += "license: MIT, source: author}\n    results:"
    text = VALID.replace("    results:", weights).replace(
        "provenance: reported, collision_filter: false, source: paper",
        "provenance: author-ckpt, collision_filter: false, evaluator: GraspNet1BEvaluator, "
        "record: records/demo_raw.json",
    )
    path = write("demo/metafile.yaml", text)
    with pytest.raises(ConfigError) as error:
        load_metafile(path)
    (issue,) = error.value.issues
    assert issue.path == "models.0.results.0.record"
    assert "no such result record next to the metafile: records/demo_raw.json" in issue.message
    write("demo/records/demo_raw.json", "{}")  # the loader checks presence; tests check content
    (model,) = load_metafile(path).models
    assert model.results[0].record == "records/demo_raw.json"


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
