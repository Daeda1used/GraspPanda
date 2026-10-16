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


UNPUBLISHED = """    weights:
      url: null
      sha256: {sha}
      license: null
      source: {source}
{note}    results:"""


def test_weights_trained_here_may_be_listed_before_they_are_published(write):
    def metafile(name: str, source: str = "grasppanda", note: str | None = "Not published yet."):
        line = f"      note: {note!r}\n" if note is not None else ""
        block = UNPUBLISHED.format(sha="b" * 64, source=source, note=line)
        text = VALID.replace("    results:", block).replace(
            "provenance: reported, collision_filter: false, source: paper",
            "provenance: reproduced, collision_filter: false, evaluator: GraspNet1BEvaluator, "
            "record: records/demo_raw.json",
        )
        write(f"{name}/records/demo_raw.json", "{}")
        return write(f"{name}/metafile.yaml", text)

    (model,) = load_metafile(metafile("listed")).models
    assert model.weights.url is None and model.weights.license is None
    assert model.weights.published is False and model.weights.note == "Not published yet."
    assert model.results[0].provenance == "reproduced"

    cases = [
        # without a url the entry has to say that the weights are not published
        (metafile("silent", note=None), "models.0.weights.note", "not published yet"),
        (metafile("vague", note="soon"), "models.0.weights.note", "not published yet"),
        # only weights trained here may be listed without a url; the authors' need one
        (metafile("author", source="author"), "models.0.weights.url", "is required"),
        (metafile("author2", source="author"), "models.0.weights.license", "is required"),
    ]
    for path, where, fragment in cases:
        with pytest.raises(ConfigError) as error:
            load_metafile(path)
        assert fragment in {issue.path: issue.message for issue in error.value.issues}[where]


CONVERT = (
    "python -m tools.convert_checkpoint authors.tar demo_realsense.pth "
    "--source-format economicgrasp --camera realsense"
)


def _issues(path) -> dict[str, str]:
    with pytest.raises(ConfigError) as error:
        load_metafile(path)
    return {issue.path: issue.message for issue in error.value.issues}


def test_the_authors_checkpoint_says_how_it_is_converted(write):
    def metafile(name: str, convert: str, source: str = "author") -> object:
        block = (
            "    weights: {url: 'https://example.org/authors.tar', sha256: " + "a" * 64 + ", "
            f"license: MIT, source: {source}, convert: '{convert}'}}\n    results:"
        )
        return write(f"{name}/metafile.yaml", VALID.replace("    results:", block))

    (model,) = load_metafile(metafile("ok", CONVERT)).models
    assert model.weights.convert == CONVERT
    assert model.weights.downloaded_name == "authors.tar"
    assert model.weights.converted_name == "demo_realsense.pth"

    where = "models.0.weights.convert"
    trained = _issues(metafile("trained", CONVERT, source="grasppanda"))
    assert "only the authors' checkpoint is converted" in trained[where]
    for name, convert in (
        ("script", "python tools/convert_checkpoint.py authors.tar demo.pth"),
        ("one_file", "python -m tools.convert_checkpoint authors.tar"),
        ("unknown_option", CONVERT + " --strict"),
    ):
        assert (
            "must read 'python -m tools.convert_checkpoint"
            in _issues(metafile(name, convert))[where]
        ), name


def test_the_authors_file_may_be_listed_beside_weights_trained_here(write):
    def metafile(name: str, weights_source: str = "grasppanda", source: str = "author"):
        if weights_source == "grasppanda":
            weights = UNPUBLISHED.format(
                sha="b" * 64, source="grasppanda", note="      note: 'Not published yet.'\n"
            ).replace("    results:", "")
        else:
            weights = (
                "    weights: {url: 'https://example.org/a.tar', sha256: " + "a" * 64 + ", "
                "license: MIT, source: author}\n"
            )
        block = (
            weights + "    author_weights: {url: 'https://example.org/released.tar', "
            "sha256: " + "c" * 64 + f", license: non-commercial, source: {source}, "
            f"convert: '{CONVERT}', note: 'Far below the paper.'}}\n    results:"
        )
        return write(f"{name}/metafile.yaml", VALID.replace("    results:", block))

    (model,) = load_metafile(metafile("ok")).models
    assert model.weights.source == "grasppanda" and model.author_weights.source == "author"
    assert model.author_weights.converted_name == "demo_realsense.pth"
    assert model.author_weights.note == "Far below the paper."
    # the authors' file of a model whose weights are the authors' file is its weights entry
    assert (
        "is for a model whose weights are another file"
        in (_issues(metafile("twice", weights_source="author"))["models.0.author_weights"])
    )
    issues = _issues(metafile("not_theirs", source="grasppanda"))
    assert "must be 'author'" in issues["models.0.author_weights.source"]


def test_fetch_refuses_an_entry_without_a_url_with_a_reason(tmp_path):
    with pytest.raises(
        ValueError, match="no url to fetch the weights with sha256 b+ from: they are not published"
    ):
        fetch(None, "b" * 64, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


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
