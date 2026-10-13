"""``tools/record_result.py`` on synthetic runs: what it checks, what it writes into
``configs/<method>/records/`` and the metafile rows it prints.

A synthetic run has the files of ``grasppanda test`` (``run.json``, ``git.json``,
``results/<variant>.json``) with a per-scene table whose means are known by hand. No
dataset, no model: the tool reads records, it never scores anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from grasppanda.hub import load_metafile, sha256_of
from tools import record_result

SOURCE_SHA = "5" * 64  # the "released file" the synthetic weights were converted from
FRICTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
#: Per-scene mean precision of the synthetic GraspNet-1Billion run: scenes alternate
#: between the two values, so the three split means are 0.61, 0.41 and 0.21.
G1B_LEVELS = {
    "seen": (range(100, 130), 0.60),
    "similar": (range(130, 160), 0.40),
    "novel": (range(160, 190), 0.20),
}

METAFILE = """
method: Demo
title: A demo method
venue: Nowhere 2026
upstream: {{repository: someone/demo, commit: abc1234, license: MIT}}
implementation: clean-room
models:
  - name: demo_model
    config: _base_/default_runtime.yaml
    protocol: {protocol}
    camera: {camera}
{weights}    results:
      - provenance: reported
        collision_filter: false
        source: paper
        AP: {reported}
"""
WEIGHTS = (
    "    weights: {{url: 'https://example.org/authors.tar', sha256: {sha}, license: MIT, "
    "source: {source}}}\n"
)


def scene_table(camera: str, scenes: dict[int, float], frames: int) -> dict:
    return {
        f"{camera}/scene_{scene:04d}": {
            "frames": frames,
            "mean_precision": value,
            # column 1 is friction 0.4 (half the value), column 3 friction 0.8 (the value + 0.1)
            "per_column": [0.0, value / 2, value, value + 0.1, value, value],
        }
        for scene, value in scenes.items()
    }


def g1b_scenes(shift: float = 0.0) -> dict[int, float]:
    return {
        scene: level + shift + 0.02 * (scene % 2)
        for scenes, level in G1B_LEVELS.values()
        for scene in scenes
    }


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    """An empty cache directory, and the only ``GRASPPANDA_*`` variable of the process."""
    for name in [name for name in os.environ if name.startswith("GRASPPANDA_")]:
        monkeypatch.delenv(name)
    root = tmp_path / "cache"
    monkeypatch.setenv("GRASPPANDA_CACHE", str(root))
    return root


def make_run(
    cache: Path,
    *,
    dataset: str = "graspnet1b",
    camera: str = "realsense",
    frames: int = 256,
    scenes: dict[int, float] | None = None,
    converted_from: str | None = SOURCE_SHA,
) -> Path:
    """A finished run below ``cache``: two variants, clean code, hashed weights."""
    weights = cache / "weights" / "converted" / "demo.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source_sha256": converted_from, "source_file": "authors.tar", "source_format": "demo"}
    torch.save({"state_dict": {}, **({"meta": meta} if converted_from else {})}, weights)
    protocol = f"{dataset}-sv-v1"
    run = cache / "runs" / "demo_test" / "20260101-000000-test"
    (run / "results").mkdir(parents=True)
    scenes = g1b_scenes() if scenes is None else scenes
    for variant, steps, shift in (
        ("raw", [], 0.0),
        ("cd", [{"type": "ModelFreeCollisionFilter", "voxel_size": 0.01}], 0.05),
    ):
        table = scene_table(
            camera, {scene: value + shift for scene, value in scenes.items()}, frames
        )
        total = frames * len(table)
        mean = sum(entry["mean_precision"] for entry in table.values()) / len(table)
        record = {
            "schema": 1,
            "variant": variant,
            "protocol": protocol,
            "dataset": dataset,
            "split": "test",
            "cameras": [camera],
            "requested": total,
            "scored": total,
            "missing": [],
            "coverage": 1.0,
            "partial": False,
            "metrics": {"AP": mean},
            "partial_metrics": None,
            "postprocessing": steps,
            "gt_assisted": ["workspace"],
            "evaluator": {
                "type": "DemoEvaluator",
                "config": {"data_root": None},
                "info": {
                    "cache": {"directory": str(cache / dataset / "eval" / "abc")},
                    "constants": {
                        "frictions": FRICTIONS,
                        "reported": [["AP0.8", 0.8], ["AP0.4", 0.4]],
                    },
                },
            },
            "config_hash": "c" * 64,
            "code": {"commit": "a" * 40, "dirty": False, "version": "0.1.0"},
            "weights": {
                "path": str(weights),
                "sha256": sha256_of(weights),
                "kind": "model",
                "clean_load": True,
            },
            "seed": 0,
            "per_scene": table,
            "created": "2026-01-01T00:00:00+00:00",
        }
        (run / "results" / f"{variant}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True)
        )
    (run / "git.json").write_text(
        json.dumps({"available": True, "commit": "a" * 40, "dirty": False})
    )
    info = {
        "command_name": "test",
        "config": "/somewhere/configs/_base_/default_runtime.yaml",
        "config_hash": "c" * 64,
    }
    (run / "run.json").write_text(json.dumps(info))
    return run


def edit(run: Path, variant: str, change) -> None:
    """Apply ``change(record)`` to one stored record of a run."""
    path = run / "results" / f"{variant}.json"
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))


def write_metafile(
    tmp_path: Path,
    *,
    weights: str | None = "author",
    sha: str = SOURCE_SHA,
    protocol: str = "graspnet1b-sv-v1",
    camera: str = "realsense",
    reported: str = "{seen: 60.0, similar: 40.0, novel: 20.0}",
) -> Path:
    """``<tmp>/configs/demo/metafile.yaml`` with one model; returns the configs directory."""
    block = WEIGHTS.format(sha=sha, source=weights) if weights else ""
    path = tmp_path / "configs" / "demo" / "metafile.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        METAFILE.format(protocol=protocol, camera=camera, weights=block, reported=reported)
    )
    return path.parent.parent


def record(run: Path, configs: Path, *extra: str) -> int:
    return record_result.main([str(run), "--method", "demo", "--configs", str(configs), *extra])


# ----------------------------------------------------------- the complete procedure


def test_a_whole_split_run_is_copied_and_its_rows_are_computed(cache, tmp_path, capsys):
    run, configs = make_run(cache), write_metafile(tmp_path)
    assert record(run.parent, configs) == 0  # the work directory holds one run with records
    out = capsys.readouterr().out

    # the two records, named after model, provenance, split and variant
    files = sorted((configs / "demo" / "records").iterdir())
    assert [file.name for file in files] == [
        "demo_model_author-ckpt_test_cd.json",
        "demo_model_author-ckpt_test_raw.json",
    ]
    for file in files:
        original = (run / "results" / f"{file.stem.rsplit('_', 1)[1]}.json").read_text()
        # the record as the run wrote it, but for the directory of this machine
        assert file.read_text() == original.replace(str(cache), "$GRASPPANDA_CACHE") + "\n"
        assert str(tmp_path) not in file.read_text()
        stored = json.loads(file.read_text())
        assert stored["weights"]["path"] == "$GRASPPANDA_CACHE/weights/converted/demo.pth"
        assert (
            stored["evaluator"]["info"]["cache"]["directory"]
            == "$GRASPPANDA_CACHE/graspnet1b/eval/abc"
        )

    # the rows: means of the per-scene values over the scenes of each split, in percent
    assert "AP: {seen: 61.00, similar: 41.00, novel: 21.00}" in out
    assert "AP: {seen: 66.00, similar: 46.00, novel: 26.00}" in out
    assert out.count("provenance: author-ckpt") == 2 and "AP0.8" not in out
    assert "collision_filter: false" in out and "collision_filter: true" in out
    assert "record: records/demo_model_author-ckpt_test_raw.json" in out
    assert "(gt_assisted: workspace)" in out
    assert "converted from authors.tar" in out and "the file the metafile lists" in out
    assert "the whole test split (90 scenes, 23,040 frames)" in out
    assert out.rstrip().endswith("paste the rows above under `results:` of the model")

    # pasted into the metafile, the rows load, and a second run changes nothing
    metafile = configs / "demo" / "metafile.yaml"
    rows = record_result.rows_yaml(
        record_result.load_run(run), "author-ckpt", "demo_model_author-ckpt_test"
    )
    metafile.write_text(metafile.read_text() + rows)
    measured = load_metafile(metafile).models[0].results[1:]
    assert [(r.provenance, r.collision_filter) for r in measured] == [
        ("author-ckpt", False),
        ("author-ckpt", True),
    ]
    assert measured[0].ap == {"seen": 61.0, "similar": 41.0, "novel": 21.0}
    assert measured[0].evaluator == "DemoEvaluator"
    before = {file: file.stat().st_mtime_ns for file in files}
    assert record(run, configs) == 0
    again = capsys.readouterr().out
    assert again.count("unchanged") == 2 and "wrote" not in again
    assert again.rstrip().endswith("the metafile states these rows")
    assert before == {file: file.stat().st_mtime_ns for file in files}

    # a metafile that states other numbers for a record is an error, not a silent state
    metafile.write_text(metafile.read_text().replace("seen: 61.00", "seen: 61.50"))
    assert record(run, configs) == 1
    assert "OTHER rows for: raw" in capsys.readouterr().err


def test_a_single_split_protocol_states_ap_and_both_friction_levels(cache, tmp_path, capsys):
    scenes = dict.fromkeys(range(1, 236), 0.25)  # 235 scenes of 13 views
    run = make_run(
        cache, dataset="graspclutter6d", camera="realsense-d435", frames=13, scenes=scenes
    )
    configs = write_metafile(tmp_path)  # lists the model under another config and protocol
    (run / "run.json").write_text(
        json.dumps(
            {
                "command_name": "test",
                "config": "/x/configs/demo/demo_to-gc6d.yaml",
                "config_hash": "c" * 64,
            }
        )
    )
    assert record(run, configs) == 0
    out = capsys.readouterr().out
    # a model the metafile does not list yet: the whole entry is printed. Its weights
    # are those of the listed model, found by the file they were converted from.
    assert "the weights are those of the metafile's model demo_model" in out
    assert "  - name: demo_to-gc6d\n    config: demo/demo_to-gc6d.yaml\n" in out
    assert "    protocol: graspclutter6d-sv-v1\n    camera: realsense-d435\n" in out
    assert "      url: https://example.org/authors.tar\n" in out
    assert f"      sha256: {SOURCE_SHA}\n" in out and "      source: author\n" in out
    assert out.count("provenance: author-ckpt") == 2
    assert "the whole test split (235 scenes, 3,055 frames)" in out
    for row in ("AP: {test: 25.00}", "AP0.8: {test: 35.00}", "AP0.4: {test: 12.50}"):
        assert row in out  # raw
    for row in ("AP: {test: 30.00}", "AP0.8: {test: 40.00}", "AP0.4: {test: 15.00}"):
        assert row in out  # cd
    assert "paste the entry above under `models:`" in out
    names = sorted(file.name for file in (configs / "demo" / "records").iterdir())
    assert names[1] == "demo_to-gc6d_author-ckpt_test_raw.json"


# ------------------------------------------------------------------------ refusals


def partial(record: dict) -> None:
    record.update(scored=record["requested"] - 1, partial=True, metrics=None, missing=["a frame"])


def one_split_only(record: dict) -> None:
    record["split"] = "test_seen"
    record["per_scene"] = {k: v for k, v in record["per_scene"].items() if int(k[-4:]) < 130}
    record["requested"] = record["scored"] = 30 * 256


@pytest.mark.parametrize(
    ("variants", "change", "message"),
    [
        (("raw",), partial, "raw: incomplete: 23039 of 23040 frames scored, 1 missing"),
        (("raw", "cd"), one_split_only, "split 'test_seen' is not the whole test split ('test')"),
        (
            ("cd",),
            lambda r: r["per_scene"]["realsense/scene_0105"].update(frames=255),
            "cd: 1 scenes without all 256 frames (first: [105])",
        ),
        (("raw", "cd"), lambda r: r["code"].update(dirty=True), "the code had uncommitted changes"),
        (("raw",), lambda r: r.pop("per_scene"), "raw: not a result record of `grasppanda test`"),
        (("cd",), lambda r: r.update(schema=7), "cd: record schema 7, this tool reads 1"),
        (("raw", "cd"), lambda r: r["weights"].pop("sha256"), "holds no sha256 of the weights"),
        (("raw",), lambda r: r["weights"].update(clean_load=False), "did not load cleanly"),
        (("cd",), lambda r: r["weights"].update(sha256="b" * 64), "disagree on weights.sha256"),
        (("cd",), lambda r: r.update(seed=1), "the variants disagree on seed"),
        (
            ("cd",),
            lambda r: r.update(postprocessing=[]),
            "would make one row; choose with --variants",
        ),
        (
            ("raw", "cd"),
            lambda r: r["evaluator"]["config"].update(data_root="/data/GraspNet-1B"),
            "must not hold a path of this machine",
        ),
    ],
)
def test_a_run_that_cannot_back_a_row_is_refused(
    cache, tmp_path, capsys, variants, change, message
):
    run, configs = make_run(cache), write_metafile(tmp_path)
    for variant in variants:
        edit(run, variant, change)
    assert record(run, configs) == 1
    error = capsys.readouterr().err
    assert error.startswith("not recorded: ") and message in error
    assert not (configs / "demo" / "records").exists()  # nothing was written


def test_the_snapshot_files_of_the_run_are_checked_too(cache, tmp_path, capsys):
    run, configs = make_run(cache), write_metafile(tmp_path)
    (run / "git.json").write_text(
        json.dumps({"available": True, "commit": "a" * 40, "dirty": True})
    )
    assert record(run, configs) == 1
    assert "git.json: the run was not started from a clean commit" in capsys.readouterr().err
    (run / "git.json").write_text(
        json.dumps({"available": True, "commit": "f" * 40, "dirty": False})
    )
    assert record(run, configs) == 1
    assert "git.json names commit ffff" in capsys.readouterr().err

    empty = cache / "runs" / "unfinished"
    (empty / "20260101-000000-test" / "predictions").mkdir(parents=True)
    assert record(empty, configs) == 1
    assert (
        "holds no run with result records: the evaluation has not finished"
        in capsys.readouterr().err
    )


def test_the_weights_are_tied_to_the_entry_of_the_metafile(cache, tmp_path, capsys):
    run = make_run(cache)
    # converted from another file than the one the metafile lists
    assert record(run, write_metafile(tmp_path / "a", sha="9" * 64)) == 1
    error = capsys.readouterr().err
    assert (
        f"converted from authors.tar (sha256 {SOURCE_SHA}), the metafile lists {'9' * 64}" in error
    )
    # the weights entry decides the provenance; a contradicting flag is refused
    assert record(run, write_metafile(tmp_path / "b"), "--provenance", "reproduced") == 1
    assert "which makes these rows 'author-ckpt'" in capsys.readouterr().err
    # weights released by the project are the evaluated file itself
    evaluated = sha256_of(cache / "weights" / "converted" / "demo.pth")
    assert record(run, write_metafile(tmp_path / "c", weights="grasppanda")) == 1
    assert "are not the released file of the metafile" in capsys.readouterr().err
    assert record(run, write_metafile(tmp_path / "d", weights="grasppanda", sha=evaluated)) == 0
    out = capsys.readouterr().out
    assert out.count("provenance: reproduced") == 2
    assert "demo_model_reproduced_test_raw.json" in out

    # without a weights entry the tool asks, and prints the entry to add
    configs = write_metafile(tmp_path / "e", weights=None)
    assert record(run, configs) == 1
    assert "has no weights entry for these weights" in capsys.readouterr().err
    assert record(run, configs, "--provenance", "reproduced", "--dry-run") == 0
    out = capsys.readouterr().out
    assert f"      sha256: {evaluated}\n" in out and "      source: grasppanda\n" in out
    assert "would write" in out and not (configs / "demo" / "records").exists()

    # the tie between the authors' file and the evaluated one is in the evaluated file
    gone = make_run(tmp_path / "gone_cache")
    (tmp_path / "gone_cache" / "weights" / "converted" / "demo.pth").unlink()
    configs = write_metafile(tmp_path / "g")
    assert record(gone, configs, "--dry-run") == 1
    assert "cannot be read and compared with the metafile" in capsys.readouterr().err
    assert record(gone, configs, "--dry-run", "--allow-unverified-weights") == 0
    assert "NOT VERIFIED: the evaluated weights file is not at" in capsys.readouterr().out

    # a checkpoint that was not converted from anything cannot be the authors' file
    plain = make_run(tmp_path / "other_cache", converted_from=None)
    assert record(plain, write_metafile(tmp_path / "f")) == 1
    assert "carry no note of tools/convert_checkpoint.py" in capsys.readouterr().err


def test_a_record_file_with_other_content_is_not_overwritten(cache, tmp_path, capsys):
    run, configs = make_run(cache), write_metafile(tmp_path)
    assert record(run, configs) == 0
    edit(run, "raw", lambda r: r.update(created="2026-02-02T00:00:00+00:00"))
    assert record(run, configs) == 1
    assert "exists with other content; pass --force" in capsys.readouterr().err
    assert record(run, configs, "--force") == 0
    stored = configs / "demo" / "records" / "demo_model_author-ckpt_test_raw.json"
    assert json.loads(stored.read_text())["created"].startswith("2026-02-02")


# ------------------------------------------------------------------- its parts


def test_machine_paths_are_named_by_their_variable_longest_first():
    roots = record_result.machine_roots(
        {
            "GRASPPANDA_CACHE": "/srv/cache/",
            "GRASPPANDA_WEIGHTS": "/srv/cache/weights_view",
            "GRASPPANDA_SEED": "7",  # not a path
            "HOME": "/srv",  # not ours
        }
    )
    assert [name for _, name in roots] == ["$GRASPPANDA_WEIGHTS", "$GRASPPANDA_CACHE"]
    value = {
        "a": "/srv/cache/weights_view/x.pth",
        "b": ["/srv/cache/runs/r1", "/srv/cache2/elsewhere"],
        "c": 3,
    }
    clean = record_result.without_machine_paths(value, roots)
    assert clean == {
        "a": "$GRASPPANDA_WEIGHTS/x.pth",
        "b": ["$GRASPPANDA_CACHE/runs/r1", "/srv/cache2/elsewhere"],  # another directory
        "c": 3,
    }
    assert record_result.absolute_paths(clean) == ["b.1: /srv/cache2/elsewhere"]


@pytest.mark.parametrize(
    ("text", "absolute"),
    [
        ("/mnt/disk/someone/weights.pth", True),
        ("read from /data/sets/g1b at night", True),
        ("~/.cache/grasppanda/x", True),
        ("C:/Users/someone/x", True),
        ("$GRASPPANDA_CACHE/graspnet1b/eval/abc", False),
        ("scene_0100/realsense", False),
        ("fraction in [0, 1]; AP0.8 / AP0.4", False),
        ("https://example.org/a/b.tar", False),
        ("records/demo_raw.json", False),
    ],
)
def test_absolute_paths_are_told_from_other_text(text, absolute):
    assert bool(record_result.absolute_paths({"key": text})) is absolute


def test_one_run_is_found_below_a_work_directory_and_two_are_not_guessed(cache):
    run = make_run(cache)
    assert record_result.load_run(run.parent).directory == run
    assert list(record_result.load_run(run).records) == ["raw", "cd"]  # raw first
    assert list(record_result.load_run(run, ["cd"]).records) == ["cd"]
    second = run.parent / "20260102-000000-test" / "results"
    second.mkdir(parents=True)
    (second / "raw.json").write_text("{}")
    with pytest.raises(record_result.RecordError, match="holds 2 runs with result records"):
        record_result.load_run(run.parent)
    with pytest.raises(record_result.RecordError, match="has no record of nms; it has cd, raw"):
        record_result.load_run(run, ["nms"])
