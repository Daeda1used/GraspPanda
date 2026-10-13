"""The scale statistic as a derived asset: the store, the safe reader of the reference's
file format, ``grasppanda prepare --scale-statistic`` on the synthetic tree, and the
dataset that serves the histogram with every training sample."""

from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pytest
import torch
import yaml

from grasppanda.cli.main import main
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.scale_statistic import (
    ScaleStatisticConfig,
    ScaleStatisticStore,
    read_reference_statistic,
)
from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY, collect_seed_scales
from grasppanda.structures import ContractError, collate

from .test_data_graspnet1b import PREPROCESS, TinyGraspNet, build, tree  # noqa: F401

RECIPE = [
    {"type": "RandomSample", "num_points": 64},
    {"type": "FilterObjects", "min_visible_points": 2},
    {
        "type": "SampleGraspLabels",
        "point_fraction": 1.0,
        "min_points": 4,
        "collision": "zero_scores",
    },
]
NATIVE = {
    "source": "native",
    "bins": 3,
    "frame_step": 2,
    "num_seeds": 16,
    "min_graspable_views": 0,
    "graspable_score": 0.1,
    "max_width": 0.1,
    "seed": 0,
}
EDGES = np.linspace(0.003, 0.1, 5)
COUNTS = np.array([4, 9, 30, 12])


def reference_file(path, counts=COUNTS, edges=EDGES):
    """A file in the reference's format: a pickled mapping saved with numpy.save."""
    np.save(path, {"num": counts, "interval": edges})
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------- store


def test_store_keeps_a_histogram_and_its_hash_covers_what_decides_it(tmp_path):
    config = ScaleStatisticConfig(**NATIVE)
    inputs = {"dataset": "unit", "grasp_labels": "abc", "split": "train"}
    store = ScaleStatisticStore("unit", config, inputs, root=tmp_path)
    assert not store.exists()
    with pytest.raises(AssetError, match="Build it with: prepare"):
        store.require(build="prepare")
    with pytest.raises(AssetError, match="has not been built"):
        store.load()
    edges, counts = np.linspace(0.01, 0.1, 4), np.array([5, 2, 9])
    store.save(edges, counts, frames=7)
    assert store.exists()
    loaded = ScaleStatisticStore("unit", config, inputs, root=tmp_path).load()
    assert np.array_equal(loaded[0], edges) and np.array_equal(loaded[1], counts)
    manifest = json.loads((store.asset.path / "manifest.json").read_text())
    assert manifest["provenance"] == "native" and manifest["uses_ground_truth"] is True
    assert manifest["shards"]["histogram"]["total"] == 16
    assert manifest["shards"]["histogram"]["frames"] == 7
    others = [
        ScaleStatisticStore("unit", ScaleStatisticConfig(**{**NATIVE, "frame_step": 4}), inputs),
        ScaleStatisticStore("unit", config, {**inputs, "split": "all"}),
        ScaleStatisticStore("unit", config, {**inputs, "grasp_labels": "xyz"}),
    ]
    assert len({store.asset.spec.hash, *(other.asset.spec.hash for other in others)}) == 4
    for bad_edges, bad_counts in (
        (edges, np.array([5, 0, 9])),
        (edges[::-1], counts),
        (edges, counts[:2]),
    ):
        with pytest.raises(AssetError, match="K \\+ 1 increasing edges and K positive counts"):
            store.save(bad_edges, bad_counts)
    with pytest.raises(AssetError, match="has 2 bins, but the configuration asks for 3"):
        store.save(edges[:3], counts[:2])


def test_config_validates_itself():
    assert ScaleStatisticConfig(**NATIVE).problems() == []
    legacy = ScaleStatisticConfig(source="legacy_import", reference_sha256="a" * 64)
    assert legacy.problems() == [] and legacy.parameters() == {
        "bins": 32,
        "reference_sha256": "a" * 64,
    }
    assert "frame_step" in ScaleStatisticConfig(**NATIVE).parameters()
    for changes, message in (
        ({"source": "guessed"}, "native or legacy_import"),
        ({"bins": 0}, "at least 1"),
        ({"source": "legacy_import"}, "reference_sha256"),
        ({"reference_sha256": "a" * 64}, "belongs to source: legacy_import"),
    ):
        problems = ScaleStatisticConfig(**{**NATIVE, **changes}).problems()
        assert len(problems) == 1 and message in problems[0], changes


# --------------------------------------------------------------- the reference file


def test_reference_format_is_read_without_running_pickled_code(tmp_path, capsys):
    path, _ = reference_file(tmp_path / "scale_distribution.npy")
    edges, counts = read_reference_statistic(path)
    assert np.array_equal(edges, EDGES) and np.array_equal(counts, COUNTS)
    assert edges.dtype == np.float64 and counts.dtype == np.int64

    class Evil:
        def __reduce__(self):
            return (print, ("this must never run",))

    array = np.empty((), dtype=object)
    array[()] = {"num": Evil(), "interval": EDGES}
    evil = tmp_path / "evil.npy"
    with evil.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, {"descr": "|O", "fortran_order": False, "shape": ()}
        )
        pickle.dump(array, handle, protocol=2)
    with pytest.raises(AssetError, match=r"refusing to load (__builtin__|builtins)\.print"):
        read_reference_statistic(evil)
    assert "this must never run" not in capsys.readouterr().out
    plain = tmp_path / "plain.npy"
    np.save(plain, np.arange(4.0))
    with pytest.raises(AssetError, match="not a scale statistic in the reference's format"):
        read_reference_statistic(plain)
    path, _ = reference_file(tmp_path / "empty_bin.npy", counts=np.array([4, 0, 30, 12]))
    with pytest.raises(AssetError, match="K positive counts"):
        read_reference_statistic(path)


def test_legacy_import_takes_only_the_file_the_config_names(tmp_path):
    path, digest = reference_file(tmp_path / "scale_distribution.npy")
    config = ScaleStatisticConfig(source="legacy_import", bins=4, reference_sha256=digest)
    store = ScaleStatisticStore("unit", config, {"dataset": "unit"}, root=tmp_path / "cache")
    store.import_reference(path)
    edges, counts = store.load()
    assert np.array_equal(edges, EDGES) and np.array_equal(counts, COUNTS)
    manifest = json.loads((store.asset.path / "manifest.json").read_text())
    assert manifest["provenance"] == "legacy-import"
    assert manifest["parameters"]["reference_sha256"] == digest
    other, _ = reference_file(tmp_path / "other.npy", counts=COUNTS + 1)
    with pytest.raises(AssetError, match="but the configuration names the reference file"):
        store.import_reference(other)
    native = ScaleStatisticStore("unit", ScaleStatisticConfig(**NATIVE), {}, root=tmp_path / "c")
    with pytest.raises(AssetError, match="only a legacy_import statistic can be imported"):
        native.import_reference(path)


# ------------------------------------------------------------- prepare and dataset


def write_config(tree, tmp_path, statistic):  # noqa: F811
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "root": str(tree["root"]),
        "preprocess": PREPROCESS,
        "scale_statistic": statistic,
        "transforms": RECIPE,
    }
    file = tmp_path / "tiny_scale.yaml"
    document = {
        "custom_imports": ["tests.unit.test_data_graspnet1b"],
        "data": {"train": {"dataset": node}},
    }
    file.write_text(yaml.safe_dump(document))
    return str(file)


def test_prepare_generates_the_native_statistic_and_the_dataset_serves_it(tree, tmp_path, capsys):  # noqa: F811
    config = write_config(tree, tmp_path, NATIVE)
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--scale-statistic"]) == 1  # labels come first
    assert "grasppanda prepare tinygraspnet --labels" in capsys.readouterr().err
    assert main([*base, "--labels"]) == 0
    with pytest.raises(AssetError, match="grasppanda prepare tinygraspnet --scale-statistic"):
        build(tree, scale_statistic=NATIVE, transforms=RECIPE)
    assert main([*base, "--scale-statistic"]) == 0
    out = capsys.readouterr().out
    assert "3 frames of train (every 2th)" in out and "graspable seeds of 3 frames in 3 bins" in out
    assert main([*base, "--scale-statistic"]) == 0
    assert "nothing to do" in capsys.readouterr().out

    dataset = build(tree, scale_statistic=NATIVE, transforms=RECIPE)
    assert dataset.asset_records()[-1]["kind"] == "scale_statistic"
    edges, counts = dataset.scale_statistic.load()
    # the generator, called directly on the same configuration, gives the same histogram
    probe = build(tree, transforms=RECIPE, seed=NATIVE["seed"])
    arguments = {
        k: NATIVE[k]
        for k in ("frame_step", "num_seeds", "min_graspable_views", "graspable_score", "max_width")
    }
    seen = []
    widths = collect_seed_scales(probe, **arguments, progress=seen.append).numpy()
    expected = np.histogram(widths, bins=3)
    assert np.array_equal(counts, expected[0]) and np.array_equal(edges, expected[1])
    assert seen == [1, 2, 3] and counts.sum() == len(widths) > 0
    assert (widths <= 0.1).all() and (widths > 0).all()

    first, second = dataset[0], dataset[1]
    (histogram,) = first.supervision.extras[HISTOGRAM_KEY]
    assert torch.equal(histogram["edges"], torch.from_numpy(edges))
    assert torch.equal(histogram["counts"], torch.from_numpy(counts))
    merged = collate([first, second])
    assert len(merged.supervision.extras[HISTOGRAM_KEY]) == 2  # one entry per frame
    plain = build(tree, transforms=RECIPE)
    assert plain[0].supervision.extras == {} and plain.scale_statistic is None


def test_prepare_imports_the_reference_file(tree, tmp_path, capsys):  # noqa: F811
    path, digest = reference_file(tmp_path / "scale_distribution.npy")
    statistic = {"source": "legacy_import", "bins": 4, "reference_sha256": digest}
    config = write_config(tree, tmp_path, statistic)
    base = ["prepare", "tinygraspnet", "--config", config, "--scale-statistic", "--import-legacy"]
    assert main(["prepare", "tinygraspnet", "--config", config, "--labels"]) == 0
    with pytest.raises(SystemExit, match="needs --scale-statistic-file"):
        main(base)
    assert main([*base, "--scale-statistic-file", str(path)]) == 0
    assert "imported 4 bins, 55 entries" in capsys.readouterr().out
    dataset = build(tree, scale_statistic=statistic, transforms=RECIPE)
    record = dataset.asset_records()[-1]
    assert record["provenance"] == "legacy-import"
    (histogram,) = dataset[0].supervision.extras[HISTOGRAM_KEY]
    assert histogram["counts"].tolist() == COUNTS.tolist()


def test_the_statistic_is_supervision_and_its_parameters_are_checked(tree):  # noqa: F811
    with pytest.raises(ContractError, match="`scale_statistic` is supervision"):
        build(tree, scale_statistic=NATIVE, load_labels=False, transforms=RECIPE[:1])
    with pytest.raises(ContractError, match="reference_sha256"):
        build(tree, scale_statistic={"source": "legacy_import"}, transforms=RECIPE)
    node = {"preprocess": PREPROCESS, "transforms": RECIPE, "scale_statistic": {"bins": 4}}
    assert any("scale_statistic" in problem for problem in TinyGraspNet.check_config(node))
