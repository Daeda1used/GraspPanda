"""Grasp tolerance labels: the generator against the definition written out in loops,
the asset store, ``grasppanda prepare --tolerance`` on the synthetic tree, and the data
transforms of the GraspNet-baseline recipe that carry the labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from grasppanda.cli.main import main
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.tolerance import GraspToleranceStore, ToleranceConfig, object_tolerance
from grasppanda.registry import DATASETS, TRANSFORMS
from grasppanda.structures import ContractError

from .test_data_graspnet1b import GRID, PREPROCESS, TinyGraspNet, build, tree  # noqa: F401

REFERENCE = ToleranceConfig(
    max_friction=0.55, min_good_ratio=0.8, radius_step=0.001, max_radius=0.05
)
TINY = {"max_friction": 0.55, "min_good_ratio": 0.6, "radius_step": 0.01, "max_radius": 0.05}
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


def definition(points, scores, config: ToleranceConfig) -> np.ndarray:
    """Tolerance in metres, cell by cell, straight from the definition."""
    radii = [config.radius_step * k for k in range(config.steps + 1)]
    count, cells = len(points), scores[0].size
    flat = scores.reshape(count, cells)
    out = np.zeros((count, cells), dtype=np.float32)
    for point in range(count):
        distance = [float(np.linalg.norm(points[point] - points[other])) for other in range(count)]
        for radius in radii:
            members = [other for other in range(count) if distance[other] <= np.float32(radius)]
            passed = False
            for cell in range(cells):
                good = sum(
                    0 < flat[other, cell] <= np.float32(config.max_friction) for other in members
                )
                if good / len(members) >= config.min_good_ratio:
                    out[point, cell] = radius
                    passed = True
            if not passed:
                break  # the walk ends at the first radius where no cell of the point passes
    return out.reshape(scores.shape)


def random_object(seed: int, count: int = 24, good: float = 0.75):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.03, 0.03, (count, 3)).astype(np.float32)
    points[5] = points[4]  # two annotated points may coincide
    levels = np.array([-1.0, 0.0, 0.1, 0.3, 0.5, 0.6, 1.0], dtype=np.float32)
    weight = np.array([1, 1, good * 6, good * 6, good * 6, 1, 1], dtype=float)
    scores = rng.choice(levels, (count, 3, 2, 2), p=weight / weight.sum())
    return points, scores


# ------------------------------------------------------------------------ generator


@pytest.mark.parametrize(("seed", "ratio"), [(0, 0.8), (1, 0.6), (2, 1.0), (3, 0.5)])
def test_generator_follows_the_definition(seed, ratio):
    config = ToleranceConfig(
        max_friction=0.55, min_good_ratio=ratio, radius_step=0.004, max_radius=0.04
    )
    points, scores = random_object(seed)
    levels = object_tolerance(points, scores, config)
    assert levels.dtype == np.uint8 and levels.shape == scores.shape
    expected = definition(points, scores, config)
    assert np.array_equal(config.radii[levels], expected)
    assert np.count_nonzero(levels) > 0 and levels.max() >= 3
    assert ratio < 0.8 or np.count_nonzero(levels) < levels.size


def test_reference_behaviours_are_kept():
    """A grasp that is itself bad can be tolerant, a cell can pass again after failing,
    and the walk ends where no cell of the point passes."""
    config = ToleranceConfig(
        max_friction=0.55, min_good_ratio=0.6, radius_step=0.01, max_radius=0.04
    )
    # five points on a line, 1 cm apart; four grasp cells per point
    points = np.array([[0.01 * k, 0, 0] for k in range(5)], dtype=np.float32)
    good, bad = 0.2, 0.9
    scores = np.array(
        [
            # cell 0: bad at the point itself, good around it
            # cell 1: good, then diluted below the ratio, then good again further out
            # cell 2: good at the point only
            # cell 3: good everywhere, which keeps the walk of every point going
            [bad, good, good, good],
            [good, bad, bad, good],
            [good, bad, bad, good],
            [good, good, bad, good],
            [good, good, bad, good],
        ],
        dtype=np.float32,
    ).reshape(5, 4, 1, 1)
    levels = object_tolerance(points, scores, config)[0, :, 0, 0]
    # radius 0 → {0}; 0.01 → {0, 1}; 0.02 → {0, 1, 2}; 0.03 → {0..3}; 0.04 → {0..4}
    # cell 0: 0/1, 1/2, 2/3, 3/4, 4/5;  cell 1: 1/1, 1/2, 1/3, 2/4, 3/5;  cell 2: 1/1, then less
    assert levels.tolist() == [4, 4, 0, 4]
    assert np.array_equal(config.radii[levels], definition(points, scores, config)[0, :, 0, 0])
    stopped = scores[:, :3]  # without cell 3 no cell of point 0 passes at 0.01: its walk ends
    assert object_tolerance(points, stopped, config)[0, :, 0, 0].tolist() == [0, 0, 0]
    isolated = scores.copy()
    isolated[0] = bad  # no cell of point 0 passes at radius 0: its walk ends at once
    assert object_tolerance(points, isolated, config)[0].max() == 0
    assert object_tolerance(points, isolated, config)[1].max() > 0


def test_rows_restrict_the_output_not_the_neighbourhood():
    points, scores = random_object(4, count=30)
    config = ToleranceConfig(
        max_friction=0.55, min_good_ratio=0.7, radius_step=0.005, max_radius=0.03
    )
    full = object_tolerance(points, scores, config)
    rows = np.array([17, 3, 3, 29])
    seen = []
    part = object_tolerance(points, scores, config, rows=rows, progress=seen.append)
    assert np.array_equal(part, full[rows]) and seen == [1, 2, 3, 4]
    with pytest.raises(ValueError, match=r"expected points \(P, 3\) and scores"):
        object_tolerance(points[:, :2], scores, config)


def test_config_owns_the_radius_list_and_validates_itself():
    assert REFERENCE.steps == 50 and REFERENCE.problems() == []
    radii = REFERENCE.radii
    assert radii.dtype == np.float32 and len(radii) == 51
    assert radii.tolist() == [np.float32(0.001 * k) for k in range(51)]  # as the reference rounds
    for changes, message in (
        ({"max_friction": 0.0}, "max_friction"),
        ({"min_good_ratio": 1.5}, "min_good_ratio"),
        ({"radius_step": 0.1}, "radius_step"),
        ({"radius_step": 0.0001}, "at most 256"),
    ):
        problems = ToleranceConfig(**{**TINY, **changes}).problems()
        assert len(problems) == 1 and message in problems[0]


# ---------------------------------------------------------------------------- store


def test_store_writes_verifies_and_serves_levels(tmp_path):
    points, scores = random_object(0)
    config = ToleranceConfig(**TINY)
    store = GraspToleranceStore("unit", config, "labelhash", grid=_grid(scores), root=tmp_path)
    assert store.missing([4, 2]) == [2, 4]
    with pytest.raises(AssetError, match="Build it with: prepare"):
        store.require([2], build="prepare")
    store.asset.create()
    record = store.write(2, points, scores)
    assert record["points"] == len(points) and 0 < record["positive"] < 1
    with pytest.raises(AssetError, match="not in the store"):
        store.levels(2)  # written, but not recorded as complete yet
    store.mark_complete(2, record)
    assert store.missing([4, 2]) == [4]
    expected = definition(points, scores, config)
    assert np.array_equal(store.tolerance(2), expected)
    assert np.array_equal(store.tolerance(2, np.array([5, 0])), expected[[5, 0]])
    assert store.levels(2).dtype == np.uint8 and store.tolerance(2).dtype == np.float32
    other = GraspToleranceStore(
        "unit", ToleranceConfig(**{**TINY, "min_good_ratio": 0.9}), "labelhash", root=tmp_path
    )
    relabelled = GraspToleranceStore("unit", config, "otherhash", root=tmp_path)
    assert len({store.asset.spec.hash, other.asset.spec.hash, relabelled.asset.spec.hash}) == 3


def _grid(scores):
    from grasppanda.datasets.labels import LabelGrid

    return LabelGrid(num_view=scores.shape[1], num_angle=scores.shape[2], num_depth=scores.shape[3])


# ------------------------------------------------------------- prepare and dataset


@pytest.fixture
def config(tree, tmp_path):  # noqa: F811
    """A dataset config with a tolerance section for the synthetic tree."""
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "root": str(tree["root"]),
        "preprocess": PREPROCESS,
        "tolerance": TINY,
        "transforms": RECIPE,
    }
    file = tmp_path / "tiny_tolerance.yaml"
    document = {
        "custom_imports": ["tests.unit.test_data_graspnet1b"],
        "data": {"train": {"dataset": node}},
    }
    file.write_text(yaml.safe_dump(document))
    return str(file)


def test_prepare_builds_the_asset_and_the_dataset_serves_it(tree, config, capsys):  # noqa: F811
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--tolerance", "--workers", "1"]) == 1  # labels come first
    assert "grasppanda prepare tinygraspnet --labels" in capsys.readouterr().err
    assert main([*base, "--labels"]) == 0
    with pytest.raises(AssetError, match="grasppanda prepare tinygraspnet --tolerance"):
        DATASETS.build(yaml.safe_load(Path(config).read_text())["data"]["train"]["dataset"])
    assert main([*base, "--check"]) == 1
    assert "tolerance 0/3 objects" in capsys.readouterr().out
    assert main([*base, "--tolerance", "--workers", "1", "--scenes", "0"]) == 0
    assert "2 objects written and verified" in capsys.readouterr().out
    assert main([*base, "--tolerance", "--workers", "1"]) == 0  # resumes: one object left
    out = capsys.readouterr().out
    assert "2 of 3 objects already present" in out and "1 objects written" in out
    assert main([*base, "--tolerance", "--workers", "1"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert main([*base, "--check"]) == 0
    assert "tolerance 3/3 objects" in capsys.readouterr().out

    dataset = build(tree, tolerance=TINY, transforms=RECIPE)
    records = dataset.asset_records()
    assert [record["kind"] for record in records] == ["grasp_labels", "grasp_tolerance"]
    manifest = json.loads((dataset.tolerance.asset.path / "manifest.json").read_text())
    assert (
        manifest["parameters"]["min_good_ratio"] == 0.6 and manifest["uses_ground_truth"] is False
    )
    assert manifest["inputs"]["grasp_labels"] == dataset.label_store.asset.spec.hash
    reference = ToleranceConfig(**TINY)
    for key in dataset.keys[:2]:
        batch = dataset[dataset.keys.index(key)]
        poses = dataset.object_poses(key)
        (objects,) = batch.supervision.objects
        assert len(objects) == len(poses) == 2
        for position, obj in enumerate(objects):
            points, scores, _ = tree["labels"][obj.object_id]
            expected = definition(points, scores, reference)
            collision = tree["collision"][key.scene][position]
            # every label point is kept (point_fraction 1): find each by its coordinates
            rows = [int(np.flatnonzero((points == p).all(axis=1))[0]) for p in obj.points.numpy()]
            assert sorted(rows) == list(range(len(points)))
            served = obj.tolerance.numpy()
            assert served.shape == (len(points), *GRID.shape) and served.dtype == np.float32
            assert np.array_equal(served, np.where(collision[rows], 0, expected[rows]))
            assert (obj.scores.numpy()[collision[rows]] == 0).all()
    plain = build(tree, transforms=RECIPE)  # no tolerance section: none is served or required
    assert plain[0].supervision.objects[0][0].tolerance is None and plain.tolerance is None


def test_tolerance_is_a_label_and_its_parameters_are_checked(tree):  # noqa: F811
    with pytest.raises(ContractError, match="`tolerance` is a label and needs load_labels=True"):
        build(tree, tolerance=TINY, load_labels=False, transforms=RECIPE[:1])
    with pytest.raises(ContractError, match="min_good_ratio"):
        build(tree, tolerance={**TINY, "min_good_ratio": 0.0}, transforms=RECIPE)
    node = {"preprocess": PREPROCESS, "transforms": RECIPE, "tolerance": {"max_friction": 0.55}}
    assert any("tolerance" in problem for problem in TinyGraspNet.check_config(node))
    node["tolerance"] = TINY
    assert list(TinyGraspNet.check_config({**node, "load_labels": False})) == [
        "`tolerance` is a label and needs `load_labels: true`."
    ]


# ----------------------------------------------------------------------- transforms


def test_label_points_can_be_drawn_from_the_visible_side_only(tree):  # noqa: F811
    recipe = [RECIPE[0], RECIPE[1]]
    dataset = build(tree, transforms=recipe, keep_full_points=False)
    step = TRANSFORMS.build(
        {**RECIPE[2], "point_fraction": 0.5, "min_points": 2, "visible_within": 0.012}
    )
    assert step.config()["visible_within"] == 0.012
    sample = dataset.sample(dataset.keys[0])
    for transform in dataset.transforms:
        sample = transform(sample)
    drawn = step.draw(sample)
    kept_fewer = False
    for annotation, rows in zip(sample.objects, drawn):
        points = annotation.source.points
        placed = points @ annotation.pose[:3, :3].T.astype(np.float64) + annotation.pose[:3, 3]
        seen = sample.points[sample.point_labels["instance"] == annotation.instance_id]
        distance = np.linalg.norm(placed[:, None, :] - seen[None, :, :], axis=-1).min(axis=1)
        visible = np.flatnonzero(distance < 0.012)
        assert np.array_equal(step.visible_rows(sample, annotation), visible)
        assert set(rows.tolist()) <= set(visible.tolist()) and len(set(rows.tolist())) == len(rows)
        assert len(rows) == min(max(int(len(visible) * 0.5), 2), len(visible))
        kept_fewer |= len(visible) < len(points)
    assert kept_fewer, (
        "the synthetic objects are meant to have points away from the visible surface"
    )
    nothing = TRANSFORMS.build({**RECIPE[2], "visible_within": 1e-9})
    sample = nothing(sample)  # no annotated point is that close: objects keep zero label points
    assert all(len(a.points) == 0 and a.scores.shape == (0, *GRID.shape) for a in sample.objects)
    with pytest.raises(ValueError, match="visible_within must be positive or null"):
        TRANSFORMS.build({**RECIPE[2], "visible_within": 0.0})


def test_the_default_draw_is_unchanged_and_objects_can_be_excluded(tree):  # noqa: F811
    dataset = build(tree, transforms=RECIPE[:1], keep_full_points=False)
    default = TRANSFORMS.build({**RECIPE[2], "point_fraction": 0.5})
    assert default.visible_within is None
    sample = dataset.sample(dataset.keys[0])
    sample = dataset.transforms[0](sample)
    state = sample.rng.bit_generator.state
    drawn = default.draw(sample)
    sample.rng.bit_generator.state = state
    for annotation, rows in zip(sample.objects, drawn):  # one draw per object over all its points
        count = annotation.source.num_points
        assert np.array_equal(rows, sample.rng.choice(count, default.count(count), replace=False))
    ids = [annotation.object_id for annotation in sample.objects]
    keep_all = TRANSFORMS.build({"type": "FilterObjects", "min_visible_points": 0})
    assert keep_all.exclude_object_ids == [] and len(keep_all(sample).objects) == len(ids)
    exclude = TRANSFORMS.build(
        {"type": "FilterObjects", "min_visible_points": 0, "exclude_object_ids": [ids[0], 99]}
    )
    assert [a.object_id for a in exclude(sample).objects] == ids[1:]
    assert sample.point_labels["instance"].max() > 0  # the points of the object stay labelled
