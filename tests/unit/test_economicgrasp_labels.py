"""Compact grasp labels: the definition against a cell-by-cell oracle, the reference
file layout, the asset store, the dataset that serves them and ``grasppanda prepare``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from grasppanda.cli.main import main
from grasppanda.datasets.assets import Asset, AssetError
from grasppanda.datasets.compact_labels import (
    VIEW_ARRAYS,
    CompactLabelConfig,
    CompactLabelStore,
    collapse_labels,
    from_reference_arrays,
    generate_compact_labels,
    scene_shard,
)
from grasppanda.datasets.labels import StoredGraspLabels
from grasppanda.structures import (
    COMPACT_LABELS,
    CompactGraspLabels,
    ContractError,
    Supervision,
    collate,
)

from .test_data_graspnet1b import (  # noqa: F401
    GRID,
    POINTS,
    PREPROCESS,
    SCENES,
    TinyGraspNet,
    build,
    tree,
)

CONFIG = {"point_friction_threshold": 0.4, "view_friction_threshold": 0.6, "max_width": 0.08}
#: The EconomicGrasp data recipe on the synthetic tree.
TRANSFORMS = [
    {"type": "RandomSample", "num_points": 64},
    {"type": "RandomFlip", "axis": "x", "prob": 0.5},
    {"type": "ConstantFeatures", "channels": 3},
]


def oracle(scores, widths, collision, config: CompactLabelConfig):
    """The definition, one annotated point and one view at a time, in single precision."""
    rows, graspness, angle, depth, score, width = [], [], [], [], [], []
    for point in range(len(scores)):
        friction = scores[point].astype(np.float32).copy()
        friction[collision[point]] = 0  # colliding and invalid grasps are unusable
        friction[friction < 0] = 0
        usable = friction > 0
        if not (usable & (friction <= np.float32(config.point_friction_threshold))).any():
            continue
        rows.append(point)
        good = usable & (friction <= np.float32(config.view_friction_threshold))
        fraction = good.reshape(len(good), -1).sum(axis=1).astype(np.float32) / np.float32(
            GRID.num_angle * GRID.num_depth
        )
        span = fraction.max() - fraction.min() + np.float32(1e-5)
        graspness.append((fraction - fraction.min()) / span)
        per_view = {name: [] for name in ("angle", "depth", "score", "width")}
        for view in range(GRID.num_view):
            # Ties: per angle the first best depth, then the first angle that reaches the
            # best quality of the view.
            per_angle = []
            for a in range(GRID.num_angle):
                qualities = [
                    np.float32(1.1) - friction[view, a, d]
                    if usable[view, a, d]
                    and widths[point, view, a, d] <= np.float32(config.max_width)
                    else np.float32(0.0)
                    for d in range(GRID.num_depth)
                ]
                top = max(qualities)
                per_angle.append((top, qualities.index(top)))
            top = max(q for q, _ in per_angle)
            a = [q for q, _ in per_angle].index(top)
            d = per_angle[a][1]
            per_view["angle"].append(a)
            per_view["depth"].append(d)
            per_view["score"].append(int(top * np.float32(10)))
            per_view["width"].append(int(widths[point, view, a, d] * np.float32(1000)))
        angle.append(per_view["angle"])
        depth.append(per_view["depth"])
        score.append(per_view["score"])
        width.append(per_view["width"])
    return {
        "rows": np.array(rows, dtype=np.int64),
        "view_graspness": np.array(graspness, dtype=np.float32).reshape(-1, GRID.num_view),
        "angle": np.array(angle, dtype=np.uint8).reshape(-1, GRID.num_view),
        "depth": np.array(depth, dtype=np.uint8).reshape(-1, GRID.num_view),
        "score": np.array(score, dtype=np.uint8).reshape(-1, GRID.num_view),
        "width": np.array(width, dtype=np.uint8).reshape(-1, GRID.num_view),
    }


def compact(tree, **overrides) -> TinyGraspNet:  # noqa: F811
    return build(tree, **{"compact_labels": CONFIG, "transforms": TRANSFORMS, **overrides})


# -------------------------------------------------------------------------- definition


def test_collapse_matches_the_definition_cell_by_cell(tree):  # noqa: F811
    dataset = build(tree)
    config = CompactLabelConfig(**CONFIG)
    kept = dropped = empty_views = 0
    for scene, object_ids in SCENES.items():
        for index, (object_id, labels) in enumerate(zip(object_ids, dataset.scene_labels(scene))):
            points, scores, widths = tree["labels"][object_id]
            expected = oracle(scores, widths, tree["collision"][scene][index], config)
            mine = collapse_labels(labels, config)
            assert set(mine) == {"rows", "points", *VIEW_ARRAYS}
            for name, values in expected.items():
                assert mine[name].dtype == values.dtype, name
                assert np.array_equal(mine[name], values), (scene, object_id, name)
            assert np.array_equal(mine["points"], points[expected["rows"]])
            kept += len(expected["rows"])
            dropped += POINTS[object_id] - len(expected["rows"])
            empty_views += int((mine["score"] == 0).sum())
            # A view without a usable grasp keeps cell (0, 0) with quality 0.
            assert (mine["angle"][mine["score"] == 0] == 0).all()
            assert (mine["depth"][mine["score"] == 0] == 0).all()
            assert mine["score"].max() <= 10 and mine["view_graspness"].min() >= 0
            assert mine["view_graspness"].max() < 1  # the epsilon keeps the maximum below 1
    assert kept > 20 and empty_views > 0  # the synthetic labels exercise both branches
    strict = CompactLabelConfig(0.05, 0.6, 0.08)  # no grasp is that good: every point is dropped
    nothing = collapse_labels(dataset.scene_labels(0)[0], strict)
    assert len(nothing["rows"]) == 0 and nothing["angle"].shape == (0, GRID.num_view)
    assert nothing["view_graspness"].dtype == np.float32 and nothing["points"].shape == (0, 3)
    assert dropped >= 0


def handmade(scores: np.ndarray, widths: np.ndarray) -> StoredGraspLabels:
    """Labels of one annotated point without collisions, from ``(V, A, D)`` grids."""
    packed = np.packbits(np.zeros((1, GRID.size), dtype=bool), axis=1)
    points = np.zeros((1, 3), dtype=np.float32)
    return StoredGraspLabels(points, scores[None], widths[None], packed, GRID)


def test_thresholds_are_compared_in_the_precision_of_the_labels():
    """A friction of exactly 0.4, a float32 like every label, passes a threshold of 0.4."""
    assert float(np.float32(0.4)) > 0.4  # compared in double precision it would not
    scores = np.full(GRID.shape, -1.0, dtype=np.float32)
    widths = np.full(GRID.shape, 0.05, dtype=np.float32)
    scores[2, 1, 0] = 0.4  # the only grasp good enough to keep the point
    scores[3, 2, 1] = 0.6  # counts towards the view graspness, exactly at its threshold
    scores[4, 0, 1] = 0.7  # counts towards nothing, but is the kept grasp of its view
    widths[4, 0, 1] = 0.08  # exactly as wide as allowed
    labels = handmade(scores, widths)
    kept = collapse_labels(labels, CompactLabelConfig(0.4, 0.6, 0.08))
    assert kept["rows"].tolist() == [0]
    assert kept["view_graspness"][0].tolist() == pytest.approx([0, 0, 1, 1, 0], abs=1e-4)
    assert kept["score"][0].tolist() == [0, 0, 7, 5, 4]  # (1.1 - friction) in tenths
    assert kept["angle"][0].tolist() == [0, 0, 1, 2, 0]
    assert kept["depth"][0].tolist() == [0, 0, 0, 1, 1]
    assert kept["width"][0].tolist() == [50, 50, 50, 50, 80]  # empty views keep cell (0, 0)
    assert len(collapse_labels(labels, CompactLabelConfig(0.39, 0.6, 0.08))["rows"]) == 0
    lower = collapse_labels(labels, CompactLabelConfig(0.4, 0.59, 0.08))
    assert lower["view_graspness"][0].tolist() == pytest.approx([0, 0, 1, 0, 0], abs=1e-4)
    narrow = collapse_labels(labels, CompactLabelConfig(0.4, 0.6, 0.079))
    assert narrow["score"][0].tolist() == [0, 0, 7, 5, 0]  # too wide: unusable for its view


def test_equal_cells_resolve_to_the_first_depth_and_then_the_first_angle():
    scores = np.full(GRID.shape, -1.0, dtype=np.float32)
    widths = np.linspace(0.01, 0.07, GRID.size, dtype=np.float32).reshape(GRID.shape)
    scores[0] = 0.3  # every cell of view 0 is equally good
    scores[1, 2, 1] = scores[1, 1, 1] = 0.2  # two angles tie, both at the second depth
    scores[2, 2, 0] = 0.5
    scores[2, 0, 1] = 0.2  # the better grasp wins wherever it is
    kept = collapse_labels(handmade(scores, widths), CompactLabelConfig(0.4, 0.6, 0.08))
    assert kept["angle"][0, :3].tolist() == [0, 1, 0]
    assert kept["depth"][0, :3].tolist() == [0, 1, 1]
    assert kept["score"][0, :3].tolist() == [8, 9, 9]
    expected = [widths[0, 0, 0], widths[1, 1, 1], widths[2, 0, 1]]
    assert kept["width"][0, :3].tolist() == [int(w * np.float32(1000)) for w in expected]


def test_widths_beyond_the_storage_range_are_refused(tree):  # noqa: F811
    dataset = build(tree)
    labels = dataset.scene_labels(0)[0]
    labels.widths = np.asarray(labels.widths) + np.float32(0.3)
    with pytest.raises(AssetError, match="cannot exceed 0.255 m"):
        collapse_labels(labels, CompactLabelConfig(0.4, 0.6, 1.0))


def test_reference_files_are_read_by_undoing_their_view_permutation():
    rng = np.random.default_rng(0)
    count, views = 6, GRID.num_view
    canonical = {
        "angle": rng.integers(0, 3, (count, views)).astype(np.uint8),
        "depth": rng.integers(0, 2, (count, views)).astype(np.uint8),
        "score": rng.integers(0, 11, (count, views)).astype(np.uint8),
        "width": rng.integers(0, 150, (count, views)).astype(np.uint8),
    }
    order = np.stack([rng.permutation(views) for _ in range(count)]).astype(np.uint16)
    archive = {
        "points": rng.random((count, 3)).astype(np.float32),
        "pointid": np.array([0, 0, 1, 1, 1, 2], dtype=np.uint8),
        "vgraspness": rng.random((count, views)).astype(np.float32),  # stored unpermuted
        "topview": order,
        "rotations": np.take_along_axis(canonical["angle"], order.astype(np.int64), axis=1),
        "depth": np.take_along_axis(canonical["depth"], order.astype(np.int64), axis=1),
        "scores": np.take_along_axis(canonical["score"], order.astype(np.int64), axis=1),
        "widths": np.take_along_axis(canonical["width"], order.astype(np.int64), axis=1),
    }
    restored = from_reference_arrays(archive)
    for name, values in canonical.items():
        assert np.array_equal(restored[name], values) and restored[name].dtype == np.uint8
    assert np.array_equal(restored["view_graspness"], archive["vgraspness"])
    assert restored["instance"].tolist() == [0, 0, 1, 1, 1, 2]
    archive["topview"] = order[:, :3]
    with pytest.raises(AssetError, match="keeps a subset of the views"):
        from_reference_arrays(
            {k: v[:, :3] if v.ndim == 2 and k != "points" else v for k, v in archive.items()}
        )


# ------------------------------------------------------------------------------- store


def test_store_writes_verifies_reads_and_refuses_what_is_missing(tree, tmp_path):  # noqa: F811
    dataset = build(tree)
    config = CompactLabelConfig(**CONFIG)
    spec = CompactLabelStore.spec("tinygraspnet", config, GRID, dataset.label_store.asset.spec.hash)
    store = CompactLabelStore(Asset(spec, tmp_path / "assets"), config)
    assert not store.asset.exists()
    store.asset.create()
    with pytest.raises(AssetError, match="scene 0 are not in the asset .* has not been generated"):
        store.read_scene(0)
    record = generate_compact_labels(dataset, store, 0)
    assert not store.asset.is_complete(scene_shard(0))  # written, not yet marked
    store.mark_complete(0, record)
    assert record["objects"] == SCENES[0]
    assert record["annotated"] == [POINTS[i] for i in SCENES[0]]
    arrays = store.read_scene(0)
    assert len(arrays["points"]) == sum(record["kept"]) and arrays["instance"].dtype == np.int16
    assert np.array_equal(np.bincount(arrays["instance"], minlength=2), record["kept"])
    for index, labels in enumerate(dataset.scene_labels(0)):
        expected = collapse_labels(labels, config)
        rows = arrays["instance"] == index
        for name in ("points", *VIEW_ARRAYS):
            assert np.array_equal(arrays[name][rows], expected[name])
    manifest = json.loads((store.asset.path / "manifest.json").read_text())
    assert manifest["kind"] == "compact_grasp_labels" and manifest["preprocessing"] is None
    assert manifest["parameters"]["point_friction_threshold"] == 0.4
    assert manifest["inputs"]["grasp_labels"] == dataset.label_store.asset.spec.hash
    assert manifest["shards"][scene_shard(0)]["bytes"] == store.shard_file(0).stat().st_size
    assert not list(store.asset.path.glob(".*tmp"))  # atomic writes leave nothing behind

    labels = store.labels(0, 1, SCENES[0][1], np.eye(4))
    assert isinstance(labels, CompactGraspLabels) and labels.view_max_friction == 0.6
    assert labels.angle.dtype == torch.uint8 and labels.pose.dtype == torch.float32
    with pytest.raises(AssetError, match="instance 1 is object 99 in the frame"):
        store.labels(0, 1, 99, np.eye(4))
    with pytest.raises(AssetError, match="expected arrays"):
        store.write_scene(1, {"points": arrays["points"]})
    with pytest.raises(AssetError, match="'angle' has 1 rows for"):
        store.write_scene(1, {**arrays, "angle": arrays["angle"][:1]})
    with pytest.raises(AssetError, match="cannot mark scene 1 complete"):
        store.mark_complete(1, record)


def test_asset_identity_follows_the_parameters_and_the_dense_labels():
    base = CompactLabelStore.spec("tinygraspnet", CompactLabelConfig(**CONFIG), GRID, "abc")
    same = CompactLabelStore.spec("tinygraspnet", CompactLabelConfig(**CONFIG), GRID, "abc")
    assert base.hash == same.hash and base.preprocessing is None and not base.uses_ground_truth
    for change in (
        {"point_friction_threshold": 0.5},
        {"view_friction_threshold": 0.8},
        {"max_width": 0.1},
    ):
        other = CompactLabelStore.spec(
            "tinygraspnet", CompactLabelConfig(**{**CONFIG, **change}), GRID, "abc"
        )
        assert other.hash != base.hash, change
    moved = CompactLabelStore.spec("tinygraspnet", CompactLabelConfig(**CONFIG), GRID, "other")
    assert moved.hash != base.hash
    assert CompactLabelConfig(0.4, -1.0, 0.1).problems() == [
        "`compact_labels.view_friction_threshold` must be a positive number, got -1.0."
    ]


# ----------------------------------------------------------------------------- dataset


def prepared(tree) -> TinyGraspNet:  # noqa: F811
    """A dataset with compact labels whose asset has been generated for every scene."""
    probe = compact(tree, check_assets=False)
    probe.compact_labels.asset.create()
    for scene in probe.manifest.all_scenes():
        probe.compact_labels.mark_complete(
            scene, generate_compact_labels(probe, probe.compact_labels, scene)
        )
    return compact(tree)


def test_the_dataset_serves_compact_labels_in_place_of_dense_grids(tree):  # noqa: F811
    dataset = prepared(tree)
    batch = dataset[4]
    supervision = batch.supervision
    assert supervision.objects == [[]]
    assert set(supervision.point_labels) == {"objectness", "instance"}
    (objects,) = supervision.extras[COMPACT_LABELS]
    key = dataset.keys[4]
    assert [labels.object_id for labels in objects] == SCENES[key.scene]
    stored = dataset.compact_labels.read_scene(key.scene)
    truth = tree["frames"][key.scene, key.camera, key.frame]["poses"]
    for index, labels in enumerate(objects):
        rows = stored["instance"] == index
        assert np.array_equal(labels.points.numpy(), stored["points"][rows])
        for name in VIEW_ARRAYS:
            assert np.array_equal(getattr(labels, name).numpy(), stored[name][rows])
        # The pose is the frame's, mirrored exactly when the flip augmentation fired.
        pose = labels.pose[:3].numpy()
        flipped = truth[:, :, index] * np.array([[-1.0], [1.0], [1.0]], dtype=np.float32)
        assert np.array_equal(pose, truth[:, :, index]) or np.array_equal(pose, flipped)
    records = dataset.asset_records()
    assert [record["kind"] for record in records] == ["compact_grasp_labels"]

    merged = collate([dataset[0], dataset[4]])
    assert len(merged.supervision.extras[COMPACT_LABELS]) == 2
    moved = merged.supervision.to("cpu")
    assert isinstance(moved.extras[COMPACT_LABELS][1][0], CompactGraspLabels)
    with pytest.raises(ContractError, match="different label sets"):
        collate([dataset[0], build(tree)[0]])


def test_poses_follow_the_flip_like_the_points(tree):  # noqa: F811
    dataset = prepared(tree)
    flip = dataset.transforms[1]
    flip.prob = 0.0
    plain = dataset[2]
    flip.prob = 1.0
    mirrored = dataset[2]
    for before, after in zip(
        plain.supervision.extras[COMPACT_LABELS][0], mirrored.supervision.extras[COMPACT_LABELS][0]
    ):
        assert torch.equal(after.pose[0], -before.pose[0])
        assert torch.equal(after.pose[1:], before.pose[1:])
        assert torch.equal(after.angle, before.angle)  # labels live in the object frame
    assert torch.equal(
        mirrored.observation.points.coords[:, 0], -plain.observation.points.coords[:, 0]
    )


def test_missing_scenes_and_contradicting_settings_are_reported(tree):  # noqa: F811
    with pytest.raises(AssetError, match=r"(?s)No compact_grasp_labels asset .*--compact-labels"):
        compact(tree)
    probe = compact(tree, check_assets=False)
    probe.compact_labels.asset.create()
    probe.compact_labels.mark_complete(0, generate_compact_labels(probe, probe.compact_labels, 0))
    with pytest.raises(AssetError, match=r"(?s)lacks 1 of the 2 shards .*prepare tinygraspnet"):
        compact(tree)
    with pytest.raises(ContractError, match="`compact_labels` is a label and needs"):
        build(tree, compact_labels=CONFIG, transforms=TRANSFORMS, load_labels=False)
    with pytest.raises(ContractError, match="reads the dense label grids"):
        build(tree, compact_labels=CONFIG, check_assets=False)  # TRAIN samples dense labels
    with pytest.raises(ContractError, match="max_width` must be a positive number"):
        compact(tree, compact_labels={**CONFIG, "max_width": 0.0}, check_assets=False)
    node = {"compact_labels": CONFIG, "preprocess": PREPROCESS, "transforms": TRANSFORMS}
    assert list(TinyGraspNet.check_config(node)) == []
    assert "needs `load_labels: true`" in " ".join(
        TinyGraspNet.check_config({**node, "load_labels": False})
    )
    dense = [
        *TRANSFORMS,
        {
            "type": "SampleGraspLabels",
            "point_fraction": 1.0,
            "min_points": 1,
            "collision": "ignore",
        },
    ]
    assert "transforms[3] (SampleGraspLabels) reads the dense label grids" in " ".join(
        TinyGraspNet.check_config({**node, "transforms": dense})
    )


# ----------------------------------------------------------------------------- prepare


def test_prepare_generates_the_asset_a_config_asks_for(tree, tmp_path, capsys):  # noqa: F811
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "root": str(tree["root"]),
        "preprocess": PREPROCESS,
        "compact_labels": CONFIG,
        "transforms": TRANSFORMS,
    }
    file = tmp_path / "tiny.yaml"
    document = {
        "custom_imports": ["tests.unit.test_data_graspnet1b"],
        "data": {"train": {"dataset": node}},
    }
    file.write_text(yaml.safe_dump(document))
    base = ["prepare", "tinygraspnet", "--config", str(file)]

    assert main([*base, "--compact-labels"]) == 1  # the dense labels come first
    assert "grasppanda prepare tinygraspnet --labels" in capsys.readouterr().err
    assert main([*base, "--labels"]) == 0
    capsys.readouterr()
    assert main([*base, "--check", "--camera", "realsense"]) == 1
    assert "compact grasp labels 0/2 scenes" in capsys.readouterr().out

    assert main([*base, "--compact-labels", "--scenes", "0"]) == 0
    out = capsys.readouterr().out
    assert "0 of 1 scenes already present" in out and "1 scenes written and read back" in out
    assert main([*base, "--compact-labels"]) == 0  # resumes: scene 1 only
    out = capsys.readouterr().out
    assert "1 of 2 scenes already present" in out and "1 scenes written and read back" in out
    assert main([*base, "--compact-labels"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert main([*base, "--check", "--camera", "realsense"]) == 0
    assert "compact grasp labels 2/2 scenes" in capsys.readouterr().out

    dataset = compact(tree)  # what the config describes can now be read
    assert COMPACT_LABELS in dataset[0].supervision.extras

    plain = dict(node, graspness=None)
    del plain["compact_labels"]
    document["data"]["train"]["dataset"] = plain
    file.write_text(yaml.safe_dump(document))
    with pytest.raises(SystemExit, match="no `compact_labels` section"):
        main([*base, "--compact-labels"])


# -------------------------------------------------------------------------- structures


def test_compact_grasp_labels_check_their_shapes_and_convert_units():
    def make(**overrides):
        fields = {
            "object_id": 3,
            "pose": torch.eye(4),
            "points": torch.zeros(5, 3),
            "view_graspness": torch.zeros(5, 7),
            "angle": torch.zeros(5, 7, dtype=torch.uint8),
            "depth": torch.zeros(5, 7, dtype=torch.uint8),
            "score": torch.full((5, 7), 3, dtype=torch.uint8),
            "width": torch.full((5, 7), 87, dtype=torch.uint8),
            "view_max_friction": 0.6,
        }
        return CompactGraspLabels(**{**fields, **overrides})

    labels = make()
    assert labels.num_views == 7
    assert labels.quality(labels.score).unique().item() == np.float32(3) / np.float32(10)
    assert labels.metres(labels.width).unique().item() == np.float32(87) / np.float32(1000)
    assert labels.quality(labels.score).dtype == torch.float32
    moved = labels.to("cpu")
    assert moved is not labels and moved.object_id == 3 and torch.equal(moved.score, labels.score)
    supervision = Supervision(extras={COMPACT_LABELS: [[labels]]}).to("cpu")
    assert supervision.extras[COMPACT_LABELS][0][0].view_max_friction == 0.6
    with pytest.raises(ValueError, match=r"view_graspness must be \(P, V\) with P = 5"):
        make(view_graspness=torch.zeros(4, 7))
    with pytest.raises(ValueError, match="depth must be an integer tensor of shape"):
        make(depth=torch.zeros(5, 6, dtype=torch.uint8))
    with pytest.raises(ValueError, match="score must be an integer tensor"):
        make(score=torch.zeros(5, 7))
