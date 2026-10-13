"""The memory-mapped grasp-label store."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.labels import GRASPNET_LABEL_GRID, GraspLabelStore, LabelGrid

from .test_data_graspnet1b import GRID, POINTS, PREPROCESS, SCENES, TinyGraspNet, make_tree


def unbuilt(root, **overrides) -> TinyGraspNet:
    return TinyGraspNet(
        "realsense", "all", PREPROCESS, [], root=str(root), check_assets=False, **overrides
    )


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    truth = make_tree(tmp_path / "data")
    truth["root"] = tmp_path / "data"
    return truth


def test_label_grid_owns_the_angle_and_depth_bins():
    grid = GRASPNET_LABEL_GRID
    assert grid.shape == (300, 12, 4) and grid.size == 14400
    assert grid.angles == tuple(a * np.pi / 12 for a in range(12))
    assert np.allclose(grid.depths, [0.01, 0.02, 0.03, 0.04])
    assert LabelGrid(5, 3, 2).size == 30


def test_store_holds_the_source_arrays_value_for_value(data):
    store = unbuilt(data["root"]).label_store
    counts = store.build(POINTS, SCENES)
    assert counts == {"built": len(POINTS) + len(SCENES), "skipped": 0}
    for object_id, (points, scores, widths) in data["labels"].items():
        stored = store.object_arrays(object_id)
        for ours, source in zip(stored, (points, scores, widths)):
            assert ours.dtype == source.dtype and np.array_equal(ours, source)
        assert isinstance(stored[1], np.memmap) and isinstance(stored[2], np.memmap)
    for scene, object_ids in SCENES.items():
        assert list(store.scene_object_ids(scene)) == object_ids
        for index, object_id in enumerate(object_ids):
            labels = store.labels(scene, index, object_id)
            assert labels.num_points == POINTS[object_id]
            assert np.array_equal(labels.collision(), data["collision"][scene][index])
    shards = store.asset.shards()
    assert shards["object_000005"]["points"] == POINTS[5]
    assert shards["object_000005"]["source"]["file"] == "grasp_label/005_labels.npz"
    assert shards["scene_000001"]["rows"] == [POINTS[i] for i in SCENES[1]]
    assert shards["scene_000001"]["source"]["bytes"] > 0


def test_take_returns_copies_of_the_requested_rows_in_order(data):
    store = unbuilt(data["root"]).label_store
    store.build(POINTS, SCENES)
    labels = store.labels(2, 1, 3)
    rows = np.array([5, 0, 3])
    points, scores, widths, collision = labels.take(rows, collision=True)
    source = data["labels"][3]
    assert np.array_equal(points, source[0][rows]) and np.array_equal(scores, source[1][rows])
    assert np.array_equal(widths, source[2][rows])
    assert np.array_equal(collision, data["collision"][2][1][rows])
    assert type(scores) is np.ndarray and scores.flags.writeable and collision.dtype == np.bool_
    scores[:] = 0  # a copy: the store is untouched
    assert np.array_equal(store.object_arrays(3)[1], source[1])
    assert labels.take(rows, collision=False)[3] is None


def test_simplified_and_official_label_files_give_the_same_store_content(tmp_path, monkeypatch):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    official = make_tree(tmp_path / "a", simplified=False, seed=3)
    make_tree(tmp_path / "b", simplified=True, seed=3)
    one = unbuilt(tmp_path / "a", label_source="grasp_label").label_store
    two = unbuilt(tmp_path / "b", label_source="grasp_label_simplified").label_store
    assert one.asset.spec.hash != two.asset.spec.hash  # the source is part of the identity
    one.build(POINTS, SCENES)
    two.build(POINTS, SCENES)
    for object_id in POINTS:
        for ours, theirs in zip(one.object_arrays(object_id), two.object_arrays(object_id)):
            assert np.array_equal(ours, theirs)
        assert np.array_equal(one.object_arrays(object_id)[2], official["labels"][object_id][2])


def test_build_is_resumable(data):
    store = unbuilt(data["root"]).label_store
    assert store.missing(POINTS, SCENES) == [
        *(f"object_{i:06d}" for i in sorted(POINTS)),
        *(f"scene_{s:06d}" for s in SCENES),
    ]
    store.build([3], [])
    assert store.missing(POINTS, SCENES)[0] == "object_000005"
    seen = []
    counts = unbuilt(data["root"]).label_store.build(POINTS, SCENES, progress=seen.append)
    assert counts == {"built": len(POINTS) + len(SCENES) - 1, "skipped": 1}
    assert len(seen) == len(POINTS) + len(SCENES)
    assert unbuilt(data["root"]).label_store.build(POINTS, SCENES)["built"] == 0


def test_reading_before_building_is_an_error(data):
    store = unbuilt(data["root"]).label_store
    with pytest.raises(AssetError, match="Build it with: tool --labels"):
        store.require(POINTS, SCENES, build="tool --labels")
    store.build([3], [])
    with pytest.raises(AssetError, match="grasp labels of object 7 are not in the store"):
        store.object_arrays(7)
    with pytest.raises(AssetError, match="collision labels of scene 0 are not in the store"):
        store.labels(0, 0, 3)
    with pytest.raises(AssetError, match="lacks 5 of the 6 shards"):
        store.require(POINTS, SCENES, build="tool --labels")


def test_malformed_sources_are_rejected_with_the_reason(data):
    root = data["root"]
    points, scores, _ = data["labels"][3]
    np.savez(root / "grasp_label/003_labels.npz", points=points, scores=scores[:, :4])
    with pytest.raises(AssetError, match="expected arrays 'points', 'scores' and 'offsets' or"):
        unbuilt(root).label_store.build([3], [])
    np.savez(root / "grasp_label/003_labels.npz", points=points, scores=scores, width=scores[:, :4])
    with pytest.raises(AssetError, match=r"'widths' has shape \(7, 4, 3, 2\).*need \(7, 5, 3, 2\)"):
        unbuilt(root).label_store.build([3], [])
    with pytest.raises(FileNotFoundError, match="grasp labels of object 9 not found"):
        unbuilt(root).label_store.build([9], [])

    store = unbuilt(root).label_store
    store.build([5, 7], [])
    collisions = data["collision"][1]
    file = root / "collision_label/scene_0001/collision_labels.npz"
    np.savez(file, arr_0=collisions[0])
    with pytest.raises(AssetError, match="holds 1 collision arrays, but scene 1 has 2 object"):
        store.build([], [1])
    np.savez(file, arr_0=collisions[0][:-1], arr_1=collisions[1])
    with pytest.raises(AssetError, match="arr_0 has 10 rows, but object 7 has 11 annotated"):
        store.build([], [1])
    np.savez(file, arr_0=collisions[0].astype(np.uint8), arr_1=collisions[1])
    with pytest.raises(AssetError, match="arr_0 is uint8"):
        store.build([], [1])
    assert store.missing([], [1]) == ["scene_000001"]  # nothing half-built was recorded


def test_a_frame_that_disagrees_with_the_stored_scene_is_refused(data):
    store = unbuilt(data["root"]).label_store
    store.build(POINTS, SCENES)
    with pytest.raises(AssetError, match=r"instance 0 is object 7 in the frame, but .* \[3, 7\]"):
        store.labels(0, 0, 7)


def test_store_survives_pickling_and_bounds_its_open_maps(data):
    dataset = unbuilt(data["root"])
    dataset.label_store.build(POINTS, SCENES)
    store = GraspLabelStore(dataset, GRID, max_open_objects=2, max_open_scenes=1)
    handed_out = [
        store.labels(scene, index, object_id)
        for scene, object_ids in SCENES.items()
        for index, object_id in enumerate(object_ids)
    ]
    # Handing out labels reads the points only; a grid is mapped when it is first read.
    assert len(store._objects) == 0 and len(store._scenes) == 0
    for labels in handed_out:
        assert labels.scores.shape == labels.widths.shape == (labels.num_points, *GRID.shape)
        assert len(labels.packed_collision) == labels.num_points
    assert len(store._objects) == 2 and len(store._scenes) == 1
    clone = pickle.loads(pickle.dumps(store))
    assert len(clone._objects) == 0 and len(clone._scenes) == 0  # maps reopen lazily per process
    assert np.array_equal(clone.labels(1, 0, 7).scores, data["labels"][7][1])
