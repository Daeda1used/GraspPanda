"""Point-wise graspness: definition, pixel-keyed storage, legacy reader, alignment checks."""

from __future__ import annotations

import numpy as np
import pytest

from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspness import (
    GraspnessConfig,
    generate_graspness,
    import_legacy_graspness,
    nearest_label_point,
    object_graspness,
)
from grasppanda.structures import SampleKey

from .test_data_graspnet1b import FRAMES, GRID, PREPROCESS, SCENES, build, tree  # noqa: F401

NATIVE = {"source": "native", "friction_threshold": 0.8, "poses": "annotations"}


def float32_search(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The reference search: float32 squared distances, first minimum wins."""
    query, reference = query.astype(np.float32), reference.astype(np.float32)
    out = np.empty(len(query), dtype=np.int64)
    for row, point in enumerate(query):
        best, best_distance = 0, np.float32(np.inf)
        for index, other in enumerate(reference):
            delta = point - other
            distance = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            if distance < best_distance:
                best, best_distance = index, distance
        out[row] = best
    return out


def expected_maps(dataset, truth, scene: int, config: GraspnessConfig) -> dict[int, np.ndarray]:
    """The definition, written out per frame with exhaustive searches."""
    object_ids = SCENES[scene]
    values = []
    for index, object_id in enumerate(object_ids):
        scores = truth["labels"][object_id][1]
        usable = (scores > 0) & (scores <= np.float32(config.friction_threshold))
        usable &= ~truth["collision"][scene][index]
        values.append(usable.reshape(len(scores), -1).sum(axis=1) / GRID.size)
    maps = {}
    for key in dataset.scene_keys(scene):
        frame = dataset.frame(key)
        placed = []
        for (object_id, pose), _ in zip(dataset.object_poses(key, config.poses), object_ids):
            points = truth["labels"][object_id][0].astype(np.float64)
            placed.append(points @ pose[:3, :3].astype(np.float64).T + pose[:3, 3])
        nearest = float32_search(frame.cloud[frame.valid], np.concatenate(placed))
        raw = np.concatenate(values)[nearest]
        out = np.full(frame.valid.shape, np.nan)
        out[frame.valid] = (raw - raw.min()) / (raw.max() - raw.min())
        maps[key.frame] = out
    return maps


def generate(dataset, scenes=SCENES) -> None:
    maps = dataset.graspness
    maps.asset.create()
    for scene in scenes:
        record = generate_graspness(dataset, maps, scene, dataset.graspness_config)
        maps.mark_complete(scene, record)


# --------------------------------------------------------------------------- definition


def test_object_graspness_counts_usable_grasps(tree):  # noqa: F811
    dataset = build(tree, transforms=[])
    for index, object_id in enumerate(SCENES[1]):
        labels = dataset.scene_labels(1)[index]
        scores, collision = tree["labels"][object_id][1], tree["collision"][1][index]
        expected = [
            sum(
                1
                for score, collides in zip(scores[p].ravel(), collision[p].ravel())
                if 0 < score <= np.float32(0.8) and not collides
            )
            / GRID.size
            for p in range(len(scores))
        ]
        assert object_graspness(labels, 0.8).tolist() == expected
        assert (object_graspness(labels, 0.4) <= object_graspness(labels, 0.8)).all()
        assert object_graspness(labels, 0.8).max() > 0


def test_nearest_label_point_equals_the_float32_exhaustive_search():
    rng = np.random.default_rng(0)
    centre = np.array([0.0, 0.0, 0.5])
    reference = rng.uniform(-0.2, 0.2, (300, 3)) + centre
    query = rng.uniform(-0.3, 0.3, (4000, 3)) + centre
    assert np.array_equal(nearest_label_point(query, reference), float32_search(query, reference))
    assert nearest_label_point(query, reference[:1]).tolist() == [0] * len(query)
    with pytest.raises(ValueError):
        nearest_label_point(query, reference[:0])


def test_nearest_label_point_breaks_ties_like_the_reference():
    rng = np.random.default_rng(1)
    reference = rng.uniform(-0.2, 0.2, (60, 3))
    reference[40] = reference[7]  # an exact duplicate: the lower index must win
    reference[55] = reference[20] + 1e-9  # equal after rounding to float32
    query = np.concatenate([reference[[7, 40, 20, 55]] + 1e-4, rng.uniform(-0.2, 0.2, (500, 3))])
    # points on the bisector plane of two reference points: decided by float32 rounding
    midpoints = (reference[:25] + reference[25:50]) / 2
    query = np.concatenate([query, midpoints])
    nearest = nearest_label_point(query, reference)
    assert nearest[:4].tolist() == [7, 7, 20, 20]
    assert np.array_equal(nearest, float32_search(query, reference))


# ------------------------------------------------------------------------------ assets


@pytest.mark.parametrize("poses", ["annotations", "meta"])
def test_generated_maps_follow_the_definition(tree, poses):  # noqa: F811
    config = {**NATIVE, "poses": poses, "dtype": "float32"}
    dataset = build(tree, split="all", transforms=[], graspness=config, check_assets=False)
    generate(dataset)
    for scene in SCENES:
        expected = expected_maps(dataset, tree, scene, dataset.graspness_config)
        for key in dataset.scene_keys(scene):
            stored = dataset.graspness.read_map(key)
            assert stored.dtype == np.float32 and stored.shape == dataset.image_size
            assert np.array_equal(np.isnan(stored), np.isnan(expected[key.frame]))
            valid = ~np.isnan(stored)
            assert np.array_equal(stored[valid], expected[key.frame][valid].astype(np.float32))
            assert stored[valid].min() == 0 and stored[valid].max() == 1
    shard = dataset.graspness.asset.shards()["scene_000001"]
    assert shard["frames"] == FRAMES and shard["points"] > 0 and shard["bytes"] > 0


def test_float16_storage_is_the_default_and_bounds_the_error(tree):  # noqa: F811
    dataset = build(tree, split="all", transforms=[], graspness=NATIVE, check_assets=False)
    generate(dataset, [0])
    key = dataset.scene_keys(0)[1]
    stored = dataset.graspness.read_map(key)
    expected = expected_maps(dataset, tree, 0, dataset.graspness_config)[1]
    valid = ~np.isnan(stored)
    assert stored.dtype == np.float16
    assert np.abs(stored[valid].astype(np.float64) - expected[valid]).max() <= 2.5e-4
    assert dataset.graspness.asset.spec.uses_ground_truth  # the crop came from the segmentation


def test_dataset_serves_graspness_aligned_with_its_points(tree):  # noqa: F811
    build(tree, graspness=NATIVE, check_assets=False)  # builds the label store
    probe = build(tree, graspness=NATIVE, check_assets=False)
    generate(probe, [0, 1])
    sampler = [{"type": "RandomSample", "num_points": 50}]
    dataset = build(tree, graspness=NATIVE, transforms=sampler)
    key = dataset.keys[4]
    sample = dataset.transforms[0](dataset.sample(key))
    stored = dataset.graspness.read_map(key)
    assert np.array_equal(sample.point_labels["graspness"], stored.reshape(-1)[sample.pixels])
    assert sample.point_labels["graspness"].dtype == np.float32
    records = dataset.asset_records()
    assert [r["kind"] for r in records] == ["grasp_labels", "graspness"]
    assert records[1]["provenance"] == "native" and records[1]["uses_ground_truth"] is True
    with pytest.raises(AssetError, match="lacks 1 of the 1 shards"):
        build(tree, graspness=NATIVE, split="test")


def test_changed_preprocessing_asks_for_a_different_asset(tree):  # noqa: F811
    probe = build(tree, graspness=NATIVE, check_assets=False)
    generate(probe, [0, 1])
    wider = [PREPROCESS[0], {**PREPROCESS[1], "margin": 0.05}]
    with pytest.raises(AssetError) as error:
        build(tree, graspness=NATIVE, preprocess=wider)
    message = str(error.value)
    assert "No graspness asset of tinygraspnet for this configuration" in message
    assert "preprocessing[1].margin: 0.02 vs 0.05" in message
    assert "grasppanda prepare tinygraspnet --graspness" in message
    other = build(tree, graspness=NATIVE, preprocess=wider, check_assets=False)
    assert other.fingerprint != probe.fingerprint
    assert other.graspness.asset.path != probe.graspness.asset.path
    for changed in ({"friction_threshold": 0.6}, {"poses": "meta"}, {"dtype": "float32"}):
        variant = build(tree, graspness={**NATIVE, **changed}, check_assets=False)
        assert variant.graspness.asset.path != probe.graspness.asset.path
    kinect = build(tree, graspness=NATIVE, camera="kinect", check_assets=False)
    assert kinect.graspness.asset.path != probe.graspness.asset.path


def test_a_map_generated_for_another_mask_is_refused(tree):  # noqa: F811
    dataset = build(tree, graspness=NATIVE, check_assets=False)
    generate(dataset, [0, 1])
    key = dataset.keys[0]
    sample = dataset.sample(key)
    with pytest.raises(AssetError, match="was generated for a different mask: it labels"):
        dataset.graspness.read(key, sample.pixels[:-1], sample.image_size)
    shifted = sample.pixels.copy()
    shifted[-1] = shifted[-1] + 1 if shifted[-1] + 1 not in shifted else 0
    with pytest.raises(AssetError, match="same count, different pixels"):
        dataset.graspness.read(key, np.sort(shifted), sample.image_size)
    with pytest.raises(AssetError, match=r"is \(24, 32\), but the frame is \(12, 64\)"):
        dataset.graspness.read(key, sample.pixels, (12, 64))
    with pytest.raises(AssetError, match="scene 2 has not been generated"):
        dataset.graspness.read_map(SampleKey("tinygraspnet", 2, "realsense", 0))


def test_a_frame_without_contrast_cannot_be_normalised(tree):  # noqa: F811
    for file in (tree["root"] / "collision_label").rglob("*.npz"):
        with np.load(file) as archive:
            np.savez(file, **{name: np.ones_like(archive[name]) for name in archive.files})
    dataset = build(tree, graspness=NATIVE, check_assets=False)
    dataset.graspness.asset.create()
    with pytest.raises(AssetError, match="every point has graspness 0.0"):
        generate_graspness(dataset, dataset.graspness, 0, dataset.graspness_config)
    assert not dataset.graspness.shard_file(0).exists()  # no partial shard is left behind


# ------------------------------------------------------------------------------ legacy


def write_legacy(dataset, truth, scenes, drop_rows: int = 0) -> None:
    """Upstream-format files: (M, 1) float64, row-aligned to the masked cloud."""
    config = GraspnessConfig(**NATIVE)
    for scene in scenes:
        expected = expected_maps(dataset, truth, scene, config)
        for key in dataset.scene_keys(scene):
            values = expected[key.frame][~np.isnan(expected[key.frame])]
            file = dataset.legacy_graspness_file(key)
            file.parent.mkdir(parents=True, exist_ok=True)
            np.save(file, values[: len(values) - drop_rows, None])


def test_legacy_files_are_served_only_when_the_row_count_matches(tree):  # noqa: F811
    probe = build(tree, transforms=[])
    write_legacy(probe, tree, [0, 1])
    dataset = build(tree, transforms=[], graspness={"source": "legacy"})
    key = dataset.keys[2]
    sample = dataset.sample(key)
    expected = expected_maps(probe, tree, key.scene, GraspnessConfig(**NATIVE))[key.frame]
    assert np.array_equal(
        sample.point_labels["graspness"], expected[~np.isnan(expected)].astype(np.float32)
    )
    record = dataset.asset_records()[1]
    assert record["provenance"] == "legacy" and record["generator"] == "unknown"

    # the same files under a different crop: every frame is refused
    wider = [PREPROCESS[0], {**PREPROCESS[1], "margin": 0.2}]
    misaligned = build(tree, transforms=[], graspness={"source": "legacy"}, preprocess=wider)
    with pytest.raises(AssetError) as error:
        misaligned.sample(key)
    message = str(error.value)
    assert "does not align with this dataset's preprocessing" in message
    assert f"preprocessing fingerprint {misaligned.fingerprint}" in message
    assert "grasppanda prepare <dataset> --graspness" in message
    dataset.legacy_graspness_file(key).unlink()
    with pytest.raises(AssetError, match="no legacy graspness file"):
        dataset.sample(key)


def test_legacy_import_rekeys_to_pixels_and_is_marked(tree):  # noqa: F811
    probe = build(tree, transforms=[])
    write_legacy(probe, tree, [0, 1])
    config = {"source": "legacy_import", "dtype": "float32"}
    importer = build(tree, transforms=[], graspness=config, check_assets=False)
    maps = importer.graspness
    maps.asset.create()
    for scene in (0, 1):
        record = import_legacy_graspness(
            importer, maps, scene, importer.legacy_graspness(), "float32"
        )
        maps.mark_complete(scene, record)
    dataset = build(tree, transforms=[], graspness=config)
    direct = build(tree, transforms=[], graspness={"source": "legacy"})
    for key in dataset.keys:
        assert np.array_equal(
            dataset.sample(key).point_labels["graspness"],
            direct.sample(key).point_labels["graspness"],
        )
    record = dataset.asset_records()[1]
    assert record["provenance"] == "legacy-import" and record["generator"] == "legacy-import@1"
    native = build(tree, transforms=[], graspness=NATIVE, check_assets=False)
    assert native.graspness.asset.path != maps.asset.path

    write_legacy(probe, tree, [0], drop_rows=1)  # a file from some other pipeline
    fresh = build(tree, transforms=[], graspness={**config, "dtype": "float16"}, check_assets=False)
    fresh.graspness.asset.create()
    with pytest.raises(AssetError, match="does not align with this dataset's preprocessing"):
        import_legacy_graspness(fresh, fresh.graspness, 0, fresh.legacy_graspness(), "float16")
    assert not fresh.graspness.shard_file(0).exists()
