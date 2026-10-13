"""Instance-normalised graspness: the definition, the asset it is, and that the
scene-normalised asset of GSNet and EconomicGrasp is untouched by the option."""

from __future__ import annotations

import numpy as np
import pytest

from grasppanda.datasets.graspness import (
    GraspnessConfig,
    GraspnessMaps,
    frame_graspness,
    instance_normalised,
    object_graspness,
)
from grasppanda.structures import ContractError

from .test_data_graspness import NATIVE, float32_search, generate
from .test_data_graspnet1b import GRID, SCENES, build, tree  # noqa: F401

INSTANCE = {**NATIVE, "normalization": "instance"}


def test_instance_normalisation_is_a_min_max_over_the_points_of_one_object():
    values = np.array([0.25, 0.75, 0.5, 0.25])
    assert instance_normalised(values).tolist() == [0.0, 1.0, 0.5, 0.0]
    assert instance_normalised(values).dtype == np.float64
    # No best point: the object keeps 0 (the published generator divides by zero here).
    assert instance_normalised(np.full(4, 0.3)).tolist() == [0.0] * 4
    assert instance_normalised(np.zeros(3)).tolist() == [0.0] * 3
    assert instance_normalised(np.zeros(0)).shape == (0,)
    assert np.isfinite(instance_normalised(np.array([0.7]))).all()


def expected_maps(dataset, truth, scene: int, friction: float) -> dict[int, np.ndarray]:
    """Equations 1 and 2 of the FineGrasp paper, per frame, with exhaustive searches."""
    object_ids = SCENES[scene]
    values = []
    for index, object_id in enumerate(object_ids):
        scores = truth["labels"][object_id][1]
        usable = (scores > 0) & (scores <= np.float32(friction)) & ~truth["collision"][scene][index]
        raw = usable.reshape(len(scores), -1).sum(axis=1) / GRID.size
        spread = raw.max() - raw.min()
        values.append((raw - raw.min()) / spread if spread > 0 else np.zeros_like(raw))  # eq. 1
    maps = {}
    for key in dataset.scene_keys(scene):
        frame = dataset.frame(key)
        placed = []
        for (object_id, pose), _ in zip(dataset.object_poses(key, "annotations"), object_ids):
            points = truth["labels"][object_id][0].astype(np.float64)
            placed.append(points @ pose[:3, :3].astype(np.float64).T + pose[:3, 3])
        nearest = float32_search(frame.cloud[frame.valid], np.concatenate(placed))
        raw = np.concatenate(values)[nearest]
        out = np.full(frame.valid.shape, np.nan)
        out[frame.valid] = (raw - raw.min()) / (raw.max() - raw.min())  # eq. 2
        maps[key.frame] = out
    return maps


def test_generated_maps_follow_the_definition(tree):  # noqa: F811
    config = {**INSTANCE, "dtype": "float32"}
    dataset = build(tree, split="all", transforms=[], graspness=config, check_assets=False)
    assert dataset.graspness_config.normalization == "instance"
    generate(dataset)
    plain = build(
        tree,
        split="all",
        transforms=[],
        graspness={**NATIVE, "dtype": "float32"},
        check_assets=False,
    )
    generate(plain)
    different = 0
    for scene in SCENES:
        expected = expected_maps(dataset, tree, scene, 0.8)
        for key in dataset.scene_keys(scene):
            stored = dataset.graspness.read_map(key)
            valid = ~np.isnan(stored)
            assert np.array_equal(valid, ~np.isnan(expected[key.frame]))
            assert np.array_equal(stored[valid], expected[key.frame][valid].astype(np.float32))
            assert stored[valid].min() == 0 and stored[valid].max() == 1
            different += int(
                not np.array_equal(stored[valid], plain.graspness.read_map(key)[valid])
            )
    assert different > 0  # it is another label, not another name for the same one


def test_it_lifts_the_object_that_scene_normalisation_suppresses():
    """A weak object next to a strong one: its best point reaches 1 instead of 0.14."""
    weak, strong = np.array([0.1, 0.15, 0.2]), np.array([0.2, 0.5, 0.8])
    both = np.concatenate([weak, strong])
    scene_only = (both - both.min()) / (both.max() - both.min())
    assert scene_only[:3].max() == pytest.approx(1 / 7)
    normalised = np.concatenate([instance_normalised(weak), instance_normalised(strong)])
    per_instance = (normalised - normalised.min()) / (normalised.max() - normalised.min())
    assert per_instance[:3].max() == per_instance[3:].max() == 1.0


def test_an_object_without_contrast_does_not_poison_the_frame(tree):  # noqa: F811
    """With the published generator such an object makes every value of the frame NaN."""
    dataset = build(tree, split="all", transforms=[], graspness=INSTANCE, check_assets=False)
    labels = dataset.scene_labels(0)
    values = [instance_normalised(object_graspness(instance, 0.8)) for instance in labels]
    values[0] = instance_normalised(np.full_like(values[0], 0.25))  # a constant object
    assert not values[0].any()
    key = dataset.scene_keys(0)[0]
    stored = frame_graspness(
        dataset.frame(key), dataset.object_poses(key, "annotations"), labels, values, "float32"
    )
    valid = ~np.isnan(stored)
    assert valid.any() and stored[valid].min() == 0 and stored[valid].max() == 1


def test_the_variant_is_another_asset_and_the_default_keeps_its_identity(tree):  # noqa: F811
    scene = build(tree, graspness=NATIVE, check_assets=False)
    spelled = build(tree, graspness={**NATIVE, "normalization": "scene"}, check_assets=False)
    instance = build(tree, graspness=INSTANCE, check_assets=False)
    assert scene.graspness.asset.path == spelled.graspness.asset.path
    assert instance.graspness.asset.path != scene.graspness.asset.path
    # The scene-normalised identity is what it was before the option existed: assets that
    # were generated then are still found.
    arguments = ("tinygraspnet", "realsense")
    preprocessing = [step.config() for step in scene.preprocess]
    spec = GraspnessMaps.spec(*arguments, GraspnessConfig(**NATIVE), preprocessing, True, "labels")
    assert spec.identity()["parameters"] == {
        "camera": "realsense",
        "friction_threshold": 0.8,
        "poses": "annotations",
        "dtype": "float16",
        "neighbour": "nearest, float32 squared distance, lowest index on ties",
        "normalisation": "per-frame min-max over the mask",
    }
    variant = GraspnessMaps.spec(
        *arguments, GraspnessConfig(**INSTANCE), preprocessing, True, "labels"
    )
    added = variant.identity()["parameters"].keys() - spec.identity()["parameters"].keys()
    assert added == {"instance_normalisation"} and variant.hash != spec.hash
    assert "within every object instance" in variant.about["values"]
    assert "within every object instance" not in spec.about["values"]
    record = instance.graspness.record()
    assert record["kind"] == "graspness" and record["provenance"] == "native"


def test_the_dataset_serves_whichever_asset_its_config_names(tree):  # noqa: F811
    probe = build(tree, graspness=INSTANCE, check_assets=False)
    generate(probe, [0, 1])
    dataset = build(tree, graspness=INSTANCE, transforms=[])
    key = dataset.keys[4]
    sample = dataset.sample(key)
    stored = dataset.graspness.read_map(key)
    assert np.array_equal(sample.point_labels["graspness"], stored.reshape(-1)[sample.pixels])
    with pytest.raises(Exception, match="No graspness asset of tinygraspnet"):
        build(tree, graspness=NATIVE, transforms=[])  # the scene-normalised one was never built


def test_only_the_native_generator_takes_the_option(tree):  # noqa: F811
    pose_sources = ("annotations", "meta")
    assert GraspnessConfig(**INSTANCE).problems(pose_sources) == []
    (problem,) = GraspnessConfig(**{**NATIVE, "normalization": "object"}).problems(pose_sources)
    assert "must be 'scene' or 'instance', got 'object'" in problem
    for source in ("legacy", "legacy_import"):
        (problem,) = GraspnessConfig(source=source, normalization="instance").problems(pose_sources)
        assert "parameter of the native generator" in problem and source in problem
        assert GraspnessConfig(source=source).problems(pose_sources) == []
    with pytest.raises(ContractError, match="parameter of the native generator"):
        build(tree, graspness={"source": "legacy", "normalization": "instance"}, check_assets=False)
