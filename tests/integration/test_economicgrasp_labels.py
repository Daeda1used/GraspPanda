"""The compact grasp-label asset against the output of the upstream generator, and the
dataset reader that serves it against the upstream loader.

The reference files are what ``dataset/generate_economic.py`` of
``iSEE-Laboratory/EconomicGrasp`` (commit ``7361bb6``) wrote for the scenes of the
recorded training batch; ``tools/make_economicgrasp_fixtures.py`` runs it and keeps its
output under ``$GRASPPANDA_CACHE/fixtures/economicgrasp/`` (never committed). The dense
labels are read from the grasp-label store (``grasppanda prepare graspnet1b --labels``).

Nothing is written to the shared cache: ``grasppanda prepare --compact-labels`` runs
against a temporary cache root that links to the dense store.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.cli.main import main
from grasppanda.config import Config
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.compact_labels import (
    VIEW_ARRAYS,
    CompactLabelConfig,
    CompactLabelStore,
    from_reference_arrays,
    scene_shard,
)
from grasppanda.datasets.labels import GraspLabelStore
from grasppanda.registry import DATASETS, DETECTORS
from grasppanda.structures import COMPACT_LABELS, CompactGraspLabels, SampleKey, collate

pytestmark = [pytest.mark.dataset, pytest.mark.oracle]

RUN_CONFIG = "economicgrasp/economicgrasp_sunet14_g1b-kinect_10e"


@pytest.fixture(scope="module")
def recorded(asset_cache) -> tuple[Path, dict]:
    directory = asset_cache / "fixtures" / "economicgrasp"
    manifest = directory / "manifest.json"
    if not manifest.is_file() or "labels" not in json.loads(manifest.read_text()):
        pytest.skip(
            f"no reference label files at {directory}; record them with "
            "tools/make_economicgrasp_fixtures.py"
        )
    return directory, json.loads(manifest.read_text())


@pytest.fixture(scope="module")
def reference(recorded) -> dict[int, dict[str, np.ndarray]]:
    """Per scene, the arrays of the upstream generator in lattice view order."""
    directory, manifest = recorded
    scenes = {}
    for scene, entry in manifest["labels"]["scenes"].items():
        with np.load(directory / entry["file"]) as archive:
            scenes[int(scene)] = from_reference_arrays(archive)
    return scenes


@pytest.fixture(scope="module")
def private_cache(graspnet1b_root, asset_cache, tmp_path_factory) -> Iterator[Path]:
    """A cache root of its own that reads the dense label store of the shared one."""
    dense = asset_cache / "graspnet1b" / GraspLabelStore.KIND
    if not dense.is_dir():
        pytest.skip("no grasp-label store: grasppanda prepare graspnet1b --labels")
    root = tmp_path_factory.mktemp("cache")
    (root / "graspnet1b").mkdir()
    (root / "graspnet1b" / GraspLabelStore.KIND).symlink_to(dense, target_is_directory=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GRASPPANDA_CACHE", str(root))
        yield root


@pytest.fixture(scope="module")
def prepared(private_cache, reference) -> list[str]:
    """``grasppanda prepare --compact-labels`` for the sample scenes, run twice."""
    first, *_ = scenes = sorted(reference)
    command = ["prepare", "graspnet1b", "--compact-labels", "--config", RUN_CONFIG, "--scenes"]
    outputs = []
    for selection in (str(first), ",".join(map(str, scenes))) * 2:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            status = main([*command, selection])
        assert status == 0, captured.getvalue()
        outputs.append(captured.getvalue())
    return outputs


@pytest.fixture(scope="module")
def dataset(private_cache, prepared):
    """The training dataset of the Kinect config, reading the freshly built asset."""
    node = dict(Config.load(RUN_CONFIG).select("data.train.dataset"))
    # Upstream-format graspness files cover every scene; the native asset may not yet.
    node.update(check_assets=False, seed=0, graspness={"source": "legacy"})
    return DATASETS.build(node)


def test_prepare_builds_resumes_and_then_has_nothing_to_do(prepared, private_cache, dataset):
    first, second, third, fourth = prepared
    assert "0 of 1 scenes already present" in first and "1 scenes written and read back" in first
    assert "1 of 2 scenes already present" in second and "1 scenes written" in second
    assert "nothing to do" in third and "nothing to do" in fourth
    store = dataset.compact_labels
    assert private_cache in store.asset.path.parents  # nothing went to the shared cache
    assert store.asset.path.name == store.asset.spec.hash
    assert sorted(store.asset.shards()) == [scene_shard(5), scene_shard(63)]
    with pytest.raises(SystemExit, match="no `compact_labels` section"):
        main(["prepare", "graspnet1b", "--compact-labels", "--camera", "kinect"])


def test_the_asset_equals_the_output_of_the_upstream_generator(dataset, reference):
    """Every value of every array, for every object of the sample scenes."""
    store = dataset.compact_labels
    compared = 0
    for scene, expected in reference.items():
        stored = store.read_scene(scene)
        record = store.asset.shards()[scene_shard(scene)]
        assert sum(record["kept"]) == len(expected["points"]) < sum(record["annotated"])
        assert record["objects"] == dataset.scene_object_ids(scene)
        assert set(stored) == {"points", "instance", *VIEW_ARRAYS}
        for name, values in expected.items():
            assert stored[name].dtype == values.dtype, (scene, name)
            assert np.array_equal(stored[name], values), (scene, name)
            compared += values.size
        # The point filter removed something from every object.
        assert all(0 < kept < total for kept, total in zip(record["kept"], record["annotated"]))
        assert (np.diff(stored["instance"]) >= 0).all()
    assert compared > 10_000_000


def test_the_asset_identity_is_camera_independent_and_parameter_dependent(dataset):
    store = dataset.compact_labels
    spec = store.asset.spec
    other = dict(
        Config.load(RUN_CONFIG.replace("kinect", "realsense")).select("data.train.dataset")
    )
    realsense = DATASETS.build({**other, "check_assets": False})
    assert realsense.compact_labels.asset.spec.hash == spec.hash  # one asset for both cameras
    assert spec.preprocessing is None and not spec.uses_ground_truth
    assert spec.inputs["grasp_labels"] == dataset.label_store.asset.spec.hash
    changed = CompactLabelStore.spec(
        "graspnet1b", CompactLabelConfig(0.5, 0.6, 0.1), dataset.label_grid, "0" * 16
    )
    assert changed.hash != spec.hash
    with pytest.raises(AssetError, match="labels of scene 0 are not in the asset"):
        store.read_scene(0)  # a missing scene is an error, never an empty label
    with pytest.raises(AssetError, match=r"(?s)lacks 98 of the 100 shards .*--compact-labels"):
        dataset.require_assets()
    assert next(record["kind"] for record in dataset.asset_records()) == CompactLabelStore.KIND


def test_the_reader_serves_what_the_upstream_loader_served(dataset, recorded, reference):
    """The frames of the recorded training batch: same poses, same labels per object."""
    directory, manifest = recorded
    if "training" not in manifest:
        pytest.skip("the fixtures were recorded without a training batch")
    record = np.load(directory / manifest["training"]["file"])
    flip = dataset.transforms[1]
    assert type(flip).__name__ == "RandomFlip" and flip.prob == 0.5
    checked = 0
    try:
        for position, ((scene, frame), mirrored) in enumerate(
            zip(manifest["training"]["frames"], manifest["training"]["mirrored_frames"])
        ):
            index = list(dataset.keys).index(SampleKey("graspnet1b", scene, "kinect", frame))
            flip.prob = 1.0 if mirrored else 0.0  # the draw of the reference loader
            batch = dataset[index]
            assert batch.supervision.objects == [[]]  # no dense grids travel with compact labels
            assert len(batch.observation.points) == manifest["num_points"]
            assert set(batch.supervision.point_labels) == {"objectness", "instance", "graspness"}
            (objects,) = batch.supervision.extras[COMPACT_LABELS]
            poses = record[f"poses_{position}"]
            assert len(objects) == len(poses) == len(dataset.scene_object_ids(scene))
            assert (np.linalg.det(poses[0][:, :3]) < 0) == mirrored
            expected = reference[scene]
            for slot, labels in enumerate(objects):
                assert isinstance(labels, CompactGraspLabels) and labels.view_max_friction == 0.6
                assert labels.object_id == dataset.scene_object_ids(scene)[slot]
                assert np.array_equal(labels.pose[:3].numpy(), poses[slot]), (scene, frame, slot)
                assert labels.pose[3].tolist() == [0.0, 0.0, 0.0, 1.0]
                rows = expected["instance"] == slot
                assert np.array_equal(labels.points.numpy(), expected["points"][rows])
                for name in VIEW_ARRAYS:
                    assert np.array_equal(getattr(labels, name).numpy(), expected[name][rows])
                checked += 1
    finally:
        flip.prob = 0.5
    assert checked == sum(manifest["training"]["objects_per_frame"])


@pytest.mark.gpu
def test_a_collated_batch_reaches_the_loss(dataset, reference, cuda):
    """Dataset → collate → EconomicGrasp.loss on real frames: every term is finite."""
    pytest.importorskip("spconv")
    keys = list(dataset.keys)
    indices = [
        keys.index(SampleKey("graspnet1b", scene, "kinect", 40)) for scene in sorted(reference)
    ]
    batch = collate([dataset[index] for index in indices]).to(cuda)
    labels = batch.supervision.extras[COMPACT_LABELS]
    assert len(labels) == len(reference) and labels[0][0].angle.device.type == "cuda"
    torch.manual_seed(0)
    model = DETECTORS.build(Config.load(RUN_CONFIG)["model"]).to(cuda).train()
    losses = model.loss(batch.observation, batch.supervision)
    expected = {"objectness", "graspness", "view", "angle", "depth", "score", "width"}
    assert {key.split("/", 1)[1] for key in losses if key.startswith("loss/")} == expected
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    assert 0.0 < float(losses["metric/supervised_seeds"]) < 1.0
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    assert all(p.grad is not None for p in model.parameters())
