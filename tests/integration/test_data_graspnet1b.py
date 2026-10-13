"""GraspNet-1Billion reader against the upstream Graspness pipeline, on real frames.

Reference values are recorded once by ``tools/make_data_fixtures.py`` (run with the
oracle interpreter; it imports the upstream code unmodified) and kept under
``$GRASPPANDA_CACHE/fixtures/data_pipeline``. The tests feed the recorded sampling
indices and augmentation draws to this pipeline and compare.

Tolerances:

* masks, masked clouds, sampled clouds, point labels, object poses and per-object label
  arrays without augmentation: **exact** equality.
* after the rotation augmentation: at most one float32 unit in the last place. Upstream
  rotates with a BLAS matrix product, here each coordinate is evaluated in a fixed
  order; the two may round the last bit of the float64 result differently, which
  survives the cast to float32 only at a rounding boundary.
* native graspness against the upstream generator: exact in double precision; the
  float16 storage adds at most 2.5e-4.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from grasppanda.config import Config
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspness import frame_graspness, object_graspness
from grasppanda.datasets.graspnet1b import GraspNet1B
from grasppanda.structures import Provenance, SampleKey
from grasppanda.transforms.frame import to_batch
from grasppanda.transforms.points import (
    FilterObjects,
    RandomFlip,
    RandomRotation,
    RandomSample,
    SampleGraspLabels,
)

pytestmark = [pytest.mark.dataset, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]


def load_recorder() -> Any:
    """The recorder script as a module: it owns the list of recorded frames."""
    file = REPO / "tools" / "make_data_fixtures.py"
    if not file.is_file():
        pytest.skip(
            "tools/make_data_fixtures.py is not part of this checkout", allow_module_level=True
        )
    spec = importlib.util.spec_from_file_location("make_data_fixtures", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = load_recorder()
#: (split, camera, scene, frame). Upstream serves labels only where an upstream-format
#: graspness file exists; the recorder marks the frames for which it does.
DATASET_FRAMES = [frame[:4] for frame in RECORDER.DATASET_FRAMES]
LABELLED_FRAMES = [frame[:4] for frame in RECORDER.DATASET_FRAMES if frame[4]]
GENERATOR_FRAMES = [
    (camera, scene, frame)
    for camera, scene, frames in RECORDER.GENERATOR_FRAMES
    for frame in frames
]
PREPROCESS = [
    {"type": "DepthToCloud"},
    {"type": "SegmentationBoxWorkspace", "margin": 0.02, "invalid_depth_foreground": "include"},
]
FLOAT32_ULP = float(np.finfo(np.float32).eps)


def digest(array: np.ndarray) -> str:
    """The digest the recorder uses: dtype, shape and bytes."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


@pytest.fixture(scope="session")
def fixtures(graspnet1b_root, asset_cache) -> Path:
    """Directory of recorded upstream values; recorded now if absent and possible."""
    directory = asset_cache / "fixtures" / "data_pipeline"
    wanted = [RECORDER.dataset_fixture(*frame) for frame in DATASET_FRAMES]
    wanted += [RECORDER.generator_fixture(*frame) for frame in GENERATOR_FRAMES]
    index = directory / "index.json"

    def complete() -> bool:
        if not index.is_file() or json.loads(index.read_text()).get("format") != RECORDER.FORMAT:
            return False
        return all((directory / name).is_file() for name in wanted)

    if not complete():
        interpreter = os.environ.get("GRASPPANDA_ORACLE_PYTHON", "")
        upstream = os.environ.get("GRASPPANDA_UPSTREAM_GRASPNESS", "")
        if not Path(interpreter).is_file() or not Path(upstream).is_dir():
            pytest.skip(
                f"no recorded fixtures in {directory}; to record them set "
                "GRASPPANDA_ORACLE_PYTHON and GRASPPANDA_UPSTREAM_GRASPNESS (a pristine "
                "graspness_unofficial checkout) or run tools/make_data_fixtures.py"
            )
        subprocess.run(
            [
                interpreter,
                "-B",
                str(REPO / "tools" / "make_data_fixtures.py"),
                "--upstream",
                upstream,
                "--dataset-root",
                str(graspnet1b_root),
                "--out",
                str(directory),
            ],
            check=True,
        )
    return directory


_datasets: dict[tuple, GraspNet1B] = {}


def dataset(split: str, camera: str, labels: bool) -> GraspNet1B:
    """A reader configured like the reference pipeline; transforms are applied by hand."""
    key = (split, camera, labels)
    if key not in _datasets:
        try:
            _datasets[key] = GraspNet1B(
                camera=camera,
                split=split,
                preprocess=PREPROCESS,
                transforms=[],
                load_labels=labels,
                graspness={"source": "legacy"} if labels else None,
                label_source="grasp_label",
            )
        except AssetError as error:
            pytest.skip(str(error).splitlines()[-1])
    return _datasets[key]


def unpack(packed: np.ndarray, size: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, count=int(size[0] * size[1])).reshape(size).astype(bool)


@pytest.mark.parametrize(("split", "camera", "scene", "frame"), DATASET_FRAMES)
def test_mask_and_masked_cloud_equal_upstream(fixtures, split, camera, scene, frame):
    reference = np.load(fixtures / RECORDER.dataset_fixture(split, camera, scene, frame))
    reader = dataset(split, camera, labels=False)
    key = SampleKey("graspnet1b", scene, camera, frame)
    observed = reader.frame(key)
    size = reference["image_size"]
    assert tuple(size) == reader.image_size == observed.depth.shape
    fx, fy, cx, cy, scale = reference["camera"]
    assert (observed.intrinsics[0, 0], observed.intrinsics[1, 1]) == (fx, fy)
    assert (observed.intrinsics[0, 2], observed.intrinsics[1, 2]) == (cx, cy)
    assert observed.depth_scale == scale
    assert np.allclose(
        observed.table_from_camera, reference["table_from_camera"], rtol=0, atol=1e-15
    )

    assert np.array_equal(observed.valid, unpack(reference["mask"], size))
    workspace = reader.preprocess[1].inside(observed)
    assert np.array_equal(workspace, unpack(reference["workspace_mask"], size))
    masked = observed.cloud[observed.valid]
    assert masked.dtype == np.float64 and np.array_equal(masked, reference["cloud_masked"])
    assert digest(masked) == str(reference["raw_cloud"])  # what upstream's collision filter gets
    assert observed.provenance == {"workspace": Provenance.GROUND_TRUTH}


@pytest.mark.parametrize(("split", "camera", "scene", "frame"), DATASET_FRAMES)
@pytest.mark.parametrize("variant", ["inference", "padded"])
def test_sampled_cloud_equals_upstream(fixtures, split, camera, scene, frame, variant):
    reference = np.load(fixtures / RECORDER.dataset_fixture(split, camera, scene, frame))
    reader = dataset(split, camera, labels=False)
    index = reference[f"{variant}/index"]
    sample = reader.sample(SampleKey("graspnet1b", scene, camera, frame))
    masked = len(sample)
    step = RandomSample(len(index))
    if variant == "padded":  # more points requested than the frame has
        assert len(index) > masked and np.array_equal(index[:masked], np.arange(masked))
        drawn = step.draw(masked, np.random.default_rng(0))
        assert np.array_equal(drawn[:masked], np.arange(masked)) and len(drawn) == len(index)
    else:
        assert len(index) == 15000 < masked and len(np.unique(index)) == len(index)
    batch = to_batch(step.apply(sample, index))
    assert np.array_equal(
        batch.observation.points.coords.numpy(), reference[f"{variant}/point_clouds"]
    )
    assert len(batch.observation.full_points) == masked and batch.supervision is None
    assert bool(reference[f"{variant}/feats_ones"])


@pytest.mark.parametrize(("split", "camera", "scene", "frame"), LABELLED_FRAMES)
@pytest.mark.parametrize("variant", ["plain", "augmented"])
def test_training_sample_equals_upstream(fixtures, split, camera, scene, frame, variant):
    reference = np.load(fixtures / RECORDER.dataset_fixture(split, camera, scene, frame))
    reader = dataset(split, camera, labels=True)
    sample = reader.sample(SampleKey("graspnet1b", scene, camera, frame))
    RandomSample(15000).apply(sample, reference[f"{variant}/index"])
    FilterObjects(50)(sample)
    labels = SampleGraspLabels(0.25, 300, "zero_scores")
    kept = int(reference[f"{variant}/num_label_draws"])
    rows = [reference[f"{variant}/label_rows_{k}"] for k in range(kept)]
    assert len(sample.objects) == kept  # the same objects pass the visibility rule
    assert [labels.count(a.source.num_points) for a in sample.objects] == [len(r) for r in rows]
    labels.apply(sample, rows)
    randoms = reference[f"{variant}/randoms"]
    tolerance = 0.0
    if variant == "augmented":
        flip, angle = randoms[0] > 0.5, randoms[1] * np.pi / 3 - np.pi / 6
        RandomFlip("x", 0.5).apply(sample, bool(flip))
        RandomRotation("x", 30.0).apply(sample, float(angle))
        tolerance = FLOAT32_ULP
    else:
        assert len(randoms) == 0
    batch = to_batch(sample)
    points = batch.observation.points.coords.numpy()
    expected = reference[f"{variant}/point_clouds"]
    assert np.abs(points - expected).max() <= tolerance * np.abs(expected).max()

    supervision = batch.supervision
    graspness = reference[f"{variant}/graspness_label"]
    assert graspness.shape == (15000, 1)
    assert np.array_equal(supervision.point_labels["graspness"].numpy(), graspness[:, 0])
    objectness = supervision.point_labels["objectness"].numpy()
    assert objectness.dtype == np.int64
    assert np.array_equal(objectness, reference[f"{variant}/objectness_label"])
    (objects,) = supervision.objects
    poses = np.stack([o.pose.numpy()[:3] for o in objects])
    assert np.abs(poses - reference[f"{variant}/object_poses"]).max() <= tolerance * 2
    for position, labelled in enumerate(objects):
        for name, ours in (
            ("grasp_points", labelled.points),
            ("grasp_widths", labelled.widths),
            ("grasp_scores", labelled.scores),
        ):
            ours = ours.numpy()
            prefix = f"{variant}/{name}_{position}"
            assert ours.shape == tuple(reference[f"{prefix}_shape"])
            assert np.array_equal(ours[:4], reference[f"{prefix}_head"])
            assert digest(ours) == str(reference[f"{prefix}_sha256"]), prefix
        assert labelled.collision is None
        assert (labelled.scores == 0).any()  # colliding grasps were invalidated


@pytest.mark.parametrize(("camera", "scene", "frame"), GENERATOR_FRAMES)
def test_native_graspness_equals_the_upstream_generator(fixtures, camera, scene, frame):
    reference = np.load(fixtures / RECORDER.generator_fixture(camera, scene, frame))["graspness"]
    assert reference.dtype == np.float64 and reference.shape[1] == 1
    split = "train" if scene < 100 else "test"
    reader = dataset(split, camera, labels=True)
    key = SampleKey("graspnet1b", scene, camera, frame)
    observed = reader.frame(key)
    labels = reader.scene_labels(scene)
    values = [object_graspness(instance, 0.8) for instance in labels]
    poses = reader.object_poses(key, "annotations")
    exact = frame_graspness(observed, poses, labels, values, "float64", workers=-1)
    assert np.array_equal(np.isnan(exact), ~observed.valid)
    assert np.array_equal(exact[observed.valid], reference[:, 0])
    stored = frame_graspness(observed, poses, labels, values, "float16", workers=-1)
    error = np.abs(stored[observed.valid].astype(np.float64) - reference[:, 0]).max()
    assert error <= 2.5e-4


def test_label_store_holds_the_dataset_label_files(graspnet1b_root):
    """Stored arrays against both upstream label variants and the collision file."""
    reader = dataset("train", "realsense", labels=True)
    store = reader.label_store
    for object_id in (14, 15):  # two small objects of scene 0
        points, scores, widths = store.object_arrays(object_id)
        with np.load(
            graspnet1b_root / "grasp_label_simplified" / f"{object_id:03d}_labels.npz"
        ) as f:
            assert np.array_equal(points, f["points"]) and np.array_equal(scores, f["scores"])
            assert np.array_equal(widths, f["width"])
        with np.load(graspnet1b_root / "grasp_label" / f"{object_id:03d}_labels.npz") as f:
            assert np.array_equal(widths, f["offsets"][..., 2])
            assert np.array_equal(scores, f["scores"])
    with np.load(reader.collision_label_file(0)) as collisions:
        for index, object_id in enumerate(reader.scene_object_ids(0)):
            labels = store.labels(0, index, object_id)
            assert np.array_equal(labels.collision(), collisions[f"arr_{index}"])
    manifest = store.asset.manifest()
    assert len(manifest["shards"]) == 88 + 190 and manifest["parameters"]["source"] == "grasp_label"


@pytest.mark.parametrize("camera", ["realsense", "kinect"])
def test_shipped_config_serves_training_batches(graspnet1b_root, camera):
    """The base config, verbatim, on a few frames (needs the native graspness asset)."""
    node = Config.load(GraspNet1B.default_config(camera)).select("data.train.dataset")
    try:
        reader = GraspNet1B(**{**{k: v for k, v in node.items() if k != "type"}, "split": "sample"})
    except AssetError as error:
        pytest.skip(str(error).splitlines()[-1])
    items = [reader[index] for index in (0, 3, 7)]
    batch = reader.collate(items)
    assert batch.observation.points.batch_offsets.tolist() == [0, 15000, 30000, 45000]
    assert batch.observation.point_feats.shape == (45000, 3)
    assert batch.observation.gt_assisted() == ["workspace"]
    labels = batch.supervision.point_labels
    assert set(labels) == {"objectness", "instance", "graspness"}
    graspness = labels["graspness"]
    assert graspness.shape == (45000,) and graspness.min() >= 0 and graspness.max() <= 1
    # graspness is high on objects and low on the table
    assert graspness[labels["objectness"] == 1].mean() > graspness[labels["objectness"] == 0].mean()
    for objects in batch.supervision.objects:
        assert 1 <= len(objects) <= len(reader.scene_object_ids(0))
        for labelled in objects:
            # the shipped config leaves the grids in the label store (`grids: deferred`)
            assert labelled.deferred is not None and labelled.scores is None
            shapes = labelled.shapes()
            assert set(shapes) == {"scores", "widths"}
            assert shapes["scores"][1:] == (300, 12, 4) == shapes["widths"][1:]
            assert len(labelled.points) >= 300  # a quarter of the label points, at least 300
    for labelled in batch.supervision.objects[0]:  # read through the store, they are the grids
        grids = labelled.dense()
        assert grids.scores.shape == grids.widths.shape == (len(labelled.points), 300, 12, 4)
        assert grids.widths.min() >= 0 and grids.scores.max() <= 1.0 + 1e-6
        assert (grids.scores == 0).any()  # grasps that collide in this scene
    records = reader.asset_records()
    assert [r["provenance"] for r in records] == ["native", "native"]
    assert records[1]["uses_ground_truth"] is True
