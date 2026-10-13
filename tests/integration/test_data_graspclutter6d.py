"""GraspClutter6D reader on real frames.

Stated tolerances:

* scene cloud against the official tool (recorded by ``tools/make_data_fixtures.py``):
  the same points in the same order, coordinates within 1e-12 m. The two evaluate the
  pinhole formula in a different order, so the last bits of a float64 may differ.
* object poses against the sensor: for every instance, the median distance of the
  observed points inside its visibility mask to the posed object model. Over the
  instances of three frames its median must stay below 2 mm for the Zivid and 5 mm for
  the RealSense D435, and its 90th percentile below 4 mm and 12 mm. (Measured when the
  test was written, about 40 instances per camera: Zivid 0.9 / 1.9 mm, D435 3.3 /
  8.9 mm; the D435 figures are the accuracy of that sensor at one metre.) This bounds
  pose units, pose direction, intrinsics and depth scale at once.
* object positions in the world frame: constant across the 13 views of a scene to
  0.1 mm, which holds only when ``cam_*_w2c`` is read as the camera pose in the world.
"""

from __future__ import annotations

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
from grasppanda.datasets.graspclutter6d import IMAGE_SIZES, GraspClutter6D
from grasppanda.structures import Provenance, SampleKey
from grasppanda.transforms.frame import transform_points

pytestmark = pytest.mark.dataset

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
#: (camera, scene, view) recorded through the official tool.
OFFICIAL_FRAMES = list(RECORDER.CLUTTER_FRAMES)
PREPROCESS = [{"type": "DepthToCloud"}, {"type": "SegmentationRectWorkspace", "margin": 0.1}]

_readers: dict[tuple, GraspClutter6D] = {}


def reader(camera: str, labels: bool = False, split: str = "all") -> GraspClutter6D:
    key = (camera, labels, split)
    if key not in _readers:
        _readers[key] = GraspClutter6D(camera, split, PREPROCESS, [], load_labels=labels)
    return _readers[key]


@pytest.fixture(scope="session")
def official(graspclutter6d_root, asset_cache) -> Path:
    """Scene clouds recorded from the official tool; recorded now if absent and possible."""
    directory = asset_cache / "fixtures" / "data_pipeline"
    wanted = [directory / RECORDER.clutter_fixture(*frame) for frame in OFFICIAL_FRAMES]
    if not all(file.is_file() for file in wanted):
        interpreter = os.environ.get("GRASPPANDA_ORACLE_PYTHON", "")
        api = os.environ.get("GRASPPANDA_GRASPCLUTTER6D_API", "")
        if not Path(interpreter).is_file() or not Path(api).is_dir():
            pytest.skip(
                f"no recorded official-tool clouds in {directory}; to record them set "
                "GRASPPANDA_ORACLE_PYTHON and GRASPPANDA_GRASPCLUTTER6D_API (a graspclutter6dAPI "
                "checkout) or run tools/make_data_fixtures.py --clutter-api ... --clutter-root ..."
            )
        subprocess.run(
            [
                interpreter,
                "-B",
                str(REPO / "tools" / "make_data_fixtures.py"),
                "--clutter-api",
                api,
                "--clutter-root",
                str(graspclutter6d_root),
                "--out",
                str(directory),
            ],
            check=True,
        )
    return directory


def read_ply_vertices(file: Path) -> np.ndarray:
    """``(V, 3)`` vertex positions of a binary little-endian PLY file."""
    types = {"double": "<f8", "float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    with open(file, "rb") as handle:
        fields, count, in_vertex = [], 0, False
        while True:
            line = handle.readline().decode().strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format":
                assert parts[1] == "binary_little_endian", line
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                count = int(parts[2]) if in_vertex else count
            elif parts[0] == "property" and in_vertex:
                fields.append((parts[2], types[parts[1]]))
        vertices = np.frombuffer(handle.read(count * np.dtype(fields).itemsize), dtype=fields)
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)


def test_manifest_follows_the_split_files(graspclutter6d_root):
    manifest = GraspClutter6D.load_manifest(graspclutter6d_root)
    train, test = manifest.split("train"), manifest.split("test")
    assert len(train.scenes) == 413 and len(test.scenes) == 235
    assert not set(train.scenes) & set(test.scenes)
    assert manifest.frames == tuple(range(13)) and len(manifest.keys("test", "zivid")) == 235 * 13
    dataset = reader("realsense-d435")
    assert len(dataset) == (413 + 235) * 13
    key = SampleKey("graspclutter6d", 2, "realsense-d435", 7)
    assert dataset.image_id(key) == 30


@pytest.mark.oracle
@pytest.mark.parametrize(("camera", "scene", "view"), OFFICIAL_FRAMES)
def test_scene_cloud_equals_the_official_tool(official, camera, scene, view):
    expected = np.load(official / RECORDER.clutter_fixture(camera, scene, view))["points"]
    frame = reader(camera).frame(SampleKey("graspclutter6d", scene, camera, view))
    assert frame.depth.shape == IMAGE_SIZES[camera]
    ours = frame.cloud[frame.valid]
    assert ours.shape == expected.shape  # the same pixels survive the crop
    assert np.abs(ours - expected).max() <= 1e-12
    assert frame.provenance == {"workspace": Provenance.GROUND_TRUTH}


@pytest.mark.parametrize("camera", list(IMAGE_SIZES))
def test_point_cloud_bounds(graspclutter6d_root, camera):
    """Tabletop scenes seen from 0.5 to 1.5 m: metres, whatever the raw depth unit."""
    dataset = reader(camera)
    for scene, view in ((2, 0), (10, 7), (5, 12)):
        frame = dataset.frame(SampleKey("graspclutter6d", scene, camera, view))
        points = frame.cloud[frame.valid]
        assert len(points) > 100_000 and (points[:, 2] > 0).all()
        inside = (points[:, 2] > 0.3) & (points[:, 2] < 2.0) & (np.abs(points[:, :2]) < 1.0).all(1)
        assert inside.mean() > 0.995
        assert 0.5 < np.median(points[:, 2]) < 1.5


@pytest.mark.parametrize(
    ("camera", "typical", "most"), [("realsense-d435", 0.005, 0.012), ("zivid", 0.002, 0.004)]
)
def test_object_poses_place_the_models_on_the_depth_surface(
    graspclutter6d_root, camera, typical, most
):
    from scipy.spatial import cKDTree

    dataset = reader(camera, labels=False)
    models: dict[int, np.ndarray] = {}
    medians = []
    for scene, view in ((2, 0), (10, 7), (5, 5)):
        key = SampleKey("graspclutter6d", scene, camera, view)
        frame = dataset.read_frame(key)
        frame = dataset.preprocess[0](frame)
        instances = dataset._instance_map(key, frame.depth.shape)
        for index, (object_id, pose) in enumerate(dataset.object_poses(key)):
            mask = (instances == index + 1) & frame.valid
            if mask.sum() < 500:
                continue
            if object_id not in models:
                file = graspclutter6d_root / "models_m" / f"obj_{object_id:06d}.ply"
                models[object_id] = read_ply_vertices(file)
            posed = transform_points(models[object_id], pose)
            distance, _ = cKDTree(posed).query(frame.cloud[mask])
            medians.append(float(np.median(distance)))
    medians = np.array(medians)
    assert len(medians) >= 20
    assert np.median(medians) < typical, medians.round(4).tolist()
    assert np.percentile(medians, 90) < most, medians.round(4).tolist()


def test_world_frame_keeps_objects_fixed_across_views(graspclutter6d_root):
    dataset = reader("zivid")
    for scene in (2, 5, 10):
        positions = []
        for view in range(13):
            key = SampleKey("graspclutter6d", scene, "zivid", view)
            frame = dataset.read_frame(key)
            centres = np.stack([pose[:3, 3] for _, pose in dataset.object_poses(key)])
            positions.append(transform_points(centres, frame.table_from_camera))
        spread = np.ptp(np.stack(positions), axis=0).max()
        assert spread < 1e-4, f"scene {scene}: object positions move by {spread} m between views"
        # read literally as world-to-camera, the same annotations make the objects wander
        camera = json.loads((dataset.scene_dir(scene) / "scene_camera.json").read_text())
        literal = []
        for view in range(13):
            key = SampleKey("graspclutter6d", scene, "zivid", view)
            entry = camera[str(dataset.image_id(key))]
            rotation = np.array(entry["cam_R_w2c"]).reshape(3, 3)
            translation = np.array(entry["cam_t_w2c"]) / 1000
            centre = dataset.object_poses(key)[0][1][:3, 3]
            literal.append(rotation.T @ (centre - translation))
        assert np.ptp(np.stack(literal), axis=0).max() > 0.05


def test_instance_keyed_labels_on_a_real_scene(graspclutter6d_root):
    """Scene 5 holds repeated objects; needs `prepare graspclutter6d --labels --split sample`."""
    try:
        dataset = reader("realsense-d435", labels=True, split="sample")
    except AssetError as error:
        pytest.skip(str(error).splitlines()[-1])
    key = dataset.keys[5]
    object_ids = dataset.scene_object_ids(key.scene)
    assert len(set(object_ids)) < len(object_ids)  # at least one model occurs twice
    sample = dataset.sample(key)
    assert [a.object_id for a in sample.objects] == object_ids
    assert [a.instance_id for a in sample.objects] == list(range(1, len(object_ids) + 1))
    instance = sample.point_labels["instance"]
    assert set(np.unique(instance)) <= set(range(len(object_ids) + 1))
    assert np.array_equal(sample.point_labels["objectness"], (instance > 0).astype(np.int64))
    with np.load(dataset.collision_label_file(key.scene)) as collisions:
        for index, annotation in enumerate(sample.objects):
            stored = annotation.source.collision()
            assert np.array_equal(stored, collisions[f"arr_{index}"])
    repeated = next(i for i in object_ids if object_ids.count(i) > 1)
    first, second = [a for a in sample.objects if a.object_id == repeated][:2]
    assert first.source.scores is second.source.scores  # one model, mapped once
    assert not np.array_equal(first.source.collision(), second.source.collision())
    with np.load(dataset.object_label_file(repeated)) as labels:
        assert np.array_equal(first.source.points, labels["points"])
        assert np.array_equal(first.source.scores, labels["scores"])
        assert np.array_equal(first.source.widths, labels["offsets"][..., 2])


def test_shipped_config_serves_test_frames(graspclutter6d_root):
    node = Config.load(GraspClutter6D.default_config()).select("data.test.dataset")
    dataset = GraspClutter6D(**{k: v for k, v in node.items() if k != "type"})
    assert len(dataset) == 235 * 13
    items = [dataset[0], dataset[14]]
    batch = dataset.collate(items)
    assert batch.supervision is None and batch.observation.gt_assisted() == ["workspace"]
    assert batch.observation.points.batch_offsets.tolist() == [0, 15000, 30000]
    assert batch.observation.intrinsics.shape == (2, 3, 3)
    sizes = batch.observation.full_points.sizes.tolist()
    assert all(size > 100_000 for size in sizes) and sizes[0] != sizes[1]
    again = dataset[0]  # seed 0: evaluation sees the same points every time
    assert (again.observation.points.coords == items[0].observation.points.coords).all()
