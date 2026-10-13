"""Parity gate: the clean-room GraspNet-1Billion evaluator against the official one.

Reference values come from ``tools/make_eval_fixtures.py``, which runs the official
``graspnetAPI`` in its own environment and stores, per frame, every stage of the
evaluation. Three kinds of tests use them:

* ``test_committed_frames_*`` needs only the dataset. It replays 24 frames whose
  official results are kept under ``$GRASPPANDA_CACHE/fixtures/eval_graspnet1b`` (168 KiB).
* ``test_recorded_*`` replays everything recorded under
  ``$GRASPPANDA_CACHE/graspnet1b/eval_fixtures/v1``: whole scenes of a detector dump,
  grasps drawn from the ground-truth labels (both cameras), constructed edge cases,
  and stage-level references (NMS extension, Open3D, signed-distance queries, poses).
  Missing recordings are produced on the spot when the oracle interpreter is
  configured (minutes per scene), otherwise the test is skipped with the command.
* ``test_dataset_*`` checks the facts about the dataset that the evaluator relies on.

Criterion (``docs/design/ARCHITECTURE.md`` §11.2): for tie-free predictions every
stage must be *equal* (selection after NMS, object assignment, grasps kept by the
cuts, collision and emptiness masks, friction scores, the ``(50, 6)`` table). A frame
whose prediction has tied scores has no single official result (the official sorts
are unstable); such frames are counted and reported, not compared.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from grasppanda.evaluation import prediction_path
from grasppanda.evaluation.graspnet1b import GRASPNET1B_SV_V1, GraspNet1BEvaluator, ObjectCache
from grasppanda.evaluation.graspnet1b.frame import evaluate_frame, prepare_prediction
from grasppanda.evaluation.graspnet1b.nms import grasp_nms
from grasppanda.evaluation.graspnet1b.poses import read_annotation
from grasppanda.evaluation.graspnet1b.scene import (
    SceneGroundTruth,
    scene_directory,
    scene_object_ids,
)
from grasppanda.structures import SampleKey
from grasppanda.utils.paths import cache_root

pytestmark = pytest.mark.dataset

PROTOCOL = GRASPNET1B_SV_V1
REPO = Path(__file__).resolve().parents[2]
RECORDER = REPO / "tools" / "make_eval_fixtures.py"
#: The small reference set (24 frames). It is derived from the dataset and from a
#: non-commercial checkpoint's predictions, so it lives in the cache, not in the repository.
SMALL_REFERENCE = "fixtures/eval_graspnet1b"
STAGES = ("selected", "owner", "kept", "kept_owner", "collision", "empty", "friction", "precision")
#: A detector dump in the official layout, needed only to (re)record the dump suites.
DUMP_VARIABLE = "GRASPPANDA_ORACLE_DUMP_KINECT"
LABEL_FRAMES = ("0", "32", "64", "96", "128", "160", "192", "224")
LABEL_FRAMES_B = ("5", "37", "69", "101", "133", "165", "197", "229")

#: (suite, scene, camera) -> recorder arguments. Seen: 100-129, similar: 130-159, novel: 160-189.
RECORDINGS: dict[tuple[str, int, str], tuple[str, ...]] = {
    ("dump", 100, "kinect"): ("dump", "--camera", "kinect", "--scenes", "100"),
    ("dump", 115, "kinect"): ("dump", "--camera", "kinect", "--scenes", "115"),
    ("dump", 130, "kinect"): ("dump", "--camera", "kinect", "--scenes", "130"),
    ("dump", 145, "kinect"): ("dump", "--camera", "kinect", "--scenes", "145"),
    ("dump", 160, "kinect"): ("dump", "--camera", "kinect", "--scenes", "160"),
    ("dump", 170, "kinect"): ("dump", "--camera", "kinect", "--scenes", "170"),
    ("edge", 112, "kinect"): ("edge", "--camera", "kinect", "--scene", "112"),
    ("label", 105, "realsense"): ("label", "--camera", "realsense", "--scenes", "105", "--frames", *LABEL_FRAMES),
    ("label", 135, "realsense"): ("label", "--camera", "realsense", "--scenes", "135", "--frames", *LABEL_FRAMES),
    ("label", 165, "realsense"): ("label", "--camera", "realsense", "--scenes", "165", "--frames", *LABEL_FRAMES),
    ("label", 121, "kinect"): ("label", "--camera", "kinect", "--scenes", "121", "--frames", *LABEL_FRAMES_B, "--count", "3000", "--friction", "1.0"),
    ("label", 152, "kinect"): ("label", "--camera", "kinect", "--scenes", "152", "--frames", *LABEL_FRAMES_B, "--count", "3000", "--friction", "1.0"),
    ("label", 183, "kinect"): ("label", "--camera", "kinect", "--scenes", "183", "--frames", *LABEL_FRAMES_B, "--count", "3000", "--friction", "1.0"),
}  # fmt: skip
PRIMITIVE_OBJECTS = (0, 9, 18, 62)
PRIMITIVE_SCENES = (100, 145)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def recorded_root(asset_cache: Path) -> Path:
    return asset_cache / "graspnet1b" / "eval_fixtures" / "v1"


@pytest.fixture(scope="session")
def object_cache(graspnet1b_root: Path, asset_cache: Path) -> ObjectCache:
    return ObjectCache(
        graspnet1b_root, model_voxel_size=PROTOCOL.model_voxel_size, root=asset_cache
    )


@pytest.fixture(scope="session")
def record(request, graspnet1b_root: Path, recorded_root: Path):
    """Return a function that makes sure a recording exists, running the oracle if needed."""

    def ensure(marker: Path, arguments: tuple[str, ...]) -> None:
        if marker.exists():
            return
        command = [str(RECORDER), "--data-root", str(graspnet1b_root), "--out", str(recorded_root)]
        command += list(arguments)
        if arguments[0] in ("dump", "edge"):
            dump = os.environ.get(DUMP_VARIABLE, "")
            if not Path(dump).is_dir():
                pytest.skip(
                    f"{marker} is not recorded and ${DUMP_VARIABLE} does not name a prediction "
                    f"dump to record it from. Command: $GRASPPANDA_ORACLE_PYTHON {' '.join(command)}"
                    " --dump <dump>"
                )
            command += ["--dump", dump]
        interpreter = request.getfixturevalue("oracle_python")  # skips when not configured
        done = subprocess.run([interpreter, *command], capture_output=True, text=True)
        if done.returncode != 0 or not marker.exists():
            pytest.fail(f"oracle run failed: {' '.join(command)}\n{done.stderr[-3000:]}")

    return ensure


def compare(result, grasps: np.ndarray, reference) -> dict[str, bool]:
    """Stage-by-stage equality of one frame with recorded official arrays."""
    equal = {
        "selected": np.array_equal(grasps[result.selected], reference["nms"]),
        "owner": np.array_equal(result.owner, reference["owner"]),
        "kept": np.array_equal(grasps[result.kept], reference["kept"]),
        "kept_owner": np.array_equal(result.kept_owner, reference["kept_owner"]),
        "collision": np.array_equal(result.collision, reference["collision"]),
        "empty": np.array_equal(result.empty, reference["empty"]),
        "friction": np.array_equal(result.friction, reference["friction"]),
        "precision": np.array_equal(result.precision, reference["precision"]),
    }
    assert tuple(equal) == STAGES
    return equal


def lexsorted(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


# ------------------------------------------------------------ committed reference


def small_reference() -> Path:
    directory = cache_root() / SMALL_REFERENCE
    if not (directory / "official_frames.npz").is_file():
        pytest.skip(f"no small reference set at {directory}; record it with {RECORDER.name} pack")
    return directory


def committed_frames() -> list[tuple[int, str, int]]:
    provenance = json.loads((small_reference() / "official_frames.json").read_text())
    return [
        (scene, provenance["camera"], frame)
        for scene in provenance["scenes"]
        for frame in provenance["frames"]
    ]


def test_committed_frames_match_the_official_evaluator(graspnet1b_root, object_cache):
    """24 frames of seen, similar and novel scenes: every stage equals the official tool."""
    frames = committed_frames()
    assert len({scene for scene, _, _ in frames}) >= 3 and len(frames) >= 24
    truths: dict[tuple[int, str], SceneGroundTruth] = {}
    kept_total = held_total = 0
    with np.load(small_reference() / "official_frames.npz") as packed:
        for scene, camera, frame in frames:
            if (scene, camera) not in truths:
                object_cache.ensure(scene_object_ids(graspnet1b_root, scene, camera))
                truths[scene, camera] = SceneGroundTruth(
                    graspnet1b_root, scene, camera, object_cache, PROTOCOL
                )
            truth = truths[scene, camera]
            prefix = f"{scene:04d}/{camera}/{frame:04d}/"
            prediction = packed[prefix + "prediction"]
            assert len(np.unique(prediction[:, 0])) == len(prediction)  # tie-free
            result = evaluate_frame(prediction, truth.objects, *truth.frame(frame), PROTOCOL)
            for stage in STAGES:
                mine, official = getattr(result, stage), packed[prefix + stage]
                assert np.array_equal(mine, official), f"{prefix}{stage}"
            kept_total += len(result.kept)
            held_total += int((result.friction > 0).sum())
    assert kept_total > 900 and held_total > 250  # the frames exercise the success path


def test_committed_frames_through_the_registered_evaluator(graspnet1b_root, asset_cache, tmp_path):
    evaluator = GraspNet1BEvaluator(data_root=str(graspnet1b_root), cache_dir=str(asset_cache))
    keys, official = [], {}
    with np.load(small_reference() / "official_frames.npz") as packed:
        for scene, camera, frame in committed_frames():
            key = SampleKey("graspnet1b", scene, camera, frame)
            prefix = f"{scene:04d}/{camera}/{frame:04d}/"
            path = prediction_path(tmp_path, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, packed[prefix + "prediction"])
            keys.append(key)
            official[key] = packed[prefix + "precision"]
    result = evaluator.evaluate(tmp_path, keys, num_workers=3)
    assert result.complete and result.missing == []
    for key in keys:
        assert np.array_equal(result.precision[key], official[key]), str(key)
    stack = np.stack([official[key] for key in keys])
    assert result.metrics["AP"] == pytest.approx(stack.mean(), rel=1e-13)
    assert result.metrics["AP0.8"] == pytest.approx(stack[:, :, 3].mean(), rel=1e-13)
    assert result.metrics["AP0.4"] == pytest.approx(stack[:, :, 1].mean(), rel=1e-13)


# -------------------------------------------------------------- recorded stages


@pytest.mark.oracle
@pytest.mark.slow
@pytest.mark.parametrize("suite,scene,camera", list(RECORDINGS), ids=lambda value: str(value))
def test_recorded_frames_match_stage_by_stage(
    suite, scene, camera, graspnet1b_root, object_cache, recorded_root, record, record_property
):
    folder = recorded_root / suite / f"scene_{scene:04d}" / camera
    record(folder / "run.json", RECORDINGS[suite, scene, camera])
    files = sorted(folder.glob("[0-9]*.npz"))
    assert len(files) >= 8

    object_cache.ensure(scene_object_ids(graspnet1b_root, scene, camera))
    truth = SceneGroundTruth(graspnet1b_root, scene, camera, object_cache, PROTOCOL)
    counts: Counter[str] = Counter()
    mismatches, tied = [], []
    for file in files:
        frame = int(file.stem)
        with np.load(file) as reference:
            prediction = reference["prediction"]
            poses, table = truth.frame(frame)
            assert np.array_equal(np.array(poses), reference["poses"]), f"poses of frame {frame}"
            result = evaluate_frame(prediction, truth.objects, poses, table, PROTOCOL)
            grasps = prepare_prediction(prediction, PROTOCOL.gripper.max_width)
            equal = compare(result, grasps, reference)
            duplicates = len(prediction) - len(np.unique(prediction, axis=0))
            if len(np.unique(prediction[:, 0])) + duplicates != len(prediction):
                tied.append(frame)  # tied scores between different grasps: no unique reference
                continue
            if not all(equal.values()):
                mismatches.append((frame, [stage for stage, ok in equal.items() if not ok]))
            counts["frames"] += 1
            counts["grasps_in"] += len(prediction)
            counts["after_nms"] += len(reference["nms"])
            counts["kept"] += len(reference["kept"])
            counts["colliding_or_empty"] += int(reference["collision"].sum())
            counts["force_closure"] += int((reference["friction"] > 0).sum())
    summary = {"suite": suite, "scene": scene, "camera": camera, "tied_frames": tied, **counts}
    record_property("parity", json.dumps(summary))
    print(f"\nparity {json.dumps(summary)}")
    assert mismatches == []
    assert counts["frames"] >= 8 and len(tied) <= 2


@pytest.mark.oracle
@pytest.mark.slow
def test_recorded_scenes_through_the_registered_evaluator(
    graspnet1b_root, asset_cache, recorded_root, record, tmp_path
):
    """Dump, label and edge recordings of both cameras, scored by worker processes."""
    keys, official = [], {}
    for (suite, scene, camera), arguments in RECORDINGS.items():
        folder = recorded_root / suite / f"scene_{scene:04d}" / camera
        record(folder / "run.json", arguments)
        for file in sorted(folder.glob("[0-9]*.npz"))[:: 4 if suite == "dump" else 1]:
            with np.load(file) as reference:
                prediction = reference["prediction"]
                if len(np.unique(prediction[:, 0])) != len(prediction):
                    continue
                key = SampleKey("graspnet1b", scene, camera, int(file.stem))
                path = prediction_path(tmp_path, key)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, prediction)
                keys.append(key)
                official[key] = reference["precision"]
    evaluator = GraspNet1BEvaluator(data_root=str(graspnet1b_root), cache_dir=str(asset_cache))
    result = evaluator.evaluate(tmp_path, keys, num_workers=4)
    assert result.complete and len(keys) > 200
    assert {key.camera for key in keys} == {"kinect", "realsense"}
    different = [
        str(key) for key in keys if not np.array_equal(result.precision[key], official[key])
    ]
    assert different == []
    stack = np.stack([official[key] for key in keys])
    assert result.metrics["AP"] == pytest.approx(stack.mean(), rel=1e-13)
    assert result.info["peak_rss_mb"] < 2048  # a worker holds one scene, not gigabytes of pickles
    print(f"\nevaluator info {json.dumps({k: result.info[k] for k in ('seconds', 'peak_rss_mb', 'start_rss_mb', 'frames_scored')})}")  # fmt: skip


# ---------------------------------------------------------- recorded primitives


@pytest.fixture(scope="session")
def primitives(recorded_root: Path, record) -> Path:
    arguments = ("primitives", "--objects", *map(str, PRIMITIVE_OBJECTS))
    arguments += ("--scenes", *map(str, PRIMITIVE_SCENES))
    record(recorded_root / "primitives" / "run.json", arguments)
    return recorded_root / "primitives"


@pytest.mark.oracle
def test_recorded_nms_equals_the_compiled_extension(primitives):
    """Random sets up to 3333 grasps, with pairs placed on both thresholds."""
    with np.load(primitives / "nms.npz") as reference:
        cases = sorted(name[:-3] for name in reference.files if name.endswith("_in"))
        assert len(cases) == 5
        for case in cases:
            grasps, official = reference[f"{case}_in"], reference[f"{case}_out"]
            keep = grasp_nms(
                grasps[:, 13:16],
                grasps[:, 4:13].reshape(-1, 3, 3),
                grasps[:, 0],
                translation_threshold=PROTOCOL.nms_translation,
                rotation_threshold=PROTOCOL.nms_rotation,
            )
            assert np.array_equal(grasps[keep], official), case
        assert len(official) < len(grasps) > 3000  # the largest case does suppress


@pytest.mark.oracle
@pytest.mark.parametrize("object_id", PRIMITIVE_OBJECTS)
def test_recorded_object_queries_are_bit_identical(object_id, primitives, object_cache):
    asset = object_cache.load(object_id)
    sdf = asset.sdf
    with np.load(primitives / f"object_{object_id:03d}.npz") as reference:
        # inside, last cell, lattice points, near the surface, outside the grid, corner cases
        assert np.array_equal(sdf.signed_distance(reference["coords"]), reference["values"])
        assert np.array_equal(sdf.object_to_grid(reference["points"]), reference["to_grid"])
        assert np.array_equal(sdf.grid_to_object(reference["coords"][:200]), reference["to_object"])
        assert np.array_equal(sdf.direction_to_object(reference["unit"]), reference["directions"])
        assert sdf.surface_threshold == float(reference["surface_threshold"])
        assert sdf.object_to_grid_scale == float(reference["object_to_grid_scale"])
        assert np.array_equal(sdf.surface_sample, reference["surface_sample"])
        # Open3D returns voxels in hash order; the set of points must be identical
        assert np.array_equal(lexsorted(asset.model_points), lexsorted(reference["model_points"]))
        # Normals come from an SVD: equal up to sign and LAPACK rounding. The sign cannot
        # change a score (see SdfGrid.surface_normals).
        normals = sdf.surface_normals(reference["normal_at"])
        official = reference["normals"]
        assert len(normals) > 300
        sign = np.sign((normals * official).sum(axis=1, keepdims=True))
        assert np.abs(normals - sign * official).max() < 1e-12


@pytest.mark.oracle
@pytest.mark.parametrize("scene", PRIMITIVE_SCENES)
@pytest.mark.parametrize("camera", ["kinect", "realsense"])
def test_recorded_poses_and_table(scene, camera, primitives, graspnet1b_root, object_cache):
    with np.load(primitives / f"scene_{scene:04d}_{camera}.npz") as reference:
        object_cache.ensure(reference["object_ids"].tolist())
        truth = SceneGroundTruth(graspnet1b_root, scene, camera, object_cache, PROTOCOL)
        assert truth.object_ids == reference["object_ids"].tolist()
        for frame in range(PROTOCOL.frames_per_scene):
            poses, table = truth.frame(frame)
            assert np.array(poses).dtype == np.float32
            assert np.array_equal(np.array(poses), reference["poses"][frame]), frame  # bit for bit
            if f"table_{frame:04d}" in reference.files:
                official = reference[f"table_{frame:04d}"]
                # The float64 matrix inverse comes from LAPACK and differs in the last bit
                # between NumPy builds (measured: at most 3.4e-16 m between 1.24 and 2.5).
                assert np.abs(lexsorted(table) - lexsorted(official)).max() < 1e-14


# ------------------------------------------------------------------------ dataset


def test_dataset_frames_list_the_objects_of_the_first_frame(graspnet1b_root):
    """Why a deviating frame is an error: in the released test scenes none deviates."""
    checked = 0
    for scene in range(100, 190):
        for camera in ("kinect", "realsense"):
            reference = scene_object_ids(graspnet1b_root, scene, camera)
            assert len(set(reference)) == len(reference) >= 1
            annotations = scene_directory(graspnet1b_root, scene, camera) / "annotations"
            for frame in (1, 77, 128, 255):
                entries = read_annotation(annotations / f"{frame:04d}.xml")
                assert [entry.object_id for entry in entries] == reference, (scene, camera, frame)
                checked += 1
    assert checked == 720


def test_dataset_pickled_grid_equals_the_sdf_file(graspnet1b_root, object_cache):
    """Only the surface sample is unique to the pickles; the grid itself is ``textured.sdf``."""
    asset = object_cache.load(18)  # the smallest pickle
    with (graspnet1b_root / "models" / "018" / "textured.sdf").open() as stream:
        shape = tuple(int(value) for value in stream.readline().split())
        origin = np.array(stream.readline().split(), dtype=np.float64)
        resolution = float(stream.readline())
        values = np.array(stream.read().split(), dtype=np.float64)
    grid = values.reshape(shape[::-1]).transpose(2, 1, 0)  # the file runs x fastest
    assert np.array_equal(grid, asset.sdf.values)
    assert np.array_equal(origin, asset.sdf.grid_to_object_translation)
    assert resolution == asset.sdf.grid_to_object_scale
    sample = asset.sdf.surface_sample
    assert sample.shape == (1000, 3)
    assert asset.sdf.on_surface(asset.sdf.values[tuple(sample.T)]).all()
    object_cache.verify(18)
