"""The GraspClutter6D evaluator: what it shares with GraspNet-1Billion and where it differs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from grasppanda.evaluation import prediction_path, save_prediction
from grasppanda.evaluation.graspclutter6d import (
    CAMERAS,
    GRASPCLUTTER6D_SOURCES,
    GRASPCLUTTER6D_SV_V1,
    ClutterGroundTruth,
    GraspClutter6DEvaluator,
    export_official_layout,
    image_id,
    prediction_file,
)
from grasppanda.evaluation.graspnet1b import (
    GRASPNET1B_SV_V1,
    EvaluationError,
    ObjectCache,
    SceneEvaluator,
)
from grasppanda.registry import EVALUATORS, component_parameters
from grasppanda.structures import GraspGroup, SampleKey

from .eval_synthetic import CLUTTER_INSTANCES, top_grasp, write_clutter_dataset

FRAMES = 2
SPHERE_A, SPHERE_B, BOX = (centre for _, centre in CLUTTER_INSTANCES)


def key(frame: int = 0, camera: str = "zivid", scene: int = 0) -> SampleKey:
    return SampleKey("graspclutter6d", scene, camera, frame)


def write(root, sample: SampleKey, rows, locate=prediction_path) -> None:
    path = locate(root, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows, dtype=np.float32).reshape(-1, 17))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_clutter_dataset(tmp_path_factory.mktemp("clutter"), scenes=(0, 5), frames=FRAMES)


@pytest.fixture(scope="module")
def evaluator(dataset, tmp_path_factory) -> GraspClutter6DEvaluator:
    return GraspClutter6DEvaluator(
        data_root=str(dataset), cache_dir=str(tmp_path_factory.mktemp("cache"))
    )


def friction_of(evaluator, tmp_path, rows) -> np.ndarray:
    """Precision at rank 1 for every friction: tells whether the single best grasp holds."""
    write(tmp_path, key(), rows)
    return evaluator.evaluate(tmp_path, [key()]).precision[key()][0]


# ------------------------------------------------------------------ protocol, layout


def test_protocol_differs_from_graspnet_exactly_where_documented():
    clutter, graspnet = GRASPCLUTTER6D_SV_V1, GRASPNET1B_SV_V1
    assert clutter.name == "graspclutter6d-sv-v1" == GraspClutter6DEvaluator.protocol
    assert (clutter.gripper.max_width, graspnet.gripper.max_width) == (0.14, 0.1)
    assert (clutter.table, clutter.pre_nms_max_distance) == (None, 0.05)
    assert clutter.normalize_closing_axis and not graspnet.normalize_closing_axis
    assert (clutter.frames_per_scene, graspnet.frames_per_scene) == (13, 256)
    same = ("top_k", "frictions", "reported", "nms_translation", "nms_rotation_deg",
            "model_voxel_size", "per_object_top", "crop_margin", "empty_threshold")  # fmt: skip
    assert all(getattr(clutter, name) == getattr(graspnet, name) for name in same)
    for part in ("finger_width", "finger_height", "depth_base"):
        assert getattr(clutter.gripper, part) == getattr(graspnet.gripper, part)


def test_images_interleave_four_cameras():
    assert [image_id(camera, 0) for camera in CAMERAS] == [1, 2, 3, 4]
    assert image_id("zivid", 12) == 52 and image_id("realsense-d415", 12) == 49
    path = prediction_file("/dump", key(frame=3, camera="azure-kinect", scene=42))
    assert str(path) == "/dump/000042/azure-kinect/000015.npy"
    with pytest.raises(ValueError, match="unknown GraspClutter6D camera 'kinect'"):
        image_id("kinect", 0)
    with pytest.raises(ValueError, match="negative"):
        image_id("zivid", -1)


def test_reads_the_dump_the_engine_writes_and_on_request_the_official_layout(
    dataset, tmp_path_factory
):
    """`grasppanda test` writes scene_NNNN/<camera>/<view>; the official tool reads
    <scene, 6 digits>/<camera>/<image id, 6 digits>. The default must be the former."""
    cache = str(tmp_path_factory.mktemp("cache"))
    ours, theirs, linked = (tmp_path_factory.mktemp(name) for name in ("ours", "theirs", "link"))
    keys = [key(frame=1, camera="azure-kinect"), key(frame=0, camera="zivid", scene=5)]
    rows = np.stack([top_grasp(SPHERE_A, score=0.9, width=0.08)]).astype(np.float32)
    for sample in keys:  # exactly what the engine does with a detector's output
        save_prediction(ours, GraspGroup.from_graspnet_array(rows, key=sample))
        write(theirs, sample, rows, locate=prediction_file)
    assert (ours / "scene_0000" / "azure-kinect" / "0001.npy").is_file()
    assert (theirs / "000000" / "azure-kinect" / "000007.npy").is_file()

    default = GraspClutter6DEvaluator(data_root=str(dataset), cache_dir=cache)
    official = GraspClutter6DEvaluator(data_root=str(dataset), cache_dir=cache, layout="official")
    assert default.layout == "grasppanda"
    expected = default.evaluate(ours, keys)
    assert expected.complete and expected.metrics["AP"] > 0
    read_official = official.evaluate(theirs, keys, num_workers=2)
    assert read_official.complete
    for sample in keys:
        assert np.array_equal(read_official.precision[sample], expected.precision[sample])
    # each layout is invisible to the other reader: missing frames, never zeros
    assert default.evaluate(theirs, keys).missing == keys
    assert official.evaluate(ours, keys).missing == keys

    absent = key(frame=1, camera="zivid")  # never predicted: not linked, still missing
    assert export_official_layout(ours, linked, [*keys, absent]) == 2
    assert (linked / "000005" / "zivid" / "000004.npy").is_symlink()
    assert export_official_layout(ours, linked, keys) == 2  # repeatable
    relinked = official.evaluate(linked, [*keys, absent])
    assert relinked.missing == [absent]
    assert np.array_equal(relinked.precision[keys[0]], expected.precision[keys[0]])
    with pytest.raises(ValueError, match="layout must be one of"):
        GraspClutter6DEvaluator(data_root=str(dataset), cache_dir=cache, layout="bop")


def test_is_a_registered_evaluator_sharing_the_scene_machinery(dataset, tmp_path):
    assert EVALUATORS.get("GraspClutter6DEvaluator") is GraspClutter6DEvaluator
    assert issubclass(GraspClutter6DEvaluator, SceneEvaluator)
    parameters, var_kwargs = component_parameters(GraspClutter6DEvaluator)
    assert not var_kwargs and all(p.annotation is not p.empty for p in parameters.values())
    built = EVALUATORS.build(
        {"type": "GraspClutter6DEvaluator", "data_root": str(dataset), "cache_dir": str(tmp_path)}
    )
    assert built.data_root == dataset


def test_environment_variable_names_the_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("GRASPPANDA_GRASPCLUTTER6D_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(EvaluationError, match="GRASPPANDA_GRASPCLUTTER6D_ROOT"):
        GraspClutter6DEvaluator(cache_dir=str(tmp_path)).evaluate(tmp_path, [key()])


# -------------------------------------------------------------------- ground truth


def test_ground_truth_keeps_repeated_instances_and_float64_poses(dataset, tmp_path):
    assert ClutterGroundTruth.list_object_ids(dataset, 0, "zivid") == [1, 1, 2]
    cache = ObjectCache(
        dataset, model_voxel_size=0.008, sources=GRASPCLUTTER6D_SOURCES, root=tmp_path
    )
    assert cache.directory.parent.parent.name == "graspclutter6d"
    truth = ClutterGroundTruth(dataset, 0, "realsense-d435", cache, GRASPCLUTTER6D_SV_V1)
    assert truth.object_ids == [1, 1, 2] and truth.objects[0] is truth.objects[1]
    assert sorted(p.name for p in (cache.directory / "objects").glob("*.npz")) == [
        "000001.npz", "000002.npz",
    ]  # fmt: skip
    poses, table = truth.frame(1)
    assert table is None and all(pose.dtype == np.float64 for pose in poses)
    assert np.allclose(poses[1][:3, 3], SPHERE_B) and np.array_equal(poses[2][:3, :3], np.eye(3))
    with pytest.raises(ValueError, match="no image 10"):
        truth.frame(FRAMES)


def test_a_frame_with_other_objects_is_an_error(tmp_path):
    broken = write_clutter_dataset(tmp_path / "data", frames=FRAMES)
    path = broken / "scenes" / "000000" / "scene_gt.json"
    scene_gt = json.loads(path.read_text())
    scene_gt["8"] = scene_gt["8"][::-1]  # zivid, frame 1
    path.write_text(json.dumps(scene_gt))
    evaluator = GraspClutter6DEvaluator(data_root=str(broken), cache_dir=str(tmp_path / "cache"))
    write(tmp_path, key(1), [top_grasp(SPHERE_A, score=0.9, width=0.08)])
    with pytest.raises(EvaluationError, match=r"zivid/0001: image 8 lists objects \[2, 1, 1\]"):
        evaluator.evaluate(tmp_path, [key(1)])


# ------------------------------------------------------------------------ scoring


def test_scores_every_instance_and_reports_missing_frames(evaluator, tmp_path):
    rows = [
        top_grasp(SPHERE_A, score=0.9, width=0.08),
        top_grasp(SPHERE_B, score=0.8, width=0.08),  # the second instance of the same object
        top_grasp(BOX, score=0.7, width=0.06, closing=(1.0, 0.0, 0.0)),
        top_grasp(SPHERE_A, score=0.6, width=0.04, closing=(1.0, 0.0, 0.0)),  # collides
    ]
    for camera in CAMERAS:
        write(tmp_path, key(camera=camera), rows)
    keys = [key(camera=camera) for camera in CAMERAS] + [key(frame=1)]
    result = evaluator.evaluate(tmp_path, keys, num_workers=2)
    assert result.missing == [key(frame=1)] and len(result.precision) == 4
    expected = np.repeat((np.cumsum([1, 1, 1] + [0] * 47) / np.arange(1, 51))[:, None], 6, axis=1)
    for camera in CAMERAS:  # identical poses for every camera in the toy dataset
        assert np.array_equal(result.precision[key(camera=camera)], expected)
    assert result.info["evaluator"] == "GraspClutter6DEvaluator"
    assert result.info["protocol"] == "graspclutter6d-sv-v1"
    assert result.info["constants"]["gripper"]["max_width"] == 0.14


def test_background_grasps_are_dropped_before_suppression(evaluator, tmp_path):
    """A high-scoring grasp far from every object must neither rank nor suppress."""
    good = top_grasp(SPHERE_A, score=0.5, width=0.08)
    far = top_grasp(SPHERE_A + np.array([0.0, 0.3, 0.0]), score=0.9, width=0.08)
    assert friction_of(evaluator, tmp_path, [far, good]).tolist() == [1.0] * 6
    assert friction_of(evaluator, tmp_path, [far]).tolist() == [0.0] * 6
    # 6 cm above the sphere's top: further than 5 cm from every model point
    hovering = top_grasp(SPHERE_A - np.array([0.0, 0.0, 0.09]), score=0.9, width=0.08)
    assert friction_of(evaluator, tmp_path, [hovering, good]).tolist() == [1.0] * 6


def test_closing_axis_is_normalised_instead_of_rejected(evaluator, tmp_path):
    skewed = top_grasp(SPHERE_A, score=0.9, width=0.08)
    skewed[4:13].reshape(3, 3)[:, 1] *= 1.01
    assert friction_of(evaluator, tmp_path, [skewed]).tolist() == [1.0] * 6


def test_width_limit_is_fourteen_centimetres(evaluator, tmp_path):
    """Widths are clipped to 0.14 m, not to GraspNet's 0.10 m and not left alone.

    The grasp closes along x on the middle sphere. Unclipped, at 0.20 m, its fingers
    would sit at |x| in (0.10, 0.11), inside the neighbouring sphere and the box;
    clipped to 0.14 m they sit at (0.07, 0.08), in free space.
    """
    across = top_grasp(SPHERE_B, score=0.9, width=0.2, closing=(1.0, 0.0, 0.0))
    assert friction_of(evaluator, tmp_path, [across]).tolist() == [1.0] * 6
