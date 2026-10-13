"""The registered evaluator on a synthetic dataset: results, coverage, errors, workers."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.evaluation import EvaluationResult, Evaluator, prediction_path, save_prediction
from grasppanda.evaluation.graspnet1b import (
    EVALUATOR_VERSION,
    GRASPNET1B_SV_V1,
    EvaluationError,
    GraspNet1BEvaluator,
)
from grasppanda.registry import EVALUATORS, component_parameters
from grasppanda.structures import GraspGroup, SampleKey

from .eval_synthetic import BOX_CENTRE, SPHERE_CENTRE, annotation_xml, top_grasp, write_dataset

FRAMES = 3


def key(frame: int, scene: int = 0, camera: str = "kinect") -> SampleKey:
    return SampleKey("graspnet1b", scene, camera, frame)


def toy_prediction() -> np.ndarray:
    """Ranks 1 and 3 hold their object at every friction; ranks 2 and 4 fail."""
    rows = [
        top_grasp(SPHERE_CENTRE, score=0.9, width=0.08),
        top_grasp(SPHERE_CENTRE, score=0.8, width=0.04, closing=(1.0, 0.0, 0.0)),
        top_grasp(BOX_CENTRE, score=0.7, width=0.06, closing=(1.0, 0.0, 0.0)),
        top_grasp(np.array([0.0, 0.25, 0.3]), score=0.6, width=0.08),
    ]
    return np.array(rows, dtype=np.float32)


def toy_table() -> np.ndarray:
    hits = np.zeros(50)
    hits[[0, 2]] = 1
    return np.repeat((np.cumsum(hits) / np.arange(1, 51))[:, None], 6, axis=1)


def write(root, sample: SampleKey, array: np.ndarray) -> None:
    path = prediction_path(root, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("data"), scenes=(0, 1), frames=FRAMES)


@pytest.fixture(scope="module")
def evaluator(dataset, tmp_path_factory) -> GraspNet1BEvaluator:
    return GraspNet1BEvaluator(
        data_root=str(dataset), cache_dir=str(tmp_path_factory.mktemp("cache"))
    )


@pytest.fixture
def dump(tmp_path):
    """Frame 0: four grasps; frame 1: an empty prediction; frame 2: no file."""
    write(tmp_path, key(0), toy_prediction())
    write(tmp_path, key(1), np.zeros((0, 17), dtype=np.float32))
    return tmp_path


# ------------------------------------------------------------------- component


def test_is_a_registered_evaluator_with_checkable_parameters(dataset, tmp_path):
    assert EVALUATORS.get("GraspNet1BEvaluator") is GraspNet1BEvaluator
    assert issubclass(GraspNet1BEvaluator, Evaluator)
    assert GraspNet1BEvaluator.protocol == "graspnet1b-sv-v1" == GRASPNET1B_SV_V1.name
    parameters, var_kwargs = component_parameters(GraspNet1BEvaluator)
    assert not var_kwargs
    assert all(p.annotation is not p.empty for p in parameters.values())
    built = EVALUATORS.build(
        {"type": "GraspNet1BEvaluator", "data_root": str(dataset), "cache_dir": str(tmp_path)}
    )
    assert isinstance(built, GraspNet1BEvaluator) and built.data_root == dataset
    with pytest.raises(ValueError, match="scene_timeout must be positive"):
        GraspNet1BEvaluator(scene_timeout=0.0)


def test_locations_default_to_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GRASPPANDA_GRASPNET1B_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    default = GraspNet1BEvaluator()
    assert (default.data_root, default.cache_dir) == (tmp_path / "data", tmp_path / "cache")
    with pytest.raises(
        EvaluationError, match="dataset 'graspnet1b' not found .*GRASPPANDA_GRASPNET1B_ROOT"
    ):
        default.evaluate(tmp_path, [key(0)])


# --------------------------------------------------------------------- results


def test_tables_metrics_and_coverage(evaluator, dump):
    keys = [key(0), key(1), key(2)]
    result = evaluator.evaluate(dump, keys)
    assert isinstance(result, EvaluationResult)
    assert list(result.precision) == [key(0), key(1)]
    assert np.array_equal(result.precision[key(0)], toy_table())
    assert np.array_equal(result.precision[key(1)], np.zeros((50, 6)))  # empty file: scored
    assert result.missing == [key(2)]  # missing file: not scored
    assert (result.requested, result.complete) == (3, False)
    assert math.isclose(result.coverage, 2 / 3)

    stack = np.stack([toy_table(), np.zeros((50, 6))])
    assert result.metrics == {
        "AP": float(stack.mean()),
        "AP0.8": float(stack[:, :, 3].mean()),
        "AP0.4": float(stack[:, :, 1].mean()),
    }
    assert 0.0 < result.metrics["AP"] < 1.0  # a fraction, as the reference returns


def test_info_names_the_evaluator_and_its_constants(evaluator, dump):
    info = evaluator.evaluate(dump, [key(0), key(1), key(2)]).info
    assert (info["evaluator"], info["version"]) == ("GraspNet1BEvaluator", EVALUATOR_VERSION)
    assert info["protocol"] == "graspnet1b-sv-v1"
    constants = info["constants"]
    assert constants["top_k"] == 50 and constants["frictions"] == (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    assert constants["gripper"]["finger_width"] == 0.01 and constants["per_object_top"] == 10
    assert constants["table"]["grid_size"] == 0.008 and constants["nms_translation"] == 0.03
    assert (info["frames_scored"], info["frames_missing"], info["empty_predictions"]) == (2, 1, 1)
    assert info["grasps_read"] == 4 and info["num_workers"] == 0
    assert "row index" in info["tie_rule"] and "all-zero" in info["empty_prediction_rule"]
    assert info["cache"]["hash"] in info["cache"]["directory"]
    assert info["peak_rss_mb"] > 0 and info["seconds"] > 0
    assert set(info["seconds_per_scene"]) == {"scene_0000/kinect"}


def test_nothing_scored_gives_nan_metrics(evaluator, tmp_path):
    result = evaluator.evaluate(tmp_path, [key(0), key(1)])
    assert result.precision == {} and result.missing == [key(0), key(1)]
    assert result.coverage == 0.0 and all(math.isnan(v) for v in result.metrics.values())
    empty = evaluator.evaluate(tmp_path, [])
    assert empty.requested == 0 and not empty.complete


def test_reads_what_save_prediction_writes(evaluator, tmp_path):
    rows = torch.as_tensor(toy_prediction())
    grasps = GraspGroup.from_graspnet_array(rows, key=key(0, camera="realsense"))
    save_prediction(tmp_path, grasps)
    result = evaluator.evaluate(tmp_path, [key(0, camera="realsense")])
    assert result.complete and np.array_equal(result.precision[grasps.key], toy_table())


def test_key_order_does_not_change_metrics(evaluator, tmp_path):
    for frame in range(FRAMES):
        write(tmp_path, key(frame), toy_prediction()[: frame + 2])
    keys = [key(frame) for frame in range(FRAMES)]
    forward = evaluator.evaluate(tmp_path, keys)
    backward = evaluator.evaluate(tmp_path, keys[::-1])
    assert forward.metrics == backward.metrics
    assert list(backward.precision) == keys[::-1]


# ---------------------------------------------------------------------- errors


def test_malformed_prediction_names_the_frame(evaluator, dump):
    write(dump, key(2), np.zeros((5, 16), dtype=np.float32))
    with pytest.raises(EvaluationError, match=r"graspnet1b/scene_0000/kinect/0002: .*\(G, 17\)"):
        evaluator.evaluate(dump, [key(0), key(1), key(2)])
    write(dump, key(2), np.full((5, 17), np.inf, dtype=np.float32))
    with pytest.raises(EvaluationError, match=r"kinect/0002: 5 grasp\(s\) contain NaN or Inf"):
        evaluator.evaluate(dump, [key(2)])


def test_bad_requests_are_rejected(evaluator, dump):
    with pytest.raises(EvaluationError, match="more than once"):
        evaluator.evaluate(dump, [key(0), key(0)])
    with pytest.raises(EvaluationError, match="scores 'graspnet1b' frames, not 'graspclutter6d'"):
        evaluator.evaluate(dump, [SampleKey("graspclutter6d", 0, "kinect", 0)])
    with pytest.raises(
        EvaluationError, match="scene_0000/webcam: .*unknown GraspNet-1Billion camera"
    ):
        evaluator.evaluate(dump, [key(0, camera="webcam")])
    with pytest.raises(EvaluationError, match="scene_0042/kinect: cannot read the object list"):
        evaluator.evaluate(dump, [key(0, scene=42)])


def test_inconsistent_ground_truth_is_an_error(dataset, tmp_path):
    broken = write_dataset(tmp_path / "data", frames=FRAMES)
    annotations = broken / "scenes" / "scene_0000" / "kinect" / "annotations"
    swapped = annotation_xml([(1, BOX_CENTRE, (1, 0, 0, 0)), (0, SPHERE_CENTRE, (1, 0, 0, 0))])
    (annotations / "0001.xml").write_text(swapped)
    evaluator = GraspNet1BEvaluator(data_root=str(broken), cache_dir=str(tmp_path / "cache"))
    for frame in (0, 1, FRAMES):
        write(tmp_path, key(frame), toy_prediction())
    assert evaluator.evaluate(tmp_path, [key(0)]).complete
    with pytest.raises(EvaluationError, match=r"kinect/0001: .*lists objects \[1, 0\]"):
        evaluator.evaluate(tmp_path, [key(1)])
    with pytest.raises(EvaluationError, match=f"kinect/000{FRAMES}: "):  # no such frame
        evaluator.evaluate(tmp_path, [key(FRAMES)])


# --------------------------------------------------------------------- workers


def everything(tmp_path) -> list[SampleKey]:
    keys = []
    for scene in (0, 1):
        for camera in ("kinect", "realsense"):
            for frame in range(FRAMES):
                sample = key(frame, scene, camera)
                keys.append(sample)
                if frame < 2:
                    write(tmp_path, sample, toy_prediction()[: 4 - frame - scene])
    return keys


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("start_method", ["fork", "forkserver"])
def test_workers_give_the_same_result(dataset, evaluator, tmp_path, start_method):
    keys = everything(tmp_path)
    sequential = evaluator.evaluate(tmp_path, keys)
    parallel_evaluator = GraspNet1BEvaluator(
        data_root=str(dataset), cache_dir=str(evaluator.cache_dir), start_method=start_method
    )
    parallel = parallel_evaluator.evaluate(tmp_path, keys, num_workers=3)
    assert list(parallel.precision) == list(sequential.precision) and len(parallel.precision) == 8
    for sample, table in sequential.precision.items():
        assert np.array_equal(parallel.precision[sample], table)
    assert parallel.metrics == sequential.metrics and parallel.missing == sequential.missing
    assert parallel.info["num_workers"] == 3 and len(parallel.info["seconds_per_scene"]) == 4
    assert parallel.info["peak_rss_mb"] > 0


def test_worker_errors_name_the_frame(evaluator, tmp_path):
    keys = everything(tmp_path)
    bad = key(1, scene=1, camera="realsense")
    write(tmp_path, bad, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(EvaluationError) as caught:
        evaluator.evaluate(tmp_path, keys, num_workers=2)
    message = str(caught.value)
    assert message.startswith("graspnet1b/scene_0001/realsense/0001: ")
    assert "(G, 17)" in message and "worker traceback" in message
