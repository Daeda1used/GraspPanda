"""Parity of the GraspClutter6D evaluator with the official ``graspclutter6dAPI``.

Reference values are recorded by ``tools/make_eval_fixtures.py clutter`` (the official
tool, run from a clone on ``sys.path`` in the oracle environment) into
``$GRASPPANDA_CACHE/graspclutter6d/eval_fixtures/v1``. Two kinds of predictions are
recorded:

* **label**: grasps drawn from the ground-truth labels, of which a tenth each is moved
  into the background, widened beyond the 0.14 m limit and given a non-unit closing
  axis, so the three rules that are specific to this benchmark all decide results. Four
  scenes, one per camera, five views each.
* **dump_***: what ``grasppanda test`` wrote for a detector on the ``test_mini`` subset,
  linked into the official layout with ``export_official_layout`` and scored by the
  official tool unchanged: all 13 views of the scenes listed in :data:`DUMPS`. These
  are the frames, the score distributions and the failure modes of real runs (most of
  their grasps collide or are not in force closure), and they cover every camera.

A missing label recording is produced on the spot when ``$GRASPPANDA_ORACLE_PYTHON`` and
``$GRASPPANDA_GRASPCLUTTER6D_API`` (a clone of the official repository) are set;
otherwise the test is skipped with the command to run. A dump recording needs the dump
of a run, so it is only ever skipped::

    grasppanda test <transfer config> --weights <checkpoint> \\
        --opts data.test.dataset.split=test_mini [data.test.dataset.camera=<camera>]
    python -c "from grasppanda.evaluation.graspclutter6d import export_official_layout; ..."
    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py clutter --api <clone> \\
        --camera <camera> --scenes <scenes> --frames 0 1 2 3 4 5 6 7 8 9 10 11 12 \\
        --dump <linked dump> --suite dump_<name>

The official tool breaks ties between equal scores with unstable sorts, so a frame with
tied scores has no unique reference result. Such a frame is compared like any other and
reported separately if it differs; a tie-free frame must be equal at every stage.
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
from grasppanda.evaluation.graspclutter6d import (
    GRASPCLUTTER6D_SOURCES,
    GRASPCLUTTER6D_SV_V1,
    ClutterGroundTruth,
    GraspClutter6DEvaluator,
    prediction_file,
)
from grasppanda.evaluation.graspnet1b import ObjectCache
from grasppanda.evaluation.graspnet1b.frame import evaluate_frame, prepare_prediction
from grasppanda.structures import SampleKey

pytestmark = [pytest.mark.dataset, pytest.mark.oracle, pytest.mark.slow]

PROTOCOL = GRASPCLUTTER6D_SV_V1
RECORDER = Path(__file__).resolve().parents[2] / "tools" / "make_eval_fixtures.py"
API_VARIABLE = "GRASPPANDA_GRASPCLUTTER6D_API"

#: (scene, camera) -> recorder arguments. All four cameras, four test scenes.
RECORDINGS: dict[tuple[int, str], tuple[str, ...]] = {
    (2, "zivid"): ("--frames", "0", "3", "6", "9", "12"),
    (3, "realsense-d435"): ("--frames", "0", "3", "6", "9", "12"),
    (6, "azure-kinect"): ("--frames", "1", "4", "7", "10", "12"),
    (7, "realsense-d415"): ("--frames", "0", "2", "5", "8", "11", "--count", "4000", "--friction", "1.0"),
}  # fmt: skip

#: suite -> (camera, scenes): detector dumps of ``grasppanda test`` on ``test_mini``, all
#: 13 views of each scene. The suite names say which detector and which variant.
DUMPS: dict[str, tuple[str, tuple[int, ...]]] = {
    "dump_gsnet_raw": ("realsense-d435", (2, 18, 35, 54, 71, 91)),
    "dump_baseline_cd": ("realsense-d435", (108, 127, 142)),
    "dump_hggd_cd": ("realsense-d435", (158, 178, 200)),
    "dump_economicgrasp_raw_d415": ("realsense-d415", (217, 235)),
    "dump_economicgrasp_raw_zivid": ("zivid", (252, 272)),
    "dump_economicgrasp_raw_kinect": ("azure-kinect", (291, 304)),
}
DUMP_CASES = [
    (suite, camera, scene) for suite, (camera, scenes) in DUMPS.items() for scene in scenes
]


def fixture_root(asset_cache: Path) -> Path:
    return asset_cache / "graspclutter6d" / "eval_fixtures" / "v1"


@pytest.fixture(scope="module")
def recorded(request, graspclutter6d_root: Path, asset_cache: Path):
    """Return the folder of a label recording, running the official tool first if it is missing."""
    root = fixture_root(asset_cache)

    def ensure(scene: int, camera: str) -> Path:
        folder = root / "label" / f"scene_{scene:06d}" / camera
        if (folder / "run.json").exists():
            return folder
        command = [str(RECORDER), "--data-root", str(graspclutter6d_root), "--out", str(root)]
        command += ["clutter", "--camera", camera, "--scenes", str(scene)]
        command += list(RECORDINGS[scene, camera])
        api = os.environ.get(API_VARIABLE, "")
        if not Path(api).is_dir():
            pytest.skip(
                f"{folder} is not recorded and ${API_VARIABLE} does not name a clone of "
                f"graspclutter6dAPI. Command: $GRASPPANDA_ORACLE_PYTHON {' '.join(command)} --api <clone>"
            )
        interpreter = request.getfixturevalue("oracle_python")
        done = subprocess.run([interpreter, *command, "--api", api], capture_output=True, text=True)
        if done.returncode != 0 or not (folder / "run.json").exists():
            pytest.fail(f"oracle run failed: {' '.join(command)}\n{done.stderr[-3000:]}")
        return folder

    return ensure


def compare(folder: Path, data_root: Path, asset_cache: Path, scene: int, camera: str):
    """Score every recorded frame of a folder and compare each stage with the recording.

    Returns:
        ``(counts, mismatches)``; a mismatch is ``(frame, tied scores?, stages that differ)``.
    """
    cache = ObjectCache(
        data_root,
        model_voxel_size=PROTOCOL.model_voxel_size,
        sources=GRASPCLUTTER6D_SOURCES,
        root=asset_cache,
    )
    cache.ensure(ClutterGroundTruth.list_object_ids(data_root, scene, camera))
    truth = ClutterGroundTruth(data_root, scene, camera, cache, PROTOCOL)
    counts: Counter[str] = Counter(instances=len(truth.objects))
    mismatches = []
    for file in sorted(folder.glob("[0-9]*.npz")):
        with np.load(file) as reference:
            prediction = reference["prediction"]
            tied = len(np.unique(prediction[:, 0])) != len(prediction)
            poses, table = truth.frame(int(file.stem))
            assert table is None
            assert np.array_equal(np.array(poses), reference["poses"])  # float64, bit for bit
            result = evaluate_frame(prediction, truth.objects, poses, table, PROTOCOL)
            grasps = prepare_prediction(prediction, PROTOCOL.gripper.max_width)
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
            if not all(equal.values()):
                mismatches.append((file.stem, tied, [s for s, ok in equal.items() if not ok]))
            counts["frames"] += 1
            counts["frames_with_tied_scores"] += tied
            counts["grasps_in"] += len(prediction)
            counts["after_filter_and_nms"] += len(reference["nms"])
            counts["kept"] += len(reference["kept"])
            counts["colliding_or_empty"] += int(reference["collision"].sum())
            counts["force_closure"] += int((reference["friction"] > 0).sum())
            counts["clipped_widths"] += int((prediction[:, 1] > PROTOCOL.gripper.max_width).sum())
    return counts, mismatches


@pytest.mark.parametrize("scene,camera", list(RECORDINGS), ids=lambda value: str(value))
def test_recorded_frames_match_stage_by_stage(
    scene, camera, graspclutter6d_root, asset_cache, recorded, record_property
):
    folder = recorded(scene, camera)
    counts, mismatches = compare(folder, graspclutter6d_root, asset_cache, scene, camera)
    summary = {"scene": scene, "camera": camera, **counts}
    record_property("parity", json.dumps(summary))
    print(f"\nclutter parity {json.dumps(summary)}")
    assert counts["frames_with_tied_scores"] == 0  # tie-free by construction
    assert mismatches == []
    assert counts["frames"] >= 5 and counts["force_closure"] > 50 and counts["clipped_widths"] > 0


@pytest.mark.parametrize("suite,camera,scene", DUMP_CASES, ids=lambda value: str(value))
def test_detector_dumps_match_stage_by_stage(
    suite, camera, scene, graspclutter6d_root, asset_cache, record_property
):
    """What ``grasppanda test`` wrote, scored by the official tool: equal at every stage."""
    folder = fixture_root(asset_cache) / suite / f"scene_{scene:06d}" / camera
    if not (folder / "run.json").exists():
        pytest.skip(f"{folder} is not recorded; see the module docstring for the commands")
    counts, mismatches = compare(folder, graspclutter6d_root, asset_cache, scene, camera)
    summary = {"suite": suite, "scene": scene, "camera": camera, **counts}
    record_property("parity", json.dumps(summary))
    print(f"\nclutter parity {json.dumps(summary)}")
    assert counts["frames"] >= 12  # an empty prediction cannot be recorded; at most one
    untied = [entry for entry in mismatches if not entry[1]]
    assert untied == []
    assert len(mismatches) <= counts["frames_with_tied_scores"]


def test_recorded_scenes_through_the_registered_evaluator(
    graspclutter6d_root, asset_cache, recorded, tmp_path
):
    """Both dump layouts, the process pool and the aggregate metrics."""
    official, ours = tmp_path / "official", tmp_path / "ours"
    keys, reference = [], {}
    for scene, camera in RECORDINGS:
        for file in sorted(recorded(scene, camera).glob("[0-9]*.npz")):
            with np.load(file) as stored:
                key = SampleKey("graspclutter6d", scene, camera, int(file.stem))
                for path in (prediction_file(official, key), prediction_path(ours, key)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, stored["prediction"])
                keys.append(key)
                reference[key] = stored["precision"]
    absent = SampleKey("graspclutter6d", 2, "zivid", 1)  # no dump: missing, not a zero
    for layout, root in (("official", official), ("grasppanda", ours)):
        evaluator = GraspClutter6DEvaluator(
            data_root=str(graspclutter6d_root), cache_dir=str(asset_cache), layout=layout
        )
        result = evaluator.evaluate(root, [*keys, absent], num_workers=4)
        assert result.missing == [absent] and len(result.precision) == len(keys) >= 20
        different = [
            str(key) for key in keys if not np.array_equal(result.precision[key], reference[key])
        ]
        assert different == []
        stack = np.stack([reference[key] for key in keys])
        assert result.metrics["AP"] == pytest.approx(stack.mean(), rel=1e-13)
        assert result.metrics["AP0.4"] == pytest.approx(stack[:, :, 1].mean(), rel=1e-13)
        assert result.metrics["AP0.8"] == pytest.approx(stack[:, :, 3].mean(), rel=1e-13)


def test_a_whole_dump_scores_like_the_official_tool(graspclutter6d_root, asset_cache, tmp_path):
    """Every recorded detector frame through the registered evaluator, in the layout
    ``grasppanda test`` writes: the path of a real run, compared table by table."""
    keys, reference = [], {}
    for suite, camera, scene in DUMP_CASES:
        folder = fixture_root(asset_cache) / suite / f"scene_{scene:06d}" / camera
        for file in sorted(folder.glob("[0-9]*.npz")) if (folder / "run.json").exists() else ():
            with np.load(file) as stored:
                if len(np.unique(stored["prediction"][:, 0])) != len(stored["prediction"]):
                    continue  # tied scores: no unique reference result
                key = SampleKey("graspclutter6d", scene, camera, int(file.stem))
                path = prediction_path(tmp_path, key)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, stored["prediction"])
                keys.append(key)
                reference[key] = stored["precision"]
    if not keys:
        pytest.skip("no detector dump is recorded; see the module docstring for the commands")
    evaluator = GraspClutter6DEvaluator(
        data_root=str(graspclutter6d_root), cache_dir=str(asset_cache)
    )
    result = evaluator.evaluate(tmp_path, keys, num_workers=4)
    assert result.complete
    different = [
        str(key) for key in keys if not np.array_equal(result.precision[key], reference[key])
    ]
    stack = np.stack([reference[key] for key in keys])
    print(
        f"\n{len(keys)} detector frames of {len({(k.scene, k.camera) for k in keys})} scenes and "
        f"{len({k.camera for k in keys})} cameras: AP {100 * result.metrics['AP']:.4f} here, "
        f"{100 * stack.mean():.4f} by the official tool; {len(different)} tables differ"
    )
    assert different == []
    assert result.metrics["AP"] == pytest.approx(stack.mean(), rel=1e-13)
