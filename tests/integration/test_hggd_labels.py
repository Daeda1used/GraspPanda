"""Per-view grasp labels against HGGD's label converter, on real frames.

Reference values are recorded once by ``tools/oracle/hggd_convert_reference.py`` (run
with an interpreter that has the upstream dependencies; it runs the pristine
``convert_grasp.run``) and kept under ``$GRASPPANDA_CACHE/fixtures/hggd/labels``. A
fixture holds, for one view: the arrays the converter saved, every ``random.sample`` it
made, the order in which its compiled NMS visited the grasps, and the grasp sets between
its steps. The tests replay the choices through :func:`frame_view_grasps` and compare.

Tolerances, with what was observed on the development machine in brackets:

* grasps after filtering, before sub-sampling: **exact**, value for value and in order
  (single-precision poses, widths, scores; 4871 to 23837 grasps per view);
* drawn object-model points: 1e-15 m [exact];
* refined centres: 1e-14 m [2e-15]. The reference maps the centre back with a numerically
  inverted 4 x 4 matrix, here it is a closed form;
* survivors of the suppression: **the same rows in the same order**;
* saved labels: centre 1e-10 px [2e-12], depth 1e-12 relative [2e-15], ``gamma`` and
  ``beta`` 1e-11 rad [1e-13], ``theta``, ``width`` and ``height`` one unit in the last
  place of the float32 the reference stores [exact], score and object id **exact**.

The last test prints the differences per field; run with ``-s`` to see them.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from grasppanda.cli.main import main
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspnet1b import GraspNet1B
from grasppanda.datasets.view_grasps import (
    FIELDS,
    REFERENCE_INTRINSICS,
    SeededDraws,
    ViewGraspConfig,
    ViewGraspTrace,
    frame_view_grasps,
    scene_grasps,
    scene_models,
)
from grasppanda.structures import SampleKey

pytestmark = [pytest.mark.dataset, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
PREPROCESS = [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}]
#: Field here, array the converter saves, factor from our unit to its unit.
SAVED = (
    ("centers", "centers_2d", 1.0),
    ("depth", "center_z_depths", 1000.0),  # the converter stores millimetres
    ("theta", "thetas_rad", 1.0),
    ("gamma", "gammas_rad", 1.0),
    ("beta", "betas_rad", 1.0),
    ("width", "widths_2d", 1.0),
    ("height", "heights_2d", 1.0),
    ("score", "scores_from_6d", 1.0),
    ("object_id", "object_ids", 1.0),
)
#: Columns of the GraspNet grasp array: score, width, height, depth, rotation, translation, id.
SCORE, WIDTH, DEPTH, ROTATION, TRANSLATION, OBJECT = 0, 1, 3, slice(4, 13), slice(13, 16), 16


def load_recorder() -> Any:
    """The recorder script as a module: it owns the list of recorded views."""
    file = REPO / "tools" / "oracle" / "hggd_convert_reference.py"
    if not file.is_file():
        pytest.skip(f"{file} is not part of this checkout", allow_module_level=True)
    spec = importlib.util.spec_from_file_location("hggd_convert_reference", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = load_recorder()
VIEWS = [tuple(view) for view in RECORDER.VIEWS]
#: (camera, scene, view) -> {field: (max abs, max rel, rows)}; filled by the comparison.
DIFFERENCES: dict[tuple, dict[str, tuple[float, float, int]]] = {}


@pytest.fixture(scope="session")
def fixtures(graspnet1b_root, asset_cache) -> Path:
    """Directory of the recorded converter output."""
    directory = asset_cache / "fixtures" / "hggd" / "labels"
    index = directory / "index.json"
    wanted = [directory / RECORDER.fixture_name(*view) for view in VIEWS]
    recorded = index.is_file() and json.loads(index.read_text()).get("format") == RECORDER.FORMAT
    if not recorded or not all(file.is_file() for file in wanted):
        pytest.skip(
            f"no recorded converter output in {directory}; record it with the interpreter of an "
            "HGGD environment: python -B tools/oracle/hggd_convert_reference.py --upstream "
            f"<pristine HGGD checkout> --dataset-root {graspnet1b_root} --out {directory}"
        )
    return directory


_readers: dict[str, GraspNet1B] = {}
_scenes: dict[tuple[str, int], tuple[Any, Any]] = {}


def reader(camera: str) -> GraspNet1B:
    if camera not in _readers:
        _readers[camera] = GraspNet1B(
            camera=camera,
            split="train",
            preprocess=PREPROCESS,
            transforms=[],
            object_labels=False,
            check_assets=False,
        )
    return _readers[camera]


def scene_state(camera: str, scene: int, key: SampleKey) -> tuple[Any, Any]:
    """Candidates and object models of a scene, computed once."""
    if (camera, scene) not in _scenes:
        dataset = reader(camera)
        store = dataset.label_store
        missing = store.missing(dataset.object_ids([scene]), [scene])
        if missing:
            pytest.skip(
                f"the grasp-label store lacks {len(missing)} shards of scene {scene}; build "
                "it with: grasppanda prepare graspnet1b --labels"
            )
        poses = dataset.object_poses(key, "annotations")
        _scenes[camera, scene] = (
            scene_grasps(dataset.scene_labels(scene), ViewGraspConfig()),
            scene_models(dataset, [object_id for object_id, _ in poses]),
        )
    return _scenes[camera, scene]


class Replay:
    """The choices the converter made, in the order it made them.

    Its first ``random.sample`` draws the model points, the last one the depth-cloud
    points of the visibility statistic, and one in between (present only for views with
    more than 5000 grasps) the sub-sample. The visiting order of its NMS was recorded by
    probing the compiled extension.
    """

    def __init__(self, reference: Any, ties: np.ndarray | None = None) -> None:
        self.reference = reference
        count = int(reference["draws"])
        self.position = {"model_points": 0, "depth_points": count - 1}
        if count == 3:
            self.position["grasps"] = 1
        self.ties = reference["nms_order"] if ties is None else ties

    def sample(self, what: str, available: int, count: int) -> np.ndarray:
        if what == "nms_ties":
            return self.ties
        draw = f"draw_{self.position[what]}"
        assert int(self.reference[f"{draw}/population"]) == available, what
        return self.reference[f"{draw}/indices"]


def replay(
    camera: str,
    scene: int,
    view: int,
    reference: Any,
    config: ViewGraspConfig,
    ties: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], ViewGraspTrace]:
    dataset = reader(camera)
    key = SampleKey("graspnet1b", scene, camera, view)
    candidates, models = scene_state(camera, scene, key)
    depth_points = None
    if config.visibility_filter:
        frame = dataset.frame(key)
        depth_points = frame.cloud[frame.depth > 0]
    return frame_view_grasps(
        candidates,
        models,
        dataset.object_poses(key, config.poses),
        config,
        REFERENCE_INTRINSICS[camera],
        dataset.image_size,
        Replay(reference, ties),
        depth_points,
    )


def difference(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float, int]:
    """Largest absolute and relative difference of two arrays, and their length."""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    if ours.size == 0:
        return 0.0, 0.0, 0
    absolute = np.abs(ours - theirs)
    relative = absolute / np.maximum(np.abs(theirs), np.finfo(np.float64).tiny)
    return float(absolute.max()), float(relative.max()), len(ours)


def float32_ulp(values: np.ndarray) -> np.ndarray:
    """One unit in the last place of the float32 numbers nearest to ``values``."""
    return np.spacing(np.abs(values).astype(np.float32)).astype(np.float64)


@pytest.mark.parametrize(("camera", "scene", "view"), VIEWS)
def test_generator_reproduces_the_converter(fixtures, camera, scene, view):
    reference = np.load(fixtures / RECORDER.fixture_name(camera, scene, view))
    assert int(reference["seed"]) == RECORDER.seed_for(scene, view)
    labels, trace = replay(camera, scene, view, reference, ViewGraspConfig())

    # steps 1 and 2: the same grasps in the same order, bit for bit
    masked = reference["masked"]
    candidates = trace.candidates
    assert len(candidates) == len(masked) > 4000
    assert np.array_equal(candidates.score, masked[:, SCORE])
    assert np.array_equal(candidates.width, masked[:, WIDTH])
    assert np.array_equal(candidates.depth, masked[:, DEPTH])
    assert np.array_equal(candidates.rotation.reshape(-1, 9), masked[:, ROTATION])
    assert np.array_equal(candidates.translation, masked[:, TRANSLATION])
    assert np.array_equal(candidates.object_id, masked[:, OBJECT])
    assert set(np.unique(candidates.score)) <= {1.0, float(np.float32(1.1) - np.float32(0.2))}

    # step 3: the converter sub-samples exactly the views with more than 5000 grasps
    refined = reference["refined"]
    assert (int(reference["draws"]) == 3) == (len(masked) > 5000)
    assert len(trace.refined) == len(refined) == min(len(masked), 5000)

    # step 4: the drawn model points and the refined centres; nothing else changes
    assert np.abs(trace.model_points - reference["model_points"]).max() <= 1e-15
    assert np.array_equal(trace.refined.rotation.reshape(-1, 9), refined[:, ROTATION])
    assert np.array_equal(trace.refined.width, refined[:, WIDTH])  # the width is not refined
    centre = difference(trace.refined.translation, refined[:, TRANSLATION])
    assert centre[0] <= 1e-14
    moved = np.abs(
        refined[:, TRANSLATION]
        - trace.candidates.take(
            reference["draw_1/indices"] if len(masked) > 5000 else slice(None)
        ).translation
    ).max(axis=1)
    assert (moved > 1e-6).mean() > 0.5  # most centres do move

    # step 5: the same survivors in the same order
    assert np.array_equal(trace.survivors, reference["nms_keep"])
    assert np.array_equal(
        trace.refined.take(trace.survivors).rotation.reshape(-1, 9),
        reference["nms_survivors"][:, ROTATION],
    )

    # steps 6 and 7: what the converter saved
    assert trace.finite.all() and trace.visible is None
    saved = {name: reference[f"saved/{theirs}"] for name, theirs, _ in SAVED}
    assert len(labels["score"]) == len(saved["score"]) == int(trace.inside.sum()) > 50
    found = {name: labels[name] * scale for name, _, scale in SAVED}
    report = {"refined centre [m]": centre}
    for name in FIELDS:
        report[name] = difference(found[name], saved[name])
    DIFFERENCES[camera, scene, view] = report
    assert report["centers"][0] <= 1e-10
    assert report["depth"][1] <= 1e-12
    assert report["gamma"][0] <= 1e-11 and report["beta"][0] <= 1e-11
    for name in ("theta", "width", "height"):
        assert saved[name].dtype == np.float32
        assert (np.abs(found[name] - saved[name]) <= float32_ulp(saved[name])).all(), name
    assert np.array_equal(found["score"], saved["score"])
    assert np.array_equal(found["object_id"], saved["object_id"])
    assert labels["object_id"].dtype == np.int64
    # the units and ranges the asset documents
    assert labels["depth"].min() > 0.2 and labels["depth"].max() < 1.0  # metres
    assert np.abs(labels["theta"]).max() > math.pi / 2  # theta is not folded to a half turn


@pytest.mark.parametrize(("camera", "scene", "view"), VIEWS)
def test_visibility_is_computed_by_the_converter_but_not_applied(fixtures, camera, scene, view):
    """With ``visibility_filter`` the labels are the subset the converter only counts."""
    reference = np.load(fixtures / RECORDER.fixture_name(camera, scene, view))
    assert float(reference["visible_distance"]) == ViewGraspConfig().visibility_distance
    config = ViewGraspConfig(visibility_filter=True)
    labels, trace = replay(camera, scene, view, reference, config)
    key = SampleKey("graspnet1b", scene, camera, view)
    frame = reader(camera).frame(key)
    drawn = frame.cloud[frame.depth > 0][reference[f"draw_{int(reference['draws']) - 1}/indices"]]
    # The same pixels. The converter back-projects with the scene's `camK.npy`, the reader
    # with the frame's calibration; for the RealSense the two differ by 0.005 px (5e-6 m).
    assert np.abs(drawn - reference["visible_cloud"]).max() <= 1e-5
    assert np.array_equal(drawn[:, 2], reference["visible_cloud"][:, 2])
    visible = reference["visible_mask"]
    assert np.array_equal(trace.visible, visible)
    assert json.loads(str(reference["info"]))["numgrasp"] == int(visible.sum())  # its statistic
    # the converter saved every grasp inside the image, visible or not
    assert len(reference["saved/centers_2d"]) == int(trace.inside.sum())
    assert len(labels["score"]) == int((trace.inside & visible).sum()) < int(trace.inside.sum())
    plain, _ = replay(camera, scene, view, reference, ViewGraspConfig())
    keep = visible[trace.inside]
    for name in FIELDS:
        assert np.array_equal(labels[name], plain[name][keep]), name


def within(kept: Any, grasps: Any, distance: float, angle: float) -> np.ndarray:
    """``(K, G)`` whether each kept grasp suppresses each grasp (both tests strict)."""
    apart = np.linalg.norm(kept.translation[:, None] - grasps.translation[None], axis=-1)
    trace = np.einsum("kij,gij->kg", kept.rotation, grasps.rotation)
    turned = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return (apart < distance) & (turned < angle)


@pytest.mark.parametrize(("camera", "scene", "view"), VIEWS)
def test_own_tie_order_gives_a_valid_suppression_of_similar_size(fixtures, camera, scene, view):
    """The converter's order among equal scores is an artefact; ours is a seeded draw."""
    reference = np.load(fixtures / RECORDER.fixture_name(camera, scene, view))
    order, score = reference["nms_order"], reference["refined"][:, SCORE]
    assert sorted(order.tolist()) == list(range(len(score)))
    assert (np.diff(score[order]) <= 0).all()  # descending score ...
    assert not np.array_equal(order, np.argsort(-score, kind="stable"))  # ... ties unstable
    assert len(np.unique(score)) == 2  # nearly everything is a tie

    key = SampleKey("graspnet1b", scene, camera, view)
    ties = SeededDraws(0, key).sample("nms_ties", len(score), len(score))
    _, theirs = replay(camera, scene, view, reference, ViewGraspConfig())
    _, ours = replay(camera, scene, view, reference, ViewGraspConfig(), ties)
    assert np.array_equal(ours.refined.translation, theirs.refined.translation)
    count, expected = len(ours.survivors), len(theirs.survivors)
    assert abs(count - expected) <= 0.2 * expected
    assert not np.array_equal(ours.survivors, theirs.survivors)

    # a correct greedy result for the order we visited: survivors are mutually apart, and
    # every other grasp is suppressed by a survivor visited before it
    grasps, config = ours.refined, ViewGraspConfig()
    visit = ties[np.argsort(-grasps.score[ties], kind="stable")]
    rank = np.empty(len(visit), dtype=np.int64)
    rank[visit] = np.arange(len(visit))
    assert (np.diff(rank[ours.survivors]) > 0).all()
    hit = within(
        grasps.take(ours.survivors), grasps, config.nms_distance, math.radians(config.nms_angle)
    )
    earlier = rank[ours.survivors][:, None] < rank[None, :]
    suppressed = (hit & earlier).any(axis=0)
    survivor = np.zeros(len(grasps), dtype=bool)
    survivor[ours.survivors] = True
    assert np.array_equal(suppressed, ~survivor)


def test_prepare_builds_a_scene_of_the_real_dataset(
    fixtures, graspnet1b_root, asset_cache, tmp_path, monkeypatch, capsys
):
    """The ``prepare`` hook on scene 0, restricted to two frames, in a throw-away cache."""
    labels = asset_cache / "graspnet1b" / "grasp_labels"
    if not labels.is_dir():
        pytest.skip("no grasp-label store; build it with: grasppanda prepare graspnet1b --labels")
    cache = tmp_path / "cache"
    (cache / "graspnet1b").mkdir(parents=True)
    (cache / "graspnet1b" / "grasp_labels").symlink_to(labels, target_is_directory=True)
    monkeypatch.setenv("GRASPPANDA_CACHE", str(cache))  # the real asset store is only read
    frames = (0, 128)
    monkeypatch.setattr(
        GraspNet1B,
        "scene_keys",
        lambda self, scene: [SampleKey(self.name, scene, self.camera, frame) for frame in frames],
    )
    node = {
        "type": "GraspNet1B",
        "camera": "realsense",
        "split": "train",
        "object_labels": False,
        "preprocess": PREPROCESS,
        "transforms": [],
        "view_grasps": {"seed": 0},
    }
    file = tmp_path / "views.yaml"
    file.write_text(yaml.safe_dump({"data": {"train": {"dataset": node}}}))
    base = ["prepare", "graspnet1b", "--config", str(file), "--camera", "realsense"]
    assert main([*base, "--view-grasps", "--scenes", "0", "--workers", "1"]) == 0
    assert "1 scenes written" in capsys.readouterr().out
    assert main([*base, "--view-grasps", "--scenes", "0", "--workers", "1"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert main([*base, "--check", "--scenes", "0"]) == 0
    assert "view grasps 1/1 scenes" in capsys.readouterr().out

    dataset = GraspNet1B(**{k: v for k, v in node.items() if k != "type"}, check_assets=False)
    assert dataset.view_grasps.asset.path.is_relative_to(cache)
    record = dataset.view_grasps.asset.shards()["scene_0000"]
    assert record["frames"] == 2 and record["degenerate"] == 0 and record["invisible"] == 0
    total = 0
    for frame in frames:
        reference = np.load(fixtures / RECORDER.fixture_name("realsense", 0, frame))
        stored = dataset.view_grasps.read(SampleKey("graspnet1b", 0, "realsense", frame))
        assert tuple(stored) == FIELDS
        count, expected = len(stored["score"]), len(reference["saved/centers_2d"])
        total += count
        assert abs(count - expected) <= 0.25 * expected  # other draws, the same statistics
        u, v = stored["centers"].T
        assert ((u > 0) & (u < 1280) & (v > 0) & (v < 720)).all()
        low, high = np.percentile(reference["saved/center_z_depths"], [0, 100]) / 1000.0
        assert stored["depth"].min() > low - 0.05 and stored["depth"].max() < high + 0.05
        assert set(stored["object_id"]) <= set(dataset.scene_object_ids(0))
        assert set(stored["score"]) <= set(reference["saved/scores_from_6d"].astype(np.float64))
    assert record["grasps"] == total
    with pytest.raises(AssetError, match="has no labels for frame 5"):
        dataset.view_grasps.read(SampleKey("graspnet1b", 0, "realsense", 5))


def test_report_of_differences(fixtures, capsys):
    """Print, per view and field, how far the replayed generator is from the converter."""
    if len(DIFFERENCES) != len(VIEWS):
        pytest.skip("run together with test_generator_reproduces_the_converter")
    names = ["refined centre [m]", *FIELDS]
    with capsys.disabled():
        print("\nper-view grasp labels: replayed generator vs HGGD convert_grasp.py")
        print(f"{'view':<22}{'field':<20}{'rows':>6}{'max abs':>12}{'max rel':>12}")
        for (camera, scene, view), report in DIFFERENCES.items():
            for name in names:
                absolute, relative, rows = report[name]
                unit = f"{camera} {scene}/{view}"
                print(f"{unit:<22}{name:<20}{rows:>6}{absolute:>12.2e}{relative:>12.2e}")
        print(f"{'all ' + str(len(VIEWS)) + ' views':<22}", end="")
        worst = {n: max(report[n][0] for report in DIFFERENCES.values()) for n in names}
        print("  ".join(f"{name} {value:.1e}" for name, value in worst.items()))
    assert all(
        report["score"][0] == 0 and report["object_id"][0] == 0 for report in DIFFERENCES.values()
    )
