"""Record reference values from the official GraspNet evaluator (the parity oracle).

GraspPanda re-implements the GraspNet-1Billion evaluator (``docs/design/ARCHITECTURE.md``
§11). This script is the other half of the parity gate: it runs the *official*
``graspnetAPI`` and stores what it computes, stage by stage, so that
``tests/integration/test_eval_parity.py`` can compare value for value.

It must be executed with an interpreter that has the official packages installed
(``graspnetAPI`` 1.2.x, ``grasp_nms``, ``open3d``), i.e. ``$GRASPPANDA_ORACLE_PYTHON``,
and it deliberately imports nothing from GraspPanda, so it stays valid Python 3.8::

    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py dump  --dump DIR --camera kinect --scenes 100 144 189
    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py label --camera realsense --scenes 110 --frames 0 32 64
    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py edge  --dump DIR --camera kinect --scene 100
    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py primitives --objects 0 18 --scenes 100
    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py clutter --api /path/to/graspclutter6dAPI \
        --camera zivid --scenes 2 --frames 0 4 8 12
    $GRASPPANDA_ORACLE_PYTHON tools/make_eval_fixtures.py clutter --api /path/to/graspclutter6dAPI \
        --camera realsense-d435 --scenes 2 18 --frames 0 1 2 --dump DIR --suite dump_gsnet
    python tools/make_eval_fixtures.py pack --suite dump --camera kinect --scenes 100 145 170 \
        --frames 0 32 64 96 128 160 192 224 --to tests/integration/fixtures/eval_graspnet1b

How values are captured: the official ``GraspNetEval.eval_scene`` is called unchanged.
Its helper functions are wrapped by pass-through taps that copy their arguments and
results (post-NMS grasps, nearest-point indices, kept grasps, collision and emptiness
masks, friction scores). No official logic is re-typed here, so a recorded value is
what the official code produced.

Safety: the dataset is read-only. A guard makes any attempt to open a file for
writing, or to create or remove anything, below the dataset root raise before it
reaches the file system; the listing of the directories the official tools are known
to cache into is compared before and after a run.

Output (default ``$GRASPPANDA_CACHE/graspnet1b/eval_fixtures/v1``)::

    <suite>/scene_XXXX/<camera>/NNNN.npz     one frame: prediction and all stages
    <suite>/scene_XXXX/<camera>/run.json     versions, timing, peak memory, case names
    primitives/*.npz                         stage-level references on real objects

``clutter`` records the same stages from the official GraspClutter6D tool (a clone of
``graspclutter6dAPI`` is put on ``sys.path``; it runs in the same environment) into
``$GRASPPANDA_CACHE/graspclutter6d/eval_fixtures/v1``. That tool would pickle Dex-Net
models into the dataset if they were missing; the write guard turns that into an error.

``pack`` needs no official package. It condenses a few recorded frames into one small
file that is committed, so that parity can be re-checked wherever the dataset is,
without the oracle environment.
"""

from __future__ import annotations

import argparse
import builtins
import importlib
import json
import os
import pickle
import platform
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FIXTURE_VERSION = "v1"
FRAMES_PER_SCENE = 256
NMS_TRANSLATION = 0.03
NMS_ROTATION = 30.0 / 180 * np.pi


def default_output(dataset: str) -> Path:
    """Same rule as ``grasppanda.utils.paths.cache_root`` (not importable here)."""
    configured = os.environ.get("GRASPPANDA_CACHE")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache" / "grasppanda"
    return root / dataset / "eval_fixtures" / FIXTURE_VERSION


def default_data_root(dataset: str) -> str:
    return os.environ.get(f"GRASPPANDA_{dataset.upper()}_ROOT", "")


# ------------------------------------------------------------------------ write guard


def install_write_guard(protected: Path) -> None:
    """Make every write below ``protected`` fail loudly instead of touching the dataset."""
    protected = protected.resolve()

    def inside(path: object) -> bool:
        try:
            resolved = Path(os.fspath(path)).resolve()
        except TypeError:  # file descriptors and other non-path arguments
            return False
        return resolved == protected or protected in resolved.parents

    def refuse(path: object, action: str) -> None:
        raise PermissionError(
            f"oracle write guard: refusing to {action} {path!r}; the dataset is read-only"
        )

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+") and inside(file):
            refuse(file, "open for writing")
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open
    for name in ("mkdir", "makedirs", "remove", "unlink", "rmdir", "rename", "replace"):
        real = getattr(os, name)

        def guarded(path, *args, _real=real, _name=name, **kwargs):
            if inside(path) or any(inside(a) for a in args if isinstance(a, (str, os.PathLike))):
                refuse(path, _name)
            return _real(path, *args, **kwargs)

        setattr(os, name, guarded)


def dataset_listing(root: Path) -> dict:
    """Names and sizes in the places an evaluator could plausibly cache into."""
    listing = {}
    for folder in (root, root / "dex_models", root / "models", root / "models_m"):
        if folder.is_dir():
            listing[str(folder)] = sorted(
                (entry.name, entry.stat().st_size if entry.is_file() else -1)
                for entry in folder.iterdir()
            )
    return listing


# ------------------------------------------------------------------------------- taps


class Taps:
    """Pass-through wrappers around the official evaluation helpers."""

    def __init__(self, with_contacts: bool) -> None:
        self.frames = []
        self.with_contacts = with_contacts
        self._current = None

    def install(self, package: str = "graspnetAPI", eval_module: str = "graspnet_eval") -> None:
        official_eval = importlib.import_module(f"{package}.{eval_module}")
        official_utils = importlib.import_module(f"{package}.utils.eval_utils")
        GraspGroup = importlib.import_module(f"{package}.grasp").GraspGroup

        taps = self
        real_nms = GraspGroup.nms
        real_closest = official_utils.compute_closest_points
        real_collision = official_utils.collision_detection
        real_eval = official_utils.eval_grasp

        def nms(group, *args, **kwargs):
            result = real_nms(group, *args, **kwargs)
            if taps._current is not None:
                taps._current["nms"] = np.array(result.grasp_group_array, copy=True)
            return result

        def closest(queries, points):
            result = real_closest(queries, points)
            if taps._current is not None:
                taps._current["closest"] = np.array(result, copy=True)
            return result

        def collision(grasp_list, model_list, dexnet_models, poses, scene_points, **kwargs):
            result = real_collision(
                grasp_list, model_list, dexnet_models, poses, scene_points, **kwargs
            )
            if taps._current is not None:
                taps._record_collision(grasp_list, dexnet_models, scene_points, result)
            return result

        def eval_grasp(grasp_group, models, dexnet_models, poses, config, **kwargs):
            taps._current = {
                "clipped": np.array(grasp_group.grasp_group_array, copy=True),
                "model_sizes": np.array([len(model) for model in models]),
                "poses": np.array(poses),
            }
            result = real_eval(grasp_group, models, dexnet_models, poses, config, **kwargs)
            taps._current["scores"] = [np.array(scores, dtype=np.float64) for scores in result[1]]
            taps.frames.append(taps._current)
            taps._current = None
            return result

        GraspGroup.nms = nms
        official_utils.compute_closest_points = closest
        official_utils.collision_detection = collision
        official_utils.eval_grasp = eval_grasp
        official_eval.eval_grasp = eval_grasp

    def _record_collision(self, grasp_list, dexnet_models, scene_points, result) -> None:
        collision_masks, empty_masks, dexgrasps = result
        record = self._current
        record["kept"] = [np.array(grasps, copy=True).reshape(-1, 17) for grasps in grasp_list]
        record["collision"] = [np.array(mask, dtype=bool) for mask in collision_masks]
        record["empty"] = [np.array(mask, dtype=bool) for mask in empty_masks]
        record["scene_points"] = len(scene_points)
        if not self.with_contacts:
            return
        found, points, normals = [], [], []
        for position, grasps in enumerate(dexgrasps):
            for index, grasp in enumerate(grasps):
                ok, point, normal = False, np.full((2, 3), np.nan), np.full((2, 3), np.nan)
                if grasp is not None and not collision_masks[position][index]:
                    ok, contacts = grasp.close_fingers(
                        dexnet_models[position], check_approach=False
                    )
                    if ok:
                        point = np.array([contact.point for contact in contacts])
                        normal = np.array([contact.normal for contact in contacts])
                found.append(bool(ok))
                points.append(point)
                normals.append(normal)
        record["contact_found"] = np.array(found, dtype=bool)
        record["contact_points"] = np.array(points).reshape(-1, 2, 3)
        record["contact_normals"] = np.array(normals).reshape(-1, 2, 3)


def frame_arrays(record: dict, prediction: np.ndarray, precision: np.ndarray) -> dict:
    """Flatten one tapped frame into the arrays stored on disk."""
    segment = np.repeat(np.arange(len(record["model_sizes"])), record["model_sizes"])
    sizes = [len(grasps) for grasps in record["kept"]]

    def flat(parts, dtype):
        parts = [np.asarray(part, dtype=dtype).reshape(-1) for part in parts]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

    arrays = {
        "prediction": prediction,
        "nms": record["nms"],
        "owner": segment[record["closest"]].astype(np.int64),
        "kept": np.concatenate(record["kept"]).reshape(-1, 17),
        "kept_owner": np.repeat(np.arange(len(sizes)), sizes).astype(np.int64),
        "collision": flat(record["collision"], bool),
        "empty": flat(record["empty"], bool),
        "friction": flat(record["scores"], np.float64),
        "precision": np.asarray(precision, dtype=np.float64),
        "poses": record["poses"],
        "model_sizes": record["model_sizes"],
        "scene_points": np.int64(record["scene_points"]),
    }
    for name in ("contact_found", "contact_points", "contact_normals"):
        if name in record:
            arrays[name] = record[name]
    return arrays


# ------------------------------------------------------------------------ scene runs


def versions(toolkit: str = "graspnetAPI") -> dict:
    import open3d

    found = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "open3d": open3d.__version__,
        f"{toolkit}_file": importlib.import_module(toolkit).__file__,
        "platform": platform.platform(),
    }
    from importlib.metadata import PackageNotFoundError, version

    for package in (toolkit, "grasp_nms", "autolab_core", "transforms3d", "scipy"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:  # e.g. a toolkit used from a clone on sys.path
            found[package] = "not installed (imported from a source tree)"
    return found


def run_scene(
    data_root: Path,
    out: Path,
    suite: str,
    scene: int,
    camera: str,
    predictions: dict,
    cases: dict,
    with_contacts: bool,
    taps: Taps,
) -> None:
    """Evaluate one scene with the official tool and store the requested frames.

    ``predictions`` maps frame index to the ``(G, 17)`` array to evaluate. The official
    loop always visits all 256 frames, so frames that are not requested get a one-grasp
    placeholder and are not stored.
    """
    from graspnetAPI import GraspNetEval

    if not predictions:
        print(f"[{suite}] scene {scene} {camera}: nothing to evaluate")
        return
    placeholder = next(iter(predictions.values()))[:1]
    folder = out / suite / f"scene_{scene:04d}" / camera
    folder.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="oracle_dump_", dir=str(out)))
    try:
        dump = work / f"scene_{scene:04d}" / camera
        dump.mkdir(parents=True)
        for frame in range(FRAMES_PER_SCENE):
            np.save(str(dump / f"{frame:04d}.npy"), predictions.get(frame, placeholder))
        taps.frames.clear()
        evaluator = GraspNetEval(str(data_root), camera, split="custom")
        started, cpu_started = time.perf_counter(), time.process_time()
        accuracy = evaluator.eval_scene(scene, str(work), return_list=True)[0]
        seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started  # all threads of this process
        print()
        if len(taps.frames) != FRAMES_PER_SCENE or len(accuracy) != FRAMES_PER_SCENE:
            raise RuntimeError(
                f"expected {FRAMES_PER_SCENE} tapped frames, got {len(taps.frames)} "
                f"and {len(accuracy)} tables"
            )
        for frame in sorted(predictions):
            arrays = frame_arrays(taps.frames[frame], predictions[frame], accuracy[frame])
            temporary = folder / f".{frame:04d}.tmp.npz"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(str(temporary), str(folder / f"{frame:04d}.npz"))
    finally:
        shutil.rmtree(str(work), ignore_errors=True)
    scene_mean = float(np.mean([accuracy[frame] for frame in sorted(predictions)]))
    run = {
        "suite": suite,
        "scene": scene,
        "camera": camera,
        "frames": sorted(predictions),
        "cases": {str(frame): name for frame, name in cases.items()},
        "placeholder_frames": FRAMES_PER_SCENE - len(predictions),
        "official_seconds_256_frames": round(seconds, 2),
        "official_cpu_seconds_256_frames": round(cpu_seconds, 2),
        "peak_rss_mb_process_so_far": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "mean_precision_of_stored_frames": scene_mean,
        "with_contacts": with_contacts,
        "versions": versions(),
        "command": " ".join(sys.argv),
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    print(
        f"[{suite}] scene {scene} {camera}: {len(predictions)} frame(s) stored, "
        f"official time {seconds:.1f} s (CPU {cpu_seconds:.1f} s), mean precision {scene_mean:.4f}"
    )


def load_dump(dump: Path, scene: int, camera: str, frame: int) -> np.ndarray:
    return np.load(str(dump / f"scene_{scene:04d}" / camera / f"{frame:04d}.npy"))


def suite_dump(args, taps: Taps) -> None:
    """Whole scenes of an existing prediction dump, unchanged."""
    for scene in args.scenes:
        frames = args.frames if args.frames else range(FRAMES_PER_SCENE)
        predictions = {frame: load_dump(args.dump, scene, args.camera, frame) for frame in frames}
        empty = [frame for frame, array in predictions.items() if len(array) == 0]
        for frame in empty:  # the official tool raises IndexError on an empty frame
            del predictions[frame]
        if empty:
            print(f"skipping empty prediction frames {empty} of scene {scene}")
        run_scene(
            args.data_root, args.out, args.suite, scene, args.camera, predictions, {},
            args.contacts, taps,
        )  # fmt: skip


def suite_label(args, taps: Taps) -> None:
    """Grasps drawn from the ground-truth labels through the official ``loadGrasp``.

    Label grasps are collision-free and in force closure by construction, so they
    exercise the success path of the evaluator, which detector dumps rarely do in
    bulk. Scores are made tie-free with a small seeded jitter.
    """
    from graspnetAPI import GraspNet

    for scene in args.scenes:
        api = GraspNet(str(args.data_root), camera=args.camera, split="custom", sceneIds=[scene])
        grasp_labels = api.loadGraspLabels(api.getObjIds([scene]))
        collision_labels = api.loadCollisionLabels(scene)
        predictions = {}
        for frame in args.frames:
            group = api.loadGrasp(
                scene, annId=frame, format="6d", camera=args.camera, grasp_labels=grasp_labels,
                collision_labels=collision_labels, fric_coef_thresh=args.friction,
            )  # fmt: skip
            array = np.array(group.grasp_group_array, dtype=np.float32)
            rng = np.random.RandomState(args.seed + 1000 * scene + frame)
            chosen = rng.choice(len(array), size=min(args.count, len(array)), replace=False)
            array = array[np.sort(chosen)]
            array[:, 0] += rng.uniform(0.0, 0.05, size=len(array)).astype(np.float32)
            while len(np.unique(array[:, 0])) != len(array):  # float32 collisions are rare
                array[:, 0] += rng.uniform(0.0, 1e-4, size=len(array)).astype(np.float32)
            predictions[frame] = array
            print(f"label grasps: scene {scene} frame {frame}: {len(array)} of {len(group)}")
        del grasp_labels, collision_labels
        cases = dict.fromkeys(predictions, "label")
        run_scene(
            args.data_root, args.out, args.suite, scene, args.camera, predictions, cases,
            args.contacts, taps,
        )  # fmt: skip


def tie_free(array: np.ndarray) -> np.ndarray:
    """Fail rather than record a reference for input whose official result is not unique."""
    if len(np.unique(array[:, 0])) != len(array):
        raise RuntimeError("constructed case has tied scores; change the seed")
    return array


def edge_cases(base: np.ndarray, seed: int) -> dict:
    """Constructed inputs. Each must have one well-defined official result.

    Tied scores are represented by exact duplicates only: any other tie makes the
    official (unstable) sorts platform-dependent, so it cannot serve as a reference.
    """
    rng = np.random.RandomState(seed)
    base = tie_free(np.array(base, dtype=np.float32))
    cases = {}
    cases["few"] = base[:20].copy()
    cases["single"] = base[:1].copy()

    widths = base.copy()
    widths[::2, 1] *= 1.8  # beyond the 0.1 m gripper for most
    widths[::7, 1] = -0.01  # negative widths are clipped to zero
    widths[3::11, 1] = 0.1  # exactly the limit as float32
    cases["widths"] = widths

    cases["duplicates"] = np.concatenate([base, base[: len(base) // 2]])

    skewed = base.copy()
    skewed[:, 4:13] += rng.uniform(-2e-6, 2e-6, size=(len(base), 9)).astype(np.float32)
    cases["nonorthonormal"] = skewed

    double = base.astype(np.float64)
    double[:, 13:16] += rng.uniform(-1e-9, 1e-9, size=(len(base), 3))
    double[:, 0] += rng.uniform(0, 1e-9, size=len(base))
    double[:, 1] += rng.uniform(0, 1e-9, size=len(base))
    cases["float64"] = double

    cases["shuffled"] = base[rng.permutation(len(base))]

    dense = np.concatenate([base] * 6)
    dense[:, 13:16] += rng.normal(0, 0.006, size=(len(dense), 3)).astype(np.float32)
    dense[:, 0] += rng.uniform(0, 1e-3, size=len(dense)).astype(np.float32)
    while len(np.unique(dense[:, 0])) != len(dense):  # float32 collisions between copies
        dense[:, 0] += rng.uniform(0, 1e-5, size=len(dense)).astype(np.float32)
    cases["dense"] = dense

    far = base.copy()
    far[::5, 13:16] += np.float32(0.4)
    cases["far"] = far

    deep = base.copy()
    deep[:, 3] = rng.choice([0.0, 0.005, 0.035, 0.06], size=len(base)).astype(np.float32)
    cases["depths"] = deep
    for name, array in cases.items():
        if name != "duplicates":
            tie_free(array)
    return cases


def suite_edge(args, taps: Taps) -> None:
    """Constructed edge cases, one per frame of one scene.

    Case ``i`` is built from the dump's own prediction for frame ``i``, so the grasps
    still belong to the frame they are scored in and reach every stage.
    """
    names = sorted(edge_cases(load_dump(args.dump, args.scene, args.camera, 0), args.seed))
    predictions = {}
    for frame, name in enumerate(names):
        base = load_dump(args.dump, args.scene, args.camera, frame)
        predictions[frame] = edge_cases(base, args.seed + frame)[name]
    run_scene(
        args.data_root, args.out, args.suite, args.scene, args.camera, predictions,
        dict(enumerate(names)), args.contacts, taps,
    )  # fmt: skip


# ---------------------------------------------------------------------- GraspClutter6D


def clutter_variations(array: np.ndarray, rng) -> np.ndarray:
    """Make label grasps exercise what the GraspClutter6D tool does differently.

    A tenth of the rows each: moved away from every object (the background filter must
    drop them), widened beyond the 0.14 m limit (the clip), and given a closing axis
    that is not of unit length (the tool normalises it instead of failing).
    """
    array = array.copy()
    groups = np.array_split(rng.permutation(len(array)), 10)
    array[groups[0], 13] += np.float32(0.3)
    array[groups[1], 1] *= np.float32(1.6)
    rotation = array[groups[2], 4:13].reshape(-1, 3, 3)
    rotation[:, :, 1] *= np.float32(1.0003)
    array[groups[2], 4:13] = rotation.reshape(-1, 9)
    return array


def clutter_dump(args, scene: int, evaluator) -> dict:
    """Frames of an existing prediction dump in the layout of the official tool."""
    predictions = {}
    for frame in args.frames:
        file = args.dump / f"{scene:06d}" / args.camera / f"{evaluator.annId2ImgId(frame):06d}.npy"
        array = np.array(np.load(str(file)), dtype=np.float32)
        if len(array) == 0:  # the official tool cannot score an empty frame
            print(f"skipping empty prediction: scene {scene} frame {frame}")
            continue
        predictions[frame] = array
    return predictions


def suite_clutter(args, taps: Taps) -> None:
    """Grasps scored by the official GraspClutter6D tool: label-derived, or of a dump."""
    from graspclutter6dAPI import GraspClutter6D, GraspClutter6DEval

    for scene in args.scenes:
        if args.dump is not None:
            evaluator = GraspClutter6DEval(str(args.data_root), args.camera, split="test")
            clutter_scene(args, taps, scene, clutter_dump(args, scene, evaluator), evaluator)
            continue
        api = GraspClutter6D(str(args.data_root), camera=args.camera, split="test")
        with open(str(args.data_root / "scenes" / f"{scene:06d}" / "scene_gt.json")) as stream:
            scene_gt = json.load(stream)
        object_ids = sorted({entry["obj_id"] for entries in scene_gt.values() for entry in entries})
        grasp_labels = api.loadGraspLabels(object_ids)
        collision_labels = api.loadCollisionLabels(scene)
        predictions = {}
        for frame in args.frames:
            group = api.loadGrasp(
                scene, annId=frame, format="6d", camera=args.camera, grasp_labels=grasp_labels,
                collision_labels=collision_labels, fric_coef_thresh=args.friction,
                remove_invisible=False,
            )  # fmt: skip
            array = np.array(group.grasp_group_array, dtype=np.float32)
            rng = np.random.RandomState(args.seed + 1000 * scene + frame)
            chosen = rng.choice(len(array), size=min(args.count, len(array)), replace=False)
            array = clutter_variations(array[np.sort(chosen)], rng)
            array[:, 0] += rng.uniform(0.0, 0.05, size=len(array)).astype(np.float32)
            while len(np.unique(array[:, 0])) != len(array):
                array[:, 0] += rng.uniform(0.0, 1e-4, size=len(array)).astype(np.float32)
            predictions[frame] = array
            print(f"label grasps: scene {scene} frame {frame}: {len(array)} of {len(group)}")
        del grasp_labels, collision_labels
        evaluator = GraspClutter6DEval(str(args.data_root), args.camera, split="test")
        clutter_scene(args, taps, scene, predictions, evaluator)


def clutter_scene(args, taps: Taps, scene: int, predictions: dict, evaluator) -> None:
    """Score the frames of one scene with the official tool and store every stage."""
    folder = args.out / args.suite / f"scene_{scene:06d}" / args.camera
    folder.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="oracle_dump_", dir=str(args.out)))
    try:
        dump = work / f"{scene:06d}" / args.camera
        dump.mkdir(parents=True)
        for frame, array in predictions.items():
            np.save(str(dump / f"{evaluator.annId2ImgId(frame):06d}.npy"), array)
        taps.frames.clear()
        started, cpu_started = time.perf_counter(), time.process_time()
        accuracy = evaluator.eval_scene(scene, str(work), return_list=True)[0]
        seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        print()
        if len(taps.frames) != len(predictions):  # frames without a dump are not scored
            raise RuntimeError(f"expected {len(predictions)} tapped frames, got {len(taps.frames)}")
        for record, frame in zip(taps.frames, sorted(predictions)):
            arrays = frame_arrays(record, predictions[frame], accuracy[frame])
            temporary = folder / f".{frame:04d}.tmp.npz"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(str(temporary), str(folder / f"{frame:04d}.npz"))
    finally:
        shutil.rmtree(str(work), ignore_errors=True)
    run = {
        "suite": args.suite,
        "scene": scene,
        "camera": args.camera,
        "frames": sorted(predictions),
        "official_seconds": round(seconds, 2),
        "official_cpu_seconds": round(cpu_seconds, 2),
        "peak_rss_mb_process_so_far": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "mean_precision_of_stored_frames": float(
            np.mean([accuracy[frame] for frame in sorted(predictions)])
        ),
        "versions": versions("graspclutter6dAPI"),
        "api_path": str(args.api),
        "command": " ".join(sys.argv),
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    print(
        f"[{args.suite}] scene {scene} {args.camera}: {len(predictions)} frame(s) stored, "
        f"official time {seconds:.1f} s, mean precision "
        f"{run['mean_precision_of_stored_frames']:.4f}"
    )


# ------------------------------------------------------------------------- primitives


def save(out: Path, name: str, **arrays) -> None:
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / f".{name}.tmp.npz"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temporary), str(out / f"{name}.npz"))
    print(f"[primitives] wrote {name}.npz")


def nms_cases(seed: int) -> dict:
    """Random grasp sets, including pairs placed next to both thresholds."""
    from grasp_nms import nms_grasp

    rng = np.random.RandomState(seed)
    cases = {}
    for index, count in enumerate((1, 2, 17, 300, 2500)):
        grasps = np.zeros((count, 17))
        grasps[:, 0] = rng.permutation(count) / count + rng.uniform(0, 1e-6, count)
        grasps[:, 13:16] = rng.uniform(-0.08, 0.08, size=(count, 3))
        rotation = np.linalg.qr(rng.normal(size=(count, 3, 3)))[0]
        grasps[:, 4:13] = rotation.reshape(count, 9)
        if count >= 300:  # clusters around the thresholds
            twin = grasps[: count // 3].copy()
            direction = rng.normal(size=(len(twin), 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            offset = NMS_TRANSLATION + rng.choice([-1e-9, 0.0, 1e-9, -3e-3], size=(len(twin), 1))
            twin[:, 13:16] += direction * offset
            angle = NMS_ROTATION + rng.choice([-1e-9, 0.0, 1e-9, -0.2], size=len(twin))
            spin = np.zeros((len(twin), 3, 3))
            spin[:, 0, 0] = spin[:, 1, 1] = np.cos(angle)
            spin[:, 0, 1], spin[:, 1, 0] = -np.sin(angle), np.sin(angle)
            spin[:, 2, 2] = 1.0
            twin[:, 4:13] = (twin[:, 4:13].reshape(-1, 3, 3) @ spin).reshape(-1, 9)
            twin[:, 0] = rng.uniform(0, 1, len(twin))
            grasps = np.concatenate([grasps, twin])
        cases[f"nms{index}_in"] = grasps
        cases[f"nms{index}_out"] = np.array(nms_grasp(grasps, NMS_TRANSLATION, NMS_ROTATION))
    return cases


def object_cases(data_root: Path, object_id: int, seed: int) -> dict:
    """Signed-distance queries, transforms, normals and model points of one object."""
    import open3d
    from graspnetAPI.utils.eval_utils import voxel_sample_points

    rng = np.random.RandomState(seed + object_id)
    with open(str(data_root / "dex_models" / f"{object_id:03d}.pkl"), "rb") as stream:
        dexnet = pickle.load(stream)
    sdf = dexnet.sdf
    dims = np.array(sdf.dimensions)
    surface = sdf.surface_points(grid_basis=True)[0]

    inside = rng.uniform(0, dims - 1, size=(400, 3))
    last_cell = rng.uniform(dims - 1.2, dims - 1e-9, size=(100, 3))
    lattice = rng.randint(0, dims, size=(100, 3)).astype(np.float64)
    near = surface[rng.choice(len(surface), 300)] + rng.uniform(-1, 1, size=(300, 3))
    outside = rng.uniform(-30, dims + 30, size=(300, 3))
    edge = np.array([[0.0, 0.0, 0.0], [-0.0, 5.0, 5.0], [99.0, 99.0, 99.0], [99.5, 3.2, 7.7]])
    coords = np.concatenate([inside, last_cell, lattice, near, outside, edge])
    values = np.array([sdf[coord] for coord in coords], dtype=np.float64)

    points = rng.uniform(-0.15, 0.15, size=(200, 3))
    to_grid = sdf.transform_pt_obj_to_grid(points.T).T
    to_object = sdf.transform_pt_grid_to_obj(coords[:200].T).T
    unit = rng.normal(size=(50, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    directions = sdf.transform_pt_grid_to_obj(unit.T, direction=True).T

    candidates = surface[rng.choice(len(surface), 400)] + rng.uniform(-0.4, 0.4, size=(400, 3))
    normal_at, normals = [], []
    for coord in candidates:
        if sdf.on_surface(coord)[0]:
            normal_at.append(coord)
            normals.append(sdf.surface_normal(coord))
    cloud = open3d.io.read_point_cloud(
        str(data_root / "models" / f"{object_id:03d}" / "nontextured.ply")
    )
    sampled = voxel_sample_points(np.array(cloud.points), 0.008)
    return {
        "coords": coords,
        "values": values,
        "points": points,
        "to_grid": to_grid,
        "to_object": to_object,
        "unit": unit,
        "directions": directions,
        "normal_at": np.array(normal_at),
        "normals": np.array(normals),
        "model_points": sampled,
        "surface_threshold": np.float64(sdf.surface_thresh_),
        "object_to_grid_scale": np.float64(sdf.transform_pt_obj_to_grid(1.0)),
        "surface_sample": np.array(sdf.surface_for_signed_val),
    }


def scene_cases(data_root: Path, scene: int, camera: str) -> dict:
    """Object poses and the transformed table for every frame of one scene."""
    from graspnetAPI import GraspNetEval
    from graspnetAPI.utils.eval_utils import create_table_points, transform_points

    evaluator = GraspNetEval(str(data_root), camera, split="custom")
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    object_ids, poses, tables = None, [], {}
    for frame in range(FRAMES_PER_SCENE):
        ids, pose_list, camera_pose, align = evaluator.get_model_poses(scene, frame)
        object_ids = ids if object_ids is None else object_ids
        poses.append(np.array(pose_list))
        if frame % 64 == 0:
            moved = transform_points(table, np.linalg.inv(np.matmul(align, camera_pose)))
            tables[f"table_{frame:04d}"] = moved
    return dict(object_ids=np.array(object_ids), poses=np.array(poses), table=table, **tables)


def suite_primitives(args, taps: Taps) -> None:
    out = args.out / "primitives"
    save(out, "nms", **nms_cases(args.seed))
    for object_id in args.objects:
        save(out, f"object_{object_id:03d}", **object_cases(args.data_root, object_id, args.seed))
    for scene in args.scenes:
        for camera in ("kinect", "realsense"):
            cases = scene_cases(args.data_root, scene, camera)
            save(out, f"scene_{scene:04d}_{camera}", **cases)
    (out / "run.json").write_text(json.dumps({"versions": versions()}, indent=2, sort_keys=True))


# ------------------------------------------------------------------------------- pack


def rows_of(prediction: np.ndarray, grasps: np.ndarray) -> np.ndarray:
    """Rows of a tie-free prediction that the given grasps came from (matched by score)."""
    scores = prediction[:, 0].astype(np.float64)
    order = np.argsort(scores)
    position = np.searchsorted(scores[order], grasps[:, 0])
    rows = order[np.minimum(position, len(order) - 1)]
    if not np.array_equal(scores[rows], grasps[:, 0]):
        raise RuntimeError("a recorded grasp has no row with the same score in its prediction")
    return rows.astype(np.int32)


def suite_pack(args) -> None:
    """Condense recorded frames into one committed file of row indices and tables."""
    arrays, runs = {}, {}
    for scene in args.scenes:
        folder = args.out / args.suite / f"scene_{scene:04d}" / args.camera
        runs[f"scene_{scene:04d}/{args.camera}"] = json.loads((folder / "run.json").read_text())
        for frame in args.frames:
            with np.load(str(folder / f"{frame:04d}.npz")) as recorded:
                prediction = recorded["prediction"]
                if len(np.unique(prediction[:, 0])) != len(prediction):
                    raise SystemExit(f"scene {scene} frame {frame} has tied scores; pick another")
                prefix = f"{scene:04d}/{args.camera}/{frame:04d}/"
                arrays[prefix + "prediction"] = prediction
                arrays[prefix + "selected"] = rows_of(prediction, recorded["nms"])
                arrays[prefix + "owner"] = recorded["owner"].astype(np.int8)
                arrays[prefix + "kept"] = rows_of(prediction, recorded["kept"])
                arrays[prefix + "kept_owner"] = recorded["kept_owner"].astype(np.int8)
                arrays[prefix + "collision"] = recorded["collision"]
                arrays[prefix + "empty"] = recorded["empty"]
                arrays[prefix + "friction"] = recorded["friction"]
                arrays[prefix + "precision"] = recorded["precision"]
    args.to.mkdir(parents=True, exist_ok=True)
    with (args.to / "official_frames.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    provenance = {
        "what": "values recorded from the official graspnetAPI by tools/make_eval_fixtures.py",
        "suite": args.suite,
        "camera": args.camera,
        "scenes": args.scenes,
        "frames": args.frames,
        "predictions": "graspnet-baseline kinect checkpoint dump (network output, no dataset content)",
        "runs": {
            name: {key: run[key] for key in ("command", "recorded", "versions")}
            for name, run in runs.items()
        },
    }
    (args.to / "official_frames.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    size = (args.to / "official_frames.npz").stat().st_size
    print(f"packed {len(args.scenes) * len(args.frames)} frames into {size / 1024:.0f} KiB")


# ------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=None, help="dataset root")
    parser.add_argument("--out", type=Path, default=None, help="fixture directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contacts", action="store_true", help="also store contact points")
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help=suite_dump.__doc__.splitlines()[0])
    dump.add_argument("--dump", type=Path, required=True)
    dump.add_argument("--camera", required=True, choices=("kinect", "realsense"))
    dump.add_argument("--scenes", type=int, nargs="+", required=True)
    dump.add_argument("--frames", type=int, nargs="*", default=None)
    dump.add_argument("--suite", default="dump")
    dump.set_defaults(run=suite_dump)

    label = sub.add_parser("label", help=suite_label.__doc__.splitlines()[0])
    label.add_argument("--camera", required=True, choices=("kinect", "realsense"))
    label.add_argument("--scenes", type=int, nargs="+", required=True)
    label.add_argument("--frames", type=int, nargs="+", required=True)
    label.add_argument("--count", type=int, default=1500)
    label.add_argument("--friction", type=float, default=0.8)
    label.add_argument("--suite", default="label")
    label.set_defaults(run=suite_label)

    edge = sub.add_parser("edge", help=suite_edge.__doc__.splitlines()[0])
    edge.add_argument("--dump", type=Path, required=True)
    edge.add_argument("--camera", required=True, choices=("kinect", "realsense"))
    edge.add_argument("--scene", type=int, required=True)
    edge.add_argument("--suite", default="edge")
    edge.set_defaults(run=suite_edge)

    primitives = sub.add_parser("primitives", help="stage-level references on real objects")
    primitives.add_argument("--objects", type=int, nargs="*", default=[])
    primitives.add_argument("--scenes", type=int, nargs="*", default=[])
    primitives.set_defaults(run=suite_primitives)

    pack = sub.add_parser("pack", help=suite_pack.__doc__.splitlines()[0])
    pack.add_argument("--suite", default="dump")
    pack.add_argument("--camera", required=True, choices=("kinect", "realsense"))
    pack.add_argument("--scenes", type=int, nargs="+", required=True)
    pack.add_argument("--frames", type=int, nargs="+", required=True)
    pack.add_argument("--to", type=Path, required=True)

    clutter = sub.add_parser("clutter", help=suite_clutter.__doc__.splitlines()[0])
    clutter.add_argument("--api", type=Path, required=True, help="clone of graspclutter6dAPI")
    clutter.add_argument("--camera", required=True)
    clutter.add_argument("--scenes", type=int, nargs="+", required=True)
    clutter.add_argument("--frames", type=int, nargs="+", required=True)
    clutter.add_argument("--count", type=int, default=2000)
    clutter.add_argument("--friction", type=float, default=0.8)
    clutter.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="score this prediction dump (layout of the official tool: <scene, 6 digits>/"
        "<camera>/<image id, 6 digits>.npy) instead of label-derived grasps",
    )
    clutter.add_argument("--suite", default="label")
    clutter.set_defaults(run=suite_clutter)

    args = parser.parse_args()
    dataset = "graspclutter6d" if args.command == "clutter" else "graspnet1b"
    if args.out is None:
        args.out = default_output(dataset)
    if args.data_root is None and default_data_root(dataset):
        args.data_root = Path(default_data_root(dataset))
    if args.command == "pack":
        args.out = args.out.resolve()
        suite_pack(args)
        return
    if args.data_root is None or not args.data_root.is_dir():
        parser.error(f"no dataset: pass --data-root or set GRASPPANDA_{dataset.upper()}_ROOT")
    args.data_root = args.data_root.resolve()
    args.out = args.out.resolve()
    if args.out == args.data_root or args.data_root in args.out.parents:
        parser.error("--out must not lie inside the dataset")
    args.out.mkdir(parents=True, exist_ok=True)

    before = dataset_listing(args.data_root)
    install_write_guard(args.data_root)
    taps = Taps(with_contacts=args.contacts)
    if args.command == "clutter":
        sys.path.insert(0, str(args.api.resolve()))
        taps.install("graspclutter6dAPI", "graspclutter6d_eval")
    else:
        taps.install()
    try:
        args.run(args, taps)
    finally:
        if dataset_listing(args.data_root) != before:
            raise SystemExit("the dataset directory listing changed during the oracle run")
    print(f"dataset untouched; fixtures in {args.out}")


if __name__ == "__main__":
    main()
