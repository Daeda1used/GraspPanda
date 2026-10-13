#!/usr/bin/env python
"""Record what the upstream Graspness data pipeline produces, for parity tests.

Run with the interpreter of an environment that can import the reference
implementation (graspnet/graspness_unofficial with its dependencies), not with the
GraspPanda environment::

    $GRASPPANDA_ORACLE_PYTHON tools/make_data_fixtures.py \
        --upstream /path/to/graspness_unofficial \
        --dataset-root $GRASPPANDA_GRASPNET1B_ROOT \
        --out $GRASPPANDA_CACHE/fixtures/data_pipeline

It can also record scene clouds as the official GraspClutter6D tool loads them
(``--clutter-api /path/to/graspclutter6dAPI --clutter-root $GRASPPANDA_GRASPCLUTTER6D_ROOT``).

The script contains no pipeline logic of its own. It imports the upstream dataset class
unmodified, asks it for frames, and observes it from outside:

* ``numpy.random.choice`` and ``numpy.random.random`` are wrapped to log every draw, so
  a test can feed the same indices and augmentation parameters to another pipeline;
* the two upstream helpers that build the organised cloud and the workspace mask are
  wrapped to capture their return values;
* the upstream graspness generator, a script without functions, is executed as it is on
  a throw-away dataset root made of symlinks that exposes only the requested frames
  (it stops with an error at the first frame that is not there, which is expected).

Nothing is written into the upstream tree (bytecode caching is off) or into the dataset;
everything goes to ``--out`` and to a temporary working directory. Fixtures that already
exist are kept, so an interrupted run can be repeated; delete a file to record it again.
The file must stay valid Python 3.8 and must not import grasppanda.
"""

# ruff: noqa: UP006, UP035
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FORMAT = 2
NUM_POINTS = 15000
FRAMES_PER_SCENE = 256

#: (split, camera, scene, frame, with labels): training scenes of both cameras and two
#: test scenes. Upstream cannot serve labels where it has no graspness file for the frame.
DATASET_FRAMES: List[Tuple[str, str, int, int, bool]] = [
    ("train", "realsense", 0, 0, True),
    ("train", "realsense", 37, 100, True),
    ("train", "realsense", 99, 255, True),
    ("train", "kinect", 0, 0, True),
    ("train", "kinect", 62, 31, True),
    ("test_similar", "realsense", 135, 64, True),
    ("test_novel", "kinect", 170, 200, False),
]
#: (camera, scene, frames) for the graspness generator.
GENERATOR_FRAMES: List[Tuple[str, int, List[int]]] = [
    ("realsense", 0, [0, 128]),
    ("realsense", 37, [100]),
    ("kinect", 0, [0]),
    ("kinect", 62, [31]),
    ("realsense", 135, [64]),
]
#: (camera, scene, view) loaded through the official GraspClutter6D tool: every camera,
#: test scenes and a training scene.
CLUTTER_FRAMES: List[Tuple[str, int, int]] = [
    ("realsense-d435", 2, 0),
    ("realsense-d435", 10, 7),
    ("realsense-d435", 5, 5),
    ("realsense-d415", 2, 3),
    ("azure-kinect", 2, 0),
    ("zivid", 10, 12),
]


def dataset_fixture(split: str, camera: str, scene: int, frame: int) -> str:
    return f"dataset_{split}_{camera}_{scene:04d}_{frame:04d}.npz"


def generator_fixture(camera: str, scene: int, frame: int) -> str:
    return f"graspness_{camera}_{scene:04d}_{frame:04d}.npz"


def digest(array: np.ndarray) -> str:
    """Hash of an array's dtype, shape and bytes: equal digests mean equal arrays."""
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


class Spy:
    """Observes the upstream dataset module from outside while it serves one item."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.draws: List[Tuple[str, Any]] = []
        self.captured: Dict[str, Any] = {}
        self._cloud = module.create_point_cloud_from_depth_image
        self._workspace = module.get_workspace_mask
        self._choice = np.random.choice
        self._random = np.random.random

    def __enter__(self) -> Spy:
        def cloud(depth: Any, camera: Any, organized: bool = True) -> Any:
            out = self._cloud(depth, camera, organized=organized)
            self.captured["cloud"] = out
            self.captured["depth_positive"] = np.asarray(depth) > 0
            self.captured["camera"] = np.array(
                [float(v) for v in (camera.fx, camera.fy, camera.cx, camera.cy, camera.scale)]
            )
            return out

        def workspace(
            cloud: Any, seg: Any, trans: Any = None, organized: bool = True, outlier: float = 0
        ) -> Any:
            out = self._workspace(cloud, seg, trans=trans, organized=organized, outlier=outlier)
            self.captured["workspace"] = out
            self.captured["table_from_camera"] = np.array(trans)
            self.captured["outlier"] = outlier
            return out

        def choice(a: Any, size: Any = None, replace: bool = True, p: Any = None) -> Any:
            out = self._choice(a, size, replace, p)
            self.draws.append(("choice", np.array(out)))
            return out

        def random(size: Any = None) -> Any:
            out = self._random(size)
            self.draws.append(("random", float(out)))
            return out

        self.module.create_point_cloud_from_depth_image = cloud
        self.module.get_workspace_mask = workspace
        np.random.choice = choice
        np.random.random = random
        return self

    def __exit__(self, *exc: Any) -> None:
        self.module.create_point_cloud_from_depth_image = self._cloud
        self.module.get_workspace_mask = self._workspace
        np.random.choice = self._choice
        np.random.random = self._random


def record_item(module: Any, dataset: Any, index: int, seed: int) -> Dict[str, np.ndarray]:
    """Ask the upstream dataset for one item and return its output with what was observed."""
    np.random.seed(seed)
    with Spy(module) as spy:
        item = dataset[index]
    out: Dict[str, np.ndarray] = {}
    choices = [value for kind, value in spy.draws if kind == "choice"]
    randoms = [value for kind, value in spy.draws if kind == "random"]
    mask = spy.captured["depth_positive"] & spy.captured["workspace"]
    count = int(mask.sum())
    if dataset.num_points > count:  # upstream keeps every point once, then pads
        out["index"] = np.concatenate([np.arange(count), choices[0]])
    else:
        out["index"] = choices[0]
    for position, rows in enumerate(choices[1:]):
        out[f"label_rows_{position}"] = rows
    out["num_label_draws"] = np.array(len(choices) - 1)
    out["randoms"] = np.array(randoms, dtype=np.float64)
    out["point_clouds"] = item["point_clouds"]
    if "objectness_label" in item:
        out["graspness_label"] = item["graspness_label"]
        out["objectness_label"] = item["objectness_label"]
        out["object_poses"] = np.array(item["object_poses_list"], dtype=np.float32).reshape(
            -1, 3, 4
        )
        for name in ("grasp_points", "grasp_widths", "grasp_scores"):
            for position, array in enumerate(item[name + "_list"]):
                key = f"{name}_{position}"
                out[key + "_sha256"] = np.array(digest(array))
                out[key + "_shape"] = np.array(array.shape)
                out[key + "_head"] = np.array(array[:4])
    out["feats_ones"] = np.array(bool((item["feats"] == 1).all()) and item["feats"].shape[1] == 3)
    out["_mask"] = mask
    out["_cloud"] = spy.captured["cloud"]
    out["_camera"] = spy.captured["camera"]
    out["_workspace"] = spy.captured["workspace"]
    out["_table_from_camera"] = spy.captured["table_from_camera"]
    return out


def record_dataset(
    upstream: Path, root: Path, out: Path, frames: List[Tuple[str, str, int, int, bool]]
) -> List[str]:
    """Record frames through the upstream ``GraspNetDataset``; existing files are kept."""
    written: List[str] = []
    groups: Dict[Tuple[str, str], List[Tuple[int, int, bool]]] = {}
    for split, camera, scene, frame, with_labels in frames:
        if not (out / dataset_fixture(split, camera, scene, frame)).is_file():
            groups.setdefault((split, camera), []).append((scene, frame, with_labels))
    if not groups:
        return written
    sys.path[:0] = [str(upstream / "dataset"), str(upstream / "utils"), str(upstream)]
    import graspnet_dataset as module  # the upstream module, imported as its scripts do

    labels = module.load_grasp_labels(str(root))
    for (split, camera), members in groups.items():
        dataset = module.GraspNetDataset(
            str(root),
            grasp_labels=labels,
            camera=camera,
            split=split,
            num_points=NUM_POINTS,
            remove_outlier=True,
            augment=False,
            load_label=True,
        )
        first_scene = int(dataset.scenename[0].split("_")[1])
        for scene, frame, with_labels in members:
            index = (scene - first_scene) * FRAMES_PER_SCENE + frame
            assert dataset.scenename[index] == f"scene_{scene:04d}"
            assert dataset.frameid[index] == frame
            arrays: Dict[str, np.ndarray] = {}
            variants = {
                # name: (augment, load_label, num_points)
                "plain": (False, True, NUM_POINTS),
                "augmented": (True, True, NUM_POINTS),
                "inference": (False, False, NUM_POINTS),
                "padded": (False, False, 1000000),
            }
            for number, (name, (augment, load_label, num_points)) in enumerate(variants.items()):
                if load_label and not with_labels:
                    continue
                dataset.augment, dataset.load_label, dataset.num_points = (
                    augment,
                    load_label,
                    num_points,
                )
                item = record_item(module, dataset, index, seed=1000 * scene + 10 * frame + number)
                shared = {key: item.pop(key) for key in list(item) if key.startswith("_")}
                for key, value in item.items():
                    arrays[f"{name}/{key}"] = value
                mask = shared["_mask"]
                if "mask" not in arrays:
                    arrays["mask"] = np.packbits(mask)
                    arrays["workspace_mask"] = np.packbits(shared["_workspace"])
                    arrays["image_size"] = np.array(mask.shape)
                    arrays["cloud_masked"] = shared["_cloud"][mask]
                    arrays["camera"] = shared["_camera"]
                    arrays["table_from_camera"] = shared["_table_from_camera"]
                else:  # every variant of a frame must see the same mask
                    assert np.array_equal(arrays["mask"], np.packbits(mask))
            dataset.augment, dataset.load_label, dataset.num_points = False, True, NUM_POINTS
            arrays["raw_cloud"] = np.array(digest(dataset.get_data(index, return_raw_cloud=True)))
            name = dataset_fixture(split, camera, scene, frame)
            np.savez_compressed(str(out / name), **arrays)
            written.append(name)
            print("recorded", name, flush=True)
        del dataset
    return written


def record_generator(
    upstream: Path, root: Path, out: Path, work: Path, frames: List[Tuple[str, int, List[int]]]
) -> List[str]:
    """Run the upstream graspness generator, unmodified; existing files are kept."""
    written = []
    for camera, scene, requested in frames:
        chosen = [f for f in requested if not (out / generator_fixture(camera, scene, f)).is_file()]
        if not chosen:
            continue
        box = work / f"box_{camera}_{scene:04d}"
        source = root / "scenes" / f"scene_{scene:04d}" / camera
        target = box / "scenes" / "scene_0000" / camera
        for sub, suffix in (
            ("depth", "png"),
            ("label", "png"),
            ("meta", "mat"),
            ("annotations", "xml"),
        ):
            (target / sub).mkdir(parents=True)
            for position, frame in enumerate(chosen):
                os.symlink(
                    str(source / sub / f"{frame:04d}.{suffix}"),
                    str(target / sub / f"{position:04d}.{suffix}"),
                )
        # The generator indexes camera poses by frame number: expose the chosen rows.
        np.save(str(target / "camera_poses.npy"), np.load(str(source / "camera_poses.npy"))[chosen])
        os.symlink(str(source / "cam0_wrt_table.npy"), str(target / "cam0_wrt_table.npy"))
        os.symlink(str(root / "grasp_label"), str(box / "grasp_label"))
        (box / "collision_label").mkdir()
        os.symlink(
            str(root / "collision_label" / f"scene_{scene:04d}"),
            str(box / "collision_label" / "scene_0000"),
        )
        command = [
            sys.executable,
            "-B",
            str(upstream / "dataset" / "generate_graspness.py"),
            "--dataset_root",
            str(box),
            "--camera_type",
            camera,
        ]
        result = subprocess.run(command, cwd=str(work), capture_output=True)
        produced = box / "graspness" / "scene_0000" / camera
        for position, frame in enumerate(chosen):
            file = produced / f"{position:04d}.npy"
            if not file.is_file():
                sys.stderr.write(result.stderr.decode(errors="replace")[-3000:])
                raise SystemExit(f"the upstream generator did not write {file}")
            name = generator_fixture(camera, scene, frame)
            np.savez_compressed(str(out / name), graspness=np.load(str(file)))
            written.append(name)
            print("recorded", name, flush=True)
        shutil.rmtree(str(box))
    return written


def clutter_fixture(camera: str, scene: int, view: int) -> str:
    return f"clutter_{camera}_{scene:06d}_{view:02d}.npz"


def record_clutter(
    api: Path, root: Path, out: Path, frames: List[Tuple[str, int, int]]
) -> List[str]:
    """Load scene clouds with the official GraspClutter6D tool; existing files are kept.

    Its defaults are used: a crop to the image rectangle around the segmentation with a
    margin of 0.1, and only pixels with depth.
    """
    pending = [f for f in frames if not (out / clutter_fixture(*f)).is_file()]
    if not pending:
        return []
    sys.path.insert(0, str(api))
    from graspclutter6dAPI.graspclutter6d import GraspClutter6D  # the official class

    written = []
    readers: Dict[str, Any] = {}
    for camera, scene, view in pending:
        if camera not in readers:
            readers[camera] = GraspClutter6D(str(root), camera=camera, split="all")
        points, _ = readers[camera].loadScenePointCloud(scene, camera, view, format="numpy")
        name = clutter_fixture(camera, scene, view)
        np.savez_compressed(str(out / name), points=points)
        written.append(name)
        print("recorded", name, flush=True)
    return written


def source_digest(upstream: Path) -> Dict[str, str]:
    """Hashes of the upstream files that define the recorded behaviour."""
    files = (
        "dataset/graspnet_dataset.py",
        "dataset/generate_graspness.py",
        "utils/data_utils.py",
        "knn/knn_modules.py",
    )
    return {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest() for name in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--upstream", type=Path, help="graspness_unofficial checkout")
    parser.add_argument("--dataset-root", type=Path, help="GraspNet-1Billion")
    parser.add_argument("--clutter-api", type=Path, help="graspclutter6dAPI checkout")
    parser.add_argument("--clutter-root", type=Path, help="GraspClutter6D")
    parser.add_argument("--out", required=True, type=Path, help="directory for the fixtures")
    parser.add_argument("--only", choices=["dataset", "generator", "clutter"], help="one part")
    args = parser.parse_args()
    sys.dont_write_bytecode = True  # upstream trees are read-only: leave no __pycache__ there
    out = args.out.resolve()
    graspnet = args.upstream is not None and args.dataset_root is not None
    clutter = args.clutter_api is not None and args.clutter_root is not None
    if not graspnet and not clutter:
        parser.error("give --upstream with --dataset-root, or --clutter-api with --clutter-root")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="grasppanda-oracle-"))
    os.chdir(str(work))  # whatever upstream writes relative to the working directory lands here
    index_file = out / "index.json"
    index = json.loads(index_file.read_text()) if index_file.is_file() else {}
    try:
        written: List[str] = []
        if graspnet:
            upstream, root = args.upstream.resolve(), args.dataset_root.resolve()
            if args.only in (None, "generator"):
                written += record_generator(upstream, root, out, work, GENERATOR_FRAMES)
            if args.only in (None, "dataset"):
                written += record_dataset(upstream, root, out, DATASET_FRAMES)
            index.update({"upstream": source_digest(upstream), "num_points": NUM_POINTS})
        if clutter and args.only in (None, "clutter"):
            api = args.clutter_api.resolve()
            written += record_clutter(api, args.clutter_root.resolve(), out, CLUTTER_FRAMES)
            source = api / "graspclutter6dAPI" / "graspclutter6d.py"
            index["clutter_api"] = hashlib.sha256(source.read_bytes()).hexdigest()
    finally:
        os.chdir("/")
        shutil.rmtree(str(work), ignore_errors=True)
    index.update(
        {
            "format": FORMAT,
            "files": sorted(path.name for path in out.glob("*.npz")),
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
    )
    index_file.write_text(json.dumps(index, indent=1, sort_keys=True))
    print("recorded", len(written), "new fixtures;", len(index["files"]), "in", out)


if __name__ == "__main__":
    main()
