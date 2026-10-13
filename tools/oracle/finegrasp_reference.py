"""Record what the reference FineGrasp implementation produces on this machine.

Runs the authors' code (``robo_orchard_lab``, Horizon Robotics) with the authors'
weights over chosen GraspNet-1Billion test frames and writes the decoded grasps in the
layout of the official tools, once as the model emits them (``raw``) and once after the
model-free collision filter of the authors' metric (``cd``: threshold 0.01, voxel
0.01 m, approach 0.05 m). The dumps are scored afterwards with GraspPanda's evaluator.

This script is an oracle: it never imports GraspPanda and is executed with an
interpreter that has the reference's compiled dependencies (Python >= 3.10, torch,
MinkowskiEngine 0.5.4, the pointnet2 and knn extensions, Open3D, graspnetAPI)::

    /path/to/env/bin/python -B tools/oracle/finegrasp_reference.py \
        --site /scratch/site --pointnet2 /path/to/graspness_unofficial \
        --project /scratch/robo_orchard_lab/projects/finegrasp_graspnet1b \
        --pipeline "$GRASPPANDA_WEIGHTS/finegrasp/finegrasp_pipeline" \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --scenes 100-129 --frame-step 64 \
        --save-inputs --out "$GRASPPANDA_CACHE/oracle/finegrasp_reference_step64"

What the reference is, and what this process adds to make it run
-----------------------------------------------------------------

``--site`` directories are put in front of ``sys.path``. They hold the released package
(``pip install --no-deps --target <dir> robo_orchard_lab==0.4.0 robo_orchard_core==0.4.0``)
and the pure-Python packages it imports that the interpreter lacks; nothing is installed
into the interpreter's environment. ``--project`` is ``projects/finegrasp_graspnet1b``
of the public repository (``HorizonRobotics/robo_orchard_lab`` at ``221ec0e``), whose
configuration builds the dataset, the model and the metric. The FineGrasp sources of
that commit and of release 0.4.0 are the same files except for the base class of the
processor; the manifest records the SHA-256 of every reference file that ran.

The records in the cache were made with an interpreter that had torch 2.4.0,
MinkowskiEngine 0.5.4, the two extensions, Open3D, graspnetAPI, safetensors and
huggingface_hub 1.x. Beside the two wheels above, its site directory held (all installed
with ``--no-deps --target``): pydantic 2.10.6, pydantic-core 2.27.2, annotated-types,
pydantic-settings, python-dotenv, numpydantic, deprecated, wrapt, rtoml, ordered-set,
gymnasium, farama-notifications, cloudpickle, accelerate, timeout-decorator,
sortedcontainers, pyzstd, transformers 4.49.0 with tokenizers, regex and
huggingface_hub < 1.0, and, because the FineGrasp package imports a second backbone that
the released model does not use, spconv-cu118 2.3.8 with cumm-cu118, pccm, ccimport,
fire, lark, portalocker, pybind11, termcolor and ninja, and torch_geometric with xxhash,
aiohttp and jinja2.

Three things are supplied here because the reference does not ship them:

* ``robo_orchard_lab.ops.knn.knn_modules``. The label matching imports it, the
  repository and the wheel do not contain it. It is the usual wrapper around the
  ``knn_pytorch`` extension (the label repository of the same authors carries one), so
  this process registers a module of that name that calls the compiled extension.
* ``pointnet2``. The project README builds it from Scale-Balanced-Grasp; any tree of
  the GraspNet-baseline lineage has the same functions. ``--pointnet2`` names the
  directory that contains the ``pointnet2`` sources matching the compiled ``_ext``.
* The normal maps ``scenes/<scene>/<camera>/normal/<frame>.npy``. The dataset class
  reads them and divides by 255; neither the files nor their generator are published.
  With ``--normals image`` (the default) this script writes the map of every frame it
  loads itself: the estimator of the paper's equation 5 (cross product of the two
  tangents of the organised cloud, :func:`normal_image`), oriented towards the camera
  and stored as an 8-bit normal image, ``(n + 1) / 2 * 255``. Orientation and encoding
  were identified with the released weights: on 20 test frames this input scores 69.4
  AP, the same normals pointing away 59.5, unit vectors instead of the image encoding
  31.7 to 36.8, and no normals at all 43.1. ``--normals processor`` is what the released
  inference processor feeds instead, unit normals estimated by Open3D on the sampled
  cloud (35.0 AP on the same frames), times ``--normal-scale``.

Point sampling is made reproducible: ``numpy.random.seed(scene * 1000 + frame)`` is
called before each frame is loaded. ``--save-inputs`` stores the sampled cloud and its
normals, so that another implementation can be run on the very same input.

Precision. TF32 is switched off (``--tf32 default`` leaves the flags as users get them).
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import types

import numpy as np

REFERENCE_FILES = (
    "models/finegrasp/finegrasp.py",
    "models/finegrasp/head/head.py",
    "models/finegrasp/backbone/minkunet.py",
    "models/finegrasp/utils.py",
    "models/finegrasp/losses.py",
    "models/finegrasp/processor.py",
    "dataset/graspnet1b/ecograsp1b_dataset.py",
    "dataset/graspnet1b/metrics.py",
)
PROJECT_FILES = ("configs/base_config.py", "configs/config_finegrasp_minkunet.py")
COLLISION = {"voxel_size": 0.01, "approach": 0.05, "threshold": 0.01}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand(ranges):
    """``"100-104,130"`` -> [100, 101, 102, 103, 104, 130]."""
    scenes = []
    for part in ranges.split(","):
        first, _, last = part.partition("-")
        scenes.extend(range(int(first), int(last or first) + 1))
    return scenes


def add_reference_arguments(parser):
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="directory put in front of sys.path (the released reference package and the "
        "pure-Python packages it needs); may be repeated",
    )
    parser.add_argument(
        "--pointnet2", required=True, help="directory that contains the `pointnet2` sources"
    )
    parser.add_argument(
        "--project", required=True, help="projects/finegrasp_graspnet1b of the reference repository"
    )
    parser.add_argument(
        "--project-commit",
        default="221ec0e",
        help="commit of that tree, recorded in the manifest",
    )
    parser.add_argument("--pipeline", required=True, help="directory with model.safetensors")
    parser.add_argument("--dataset-root", required=True, help="GraspNet-1Billion root")
    parser.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--tf32", default="off", choices=["off", "default"])
    parser.add_argument(
        "--normals",
        default="image",
        choices=["image", "processor"],
        help="where the normals come from, see the module docstring",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=1.0,
        help="with --normals processor: factor on the unit normals (the dataset class "
        "divides stored maps by 255, the released processor does not)",
    )


def normal_image(depth, intrinsics, depth_scale, step=1, towards_camera=True):
    """The 8-bit normal image ``(H, W, 3)`` this oracle hands the reference dataset.

    ``n = (p(v + s, u) - p(v - s, u)) x (p(v, u + s) - p(v, u - s))``, normalised, on the
    organised cloud of the whole image (edge-replicated borders, missing pixels on the
    camera centre, zero where the product vanishes); the opposite sign without
    ``towards_camera``. Stored as ``uint8((n + 1) / 2 * 255)``.
    """
    z = (depth / np.float32(depth_scale)).astype(np.float32).astype(np.float64)
    height, width = z.shape
    cloud = np.empty((height, width, 3), dtype=np.float64)
    cloud[..., 0] = (np.arange(width) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    cloud[..., 1] = (np.arange(height) - intrinsics[1, 2])[:, None] * z / intrinsics[1, 1]
    cloud[..., 2] = z
    padded = np.pad(cloud, ((step, step), (step, step), (0, 0)), mode="edge")
    across = padded[step:-step, 2 * step :] - padded[step:-step, : -2 * step]
    down = padded[2 * step :, step:-step] - padded[: -2 * step, step:-step]
    normal = np.cross(down, across) if towards_camera else np.cross(across, down)
    length = np.sqrt((normal * normal).sum(axis=-1, keepdims=True))
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)
    return ((normal + 1.0) / 2.0 * 255.0).astype(np.uint8)


def install_knn():
    """Register ``robo_orchard_lab.ops.knn.knn_modules``, which the reference imports and
    does not ship: the standard wrapper around the compiled ``knn_pytorch`` extension."""
    import torch

    try:  # a package around the extension, or the bare extension module
        extension = importlib.import_module("knn_pytorch.knn_pytorch")
    except ImportError:
        extension = importlib.import_module("knn_pytorch")

    def knn(ref, query, k=1):
        """Indices (1-based) of the ``k`` nearest ``ref`` points of every ``query`` point."""
        device = ref.device
        ref = ref.float().to(device)
        query = query.float().to(device)
        inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
        extension.knn(ref, query, inds)
        return inds

    import robo_orchard_lab.ops as ops

    package = types.ModuleType("robo_orchard_lab.ops.knn")
    package.__path__ = []
    module = types.ModuleType("robo_orchard_lab.ops.knn.knn_modules")
    module.knn = knn
    package.knn_modules = module
    ops.knn = package
    sys.modules[package.__name__] = package
    sys.modules[module.__name__] = module
    return getattr(extension, "__file__", "builtin")


class Reference:
    """The reference package, its project configuration and a few facts about both."""

    def __init__(self, args):
        sys.dont_write_bytecode = True
        self.args = args
        os.chdir(tempfile.mkdtemp(prefix="finegrasp_oracle_"))
        for directory in reversed(args.site):
            sys.path.insert(0, os.path.abspath(directory))
        pointnet2 = os.path.abspath(args.pointnet2)
        sys.path.append(pointnet2)  # the package `pointnet2`
        sys.path.append(os.path.join(pointnet2, "pointnet2"))  # its bare `import pytorch_utils`

        import torch

        self.torch = torch
        self.tf32_defaults = {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        }
        if args.tf32 == "off":
            self.set_tf32(False)

        import robo_orchard_lab

        self.package = robo_orchard_lab
        self.knn_extension = install_knn()
        import robo_orchard_lab.models.finegrasp.finegrasp as model_module
        import robo_orchard_lab.models.finegrasp.utils as utils_module
        from robo_orchard_lab.dataset.collates import collate_batch_dict
        from robo_orchard_lab.models.finegrasp import FineGraspProcessorCfg

        self.model_module = model_module
        self.utils_module = utils_module
        self.collate = collate_batch_dict
        self.processor = FineGraspProcessorCfg()()  # seeds every generator with 2025

        project = os.path.abspath(args.project)
        path = os.path.join(project, "configs", "config_finegrasp_minkunet.py")
        spec = importlib.util.spec_from_file_location("config_finegrasp_minkunet", path)
        self.config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config)
        self.project = project
        self.root = os.path.abspath(args.dataset_root)
        self.pipeline = os.path.abspath(args.pipeline)
        self.device = torch.device("cuda:0")
        self._views = []

    def set_tf32(self, enabled):
        """Switch TF32 off (``False``) or back to what the library sets by default (``None``)."""
        defaults = self.tf32_defaults
        self.torch.backends.cuda.matmul.allow_tf32 = (
            defaults["matmul"] if enabled is None else enabled
        )
        self.torch.backends.cudnn.allow_tf32 = defaults["cudnn"] if enabled is None else enabled

    # ---------------------------------------------------------------------- model

    def trainer_cfg(self, data_root):
        cfg = self.config.get_trainer_cfg()
        return cfg.model_copy(update={"data_root": str(data_root)})

    def build_model(self, training):
        """The model of the project configuration (with its losses) and the released weights."""
        from safetensors.torch import load_file

        model_cfg, model = self.config.build_model()
        with open(os.path.join(self.pipeline, "model.config.json")) as handle:
            released = json.load(handle)
        for key, value in released.items():
            if key in ("__config_type__", "class_type", "loss"):
                continue
            if getattr(model_cfg, key) != value:
                raise RuntimeError(
                    f"the project configuration has {key} = {getattr(model_cfg, key)!r}, the "
                    f"released model.config.json {value!r}"
                )
        state = load_file(os.path.join(self.pipeline, "model.safetensors"))
        model.load_state_dict(state, strict=True)
        model = model.to(self.device)
        return model_cfg, (model.train() if training else model.eval())

    # -------------------------------------------------------------------- dataset

    def dataset_view(self, scenes, extra=()):
        """A directory that shows the reference dataset class the chosen scenes.

        Everything is a symbolic link into the dataset except ``normal.npy``, the map
        every frame's normal path is pointed at (see the module docstring).
        """
        view = tempfile.mkdtemp(prefix="finegrasp_dataset_")
        os.mkdir(os.path.join(view, "scenes"))
        for scene in scenes:
            name = "scene_%04d" % scene
            os.symlink(os.path.join(self.root, "scenes", name), os.path.join(view, "scenes", name))
        for name, target in extra:
            os.symlink(target, os.path.join(view, name))
        np.save(os.path.join(view, "normal.npy"), np.zeros((720, 1280, 3), dtype=np.uint8))
        self._views.append(view)
        return view

    def datasets(self, view):
        """The training and the evaluation dataset of the project configuration."""
        _, _, train_loader, test_loader = self.config.build_dataset(self.trainer_cfg(view))
        datasets = train_loader.dataset, test_loader.dataset
        for dataset in datasets:
            dataset.normal_map = os.path.join(view, "normal.npy")
            dataset.normalpath = [dataset.normal_map] * len(dataset.normalpath)
        return datasets

    def normal_map(self, dataset, index):
        """The normal image of a frame, as this oracle generates it."""
        import scipy.io as scio
        from PIL import Image

        depth = np.array(Image.open(dataset.depthpath[index]))
        meta = scio.loadmat(dataset.metapath[index])
        scale = float(meta["factor_depth"].reshape(-1)[0])
        return normal_image(depth, np.asarray(meta["intrinsic_matrix"], dtype=np.float64), scale)

    def load(self, dataset, index):
        """One sample of the reference dataset, with the normals of ``--normals``."""
        image = self.args.normals == "image"
        stored = self.normal_map(dataset, index) if image else np.zeros((720, 1280, 3), np.uint8)
        np.save(dataset.normal_map + ".tmp.npy", stored)
        os.replace(dataset.normal_map + ".tmp.npy", dataset.normal_map)
        sample = dataset[index]
        if not image:
            normal = self.processor.get_normal(sample["point_clouds"]) * self.args.normal_scale
            sample["cloud_normal"] = normal.astype(np.float32)
        return sample

    def to_device(self, batch):
        moved = {}
        for key, value in batch.items():
            if self.torch.is_tensor(value):
                moved[key] = value.to(self.device)
            elif isinstance(value, list) and value and isinstance(value[0], list):
                moved[key] = [
                    [item.to(self.device) if self.torch.is_tensor(item) else item for item in frame]
                    for frame in value
                ]
            else:
                moved[key] = value
        return moved

    # ------------------------------------------------------------------- manifest

    def manifest(self):
        import MinkowskiEngine as ME
        import open3d

        package = os.path.dirname(self.package.__file__)
        return {
            "reference_package": {
                "name": "robo_orchard_lab",
                "version": self.package.__version__,
                "location": package,
                "sha256": {name: sha256(os.path.join(package, name)) for name in REFERENCE_FILES},
            },
            "reference_project": {
                "location": self.project,
                "commit": f"{self.args.project_commit} (as stated)",
                "sha256": {
                    name: sha256(os.path.join(self.project, name)) for name in PROJECT_FILES
                },
            },
            "supplied_by_this_process": {
                "robo_orchard_lab.ops.knn.knn_modules": f"wrapper around {self.knn_extension}",
                "pointnet2": os.path.abspath(self.args.pointnet2),
                "normals": (
                    "8-bit normal image of normal_image(): cross product of the tangents of "
                    "the organised cloud, towards the camera, uint8((n + 1) / 2 * 255)"
                    if self.args.normals == "image"
                    else "FineGraspProcessor.get_normal on the sampled cloud (Open3D hybrid "
                    f"search, radius 0.1, 30 neighbours) x {self.args.normal_scale}"
                ),
            },
            "weights": os.path.join(self.pipeline, "model.safetensors"),
            "weights_sha256": sha256(os.path.join(self.pipeline, "model.safetensors")),
            "dataset_root": self.root,
            "camera": self.args.camera,
            "normals": self.args.normals,
            "normal_scale": self.args.normal_scale if self.args.normals == "processor" else None,
            "tf32": self.args.tf32,
            "tf32_defaults": self.tf32_defaults,
            "versions": {
                "python": sys.version.split()[0],
                "torch": self.torch.__version__,
                "MinkowskiEngine": ME.__version__,
                "open3d": open3d.__version__,
                "numpy": np.__version__,
                "gpu": self.torch.cuda.get_device_name(0),
            },
        }

    def close(self):
        """Remove the temporary dataset views: links and the normal map, never their targets."""
        for view in self._views:
            scenes = os.path.join(view, "scenes")
            for folder in (scenes, view):
                for name in sorted(os.listdir(folder)):
                    path = os.path.join(folder, name)
                    if os.path.islink(path) or name == "normal.npy":
                        os.unlink(path)
            os.rmdir(scenes)
            os.rmdir(view)
        self._views = []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_reference_arguments(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", default="100-129", help="test scenes, e.g. 100-104,130")
    parser.add_argument("--frame-step", type=int, default=64)
    parser.add_argument("--save-inputs", action="store_true", help="store clouds and normals")
    args = parser.parse_args()
    if args.camera != "realsense":
        parser.error("the project configuration and the released weights are RealSense only")

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    reference = Reference(args)
    torch = reference.torch
    scenes = expand(args.scenes)
    view = reference.dataset_view(range(100, 190))
    _, dataset = reference.datasets(view)
    model_cfg, model = reference.build_model(training=False)
    # The metric of the project configuration leaves the filter off (threshold -1); the
    # filtered dump uses the threshold of its commented-out line and of the processor.
    metrics = {}
    for variant, threshold in (("raw", -1), ("cd", COLLISION["threshold"])):
        _, metric = reference.config.build_metric(
            model_cfg, reference.trainer_cfg(view), os.path.join(out, variant)
        )
        if metric.voxel_size_cd != COLLISION["voxel_size"]:
            raise RuntimeError(f"unexpected collision voxel size {metric.voxel_size_cd}")
        metric.collision_thresh = threshold
        metrics[variant] = metric

    frames = []
    tic = time.time()
    for scene in scenes:
        for frame in range(0, 256, args.frame_step):
            index = (scene - 100) * 256 + frame
            name = "scene_%04d" % scene
            assert dataset.scenename[index] == name and dataset.frameid[index] == frame
            np.random.seed(scene * 1000 + frame)
            sample = reference.load(dataset, index)
            batch = reference.to_device(reference.collate([sample]))
            with torch.no_grad():
                end_points = model(batch)
                for metric in metrics.values():
                    metric.update(batch, end_points)
                angle_class = end_points["grasp_angle_pred"][0].argmax(dim=0)
                depth_class = end_points["grasp_depth_pred"][0].argmax(dim=0)
            counts = {}
            for variant in metrics:
                path = os.path.join(out, variant, name, args.camera, "%04d.npy" % frame)
                counts[variant] = int(np.load(path).shape[0])
            if args.save_inputs:
                path = os.path.join(out, "inputs", name, args.camera, "%04d.npz" % frame)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(
                    path + ".tmp.npz",
                    points=sample["point_clouds"],
                    normals=sample["cloud_normal"],
                )
                os.replace(path + ".tmp.npz", path)
            frames.append(
                {
                    "scene": scene,
                    "frame": frame,
                    "grasps": counts["raw"],
                    "collisions": counts["raw"] - counts["cd"],
                    # how often the never-supervised extra class won the decoder arg-max
                    "extra_angle_class": int((angle_class == model_cfg.num_angle).sum()),
                    "extra_depth_class": int((depth_class == model_cfg.num_depth).sum()),
                }
            )
            print(json.dumps(frames[-1]), flush=True)

    manifest = reference.manifest()
    manifest.update(
        {
            "num_points": int(dataset.num_sample_points),
            "collision_filter": COLLISION,
            "sampling_seed": "numpy.random.seed(scene * 1000 + frame) before each frame",
            "seconds": round(time.time() - tic, 1),
            "frames": frames,
        }
    )
    reference.close()
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("wrote", os.path.join(out, "manifest.json"))


if __name__ == "__main__":
    main()
