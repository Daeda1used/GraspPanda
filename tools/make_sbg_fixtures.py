"""Record reference values of Scale-Balanced-Grasp from the upstream implementation.

This script is the *oracle recorder* of the Scale-Balanced-Grasp parity tests. It does
not import GraspPanda and is run with the interpreter of an environment that has the
upstream dependencies (PyTorch, the compiled ``pointnet2`` and ``knn_pytorch``
extensions, graspnetAPI, Open3D), with ``-B`` so that no bytecode is written. It is
written for the Python 3.7 of that environment:

    $GRASPPANDA_SBG_ORACLE_PYTHON -B tools/make_sbg_fixtures.py \
        --reference /path/to/pristine/Scale-Balanced-Grasp \
        --checkpoint /path/to/checkpoint-rs.tar \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" \
        --tolerance-dir /path/to/upstream/tolerance \
        --health --out "$GRASPPANDA_CACHE/fixtures/scale_balanced_grasp"

``--reference`` must be the pinned commit (``mahaoxiang822/Scale-Balanced-Grasp`` @
``995d6f8``) without local edits. It is used read-only: the working directory is a
fresh temporary directory and nothing is written into the tree. The network recorded is
``GraspNet_MSCQ`` with the scale-balanced loss, as the upstream ``train.py`` and
``test.py`` build it; object-balanced sampling and noisy-clean mixing are not involved.
The upstream forward pass is run unmodified; what flows between its stages is recorded
through forward hooks and by wrapping the few free functions it calls.

The upstream loss module reads ``statistic/scale_distribution.npy`` relative to the
working directory when it is imported. The temporary working directory therefore
contains a link named ``statistic`` to that directory of the reference tree.

Two things about the environment, neither of which touches an upstream file. The
upstream dataset module imports ``torch._six.container_abcs``, which PyTorch 1.9
removed; the name is supplied before the import and the manifest records it. And the
matrix products of the reference need the cuBLAS libraries of its own environment: if
``LD_LIBRARY_PATH`` puts another CUDA installation first, every ``nn.Linear`` fails with
``CUBLAS_STATUS_NOT_SUPPORTED``; start the interpreter with the environment's ``lib``
directory first on that path.

What is recorded (float32 unless stated); ``manifest.json`` lists every file:

``health/{raw,cd}/scene_XXXX/<camera>/NNNN.npy`` and ``health_inputs.npz`` (``--health``)
    Step 0, "is the checkpoint healthy?": the upstream network's decoded grasps on a
    small fixed set of test frames, without (``raw``, the shipped test command) and with
    (``cd``) the upstream model-free collision detection, in the layout the benchmark
    evaluator reads, plus the sampled input clouds.
``infer_<scene>_<frame>.npz``
    One test frame: the input cloud as the upstream loader produced it, the neighbour
    lists and outputs of the two transformer blocks, the seeds and seed features, the
    approach head's hidden activation, objectness and view scores, the chosen views,
    per radius and depth the cylinder neighbours and grouped offsets, the crop features
    of every radius, their fusion, the gated seed residual, the features the heads
    receive, the head outputs, the decoded grasps and the collision decisions.
``train_batch.npz`` and ``train_labels.npz``
    One training batch with labels: the same stage outputs in training mode, the
    label-matching results, the scale-balanced weight of every seed with the prior it
    was read from, the graspable target, every loss term and logged metric, a few
    parameter gradients and the total gradient norm.
``scale_distribution.npy``
    A byte-for-byte copy of the reference's scale statistic, so that the tests of its
    reader do not depend on where the reference tree was when the fixtures were made.

Precision. PyTorch 1.10 enables TF32 for matrix products and cuDNN convolutions by
default on Ampere and newer GPUs. The fixtures are recorded with TF32 off, in true
float32; the manifest reports how far a second identical run and the PyTorch defaults
move every stage, which is the scale a comparison has to be read against.
"""

from __future__ import annotations

import argparse
import collections.abc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REFERENCE_COMMIT = "995d6f892474c91fbfcdf58f7d79589be4cbb41c"
INFERENCE_FRAMES = (
    (100, 0),
    (118, 137),
    (145, 64),
    (175, 200),
)  # (scene, frame): seen ×2, similar, novel
TRAINING_FRAMES = ((5, 17), (63, 100))  # the reference trains with batches of two
HEALTH_SCENES = (100, 101, 102, 103, 104)
HEALTH_FRAMES = (0, 64, 128, 192)
CYLINDER_RADIUS = 0.08
#: Name of the copy of the reference's scale statistic in the output directory.
STATISTIC_COPY = "scale_distribution.npy"
STAGES = (
    "transformer1_out",
    "seed_features",
    "approach_hidden",
    "objectness",
    "view_score",
    "view_index",
    "grouped",
    "fused",
    "gate",
    "crop",
    "score_pred",
    "angle_pred",
    "width_pred",
    "tolerance_pred",
)
GRADIENTS = (
    "view_estimator.backbone.sa1.mlp_module.layer0.conv.weight",
    "view_estimator.backbone.transformer1.fc1.weight",
    "view_estimator.backbone.transformer2.w_qs.weight",
    "view_estimator.backbone.transformer2.fc_gamma.2.weight",
    "view_estimator.vpmodule.conv3.weight",
    "grasp_generator.crop1.mlps.layer0.conv.weight",
    "grasp_generator.crop4.mlps.layer2.conv.weight",
    "grasp_generator.fuse_multi_scale.weight",
    "grasp_generator.gate_fusion.0.weight",
    "grasp_generator.operation.conv3.weight",
    "grasp_generator.tolerance.conv3.weight",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_commit(reference):
    """Commit of the reference tree: its ``.git/HEAD`` if it is a checkout, else unknown."""
    head_file = Path(reference) / ".git" / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text().strip()
    if head.startswith("ref:"):
        ref = Path(reference) / ".git" / head.split(" ", 1)[1]
        return ref.read_text().strip() if ref.is_file() else head
    return head


def tree_digest(reference, folders=("models", "utils", "dataset", "pointnet2", "knn")):
    """One hash over the Python sources the oracle imports, to pin what was run."""
    digest = hashlib.sha256()
    for folder in folders:
        for path in sorted((Path(reference) / folder).glob("*.py")):
            digest.update(path.relative_to(reference).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def numpy(value):
    return value.detach().cpu().numpy()


def set_tf32(torch, matmul, cudnn):
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def to_device(batch, device):
    """Move a collated upstream batch (tensors and nested lists of tensors) to a device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [[item.to(device) for item in frame] for frame in value]
        else:
            moved[key] = value.to(device)
    return moved


def difference(first, second, keys):
    """Largest absolute difference per recorded stage between two runs."""
    report = {}
    for key in keys:
        if key not in first or key not in second:
            continue
        if first[key].shape != second[key].shape:
            report[key] = f"shape {first[key].shape} vs {second[key].shape}"
        elif first[key].dtype.kind in "iub":
            report[key + "_mismatches"] = int((first[key] != second[key]).sum())
        else:
            report[key] = float(np.abs(first[key].astype(np.float64) - second[key]).max())
    return report


class Recorder:
    """Collects tensors from hooks and wrapped functions during one forward pass."""

    def __init__(self, net, query_module, transformer_module):
        self.net = net
        self.query_module = query_module
        self.transformer_module = transformer_module
        self.data = {}
        self.cylinder_index = []
        self.grouped = []
        self.neighbours = []
        self.handles = []
        self.originals = []

    def _wrap(self, owner, name, factory):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, factory(original))

    def install(self):
        net = self.net
        backbone = net.view_estimator.backbone

        def transformer(name):
            def hook(module, inputs, output):
                self.data[name + "_out"] = numpy(output).transpose(0, 2, 1).copy()  # (B, N, C)

            return hook

        for name in ("transformer1", "transformer2"):
            self.handles.append(getattr(backbone, name).register_forward_hook(transformer(name)))

        def index_factory(original):
            def wrapper(points, idx):
                # A block first gathers the coordinates of its neighbours: (B, N, k) indices.
                if idx.dim() == 3 and points.size(-1) == 3:
                    self.neighbours.append(numpy(idx).astype(np.int32))
                return original(points, idx)

            return wrapper

        self._wrap(self.transformer_module, "index_points", index_factory)

        def approach_hidden(module, inputs):
            self.data["approach_hidden"] = numpy(inputs[0]).transpose(0, 2, 1).copy()  # (B, M, C)

        approach = net.view_estimator.vpmodule
        self.handles.append(approach.conv3.register_forward_pre_hook(approach_hidden))

        generator = net.grasp_generator

        def grouped(module, inputs, output):
            # (B, 3, M, K) of one radius and depth; inputs: scene cloud, crop centres, frames
            self.grouped.append(numpy(output).transpose(0, 2, 3, 1).copy())
            self.data["crop_centre"] = numpy(inputs[1])
            self.data["crop_rotation"] = numpy(inputs[2])

        def scale(index):
            def hook(module, inputs, output):
                self.data[f"crop_scale{index}"] = numpy(output).transpose(0, 2, 3, 1).copy()

            return hook

        for index, crop in enumerate(
            (generator.crop1, generator.crop2, generator.crop3, generator.crop4)
        ):
            for grouper in crop.groupers:
                self.handles.append(grouper.register_forward_hook(grouped))
            self.handles.append(crop.register_forward_hook(scale(index)))  # (B, M, D, C)

        def fused(module, inputs, output):
            depths = len(generator.crop1.groupers)
            batch, channels, rows = output.shape
            value = output.view(batch, channels, rows // depths, depths)
            self.data["fused"] = numpy(value).transpose(0, 2, 3, 1).copy()  # (B, M, D, C)

        self.handles.append(generator.fuse_multi_scale.register_forward_hook(fused))

        def gate(module, inputs, output):
            self.data["gate"] = numpy(output).transpose(0, 2, 1).copy()  # (B, M, C)

        self.handles.append(generator.gate_fusion.register_forward_hook(gate))

        def head_input(module, inputs):
            self.data["crop"] = numpy(inputs[0]).transpose(0, 2, 3, 1).copy()  # (B, M, D, C)

        self.handles.append(generator.operation.register_forward_pre_hook(head_input))

        def cylinder_factory(original):
            def wrapper(*args):
                index = original(*args)
                self.cylinder_index.append(numpy(index).astype(np.int32))
                return index

            return wrapper

        self._wrap(self.query_module, "cylinder_query", cylinder_factory)
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def collect(self, batch, end_points):
        """Merge hook data with the entries the model wrote into ``end_points``."""
        data = dict(self.data)
        data["point_clouds"] = numpy(batch["point_clouds"])
        data["seed_index"] = numpy(end_points["fp2_inds"]).astype(np.int64)
        data["seed_xyz"] = numpy(end_points["fp2_xyz"])
        data["seed_features"] = numpy(end_points["fp2_features"]).transpose(0, 2, 1).copy()
        data["objectness"] = numpy(end_points["objectness_score"]).transpose(0, 2, 1).copy()
        data["view_score"] = numpy(end_points["view_score"])
        data["view_index"] = numpy(end_points["grasp_top_view_inds"]).astype(np.int64)
        data["view_xyz"] = numpy(end_points["grasp_top_view_xyz"])
        data["view_rot"] = numpy(end_points["grasp_top_view_rot"])
        if len(self.neighbours) != 2:
            raise RuntimeError("expected the neighbour lists of two transformer blocks")
        data["transformer1_neighbours"], data["transformer2_neighbours"] = self.neighbours
        radii = 4
        depths = len(self.cylinder_index) // radii
        index = np.stack(self.cylinder_index, axis=2)  # (B, M, radius · depth, K), radius-major
        data["cylinder_index"] = index.reshape((*index.shape[:2], radii, depths, index.shape[-1]))
        grouped = np.stack(self.grouped, axis=2)
        data["grouped"] = grouped.reshape((*grouped.shape[:2], radii, depths, *grouped.shape[3:]))
        for key, name in (
            ("grasp_score_pred", "score_pred"),
            ("grasp_angle_cls_pred", "angle_pred"),
            ("grasp_width_pred", "width_pred"),
            ("grasp_tolerance_pred", "tolerance_pred"),
        ):
            data[name] = numpy(end_points[key]).transpose(0, 2, 1, 3).copy()  # (B, M, A, D)
        return data


def install_shims():
    """Make the pinned upstream sources importable under a recent PyTorch."""
    shims = []
    try:
        from torch._six import container_abcs  # noqa: F401
    except ImportError:  # PyTorch 1.9 dropped the name, PyTorch 2 the module
        try:
            import torch._six as module
        except ImportError:
            module = types.ModuleType("torch._six")
            sys.modules["torch._six"] = module
        module.container_abcs = collections.abc
        shims.append("torch._six.container_abcs -> collections.abc")
    return shims


def seed_for(scene, frame):
    return scene * 1000 + frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, help="pristine Scale-Balanced-Grasp tree")
    parser.add_argument("--checkpoint", required=True, help="upstream checkpoint (.tar)")
    parser.add_argument("--dataset-root", required=True, help="GraspNet-1Billion root")
    parser.add_argument("--out", required=True, help="output directory (under the cache root)")
    parser.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance-dir",
        help="directory of upstream-format tolerance labels (NNN_tolerance.npy); needed for "
        "the training batch",
    )
    parser.add_argument("--health", action="store_true", help="also dump the step-0 frames")
    parser.add_argument("--health-only", action="store_true", help="dump the step-0 frames only")
    parser.add_argument("--skip-training", action="store_true", help="no training batch")
    args = parser.parse_args(argv)

    reference = str(Path(args.reference).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())
    root = str(Path(args.dataset_root).resolve())
    tolerance_dir = None if args.tolerance_dir is None else str(Path(args.tolerance_dir).resolve())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The reference is used read-only: no bytecode next to its sources, no relative
    # output inside its tree. Its loss module reads the scale statistic relative to the
    # working directory, so the temporary one links to that directory of the reference.
    sys.dont_write_bytecode = True
    os.chdir(tempfile.mkdtemp(prefix="sbg_oracle_"))
    os.symlink(os.path.join(reference, "statistic"), "statistic")
    shims = install_shims()
    for sub in ("dataset", "models", "utils", "pointnet2", "knn", ""):
        sys.path.insert(0, os.path.join(reference, sub) if sub else reference)

    import pct_zh as transformer_module
    import pointnet2_utils as query_module
    import scipy.io as scio
    import torch
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import GraspNet_MSCQ, pred_decode
    from graspnet_dataset import GraspNetDataset, collate_fn
    from graspnetAPI import GraspGroup

    device = torch.device("cuda:0")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    defaults = (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
    )
    shutil.copyfile(
        os.path.join(reference, "statistic", "scale_distribution.npy"), str(out / STATISTIC_COPY)
    )
    manifest.update(
        {
            "reference": reference,
            "reference_commit": reference_commit(reference) or REFERENCE_COMMIT + " (pinned)",
            "reference_sources_sha256": tree_digest(reference),
            "network": f"GraspNet_MSCQ (cylinder_radius {CYLINDER_RADIUS:.2f}), no object-balanced sampling",
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(checkpoint),
            "scale_statistic_file": STATISTIC_COPY,
            "scale_statistic_sha256": sha256(str(out / STATISTIC_COPY)),
            "dataset_root": root,
            "camera": args.camera,
            "num_points": args.num_points,
            "seed": args.seed,
            "precision": "float32 (TF32 disabled)",
            "shims": shims,
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "pointnet2_ext": getattr(query_module._ext, "__file__", "?"),
            },
            "tf32_defaults": {"matmul": defaults[0], "cudnn": defaults[1]},
        }
    )
    set_tf32(torch, False, False)

    def load_network(training):
        net = GraspNet_MSCQ(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=CYLINDER_RADIUS,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=training,
        ).to(device)
        state = torch.load(checkpoint, map_location=device)
        net.load_state_dict(state["model_state_dict"])
        manifest["checkpoint_epoch"] = int(state.get("epoch", -1))
        manifest["parameters"] = int(sum(p.numel() for p in net.parameters()))
        manifest["tensors"] = len(net.state_dict())
        return net.train() if training else net.eval()

    dataset = GraspNetDataset(
        root,
        valid_obj_idxs=None,
        grasp_labels=None,
        split="test",
        camera=args.camera,
        num_points=args.num_points,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )

    def test_sample(scene, frame):
        index = (scene - 100) * 256 + frame
        assert dataset.scenename[index] == f"scene_{scene:04d}" and dataset.frameid[index] == frame
        np.random.seed(args.seed + seed_for(scene, frame))
        return index, dataset[index]

    net = load_network(False)

    def run_inference(sample):
        recorder = Recorder(net, query_module, transformer_module).install()
        try:
            batch = to_device(collate_fn([sample]), device)
            with torch.no_grad():
                end_points = net(batch)
                decoded = pred_decode(end_points)
            data = recorder.collect(batch, end_points)
        finally:
            recorder.remove()
        data["decoded"] = numpy(decoded[0])
        data["objectness_mask"] = numpy(torch.argmax(end_points["objectness_score"][0], 0) == 1)
        return data

    def collision(index, decoded):
        full_cloud, _ = dataset.get_data(index, return_raw_cloud=True)
        detector = ModelFreeCollisionDetector(full_cloud, voxel_size=0.01)
        mask = detector.detect(GraspGroup(decoded), approach_dist=0.05, collision_thresh=0.01)
        return full_cloud, detector, np.asarray(mask, dtype=bool)

    # ----------------------------------------------------------------- step 0: health
    if args.health or args.health_only:
        clouds, listed = [], []
        for scene in HEALTH_SCENES:
            for frame in HEALTH_FRAMES:
                index, sample = test_sample(scene, frame)
                batch = to_device(collate_fn([sample]), device)
                with torch.no_grad():
                    decoded = numpy(pred_decode(net(batch))[0])
                _, _, mask = collision(index, decoded)
                for variant, grasps in (("raw", decoded), ("cd", decoded[~mask])):
                    target = out / "health" / variant / f"scene_{scene:04d}" / args.camera
                    target.mkdir(parents=True, exist_ok=True)
                    np.save(str(target / f"{frame:04d}.npy"), grasps.astype(np.float32))
                clouds.append(sample["point_clouds"])
                listed.append([scene, frame, len(decoded), int(mask.sum())])
                print(
                    f"health scene {scene} frame {frame}: {len(decoded)} grasps, "
                    f"{mask.sum()} in collision"
                )
        np.savez(
            str(out / "health_inputs.npz"), point_clouds=np.stack(clouds), frames=np.array(listed)
        )
        manifest["health"] = {
            "inputs": "health_inputs.npz",
            "variants": {"raw": "health/raw", "cd": "health/cd"},
            "frames": listed,
            "collision": {"voxel_size": 0.01, "approach_dist": 0.05, "collision_thresh": 0.01},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if args.health_only:
            print("wrote", manifest_path)
            return 0

    # ---------------------------------------------------------------------- inference
    manifest["inference"] = []
    for scene, frame in INFERENCE_FRAMES:
        index, sample = test_sample(scene, frame)
        data = run_inference(sample)
        again = run_inference(sample)
        set_tf32(torch, *defaults)
        default = run_inference(sample)
        set_tf32(torch, False, False)

        full_cloud, detector, mask = collision(index, data["decoded"])
        # float64, exactly what the reference test script hands to its collision detector
        data["full_cloud"] = np.asarray(full_cloud, dtype=np.float64)
        data["collision_scene"] = np.asarray(detector.scene_points, dtype=np.float64)
        data["collision_mask"] = mask

        name = f"infer_{scene:04d}_{frame:04d}.npz"
        np.savez(str(out / name), **data)
        compared = (*STAGES, "decoded", "transformer1_neighbours", "transformer2_neighbours")
        entry = {
            "file": name,
            "scene": scene,
            "frame": frame,
            "points": int(data["point_clouds"].shape[1]),
            "grasps": int(data["decoded"].shape[0]),
            "empty_crops": int((data["cylinder_index"] == 0).all(axis=-1).sum()),
            "full_cloud_points": int(full_cloud.shape[0]),
            "collisions": int(mask.sum()),
            "repeat_run_max_abs_diff": difference(data, again, compared),
            "pytorch_default_tf32_max_abs_diff": difference(data, default, compared),
        }
        manifest["inference"].append(entry)
        print(json.dumps(entry))
    del net
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------------- training
    if not args.skip_training:
        if tolerance_dir is None:
            raise SystemExit("--tolerance-dir is needed to record the training batch")
        import loss as loss_module  # reads statistic/scale_distribution.npy from the cwd

        train = GraspNetDataset(
            root,
            valid_obj_idxs=None,
            grasp_labels=None,
            split="train",
            camera=args.camera,
            num_points=args.num_points,
            remove_outlier=True,
            remove_invisible=True,
            augment=True,
            load_label=False,  # the constructor would otherwise read the labels of all scenes
        )
        indices = [scene * 256 + frame for scene, frame in TRAINING_FRAMES]
        # Give the dataset only the labels its chosen frames need, in the layout its own
        # loader builds (load_grasp_labels), then let its own label path build the samples.
        train.grasp_labels, train.collision_labels, valid = {}, {}, []
        for index in indices:
            scene = train.scenename[index]
            if scene not in train.collision_labels:
                archive = np.load(
                    os.path.join(root, "collision_label", scene, "collision_labels.npz")
                )
                train.collision_labels[scene] = {
                    i: archive[f"arr_{i}"] for i in range(len(archive.files))
                }
            for obj in scio.loadmat(train.metapath[index])["cls_indexes"].flatten().astype(int):
                if obj - 1 == 18 or obj in train.grasp_labels:  # upstream skips object 18
                    continue
                label = np.load(os.path.join(root, "grasp_label", f"{obj - 1:03d}_labels.npz"))
                tolerance = np.load(os.path.join(tolerance_dir, f"{obj - 1:03d}_tolerance.npy"))
                train.grasp_labels[obj] = (
                    label["points"].astype(np.float32),
                    label["offsets"].astype(np.float32),
                    label["scores"].astype(np.float32),
                    tolerance,
                )
                valid.append(int(obj))
        train.valid_obj_idxs = valid
        train.load_label = True
        np.random.seed(args.seed)
        samples = [train[index] for index in indices]
        train.grasp_labels, train.collision_labels = {}, {}

        net = load_network(True)
        torch.manual_seed(args.seed)
        recorder = Recorder(net, query_module, transformer_module).install()
        weights = {}

        def weight_factory(original):
            def wrapper(end_points):
                weight = original(end_points)
                weights["seed_weight"] = numpy(weight)
                return weight

            return wrapper

        recorder._wrap(loss_module, "generate_reweight_mask", weight_factory)
        try:
            batch = to_device(collate_fn(samples), device)
            end_points = net(batch)
            data = recorder.collect(batch, end_points)
            for key, name in (
                ("batch_grasp_point", "label_point"),
                ("batch_grasp_view_rot", "label_crop_rotation"),
                ("batch_grasp_label", "label_score_target"),
                ("batch_grasp_tolerance", "label_tolerance"),
                ("batch_grasp_view_label", "label_view_score"),
            ):
                data[name] = numpy(end_points[key])
            data["label_width"] = numpy(end_points["batch_grasp_offset"][..., 2])
            total, end_points = loss_module.get_loss(end_points)
            total.backward()
        finally:
            recorder.remove()
        data.update(weights)
        data["graspable_label"] = numpy(end_points["graspable_mask"]).astype(np.int64)
        data["scale_weights"] = numpy(loss_module.scale_distribution_prior)
        data["scale_intervals"] = np.asarray(loss_module.intervals, dtype=np.float64)
        data["scale_counts"] = np.asarray(loss_module.scale_distribution_prior_num)
        scalars = {
            key: float(value)
            for key, value in end_points.items()
            if key.startswith(("loss/", "stage1_", "stage2_"))
        }
        for key, value in scalars.items():
            data["scalar_" + key.replace("/", "__")] = np.float64(value)
        parameters = dict(net.named_parameters())
        for key in GRADIENTS:
            data["grad_" + key] = numpy(parameters[key].grad)
        squared = sum(
            float(p.grad.double().pow(2).sum()) for p in net.parameters() if p.grad is not None
        )
        data["grad_norm"] = np.float64(squared**0.5)
        data["objectness_label"] = numpy(batch["objectness_label"]).astype(np.int64)
        np.savez(str(out / "train_batch.npz"), **data)

        labels = {"objects_per_frame": np.array([len(p) for p in batch["object_poses_list"]])}
        for b in range(len(samples)):
            for o in range(len(batch["object_poses_list"][b])):
                prefix = f"f{b}_o{o}_"
                labels[prefix + "pose"] = numpy(batch["object_poses_list"][b][o])
                labels[prefix + "points"] = numpy(batch["grasp_points_list"][b][o])
                labels[prefix + "scores"] = numpy(batch["grasp_labels_list"][b][o])
                labels[prefix + "widths"] = numpy(batch["grasp_offsets_list"][b][o][..., 2])
                labels[prefix + "tolerance"] = numpy(batch["grasp_tolerance_list"][b][o])
        np.savez_compressed(str(out / "train_labels.npz"), **labels)
        manifest["training"] = {
            "file": "train_batch.npz",
            "labels": "train_labels.npz",
            "tolerance_dir": tolerance_dir,
            "frames": [list(pair) for pair in TRAINING_FRAMES],
            "augment": True,
            "objects_per_frame": labels["objects_per_frame"].tolist(),
            "mirrored_frames": [
                bool(np.linalg.det(numpy(poses[0])[:3, :3]) < 0)
                for poses in batch["object_poses_list"]
            ],
            "empty_crops": int((data["cylinder_index"] == 0).all(axis=-1).sum()),
            "scalars": scalars,
            "grad_norm": float(data["grad_norm"]),
        }
        print(json.dumps(manifest["training"]))

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
