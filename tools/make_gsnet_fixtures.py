"""Record reference activations of GSNet from the upstream implementation.

This script is the *oracle recorder* of the GSNet parity tests. It does not import
GraspPanda and is run with the interpreter of the legacy environment (torch 1.9,
MinkowskiEngine 0.5.4, the compiled pointnet2 / knn extensions, graspnetAPI):

    $GRASPPANDA_ORACLE_PYTHON tools/make_gsnet_fixtures.py \
        --reference /path/to/graspness_unofficial \
        --checkpoint /path/to/checkpoint-rs.tar \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" \
        --out "$GRASPPANDA_CACHE/fixtures/gsnet"

It loads the pristine upstream code read-only (no bytecode is written, the working
directory is a fresh temporary directory), runs it unmodified, and records what flows
between its stages through forward hooks and by wrapping the few free functions the
model calls. Nothing of the upstream forward pass is re-implemented here.

Recorded (float32 unless stated), see ``manifest.json`` in the output directory:

``infer_<scene>_<frame>.npz``
    Inputs of one test frame exactly as the upstream loader produced them, the
    voxelisation, backbone activations at every level, per-point features,
    objectness / graspness, the sampling candidates and the seed indices, view scores
    and chosen views, cylinder neighbours, grouped and pooled crop features, head
    outputs, the decoded ``(1024, 17)`` array, and the model-free collision mask.
``sweep.npz``
    Inputs, per-point scores, candidate masks, seed indices and decoded grasps of
    ``--sweep`` further test frames spread over all test scenes. Seed selection sits
    behind a threshold and an arg-max, so whether two implementations pick the same
    seeds is a matter of frequency; the sweep measures it on more frames than the
    detailed records cover.
``train_batch.npz`` and ``train_labels.npz``
    One training batch with labels, the view indices drawn by ``torch.multinomial``
    (so the stochastic step can be replayed), the label-matching outputs, every loss
    term, and a few parameter gradients.
``--layer-probe FILE``
    A tiny record of single MinkowskiEngine layers on random inputs, small enough to
    commit; it pins the kernel layout for the CPU unit tests.
``--fps-probe FILE``
    The reference furthest-point-sampling kernel on small lattice clouds, where equal
    distances abound and every distance is exact in float32; also small enough to
    commit. It pins the order in which that kernel resolves ties.

``--probes-only`` writes the two probes and stops (no dataset is read).

Precision. On Ampere and newer GPUs torch 1.9 enables TF32 for matrix products by
default, which MinkowskiEngine's 1×1 convolutions and the label transforms go through;
activations then carry errors around 1e-3. The fixtures are recorded with TF32 switched
off, i.e. in true float32. The manifest additionally reports how far the upstream
defaults (TF32 on) and a second identical run move the results, which is the scale any
comparison has to be read against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

INFERENCE_FRAMES = (
    (100, 0),
    (118, 137),
    (145, 64),
    (175, 200),
)  # (scene, frame): seen ×2, similar, novel
TRAINING_FRAMES = ((5, 17), (5, 201), (63, 100), (63, 31))
BACKBONE_TAPS = (
    ["bn0", "final"]
    + [f"bn{i}" for i in range(1, 5)]
    + [f"bntr{i}" for i in range(4, 8)]
    + [f"block{i}" for i in range(1, 9)]
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_commit(reference):
    """HEAD of the reference checkout, read from the files (git is not invoked)."""
    head = (Path(reference) / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = Path(reference) / ".git" / head.split(" ", 1)[1]
        return ref.read_text().strip() if ref.is_file() else head
    return head


def numpy(value):
    return value.detach().cpu().numpy()


class Recorder:
    """Collects tensors from hooks and wrapped functions during one forward pass."""

    def __init__(self, torch, net, graspnet_module, query_module):
        self.torch = torch
        self.net = net
        self.graspnet_module = graspnet_module
        self.query_module = query_module
        self.data = {}
        self.calls = {"fps_input": [], "fps_index": [], "cylinder_index": [], "multinomial": []}
        self.handles = []
        self.originals = []

    # -- installation ------------------------------------------------------------

    def _wrap(self, owner, name, wrapper_factory):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, wrapper_factory(original))

    def install(self):
        net, calls = self.net, self.calls

        def tap_sparse(key):
            def hook(module, inputs, output):
                # Clone at once: the shared in-place ReLU of the reference overwrites .F later.
                self.data[f"tap_{key}_coords"] = numpy(output.C).astype(np.int32)
                self.data[f"tap_{key}_feats"] = numpy(output.F).copy()

            return hook

        for name in BACKBONE_TAPS:
            self.handles.append(getattr(net.backbone, name).register_forward_hook(tap_sparse(name)))

        def point_features(module, inputs):
            self.data["point_feats"] = numpy(inputs[0]).transpose(0, 2, 1).copy()  # (B, N, C)

        self.handles.append(net.graspable.register_forward_pre_hook(point_features))

        def view_stage(module, inputs, output):
            self.data["view_residual"] = numpy(output[1]).transpose(0, 2, 1).copy()  # (B, M, C)

        self.handles.append(net.rotation.register_forward_hook(view_stage))

        def grouped(module, inputs, output):
            self.data["grouped"] = numpy(output).transpose(0, 2, 3, 1).copy()  # (B, M, K, 3 + C)
            self.data["crop_centre"] = numpy(inputs[1])
            self.data["crop_rotation"] = numpy(inputs[2])

        self.handles.append(net.crop.grouper.register_forward_hook(grouped))

        def pooled(module, inputs, output):
            self.data["crop"] = numpy(output).transpose(0, 2, 1).copy()  # (B, M, C)

        self.handles.append(net.crop.register_forward_hook(pooled))

        def fps_factory(original):
            def wrapper(xyz, npoint):
                index = original(xyz, npoint)
                calls["fps_input"].append(numpy(xyz)[0].copy())
                calls["fps_index"].append(numpy(index)[0].astype(np.int64))
                return index

            return wrapper

        self._wrap(self.graspnet_module, "furthest_point_sample", fps_factory)

        def cylinder_factory(original):
            def wrapper(*args):
                index = original(*args)
                calls["cylinder_index"].append(numpy(index).astype(np.int64))
                return index

            return wrapper

        self._wrap(self.query_module, "cylinder_query", cylinder_factory)

        def multinomial_factory(original):
            def wrapper(*args, **kwargs):
                drawn = original(*args, **kwargs)
                calls["multinomial"].append(numpy(drawn)[:, 0].astype(np.int64))
                return drawn

            return wrapper

        self._wrap(self.torch, "multinomial", multinomial_factory)

        if hasattr(self.graspnet_module, "match_grasp_view_and_label"):

            def match_factory(original):
                def wrapper(end_points):
                    rotation, end_points = original(end_points)
                    self.data["label_crop_rotation"] = numpy(rotation)
                    return rotation, end_points

                return wrapper

            self._wrap(self.graspnet_module, "match_grasp_view_and_label", match_factory)
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    # -- results -----------------------------------------------------------------

    def collect(self, batch, end_points, threshold):
        """Merge hook data with the entries the model wrote into ``end_points``."""
        torch = self.torch
        data = dict(self.data)
        clouds = batch["point_clouds"]
        data["point_clouds"] = numpy(clouds)
        data["coors"] = numpy(batch["coors"]).astype(np.int32)
        data["quantize2original"] = numpy(batch["quantize2original"]).astype(np.int64)
        objectness = end_points["objectness_score"]  # (B, 2, N)
        graspness = end_points["graspness_score"]  # (B, N)
        data["objectness"] = numpy(objectness).transpose(0, 2, 1).copy()
        data["graspness"] = numpy(graspness)
        # Seed indices into the input cloud: the sampler ran on the candidate subset. The
        # subset is re-derived from the recorded scores only to translate indices, and it is
        # checked against the points the sampler really received.
        candidate = (torch.argmax(objectness, 1) == 1) & (graspness > threshold)
        seed_index = []
        for b in range(clouds.shape[0]):
            members = torch.nonzero(candidate[b]).flatten().cpu().numpy()
            received = self.calls["fps_input"][b]
            if received.shape[0] != len(members) or not np.array_equal(
                received, data["point_clouds"][b][members]
            ):
                raise RuntimeError("candidate subset does not match what the sampler received")
            seed_index.append(members[self.calls["fps_index"][b]])
            data[f"candidates_{b}"] = members.astype(np.int64)
            data[f"fps_index_{b}"] = self.calls["fps_index"][b]
        data["seed_index"] = np.stack(seed_index)
        data["seed_xyz"] = numpy(end_points["xyz_graspable"])
        data["view_score"] = numpy(end_points["view_score"])
        data["view_index"] = numpy(end_points["grasp_top_view_inds"]).astype(np.int64)
        for key, name in (("grasp_top_view_xyz", "view_xyz"), ("grasp_top_view_rot", "view_rot")):
            if key in end_points:
                data[name] = numpy(end_points[key])
        data["cylinder_index"] = np.concatenate(self.calls["cylinder_index"])
        data["score_pred"] = numpy(end_points["grasp_score_pred"])
        data["width_pred"] = numpy(end_points["grasp_width_pred"])
        if self.calls["multinomial"]:
            data["sampled_view_index"] = np.stack(self.calls["multinomial"])
        return data


def to_device(batch, device):
    """Move a collated upstream batch (tensors and nested lists of tensors) to a device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [[item.to(device) for item in frame] for frame in value]
        else:
            moved[key] = value.to(device)
    return moved


def set_tf32(torch, enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def load_network(torch, graspnet_module, checkpoint, training, device):
    net = graspnet_module.GraspNet(seed_feat_dim=512, is_training=training).to(device)
    state = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state["model_state_dict"])
    return net.train() if training else net.eval()


def difference(first, second, keys):
    """Largest absolute difference per recorded stage between two runs."""
    report = {}
    for key in keys:
        if first[key].shape != second[key].shape:
            report[key] = f"shape {first[key].shape} vs {second[key].shape}"
        elif first[key].dtype.kind in "iub":
            report[key + "_mismatches"] = int((first[key] != second[key]).sum())
        else:
            report[key] = float(np.abs(first[key].astype(np.float64) - second[key]).max())
    return report


def record_layer_probe(torch, ME, path, device):
    """Single MinkowskiEngine layers on a tiny random input (committed with the tests)."""
    generator = np.random.default_rng(0)
    torch.manual_seed(0)
    coords = []
    for sample in range(2):
        cells = np.unique(generator.integers(-9, 14, size=(48, 3)), axis=0)
        coords.append(np.concatenate([np.full((len(cells), 1), sample), cells], axis=1))
    coords = torch.tensor(np.concatenate(coords), dtype=torch.int32)
    feats = torch.randn(len(coords), 3)
    layers = [
        ("stem", ME.MinkowskiConvolution(3, 4, kernel_size=5, dimension=3)),
        ("down1", ME.MinkowskiConvolution(4, 4, kernel_size=2, stride=2, dimension=3)),
        ("mid1", ME.MinkowskiConvolution(4, 6, kernel_size=3, dimension=3)),
        ("down2", ME.MinkowskiConvolution(6, 6, kernel_size=2, stride=2, dimension=3)),
        ("mid2", ME.MinkowskiConvolution(6, 5, kernel_size=3, dimension=3)),
        ("up2", ME.MinkowskiConvolutionTranspose(5, 6, kernel_size=2, stride=2, dimension=3)),
        ("up1", ME.MinkowskiConvolutionTranspose(6, 4, kernel_size=2, stride=2, dimension=3)),
        ("point", ME.MinkowskiConvolution(4, 2, kernel_size=1, bias=True, dimension=3)),
    ]
    record = {"coords": coords.numpy(), "feats": feats.numpy()}
    tensor = ME.SparseTensor(feats.to(device), coordinates=coords.to(device))
    with torch.no_grad():
        for name, layer in layers:
            layer = layer.to(device)
            torch.nn.init.normal_(layer.kernel)
            record["kernel_" + name] = numpy(layer.kernel)
            if layer.bias is not None:
                torch.nn.init.normal_(layer.bias)
                record["bias_" + name] = numpy(layer.bias)
            tensor = layer(tensor)
            record["out_coords_" + name] = numpy(tensor.C).astype(np.int32)
            record["out_feats_" + name] = numpy(tensor.F)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **record)
    return os.path.getsize(path)


def record_fps_probe(torch, sampler, path, device):
    """The reference sampling kernel on lattice clouds (committed with the tests).

    Coordinates are multiples of 1/16 next to (1, 1, 1): squared distances are exact in
    float32, so the picks depend on the tie order of the kernel and on nothing else.
    The cases cover the kernel's thread counts (64, 256, 512), clouds with repeated
    points, a cloud with fewer distinct points than samples, and a point the kernel
    skips because it lies next to the origin.
    """
    generator = np.random.default_rng(0)
    record = {}
    cases = ((100, 64, 6), (300, 128, 8), (511, 200, 10), (512, 200, 10), (700, 256, 10))
    cases += ((3000, 512, 12), (6000, 1024, 12), (200, 150, 4))  # (points, samples, lattice edge)
    for case, (count, samples, edge) in enumerate(cases):
        cells = generator.integers(0, edge, size=(count, 3))
        points = (cells / 16.0 + 1.0).astype(np.float32)
        points[count // 3] = 0.01  # squared norm 3e-4: never picked by the kernel
        index = sampler(torch.from_numpy(points)[None].to(device), samples)
        record[f"points_{case}"] = points
        record[f"index_{case}"] = numpy(index)[0].astype(np.int32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **record)
    return os.path.getsize(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, help="pristine graspness_unofficial checkout")
    parser.add_argument("--checkpoint", required=True, help="upstream checkpoint (.tar)")
    parser.add_argument("--dataset-root", required=True, help="GraspNet-1Billion root")
    parser.add_argument("--out", required=True, help="output directory (under the cache root)")
    parser.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--num-points", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", type=int, default=64, help="frames of the seed sweep (0: none)")
    parser.add_argument("--layer-probe", help="also write the tiny single-layer record here")
    parser.add_argument("--fps-probe", help="also write the tiny sampling-kernel record here")
    parser.add_argument("--probes-only", action="store_true", help="write the probes and stop")
    parser.add_argument("--skip-training", action="store_true", help="record inference frames only")
    args = parser.parse_args(argv)

    reference = str(Path(args.reference).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())
    root = str(Path(args.dataset_root).resolve())
    out = Path(args.out).resolve()
    probe = None if args.layer_probe is None else str(Path(args.layer_probe).resolve())
    fps_probe = None if args.fps_probe is None else str(Path(args.fps_probe).resolve())
    out.mkdir(parents=True, exist_ok=True)

    # The reference is used read-only: no bytecode next to its sources, no relative
    # output inside its tree.
    sys.dont_write_bytecode = True
    os.chdir(tempfile.mkdtemp(prefix="gsnet_oracle_"))
    for sub in ("dataset", "models", "utils", "pointnet2", ""):
        sys.path.insert(0, os.path.join(reference, sub) if sub else reference)

    import MinkowskiEngine as ME
    import models.graspnet as graspnet_module
    import pointnet2.pointnet2_utils as query_module
    import scipy.io as scio
    import torch

    # Imported as a top-level module from <reference>/utils: MinkowskiEngine appends its own
    # package directory to sys.path, where a regular package called ``utils`` shadows the
    # reference's ``utils`` namespace.
    from collision_detector import ModelFreeCollisionDetector
    from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
    from graspnetAPI import GraspGroup
    from loss_utils import GRASPNESS_THRESHOLD
    from models.loss import get_loss

    device = torch.device("cuda:0")
    manifest = {
        "reference": reference,
        "reference_commit": reference_commit(reference),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_root": root,
        "camera": args.camera,
        "num_points": args.num_points,
        "voxel_size": 0.005,
        "seed": args.seed,
        "precision": "float32 (TF32 disabled)",
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "MinkowskiEngine": ME.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "tf32_defaults": {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "inference": [],
    }
    set_tf32(torch, False)

    if probe is not None:
        manifest["layer_probe_bytes"] = record_layer_probe(torch, ME, probe, device)
        print(f"layer probe: {probe} ({manifest['layer_probe_bytes']} bytes)")
    if fps_probe is not None:
        sampler = query_module.furthest_point_sample
        manifest["fps_probe_bytes"] = record_fps_probe(torch, sampler, fps_probe, device)
        print(f"sampling probe: {fps_probe} ({manifest['fps_probe_bytes']} bytes)")
    if args.probes_only:
        return 0

    # ------------------------------------------------------------------ inference
    dataset = GraspNetDataset(
        root,
        split="test",
        camera=args.camera,
        num_points=args.num_points,
        voxel_size=0.005,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    net = load_network(torch, graspnet_module, checkpoint, False, device)
    stages = [
        "point_feats",
        "objectness",
        "graspness",
        "seed_index",
        "view_score",
        "view_index",
        "crop",
        "score_pred",
        "width_pred",
        "decoded",
    ]

    def run_inference(sample):
        recorder = Recorder(torch, net, graspnet_module, query_module).install()
        try:
            batch = to_device(minkowski_collate_fn([sample]), device)
            with torch.no_grad():
                end_points = net(batch)
                decoded = graspnet_module.pred_decode(end_points)
            data = recorder.collect(batch, end_points, GRASPNESS_THRESHOLD)
        finally:
            recorder.remove()
        data["decoded"] = np.stack([numpy(d) for d in decoded])
        return data

    for scene, frame in INFERENCE_FRAMES:
        index = (scene - 100) * 256 + frame
        assert dataset.scenename[index] == f"scene_{scene:04d}" and dataset.frameid[index] == frame
        np.random.seed(args.seed + index)
        sample = dataset[index]
        data = run_inference(sample)
        again = run_inference(sample)
        set_tf32(torch, True)
        default = run_inference(sample)
        set_tf32(torch, False)

        full_cloud = dataset.get_data(index, return_raw_cloud=True)
        grasps = GraspGroup(data["decoded"][0])
        detector = ModelFreeCollisionDetector(full_cloud, voxel_size=0.01)
        collision = detector.detect(grasps, approach_dist=0.05, collision_thresh=0.01)
        # float64, exactly what the reference test script hands to its collision detector
        data["full_cloud"] = np.asarray(full_cloud, dtype=np.float64)
        data["collision_scene"] = np.asarray(detector.scene_points, dtype=np.float64)
        data["collision_mask"] = np.asarray(collision, dtype=bool)

        name = f"infer_{scene:04d}_{frame:04d}.npz"
        np.savez(out / name, **data)
        entry = {
            "file": name,
            "scene": scene,
            "frame": frame,
            "points": int(data["point_clouds"].shape[1]),
            "voxels": int(data["coors"].shape[0]),
            "candidates": int(data["candidates_0"].shape[0]),
            "full_cloud_points": int(full_cloud.shape[0]),
            "full_cloud_dtype": str(full_cloud.dtype),
            "collisions": int(collision.sum()),
            "repeat_run_max_abs_diff": difference(data, again, stages),
            "tf32_default_max_abs_diff": difference(data, default, stages),
        }
        manifest["inference"].append(entry)
        print(json.dumps(entry))

    # ---------------------------------------------------------------------- sweep
    if args.sweep > 0:
        chosen = np.unique(np.linspace(0, len(dataset) - 1, args.sweep).round().astype(int))
        names = ("scene", "frame", "point_clouds", "objectness", "graspness", "candidate_mask")
        sweep = {name: [] for name in (*names, "seed_index", "decoded")}
        tf32_other_seeds = 0
        for index in chosen.tolist():
            np.random.seed(args.seed + index)
            sample = dataset[index]
            data = run_inference(sample)
            set_tf32(torch, True)
            default = run_inference(sample)
            set_tf32(torch, False)
            tf32_other_seeds += int(not np.array_equal(data["seed_index"], default["seed_index"]))
            sweep["scene"].append(int(dataset.scenename[index].split("_")[1]))
            sweep["frame"].append(int(dataset.frameid[index]))
            mask = np.zeros(data["point_clouds"].shape[1], dtype=bool)
            mask[data["candidates_0"]] = True
            sweep["candidate_mask"].append(mask)
            for name in ("point_clouds", "objectness", "graspness", "seed_index", "decoded"):
                sweep[name].append(data[name][0])
        np.savez(out / "sweep.npz", **{name: np.stack(values) for name, values in sweep.items()})
        manifest["sweep"] = {
            "file": "sweep.npz",
            "frames": len(chosen),
            "min_candidates": int(min(mask.sum() for mask in sweep["candidate_mask"])),
            # how often the reference picks other seeds than itself under its default TF32
            "tf32_default_frames_with_other_seeds": tf32_other_seeds,
        }
        print(json.dumps(manifest["sweep"]))
    del net
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- training
    if not args.skip_training:
        train = GraspNetDataset(
            root,
            grasp_labels=None,
            split="train",
            camera=args.camera,
            num_points=args.num_points,
            voxel_size=0.005,
            remove_outlier=True,
            augment=True,
            load_label=False,  # the constructor would otherwise read the labels of all scenes
        )
        indices = [scene * 256 + frame for scene, frame in TRAINING_FRAMES]
        # Give the dataset only the labels its chosen frames need, then let its own
        # label path build the samples.
        train.grasp_labels, train.collision_labels = {}, {}
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
                if obj not in train.grasp_labels:
                    label = np.load(
                        os.path.join(root, "grasp_label_simplified", f"{obj - 1:03d}_labels.npz")
                    )
                    train.grasp_labels[obj] = (
                        label["points"].astype(np.float32),
                        label["width"].astype(np.float32),
                        label["scores"].astype(np.float32),
                    )
        train.load_label = True
        np.random.seed(args.seed)
        samples = [train[index] for index in indices]
        train.grasp_labels, train.collision_labels = {}, {}

        net = load_network(torch, graspnet_module, checkpoint, True, device)
        torch.manual_seed(args.seed)
        recorder = Recorder(torch, net, graspnet_module, query_module).install()
        try:
            batch = to_device(minkowski_collate_fn(samples), device)
            end_points = net(batch)
            data = recorder.collect(batch, end_points, GRASPNESS_THRESHOLD)
            for key, name in (
                ("batch_grasp_point", "label_point"),
                ("batch_grasp_view_graspness", "label_view_graspness"),
                ("batch_grasp_score", "label_score_target"),
                ("batch_grasp_width", "label_width"),
            ):
                data[name] = numpy(end_points[key])
            total, end_points = get_loss(end_points)
            total.backward()
        finally:
            recorder.remove()
        losses = {key: float(value) for key, value in end_points.items() if key.startswith("loss/")}
        for key, value in losses.items():
            data["loss_" + key.split("/", 1)[1]] = np.float64(value)
        gradients = {
            "backbone.conv0p1s1.kernel": net.backbone.conv0p1s1.kernel,
            "backbone.block4.0.conv2.kernel": net.backbone.block4[0].conv2.kernel,
            "backbone.convtr7p2s2.kernel": net.backbone.convtr7p2s2.kernel,
            "backbone.final.kernel": net.backbone.final.kernel,
            "graspable.conv_graspable.weight": net.graspable.conv_graspable.weight,
            "rotation.conv2.weight": net.rotation.conv2.weight,
            "crop.mlps.layer0.conv.weight": net.crop.mlps.layer0.conv.weight,
            "swad.conv_swad.weight": net.swad.conv_swad.weight,
        }
        for key, parameter in gradients.items():
            data["grad_" + key] = numpy(parameter.grad)
        squared = sum(
            float(p.grad.double().pow(2).sum()) for p in net.parameters() if p.grad is not None
        )
        data["grad_norm"] = np.float64(squared**0.5)
        data["objectness_label"] = numpy(batch["objectness_label"]).astype(np.int64)
        data["graspness_label"] = numpy(batch["graspness_label"])
        np.savez(out / "train_batch.npz", **data)

        labels = {"objects_per_frame": np.array([len(p) for p in batch["object_poses_list"]])}
        for b in range(len(samples)):
            for o in range(len(batch["object_poses_list"][b])):
                prefix = f"f{b}_o{o}_"
                labels[prefix + "pose"] = numpy(batch["object_poses_list"][b][o])
                labels[prefix + "points"] = numpy(batch["grasp_points_list"][b][o])
                labels[prefix + "scores"] = numpy(batch["grasp_scores_list"][b][o])
                labels[prefix + "widths"] = numpy(batch["grasp_widths_list"][b][o])
        start = time.time()
        np.savez_compressed(out / "train_labels.npz", **labels)
        manifest["training"] = {
            "file": "train_batch.npz",
            "labels": "train_labels.npz",
            "frames": [list(pair) for pair in TRAINING_FRAMES],
            "augment": True,
            "objects_per_frame": labels["objects_per_frame"].tolist(),
            "mirrored_frames": [
                bool(np.linalg.det(numpy(poses[0])[:3, :3]) < 0)
                for poses in batch["object_poses_list"]
            ],
            "candidates": [int(data[f"candidates_{b}"].shape[0]) for b in range(len(samples))],
            "losses": losses,
            "grad_norm": float(data["grad_norm"]),
            "labels_bytes": os.path.getsize(out / "train_labels.npz"),
            "labels_write_seconds": round(time.time() - start, 1),
        }
        print(json.dumps(manifest["training"]))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
