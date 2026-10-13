"""Record reference activations of EconomicGrasp from the upstream implementation.

This script is the *oracle recorder* of the EconomicGrasp parity tests. It does not
import GraspPanda and is run with the interpreter of the legacy environment (torch 1.9,
MinkowskiEngine 0.5.4, the compiled pointnet2 / knn extensions, graspnetAPI), from a
scratch directory, on a pristine tree of the pinned commit
(``git archive 7361bb6 | tar -x -C <scratch>``)::

    $GRASPPANDA_ORACLE_PYTHON -B tools/make_economicgrasp_fixtures.py \
        --reference /scratch/EconomicGrasp \
        --checkpoint /path/to/checkpoint-kinect.tar \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" \
        --out "$GRASPPANDA_CACHE/fixtures/economicgrasp"

It loads the upstream code read-only (no bytecode is written, the working directory is
a fresh temporary directory), runs it unmodified, and records what flows between its
stages through forward hooks and by wrapping the few free functions the model calls.
Nothing of the upstream forward pass is re-implemented here. The dataset is read in
place; the label files the upstream generator writes go to the temporary directory,
which presents the dataset to it through symbolic links.

Recorded (float32 unless stated), see ``manifest.json`` in the output directory:

``infer_<scene>_<frame>.npz``
    Inputs of one test frame exactly as the upstream loader produced them, the
    voxelisation, backbone activations at every level, per-point features,
    objectness / graspness, the sampling candidates and the seed indices, view scores
    and chosen views, cylinder neighbours, the crop at every step (grouped features,
    first shared MLP, tokens before and after the neighbour attention, pooled result),
    the task tokens of the head before and after their attention, the four head
    outputs, the decoded ``(1024, 17)`` array, and the model-free collision mask of the
    upstream test script.
``sweep.npz``
    Inputs, per-point scores, candidate masks, seed indices and decoded grasps of
    ``--sweep`` further test frames spread over all test scenes, plus the three arg-max
    decisions behind every decoded grasp (view, angle class, depth class) with the
    margin between the best and the second-best value. Seed selection sits behind a
    threshold and the decoder behind arg-maxes, so whether two implementations decide
    alike is a matter of frequency; the sweep measures it on more frames than the
    detailed records cover.
``economic_scene_<scene>_labels.npz``
    What the upstream label generator (``dataset/generate_economic.py``) writes for the
    scenes of the training batch, unchanged.
``train_batch.npz``
    One training batch with its object poses, the view indices drawn by
    ``torch.multinomial`` and the attention weights dropped by the two dropout layers
    (so the stochastic steps can be replayed), the label-matching outputs, every loss
    term, and a few parameter gradients.

Two things about the upstream code that the records have to be read with:

* It cannot run on torch 1.9 as it is: its forward floors voxel coordinates that are
  already integers, which torch 2.x (the upstream README installs 2.5.1) defines as the
  identity and torch 1.9 rejects. This process extends ``torch.floor`` accordingly; the
  upstream sources stay untouched.
* Its label matching assigns with repeated indices, so which of two template views
  that share a labelled view receives the label is undefined on a GPU. The manifest
  reports how many seeds change between two runs of the matching on the same inputs.

Precision. On Ampere and newer GPUs torch 1.9 enables TF32 for matrix products by
default; activations then carry errors around 1e-2. The fixtures are recorded with
TF32 switched off, i.e. in true float32. The manifest additionally reports how far the
upstream defaults (TF32 on) and a second identical run move the results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
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
LABEL_KEYS = (
    ("batch_grasp_point", "label_point"),
    ("batch_grasp_view_graspness", "label_view_graspness"),
    ("batch_grasp_rotations", "label_angle"),
    ("batch_grasp_depth", "label_depth"),
    ("batch_grasp_score", "label_score"),
    ("batch_grasp_width", "label_width"),
    ("batch_valid_mask", "label_valid"),
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numpy(value):
    return value.detach().cpu().numpy()


def allow_integer_floor(torch):
    """Make ``torch.floor`` accept integer tensors (the identity), as torch 2.x does."""
    try:
        torch.floor(torch.zeros(1, dtype=torch.int32))
        return False
    except RuntimeError:
        original = torch.floor

        def floor(tensor, *args, **kwargs):
            if not tensor.is_floating_point():
                return tensor.clone()
            return original(tensor, *args, **kwargs)

        torch.floor = floor
        return True


def set_tf32(torch, enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


class Recorder:
    """Collects tensors from hooks and wrapped functions during one forward pass."""

    def __init__(self, torch, net, model_module, query_module, detailed=True):
        self.torch = torch
        self.net = net
        self.model_module = model_module
        self.query_module = query_module
        self.detailed = detailed
        self.data = {}
        self.calls = {"fps_input": [], "fps_index": [], "cylinder_index": [], "multinomial": []}
        self.handles = []
        self.originals = []

    # -- installation ------------------------------------------------------------

    def _wrap(self, owner, name, wrapper_factory):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, wrapper_factory(original))

    def _hook(self, module, function, pre=False):
        register = module.register_forward_pre_hook if pre else module.register_forward_hook
        self.handles.append(register(function))

    def install(self):
        net, calls, data = self.net, self.calls, self.data

        def tap_sparse(key):
            def hook(module, inputs, output):
                # Clone at once: the shared in-place ReLU of the reference overwrites .F later.
                data[f"tap_{key}_coords"] = numpy(output.C).astype(np.int32)
                data[f"tap_{key}_feats"] = numpy(output.F).copy()

            return hook

        if self.detailed:
            for name in BACKBONE_TAPS:
                self._hook(getattr(net.backbone, name), tap_sparse(name))

        def point_features(module, inputs):
            data["point_feats"] = numpy(inputs[0]).transpose(0, 2, 1).copy()  # (B, N, C)

        self._hook(net.graspable, point_features, pre=True)

        def view_stage(module, inputs, output):
            data["view_residual"] = numpy(output[1]).transpose(0, 2, 1).copy()  # (B, M, C)

        self._hook(net.view, view_stage)

        def grouped(module, inputs, output):
            data["crop_centre"] = numpy(inputs[1])
            data["crop_rotation"] = numpy(inputs[2])
            if self.detailed:
                data["grouped"] = numpy(output).transpose(0, 2, 3, 1).copy()  # (B, M, K, 3 + C)

        self._hook(net.cy_group.grouper, grouped)

        def shared_mlp(key):
            def hook(module, inputs, output):
                data[key] = numpy(output).transpose(0, 2, 3, 1).copy()  # (B, M, K, C)

            return hook

        def interaction(key):
            def hook(module, inputs, output):
                data[key + "_in"] = numpy(inputs[0]).copy()  # (B * M, tokens, C)
                data[key + "_out"] = numpy(output).copy()

            return hook

        def dropped(key):
            def hook(module, inputs, output):
                if module.training:  # which attention weights survived, (B * M, 1, T, T)
                    data[key] = numpy(output != 0)
                    data[key + "_zero_inputs"] = np.int64((inputs[0] == 0).sum().item())

            return hook

        if self.detailed:
            self._hook(net.cy_group.mlps, shared_mlp("crop_mlp"))
            self._hook(net.cy_group.mlps2, shared_mlp("crop_post_mlp"))
            self._hook(net.cy_group.local_interaction_module, interaction("crop_tokens"))
        self._hook(net.grasp_head.global_interaction_module, interaction("head_tokens"))
        self._hook(net.cy_group.local_interaction_module.msa.dropout, dropped("crop_dropout_keep"))
        self._hook(
            net.grasp_head.global_interaction_module.msa.dropout, dropped("head_dropout_keep")
        )

        def pooled(module, inputs, output):
            data["crop"] = numpy(output).transpose(0, 2, 1).copy()  # (B, M, C)

        self._hook(net.cy_group, pooled)

        def fps_factory(original):
            def wrapper(xyz, npoint):
                index = original(xyz, npoint)
                calls["fps_input"].append(numpy(xyz)[0].copy())
                calls["fps_index"].append(numpy(index)[0].astype(np.int64))
                return index

            return wrapper

        self._wrap(self.model_module, "furthest_point_sample", fps_factory)

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

        def match_factory(original):
            def wrapper(end_points):
                rotation, end_points = original(end_points)
                data["label_crop_rotation"] = numpy(rotation)
                for key, name in LABEL_KEYS:
                    data[name] = numpy(end_points[key])
                return rotation, end_points

            return wrapper

        self._wrap(self.model_module, "process_grasp_labels", match_factory)
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
        data["coors"] = numpy(end_points["coors"]).astype(np.int32)
        data["quantize2original"] = numpy(end_points["quantize2original"]).astype(np.int64)
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
        for what in ("angle", "depth", "score", "width"):  # (B, classes, M) -> (B, M, classes)
            data[what + "_pred"] = numpy(end_points[f"grasp_{what}_pred"]).transpose(0, 2, 1).copy()
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


def dataset_view(root, scenes, camera):
    """A directory that shows the upstream scripts the dataset, restricted to ``scenes``.

    The label generator walks every scene it finds and writes next to them; here it
    finds only the scenes of the training batch and writes into the temporary tree.
    """
    view = Path(tempfile.mkdtemp(prefix="economicgrasp_dataset_"))
    (view / "scenes").mkdir()
    for scene in scenes:
        name = f"scene_{scene:04d}"
        os.symlink(os.path.join(root, "scenes", name), view / "scenes" / name)
    for folder in ("grasp_label", "collision_label", "graspness"):
        os.symlink(os.path.join(root, folder), view / folder)
    return view


def remove_view(view):
    """Remove the temporary tree: the links themselves and the generated label files."""
    for link in [*sorted((view / "scenes").iterdir()), *sorted(view.iterdir())]:
        if link.is_symlink():
            link.unlink()
    (view / "scenes").rmdir()
    generated = view / "economic_grasp_label_300views"
    for file in sorted(generated.glob("scene_*_labels.npz")):
        file.unlink()
    generated.rmdir()
    view.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, help="pristine EconomicGrasp tree")
    parser.add_argument(
        "--commit", default="7361bb6", help="commit of that tree, recorded in the manifest"
    )
    parser.add_argument("--checkpoint", required=True, help="upstream checkpoint (.tar)")
    parser.add_argument("--dataset-root", required=True, help="GraspNet-1Billion root")
    parser.add_argument("--out", required=True, help="output directory (under the cache root)")
    parser.add_argument("--camera", default="kinect", choices=["realsense", "kinect"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", type=int, default=32, help="frames of the seed sweep (0: none)")
    parser.add_argument("--skip-training", action="store_true", help="record inference only")
    args = parser.parse_args(argv)

    reference = str(Path(args.reference).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())
    root = str(Path(args.dataset_root).resolve())
    out = Path(args.out).resolve()
    script = sys.argv[0]
    out.mkdir(parents=True, exist_ok=True)

    # The reference is used read-only: no bytecode next to its sources, no relative
    # output inside its tree.
    sys.dont_write_bytecode = True
    os.chdir(tempfile.mkdtemp(prefix="economicgrasp_oracle_"))
    sys.path.insert(0, os.path.join(reference, "libs", "pointnet2"))  # bare `import pytorch_utils`
    sys.path.insert(0, reference)

    import torch

    floor_extended = allow_integer_floor(torch)
    manifest = {
        "reference": reference,
        "reference_commit": f"{args.commit} (as stated; an exported tree has no git metadata)",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_root": root,
        "camera": args.camera,
        "voxel_size": 0.005,
        "seed": args.seed,
        "precision": "float32 (TF32 disabled)",
        "torch_floor_extended_to_integers": floor_extended,
        "tf32_defaults": {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "inference": [],
    }
    set_tf32(torch, False)

    # ------------------------------------------------- labels of the training scenes
    view = None
    if not args.skip_training:
        scenes = sorted({scene for scene, _ in TRAINING_FRAMES})
        view = dataset_view(root, scenes, args.camera)
        tic = time.time()
        # The generator is a script that parses its own command line.
        sys.argv = [script, "--dataset_root", str(view), "--camera_type", args.camera]
        generator = os.path.join(reference, "dataset", "generate_economic.py")
        runpy.run_path(generator, run_name="__main__")
        manifest["labels"] = {"seconds": round(time.time() - tic, 1), "scenes": {}}
        for scene in scenes:
            name = f"scene_{scene:04d}_labels.npz"
            shutil.copyfile(view / "economic_grasp_label_300views" / name, out / f"economic_{name}")
            with np.load(out / f"economic_{name}") as archive:
                manifest["labels"]["scenes"][str(scene)] = {
                    "file": f"economic_{name}",
                    "points": int(archive["points"].shape[0]),
                    "arrays": {
                        key: [str(archive[key].dtype), *archive[key].shape] for key in archive
                    },
                }
        print(json.dumps(manifest["labels"]), flush=True)

    # The model modules parse the command line when they are imported.
    sys.argv = [script, "--dataset_root", root, "--camera", args.camera]
    import libs.pointnet2.pointnet2_utils as query_module
    import MinkowskiEngine as ME
    import models.economicgrasp as model_module
    from dataset.graspnet_dataset import GraspNetDataset, collate_fn
    from graspnetAPI import GraspGroup
    from models.loss_economicgrasp import get_loss
    from utils.arguments import cfgs
    from utils.collision_detector import ModelFreeCollisionDetector

    manifest["num_points"] = cfgs.num_point
    manifest["versions"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "MinkowskiEngine": ME.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    device = torch.device("cuda:0")

    def load_network(training):
        net = model_module.economicgrasp(seed_feat_dim=512, is_training=training).to(device)
        net.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
        return net.train() if training else net.eval()

    manifest["reference_parameters"] = int(
        sum(p.numel() for p in load_network(False).parameters() if p.requires_grad)
    )

    # ------------------------------------------------------------------ inference
    dataset = GraspNetDataset(
        root,
        split="test",
        camera=args.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    net = load_network(False)
    stages = [
        "point_feats",
        "objectness",
        "graspness",
        "seed_index",
        "view_score",
        "view_index",
        "crop",
        "angle_pred",
        "depth_pred",
        "score_pred",
        "width_pred",
        "decoded",
    ]

    def run_inference(sample, detailed=True):
        recorder = Recorder(torch, net, model_module, query_module, detailed).install()
        try:
            batch = to_device(collate_fn([sample]), device)
            with torch.no_grad():
                end_points = net(batch)
                decoded = model_module.pred_decode(end_points)
            data = recorder.collect(batch, end_points, cfgs.graspness_threshold)
        finally:
            recorder.remove()
        data["decoded"] = np.stack([numpy(d) for d in decoded])
        return data

    def extra_classes(data):
        """How often the never-supervised extra class won the decoder's arg-max."""
        return {
            "angle": int((data["angle_pred"].argmax(-1) == cfgs.num_angle).sum()),
            "depth": int((data["depth_pred"].argmax(-1) == cfgs.num_depth).sum()),
        }

    for scene, frame in INFERENCE_FRAMES:
        index = (scene - 100) * 256 + frame
        assert dataset.scenename[index] == f"scene_{scene:04d}" and dataset.frameid[index] == frame
        np.random.seed(args.seed + index)
        sample = dataset[index]
        data = run_inference(sample)
        again = run_inference(sample, detailed=False)
        set_tf32(torch, True)
        default = run_inference(sample, detailed=False)
        set_tf32(torch, False)

        # The collision filter of the upstream test script: the sparse-conv voxel size.
        full_cloud, _ = dataset.get_data(index, return_raw_cloud=True)
        grasps = GraspGroup(data["decoded"][0])
        detector = ModelFreeCollisionDetector(full_cloud, voxel_size=cfgs.voxel_size)
        collision = detector.detect(grasps, approach_dist=0.05, collision_thresh=0.01)
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
            "extra_class_wins": extra_classes(data),
            "repeat_run_max_abs_diff": difference(data, again, stages),
            "tf32_default_max_abs_diff": difference(data, default, stages),
        }
        manifest["inference"].append(entry)
        print(json.dumps(entry), flush=True)

    # ---------------------------------------------------------------------- sweep
    if args.sweep > 0:
        chosen = np.unique(np.linspace(0, len(dataset) - 1, args.sweep).round().astype(int))
        names = ("scene", "frame", "point_clouds", "objectness", "graspness", "candidate_mask")
        sweep = {name: [] for name in (*names, "seed_index", "decoded")}
        decisions = {"view": "view_score", "angle": "angle_pred", "depth": "depth_pred"}
        for what in decisions:
            sweep.update({f"{what}_{part}": [] for part in ("choice", "margin", "scale")})
        tf32_other_seeds = 0
        wins = {"angle": 0, "depth": 0}
        for index in chosen.tolist():
            np.random.seed(args.seed + index)
            sample = dataset[index]
            data = run_inference(sample, detailed=False)
            set_tf32(torch, True)
            default = run_inference(sample, detailed=False)
            set_tf32(torch, False)
            tf32_other_seeds += int(not np.array_equal(data["seed_index"], default["seed_index"]))
            for what, count in extra_classes(data).items():
                wins[what] += count
            sweep["scene"].append(int(dataset.scenename[index].split("_")[1]))
            sweep["frame"].append(int(dataset.frameid[index]))
            mask = np.zeros(data["point_clouds"].shape[1], dtype=bool)
            mask[data["candidates_0"]] = True
            sweep["candidate_mask"].append(mask)
            for name in ("point_clouds", "objectness", "graspness", "seed_index", "decoded"):
                sweep[name].append(data[name][0])
            for what, stage in decisions.items():
                values = data[stage][0]  # (M, classes)
                ranked = np.sort(values, axis=-1)
                sweep[f"{what}_choice"].append(values.argmax(axis=-1))
                sweep[f"{what}_margin"].append(ranked[:, -1] - ranked[:, -2])
                sweep[f"{what}_scale"].append(np.abs(values).max())
        np.savez(out / "sweep.npz", **{name: np.stack(values) for name, values in sweep.items()})
        manifest["sweep"] = {
            "file": "sweep.npz",
            "frames": len(chosen),
            "min_candidates": int(min(mask.sum() for mask in sweep["candidate_mask"])),
            "extra_class_wins": wins,
            # how often the reference picks other seeds than itself under its default TF32
            "tf32_default_frames_with_other_seeds": tf32_other_seeds,
        }
        print(json.dumps(manifest["sweep"]), flush=True)
    del net
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- training
    if not args.skip_training:
        train = GraspNetDataset(
            str(view),
            split="train",
            camera=args.camera,
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=True,
            load_label=True,
        )
        indices = [scene * 256 + frame for scene, frame in TRAINING_FRAMES]
        for index, (scene, frame) in zip(indices, TRAINING_FRAMES):
            assert train.scenename[index] == f"scene_{scene:04d}" and train.frameid[index] == frame
        np.random.seed(args.seed)
        samples = [train[index] for index in indices]

        net = load_network(True)
        torch.manual_seed(args.seed)
        recorder = Recorder(torch, net, model_module, query_module, detailed=False).install()
        try:
            batch = to_device(collate_fn(samples), device)
            end_points = net(batch)
            data = recorder.collect(batch, end_points, cfgs.graspness_threshold)
            total, end_points = get_loss(end_points)
            total.backward()
        finally:
            recorder.remove()
        losses = {
            key: float(value)
            for key, value in end_points.items()
            if key[:2] in ("A:", "B:", "C:", "D:")
        }
        for key, value in losses.items():
            data["stat_" + key] = np.float64(value)
        gradients = {
            "backbone.conv0p1s1.kernel": net.backbone.conv0p1s1.kernel,
            "backbone.block4.0.conv2.kernel": net.backbone.block4[0].conv2.kernel,
            "backbone.convtr7p2s2.kernel": net.backbone.convtr7p2s2.kernel,
            "backbone.final.kernel": net.backbone.final.kernel,
            "graspable.conv_graspable.weight": net.graspable.conv_graspable.weight,
            "view.conv2.weight": net.view.conv2.weight,
            "cy_group.mlps.layer0.conv.weight": net.cy_group.mlps.layer0.conv.weight,
            "cy_group.local_interaction_module.msa.linears.0.weight": (
                net.cy_group.local_interaction_module.msa.linears[0].weight
            ),
            "cy_group.mlps2.layer1.conv.weight": net.cy_group.mlps2.layer1.conv.weight,
            "grasp_head.conv_angle_feature.weight": net.grasp_head.conv_angle_feature.weight,
            "grasp_head.global_interaction_module.msa.linears.3.weight": (
                net.grasp_head.global_interaction_module.msa.linears[3].weight
            ),
            "grasp_head.conv_score.weight": net.grasp_head.conv_score.weight,
        }
        for key, parameter in gradients.items():
            data["grad_" + key] = numpy(parameter.grad)
        squared = sum(
            float(p.grad.double().pow(2).sum()) for p in net.parameters() if p.grad is not None
        )
        data["grad_norm"] = np.float64(squared**0.5)
        data["objectness_label"] = numpy(batch["objectness_label"]).astype(np.int64)
        data["graspness_label"] = numpy(batch["graspness_label"])
        data["objects_per_frame"] = np.array([len(p) for p in batch["object_poses_list"]])
        for b in range(len(samples)):
            data[f"poses_{b}"] = np.stack([numpy(pose) for pose in batch["object_poses_list"][b]])
            data[f"label_points_{b}"] = np.array(
                [len(points) for points in batch["grasp_points_list"][b]]
            )

        # The matching once more on the same inputs: its indexed assignment with repeated
        # indices makes the outcome differ from run to run.
        with torch.no_grad():
            _, repeated = model_module.process_grasp_labels(dict(end_points))
        changed = numpy(repeated["batch_valid_mask"]) != data["label_valid"]
        relabelled = (numpy(repeated["batch_grasp_rotations"]) == cfgs.num_angle) != (
            data["label_angle"] == cfgs.num_angle
        )
        np.savez(out / "train_batch.npz", **data)
        manifest["training"] = {
            "file": "train_batch.npz",
            "frames": [list(pair) for pair in TRAINING_FRAMES],
            "augment": True,
            "objects_per_frame": data["objects_per_frame"].tolist(),
            "mirrored_frames": [
                bool(np.linalg.det(data[f"poses_{b}"][0][:3, :3]) < 0) for b in range(len(samples))
            ],
            "candidates": [int(data[f"candidates_{b}"].shape[0]) for b in range(len(samples))],
            "valid_seeds": int(data["label_valid"].sum()),
            "labelled_seeds": int((data["label_angle"] != cfgs.num_angle).sum()),
            "repeat_matching": {
                "seeds_with_other_validity": int(changed.sum()),
                "seeds_with_other_labelled_state": int(relabelled.sum()),
            },
            "dropped_attention_weights": {
                key: int((~data[key]).sum()) for key in ("crop_dropout_keep", "head_dropout_keep")
            },
            "statistics": losses,
            "grad_norm": float(data["grad_norm"]),
        }
        print(json.dumps(manifest["training"]), flush=True)
        remove_view(view)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
