"""Record reference activations of FineGrasp from the authors' implementation.

This script is the *oracle recorder* of the FineGrasp parity tests. It does not import
GraspPanda and is run with an interpreter that has the reference's compiled
dependencies (Python >= 3.10, torch, MinkowskiEngine 0.5.4, the pointnet2 and knn
extensions, Open3D, graspnetAPI), exactly like ``tools/oracle/finegrasp_reference.py``,
whose module docstring says what the reference is and which three things are supplied
because it does not ship them (the knn wrapper, pointnet2, the normal images)::

    /path/to/env/bin/python -B tools/make_finegrasp_fixtures.py \
        --site /scratch/site --pointnet2 /path/to/graspness_unofficial \
        --project /scratch/robo_orchard_lab/projects/finegrasp_graspnet1b \
        --pipeline "$GRASPPANDA_WEIGHTS/finegrasp/finegrasp_pipeline" \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" \
        --economic-labels "$GRASPPANDA_CACHE/fixtures/economicgrasp" \
        --out "$GRASPPANDA_CACHE/fixtures/finegrasp"

It runs the reference unmodified and records what flows between its stages through
forward hooks and by wrapping the few free functions the model calls. Nothing of the
reference's forward pass is re-implemented here.

Recorded (float32 unless stated), see ``manifest.json`` in the output directory:

``infer_<scene>_<frame>.npz``
    Inputs of one test frame as the reference loader produced them (cloud, normals, the
    normal image they were read from), the voxelisation with the input feature of every
    voxel, backbone activations at every level, per-point features, objectness /
    graspness, the sampling candidates and the seed indices, view scores and chosen
    views, and for each of the four crop scales the cylinder neighbours, the local
    offsets and the pooled crop; the scale tokens after each fusion layer, the fusion
    weights and the fused feature; the task tokens of the head before and after their
    attention, the four head outputs, the decoded ``(1024, 17)`` array, and the
    collision mask of the reference's detector. The per-neighbour activations of a crop
    (shared MLP, tokens before and after the neighbour attention, second MLP) are kept
    for every ``DETAIL_STRIDE``-th seed.
``sweep.npz``
    Inputs, per-point scores, candidate masks, seed indices, decoded grasps and the
    three arg-max decisions behind every grasp (with their margins) of ``--sweep``
    further test frames spread over all test scenes.
``train_batch.npz``
    One training batch with its object poses, the view indices drawn by
    ``torch.multinomial``, the entries dropped by every dropout module (so the
    stochastic steps can be replayed), the label-matching outputs, every loss term, and
    a few parameter gradients.

Three things about the training record:

* The compact labels are the files the EconomicGrasp generator wrote for the same
  scenes (``--economic-labels``, recorded by ``tools/make_economicgrasp_fixtures.py``);
  the reference README has them generated with that very script.
* The graspness files are the dataset's ``graspness`` folder under the name the
  reference reads (``instance_norm_graspness``). The authors' instance-normalised files
  for GraspNet-1Billion are not published; the loss consumes these values as numbers,
  and what they contain is not under test here.
* The two transformer layers of the scale fusion drop attention weights inside a
  compiled attention kernel, where the draw cannot be observed. Their attention dropout
  is switched off for the recorded batch; the three dropout modules of each layer are
  recorded like all others.

Precision. TF32 is switched off for the records; the manifest reports how far the
library defaults and a second identical run move the results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
#: Per-neighbour activations of a crop are kept for every DETAIL_STRIDE-th seed.
DETAIL_STRIDE = 8
COLLISION = {"voxel_size": 0.01, "approach": 0.05, "threshold": 0.01}


def load_oracle():
    """The helpers shared with ``tools/oracle/finegrasp_reference.py``."""
    path = Path(__file__).resolve().parent / "oracle" / "finegrasp_reference.py"
    spec = importlib.util.spec_from_file_location("finegrasp_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numpy(value):
    return value.detach().cpu().numpy()


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

    def _per_seed(self, tensor, batch):
        """``(G, N * B, …)`` of the fusion (seed-major) → ``(B * N, G, …)`` (frame-major)."""
        groups, rows = tensor.shape[:2]
        seeds = rows // batch
        shaped = tensor.reshape(groups, seeds, batch, *tensor.shape[2:])
        order = (2, 1, 0, *range(3, shaped.ndim))
        return shaped.permute(*order).reshape(batch * seeds, groups, *tensor.shape[2:])

    def install(self):
        net, calls, data = self.net, self.calls, self.data
        stride = DETAIL_STRIDE

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

        def grouped(scale):
            def hook(module, inputs, output):
                if scale == 0:
                    data["crop_centre"] = numpy(inputs[1])
                    data["crop_rotation"] = numpy(inputs[2])
                if self.detailed:  # (B, 3 + C, M, K) -> (B, M, K, 3 + C)
                    local = numpy(output[:, :3]).transpose(0, 2, 3, 1)
                    data[f"grouped_xyz_{scale}"] = local.copy()
                    feats = numpy(output[:, 3:, ::stride]).transpose(0, 2, 3, 1)
                    data[f"grouped_feats_{scale}"] = feats.copy()

            return hook

        def shared_mlp(key):
            def hook(module, inputs, output):  # (B, C, M, K) -> (B, M / stride, K, C)
                data[key] = numpy(output[:, :, ::stride]).transpose(0, 2, 3, 1).copy()

            return hook

        def crop_interaction(key):
            def hook(module, inputs, output):  # (B * M, K, C) -> (B, M / stride, K, C)
                batch = data["crop_centre"].shape[0]
                for name, tensor in (("_in", inputs[0]), ("_out", output)):
                    shaped = tensor.reshape(batch, -1, *tensor.shape[1:])[:, ::stride]
                    data[key + name] = numpy(shaped).copy()

            return hook

        def interaction(key):
            def hook(module, inputs, output):
                data[key + "_in"] = numpy(inputs[0]).copy()  # (B * M, tokens, C)
                data[key + "_out"] = numpy(output).copy()

            return hook

        def dropped(key, reorder=False):
            def hook(module, inputs, output):
                if not module.training:
                    return
                # An entry that was zero before the dropout is zero either way; it is
                # recorded as kept.
                keep = (output != 0) | (inputs[0] == 0)
                if reorder:
                    keep = self._per_seed(keep, data["crop_centre"].shape[0])
                data[key] = numpy(keep)
                data[key + "_dropped"] = np.int64((~keep).sum().item())

            return hook

        def pooled(scale):
            def hook(module, inputs, output):
                data[f"crop_{scale}"] = numpy(output).transpose(0, 2, 1).copy()  # (B, M, C)

            return hook

        for scale, group in enumerate(net.cy_groups):
            self._hook(group.grouper, grouped(scale))
            if self.detailed:
                self._hook(group.mlps, shared_mlp(f"crop_mlp_{scale}"))
                self._hook(group.mlps2, shared_mlp(f"crop_post_mlp_{scale}"))
                self._hook(group.local_interaction_module, crop_interaction(f"crop_tokens_{scale}"))
            self._hook(
                group.local_interaction_module.msa.dropout, dropped(f"crop_dropout_keep_{scale}")
            )
            self._hook(group, pooled(scale))

        fusion = net.fuse_multi_scale

        def fused(module, inputs, output):
            data["fused"] = numpy(output).transpose(0, 2, 1).copy()  # (B, M, C)

        self._hook(fusion, fused)

        def fusion_layer(index):
            def hook(module, inputs, output):  # (G, N * B, C) -> (B * N, G, C)
                batch = data["crop_centre"].shape[0]
                data[f"fusion_layer_{index}"] = numpy(self._per_seed(output, batch)).copy()

            return hook

        def fusion_score(module, inputs, output):  # (G, N * B, 1) logits
            batch = data["crop_centre"].shape[0]
            weights = self.torch.softmax(output, dim=0)
            data["fusion_weights"] = numpy(self._per_seed(weights, batch))[..., 0].copy()

        for index, layer in enumerate(fusion.transformer.layers):
            self._hook(layer, fusion_layer(index))
            for name in ("dropout1", "dropout", "dropout2"):
                key = f"fusion_{index}_{name}_keep"
                self._hook(getattr(layer, name), dropped(key, reorder=True))
        self._hook(fusion.attn, fusion_score)

        self._hook(net.grasp_head.global_interaction_module, interaction("head_tokens"))
        self._hook(
            net.grasp_head.global_interaction_module.msa.dropout, dropped("head_dropout_keep")
        )

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
            def wrapper(end_points, model_cfgs):
                rotation, end_points = original(end_points, model_cfgs)
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
        data["cloud_normal"] = numpy(batch["cloud_normal"])
        data["coors"] = numpy(end_points["coors"]).astype(np.int32)
        data["voxel_feats"] = numpy(end_points["feats"])
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
        data["cylinder_index"] = np.stack(self.calls["cylinder_index"])  # (scales, B, M, K)
        for what in ("angle", "depth", "score", "width"):  # (B, classes, M) -> (B, M, classes)
            data[what + "_pred"] = numpy(end_points[f"grasp_{what}_pred"]).transpose(0, 2, 1).copy()
        if self.calls["multinomial"]:
            data["sampled_view_index"] = np.stack(self.calls["multinomial"])
        return data


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


def main(argv=None):
    oracle = load_oracle()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    oracle.add_reference_arguments(parser)
    parser.add_argument("--out", required=True, help="output directory (under the cache root)")
    parser.add_argument(
        "--economic-labels",
        help="directory with economic_scene_<id>_labels.npz of the training scenes, as "
        "tools/make_economicgrasp_fixtures.py records them; needed for the training batch",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", type=int, default=32, help="frames of the seed sweep (0: none)")
    parser.add_argument("--skip-training", action="store_true", help="record inference only")
    args = parser.parse_args(argv)
    if args.normals != "image":
        parser.error("the fixtures are recorded with --normals image")
    if not args.skip_training and not args.economic_labels:
        parser.error("--economic-labels is needed for the training batch (or --skip-training)")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = None if args.economic_labels is None else Path(args.economic_labels).resolve()

    reference = oracle.Reference(args)
    torch = reference.torch
    import pointnet2.pointnet2_utils as query_module
    from graspnetAPI import GraspGroup

    model_module, utils_module = reference.model_module, reference.utils_module
    manifest = reference.manifest()
    manifest.update(
        {
            "voxel_size": 0.005,
            "seed": args.seed,
            "precision": "float32 (TF32 disabled)",
            "detail_stride": DETAIL_STRIDE,
            "collision_filter": COLLISION,
            "inference": [],
        }
    )
    reference.set_tf32(False)

    # ------------------------------------------------------------------ inference
    view = reference.dataset_view(range(100, 190))
    _, dataset = reference.datasets(view)
    model_cfg, net = reference.build_model(training=False)
    manifest["num_points"] = int(dataset.num_sample_points)
    manifest["reference_parameters"] = int(
        sum(p.numel() for p in net.parameters() if p.requires_grad)
    )
    manifest["reference_tensors"] = len(net.state_dict())
    manifest["model_config"] = {
        key: getattr(model_cfg, key)
        for key in (
            "seed_feat_dim",
            "graspness_threshold",
            "grasp_max_width",
            "num_depth",
            "num_view",
            "num_angle",
            "num_seed_points",
            "voxel_size",
            "cylinder_radius",
            "cylinder_groups",
            "use_normal",
        )
    }
    stages = [
        "point_feats",
        "objectness",
        "graspness",
        "seed_index",
        "view_score",
        "view_index",
        "fused",
        "angle_pred",
        "depth_pred",
        "score_pred",
        "width_pred",
        "decoded",
    ]

    def run_inference(sample, detailed=True):
        recorder = Recorder(torch, net, model_module, query_module, detailed).install()
        try:
            batch = reference.to_device(reference.collate([sample]))
            with torch.no_grad():
                end_points = net(batch)
                decoded = utils_module.pred_decode(
                    end_points, model_cfg.grasp_max_width, model_cfg.num_seed_points
                )
            data = recorder.collect(batch, end_points, model_cfg.graspness_threshold)
        finally:
            recorder.remove()
        data["decoded"] = np.stack([numpy(d) for d in decoded])
        return data

    def extra_classes(data):
        """How often the never-supervised extra class won the decoder's arg-max."""
        return {
            "angle": int((data["angle_pred"].argmax(-1) == model_cfg.num_angle).sum()),
            "depth": int((data["depth_pred"].argmax(-1) == model_cfg.num_depth).sum()),
        }

    for scene, frame in INFERENCE_FRAMES:
        index = (scene - 100) * 256 + frame
        assert dataset.scenename[index] == f"scene_{scene:04d}" and dataset.frameid[index] == frame
        np.random.seed(args.seed + index)
        sample = reference.load(dataset, index)
        data = run_inference(sample)
        again = run_inference(sample, detailed=False)
        reference.set_tf32(None)  # the library defaults
        default = run_inference(sample, detailed=False)
        reference.set_tf32(False)

        # The collision filter of the reference metric and of the released processor.
        full_cloud = np.asarray(sample["cloud_masked"][0])
        grasps = GraspGroup(data["decoded"][0])
        detector = utils_module.ModelFreeCollisionDetector(
            full_cloud, voxel_size=COLLISION["voxel_size"]
        )
        collision = detector.detect(
            grasps, approach_dist=COLLISION["approach"], collision_thresh=COLLISION["threshold"]
        )
        data["full_cloud"] = np.asarray(full_cloud, dtype=np.float64)
        data["collision_scene"] = np.asarray(detector.scene_points, dtype=np.float64)
        data["collision_mask"] = np.asarray(collision, dtype=bool)
        data["normal_image"] = reference.normal_map(dataset, index)

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
        names = ("scene", "frame", "point_clouds", "cloud_normal", "objectness", "graspness")
        sweep = {name: [] for name in (*names, "candidate_mask", "seed_index", "decoded")}
        decisions = {"view": "view_score", "angle": "angle_pred", "depth": "depth_pred"}
        for what in decisions:
            sweep.update({f"{what}_{part}": [] for part in ("choice", "margin", "scale")})
        wins = {"angle": 0, "depth": 0}
        for index in chosen.tolist():
            np.random.seed(args.seed + index)
            data = run_inference(reference.load(dataset, index), detailed=False)
            for what, count in extra_classes(data).items():
                wins[what] += count
            sweep["scene"].append(int(dataset.scenename[index].split("_")[1]))
            sweep["frame"].append(int(dataset.frameid[index]))
            mask = np.zeros(data["point_clouds"].shape[1], dtype=bool)
            mask[data["candidates_0"]] = True
            sweep["candidate_mask"].append(mask)
            for name in ("point_clouds", "cloud_normal", "objectness", "graspness"):
                sweep[name].append(data[name][0])
            for name in ("seed_index", "decoded"):
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
        }
        print(json.dumps(manifest["sweep"]), flush=True)
    del net
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- training
    if not args.skip_training:
        scenes = sorted({scene for scene, _ in TRAINING_FRAMES})
        files = {scene: labels / f"economic_scene_{scene:04d}_labels.npz" for scene in scenes}
        for file in files.values():
            if not file.is_file():
                raise SystemExit(f"no compact labels of the training scenes: {file}")
        compact = Path(reference.dataset_view([], extra=())) / "labels"
        compact.mkdir()
        for scene, file in files.items():
            os.symlink(file, compact / f"scene_{scene:04d}_labels.npz")
        view = reference.dataset_view(
            scenes,
            extra=(
                ("economic_grasp_label_300views", str(compact)),
                ("instance_norm_graspness", os.path.join(reference.root, "graspness")),
            ),
        )
        train, _ = reference.datasets(view)
        assert train.use_new_graspness and train.augment and train.load_label
        indices = [scene * 256 + frame for scene, frame in TRAINING_FRAMES]
        for index, (scene, frame) in zip(indices, TRAINING_FRAMES):
            assert train.scenename[index] == f"scene_{scene:04d}" and train.frameid[index] == frame
        # The batch is meant to cover the flip augmentation: the first sampling seed from
        # --seed on whose batch holds a mirrored and an unmirrored frame is used.
        for sampling_seed in range(args.seed, args.seed + 64):
            np.random.seed(sampling_seed)
            samples = [reference.load(train, index) for index in indices]
            mirrored = [
                bool(np.linalg.det(sample["object_poses_list"][0][:3, :3]) < 0)
                for sample in samples
            ]
            if any(mirrored) and not all(mirrored):
                break
        else:
            raise SystemExit("no sampling seed gave a batch with and without a flip")

        model_cfg, net = reference.build_model(training=True)
        weights = {type(loss).__name__: float(loss.loss_weight) for loss in net.loss}
        for layer in net.fuse_multi_scale.transformer.layers:
            layer.self_attn.dropout = 0.0  # see the module docstring
        torch.manual_seed(args.seed)
        recorder = Recorder(torch, net, model_module, query_module, detailed=False).install()
        try:
            batch = reference.to_device(reference.collate(samples))
            end_points = net(batch)
            data = recorder.collect(batch, end_points, model_cfg.graspness_threshold)
            terms = {key: value for key, value in end_points.items() if "loss" in key}
            total = sum(terms.values())  # what the reference's training step sums
            total.backward()
        finally:
            recorder.remove()
        statistics = {key: float(value) for key, value in terms.items()}
        statistics["total"] = float(total)
        for key in end_points:
            if key.startswith(("stage1_", "C: ", "D: ")):
                statistics[key] = float(end_points[key])
        for key, value in statistics.items():
            data["stat_" + key] = np.float64(value)
        gradients = {
            "backbone.conv0p1s1.kernel": net.backbone.conv0p1s1.kernel,
            "backbone.block4.0.conv2.kernel": net.backbone.block4[0].conv2.kernel,
            "backbone.convtr7p2s2.kernel": net.backbone.convtr7p2s2.kernel,
            "backbone.final.kernel": net.backbone.final.kernel,
            "graspable.conv_graspable.weight": net.graspable.conv_graspable.weight,
            "view.conv2.weight": net.view.conv2.weight,
            "cy_groups.0.mlps.layer0.conv.weight": net.cy_groups[0].mlps.layer0.conv.weight,
            "cy_groups.3.mlps.layer0.conv.weight": net.cy_groups[3].mlps.layer0.conv.weight,
            "cy_groups.1.local_interaction_module.msa.linears.0.weight": (
                net.cy_groups[1].local_interaction_module.msa.linears[0].weight
            ),
            "cy_groups.2.mlps2.layer1.conv.weight": net.cy_groups[2].mlps2.layer1.conv.weight,
            "fuse_multi_scale.transformer.layers.0.self_attn.in_proj_weight": (
                net.fuse_multi_scale.transformer.layers[0].self_attn.in_proj_weight
            ),
            "fuse_multi_scale.transformer.layers.0.self_attn.out_proj.weight": (
                net.fuse_multi_scale.transformer.layers[0].self_attn.out_proj.weight
            ),
            "fuse_multi_scale.transformer.layers.1.linear2.weight": (
                net.fuse_multi_scale.transformer.layers[1].linear2.weight
            ),
            "fuse_multi_scale.attn.weight": net.fuse_multi_scale.attn.weight,
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
            _, repeated = model_module.process_grasp_labels(dict(end_points), model_cfg)
        changed = numpy(repeated["batch_valid_mask"]) != data["label_valid"]
        relabelled = (numpy(repeated["batch_grasp_rotations"]) == model_cfg.num_angle) != (
            data["label_angle"] == model_cfg.num_angle
        )
        np.savez(out / "train_batch.npz", **data)
        manifest["training"] = {
            "file": "train_batch.npz",
            "frames": [list(pair) for pair in TRAINING_FRAMES],
            "augment": True,
            "sampling_seed": sampling_seed,
            "labels": {str(scene): str(file) for scene, file in files.items()},
            "graspness_files": "the dataset's `graspness` folder, as a stand-in (see docstring)",
            "fusion_attention_dropout": "switched off (see docstring)",
            "loss_weights": weights,
            "objects_per_frame": data["objects_per_frame"].tolist(),
            "mirrored_frames": mirrored,
            "candidates": [int(data[f"candidates_{b}"].shape[0]) for b in range(len(samples))],
            "valid_seeds": int(data["label_valid"].sum()),
            "labelled_seeds": int((data["label_angle"] != model_cfg.num_angle).sum()),
            "repeat_matching": {
                "seeds_with_other_validity": int(changed.sum()),
                "seeds_with_other_labelled_state": int(relabelled.sum()),
            },
            "dropped_entries": {
                key[: -len("_dropped")]: int(value)
                for key, value in data.items()
                if key.endswith("_dropped")
            },
            "statistics": statistics,
            "grad_norm": float(data["grad_norm"]),
        }
        print(json.dumps(manifest["training"]), flush=True)
        for link in sorted(compact.iterdir()):
            link.unlink()
        compact.rmdir()

    reference.close()
    manifest["seconds"] = round(time.time() - START, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", out / "manifest.json")
    return 0


START = time.time()

if __name__ == "__main__":
    raise SystemExit(main())
