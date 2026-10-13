"""Record reference activations of HGGD from the upstream implementation.

This script is the *oracle recorder* of the HGGD parity tests. It does not import
GraspPanda and is run with the interpreter of the upstream environment (torch 1.11,
pytorch3d, numba, cupoch, torchvision, the bundled ``customgraspnetAPI``):

    $GRASPPANDA_ORACLE_PYTHON -B tools/make_hggd_fixtures.py \
        --reference /path/to/HGGD --checkpoint /path/to/checkpoint-realsense.tar \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --camera realsense \
        --out "$GRASPPANDA_CACHE/fixtures/hggd"

It loads the pristine upstream code read-only (no bytecode is written, the working
directory is a fresh temporary directory), calls its functions in the order of
``test_graspnet.py`` / ``train_graspnet.py`` with the options of the shipped shell
scripts, and records what flows between them through forward hooks and by wrapping the
free functions and the random number sources they call. Nothing of the upstream
computation is re-implemented here.

Recorded (see ``manifest_<camera>.json`` in the output directory):

``infer_<camera>_<scene>_<frame>.npz``
    One test frame: the network input and the full-resolution depth exactly as the
    upstream loader produced them, the sampled scene cloud with the indices that were
    drawn, the raw and processed outputs of the image network, backbone activations,
    the candidate pixels and the selected partial grasps, the lifted centres, the
    point-to-pixel neighbours and the fused cloud, every random draw of the region
    stage (re-sampling indices, jitter, sampling start points), the regions, the local
    network outputs, the decoded grasps, the voxelised cloud of the collision filter,
    its decisions, and the grasps kept by NMS with and without the filter.
``train_<camera>.npz``, ``train_<camera>_labels.npz`` (with ``--training``)
    One training batch: inputs, the dense targets, every loss term, the proposals,
    regions and matched labels, the dropout masks that were drawn, a few parameter
    gradients, and the result of one anchor-shifting step on the batch's labels. The
    per-view label files the upstream loader reads are a separate download; when
    ``--label-root`` does not name a directory that has them, the pristine upstream
    converter (``convert_grasp.run``) is run for the four frames of the batch and its
    files are kept under ``<out>/upstream_labels/<camera>``.

The upstream test loop opens a label file for every test frame without using it for
the dump; the recorder's dataset returns an empty label set there instead (see
``tools/oracle/hggd_reference.py``).

Precision. TF32 is switched off, so matrix products and convolutions run in true
float32, and cuDNN auto-tuning is off.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

INFERENCE_FRAMES = {
    "realsense": ((100, 0), (118, 137), (145, 64), (175, 200)),  # seen x2, similar, novel
    "kinect": ((100, 0), (160, 128)),
}
TRAINING_FRAMES = ((5, 17), (5, 201), (63, 100), (63, 31))
OPTIONS = {  # test_graspnet.sh / train_graspnet.sh
    "anchor_num": 7,
    "anchor_k": 6,
    "anchor_w": 50.0,
    "anchor_z": 20.0,
    "grid_size": 8,
    "all_points_num": 25600,
    "group_num": 512,
    "ratio": 8,
    "input_h": 360,
    "input_w": 640,
    "sigma": 10,
    "grasp_count": 5000,
    "test": {"center_num": 48, "local_k": 10, "heatmap_thres": 0.01},
    "train": {
        "center_num": 128,
        "batch_size": 4,
        "loc_a": 1.0,
        "reg_b": 5.0,
        "cls_c": 1.0,
        "offset_d": 1.0,
        "local_grasp_num": 500,  # argparse default; the shell script does not set it
    },
}
LABEL_FIELDS = (
    "centers_2d",
    "thetas_rad",
    "gammas_rad",
    "betas_rad",
    "widths_2d",
    "heights_2d",
    "scores_from_6d",
    "object_ids",
    "center_z_depths",
)
GRADIENT_TAPS = {
    "anchor": ("backbone.net.conv1.0.weight", "hmap.1.weight", "cls_mask_conv.1.weight"),
    "local": ("pointnet.conv1.weight", "anchor_mlp.3.weight", "offset_mlp.3.weight"),
}


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


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Taps:
    """Wraps functions of the upstream modules and records what passes through them."""

    def __init__(self):
        self.calls = {}
        self._originals = []

    def record(self, name, value):
        self.calls.setdefault(name, []).append(value)

    def wrap(self, owner, attribute, name, capture):
        """Replace ``owner.attribute`` by a wrapper that calls ``capture(args, kwargs, out)``."""
        original = getattr(owner, attribute)
        self._originals.append((owner, attribute, original))

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            value = capture(args, kwargs, out)
            if value is not None:
                self.record(name, value)
            return out

        setattr(owner, attribute, wrapper)

    def restore(self):
        for owner, attribute, original in reversed(self._originals):
            setattr(owner, attribute, original)
        self._originals.clear()

    def take(self, name):
        return self.calls.pop(name, [])


@contextlib.contextmanager
def tapped(install):
    taps = Taps()
    install(taps)
    try:
        yield taps
    finally:
        taps.restore()


def ragged(arrays, dtype):
    """Concatenate arrays of different length; return ``(values, offsets)``."""
    arrays = [np.asarray(a, dtype=dtype) for a in arrays]
    sizes = np.array([len(a) for a in arrays], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    tail = arrays[0].shape[1:] if arrays else ()
    values = np.concatenate(arrays) if arrays else np.zeros((0, *tail), dtype)
    return values, offsets


# --------------------------------------------------------------------- upstream set-up


def select_camera(camera):
    """Upstream fixes the camera in ``dataset/config.py``; a user edits that file."""
    from dataset import config

    config.camera = camera
    config.get_camera_intrinsic.__defaults__ = (camera,)
    return config.get_camera_intrinsic()


def build(args, scenes, label_root):
    import torch
    from dataset.graspnet_dataset import GraspnetPointDataset
    from models.anchornet import AnchorGraspNet
    from models.localgraspnet import PointMultiGraspNet

    empty = {name: np.zeros((0, 2) if name == "centers_2d" else (0,)) for name in LABEL_FIELDS}

    class Dataset(GraspnetPointDataset):
        def load_grasp_labels(self, index):
            if label_root is None:  # the test loop never uses them for the dump
                return empty
            return super().load_grasp_labels(index)

    dataset = Dataset(
        OPTIONS["all_points_num"],
        label_root or "/nonexistent-hggd-labels",
        args.dataset_root,
        list(scenes),
        sigma=OPTIONS["sigma"],
        ratio=OPTIONS["ratio"],
        anchor_k=OPTIONS["anchor_k"],
        anchor_z=OPTIONS["anchor_z"],
        anchor_w=OPTIONS["anchor_w"],
        grasp_count=OPTIONS["grasp_count"],
        output_size=(OPTIONS["input_w"], OPTIONS["input_h"]),
        random_rotate=False,
        random_zoom=False,
    )
    anchornet = AnchorGraspNet(in_dim=4, ratio=OPTIONS["ratio"], anchor_k=OPTIONS["anchor_k"])
    localnet = PointMultiGraspNet(info_size=3, k_cls=OPTIONS["anchor_num"] ** 2)
    anchornet, localnet = anchornet.cuda(), localnet.cuda()
    checkpoint = torch.load(args.checkpoint)
    anchornet.load_state_dict(checkpoint["anchor"])
    localnet.load_state_dict(checkpoint["local"])
    anchors = {"gamma": checkpoint["gamma"], "beta": checkpoint["beta"]}
    return dataset, anchornet, localnet, anchors


def install_region_taps(taps):
    """Everything random or library-defined between the proposals and the regions."""
    import torch
    from dataset import evaluation, pc_dataset_tools
    from dataset import utils as dataset_utils

    taps.wrap(
        dataset_utils,
        "fast_sample",
        "sample_index",
        lambda a, k, out: np.asarray(out, dtype=np.int64),
    )
    taps.wrap(
        evaluation,
        "select_2d_center",
        "candidates",
        lambda a, k, out: np.asarray(out[0], dtype=np.int64),
    )
    taps.wrap(
        pc_dataset_tools,
        "knn_points",
        "knn_index",
        lambda a, k, out: numpy(out[1]).astype(np.int32),
    )
    taps.wrap(
        pc_dataset_tools,
        "center2dtopc",
        "lifted",
        lambda a, k, out: [numpy(centres).astype(np.float64) for centres in out],
    )
    taps.wrap(
        pc_dataset_tools,
        "sample_farthest_points",
        "fps",
        lambda a, k, out: {
            "points": numpy(a[0]),
            "lengths": numpy(k["lengths"]).astype(np.int64),
            "index": numpy(out[1]).astype(np.int64),
        },
    )
    # Draws in call order: per region with too few points the re-sampling indices, then
    # (inside pytorch3d) one start index per region.
    taps.wrap(
        torch,
        "randint",
        "randint",
        lambda a, k, out: numpy(out).astype(np.int64).reshape(-1),
    )
    taps.wrap(torch, "randn", "randn", lambda a, k, out: numpy(out))


def region_record(taps, prefix=""):
    """Turn the taps of one ``data_process`` call into arrays."""
    data = {}
    fps = taps.take("fps")
    randints = taps.take("randint")
    noise = taps.take("randn")
    # One call of get_group_pc per frame of the batch that has centres: per call one
    # jitter tensor, one sampling call, and randint draws = re-sampled regions + regions.
    jitter, starts, resample = [], [], []
    cursor = 0
    for call, one in enumerate(fps):
        regions = len(one["lengths"])
        mine = []
        while cursor < len(randints) and len(randints[cursor]) == OPTIONS["group_num"]:
            mine.append(randints[cursor])
            cursor += 1
        resample.append(mine)
        starts.append(np.concatenate(randints[cursor : cursor + regions]))
        cursor += regions
        jitter.append(noise[call])
        data[f"{prefix}fps_points_{call}"] = one["points"]
        data[f"{prefix}fps_lengths_{call}"] = one["lengths"]
        data[f"{prefix}fps_index_{call}"] = one["index"]
        data[f"{prefix}jitter_{call}"] = noise[call]
        data[f"{prefix}fps_start_{call}"] = starts[-1]
        data[f"{prefix}resample_{call}"] = (
            np.stack(mine) if mine else np.zeros((0, OPTIONS["group_num"]), np.int64)
        )
    if cursor != len(randints) or len(noise) != len(fps):
        raise RuntimeError("the random draws of the region stage were not attributed completely")
    data[f"{prefix}region_calls"] = np.int64(len(fps))
    return data


def rect_arrays(rect_gg, prefix):
    return {
        f"{prefix}pixel": np.asarray(rect_gg.centers, dtype=np.float64),
        f"{prefix}theta": np.asarray(rect_gg.thetas, dtype=np.float64),
        f"{prefix}width": np.asarray(rect_gg.widths, dtype=np.float64),
        f"{prefix}depth_offset": np.asarray(rect_gg.depths, dtype=np.float64),
        f"{prefix}score": np.asarray(rect_gg.scores, dtype=np.float64),
    }


# ------------------------------------------------------------------------- inference


def record_inference(args, torch, dataset, anchornet, localnet, anchors, index, key):
    import cupoch
    from customgraspnetAPI import Grasp as GraspNetGrasp
    from customgraspnetAPI import GraspGroup as GraspNetGraspGroup
    from dataset import collision_detector
    from dataset.evaluation import anchor_output_process, detect_2d_grasp, detect_6d_grasp_multi
    from dataset.pc_dataset_tools import data_process, feature_fusion
    from torch.utils.data.dataloader import default_collate
    from torch.utils.dlpack import to_dlpack

    test, size = OPTIONS["test"], (OPTIONS["input_w"], OPTIONS["input_h"])
    data = {}
    activations = {}
    hooks = [
        anchornet.backbone.net.register_forward_hook(
            lambda module, inputs, output: activations.update(
                {f"backbone_{level}": numpy(o) for level, o in enumerate(output)}
            )
        ),
        localnet.pointnet.stn.register_forward_hook(
            lambda module, inputs, output: activations.update({"stn": numpy(output)})
        ),
    ]
    seed_everything(torch, key[0] * 1000 + key[1])
    with tapped(install_region_taps) as taps:
        anchor_data, rgb, ori_depth, _ = default_collate([dataset[index]])
        depth = torch.from_numpy(ori_depth.numpy().squeeze()).float()[None]
        view_points, _, _ = dataset.helper.to_scene_points(
            rgb.cuda(), depth.cuda(), include_rgb=True
        )
        points = view_points[..., :3]
        xyzs = dataset.helper.to_xyz_maps(depth.cuda())
        x = anchor_data[0].cuda()
        pred_2d, features = anchornet(x)
        data.update(
            image=numpy(x)[0],
            depth=depth.numpy()[0],
            sample_index=taps.take("sample_index")[0],
            points=numpy(points)[0],
            xyz_map=numpy(xyzs)[0],
            heatmap_logits=numpy(pred_2d[0])[0, 0],
            angle_logits=numpy(pred_2d[1])[0],
            angle_offset_raw=numpy(pred_2d[2])[0],
            depth_offset_raw=numpy(pred_2d[3])[0],
            width_offset_raw=numpy(pred_2d[4])[0],
            features=numpy(features)[0],
        )
        data.update(activations)
        loc_map, cls_mask, theta_offset, depth_offset, width_offset = anchor_output_process(
            *pred_2d, sigma=OPTIONS["sigma"]
        )
        data.update(
            heatmap=loc_map,
            angle_score=cls_mask,
            angle_offset=theta_offset,
            depth_offset=depth_offset,
            width_offset=width_offset,
        )
        rect_gg = detect_2d_grasp(
            loc_map,
            cls_mask,
            theta_offset,
            depth_offset,
            width_offset,
            ratio=OPTIONS["ratio"],
            anchor_k=OPTIONS["anchor_k"],
            anchor_w=OPTIONS["anchor_w"],
            anchor_z=OPTIONS["anchor_z"],
            mask_thre=test["heatmap_thres"],
            center_num=test["center_num"],
            grid_size=OPTIONS["grid_size"],
            grasp_nms=OPTIONS["grid_size"],
            reduce="max",
        )
        data["candidates"] = taps.take("candidates")[0]
        data.update(rect_arrays(rect_gg, "proposal_"))
        if rect_gg.size == 0:
            raise RuntimeError(f"{key}: no 2-D proposal; choose another frame for the fixture")
        points_all = feature_fusion(points, features, xyzs)
        data["knn_index"] = taps.take("knn_index")[0][0]
        data["fused"] = numpy(points_all)[0]
        rect_ggs = [rect_gg]
        taps.take("randint"), taps.take("randn")  # nothing before this point belongs to regions
        pc_group, valid_local_centers = data_process(
            points_all,
            depth.cuda(),
            rect_ggs,
            test["center_num"],
            OPTIONS["group_num"],
            size,
            min_points=32,
            is_training=False,
        )
        data["lifted_centre"] = taps.take("lifted")[0][0]
        data.update(region_record(taps))
        rect_gg = rect_ggs[0]
        points_all = points_all.squeeze()
        data["region"] = numpy(pc_group)
        data["region_centre"] = numpy(valid_local_centers[0]).astype(np.float64)
        data.update(rect_arrays(rect_gg, "region_"))
        grasp_info = np.vstack([rect_gg.thetas[None], rect_gg.widths[None], rect_gg.depths[None]]).T
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device="cuda")
        data["grasp_info"] = numpy(grasp_info)
        local_feature, pred, offset = localnet(pc_group, grasp_info)
        data.update(
            local_feature=numpy(local_feature),
            anchor_logits=numpy(pred),
            offset_raw=numpy(offset),
        )
        data["stn"] = activations["stn"]
        pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
            rect_gg, pred, offset, valid_local_centers, size, anchors, k=test["local_k"]
        )
        data["decoded_parameters"] = np.asarray(pred_grasp, dtype=np.float64)
        decoded = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        data.update(
            decoded_pixel=np.asarray(pred_rect_gg.centers, dtype=np.float64),
            decoded_translation=np.asarray(decoded.translations, dtype=np.float64),
            decoded_rotation=np.asarray(decoded.rotations, dtype=np.float64),
            decoded_width=np.asarray(decoded.widths, dtype=np.float64),
            decoded_height=np.asarray(decoded.heights, dtype=np.float64),
            decoded_depth=np.asarray(decoded.depths, dtype=np.float64),
            decoded_score=np.asarray(decoded.scores, dtype=np.float64),
        )
    for hook in hooks:
        hook.remove()

    # Collision filter of the test script: mode "graspnet", voxel 0.01, approach 0.05.
    cloud = points_all[:, :3].clone()
    voxels = cupoch.geometry.PointCloud()
    voxels.from_points_dlpack(to_dlpack(cloud.clone()))
    data["voxel_points"] = np.array(voxels.voxel_down_sample(0.01).points.cpu(), dtype=np.float32)
    products = []
    with tapped(
        lambda taps: taps.wrap(
            torch, "matmul", "local", lambda a, k, out: products.append(numpy(out)[:32])
        )
    ):
        detector = collision_detector.ModelFreeCollisionDetector(
            cloud, voxel_size=0.01, mode="graspnet"
        )
        no_collision = detector.detect(decoded, approach_dist=0.05)
    data["collision_scene"] = np.asarray(detector.scene_points)  # float16, as the filter uses it
    data["collision_local"] = products[0]  # float16 gripper-frame coordinates, first 32 grasps
    data["collision_free"] = np.asarray(no_collision, dtype=bool)

    def benchmark_group(group):
        out = GraspNetGraspGroup()
        for grasp in group:
            out.add(
                GraspNetGrasp(
                    grasp.score,
                    grasp.width,
                    grasp.height,
                    grasp.depth,
                    grasp.rotation.reshape(9),
                    grasp.translation,
                    -1,
                )
            )
        return out

    def finish(group):
        if len(group) == 0:
            return np.zeros((0, 17))
        kept = group.nms(0.03, 30 / 180 * np.pi).sort_by_score()
        return np.asarray(kept.grasp_group_array, dtype=np.float64).reshape(-1, 17)

    data["decoded_array"] = np.asarray(
        benchmark_group(decoded).grasp_group_array, dtype=np.float64
    ).reshape(-1, 17)
    data["final_cd"] = finish(benchmark_group(decoded[no_collision]))
    data["final_raw"] = finish(benchmark_group(decoded))
    return data


# -------------------------------------------------------------------------- training


def label_file(root, scene, frame):
    return os.path.join(root, "6d_dataset", f"scene_{scene}", "grasp_labels", f"{frame}_view.npz")


def convert_labels(reference, dataset_root, camera, root):
    """Run the pristine label converter for the training frames that have no file yet.

    ``convert_grasp`` builds its dataset from a hard-coded path and reads its settings
    from module constants at import; the dataset root is substituted while it is
    imported and the constants are set before ``run`` is called. No upstream file is
    edited. Python's ``random`` is seeded per view, so the files can be regenerated.
    """
    missing = [key for key in TRAINING_FRAMES if not os.path.isfile(label_file(root, *key))]
    if not missing:
        return
    import customgraspnetAPI

    original = customgraspnetAPI.GraspNet.__init__

    def redirected(self, root, camera="kinect", split="train"):
        original(self, dataset_root, camera=camera, split=split)

    customgraspnetAPI.GraspNet.__init__ = redirected
    try:
        import convert_grasp
    finally:
        customgraspnetAPI.GraspNet.__init__ = original
    convert_grasp.graspnet_root = dataset_root
    convert_grasp.camera = camera
    convert_grasp.output_path = root
    convert_grasp.skip = False
    for scene, frame in missing:
        random.seed(scene * 1000 + frame)
        convert_grasp.annIds = [frame]
        convert_grasp.run([scene])
        if not os.path.isfile(label_file(root, scene, frame)):
            raise RuntimeError(f"the converter wrote no label file for scene {scene} view {frame}")


def record_training(args, torch, dataset, anchornet, localnet, anchors, intrinsics):
    from dataset.evaluation import anchor_output_process, detect_2d_grasp
    from dataset.pc_dataset_tools import data_process, feature_fusion, get_center_group_label
    from dataset.utils import shift_anchors
    from models import losses
    from torch.utils.data.dataloader import default_collate

    train, size = OPTIONS["train"], (OPTIONS["input_w"], OPTIONS["input_h"])
    options = argparse.Namespace(offset_d=train["offset_d"])
    data, labels = {}, {}
    anchornet.train()
    localnet.train()
    dataset.unaug()  # joint training
    dataset.train()
    dropout = {}
    hook = localnet.point_layer[2].register_forward_hook(
        lambda module, inputs, output: dropout.update(
            {"before": numpy(inputs[0]), "after": numpy(output)}
        )
    )
    seed_everything(torch, 20231017)

    def install(taps):
        install_region_taps(taps)
        taps.wrap(
            losses,
            "faster_get_local_labels",
            "local_targets",
            lambda a, k, out: (np.asarray(out[0]), np.asarray(out[1])),
        )

    with tapped(install) as taps:
        anchor_data, rgbs, depths, grasppaths = default_collate(
            [dataset[index] for index in range(len(TRAINING_FRAMES))]
        )
        points, _, _ = dataset.helper.to_scene_points(rgbs.cuda(), depths.cuda(), include_rgb=False)
        xyzs = dataset.helper.to_xyz_maps(depths.cuda())
        all_grasp_labels = [np.load(path) for path in grasppaths]
        for frame, label in enumerate(all_grasp_labels):
            for name in LABEL_FIELDS:
                labels[f"{name}_{frame}"] = np.asarray(label[name])
        x, y = anchor_data[0].cuda(), [target.cuda() for target in anchor_data[1]]
        pred_2d, features = anchornet(x)
        anchor = losses.compute_anchor_loss(
            pred_2d, y, loc_a=train["loc_a"], reg_b=train["reg_b"], cls_c=train["cls_c"]
        )
        data.update(
            image=numpy(x),
            depth=depths.numpy(),
            sample_index=np.stack(taps.take("sample_index")),
            points=numpy(points),
            target_heatmap=numpy(y[0])[:, 0],
            target_angle=numpy(y[1]),
            target_angle_offset=numpy(y[2]),
            target_depth_offset=numpy(y[3]),
            target_width_offset=numpy(y[4]),
            heatmap_logits=numpy(pred_2d[0])[:, 0],
            angle_logits=numpy(pred_2d[1]),
            angle_offset_raw=numpy(pred_2d[2]),
            depth_offset_raw=numpy(pred_2d[3]),
            width_offset_raw=numpy(pred_2d[4]),
            features=numpy(features),
            loss_heatmap=numpy(anchor["losses"]["loc_map_loss"]),
            loss_regression=numpy(anchor["losses"]["reg_loss"]),
            loss_angle=numpy(anchor["losses"]["cls_loss"]),
        )
        loc_maps, theta_cls, theta_offset, depth_offset, width_offset = anchor_output_process(
            *pred_2d, sigma=OPTIONS["sigma"]
        )
        data["heatmap"] = loc_maps
        rect_ggs = []
        for frame in range(train["batch_size"]):
            rect_ggs.append(
                detect_2d_grasp(
                    loc_maps[frame],
                    theta_cls[frame],
                    theta_offset[frame],
                    depth_offset[frame],
                    width_offset[frame],
                    ratio=OPTIONS["ratio"],
                    anchor_k=OPTIONS["anchor_k"],
                    anchor_w=OPTIONS["anchor_w"],
                    anchor_z=OPTIONS["anchor_z"],
                    mask_thre=0,
                    center_num=train["center_num"],
                    grid_size=OPTIONS["grid_size"],
                    grasp_nms=OPTIONS["grid_size"],
                )
            )
            data.update(rect_arrays(rect_ggs[-1], f"proposal_{frame}_"))
        taps.take("candidates")
        points_all = feature_fusion(points, features, xyzs)
        data["knn_index"] = taps.take("knn_index")[0]
        taps.take("randint"), taps.take("randn")
        pc_group, valid_local_centers = data_process(
            points_all,
            depths.cuda(),
            rect_ggs,
            train["center_num"],
            OPTIONS["group_num"],
            size,
            is_training=False,  # sic: the training loop passes False
        )
        lifted = taps.take("lifted")[0]
        for frame in range(train["batch_size"]):
            data[f"lifted_centre_{frame}"] = lifted[frame]
            data[f"region_centre_{frame}"] = numpy(valid_local_centers[frame]).astype(np.float64)
            data.update(rect_arrays(rect_ggs[frame], f"region_{frame}_"))
        data.update(region_record(taps))
        grasp_info = np.zeros((0, 3), dtype=np.float32)
        for frame in range(train["batch_size"]):
            current = np.vstack(
                [
                    rect_ggs[frame].thetas[None],
                    rect_ggs[frame].widths[None],
                    rect_ggs[frame].depths[None],
                ]
            )
            grasp_info = np.vstack([grasp_info, current.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device="cuda")
        _, pred_view, offset = localnet(pc_group, grasp_info)
        gg_labels, total_labels = get_center_group_label(
            valid_local_centers, all_grasp_labels, train["local_grasp_num"]
        )
        matched, matched_offsets = ragged([numpy(g) for g in gg_labels], np.float32)
        multi_cls_loss, offset_loss = losses.compute_multicls_loss(
            pred_view, offset, gg_labels, grasp_info, anchors, options
        )
        multi_labels, offset_labels = taps.take("local_targets")[0]
        data.update(
            region=numpy(pc_group),
            grasp_info=numpy(grasp_info),
            dropout_before=dropout["before"],
            dropout_after=dropout["after"],
            anchor_logits=numpy(pred_view),
            offset_raw=numpy(offset),
            matched_labels=matched,
            matched_offsets=matched_offsets,
            total_labels=numpy(total_labels),
            target_anchor=multi_labels,
            target_offset=offset_labels,  # metres; the loss divides by 0.02
            loss_anchor=numpy(multi_cls_loss),
            loss_offset=numpy(offset_loss),
        )
        loss = anchor["loss"] + multi_cls_loss + offset_loss
        data["loss"] = numpy(loss)
        anchornet.zero_grad()
        localnet.zero_grad()
        loss.backward()
        for owner, net in (("anchor", anchornet), ("local", localnet)):
            parameters = dict(net.named_parameters())
            for name in GRADIENT_TAPS[owner]:
                data[f"grad_{owner}.{name}"] = numpy(parameters[name].grad)
        shifted = shift_anchors(total_labels.cpu(), {k: v.clone() for k, v in anchors.items()})
        data["shifted_gamma"] = numpy(shifted["gamma"]).astype(np.float64)
        data["shifted_beta"] = numpy(shifted["beta"]).astype(np.float64)
    hook.remove()
    data["intrinsics"] = np.asarray(intrinsics, dtype=np.float64)
    return data, labels


# ------------------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", required=True, help="pristine HGGD checkout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--camera", default="realsense", choices=sorted(INFERENCE_FRAMES))
    parser.add_argument("--out", required=True)
    parser.add_argument("--training", action="store_true", help="record the training batch")
    parser.add_argument(
        "--label-root",
        help="with --training: directory holding 6d_dataset/scene_<id>/grasp_labels/"
        "<ann>_view.npz for the training frames (default: <out>/upstream_labels/<camera>, "
        "filled by the upstream converter when files are missing)",
    )
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    sys.dont_write_bytecode = True
    os.makedirs(args.out, exist_ok=True)
    out = os.path.abspath(args.out)
    label_root = None
    if args.training:
        label_root = os.path.abspath(
            args.label_root or os.path.join(out, "upstream_labels", args.camera)
        )
    os.chdir(tempfile.mkdtemp(prefix="hggd-oracle-"))  # upstream writes nothing, but be sure
    sys.path.insert(0, args.reference)
    import torch
    import torchvision

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    intrinsics = select_camera(args.camera)

    manifest = {
        "reference": args.reference,
        "reference_commit": reference_commit(args.reference),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "camera": args.camera,
        "options": OPTIONS,
        "intrinsics": np.asarray(intrinsics).tolist(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0),
        "tf32": False,
        "inference": [],
        "training": None,
    }
    manifest_file = os.path.join(out, f"manifest_{args.camera}.json")
    if os.path.isfile(manifest_file):  # keep what this run does not record again
        with open(manifest_file) as handle:
            previous = json.load(handle)
        if previous.get("checkpoint_sha256") == manifest["checkpoint_sha256"]:
            manifest["inference"] = previous.get("inference", []) if args.skip_inference else []
            manifest["training"] = previous.get("training")
    tic = time.time()
    if not args.skip_inference:
        frames = INFERENCE_FRAMES[args.camera]
        scenes = sorted({scene for scene, _ in frames})
        dataset, anchornet, localnet, anchors = build(args, scenes, None)
        dataset.unaug()
        dataset.eval()
        anchornet.eval()
        localnet.eval()
        with torch.no_grad():
            for scene, frame in frames:
                index = scenes.index(scene) * 256 + frame
                data = record_inference(
                    args, torch, dataset, anchornet, localnet, anchors, index, (scene, frame)
                )
                data["intrinsics"] = np.asarray(intrinsics, dtype=np.float64)
                data["anchor_gamma"] = numpy(anchors["gamma"]).astype(np.float64)
                data["anchor_beta"] = numpy(anchors["beta"]).astype(np.float64)
                name = f"infer_{args.camera}_{scene:04d}_{frame:04d}.npz"
                np.savez_compressed(os.path.join(out, name), **data)
                manifest["inference"].append(
                    {
                        "file": name,
                        "scene": scene,
                        "frame": frame,
                        "proposals": len(data["proposal_theta"]),
                        "regions": len(data["region"]),
                        "decoded": len(data["decoded_score"]),
                        "collision_free": int(data["collision_free"].sum()),
                        "final_cd": len(data["final_cd"]),
                        "final_raw": len(data["final_raw"]),
                    }
                )
                print(manifest["inference"][-1], flush=True)
    if label_root is not None:
        convert_labels(args.reference, args.dataset_root, args.camera, label_root)
        scenes = [scene for scene, _ in TRAINING_FRAMES]
        if len(set(scenes)) != 2 or scenes != sorted(scenes):
            raise RuntimeError("TRAINING_FRAMES must list two scenes, grouped")
        unique = sorted(set(scenes))
        dataset, anchornet, localnet, anchors = build(args, unique, label_root)
        order = [unique.index(scene) * 256 + frame for scene, frame in TRAINING_FRAMES]

        class Batch:
            """The four training frames, addressed 0..3."""

            helper = dataset.helper
            unaug, train = dataset.unaug, dataset.train

            def __getitem__(self, position):
                return dataset[order[position]]

        data, labels = record_training(
            args, torch, Batch(), anchornet, localnet, anchors, intrinsics
        )
        data["anchor_gamma"] = numpy(anchors["gamma"]).astype(np.float64)
        data["anchor_beta"] = numpy(anchors["beta"]).astype(np.float64)
        np.savez_compressed(os.path.join(out, f"train_{args.camera}.npz"), **data)
        np.savez_compressed(os.path.join(out, f"train_{args.camera}_labels.npz"), **labels)
        manifest["training"] = {
            "file": f"train_{args.camera}.npz",
            "labels": f"train_{args.camera}_labels.npz",
            "frames": [list(frame) for frame in TRAINING_FRAMES],
            "regions": len(data["region"]),
            "loss": float(data["loss"]),
        }
        print(manifest["training"], flush=True)
    manifest["seconds"] = round(time.time() - tic, 1)
    with open(manifest_file, "w") as handle:
        json.dump(manifest, handle, indent=1)


if __name__ == "__main__":
    main()
