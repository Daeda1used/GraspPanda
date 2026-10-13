"""Run the pristine HGGD test path with an author checkpoint and dump its grasps.

This script is an oracle: it is executed with an interpreter that has the upstream
dependencies (torch 1.11, pytorch3d, numba, cupoch, the bundled ``customgraspnetAPI``),
never with GraspPanda's environment, and it imports the upstream modules read-only (run
it with ``-B`` so no bytecode is written next to them).

It follows ``test_graspnet.py`` call by call with the options of ``test_graspnet.sh``.
Two things differ, both stated here because they are the reason this script exists:

* The upstream test loop opens a per-view label file (``6d_dataset/scene_<id>/
  grasp_labels/<ann>_view.npz``) for every test frame, only to compute statistics that
  never reach the dump. Those files are a separate download. Here the dataset's label
  loader returns an empty label set, so no label file is needed and the image inputs
  are still produced by the unmodified upstream loader.
* The random draws (scene-point sub-sampling, the start point and jitter of the region
  sampling) are seeded per frame with ``scene * 1000 + frame + seed_offset`` instead of
  once per run, so a single frame can be reproduced and the spread over draws measured.

For every frame three dumps are written, all after the script's NMS (0.03 m, 30°) and
sorted by score, as the script does:

``cd``     the script's output: collision filter (which also rejects grasps with at most
           two points between the fingers), then NMS.
``raw``    the same without the collision filter.
``raw_no_nms``  every decoded grasp.

A frame without 2-D proposals gets an empty ``(0, 17)`` dump (upstream writes no file).

    /path/to/envs/HGGD/bin/python -B tools/oracle/hggd_reference.py \
        --upstream /path/to/HGGD --checkpoint checkpoint-realsense.tar \
        --dataset-root /path/to/GraspNet-1B --camera realsense \
        --scenes 100 101 102 103 104 --frame-step 64 --out /path/to/out
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

OPTIONS = {  # test_graspnet.sh
    "center_num": 48,
    "anchor_num": 7,
    "anchor_k": 6,
    "anchor_w": 50.0,
    "anchor_z": 20.0,
    "grid_size": 8,
    "all_points_num": 25600,
    "group_num": 512,
    "local_k": 10,
    "ratio": 8,
    "input_h": 360,
    "input_w": 640,
    "heatmap_thres": 0.01,
    "sigma": 10,
    "grasp_count": 5000,
}
EMPTY_LABELS = {
    name: np.zeros((0, 2) if name == "centers_2d" else (0,))
    for name in (
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
}


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_camera(camera):
    """Upstream fixes the camera in ``dataset/config.py``; a user edits that file."""
    from dataset import config

    config.camera = camera
    config.get_camera_intrinsic.__defaults__ = (camera,)


def build(args):
    import torch

    select_camera(args.camera)
    from dataset.graspnet_dataset import GraspnetPointDataset
    from models.anchornet import AnchorGraspNet
    from models.localgraspnet import PointMultiGraspNet

    class LabelFreeDataset(GraspnetPointDataset):
        def load_grasp_labels(self, index):
            return EMPTY_LABELS

    dataset = LabelFreeDataset(
        OPTIONS["all_points_num"],
        "/nonexistent-hggd-labels",
        args.dataset_root,
        list(args.scenes),
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
    dataset.unaug()
    dataset.eval()
    anchornet = AnchorGraspNet(in_dim=4, ratio=OPTIONS["ratio"], anchor_k=OPTIONS["anchor_k"])
    localnet = PointMultiGraspNet(info_size=3, k_cls=OPTIONS["anchor_num"] ** 2)
    anchornet, localnet = anchornet.cuda(), localnet.cuda()
    checkpoint = torch.load(args.checkpoint)
    anchornet.load_state_dict(checkpoint["anchor"])
    localnet.load_state_dict(checkpoint["local"])
    anchors = {"gamma": checkpoint["gamma"], "beta": checkpoint["beta"]}
    anchornet.eval()
    localnet.eval()
    return dataset, anchornet, localnet, anchors


def to_array(group):
    """A ``customgraspnetAPI`` group as the ``(G, 17)`` array the benchmark reads."""
    return np.asarray(group.grasp_group_array, dtype=np.float64).reshape(-1, 17)


def run_frame(torch, dataset, anchornet, localnet, anchors, index):
    """One iteration of the upstream test loop; returns the three dumps and counts."""
    from customgraspnetAPI import Grasp as GraspNetGrasp
    from customgraspnetAPI import GraspGroup as GraspNetGraspGroup
    from dataset.evaluation import (
        anchor_output_process,
        collision_detect,
        detect_2d_grasp,
        detect_6d_grasp_multi,
    )
    from dataset.pc_dataset_tools import data_process, feature_fusion
    from torch.utils.data.dataloader import default_collate

    size = (OPTIONS["input_w"], OPTIONS["input_h"])
    anchor_data, rgb, ori_depth, _ = default_collate([dataset[index]])
    depth = torch.from_numpy(ori_depth.numpy().squeeze()).float()[None]
    view_points, _, _ = dataset.helper.to_scene_points(rgb.cuda(), depth.cuda(), include_rgb=True)
    points = view_points[..., :3]
    xyzs = dataset.helper.to_xyz_maps(depth.cuda())
    pred_2d, perpoint_features = anchornet(anchor_data[0].cuda())
    loc_map, cls_mask, theta_offset, height_offset, width_offset = anchor_output_process(
        *pred_2d, sigma=OPTIONS["sigma"]
    )
    rect_gg = detect_2d_grasp(
        loc_map,
        cls_mask,
        theta_offset,
        height_offset,
        width_offset,
        ratio=OPTIONS["ratio"],
        anchor_k=OPTIONS["anchor_k"],
        anchor_w=OPTIONS["anchor_w"],
        anchor_z=OPTIONS["anchor_z"],
        mask_thre=OPTIONS["heatmap_thres"],
        center_num=OPTIONS["center_num"],
        grid_size=OPTIONS["grid_size"],
        grasp_nms=OPTIONS["grid_size"],
        reduce="max",
    )
    empty = np.zeros((0, 17))
    counts = {"proposals": int(rect_gg.size), "regions": 0, "decoded": 0, "collision_free": 0}
    if rect_gg.size == 0:
        return {"cd": empty, "raw": empty, "raw_no_nms": empty}, counts
    points_all = feature_fusion(points, perpoint_features, xyzs)
    rect_ggs = [rect_gg]
    pc_group, valid_local_centers = data_process(
        points_all,
        depth.cuda(),
        rect_ggs,
        OPTIONS["center_num"],
        OPTIONS["group_num"],
        size,
        min_points=32,
        is_training=False,
    )
    rect_gg = rect_ggs[0]
    points_all = points_all.squeeze()
    counts["regions"] = int(pc_group.shape[0])
    if pc_group.shape[0] == 0:  # upstream would fail here; an empty frame is the result
        return {"cd": empty, "raw": empty, "raw_no_nms": empty}, counts
    grasp_info = np.vstack([rect_gg.thetas[None], rect_gg.widths[None], rect_gg.depths[None]]).T
    grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device="cuda")
    _, pred, offset = localnet(pc_group, grasp_info)
    _, pred_rect_gg = detect_6d_grasp_multi(
        rect_gg, pred, offset, valid_local_centers, size, anchors, k=OPTIONS["local_k"]
    )
    if len(pred_rect_gg) == 0:
        return {"cd": empty, "raw": empty, "raw_no_nms": empty}, counts
    decoded = pred_rect_gg.to_6d_grasp_group(depth=0.02)
    filtered, valid_mask = collision_detect(points_all, decoded, mode="graspnet")
    counts["decoded"], counts["collision_free"] = len(decoded), int(valid_mask.sum())

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
            return empty
        return to_array(group.nms(0.03, 30 / 180 * np.pi).sort_by_score())

    everything = benchmark_group(decoded)
    dumps = {
        "cd": finish(benchmark_group(filtered)),
        "raw": finish(everything),
        "raw_no_nms": to_array(everything.sort_by_score()),
    }
    return dumps, counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--camera", default="realsense", choices=("realsense", "kinect"))
    parser.add_argument("--scenes", type=int, nargs="+", required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.upstream)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dataset, anchornet, localnet, anchors = build(args)
    summary = {
        "options": OPTIONS,
        "camera": args.camera,
        "seed_offset": args.seed_offset,
        "frames": {},
    }
    tic = time.time()
    with torch.no_grad():
        for position, scene in enumerate(args.scenes):
            for frame in range(0, 256, args.frame_step):
                seed_everything(torch, scene * 1000 + frame + args.seed_offset)
                dumps, counts = run_frame(
                    torch, dataset, anchornet, localnet, anchors, position * 256 + frame
                )
                for variant, array in dumps.items():
                    directory = os.path.join(args.out, variant, "scene_%04d" % scene, args.camera)
                    os.makedirs(directory, exist_ok=True)
                    path = os.path.join(directory, "%04d.npy" % frame)
                    np.save(path + ".tmp.npy", array.astype(np.float32))
                    os.replace(path + ".tmp.npy", path)
                counts.update({name: len(array) for name, array in dumps.items()})
                summary["frames"]["%04d/%04d" % (scene, frame)] = counts
                print("scene %d frame %d: %s" % (scene, frame, counts), flush=True)
    summary["seconds"] = round(time.time() - tic, 1)
    with open(os.path.join(args.out, "summary_%s.json" % args.camera), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
