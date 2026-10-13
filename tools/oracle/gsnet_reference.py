"""Record what the reference GSNet implementation produces on this machine.

Runs the pristine upstream code (graspnet/graspness_unofficial) with the author's
checkpoint over the GraspNet-1Billion test splits and scores the dumps with the official
graspnetAPI. The numbers are the ``author-ckpt`` reference that GraspPanda's own model
and evaluator are compared against.

This script is an oracle: it is executed with an interpreter that has the upstream
dependencies (torch 1.9, MinkowskiEngine 0.5.4, the compiled pointnet2 and knn
extensions, graspnetAPI), never with GraspPanda's environment, and it imports upstream
modules read-only (run it with ``-B`` so no bytecode is written next to them).

Point sampling is made reproducible: ``numpy.random.seed(scene * 1000 + frame)`` is
called before each frame is loaded, so any pipeline can draw the same 15000 points.

    $GRASPPANDA_ORACLE_PYTHON -B tools/oracle/gsnet_reference.py \
        --upstream /path/to/graspness_unofficial --checkpoint checkpoint-rs.tar \
        --dataset-root /path/to/GraspNet-1B --camera realsense --out /path/to/out
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SPLITS = ("test_seen", "test_similar", "test_novel")


def seed_for(scene_name, frame):
    return int(scene_name.split("_")[1]) * 1000 + frame


def infer(args, split):
    import torch
    from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
    from graspnetAPI import GraspGroup
    from models.graspnet import GraspNet, pred_decode
    from collision_detector import ModelFreeCollisionDetector  # upstream imports it bare

    dataset = GraspNetDataset(
        args.dataset_root,
        split=split,
        camera=args.camera,
        num_points=15000,
        voxel_size=0.005,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    scenes = dataset.scene_list()
    net = GraspNet(seed_feat_dim=512, is_training=False).cuda()
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print(
        "%s: %d frames, checkpoint epoch %s" % (split, len(dataset), checkpoint.get("epoch")),
        flush=True,
    )
    tic = time.time()
    done = 0
    for index in range(len(dataset)):
        frame = index % 256
        paths = {
            variant: os.path.join(args.out, variant, scenes[index], args.camera, "%04d.npy" % frame)
            for variant in ("raw", "cd")
        }
        if all(os.path.exists(p) for p in paths.values()):
            continue
        np.random.seed(seed_for(scenes[index], frame))
        batch = minkowski_collate_fn([dataset[index]])
        for key in batch:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            preds = pred_decode(net(batch))[0].detach().cpu().numpy()
        grasps = GraspGroup(preds)
        cloud = dataset.get_data(index, return_raw_cloud=True)
        detector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        collision = detector.detect(grasps, approach_dist=0.05, collision_thresh=0.01)
        for variant, group in (("raw", grasps), ("cd", grasps[~collision])):
            os.makedirs(os.path.dirname(paths[variant]), exist_ok=True)
            temporary = paths[variant] + ".tmp.npy"
            np.save(temporary, group.grasp_group_array)
            os.replace(temporary, paths[variant])
        done += 1
        if done % 500 == 0:
            print(
                "  %s %d/%d  %.3f s/frame"
                % (split, index + 1, len(dataset), (time.time() - tic) / done),
                flush=True,
            )


def evaluate(args, split, variant):
    from graspnetAPI import GraspNetEval

    target = os.path.join(args.out, "official_%s_%s_%s.npy" % (args.camera, split, variant))
    if os.path.exists(target):
        res = np.load(target)
    else:
        evaluator = GraspNetEval(root=args.dataset_root, camera=args.camera, split=split)
        # Call the scene loop directly: the convenience methods (eval_seen, ...) of an
        # installed copy may have been edited to cover fewer scenes.
        first, last = {
            "test_seen": (100, 130),
            "test_similar": (130, 160),
            "test_novel": (160, 190),
        }[split]
        tic = time.time()
        res = evaluator.parallel_eval_scenes(
            scene_ids=list(range(first, last)),
            dump_folder=os.path.join(args.out, variant),
            proc=args.proc,
        )
        res = np.asarray(res)
        np.save(target, res)
        print("  official eval %s/%s took %.0f s" % (split, variant, time.time() - tic), flush=True)
    return {
        "AP": float(res.mean() * 100),
        "AP0.8": float(res[..., 3].mean() * 100),
        "AP0.4": float(res[..., 1].mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--out", required=True)
    parser.add_argument("--proc", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.upstream)
    for sub in (
        "pointnet2",
        "utils",
        "models",
        "dataset",
    ):  # the upstream modules import each other bare
        sys.path.append(os.path.join(args.upstream, sub))
    for split in args.splits:
        infer(args, split)
    if args.skip_eval:
        return
    summary = {}
    for split in args.splits:
        for variant in ("raw", "cd"):
            summary["%s/%s" % (split, variant)] = evaluate(args, split, variant)
            print(split, variant, summary["%s/%s" % (split, variant)], flush=True)
            with open(os.path.join(args.out, "summary_%s.json" % args.camera), "w") as handle:
                json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
