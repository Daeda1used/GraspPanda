"""Record what the reference EconomicGrasp implementation produces on this machine.

Runs the pristine upstream code (iSEE-Laboratory/EconomicGrasp at the pinned commit)
with an author checkpoint over chosen GraspNet-1Billion test frames and writes the
decoded grasps in the layout of the official tools, once as the model emits them
(``raw``) and once after the model-free collision filter of the upstream test script
(``cd``: collision threshold 0.01 m, voxel 0.005 m). The dumps are scored afterwards
with GraspPanda's evaluator; the numbers are the ``author-ckpt`` reference that
GraspPanda's own model is compared against, and the first thing to look at before
building on a checkpoint: is it healthy in the code it was trained with?

This script is an oracle: it is executed with an interpreter that has the upstream
dependencies (torch 1.9, MinkowskiEngine 0.5.4, the compiled pointnet2 and knn
extensions, graspnetAPI), never with GraspPanda's environment. It imports upstream
modules read-only; run it with ``-B`` and from a scratch directory, on a pristine tree
(``git archive <commit> | tar -x -C <scratch>``)::

    $GRASPPANDA_ORACLE_PYTHON -B tools/oracle/economicgrasp_reference.py \
        --upstream /scratch/EconomicGrasp --checkpoint checkpoint-kinect.tar \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --camera kinect \
        --scenes 100-104 --frame-step 64 --out /scratch/economicgrasp_reference

All 90 test scenes (``--scenes 100-189 --frame-step 1``) are 23 040 frames; the sample
above ran at 1.7 s per frame on a busy development machine, which makes about 11 hours.

Point sampling is made reproducible: ``numpy.random.seed(scene * 1000 + frame)`` is
called before each frame is loaded. ``--save-inputs`` additionally stores the sampled
cloud of every frame, so that another implementation can be run on the very same points.

Precision. torch 1.9 enables TF32 on Ampere and newer GPUs, which moves activations by
about 1e-2 and with them a few seeds. ``--tf32 off`` (the default here) records in true
float32; ``--tf32 default`` leaves the flags as upstream users get them.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def allow_integer_floor(torch):
    """Make ``torch.floor`` accept integer tensors, as the torch releases upstream targets do.

    The upstream forward floors voxel coordinates that are already integers. torch 2.x
    (the upstream README installs 2.5.1) defines that as the identity; torch 1.9 raises.
    The upstream sources stay untouched: only this process's ``torch.floor`` is extended.
    Returns whether the extension was needed.
    """
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


def expand(ranges):
    """``"100-104,130"`` -> [100, 101, 102, 103, 104, 130]."""
    scenes = []
    for part in ranges.split(","):
        first, _, last = part.partition("-")
        scenes.extend(range(int(first), int(last or first) + 1))
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", required=True, help="pristine EconomicGrasp tree")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--camera", default="kinect", choices=["realsense", "kinect"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", default="100-104", help="test scenes, e.g. 100-104,130")
    parser.add_argument("--frame-step", type=int, default=64)
    parser.add_argument("--tf32", default="off", choices=["off", "default"])
    parser.add_argument("--save-inputs", action="store_true", help="store the sampled clouds")
    args = parser.parse_args()

    upstream = os.path.abspath(args.upstream)
    checkpoint_path = os.path.abspath(args.checkpoint)
    root = os.path.abspath(args.dataset_root)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # The upstream modules parse the command line when they are imported.
    sys.argv = [sys.argv[0], "--dataset_root", root, "--camera", args.camera]
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(upstream, "libs", "pointnet2"))  # bare `import pytorch_utils`
    sys.path.insert(0, upstream)

    import torch
    from dataset.graspnet_dataset import GraspNetDataset, collate_fn
    from graspnetAPI import GraspGroup
    from models.economicgrasp import economicgrasp, pred_decode
    from utils.arguments import cfgs
    from utils.collision_detector import ModelFreeCollisionDetector

    floor_extended = allow_integer_floor(torch)
    defaults = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    if args.tf32 == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dataset = GraspNetDataset(
        root,
        split="test",
        camera=args.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    net = economicgrasp(seed_feat_dim=512, is_training=False).cuda()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    frames = []
    tic = time.time()
    for scene in expand(args.scenes):
        for frame in range(0, 256, args.frame_step):
            index = (scene - 100) * 256 + frame
            name = "scene_%04d" % scene
            assert dataset.scenename[index] == name and dataset.frameid[index] == frame
            np.random.seed(scene * 1000 + frame)
            sample = dataset[index]
            batch = collate_fn([sample])
            for key in batch:
                batch[key] = batch[key].cuda()
            with torch.no_grad():
                end_points = net(batch)
                preds = pred_decode(end_points)[0].detach().cpu().numpy()
                angle_class = end_points["grasp_angle_pred"][0].argmax(dim=0)
                depth_class = end_points["grasp_depth_pred"][0].argmax(dim=0)
            grasps = GraspGroup(preds)
            cloud, _ = dataset.get_data(index, return_raw_cloud=True)
            detector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
            collision = detector.detect(grasps, approach_dist=0.05, collision_thresh=0.01)
            variants = {"raw": grasps.grasp_group_array, "cd": grasps[~collision].grasp_group_array}
            if args.save_inputs:
                variants["inputs"] = sample["point_clouds"]
            for variant, array in variants.items():
                path = os.path.join(out, variant, name, args.camera, "%04d.npy" % frame)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path + ".tmp.npy", array)
                os.replace(path + ".tmp.npy", path)
            frames.append(
                {
                    "scene": scene,
                    "frame": frame,
                    "grasps": len(preds),
                    "collisions": int(collision.sum()),
                    # how often the never-supervised extra class won the decoder arg-max
                    "extra_angle_class": int((angle_class == cfgs.num_angle).sum()),
                    "extra_depth_class": int((depth_class == cfgs.num_depth).sum()),
                }
            )
            print(json.dumps(frames[-1]), flush=True)

    manifest = {
        "upstream": upstream,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_root": root,
        "camera": args.camera,
        "num_points": cfgs.num_point,
        "collision_filter": {"voxel_size": cfgs.voxel_size, "approach": 0.05, "threshold": 0.01},
        "sampling_seed": "numpy.random.seed(scene * 1000 + frame) before each frame",
        "tf32": args.tf32,
        "tf32_defaults": defaults,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
        "torch_floor_extended_to_integers": floor_extended,
        "seconds": round(time.time() - tic, 1),
        "frames": frames,
    }
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("wrote", os.path.join(out, "manifest.json"))


if __name__ == "__main__":
    main()
