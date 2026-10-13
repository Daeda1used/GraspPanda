"""Render dumped grasp predictions on the RGB-D frame they belong to.

Reads a prediction file in the GraspNet layout (as written by `grasppanda test` or
`grasppanda infer`), back-projects the frame's depth image with its colours, and draws
the highest-scoring grasps: a still image, or a short swing animation.

    python tools/render_predictions.py --predictions <run>/predictions/cd \
        --scene 100 --frame 0 --camera realsense --out demo.webp --frames 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

from grasppanda.evaluation import load_prediction
from grasppanda.structures import SampleKey
from grasppanda.utils.paths import dataset_root
from grasppanda.visualization import Camera, render


def load_frame(
    root: Path, scene: int, camera: str, frame: int, max_depth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Coloured point cloud of one GraspNet-1Billion frame, in the camera frame."""
    base = root / "scenes" / f"scene_{scene:04d}" / camera
    depth = np.asarray(Image.open(base / "depth" / f"{frame:04d}.png")).astype(np.float64)
    color = np.asarray(Image.open(base / "rgb" / f"{frame:04d}.png"))
    meta = scipy.io.loadmat(base / "meta" / f"{frame:04d}.mat")
    intrinsics, scale = meta["intrinsic_matrix"], float(meta["factor_depth"].squeeze())
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    z = depth / scale
    keep = (z > 0) & (z < max_depth)
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.stack([x, y, z], axis=-1)[keep], color[keep]


def swing(points: np.ndarray, frames: int, degrees: float, lift: float) -> list[Camera]:
    """Cameras swinging about the scene centre, raised a little above the sensor's view."""
    target = np.median(points, axis=0)
    up = np.array([0.0, -1.0, 0.0])
    offset = -target * 0.92 + up * lift * np.linalg.norm(target)
    cameras = []
    for index in range(frames):
        angle = np.radians(degrees) * np.sin(2.0 * np.pi * index / frames)
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = offset * cos + np.cross(up, offset) * sin + up * (up @ offset) * (1.0 - cos)
        cameras.append(Camera(target + rotated, target, up, fov_degrees=42.0))
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions", required=True, help="directory holding scene_XXXX/<camera>/NNNN.npy"
    )
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument(
        "--root", help="GraspNet-1Billion root (default: resolved like the dataset reader)"
    )
    parser.add_argument(
        "--out", required=True, help=".png for a still; .webp or .gif for an animation"
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--frames", type=int, default=1, help="animation frames (1 = still image)")
    parser.add_argument("--size", type=int, nargs=2, default=(800, 500), metavar=("W", "H"))
    parser.add_argument("--max-depth", type=float, default=0.8)
    args = parser.parse_args()

    root = dataset_root("graspnet1b", args.root)
    points, colors = load_frame(root, args.scene, args.camera, args.frame, args.max_depth)
    grasps = load_prediction(
        args.predictions, SampleKey("graspnet1b", args.scene, args.camera, args.frame)
    )
    cameras = swing(points, max(args.frames, 1), degrees=26.0, lift=0.22)
    images = [
        render(points, colors, grasps, camera=camera, size=tuple(args.size), top_k=args.top_k)
        for camera in (cameras if args.frames > 1 else cameras[:1])
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(out)
    else:
        images[0].save(
            out, save_all=True, append_images=images[1:], duration=90, loop=0, quality=72
        )
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(images)} frame(s), {len(grasps)} grasps in the file)"
    )


if __name__ == "__main__":
    main()
