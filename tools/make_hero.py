"""The hero picture of the README and the documentation: grasps on a GraspNet-1Billion scene.

    grasppanda demo economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \\
        --weights economicgrasp_realsense.pth --frame graspnet1b:scene_0163/realsense/0000 \\
        --collision-filter --post nms --out work_dirs/hero
    python -m tools.make_hero work_dirs/hero/grasps.npy --scene 163 --out resources/hero.webp

The grasps are those ``demo`` wrote, best first; twelve of them are drawn, each at least
5 cm from a better one. The scene is the frame's own depth and colour image, in a frame of
the table fitted to the cloud (z up). Scene 163 is of the ``test_novel`` split: none of its
objects is in the training scenes. The picture is drawn with NumPy and Pillow only: points
splatted into a depth buffer, the table behind them as a plane (it fills the holes a depth
sensor leaves beside objects), and each gripper as four shaded boxes in the proportions of
``graspnetAPI``'s drawing, fingers a little thicker so they read at README size. The
background is transparent, for light and dark pages alike.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

# graspnetAPI's gripper drawing (plot_gripper_pro_max), metres; the fingers are drawn thicker
FINGER, HEIGHT, DEPTH_BASE, TAIL = 0.006, 0.006, 0.02, 0.04
#: Best grasp first: amber to coral, readable on the green table and on the objects.
RAMP = np.array([[255, 204, 51], [255, 140, 66], [240, 84, 79]], dtype=np.float64)
LIGHT = np.array([-0.35, -0.55, 0.76])


def load_scene(root: Path, scene: int, frame: int, camera: str = "realsense"):
    """Points of the frame in a frame of the table (z up, metres), their colours, and the
    transform from the camera to that frame. The table is the dominant plane of the cloud;
    its in-plane axes follow the board's edges."""
    folder = root / "scenes" / f"scene_{scene:04d}" / camera
    depth = np.asarray(Image.open(folder / "depth" / f"{frame:04d}.png"), dtype=np.float64) / 1000
    rgb = np.asarray(Image.open(folder / "rgb" / f"{frame:04d}.png").convert("RGB"))
    k = np.load(folder / "camK.npy")
    v, u = np.nonzero((depth > 0) & (depth < 1.5))
    z = depth[v, u]
    points = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], 1)
    normal, offset = table_plane(points)
    if offset < 0:  # z up: the camera, at the origin, above the board
        normal, offset = -normal, -offset
    on_board = np.abs(points @ normal + offset) < 0.005
    centre = points[on_board].mean(0)
    flat = points[on_board] - centre
    flat -= np.outer(flat @ normal, normal)
    axes = np.linalg.svd(flat[:: max(1, len(flat) // 20000)], full_matrices=False)[2]
    x_axis = axes[0] - (axes[0] @ normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack([x_axis, np.cross(normal, x_axis), normal])  # rows: table axes
    table_from_camera = np.eye(4)
    table_from_camera[:3, :3] = rotation
    table_from_camera[:3, 3] = -rotation @ centre
    moved = points @ rotation.T + table_from_camera[:3, 3]
    return moved, rgb[v, u].astype(np.float64), table_from_camera


def table_plane(points: np.ndarray, rounds: int = 300, tolerance: float = 0.004):
    """``(normal, offset)`` of the plane with the most points within ``tolerance``."""
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(len(points), 60000), replace=False)]
    best, best_count = None, -1
    for _ in range(rounds):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        count = int((np.abs((sample - a) @ normal) < tolerance).sum())
        if count > best_count:
            best, best_count = (normal, -normal @ a), count
    normal, offset = best
    inliers = sample[np.abs(sample @ normal + offset) < tolerance]
    centre = inliers.mean(0)
    normal = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
    return normal, -normal @ centre


def gripper_boxes(width: float, depth: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four boxes of a parallel-jaw gripper in its own frame (x approach, y closing),
    as (lower corner, upper corner)."""
    half, back = width / 2.0, -DEPTH_BASE - FINGER
    z = (-HEIGHT / 2.0, HEIGHT / 2.0)
    return [
        (np.array([back, -half - FINGER, z[0]]), np.array([depth, -half, z[1]])),
        (np.array([back, half, z[0]]), np.array([depth, half + FINGER, z[1]])),
        (np.array([back, -half - FINGER, z[0]]), np.array([-DEPTH_BASE, half + FINGER, z[1]])),
        (np.array([back - TAIL, -FINGER / 2, z[0]]), np.array([back, FINGER / 2, z[1]])),
    ]


def box_triangles(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Twelve triangles ``(12, 3, 3)`` of an axis-aligned box and their outward normals."""
    c = np.array(
        [
            [
                lower[0] if i & 1 == 0 else upper[0],
                lower[1] if i & 2 == 0 else upper[1],
                lower[2] if i & 4 == 0 else upper[2],
            ]
            for i in range(8)
        ]
    )
    faces = [  # corner indices, counter-clockwise seen from outside, and the normal
        ((0, 2, 6, 4), (-1, 0, 0)),
        ((1, 5, 7, 3), (1, 0, 0)),
        ((0, 4, 5, 1), (0, -1, 0)),
        ((2, 3, 7, 6), (0, 1, 0)),
        ((0, 1, 3, 2), (0, 0, -1)),
        ((4, 6, 7, 5), (0, 0, 1)),
    ]
    triangles, normals = [], []
    for (a, b, cc, d), normal in faces:
        triangles += [c[[a, b, cc]], c[[a, cc, d]]]
        normals += [normal, normal]
    return np.array(triangles), np.array(normals, dtype=np.float64)


class View:
    """A pinhole camera at ``eye`` looking at ``target``, z up."""

    def __init__(self, eye, target, size, fov=32.0):
        self.eye, self.size = np.asarray(eye, float), size
        forward = np.asarray(target, float) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.stack([right, down, forward])
        self.focal = 0.5 * size[1] / np.tan(np.radians(fov) / 2)

    offset = np.zeros(2)

    def project(self, points):
        local = (np.asarray(points, float) - self.eye) @ self.rotation.T
        depth = np.maximum(local[..., 2], 1e-6)
        u = self.focal * local[..., 0] / depth + self.size[0] / 2 - self.offset[0]
        v = self.focal * local[..., 1] / depth + self.size[1] / 2 - self.offset[1]
        return np.stack([u, v], -1), local[..., 2]


def rasterise(triangles, colours, view, image, zbuffer, coverage):
    """Draw triangles with a depth test, one bounding box at a time."""
    height, width = zbuffer.shape
    pixels, depth = view.project(triangles)
    for (p0, p1, p2), (z0, z1, z2), colour in zip(pixels, depth, colours):
        lo = np.floor(np.minimum(np.minimum(p0, p1), p2)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(p0, p1), p2)).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, [width - 1, height - 1])
        if (hi < lo).any():
            continue
        ys, xs = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        px, py = xs + 0.5, ys + 0.5
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(area) < 1e-9:
            continue
        w1 = ((px - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (py - p0[1])) / area
        w2 = ((p1[0] - p0[0]) * (py - p0[1]) - (px - p0[0]) * (p1[1] - p0[1])) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0 * z0 + w1 * z1 + w2 * z2
        sub = zbuffer[ys, xs]
        draw = inside & (z < sub)
        zbuffer[ys[draw], xs[draw]] = z[draw]
        image[ys[draw], xs[draw]] = colour
        coverage[ys[draw], xs[draw]] = 1.0


def render(points, colours, grasps, view, supersample=3, top_k=12, table=None):
    width, height = view.size
    image = np.zeros((height, width, 3))
    zbuffer = np.full((height, width), np.inf)
    coverage = np.zeros((height, width))
    if table is not None:  # the board as a plane, behind every point on it
        corners, colour = table
        quad = np.array([corners[[0, 1, 2]], corners[[0, 2, 3]]])
        rasterise(quad, [colour, colour], view, image, zbuffer, coverage)
        zbuffer[:] = np.where(np.isfinite(zbuffer), zbuffer + 0.004, zbuffer)
    pixels, depth = view.project(points)
    order = np.argsort(-depth)
    u = pixels[order, 0].astype(int)
    v = pixels[order, 1].astype(int)
    z, c = depth[order], colours[order]
    near = np.median(depth)
    for dy in range(-supersample, supersample + 1):
        for dx in range(-supersample, supersample + 1):
            # a disc of about 1.5 output pixels, a little larger for near points
            reach = (dx * dx + dy * dy) <= (supersample * 1.15) ** 2 * (near / z) ** 2
            uu, vv = u + dx, v + dy
            ok = reach & (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            ok &= z < zbuffer[vv.clip(0, height - 1), uu.clip(0, width - 1)]
            image[vv[ok], uu[ok]] = c[ok]
            zbuffer[vv[ok], uu[ok]] = z[ok]
            coverage[vv[ok], uu[ok]] = 1.0
    shown = grasps[:top_k][::-1]  # the best drawn last
    for rank, grasp in zip(range(len(shown) - 1, -1, -1), shown):
        width_g, depth_g = float(grasp[1]), float(grasp[3])
        rotation, translation = grasp[4:13].reshape(3, 3), grasp[13:16]
        base = colour_of(rank, len(shown))
        tris, cols = [], []
        for lower, upper in gripper_boxes(width_g, depth_g):
            t, n = box_triangles(lower, upper)
            world = t @ rotation.T + translation
            normal = n @ rotation.T
            shade = 0.55 + 0.45 * np.clip(normal @ (LIGHT / np.linalg.norm(LIGHT)), 0, 1)
            tris.append(world)
            cols += [np.clip(base * s, 0, 255) for s in shade]
        rasterise(np.concatenate(tris), cols, view, image, zbuffer, coverage)
    return image, coverage


def colour_of(rank: int, count: int) -> np.ndarray:
    t = rank / max(count - 1, 1)
    position = t * (len(RAMP) - 1)
    low = int(np.floor(position))
    high = min(low + 1, len(RAMP) - 1)
    return RAMP[low] + (RAMP[high] - RAMP[low]) * (position - low)


def finish(image, coverage, supersample, shadow=True):
    """Premultiplied downsampling to an RGBA picture, with a soft shadow under the board."""
    height, width = coverage.shape
    rgba = np.dstack([image * coverage[..., None], coverage * 255]).astype(np.float32)
    big = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA")
    size = (width // supersample, height // supersample)
    premultiplied = np.asarray(big.resize(size, Image.LANCZOS), dtype=np.float64)
    alpha = premultiplied[..., 3:4] / 255
    colour = np.where(alpha > 1e-3, premultiplied[..., :3] / np.maximum(alpha, 1e-3), 0)
    return Image.fromarray(np.dstack([colour.clip(0, 255), alpha * 255]).astype(np.uint8), "RGBA")


def spread(grasps: np.ndarray, count: int, apart: float) -> np.ndarray:
    """The best grasps, best first, keeping only those at least ``apart`` metres from every
    one already kept."""
    kept: list[np.ndarray] = []
    for grasp in grasps:
        if all(np.linalg.norm(grasp[13:16] - other[13:16]) >= apart for other in kept):
            kept.append(grasp)
        if len(kept) == count:
            break
    return np.array(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("grasps", type=Path, help="grasps.npy written by `grasppanda demo`")
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--root", type=Path, default=None, help="GraspNet-1Billion root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(1600, 800))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--apart", type=float, default=0.05, help="metres between grasp centres")
    parser.add_argument(
        "--margin", type=float, default=0.07, help="board kept around the objects, m"
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=25.0,
        help="degrees around the board from where the camera of the frame stood",
    )
    parser.add_argument("--elevation", type=float, default=40.0, help="degrees above the board")
    parser.add_argument("--fov", type=float, default=28.0)
    parser.add_argument(
        "--fill", type=float, default=0.94, help="share of the width the board spans"
    )
    parser.add_argument("--frames", type=int, default=0, help="an animated swing of N frames")
    parser.add_argument("--swing", type=float, default=18.0, help="degrees either way")
    args = parser.parse_args(argv)

    root = args.root or Path(os.environ["GRASPPANDA_GRASPNET1B_ROOT"])
    points, colours, table_from_camera = load_scene(root, args.scene, args.frame)
    grasps = np.load(args.grasps).astype(np.float64)
    rotation = table_from_camera[:3, :3]
    moved = grasps.copy()
    moved[:, 4:13] = np.einsum("ij,njk->nik", rotation, grasps[:, 4:13].reshape(-1, 3, 3)).reshape(
        -1, 9
    )
    moved[:, 13:16] = grasps[:, 13:16] @ rotation.T + table_from_camera[:3, 3]
    moved = spread(moved, args.top_k, args.apart)

    board = np.abs(points[:, 2]) < 0.004
    keep_box = np.percentile(points[board][:, :2], [3.0, 97.0], axis=0)
    above = points[
        (points[:, 2] > 0.01)
        & np.all((points[:, :2] >= keep_box[0]) & (points[:, :2] <= keep_box[1]), axis=1)
    ]
    around = (
        np.percentile(above[:, :2], [1.0, 99.0], axis=0) + np.array([[-1.0], [1.0]]) * args.margin
    )
    keep_box = np.stack([np.maximum(keep_box[0], around[0]), np.minimum(keep_box[1], around[1])])
    inside = np.all((points[:, :2] >= keep_box[0]) & (points[:, :2] <= keep_box[1]), axis=1)
    inside &= (points[:, 2] > -0.004) & (points[:, 2] < 0.30)
    points, colours = points[inside], colours[inside]
    (x0, y0), (x1, y1) = keep_box
    corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
    board_colour = np.median(colours[np.abs(points[:, 2]) < 0.003], axis=0) * 0.80
    objects = points[points[:, 2] > 0.01]
    target = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
    target[:2] = 0.5 * target[:2] + 0.5 * np.median(objects[:, :2], axis=0)
    sensor = table_from_camera[:3, 3]  # where the camera of the frame stood, in the table frame
    base = np.degrees(np.arctan2(sensor[1] - target[1], sensor[0] - target[0]))

    supersample = 3
    size = (args.size[0] * supersample, args.size[1] * supersample)
    hands = [
        box_triangles(lower, upper)[0].reshape(-1, 3) @ grasp[4:13].reshape(3, 3).T + grasp[13:16]
        for grasp in moved
        for lower, upper in gripper_boxes(grasp[1], grasp[3])
    ]
    extent = np.concatenate([corners, objects[:: max(1, len(objects) // 4000)], *hands])

    def view_at(azimuth: float) -> View:
        a, e = np.radians(base + azimuth), np.radians(args.elevation)
        direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        low, high = 0.2, 5.0
        for _ in range(40):  # the distance at which the scene spans `fill` of the picture
            middle = (low + high) / 2
            pixels, _ = View(target + middle * direction, target, size, args.fov).project(extent)
            span = (pixels.max(0) - pixels.min(0)) / np.array(size)
            low, high = (
                (middle, high) if max(span[0] / args.fill, span[1] / 0.9) > 1 else (low, middle)
            )
        view = View(target + high * direction, target, size, args.fov)
        pixels, _ = view.project(extent)  # centre what is drawn
        shift = (pixels.max(0) + pixels.min(0)) / 2 - np.array(size) / 2
        view.offset = shift
        return view

    def picture(azimuth: float) -> Image.Image:
        view = view_at(azimuth)
        image, coverage = render(
            points, colours, moved, view, supersample, args.top_k, (corners, board_colour)
        )
        return finish(image, coverage, supersample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.frames:
        phase = np.linspace(0, 2 * np.pi, args.frames, endpoint=False)
        frames = [picture(args.azimuth + args.swing * np.sin(angle)) for angle in phase]
        frames[0].save(
            args.out,
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0,
            quality=80,
            method=6,
        )
    else:
        options = {"quality": 92, "method": 6} if args.out.suffix == ".webp" else {"optimize": True}
        picture(args.azimuth).save(args.out, **options)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB), {len(moved)} grasps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
