"""A small software renderer for point clouds and grasps.

It needs only NumPy and Pillow, runs headless, and is deterministic, which makes it
usable in tests, in CI and for generating documentation images. For interactive
inspection use a real viewer; this module is for pictures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from grasppanda.structures import GraspGroup

__all__ = [
    "Camera",
    "fit_camera",
    "gripper_segments",
    "orbit_cameras",
    "render",
    "save_gif",
    "score_colors",
]

#: Low score → high score.
_SCORE_RAMP = np.array([[215, 58, 73], [249, 197, 19], [45, 164, 78]], dtype=np.float64)
_TAIL = 0.04  # length of the stem drawn behind the palm, metres


@dataclass(frozen=True)
class Camera:
    """A pinhole camera looking from ``eye`` at ``target``."""

    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_degrees: float = 40.0

    def rotation(self) -> np.ndarray:
        """Rows are the camera's right, down and forward axes in world coordinates."""
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward])

    def project(self, points: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return pixel coordinates ``(N, 2)`` and depth ``(N,)`` for an image of ``size``."""
        width, height = size
        local = (np.asarray(points, dtype=np.float64) - self.eye) @ self.rotation().T
        depth = local[:, 2]
        focal = 0.5 * height / np.tan(np.radians(self.fov_degrees) / 2.0)
        safe = np.where(depth > 1e-6, depth, 1e-6)
        pixels = np.stack(
            [focal * local[:, 0] / safe + width / 2.0, focal * local[:, 1] / safe + height / 2.0],
            axis=1,
        )
        return pixels, depth


def _default_camera(points: np.ndarray) -> Camera:
    """Look at the cloud from where a depth sensor at the origin saw it, slightly raised."""
    if len(points) == 0:
        return Camera(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))
    centre = np.median(points, axis=0)
    eye = centre * 0.12 + np.array([0.0, -0.12, 0.0]) * np.linalg.norm(centre)
    return Camera(eye=eye, target=centre, up=np.array([0.0, -1.0, 0.0]))


def fit_camera(
    points: np.ndarray,
    focus: np.ndarray | None = None,
    *,
    aspect: float = 4.0 / 3.0,
    coverage: float = 0.98,
    margin: float = 1.08,
) -> Camera:
    """The default view, moved along its line of sight until the subject fills the picture.

    Args:
        points: ``(N, 3)`` cloud in the camera frame of the sensor; it fixes the direction
            of view (from the sensor's side, slightly raised).
        focus: ``(M, 3)`` points the picture is about, for example grasp positions.
            Default: the cloud itself.
        aspect: Width over height of the picture.
        coverage: Share of the subject that has to be inside the picture.
        margin: Step back by this factor from the closest position that achieves it.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    base = _default_camera(points)
    subject = points if focus is None or len(focus) == 0 else np.asarray(focus, dtype=np.float64)
    if len(subject) == 0:
        return base
    target = np.median(subject, axis=0)
    direction = base.eye - base.target
    direction = direction / np.linalg.norm(direction)
    size = (round(1000 * aspect), 1000)

    def enough(distance: float) -> bool:
        camera = Camera(target + direction * distance, target, base.up, base.fov_degrees)
        pixels, depth = camera.project(subject, size)
        inside = (depth > 1e-3) & (pixels[:, 0] >= 0) & (pixels[:, 0] < size[0])
        inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < size[1])
        return inside.mean() >= coverage

    near, far = 0.02, 0.04
    while not enough(far) and far < 50.0:
        near, far = far, far * 2.0
    for _ in range(24):
        middle = 0.5 * (near + far)
        near, far = (near, middle) if enough(middle) else (middle, far)
    return Camera(target + direction * far * margin, target, base.up, base.fov_degrees)


def orbit_cameras(
    points: np.ndarray,
    frames: int = 36,
    *,
    swing_degrees: float = 28.0,
    base: Camera | None = None,
) -> list[Camera]:
    """Cameras swinging left and right around the cloud's centre.

    A single-view cloud has no back side, so a full turn would mostly show holes; a
    swing keeps the observed surfaces in view. ``base`` is the view at the middle of the
    swing; the default view of the cloud when omitted.
    """
    base = base or _default_camera(np.asarray(points, dtype=np.float64))
    offset = base.eye - base.target
    axis = base.up / np.linalg.norm(base.up)
    cameras = []
    for index in range(frames):
        angle = np.radians(swing_degrees) * np.sin(2.0 * np.pi * index / frames)
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = offset * cos + np.cross(axis, offset) * sin + axis * (axis @ offset) * (1.0 - cos)
        cameras.append(Camera(base.target + rotated, base.target, base.up, base.fov_degrees))
    return cameras


def score_colors(scores: np.ndarray) -> np.ndarray:
    """Map scores to ``(G, 3)`` uint8 colours by rank: red (lowest) to green (highest)."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rank = np.argsort(np.argsort(scores, kind="stable"), kind="stable") / max(len(scores) - 1, 1)
    position = rank * (len(_SCORE_RAMP) - 1)
    lower = np.clip(np.floor(position).astype(int), 0, len(_SCORE_RAMP) - 2)
    blend = (position - lower)[:, None]
    return (_SCORE_RAMP[lower] * (1 - blend) + _SCORE_RAMP[lower + 1] * blend).astype(np.uint8)


def gripper_segments(grasps: GraspGroup) -> np.ndarray:
    """Wire-frame of each gripper as ``(G, 4, 2, 3)`` line segments in the grasp's frame.

    Per grasp: left finger, right finger, palm, stem. Fingers reach ``depth`` past the
    grasp point along the approach axis and start ``depth_base`` behind it.
    """
    translation = grasps.translation.detach().cpu().numpy().astype(np.float64)
    rotation = grasps.rotation.detach().cpu().numpy().astype(np.float64)
    half = grasps.width.detach().cpu().numpy().astype(np.float64)[:, None] / 2.0
    depth = grasps.depth.detach().cpu().numpy().astype(np.float64)[:, None]
    back = np.full_like(depth, -grasps.gripper.depth_base)
    zero = np.zeros_like(depth)

    def local(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.concatenate([x, y, zero], axis=1)

    endpoints = np.stack(
        [
            np.stack([local(back, -half), local(depth, -half)], axis=1),
            np.stack([local(back, half), local(depth, half)], axis=1),
            np.stack([local(back, -half), local(back, half)], axis=1),
            np.stack([local(back - _TAIL, zero), local(back, zero)], axis=1),
        ],
        axis=1,
    )  # (G, 4, 2, 3) in the gripper frame
    return np.einsum("gij,gsej->gsei", rotation, endpoints) + translation[:, None, None, :]


def _colour_points(points: np.ndarray, colors: np.ndarray | None) -> np.ndarray:
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
        return colors
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    depth = points[:, 2]
    span = max(float(depth.max() - depth.min()), 1e-9)
    shade = (1.0 - (depth - depth.min()) / span)[:, None]
    return (np.array([70.0, 80.0, 95.0]) + shade * np.array([120.0, 125.0, 125.0])).astype(np.uint8)


def render(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    grasps: GraspGroup | None = None,
    *,
    camera: Camera | None = None,
    size: tuple[int, int] = (960, 720),
    top_k: int | None = 50,
    point_size: int = 2,
    line_width: int = 3,
    background: tuple[int, int, int] = (255, 255, 255),
    supersample: int = 2,
) -> Image.Image:
    """Draw a point cloud and, on top of it, the ``top_k`` highest-scoring grasps.

    Args:
        points: ``(N, 3)`` in the same frame as the grasps.
        colors: ``(N, 3)`` uint8 or float in ``[0, 1]``; shaded by depth when omitted.
        grasps: Grasps to draw, coloured by score rank (red low, green high).
        camera: Viewpoint; chosen automatically for a camera-frame cloud when omitted.
        size: ``(width, height)`` of the returned image.
        top_k: Number of grasps to draw; ``None`` draws all.
        supersample: Render at this multiple of ``size`` and scale down, for smooth edges.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera = camera or _default_camera(points)
    width, height = size[0] * supersample, size[1] * supersample
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = background
    zbuffer = np.full((height, width), np.inf)

    pixels, depth = camera.project(points, (width, height))
    palette = _colour_points(points, colors)
    visible = depth > 1e-3
    order = np.argsort(-depth[visible], kind="stable")  # far points first, near points on top
    u = np.round(pixels[visible][order, 0]).astype(np.int64)
    v = np.round(pixels[visible][order, 1]).astype(np.int64)
    z = depth[visible][order]
    shade = palette[visible][order]
    splat = point_size * supersample
    for dy in range(splat):
        for dx in range(splat):
            uu, vv = u + dx - splat // 2, v + dy - splat // 2
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            canvas[vv[inside], uu[inside]] = shade[inside]
            zbuffer[vv[inside], uu[inside]] = z[inside]

    image = Image.fromarray(canvas)
    if grasps is not None and len(grasps):
        chosen = grasps.sort_by_score()
        if top_k is not None:
            chosen = chosen[:top_k]
        segments = gripper_segments(chosen)[::-1]  # best grasp drawn last, on top
        tint = score_colors(chosen.score.detach().cpu().numpy())[::-1]
        draw = ImageDraw.Draw(image)
        flat, flat_depth = camera.project(segments.reshape(-1, 3), (width, height))
        flat = flat.reshape(-1, 4, 2, 2)
        flat_depth = flat_depth.reshape(-1, 4, 2)
        stroke = line_width * supersample
        for lines, depths, color in zip(flat, flat_depth, tint):
            shown = []
            for (start, end), (z0, z1) in zip(lines, depths):
                if min(z0, z1) <= 1e-3:
                    continue
                mid_u, mid_v = np.round((start + end) / 2.0).astype(int)
                behind = (
                    0 <= mid_u < width
                    and 0 <= mid_v < height
                    and (z0 + z1) / 2.0 > zbuffer[mid_v, mid_u] + 0.03
                )
                if not behind:  # skip segments clearly behind the observed surface
                    shown.append([tuple(start), tuple(end)])
            # A dark rim under each gripper keeps it readable on any background.
            for segment in shown:
                draw.line(segment, fill=(20, 22, 26), width=stroke + 2 * supersample)
            for segment in shown:
                draw.line(segment, fill=tuple(int(c) for c in color), width=stroke)
    if supersample > 1:
        image = image.resize(size, Image.LANCZOS)
    return image


def save_gif(frames: Sequence[Image.Image], path: str | Path, *, duration_ms: int = 80) -> Path:
    """Write frames as a looping GIF with one shared palette, so colours do not flicker."""
    if not frames:
        raise ValueError("save_gif() needs at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[0].convert("P", palette=Image.ADAPTIVE, colors=128)
    quantised = [
        frame.convert("RGB").quantize(palette=palette, dither=Image.NONE) for frame in frames
    ]
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return path
