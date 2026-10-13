"""An interactive window for a cloud and its grasps, for machines that have a display.

The pictures of this package come from :mod:`.render`, which needs nothing but NumPy
and Pillow. This module adds a viewer one can turn with the mouse. It uses Open3D, an
optional dependency (``pip install -e ".[vis]"``) that is imported only when a
window is actually opened.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grasppanda.structures import GraspGroup
from grasppanda.utils.imports import require

from .render import gripper_segments, score_colors

__all__ = ["colors_from_image", "show", "wireframe"]


def colors_from_image(points: np.ndarray, rgb: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Colour camera-frame points with the pixels of a registered colour image.

    Args:
        points: ``(N, 3)`` in the camera frame of the image, metres.
        rgb: ``(H, W, 3)`` uint8.
        intrinsics: ``(3, 3)`` camera matrix of the image.

    Returns:
        ``(N, 3)`` uint8; mid grey for a point that projects outside the image.
    """
    points = np.asarray(points, dtype=np.float64)
    height, width = rgb.shape[:2]
    depth = np.where(points[:, 2] > 1e-9, points[:, 2], 1.0)
    u = np.round(points[:, 0] * intrinsics[0, 0] / depth + intrinsics[0, 2]).astype(np.int64)
    v = np.round(points[:, 1] * intrinsics[1, 1] / depth + intrinsics[1, 2]).astype(np.int64)
    inside = (points[:, 2] > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    colors[inside] = rgb[v[inside], u[inside]]
    return colors


def wireframe(
    grasps: GraspGroup, top_k: int | None = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Line geometry of the ``top_k`` best grasps, best first.

    Returns:
        ``(V, 3)`` vertices, ``(L, 2)`` vertex index pairs and ``(L, 3)`` colours in
        ``[0, 1]``: four lines (two fingers, palm, stem) per grasp, coloured by score
        rank from red (lowest shown) to green (highest).
    """
    chosen = grasps.sort_by_score()
    if top_k is not None:
        chosen = chosen[:top_k]
    segments = gripper_segments(chosen)  # (G, 4, 2, 3)
    vertices = segments.reshape(-1, 3)
    pairs = np.arange(len(vertices)).reshape(-1, 2)
    tint = score_colors(chosen.score.detach().cpu().numpy()).astype(np.float64) / 255.0
    return vertices, pairs, np.repeat(tint, 4, axis=0)


def show(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    grasps: GraspGroup | None = None,
    *,
    top_k: int | None = 50,
    title: str = "GraspPanda",
) -> None:
    """Open a window with the cloud and the ``top_k`` best grasps; returns when it is closed.

    Args:
        points: ``(N, 3)`` in the frame of the grasps.
        colors: ``(N, 3)`` uint8 or float in ``[0, 1]``; grey when omitted.
        grasps: Grasps to draw as wire-frame grippers.
        top_k: Number of grasps to draw; ``None`` draws all.
        title: Window title.

    Raises:
        MissingDependencyError: Open3D is not installed.
    """
    o3d: Any = require(
        "open3d", extra="vis", purpose="The interactive viewer (grasppanda demo --show)"
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float64) / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)
    geometries = [cloud]
    if grasps is not None and len(grasps):
        vertices, pairs, tint = wireframe(grasps, top_k)
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(vertices)
        lines.lines = o3d.utility.Vector2iVector(pairs)
        lines.colors = o3d.utility.Vector3dVector(tint)
        geometries.append(lines)
    o3d.visualization.draw_geometries(geometries, window_name=title)
