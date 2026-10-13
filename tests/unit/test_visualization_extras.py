"""Framing, colours from an image, the optional viewer and the charts of ``analyze``."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from grasppanda.structures import GraspGroup
from grasppanda.utils.imports import MissingDependencyError
from grasppanda.visualization import fit_camera, orbit_cameras
from grasppanda.visualization.charts import draw_analysis, draw_comparison
from grasppanda.visualization.viewer import colors_from_image, show, wireframe


def grasps(count: int = 3) -> GraspGroup:
    return GraspGroup(
        translation=torch.tensor([[0.02 * i, 0.0, 0.5] for i in range(count)]),
        rotation=torch.eye(3).expand(count, 3, 3).clone(),
        width=torch.full((count,), 0.06),
        depth=torch.full((count,), 0.02),
        score=torch.linspace(0.1, 0.9, count),
    )


def inside(camera, points: np.ndarray, size=(400, 300)) -> np.ndarray:
    pixels, depth = camera.project(points, size)
    return (
        (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < 400)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < 300)
    )


def test_fit_camera_frames_the_subject_and_no_more():
    generator = np.random.default_rng(0)
    table = generator.uniform([-0.6, -0.4, 0.5], [0.6, 0.4, 0.9], size=(4000, 3))
    whole = fit_camera(table)
    assert inside(whole, table).mean() >= 0.98
    corner = table[(table[:, 0] > 0.3) & (table[:, 1] > 0.1)]
    close = fit_camera(table, corner)
    assert inside(close, corner).mean() >= 0.98
    assert inside(close, table).mean() < 0.6  # it moved in: most of the table is out of view
    assert np.linalg.norm(close.eye - close.target) < np.linalg.norm(whole.eye - whole.target)
    assert np.allclose(close.target, np.median(corner, axis=0))
    swing = orbit_cameras(table, frames=4, base=close)
    assert np.allclose(swing[0].eye, close.eye) and np.allclose(swing[2].target, close.target)
    assert not np.allclose(swing[1].eye, close.eye)
    assert fit_camera(np.zeros((0, 3))).fov_degrees == 40.0  # nothing to frame: the default view


def test_points_take_the_colour_of_their_pixel():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[1, 2] = (10, 20, 30)
    rgb[3, 5] = (200, 100, 50)
    intrinsics = np.array([[2.0, 0, 3.0], [0, 2.0, 2.0], [0, 0, 1.0]])
    points = np.array(
        [
            [(2 - 3.0) * 0.5 / 2.0, (1 - 2.0) * 0.5 / 2.0, 0.5],  # pixel (u 2, v 1)
            [(5 - 3.0) * 2.0 / 2.0, (3 - 2.0) * 2.0 / 2.0, 2.0],  # pixel (u 5, v 3)
            [5.0, 0.0, 1.0],  # outside the image
            [0.0, 0.0, 0.0],  # no depth
        ]
    )
    colours = colors_from_image(points, rgb, intrinsics)
    assert colours.tolist() == [[10, 20, 30], [200, 100, 50], [128, 128, 128], [128, 128, 128]]


def test_wireframe_holds_four_lines_per_shown_grasp_best_first():
    vertices, pairs, tint = wireframe(grasps(3), top_k=2)
    assert vertices.shape == (16, 3) and pairs.shape == (8, 2) and tint.shape == (8, 3)
    assert pairs.tolist()[:2] == [[0, 1], [2, 3]]
    assert np.allclose(vertices[1], [0.04 + 0.02, -0.03, 0.5])  # the best grasp's left fingertip
    assert tint[0, 1] > tint[0, 0] and tint[-1, 0] > tint[-1, 1]  # best green, last shown red
    assert tint.min() >= 0.0 and tint.max() <= 1.0


def test_show_needs_open3d_and_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)  # as if it were not installed
    with pytest.raises(MissingDependencyError, match='pip install "grasppanda\\[vis\\]"'):
        show(np.zeros((3, 3)))


def test_show_hands_cloud_and_grippers_to_open3d(monkeypatch):
    drawn = {}

    class Holder:
        pass

    fake = types.SimpleNamespace(
        geometry=types.SimpleNamespace(PointCloud=Holder, LineSet=Holder),
        utility=types.SimpleNamespace(Vector3dVector=np.asarray, Vector2iVector=np.asarray),
        visualization=types.SimpleNamespace(
            draw_geometries=lambda geometries, window_name: drawn.update(
                geometries=geometries, title=window_name
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake)
    points = np.random.default_rng(0).uniform(size=(50, 3))
    colours = np.full((50, 3), 255, dtype=np.uint8)
    show(points, colours, grasps(3), top_k=2, title="test")
    cloud, lines = drawn["geometries"]
    assert drawn["title"] == "test" and cloud.points.shape == (50, 3)
    assert cloud.colors.max() == 1.0  # uint8 colours are scaled for Open3D
    assert (
        lines.points.shape == (16, 3)
        and lines.lines.shape == (8, 2)
        and lines.colors.shape == (8, 3)
    )
    show(points)
    assert len(drawn["geometries"]) == 1  # no grasps: the cloud alone


def non_background(image) -> float:
    pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
    return float((np.abs(pixels.astype(int) - np.array([252, 252, 251])).sum(axis=1) > 12).mean())


def test_analysis_chart_grows_with_the_number_of_scenes():
    names = [f"cam/scene_{i:04d}" for i in range(30)]
    values = np.linspace(0.1, 0.8, 30).tolist()
    shares = {"earned": 0.5, "lost: collision": 0.3, "lost: no force closure": 0.2}
    labels = ["0.2", "0.4", "0.6", "0.8", "1", "1.2"]
    image = draw_analysis(
        "run · cd", "conditions", labels, [0.2, 0.4, 0.5, 0.6, 0.6, 0.7], names, values, shares
    )
    assert image.size[0] == 1100 and non_background(image) > 0.05
    more = draw_analysis("run", "c", labels, [0.2] * 6, names * 4, values * 4, None)
    assert more.size[1] > image.size[1]
    pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
    for colour in ((42, 120, 214), (235, 104, 52), (27, 175, 122)):  # one hue per share, in order
        assert (np.abs(pixels.astype(int) - colour).sum(axis=1) < 6).any()
    empty = draw_analysis("run", "c", [], [], [], [], None)  # a record without scenes still draws
    assert empty.size[0] == 1100


def test_comparison_chart_marks_the_sign_by_colour():
    names = [f"cam/scene_{i:04d}" for i in range(8)]
    image = draw_comparison(
        "A · B", "summary", names, [0.05, -0.04, 0.03, 0.02, -0.01, 0.01, 0.0, 0.0]
    )
    pixels = np.asarray(image.convert("RGB")).reshape(-1, 3).astype(int)
    blue = (np.abs(pixels - (42, 120, 214)).sum(axis=1) < 6).sum()
    red = (np.abs(pixels - (227, 73, 72)).sum(axis=1) < 6).sum()
    assert blue > red > 0  # more and larger gains than losses
    only_gains = draw_comparison("A · B", "s", names[:2], [0.02, 0.01])
    gains = np.asarray(only_gains.convert("RGB")).reshape(-1, 3).astype(int)
    assert (np.abs(gains - (42, 120, 214)).sum(axis=1) < 6).any()
