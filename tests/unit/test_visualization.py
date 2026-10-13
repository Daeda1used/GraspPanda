"""The software renderer: geometry first, pixels second."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from grasppanda.structures import GraspGroup
from grasppanda.visualization import (
    Camera,
    gripper_segments,
    orbit_cameras,
    render,
    save_gif,
    score_colors,
)


def one_grasp(width: float = 0.08, depth: float = 0.03) -> GraspGroup:
    return GraspGroup(
        translation=torch.tensor([[0.1, 0.0, 0.5]]),
        rotation=torch.eye(3)[None],
        width=torch.tensor([width]),
        depth=torch.tensor([depth]),
        score=torch.tensor([0.9]),
    )


def test_camera_projects_the_target_to_the_image_centre():
    camera = Camera(
        eye=np.zeros(3), target=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, -1.0, 0.0])
    )
    pixels, depth = camera.project(
        np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]]), (640, 480)
    )
    assert np.allclose(pixels[0], [320, 240]) and np.allclose(depth, 2.0)
    assert pixels[1, 0] > 320 and np.isclose(pixels[1, 1], 240)  # +x is right
    assert pixels[2, 1] > 240 and np.isclose(pixels[2, 0], 320)  # +y is down, as in a camera frame


def test_gripper_wireframe_has_the_declared_dimensions():
    grasp = one_grasp(width=0.08, depth=0.03)
    left, right, palm, stem = gripper_segments(grasp)[0]
    base = grasp.gripper.depth_base
    assert np.allclose(np.linalg.norm(left[1] - left[0]), 0.03 + base)  # finger length
    assert np.allclose(np.linalg.norm(palm[1] - palm[0]), 0.08)  # jaw opening
    assert np.allclose(left[1], [0.1 + 0.03, -0.04, 0.5]) and np.allclose(
        right[1], [0.13, 0.04, 0.5]
    )
    assert np.allclose(stem[1], [0.1 - base, 0.0, 0.5])  # stem ends at the palm centre


def test_score_colours_rank_from_red_to_green():
    colors = score_colors(np.array([0.1, 0.9, 0.5]))
    assert colors[0, 0] > colors[0, 1] and colors[1, 1] > colors[1, 0]  # lowest red, highest green
    assert score_colors(np.array([])).shape == (0, 3)


def test_render_draws_points_and_grasps():
    generator = np.random.default_rng(0)
    points = generator.uniform([-0.3, -0.2, 0.4], [0.3, 0.2, 0.6], size=(5000, 3))
    blank = render(points[:0], size=(160, 120))
    cloud = render(points, size=(160, 120))
    with_grasp = render(points, grasps=one_grasp(), size=(160, 120))
    assert cloud.size == (160, 120) and isinstance(cloud, Image.Image)
    assert np.asarray(blank).min() == 255  # nothing drawn on an empty scene
    assert (np.asarray(cloud) != 255).any()
    assert (np.asarray(cloud) != np.asarray(with_grasp)).any()
    assert np.array_equal(
        np.asarray(cloud), np.asarray(render(points, size=(160, 120)))
    )  # deterministic


def test_orbit_and_gif(tmp_path):
    points = np.random.default_rng(1).uniform([-0.3, -0.2, 0.4], [0.3, 0.2, 0.6], size=(2000, 3))
    cameras = orbit_cameras(points, frames=5)  # five distinct angles; a GIF merges equal frames
    assert len(cameras) == 5 and np.allclose(cameras[0].target, cameras[3].target)
    distances = [np.linalg.norm(camera.eye - camera.target) for camera in cameras]
    assert np.allclose(distances, distances[0])  # the swing keeps its radius
    frames = [render(points, camera=camera, size=(96, 72), supersample=1) for camera in cameras]
    path = save_gif(frames, tmp_path / "swing.gif")
    with Image.open(path) as gif:
        assert gif.n_frames == 5 and gif.size == (96, 72)
