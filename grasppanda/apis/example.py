"""A synthetic table-top scene, so that inference can be tried without any dataset.

:func:`example_scene` ray-casts a handful of geometric primitives standing on a table
and returns what a depth camera would deliver: a 16-bit depth image in millimetres, a
colour image, the camera matrix, an object mask and the camera-to-table transform.
Everything is computed from the formulas in this file; no dataset image, object model
or recorded sensor data is involved, so the scene and pictures of it carry no dataset
licence.

It is **not** a benchmark and looks nothing like a real sensor image: surfaces are
perfectly smooth, there is no noise, no missing depth and no reflection. A detector
trained on real scenes is outside its training distribution here. The scene exists to
show that an installation works end to end and to give tests an input.

Frames: the camera frame is the optical frame of the depth sensor (x right, y down, z
forward); the table frame has its origin on the table top with z up. Lengths are metres
except for the depth image, which is in millimetres like the PNG files of most sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ExampleScene", "example_scene"]

#: ``(kind, name, centre xyz, size, yaw degrees, RGB)`` in the table frame, metres.
#: ``size`` is half extents for a box, ``(radius, half length)`` for a cylinder and
#: ``(radius,)`` for a sphere. A lying cylinder has its axis in the table plane.
_OBJECTS = (
    ("box", "box", (-0.13, 0.02, 0.030), (0.035, 0.050, 0.030), 25.0, (214, 96, 77)),
    ("box", "tall box", (0.02, 0.10, 0.055), (0.025, 0.025, 0.055), -15.0, (67, 147, 195)),
    ("cylinder", "can", (0.14, 0.03, 0.045), (0.030, 0.045), 0.0, (244, 165, 60)),
    ("lying", "tube", (-0.02, -0.08, 0.025), (0.025, 0.060), 70.0, (90, 174, 97)),
    ("sphere", "ball", (0.10, -0.09, 0.035), (0.035,), 0.0, (153, 112, 171)),
    ("box", "flat box", (-0.15, -0.11, 0.0125), (0.050, 0.035, 0.0125), -35.0, (140, 140, 140)),
)
_TABLE_HALF = (0.45, 0.35)  # the table top ends there; beyond it the sensor sees nothing
_TABLE_RGB = (236, 229, 215)
_EYE = np.array([0.0, -0.42, 0.40])  # camera position in the table frame
_TARGET = np.array([0.0, 0.02, 0.03])
_LIGHT = np.array([0.35, -0.45, 0.82])
_FAR = 1e9


@dataclass(frozen=True, eq=False)
class ExampleScene:
    """What a depth camera would deliver for the synthetic scene.

    Attributes:
        depth: ``(H, W)`` uint16 depth in millimetres; zero where the camera sees past
            the table.
        rgb: ``(H, W, 3)`` uint8 colour image, registered to ``depth``.
        intrinsics: ``(3, 3)`` camera matrix in pixels.
        mask: ``(H, W)`` uint8, zero for table and background and ``i + 1`` on the
            ``i``-th entry of ``objects``. It is exact because the scene is synthetic;
            on a real camera a mask comes from a segmentation model.
        table_from_camera: ``(4, 4)`` transform from the camera frame to the table frame.
        depth_scale: Depth units per metre (1000: millimetres).
        objects: Names of the primitives, in mask order.
    """

    depth: np.ndarray
    rgb: np.ndarray
    intrinsics: np.ndarray
    mask: np.ndarray
    table_from_camera: np.ndarray
    depth_scale: float
    objects: tuple[str, ...]

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write ``depth.png`` (16 bit), ``rgb.png``, ``mask.png`` and the calibration.

        ``intrinsics.txt`` holds ``fx,fy,cx,cy`` on one line, the form
        ``grasppanda demo --intrinsics`` reads; ``table_from_camera.txt`` the 4 x 4 matrix.
        """
        from PIL import Image

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {name: directory / f"{name}.png" for name in ("depth", "rgb", "mask")}
        Image.fromarray(self.depth).save(paths["depth"])
        Image.fromarray(self.rgb).save(paths["rgb"])
        Image.fromarray(self.mask).save(paths["mask"])
        k = self.intrinsics
        paths["intrinsics"] = directory / "intrinsics.txt"
        paths["intrinsics"].write_text(f"{k[0, 0]:g},{k[1, 1]:g},{k[0, 2]:g},{k[1, 2]:g}\n")
        paths["table_from_camera"] = directory / "table_from_camera.txt"
        np.savetxt(paths["table_from_camera"], self.table_from_camera, fmt="%.9f")
        return paths


def _rotation_z(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hit_box(origin: np.ndarray, ray: np.ndarray, half: np.ndarray):
    """Slab test against ``|x_i| <= half_i``; returns distance and outward normal."""
    parallel = ray == 0
    safe = np.where(parallel, 1.0, ray)
    near = (-np.sign(safe) * half - origin) / safe
    far = (np.sign(safe) * half - origin) / safe
    # A ray parallel to a slab is inside it for its whole length, or never.
    inside = np.abs(origin) <= half
    near = np.where(parallel, np.where(inside, -_FAR, _FAR), near)
    far = np.where(parallel, np.where(inside, _FAR, -_FAR), far)
    axis = near.argmax(axis=1)
    enter, leave = near.max(axis=1), far.min(axis=1)
    hit = (enter < leave) & (enter > 0)
    normal = np.zeros_like(ray)
    rows = np.arange(len(ray))
    normal[rows, axis] = -np.sign(ray[rows, axis])
    return np.where(hit, enter, _FAR), normal


def _hit_sphere(origin: np.ndarray, ray: np.ndarray, radius: float):
    a = (ray * ray).sum(axis=1)
    b = (origin * ray).sum(axis=1)
    c = (origin * origin).sum(axis=1) - radius * radius
    root = b * b - a * c
    distance = (-b - np.sqrt(np.maximum(root, 0.0))) / a
    hit = (root > 0) & (distance > 0)
    normal = (origin + distance[:, None] * ray) / radius
    return np.where(hit, distance, _FAR), normal


def _hit_cylinder(origin: np.ndarray, ray: np.ndarray, radius: float, half: float):
    """Finite cylinder about the local z-axis, with caps."""
    a = ray[:, 0] ** 2 + ray[:, 1] ** 2
    b = origin[:, 0] * ray[:, 0] + origin[:, 1] * ray[:, 1]
    c = origin[:, 0] ** 2 + origin[:, 1] ** 2 - radius * radius
    root = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        side = (-b - np.sqrt(np.maximum(root, 0.0))) / a
        height = origin[:, 2] + side * ray[:, 2]
        on_side = (root > 0) & (a > 0) & (side > 0) & (np.abs(height) <= half)
        cap = (np.sign(-ray[:, 2]) * half - origin[:, 2]) / ray[:, 2]
    at_cap = origin[:, :2] + cap[:, None] * ray[:, :2]
    on_cap = np.isfinite(cap) & (cap > 0) & ((at_cap**2).sum(axis=1) <= radius * radius)
    side = np.where(on_side, side, _FAR)
    cap = np.where(on_cap, cap, _FAR)
    normal = np.zeros_like(ray)
    radial = origin[:, :2] + side[:, None] * ray[:, :2]
    normal[:, :2] = radial / radius
    lid = cap < side
    normal[lid] = 0.0
    normal[lid, 2] = -np.sign(ray[lid, 2])
    return np.minimum(side, cap), normal


def _placements(seed: int) -> list[tuple[str, str, np.ndarray, tuple, np.ndarray, tuple]]:
    """Objects with their table-from-object rotation; ``seed`` 0 is the documented layout."""
    rng = np.random.default_rng(seed)
    placed = []
    for kind, name, centre, size, yaw, colour in _OBJECTS:
        centre = np.array(centre)
        if seed:
            centre[:2] += rng.uniform(-0.015, 0.015, size=2)
            yaw = yaw + rng.uniform(-20.0, 20.0)
        rotation = _rotation_z(yaw)
        if kind == "lying":  # local z (the cylinder axis) laid along the table-frame x-axis
            rotation = rotation @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        placed.append((kind, name, centre, size, rotation, colour))
    return placed


def example_scene(
    seed: int = 0, *, size: tuple[int, int] = (1280, 720), focal: float = 925.0
) -> ExampleScene:
    """Render the synthetic table-top scene.

    Args:
        seed: ``0`` gives the documented layout; any other value moves every primitive
            by up to 1.5 cm and turns it by up to 20 degrees, reproducibly.
        size: ``(width, height)`` of the images in pixels. The default is the 1280 x 720
            of the RealSense and Kinect images of GraspNet-1Billion.
        focal: Focal length in pixels at a width of 1280; scaled with ``size``.

    Returns:
        The scene as an :class:`ExampleScene`. Two calls with the same arguments return
        identical arrays.
    """
    width, height = size
    fx = fy = focal * width / 1280.0
    intrinsics = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])

    forward = (_TARGET - _EYE) / np.linalg.norm(_TARGET - _EYE)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    table_from_camera = np.eye(4)
    table_from_camera[:3, :3] = np.stack([right, down, forward], axis=1)
    table_from_camera[:3, 3] = _EYE

    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack(
        [(u.ravel() - intrinsics[0, 2]) / fx, (v.ravel() - intrinsics[1, 2]) / fy], axis=1
    )
    rays = np.concatenate([rays, np.ones((len(rays), 1))], axis=1)  # z = 1: distance is depth
    rays = rays @ table_from_camera[:3, :3].T  # in the table frame
    origin = np.broadcast_to(_EYE, rays.shape)

    # The table top: the plane z = 0 within the table's outline.
    with np.errstate(divide="ignore"):
        distance = np.where(rays[:, 2] < 0, -_EYE[2] / rays[:, 2], _FAR)
    landing = _EYE[:2] + distance[:, None] * rays[:, :2]
    on_table = (np.abs(landing[:, 0]) <= _TABLE_HALF[0]) & (np.abs(landing[:, 1]) <= _TABLE_HALF[1])
    distance = np.where(on_table, distance, _FAR)
    normal = np.tile(np.array([0.0, 0.0, 1.0]), (len(rays), 1))
    # Faint 5 cm tiles: enough texture for an image network, and plainly not a photograph.
    tiles = (np.floor(landing[:, 0] / 0.05) + np.floor(landing[:, 1] / 0.05)) % 2
    colour = np.array(_TABLE_RGB, dtype=np.float64) - 14.0 * tiles[:, None]
    mask = np.zeros(len(rays), dtype=np.uint8)

    placed = _placements(seed)
    for index, (kind, _, centre, extent, rotation, rgb) in enumerate(placed, start=1):
        local_origin = (origin - centre) @ rotation
        local_ray = rays @ rotation
        if kind == "box":
            hit, local_normal = _hit_box(local_origin, local_ray, np.array(extent))
        elif kind == "sphere":
            hit, local_normal = _hit_sphere(local_origin, local_ray, extent[0])
        else:
            hit, local_normal = _hit_cylinder(local_origin, local_ray, extent[0], extent[1])
        nearer = hit < distance
        distance = np.where(nearer, hit, distance)
        normal[nearer] = (local_normal @ rotation.T)[nearer]
        colour[nearer] = rgb
        mask[nearer] = index

    seen = distance < _FAR
    light = _LIGHT / np.linalg.norm(_LIGHT)
    shade = 0.45 + 0.55 * np.clip(normal @ light, 0.0, 1.0)
    image = np.where(seen[:, None], np.clip(colour * shade[:, None], 0, 255), 18.0)
    depth = np.where(seen, np.round(distance * 1000.0), 0.0)
    return ExampleScene(
        depth=depth.astype(np.uint16).reshape(height, width),
        rgb=image.astype(np.uint8).reshape(height, width, 3),
        intrinsics=intrinsics,
        mask=mask.reshape(height, width),
        table_from_camera=table_from_camera,
        depth_scale=1000.0,
        objects=tuple(name for _, name, *_ in placed),
    )
