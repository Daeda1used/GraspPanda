"""Object poses of a GraspNet-1Billion frame, with the reference tool's rounding.

The annotation XML stores each object's position and a quaternion with four decimals.
The reference evaluator does not use that quaternion directly. It converts it to a
rotation matrix, the matrix to static-xyz Euler angles, the angles to degrees, back to
radians, back to a matrix, and finally stores the pose as float32. Every later stage
(model placement, collision crop, the object-frame grasp handed to the force-closure
test) starts from those float32 values, so a "cleaner" conversion would change scores
for grasps close to a threshold. This module reproduces the detour on purpose.

All trigonometry goes through :mod:`math`, as it does upstream, and each product is
associated as written here; do not "simplify" the expressions.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AnnotatedObject",
    "euler_to_matrix",
    "matrix_to_euler",
    "pose_from_annotation",
    "quaternion_to_matrix",
    "read_annotation",
]

#: Below this squared norm a quaternion is treated as the identity rotation.
_QUATERNION_EPS = float(np.finfo(np.float64).eps)
#: Below this cosine of the middle angle the Euler decomposition is in gimbal lock.
_GIMBAL_EPS = float(np.finfo(np.float64).eps) * 4.0

_Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class AnnotatedObject:
    """One ``<obj>`` entry of an annotation file."""

    object_id: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # w, x, y, z


def read_annotation(path: str | Path) -> list[AnnotatedObject]:
    """Parse ``annotations/NNNN.xml`` into its objects, in file order."""
    try:
        scene = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"{path} is not a valid annotation file: {error}") from error
    objects = []
    for index, node in enumerate(scene):
        fields = {}
        for tag, count in (("obj_id", 1), ("pos_in_world", 3), ("ori_in_world", 4)):
            child = node.find(tag)
            values = child.text.split() if child is not None and child.text else []
            if len(values) != count:
                raise ValueError(
                    f"{path}: object {index} needs <{tag}> with {count} value(s), "
                    f"found {len(values)}"
                )
            fields[tag] = values
        objects.append(
            AnnotatedObject(
                object_id=int(fields["obj_id"][0]),
                position=tuple(float(v) for v in fields["pos_in_world"]),
                quaternion=tuple(float(v) for v in fields["ori_in_world"]),
            )
        )
    return objects


def quaternion_to_matrix(w: float, x: float, y: float, z: float) -> _Matrix:
    """Rotation matrix of a quaternion that need not be normalised."""
    squared_norm = w * w + x * x + y * y + z * z
    if squared_norm < _QUATERNION_EPS:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    scale = 2.0 / squared_norm
    xs, ys, zs = x * scale, y * scale, z * scale
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return (
        (1.0 - (yy + zz), xy - wz, xz + wy),
        (xy + wz, 1.0 - (xx + zz), yz - wx),
        (xz - wy, yz + wx, 1.0 - (xx + yy)),
    )


def matrix_to_euler(matrix: _Matrix) -> tuple[float, float, float]:
    """Static-frame x, y, z Euler angles (radians) of a rotation matrix."""
    cos_y = math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])
    angle_y = math.atan2(-matrix[2][0], cos_y)
    if cos_y > _GIMBAL_EPS:
        angle_x = math.atan2(matrix[2][1], matrix[2][2])
        angle_z = math.atan2(matrix[1][0], matrix[0][0])
    else:  # gimbal lock: the z rotation is folded into x
        angle_x = math.atan2(-matrix[1][2], matrix[1][1])
        angle_z = 0.0
    return angle_x, angle_y, angle_z


def euler_to_matrix(angle_x: float, angle_y: float, angle_z: float) -> _Matrix:
    """Rotation matrix ``Rz(angle_z) @ Ry(angle_y) @ Rx(angle_x)`` (static x, y, z)."""
    sin_x, sin_y, sin_z = math.sin(angle_x), math.sin(angle_y), math.sin(angle_z)
    cos_x, cos_y, cos_z = math.cos(angle_x), math.cos(angle_y), math.cos(angle_z)
    cos_x_cos_z, cos_x_sin_z = cos_x * cos_z, cos_x * sin_z
    sin_x_cos_z, sin_x_sin_z = sin_x * cos_z, sin_x * sin_z
    return (
        (cos_y * cos_z, sin_y * sin_x_cos_z - cos_x_sin_z, sin_y * cos_x_cos_z + sin_x_sin_z),
        (cos_y * sin_z, sin_y * sin_x_sin_z + cos_x_cos_z, sin_y * cos_x_sin_z - sin_x_cos_z),
        (-sin_y, cos_y * sin_x, cos_y * cos_x),
    )


def pose_from_annotation(entry: AnnotatedObject) -> np.ndarray:
    """``(4, 4)`` float32 object-to-camera pose exactly as the reference tool builds it."""
    angles = matrix_to_euler(quaternion_to_matrix(*entry.quaternion))
    to_degrees = 180.0 / math.pi
    degrees = [angle * to_degrees for angle in angles]
    radians = [angle / 180.0 * math.pi for angle in degrees]
    pose = np.zeros((4, 4), dtype=np.float32)
    pose[:3, :3] = euler_to_matrix(*radians)
    pose[:3, 3] = entry.position
    pose[3, 3] = 1
    return pose
