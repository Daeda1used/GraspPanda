"""Evaluator geometry inputs: pinned arithmetic, annotation poses, model points, table."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b.arithmetic import (
    cross3,
    dot3,
    matmul3,
    norm3,
    rotate_rows,
    transform_points,
)
from grasppanda.evaluation.graspnet1b.models import read_ply_vertices, voxel_downsample
from grasppanda.evaluation.graspnet1b.poses import (
    AnnotatedObject,
    euler_to_matrix,
    matrix_to_euler,
    pose_from_annotation,
    quaternion_to_matrix,
    read_annotation,
)
from grasppanda.evaluation.graspnet1b.protocol import GRASPNET1B_SV_V1, Protocol, TableSlab
from grasppanda.evaluation.graspnet1b.scene import table_lattice

from .eval_synthetic import annotation_xml, write_ply

# --------------------------------------------------------------------- arithmetic


def test_products_agree_with_numpy_to_rounding():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    assert np.allclose(dot3(a, b), np.einsum("ij,ij->i", a, b), rtol=1e-14, atol=1e-15)
    assert np.allclose(norm3(a), np.linalg.norm(a, axis=1), rtol=1e-14)
    assert np.allclose(cross3(a, b), np.cross(a, b), rtol=1e-14, atol=1e-15)
    m, n = rng.normal(size=(7, 3, 3)), rng.normal(size=(7, 3, 3))
    assert np.allclose(matmul3(m, n), m @ n, rtol=1e-13, atol=1e-15)
    assert np.allclose(matmul3(m[:1], n), m[:1] @ n, rtol=1e-13, atol=1e-15)  # broadcasting
    points = rng.normal(size=(7, 11, 3))
    assert np.allclose(rotate_rows(points, m), points @ m, rtol=1e-13, atol=1e-15)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = m[0], [0.1, -0.2, 0.3]
    assert np.allclose(transform_points(pose, a), a @ m[0].T + pose[:3, 3], rtol=1e-13)


def test_sums_run_left_to_right():
    """(1e16 + 1) + 1 loses both ones; 1e16 + (1 + 1) keeps them. The order is pinned."""
    a = np.array([[1e8, 1.0, 1.0]])
    assert dot3(a, a)[0] == (1e16 + 1.0) + 1.0 != 1e16 + (1.0 + 1.0)
    row = np.array([[[1e8, 1.0, 1.0]]])
    column = np.array([[[1e8], [1.0], [1.0]]]) * np.ones((1, 3, 3))
    assert (rotate_rows(row, column) == (1e16 + 1.0) + 1.0).all()
    assert (matmul3(row * np.ones((1, 3, 3)), column) == (1e16 + 1.0) + 1.0).all()
    matrix = np.array([[1e8, 1.0, 1.0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    assert transform_points(matrix, a)[0, 0] == ((1e16 + 1.0) + 1.0) + 1.0


def test_transform_points_widens_a_float32_pose_without_rounding_it_again():
    rng = np.random.default_rng(1)
    pose = rng.normal(size=(4, 4)).astype(np.float32)
    points = rng.normal(size=(20, 3))
    out = transform_points(pose, points)
    assert out.dtype == np.float64
    assert np.array_equal(out, transform_points(pose.astype(np.float64), points))


# --------------------------------------------------------------------------- poses


def test_euler_detour_reproduces_the_rotation():
    rng = np.random.default_rng(2)
    for quaternion in rng.normal(size=(200, 4)):
        direct = np.array(quaternion_to_matrix(*quaternion))
        assert np.allclose(direct @ direct.T, np.eye(3), atol=1e-12)
        assert math.isclose(np.linalg.det(direct), 1.0, abs_tol=1e-12)
        detour = np.array(euler_to_matrix(*matrix_to_euler(tuple(map(tuple, direct)))))
        assert np.allclose(detour, direct, atol=1e-12)


def test_known_rotations():
    quarter = math.sqrt(0.5)
    about_z = np.array(quaternion_to_matrix(quarter, 0.0, 0.0, quarter))
    assert np.allclose(about_z, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    assert np.allclose(euler_to_matrix(0.0, 0.0, math.pi / 2), about_z, atol=1e-15)
    assert np.allclose(matrix_to_euler(tuple(map(tuple, about_z))), (0.0, 0.0, math.pi / 2))
    assert quaternion_to_matrix(0.0, 0.0, 0.0, 0.0) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def test_gimbal_lock_branch():
    lock = euler_to_matrix(0.3, math.pi / 2, 0.0)
    angles = matrix_to_euler(lock)
    assert angles[2] == 0.0 and math.isclose(angles[1], math.pi / 2)
    assert np.allclose(euler_to_matrix(*angles), lock, atol=1e-12)


def test_pose_is_float32_and_differs_from_the_direct_conversion():
    entry = AnnotatedObject(7, (0.0426, 0.0804, 0.5144), (0.5707, 0.0606, -0.0763, 0.8153))
    pose = pose_from_annotation(entry)
    assert pose.dtype == np.float32 and pose.shape == (4, 4)
    assert np.array_equal(pose[3], [0, 0, 0, 1])
    assert np.array_equal(pose[:3, 3], np.float32(entry.position))
    direct = np.array(quaternion_to_matrix(*entry.quaternion))
    assert np.allclose(pose[:3, :3], direct, atol=1e-6)
    # The detour through degrees is not the identity in float64; storing float32 is the
    # documented behaviour, so the result must equal the *detour* rounded, bit for bit.
    angles = matrix_to_euler(tuple(map(tuple, direct)))
    radians = [a * (180.0 / math.pi) / 180.0 * math.pi for a in angles]
    assert np.array_equal(pose[:3, :3], np.float32(euler_to_matrix(*radians)))


def test_read_annotation(tmp_path):
    path = tmp_path / "0000.xml"
    path.write_text(
        annotation_xml([(9, (0.1, 0.2, 0.3), (1, 0, 0, 0)), (62, (-0.1, 0, 0.5), (0, 1, 0, 0))])
    )
    entries = read_annotation(path)
    assert [e.object_id for e in entries] == [9, 62]
    assert entries[1].position == (-0.1, 0.0, 0.5) and entries[1].quaternion == (0, 1, 0, 0)


@pytest.mark.parametrize(
    "body,message",
    [
        ("<scene><obj><obj_id>1</obj_id></obj></scene>", "pos_in_world"),
        (
            "<scene><obj><obj_id>1</obj_id><pos_in_world>0 0</pos_in_world>"
            "<ori_in_world>1 0 0 0</ori_in_world></obj></scene>",
            "3 value",
        ),
        ("<scene><obj>", "not a valid annotation"),
    ],
)
def test_read_annotation_rejects_malformed_files(tmp_path, body, message):
    path = tmp_path / "bad.xml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        read_annotation(path)


# -------------------------------------------------------------------------- models


def voxel_reference(points: np.ndarray, size: float) -> np.ndarray:
    """Open3D's rule, one point at a time."""
    anchor = points.min(axis=0) - size * 0.5
    cells: dict[tuple[int, ...], list] = {}
    for point in points:
        key = tuple(math.floor(c) for c in (point - anchor) / size)
        entry = cells.setdefault(key, [np.zeros(3), 0])
        entry[0] = entry[0] + point
        entry[1] += 1
    return np.array([cells[key][0] / cells[key][1] for key in sorted(cells)])


def test_voxel_downsample_matches_the_point_by_point_rule():
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.05, 0.05, size=(4000, 3)).astype(np.float32).astype(np.float64)
    out = voxel_downsample(points, 0.008)
    assert np.array_equal(out, voxel_reference(points, 0.008))  # same order of accumulation
    assert len(out) < len(points)


def test_voxel_grid_is_anchored_half_a_voxel_below_the_minimum():
    points = np.array([[0.0, 0, 0], [0.003, 0, 0], [0.005, 0, 0], [0.011, 0, 0]])
    out = voxel_downsample(points, 0.008)  # cells [-0.004, 0.004), [0.004, 0.012)
    assert np.allclose(out[:, 0], [0.0015, 0.008])


def test_voxel_downsample_edge_cases():
    assert voxel_downsample(np.zeros((0, 3)), 0.01).shape == (0, 3)
    assert np.array_equal(voxel_downsample(np.ones((5, 3)), 0.01), np.ones((1, 3)))
    with pytest.raises(ValueError, match="positive"):
        voxel_downsample(np.ones((5, 3)), 0.0)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        voxel_downsample(np.ones((5, 2)), 0.01)
    with pytest.raises(ValueError, match="too small"):
        voxel_downsample(np.array([[0.0, 0, 0], [1e9, 1e9, 1e9]]), 1e-12)


def test_read_ply_vertices(tmp_path):
    points = np.random.default_rng(4).normal(size=(37, 3))
    write_ply(tmp_path / "model.ply", points)
    vertices = read_ply_vertices(tmp_path / "model.ply")
    assert vertices.dtype == np.float64
    assert np.array_equal(vertices, points.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize(
    "header,message",
    [
        ("plx\n", "not a PLY"),
        ("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n", "binary"),
        ("ply\nformat binary_little_endian 1.0\nelement face 1\nend_header\n", "must be 'vertex'"),
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            "property list uchar int k\nend_header\n",
            "fixed-size",
        ),
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nend_header\n",
            r"lacks \['z'\]",
        ),
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 9\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n",
            "expected 9 vertices",
        ),
    ],
)
def test_read_ply_rejects_what_it_does_not_support(tmp_path, header, message):
    path = tmp_path / "bad.ply"
    path.write_bytes(header.encode("ascii") + b"\x00" * 12)
    with pytest.raises(ValueError, match=message):
        read_ply_vertices(path)


# ------------------------------------------------------------------ table, protocol


def test_table_lattice():
    table = table_lattice(GRASPNET1B_SV_V1.table)
    assert table.shape == (125 * 125 * 6, 3)
    assert np.allclose(table.min(axis=0), [-0.5, -0.5, -0.05])
    assert np.allclose(table.max(axis=0), [0.5, 0.5, 0.0])
    assert len(np.unique(table, axis=0)) == len(table)
    tiny = table_lattice(TableSlab(size=(0.1, 0.1, 0.1), offset=(0.0, 0.0, 0.0), grid_size=0.05))
    assert sorted(set(tiny[:, 0])) == [0.0, 0.1]  # both ends are lattice points


def test_protocol_owns_the_constants():
    p = GRASPNET1B_SV_V1
    assert (p.name, p.top_k, p.per_object_top, p.empty_threshold) == (
        "graspnet1b-sv-v1",
        50,
        10,
        10,
    )
    assert p.frictions == (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    assert (p.friction_column(0.8), p.friction_column(0.4)) == (3, 1)
    assert p.nms_rotation == 30.0 / 180 * math.pi and p.nms_translation == 0.03
    gripper = p.gripper
    assert (gripper.max_width, gripper.finger_width, gripper.finger_height, gripper.depth_base) == (
        0.1, 0.01, 0.02, 0.02,
    )  # fmt: skip
    assert p.as_dict()["gripper"]["max_width"] == 0.1
    with pytest.raises(ValueError, match="ascending"):
        Protocol(name="x", frictions=(0.4, 0.2))
    with pytest.raises(ValueError, match="not a column"):
        Protocol(name="x", reported=(("AP0.5", 0.5),))
