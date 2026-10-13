"""The synthetic example scene and the file readers of the inference API."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image

from grasppanda.apis import InputError, example_scene, io
from grasppanda.transforms.frame import transform_points

# ------------------------------------------------------------------ the example scene


def cloud_of(scene) -> tuple[np.ndarray, np.ndarray]:
    """Back-projected points of the scene and the mask value of each."""
    v, u = np.nonzero(scene.depth > 0)
    z = scene.depth[v, u] / scene.depth_scale
    k = scene.intrinsics
    points = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)
    return points, scene.mask[v, u]


def test_the_scene_is_what_a_depth_camera_would_deliver():
    scene = example_scene(size=(320, 180))
    assert scene.depth.shape == (180, 320) and scene.depth.dtype == np.uint16
    assert scene.rgb.shape == (180, 320, 3) and scene.rgb.dtype == np.uint8
    assert scene.mask.shape == (180, 320) and scene.depth_scale == 1000.0
    assert scene.objects == ("box", "tall box", "can", "tube", "ball", "flat box")
    assert set(np.unique(scene.mask)) == set(range(7))  # table and six visible primitives
    measured = scene.depth[scene.depth > 0] / 1000.0
    assert measured.min() > 0.35 and measured.max() < 1.0  # a table-top at arm's length
    assert (scene.depth == 0).any()  # beyond the table edge the sensor sees nothing
    assert np.allclose(scene.intrinsics, [[231.25, 0, 160], [0, 231.25, 90], [0, 0, 1]])
    rotation = scene.table_from_camera[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3)) and np.isclose(
        np.linalg.det(rotation), 1.0
    )


def test_geometry_agrees_with_the_calibration():
    scene = example_scene(size=(320, 180))
    points, mask = cloud_of(scene)
    on_table = transform_points(points, scene.table_from_camera)
    assert (
        np.abs(on_table[mask == 0][:, 2]).max() < 1.5e-3
    )  # the table top is z = 0 (1 mm depth steps)
    objects = on_table[mask > 0]
    assert (
        objects[:, 2].min() > -1.5e-3 and 0.10 < objects[:, 2].max() < 0.12
    )  # the tall box: 11 cm
    ball = on_table[mask == 5]  # centre (0.10, -0.09, 0.035), radius 3.5 cm
    radius = np.linalg.norm(ball - np.array([0.10, -0.09, 0.035]), axis=1)
    assert np.abs(radius - 0.035).max() < 2e-3


def test_the_scene_is_deterministic_and_seeded():
    first, second = example_scene(size=(160, 90)), example_scene(size=(160, 90))
    assert np.array_equal(first.depth, second.depth) and np.array_equal(first.rgb, second.rgb)
    moved = example_scene(3, size=(160, 90))
    assert not np.array_equal(first.depth, moved.depth)
    assert np.array_equal(moved.depth, example_scene(3, size=(160, 90)).depth)


def test_saved_files_read_back(tmp_path):
    scene = example_scene(size=(160, 90))
    files = scene.save(tmp_path / "scene")
    assert np.array_equal(io.read_depth(files["depth"]), scene.depth)
    assert io.read_depth(files["depth"]).dtype == np.uint16  # 16 bit survive the PNG
    assert np.array_equal(io.read_rgb(files["rgb"]), scene.rgb)
    assert np.array_equal(io.read_mask(files["mask"]), scene.mask)
    assert np.allclose(io.read_intrinsics(str(files["intrinsics"])), [115.625, 115.625, 80, 45])
    assert np.allclose(np.loadtxt(files["table_from_camera"]), scene.table_from_camera)


# -------------------------------------------------------------------------- readers


def test_intrinsics_from_text_and_files(tmp_path):
    assert np.array_equal(io.read_intrinsics("925, 925,640,360"), [925, 925, 640, 360])
    matrix = np.array([[900.0, 0, 640], [0, 901.0, 360], [0, 0, 1]])
    np.savetxt(tmp_path / "k.txt", matrix)
    np.save(tmp_path / "k.npy", matrix)
    (tmp_path / "k.json").write_text("[[900, 0, 640], [0, 901, 360], [0, 0, 1]]")
    for name in ("k.txt", "k.npy", "k.json"):
        assert np.array_equal(io.read_intrinsics(str(tmp_path / name)), matrix), name
    with pytest.raises(InputError, match="neither a file nor numbers.*fx,fy,cx,cy"):
        io.read_intrinsics("realsense")
    with pytest.raises(InputError, match="hold 3 numbers"):
        io.read_intrinsics("1,2,3")


def test_depth_must_be_a_depth_image(tmp_path):
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(tmp_path / "colour.png")
    with pytest.raises(InputError, match="picture of depth, not a depth image"):
        io.read_depth(tmp_path / "colour.png")
    with pytest.raises(InputError, match="depth image not found"):
        io.read_depth(tmp_path / "none.png")
    np.save(tmp_path / "metres.npy", np.full((4, 5), 0.5, dtype=np.float32))
    assert io.read_depth(tmp_path / "metres.npy").dtype == np.float32


POINTS = np.array([[0.1, -0.2, 0.5], [0.0, 0.25, 0.75], [-0.3, 0.0, 1.0]])
COLOURS = np.array([[255, 0, 0], [0, 128, 0], [10, 20, 30]], dtype=np.uint8)


def test_point_clouds_from_npy(tmp_path):
    np.save(tmp_path / "xyz.npy", POINTS.astype(np.float32))
    points, colours = io.read_points(tmp_path / "xyz.npy")
    assert np.allclose(points, POINTS) and points.dtype == np.float64 and colours is None
    np.save(tmp_path / "xyzrgb.npy", np.concatenate([POINTS, COLOURS / 255.0], axis=1))
    points, colours = io.read_points(tmp_path / "xyzrgb.npy")
    assert np.allclose(points, POINTS) and np.array_equal(colours, COLOURS)
    np.save(tmp_path / "organised.npy", np.zeros((4, 5, 3)))
    assert io.read_points(tmp_path / "organised.npy")[0].shape == (20, 3)
    np.save(tmp_path / "flat.npy", np.zeros(9))
    with pytest.raises(InputError, match=r"expected \(N, 3\) points"):
        io.read_points(tmp_path / "flat.npy")
    with pytest.raises(InputError, match="unknown point cloud format '.xyz'"):
        (tmp_path / "cloud.xyz").write_text("0 0 0\n")
        io.read_points(tmp_path / "cloud.xyz")


def test_point_clouds_from_ply(tmp_path):
    header = (
        "ply\nformat {layout} 1.0\ncomment made by a test\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    rows = "".join(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n" for p, c in zip(POINTS, COLOURS))
    (tmp_path / "ascii.ply").write_text(header.format(layout="ascii") + rows + "3 0 1 2\n")
    for layout, order in (("binary_little_endian", "<"), ("binary_big_endian", ">")):
        body = b"".join(
            struct.pack(f"{order}fffBBB", *p, *c) for p, c in zip(POINTS, COLOURS.tolist())
        )
        payload = (
            header.format(layout=layout).encode() + body + struct.pack(f"{order}Biii", 3, 0, 1, 2)
        )
        (tmp_path / f"{layout}.ply").write_bytes(payload)
    for name in ("ascii", "binary_little_endian", "binary_big_endian"):
        points, colours = io.read_points(tmp_path / f"{name}.ply")
        assert np.allclose(points, POINTS, atol=1e-6), name
        assert np.array_equal(colours, COLOURS), name
    (tmp_path / "broken.ply").write_text("not a ply\n")
    with pytest.raises(InputError, match="is not a PLY file"):
        io.read_points(tmp_path / "broken.ply")


def test_point_clouds_from_pcd(tmp_path):
    packed = (
        (COLOURS[:, 0].astype(np.uint32) << 16)
        | (COLOURS[:, 1].astype(np.uint32) << 8)
        | COLOURS[:, 2]
    )
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n"
        "SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA {layout}\n"
    )
    rows = "".join(f"{p[0]} {p[1]} {p[2]} {w}\n" for p, w in zip(POINTS, packed))
    (tmp_path / "ascii.pcd").write_text(header.format(layout="ascii") + rows)
    body = b"".join(struct.pack("<fffI", *p, int(w)) for p, w in zip(POINTS, packed))
    (tmp_path / "binary.pcd").write_bytes(header.format(layout="binary").encode() + body)
    for name in ("ascii", "binary"):
        points, colours = io.read_points(tmp_path / f"{name}.pcd")
        assert np.allclose(points, POINTS, atol=1e-6), name
        assert np.array_equal(colours, COLOURS), name
    as_float = header.replace("TYPE F F F U", "TYPE F F F F").format(layout="binary")
    body = b"".join(struct.pack("<fffI", *p, int(w)) for p, w in zip(POINTS, packed))
    (tmp_path / "pcl.pcd").write_bytes(as_float.encode() + body)  # PCL keeps the word in a float
    assert np.array_equal(io.read_points(tmp_path / "pcl.pcd")[1], COLOURS)
    (tmp_path / "packed.pcd").write_bytes(header.format(layout="binary_compressed").encode())
    with pytest.raises(InputError, match="binary_compressed.*not supported"):
        io.read_points(tmp_path / "packed.pcd")
