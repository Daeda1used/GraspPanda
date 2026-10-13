"""Object model points: PLY vertices and Open3D-compatible voxel down-sampling.

The reference evaluator assigns grasps to objects and checks collisions against model
vertices that were down-sampled with ``open3d.geometry.PointCloud.voxel_down_sample``.
Which vertices fall into one voxel, and the order in which they are averaged, decide
the resulting coordinates to the last bit, so the rule is reproduced here with NumPy
only and compared against Open3D in the integration tests:

* voxels are anchored at ``min(points) - voxel_size / 2`` per axis;
* a point belongs to voxel ``floor((point - anchor) / voxel_size)``;
* a voxel yields the mean of its points, accumulated in input order.

Open3D returns the voxels in hash-table order. That order carries no meaning and is
not reproduced: points come back sorted by voxel index. No stage of the evaluator
depends on the order of the points inside one object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_ply_vertices", "voxel_downsample"]

_PLY_SCALARS = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_PLY_HEADER_LIMIT = 1 << 16


def read_ply_vertices(path: str | Path) -> np.ndarray:
    """Read the ``x, y, z`` vertex coordinates of a binary PLY file as ``(N, 3)`` float64.

    Only what the dataset's ``nontextured.ply`` files need is supported: a binary
    ``vertex`` element that comes first and has fixed-size properties. Anything else
    raises with the reason instead of guessing.
    """
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(_PLY_HEADER_LIMIT)
    end = head.find(b"end_header")
    newline = head.find(b"\n", end)
    if not head.startswith(b"ply") or end < 0 or newline < 0:
        raise ValueError(f"{path} is not a PLY file (no 'ply' ... 'end_header' header)")
    lines = [line.split() for line in head[:end].decode("ascii").splitlines()[1:]]
    lines = [line for line in lines if line and line[0] not in ("comment", "obj_info")]

    if not lines or lines[0][0] != "format":
        raise ValueError(f"{path}: the PLY header has no format line")
    byte_order = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(lines[0][1])
    if byte_order is None:
        raise ValueError(f"{path}: only binary PLY files are supported, got '{lines[0][1]}'")
    if len(lines) < 2 or lines[1][:2] != ["element", "vertex"]:
        raise ValueError(f"{path}: the first PLY element must be 'vertex'")
    count = int(lines[1][2])

    fields = []
    for line in lines[2:]:
        if line[0] == "element":
            break
        if line[0] != "property":
            raise ValueError(f"{path}: unexpected PLY header line '{' '.join(line)}'")
        if line[1] == "list" or line[1] not in _PLY_SCALARS:
            raise ValueError(
                f"{path}: vertex property '{' '.join(line[1:])}' is not a fixed-size scalar"
            )
        fields.append((line[-1], byte_order + _PLY_SCALARS[line[1]]))
    missing = {"x", "y", "z"} - {name for name, _ in fields}
    if missing:
        raise ValueError(f"{path}: the vertex element lacks {sorted(missing)}")

    vertices = np.fromfile(path, dtype=np.dtype(fields), count=count, offset=newline + 1)
    if len(vertices) != count:
        raise ValueError(f"{path}: expected {count} vertices, the file holds {len(vertices)}")
    return np.stack([vertices[axis].astype(np.float64) for axis in "xyz"], axis=1)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Average the points of each occupied voxel, with Open3D's voxel rule.

    Args:
        points: ``(N, 3)``; converted to float64 as Open3D does.
        voxel_size: Voxel edge, same unit as ``points``.

    Returns:
        ``(M, 3)`` float64 voxel means, sorted by voxel index (x, then y, then z).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if len(points) == 0:
        return points.copy()
    anchor = points.min(axis=0) - voxel_size * 0.5
    cell = np.floor((points - anchor) / voxel_size)
    cells = cell.max(axis=0) + 1
    if not np.isfinite(cells).all() or cells.prod() >= 2.0**62:
        raise ValueError(
            f"voxel_size {voxel_size} is too small for the extent of the points: "
            f"the grid would have {cells[0]:.3g} x {cells[1]:.3g} x {cells[2]:.3g} cells"
        )
    index, extent = cell.astype(np.int64), cells.astype(np.int64)
    key = (index[:, 0] * extent[1] + index[:, 1]) * extent[2] + index[:, 2]
    _, voxel = np.unique(key, return_inverse=True)  # ascending key = sorted by (x, y, z) index
    count = np.bincount(voxel)
    # bincount adds its weights one by one in input order: the accumulation Open3D performs.
    total = np.stack([np.bincount(voxel, weights=points[:, axis]) for axis in range(3)], axis=1)
    return total / count[:, None]
