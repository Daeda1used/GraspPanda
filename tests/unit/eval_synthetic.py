"""Analytic toy objects and a miniature GraspNet-style dataset for the evaluator tests.

Everything here is generated from formulas, so the unit suite covers the evaluator's
logic without the real dataset: signed-distance grids of a sphere and a box, their
surface points, and a tiny dataset tree with the files the evaluator reads
(annotation XML, camera poses, binary PLY models, Dex-Net-style pickles).

The pickles are written from stand-in classes that only *carry the names* of the
upstream ones; see :func:`upstream_stand_ins`.
"""

from __future__ import annotations

import contextlib
import json
import pickle
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from grasppanda.evaluation.graspnet1b.cache import ObjectAsset
from grasppanda.evaluation.graspnet1b.sdf import SdfGrid

RESOLUTION = 0.002
CELLS = 64
SPHERE_RADIUS = 0.03
BOX_HALF = (0.02, 0.03, 0.015)
#: Table top in the camera frame of the toy scene (camera looks straight down).
TABLE_DEPTH = 0.5
SPHERE_CENTRE = np.array([-0.08, 0.0, TABLE_DEPTH - SPHERE_RADIUS])
BOX_CENTRE = np.array([0.08, 0.0, TABLE_DEPTH - BOX_HALF[2]])


def _upstream(toolkit: str) -> dict[str, tuple[str, ...]]:
    """Module paths and class names an official ``dex_models`` pickle refers to."""
    return {
        f"{toolkit}.utils.dexnet.grasping.graspable_object": ("GraspableObject3D",),
        f"{toolkit}.utils.dexnet.grasping.meshpy.sdf": ("Sdf3D",),
        f"{toolkit}.utils.dexnet.grasping.meshpy.mesh": ("Mesh3D",),
        "autolab_core.rigid_transformations": ("SimilarityTransform", "RigidTransform"),
    }


# ---------------------------------------------------------------------------- shapes


def grid_centres(
    cells: int = CELLS, resolution: float = RESOLUTION
) -> tuple[np.ndarray, np.ndarray]:
    """Object-frame positions of all voxels of a cube grid centred on the origin, and its origin."""
    origin = np.full(3, -(cells - 1) / 2 * resolution)
    index = np.stack(np.meshgrid(*[np.arange(cells)] * 3, indexing="ij"), axis=-1)
    return origin + resolution * index, origin


def sphere_distance(points: np.ndarray, radius: float = SPHERE_RADIUS) -> np.ndarray:
    return np.linalg.norm(points, axis=-1) - radius


def box_distance(points: np.ndarray, half: tuple[float, float, float] = BOX_HALF) -> np.ndarray:
    excess = np.abs(points) - np.array(half)
    outside = np.linalg.norm(np.maximum(excess, 0.0), axis=-1)
    return outside + np.minimum(excess.max(axis=-1), 0.0)


def curved_box_distance(
    points: np.ndarray, half: tuple[float, float, float] = BOX_HALF
) -> np.ndarray:
    """Box distance with a little curvature added: same surface, same sign, not linear.

    The contact search fits a parabola to three distance samples and gives up when the
    fit is flat. Along a straight line towards a flat face the exact distance is linear,
    so on a perfect box no contact is ever found; real grids are never that perfect.
    """
    distance = box_distance(points, half)
    return distance + 4.0 * distance * distance


def make_grid(
    distance, *, cells: int = CELLS, resolution: float = RESOLUTION, seed: int = 0
) -> SdfGrid:
    """Sample an analytic distance function into an :class:`SdfGrid` with a seeded surface sample."""
    centres, origin = grid_centres(cells, resolution)
    values = np.ascontiguousarray(distance(centres), dtype=np.float64)
    threshold = resolution * np.sqrt(2) / 2
    surface = np.argwhere(np.abs(values) < threshold)
    sample = surface[np.random.default_rng(seed).integers(0, len(surface), size=1000)]
    return SdfGrid(
        values=values,
        grid_to_object_scale=resolution,
        grid_to_object_translation=origin,
        object_to_grid_scale=1.0 / resolution,
        object_to_grid_translation=-(1.0 / resolution) * origin,
        surface_threshold=float(threshold),
        surface_sample=sample.astype(np.int64),
    )


def sphere_points(radius: float = SPHERE_RADIUS, count: int = 6000) -> np.ndarray:
    """Evenly spread points on a sphere (Fibonacci lattice)."""
    k = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * k / count)
    azimuth = np.pi * (1 + 5**0.5) * k
    return radius * np.stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], axis=1
    )


def box_points(half: tuple[float, float, float] = BOX_HALF, spacing: float = 0.002) -> np.ndarray:
    """A lattice of points on the six faces of a box."""
    faces = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        ticks = [np.arange(-half[a], half[a] + spacing / 2, spacing) for a in others]
        u, v = (grid.ravel() for grid in np.meshgrid(*ticks, indexing="ij"))
        for side in (-1.0, 1.0):
            face = np.zeros((len(u), 3))
            face[:, others[0]], face[:, others[1]], face[:, axis] = u, v, side * half[axis]
            faces.append(face)
    return np.concatenate(faces)


def sphere_asset(object_id: int = 0) -> ObjectAsset:
    return ObjectAsset(object_id, make_grid(sphere_distance, seed=object_id), sphere_points())


def box_asset(object_id: int = 1) -> ObjectAsset:
    return ObjectAsset(object_id, make_grid(curved_box_distance, seed=object_id), box_points())


# ---------------------------------------------------------------------------- grasps


def grasp_row(
    point: np.ndarray,
    *,
    approach: tuple[float, float, float] = (0.0, 0.0, 1.0),
    closing: tuple[float, float, float] = (0.0, 1.0, 0.0),
    width: float = 0.08,
    depth: float = 0.02,
    score: float = 0.5,
) -> np.ndarray:
    """One GraspNet 17-vector. ``point`` is the grasp point; the fingertips are ``depth`` ahead."""
    x = np.array(approach, dtype=np.float64)
    y = np.array(closing, dtype=np.float64)
    rotation = np.stack([x, y, np.cross(x, y)], axis=1)
    return np.concatenate([[score, width, 0.02, depth], rotation.reshape(9), point, [-1.0]])


def top_grasp(centre: np.ndarray, **kwargs) -> np.ndarray:
    """A top-down grasp whose fingertip line passes through ``centre`` (camera looks down +z)."""
    depth = kwargs.get("depth", 0.02)
    return grasp_row(centre - np.array([0.0, 0.0, depth]), **kwargs)


# --------------------------------------------------------------------------- dataset


@contextlib.contextmanager
def stand_in_modules(layout: dict[str, tuple[str, ...]]) -> Iterator[dict[str, type]]:
    """Temporarily make empty classes importable under the given module paths."""
    created: dict[str, types.ModuleType] = {}
    classes: dict[str, type] = {}
    for module_name, names in layout.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):  # parent packages must be importable too
            name = ".".join(parts[:depth])
            if name not in created:
                created[name] = types.ModuleType(name)
                created[name].__path__ = []
        for name in names:
            classes[name] = type(name, (), {"__module__": module_name})
            setattr(created[module_name], name, classes[name])
    previous = {name: sys.modules.get(name) for name in created}
    sys.modules.update(created)
    try:
        yield classes
    finally:
        for name, module in previous.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


def upstream_stand_ins(toolkit: str = "graspnetAPI"):
    """Stand-ins for the classes an official ``dex_models`` pickle names, for dumping only.

    ``pickle`` records a class by module path and name. To write a file of the same
    shape as the official ones, classes with those names must be importable while
    dumping. They are gone again on exit, so reading such a pickle back proves that no
    upstream module is needed.
    """
    return stand_in_modules(_upstream(toolkit))


def dexnet_pickle_bytes(
    grid: SdfGrid, *, rotation: np.ndarray | None = None, toolkit: str = "graspnetAPI"
) -> bytes:
    """Serialise a grid the way an official ``dex_models`` pickle of ``toolkit`` stores it."""
    rotation = np.eye(3) if rotation is None else rotation
    with upstream_stand_ins(toolkit) as classes:

        def similarity(scale: float, translation: np.ndarray, source: str, target: str):
            transform = classes["SimilarityTransform"]()
            transform.__dict__.update(
                _scale=float(scale), _rotation=rotation.copy(), _translation=np.array(translation),
                _from_frame=source, _to_frame=target,
            )  # fmt: skip
            return transform

        field = classes["Sdf3D"]()
        field.__dict__.update(
            data_=grid.values,
            origin_=np.array(grid.grid_to_object_translation),
            resolution_=float(grid.grid_to_object_scale),
            dims_=grid.values.shape,
            surface_thresh_=np.float64(grid.surface_threshold),
            use_abs_=False,
            T_grid_world_=similarity(
                grid.grid_to_object_scale, grid.grid_to_object_translation, "grid", "world"
            ),
            T_world_grid_=similarity(
                grid.object_to_grid_scale, grid.object_to_grid_translation, "world", "grid"
            ),
            surface_for_signed_val=grid.surface_sample,
        )
        mesh = classes["Mesh3D"]()
        mesh.__dict__.update(vertices_=np.zeros((3, 3)), center_of_mass_=np.zeros(3))
        graspable = classes["GraspableObject3D"]()
        graspable.__dict__.update(sdf_=field, mesh_=mesh, key_="", mass_=1.0)
        return pickle.dumps(graspable, protocol=4)  # bytes opcodes, like the official files


def write_ply(path: Path, points: np.ndarray) -> None:
    """Binary PLY with the vertex layout of the dataset's ``nontextured.ply`` plus one face."""
    vertex = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
         ("flags", "<i4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")]
    )  # fmt: skip
    data = np.zeros(len(points), dtype=vertex)
    data["x"], data["y"], data["z"] = points.astype(np.float32).T
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", "comment synthetic", f"element vertex {len(points)}"]
        + [f"property {'float' if kind == '<f4' else 'int' if kind == '<i4' else 'uchar'} {name}"
           for name, kind in vertex.descr]
        + ["element face 1", "property list uchar int vertex_indices", "end_header", ""]
    )  # fmt: skip
    face = np.array([3], dtype="u1").tobytes() + np.array([0, 1, 2], dtype="<i4").tobytes()
    path.write_bytes(header.encode("ascii") + data.tobytes() + face)


def annotation_xml(objects: list[tuple[int, np.ndarray, tuple[float, float, float, float]]]) -> str:
    entries = "".join(
        f"<obj><obj_id>{object_id}</obj_id><obj_name>toy.ply</obj_name><obj_path>toy</obj_path>"
        f"<pos_in_world>{position[0]:.4f} {position[1]:.4f} {position[2]:.4f}</pos_in_world>"
        f"<ori_in_world>{' '.join(f'{q:.4f}' for q in quaternion)}</ori_in_world></obj>"
        for object_id, position, quaternion in objects
    )
    return f'<?xml version="1.0" ?><scene>{entries}</scene>'


def write_dataset(root: Path, *, scenes: tuple[int, ...] = (0,), frames: int = 3) -> Path:
    """Create a miniature dataset: a sphere (object 0) and a box (object 1) on a table.

    The camera looks straight down from 0.5 m; all frames share the same poses.
    """
    assets = {0: sphere_asset(0), 1: box_asset(1)}
    for object_id, asset in assets.items():
        (root / "dex_models").mkdir(parents=True, exist_ok=True)
        (root / "dex_models" / f"{object_id:03d}.pkl").write_bytes(dexnet_pickle_bytes(asset.sdf))
        model = root / "models" / f"{object_id:03d}"
        model.mkdir(parents=True, exist_ok=True)
        write_ply(model / "nontextured.ply", asset.model_points)
    camera_in_table = np.array(
        [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, TABLE_DEPTH], [0, 0, 0, 1.0]]
    )
    xml = annotation_xml([(0, SPHERE_CENTRE, (1, 0, 0, 0)), (1, BOX_CENTRE, (1, 0, 0, 0))])
    for scene in scenes:
        for camera in ("kinect", "realsense"):
            folder = root / "scenes" / f"scene_{scene:04d}" / camera
            (folder / "annotations").mkdir(parents=True, exist_ok=True)
            for frame in range(frames):
                (folder / "annotations" / f"{frame:04d}.xml").write_text(xml)
            np.save(folder / "camera_poses.npy", np.tile(np.eye(4), (frames, 1, 1)))
            np.save(folder / "cam0_wrt_table.npy", camera_in_table)
    return root


# ------------------------------------------------------------ GraspClutter6D layout

#: Instances of the toy clutter scene: (object id, centre in the camera frame, metres).
CLUTTER_INSTANCES = (
    (1, np.array([-0.12, 0.0, 0.47])),  # a sphere
    (1, np.array([0.0, 0.0, 0.47])),  # a second instance of the same sphere
    (2, np.array([0.12, 0.0, 0.485])),  # the box
)


def write_clutter_dataset(root: Path, *, scenes: tuple[int, ...] = (0,), frames: int = 2) -> Path:
    """Create a miniature dataset in the GraspClutter6D (BOP) layout.

    Object 1 is the sphere, object 2 the box; the sphere appears twice in every image.
    Poses are identical for all cameras and frames; translations are in millimetres.
    """
    assets = {1: sphere_asset(1), 2: box_asset(2)}
    (root / "dex_models").mkdir(parents=True, exist_ok=True)
    (root / "models_m").mkdir(parents=True, exist_ok=True)
    for object_id, asset in assets.items():
        payload = dexnet_pickle_bytes(asset.sdf, toolkit="graspclutter6dAPI")
        (root / "dex_models" / f"obj_{object_id:06d}.pkl").write_bytes(payload)
        write_ply(root / "models_m" / f"obj_{object_id:06d}.ply", asset.model_points)
    instances = [
        {
            "obj_id": object_id,
            "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
            "cam_t_m2c": (centre * 1000).tolist(),
        }
        for object_id, centre in CLUTTER_INSTANCES
    ]
    scene_gt = {str(image): instances for image in range(1, 4 * frames + 1)}
    for scene in scenes:
        folder = root / "scenes" / f"{scene:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scene_gt.json").write_text(json.dumps(scene_gt))
    return root
