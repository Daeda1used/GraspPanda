"""Ground truth of one GraspNet-1Billion scene, as the evaluator consumes it.

Layout read (never written)::

    scenes/scene_XXXX/<camera>/annotations/NNNN.xml   object ids and poses per frame
    scenes/scene_XXXX/<camera>/camera_poses.npy       (256, 4, 4) camera w.r.t. frame 0
    scenes/scene_XXXX/<camera>/cam0_wrt_table.npy     (4, 4) frame-0 camera w.r.t. table

Two conventions of the reference tool are kept:

* the object list and its order come from ``annotations/0000.xml``; a later frame
  contributes poses only. In the released dataset every frame lists the same objects
  in the same order (checked for all test scenes and both cameras), so a frame that
  does not is reported as an error instead of being paired up by position;
* the synthetic table is a lattice in the table frame, moved into each camera frame
  with ``inv(cam0_wrt_table @ camera_pose)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .arithmetic import transform_points
from .cache import ObjectAsset, ObjectCache
from .poses import pose_from_annotation, read_annotation
from .protocol import Protocol, TableSlab

__all__ = ["SceneGroundTruth", "scene_directory", "scene_object_ids", "table_lattice"]

CAMERAS = ("realsense", "kinect")


def scene_directory(data_root: str | Path, scene: int, camera: str) -> Path:
    """``<root>/scenes/scene_0100/realsense``; rejects unknown cameras."""
    if camera not in CAMERAS:
        raise ValueError(f"unknown GraspNet-1Billion camera '{camera}'; expected one of {CAMERAS}")
    return Path(data_root) / "scenes" / f"scene_{scene:04d}" / camera


def scene_object_ids(data_root: str | Path, scene: int, camera: str) -> list[int]:
    """Object ids of a scene in the order of its first annotation file."""
    path = scene_directory(data_root, scene, camera) / "annotations" / "0000.xml"
    return [entry.object_id for entry in read_annotation(path)]


def table_lattice(slab: TableSlab) -> np.ndarray:
    """``(T, 3)`` float64 lattice points of the table slab, in the table frame."""
    axes = [
        np.linspace(0, size, int(size / slab.grid_size)) + offset
        for size, offset in zip(slab.size, slab.offset)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _matmul4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` for ``(4, 4)`` float64 matrices, each entry summed over k in order."""
    product = a[:, 0:1] * b[0:1, :]
    for k in range(1, 4):
        product = product + a[:, k : k + 1] * b[k : k + 1, :]
    return product


class SceneGroundTruth:
    """Objects, poses and table of one scene seen by one camera.

    Args:
        data_root: Dataset root.
        scene: Scene index.
        camera: ``realsense`` or ``kinect``.
        cache: Source of the per-object scoring data.
        protocol: Supplies the table slab.
    """

    def __init__(
        self, data_root: str | Path, scene: int, camera: str, cache: ObjectCache, protocol: Protocol
    ) -> None:
        self.directory = scene_directory(data_root, scene, camera)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"scene directory {self.directory} does not exist")
        self.object_ids = scene_object_ids(data_root, scene, camera)
        if not self.object_ids:
            raise ValueError(f"{self.directory / 'annotations' / '0000.xml'} lists no objects")
        self.objects: list[ObjectAsset] = [cache.load(object_id) for object_id in self.object_ids]
        self._camera_poses = np.load(self.directory / "camera_poses.npy").astype(np.float64)
        self._camera_to_table = np.load(self.directory / "cam0_wrt_table.npy").astype(np.float64)
        self._table = None if protocol.table is None else table_lattice(protocol.table)

    list_object_ids = staticmethod(scene_object_ids)

    def frame(self, frame: int) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Object poses (float32, object to camera) and camera-frame table points of a frame."""
        path = self.directory / "annotations" / f"{frame:04d}.xml"
        entries = read_annotation(path)
        listed = [entry.object_id for entry in entries]
        if listed != self.object_ids:
            raise ValueError(
                f"{path} lists objects {listed}, but the scene's first frame lists "
                f"{self.object_ids}; the evaluator pairs poses with objects by that order"
            )
        if not 0 <= frame < len(self._camera_poses):
            raise ValueError(
                f"frame {frame} has no camera pose ({len(self._camera_poses)} poses in "
                f"{self.directory / 'camera_poses.npy'})"
            )
        poses = [pose_from_annotation(entry) for entry in entries]
        if self._table is None:
            return poses, None
        table_to_camera = np.linalg.inv(_matmul4(self._camera_to_table, self._camera_poses[frame]))
        return poses, transform_points(table_to_camera, self._table)
