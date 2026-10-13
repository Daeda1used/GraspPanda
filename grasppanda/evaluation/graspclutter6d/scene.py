"""Ground truth of one GraspClutter6D scene, as the evaluator consumes it.

The dataset follows the BOP layout. One ``scene_gt.json`` per scene holds, for each of
its 52 images, the object instances with their model-to-camera poses (translation in
millimetres). Images interleave four cameras::

    image_id = 4 * frame + c,   frame = 0 .. 12,
    c = 1 realsense-d415, 2 realsense-d435, 3 azure-kinect, 4 zivid

so a :class:`~grasppanda.structures.SampleKey` addresses a frame by camera name and
frame index, never by directory.

Conventions of the reference tool that are kept:

* the instance list and its order come from the camera's first image; later frames
  contribute poses, paired by position. A frame that lists different objects is an
  error here instead of a silent mismatch;
* poses are used as float64, straight from the JSON (no Euler detour, no float32);
* there is no table in the collision scene;
* a scene can contain several instances of one object id; each is its own entry.

Layout read (never written)::

    scenes/<6-digit scene>/scene_gt.json
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from grasppanda.evaluation.base import prediction_path
from grasppanda.evaluation.graspnet1b.cache import ObjectAsset, ObjectCache
from grasppanda.evaluation.graspnet1b.protocol import Protocol
from grasppanda.structures.common import SampleKey

__all__ = [
    "CAMERAS",
    "ClutterGroundTruth",
    "export_official_layout",
    "image_id",
    "prediction_file",
]

#: Camera name -> offset of its images inside a group of four.
CAMERAS = {"realsense-d415": 1, "realsense-d435": 2, "azure-kinect": 3, "zivid": 4}
_MILLIMETRE = 1000


def image_id(camera: str, frame: int) -> int:
    """BOP image id of a camera's frame; rejects unknown cameras and negative frames."""
    if camera not in CAMERAS:
        raise ValueError(
            f"unknown GraspClutter6D camera '{camera}'; expected one of {sorted(CAMERAS)}"
        )
    if frame < 0:
        raise ValueError(f"frame index must not be negative, got {frame}")
    return 4 * frame + CAMERAS[camera]


def prediction_file(root: str | Path, key: SampleKey) -> Path:
    """``<root>/000042/zivid/000004.npy``: the dump layout of the official GraspClutter6D tool."""
    name = f"{image_id(key.camera, key.frame):06d}.npy"
    return Path(root) / f"{key.scene:06d}" / key.camera / name


def export_official_layout(
    predictions: str | Path, target: str | Path, keys: Iterable[SampleKey]
) -> int:
    """Link GraspPanda's dump of ``keys`` into ``target`` in the official tool's layout.

    ``grasppanda test`` writes ``<predictions>/scene_0042/zivid/0000.npy`` (scene, camera,
    view); ``graspclutter6dAPI`` reads ``<target>/000042/zivid/000004.npy`` (scene, camera,
    image id). Frames without a dump are skipped, as the official tool scores them as
    zero by itself.

    Returns:
        The number of files linked.
    """
    linked = 0
    for key in keys:
        source = prediction_path(predictions, key).resolve()
        if not source.is_file():
            continue
        link = prediction_file(target, key)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source)
        linked += 1
    return linked


def _read_scene_gt(data_root: str | Path, scene: int) -> dict[str, list[dict]]:
    path = Path(data_root) / "scenes" / f"{scene:06d}" / "scene_gt.json"
    with path.open() as stream:
        return json.load(stream)


def _instances(scene_gt: dict[str, list[dict]], camera: str, frame: int) -> list[dict]:
    image = str(image_id(camera, frame))
    if image not in scene_gt:
        raise ValueError(f"scene_gt.json has no image {image} (camera {camera}, frame {frame})")
    return scene_gt[image]


class ClutterGroundTruth:
    """Object instances and poses of one scene seen by one camera.

    Args:
        data_root: Dataset root.
        scene: Scene index.
        camera: One of :data:`CAMERAS`.
        cache: Source of the per-object scoring data.
        protocol: Unused here (the benchmark has no table); part of the shared interface.
    """

    def __init__(
        self, data_root: str | Path, scene: int, camera: str, cache: ObjectCache, protocol: Protocol
    ) -> None:
        self.camera = camera
        self._scene_gt = _read_scene_gt(data_root, scene)
        self.object_ids = [int(entry["obj_id"]) for entry in _instances(self._scene_gt, camera, 0)]
        if not self.object_ids:
            raise ValueError(f"scene {scene} lists no objects for camera {camera}")
        self.objects: list[ObjectAsset] = [cache.load(object_id) for object_id in self.object_ids]

    @staticmethod
    def list_object_ids(data_root: str | Path, scene: int, camera: str) -> list[int]:
        """Object id of every instance of the scene, in the order of the camera's first image."""
        entries = _instances(_read_scene_gt(data_root, scene), camera, 0)
        return [int(entry["obj_id"]) for entry in entries]

    def frame(self, frame: int) -> tuple[list[np.ndarray], None]:
        """Float64 object-to-camera poses of a frame; there is no table."""
        entries = _instances(self._scene_gt, self.camera, frame)
        listed = [int(entry["obj_id"]) for entry in entries]
        if listed != self.object_ids:
            raise ValueError(
                f"image {image_id(self.camera, frame)} lists objects {listed}, but the "
                f"camera's first image lists {self.object_ids}; the evaluator pairs poses "
                "with objects by that order"
            )
        poses = []
        for entry in entries:
            pose = np.eye(4)
            pose[:3, :3] = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            pose[:3, 3] = np.array(entry["cam_t_m2c"], dtype=np.float64).reshape(3) / _MILLIMETRE
            poses.append(pose)
        return poses, None
