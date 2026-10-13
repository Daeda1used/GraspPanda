"""GraspNet-1Billion reader.

Expected tree (the official download, read in place and never modified)::

    scenes/scene_XXXX/object_id_list.txt
    scenes/scene_XXXX/{realsense,kinect}/{depth,label,rgb}/NNNN.png
    scenes/scene_XXXX/{realsense,kinect}/meta/NNNN.mat          intrinsics, depth factor, poses
    scenes/scene_XXXX/{realsense,kinect}/annotations/NNNN.xml   poses, four decimals
    scenes/scene_XXXX/{realsense,kinect}/{camera_poses,cam0_wrt_table}.npy
    grasp_label/NNN_labels.npz            (or grasp_label_simplified/, same values)
    collision_label/scene_XXXX/collision_labels.npz

190 scenes, 256 frames per scene and camera, images of 1280 x 720. Object ids are
zero-based (``grasp_label/000`` to ``087``); the segmentation image and ``cls_indexes``
store ``object_id + 1``. An object occurs at most once per scene, so the segmentation
is an instance map as well.

The pipeline a frame goes through is described in :mod:`grasppanda.datasets.scene`.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from grasppanda.registry import DATASETS, Cfg
from grasppanda.structures import ContractError, SampleKey
from grasppanda.transforms.base import Transform
from grasppanda.transforms.frame import Frame

from .compact_labels import CompactLabelConfig
from .graspness import GraspnessConfig, LegacyGraspness
from .manifest import SplitManifest
from .scale_statistic import ScaleStatisticConfig
from .scene import GraspSceneDataset, read_image
from .tolerance import ToleranceConfig
from .view_grasps import ViewGraspConfig

__all__ = ["GraspNet1B", "check_tree", "quaternion_to_matrix"]

IMAGE_SIZE = (720, 1280)
CAMERAS = ("realsense", "kinect")


def quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Rotation matrix of a quaternion that need not be normalised (float64)."""
    scale = 2.0 / (w * w + x * x + y * y + z * z)
    return np.array(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - w * z), scale * (x * z + w * y)],
            [scale * (x * y + w * z), 1.0 - scale * (x * x + z * z), scale * (y * z - w * x)],
            [scale * (x * z - w * y), scale * (y * z + w * x), 1.0 - scale * (x * x + y * y)],
        ]
    )


@DATASETS.register()
class GraspNet1B(GraspSceneDataset):
    """GraspNet-1Billion frames as point clouds with optional supervision.

    See :class:`~grasppanda.datasets.scene.GraspSceneDataset` for what an item holds and
    for the arguments not listed here.

    Args:
        camera: ``realsense`` or ``kinect``.
        split: ``train``, ``test_seen``, ``test_similar``, ``test_novel``, ``test``,
            ``train_dev``, ``val_dev`` or ``sample`` (see the shipped manifest).
        label_source: Directory of the object grasp labels the label store is built
            from. ``grasp_label`` is the official one; ``grasp_label_simplified`` is the
            output of the Graspness ``simplify_dataset`` script and holds the same values.
        images, object_labels, view_grasps: See the base class.
    """

    name = "graspnet1b"
    title = "GraspNet-1Billion"
    status = "tested"
    source = "GraspNet-1Billion (Fang et al., CVPR 2020)"

    #: ``(H, W)`` of every image.
    image_size = IMAGE_SIZE
    cameras = CAMERAS
    served_images = ("rgb", "depth")
    #: ``meta`` are the poses of ``meta/NNNN.mat``, which the grasp labels use;
    #: ``annotations`` are those of ``annotations/NNNN.xml``, rounded to four decimals.
    pose_sources = ("meta", "annotations")

    def __init__(
        self,
        camera: Literal["realsense", "kinect"],
        split: str,
        preprocess: list[Cfg[Transform]],
        transforms: list[Cfg[Transform]],
        root: str | None = None,
        frame_step: int = 1,
        load_labels: bool = True,
        graspness: GraspnessConfig | None = None,
        label_source: Literal["grasp_label", "grasp_label_simplified"] = "grasp_label",
        keep_full_points: bool = True,
        seed: int | None = None,
        check_assets: bool = True,
        compact_labels: CompactLabelConfig | None = None,
        images: list[Literal["rgb", "depth"]] | None = None,
        object_labels: bool = True,
        view_grasps: ViewGraspConfig | None = None,
        tolerance: ToleranceConfig | None = None,
        scale_statistic: ScaleStatisticConfig | None = None,
    ) -> None:
        self.label_source = label_source
        self._calibration: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._scene_objects: dict[int, list[int]] = {}
        self._meta_cache: tuple[SampleKey, dict[str, np.ndarray]] | None = None
        super().__init__(
            camera,
            split,
            preprocess,
            transforms,
            root=root,
            frame_step=frame_step,
            load_labels=load_labels,
            graspness=graspness,
            keep_full_points=keep_full_points,
            seed=seed,
            check_assets=check_assets,
            compact_labels=compact_labels,
            images=images or (),
            object_labels=object_labels,
            view_grasps=view_grasps,
            tolerance=tolerance,
            scale_statistic=scale_statistic,
        )

    # -------------------------------------------------------------------------- paths

    @classmethod
    def load_manifest(cls, root: Path | None = None) -> SplitManifest:
        """The split manifest shipped with the package."""
        return SplitManifest.load(cls.name)

    @classmethod
    def check_tree(
        cls, root: str | os.PathLike[str], cameras: Sequence[str] | None = None
    ) -> list[str]:
        return check_tree(Path(root), cls.load_manifest(), cameras or cls.cameras)

    def scene_dir(self, scene: int) -> Path:
        return self.root / "scenes" / f"scene_{scene:04d}"

    def camera_dir(self, scene: int) -> Path:
        return self.scene_dir(scene) / self.camera

    def object_label_file(self, object_id: int) -> Path:
        return self.root / self.label_source / f"{object_id:03d}_labels.npz"

    def collision_label_file(self, scene: int) -> Path:
        return self.root / "collision_label" / f"scene_{scene:04d}" / "collision_labels.npz"

    def object_model_file(self, object_id: int) -> Path:
        """Vertices of an object model; the per-view grasp labels are refined against them."""
        return self.root / "models" / f"{object_id:03d}" / "nontextured.ply"

    def legacy_graspness_file(self, key: SampleKey) -> Path:
        """Where the Graspness reference pipeline keeps the graspness of a frame."""
        scene = f"scene_{key.scene:04d}"
        return self.root / "graspness" / scene / key.camera / f"{key.frame:04d}.npy"

    def legacy_graspness(self) -> LegacyGraspness:
        """Reader of the upstream-format graspness files inside the dataset tree."""
        return LegacyGraspness(self.legacy_graspness_file, self.fingerprint)

    # ------------------------------------------------------------------ scene content

    def scene_object_ids(self, scene: int) -> list[int]:
        """Zero-based object ids of a scene, in the order of its collision labels."""
        if scene not in self._scene_objects:
            file = self.scene_dir(scene) / "object_id_list.txt"
            if not file.is_file():
                raise FileNotFoundError(f"object list of scene {scene} not found: {file}")
            self._scene_objects[scene] = [int(token) for token in file.read_text().split()]
        return self._scene_objects[scene]

    def instance_ids(self, object_ids: Sequence[int]) -> list[int]:
        """The segmentation stores ``object_id + 1``."""
        return [object_id + 1 for object_id in object_ids]

    def _calibrated(self, scene: int) -> tuple[np.ndarray, np.ndarray]:
        if scene not in self._calibration:
            directory = self.camera_dir(scene)
            self._calibration[scene] = (
                np.load(directory / "camera_poses.npy"),
                np.load(directory / "cam0_wrt_table.npy"),
            )
        return self._calibration[scene]

    def _meta(self, key: SampleKey) -> dict[str, np.ndarray]:
        """Content of ``meta/NNNN.mat``; the last one read is kept for the pose lookup."""
        if self._meta_cache is None or self._meta_cache[0] != key:
            from scipy.io import loadmat

            file = self.camera_dir(key.scene) / "meta" / f"{key.frame:04d}.mat"
            self._meta_cache = (key, loadmat(str(file)))
        return self._meta_cache[1]

    # ------------------------------------------------------------------------- frames

    def read_frame(self, key: SampleKey) -> Frame:
        directory = self.camera_dir(key.scene)
        meta = self._meta(key)
        depth = read_image(directory / "depth" / f"{key.frame:04d}.png")
        if depth.shape != tuple(self.image_size):
            raise ContractError(f"{key}: depth image is {depth.shape}, expected {self.image_size}")
        camera_poses, table_from_first = self._calibrated(key.scene)
        frame = Frame(
            key=key,
            depth=depth,
            depth_scale=float(meta["factor_depth"].reshape(-1)[0]),
            intrinsics=np.asarray(meta["intrinsic_matrix"], dtype=np.float64),
            table_from_camera=table_from_first @ camera_poses[key.frame],
        )
        if "rgb" in self.images:
            frame.rgb = read_image(directory / "rgb" / f"{key.frame:04d}.png")
            if frame.rgb.shape != (*self.image_size, 3):
                raise ContractError(
                    f"{key}: colour image is {frame.rgb.shape}, expected {(*self.image_size, 3)}"
                )
        if self.reads_ground_truth:
            frame.segmentation = read_image(directory / "label" / f"{key.frame:04d}.png")
        return frame

    def object_poses(
        self, key: SampleKey, source: str | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """``(object_id, (4, 4) float32 object-to-camera pose)`` of every object.

        Args:
            key: The frame.
            source: ``meta`` (default) or ``annotations``; see :attr:`pose_sources`.
        """
        source = source or self.pose_sources[0]
        expected = self.scene_object_ids(key.scene)
        if source == "meta":
            meta = self._meta(key)
            ids = [int(index) - 1 for index in meta["cls_indexes"].reshape(-1)]
            poses = []
            for column in range(len(ids)):
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :] = meta["poses"][:, :, column]
                poses.append(pose)
        elif source == "annotations":
            file = self.camera_dir(key.scene) / "annotations" / f"{key.frame:04d}.xml"
            ids, poses = [], []
            for entry in ElementTree.parse(file).getroot().findall("obj"):
                ids.append(int(entry.findtext("obj_id")))
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = quaternion_to_matrix(
                    *(float(value) for value in entry.findtext("ori_in_world").split())
                )
                pose[:3, 3] = [float(value) for value in entry.findtext("pos_in_world").split()]
                poses.append(pose)
        else:
            raise ValueError(f"pose source must be one of {self.pose_sources}, got {source!r}")
        if ids != expected:
            raise ContractError(
                f"{key}: the {source} list objects {ids}, but the scene's object list (and the "
                f"order of its collision labels) is {expected}"
            )
        return list(zip(ids, poses))


# ------------------------------------------------------------------------- tree check


def check_tree(root: Path, manifest: SplitManifest, cameras: Sequence[str]) -> list[str]:
    """List what a GraspNet-1Billion tree lacks for the scenes of a manifest."""
    if not root.is_dir():
        return [f"dataset root {root} does not exist"]
    problems: list[str] = []
    objects: set[int] = set()
    for scene in manifest.all_scenes():
        name = f"scene_{scene:04d}"
        scene_dir = root / "scenes" / name
        if not scene_dir.is_dir():
            problems.append(f"scenes/{name}/ is missing")
            continue
        object_list = scene_dir / "object_id_list.txt"
        if object_list.is_file():
            objects.update(int(token) for token in object_list.read_text().split())
        else:
            problems.append(f"scenes/{name}/object_id_list.txt is missing")
        if not (root / "collision_label" / name / "collision_labels.npz").is_file():
            problems.append(f"collision_label/{name}/collision_labels.npz is missing")
        for camera in cameras:
            camera_dir = scene_dir / camera
            if not camera_dir.is_dir():
                problems.append(f"scenes/{name}/{camera}/ is missing")
                continue
            for file in ("camera_poses.npy", "cam0_wrt_table.npy"):
                if not (camera_dir / file).is_file():
                    problems.append(f"scenes/{name}/{camera}/{file} is missing")
            for sub, suffix in (
                ("depth", ".png"),
                ("label", ".png"),
                ("meta", ".mat"),
                ("annotations", ".xml"),
            ):
                present = (
                    {entry.name for entry in os.scandir(camera_dir / sub)}
                    if (camera_dir / sub).is_dir()
                    else set()
                )
                absent = [f for f in manifest.frames if f"{f:04d}{suffix}" not in present]
                if absent:
                    problems.append(
                        f"scenes/{name}/{camera}/{sub}/: {len(absent)} of {len(manifest.frames)} "
                        f"{suffix} files missing, e.g. {absent[0]:04d}{suffix}"
                    )
    label_dirs = [d for d in ("grasp_label", "grasp_label_simplified") if (root / d).is_dir()]
    if not label_dirs:
        problems.append("grasp_label/ (or grasp_label_simplified/) is missing")
    for directory in label_dirs:
        present = {path.name for path in (root / directory).glob("*_labels.npz")}
        absent = sorted(i for i in objects if f"{i:03d}_labels.npz" not in present)
        if absent:
            problems.append(
                f"{directory}/: labels of {len(absent)} objects missing, e.g. {absent[:5]}"
            )
    return problems
