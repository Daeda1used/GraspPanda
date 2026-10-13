"""GraspClutter6D reader.

The dataset is laid out in BOP style and read in place::

    split_info/grasp_{train,test}_scene_ids.json    413 and 235 scenes (authoritative)
    split_info/obj_ids_per_scene.json
    scenes/<scene:06d>/{rgb,depth,label}/<image:06d>.png
    scenes/<scene:06d>/{mask,mask_visib}/<image:06d>_<instance:06d>.png
    scenes/<scene:06d>/scene_camera.json    per image: cam_K, depth_scale, cam_{R,t}_w2c
    scenes/<scene:06d>/scene_gt.json        per image: per instance cam_{R,t}_m2c, obj_id
    grasp_label/obj_<object:06d>_labels.npz
    collision_label/<scene:06d>.npz         one array per instance, in scene_gt order
    models_m/obj_<object:06d>.ply           object models in metres (evaluator, view grasps)
    dex_models/obj_<object:06d>.pkl         force-closure data of the evaluator

What differs from GraspNet-1Billion, and how the reader deals with it:

* **Cameras are told apart by image id, not by directory.** A scene has 13 annotated
  views, each taken by four cameras: ``image = 4 * frame + c`` with ``c`` = 1 RealSense
  D415, 2 RealSense D435, 3 Azure Kinect, 4 Zivid. ``SampleKey.frame`` is the view
  (0 to 12) and ``SampleKey.camera`` the camera name.
* **Intrinsics and depth scale are per image.** ``depth_scale`` converts raw depth to
  millimetres (1 for the RealSense cameras, 0.1 for Azure Kinect and Zivid), and even
  the intrinsics of one camera vary between views. Image size differs per camera.
* **Lengths are millimetres** in ``scene_gt.json`` and ``scene_camera.json``; poses
  leave this module in metres.
* **``cam_R_w2c`` / ``cam_t_w2c`` hold the camera pose in the world frame**, that is,
  they map camera coordinates to world coordinates, contrary to what the BOP names say.
  The official tool uses them that way, and only under this reading do the table and
  the objects stay put across the views of a scene. The world frame is fixed to the
  rig, not to the table top: its z-axis points up to within a few degrees and the table
  lies well below z = 0. It is what ``Frame.table_from_camera`` holds for this dataset,
  so the bounds of a ``TableBoxWorkspace`` are world coordinates here.
* **One object model can occur several times in a scene.** The ``label`` image stores
  object ids and cannot tell instances apart, so instance ids come from the per-instance
  visibility masks, and grasp labels are keyed by instance (position in ``scene_gt``).

Nothing is written into the dataset tree.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import yaml

from grasppanda.registry import DATASETS, Cfg
from grasppanda.structures import ContractError, SampleKey
from grasppanda.transforms.base import Transform
from grasppanda.transforms.frame import Frame

from .compact_labels import CompactLabelConfig
from .graspness import GraspnessConfig
from .manifest import SplitManifest
from .scale_statistic import ScaleStatisticConfig
from .scene import GraspSceneDataset, read_image
from .tolerance import ToleranceConfig
from .view_grasps import ViewGraspConfig

__all__ = ["GraspClutter6D", "check_tree"]

#: Camera names of the official tool, in the order of their image-id offsets 1 to 4.
CAMERAS = ("realsense-d415", "realsense-d435", "azure-kinect", "zivid")
#: ``(H, W)`` of the images of each camera.
IMAGE_SIZES = {
    "realsense-d415": (1080, 1920),
    "realsense-d435": (1080, 1920),
    "azure-kinect": (2160, 3840),
    "zivid": (1200, 1920),
}
#: Annotated views per scene; each is seen by every camera.
VIEWS_PER_SCENE = 13
_SHORT = {"realsense-d415": "d415", "realsense-d435": "d435", "azure-kinect": "kinect"}


def _read_json(file: Path) -> Any:
    if not file.is_file():
        raise FileNotFoundError(f"{file} not found")
    return json.loads(file.read_text())


def _scene_ids(root: Path, split: str) -> list[int]:
    return [
        int(scene) for scene in _read_json(root / "split_info" / f"grasp_{split}_scene_ids.json")
    ]


def _subsets(dataset: str, scenes: dict[str, list[int]]) -> dict[str, list[int]]:
    """Scenes of the subsets shipped for ``dataset`` (``splits/<dataset>.yaml``), if any.

    A subset is offered only when the split of the dataset tree it is declared a part of
    contains every scene it names. On a tree with other split files (a partial copy, a
    synthetic tree) the subset does not exist, and asking for it says so; it never
    silently means other frames.
    """
    file = resources.files("grasppanda.datasets") / "splits" / f"{dataset}.yaml"
    if not file.is_file():
        return {}
    found: dict[str, list[int]] = {}
    for name, entry in yaml.safe_load(file.read_text(encoding="utf-8"))["subsets"].items():
        parent = set(scenes[entry["of"]])
        if "exclude" in entry:
            if entry["exclude"] in found:
                found[name] = sorted(parent - set(found[entry["exclude"]]))
        elif set(entry["scenes"]) <= parent:
            found[name] = [int(scene) for scene in entry["scenes"]]
    return found


@DATASETS.register()
class GraspClutter6D(GraspSceneDataset):
    """GraspClutter6D frames as point clouds with optional supervision.

    See :class:`~grasppanda.datasets.scene.GraspSceneDataset` for what an item holds and
    for the arguments not listed here.

    Args:
        camera: ``realsense-d415``, ``realsense-d435``, ``azure-kinect`` or ``zivid``.
        split: ``train`` and ``test`` are the grasp splits of the dataset's own split
            files; ``all`` is their union and ``sample`` the first training scene.
            ``train_dev`` / ``val_dev`` partition ``train`` (every tenth training scene,
            by rank, is ``val_dev``), and ``test_mini`` is every tenth test scene, by
            rank: a fixed subset for quick checks, never a benchmark split.
        images, object_labels, compact_labels, view_grasps, tolerance, scale_statistic:
            See the base class. ``view_grasps`` needs ``intrinsics: frame`` here (the
            calibration differs from view to view) and ``poses: scene_gt``.
    """

    name = "graspclutter6d"
    title = "GraspClutter6D"
    status = "tested"
    source = "GraspClutter6D (arXiv 2504.06866)"

    cameras = CAMERAS
    #: ``(H, W)`` of the images, per camera.
    image_sizes: ClassVar[dict[str, tuple[int, int]]] = IMAGE_SIZES
    #: Annotated views per scene.
    views_per_scene: ClassVar[int] = VIEWS_PER_SCENE
    served_images = ("rgb", "depth")
    pose_sources = ("scene_gt",)

    def __init__(
        self,
        camera: Literal["realsense-d415", "realsense-d435", "azure-kinect", "zivid"],
        split: str,
        preprocess: list[Cfg[Transform]],
        transforms: list[Cfg[Transform]],
        root: str | None = None,
        frame_step: int = 1,
        load_labels: bool = True,
        graspness: GraspnessConfig | None = None,
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
        self._json: OrderedDict[tuple[int, str], Any] = OrderedDict()
        self._objects_per_scene: dict[str, list[int]] | None = None
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
    def load_manifest(cls, root: Path) -> SplitManifest:
        """Build the manifest from the split files inside the dataset tree."""
        train, test = _scene_ids(root, "train"), _scene_ids(root, "test")
        if set(train) & set(test):
            raise ContractError(f"{root}/split_info: train and test scenes overlap")
        scenes = {"train": train, "test": test, "all": train + test, "sample": [min(train)]}
        scenes.update(_subsets(cls.name, scenes))
        return SplitManifest.build(cls.name, cls.cameras, range(cls.views_per_scene), scenes)

    @classmethod
    def default_config(cls, camera: str | None = None) -> str:
        camera = camera or "realsense-d435"
        return f"_base_/datasets/{cls.name}_{_SHORT.get(camera, camera)}.yaml"

    @classmethod
    def check_tree(
        cls, root: str | os.PathLike[str], cameras: Sequence[str] | None = None
    ) -> list[str]:
        return check_tree(Path(root), cameras or cls.cameras, cls.views_per_scene)

    def scene_dir(self, scene: int) -> Path:
        return self.root / "scenes" / f"{scene:06d}"

    def image_id(self, key: SampleKey) -> int:
        """Image id of a view as seen by the key's camera."""
        return 4 * key.frame + self.cameras.index(key.camera) + 1

    def object_label_file(self, object_id: int) -> Path:
        return self.root / "grasp_label" / f"obj_{object_id:06d}_labels.npz"

    def collision_label_file(self, scene: int) -> Path:
        return self.root / "collision_label" / f"{scene:06d}.npz"

    def object_model_file(self, object_id: int) -> Path:
        """Vertices of an object model in metres (``models/`` holds millimetres)."""
        return self.root / "models_m" / f"obj_{object_id:06d}.ply"

    @property
    def image_size(self) -> tuple[int, int]:
        """``(H, W)`` of the images of this reader's camera."""
        return self.image_sizes[self.camera]

    # ------------------------------------------------------------------ scene content

    def _scene_json(self, scene: int, name: str) -> dict[str, Any]:
        """A per-scene JSON file; the most recently used ones stay parsed."""
        key = (scene, name)
        if key in self._json:
            self._json.move_to_end(key)
        else:
            self._json[key] = _read_json(self.scene_dir(scene) / name)
            while len(self._json) > 32:
                self._json.popitem(last=False)
        return self._json[key]

    def _image_entry(self, key: SampleKey, name: str) -> Any:
        image = self.image_id(key)
        content = self._scene_json(key.scene, name)
        if str(image) not in content:
            raise ContractError(f"{key}: {name} has no entry for image {image}")
        return content[str(image)]

    def scene_object_ids(self, scene: int) -> list[int]:
        """Object id of every instance, in ``scene_gt`` order (that of the collision labels)."""
        first = str(self.cameras.index(self.camera) + 1)  # view 0 of this camera
        return [
            int(instance["obj_id"]) for instance in self._scene_json(scene, "scene_gt.json")[first]
        ]

    def object_ids(self, scenes: Iterable[int]) -> set[int]:
        """Object models of the given scenes, from the dataset's per-scene index."""
        if self._objects_per_scene is None:
            self._objects_per_scene = _read_json(
                self.root / "split_info" / "obj_ids_per_scene.json"
            )
        return {int(i) for scene in scenes for i in self._objects_per_scene[str(scene)]}

    def instance_ids(self, object_ids: Sequence[int]) -> list[int]:
        """Instances are numbered from 1 in ``scene_gt`` order."""
        return list(range(1, len(object_ids) + 1))

    # ------------------------------------------------------------------------- frames

    def read_frame(self, key: SampleKey) -> Frame:
        directory, image = self.scene_dir(key.scene), self.image_id(key)
        camera = self._image_entry(key, "scene_camera.json")
        depth = read_image(directory / "depth" / f"{image:06d}.png")
        expected = tuple(self.image_sizes[key.camera])
        if depth.shape != expected:
            raise ContractError(
                f"{key}: depth image {image:06d}.png is {depth.shape}, expected {expected} for "
                f"{key.camera}"
            )
        # Camera pose in the world frame, translation in millimetres (see the module
        # docstring: the fields are named w2c but map camera coordinates to the world).
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = np.array(camera["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
        world_from_camera[:3, 3] = np.array(camera["cam_t_w2c"], dtype=np.float64) / 1000.0
        frame = Frame(
            key=key,
            depth=depth,
            # raw units per metre: depth_scale is millimetres per raw unit
            depth_scale=1000.0 / float(camera["depth_scale"]),
            intrinsics=np.array(camera["cam_K"], dtype=np.float64).reshape(3, 3),
            table_from_camera=world_from_camera,
        )
        if "rgb" in self.images:
            frame.rgb = read_image(directory / "rgb" / f"{image:06d}.png")
            if frame.rgb.shape != (*expected, 3):
                raise ContractError(
                    f"{key}: colour image {image:06d}.png is {frame.rgb.shape}, expected "
                    f"{(*expected, 3)} for {key.camera}"
                )
        if self.reads_ground_truth:
            label = read_image(directory / "label" / f"{image:06d}.png")
            frame.segmentation = label[..., 0] if label.ndim == 3 else label
        if self.load_labels:
            frame.instances = self._instance_map(key, depth.shape)
        return frame

    def _instance_map(self, key: SampleKey, shape: tuple[int, int]) -> np.ndarray:
        """Instance ids from the per-instance visibility masks; first instance wins a pixel."""
        directory, image = self.scene_dir(key.scene), self.image_id(key)
        count = len(self._image_entry(key, "scene_gt.json"))
        instances = np.zeros(shape, dtype=np.uint16)
        for index in range(count):
            mask = read_image(directory / "mask_visib" / f"{image:06d}_{index:06d}.png")
            if mask.shape != shape:
                raise ContractError(
                    f"{key}: visibility mask of instance {index} is {mask.shape}, expected {shape}"
                )
            instances[(mask > 0) & (instances == 0)] = index + 1
        return instances

    def object_poses(
        self, key: SampleKey, source: str | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """``(object_id, (4, 4) float32 object-to-camera pose in metres)`` of every instance."""
        if (source or "scene_gt") != "scene_gt":
            raise ValueError(f"pose source must be one of {self.pose_sources}, got {source!r}")
        ids, poses = [], []
        for instance in self._image_entry(key, "scene_gt.json"):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = np.array(instance["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            pose[:3, 3] = np.array(instance["cam_t_m2c"], dtype=np.float64) / 1000.0
            ids.append(int(instance["obj_id"]))
            poses.append(pose)
        expected = self.scene_object_ids(key.scene)
        if ids != expected:
            raise ContractError(
                f"{key}: scene_gt lists objects {ids} for this image, but {expected} for the "
                "first view; the collision labels follow the latter"
            )
        return list(zip(ids, poses))


# ------------------------------------------------------------------------- tree check


def check_tree(root: Path, cameras: Sequence[str], views: int = VIEWS_PER_SCENE) -> list[str]:
    """List what a GraspClutter6D tree lacks for the scenes of its grasp splits."""
    if not root.is_dir():
        return [f"dataset root {root} does not exist"]
    problems: list[str] = []
    for name in (
        "grasp_train_scene_ids.json",
        "grasp_test_scene_ids.json",
        "obj_ids_per_scene.json",
    ):
        if not (root / "split_info" / name).is_file():
            problems.append(f"split_info/{name} is missing")
    if problems:
        return problems
    scenes = sorted(_scene_ids(root, "train") + _scene_ids(root, "test"))
    per_scene = _read_json(root / "split_info" / "obj_ids_per_scene.json")
    images = [4 * view + CAMERAS.index(camera) + 1 for camera in cameras for view in range(views)]
    objects: set[int] = set()
    for scene in scenes:
        name = f"{scene:06d}"
        directory = root / "scenes" / name
        if not directory.is_dir():
            problems.append(f"scenes/{name}/ is missing")
            continue
        objects.update(int(i) for i in per_scene.get(str(scene), []))
        for file in ("scene_camera.json", "scene_gt.json"):
            if not (directory / file).is_file():
                problems.append(f"scenes/{name}/{file} is missing")
        for sub in ("depth", "label", "rgb"):
            present = (
                {entry.name for entry in os.scandir(directory / sub)}
                if (directory / sub).is_dir()
                else set()
            )
            absent = [image for image in images if f"{image:06d}.png" not in present]
            if absent:
                problems.append(
                    f"scenes/{name}/{sub}/: {len(absent)} of {len(images)} images missing, "
                    f"e.g. {absent[0]:06d}.png"
                )
        if not (directory / "mask_visib").is_dir():
            problems.append(f"scenes/{name}/mask_visib/ is missing")
        if not (root / "collision_label" / f"{name}.npz").is_file():
            problems.append(f"collision_label/{name}.npz is missing")
    present = (
        {path.name for path in (root / "grasp_label").glob("obj_*_labels.npz")}
        if (root / "grasp_label").is_dir()
        else set()
    )
    absent = sorted(i for i in objects if f"obj_{i:06d}_labels.npz" not in present)
    if absent:
        problems.append(f"grasp_label/: labels of {len(absent)} objects missing, e.g. {absent[:5]}")
    # What the evaluator reads (and, for the models, the view-grasp generator).
    for sub, suffix, what in (("models_m", ".ply", "models"), ("dex_models", ".pkl", "pickles")):
        present = (
            {entry.name for entry in os.scandir(root / sub)} if (root / sub).is_dir() else set()
        )
        absent = sorted(i for i in objects if f"obj_{i:06d}{suffix}" not in present)
        if absent:
            problems.append(f"{sub}/: {what} of {len(absent)} objects missing, e.g. {absent[:5]}")
    return problems
