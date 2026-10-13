"""Shared machinery of RGB-D scene datasets with GraspNet-format grasp labels.

GraspNet-1Billion and GraspClutter6D differ in how a frame is stored on disk and agree
in everything after that. A reader subclasses :class:`GraspSceneDataset`, says how to
read a raw frame, the object poses and the label files, and inherits the pipeline:

1. ``preprocess``: image-space steps that decide which pixels become points. Their
   configuration is the preprocessing fingerprint that derived labels are tied to.
2. the label-alignment boundary: the masked cloud is formed and per-point labels
   (objectness, instance, graspness) are attached to it;
3. ``transforms``: sampling, grasp-label selection and augmentation, which carry the
   labels along.

Grasp labels come in one of two kinds. By default the objects of a frame carry the dense
label grids, read lazily by a label transform. A dataset configured with
``compact_labels`` serves the compact labels of :mod:`grasppanda.datasets.compact_labels`
*instead*: they are bound to the object poses as the transforms left them and travel in
``Supervision.extras``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from grasppanda.registry import TRANSFORMS, RegistryError
from grasppanda.structures import COMPACT_LABELS, Batch, ContractError, SampleKey
from grasppanda.transforms.base import ImageTransform, PointTransform, Transform, fingerprint
from grasppanda.transforms.frame import Frame, ObjectAnnotation, PointSample, align, to_batch
from grasppanda.transforms.image import WorkspaceProvider
from grasppanda.transforms.points import SampleGraspLabels
from grasppanda.utils.paths import dataset_root

from .assets import Asset
from .base import BaseDataset
from .compact_labels import CompactLabelConfig, CompactLabelStore
from .compact_labels import scene_shard as compact_shard
from .graspness import GraspnessConfig, GraspnessMaps, GraspnessSource, LegacyGraspness, scene_shard
from .labels import GRASPNET_LABEL_GRID, GraspLabelStore, LabelGrid, StoredGraspLabels
from .manifest import SplitManifest
from .scale_statistic import ScaleStatisticConfig, ScaleStatisticStore
from .tolerance import GraspToleranceStore, ToleranceConfig
from .view_grasps import ViewGraspConfig, ViewGrasps
from .view_grasps import scene_shard as view_grasp_shard

__all__ = ["GraspSceneDataset", "read_image"]


def read_image(path: Path) -> np.ndarray:
    """Read a PNG as an array, without any conversion."""
    from PIL import Image

    with Image.open(path) as image:
        return np.array(image)


def _expect_transforms(
    node: Any, kind: type[Transform], slot: str, meaning: str
) -> Iterator[tuple[str | None, type | None]]:
    """Yield ``(problem, class)`` for every entry of a transform list in a config node."""
    for index, entry in enumerate(node or []):
        name = entry.get("type") if isinstance(entry, dict) else type(entry).__name__
        try:
            cls = TRANSFORMS.get(name) if isinstance(entry, dict) else type(entry)
        except RegistryError:
            yield None, None  # reported by the signature check
            continue
        if not issubclass(cls, kind):
            yield (
                f"{slot}[{index}] is {name}, which is not {meaning}. `preprocess` takes "
                "image-space steps (they are fingerprinted and fix label alignment); "
                "`transforms` takes point-space steps that run after it.",
                cls,
            )
        else:
            yield None, cls


class GraspSceneDataset(BaseDataset):
    """Frames of a scene dataset as point clouds with optional supervision.

    ``dataset[i]`` returns a :class:`~grasppanda.structures.Batch` of one frame:
    ``Observation(keys, intrinsics, points, point_feats, full_points, provenance)`` and,
    with ``load_labels``, ``Supervision(objects, point_labels)`` where ``point_labels``
    holds ``objectness``, ``instance`` and, if configured, ``graspness``, each ``(N,)``.
    With ``compact_labels`` the supervision carries ``extras[COMPACT_LABELS]`` (one
    :class:`~grasppanda.structures.CompactGraspLabels` per object) and no dense grids.
    With ``images`` the observation also carries ``rgb`` and / or ``depth``; with
    ``view_grasps`` the supervision carries ``extras["view_grasps"]``.

    Args:
        camera: One of the dataset's cameras.
        split: Name of a split of the dataset's manifest.
        preprocess: Image-space transforms up to and including the workspace provider.
            They define the preprocessing fingerprint.
        transforms: Point-space transforms applied after label alignment.
        root: Dataset directory. Defaults to ``$GRASPPANDA_<NAME>_ROOT``, then
            ``$GRASPPANDA_DATA/<name>``, then ``./data/<name>``.
        frame_step: Use every ``frame_step``-th frame of the split.
        load_labels: Attach supervision. With ``False`` no label file is opened, and
            ground truth is read only if ``preprocess`` declares that it uses it.
        graspness: Which point-wise graspness to serve; ``None`` serves none.
        compact_labels: Serve compact grasp labels (one grasp per annotated point and
            view) in place of the dense grids; ``None`` serves the dense grids.
        tolerance: Parameters of the grasp tolerance labels to attach to every object's
            grasp labels (see :mod:`grasppanda.datasets.tolerance`); ``None`` serves none.
        scale_statistic: Which histogram of grasp widths to attach to every sample as
            ``supervision.extras['scale_histogram']`` (see
            :mod:`grasppanda.datasets.scale_statistic`); ``None`` serves none.
        keep_full_points: Return the masked cloud before sampling as
            ``observation.full_points`` (a collision filter needs it).
        images: Sensor images to return beside the cloud: ``rgb`` as
            ``observation.rgb`` (``(3, H, W)`` in ``[0, 1]``) and ``depth`` as
            ``observation.depth`` (``(H, W)`` metres, the unmasked sensor image). Empty
            by default, so a point-cloud method reads no image it does not use. A sample
            that carries images refuses point-space augmentations, which would move the
            cloud away from them.
        object_labels: Attach the dense per-object grasp labels (``supervision.objects``).
            A method trained on another label asset turns this off; ``objects`` is then
            empty and the grasp-label store is not needed.
        view_grasps: Which per-view grasp labels to serve as
            ``supervision.extras["view_grasps"]``; ``None`` serves none.
        seed: ``None`` draws the randomness of each sample from PyTorch's generator, so
            it changes every epoch and follows ``torch.manual_seed``. An integer makes
            every frame's sample a fixed function of the seed and the frame.
        check_assets: Verify at construction that the derived assets this configuration
            reads are complete. ``grasppanda prepare`` turns it off to build them.
    """

    #: Name for messages, e.g. ``GraspNet-1Billion``.
    title: ClassVar[str] = "dataset"
    #: Camera names; ``SampleKey.camera`` is one of them.
    cameras: ClassVar[tuple[str, ...]] = ()
    #: Grid of the dense grasp labels.
    label_grid: ClassVar[LabelGrid] = GRASPNET_LABEL_GRID
    #: Where object poses can be read from; the first is what the grasp labels use.
    pose_sources: ClassVar[tuple[str, ...]] = ()
    #: Name of the label variant the store is built from; part of its hash.
    label_source: str = "grasp_label"
    #: Sensor images this reader can serve through ``images``.
    served_images: ClassVar[tuple[str, ...]] = ("depth",)

    def __init__(
        self,
        camera: str,
        split: str,
        preprocess: Sequence[Any],
        transforms: Sequence[Any],
        root: str | None = None,
        frame_step: int = 1,
        load_labels: bool = True,
        graspness: GraspnessConfig | dict[str, Any] | None = None,
        keep_full_points: bool = True,
        seed: int | None = None,
        check_assets: bool = True,
        compact_labels: CompactLabelConfig | dict[str, Any] | None = None,
        images: Sequence[str] = (),
        object_labels: bool = True,
        view_grasps: ViewGraspConfig | dict[str, Any] | None = None,
        tolerance: ToleranceConfig | dict[str, Any] | None = None,
        scale_statistic: ScaleStatisticConfig | dict[str, Any] | None = None,
    ) -> None:
        who = type(self).__name__
        self.root = dataset_root(self.name, root)
        self.camera = camera
        self.split = split
        self.frame_step = frame_step
        self.load_labels = load_labels
        self.keep_full_points = keep_full_points
        self.seed = seed
        self.images = tuple(images)
        self.object_labels = object_labels
        for problem in self._label_problems(self.images, load_labels, object_labels, view_grasps):
            raise ContractError(f"{who}: {problem}")
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.title} not found at {self.root}. Set `root` in the dataset config, or "
                f"the environment variable GRASPPANDA_{self.name.upper()}_ROOT."
            )
        self.manifest = self.load_manifest(self.root)
        self._keys = self.manifest.keys(split, camera, frame_step)

        self.preprocess: list[ImageTransform] = [TRANSFORMS.build(node) for node in preprocess]
        self.transforms: list[PointTransform] = [TRANSFORMS.build(node) for node in transforms]
        for problem in self._pipeline_problems(self.preprocess, self.transforms):
            raise ContractError(f"{who}: {problem}")
        self.fingerprint = fingerprint(self.preprocess)
        self.uses_ground_truth = any(step.uses_ground_truth for step in self.preprocess)

        if isinstance(graspness, dict):
            graspness = GraspnessConfig(**graspness)
        if graspness is not None and not load_labels:
            raise ContractError(
                f"{who}: `graspness` is a label and needs load_labels=True; remove it or turn "
                "labels on."
            )
        for problem in graspness.problems(self.pose_sources) if graspness else ():
            raise ContractError(f"{who}: {problem}")
        if isinstance(tolerance, dict):
            tolerance = ToleranceConfig(**tolerance)
        if tolerance is not None and not load_labels:
            raise ContractError(
                f"{who}: `tolerance` is a label and needs load_labels=True; remove it or turn "
                "labels on."
            )
        for problem in tolerance.problems() if tolerance else ():
            raise ContractError(f"{who}: {problem}")
        if isinstance(scale_statistic, dict):
            scale_statistic = ScaleStatisticConfig(**scale_statistic)
        if scale_statistic is not None and not load_labels:
            raise ContractError(
                f"{who}: `scale_statistic` is supervision and needs load_labels=True; remove it "
                "or turn labels on."
            )
        for problem in scale_statistic.problems() if scale_statistic else ():
            raise ContractError(f"{who}: {problem}")
        self.graspness_config = graspness
        if isinstance(compact_labels, dict):
            compact_labels = CompactLabelConfig(**compact_labels)
        for problem in self._compact_problems(compact_labels, load_labels, self.transforms):
            raise ContractError(f"{who}: {problem}")
        if isinstance(view_grasps, dict):
            view_grasps = ViewGraspConfig(**view_grasps)
        self.view_grasp_config = view_grasps
        self.tolerance_config = tolerance
        self.label_store: GraspLabelStore | None = None
        self.graspness: GraspnessSource | None = None
        self.compact_labels: CompactLabelStore | None = None
        self.view_grasps: ViewGrasps | None = None
        self.tolerance: GraspToleranceStore | None = None
        self.scale_statistic: ScaleStatisticStore | None = None
        if load_labels:
            # The store is the identity of the dense labels that every derived label asset
            # is computed from; creating the object opens nothing.
            self.label_store = GraspLabelStore(self, self.label_grid)
            self.graspness = self._graspness_source(graspness)
            if compact_labels is not None:
                spec = CompactLabelStore.spec(
                    self.name, compact_labels, self.label_grid, self.label_store.asset.spec.hash
                )
                self.compact_labels = CompactLabelStore(Asset(spec), compact_labels)
            if view_grasps is not None:
                spec = ViewGrasps.spec(
                    self.name, self.camera, view_grasps, self.label_store.asset.spec.hash
                )
                self.view_grasps = ViewGrasps(Asset(spec))
            if tolerance is not None:
                self.tolerance = GraspToleranceStore(
                    self.name, tolerance, self.label_store.asset.spec.hash, self.label_grid
                )
            if scale_statistic is not None:
                self.scale_statistic = ScaleStatisticStore(
                    self.name, scale_statistic, self._scale_statistic_inputs()
                )
            if check_assets:
                self.require_assets()

    # ------------------------------------------------------ what a reader implements

    @classmethod
    def load_manifest(cls, root: Path) -> SplitManifest:
        """The split manifest; shipped with the package or read from the dataset tree."""
        raise NotImplementedError

    def read_frame(self, key: SampleKey) -> Frame:
        """The raw frame: depth, intrinsics, calibration and, when
        :attr:`reads_ground_truth`, the segmentation."""
        raise NotImplementedError

    def object_poses(
        self, key: SampleKey, source: str | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """``(object_id, (4, 4) float32 object-to-camera pose)`` of every object instance,
        in the order of the scene's collision labels. ``source`` is one of
        :attr:`pose_sources` and defaults to the first."""
        raise NotImplementedError

    def instance_ids(self, object_ids: Sequence[int]) -> list[int]:
        """Value that marks each instance in the ``instance`` point label."""
        raise NotImplementedError

    def scene_object_ids(self, scene: int) -> list[int]:
        """Object id of every instance of a scene, in the order of its collision labels."""
        raise NotImplementedError

    def object_label_file(self, object_id: int) -> Path:
        raise NotImplementedError

    def collision_label_file(self, scene: int) -> Path:
        raise NotImplementedError

    def legacy_graspness(self) -> LegacyGraspness:
        """Reader of upstream-format graspness files, where a dataset has such a layout."""
        raise ContractError(f"no upstream-format graspness layout is known for {self.name}")

    @classmethod
    def default_config(cls, camera: str | None = None) -> str:
        """Name of the shipped base config that ``grasppanda prepare`` works for."""
        return f"_base_/datasets/{cls.name}_{camera or cls.cameras[0]}.yaml"

    @classmethod
    def check_tree(
        cls, root: str | os.PathLike[str], cameras: Sequence[str] | None = None
    ) -> list[str]:
        """List what a dataset tree lacks for this reader. Counts files; opens none."""
        raise NotImplementedError

    # ------------------------------------------------------------------- config check

    @classmethod
    def _label_problems(
        cls, images: Sequence[str], load_labels: bool, object_labels: bool, view_grasps: Any
    ) -> list[str]:
        problems = []
        unserved = [name for name in images if name not in cls.served_images]
        if unserved:
            problems.append(
                f"`images` asks for {unserved}, but this reader serves {list(cls.served_images)}."
            )
        if view_grasps is not None and not load_labels:
            problems.append("`view_grasps` is a label and needs `load_labels: true`.")
        if isinstance(view_grasps, dict):
            try:
                view_grasps = ViewGraspConfig(**view_grasps)
            except (ContractError, TypeError) as error:
                problems.append(str(error))
        if isinstance(view_grasps, ViewGraspConfig) and view_grasps.poses not in cls.pose_sources:
            problems.append(
                f"`view_grasps.poses` is '{view_grasps.poses}', but this reader has the pose "
                f"sources {list(cls.pose_sources)}."
            )
        if not object_labels and not load_labels:
            problems.append(
                "`object_labels: false` only makes sense with `load_labels: true`; without "
                "labels there are no object labels to leave out."
            )
        return problems

    @staticmethod
    def _pipeline_problems(preprocess: Sequence[Any], transforms: Sequence[Any]) -> list[str]:
        problems, classes = [], []
        for problem, cls in _expect_transforms(
            preprocess, ImageTransform, "preprocess", "an image-space transform"
        ):
            classes.append(cls)
            if problem:
                problems.append(problem)
        for problem, _ in _expect_transforms(
            transforms, PointTransform, "transforms", "a point-space transform"
        ):
            if problem:
                problems.append(problem)
        if None not in classes:
            providers = [cls.__name__ for cls in classes if issubclass(cls, WorkspaceProvider)]
            if len(providers) != 1:
                found = ", ".join(providers) or "none"
                problems.append(
                    f"`preprocess` must name exactly one workspace provider (found: {found}). "
                    "Providers that use ground truth say so in their name (Segmentation…); "
                    "TableBoxWorkspace uses calibration only and NoWorkspace nothing."
                )
        return problems

    @staticmethod
    def _compact_problems(
        config: CompactLabelConfig | None, load_labels: bool, transforms: Sequence[Any]
    ) -> list[str]:
        """Problems of a ``compact_labels`` section; ``transforms`` are nodes or built steps."""
        if config is None:
            return []
        if not load_labels:
            return ["`compact_labels` is a label and needs `load_labels: true`."]
        problems = config.problems()
        for index, (_, cls) in enumerate(
            _expect_transforms(transforms, PointTransform, "transforms", "a point-space transform")
        ):
            if cls is not None and issubclass(cls, SampleGraspLabels):
                problems.append(
                    f"transforms[{index}] ({cls.__name__}) reads the dense label grids, which a "
                    "dataset with `compact_labels` does not serve. Remove one of the two."
                )
        return problems

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Problems visible in the config node alone, before anything is opened."""
        yield from cls._pipeline_problems(node.get("preprocess"), node.get("transforms"))
        compact = node.get("compact_labels")
        if isinstance(compact, dict):
            try:
                compact = CompactLabelConfig(**compact)
            except TypeError:
                compact = None  # missing or unknown keys: reported by the signature check
        if isinstance(compact, CompactLabelConfig):
            yield from cls._compact_problems(
                compact, node.get("load_labels", True) is not False, node.get("transforms") or []
            )
        yield from cls._label_problems(
            node.get("images") or (),
            node.get("load_labels", True) is not False,
            node.get("object_labels", True) is not False,
            node.get("view_grasps"),
        )
        graspness = node.get("graspness")
        if graspness is not None and node.get("load_labels", True) is False:
            yield "`graspness` is a label and needs `load_labels: true`."
        if isinstance(graspness, dict):
            yield from GraspnessConfig(**graspness).problems(cls.pose_sources)
        statistic = node.get("scale_statistic")
        if statistic is not None and node.get("load_labels", True) is False:
            yield "`scale_statistic` is supervision and needs `load_labels: true`."
        if isinstance(statistic, dict):
            try:
                yield from ScaleStatisticConfig(**statistic).problems()
            except TypeError as error:
                yield f"`scale_statistic`: {error}."
        tolerance = node.get("tolerance")
        if tolerance is not None and node.get("load_labels", True) is False:
            yield "`tolerance` is a label and needs `load_labels: true`."
        if isinstance(tolerance, dict):
            try:
                yield from ToleranceConfig(**tolerance).problems()
            except TypeError as error:
                yield f"`tolerance`: {error}."
        if node.get("frame_step", 1) < 1:
            yield "`frame_step` must be at least 1."

    # ---------------------------------------------------------- what the batches carry

    @classmethod
    def _serves_dense_grids(cls, node: Mapping[str, Any]) -> bool:
        """Whether ``supervision.objects`` holds the dense label grids: object labels are
        on, compact labels do not replace them, and a transform loads the grids."""
        loaders = [
            kind
            for _, kind in _expect_transforms(node.get("transforms"), PointTransform, "", "")
            if kind is not None and issubclass(kind, SampleGraspLabels)
        ]
        return (
            node.get("object_labels", True) is not False
            and node.get("compact_labels") is None
            and bool(loaders)
        )

    @classmethod
    def served_supervision(cls, node: Mapping[str, Any]) -> frozenset[str]:
        """See :meth:`BaseDataset.served_supervision`; read from the config node alone."""
        if node.get("load_labels", True) is False:
            return frozenset()
        served = {"point_labels.objectness", "point_labels.instance"}
        dense = cls._serves_dense_grids(node)
        sections = {
            "graspness": "point_labels.graspness",
            "view_grasps": "extras.view_grasps",
            "scale_statistic": "extras.scale_histogram",
        }
        served |= {field for section, field in sections.items() if node.get(section) is not None}
        if dense:
            served.add("objects")
            if node.get("tolerance") is not None:
                served.add("objects.tolerance")
        elif node.get("compact_labels") is not None and node.get("object_labels", True):
            served.add(f"extras.{COMPACT_LABELS}")
        return frozenset(served)

    @classmethod
    def supervision_hint(cls, field: str, node: Mapping[str, Any]) -> str:
        """See :meth:`BaseDataset.supervision_hint`."""
        prepare = f"`grasppanda prepare {cls.name} --"
        if node.get("load_labels", True) is False:
            return "It is configured with `load_labels: false` and serves no supervision at all."
        one_of_two = (
            "A dataset serves one of the two label sets, and its transforms differ with it: "
            "the dense grids need a `SampleGraspLabels` transform and no `compact_labels` "
            "section, compact labels the section and no such transform."
        )
        if field in ("objects", "objects.tolerance") and not cls._serves_dense_grids(node):
            if node.get("object_labels", True) is False:
                return "It is configured with `object_labels: false`: no grasp labels per object."
            if node.get("compact_labels") is not None:
                return (
                    "It serves compact labels in place of the dense label grids (its "
                    f"`compact_labels` section). {one_of_two}"
                )
            return (
                "No transform loads the dense label grids: add `SampleGraspLabels` to its "
                "`transforms` (configs/_base_/datasets/ holds the recipes of the shipped methods)."
            )
        if field == f"extras.{COMPACT_LABELS}":
            return (
                f"Compact labels are served with a `compact_labels:` section (asset: {prepare}"
                f"compact-labels --config <this config>`). {one_of_two}"
            )
        sections = {
            "objects.tolerance": ("tolerance", "tolerance"),
            "point_labels.graspness": ("graspness", "graspness"),
            "extras.scale_histogram": ("scale_statistic", "scale-statistic"),
            "extras.view_grasps": ("view_grasps", "view-grasps"),
        }
        if field in sections:
            section, flag = sections[field]
            return (
                f"It is served with a `{section}:` section of the dataset (asset: {prepare}{flag} "
                "--config <this config>`)."
            )
        return f"{cls.__name__} has no option that serves it."

    # ------------------------------------------------------------------------- assets

    def _graspness_source(self, config: GraspnessConfig | None) -> GraspnessSource | None:
        if config is None:
            return None
        if config.source == "legacy":
            return self.legacy_graspness()
        assert self.label_store is not None
        spec = GraspnessMaps.spec(
            self.name,
            self.camera,
            config,
            [step.config() for step in self.preprocess],
            self.uses_ground_truth,
            self.label_store.asset.spec.hash,
        )
        return GraspnessMaps(Asset(spec))

    def _scale_statistic_inputs(self) -> dict[str, Any]:
        """What a natively generated scale statistic is a function of, besides its own
        parameters: the labels, the frames and the recipe that turns them into samples."""
        assert self.label_store is not None
        return {
            "dataset": self.name,
            "grasp_labels": self.label_store.asset.spec.hash,
            "camera": self.camera,
            "split": self.split,
            "preprocessing": [step.config() for step in self.preprocess],
            "transforms": [step.config() for step in self.transforms],
        }

    def scenes(self) -> tuple[int, ...]:
        """Scene ids of the split."""
        return self.manifest.split(self.split).scenes

    def object_ids(self, scenes: Iterable[int]) -> set[int]:
        """Ids of the object models that occur in the given scenes."""
        return {object_id for scene in scenes for object_id in self.scene_object_ids(scene)}

    def require_assets(self) -> None:
        """Raise ``AssetError`` unless every derived asset of this split is complete."""
        if self.label_store is None:
            return
        scenes = self.scenes()
        if self.compact_labels is not None:  # served in place of the dense grids
            self.compact_labels.asset.require(
                [compact_shard(scene) for scene in scenes],
                build=f"grasppanda prepare {self.name} --compact-labels --split {self.split} "
                "--config <the config that declares these compact_labels>",
            )
        elif self.object_labels or self.graspness is not None:
            self.label_store.require(
                self.object_ids(scenes), scenes, build=f"grasppanda prepare {self.name} --labels"
            )
        if self.view_grasps is not None:
            self.view_grasps.asset.require(
                [view_grasp_shard(scene) for scene in scenes],
                build=f"grasppanda prepare {self.name} --view-grasps --camera {self.camera} "
                f"--split {self.split}   (add --config <file> when the `view_grasps` settings "
                "differ from the shipped base config)",
            )
        if isinstance(self.graspness, GraspnessMaps):
            assert self.graspness_config is not None
            flag = " --import-legacy" if self.graspness_config.source == "legacy_import" else ""
            self.graspness.asset.require(
                [scene_shard(scene) for scene in scenes],
                build=f"grasppanda prepare {self.name} --graspness{flag} --camera {self.camera} "
                f"--split {self.split}   (add --config <file> when the preprocessing or the "
                "graspness settings differ from the shipped base config)",
            )
        if self.tolerance is not None:
            self.tolerance.require(
                self.object_ids(scenes),
                build=f"grasppanda prepare {self.name} --tolerance --config <a config whose "
                f"dataset has this `tolerance` section> --split {self.split}",
            )
        if self.scale_statistic is not None:
            legacy = self.scale_statistic.config.source == "legacy_import"
            flag = " --import-legacy --scale-statistic-file <reference file>" if legacy else ""
            self.scale_statistic.require(
                build=f"grasppanda prepare {self.name} --scale-statistic{flag} --config <a config "
                "whose dataset has this `scale_statistic` section>"
            )

    def asset_records(self) -> list[dict[str, Any]]:
        """Derived assets this dataset reads, with their provenance, for run records."""
        records = []
        if self.compact_labels is not None:
            records.append(self.compact_labels.record())
        elif self.label_store is not None and (self.object_labels or self.graspness is not None):
            records.append(self.label_store.asset.record())
        if self.graspness is not None:
            records.append(self.graspness.record())
        if self.view_grasps is not None:
            records.append(self.view_grasps.record())
        if self.tolerance is not None:
            records.append(self.tolerance.asset.record())
        if self.scale_statistic is not None:
            records.append(self.scale_statistic.asset.record())
        return records

    # ------------------------------------------------------------------------- frames

    @property
    def reads_ground_truth(self) -> bool:
        """Whether frames carry ground truth: for labels, or for a declared pipeline step."""
        return self.load_labels or self.uses_ground_truth

    def scene_keys(self, scene: int) -> list[SampleKey]:
        """Every frame of a scene for this camera, whatever the split."""
        return [SampleKey(self.name, scene, self.camera, frame) for frame in self.manifest.frames]

    def scene_labels(self, scene: int) -> list[StoredGraspLabels]:
        """Grasp labels of every instance of a scene, in the order of its collision labels."""
        if self.label_store is None:
            raise ContractError(f"{type(self).__name__} was built with load_labels=False")
        return [
            self.label_store.labels(scene, index, object_id)
            for index, object_id in enumerate(self.scene_object_ids(scene))
        ]

    def frame(self, key: SampleKey) -> Frame:
        """Read a frame and run the image-space preprocessing."""
        frame = self.read_frame(key)
        for step in self.preprocess:
            frame = step(frame)
        return frame

    # -------------------------------------------------------------------------- items

    @property
    def keys(self) -> Sequence[SampleKey]:
        return self._keys

    def _rng(self, key: SampleKey) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng(int(torch.randint(2**62, ()).item()))
        camera = self.cameras.index(key.camera)
        return np.random.default_rng([self.seed, key.scene, key.frame, camera])

    def sample(self, key: SampleKey) -> PointSample:
        """The frame at the label-alignment boundary: masked cloud with aligned labels."""
        frame = self.frame(key)
        sample = align(frame, keep_full_points=self.keep_full_points, images=self.images)
        sample.rng = self._rng(key)
        if not self.load_labels:
            return sample
        assert frame.segmentation is not None and self.label_store is not None
        foreground = frame.segmentation.reshape(-1)[sample.pixels] > 0
        instances = frame.segmentation if frame.instances is None else frame.instances
        sample.point_labels["objectness"] = foreground.astype(np.int64)
        sample.point_labels["instance"] = instances.reshape(-1)[sample.pixels].astype(np.int64)
        if self.graspness is not None:
            sample.point_labels["graspness"] = self.graspness.read(
                key, sample.pixels, sample.image_size
            )
        if self.view_grasps is not None:
            sample.extras["view_grasps"] = self.view_grasps.read(key)
            sample.camera_bound.add("view_grasps")  # grasps in the camera frame of the view
        if self.scale_statistic is not None and self.scale_statistic.exists():
            edges, counts = self.scale_statistic.load()
            sample.extras["scale_histogram"] = {"edges": edges, "counts": counts}
        if not self.object_labels:
            sample.objects = []
            return sample
        poses = self.object_poses(key)
        marks = self.instance_ids([object_id for object_id, _ in poses])
        sample.objects = [
            ObjectAnnotation(
                object_id=object_id,
                instance_id=mark,
                pose=pose,
                source=(
                    self._label_source(key.scene, index, object_id)
                    if self.compact_labels is None
                    else None
                ),
            )
            for index, ((object_id, pose), mark) in enumerate(zip(poses, marks))
        ]
        return sample

    def _label_source(self, scene: int, index: int, object_id: int) -> StoredGraspLabels:
        """Grasp labels of one instance, with the tolerance levels if they are served."""
        assert self.label_store is not None
        labels = self.label_store.labels(scene, index, object_id)
        if self.tolerance is not None:
            labels.tolerance_levels = self.tolerance.levels(object_id)
            labels.tolerance_radii = self.tolerance.radii
        return labels

    def __getitem__(self, index: int) -> Batch:
        sample = self.sample(self._keys[index])
        for step in self.transforms:
            sample = step(sample)
        if self.compact_labels is None:
            return to_batch(sample)
        # Compact labels live in object frames: they are bound to the poses only now, after
        # every transform that moves the scene (and its poses) has run.
        scene = sample.key.scene
        marks = self.instance_ids(self.scene_object_ids(scene))
        compact = [
            self.compact_labels.labels(
                scene, marks.index(annotation.instance_id), annotation.object_id, annotation.pose
            )
            for annotation in sample.objects or ()
        ]
        sample.objects = []
        batch = to_batch(sample)
        assert batch.supervision is not None
        batch.supervision.extras[COMPACT_LABELS] = [compact]
        return batch
