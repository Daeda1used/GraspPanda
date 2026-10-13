"""Grasps from a depth image, a point cloud or a dataset frame, through a config's own pipeline.

    from grasppanda import GraspInferencer

    inferencer = GraspInferencer("gsnet/gsnet_sunet14_g1b-realsense_10e", "gsnet.pth")
    grasps = inferencer(depth=depth_mm, intrinsics=K, depth_scale=1000.0)
    array = grasps.to_graspnet_array()          # (G, 17), camera frame, metres

**What runs.** The transforms of the config's test pipeline (``data.<evaluation.split>``),
built from the same config nodes ``grasppanda test`` builds them from: the image-space
``preprocess`` (back-projection, workspace), the point-space ``transforms`` (sampling,
input features), then ``model.predict`` and, on request, one of the config's
post-processing chains. A dataset frame goes through the dataset reader itself, so it
yields the grasps ``grasppanda test`` writes for that frame.

**Units and frames.** Lengths are metres. Images are indexed ``[row, column]``. Points
and grasps are in the camera frame, the optical frame of the depth sensor: x to the
right, y down, z forward along the viewing direction. A grasp's rotation has the
approach direction in column 0, the closing direction in column 1 and the finger-height
direction in column 2; ``translation`` is the grasp point between the fingers and
``depth`` how far the fingers reach past it along the approach
(:class:`~grasppanda.structures.GraspGroup`).

**Randomness.** Point sampling draws from a NumPy generator and some detectors draw
inside ``predict`` (HGGD jitters its regions). Both are seeded per call, from the
config by default, so the same input gives the same grasps.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grasppanda.config import Config
from grasppanda.datasets import BaseDataset
from grasppanda.models import BaseGraspDetector, PostProcessor
from grasppanda.registry import DATASETS, POSTPROCESSORS, TRANSFORMS
from grasppanda.structures import ContractError, GraspGroup, Observation, SampleKey
from grasppanda.transforms.base import ImageTransform, PointTransform
from grasppanda.transforms.frame import Frame, PointSample, align, to_batch
from grasppanda.transforms.image import (
    DepthToCloud,
    NoWorkspace,
    SegmentationBoxWorkspace,
    TableBoxWorkspace,
    WorkspaceProvider,
)

__all__ = ["GraspInferencer", "InferenceWarning", "InputError", "parse_frame_key"]

#: Key of a sample that does not come from a dataset.
USER_KEY = SampleKey("input", 0, "camera", 0)
#: A depth image or cloud whose median range is outside these bounds (metres) is in
#: another unit; no depth camera used for grasping works outside them.
_PLAUSIBLE_RANGE = (0.03, 20.0)
_FRAME_KEY = re.compile(
    r"^(?P<dataset>[\w.-]+)[:/]scene_(?P<scene>\d+)/(?P<camera>[\w.-]+)/(?P<frame>\d+)$"
)
#: How to supply each observation field the detectors know.
_HOW_TO_SUPPLY = {
    "rgb": "the colour image registered to the depth image: rgb=... (--rgb FILE)",
    "depth": "a depth image: depth=... with intrinsics=... (--depth FILE --intrinsics ...)",
    "intrinsics": "the camera matrix of the depth image: intrinsics=... (--intrinsics ...)",
    "points": "a point cloud: points=..., or a depth image with intrinsics",
    "point_feats": "per-point input features, which a transform of the config's pipeline adds "
    "(the config's data section has none)",
}


class InputError(ContractError):
    """The input of an inference call is missing something or is not what it claims to be."""


class InferenceWarning(UserWarning):
    """The input differs from the conditions the config's test pipeline assumes."""


def parse_frame_key(text: str | SampleKey) -> SampleKey:
    """Read ``graspnet1b:scene_0100/realsense/0000`` (or the ``/`` form a key prints as)."""
    if isinstance(text, SampleKey):
        return text
    match = _FRAME_KEY.match(str(text).strip())
    if match is None:
        raise InputError(
            f"'{text}' does not name a dataset frame. Write <dataset>:scene_<N>/<camera>/<frame>, "
            "for example graspnet1b:scene_0100/realsense/0000."
        )
    return SampleKey(match["dataset"], int(match["scene"]), match["camera"], int(match["frame"]))


class CameraBoxWorkspace(WorkspaceProvider):
    """Keep the points strictly inside an axis-aligned box of the camera frame."""

    def __init__(self, lower: Sequence[float], upper: Sequence[float]) -> None:
        self.lower = [float(value) for value in lower]
        self.upper = [float(value) for value in upper]

    def inside(self, frame: Frame) -> np.ndarray:
        cloud, _ = frame.require_cloud(type(self).__name__)
        return _in_box(cloud.reshape(-1, 3), self.lower, self.upper).reshape(cloud.shape[:2])


def _in_box(points: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for axis in range(3):
        inside &= (points[:, axis] > lower[axis]) & (points[:, axis] < upper[axis])
    return inside


def _box(workspace: Any) -> tuple[list[float], list[float]]:
    values = np.asarray(workspace, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise InputError(
            "workspace must be six numbers, the corners (xmin, ymin, zmin, xmax, ymax, zmax) "
            "of a box in the camera frame in metres"
        )
    lower, upper = values[:3].tolist(), values[3:].tolist()
    if any(low >= high for low, high in zip(lower, upper)):
        raise InputError(f"workspace is empty: lower corner {lower} must be below {upper}")
    return lower, upper


def _extent(points: np.ndarray) -> str:
    low, high = points.min(axis=0), points.max(axis=0)
    return ", ".join(f"{axis} {low[i]:.2f} to {high[i]:.2f}" for i, axis in enumerate("xyz"))


def _check_range(median: float, what: str, too_large: str, too_small: str) -> None:
    """Refuse input whose typical range (metres) no depth camera used for grasping measures."""
    low, high = _PLAUSIBLE_RANGE
    if median > high:
        raise InputError(f"{what} has a median range of {median:.4g} m. {too_large}")
    if median < low:
        raise InputError(f"{what} has a median range of {median:.3g} m. {too_small}")


class GraspInferencer:
    """A config and its weights, ready to predict grasps on new input.

    Call it with exactly one kind of input::

        inferencer(depth=..., intrinsics=..., rgb=None, mask=None)   # a camera frame
        inferencer(points=...)                                       # a point cloud
        inferencer(frame="graspnet1b:scene_0100/realsense/0000")     # a dataset frame

    Each call returns one :class:`~grasppanda.structures.GraspGroup` in the camera
    frame, metres, in the order the detector emits them (not sorted; use
    ``sort_by_score()``). ``to_graspnet_array()`` gives the 17-column GraspNet layout
    ``[score, width, height, depth, R (9, row-major), t (3), object_id]``.

    Args:
        config: A config file, the name of a shipped config, or a loaded
            :class:`~grasppanda.config.Config`.
        weights: Weights for the model. ``None`` uses ``runtime.load_from`` of the
            config, then verified weights of the model zoo if it lists any for this
            config. An untrained model is never run.
        device: ``auto`` (CUDA when available), ``cpu``, ``cuda`` or ``cuda:N``; ``None``
            takes ``runtime.device`` of the config.
        overrides: ``key=value`` overrides applied when ``config`` is loaded here.
        split: The ``data`` section whose pipeline is applied; default
            ``evaluation.split`` of the config (``test``).

    Attributes:
        model: The detector, in evaluation mode on ``device``.
        required_inputs: Observation fields the detector reads.
        preprocess, transforms: The built steps of the pipeline; empty when the config's
            data section is not a scene dataset and so declares none.
        images: Sensor images the pipeline hands to the detector (``rgb``, ``depth``).
        variants: Names of the config's post-processing chains.
        weights_report: What happened to every key when the weights were loaded.
    """

    def __init__(
        self,
        config: str | Path | Config,
        weights: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        overrides: Sequence[str] = (),
        split: str | None = None,
    ) -> None:
        # Imported here: the engine imports the config package, which types its schema
        # with classes of the engine.
        from grasppanda.engine.builders import build_model
        from grasppanda.engine.checkpoint import load_weights
        from grasppanda.engine.errors import EngineError
        from grasppanda.engine.run import resolve_device
        from grasppanda.engine.seeding import determinism_restored, preserved_rng
        from grasppanda.ops import backends

        if isinstance(config, Config):
            if overrides:
                raise ValueError("overrides apply when a config is loaded; pass a path or a name")
            self.config = config
        else:
            self.config = Config.load(config, overrides, command="demo")
        self.cfg = cfg = self.config.resolved()
        runtime = cfg["runtime"]
        self.seed: int = int(runtime["seed"])
        self.device = resolve_device(str(device) if device is not None else runtime["device"])
        self.weights = self._resolve_weights(weights, runtime["load_from"])

        backends.configure(runtime["ops"])
        with determinism_restored(), preserved_rng(self.seed):
            self.model: BaseGraspDetector = build_model(cfg).to(self.device)
        try:
            self.weights_report = load_weights(
                self.model,
                self.weights,
                kind=cfg["evaluation"]["weights"],
                allow_mismatch=runtime["allow_weight_mismatch"],
            )
        except EngineError as error:
            raise type(error)(
                f"{error}\nThe weights must have been trained for (or converted to) the model "
                f"of this config: {self.config.path or 'the given config'}."
            ) from None
        self.model.eval()
        self.required_inputs = frozenset(self.model.required_inputs)

        self.section = split or cfg["evaluation"]["split"]
        self._node: Mapping[str, Any] | None = None
        loader = (cfg.get("data") or {}).get(self.section)
        if loader is not None:
            self._node = loader["dataset"]
        elif split is not None:
            raise InputError(f"the config has no data.{split} section to take a pipeline from")
        self.preprocess: list[ImageTransform] = []
        self.transforms: list[PointTransform] = []
        self.images: tuple[str, ...] = ()
        self.keep_full_points = True
        self.sample_seed: int | None = 0
        self._scene_pipeline = False
        if self._node is not None and self._is_scene_dataset(self._node):
            node = self._node
            self._scene_pipeline = True
            self.preprocess = [TRANSFORMS.build(step) for step in node.get("preprocess") or ()]
            self.transforms = [TRANSFORMS.build(step) for step in node.get("transforms") or ()]
            self.images = tuple(node.get("images") or ())
            self.keep_full_points = bool(node.get("keep_full_points", True))
            self.sample_seed = node.get("seed")
        self._chains: dict[str, list[PostProcessor]] = {
            name: [POSTPROCESSORS.build(step) for step in chain]
            for name, chain in cfg["evaluation"]["variants"].items()
        }
        self._dataset: BaseDataset | None = None
        self._warned: set[str] = set()

    # ------------------------------------------------------------------- construction

    def _resolve_weights(self, weights: str | Path | None, configured: str | None) -> Path:
        from grasppanda.engine.errors import EngineError
        from grasppanda.hub import zoo_weights

        chosen = weights or configured
        if chosen:
            path = Path(chosen).expanduser()
            if not path.is_file():
                raise EngineError(f"weights file not found: {path}")
            return path
        found, reason = zoo_weights(self.config.path) if self.config.path else (None, "")
        if found is not None:
            return found
        raise EngineError(
            "GraspInferencer needs weights (weights=..., --weights FILE): an untrained model "
            f"is never run. {reason}".strip()
        )

    @staticmethod
    def _is_scene_dataset(node: Mapping[str, Any]) -> bool:
        from grasppanda.datasets.scene import GraspSceneDataset

        cls = DATASETS.get(node["type"])
        return isinstance(cls, type) and issubclass(cls, GraspSceneDataset)

    @property
    def variants(self) -> list[str]:
        return list(self._chains)

    @property
    def dataset(self) -> BaseDataset:
        """The dataset of the pipeline's ``data`` section, built on first use."""
        if self._dataset is None:
            if self._node is None:
                raise InputError(
                    f"the config has no data.{self.section} section, so it names no dataset to "
                    "read a frame from"
                )
            self._dataset = DATASETS.build(self._node)
        return self._dataset

    def describe(self) -> str:
        """The pipeline in a few lines, for a console header."""
        steps = [repr(step) for step in (*self.preprocess, *self.transforms)]
        if not self._scene_pipeline:
            steps = ["none declared: a depth image is back-projected, a cloud is used as it is"]
        chains = ", ".join(
            f"{name} ({' + '.join(type(step).__name__ for step in chain) or 'no post-processing'})"
            for name, chain in self._chains.items()
        )
        return "\n".join(
            [
                f"model     {type(self.model).__name__} on {self.device}, reads "
                f"{', '.join(sorted(self.required_inputs))}",
                f"pipeline  data.{self.section}: " + " -> ".join(steps),
                f"variants  {chains}",
            ]
        )

    # ------------------------------------------------------------------------- inputs

    def _warn(self, kind: str, message: str) -> None:
        if kind not in self._warned:  # once per inferencer: a loop over frames stays readable
            self._warned.add(kind)
            warnings.warn(message, InferenceWarning, stacklevel=4)

    def _workspace(
        self, mask: np.ndarray | None, table_from_camera: np.ndarray | None, workspace: Any
    ) -> tuple[WorkspaceProvider, bool]:
        """The workspace provider of this call and whether the frame needs a stand-in table."""
        configured = next(
            (step for step in self.preprocess if isinstance(step, WorkspaceProvider)), None
        )
        name = type(configured).__name__ if configured is not None else "none"
        if workspace is not None:
            if mask is not None:
                self._warn("mask+box", "both mask and workspace were given; the box is used")
            return CameraBoxWorkspace(*_box(workspace)), False
        if configured is None:
            if mask is not None:
                self._warn("mask-unused", "mask is ignored: this config declares no workspace")
            return NoWorkspace(), False
        needs_mask = configured.uses_ground_truth
        needs_table = isinstance(configured, (SegmentationBoxWorkspace, TableBoxWorkspace))
        if not needs_mask and mask is not None:
            self._warn(
                "mask-unused",
                f"mask is ignored: the pipeline of this config has no workspace that reads a "
                f"segmentation (its provider is {name}). Use workspace=... to restrict the "
                "region.",
            )
        if needs_mask and mask is None:
            self._warn(
                "no-mask",
                f"The test pipeline of this config crops the cloud with {name}, which reads a "
                "ground-truth segmentation; the weights were trained and tested on clouds "
                "cropped that way. No mask was given, so every pixel with depth is used. "
                "Expect grasps on the surroundings too and fewer points on the objects; pass "
                "mask=... (object pixels) or workspace=... (a box in the camera frame) to "
                "restrict the input.",
            )
            return NoWorkspace(), False
        if needs_table and table_from_camera is None:
            if needs_mask:
                self._warn(
                    "no-table",
                    f"{name} builds its box in the table frame, but table_from_camera was not "
                    "given: the box around the mask is aligned with the camera axes instead. "
                    "It is the same kind of crop, not the identical one.",
                )
                return configured, True
            self._warn(
                "no-table",
                f"{name} needs table_from_camera, which was not given; every pixel with depth "
                "is used.",
            )
            return NoWorkspace(), False
        return configured, False

    def _rng(self, seed: int | Sequence[int] | np.random.Generator | None) -> np.random.Generator:
        if isinstance(seed, np.random.Generator):
            return seed
        if seed is None:
            seed = self.sample_seed
        if seed is None:  # the config asks for a fresh sample every time, as its dataset does
            return np.random.default_rng(int(torch.randint(2**62, ()).item()))
        return np.random.default_rng(seed)

    def _finish(self, sample: PointSample, rng: np.random.Generator) -> Observation:
        sample.rng = rng
        for step in self.transforms:
            sample = step(sample)
        return to_batch(sample).observation

    def _from_depth(
        self,
        depth: Any,
        intrinsics: Any,
        rgb: Any,
        mask: Any,
        depth_scale: float,
        table_from_camera: Any,
        workspace: Any,
        rng: np.random.Generator,
    ) -> Observation:
        depth = np.asarray(depth)
        if depth.ndim != 2 or depth.dtype.kind not in "uif":
            raise InputError(
                f"depth must be a (H, W) array of numbers, got shape {depth.shape} and dtype "
                f"{depth.dtype}. A colour-mapped depth picture is not a depth image."
            )
        if intrinsics is None:
            raise InputError(
                "a depth image needs the intrinsics of the camera that took it: intrinsics=... "
                "as a 3 x 3 matrix or (fx, fy, cx, cy) in pixels (--intrinsics fx,fy,cx,cy)"
            )
        matrix = _camera_matrix(intrinsics, depth.shape)
        if not depth_scale > 0:
            raise InputError(
                f"depth_scale must be positive (depth units per metre), got {depth_scale}"
            )
        if depth.dtype.kind == "f" and not np.isfinite(depth).all():
            depth = np.where(np.isfinite(depth), depth, 0)  # NaN and Inf mean "no measurement"
        measured = depth[depth > 0]
        if measured.size == 0:
            raise InputError("the depth image has no positive value: there is nothing to see")
        raw = float(np.median(measured))
        _check_range(
            raw / depth_scale,
            f"the depth image (median value {raw:.4g}, depth_scale={depth_scale:g})",
            "The values are probably millimetres: pass depth_scale=1000 (--depth-scale 1000).",
            "The declared unit is too small: depth_scale is the number of depth units per "
            "metre, 1 for an image in metres and 1000 for millimetres.",
        )

        frame = Frame(key=USER_KEY, depth=depth, depth_scale=float(depth_scale), intrinsics=matrix)
        if rgb is not None:
            frame.rgb = _colour_image(rgb, depth.shape)
        if "rgb" in self.images and frame.rgb is None:
            raise InputError(
                f"{type(self.model).__name__} reads the colour image; pass " + _HOW_TO_SUPPLY["rgb"]
            )
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != depth.shape:
                raise InputError(
                    f"mask is {mask.shape} but the depth image is {depth.shape}; they must be "
                    "registered pixel for pixel"
                )
            if not (mask > 0).any():
                raise InputError("mask has no positive pixel: it marks no object")
        if table_from_camera is not None:
            table_from_camera = np.asarray(table_from_camera, dtype=np.float64)
            if table_from_camera.shape != (4, 4):
                raise InputError(
                    f"table_from_camera must be a (4, 4) transform, got {table_from_camera.shape}"
                )
        provider, stand_in = self._workspace(mask, table_from_camera, workspace)
        if provider.uses_ground_truth:  # a segmentation-based provider: the mask plays that part
            frame.segmentation = mask
        frame.table_from_camera = np.eye(4) if stand_in else table_from_camera

        steps = self.preprocess or [DepthToCloud(), provider]
        if not any(isinstance(step, DepthToCloud) for step in steps):
            raise InputError("the config's `preprocess` has no DepthToCloud step")
        for step in steps:
            frame = (provider if isinstance(step, WorkspaceProvider) else step)(frame)
        assert frame.cloud is not None and frame.valid is not None
        if not frame.valid.any():
            seen = frame.cloud[frame.depth > 0]
            raise InputError(
                f"the workspace is empty: no pixel with depth lies inside it "
                f"({type(provider).__name__}). The cloud spans {_extent(seen)} metres in the "
                "camera frame (x right, y down, z forward)."
            )
        sample = align(frame, keep_full_points=self.keep_full_points, images=self.images)
        return self._finish(sample, rng)

    def _from_points(self, points: Any, workspace: Any, rng: np.random.Generator) -> Observation:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3 or points.dtype.kind not in "uif":
            raise InputError(
                f"points must be a (N, 3) array of x, y, z in the camera frame, got shape "
                f"{points.shape}"
            )
        points = points[:, :3].astype(np.float64)
        points = points[np.isfinite(points).all(axis=1)]  # organised clouds mark holes with NaN
        if len(points) == 0:
            raise InputError("the point cloud has no finite point")
        _check_range(
            float(np.median(np.linalg.norm(points, axis=1))),
            "the point cloud",
            "Points are metres in the camera frame; a millimetre cloud must be divided by 1000.",
            "Points are metres in the camera frame, so a table-top scene lies some decimetres "
            "from the origin; this cloud is in another unit or another frame.",
        )
        if workspace is not None:
            inside = _in_box(points, *_box(workspace))
            if not inside.any():
                raise InputError(
                    f"the workspace is empty: no point lies inside it. The cloud spans "
                    f"{_extent(points)} metres in the camera frame (x right, y down, z forward)."
                )
            points = points[inside]
        elif any(step.uses_ground_truth for step in self.preprocess):
            self._warn(
                "points-uncropped",
                "The test pipeline of this config crops the cloud around the objects with a "
                "ground-truth segmentation. A point cloud is used as it is given: crop it to "
                "the workspace yourself, or pass workspace=....",
            )
        sample = PointSample(
            key=USER_KEY,
            intrinsics=np.eye(3),
            points=points,
            pixels=np.arange(len(points)),
            image_size=(1, len(points)),
            full_points=points if self.keep_full_points else None,
        )
        observation = self._finish(sample, rng)
        observation.intrinsics = None  # a bare cloud has no camera matrix
        return observation

    def _from_frame(self, frame: str | SampleKey) -> Observation:
        key = parse_frame_key(frame)
        dataset = self.dataset
        if key.dataset != dataset.name:
            raise InputError(
                f"{key} is a frame of '{key.dataset}', but data.{self.section} of this config "
                f"reads '{dataset.name}'"
            )
        camera = getattr(dataset, "camera", key.camera)
        if key.camera != camera:
            raise InputError(
                f"{key} was taken by camera '{key.camera}', but data.{self.section} of this "
                f"config reads '{camera}'. Use the config of that camera, or override "
                f"data.{self.section}.dataset.camera={key.camera}."
            )
        keys = dataset.keys
        if key in keys:
            return dataset[keys.index(key)].observation
        if not self._scene_pipeline:
            raise InputError(f"{key} is not one of the {len(keys)} frames of data.{self.section}")
        # A frame of the dataset outside the configured split: same reader, same steps.
        try:
            sample = dataset.sample(key)
        except FileNotFoundError as error:
            raise InputError(f"{key} does not exist in the dataset: {error}") from error
        for step in dataset.transforms:
            sample = step(sample)
        return to_batch(sample).observation

    # ----------------------------------------------------------------------- the calls

    def observe(
        self,
        *,
        depth: Any = None,
        intrinsics: Any = None,
        rgb: Any = None,
        mask: Any = None,
        points: Any = None,
        frame: str | SampleKey | None = None,
        depth_scale: float = 1.0,
        table_from_camera: Any = None,
        workspace: Any = None,
        seed: int | Sequence[int] | np.random.Generator | None = None,
    ) -> Observation:
        """Turn the input into the :class:`~grasppanda.structures.Observation` the model reads.

        This is the preprocessing half of a call; see :meth:`__call__` for the arguments.
        The result holds what the config's pipeline produces: ``points`` (the sampled
        cloud), ``full_points`` (the cloud before sampling, when the config keeps it) and
        the images the detector reads.
        """
        given = [
            name
            for name, value in (("depth", depth), ("points", points), ("frame", frame))
            if value is not None
        ]
        if len(given) != 1:
            raise InputError(
                "give exactly one input: depth= (with intrinsics=), points= or frame=; got "
                + (" and ".join(given) or "none")
            )
        if frame is not None:
            extra = [
                name
                for name, value in (
                    ("intrinsics", intrinsics),
                    ("rgb", rgb),
                    ("mask", mask),
                    ("workspace", workspace),
                    ("table_from_camera", table_from_camera),
                    ("seed", seed),
                )
                if value is not None
            ]
            if extra:
                raise InputError(
                    f"a dataset frame is read exactly as `grasppanda test` reads it; "
                    f"{', '.join(extra)} cannot be combined with frame="
                )
            observation = self._from_frame(frame)
        elif points is not None:
            extra = [
                n
                for n, v in (("rgb", rgb), ("mask", mask), ("intrinsics", intrinsics))
                if v is not None
            ]
            if extra:
                raise InputError(
                    f"{', '.join(extra)} belong to a depth image; with points= only "
                    "workspace= and seed= apply"
                )
            observation = self._from_points(points, workspace, self._rng(seed))
        else:
            observation = self._from_depth(
                depth,
                intrinsics,
                rgb,
                mask,
                depth_scale,
                table_from_camera,
                workspace,
                self._rng(seed),
            )
        self._require(observation, given[0])
        return observation

    def _require(self, observation: Observation, given: str) -> None:
        missing = sorted(self.required_inputs - observation.provides())
        if not missing:
            return
        name = type(self.model).__name__
        needs = "; ".join(f"{field}: {_HOW_TO_SUPPLY.get(field, field)}" for field in missing)
        raise InputError(
            f"{name} reads {', '.join(sorted(self.required_inputs))} and the {given} input "
            f"provides {', '.join(sorted(observation.provides())) or 'nothing'}. Missing: {needs}."
        )

    def chain(
        self,
        *,
        collision_filter: bool = False,
        variant: str | None = None,
        post: Sequence[str] | str | None = None,
    ) -> list[PostProcessor]:
        """The post-processing chain a call applies.

        A named variant of the config or its collision filter, followed by the steps of
        ``post``: short text such as ``"nms+top-k=20"`` or ``["gripper-limits=0.08",
        "top-k=20"]`` (:mod:`grasppanda.combine.post`), built on the spot.
        """
        from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter

        if post:
            from grasppanda.combine.post import PostSpecError, chain_nodes

            try:
                extra = [POSTPROCESSORS.build(node) for node in chain_nodes(post)]
            except PostSpecError as error:
                raise InputError(str(error)) from None
            return [*self.chain(collision_filter=collision_filter, variant=variant), *extra]
        if variant is not None:
            if variant not in self._chains:
                raise InputError(
                    f"the config has no post-processing variant '{variant}'; it defines "
                    f"{', '.join(self._chains) or 'none'} under evaluation.variants"
                )
            return self._chains[variant]
        if not collision_filter:
            return []
        for chain in self._chains.values():
            if any(isinstance(step, ModelFreeCollisionFilter) for step in chain):
                return chain
        raise InputError(
            "collision_filter=True applies the config's variant that holds a "
            "ModelFreeCollisionFilter, and this config has none (evaluation.variants: "
            f"{', '.join(self._chains) or 'none'}). Name a variant with variant=..., or add "
            "one to the config."
        )

    @torch.no_grad()
    def predict(
        self,
        observation: Observation,
        *,
        collision_filter: bool = False,
        variant: str | None = None,
        seed: int | None = None,
        post: Sequence[str] | str | None = None,
    ) -> GraspGroup:
        """Run the detector, then the requested post-processing, on one observation.

        Args:
            observation: One frame, as :meth:`observe` returns it.
            collision_filter: Apply the config's variant that holds a model-free
                collision filter (``cd`` in the shipped configs): grasps whose gripper
                would hit observed points are removed.
            variant: Apply this chain of ``evaluation.variants`` instead, by name.
            seed: Seed of the random draws inside the detector; default ``runtime.seed``.
            post: Further steps, applied after the variant or the collision filter:
                ``"nms+top-k=20"`` (see :meth:`chain`).
        """
        from grasppanda.engine.seeding import determinism_restored, preserved_rng, seed_everything

        if len(observation) != 1:
            raise InputError(f"predict() takes one frame, got a batch of {len(observation)}")
        steps = self.chain(collision_filter=collision_filter, variant=variant, post=post)
        observation = observation.to(self.device)
        seed = self.seed if seed is None else seed
        with determinism_restored(), preserved_rng(seed):
            if self.cfg["runtime"]["deterministic"]:  # as a run of this config would predict
                seed_everything(seed, deterministic=True)
            groups = self.model.predict(observation)
        if len(groups) != 1 or not isinstance(groups[0], GraspGroup):
            raise ContractError(
                f"{type(self.model).__name__}.predict returned {len(groups)} result(s) for one frame"
            )
        grasps = groups[0]
        grasps.key = grasps.key or observation.keys[0]
        return self.postprocess(grasps, observation, steps)

    @torch.no_grad()
    def postprocess(
        self, grasps: GraspGroup, observation: Observation, steps: Sequence[PostProcessor]
    ) -> GraspGroup:
        """Apply post-processors (see :meth:`chain`) to the grasps of one observation."""
        observation = observation.to(self.device)
        for step in steps:
            grasps = step(grasps, observation)
        return grasps

    def __call__(
        self,
        *,
        depth: Any = None,
        intrinsics: Any = None,
        rgb: Any = None,
        mask: Any = None,
        points: Any = None,
        frame: str | SampleKey | None = None,
        depth_scale: float = 1.0,
        table_from_camera: Any = None,
        workspace: Any = None,
        seed: int | None = None,
        collision_filter: bool = False,
        variant: str | None = None,
        post: Sequence[str] | str | None = None,
    ) -> GraspGroup:
        """Predict grasps on one input.

        Args:
            depth: ``(H, W)`` depth image, zero (or NaN) where the sensor measured
                nothing. Metres by default; for a 16-bit millimetre image pass the
                integer array and ``depth_scale=1000``.
            intrinsics: Camera matrix of the depth image: ``(3, 3)``, or
                ``(fx, fy, cx, cy)`` in pixels.
            rgb: ``(H, W, 3)`` colour image registered to ``depth``, uint8 or float in
                ``[0, 1]``. Required by detectors that read it (HGGD); ignored by
                point-cloud detectors.
            mask: ``(H, W)`` object pixels (non-zero), from any segmentation. It takes
                the place of the dataset's segmentation in the config's workspace
                provider, which crops the cloud to a box around the marked pixels (the
                table below the objects stays in). Ignored, with a warning, by a
                pipeline without such a provider.
            points: ``(N, 3)`` cloud in the camera frame, metres, already cropped to
                the workspace. Rows with NaN are dropped. The image-space steps of the
                pipeline do not apply; sampling and input features do.
            frame: A dataset frame, ``<dataset>:scene_<N>/<camera>/<frame>`` or a
                :class:`~grasppanda.structures.SampleKey`. It is read by the dataset
                reader of the config, so the result is what ``grasppanda test`` dumps
                for that frame. Needs the dataset on disk.
            depth_scale: Depth units per metre: ``1`` for metres, ``1000`` for millimetres.
            table_from_camera: ``(4, 4)`` transform from the camera frame to a frame
                fixed to the table (z up). Only workspace providers that work in the
                table frame use it.
            workspace: ``(xmin, ymin, zmin, xmax, ymax, zmax)``, a box in the camera
                frame in metres. Replaces the config's workspace provider.
            seed: Seed of this call: the point sampling and the detector's random
                draws. ``None`` uses the config's seeds (``data.<split>.dataset.seed``
                and ``runtime.seed``), which is what ``grasppanda test`` uses.
            collision_filter: Apply the config's collision-filter variant.
            variant: Apply this post-processing chain of the config, by name.
            post: Further post-processing steps as short text, applied afterwards:
                ``"gripper-limits=0.08+nms+top-k=20"`` (see :meth:`chain`).

        Returns:
            The grasps of this input, camera frame, metres; possibly empty.

        Raises:
            InputError: The input is incomplete, in the wrong unit, or the detector
                needs something that was not given. The message says what.
        """
        observation = self.observe(
            depth=depth,
            intrinsics=intrinsics,
            rgb=rgb,
            mask=mask,
            points=points,
            frame=frame,
            depth_scale=depth_scale,
            table_from_camera=table_from_camera,
            workspace=workspace,
            seed=seed,
        )
        return self.predict(
            observation,
            collision_filter=collision_filter,
            variant=variant,
            seed=seed,
            post=post,
        )


def _camera_matrix(intrinsics: Any, image_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape == (4,):
        fx, fy, cx, cy = values
        values = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise InputError(
            "intrinsics must be a (3, 3) camera matrix or (fx, fy, cx, cy) in pixels, got "
            f"shape {np.shape(intrinsics)}"
        )
    fx, fy, cx, cy = values[0, 0], values[1, 1], values[0, 2], values[1, 2]
    height, width = image_shape
    if fx <= 0 or fy <= 0:
        raise InputError(f"focal lengths must be positive, got fx={fx:g}, fy={fy:g}")
    if not (0 <= cx <= width and 0 <= cy <= height) or min(fx, fy) < 0.2 * width:
        raise InputError(
            f"intrinsics fx={fx:g}, fy={fy:g}, cx={cx:g}, cy={cy:g} do not fit a {width} x "
            f"{height} image: they are in pixels of the depth image (the principal point "
            "lies inside it, the focal length is of the order of the image width). "
            "Intrinsics of another resolution must be scaled with the image."
        )
    return values


def _colour_image(rgb: Any, image_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputError(f"rgb must be (H, W, 3), got shape {image.shape}")
    if image.shape[:2] != tuple(image_shape):
        raise InputError(
            f"rgb is {image.shape[1]} x {image.shape[0]} but the depth image is "
            f"{image_shape[1]} x {image_shape[0]}; the colour image must be registered to the "
            "depth image, pixel for pixel (align it with your camera's SDK)"
        )
    if image.dtype == np.uint8:
        return image
    if image.dtype.kind == "f" and image.size and image.min() >= 0.0 and image.max() <= 1.0:
        return np.round(image * 255.0).astype(np.uint8)
    raise InputError(f"rgb must be uint8 or float in [0, 1], got {image.dtype}")
