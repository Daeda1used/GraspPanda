"""Working containers of the data pipeline and the label-alignment boundary.

``Frame`` is one image-space sample: what the sensor and the calibration provide, plus
the organised cloud and validity mask that image transforms build. :func:`align` turns
it into a ``PointSample``: the masked cloud, each point keyed by its pixel. Per-point
labels are attached to that set and nothing before :func:`align` may be changed without
changing the preprocessing fingerprint.

Geometry stays in NumPy ``float64`` until :func:`to_batch`, as the reference pipelines
compute it, so that a reproduction is bit-comparable; the batch holds ``float32``.
Rigid maps are evaluated coordinate by coordinate in a fixed order (:func:`transform_points`)
rather than through a BLAS product, so the same input gives the same bits on every
machine and dataloader workers do not start BLAS thread pools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from grasppanda.structures import (
    Batch,
    ContractError,
    ObjectGraspLabels,
    Observation,
    PointSet,
    Provenance,
    SampleKey,
    Supervision,
)
from grasppanda.structures.observation import DeferredGrids

__all__ = [
    "IMAGES",
    "Frame",
    "GraspLabelSource",
    "ObjectAnnotation",
    "PointSample",
    "align",
    "to_batch",
    "transform_columns",
    "transform_points",
]


#: Sensor images a sample can carry beside its cloud.
IMAGES = ("rgb", "depth")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a ``(3, 3)`` linear or ``(3, 4)`` / ``(4, 4)`` affine map to ``(N, 3)`` points.

    Each output coordinate is ``((m0 * x + m1 * y) + m2 * z) + t`` in double precision.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape not in ((3, 3), (3, 4), (4, 4)):
        raise ValueError(f"expected a (3, 3), (3, 4) or (4, 4) matrix, got {matrix.shape}")
    points = np.asarray(points, dtype=np.float64)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    out = np.empty((len(points), 3), dtype=np.float64)
    for row in range(3):
        column = matrix[row, 0] * x + matrix[row, 1] * y + matrix[row, 2] * z
        out[:, row] = column + matrix[row, 3] if matrix.shape[1] == 4 else column
    return out


def transform_columns(
    points: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`transform_points` with every output coordinate as its own contiguous array.

    Value for value what ``transform_points(points, matrix)[:, i]`` holds: the same
    operations in the same order, written into one work array instead of fresh ones.
    For a step that looks at one coordinate at a time over a whole image (a box test,
    an extent) this takes less than half the time of building the interleaved result.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape not in ((3, 3), (3, 4), (4, 4)):
        raise ValueError(f"expected a (3, 3), (3, 4) or (4, 4) matrix, got {matrix.shape}")
    points = np.asarray(points, dtype=np.float64)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    work = np.empty(len(points), dtype=np.float64)
    columns = []
    for row in range(3):
        column = matrix[row, 0] * x
        np.multiply(matrix[row, 1], y, out=work)
        column += work
        np.multiply(matrix[row, 2], z, out=work)
        column += work
        if matrix.shape[1] == 4:
            column += matrix[row, 3]
        columns.append(column)
    return columns[0], columns[1], columns[2]


@dataclass(eq=False)
class Frame:
    """One frame in image space.

    Attributes:
        key: Identity of the frame.
        depth: ``(H, W)`` raw sensor units, zero where invalid.
        depth_scale: Raw units per metre (1000 for millimetres).
        intrinsics: ``(3, 3)`` camera matrix.
        table_from_camera: ``(4, 4)`` calibrated transform from the camera frame to a
            frame fixed to the table (z up), if the dataset provides one.
        rgb: ``(H, W, 3)`` uint8 colour image, when the dataset was asked for it.
        segmentation: ``(H, W)`` **ground-truth** object ids, zero for background.
            ``None`` unless labels are loaded or a transform declares that it uses
            ground truth.
        instances: ``(H, W)`` **ground-truth** instance ids, zero for background, for
            datasets in which one object model can occur several times in a scene.
            ``None`` means ``segmentation`` already tells instances apart.
        cloud: ``(H, W, 3)`` organised cloud in the camera frame, metres.
        valid: ``(H, W)`` pixels that are still part of the observation.
        workspace: Name of the workspace provider that was applied, if any.
        provenance: Where inputs that shaped ``valid`` came from, by name.
    """

    key: SampleKey
    depth: np.ndarray
    depth_scale: float
    intrinsics: np.ndarray
    table_from_camera: np.ndarray | None = None
    rgb: np.ndarray | None = None
    segmentation: np.ndarray | None = None
    instances: np.ndarray | None = None
    cloud: np.ndarray | None = None
    valid: np.ndarray | None = None
    workspace: str | None = None
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def require_cloud(self, user: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cloud, valid)`` or explain which step is missing."""
        if self.cloud is None or self.valid is None:
            raise ContractError(
                f"{user} needs the organised cloud of {self.key}; put DepthToCloud before it "
                "in `preprocess`"
            )
        return self.cloud, self.valid

    def table_points(self, user: str) -> np.ndarray:
        """``(H * W, 3)`` cloud in the table frame."""
        cloud, _ = self.require_cloud(user)
        if self.table_from_camera is None:
            raise ContractError(
                f"{user} works in the table frame, but {self.key} has no camera-to-table "
                "calibration"
            )
        return transform_points(cloud.reshape(-1, 3), self.table_from_camera)

    def table_coordinates(self, user: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The x, y and z of :meth:`table_points`, each as its own ``(H * W,)`` array."""
        cloud, _ = self.require_cloud(user)
        if self.table_from_camera is None:
            raise ContractError(
                f"{user} works in the table frame, but {self.key} has no camera-to-table "
                "calibration"
            )
        return transform_columns(cloud.reshape(-1, 3), self.table_from_camera)


class GraspLabelSource(Protocol):
    """Dense grasp labels of one object instance that have not been read yet."""

    num_points: int

    def take(
        self, rows: np.ndarray, *, collision: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(points, scores, widths, collision)`` of the given label points."""

    def tolerance(self, rows: np.ndarray) -> np.ndarray | None:
        """Return the ``(K, V, A, D)`` grasp tolerance of the given label points, or
        ``None`` when the source carries no tolerance labels."""

    def defer(self, rows: np.ndarray, *, collision: str) -> DeferredGrids:
        """Return a reader of the grids of the given label points that leaves them where
        they are stored (``SampleGraspLabels(grids="deferred")``). A source that offers
        it also exposes its annotated ``points``."""


@dataclass(eq=False)
class ObjectAnnotation:
    """One object instance of a frame on its way through the pipeline.

    Attributes:
        object_id: Dataset id of the object model.
        instance_id: Value that marks this instance in the ``instance`` point label.
        pose: ``(4, 4)`` float32 object-to-camera transform. After a flip augmentation
            it is a reflection, exactly as the points are.
        source: Where the dense labels can be read from; ``None`` once they are loaded
            (or when the dataset serves poses only).
        points, scores, widths, collision, tolerance: The loaded labels, see
            :class:`~grasppanda.structures.ObjectGraspLabels`.
        deferred: Reader of the grids of ``points``, when the label transform left them
            in the store; ``scores``, ``widths``, ``collision`` and ``tolerance`` are
            then ``None``.
    """

    object_id: int
    instance_id: int
    pose: np.ndarray
    source: GraspLabelSource | None = None
    points: np.ndarray | None = None
    scores: np.ndarray | None = None
    widths: np.ndarray | None = None
    collision: np.ndarray | None = None
    tolerance: np.ndarray | None = None
    deferred: DeferredGrids | None = None


@dataclass(eq=False)
class PointSample:
    """One frame after the label-alignment boundary.

    Attributes:
        key: Identity of the frame.
        intrinsics: ``(3, 3)`` camera matrix of the original image.
        points: ``(N, 3)`` float64 points in the camera frame.
        pixels: ``(N,)`` flat index ``v * W + u`` of the pixel each point came from. This
            is the key of pixel-keyed labels and survives sampling.
        image_size: ``(H, W)``.
        full_points: ``(M, 3)`` masked cloud before sampling, moved rigidly with
            ``points``. ``None`` when the dataset does not keep it.
        point_labels: Named ``(N, ...)`` labels aligned with ``points``.
        objects: Object instances with poses and (lazy) grasp labels; ``None`` when the
            sample carries no supervision.
        feats: ``(N, C)`` input features.
        index: ``(N,)`` rows of the masked cloud that ``points`` are, once sampled.
        rng: Source of randomness for this sample.
        provenance: See :class:`Frame`.
        rgb: ``(H, W, 3)`` uint8 colour image of the frame, when the dataset serves images.
        depth: ``(H, W)`` float32 sensor depth in metres, zero where invalid, when the
            dataset serves images. It is the unmasked sensor image: a workspace crop
            restricts ``points``, not this array.
        extras: Method-specific labels of the frame, by name; arrays or mappings of arrays.
            They become ``Supervision.extras``.
        camera_bound: Names in ``extras`` that are expressed in the camera frame or on the
            pixel grid. A rigid map of the cloud cannot carry them along, so it is refused
            while one is present; entries that do not depend on the pose of the cloud
            (a statistic of the frame, say) are not listed here.
    """

    key: SampleKey
    intrinsics: np.ndarray
    points: np.ndarray
    pixels: np.ndarray
    image_size: tuple[int, int]
    full_points: np.ndarray | None = None
    point_labels: dict[str, np.ndarray] = field(default_factory=dict)
    objects: list[ObjectAnnotation] | None = None
    feats: np.ndarray | None = None
    index: np.ndarray | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    camera_bound: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.points)

    def gather(self, index: np.ndarray) -> None:
        """Keep ``points[index]`` and the labels of exactly those points."""
        for name, values in self.point_labels.items():
            if len(values) != len(self.points):
                raise ContractError(
                    f"point label '{name}' of {self.key} has {len(values)} rows for "
                    f"{len(self.points)} points"
                )
        if self.feats is not None:
            self.feats = self.feats[index]
        self.point_labels = {name: values[index] for name, values in self.point_labels.items()}
        self.points = self.points[index]
        self.pixels = self.pixels[index]
        self.index = index if self.index is None else self.index[index]

    def apply_linear(self, matrix: np.ndarray) -> None:
        """Apply a ``(3, 3)`` rotation or reflection to points and object poses together."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"expected a (3, 3) matrix, got {matrix.shape}")
        carried = [name for name in ("rgb", "depth") if getattr(self, name) is not None]
        carried += sorted(self.camera_bound & self.extras.keys())
        if carried:
            raise ContractError(
                f"{self.key}: a rigid map of the cloud would leave the image-space data of "
                f"this sample behind ({', '.join(carried)}). Remove the "
                "point-space augmentation, or do not ask the dataset for images and "
                "image-keyed labels."
            )
        shared = self.full_points is self.points
        self.points = transform_points(self.points, matrix)
        if shared:
            self.full_points = self.points
        elif self.full_points is not None:
            self.full_points = transform_points(self.full_points, matrix)
        for annotation in self.objects or ():
            # The columns of a pose are points and directions of the object frame; they
            # move like the scene. Poses return to single precision after every step,
            # as the reference pipelines keep them.
            pose = annotation.pose.astype(np.float64)
            pose[:3, :] = transform_points(pose[:3, :].T, matrix).T
            annotation.pose = pose.astype(np.float32)


def align(
    frame: Frame, *, keep_full_points: bool = True, images: Sequence[str] = ()
) -> PointSample:
    """Cross the label-alignment boundary: the valid pixels become the point set.

    Points are ordered by pixel, row by row. Labels attached to the result are aligned
    to this set; later steps only gather from it. ``images`` names the sensor images
    (``rgb``, ``depth``) the sample keeps beside the cloud.
    """
    cloud, valid = frame.require_cloud("the label-alignment boundary")
    pixels = np.flatnonzero(valid)
    if len(pixels) == 0:
        raise ContractError(
            f"{frame.key}: no pixel survives preprocessing (valid depth and workspace)"
        )
    # np.take copies whole rows; indexing with an index array takes 2.5 times as long.
    points = np.take(cloud.reshape(-1, 3), pixels, axis=0)
    unknown = sorted(set(images) - set(IMAGES))
    if unknown:
        raise ValueError(f"unknown image(s) {unknown}; a sample can carry {list(IMAGES)}")
    if "rgb" in images and frame.rgb is None:
        raise ContractError(f"{frame.key}: the reader did not provide a colour image")
    return PointSample(
        key=frame.key,
        intrinsics=frame.intrinsics,
        points=points,
        pixels=pixels,
        image_size=(valid.shape[0], valid.shape[1]),
        full_points=points if keep_full_points else None,
        provenance=dict(frame.provenance),
        rgb=frame.rgb if "rgb" in images else None,
        depth=(
            (frame.depth / np.float32(frame.depth_scale)).astype(np.float32)
            if "depth" in images
            else None
        ),
    )


def _labels(annotation: ObjectAnnotation, key: SampleKey) -> ObjectGraspLabels:
    if annotation.points is not None and annotation.deferred is not None:
        return ObjectGraspLabels(
            object_id=annotation.object_id,
            pose=torch.from_numpy(annotation.pose),
            points=torch.from_numpy(annotation.points),
            deferred=annotation.deferred,
        )
    if annotation.points is None or annotation.scores is None or annotation.widths is None:
        raise ContractError(
            f"{key}: the grasp labels of object {annotation.object_id} were never loaded. Add a "
            "label transform such as SampleGraspLabels to `transforms`, or build the dataset "
            "with load_labels=False."
        )
    return ObjectGraspLabels(
        object_id=annotation.object_id,
        pose=torch.from_numpy(annotation.pose),
        points=torch.from_numpy(annotation.points),
        scores=torch.from_numpy(annotation.scores),
        widths=torch.from_numpy(annotation.widths),
        collision=None if annotation.collision is None else torch.from_numpy(annotation.collision),
        tolerance=None if annotation.tolerance is None else torch.from_numpy(annotation.tolerance),
    )


def _tensors(value: Any) -> Any:
    """Arrays of a (possibly nested) label entry as tensors; everything else unchanged."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, dict):
        return {name: _tensors(item) for name, item in value.items()}
    return value


def to_batch(sample: PointSample) -> Batch:
    """Convert a finished sample into a :class:`Batch` of one frame (float32 tensors)."""
    rgb = None
    if sample.rgb is not None:  # (H, W, 3) uint8 -> (1, 3, H, W) in [0, 1]
        rgb = torch.from_numpy(np.ascontiguousarray(sample.rgb)).permute(2, 0, 1)[None]
        rgb = rgb.to(torch.float32) / 255.0
    observation = Observation(
        keys=[sample.key],
        intrinsics=torch.from_numpy(sample.intrinsics.astype(np.float32))[None],
        depth=None if sample.depth is None else torch.from_numpy(sample.depth)[None],
        rgb=rgb,
        points=PointSet(torch.from_numpy(sample.points.astype(np.float32))),
        point_feats=None if sample.feats is None else torch.from_numpy(sample.feats),
        full_points=(
            None
            if sample.full_points is None
            else PointSet(torch.from_numpy(sample.full_points.astype(np.float32)))
        ),
        provenance=dict(sample.provenance),
    )
    if sample.objects is None:
        if sample.extras:
            raise ContractError(
                f"{sample.key}: the sample carries labels ({', '.join(sorted(sample.extras))}) "
                "but no supervision; build the dataset with load_labels=True"
            )
        return Batch(observation)
    supervision = Supervision(
        objects=[[_labels(annotation, sample.key) for annotation in sample.objects]],
        point_labels={
            name: torch.from_numpy(np.ascontiguousarray(values))
            for name, values in sample.point_labels.items()
        },
        extras={name: [_tensors(value)] for name, value in sample.extras.items()},
    )
    return Batch(observation, supervision)
