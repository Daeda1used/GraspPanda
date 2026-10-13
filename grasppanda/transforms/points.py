"""Point-space steps: sampling, grasp-label selection, augmentation, input features.

Everything here runs after the label-alignment boundary. A step that selects points
gathers every per-point label with the same indices; a step that moves points applies
the same rigid map to the object poses, so labels expressed in object frames stay
attached to the right place.

Each random step splits into ``draw`` (the randomness) and ``apply`` (the deterministic
effect). Parity tests feed indices recorded from a reference implementation to
``apply`` and compare the outputs exactly.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError

from .base import PointTransform
from .frame import PointSample, transform_points

__all__ = [
    "ConstantFeatures",
    "FilterObjects",
    "RandomFlip",
    "RandomRotation",
    "RandomSample",
    "SampleGraspLabels",
    "rotation_matrix",
]

_AXES = {"x": 0, "y": 1, "z": 2}


def rotation_matrix(axis: Literal["x", "y", "z"], angle: float) -> np.ndarray:
    """Right-handed rotation by ``angle`` radians about a coordinate axis."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")


@TRANSFORMS.register()
class RandomSample(PointTransform):
    """Reduce the cloud to exactly ``num_points`` points.

    With enough points they are drawn without replacement. With fewer, every point is
    kept once, in order, and the remainder is drawn with replacement. Both recipes of
    the GraspNet-1Billion family sample this way (15 000 points in Graspness, 20 000 in
    EconomicGrasp), at test time too.
    """

    source = "GraspNet-1Billion reference pipelines"

    def __init__(self, num_points: int) -> None:
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        self.num_points = int(num_points)

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count >= self.num_points:
            return rng.choice(count, self.num_points, replace=False)
        extra = rng.choice(count, self.num_points - count, replace=True)
        return np.concatenate([np.arange(count), extra])

    def apply(self, sample: PointSample, index: np.ndarray) -> PointSample:
        sample.gather(np.asarray(index, dtype=np.int64))
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(len(sample), sample.rng))


def _require_objects(sample: PointSample, user: str) -> None:
    if sample.objects is None:
        raise ContractError(
            f"{user} needs object annotations, but {sample.key} carries no supervision. "
            "Remove it from `transforms` or build the dataset with load_labels=True."
        )


@TRANSFORMS.register()
class FilterObjects(PointTransform):
    """Drop object instances that are barely visible in the current point set.

    An instance is kept when at least ``min_visible_points`` points carry its id in the
    ``instance`` label. Placed after sampling, this is the Graspness rule "skip objects
    with fewer than 50 sampled segmentation points".

    Args:
        min_visible_points: Points of the current sample an instance needs.
        exclude_object_ids: Object models that never supervise grasps, whatever their
            visibility. Their points stay in the cloud and keep their point labels.
            GraspNet-baseline and Scale-Balanced-Grasp train without object 18.
    """

    source = "GSNet (Wang et al., ICCV 2021)"

    def __init__(
        self, min_visible_points: int, exclude_object_ids: list[int] | None = None
    ) -> None:
        self.min_visible_points = int(min_visible_points)
        self.exclude_object_ids = sorted(int(i) for i in exclude_object_ids or ())

    def __call__(self, sample: PointSample) -> PointSample:
        name = type(self).__name__
        _require_objects(sample, name)
        if "instance" not in sample.point_labels:
            raise ContractError(f"{name} needs the 'instance' point label of {sample.key}")
        instance = sample.point_labels["instance"]
        sample.objects = [
            annotation
            for annotation in sample.objects or ()
            if annotation.object_id not in self.exclude_object_ids
            and np.count_nonzero(instance == annotation.instance_id) >= self.min_visible_points
        ]
        return sample


@TRANSFORMS.register()
class SampleGraspLabels(PointTransform):
    """Read the dense grasp labels of every remaining object, on a subset of its points.

    Args:
        point_fraction: Fraction of an object's annotated grasp points to keep.
        min_points: Keep at least this many (or all, if the object has fewer). Graspness
            keeps a random quarter with a minimum of 300: ``point_fraction: 0.25``,
            ``min_points: 300``.
        collision: What to do with the per-scene collision masks. ``"zero_scores"`` sets
            the friction score of every colliding grasp to 0, which marks it invalid
            (the Graspness recipe); ``"attach"`` leaves scores untouched and returns the
            mask in ``ObjectGraspLabels.collision``; ``"ignore"`` does not read them.
            Tolerance labels, when the source serves them, are zeroed with the scores.
        visible_within: Draw only from annotated points that lie within this distance
            (metres) of a point of the same object in the current sample, i.e. on the
            side of the object the camera sees. ``None`` (the Graspness recipe) draws
            from all annotated points; GraspNet-baseline uses 0.01. ``point_fraction``
            and ``min_points`` then apply to the visible points, and an object without
            any keeps no label point.
        grids: Where the ``(P, V, A, D)`` grids of the chosen label points are read.
            ``"dense"`` copies them into the sample, as the reference pipelines do: a
            batch of four frames then carries about 1.6 GiB through the loader and onto
            the device. ``"deferred"`` puts a reader in their place
            (:class:`~grasppanda.structures.observation.DeferredGrids`), and label
            matching reads the rows of the label points it matched, a few per cent of
            them. Both give the same labels, value for value, and draw the same random
            numbers; a run is the same run with either. ``"deferred"`` needs a label
            source that can defer, such as the memory-mapped label store.
    """

    source = "GraspNet-1Billion reference pipelines"

    def __init__(
        self,
        point_fraction: float,
        min_points: int,
        collision: Literal["zero_scores", "attach", "ignore"],
        visible_within: float | None = None,
        grids: Literal["dense", "deferred"] = "dense",
    ) -> None:
        if not 0 < point_fraction <= 1:
            raise ValueError(f"point_fraction must be in (0, 1], got {point_fraction}")
        if visible_within is not None and visible_within <= 0:
            raise ValueError(f"visible_within must be positive or null, got {visible_within}")
        if grids not in ("dense", "deferred"):
            raise ValueError(f"grids must be 'dense' or 'deferred', got {grids!r}")
        self.point_fraction = float(point_fraction)
        self.min_points = int(min_points)
        self.collision = collision
        self.visible_within = None if visible_within is None else float(visible_within)
        self.grids = grids

    def config(self) -> dict[str, Any]:
        """As :meth:`Transform.config`, without ``grids``.

        The configuration of a step identifies the labels it produces, and assets
        derived from a data recipe are keyed by it. Where the grids are read does not
        change a value, so it is not part of that identity.
        """
        node = super().config()
        del node["grids"]
        return node

    def count(self, available: int) -> int:
        """Number of label points kept of an object that has ``available``."""
        return min(max(int(available * self.point_fraction), self.min_points), available)

    def visible_rows(self, sample: PointSample, annotation) -> np.ndarray:
        """Annotated points of an object within ``visible_within`` of its sampled points."""
        name = type(self).__name__
        points = getattr(self._source(sample, annotation), "points", None)
        if points is None or "instance" not in sample.point_labels:
            raise ContractError(
                f"{name} with visible_within needs the annotated points of the label source "
                f"and the 'instance' point label of {sample.key}"
            )
        seen = sample.points[sample.point_labels["instance"] == annotation.instance_id]
        if len(seen) == 0:
            return np.zeros(0, dtype=np.int64)
        from scipy.spatial import cKDTree

        placed = transform_points(points, annotation.pose)
        distance, _ = cKDTree(seen).query(placed, k=1)
        return np.flatnonzero(distance < self.visible_within)

    def draw(self, sample: PointSample) -> list[np.ndarray]:
        _require_objects(sample, type(self).__name__)
        rows = []
        for annotation in sample.objects or ():
            source = self._source(sample, annotation)
            if self.visible_within is None:
                rows.append(
                    sample.rng.choice(
                        source.num_points, self.count(source.num_points), replace=False
                    )
                )
                continue
            visible = self.visible_rows(sample, annotation)
            rows.append(visible[sample.rng.choice(len(visible), self.count(len(visible)), False)])
        return rows

    def apply(self, sample: PointSample, rows: list[np.ndarray]) -> PointSample:
        _require_objects(sample, type(self).__name__)
        objects = sample.objects or []
        if len(rows) != len(objects):
            raise ValueError(f"got {len(rows)} index arrays for {len(objects)} objects")
        for annotation, index in zip(objects, rows):
            source = self._source(sample, annotation)
            index = np.asarray(index, dtype=np.int64)
            if self.grids == "deferred":
                annotation.points = self._points(sample, annotation)[index]
                annotation.deferred = source.defer(index, collision=self.collision)
                annotation.source = None
                continue
            points, scores, widths, collision = source.take(
                index, collision=self.collision != "ignore"
            )
            read_tolerance = getattr(source, "tolerance", None)
            tolerance = None if read_tolerance is None else read_tolerance(index)
            if self.collision == "zero_scores":
                np.putmask(scores, collision, 0)  # exact zeros, in place
                if tolerance is not None:
                    np.putmask(tolerance, collision, 0)
                collision = None
            annotation.points, annotation.scores, annotation.widths = points, scores, widths
            annotation.collision, annotation.tolerance = collision, tolerance
            annotation.source = None
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(sample))

    @staticmethod
    def _source(sample: PointSample, annotation):
        if annotation.source is None:
            raise ContractError(
                f"{sample.key}: object {annotation.object_id} has no grasp-label source "
                "(labels already loaded, or the dataset serves poses only)"
            )
        return annotation.source

    def _points(self, sample: PointSample, annotation) -> np.ndarray:
        """Annotated points of a source that can defer its grids."""
        source = self._source(sample, annotation)
        points = getattr(source, "points", None)
        if points is None or not callable(getattr(source, "defer", None)):
            raise ContractError(
                f"{sample.key}: {type(self).__name__} is configured with grids: deferred, but "
                f"the label source of object {annotation.object_id} "
                f"({type(source).__name__}) cannot leave its grids in place. Use grids: dense."
            )
        return np.asarray(points)


@TRANSFORMS.register()
class RandomFlip(PointTransform):
    """Mirror the scene along a camera axis with probability ``prob``.

    Points and object poses are reflected together; a flipped pose is therefore a
    reflection, not a rotation. Graspness and EconomicGrasp flip ``x`` with ``prob`` 0.5.
    """

    moves_points = True  # refused beside image-space data by the config check

    source = "GraspNet-1Billion reference pipelines"

    def __init__(self, axis: Literal["x", "y", "z"], prob: float) -> None:
        if axis not in _AXES:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self.axis = axis
        self.prob = float(prob)

    def draw(self, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.prob)

    def apply(self, sample: PointSample, flip: bool) -> PointSample:
        if flip:
            matrix = np.eye(3)
            matrix[_AXES[self.axis], _AXES[self.axis]] = -1.0
            sample.apply_linear(matrix)
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(sample.rng))


@TRANSFORMS.register()
class RandomRotation(PointTransform):
    """Rotate the scene about a camera axis by a uniform angle in ``[-max_angle, max_angle)``.

    Args:
        axis: Camera axis to rotate about. Graspness rotates about ``x`` (its source
            comment says z; the matrix in the code is an x-rotation).
        max_angle: Bound of the angle in degrees (Graspness: 30).
    """

    moves_points = True  # refused beside image-space data by the config check

    source = "GraspNet-1Billion reference pipelines"

    def __init__(self, axis: Literal["x", "y", "z"], max_angle: float) -> None:
        if axis not in _AXES:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
        if max_angle < 0:
            raise ValueError(f"max_angle must be non-negative, got {max_angle}")
        self.axis = axis
        self.max_angle = float(max_angle)

    def draw(self, rng: np.random.Generator) -> float:
        """An angle in radians."""
        return math.radians((2.0 * rng.random() - 1.0) * self.max_angle)

    def apply(self, sample: PointSample, angle: float) -> PointSample:
        sample.apply_linear(rotation_matrix(self.axis, angle))
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(sample.rng))


@TRANSFORMS.register()
class ConstantFeatures(PointTransform):
    """Give every point the same input feature vector (``value`` repeated ``channels`` times).

    The sparse-convolution methods of the GraspNet-1Billion family feed three ones per
    point: geometry enters only through the voxel coordinates.
    """

    source = "GSNet (Wang et al., ICCV 2021)"

    def __init__(self, channels: int, value: float = 1.0) -> None:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.value = float(value)

    def __call__(self, sample: PointSample) -> PointSample:
        sample.feats = np.full((len(sample), self.channels), self.value, dtype=np.float32)
        return sample
