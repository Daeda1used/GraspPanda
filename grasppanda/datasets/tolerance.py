"""Grasp tolerance labels: generation and storage.

Tolerance says how far a grasp may be moved over the object surface and still be good.
It is a label *derived* from the dense grasp labels of an object model, independent of
any scene, and it is supervised by the operation/tolerance head bundle of the
GraspNet-baseline family.

Definition (the GraspNet-baseline generator). A grasp is *good* when its friction score
lies in ``(0, max_friction]``. For an annotated point ``p`` and a grasp cell
``(view, angle, depth)``, walk through the radii ``0, step, 2·step, …, max_radius``:
at radius ``r`` the cell *passes* when at least ``min_good_ratio`` of the annotated
points of the object within distance ``r`` of ``p`` (``p`` included) hold a good grasp
in that same cell. The tolerance of the cell is the last radius at which it passed.
The walk ends at the first radius where no cell of ``p`` passes, so a cell can also
pass, fail and pass again before that; and a cell whose own grasp is bad still gets a
positive tolerance when enough of its neighbours are good. Both are reference
behaviour and are kept.

The reference script computes this with one process per annotated point and stores
``float32`` arrays next to its sources. Here one pass per point over its neighbours,
sorted by distance, accumulates the counts shell by shell; distances and comparisons
are evaluated in the same precision as the reference (``float32`` distances against
``float32`` radii, ``float64`` ratios), so results are equal value for value. Levels are
stored as ``uint8`` indices into the radius list, a quarter of the size, in the asset
store, keyed by a hash of the parameters and of the grasp-label store they were
computed from.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assets import Asset, AssetError, AssetSpec, save_array
from .labels import GRASPNET_LABEL_GRID, LabelGrid, open_array

__all__ = ["GraspToleranceStore", "ToleranceConfig", "object_tolerance"]


@dataclass(frozen=True)
class ToleranceConfig:
    """Parameters of the tolerance labels a dataset serves.

    The values have no defaults because the generator, not a reader, owns them; the
    GraspNet-baseline reference is ``max_friction: 0.55``, ``min_good_ratio: 0.8``,
    ``radius_step: 0.001``, ``max_radius: 0.05``.

    Attributes:
        max_friction: A grasp is good when its friction score is in ``(0, max_friction]``.
        min_good_ratio: Fraction of the annotated points inside a radius that must hold
            a good grasp for the radius to pass.
        radius_step: Spacing of the radii, metres.
        max_radius: Largest radius, metres; also the largest tolerance value.
    """

    max_friction: float
    min_good_ratio: float
    radius_step: float
    max_radius: float

    def problems(self) -> list[str]:
        found = []
        if self.max_friction <= 0:
            found.append("`tolerance.max_friction` must be positive.")
        if not 0 < self.min_good_ratio <= 1:
            found.append("`tolerance.min_good_ratio` must be in (0, 1].")
        if not 0 < self.radius_step <= self.max_radius:
            found.append("`tolerance.radius_step` must be positive and at most `max_radius`.")
        elif self.steps > 255:
            found.append(
                f"`tolerance` asks for {self.steps + 1} radii; at most 256 fit the uint8 storage."
            )
        return found

    @property
    def steps(self) -> int:
        """Index of the largest radius."""
        return round(self.max_radius / self.radius_step)

    @property
    def radii(self) -> np.ndarray:
        """``(steps + 1,)`` float32 radii; level ``k`` of a stored label means ``radii[k]``.

        Each radius is the double-precision product ``radius_step * k`` rounded to
        single precision, as the reference compares and stores it.
        """
        return np.array([self.radius_step * k for k in range(self.steps + 1)], dtype=np.float32)


def object_tolerance(
    points: np.ndarray,
    scores: np.ndarray,
    config: ToleranceConfig,
    rows: np.ndarray | None = None,
    progress: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Tolerance levels of the grasps of one object model.

    Args:
        points: ``(P, 3)`` float32 annotated grasp points, object frame.
        scores: ``(P, V, A, D)`` friction scores of the object.
        config: Generator parameters.
        rows: Compute these annotated points only (neighbours are always all points).
        progress: Called with the number of points finished so far.

    Returns:
        ``(len(rows), V, A, D)`` uint8 indices into ``config.radii`` (``(P, …)`` without
        ``rows``).
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    count, grid = len(points), scores.shape[1:]
    if points.shape != (count, 3) or len(scores) != count:
        raise ValueError(
            f"expected points (P, 3) and scores (P, ...), got {points.shape} and {scores.shape}"
        )
    radii = config.radii
    cells = int(np.prod(grid))
    good = np.empty((count, cells), dtype=np.bool_)
    limit = np.float32(config.max_friction)
    for start in range(0, count, 256):  # in blocks: `scores` may be a memory map
        block = np.asarray(scores[start : start + 256], dtype=np.float32).reshape(-1, cells)
        good[start : start + 256] = (block > 0) & (block <= limit)
    rows = np.arange(count) if rows is None else np.asarray(rows, dtype=np.int64)
    levels = np.zeros((len(rows), cells), dtype=np.uint8)
    for position, row in enumerate(rows.tolist()):
        distance = np.linalg.norm(points[row] - points, axis=-1)  # float32, as the reference
        order = np.argsort(distance, kind="stable")
        ends = np.searchsorted(distance[order], radii, side="right")  # points within each radius
        inside = np.zeros(cells, dtype=np.int64)
        start = 0
        for level, end in enumerate(ends.tolist()):
            if end > start:
                inside += good[order[start:end]].sum(axis=0, dtype=np.int64)
                start = end
            passing = inside / end >= config.min_good_ratio  # float64 ratio, as the reference
            if not passing.any():
                break
            levels[position, passing] = level
        if progress is not None:
            progress(position + 1)
    return levels.reshape(len(rows), *grid)


def _object_shard(object_id: int) -> str:
    return f"object_{object_id:06d}"


class GraspToleranceStore:
    """Memory-mapped tolerance labels of the object models of one dataset.

    Args:
        dataset: Dataset name, e.g. ``graspnet1b``.
        config: Generator parameters; part of the asset hash.
        label_asset: Hash of the grasp-label store the tolerance is computed from.
        grid: Label grid the dataset declares.
        root: Asset root; defaults to :func:`grasppanda.utils.paths.cache_root`.
        max_open: Bound on the memory maps a process keeps open.
    """

    KIND = "grasp_tolerance"
    GENERATOR = "grasppanda.grasp_tolerance"
    VERSION = 1

    def __init__(
        self,
        dataset: str,
        config: ToleranceConfig,
        label_asset: str,
        grid: LabelGrid = GRASPNET_LABEL_GRID,
        root: str | os.PathLike[str] | None = None,
        max_open: int = 128,
    ) -> None:
        self.config = config
        self.grid = grid
        self.radii = config.radii
        spec = AssetSpec(
            dataset=dataset,
            kind=self.KIND,
            generator=self.GENERATOR,
            version=self.VERSION,
            parameters={**asdict(config), "grid": list(grid.shape), "levels": "uint8"},
            inputs={"dataset": dataset, "grasp_labels": label_asset},
            about={
                "levels": "(P, V, A, D) uint8 per object: index k of the radius; the tolerance "
                "of a grasp is radius_step * k metres, rounded to float32",
            },
        )
        self.asset = Asset(spec, root)
        self._max_open = max_open
        self._objects: OrderedDict[int, np.ndarray] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        # Memory maps are reopened lazily in every process that receives the store.
        state = self.__dict__.copy()
        state["_objects"] = OrderedDict()
        return state

    def _file(self, object_id: int) -> Path:
        return self.asset.path / "objects" / f"{object_id:06d}.levels.npy"

    # ------------------------------------------------------------------------- build

    def write(self, object_id: int, points: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
        """Generate and store the labels of one object; returns its shard record.

        Writes the shard file only. The caller records completion with
        :meth:`mark_complete`, so that several processes can generate objects while one
        process alone updates the manifest.
        """
        levels = object_tolerance(points, scores, self.config)
        save_array(self._file(object_id), levels)
        stored = np.load(self._file(object_id), mmap_mode="r")
        if stored.shape != levels.shape or not np.array_equal(stored, levels):
            raise AssetError(f"verification failed: {self._file(object_id)} differs from memory")
        return {
            "points": len(levels),
            "positive": float(np.count_nonzero(levels) / max(levels.size, 1)),
            "max_level": int(levels.max()) if levels.size else 0,
        }

    def mark_complete(self, object_id: int, record: dict[str, Any]) -> None:
        self.asset.mark_complete(_object_shard(object_id), **record)

    # -------------------------------------------------------------------------- read

    def require(self, objects: Iterable[int], *, build: str) -> None:
        """Raise :class:`AssetError` unless the labels of these objects exist."""
        self.asset.require([_object_shard(i) for i in sorted(set(objects))], build=build)

    def missing(self, objects: Iterable[int]) -> list[int]:
        """Objects whose tolerance has not been generated."""
        wanted = sorted(set(objects))
        if not self.asset.exists():
            return wanted
        return [i for i in wanted if not self.asset.is_complete(_object_shard(i))]

    def levels(self, object_id: int) -> np.ndarray:
        """``(P, V, A, D)`` uint8 memory map of an object's tolerance levels."""
        if object_id in self._objects:
            self._objects.move_to_end(object_id)
            return self._objects[object_id]
        if not self.asset.is_complete(_object_shard(object_id)):
            raise AssetError(
                f"tolerance labels of object {object_id} are not in the store at {self.asset.path}"
            )
        levels = open_array(self._file(object_id))
        if levels.shape[1:] != self.grid.shape or levels.dtype != np.uint8:
            raise AssetError(
                f"{self._file(object_id)} holds {levels.dtype} {levels.shape}; expected uint8 "
                f"(P, {', '.join(map(str, self.grid.shape))})"
            )
        self._objects[object_id] = levels
        while len(self._objects) > self._max_open:
            self._objects.popitem(last=False)
        return levels

    def tolerance(self, object_id: int, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        """``(K, V, A, D)`` float32 tolerance in metres of the given label points."""
        return self.radii[np.asarray(self.levels(object_id)[rows])]
