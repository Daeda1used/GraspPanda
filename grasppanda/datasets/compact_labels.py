"""Compact grasp labels: the dense grids collapsed to one grasp per point and view.

The dense label set gives every annotated point of an object 300 views × 12 in-plane
angles × 4 depths. EconomicGrasp trains on a much smaller set that is computed once per
scene (its "economic supervision"). Definition, following the reference generator:

1. A grasp is *usable* in a scene when its friction score is positive and it does not
   collide there. Annotated points without any usable grasp of friction
   ``<= point_friction_threshold`` are dropped.
2. The view graspness of a kept point is, per view, the fraction of its angle × depth
   cells that hold a usable grasp of friction ``<= view_friction_threshold``, min-max
   normalised over the views of the point (with ``1e-5`` in the denominator). This is
   the target of the view stage, computed here because step 3 destroys its input.
3. Per point and view, only the best cell survives: the usable grasp no wider than
   ``max_width`` with the highest quality ``1.1 - friction``. Ties go to the first depth
   and then to the first angle. A view without such a grasp keeps cell (0, 0) with
   quality 0.
4. The survivors are stored as small integers: angle bin, depth bin, quality in tenths
   and width in truncated millimetres.

Differences from the reference files, none of which changes a value:

* Views are stored in lattice order. The reference sorts the views of every point by
  view graspness (keeping all 300) and stores the permutation; the order of equal
  values in that sort is not defined, so its files are not reproducible bit for bit,
  while their content is (:func:`from_reference_arrays` undoes the permutation).
* The asset lives in the asset store, keyed by its parameters and by the hash of the
  dense label store it was computed from; one asset serves every camera of a dataset.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from grasppanda.structures import CompactGraspLabels

from .assets import Asset, AssetError, AssetSpec, atomic_path
from .labels import LabelGrid, StoredGraspLabels

__all__ = [
    "CompactLabelConfig",
    "CompactLabelStore",
    "collapse_labels",
    "from_reference_arrays",
    "generate_compact_labels",
    "scene_shard",
]

#: Arrays of a scene shard that hold one value per (point, view).
VIEW_ARRAYS = ("view_graspness", "angle", "depth", "score", "width")
#: Guard of the view-graspness normalisation, as in the reference generator.
_VIEW_EPS = np.float32(1e-5)
#: Quality of a grasp is this constant minus its friction coefficient.
_QUALITY_OFFSET = np.float32(1.1)
#: Annotated points processed at a time; bounds the dense grids held in memory.
_ROWS = 256


@dataclass(frozen=True)
class CompactLabelConfig:
    """Which compact grasp labels a dataset serves.

    None of the values has a default: the generator, not this reader, owns them, and a
    dataset config states which asset it reads.

    Attributes:
        point_friction_threshold: An annotated point is kept when it has a usable grasp
            with friction in ``(0, point_friction_threshold]``. The reference uses 0.4.
        view_friction_threshold: A usable grasp counts towards the view graspness when
            its friction is in ``(0, view_friction_threshold]``. The reference uses
            0.6; the view stage of the model must be configured with the same value.
        max_width: Grasps wider than this (metres) cannot become the kept grasp of a
            view. The reference uses 0.1, the opening of the benchmark gripper.
    """

    point_friction_threshold: float
    view_friction_threshold: float
    max_width: float

    def problems(self) -> list[str]:
        found = []
        for name in ("point_friction_threshold", "view_friction_threshold", "max_width"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not value > 0:
                found.append(f"`compact_labels.{name}` must be a positive number, got {value!r}.")
        return found


class SceneLabels(Protocol):
    """What the generator needs from a dataset."""

    name: str

    def scene_labels(self, scene: int) -> list[StoredGraspLabels]:
        """Dense grasp labels of every instance of a scene, in the order of its object list."""

    def scene_object_ids(self, scene: int) -> list[int]:
        """Object id of every instance of a scene, in the same order."""


def scene_shard(scene: int) -> str:
    return f"scene_{scene:06d}"


# -------------------------------------------------------------------------- definition


def collapse_labels(labels: StoredGraspLabels, config: CompactLabelConfig) -> dict[str, np.ndarray]:
    """Compact labels of one object instance in a scene.

    All arithmetic is single precision in the order of the reference generator, so the
    result equals its output value for value.

    Returns:
        ``rows`` ``(K,)`` indices of the kept annotated points, ``points`` ``(K, 3)``,
        ``view_graspness`` ``(K, V)`` float32 and ``angle``, ``depth``, ``score``,
        ``width`` ``(K, V)`` uint8.
    """
    grid = labels.grid
    cells = np.float32(grid.num_angle * grid.num_depth)
    point_threshold = np.float32(config.point_friction_threshold)  # scores are float32;
    view_threshold = np.float32(config.view_friction_threshold)  # compare in kind
    max_width = np.float32(config.max_width)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("rows", *VIEW_ARRAYS)}
    for start in range(0, labels.num_points, _ROWS):
        rows = slice(start, start + _ROWS)
        friction = np.array(labels.scores[rows], dtype=np.float32)
        widths = np.asarray(labels.widths[rows], dtype=np.float32)
        friction[labels.collision(rows)] = 0
        friction[friction < 0] = 0
        usable = friction > 0
        kept = (usable & (friction <= point_threshold)).reshape(len(friction), -1).any(axis=1)
        if not kept.any():
            continue
        friction, widths, usable = friction[kept], widths[kept], usable[kept]

        counts = (usable & (friction <= view_threshold)).sum(axis=(2, 3))
        graspness = counts.astype(np.float32) / cells
        low = graspness.min(axis=1, keepdims=True)
        high = graspness.max(axis=1, keepdims=True)
        graspness = (graspness - low) / (high - low + _VIEW_EPS)

        quality = np.where(usable & (widths <= max_width), _QUALITY_OFFSET - friction, 0)
        quality = quality.astype(np.float32, copy=False)
        depth = quality.argmax(axis=3)  # first of equal cells
        quality = np.take_along_axis(quality, depth[..., None], axis=3)[..., 0]
        widths = np.take_along_axis(widths, depth[..., None], axis=3)[..., 0]
        angle = quality.argmax(axis=2)
        quality = np.take_along_axis(quality, angle[..., None], axis=2)[..., 0]
        widths = np.take_along_axis(widths, angle[..., None], axis=2)[..., 0]
        depth = np.take_along_axis(depth, angle[..., None], axis=2)[..., 0]

        units = widths * np.float32(CompactGraspLabels.WIDTH_UNITS)
        if units.size and units.max() >= 256:
            raise AssetError(
                f"a kept grasp is {float(widths.max()):.3f} m wide; widths are stored in whole "
                f"1/{CompactGraspLabels.WIDTH_UNITS} m as uint8 and cannot exceed 0.255 m"
            )
        parts["rows"].append(np.flatnonzero(kept) + start)
        parts["view_graspness"].append(graspness)
        parts["angle"].append(angle.astype(np.uint8))
        parts["depth"].append(depth.astype(np.uint8))
        parts["score"].append(
            (quality * np.float32(CompactGraspLabels.SCORE_LEVELS)).astype(np.uint8)
        )
        parts["width"].append(units.astype(np.uint8))
    shape = (0, grid.num_view)
    empty = {name: np.zeros(shape, dtype=np.uint8) for name in VIEW_ARRAYS}
    empty["view_graspness"] = np.zeros(shape, dtype=np.float32)
    out = {
        name: np.concatenate(parts[name]) if parts[name] else empty[name] for name in VIEW_ARRAYS
    }
    out["rows"] = np.concatenate(parts["rows"]) if parts["rows"] else np.zeros(0, dtype=np.int64)
    out["points"] = np.asarray(labels.points, dtype=np.float32)[out["rows"]]
    return out


def from_reference_arrays(archive: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Arrays of a reference label file (``<scene>_labels.npz``) in this module's layout.

    The reference stores the per-view values of a point in the order of its
    ``topview`` permutation; here they are put back in lattice order. Its view
    graspness is already in lattice order.

    Returns:
        ``points``, ``instance`` and the arrays of :data:`VIEW_ARRAYS`.
    """
    order = np.asarray(archive["topview"]).astype(np.int64)
    out = {
        "points": np.asarray(archive["points"], dtype=np.float32),
        "instance": np.asarray(archive["pointid"]).astype(np.int16),
        "view_graspness": np.asarray(archive["vgraspness"], dtype=np.float32),
    }
    if not (np.sort(order, axis=1) == np.arange(order.shape[1])).all():
        raise AssetError(
            "the reference file keeps a subset of the views of a point; only files with "
            "every view are supported"
        )
    for name, source in (
        ("angle", "rotations"),
        ("depth", "depth"),
        ("score", "scores"),
        ("width", "widths"),
    ):
        values = np.asarray(archive[source])
        restored = np.empty_like(values)
        np.put_along_axis(restored, order, values, axis=1)
        out[name] = restored
    return out


# ----------------------------------------------------------------------------- storage


class CompactLabelStore:
    """Compact labels in the asset store: one archive per scene.

    A scene archive holds the kept points of all its object instances, concatenated in
    the order of the scene's object list: ``points`` ``(N, 3)`` float32 (object frame),
    ``instance`` ``(N,)`` position of the owning instance in that list, and the arrays
    of :data:`VIEW_ARRAYS`, each ``(N, V)``.

    Args:
        asset: The asset directory.
        config: The parameters the asset was (or is to be) generated with.
        max_open_scenes: Scenes a process keeps in memory, least recently used first out.
    """

    KIND = "compact_grasp_labels"
    GENERATOR = "grasppanda.compact_grasp_labels"
    VERSION = 1

    def __init__(self, asset: Asset, config: CompactLabelConfig, max_open_scenes: int = 4) -> None:
        self.asset = asset
        self.config = config
        self._max_open = max_open_scenes
        self._scenes: OrderedDict[int, list[dict[str, torch.Tensor]]] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        # Every process that receives the store reads the scenes it needs itself.
        state = self.__dict__.copy()
        state["_scenes"] = OrderedDict()
        return state

    @classmethod
    def spec(
        cls, dataset: str, config: CompactLabelConfig, grid: LabelGrid, label_hash: str
    ) -> AssetSpec:
        """Identity of the asset a dataset configuration asks for."""
        return AssetSpec(
            dataset=dataset,
            kind=cls.KIND,
            generator=cls.GENERATOR,
            version=cls.VERSION,
            parameters={
                "point_friction_threshold": float(config.point_friction_threshold),
                "view_friction_threshold": float(config.view_friction_threshold),
                "max_width": float(config.max_width),
                "grid": list(grid.shape),
                "usable": "friction > 0 and not colliding in the scene",
                "kept_cell": "highest quality; first depth, then first angle among equals",
                "quality": f"(1.1 - friction) * {CompactGraspLabels.SCORE_LEVELS}, truncated, uint8",
                "width": f"metres * {CompactGraspLabels.WIDTH_UNITS}, truncated, uint8",
                "view_graspness": "float32, min-max over the views of a point, eps 1e-5",
                "view_order": "lattice",
            },
            inputs={"dataset": dataset, "grasp_labels": label_hash},
            about={
                "layout": "scene_<id>.npz, kept points of all instances concatenated",
                "points": "(N, 3) float32, object frame, metres",
                "instance": "(N,) int16, position of the instance in the scene's object list",
                "view_graspness": "(N, V) float32 target view scores in [0, 1)",
                "angle": "(N, V) uint8 in-plane angle bin of the kept grasp",
                "depth": "(N, V) uint8 depth bin of the kept grasp",
                "score": "(N, V) uint8 quality level; 0 = the view has no usable grasp",
                "width": "(N, V) uint8 width of the kept grasp, millimetres",
            },
        )

    def shard_file(self, scene: int) -> Path:
        return self.asset.path / f"{scene_shard(scene)}.npz"

    # ------------------------------------------------------------------------- write

    def write_scene(self, scene: int, arrays: Mapping[str, np.ndarray]) -> None:
        """Write a scene's archive atomically and read it back to verify it.

        The shard is not marked complete: several processes may write scenes at once,
        and only the one that owns the manifest calls :meth:`mark_complete`.
        """
        names = ("points", "instance", *VIEW_ARRAYS)
        if set(arrays) != set(names):
            raise AssetError(f"scene {scene}: expected arrays {names}, got {sorted(arrays)}")
        count = len(arrays["points"])
        for name in names:
            if len(arrays[name]) != count:
                raise AssetError(
                    f"scene {scene}: '{name}' has {len(arrays[name])} rows for {count} points"
                )
        file = self.shard_file(scene)
        with atomic_path(file) as temporary, open(temporary, "wb") as handle:
            np.savez(handle, **{name: arrays[name] for name in names})
        with np.load(file) as stored:
            for name in names:
                if stored[name].dtype != arrays[name].dtype or not np.array_equal(
                    stored[name], arrays[name]
                ):
                    raise AssetError(f"verification failed: '{name}' of {file} differs")

    def mark_complete(self, scene: int, record: dict[str, Any]) -> None:
        """Record a written scene, with what its generation reported, in the manifest."""
        file = self.shard_file(scene)
        if not file.is_file():
            raise AssetError(f"cannot mark scene {scene} complete: {file} is missing")
        self.asset.mark_complete(scene_shard(scene), **record, bytes=file.stat().st_size)

    # -------------------------------------------------------------------------- read

    def read_scene(self, scene: int) -> dict[str, np.ndarray]:
        """The stored arrays of a scene."""
        record = self.asset.shards().get(scene_shard(scene))
        if record is None:
            raise AssetError(
                f"compact grasp labels of scene {scene} are not in the asset at "
                f"{self.asset.path}: the scene has not been generated"
            )
        with np.load(self.shard_file(scene)) as archive:
            return {name: archive[name] for name in ("points", "instance", *VIEW_ARRAYS)}

    def scene_object_ids(self, scene: int) -> Sequence[int]:
        """Object ids of a scene's instances as recorded when its shard was built."""
        record = self.asset.shards().get(scene_shard(scene))
        if record is None:
            raise AssetError(
                f"compact grasp labels of scene {scene} are not in the asset at {self.asset.path}"
            )
        return record["objects"]

    def _instances(self, scene: int) -> list[dict[str, torch.Tensor]]:
        if scene in self._scenes:
            self._scenes.move_to_end(scene)
            return self._scenes[scene]
        arrays = self.read_scene(scene)
        instances = []
        for index in range(len(self.scene_object_ids(scene))):
            rows = arrays["instance"] == index
            instances.append(
                {
                    name: torch.from_numpy(np.ascontiguousarray(arrays[name][rows]))
                    for name in ("points", *VIEW_ARRAYS)
                }
            )
        self._scenes[scene] = instances
        while len(self._scenes) > self._max_open:
            self._scenes.popitem(last=False)
        return instances

    def labels(
        self, scene: int, index: int, object_id: int, pose: np.ndarray
    ) -> CompactGraspLabels:
        """Labels of instance ``index`` of ``scene`` (a copy of ``object_id``) at ``pose``."""
        recorded = self.scene_object_ids(scene)
        if index >= len(recorded) or recorded[index] != object_id:
            raise AssetError(
                f"scene {scene}: instance {index} is object {object_id} in the frame, but the "
                f"compact labels were generated for objects {list(recorded)}"
            )
        return CompactGraspLabels(
            object_id=object_id,
            pose=torch.from_numpy(np.asarray(pose, dtype=np.float32)),
            view_max_friction=float(self.config.view_friction_threshold),
            **self._instances(scene)[index],
        )

    def record(self) -> dict[str, Any]:
        return self.asset.record()


# -------------------------------------------------------------------------- generation


def generate_compact_labels(
    dataset: SceneLabels,
    store: CompactLabelStore,
    scene: int,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Generate and write the compact labels of one scene.

    Returns:
        The shard record (object ids, annotated and kept points per instance) for
        :meth:`CompactLabelStore.mark_complete`, which the caller invokes.
    """
    object_ids = [int(i) for i in dataset.scene_object_ids(scene)]
    instances = dataset.scene_labels(scene)
    if len(instances) != len(object_ids):
        raise AssetError(
            f"scene {scene}: {len(instances)} labelled instances for {len(object_ids)} objects"
        )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("points", "instance", *VIEW_ARRAYS)}
    kept = []
    for index, labels in enumerate(instances):
        compact = collapse_labels(labels, store.config)
        kept.append(len(compact["rows"]))
        parts["instance"].append(np.full(kept[-1], index, dtype=np.int16))
        for name in ("points", *VIEW_ARRAYS):
            parts[name].append(compact[name])
        if progress is not None:
            progress(index)
    store.write_scene(scene, {name: np.concatenate(values) for name, values in parts.items()})
    return {
        "objects": object_ids,
        "annotated": [int(labels.num_points) for labels in instances],
        "kept": kept,
    }
