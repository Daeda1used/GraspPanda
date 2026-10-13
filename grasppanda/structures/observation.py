"""What a model may see, and what only training may see."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

import torch
from torch import Tensor

from .common import ContractError, Mask, Provenance, SampleKey
from .compact_labels import CompactGraspLabels
from .points import PointSet

__all__ = [
    "GRIDS",
    "Batch",
    "DeferredGrids",
    "ObjectGraspLabels",
    "Observation",
    "Supervision",
    "collate",
    "gather_grids",
]


def _move(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, (Tensor, PointSet)):
        return value.to(device)
    if isinstance(value, Mask):
        return Mask(value.data.to(device), value.provenance)
    if isinstance(value, (ObjectGraspLabels, CompactGraspLabels)):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


@dataclass(eq=False)
class Observation:
    """Everything available at inference time for a batch of frames.

    ``predict(observation)`` gets this object and nothing else. Any input derived from
    ground truth must say so, either as a :class:`Mask` with that provenance or as an
    entry of ``provenance``; it is then reported by :meth:`gt_assisted` and stamped on
    the results of the run.

    Attributes:
        keys: One :class:`SampleKey` per frame of the batch.
        intrinsics: ``(B, 3, 3)`` camera matrices.
        depth: ``(B, H, W)`` metres; zero where invalid.
        rgb: ``(B, 3, H, W)`` in ``[0, 1]``.
        points: Scene points the model consumes, camera frame, metres.
        point_feats: ``(N, C)`` input features aligned with ``points``.
        full_points: The cloud before sub-sampling, for post-processors such as a
            collision filter. Never an input of the network.
        masks: Named per-point masks aligned with ``points`` (e.g. ``instance``).
        provenance: Where inputs that shaped this observation came from, by name. A
            workspace crop computed from ground-truth segmentation is recorded here as
            ``{"workspace": Provenance.GROUND_TRUTH}``.
    """

    keys: list[SampleKey]
    intrinsics: Tensor | None = None
    depth: Tensor | None = None
    rgb: Tensor | None = None
    points: PointSet | None = None
    point_feats: Tensor | None = None
    full_points: PointSet | None = None
    masks: dict[str, Mask] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)

    def gt_assisted(self) -> list[str]:
        """Names of inputs that were derived from ground truth."""
        sources = {name: mask.provenance for name, mask in self.masks.items()} | self.provenance
        return sorted(n for n, p in sources.items() if p is Provenance.GROUND_TRUTH)

    def provides(self) -> set[str]:
        """Names of the inputs that are present."""
        present = {
            name
            for name in ("intrinsics", "depth", "rgb", "points", "point_feats", "full_points")
            if getattr(self, name) is not None
        }
        return present | set(self.masks)

    def to(self, device: torch.device | str) -> Observation:
        moved = {f.name: _move(getattr(self, f.name), device) for f in fields(self)}
        return Observation(**moved)


#: Names of the label grids of an object; ``collision`` and ``tolerance`` are optional.
GRIDS = ("scores", "widths", "collision", "tolerance")


class DeferredGrids(Protocol):
    """The label grids of one object instance, read on demand instead of held as tensors.

    A training sample would otherwise carry the full ``(P, V, A, D)`` grids of every
    object (tens of megabytes each), of which label matching reads the rows of the few
    label points that seeds were matched to. An implementation keeps the grids where
    they are stored and returns rows when asked. Row ``i`` is label point ``i`` of
    :attr:`ObjectGraspLabels.points`.

    Reading is defined for a group of readers at once, because that is how it is fast:
    the rows of every object of a batch are read into one buffer and moved to the device
    in one transfer. The values are those the dense tensors of
    :class:`ObjectGraspLabels` would hold; reading through this interface must not change
    a single number.
    """

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(P, V, A, D)`` of every grid."""

    @property
    def has_collision(self) -> bool:
        """Whether the grid ``collision`` can be read (the labels carry a separate mask)."""

    @property
    def has_tolerance(self) -> bool:
        """Whether the grid ``tolerance`` can be read."""

    @classmethod
    def gather(
        cls,
        readers: Sequence[DeferredGrids],
        name: str,
        owner: Tensor,
        rows: Tensor,
        views: Tensor | None = None,
    ) -> Tensor:
        """Read one grid for ``K`` requests across ``readers``, all instances of ``cls``.

        Request ``k`` reads label point ``rows[k]`` of ``readers[owner[k]]``. The result is
        ``(K, V, A, D)``, or ``(K, A, D)`` at view ``views[k]`` when ``views`` is given,
        on the device of ``rows``; ``name`` is one of :data:`GRIDS`. Requests may repeat
        and come in any order.
        """


@dataclass(eq=False)
class ObjectGraspLabels:
    """Dense grasp annotations of one object instance in a frame.

    The grids are stored in one of two ways. *Dense*: ``scores`` and ``widths`` (and,
    when present, ``collision`` and ``tolerance``) are tensors. *Deferred*: ``deferred``
    reads rows from the label store when they are asked for, and the four tensors are
    ``None``. Code that consumes labels reads them through :func:`gather_grids` (many
    objects in one pass) or the ``gather_*`` methods (one object), which return the same
    values for both storages; :meth:`dense` turns deferred labels into tensors.

    Attributes:
        object_id: Dataset id of the object model.
        pose: ``(4, 4)`` object-to-camera transform for this frame.
        points: ``(P, 3)`` annotated grasp points in the object frame.
        scores: ``(P, V, A, D)`` minimum friction coefficient for force closure. Lower
            is better and values ≤ 0 are invalid. These are labels, not confidences.
        widths: ``(P, V, A, D)`` gripper widths in metres.
        collision: ``(P, V, A, D)`` true where the grasp collides in this scene.
        tolerance: ``(P, V, A, D)`` grasp tolerance in metres: how far the grasp point
            may move over the object surface while the same grasp stays good. A derived
            label; ``None`` unless the dataset is configured to serve it.
        deferred: Reader of the grids of the ``P`` label points, in place of the four
            tensors above.
    """

    object_id: int
    pose: Tensor
    points: Tensor
    scores: Tensor | None = None
    widths: Tensor | None = None
    collision: Tensor | None = None
    tolerance: Tensor | None = None
    deferred: DeferredGrids | None = None

    def __post_init__(self) -> None:
        tensors = (self.scores, self.widths, self.collision, self.tolerance)
        dense = self.scores is not None and self.widths is not None
        if self.deferred is None and dense:
            return
        if self.deferred is None or any(tensor is not None for tensor in tensors):
            raise ContractError(
                f"object {self.object_id}: grasp labels hold either the tensors `scores` and "
                "`widths` (dense) or a `deferred` reader of them, not both and not neither"
            )

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Shape of every grid the labels hold, by name, whichever way it is stored."""
        if self.deferred is not None:
            names = ["scores", "widths"]
            names += ["collision"] if self.deferred.has_collision else []
            names += ["tolerance"] if self.deferred.has_tolerance else []
            return dict.fromkeys(names, tuple(self.deferred.shape))
        return {
            name: tuple(grid.shape) for name in GRIDS if (grid := getattr(self, name)) is not None
        }

    def gather_scores(self, rows: Tensor) -> Tensor:
        """``(K, V, A, D)`` friction scores of the label points ``rows``, on their device."""
        return _gathered(gather_grids([self], "scores", torch.zeros_like(rows), rows), self)

    def gather_widths(self, rows: Tensor, views: Tensor | None = None) -> Tensor:
        """Gripper widths of the label points ``rows``, on their device.

        ``(K, V, A, D)`` without ``views``; with ``views`` the ``(K, A, D)`` grid of label
        point ``rows[k]`` at view ``views[k]``.
        """
        owner = torch.zeros_like(rows)
        return _gathered(gather_grids([self], "widths", owner, rows, views), self)

    def gather_collision(self, rows: Tensor) -> Tensor | None:
        """``(K, V, A, D)`` collision mask of ``rows``; ``None`` when the labels carry none
        (because collisions were ignored, or already written into the scores)."""
        return gather_grids([self], "collision", torch.zeros_like(rows), rows)

    def gather_tolerance(self, rows: Tensor, views: Tensor | None = None) -> Tensor:
        """Grasp tolerance of the label points ``rows``, shaped as :meth:`gather_widths`."""
        owner = torch.zeros_like(rows)
        return _gathered(gather_grids([self], "tolerance", owner, rows, views), self)

    def dense(self) -> ObjectGraspLabels:
        """These labels with every grid as a tensor, on the device of ``points``.

        Reads all rows of deferred labels, which is what deferring them avoids; meant for
        inspection and for code written against the tensors.
        """
        if self.deferred is None:
            return self
        rows = torch.arange(len(self.points), device=self.points.device)
        return ObjectGraspLabels(
            self.object_id,
            self.pose,
            self.points,
            self.gather_scores(rows),
            self.gather_widths(rows),
            self.gather_collision(rows),
            self.gather_tolerance(rows) if self.deferred.has_tolerance else None,
        )

    def to(self, device: torch.device | str) -> ObjectGraspLabels:
        """Move the tensors; a deferred reader stays where its store is."""
        return ObjectGraspLabels(
            self.object_id,
            self.pose.to(device),
            self.points.to(device),
            None if self.scores is None else self.scores.to(device),
            None if self.widths is None else self.widths.to(device),
            None if self.collision is None else self.collision.to(device),
            None if self.tolerance is None else self.tolerance.to(device),
            self.deferred,
        )


def _gathered(grid: Tensor | None, labels: ObjectGraspLabels) -> Tensor:
    assert grid is not None, f"object {labels.object_id}"
    return grid


def gather_grids(
    objects: Sequence[ObjectGraspLabels],
    name: str,
    owner: Tensor,
    rows: Tensor,
    views: Tensor | None = None,
) -> Tensor | None:
    """Read one label grid for many label points across the objects of a batch.

    Request ``k`` reads label point ``rows[k]`` of ``objects[owner[k]]``. This is how
    label matching reads labels: it needs the grids of the label points its seeds were
    matched to and of no others, and it must not care how the grids are stored. Dense
    tensors are indexed; deferred labels are read from their store, all objects in one
    pass. The values are the same either way.

    Args:
        objects: Labels of object instances, e.g. those of every frame of a batch.
        name: ``scores``, ``widths``, ``collision`` or ``tolerance``.
        owner: ``(K,)`` position in ``objects`` of the object each request reads.
        rows: ``(K,)`` label point of that object.
        views: ``(K,)`` view to read for each request; without it every view is read.

    Returns:
        ``(K, V, A, D)``, or ``(K, A, D)`` with ``views``, on the device of ``rows``. For
        ``collision``: ``None`` when no object carries a mask, and no collision for the
        requests of an object without one.
    """
    if name not in GRIDS:
        raise ValueError(f"unknown label grid '{name}'; there are {', '.join(GRIDS)}")
    if not objects:
        raise ContractError("gather_grids() needs at least one object to read labels from")
    present = [name in obj.shapes() for obj in objects]
    if not any(present) and name == "collision":
        return None
    if not all(present) and name != "collision":
        missing = [obj.object_id for obj, has in zip(objects, present) if not has]
        raise ContractError(f"objects {missing} carry no {name} labels")
    kinds = {type(obj.deferred) for obj in objects}
    if all(present) and len(kinds) == 1 and objects[0].deferred is not None:
        reader = type(objects[0].deferred)  # the usual case: every object is deferred
        return reader.gather([obj.deferred for obj in objects], name, owner, rows, views)

    # Mixed or dense storage: group the requests by object once, then fill the result.
    order = torch.argsort(owner, stable=True)
    counts = torch.bincount(owner, minlength=len(objects)).tolist()
    if len(counts) != len(objects):
        raise ContractError(
            f"gather_grids(): a request names object {len(counts) - 1}, but there are "
            f"{len(objects)} objects"
        )
    parts: list[tuple[Tensor, Tensor]] = []  # (requests, their values)
    deferred: dict[type, list[tuple[DeferredGrids, Tensor]]] = {}
    start = 0
    for obj, count, has in zip(objects, counts, present):
        member = order[start : start + count]
        start += count
        if count == 0 or not has:
            continue
        if obj.deferred is not None:
            deferred.setdefault(type(obj.deferred), []).append((obj.deferred, member))
            continue
        grid = getattr(obj, name)
        index = rows[member].to(grid.device)
        values = grid[index] if views is None else grid[index, views[member].to(grid.device)]
        parts.append((member, values.to(rows.device)))
    for reader, requests in deferred.items():
        member = torch.cat([member for _, member in requests])
        which = torch.cat(
            [torch.full_like(member, position) for position, (_, member) in enumerate(requests)]
        )
        chosen = None if views is None else views[member]
        readers = [grids for grids, _ in requests]
        parts.append((member, reader.gather(readers, name, which, rows[member], chosen)))
    if len(objects) == 1 and parts:
        return parts[0][1]  # one object: the stable sort left its requests in order
    shape = next(obj.shapes()[name] for obj, has in zip(objects, present) if has)
    kind = parts[0][1].dtype if parts else (torch.bool if name == "collision" else torch.float32)
    cell = shape[1:] if views is None else shape[2:]
    result = torch.zeros((len(rows), *cell), dtype=kind, device=rows.device)
    for member, values in parts:
        result[member] = values
    return result


@dataclass(eq=False)
class Supervision:
    """Training-only information. Never passed to ``predict``.

    Attributes:
        objects: Per frame, the grasp labels of every object instance in it.
        point_labels: Named ``(N, ...)`` labels aligned with ``observation.points``.
        extras: Method-specific targets, one list entry per frame. Compact grasp labels
            travel here under ``COMPACT_LABELS``.
    """

    objects: list[list[ObjectGraspLabels]] = field(default_factory=list)
    point_labels: dict[str, Tensor] = field(default_factory=dict)
    extras: dict[str, list[Any]] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> Supervision:
        return Supervision(
            _move_objects(self.objects, device),
            _move(self.point_labels, device),
            _move(self.extras, device),
        )


def _move_objects(
    frames: list[list[ObjectGraspLabels]], device: torch.device | str
) -> list[list[ObjectGraspLabels]]:
    """Move the object labels of a batch; the poses and label points in two transfers.

    A batch has some forty objects, each with a pose and a few hundred label points.
    Moved one tensor at a time that is eighty small transfers, each of which waits for
    the device; stacked first, it is two. The grids, if the labels hold any as
    tensors, are moved object by object as before.
    """
    flat = [obj for objects in frames for obj in objects]
    target = torch.device(device)
    small = [tensor for obj in flat for tensor in (obj.pose, obj.points)]
    stackable = (
        target.type != "cpu"
        and len({(t.device.type, t.dtype) for t in small}) == 1
        and small[0].device.type == "cpu"
        and len({obj.pose.shape for obj in flat}) == 1
        and len({obj.points.shape[1:] for obj in flat}) == 1
    )
    if not stackable:  # nothing to gain, or tensors that do not stack: one by one
        return [[obj.to(device) for obj in objects] for objects in frames]
    poses = torch.stack([obj.pose for obj in flat]).to(target)
    points = torch.cat([obj.points for obj in flat]).to(target).split([len(o.points) for o in flat])
    moved = iter(
        ObjectGraspLabels(
            obj.object_id,
            pose,
            own,
            None if obj.scores is None else obj.scores.to(target),
            None if obj.widths is None else obj.widths.to(target),
            None if obj.collision is None else obj.collision.to(target),
            None if obj.tolerance is None else obj.tolerance.to(target),
            obj.deferred,
        )
        for obj, pose, own in zip(flat, poses, points)
    )
    return [[next(moved) for _ in objects] for objects in frames]


@dataclass(eq=False)
class Batch:
    """An observation and, during training, its supervision.

    A dataset returns a batch of one frame; :func:`collate` merges them.
    """

    observation: Observation
    supervision: Supervision | None = None

    def __len__(self) -> int:
        return len(self.observation)

    def to(self, device: torch.device | str) -> Batch:
        supervision = None if self.supervision is None else self.supervision.to(device)
        return Batch(self.observation.to(device), supervision)


def _cat_optional(values: Sequence[Tensor | None], name: str) -> Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ContractError(f"cannot collate: '{name}' is present in some frames only")
    if len({tuple(value.shape[1:]) for value in present}) != 1:
        shapes = sorted({tuple(value.shape[1:]) for value in present})
        raise ContractError(f"cannot collate '{name}' of different shapes {shapes}")
    return torch.cat(present)


def _cat_points(values: Sequence[PointSet | None], name: str) -> PointSet | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ContractError(f"cannot collate: '{name}' is present in some frames only")
    return PointSet.cat(present)


def collate(items: Sequence[Batch]) -> Batch:
    """Merge batches into one. Point clouds may differ in size; images may not."""
    if not items:
        raise ValueError("collate() needs at least one item")
    observations = [item.observation for item in items]
    first = observations[0]
    for other in observations[1:]:
        if other.provenance != first.provenance or set(other.masks) != set(first.masks):
            raise ContractError("cannot collate frames whose inputs have different provenance")
    masks = {}
    for name, mask in first.masks.items():
        if any(o.masks[name].provenance is not mask.provenance for o in observations):
            raise ContractError(f"cannot collate mask '{name}' of different provenance")
        masks[name] = Mask(torch.cat([o.masks[name].data for o in observations]), mask.provenance)
    observation = Observation(
        keys=[key for o in observations for key in o.keys],
        intrinsics=_cat_optional([o.intrinsics for o in observations], "intrinsics"),
        depth=_cat_optional([o.depth for o in observations], "depth"),
        rgb=_cat_optional([o.rgb for o in observations], "rgb"),
        points=_cat_points([o.points for o in observations], "points"),
        point_feats=_cat_optional([o.point_feats for o in observations], "point_feats"),
        full_points=_cat_points([o.full_points for o in observations], "full_points"),
        masks=masks,
        provenance=dict(first.provenance),
    )
    supervisions = [item.supervision for item in items]
    if all(s is None for s in supervisions):
        return Batch(observation)
    if any(s is None for s in supervisions):
        raise ContractError("cannot collate frames with and without supervision")
    names = set(supervisions[0].point_labels)
    extras = set(supervisions[0].extras)
    if any(set(s.point_labels) != names or set(s.extras) != extras for s in supervisions):
        raise ContractError("cannot collate frames with different label sets")
    supervision = Supervision(
        objects=[objects for s in supervisions for objects in s.objects],
        point_labels={n: torch.cat([s.point_labels[n] for s in supervisions]) for n in names},
        extras={n: [entry for s in supervisions for entry in s.extras[n]] for n in extras},
    )
    return Batch(observation, supervision)
