"""Dense grasp labels in the GraspNet format, served from memory-mapped arrays.

The format, shared by GraspNet-1Billion and GraspClutter6D: per object model, ``P``
annotated grasp points, each with a ``views x angles x depths`` grid of friction scores
and gripper widths; per scene, one boolean collision grid per object instance.

Reference pipelines load every object's arrays into the RAM of the training process
(and again into each worker that touches them). Here they are converted once into
uncompressed ``.npy`` files inside the asset store and opened with ``mmap``: all
dataloader workers share the same pages of the page cache, and a process only pays for
the rows it reads. Collision grids are bit-packed (eight grasps per byte).

Conversion verifies every array value for value against its source before the shard is
marked complete.

A training sample does not have to carry the grids at all. :meth:`StoredGraspLabels.defer`
returns a :class:`DeferredGraspLabels`: the names of the files, the chosen label points
and how collisions are treated, a few kilobytes that travel through the dataloader in
place of tens of megabytes per object. The process that matches labels to seeds then
reads the rows it needs, and nothing else, from the same memory maps
(``docs/advanced_guides/data_loading.md``).
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import Tensor

from .assets import Asset, AssetError, AssetSpec, save_array

__all__ = [
    "GRASPNET_LABEL_GRID",
    "DeferredGraspLabels",
    "GraspLabelStore",
    "GridFiles",
    "LabelFiles",
    "LabelGrid",
    "StoredGraspLabels",
    "open_array",
]

CollisionMode = Literal["zero_scores", "attach", "ignore"]


@dataclass(frozen=True)
class LabelGrid:
    """The grid on which dense grasp labels are defined.

    One owner for numbers that reference code repeats in the label generator, the label
    matching and the decoder: a label array has shape ``(P, num_view, num_angle,
    num_depth)``; angle bin ``a`` is an in-plane rotation of ``a * pi / num_angle`` and
    depth bin ``d`` an insertion depth of ``(d + 1) * depth_step`` metres. Views are the
    Fibonacci lattice of :func:`grasppanda.ops.fibonacci_views`.
    """

    num_view: int = 300
    num_angle: int = 12
    num_depth: int = 4
    depth_step: float = 0.01

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_view, self.num_angle, self.num_depth)

    @property
    def size(self) -> int:
        """Grasps per annotated point."""
        return self.num_view * self.num_angle * self.num_depth

    @property
    def angles(self) -> tuple[float, ...]:
        """In-plane rotation of each angle bin, radians."""
        return tuple(a * math.pi / self.num_angle for a in range(self.num_angle))

    @property
    def depths(self) -> tuple[float, ...]:
        """Insertion depth of each depth bin, metres."""
        return tuple((d + 1) * self.depth_step for d in range(self.num_depth))


#: Label grid of GraspNet-1Billion; GraspClutter6D reuses it.
GRASPNET_LABEL_GRID = LabelGrid()


class LabelFiles(Protocol):
    """Where a dataset keeps its GraspNet-format label files."""

    #: Dataset name, e.g. ``graspnet1b``.
    name: str
    #: Dataset directory; source files are recorded relative to it.
    root: Path
    #: Name of the label variant, part of the asset hash (e.g. ``grasp_label``).
    label_source: str

    def object_label_file(self, object_id: int) -> Path:
        """``.npz`` with ``points``, ``scores`` and either ``offsets`` or ``width``."""

    def collision_label_file(self, scene: int) -> Path:
        """``.npz`` with one boolean array ``arr_<i>`` per object instance of the scene."""

    def scene_object_ids(self, scene: int) -> list[int]:
        """Object id of every instance of a scene, in the order of the collision arrays."""


def _object_shard(object_id: int) -> str:
    return f"object_{object_id:06d}"


def _scene_shard(scene: int) -> str:
    return f"scene_{scene:06d}"


def _equal(stored: np.ndarray, source: np.ndarray, rows: int = 256) -> bool:
    """Value-for-value comparison in blocks, so that memory use stays bounded."""
    if stored.shape != source.shape or stored.dtype != source.dtype:
        return False
    for start in range(0, len(source), rows):
        block = slice(start, start + rows)
        if not np.array_equal(stored[block], source[block], equal_nan=stored.dtype.kind == "f"):
            return False
    return True


def open_array(path: str | os.PathLike[str]) -> np.ndarray:
    """Memory-map an ``.npy`` file of an asset, read-only.

    The name is handed to NumPy as a string: given a ``Path`` it resolves symbolic links
    component by component, a dozen file-system calls that each cost a round trip when
    the asset lives on a network file system.
    """
    return np.load(os.fspath(path), mmap_mode="r")


def _map_budget() -> int:
    """How many memory maps deferred labels may keep open in one process.

    A map holds a file descriptor. Half of the limit on open files, at most 4096: a
    training on GraspNet-1Billion reads 220 files (two per object model of its 40, one
    per scene of its 100, one more per object with tolerance labels), which fits the
    common limit of 1024.
    """
    try:
        import resource
    except ImportError:  # no such limit to read on this platform
        return 512
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    return 4096 if soft == resource.RLIM_INFINITY else max(64, min(4096, soft // 2))


#: Set this environment variable to ``0`` to switch the read-ahead of label files off.
READ_AHEAD_VARIABLE = "GRASPPANDA_LABEL_READ_AHEAD"


class _ReadAhead:
    """Reads label files front to back in the background, to fill the page cache.

    Deferred labels are read row by row, where the matched label points are. That is
    fast once the files are in the page cache and slow before: a row that is not costs
    a round trip to the disk (about 4 ms on the network store of the development
    machine, so about 10 s for the rows of one batch), and the training process waits
    for it. The same store delivers a file read front to back six times faster. So the
    first time a file is opened it is also read once, to its end, on a background
    thread: the rows a batch needs are then usually there, and after the first minutes
    of a run started on a cold cache they always are. A file that is in the cache
    already costs a few tens of milliseconds; nothing is kept, and the threads end when
    the queue is empty.
    """

    THREADS = 2  # two sequential streams fill a gigabit link
    CHUNK = 4 << 20

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[str] = deque()
        self._requested: set[str] = set()
        self._running = 0

    def request(self, path: str) -> None:
        """Queue ``path`` unless it was queued before in this process."""
        if os.environ.get(READ_AHEAD_VARIABLE, "1").strip() == "0":
            return
        with self._lock:
            if path in self._requested:
                return
            self._requested.add(path)
            self._pending.append(path)
            if self._running < self.THREADS:
                self._running += 1
                threading.Thread(target=self._drain, name="label-read-ahead", daemon=True).start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._running -= 1
                    return
                path = self._pending.popleft()
            self.read(path)

    @classmethod
    def read(cls, path: str) -> int:
        """Read a file to its end and drop what was read; the number of bytes that were."""
        total, buffer = 0, bytearray(cls.CHUNK)
        try:
            with open(path, "rb", buffering=0) as handle:
                while count := handle.readinto(buffer):
                    total += count
        except OSError:  # best effort: the reader itself reports a file it cannot read
            pass
        return total

    def idle(self) -> bool:
        """Whether nothing is queued or being read."""
        with self._lock:
            return self._running == 0 and not self._pending


#: Memory maps of this process that deferred labels read from, least recently used
#: first. Opening one costs several round trips on a network file system, so a map
#: that falls out is expensive to get back: raise ``ulimit -n`` if a dataset has more
#: label files than :func:`_map_budget` allows.
_MAPS: OrderedDict[str, np.ndarray] = OrderedDict()
_MAX_MAPS = _map_budget()
_read_ahead = _ReadAhead()


def _start_afresh() -> None:
    """A forked process (a dataloader worker) starts without the maps of its parent, of
    which it would only hold the file descriptors, and without its read-ahead threads,
    which do not exist in it."""
    global _read_ahead
    _MAPS.clear()
    _read_ahead = _ReadAhead()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_start_afresh)


def _mapped(path: str) -> np.ndarray:
    """The array of ``path`` as a memory map, opened once per process."""
    array = _MAPS.get(path)
    if array is None:
        array = _MAPS[path] = open_array(path)
        _read_ahead.request(path)
        while len(_MAPS) > _MAX_MAPS:
            _MAPS.popitem(last=False)
    else:
        _MAPS.move_to_end(path)
    return array


@dataclass(frozen=True)
class GridFiles:
    """Where the grids of one object instance of a scene are stored.

    Attributes:
        scores, widths: ``.npy`` files of the object model, ``(P, V, A, D)`` float32.
        collision: ``.npy`` file of the scene's bit-packed collision grids.
        collision_start: Row of that file at which the rows of this instance begin.
    """

    scores: str
    widths: str
    collision: str
    collision_start: int


#: Value of each bit of a packed collision byte, first grasp first.
_BITS = (128, 64, 32, 16, 8, 4, 2, 1)

#: Page-locked host memory that rows are read into on their way to a CUDA device, kept
#: for the life of the process: a transfer from it is about twice as fast as from
#: ordinary memory, and reusing it spares the page faults of a fresh buffer per batch.
_staging: dict[str, Tensor] = {}
_staging_lock = threading.Lock()


def _host_buffer(shape: tuple[int, ...], kind: torch.dtype, device: torch.device) -> Tensor:
    """An uninitialised CPU tensor of ``shape``; for a CUDA target a view of the staging
    memory, to be copied to the device before the next call (hold ``_staging_lock``)."""
    if device.type != "cuda":
        return torch.empty(shape, dtype=kind)
    size = math.prod(shape) * torch.empty(0, dtype=kind).element_size()
    held = _staging.get("bytes")
    if held is None or held.numel() < size:
        held = torch.empty(max(size + size // 4, 1 << 20), dtype=torch.uint8)
        # If page-locked memory is refused, ordinary memory works too, only slower.
        with contextlib.suppress(RuntimeError):
            held = held.pin_memory()
        _staging["bytes"] = held
    return held[:size].view(kind).view(shape)


def _take_rows(source: np.ndarray, rows: np.ndarray, out: np.ndarray, what: str) -> None:
    """``out[:] = source[rows]`` for ascending ``rows``, copied straight into ``out``."""
    if len(rows) and (rows[0] < 0 or rows[-1] >= len(source)):
        raise AssetError(
            f"{what}: rows {int(rows[0])}..{int(rows[-1])} were asked of an array of "
            f"{len(source)} rows; the label store is not the one the labels were chosen from"
        )
    # `clip` only switches off NumPy's buffered copy; the range was checked above.
    np.take(source, rows, axis=0, out=out, mode="clip")


class DeferredGraspLabels:
    """The grids of chosen label points of one object instance, left in the store.

    Implements :class:`grasppanda.structures.observation.DeferredGrids`. The object is
    small and picklable: file names, rows and two options. :meth:`gather` runs in
    whichever process matches labels; it opens the files it needs once per process
    (:func:`_mapped`), reads every distinct row once, in file order, into one buffer,
    moves the buffer to the device of the request in one transfer, and expands rows that
    several requests share there.

    Args:
        files: Where the grids are stored.
        rows: ``(P,)`` rows of the object's stored grids that the label points are;
            label point ``i`` of the sample is stored row ``rows[i]``.
        grid: Label grid of the store.
        collision: ``zero_scores``: the score (and the tolerance) of a grasp that
            collides in this scene reads as 0. ``attach``: scores are read as stored and
            the mask can be read as the grid ``collision``. ``ignore``: collisions are
            not read.
        tolerance_file: ``.npy`` file with the object's ``(P, V, A, D)`` uint8 tolerance
            levels, when the dataset serves tolerance.
        tolerance_radii: Radius in metres of every tolerance level.
    """

    def __init__(
        self,
        files: GridFiles,
        rows: np.ndarray,
        grid: LabelGrid,
        collision: CollisionMode,
        tolerance_file: str | None = None,
        tolerance_radii: np.ndarray | None = None,
    ) -> None:
        if collision not in ("zero_scores", "attach", "ignore"):
            raise ValueError(f"collision must be zero_scores, attach or ignore, got {collision!r}")
        if (tolerance_file is None) != (tolerance_radii is None):
            raise ValueError("tolerance_file and tolerance_radii are given together or not at all")
        self.files = files
        self.rows = np.ascontiguousarray(rows, dtype=np.int64)
        self.grid = grid
        self.collision_mode = collision
        self.tolerance_file = tolerance_file
        self.tolerance_radii = tolerance_radii

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.rows), *self.grid.shape)

    @property
    def has_collision(self) -> bool:
        return self.collision_mode == "attach"

    @property
    def has_tolerance(self) -> bool:
        return self.tolerance_file is not None

    def _file(self, name: str) -> str:
        file = {
            "scores": self.files.scores,
            "widths": self.files.widths,
            "tolerance": self.tolerance_file,
        }[name]
        if file is None:
            raise AssetError("these grasp labels were deferred without tolerance")
        return file

    # ------------------------------------------------------------------- reading

    @classmethod
    def gather(
        cls,
        readers: Sequence[DeferredGraspLabels],
        name: str,
        owner: Tensor,
        rows: Tensor,
        views: Tensor | None = None,
    ) -> Tensor:
        """Read grid ``name`` for label point ``rows[k]`` of ``readers[owner[k]]``.

        ``(K, V, A, D)``, or ``(K, A, D)`` at view ``views[k]``, on the device of ``rows``.
        """
        grid = readers[0].grid
        if any(reader.grid != grid for reader in readers):
            raise AssetError("the grasp labels of one batch come from stores of different grids")
        if name == "collision" and not all(reader.has_collision for reader in readers):
            raise AssetError("a collision mask was asked of labels that were deferred without it")
        index = [owner, rows] if views is None else [owner, rows, views]
        host = torch.stack(index).cpu().numpy()  # one transfer, however many objects
        if host.shape[1] and not 0 <= host[0].min() <= host[0].max() < len(readers):
            raise AssetError(
                f"a request names object {int(host[0].max())}, but {len(readers)} were given"
            )
        order = np.argsort(host[0], kind="stable")
        bounds = np.searchsorted(host[0][order], np.arange(len(readers) + 1))
        groups = [  # (reader, its requests, the stored rows they read)
            (reader, members, reader.rows[host[1][members]])
            for position, reader in enumerate(readers)
            if len(members := order[bounds[position] : bounds[position + 1]])
        ]
        if views is not None:
            return cls._cells(groups, name, host[2], grid, len(host[0]), rows.device)

        inverse = np.empty(len(host[0]), dtype=np.int64)  # request -> row of the buffer
        plan, total = [], 0  # (reader, distinct stored rows, first row of the buffer)
        for reader, members, stored in groups:
            distinct, back = np.unique(stored, return_inverse=True)
            inverse[members] = total + back.reshape(-1)
            plan.append((reader, distinct, total))
            total += len(distinct)
        device = rows.device
        if name == "collision":
            values = cls._collides(plan, total, grid, device, "attach")
        else:
            kind = torch.uint8 if name == "tolerance" else torch.float32
            with _staging_lock:
                buffer = _host_buffer((total, *grid.shape), kind, device)
                target = buffer.numpy()
                for reader, distinct, start in plan:
                    file = reader._file(name)
                    _take_rows(_mapped(file), distinct, target[start : start + len(distinct)], file)
                values = buffer.to(device)
            if name == "tolerance":
                values = torch.from_numpy(cls._radii(readers)).to(device)[values.long()]
            if name in ("scores", "tolerance"):
                collides = cls._collides(plan, total, grid, device, "zero_scores")
                if collides is not None:
                    values.masked_fill_(collides, 0.0)
        assert values is not None
        return values[torch.from_numpy(inverse).to(device)]

    @staticmethod
    def _radii(readers: Sequence[DeferredGraspLabels]) -> np.ndarray:
        radii = readers[0].tolerance_radii
        if radii is None or any(
            reader.tolerance_radii is None or not np.array_equal(reader.tolerance_radii, radii)
            for reader in readers
        ):
            raise AssetError("the grasp labels of one batch carry different tolerance assets")
        return radii

    @staticmethod
    def _collides(
        plan: Sequence[tuple[DeferredGraspLabels, np.ndarray, int]],
        total: int,
        grid: LabelGrid,
        device: torch.device,
        mode: str,
    ) -> Tensor | None:
        """``(total, V, A, D)`` collision mask of the planned rows, unpacked on ``device``.

        Rows of readers in another collision mode read as free; ``None`` if all are.
        """
        if not any(reader.collision_mode == mode for reader, _, _ in plan):
            return None
        packed = np.zeros((total, (grid.size + 7) // 8), dtype=np.uint8)
        for reader, distinct, start in plan:
            if reader.collision_mode == mode:
                file = reader.files.collision
                rows = reader.files.collision_start + distinct
                _take_rows(_mapped(file), rows, packed[start : start + len(distinct)], file)
        bits = torch.tensor(_BITS, dtype=torch.uint8, device=device)
        mask = (torch.from_numpy(packed).to(device)[:, :, None] & bits) != 0
        return mask.reshape(total, -1)[:, : grid.size].reshape(total, *grid.shape)

    @classmethod
    def _cells(
        cls,
        groups: Sequence[tuple[DeferredGraspLabels, np.ndarray, np.ndarray]],
        name: str,
        views: np.ndarray,
        grid: LabelGrid,
        count: int,
        device: torch.device,
    ) -> Tensor:
        """``(K, A, D)``: grid ``name`` of every request at its view."""
        if name not in ("widths", "tolerance"):
            raise ValueError(f"'{name}' is read for all views at once; pass no views")
        out = np.zeros((count, grid.num_angle, grid.num_depth), dtype=np.float32)
        for reader, members, stored in groups:
            at = views[members]
            cells = _mapped(reader._file(name))[stored, at]
            if name == "tolerance":
                assert reader.tolerance_radii is not None
                cells = reader.tolerance_radii[cells]
                if reader.collision_mode == "zero_scores":
                    np.putmask(cells, reader._cell_collides(stored, at), 0)
            out[members] = cells
        return torch.from_numpy(out).to(device)

    def _cell_collides(self, stored: np.ndarray, views: np.ndarray) -> np.ndarray:
        """``(K, A, D)`` collision of the grasps of stored row ``stored[k]`` at ``views[k]``."""
        cells = self.grid.num_angle * self.grid.num_depth
        bit = views[:, None] * cells + np.arange(cells)[None, :]  # position in the row
        row = (self.files.collision_start + stored)[:, None]
        byte = _mapped(self.files.collision)[row, bit >> 3]
        collides = (byte >> (7 - (bit & 7)).astype(np.uint8)) & 1
        return collides.astype(np.bool_).reshape(len(stored), self.grid.num_angle, -1)


class StoredGraspLabels:
    """Labels of one object instance in a scene; implements ``GraspLabelSource``.

    Args:
        points: ``(P, 3)`` annotated grasp points.
        scores, widths: ``(P, V, A, D)`` grids of the object model.
        packed_collision: ``(P, ceil(V * A * D / 8))`` bit-packed collision grid of the
            instance in its scene.
        grid: Label grid of the store.
        files: Where these arrays are stored, if they are; :meth:`defer` needs it.
    """

    def __init__(
        self,
        points: np.ndarray,
        scores: np.ndarray,
        widths: np.ndarray,
        packed_collision: np.ndarray,
        grid: LabelGrid,
        files: GridFiles | None = None,
    ) -> None:
        self.points, self.scores, self.widths = points, scores, widths
        self.packed_collision = packed_collision
        self.grid = grid
        self.files = files
        self.num_points = len(points)
        #: Tolerance levels ``(P, V, A, D)`` and the radius of every level, attached by a
        #: dataset that serves tolerance labels (see :mod:`grasppanda.datasets.tolerance`).
        self.tolerance_levels: np.ndarray | None = None
        self.tolerance_radii: np.ndarray | None = None

    def collision(self, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        """``(K, V, A, D)`` boolean collision grid of the given label points."""
        packed = np.asarray(self.packed_collision[rows])
        flat = np.unpackbits(packed, axis=1, count=self.grid.size)
        return flat.view(np.bool_).reshape(len(packed), *self.grid.shape)

    def take(
        self, rows: np.ndarray, *, collision: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Copy the labels of ``rows`` out of the store, in the given order."""
        return (
            np.asarray(self.points[rows]),
            np.asarray(self.scores[rows]),
            np.asarray(self.widths[rows]),
            self.collision(rows) if collision else None,
        )

    def tolerance(self, rows: np.ndarray) -> np.ndarray | None:
        """``(K, V, A, D)`` float32 tolerance in metres of ``rows``; ``None`` if not served."""
        if self.tolerance_levels is None or self.tolerance_radii is None:
            return None
        self._check_tolerance()
        return self.tolerance_radii[np.asarray(self.tolerance_levels[rows])]

    def _check_tolerance(self) -> None:
        assert self.tolerance_levels is not None
        if len(self.tolerance_levels) != self.num_points:
            raise AssetError(
                f"tolerance labels cover {len(self.tolerance_levels)} points, but the object has "
                f"{self.num_points} annotated grasp points"
            )

    def defer(self, rows: np.ndarray, *, collision: CollisionMode) -> DeferredGraspLabels:
        """The grids of ``rows`` as a reader that leaves them in the store.

        What :meth:`take` would copy now, :class:`DeferredGraspLabels` copies later and
        only for the rows that are asked for; the values are the same.
        """
        if self.files is None:
            raise AssetError(
                "these grasp labels are held in memory, not in a label store, so there is "
                "no file their grids could be read from later; read them with take()"
            )
        tolerance_file = None
        if self.tolerance_levels is not None and self.tolerance_radii is not None:
            self._check_tolerance()
            tolerance_file = getattr(self.tolerance_levels, "filename", None)
            if tolerance_file is None:
                raise AssetError(
                    "the tolerance levels of these grasp labels are not a memory-mapped "
                    "file, so they cannot be read later; read them with tolerance()"
                )
        return DeferredGraspLabels(
            self.files,
            np.asarray(rows, dtype=np.int64),
            self.grid,
            collision,
            None if tolerance_file is None else os.fspath(tolerance_file),
            self.tolerance_radii if tolerance_file is not None else None,
        )


class _Mapped:
    """A grid of :class:`_MappedOnDemand`: read from the store until a value is assigned."""

    def __init__(self, read: Callable[[_MappedOnDemand], np.ndarray]) -> None:
        self.read = read

    def __set_name__(self, owner: type, name: str) -> None:
        self.slot = f"_assigned_{name}"

    def __get__(self, labels: _MappedOnDemand | None, owner: type | None = None) -> Any:
        if labels is None:
            return self
        assigned = vars(labels).get(self.slot)
        return self.read(labels) if assigned is None else assigned

    def __set__(self, labels: _MappedOnDemand, value: np.ndarray) -> None:
        vars(labels)[self.slot] = value
        labels.files = None  # no longer what the files hold: nothing to defer to


class _MappedOnDemand(StoredGraspLabels):
    """Labels of a store whose grids are mapped when one of them is first read.

    A dataloader worker that defers the grids only needs the points, and then never
    opens the three files behind ``scores``, ``widths`` and ``packed_collision``.
    """

    def __init__(
        self,
        store: GraspLabelStore,
        scene: int,
        index: int,
        object_id: int,
        points: np.ndarray,
        files: GridFiles,
    ) -> None:
        self._store, self._scene, self._index, self._object_id = store, scene, index, object_id
        self.points = points
        self.grid = store.grid
        self.files = files
        self.num_points = len(points)
        self.tolerance_levels = None
        self.tolerance_radii = None

    def _collision_rows(self) -> np.ndarray:
        packed, offsets = self._store._scene(self._scene)
        return packed[offsets[self._index] : offsets[self._index + 1]]

    scores = _Mapped(lambda self: self._store.object_arrays(self._object_id)[1])
    widths = _Mapped(lambda self: self._store.object_arrays(self._object_id)[2])
    packed_collision = _Mapped(_collision_rows)


class GraspLabelStore:
    """Memory-mapped grasp labels of one dataset.

    Args:
        files: Where the dataset keeps the source label files.
        grid: Label grid the dataset declares; arrays of any other shape are rejected.
        root: Asset root; defaults to :func:`grasppanda.utils.paths.cache_root`.
        max_open_objects, max_open_scenes: Bounds on the memory maps a process keeps
            open; the least recently used are closed first. An object takes two file
            descriptors and a scene one, so the defaults stay below the usual limit of
            1024 while covering every object and scene of GraspNet-1Billion.
    """

    KIND = "grasp_labels"
    GENERATOR = "grasppanda.grasp_label_store"
    VERSION = 1

    def __init__(
        self,
        files: LabelFiles,
        grid: LabelGrid = GRASPNET_LABEL_GRID,
        root: str | os.PathLike[str] | None = None,
        max_open_objects: int = 128,
        max_open_scenes: int = 256,
    ) -> None:
        self.files = files
        self.grid = grid
        spec = AssetSpec(
            dataset=files.name,
            kind=self.KIND,
            generator=self.GENERATOR,
            version=self.VERSION,
            parameters={
                "source": files.label_source,
                "grid": list(grid.shape),
                "scores": "float32",
                "widths": "float32",
                "collision": "packbits-big",
            },
            inputs={"dataset": files.name},
            about={
                "points": "(P, 3) float32, object frame, metres",
                "scores": "(P, V, A, D) float32 friction coefficient: lower is better, <= 0 invalid",
                "widths": "(P, V, A, D) float32 gripper width, metres",
                "collision": "per scene, rows of all instances concatenated; (sum P, V*A*D/8) "
                "uint8, bits packed big-endian, 1 = grasp collides in this scene",
            },
        )
        self.asset = Asset(spec, root)
        self._max_open = (max_open_objects, max_open_scenes)
        self._objects: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self._scenes: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._points: dict[int, np.ndarray] = {}

    def __getstate__(self) -> dict[str, Any]:
        # Memory maps are reopened lazily in every process that receives the store.
        state = self.__dict__.copy()
        state["_objects"], state["_scenes"], state["_points"] = OrderedDict(), OrderedDict(), {}
        return state

    # ------------------------------------------------------------------------- paths

    def _object_dir(self, object_id: int) -> Path:
        return self.asset.path / "objects" / f"{object_id:06d}"

    def _collision_file(self, scene: int) -> Path:
        return self.asset.path / "scenes" / f"{scene:06d}.collision.npy"

    # ------------------------------------------------------------------------- build

    def build(
        self,
        objects: Iterable[int],
        scenes: Iterable[int],
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """Convert the labels of the given objects and scenes; skip finished shards.

        Returns:
            Counts of ``built`` and ``skipped`` shards.
        """
        self.asset.create()
        done = self.asset.shards(refresh=True)
        counts = {"built": 0, "skipped": 0}
        jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        for object_id in sorted(set(objects)):
            jobs.append((_object_shard(object_id), lambda i=object_id: self._build_object(i)))
        for scene in sorted(set(scenes)):
            jobs.append((_scene_shard(scene), lambda s=scene: self._build_scene(s)))
        for shard, job in jobs:
            if shard in done:
                counts["skipped"] += 1
            else:
                self.asset.mark_complete(shard, **job())
                counts["built"] += 1
            if progress is not None:
                progress(shard)
        return counts

    def _read_object(self, object_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        file = self.files.object_label_file(object_id)
        if not file.is_file():
            raise FileNotFoundError(f"grasp labels of object {object_id} not found: {file}")
        with np.load(file) as archive:
            names = set(archive.files)
            if not {"points", "scores"} <= names or not names & {"offsets", "width"}:
                raise AssetError(
                    f"{file} is not a GraspNet-format label file: expected arrays 'points', "
                    f"'scores' and 'offsets' or 'width', found {sorted(names)}"
                )
            points = archive["points"].astype(np.float32, copy=False)
            scores = archive["scores"].astype(np.float32, copy=False)
            if "width" in names:
                widths = archive["width"].astype(np.float32, copy=False)
            else:  # offsets hold (angle, depth, width) per grasp; the first two are the grid
                widths = np.ascontiguousarray(archive["offsets"][..., 2], dtype=np.float32)
        expected = (len(points), *self.grid.shape)
        if points.ndim != 2 or points.shape[1] != 3:
            raise AssetError(f"{file}: 'points' must be (P, 3), got {points.shape}")
        for name, array in (("scores", scores), ("widths", widths)):
            if array.shape != expected:
                raise AssetError(
                    f"{file}: '{name}' has shape {array.shape}, but {len(points)} points on the "
                    f"declared label grid {self.grid.shape} need {expected}"
                )
        return points, scores, widths

    def _build_object(self, object_id: int) -> dict[str, Any]:
        arrays = dict(zip(("points", "scores", "widths"), self._read_object(object_id)))
        directory = self._object_dir(object_id)
        for name, array in arrays.items():
            save_array(directory / f"{name}.npy", array)
            stored = np.load(directory / f"{name}.npy", mmap_mode="r")
            if not _equal(stored, array):
                raise AssetError(
                    f"verification failed: {directory / name}.npy differs from its source "
                    f"{self.files.object_label_file(object_id)}"
                )
        return {
            "points": len(arrays["points"]),
            "source": self._source_record(self.files.object_label_file(object_id)),
        }

    def _source_record(self, file: Path) -> dict[str, Any]:
        """Fingerprint of an input file: where it was and how large."""
        try:
            name = file.relative_to(self.files.root).as_posix()
        except ValueError:
            name = file.name
        return {"file": name, "bytes": file.stat().st_size}

    def _build_scene(self, scene: int) -> dict[str, Any]:
        file = self.files.collision_label_file(scene)
        if not file.is_file():
            raise FileNotFoundError(f"collision labels of scene {scene} not found: {file}")
        object_ids = self.files.scene_object_ids(scene)
        with np.load(file) as archive:
            if len(archive.files) != len(object_ids):
                raise AssetError(
                    f"{file} holds {len(archive.files)} collision arrays, but scene {scene} has "
                    f"{len(object_ids)} object instances"
                )
            sources = [archive[f"arr_{index}"] for index in range(len(object_ids))]
        for index, (object_id, source) in enumerate(zip(object_ids, sources)):
            if source.dtype != np.bool_ or source.shape[1:] != self.grid.shape:
                raise AssetError(
                    f"{file}: arr_{index} is {source.dtype} {source.shape}; expected a boolean "
                    f"array (P, {', '.join(map(str, self.grid.shape))})"
                )
            shard = _object_shard(object_id)
            known = self.asset.shards().get(shard)
            if known is not None and known["points"] != len(source):
                raise AssetError(
                    f"{file}: arr_{index} has {len(source)} rows, but object {object_id} has "
                    f"{known['points']} annotated grasp points"
                )
        rows = [len(source) for source in sources]
        packed = np.concatenate(
            [np.packbits(source.reshape(len(source), -1), axis=1) for source in sources]
        )
        target = self._collision_file(scene)
        save_array(target, packed)
        stored = np.load(target, mmap_mode="r")
        start = 0
        for index, source in enumerate(sources):
            block = np.unpackbits(stored[start : start + len(source)], axis=1, count=self.grid.size)
            if not np.array_equal(block.view(np.bool_).reshape(source.shape), source):
                raise AssetError(f"verification failed: {target} differs from {file} arr_{index}")
            start += len(source)
        return {
            "objects": [int(i) for i in object_ids],
            "rows": rows,
            "source": self._source_record(file),
        }

    # -------------------------------------------------------------------------- read

    def require(self, objects: Iterable[int], scenes: Iterable[int], *, build: str) -> None:
        """Raise :class:`AssetError` unless the labels of these objects and scenes exist."""
        shards = [_object_shard(i) for i in sorted(set(objects))]
        shards += [_scene_shard(s) for s in sorted(set(scenes))]
        self.asset.require(shards, build=build)

    def missing(self, objects: Iterable[int], scenes: Iterable[int]) -> list[str]:
        """Shards of the given objects and scenes that have not been built."""
        shards = [_object_shard(i) for i in sorted(set(objects))]
        shards += [_scene_shard(s) for s in sorted(set(scenes))]
        return shards if not self.asset.exists() else self.asset.missing(shards)

    def object_points(self, object_id: int) -> np.ndarray:
        """``(P, 3)`` annotated grasp points of an object, held in memory."""
        points = self._points.get(object_id)
        if points is None:
            if not self.asset.is_complete(_object_shard(object_id)):
                raise AssetError(
                    f"grasp labels of object {object_id} are not in the store at {self.asset.path}"
                )
            points = np.load(os.fspath(self._object_dir(object_id) / "points.npy"))
            self._points[object_id] = points
        return points

    def object_arrays(self, object_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(points, scores, widths)`` of an object; the two grids are memory maps."""
        if object_id in self._objects:
            self._objects.move_to_end(object_id)
            return self._objects[object_id]
        points = self.object_points(object_id)
        directory = self._object_dir(object_id)
        arrays = (
            points,
            open_array(directory / "scores.npy"),
            open_array(directory / "widths.npy"),
        )
        self._objects[object_id] = arrays
        while len(self._objects) > self._max_open[0]:
            self._objects.popitem(last=False)
        return arrays

    def _scene_offsets(self, scene: int) -> np.ndarray:
        """Row at which each instance of a scene starts in its collision file, and the end."""
        record = self.asset.shards().get(_scene_shard(scene))
        if record is None:
            raise AssetError(
                f"collision labels of scene {scene} are not in the store at {self.asset.path}"
            )
        return np.concatenate([[0], np.cumsum(record["rows"])])

    def _scene(self, scene: int) -> tuple[np.ndarray, np.ndarray]:
        if scene in self._scenes:
            self._scenes.move_to_end(scene)
            return self._scenes[scene]
        entry = (open_array(self._collision_file(scene)), self._scene_offsets(scene))
        self._scenes[scene] = entry
        while len(self._scenes) > self._max_open[1]:
            self._scenes.popitem(last=False)
        return entry

    def scene_object_ids(self, scene: int) -> Sequence[int]:
        """Object ids of a scene's instances as recorded when its shard was built."""
        record = self.asset.shards().get(_scene_shard(scene))
        if record is None:
            raise AssetError(
                f"collision labels of scene {scene} are not in the store at {self.asset.path}"
            )
        return record["objects"]

    def labels(self, scene: int, index: int, object_id: int) -> StoredGraspLabels:
        """Labels of instance ``index`` of ``scene``, which is a copy of ``object_id``.

        Only the points are read here; the grids are mapped when they are first used.
        """
        offsets = self._scene_offsets(scene)
        recorded = self.scene_object_ids(scene)
        if index >= len(recorded) or recorded[index] != object_id:
            raise AssetError(
                f"scene {scene}: instance {index} is object {object_id} in the frame, but the "
                f"collision labels were stored for objects {list(recorded)}"
            )
        points = self.object_points(object_id)
        rows = int(offsets[index + 1] - offsets[index])
        if rows != len(points):
            raise AssetError(
                f"scene {scene}: collision labels of instance {index} have {rows} rows, "
                f"but object {object_id} has {len(points)} annotated grasp points"
            )
        directory = self._object_dir(object_id)
        files = GridFiles(
            scores=os.fspath(directory / "scores.npy"),
            widths=os.fspath(directory / "widths.npy"),
            collision=os.fspath(self._collision_file(scene)),
            collision_start=int(offsets[index]),
        )
        return _MappedOnDemand(self, scene, index, object_id, points, files)
