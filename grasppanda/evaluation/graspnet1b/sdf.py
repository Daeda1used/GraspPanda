"""Signed-distance grid queries with the reference tool's rounding and corner cases.

The force-closure test never touches a mesh: fingers close along a line through the
object's signed-distance grid, contacts are zero crossings on that line, and surface
normals are fitted to nearby grid samples. Three details of the reference decide
scores and are reproduced here deliberately:

* **float32 in, float64 through.** Coordinates are cast to float32 before each
  object <-> grid transform; the transform itself (``scale * x + translation``) and
  everything after it runs in float64.
* **Interpolation order.** Inside the grid the distance is a trilinear blend whose
  eight terms are accumulated in one fixed corner order. Corners beyond the last
  voxel contribute zero.
* **Outside the grid** the distance is measured to the nearest of a fixed set of
  surface voxels (``surface_sample``) plus the stored value there. That set was drawn
  at random when the official objects were built, so it is data, not an algorithm.

Coordinates in the "grid frame" are voxel indices as real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arithmetic import norm3

__all__ = ["SdfGrid"]

#: Radius, in voxels, of the neighbourhood a surface normal is fitted to.
_NORMAL_RADIUS = 1.5
#: Corners of a voxel cell, as (x, y, z) offsets, in the order their terms are summed.
_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1),
)


def _normal_offsets() -> np.ndarray:
    """The 27 sample offsets of a normal fit: the centre plus 26 directions at one radius."""
    steps = (-_NORMAL_RADIUS, 0.0, _NORMAL_RADIUS)
    offsets = []
    for dx in steps:
        for dy in steps:
            for dz in steps:
                offset = np.array([dx, dy, dz])
                if dx != 0 or dy != 0 or dz != 0:
                    offset = _NORMAL_RADIUS * offset / norm3(offset)
                offsets.append(offset)
    return np.array(offsets)


_NORMAL_OFFSETS = _normal_offsets()


@dataclass(frozen=True, eq=False)
class SdfGrid:
    """A signed-distance field sampled on a regular, axis-aligned grid.

    Attributes:
        values: ``(nx, ny, nz)`` float64 signed distances, metres, negative inside.
        grid_to_object_scale: Voxel edge, metres.
        grid_to_object_translation: ``(3,)`` object-frame position of voxel ``(0, 0, 0)``.
        object_to_grid_scale: Voxels per metre, as stored with the object.
        object_to_grid_translation: ``(3,)`` grid position of the object origin.
        surface_threshold: ``|distance|`` below this counts as on the surface.
        surface_sample: ``(S, 3)`` int64 voxels used for queries outside the grid.
    """

    values: np.ndarray
    grid_to_object_scale: float
    grid_to_object_translation: np.ndarray
    object_to_grid_scale: float
    object_to_grid_translation: np.ndarray
    surface_threshold: float
    surface_sample: np.ndarray
    _sample_object: np.ndarray = field(init=False, repr=False)
    _sample_values: np.ndarray = field(init=False, repr=False)
    _upper: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.dtype != np.float64:
            raise ValueError(
                f"values must be a 3-D float64 array, got {self.values.dtype} {self.values.shape}"
            )
        sample = np.asarray(self.surface_sample)
        shape = np.array(self.values.shape)
        if sample.ndim != 2 or sample.shape[1] != 3 or len(sample) == 0:
            raise ValueError(f"surface_sample must have shape (S, 3), got {sample.shape}")
        if not np.issubdtype(sample.dtype, np.integer):
            raise ValueError(f"surface_sample must hold voxel indices, got dtype {sample.dtype}")
        if (sample < 0).any() or (sample >= shape).any():
            raise ValueError("surface_sample indexes outside the grid")
        set_ = object.__setattr__
        set_(self, "_sample_object", self.grid_to_object(sample.astype(np.float64)))
        set_(self, "_sample_values", self.values[sample[:, 0], sample[:, 1], sample[:, 2]])
        set_(self, "_upper", (shape - 1).astype(np.float64))

    # ------------------------------------------------------------------ frames

    def object_to_grid(self, points: np.ndarray) -> np.ndarray:
        """Object-frame points ``(..., 3)`` in metres to grid coordinates."""
        single = np.asarray(points).astype(np.float32).astype(np.float64)
        return self.object_to_grid_scale * single + self.object_to_grid_translation

    def grid_to_object(self, points: np.ndarray) -> np.ndarray:
        """Grid coordinates ``(..., 3)`` to object-frame points in metres."""
        single = np.asarray(points).astype(np.float32).astype(np.float64)
        return self.grid_to_object_scale * single + self.grid_to_object_translation

    @staticmethod
    def direction_to_object(directions: np.ndarray) -> np.ndarray:
        """Grid-frame directions to the object frame.

        The grid is axis-aligned and directions are not scaled, so only the float32
        round trip of the reference remains.
        """
        return np.asarray(directions).astype(np.float32).astype(np.float64)

    # ----------------------------------------------------------------- queries

    def signed_distance(self, coords: np.ndarray) -> np.ndarray:
        """Signed distance at grid coordinates ``(N, 3)``; returns ``(N,)`` float64."""
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
        if not np.isfinite(coords).all():
            raise ValueError("signed-distance queries must be finite")
        shape = np.array(self.values.shape, dtype=np.float64)
        outside = (coords < 0).any(axis=1) | (coords >= shape).any(axis=1)
        result = np.empty(len(coords))
        if outside.any():
            result[outside] = self._distance_outside(coords[outside])
        if not outside.all():
            result[~outside] = self._interpolate(coords[~outside])
        return result

    def on_surface(self, distance: np.ndarray) -> np.ndarray:
        """Whether signed distances are within the surface band."""
        return np.abs(distance) < self.surface_threshold

    def _interpolate(self, coords: np.ndarray) -> np.ndarray:
        # A coordinate in the last cell is snapped onto the last voxel plane; -0.0 becomes 0.
        snapped = np.where(coords > self._upper, self._upper, coords)
        snapped = np.where(snapped > 0, snapped, 0.0)
        base = np.floor(snapped)
        low = base.astype(np.int64)
        limit = np.array(self.values.shape, dtype=np.int64)
        total = np.zeros(len(coords))
        for corner in _CORNERS:
            index = low + corner
            position = base + corner
            factor = -np.abs(position - snapped) + 1
            weight = (factor[:, 0] * factor[:, 1]) * factor[:, 2]
            inside = (index < limit).all(axis=1)
            index = np.minimum(index, limit - 1)
            value = np.where(inside, self.values[index[:, 0], index[:, 1], index[:, 2]], 0.0)
            total = total + weight * value
        return total

    def _distance_outside(self, coords: np.ndarray) -> np.ndarray:
        nearest = np.empty(len(coords), dtype=np.int64)
        sample = self.surface_sample
        for start in range(0, len(coords), 512):  # bounds the (chunk, S, 3) temporary
            chunk = coords[start : start + 512]
            nearest[start : start + 512] = np.argmin(
                norm3(sample[None, :, :] - chunk[:, None, :]), axis=1
            )
        gap = self._sample_object[nearest] - self.grid_to_object(coords)
        return norm3(gap) + self._sample_values[nearest]

    # ----------------------------------------------------------------- normals

    def surface_normals(self, coords: np.ndarray) -> np.ndarray:
        """Unit surface normals at on-surface grid coordinates ``(N, 3)``, sign arbitrary.

        Each normal is the direction of least variance of the on-surface samples among
        27 points around the coordinate (itself and 26 directions at 1.5 voxels),
        taken in ascending order of their signed distance.

        The reference then orients the normal along increasing distance. Its only
        caller re-orients it against the closing direction and the force-closure test
        uses an absolute value, so that step cannot change a score and is omitted.

        Raises:
            ValueError: if a coordinate is not on the surface (nothing to fit).
        """
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        samples = coords[:, None, :] + _NORMAL_OFFSETS[None, :, :]
        distance = self.signed_distance(samples.reshape(-1, 3)).reshape(len(coords), -1)
        near = self.on_surface(distance)
        centre = len(_NORMAL_OFFSETS) // 2  # the zero offset
        if not near[:, centre].all():
            raise ValueError("surface normals are only defined at points on the surface")
        normals = np.empty((len(coords), 3))
        for row in range(len(coords)):
            kept = np.flatnonzero(near[row])
            kept = kept[np.argsort(distance[row, kept], kind="stable")]
            points = samples[row, kept]
            total = points[0]
            for point in points[1:]:  # accumulate in order; pairwise summation would differ
                total = total + point
            centred = points - total / len(points)
            normals[row] = np.linalg.svd(centred.T)[0][:, 2]
        return normals
