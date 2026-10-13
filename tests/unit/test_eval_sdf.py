"""Signed-distance queries on analytic shapes, against scalar restatements of each rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b.sdf import SdfGrid

from .eval_synthetic import (
    BOX_HALF,
    CELLS,
    RESOLUTION,
    SPHERE_RADIUS,
    box_distance,
    make_grid,
    sphere_distance,
)

#: Corner order of the trilinear blend: (x, y, z) offsets, summed in this sequence.
CORNERS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


@pytest.fixture(scope="module")
def sphere() -> SdfGrid:
    return make_grid(sphere_distance)


@pytest.fixture(scope="module")
def box() -> SdfGrid:
    return make_grid(box_distance)


def scalar_distance(grid: SdfGrid, coord: np.ndarray) -> float:
    """One query, written out the slow way."""
    shape = grid.values.shape
    if any(c < 0 for c in coord) or any(c >= n for c, n in zip(coord, shape)):
        sample = grid.surface_sample
        gaps = [math.sqrt(sum(float(s - c) ** 2 for s, c in zip(row, coord))) for row in sample]
        nearest = sample[int(np.argmin(gaps))]
        there = grid.grid_to_object(nearest.astype(np.float64))
        here = grid.grid_to_object(np.asarray(coord))
        d = there - here
        return math.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) + grid.values[tuple(nearest)]
    snapped = [max(0, min(c, n - 1)) for c, n in zip(coord, shape)]
    low = [math.floor(c) for c in snapped]
    total = 0.0
    for corner in CORNERS:
        index = [lo + o for lo, o in zip(low, corner)]
        value = 0.0 if any(i >= n for i, n in zip(index, shape)) else grid.values[tuple(index)]
        weight = 1.0
        for i, c in zip(index, snapped):
            weight = weight * (1 - abs(i - c))
        total = total + weight * value
    return total


def test_lattice_points_return_the_stored_values(sphere):
    rng = np.random.default_rng(0)
    index = rng.integers(0, CELLS, size=(200, 3))
    values = sphere.signed_distance(index.astype(np.float64))
    assert np.array_equal(values, sphere.values[index[:, 0], index[:, 1], index[:, 2]])


def test_interpolation_matches_the_scalar_rule(sphere, box):
    rng = np.random.default_rng(1)
    coords = np.concatenate(
        [
            rng.uniform(0, CELLS - 1, size=(300, 3)),
            rng.uniform(CELLS - 1.5, CELLS - 1e-9, size=(100, 3)),  # the last cell
            np.array([[0.0, 0.0, 0.0], [-0.0, 3.5, 2.25], [CELLS - 1.0] * 3, [CELLS - 0.5, 1, 2]]),
        ]
    )
    for grid in (sphere, box):
        values = grid.signed_distance(coords)
        assert np.array_equal(values, [scalar_distance(grid, c) for c in coords])
    inside = coords[:300]
    world = sphere.grid_to_object_scale * inside + sphere.grid_to_object_translation
    assert np.abs(sphere.signed_distance(inside) - sphere_distance(world)).max() < RESOLUTION


def test_last_cell_is_snapped_to_the_last_plane(sphere):
    last = CELLS - 1.0
    assert (
        sphere.signed_distance(np.array([[last + 0.7, 10.0, 10.0]]))[0] == sphere.values[-1, 10, 10]
    )


def test_outside_uses_the_frozen_surface_sample(sphere):
    rng = np.random.default_rng(2)
    coords = np.concatenate(
        [rng.uniform(-40, CELLS + 40, size=(200, 3)), [[-1e-9, 5, 5], [5, float(CELLS), 5]]]
    )
    outside = (coords < 0).any(axis=1) | (coords >= CELLS).any(axis=1)
    assert outside[-2:].all() and outside.sum() > 50
    values = sphere.signed_distance(coords)
    assert np.array_equal(values, [scalar_distance(sphere, c) for c in coords])
    # far outside, the result approximates the true distance to the sphere (sample is finite)
    world = sphere.grid_to_object_scale * coords[outside] + sphere.grid_to_object_translation
    assert np.abs(values[outside] - sphere_distance(world)).max() < 0.01


def test_outside_values_change_with_the_sample(sphere):
    other = make_grid(sphere_distance, seed=123)
    assert not np.array_equal(other.surface_sample, sphere.surface_sample)
    far = np.array([[-20.0, 31.3, 30.9], [90.2, 80.1, -7.7], [31.0, 31.0, 100.5]])
    assert not np.array_equal(sphere.signed_distance(far), other.signed_distance(far))
    near = np.array([[20.2, 31.3, 30.9]])
    assert np.array_equal(sphere.signed_distance(near), other.signed_distance(near))


def test_frame_transforms_round_to_float32_first(sphere):
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.05, 0.05, size=(50, 3))
    single = points.astype(np.float32).astype(np.float64)
    assert not np.array_equal(single, points)
    expected = sphere.object_to_grid_scale * single + sphere.object_to_grid_translation
    assert np.array_equal(sphere.object_to_grid(points), expected)
    grid_coords = rng.uniform(0, CELLS, size=(50, 3))
    single = grid_coords.astype(np.float32).astype(np.float64)
    expected = sphere.grid_to_object_scale * single + sphere.grid_to_object_translation
    assert np.array_equal(sphere.grid_to_object(grid_coords), expected)
    assert np.allclose(sphere.grid_to_object(sphere.object_to_grid(points)), points, atol=1e-8)
    directions = rng.normal(size=(5, 3))
    assert np.array_equal(
        SdfGrid.direction_to_object(directions), directions.astype(np.float32).astype(np.float64)
    )


def test_surface_band(sphere):
    band = sphere.surface_threshold
    assert band == RESOLUTION * math.sqrt(2) / 2
    assert sphere.on_surface(np.array([0.0, band * 0.999, -band * 0.999, band, -band])).tolist() == [
        True, True, True, False, False,
    ]  # fmt: skip


def test_sphere_normals_are_radial(sphere):
    rng = np.random.default_rng(4)
    directions = rng.normal(size=(60, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coords = sphere.object_to_grid(SPHERE_RADIUS * directions)
    normals = sphere.surface_normals(coords)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    cosine = np.abs((normals * directions).sum(axis=1))
    # the fit uses at most 27 samples 1.5 voxels apart, so it is only roughly radial
    assert cosine.min() > math.cos(math.radians(15.0)) and np.median(cosine) > 0.99


def test_box_face_normals_are_axis_aligned(box):
    for axis in range(3):
        point = np.zeros(3)
        point[axis] = BOX_HALF[axis]
        normal = box.surface_normals(box.object_to_grid(point[None]))[0]
        assert abs(normal[axis]) > 0.999


def test_normals_need_a_surface_point(sphere):
    with pytest.raises(ValueError, match="on the surface"):
        sphere.surface_normals(sphere.object_to_grid(np.zeros((1, 3))))


def test_grid_validation(sphere):
    kwargs = {
        "grid_to_object_scale": 1.0,
        "grid_to_object_translation": np.zeros(3),
        "object_to_grid_scale": 1.0,
        "object_to_grid_translation": np.zeros(3),
        "surface_threshold": 0.7,
    }
    values = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="3-D float64"):
        SdfGrid(values=values.astype(np.float32), surface_sample=np.zeros((1, 3), int), **kwargs)
    with pytest.raises(ValueError, match="outside the grid"):
        SdfGrid(values=values, surface_sample=np.array([[0, 0, 4]]), **kwargs)
    with pytest.raises(ValueError, match="voxel indices"):
        SdfGrid(values=values, surface_sample=np.zeros((1, 3)), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        sphere.signed_distance(np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        sphere.signed_distance(np.zeros(3))
