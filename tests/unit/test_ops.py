"""Operators, each checked against a brute-force oracle written out here.

The four operators that have a compiled implementation besides the pure-PyTorch
reference (``grasppanda.ops.backends``) run every test once per implementation, so
both are pinned to the same oracles. ``test_ops_equivalence.py`` compares the two
with each other on many more inputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda._ops import library
from grasppanda.ops import (
    approach_to_rotation,
    backends,
    ball_query,
    cylinder_query,
    fibonacci_views,
    furthest_point_sample,
    gather_points,
    grasp_nms,
    group_points,
    knn,
    rotation_angle,
    voxelize,
)


@pytest.fixture(params=["reference", "compiled"])
def backend(request, monkeypatch) -> str:
    """Run a test with each implementation of the accelerated operators."""
    monkeypatch.delenv(backends.ENV, raising=False)
    if request.param == "compiled":
        kernels, detail = library.kernels()
        if kernels is None:
            pytest.skip(f"no compiled kernels on this machine: {detail}")
    with backends.override(**dict.fromkeys(backends.OPERATORS, request.param)):
        yield request.param


both_backends = pytest.mark.usefixtures("backend")


def cloud(batch: int, count: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(batch, count, 3, generator=torch.Generator().manual_seed(seed))


def first_hits_oracle(mask: np.ndarray, nsample: int) -> tuple[np.ndarray, np.ndarray]:
    """The PointNet++ kernel loop, written out: first hits in order, padded with the first."""
    batch, queries, _ = mask.shape
    index = np.zeros((batch, queries, nsample), dtype=np.int64)
    hits = np.zeros((batch, queries), dtype=np.int64)
    for b in range(batch):
        for m in range(queries):
            found = np.flatnonzero(mask[b, m])[:nsample]
            hits[b, m] = len(found)
            if len(found):
                index[b, m] = found[0]
                index[b, m, : len(found)] = found
    return index, hits


# ----------------------------------------------------------------------------- FPS


def fps_oracle(points: np.ndarray, num_samples: int) -> list[int]:
    chosen = [0]
    distance = np.full(len(points), np.inf)
    for _ in range(num_samples - 1):
        distance = np.minimum(distance, ((points - points[chosen[-1]]) ** 2).sum(-1))
        chosen.append(int(distance.argmax()))
    return chosen


@both_backends
def test_fps_matches_the_definition():
    points = cloud(3, 200)
    index = furthest_point_sample(points, 32)
    assert index.shape == (3, 32) and (index[:, 0] == 0).all()
    for b in range(3):
        assert index[b].tolist() == fps_oracle(points[b].numpy(), 32)
        assert len(set(index[b].tolist())) == 32
    assert torch.equal(furthest_point_sample(points[1], 32), index[1])


@both_backends
def test_fps_can_ignore_points_near_the_origin():
    points = cloud(1, 50) + 1.0
    points[0, 7] = 0.0
    points[0, 9] = torch.tensor([0.01, 0.01, 0.01])
    assert 7 in furthest_point_sample(points, 5)[0].tolist()  # the origin is the farthest point
    kept = furthest_point_sample(points, 48, skip_near_origin=True)[0].tolist()
    assert 7 not in kept and 9 not in kept


@both_backends
def test_fps_refuses_to_oversample():
    with pytest.raises(ValueError, match="cannot sample 9 points from 8"):
        furthest_point_sample(cloud(1, 8), 9)


def kernel_winner(tied: list[int], count: int) -> int:
    """Tie resolution of the PointNet++ CUDA kernel, written out.

    Thread ``t`` scans the indices ``k ≡ t`` and keeps its first maximum; the threads
    are then reduced in a binary tree in which slot ``t + half`` is folded into slot
    ``t`` and the value already in the lower slot survives a tie.
    """
    threads = min(1 << (count.bit_length() - 1), 512)
    slots: dict[int, int] = {}
    for index in sorted(tied):
        slots.setdefault(index % threads, index)
    half = threads // 2
    while half:
        folded: dict[int, int] = {}
        for slot in sorted(slots):  # the lower slot of a pair is visited first
            folded.setdefault(slot % half, slots[slot])
        slots, half = folded, half // 2
    return slots[0]


@both_backends
@pytest.mark.parametrize(
    ("count", "tied"),
    [
        (40, [1, 2, 3, 4, 5]),
        (300, [3, 77, 130, 259, 290]),
        (1500, [2, 513, 700, 1025, 1499]),  # 513 and 1025 share a thread
    ],
)
def test_fps_can_resolve_ties_like_the_pointnet2_kernel(count, tied):
    points = cloud(1, count, seed=count)[0] * 0.05 + 1.0  # a small blob next to (1, 1, 1) ...
    points[0] = 1.0
    offsets = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5]])
    points[tied] = points[0] + offsets  # ... and five points exactly 0.5 away from point 0
    assert furthest_point_sample(points, 2).tolist() == [0, tied[0]]  # default: lowest index
    winner = kernel_winner(tied, count)
    picks = furthest_point_sample(points, 2, tie_break="pointnet2_cuda")
    assert picks.tolist() == [0, winner] and winner != tied[0]
    with pytest.raises(ValueError, match="unknown tie_break 'random'"):
        furthest_point_sample(points, 2, tie_break="random")


@both_backends
def test_fps_kernel_tie_order_reproduces_the_recorded_pointnet2_kernel():
    """Picks of the upstream CUDA kernel on lattice clouds (tools/make_gsnet_fixtures.py).

    Squared distances on the lattice are exact in float32, so the picks depend on the
    tie order alone. The cases cover 64, 256 and 512 kernel threads, repeated points,
    fewer distinct points than samples, and a point next to the origin.
    """
    record = np.load(Path(__file__).parent / "fixtures" / "pointnet2_fps.npz")
    cases = sum(name.startswith("points_") for name in record.files)
    assert cases == 8
    for case in range(cases):
        points = torch.from_numpy(record[f"points_{case}"])
        reference = record[f"index_{case}"].tolist()
        kernel = furthest_point_sample(
            points, len(reference), skip_near_origin=True, tie_break="pointnet2_cuda"
        )
        assert kernel.tolist() == reference, case
        plain = furthest_point_sample(points, len(reference), skip_near_origin=True)
        assert plain.tolist() != reference, case  # every case hinges on the tie order


def fps_definition(
    points: np.ndarray, num_samples: int, skip_near_origin: bool, tie_break: str
) -> list[int]:
    """The operator as its docstring defines it, one step at a time in NumPy.

    Squared distances are ``x² + y² + z²`` added left to right in the precision of the
    input; a tie under ``pointnet2_cuda`` is resolved by replaying the kernel's
    reduction tree (``kernel_winner``), not by the closed form the operator uses.
    """
    count = len(points)
    squares = points * points
    ignore = (squares[:, 0] + squares[:, 1]) + squares[:, 2] <= points.dtype.type(1e-3)
    distance = np.full(count, np.inf, dtype=points.dtype)
    chosen = [0]
    for _ in range(num_samples - 1):
        offset = points - points[chosen[-1]]
        squares = offset * offset
        distance = np.minimum(distance, (squares[:, 0] + squares[:, 1]) + squares[:, 2])
        candidates = (
            np.where(ignore, points.dtype.type(-1), distance) if skip_near_origin else distance
        )
        if tie_break == "lowest_index":
            chosen.append(int(np.argmax(candidates)))
        else:
            tied = np.flatnonzero(candidates == candidates.max()).tolist()
            chosen.append(kernel_winner(tied, count) if tied else 0)
    return chosen


def assorted_cloud(generator: torch.Generator, count: int, kind: str) -> torch.Tensor:
    """One ``(count, 3)`` cloud of a kind that stresses a different rule."""
    points = torch.rand(count, 3, generator=generator)
    if kind == "lattice":  # few distinct coordinates: exact ties and repeated points
        points = torch.round(points * 3) / 4
    elif kind == "repeats":  # as a frame padded with repeats of its first candidate
        points[count // 2 :] = points[0]
    elif kind == "origin":  # centred on the origin: several points inside the 1e-3 ball
        points = (points - 0.5) * 0.2
    return points


@both_backends
@pytest.mark.parametrize("kind", ["random", "lattice", "repeats", "origin"])
def test_fps_matches_the_definition_on_random_inputs(kind):
    generator = torch.Generator().manual_seed(
        ["random", "lattice", "repeats", "origin"].index(kind)
    )
    cases = 0
    for count in (1, 2, 3, 7, 31, 32, 33, 100, 257, 600, 1100):
        points = assorted_cloud(generator, count, kind)
        wanted = min(count, 48 if count < 600 else 12)
        for skip in (False, True):
            for tie_break in ("lowest_index", "pointnet2_cuda"):
                picks = furthest_point_sample(
                    points, wanted, skip_near_origin=skip, tie_break=tie_break
                )
                expected = fps_definition(points.numpy(), wanted, skip, tie_break)
                assert picks.tolist() == expected, (kind, count, skip, tie_break)
                cases += 1
    assert cases == 44


@both_backends
def test_fps_is_defined_per_cloud_and_for_double_precision():
    generator = torch.Generator().manual_seed(5)
    batch = torch.stack([assorted_cloud(generator, 90, kind) for kind in ("random", "lattice")])
    picks = furthest_point_sample(batch, 40, skip_near_origin=True, tie_break="pointnet2_cuda")
    for b in range(2):
        assert picks[b].tolist() == fps_definition(batch[b].numpy(), 40, True, "pointnet2_cuda")
    double = batch.double()  # always the reference: the kernels take float32
    picks = furthest_point_sample(double, 40)
    for b in range(2):
        assert picks[b].tolist() == fps_definition(double[b].numpy(), 40, False, "lowest_index")
    strided = batch.flip(1)[:, ::2]  # not contiguous
    assert furthest_point_sample(strided, 20)[0].tolist() == fps_definition(
        strided[0].numpy(), 20, False, "lowest_index"
    )


# ------------------------------------------------------------------------- queries


@both_backends
def test_knn_matches_brute_force_and_orders_ties_by_index():
    query, support = cloud(2, 17, 1), cloud(2, 60, 2)
    distance, index = knn(query, support, 5, chunk=4)
    full = torch.cdist(query.double(), support.double())
    expected = torch.sort(full, dim=-1, stable=True)
    assert torch.equal(index, expected.indices[..., :5])
    assert torch.allclose(distance.double(), expected.values[..., :5], atol=1e-6)

    grid = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]]])
    assert knn(torch.zeros(1, 1, 3), grid, 4)[1].tolist() == [[[0, 1, 2, 3]]]


@both_backends
@pytest.mark.parametrize("nsample", [4, 16, 80])
def test_ball_query_matches_the_kernel_loop(nsample):
    query, support = cloud(2, 21, 3), cloud(2, 64, 4)
    radius = 0.3
    index, hits = ball_query(query, support, radius, nsample, chunk=5)
    squared = ((support[:, None] - query[:, :, None]) ** 2).sum(-1)
    expected_index, expected_hits = first_hits_oracle((squared < radius**2).numpy(), nsample)
    assert np.array_equal(index.numpy(), expected_index)
    assert np.array_equal(hits.numpy(), expected_hits)


@both_backends
def test_empty_neighbourhood_is_distinguishable():
    index, hits = ball_query(torch.full((1, 1, 3), 10.0), cloud(1, 8), 0.1, 4)
    assert hits.tolist() == [[0]] and index.tolist() == [[[0, 0, 0, 0]]]


@both_backends
def test_cylinder_query_matches_the_kernel_loop():
    query, support = cloud(2, 19, 5), cloud(2, 300, 6)
    generator = torch.Generator().manual_seed(7)
    rotation = approach_to_rotation(
        torch.randn(2, 19, 3, generator=generator), torch.rand(2, 19, generator=generator) * math.pi
    )
    radius, hmin, hmax, nsample = 0.25, -0.1, 0.3, 12
    index, hits = cylinder_query(query, support, rotation, radius, hmin, hmax, nsample, chunk=7)

    mask = np.zeros((2, 19, 300), dtype=bool)
    for b in range(2):
        for m in range(19):
            local = (support[b] - query[b, m]).numpy() @ rotation[b, m].numpy()  # Rᵀ d, row-wise
            mask[b, m] = (
                (local[:, 1] ** 2 + local[:, 2] ** 2 < radius**2)
                & (local[:, 0] > hmin)
                & (local[:, 0] < hmax)
            )
    expected_index, expected_hits = first_hits_oracle(mask, nsample)
    assert np.array_equal(index.numpy(), expected_index)
    assert np.array_equal(hits.numpy(), expected_hits)
    assert 0 < hits.float().mean() < nsample  # the test exercises both full and partial groups


@both_backends
def test_cylinder_axis_is_the_approach_column():
    support = torch.tensor([[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]])
    query = torch.zeros(1, 1, 3)
    along_y = approach_to_rotation(torch.tensor([[[0.0, 1.0, 0.0]]]))
    index, hits = cylinder_query(query, support, along_y, 0.05, 0.0, 0.3, 2)
    assert hits.tolist() == [[1]] and index.tolist() == [[[1, 1]]]


@both_backends
@pytest.mark.parametrize("kind", ["random", "lattice", "repeats"])
def test_queries_match_the_kernel_loop_on_random_inputs(kind):
    """Ball query, cylinder query and kNN against loops written out, on awkward inputs.

    On the lattice many support points lie exactly on a ball or cylinder surface (the
    radii are lattice distances), so the strict thresholds decide; ``nsample`` is larger
    than some neighbourhoods and than one whole cloud; some queries have no neighbour.
    """
    generator = torch.Generator().manual_seed(["random", "lattice", "repeats"].index(kind) + 20)
    for count, queries in ((1, 3), (2, 5), (37, 11), (300, 40)):
        support = torch.stack([assorted_cloud(generator, count, kind) for _ in range(2)])
        query = torch.stack([assorted_cloud(generator, queries, kind) for _ in range(2)])
        query[:, -1] += 7.0  # a query far away from everything
        approach = torch.randn(2, queries, 3, generator=generator)
        approach[:, ::2] = torch.tensor([0.0, 0.0, 1.0])  # frames with entries 0 and ±1
        rotation = approach_to_rotation(approach, torch.rand(2, queries, generator=generator))
        rotation[:, ::2] = approach_to_rotation(approach[:, ::2])
        squared = ((support[:, None] - query[:, :, None]) ** 2).sum(-1)
        local = torch.einsum("bmnj,bmjk->bmnk", support[:, None] - query[:, :, None], rotation)
        for radius, nsample in ((0.25, 3), (0.5, 16), (0.75, 64)):
            index, hits = ball_query(query, support, radius, nsample)
            expected = first_hits_oracle((squared < radius**2).numpy(), nsample)
            assert np.array_equal(index.numpy(), expected[0]), (kind, count, radius, nsample)
            assert np.array_equal(hits.numpy(), expected[1])

            hmin, hmax = -0.25, 0.5
            index, hits = cylinder_query(query, support, rotation, radius, hmin, hmax, nsample)
            inside = (
                (local[..., 1] ** 2 + local[..., 2] ** 2 < radius**2)
                & (local[..., 0] > hmin)
                & (local[..., 0] < hmax)
            )
            expected = first_hits_oracle(inside.numpy(), nsample)
            assert np.array_equal(index.numpy(), expected[0]), (kind, count, radius, nsample)
            assert np.array_equal(hits.numpy(), expected[1])
        for k in {1, min(3, count), min(20, count), count}:
            distance, index = knn(query, support, k)
            order = np.argsort(squared.numpy(), axis=-1, kind="stable")[..., :k]
            assert np.array_equal(index.numpy(), order), (kind, count, k)
            assert torch.equal(distance, torch.gather(squared, 2, index).sqrt())


def test_gather_and_group():
    values, index = cloud(2, 10, 8), torch.tensor([[0, 9, 3], [5, 5, 1]])
    assert torch.equal(gather_points(values, index)[1, 2], values[1, 1])
    neighbours = torch.stack([index, index.flip(-1)], dim=-1)
    grouped = group_points(values, neighbours)
    assert grouped.shape == (2, 3, 2, 3) and torch.equal(grouped[0, 0, 1], values[0, 3])


# ----------------------------------------------------------------------------- NMS


def nms_inputs():
    translation = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [0.01, 0, 0], [0.2, 0, 0], [0.0, 0, 0]])
    angles = torch.tensor([0.0, 0.1, 1.0, 0.0, 0.0])
    rotation = approach_to_rotation(torch.tensor([0.0, 0.0, 1.0]).expand(5, 3), angles)
    score = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95])
    return translation, rotation, score


def test_nms_needs_both_distance_and_angle_to_suppress():
    translation, rotation, score = nms_inputs()
    # 4 wins; 0 and 1 are close in both senses; 2 is rotated away; 3 is far away.
    assert grasp_nms(translation, rotation, score).tolist() == [4, 2, 3]


def test_nms_thresholds_are_strict_and_ties_keep_input_order():
    translation = torch.tensor([[0.0, 0, 0], [0.03, 0, 0]], dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    score = torch.tensor([0.5, 0.5])
    assert grasp_nms(translation, rotation, score).tolist() == [0, 1]  # exactly 0.03 apart: kept
    assert grasp_nms(translation * 0.99, rotation, score).tolist() == [0]
    assert grasp_nms(torch.zeros(0, 3), torch.zeros(0, 3, 3), torch.zeros(0)).numel() == 0


# ---------------------------------------------------------------------------- voxel


def test_voxelize_inverse_maps_points_to_their_cells():
    coords = cloud(1, 500, 9)[0] - 0.5
    batch_index = torch.arange(500) % 2
    grid = voxelize(coords, 0.1, batch_index)
    cells = torch.floor(coords / 0.1).long()
    assert torch.equal(grid.voxels[grid.inverse][:, 1:], cells)
    assert torch.equal(grid.voxels[grid.inverse][:, 0], batch_index)
    assert len(torch.unique(grid.voxels, dim=0)) == len(grid.voxels)
    for voxel in range(0, len(grid.voxels), 37):
        members = torch.nonzero(grid.inverse == voxel).flatten()
        assert grid.first[voxel] == members.min()


# ---------------------------------------------------------------------------- views


def test_fibonacci_views_are_the_graspnet_template():
    views = fibonacci_views(300)
    assert views.shape == (300, 3)
    assert torch.allclose(views.norm(dim=-1), torch.ones(300), atol=1e-6)
    assert views[0, 2] == pytest.approx(1 / 300 - 1) and views[-1, 2] == pytest.approx(1 - 1 / 300)
    golden = (math.sqrt(5) - 1) / 2
    z = 2 * 7 + 1
    expected = [math.sqrt(1 - (z / 300 - 1) ** 2) * math.cos(2 * math.pi * 7 * golden)]
    assert views[7, 0] == pytest.approx(expected[0], abs=1e-6)


def test_approach_to_rotation_builds_proper_frames():
    generator = torch.Generator().manual_seed(11)
    approach = torch.randn(64, 3, generator=generator)
    angle = torch.rand(64, generator=generator) * math.pi
    rotation = approach_to_rotation(approach, angle)
    identity = torch.eye(3).expand(64, 3, 3)
    assert torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(64), atol=1e-5)
    assert torch.allclose(
        rotation[:, :, 0], approach / approach.norm(dim=-1, keepdim=True), atol=1e-6
    )
    assert torch.allclose(
        rotation_angle(approach_to_rotation(approach), rotation), angle, atol=1e-3
    )

    base = approach_to_rotation(approach)
    assert torch.allclose(base[:, 2, 1], torch.zeros(64), atol=1e-6)  # closing axis is horizontal
    vertical = approach_to_rotation(torch.tensor([0.0, 0.0, -1.0]))
    assert torch.equal(vertical[:, 1], torch.tensor([0.0, 1.0, 0.0]))
