"""The compiled kernels against the pure-PyTorch reference, on randomized inputs.

A backend is accepted only if it returns what the reference returns: identical indices
and hit counts, and (kNN) identical distances. The comparisons here go through the
public operators, once per backend, on clouds chosen to hit the rules that are easy to
get wrong: exact ties and repeated points (lattice clouds, padded candidate sets),
points next to the origin, neighbourhoods that are empty or larger than ``nsample``,
support points exactly on or within round-off of a ball or cylinder surface, clouds of
one to three points, non-contiguous tensors, NaN and infinite coordinates, and batches
of clouds of different sizes.

The ``compare_*`` functions take a device, so ``tests/integration/test_ops_cuda.py``
runs the same comparisons on a GPU. Everything is skipped on a machine without the
compiled kernels.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from grasppanda._ops import library
from grasppanda.models.backbones._packed import ball_query_packed, fps_packed, knn_packed
from grasppanda.ops import (
    approach_to_rotation,
    backends,
    ball_query,
    cylinder_query,
    furthest_point_sample,
    knn,
)

KINDS = ("random", "scene", "lattice", "repeats", "origin", "nonfinite")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def compiled_kernels(monkeypatch):
    monkeypatch.delenv(backends.ENV, raising=False)
    kernels, detail = library.kernels()
    if kernels is None:
        pytest.skip(f"no compiled kernels on this machine: {detail}")
    return kernels


def make_cloud(generator: torch.Generator, batch: int, count: int, kind: str) -> Tensor:
    """``(batch, count, 3)`` float32 points of a kind that stresses a different rule."""
    points = torch.rand(batch, count, 3, generator=generator)
    if kind == "scene":  # a table top with a few objects, in metres, half a metre away
        points[..., 2] = 0.5 + 0.002 * points[..., 2]
        blob = torch.randint(0, 4, (batch, count), generator=generator)
        centre = torch.rand(batch, 4, 3, generator=generator) * torch.tensor([0.6, 0.6, 0.0])
        offset = 0.04 * torch.randn(batch, count, 3, generator=generator)
        on_object = torch.rand(batch, count, generator=generator) < 0.5
        lifted = torch.gather(centre, 1, blob[..., None].expand(-1, -1, 3)) + offset
        lifted[..., 2] = 0.45 + lifted[..., 2].abs()
        points = torch.where(on_object[..., None], lifted, points * torch.tensor([0.6, 0.6, 1.0]))
    elif kind == "lattice":  # few distinct coordinates: exact ties and repeated points
        points = torch.round(points * 4) / 8
    elif kind == "repeats":  # as a frame padded with repeats of its first candidate
        points[:, count // 2 :] = points[:, :1]
    elif kind == "origin":  # centred on the origin: several points inside the 1e-3 ball
        points = (points - 0.5) * 0.2
    elif kind == "nonfinite":
        bad = torch.rand(batch, count, generator=generator) < 0.05
        points[..., 0] = torch.where(bad, torch.tensor(float("nan")), points[..., 0])
        far = torch.rand(batch, count, generator=generator) < 0.05
        points[..., 1] = torch.where(far, torch.tensor(float("inf")), points[..., 1])
    return points


def identical(first: Tensor | tuple[Tensor, ...], second: Tensor | tuple[Tensor, ...]) -> bool:
    """Same type, shape and values; a NaN only matches a NaN."""
    first = first if isinstance(first, tuple) else (first,)
    second = second if isinstance(second, tuple) else (second,)
    return len(first) == len(second) and all(
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.device == b.device
        and bool(((a == b) | ((a != a) & (b != b))).all())
        for a, b in zip(first, second)
    )


def on_both(operator: str, function: Callable, *arguments, **options) -> bool:
    """Call ``function`` on the reference and on the kernels; are the results identical?"""
    with backends.override(**{operator: "reference"}):
        expected = function(*arguments, **options)
    with backends.override(**{operator: "compiled"}):
        got = function(*arguments, **options)
    return identical(got, expected)


def pick(generator: torch.Generator, options):
    return options[int(torch.randint(len(options), (1,), generator=generator))]


# ------------------------------------------------------------------- comparisons


def compare_fps(device: torch.device, seeds: range) -> int:
    """Both tie rules, with and without the near-origin rule; returns the number of calls."""
    calls = 0
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        kind = KINDS[seed % len(KINDS)]
        batch = pick(generator, (1, 1, 2, 4))
        count = pick(generator, (1, 2, 3, 31, 32, 33, 64, 200, 511, 512, 513, 1100))
        points = make_cloud(generator, batch, count, kind).to(device)
        wanted = min(count, pick(generator, (1, 16, 48, 80)))
        options = [
            (skip, tie) for skip in (False, True) for tie in ("lowest_index", "pointnet2_cuda")
        ]
        # Every combination on the batch; one of them also on a single cloud and on a
        # tensor that is not contiguous.
        runs = [(points, *option) for option in options]
        runs += [(points[0], *pick(generator, options))]
        runs += [(points.flip(1)[:, ::2], *pick(generator, options))]
        for cloud, skip, tie_break in runs:
            samples = min(wanted, cloud.shape[-2])
            assert on_both(
                "fps",
                furthest_point_sample,
                cloud,
                samples,
                skip_near_origin=skip,
                tie_break=tie_break,
            ), (seed, kind, tuple(cloud.shape), samples, skip, tie_break)
            calls += 1
    return calls


def query_case(generator: torch.Generator, seed: int, device: torch.device):
    kind = KINDS[seed % len(KINDS)]
    batch = pick(generator, (1, 2, 3))
    count = pick(generator, (1, 2, 3, 40, 257, 1000, 2500))
    queries = pick(generator, (1, 5, 33, 120))
    support = make_cloud(generator, batch, count, kind)
    query = make_cloud(generator, batch, queries, kind)
    own = min(count, queries) // 2
    query[:, :own] = support[:, :own]  # centres that are support points themselves
    query[:, -1] += 9.0  # and one that has no neighbour
    return kind, support.to(device), query.to(device)


def compare_ball_query(device: torch.device, seeds: range) -> int:
    calls = 0
    for seed in seeds:
        generator = torch.Generator().manual_seed(1000 + seed)
        kind, support, query = query_case(generator, seed, device)
        # 0.125 and 0.25 are lattice distances: points exactly on the sphere.
        radius = pick(generator, (0.01, 0.05, 0.125, 0.25, 0.9))
        nsample = pick(generator, (1, 2, 16, 32, 33, 64, 300))
        for q, s in ((query, support), (query.flip(1), support[:, ::2])):
            assert on_both("ball_query", ball_query, q, s, radius, nsample), (
                *(seed, kind, tuple(q.shape), tuple(s.shape), radius, nsample),
            )
            calls += 1
    return calls


def compare_knn(device: torch.device, seeds: range) -> int:
    calls = 0
    for seed in seeds:
        generator = torch.Generator().manual_seed(2000 + seed)
        kind, support, query = query_case(generator, seed, device)
        count = support.shape[1]
        for k in sorted({1, min(3, count), min(pick(generator, (8, 16, 64)), count)}):
            assert on_both("knn", knn, query, support, k), (seed, kind, count, k)
            thinned = support[:, ::2]  # not contiguous
            assert on_both("knn", knn, query.flip(1), thinned, min(k, thinned.shape[1]))
            calls += 2
    return calls


def surface_points(
    generator: torch.Generator,
    query: Tensor,
    rotation: Tensor,
    radius: float,
    hmin: float,
    hmax: float,
) -> Tensor:
    """Support points on, and a few units in the last place off, the cylinders of the queries.

    For every query one ring on the lateral surface and one disc on each cap, scaled by
    ``1 + e`` with ``|e|`` between 0 and 4e-6: exactly the pairs for which two correct
    float32 evaluations of the membership test can disagree.
    """
    batch, queries, _ = query.shape
    per_query = 6
    scale = 1.0 + torch.tensor([0.0, 1.2e-7, -1.2e-7, 1e-6, -1e-6, 4e-6])
    angle = torch.rand(batch, queries, per_query, generator=generator) * 2 * math.pi
    where = torch.randint(0, 3, (batch, queries, per_query), generator=generator)
    inner = torch.rand(batch, queries, per_query, generator=generator)
    radial = torch.where(where == 0, radius * scale, radius * inner)
    axial = torch.where(
        where == 0,
        hmin + (hmax - hmin) * inner,
        torch.where(where == 1, hmin * scale, hmax * scale),
    )
    local = torch.stack([axial, radial * torch.cos(angle), radial * torch.sin(angle)], dim=-1)
    # world = centre + R · local (local = Rᵀ (p - c) for an orthonormal frame)
    world = query[:, :, None, :] + torch.einsum("bmjk,bmnk->bmnj", rotation, local)
    return world.reshape(batch, queries * per_query, 3)


def compare_cylinder_query(
    device: torch.device, seeds: range, kernels=None
) -> tuple[int, int, int]:
    """Returns ``(calls, queries, queries the kernel handed back to the reference)``."""
    calls = total = handed_back = 0
    for seed in seeds:
        generator = torch.Generator().manual_seed(3000 + seed)
        kind, support, query = query_case(generator, seed, device)
        batch, queries, _ = query.shape
        approach = torch.randn(batch, queries, 3, generator=generator)
        approach[:, ::3] = torch.tensor([0.0, 0.0, 1.0])  # frames with entries 0 and ±1
        rotation = approach_to_rotation(approach, torch.rand(batch, queries, generator=generator))
        rotation[:, ::3] = approach_to_rotation(approach[:, ::3])
        if seed % 5 == 0:  # the operator takes any matrix, not only rotations
            rotation = rotation + 0.3 * torch.randn(batch, queries, 3, 3, generator=generator)
        radius = pick(generator, (0.02, 0.05, 0.125, 0.25))
        hmin, hmax = pick(generator, ((-0.02, 0.04), (0.0, 0.125), (-0.125, 0.25), (-0.5, -0.1)))
        nsample = pick(generator, (1, 2, 16, 33, 64, 300))
        if seed % 2 == 0 and kind != "nonfinite":
            extra = surface_points(generator, query.cpu(), rotation, radius, hmin, hmax)
            order = torch.randperm(support.shape[1] + extra.shape[1], generator=generator)
            support = torch.cat([support, extra.to(device)], dim=1)[:, order.to(device)]
        rotation = rotation.to(device)
        arguments = (query, support, rotation, radius, hmin, hmax, nsample)
        assert on_both("cylinder_query", cylinder_query, *arguments), (
            *(seed, kind, tuple(query.shape), tuple(support.shape)),
            *(radius, hmin, hmax, nsample),
        )
        calls += 1
        if kernels is not None:
            _, hits = kernels.cylinder_query(
                query, support, rotation, radius**2, hmin, hmax, nsample
            )
            total += hits.numel()
            handed_back += int((hits < 0).sum())
    return calls, total, handed_back


def compare_packed(device: torch.device, seeds: range) -> int:
    """Batches of clouds of different sizes, through the helpers the backbones use."""
    calls = 0
    for seed in seeds:
        generator = torch.Generator().manual_seed(4000 + seed)
        kind = KINDS[seed % 5]
        sizes = [pick(generator, (40, 41, 150, 700)) for _ in range(pick(generator, (2, 3, 4)))]
        coords = torch.cat([make_cloud(generator, 1, size, kind)[0] for size in sizes]).to(device)
        count = pick(generator, (8, 32, 40))
        selection = dict.fromkeys(backends.OPERATORS, "reference")
        with backends.override(**selection):
            expected = _packed_pipeline(coords, sizes, count)
        with backends.override(**dict.fromkeys(backends.OPERATORS, "compiled")):
            got = _packed_pipeline(coords, sizes, count)
        assert identical(got, expected), (seed, kind, sizes, count)
        calls += 1
    return calls


def _packed_pipeline(coords: Tensor, sizes: list[int], count: int) -> tuple[Tensor, ...]:
    """Sample, group and interpolate like one PointNet++ level."""
    chosen = fps_packed(coords, sizes, count, skip_near_origin=True)
    centres = coords[chosen]
    neighbours = ball_query_packed(centres, [count] * len(sizes), coords, sizes, 0.125, 16)
    distance, index = knn_packed(coords, sizes, centres, [count] * len(sizes), 3)
    return chosen, neighbours, distance, index


# ---------------------------------------------------------------------- CPU tests


def test_fps_is_identical_on_the_cpu():
    assert compare_fps(CPU, range(150)) == 900


def test_ball_query_is_identical_on_the_cpu():
    assert compare_ball_query(CPU, range(300)) == 600


def test_knn_is_identical_on_the_cpu():
    assert compare_knn(CPU, range(150)) >= 300


def test_cylinder_query_is_identical_on_the_cpu(compiled_kernels):
    calls, queries, handed_back = compare_cylinder_query(CPU, range(400), compiled_kernels)
    assert calls == 400
    # The inputs must exercise both paths: queries the kernel decides, and queries with
    # a point within round-off of the surface that it hands back to the reference.
    assert 0 < handed_back < queries


def test_ragged_batches_are_identical_on_the_cpu():
    assert compare_packed(CPU, range(60)) == 60


def test_cylinder_kernel_decides_almost_every_query_of_a_scene(compiled_kernels):
    """On a realistic cloud the hand-back is the exception (it costs a reference run)."""
    generator = torch.Generator().manual_seed(7)
    support = make_cloud(generator, 2, 6000, "scene")
    query = support[:, :512].contiguous()
    rotation = approach_to_rotation(torch.randn(2, 512, 3, generator=generator))
    _, hits = compiled_kernels.cylinder_query(query, support, rotation, 0.05**2, -0.02, 0.04, 64)
    assert int((hits < 0).sum()) <= 0.02 * hits.numel()
    assert on_both(
        "cylinder_query", cylinder_query, query, support, rotation, 0.05, -0.02, 0.04, 64
    )
