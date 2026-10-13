"""The compiled CUDA kernels against the pure-PyTorch reference running on the same GPU.

The randomized comparisons are those of ``tests/unit/test_ops_equivalence.py`` (ties,
repeated points, empty neighbourhoods, points within round-off of a surface, tiny and
ragged inputs, NaN and infinity), with more seeds, plus inputs of the size the models
use: 1024 samples or queries on clouds of 15 000 to 20 000 points.

Skipped without a CUDA device or without compiled CUDA kernels.
"""

from __future__ import annotations

import pytest
import torch

from grasppanda._ops import library
from grasppanda.ops import (
    approach_to_rotation,
    backends,
    ball_query,
    cylinder_query,
    furthest_point_sample,
    knn,
)

from ..unit.test_ops_equivalence import (
    compare_ball_query,
    compare_cylinder_query,
    compare_fps,
    compare_knn,
    compare_packed,
    make_cloud,
    on_both,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def kernels(cuda):
    loaded, detail = library.kernels()
    if loaded is None or not loaded.cuda:
        pytest.skip(f"no compiled CUDA kernels on this machine: {detail}")
    return loaded


@pytest.fixture(autouse=True)
def default_selection(monkeypatch, kernels):
    monkeypatch.delenv(backends.ENV, raising=False)


def test_every_operator_settles_on_the_kernels(cuda):
    report = backends.report()
    assert {report[name]["cuda"] for name in backends.OPERATORS} == {"compiled"}, report
    assert report["library"]["cuda"]


def test_fps_is_identical(cuda):
    assert compare_fps(cuda, range(300)) == 1800


def test_ball_query_is_identical(cuda):
    assert compare_ball_query(cuda, range(400)) == 800


def test_knn_is_identical(cuda):
    assert compare_knn(cuda, range(300)) >= 600


def test_cylinder_query_is_identical(cuda, kernels):
    calls, queries, handed_back = compare_cylinder_query(cuda, range(600), kernels)
    assert calls == 600
    assert 0 < handed_back < queries  # both the kernel's verdicts and the hand-back are used


def test_ragged_batches_are_identical(cuda):
    assert compare_packed(cuda, range(80)) == 80


@pytest.mark.parametrize("batch", [1, 4])
def test_operators_are_identical_at_model_scale(cuda, kernels, batch):
    """1024 seeds from a 15 000-point frame; 1024 queries on 20 000 points."""
    generator = torch.Generator().manual_seed(batch)
    for kind, count in (("scene", 15000), ("lattice", 3000), ("scene", 20000)):
        points = make_cloud(generator, batch, count, kind).to(cuda)
        for skip, tie_break in ((False, "lowest_index"), (True, "pointnet2_cuda")):
            assert on_both(
                "fps",
                furthest_point_sample,
                points,
                1024,
                skip_near_origin=skip,
                tie_break=tie_break,
            ), (kind, count, tie_break)
        centres = points[:, :: count // 1024][:, :1024].contiguous()
        rotation = approach_to_rotation(torch.randn(batch, 1024, 3, generator=generator)).to(cuda)
        for nsample in (16, 64):
            assert on_both("ball_query", ball_query, centres, points, 0.05, nsample), (kind, count)
            assert on_both(
                "cylinder_query",
                cylinder_query,
                centres,
                points,
                rotation,
                0.05,
                -0.02,
                0.04,
                nsample,
            ), (kind, count, nsample)
        assert on_both("knn", knn, points, centres, 3), (kind, count)
        assert on_both("knn", knn, centres, points, 1), (kind, count)
    # On a scene the kernel answers nearly every cylinder itself.
    _, hits = kernels.cylinder_query(centres, points, rotation, 0.05**2, -0.02, 0.04, 64)
    assert int((hits < 0).sum()) <= 0.02 * hits.numel()


def test_fps_picks_the_same_points_on_cpu_and_gpu(cuda):
    """The operator is defined to be device independent; so are both of its backends."""
    generator = torch.Generator().manual_seed(11)
    for kind in ("scene", "lattice", "origin"):
        points = make_cloud(generator, 2, 4000, kind)
        for backend in ("reference", "compiled"):
            with backends.override(fps=backend):
                options = {"skip_near_origin": True, "tie_break": "pointnet2_cuda"}
                on_cpu = furthest_point_sample(points, 256, **options)
                on_gpu = furthest_point_sample(points.to(cuda), 256, **options)
            assert torch.equal(on_cpu, on_gpu.cpu()), (kind, backend)


def test_operators_are_identical_under_autocast(cuda):
    """Mixed precision changes none of the operators' arithmetic except one matrix product.

    Sampling, the ball query and kNN stay in float32 under autocast, on both backends.
    The cylinder query of the reference rotates in half precision there, so the kernel
    is not used and the two selections return the same (reduced-precision) result.
    """
    generator = torch.Generator().manual_seed(13)
    points = make_cloud(generator, 2, 3000, "scene").to(cuda)
    centres = points[:, :256].contiguous()
    rotation = approach_to_rotation(torch.randn(2, 256, 3, generator=generator)).to(cuda)
    with torch.autocast("cuda"):
        assert on_both("fps", furthest_point_sample, points, 200, skip_near_origin=True)
        assert on_both("ball_query", ball_query, centres, points, 0.05, 32)
        assert on_both("knn", knn, centres, points, 3)
        assert on_both(
            "cylinder_query", cylinder_query, centres, points, rotation, 0.05, -0.02, 0.04, 32
        )
    with backends.override(fps="compiled"):
        plain = furthest_point_sample(points, 200, skip_near_origin=True)
        with torch.autocast("cuda"):
            assert torch.equal(furthest_point_sample(points, 200, skip_near_origin=True), plain)


def test_kernels_run_on_the_current_stream(cuda):
    generator = torch.Generator().manual_seed(12)
    points = make_cloud(generator, 2, 5000, "scene").to(cuda)
    with backends.override(fps="reference", knn="reference"):
        expected = furthest_point_sample(points, 128), knn(points[:, :64], points, 4)
    side = torch.cuda.Stream(cuda)
    with torch.cuda.stream(side):
        moved = points * 1.0  # queued on the side stream just before the kernels
        got = furthest_point_sample(moved, 128), knn(moved[:, :64], moved, 4)
    side.synchronize()  # as for any result that is read from another stream
    assert torch.equal(got[0], expected[0])
    assert all(torch.equal(a, b) for a, b in zip(got[1], expected[1]))
