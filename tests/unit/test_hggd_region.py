"""HGGD's region stage: features on points, lifting, ball regions and their sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.detectors.hggd import (
    PartialGraspSet,
    RegionDraws,
    RegionOptions,
    RegionStage,
)
from grasppanda.models.detectors.hggd.region import attach_features, cell_points
from grasppanda.structures import ContractError

K = torch.tensor([[100.0, 0.0, 63.5], [0.0, 101.0, 39.5], [0.0, 0.0, 1.0]])
SIZE = (64, 40)  # network image; the sensor image is 128 x 80


def stage(**options) -> RegionStage:
    return RegionStage(RegionOptions(**options))


def grasps(pixels, theta=0.0, width=20.0, depth_offset=0.0) -> PartialGraspSet:
    count = len(pixels)
    return PartialGraspSet(
        frame=torch.zeros(count, dtype=torch.long),
        pixel=torch.tensor(pixels, dtype=torch.long).reshape(count, 2),
        theta=torch.full((count,), float(theta), dtype=torch.float64),
        width=torch.full((count,), float(width), dtype=torch.float64),
        depth_offset=torch.full((count,), float(depth_offset), dtype=torch.float64),
        score=torch.linspace(0.9, 0.5, count, dtype=torch.float64),
    )


def back_project(u: float, v: float, z: float) -> list[float]:
    return [z * (u - 63.5) / 100.0, z * (v - 39.5) / 101.0, z]


# ------------------------------------------------------------------- image features


def test_cells_are_back_projected_at_their_corner_pixel():
    depth = torch.arange(80 * 128, dtype=torch.float32).reshape(80, 128) / 10000 + 0.3
    depth[16, 32] = 0.0  # the corner pixel of cell (2, 1) has no depth
    cells = cell_points(depth, K, (8, 5))
    assert tuple(cells.shape) == (8, 5, 3)
    for i, j in ((0, 0), (3, 2), (7, 4)):
        expected = back_project(16 * i, 16 * j, float(depth[16 * j, 16 * i]))
        assert np.allclose(cells[i, j].numpy(), expected, atol=1e-6)
    assert cells[2, 1].tolist() == [0.0, 0.0, 0.0]  # lands on the camera centre


def test_points_pool_the_maximum_over_their_nearest_cells():
    generator = torch.Generator().manual_seed(0)
    cells = torch.rand(6, 4, 3, generator=generator)
    features = torch.randn(5, 6, 4, generator=generator)
    points = torch.rand(40, 3, generator=generator)
    pooled = attach_features(points, cells, features, 3)
    flat_cells, flat_feats = cells.reshape(-1, 3), features.reshape(5, -1).T
    for row in (0, 7, 39):
        nearest = (flat_cells - points[row]).norm(dim=1).argsort()[:3]
        assert torch.equal(pooled[row], flat_feats[nearest].amax(dim=0))
    assert tuple(pooled.shape) == (40, 5)
    single = attach_features(points, cells, features, 1)
    nearest = torch.cdist(points, flat_cells).argmin(dim=1)
    assert torch.equal(single, flat_feats[nearest])


# -------------------------------------------------------------------------- lifting


def test_centres_are_the_pixel_at_sensor_depth_plus_offset():
    depth = torch.full((80, 128), 0.5)
    depth[20, 40] = 0.62
    proposals = grasps([[20, 10], [50, 30]], depth_offset=0.01)
    centre = stage().lift(proposals, depth, K, SIZE)
    assert centre.dtype == torch.float64
    assert np.allclose(centre[0].numpy(), back_project(40, 20, 0.63), atol=1e-7)  # pixel * 2
    assert np.allclose(centre[1].numpy(), back_project(100, 60, 0.51), atol=1e-7)


def test_a_pixel_without_depth_borrows_the_median_around_it():
    depth = torch.full((80, 128), 0.5)
    depth[16:24, 36:44] = 0.0  # a hole around sensor pixel (40, 20) ...
    depth[17, 37], depth[18, 38], depth[19, 39] = 0.40, 0.44, 0.48  # ... with three readings
    proposals = grasps([[20, 10]], depth_offset=-0.005)
    centre = stage().lift(proposals, depth, K, SIZE)
    assert np.allclose(centre[0].numpy(), back_project(40, 20, 0.44 - 0.005), atol=1e-6)
    # no reading at all in the window: the image mean, without the predicted offset
    depth[16:24, 36:44] = 0.0
    alone = stage().lift(proposals, depth, K, SIZE)
    assert abs(float(alone[0, 2]) - float(depth.mean())) < 1e-6
    # the window ends one pixel short of the image border
    edge = torch.zeros(80, 128)
    edge[:, 127] = 0.7
    lifted = stage(zero_depth_window=4).lift(grasps([[62, 10]]), edge, K, SIZE)
    assert abs(float(lifted[0, 2]) - float(edge.mean())) < 1e-6  # column 127 is never read


def test_reference_order_moves_centres_without_depth_to_the_end():
    depth = torch.full((80, 128), 0.5)
    depth[20, 40] = 0.0  # the first proposal sits on a pixel without depth
    depth[19, 39] = 0.6
    proposals = grasps([[20, 10], [50, 30], [10, 5]])
    appended = stage(zero_depth_centres="append").lift(proposals, depth, K, SIZE)
    in_place = stage(zero_depth_centres="in_place").lift(proposals, depth, K, SIZE)
    assert torch.allclose(
        in_place[0], torch.tensor(back_project(40, 20, 0.5), dtype=torch.float64), atol=1e-6
    )
    assert torch.allclose(appended, in_place[[1, 2, 0]])  # attributes keep their rows: off by one
    with pytest.raises(ValueError, match="unknown zero_depth_centres 'drop'"):
        stage(zero_depth_centres="drop")


def test_radius_is_the_metric_jaw_opening_at_the_centre():
    proposals = grasps([[20, 10], [20, 10]], theta=0.0, width=20.0)
    proposals.theta[1] = math.pi / 2
    centre = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]], dtype=torch.float64)
    radius = stage().radius(proposals, centre, K, 2.0)
    assert radius.dtype == torch.float32
    assert abs(float(radius[0]) - 0.5 * 40 / 100.0) < 1e-6  # 40 sensor pixels along u, fx = 100
    assert abs(float(radius[1]) - 0.8 * 40 / 101.0) < 1e-6  # along v, fy = 101


# -------------------------------------------------------------------------- regions


def scene_with(counts: list[int], centres: torch.Tensor, spread: float, channels: int = 4):
    generator = torch.Generator().manual_seed(1)
    clouds = []
    for count, centre in zip(counts, centres):
        direction = torch.nn.functional.normalize(torch.randn(count, 3, generator=generator), dim=1)
        clouds.append(
            centre.float() + direction * spread * torch.rand(count, 1, generator=generator)
        )
    points = torch.cat(clouds)
    return torch.cat([points, torch.randn(len(points), channels, generator=generator)], dim=1)


def brute_sample(points: np.ndarray, length: int, start: int, count: int) -> list[int]:
    chosen = [start]
    distance = np.full(length, np.inf)
    for _ in range(count - 1):
        delta = points[:length] - points[chosen[-1]]
        distance = np.minimum(distance, (delta * delta).sum(axis=1))
        chosen.append(int(np.argmax(distance)))
    return chosen


def test_sampling_starts_where_told_and_ignores_slots_past_the_length():
    generator = torch.Generator().manual_seed(2)
    points = torch.rand(3, 40, 3, generator=generator)
    lengths = torch.tensor([40, 25, 12])
    start = torch.tensor([0, 17, 5])
    index = stage(num_points=10).sample(points, lengths, start)
    assert tuple(index.shape) == (3, 10) and index[:, 0].tolist() == start.tolist()
    for row in range(3):
        assert index[row].tolist() == brute_sample(
            points[row].numpy(), int(lengths[row]), int(start[row]), 10
        )
        assert int(index[row].max()) < int(lengths[row]) and len(set(index[row].tolist())) == 10
    everything = stage(num_points=12).sample(points[2:], lengths[2:], start[2:])
    assert sorted(everything[0].tolist()) == list(range(12))  # a full region is a permutation


def test_regions_are_balls_relative_to_their_centre():
    centres = torch.tensor(
        [[0.0, 0.0, 0.5], [0.3, 0.0, 0.5], [-0.3, 0.1, 0.6]], dtype=torch.float64
    )
    scene = scene_with([60, 20, 5], centres, 0.02)
    grouper = stage(num_points=16, min_points=8, jitter=0.0)
    radius = torch.tensor([0.03, 0.03, 0.03])
    regions, kept, support = grouper.group(scene, centres, radius)
    assert support.tolist() == [60, 20, 5] and kept.tolist() == [True, True, False]
    assert tuple(regions.shape) == (2, 16, 7)  # the region with 5 points is dropped
    members = (scene[:, :3].double() - centres[0]).norm(dim=1) < 0.03
    inside = scene[members]
    relative = (inside[:, :3].double() - centres[0]).float()
    for point in regions[0]:
        match = (relative - point[:3]).abs().amax(dim=1) < 1e-7  # a point of the ball, re-centred
        assert match.any() and torch.equal(inside[match][0, 3:], point[3:])  # features travel along
    assert float(regions[..., :3].norm(dim=-1).max()) < 0.03
    assert len(torch.unique(regions[0], dim=0)) == 16  # 60 points: 16 distinct samples
    empty = grouper.group(scene, centres[:0], radius[:0])
    assert tuple(empty[0].shape) == (0, 16, 7) and len(empty[1]) == 0


def test_small_regions_are_resampled_with_replacement_and_jittered():
    centres = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
    scene = scene_with([12], centres, 0.02)
    grouper = stage(num_points=16, min_points=8, jitter=5e-4)
    torch.manual_seed(0)
    regions, kept, _ = grouper.group(scene, centres, torch.tensor([0.03]))
    assert kept.tolist() == [True] and tuple(regions.shape) == (1, 16, 7)
    assert len(torch.unique(regions[0, :, 3:], dim=0)) <= 12  # copies of at most 12 points
    assert len(torch.unique(regions[0, :, :3], dim=0)) == 16  # which the jitter separates
    at_limit = stage(num_points=16, min_points=12).group(scene, centres, torch.tensor([0.03]))
    assert at_limit[1].tolist() == [False]  # needs MORE than min_points


def test_supplied_draws_make_the_stage_deterministic_and_are_checked():
    centres = torch.tensor([[0.0, 0.0, 0.5], [0.3, 0.0, 0.5]], dtype=torch.float64)
    scene = scene_with([40, 10], centres, 0.02)
    grouper = stage(num_points=16, min_points=8)
    radius = torch.tensor([0.03, 0.03])
    draws = RegionDraws(
        resample=[torch.arange(16) % 10],
        jitter=torch.zeros(2, 40, 3),
        start=torch.tensor([3, 0]),
    )
    first, _, _ = grouper.group(scene, centres, radius, draws)
    again, _, _ = grouper.group(scene, centres, radius, draws)
    assert torch.equal(first, again)
    members = torch.nonzero((scene[:, :3].double() - centres[1]).norm(dim=1) < 0.03)[:, 0]
    assert {tuple(row.tolist()) for row in first[1, :, 3:]} == {
        tuple(row.tolist()) for row in scene[members[:10], 3:]
    }
    with pytest.raises(ContractError, match="too few re-sampling index sets"):
        grouper.group(scene, centres, radius, RegionDraws([], draws.jitter, draws.start))
    with pytest.raises(ContractError, match="draws are for"):
        grouper.group(scene, centres, radius, RegionDraws([], torch.zeros(2, 7, 3), draws.start))
    drawn = grouper.draw(torch.tensor([40, 10, 3]), 40, torch.device("cpu"))
    assert len(drawn.resample) == 1 and tuple(drawn.jitter.shape) == (3, 40, 3)
    assert int(drawn.start[0]) < 40 and int(drawn.start[1]) < 16 and int(drawn.start[2]) < 16


def test_batch_keeps_proposals_and_regions_aligned_per_frame():
    grouper = stage(num_points=16, min_points=4, feature_neighbours=2)
    depth = torch.full((2, 80, 128), 0.5)
    intrinsics = K[None].repeat(2, 1, 1)
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(2, 6, 8, 5, generator=generator)
    proposals = PartialGraspSet.cat(
        [grasps([[20, 10], [50, 30]]), grasps([[30, 20]])], num_frames=2
    )
    proposals.frame[2] = 1
    clouds = []
    for frame, pixels in enumerate(([[40, 20], [100, 60]], [[60, 40]])):
        centres = torch.tensor([back_project(u, v, 0.5) for u, v in pixels], dtype=torch.float64)
        count = [30, 2] if frame == 0 else [30]  # the second proposal of frame 0 has no support
        clouds.append(scene_with(count, centres, 0.01)[:, :3])
    regions = grouper(proposals, clouds, features, depth, intrinsics, SIZE)
    assert len(regions) == 2 and regions.kept.tolist() == [True, False, True]
    assert regions.grasps.frame.tolist() == [0, 1] and regions.grasps.centre is not None
    assert regions.grasps.pixel.tolist() == [[20, 10], [30, 20]]
    assert tuple(regions.points.shape) == (2, 16, 3 + 6) and regions.support.tolist()[1] == 2
    with pytest.raises(ContractError, match="1 clouds, 2 depth images"):
        grouper(proposals, clouds[:1], features, depth, intrinsics, SIZE)
    with pytest.raises(ContractError, match="aspect ratio"):
        grouper(proposals, clouds, features, depth, intrinsics, (64, 64))
