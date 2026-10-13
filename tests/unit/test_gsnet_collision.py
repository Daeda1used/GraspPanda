"""ModelFreeCollisionFilter against a point-by-point NumPy oracle, and its misuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.config import Config, ConfigError
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import (
    ContractError,
    GraspGroup,
    Gripper,
    Observation,
    PointSet,
    SampleKey,
)

KEYS = [SampleKey("unit", 0, "none", frame) for frame in range(2)]


def random_grasps(count: int, seed: int = 0, **meta) -> GraspGroup:
    generator = torch.Generator().manual_seed(seed)
    return GraspGroup(
        translation=torch.rand(count, 3, generator=generator) * 0.2 + 0.4,
        rotation=approach_to_rotation(
            torch.randn(count, 3, generator=generator),
            torch.rand(count, generator=generator) * math.pi,
        ),
        width=torch.rand(count, generator=generator) * 0.1,
        depth=(torch.randint(0, 4, (count,), generator=generator) + 1) * 0.01,
        score=torch.rand(count, generator=generator),
        **meta,
    )


def scene_cloud(count: int = 300, seed: int = 1) -> torch.Tensor:
    return torch.rand(count, 3, generator=torch.Generator().manual_seed(seed)).double() * 0.3 + 0.35


def downsample_oracle(points: np.ndarray, voxel_size: float) -> np.ndarray:
    origin = points.min(axis=0) - voxel_size * 0.5
    cells: dict[tuple[int, ...], list[np.ndarray]] = {}
    for point in points:
        cells.setdefault(tuple(np.floor((point - origin) / voxel_size).astype(int)), []).append(
            point
        )
    return np.array([np.mean(members, axis=0) for members in cells.values()])


def collision_oracle(grasps, scene, voxel_size, approach, threshold, finger_length=0.06):
    """The gripper box model, one grasp and one point at a time."""
    finger_width = grasps.gripper.finger_width
    approach = max(approach, finger_width)
    result = []
    for g in range(len(grasps)):
        rotation = grasps.rotation[g].double().numpy()
        local = (scene - grasps.translation[g].double().numpy()) @ rotation
        h, d, w = (float(getattr(grasps, name)[g]) for name in ("height", "depth", "width"))
        count = 0
        for x, y, z in local:
            if not -h / 2 < z < h / 2:
                continue
            in_finger = d - finger_length < x < d and w / 2 < abs(y) < w / 2 + finger_width
            across = abs(y) < w / 2 + finger_width
            in_palm = across and d - finger_length - finger_width < x <= d - finger_length
            behind = d - finger_length - finger_width
            in_sweep = across and behind - approach < x <= behind
            count += in_finger or in_palm or in_sweep
        cell = voxel_size**3
        volume = (
            2 * h * finger_length * finger_width
            + h * (w + 2 * finger_width) * finger_width
            + h * (w + 2 * finger_width) * approach
        ) / cell
        result.append(count / (volume + 1e-6) > threshold)
    return np.array(result)


def test_voxel_down_sample_returns_voxel_centroids_on_the_open3d_grid():
    cloud = scene_cloud(600)
    sampled = voxel_down_sample(cloud, 0.03)
    expected = downsample_oracle(cloud.numpy(), 0.03)
    assert sampled.dtype == torch.float64 and sampled.shape == expected.shape
    difference = (sampled[:, None, :] - torch.from_numpy(expected)[None, :, :]).abs().amax(dim=-1)
    assert difference.min(dim=1).values.max() < 1e-12  # every centroid has an exact partner
    assert voxel_down_sample(torch.zeros(0, 3), 0.01).shape == (0, 3)
    assert (
        voxel_down_sample(cloud.float(), 0.03).dtype == torch.float64
    )  # always computed in double
    with pytest.raises(ValueError, match=r"expected \(N, 3\)"):
        voxel_down_sample(torch.zeros(5, 2), 0.01)


@pytest.mark.parametrize(("voxel_size", "threshold"), [(0.01, 0.01), (0.02, 0.05)])
def test_collision_mask_matches_the_box_model(voxel_size, threshold):
    collision = ModelFreeCollisionFilter(voxel_size=voxel_size, collision_threshold=threshold)
    grasps = random_grasps(40)
    scene = voxel_down_sample(scene_cloud(), voxel_size)
    mask = collision.collision_mask(grasps, scene)
    expected = collision_oracle(grasps, scene.numpy(), voxel_size, 0.05, threshold)
    assert np.array_equal(mask.numpy(), expected)
    assert 0 < mask.sum() < len(grasps)  # both outcomes occur


def test_the_filter_drops_colliding_grasps_of_the_right_frame():
    collision = POSTPROCESSORS.build({"type": "ModelFreeCollisionFilter"})
    assert (collision.voxel_size, collision.approach_distance, collision.collision_threshold) == (
        0.01,
        0.05,
        0.01,
    )
    empty_scene = PointSet(torch.full((50, 3), 5.0, dtype=torch.float64))  # nothing near any grasp
    cluttered = PointSet(scene_cloud())
    observation = Observation(keys=KEYS, full_points=PointSet.cat([empty_scene, cluttered]))
    in_empty, in_clutter = random_grasps(40, key=KEYS[0]), random_grasps(40, key=KEYS[1])
    assert len(collision(in_empty, observation)) == 40
    kept = collision(in_clutter, observation)
    scene = voxel_down_sample(cluttered.coords, 0.01)
    mask = collision.collision_mask(in_clutter, scene)
    assert 0 < len(kept) < 40 and len(kept) == int((~mask).sum())
    assert torch.equal(kept.translation, in_clutter.translation[~mask])
    assert kept.key == KEYS[1] and kept.score_kind == in_clutter.score_kind


def test_the_gripper_of_the_grasps_and_the_filters_own_voxel_size_matter():
    grasps = random_grasps(60, seed=2)
    scene = voxel_down_sample(scene_cloud(), 0.01)
    reference = ModelFreeCollisionFilter().collision_mask(grasps, scene)
    wide = random_grasps(60, seed=2, gripper=Gripper(finger_width=0.03))
    assert ModelFreeCollisionFilter().collision_mask(wide, scene).sum() > reference.sum()
    # A coarser voxel size means fewer voxels per gripper volume: the same count weighs more.
    coarse = ModelFreeCollisionFilter(voxel_size=0.02).collision_mask(grasps, scene)
    assert coarse.sum() > reference.sum()
    lenient = ModelFreeCollisionFilter(collision_threshold=0.2).collision_mask(grasps, scene)
    assert lenient.sum() < reference.sum()


def test_missing_or_ambiguous_inputs_are_refused():
    collision = ModelFreeCollisionFilter()
    grasps = random_grasps(5, key=KEYS[0])
    with pytest.raises(ContractError, match="reads observation.full_points"):
        collision(grasps, Observation(keys=KEYS[:1], points=PointSet(scene_cloud(100).float())))
    both = Observation(keys=KEYS, full_points=PointSet.from_dense(torch.rand(2, 50, 3)))
    with pytest.raises(ContractError, match="cannot tell which of the 2 frames"):
        collision(random_grasps(5), both)
    stranger = random_grasps(5, key=SampleKey("unit", 9, "none", 9))
    with pytest.raises(ContractError, match="does not contain frame"):
        collision(stranger, both)
    table = Observation(keys=KEYS[:1], full_points=PointSet(torch.rand(50, 3), frame="table"))
    with pytest.raises(ContractError, match="grasps are in frame 'camera'"):
        collision(grasps, table)
    with pytest.raises(ValueError, match="must be positive"):
        ModelFreeCollisionFilter(voxel_size=0.0)


def test_an_empty_prediction_passes_through():
    collision = ModelFreeCollisionFilter()
    empty = GraspGroup.empty(key=KEYS[0])
    assert collision(empty, Observation(keys=KEYS[:1])) is empty  # nothing to check, nothing read
    single = Observation(keys=KEYS[:1], full_points=PointSet(scene_cloud(200)))
    assert len(collision(random_grasps(3), single)) <= 3  # one frame: a key is not needed


def test_the_filter_is_configured_as_an_evaluation_variant():
    """Raw and collision-filtered predictions are scored side by side, never mixed."""
    variants = {"raw": [], "cd": [{"type": "ModelFreeCollisionFilter", "voxel_size": 0.01}]}
    config = Config.from_dict({"evaluation": {"variants": variants}})
    chain = config.resolved()["evaluation"]["variants"]["cd"]
    assert chain == [
        {
            "type": "ModelFreeCollisionFilter",
            "voxel_size": 0.01,
            "approach_distance": 0.05,
            "collision_threshold": 0.01,
            "finger_length": 0.06,
            # The options of other reference filters, at the values that leave this one as it is.
            "cloud": "full_points",
            "height": None,
            "depth": None,
            "min_points_between": None,
            "precision": "float64",
        }
    ]
    typo = {"cd": [{"type": "ModelFreeCollisionFilter", "voxel": 0.01}]}
    with pytest.raises(ConfigError, match="has no parameter 'voxel'. Did you mean 'voxel_size'"):
        Config.from_dict({"evaluation": {"variants": typo}})
