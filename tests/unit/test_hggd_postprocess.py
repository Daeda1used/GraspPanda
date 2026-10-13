"""The post-processing of HGGD's test script: its collision filter variant and NMS."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.config import Config, ConfigError
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.models.postprocessors.nms import GraspNMS
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import ContractError, GraspGroup, Observation, PointSet, SampleKey

KEY = SampleKey("graspnet1b", 100, "realsense", 0)
HGGD_FILTER = {
    "type": "ModelFreeCollisionFilter",
    "cloud": "points",
    "finger_length": 0.04,
    "height": 0.01,
    "depth": 0.02,
    "collision_threshold": 0.0,
    "min_points_between": 2,
    "precision": "float16",
}


def grasp(width: float = 0.06, **fields) -> GraspGroup:
    """One grasp at (0, 0, 0.5) approaching along +z, closing along +y of the camera."""
    rotation = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]])
    return GraspGroup(
        translation=torch.tensor([[0.0, 0.0, 0.5]]),
        rotation=rotation,
        width=torch.tensor([width]),
        depth=torch.tensor([fields.get("depth", 0.02)]),
        score=torch.tensor([0.9]),
        height=torch.tensor([fields.get("height", 0.02)]),
        key=KEY,
    )


def local(points: list[list[float]]) -> torch.Tensor:
    """Points given in the gripper frame (x approach, y closing, z height) → camera frame."""
    frame = grasp()
    return torch.tensor(points) @ frame.rotation[0].T + frame.translation[0]


#: Three points between the fingers, further apart than a voxel is wide.
BETWEEN = [[0.015, -0.012, 0.0], [0.015, 0.0, 0.0], [0.015, 0.012, 0.0]]


# ----------------------------------------------------------------- voxel arithmetic


def loop_voxels(points: np.ndarray, size: float) -> dict[tuple[int, int, int], np.ndarray]:
    """cupoch's rule point by point: float32, index by multiplying with the reciprocal."""
    points = points.astype(np.float32)
    size32 = np.float32(size)
    origin = points.min(axis=0) - size32 * np.float32(0.5)
    reciprocal = np.float32(1.0) / size32
    sums: dict[tuple[int, int, int], list] = {}
    for point in points:
        cell = tuple(np.floor((point - origin) * reciprocal).astype(np.int64).tolist())
        total, count = sums.setdefault(cell, [np.zeros(3, np.float32), 0])
        sums[cell] = [total + point, count + 1]
    return {cell: total / np.float32(count) for cell, (total, count) in sums.items()}


def test_cupoch_arithmetic_is_single_precision_with_a_reciprocal():
    generator = torch.Generator().manual_seed(0)
    # a depth-camera-like lattice: many points sit exactly on voxel faces
    points = torch.round(torch.rand(4000, 3, generator=generator) * 400) / 1000 + 0.3
    mine = voxel_down_sample(points, 0.01, arithmetic="cupoch")
    reference = loop_voxels(points.numpy(), 0.01)
    assert mine.dtype == torch.float32 and len(mine) == len(reference)
    ours = {tuple(row) for row in mine.numpy().round(7).tolist()}
    theirs = {tuple(row) for row in np.stack(list(reference.values())).round(7).tolist()}
    assert ours == theirs
    other = voxel_down_sample(points, 0.01)  # Open3D's arithmetic: double precision, a division
    assert other.dtype == torch.float64 and len(other) != len(mine)  # the faces fall differently
    assert len(voxel_down_sample(points[:0], 0.01, arithmetic="cupoch")) == 0
    with pytest.raises(ValueError, match="arithmetic must be 'open3d' or 'cupoch'"):
        voxel_down_sample(points, 0.01, arithmetic="pcl")


# ------------------------------------------------------------------- the box rules


def decide(points: list[list[float]], group: GraspGroup | None = None, **options) -> bool:
    """Whether the HGGD variant rejects the grasp against points given in the gripper frame.

    The points are tested as they are (no down-sampling), so each is one occupied voxel.
    """
    collision = POSTPROCESSORS.build({**HGGD_FILTER, **options})
    return bool(collision.collision_mask(group or grasp(), local(points))[0])


def test_a_single_point_in_a_finger_or_behind_the_palm_is_a_collision():
    assert not decide(BETWEEN)  # three points between the fingers, nothing else: a grasp
    assert decide([*BETWEEN, [0.0, 0.035, 0.0]])  # one point inside the right finger
    assert decide([*BETWEEN, [0.0, -0.035, 0.0]])  # the left finger
    assert decide([*BETWEEN, [-0.03, 0.0, 0.0]])  # behind the fingers: palm and approach sweep
    assert decide([*BETWEEN, [-0.07, 0.02, 0.0]])  # 5 cm sweep reaches to -0.08
    assert not decide([*BETWEEN, [-0.09, 0.0, 0.0]])  # beyond the sweep
    assert not decide([*BETWEEN, [0.0, 0.045, 0.0]])  # outside the finger (width/2 + 1 cm)
    assert not decide([*BETWEEN, [0.03, 0.035, 0.0]])  # past the finger tips (depth 0.02)
    assert not decide([*BETWEEN, [0.0, 0.035, 0.008]])  # above the 1 cm high gripper
    # The Graspness filter tolerates sparse contact: one occupied voxel in a gripper
    # volume of 120 voxels is below its 1 % threshold.
    lenient = ModelFreeCollisionFilter(cloud="points")
    scene = local([*BETWEEN, [0.0, 0.035, 0.0]])
    assert not bool(lenient.collision_mask(grasp(), scene)[0])
    assert bool(ModelFreeCollisionFilter(collision_threshold=0.0).collision_mask(grasp(), scene)[0])


def test_an_empty_grasp_is_rejected_too():
    assert decide(BETWEEN[:2])  # at most two points between the fingers: nothing to hold
    assert decide([[0.5, 0.5, 0.5]])
    assert not decide(BETWEEN[:2], min_points_between=1)
    assert not decide([[0.5, 0.5, 0.5]], min_points_between=None)  # the Graspness rule keeps it
    with pytest.raises(ValueError, match="min_points_between must not be negative"):
        ModelFreeCollisionFilter(min_points_between=-1)


def test_one_gripper_for_every_grasp_whatever_the_grasp_says():
    tall = grasp(height=0.05, depth=0.04)  # HGGD decodes heights of about 1 cm and depth 0.02
    above = [*BETWEEN, [0.0, 0.035, 0.008]]
    assert not decide(above, tall)  # fixed 1 cm gripper: the point is above it
    assert decide(above, tall, height=None)  # the grasp's own 5 cm: inside the finger
    deep = [*BETWEEN, [0.03, 0.035, 0.0]]
    assert not decide(deep, tall) and decide(deep, tall, depth=None)
    with pytest.raises(ValueError, match="height and depth must be positive"):
        ModelFreeCollisionFilter(height=0.0)


def test_half_floats_decide_points_on_a_box_face():
    """The reference rounds everything to float16; a point 0.1 mm off a face can flip."""
    # The outer face of the right finger is at width / 2 + 0.01 = 0.04 m, which is
    # 0.0400085 as a half float. A point 5 micrometres inside the face is inside the
    # finger in double precision; rounded to a half float it lands ON the face: outside.
    face = np.float16(0.06) / np.float16(2) + np.float16(0.01)
    assert np.float16(0.039995) == face and 0.039995 < 0.04 < float(face)
    grazing = [*BETWEEN, [0.0, 0.039995, 0.0]]
    assert decide(grazing, precision="float64") and not decide(grazing)
    clear = [*BETWEEN, [0.0, 0.0395, 0.0]]
    assert decide(clear, precision="float64") and decide(clear)  # well inside: both agree
    options = {k: v for k, v in HGGD_FILTER.items() if k != "type"}
    exact = ModelFreeCollisionFilter(**{**options, "precision": "float64"})
    half = POSTPROCESSORS.build(HGGD_FILTER)
    generator = torch.Generator().manual_seed(1)
    cloud = torch.rand(3000, 3, generator=generator) * torch.tensor([0.3, 0.3, 0.2]) + torch.tensor(
        [-0.15, -0.15, 0.4]
    )
    count = 200
    angles = torch.rand(count, generator=generator) * math.pi
    rotation = torch.zeros(count, 3, 3)
    rotation[:, 2, 0] = 1.0
    rotation[:, 0, 1], rotation[:, 1, 1] = -torch.cos(angles), torch.sin(angles)
    rotation[:, 0, 2], rotation[:, 1, 2] = -torch.sin(angles), -torch.cos(angles)
    group = GraspGroup(
        translation=cloud[:count].clone(),
        rotation=rotation,
        width=torch.rand(count, generator=generator) * 0.08 + 0.02,
        depth=torch.full((count,), 0.02),
        score=torch.rand(count, generator=generator),
        key=KEY,
    )
    a = half.collision_mask(group, half.down_sample(cloud))
    b = exact.collision_mask(group, exact.down_sample(cloud))
    assert a.dtype == torch.bool and tuple(a.shape) == (count,)
    assert 0 < int(a.sum()) < count
    assert (a == b).float().mean() > 0.8  # the same rule, differently rounded
    with pytest.raises(ValueError, match="precision must be 'float64' or 'float16'"):
        ModelFreeCollisionFilter(precision="float32")


def test_the_filter_reads_the_cloud_it_is_told_to():
    sampled = PointSet(local(BETWEEN))
    full = PointSet(local([*BETWEEN, [0.0, 0.035, 0.0]]))
    observation = Observation(keys=[KEY], points=sampled, full_points=full)
    on_points = POSTPROCESSORS.build(HGGD_FILTER)
    on_full = POSTPROCESSORS.build({**HGGD_FILTER, "cloud": "full_points"})
    assert len(on_points(grasp(), observation)) == 1  # the network's cloud: free
    assert len(on_full(grasp(), observation)) == 0  # the full cloud holds the colliding point
    with pytest.raises(ContractError, match="reads observation.points, which this observation"):
        on_points(grasp(), Observation(keys=[KEY], full_points=full))
    with pytest.raises(ValueError, match="cloud must be 'full_points' or 'points'"):
        ModelFreeCollisionFilter(cloud="scene")
    assert len(on_points(grasp()[[]], Observation(keys=[KEY]))) == 0  # nothing to check


def test_the_variant_is_expressed_in_a_config_and_typos_are_caught():
    variants = {"raw": [], "cd": [dict(HGGD_FILTER), {"type": "GraspNMS"}]}
    chain = Config.from_dict({"evaluation": {"variants": variants}}).resolved()
    chain = chain["evaluation"]["variants"]["cd"]
    assert chain[0]["voxel_size"] == 0.01 and chain[0]["approach_distance"] == 0.05
    assert chain[1] == {
        "type": "GraspNMS",
        "translation_threshold": 0.03,
        "rotation_threshold": 30.0,
    }
    with pytest.raises(ConfigError, match="precision"):
        Config.from_dict(
            {"evaluation": {"variants": {"cd": [{**HGGD_FILTER, "precision": "half"}]}}}
        )
    with pytest.raises(ConfigError, match="has no parameter 'rotation'"):
        Config.from_dict(
            {"evaluation": {"variants": {"cd": [{"type": "GraspNMS", "rotation": 30}]}}}
        )


# ------------------------------------------------------------------------------ NMS


def turned(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_nms_needs_both_distance_and_angle_and_sorts_by_score():
    translation = torch.tensor(
        [[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.02, 0.0, 0.5], [0.04, 0.0, 0.5], [0.0, 0.0, 0.5]]
    )
    rotation = torch.stack(
        [turned(0.0), turned(math.radians(20)), turned(math.radians(40)), turned(0.0), turned(0.0)]
    )
    score = torch.tensor([0.5, 0.9, 0.3, 0.6, 0.4])
    group = GraspGroup(
        translation=translation,
        rotation=rotation,
        width=torch.full((5,), 0.05),
        depth=torch.full((5,), 0.02),
        score=score,
        key=KEY,
    )
    kept = GraspNMS()(group, Observation(keys=[KEY]))
    # the same rule written as two loops
    brute = []
    order = torch.argsort(score, descending=True, stable=True).tolist()
    for index in order:
        duplicate = False
        for other in brute:
            distance = float((translation[index] - translation[other]).norm())
            cosine = (float((rotation[index] * rotation[other]).sum()) - 1) / 2
            angle = math.acos(max(-1.0, min(1.0, cosine)))
            duplicate |= distance < 0.03 and angle < math.radians(30)
        if not duplicate:
            brute.append(index)
    assert kept.score.tolist() == pytest.approx(score[brute].tolist())
    assert torch.all(kept.score[:-1] >= kept.score[1:]) and kept.key == KEY
    assert len(GraspNMS(translation_threshold=0.0)(group, Observation(keys=[KEY]))) == 5
    assert (
        len(
            GraspNMS(translation_threshold=1.0, rotation_threshold=180.0)(
                group, Observation(keys=[KEY])
            )
        )
        == 1
    )
    assert len(GraspNMS()(group[[]], Observation(keys=[KEY]))) == 0
    with pytest.raises(ValueError, match=r"rotation_threshold must be\s+in \[0, 180\]"):
        GraspNMS(rotation_threshold=200.0)
