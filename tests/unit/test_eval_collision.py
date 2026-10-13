"""Evaluator-side collision and emptiness masks, one gripper region at a time."""

from __future__ import annotations

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b.arithmetic import rotate_rows
from grasppanda.evaluation.graspnet1b.collision import crop_to_box, gripper_collision
from grasppanda.evaluation.graspnet1b.protocol import GRASPNET1B_SV_V1 as PROTOCOL

GRIPPER = PROTOCOL.gripper
WIDTH, DEPTH = 0.06, 0.03


def filler(count: int = 12) -> np.ndarray:
    """Points between the fingers, so that a grasp is not "empty"."""
    return np.stack([np.linspace(0.0, 0.02, count), np.zeros(count), np.zeros(count)], axis=1)


def check(points, *, width=WIDTH, depth=DEPTH, rotation=None, translation=None):
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    scene = np.asarray(points, dtype=np.float64) @ rotation.T + translation
    collision, empty = gripper_collision(
        translation[None],
        rotation[None],
        np.array([depth]),
        np.array([width]),
        scene,
        gripper=GRIPPER,
        empty_threshold=PROTOCOL.empty_threshold,
    )
    return bool(collision[0]), bool(empty[0])


def with_point(point):
    return check(np.concatenate([filler(), [point]]))


def test_free_grasp_with_enough_points_between_the_fingers():
    assert check(filler()) == (False, False)


@pytest.mark.parametrize(
    "point,collides",
    [
        ((0.01, 0.035, 0.0), True),  # right finger: width / 2 < y < width / 2 + finger_width
        ((0.01, -0.035, 0.0), True),  # left finger
        ((0.01, 0.045, 0.0), False),  # beyond the finger
        ((0.01, 0.035, 0.015), False),  # above the finger (|z| >= height / 2)
        ((0.031, 0.035, 0.0), False),  # past the fingertip (x >= depth)
        ((-0.019, 0.035, 0.0), True),  # the finger starts at -depth_base
        ((-0.025, 0.0, 0.0), True),  # palm: -depth_base - finger_width < x < -depth_base
        ((-0.025, 0.039, 0.0), True),  # the palm spans the outer finger faces
        ((-0.025, 0.041, 0.0), False),
        ((-0.035, 0.0, 0.0), False),  # behind the palm
        ((-0.015, 0.0, 0.0), False),  # between the fingers near the palm: not a collision
    ],
)
def test_gripper_regions(point, collides):
    assert with_point(point) == (collides, False)


def test_boundaries_are_strict():
    half, finger, base = WIDTH / 2, GRIPPER.finger_width, GRIPPER.depth_base
    on_surface = [
        (0.01, half, 0.0),  # inner finger face
        (0.01, half + finger, 0.0),  # outer finger face
        (0.01, half + finger / 2, GRIPPER.finger_height / 2),  # top of the finger
        (DEPTH, half + finger / 2, 0.0),  # fingertip
        (-base, 0.0, 0.0),  # front of the palm
        (-(base + finger), 0.0, 0.0),  # back of the palm
    ]
    for point in on_surface:
        assert with_point(point) == (False, False), point
    nudge = 1e-12
    assert with_point((0.01, half + nudge, 0.0))[0]
    assert with_point((-base - nudge, 0.0, 0.0))[0]


def test_empty_grasps_are_reported_as_collisions():
    assert check(filler(PROTOCOL.empty_threshold - 1)) == (True, True)
    assert check(filler(PROTOCOL.empty_threshold)) == (False, False)
    # points on the inner finger faces still count as "between the fingers"
    faces = filler(PROTOCOL.empty_threshold)
    faces[:, 1] = WIDTH / 2
    assert check(faces) == (False, False)
    faces[:, 1] = WIDTH / 2 + 1e-12  # inside the finger: collision, and nothing in between
    assert check(faces) == (True, True)
    assert check(np.zeros((0, 3))) == (True, True)


def test_zero_width_is_always_empty():
    centred = filler()
    assert check(centred, width=0.0) == (False, False)  # only points with y == 0 exactly fit
    off_centre = centred + np.array([0.0, 1e-9, 0.0])
    assert check(off_centre, width=0.0) == (True, True)


def test_pose_invariance_and_height_column_is_not_an_input():
    rng = np.random.default_rng(0)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    translation = rng.normal(size=3)
    for point, expected in [((0.01, 0.035, 0.0), True), ((0.01, 0.045, 0.0), False)]:
        points = np.concatenate([filler(), [point]])
        assert check(points, rotation=rotation, translation=translation) == (expected, False)


def unpruned(translation, rotation, depth, width, scene):
    """The definition evaluated for every scene point, without any shortcut."""
    local = rotate_rows(scene[None, :, :] - translation[:, None, :], rotation)
    x, y, z = local[..., 0], local[..., 1], local[..., 2]
    half, fw, base = width[:, None] / 2, GRIPPER.finger_width, GRIPPER.depth_base
    mask1 = (z > -GRIPPER.finger_height / 2) & (z < GRIPPER.finger_height / 2)
    mask2 = (x > -base) & (x < depth[:, None])
    mask3, mask4 = y > -(half + fw), y < -half
    mask5, mask6 = y < (half + fw), y > half
    mask7 = (x > -(base + fw)) & (x < -base)
    left, right = mask1 & mask2 & mask3 & mask4, mask1 & mask2 & mask5 & mask6
    bottom = mask1 & mask3 & mask5 & mask7
    empty = (mask1 & mask2 & ~mask4 & ~mask6).sum(axis=1) < PROTOCOL.empty_threshold
    return (left | right | bottom).any(axis=1) | empty, empty


def test_pruning_never_changes_a_result():
    """Distant points are skipped; the outcome must equal the all-points definition."""
    rng = np.random.default_rng(1)
    count = 60
    scene = rng.uniform(-0.3, 0.3, size=(6000, 3))
    scene[:3000] *= 0.2  # dense near the grasps, sparse far away
    rotation = np.linalg.qr(rng.normal(size=(count, 3, 3)))[0]
    rotation[40:44] *= 0.3  # not rotations: shrunk, stretched, singular, zero
    rotation[44:48] *= 5.0
    rotation[48:52, :, 2] = rotation[48:52, :, 1]
    rotation[52:54] = 0.0
    arguments = (
        rng.uniform(-0.03, 0.03, size=(count, 3)),
        rotation,
        rng.choice([-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.3], size=count),
        rng.uniform(0.0, 0.1, size=count),
    )
    kwargs = {"gripper": GRIPPER, "empty_threshold": PROTOCOL.empty_threshold}
    collision, empty = gripper_collision(*arguments, scene, **kwargs)
    expected_collision, expected_empty = unpruned(*arguments, scene)
    assert np.array_equal(collision, expected_collision) and np.array_equal(empty, expected_empty)
    assert collision.any() and 0 < empty.sum() < count and (collision | ~empty).all()
    # a zero matrix maps every point to the grasp point: between the fingers iff depth > 0
    assert (empty[52:54] == (arguments[2][52:54] <= 0)).all()
    for index in range(0, count, 7):  # one grasp at a time gives the same answer
        single = gripper_collision(*(a[index : index + 1] for a in arguments), scene, **kwargs)
        assert (bool(single[0][0]), bool(single[1][0])) == (collision[index], empty[index])


def test_crop_to_box_is_strict():
    anchor = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    margin = 0.5
    scene = np.array(
        [
            [0.5, 1.0, 1.5],  # inside
            [-0.5, 1.0, 1.5],  # on the lower x face: excluded
            [1.5, 1.0, 1.5],  # on the upper x face: excluded
            [-0.4999, -0.4999, -0.4999],
            [1.4999, 2.4999, 3.4999],
            [0.5, 2.6, 1.5],
        ]
    )
    assert np.array_equal(crop_to_box(scene, anchor, margin), scene[[0, 3, 4]])
