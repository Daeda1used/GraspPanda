"""The antipodal test on analytic shapes, and "contacts once" against "close six times"."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b import force_closure
from grasppanda.evaluation.graspnet1b.force_closure import (
    Contacts,
    _first_crossing,
    _zero_crossing,
    cone_angles,
    find_contacts,
    friction_scores,
    reference_walk,
)
from grasppanda.evaluation.graspnet1b.protocol import GRASPNET1B_SV_V1 as PROTOCOL
from grasppanda.evaluation.graspnet1b.sdf import SdfGrid

from .eval_synthetic import (
    RESOLUTION,
    SPHERE_RADIUS,
    box_distance,
    curved_box_distance,
    make_grid,
    sphere_distance,
)

FRICTIONS = PROTOCOL.frictions
WIDE_BOX = (0.02, 0.045, 0.015)


@pytest.fixture(scope="module")
def sphere() -> SdfGrid:
    return make_grid(sphere_distance)


@pytest.fixture(scope="module")
def box() -> SdfGrid:
    return make_grid(lambda points: curved_box_distance(points, WIDE_BOX))


def planar_axis(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([math.cos(angle), math.sin(angle), 0.0])


def scores(grid, centers, axes, widths):
    centers, axes = np.atleast_2d(centers), np.atleast_2d(axes)
    return friction_scores(grid, centers, axes, np.atleast_1d(widths), FRICTIONS)


def expected_score(angle_degrees: float) -> float:
    needed = math.tan(math.radians(angle_degrees))
    return next((mu for mu in FRICTIONS if needed <= mu), -1.0)


# ------------------------------------------------------------------------- contacts


def test_sphere_through_the_centre(sphere):
    axis = np.array([0.0, 1.0, 0.0])
    (contacts,) = find_contacts(sphere, np.zeros((1, 3)), axis[None], np.array([0.08]))
    assert contacts is not None
    first, second = contacts.points
    assert np.allclose(first, [0, -SPHERE_RADIUS, 0], atol=RESOLUTION)
    assert np.allclose(second, [0, SPHERE_RADIUS, 0], atol=RESOLUTION)
    # outward normals: against each finger's direction of travel
    assert contacts.normals[0] @ axis < -0.98 and contacts.normals[1] @ axis > 0.98
    assert np.array_equal(contacts.normals, contacts.normals.astype(np.float32))  # float32 values
    assert cone_angles(contacts).max() < math.radians(8.0)
    assert scores(sphere, np.zeros(3), axis, 0.08)[0] == FRICTIONS[0]


@pytest.mark.parametrize("offset_degrees", [0.0, 15.0, 27.0, 34.5, 42.0, 47.5, 65.0])
def test_sphere_chord_score_follows_the_cone_angle(sphere, offset_degrees):
    """A chord at height h meets the sphere where the normal is asin(h / r) off the chord."""
    height = SPHERE_RADIUS * math.sin(math.radians(offset_degrees))
    center = np.array([height, 0.0, 0.0])
    axis = np.array([0.0, 1.0, 0.0])
    (contacts,) = find_contacts(sphere, center[None], axis[None], np.array([0.09]))
    assert contacts is not None
    angles = np.degrees(cone_angles(contacts))
    assert np.abs(angles - offset_degrees).max() < 12.0  # the normal fit is coarse
    # internal consistency: the score is the smallest friction whose cone holds both angles
    assert scores(sphere, center, axis, 0.09)[0] == expected_score(angles.max())


@pytest.mark.parametrize("degrees", [0.0, 10.0])
def test_box_grasp_across_two_faces(box, degrees):
    """A line through the box centre tilted by φ meets the ±x faces at angle φ to their normals."""
    axis = planar_axis(degrees)
    (contacts,) = find_contacts(box, np.zeros((1, 3)), axis[None], np.array([0.09]))
    assert contacts is not None
    reach = WIDE_BOX[0] / math.cos(math.radians(degrees))
    assert np.allclose(contacts.points, [-reach * axis, reach * axis], atol=RESOLUTION / 2)
    assert np.allclose(contacts.normals, [[-1, 0, 0], [1, 0, 0]], atol=1e-3)
    assert np.allclose(np.degrees(cone_angles(contacts)), degrees, atol=0.5)
    assert scores(box, np.zeros(3), axis, 0.09)[0] == 0.2


def test_an_exactly_linear_distance_profile_yields_no_contact():
    """Reference behaviour: a flat parabola fit is rejected.

    Towards a flat, grid-aligned face the interpolated distance is linear wherever three
    consecutive samples share a voxel cell; on a perfect box that is every candidate.
    """
    flat = make_grid(lambda points: box_distance(points, WIDE_BOX))
    assert scores(flat, np.zeros(3), planar_axis(0.0), 0.09)[0] == -1.0
    points = np.array([10.0, 20.0, 30.0]) + np.arange(3)[:, None] * np.array([1.0, 0.0, 0.0])
    assert _zero_crossing(points, np.array([0.375, -0.125, -0.625]), 0, 1, 2) is None


def test_no_contact_means_no_force_closure(sphere, box):
    axis = np.array([0.0, 1.0, 0.0])
    away = np.array([0.05, 0.0, 0.0])  # the closing line misses the sphere
    assert find_contacts(sphere, away[None], axis[None], np.array([0.08])) == [None]
    assert scores(sphere, away, axis, 0.08)[0] == -1.0
    # fingers that start inside the object never cross its surface
    assert scores(box, np.zeros(3), planar_axis(0.0), 0.02)[0] == -1.0
    # one finger finds the surface, the other starts inside: still no grasp
    shifted = np.array([0.012, 0.0, 0.0])
    assert find_contacts(box, shifted[None], planar_axis(0.0)[None], np.array([0.03])) == [None]


def test_batch_equals_one_by_one(sphere):
    rng = np.random.default_rng(0)
    centers = rng.uniform(-0.02, 0.02, size=(40, 3))
    axes = rng.normal(size=(40, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    widths = rng.uniform(0.004, 0.1, size=40)
    together = friction_scores(sphere, centers, axes, widths, FRICTIONS)
    single = [scores(sphere, c, a, w)[0] for c, a, w in zip(centers, axes, widths)]
    assert np.array_equal(together, single)
    assert len(set(together.tolist())) >= 4  # several friction levels and failures occur
    assert friction_scores(
        sphere, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), FRICTIONS
    ).shape == (0,)


# ---------------------------------------------- contacts once versus six closings


def close_six_times(grid, center, axis, width) -> float:
    """The reference's control flow: close the fingers again for every friction level."""
    descending = FRICTIONS[::-1]
    quality, previous, holds = -1.0, False, False
    for index, friction in enumerate(descending):
        previous = holds
        (contacts,) = find_contacts(grid, center[None], axis[None], np.array([width]))
        holds = False
        if contacts is not None:
            angles = cone_angles(contacts)
            if angles is not None:
                with np.errstate(invalid="ignore"):
                    holds = not (angles > np.arctan(friction)).any()
        if previous and not holds:
            quality = descending[index - 1]
            break
        if holds and friction == descending[-1]:
            quality = friction
            break
        if friction == descending[0] and not holds:
            break
    return quality


def test_contacts_once_gives_the_same_score_as_closing_six_times(sphere, box):
    rng = np.random.default_rng(1)
    for grid, reach in ((sphere, 0.025), (box, 0.03)):
        centers = rng.uniform(-reach, reach, size=(150, 3)) * [1.0, 1.0, 0.4]
        axes = rng.normal(size=(150, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        widths = rng.uniform(0.01, 0.1, size=150)
        fast = friction_scores(grid, centers, axes, widths, FRICTIONS)
        slow = [close_six_times(grid, c, a, w) for c, a, w in zip(centers, axes, widths)]
        assert np.array_equal(fast, slow)
        counts = {level: int((fast == level).sum()) for level in (-1.0, *FRICTIONS)}
        assert counts[-1.0] > 10 and sum(counts[mu] > 0 for mu in FRICTIONS) >= 4, counts


def test_reference_walk():
    assert reference_walk([False] * 6, FRICTIONS) == -1.0
    assert reference_walk([True] * 6, FRICTIONS) == 0.2
    assert reference_walk([False, False, True, True, True, True], FRICTIONS) == 0.6
    assert reference_walk([True, True, True, True, True, False], FRICTIONS) == -1.0
    # not monotone: the walk from the largest friction stops at the first failure
    assert reference_walk([True, False, True, True, True, True], FRICTIONS) == 0.6


# ------------------------------------------------------------------ cone fine print


def test_coincident_contacts_have_no_cone():
    point = np.array([0.01, 0.02, 0.03])
    normals = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    assert cone_angles(Contacts(points=np.stack([point, point]), normals=normals)) is None


def test_a_cosine_above_one_counts_as_inside_the_cone(sphere, monkeypatch):
    rng = np.random.default_rng(2)
    for _ in range(20000):  # find a vector whose normalised self-product rounds above 1
        v = rng.normal(size=3)
        length = math.sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
        if ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]) / length / length > 1.0:
            break
    else:
        pytest.fail("no rounding example found")
    contacts = Contacts(points=np.stack([np.zeros(3), v]), normals=np.stack([-v, v]))
    assert np.isnan(cone_angles(contacts)).any()
    monkeypatch.setattr(force_closure, "find_contacts", lambda *args: [contacts])
    assert scores(sphere, np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.05)[0] == 0.2


# -------------------------------------------------------------------- zero crossing


def line(count: int, step: float = 1.0) -> np.ndarray:
    return np.array([10.0, 20.0, 30.0]) + step * np.arange(count)[:, None] * np.array(
        [0.6, 0.0, 0.8]
    )


def test_zero_crossing_of_a_parabola():
    points = line(3)
    travel = np.arange(3.0)
    distance = 0.05 * (travel - 0.6) * (travel + 4.0)  # roots at 0.6 and -4
    crossing = _zero_crossing(points, distance, 0, 1, 2)
    assert np.allclose(crossing, points[0] + 0.6 * np.array([0.6, 0.0, 0.8]))


def test_zero_crossing_rejections():
    points = line(3)
    travel = np.arange(3.0)
    assert _zero_crossing(points, 0.3 * (travel - 0.5), 0, 1, 2) is None  # a straight line
    far = 0.05 * (travel - 1.5) * (travel + 4.0)  # root 1.5: more than one voxel away
    assert _zero_crossing(points, far, 0, 1, 2) is None
    above = 0.05 * (travel - 0.4) ** 2 + 0.01  # no real root: the vertex (0.4) is used
    assert np.allclose(
        _zero_crossing(points, above, 0, 1, 2), points[0] + 0.4 * np.array([0.6, 0, 0.8])
    )
    both = 0.05 * (travel - 0.2) * (travel - 0.9)  # two roots in range: the last one reported
    crossing = _zero_crossing(points, both, 0, 1, 2)
    reported = (crossing - points[0])[0] / 0.6
    assert min(abs(reported - 0.2), abs(reported - 0.9)) < 1e-9
    with pytest.raises(ValueError, match="coincide"):
        _zero_crossing(np.repeat(points[:1], 3, axis=0), far, 0, 1, 2)


def test_first_crossing_walks_past_rejected_candidates():
    band = 0.7
    points = line(6)
    travel = np.arange(6.0)
    # enters the band at sample 2 while still approaching (|next| < |this|): rejected there,
    # accepted at sample 3, whose fit over samples 2..4 has its root within one voxel of 2
    distance = 0.9 - 0.25 * travel
    distance = distance + 0.02 * (travel - 3.0) ** 2
    accepted = _first_crossing(points, distance, band)
    assert accepted is not None
    start = (accepted - points[0])[0] / 0.6
    assert 2.0 <= start <= 4.0
    assert _first_crossing(points, np.full(6, 5.0), band) is None  # never near the surface
    # only the last sample is in the band: fitted with the two samples before it
    late = np.array([5.0, 4.0, 3.0, 2.0, 1.3, 0.45])
    assert _first_crossing(points, late, band) is None  # the root lies beyond the last sample
