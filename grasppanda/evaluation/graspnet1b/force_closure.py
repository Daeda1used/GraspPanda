"""Two-contact antipodal force-closure test on a signed-distance grid.

A grasp, expressed in the object frame, is a centre between the fingertips, a unit
closing axis and an opening width. Each fingertip starts half a width away from the
centre and travels towards it; where its path first crosses the object surface is a
contact. The grasp is in force closure for a friction coefficient μ when, at both
contacts, the line between the contacts lies inside the friction cone
(angle to the surface normal ≤ atan μ).

The benchmark score of a grasp is the smallest μ of the protocol's list for which
this holds, or ``-1``. The reference closes the fingers again for each of its six
μ values; the contacts do not depend on μ, so here they are found once and the cone
angle is compared with all thresholds. :func:`reference_walk` keeps the reference's
level-by-level bookkeeping so that both formulations can be tested against each other.

Finding a contact (the reference rule, kept in detail because every clause moves
scores):

* The path is sampled at ``max(3, floor(width in voxels))`` evenly spaced points.
* The first sample inside the surface band starts a candidate. A parabola through
  the candidate and its neighbours (the two before it if it is the last sample,
  the two after it if it is the first) locates the zero crossing.
* The candidate is rejected, and the search continues, if the fit is degenerate, if
  the crossing lies more than one voxel from the fit's first sample, or (except at
  the last sample) if the next sample is closer to the surface than the candidate.
* The contact must itself be inside the surface band after a round trip through the
  object frame; its normal is fitted there and oriented against the motion.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .arithmetic import dot3, norm3
from .sdf import SdfGrid

__all__ = ["Contacts", "cone_angles", "find_contacts", "friction_scores", "reference_walk"]

#: Score of a grasp that is not in force closure for any friction of the protocol.
NO_FORCE_CLOSURE = -1.0
#: A fitted parabola whose leading coefficient is below this is treated as degenerate.
_FLAT_FIT = 1e-10
#: Roots of the fit are searched in this parameter range (voxels along the path).
_ROOT_RANGE = (0.0, 10.0)
#: The crossing must lie within this distance (voxels) of the fit's first sample.
_CROSSING_REACH = 1.0


@dataclass(frozen=True)
class Contacts:
    """The two contacts of a closed grasp, in the object frame.

    Attributes:
        points: ``(2, 3)`` contact positions, metres.
        normals: ``(2, 3)`` outward surface normals.
    """

    points: np.ndarray
    normals: np.ndarray


def _zero_crossing(
    points: np.ndarray, distances: np.ndarray, first: int, second: int, third: int
) -> np.ndarray | None:
    """Zero of the parabola through three samples of a path, or ``None`` if unusable."""
    step = points[second] - points[first]
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = step / norm3(step)
    if not np.isfinite(direction).all():
        raise ValueError(
            "two consecutive samples of a finger path coincide (the grasp is too narrow "
            "for float32 coordinates); the reference evaluator fails on such a grasp too"
        )
    axis = int(np.flatnonzero(direction != 0)[0])
    t_second = step[axis] / direction[axis]
    t_third = (points[third] - points[first])[axis] / direction[axis]
    system = np.array([[0.0, 0.0, 1.0], [t_second**2, t_second, 1.0], [t_third**2, t_third, 1.0]])
    try:
        fit = np.linalg.solve(system, distances[[first, second, third]])
    except np.linalg.LinAlgError:
        return None
    crossing = None
    roots = np.roots(fit)
    if not np.iscomplexobj(roots):  # a complex pair never qualifies
        for root in roots:
            if _ROOT_RANGE[0] <= root <= _ROOT_RANGE[1]:
                crossing = root  # the last qualifying root wins
    if np.abs(fit[0]) < _FLAT_FIT:
        return None
    if crossing is None:
        crossing = -fit[1] / (2 * fit[0])  # no usable root: the vertex of the parabola
    if crossing < -_CROSSING_REACH or crossing > _CROSSING_REACH:
        return None
    return points[first] + crossing * direction


def _first_crossing(points: np.ndarray, distances: np.ndarray, band: float) -> np.ndarray | None:
    """Walk one finger path (grid frame) and return its accepted zero crossing."""
    last = len(points) - 1
    magnitude = np.abs(distances)
    for index in np.flatnonzero(magnitude < band):
        start = index - 2 if index == last else max(index - 1, 0)
        if index != last and magnitude[index + 1] < magnitude[index]:
            # Still approaching the surface: rejected whatever the fit says, so the fit
            # is skipped. A degenerate path must fail here as it would in the fit.
            if (points[start + 1] == points[start]).all():
                _zero_crossing(points, distances, start, start + 1, start + 2)
            continue
        crossing = _zero_crossing(points, distances, start, start + 1, start + 2)
        if crossing is not None:
            return crossing
    return None


def find_contacts(
    grid: SdfGrid, centers: np.ndarray, axes: np.ndarray, widths: np.ndarray
) -> list[Contacts | None]:
    """Close parallel jaws on an object for a batch of grasps.

    Args:
        grid: The object's signed-distance grid.
        centers: ``(G, 3)`` fingertip midpoints in the object frame, metres.
        axes: ``(G, 3)`` unit closing axes in the object frame.
        widths: ``(G,)`` jaw openings, metres.

    Returns:
        Per grasp the two contacts, or ``None`` when either finger finds none.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    axes = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    widths = np.asarray(widths, dtype=np.float64).reshape(-1)
    count = len(widths)
    if count == 0:
        return []

    # Sample both finger paths of every grasp and evaluate them in one query.
    paths, owners = [], []
    for grasp in range(count):
        samples = max(int(grid.object_to_grid_scale * widths[grasp]), 3)
        half = widths[grasp] / 2.0
        travel = np.linspace(0, float(widths[grasp]) / 2, num=samples)[:, None]
        reach = half * axes[grasp]
        paths.append((centers[grasp] - reach) + travel * axes[grasp])
        paths.append((centers[grasp] + reach) + travel * -axes[grasp])
        owners.extend((grasp, grasp))
    lengths = [len(path) for path in paths]
    grid_paths = grid.object_to_grid(np.concatenate(paths))
    path_distance = grid.signed_distance(grid_paths)
    bounds = np.concatenate([[0], np.cumsum(lengths)])

    crossings, directions, finger_of = [], [], []
    for finger, (lo, hi) in enumerate(itertools.pairwise(bounds)):
        points = grid_paths[lo:hi]
        crossing = _first_crossing(points, path_distance[lo:hi], grid.surface_threshold)
        if crossing is None:
            continue
        motion = points[-1] - points[0]
        crossings.append(crossing)
        directions.append(motion / norm3(motion))
        finger_of.append(finger)
    if not crossings:
        return [None] * count

    # Contacts live in the object frame; the surface test and the normal use the
    # position after it has been taken there and back (two float32 round trips).
    contact_points = grid.grid_to_object(np.array(crossings))
    contact_grid = grid.object_to_grid(contact_points)
    on_surface = grid.on_surface(grid.signed_distance(contact_grid))
    inward = grid.direction_to_object(np.array(directions))
    normals = np.zeros((len(crossings), 3))
    if on_surface.any():
        fitted = grid.surface_normals(contact_grid[on_surface])
        flip = dot3(inward[on_surface], fitted) > 0
        fitted[flip] = -fitted[flip]
        normals[on_surface] = grid.direction_to_object(fitted)

    found: dict[int, int] = {finger: row for row, finger in enumerate(finger_of) if on_surface[row]}
    result: list[Contacts | None] = []
    for grasp in range(count):
        rows = [found.get(2 * grasp), found.get(2 * grasp + 1)]
        if rows[0] is None or rows[1] is None:
            result.append(None)
        else:
            result.append(Contacts(points=contact_points[rows], normals=normals[rows]))
    return result


def cone_angles(contacts: Contacts) -> np.ndarray | None:
    """Angle at each contact between the inward normal and the line to the other contact.

    Returns ``None`` when both contacts coincide (no line between them). A cosine
    that rounds above one gives ``NaN``, which never exceeds a threshold, exactly as in
    the reference.
    """
    first, second = contacts.points
    if (first == second).all():
        return None
    angles = np.empty(2)
    for index, (point, other) in enumerate(((first, second), (second, first))):
        inward = -contacts.normals[index]
        line = other - point
        projection = np.abs(dot3(inward, line)) / norm3(inward)
        with np.errstate(invalid="ignore"):
            angles[index] = np.arccos(projection / norm3(line))
    return angles


def reference_walk(holds: Sequence[bool], frictions: Sequence[float]) -> float:
    """Score from per-friction outcomes, following the reference's bookkeeping.

    ``frictions`` ascends; the reference visits it from the largest value down, stops
    at the first failure and reports the last value that held.
    """
    score = NO_FORCE_CLOSURE
    for friction, ok in zip(reversed(frictions), reversed(holds)):
        if not ok:
            break
        score = float(friction)
    return score


def friction_scores(
    grid: SdfGrid,
    centers: np.ndarray,
    axes: np.ndarray,
    widths: np.ndarray,
    frictions: Sequence[float],
) -> np.ndarray:
    """Smallest friction coefficient in ``frictions`` giving force closure, per grasp.

    Arguments are those of :func:`find_contacts`; ``frictions`` ascends. Returns
    ``(G,)`` float64 with ``-1`` for grasps without force closure at the largest value.
    """
    limits = np.arctan(np.asarray(frictions, dtype=np.float64))
    scores = np.full(len(np.atleast_1d(widths)), NO_FORCE_CLOSURE)
    for grasp, contacts in enumerate(find_contacts(grid, centers, axes, widths)):
        angles = None if contacts is None else cone_angles(contacts)
        if angles is None:
            continue
        with np.errstate(invalid="ignore"):
            outside = (angles[:, None] > limits[None, :]).any(axis=0)
        scores[grasp] = reference_walk(~outside, frictions)
    return scores
