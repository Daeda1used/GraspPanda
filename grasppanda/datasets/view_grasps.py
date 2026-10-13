"""Per-view grasp labels: 6-D grasps of a frame expressed in its image.

HGGD trains on labels of this form. Its converter (``convert_grasp.py``) derives them
from the dense GraspNet-format labels; this module is that derivation as a derived asset
of a dataset: keyed by frame, content-hashed, resumable scene by scene and never written
into the dataset tree.

Definition, per frame:

1. **Candidates.** Every annotated grasp of the scene's objects whose friction score is
   in ``(0, friction_threshold]``, that does not collide in the scene, whose width is in
   ``(min_width, max_width)`` and whose depth is in ``(min_depth, max_depth)``. With the
   reference values only the 2 cm depth bin passes. The score of a grasp is
   ``1.1 - friction``, so the reference threshold leaves two score values, 1.0 and 0.9.
2. **Camera frame.** The grasps are placed with the object poses of the frame. A grasp is
   kept when the camera-z component of its approach axis exceeds ``approach_threshold``
   (zero: the gripper moves away from the camera).
3. **Sub-sampling.** More than ``max_grasps`` grasps are reduced to ``max_grasps`` drawn
   without replacement.
4. **Centre refinement.** ``model_points`` vertices are drawn from the object models of
   the scene, placed with the same poses. Each grasp centre moves along the closing
   axis to the middle of the vertices between its fingers (``|z| < height / 2``,
   ``depth - finger_length < x < depth``, ``|y| < width / 2`` in the grasp frame). A grasp
   with no vertex between its fingers stays where it is; the width is never changed.
5. **Suppression.** Greedy NMS in descending score order: a kept grasp removes every
   later grasp closer than ``nms_distance`` *and* rotated by less than ``nms_angle``.
   Nearly all scores are tied, so the visiting order among equal scores decides which
   of them survive; it is drawn at random.
6. **Image.** The centre is projected to a pixel and a camera depth. With the grasp moved
   onto the optical axis at that depth, ``theta`` is the image direction from the
   centre to the jaw, ``gamma`` the tilt of the closing axis out of the image plane and
   ``beta`` the remaining roll; ``width`` and ``height`` are the projected jaw opening
   and finger thickness, corrected for the two tilts.
7. **Image bounds.** A label is kept when its centre pixel is strictly inside the image.

What the reference does that one might not expect, each a named option whose default is
the reference behaviour:

* ``intrinsics="reference"``: the projection uses one constant camera matrix per camera
  (:data:`REFERENCE_INTRINSICS`: the calibration rounded to two decimals), not the
  calibration of the frame. In GraspNet-1Billion the per-frame calibration of the
  RealSense holds exactly these values and that of the Kinect differs by 0.005 px
  (every ninth scene was checked).
* ``poses="annotations"``: object poses come from the per-frame annotation files (four
  decimals), not from the poses the dense labels are aligned with.
* ``visibility_filter=False``: the reference computes which grasp centres lie within
  ``visibility_distance`` of the depth cloud, uses that for a statistic and saves every
  grasp regardless. Turning the option on drops the others.

Where the reference cannot be followed to the letter:

* *Ties.* The reference suppresses with a compiled extension that orders grasps with an
  unstable sort, so its order among equal scores is an artefact of a C++ library. Here
  that order is an explicit random choice (:class:`SeededDraws`); given the same order
  the survivors are the same grasps in the same order.
* *Randomness.* The reference draws from Python's global ``random``; here every draw of
  a frame comes from a generator seeded with ``(seed, scene, frame, draw)``, so a frame
  can be generated alone and a scene can be resumed.
* *Labels without a pixel.* A grasp centre in the camera plane (zero depth) has no pixel;
  the reference would store ``inf`` for it. Such a label cannot be stored, so it is
  dropped and counted in the shard record (``degenerate``). No object in front of a
  camera produces one.

Numbers are computed with the precision the reference uses, so that a recorded reference
run can be compared value by value: label points, grasp frames and placed grasps are
single precision (the GraspNet grasp array), the refinement and the projection are
double precision, and ``theta``, ``width`` and ``height`` are derived from the image
rectangle in single precision. The label view lattice is computed in single precision
as the GraspNet API does, which differs from the exact lattice of
:func:`grasppanda.ops.fibonacci_views` by up to 8e-5 in a component. With the recorded
choices replayed, the placed grasps and the survivors of the suppression are identical
to the reference's and the stored values agree to rounding of the last bits
(``tests/integration/test_hggd_labels.py``).
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

from grasppanda.structures import ContractError, Gripper, SampleKey
from grasppanda.transforms.frame import Frame, transform_points

from .assets import Asset, AssetError, AssetSpec, atomic_path
from .labels import LabelGrid, StoredGraspLabels

__all__ = [
    "COLUMNS",
    "FIELDS",
    "REFERENCE_INTRINSICS",
    "Draws",
    "GraspSet",
    "SceneGrasps",
    "SceneModels",
    "SeededDraws",
    "ViewGraspConfig",
    "ViewGraspTrace",
    "ViewGrasps",
    "camera_grasps",
    "frame_view_grasps",
    "generate_view_grasps",
    "grasp_rectangles",
    "reference_views",
    "refine_centers",
    "scene_grasps",
    "scene_models",
    "scene_shard",
    "suppress",
]

#: Arrays of one frame, all with one row per grasp:
#:
#: ``centers``    ``(L, 2)`` grasp centre ``(u, v)`` in pixels of the full-resolution image
#: ``depth``      ``(L,)`` camera-z of the grasp centre, metres
#: ``theta``      ``(L,)`` in-plane angle, radians
#: ``gamma``      ``(L,)`` rotation about the gripper's height axis, radians
#: ``beta``       ``(L,)`` rotation about the gripper's closing axis, radians
#: ``width``      ``(L,)`` projected jaw opening, pixels of the full-resolution image
#: ``height``     ``(L,)`` projected finger thickness, pixels
#: ``score``      ``(L,)`` label quality, ``1.1 - friction`` (higher is better)
#: ``object_id``  ``(L,)`` dataset id of the object the grasp belongs to
FIELDS = ("centers", "depth", "theta", "gamma", "beta", "width", "height", "score", "object_id")

#: Columns of the ``(L, 10)`` array a frame is stored as.
COLUMNS = ("u", "v", "depth", "theta", "gamma", "beta", "width", "height", "score", "object_id")

#: ``(fx, fy, cx, cy)`` the reference converter projects with, per camera of
#: GraspNet-1Billion: the calibration rounded to two decimals, the same for every scene.
#: Whoever turns the labels back into 3-D must use the intrinsics they were projected with.
REFERENCE_INTRINSICS: dict[str, tuple[float, float, float, float]] = {
    "realsense": (927.17, 927.37, 651.32, 349.62),
    "kinect": (631.55, 631.21, 638.43, 366.50),
}

#: GraspNet convention: the score of a label grasp is this minus its friction coefficient.
_SCORE_OFFSET = 1.1
#: The reference hides the points outside the fingers by shifting the coordinate by this
#: much before taking the extremes, which rounds them to multiples of 2**-49 m. Repeated
#: here so that refined centres compare bit for bit.
_REFERENCE_SHIFT = 10.0
#: Label points read from the memory-mapped store at once.
_LABEL_ROWS = 256
#: Random draws of a frame, in the order they are used; the position seeds the draw.
_DRAWS = ("model_points", "grasps", "nms_ties", "depth_points")


def scene_shard(scene: int) -> str:
    """Name of the shard that holds every frame of a scene."""
    return f"scene_{scene:04d}"


# ------------------------------------------------------------------------ configuration


@dataclass(frozen=True)
class ViewGraspConfig:
    """Parameters of the per-view grasp labels; the defaults are HGGD's converter.

    Every field is part of the asset hash.

    Attributes:
        friction_threshold: Keep grasps whose friction score is in ``(0, threshold]``.
        min_width, max_width: Keep grasps whose width is strictly between them, metres.
        min_depth, max_depth: Keep grasps whose depth is strictly between them, metres.
        approach_threshold: Keep grasps whose approach axis has a camera-z component
            above this.
        max_grasps: Grasps of a frame that enter refinement and suppression; more are
            sub-sampled.
        model_points: Object-model vertices drawn per frame for the centre refinement.
        finger_length: Extent of the region between the fingers along the approach axis,
            measured back from the grasp depth, metres.
        nms_distance: Suppression radius, metres (strict).
        nms_angle: Suppression rotation angle, degrees (strict).
        poses: Which of the dataset's object poses place grasps and models.
        intrinsics: ``reference`` projects with :data:`REFERENCE_INTRINSICS`, ``frame``
            with the calibration of each frame.
        visibility_filter: Drop grasps whose centre is farther than
            ``visibility_distance`` from ``visibility_points`` points drawn from the
            frame's depth cloud. The reference computes this and does not apply it.
        visibility_distance: Metres (strict).
        visibility_points: Depth-cloud points drawn for the visibility test.
        seed: Seed of every random draw, together with the scene and the frame.
        dtype: Storage type of the float arrays.
    """

    friction_threshold: float = 0.2
    min_width: float = 0.03
    max_width: float = 0.15
    min_depth: float = 0.015
    max_depth: float = 0.025
    approach_threshold: float = 0.0
    max_grasps: int = 5000
    model_points: int = 40000
    finger_length: float = 0.04
    nms_distance: float = 0.02
    nms_angle: float = 60.0
    poses: str = "annotations"
    intrinsics: Literal["reference", "frame"] = "reference"
    visibility_filter: bool = False
    visibility_distance: float = 0.01
    visibility_points: int = 40000
    seed: int = 0
    dtype: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        # A config file may spell 1 for 1.0; the hash must not depend on the spelling.
        for entry in fields(self):
            value = getattr(self, entry.name)
            annotation = getattr(entry.type, "__name__", entry.type)  # a name or the type
            kind = {"float": float, "int": int, "bool": bool, "str": str}.get(annotation)
            if kind is None:
                continue
            accepted = (int, float) if kind is float else kind
            if not isinstance(value, accepted) or (kind is not bool and isinstance(value, bool)):
                raise ContractError(
                    f"view_grasps.{entry.name} must be {kind.__name__}, got {value!r}"
                )
            object.__setattr__(self, entry.name, kind(value))
        problems = []
        if not self.friction_threshold > 0:
            problems.append(f"friction_threshold must be positive, got {self.friction_threshold}")
        for name in ("width", "depth"):
            low, high = getattr(self, f"min_{name}"), getattr(self, f"max_{name}")
            if not low < high:
                problems.append(f"min_{name} ({low}) must be below max_{name} ({high})")
        for name in ("max_grasps", "model_points", "visibility_points"):
            if getattr(self, name) < 1:
                problems.append(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("finger_length", "nms_distance", "visibility_distance"):
            if not getattr(self, name) > 0:
                problems.append(f"{name} must be positive, got {getattr(self, name)}")
        if not 0 < self.nms_angle <= 180:
            problems.append(f"nms_angle must be in (0, 180] degrees, got {self.nms_angle}")
        if self.intrinsics not in ("reference", "frame"):
            problems.append(f"intrinsics must be 'reference' or 'frame', got {self.intrinsics!r}")
        if self.dtype not in ("float64", "float32"):
            problems.append(f"dtype must be 'float64' or 'float32', got {self.dtype!r}")
        if problems:
            raise ContractError("view_grasps: " + "; ".join(problems))


# ------------------------------------------------------------------------------ choices


class Draws(Protocol):
    """Source of the arbitrary choices of one frame."""

    def sample(self, what: str, available: int, count: int) -> np.ndarray:
        """``count`` distinct rows out of ``available``, in the order they are used.

        ``what`` names the choice: ``model_points`` (vertices of the scene's object
        models), ``grasps`` (the sub-sample of a frame with too many grasps),
        ``nms_ties`` (a permutation: the order in which equally scored grasps are
        visited) or ``depth_points`` (pixels with depth, for the visibility test).
        """


class SeededDraws:
    """Draws of one frame from generators seeded with ``(seed, scene, frame, draw)``.

    Each named draw has its own generator, so turning an optional step on or off does
    not change the others.
    """

    def __init__(self, seed: int, key: SampleKey) -> None:
        self.seed, self.key = seed, key

    def sample(self, what: str, available: int, count: int) -> np.ndarray:
        if count == 0:
            return np.zeros(0, dtype=np.int64)
        rng = np.random.default_rng([self.seed, self.key.scene, self.key.frame, _DRAWS.index(what)])
        return rng.choice(available, count, replace=False)


def _checked_draw(draws: Draws, what: str, available: int, count: int) -> np.ndarray:
    """A draw, verified: a replayed choice recorded for other data must not pass."""
    index = np.asarray(draws.sample(what, available, count), dtype=np.int64)
    if index.shape != (count,) or (count and (index.min() < 0 or index.max() >= available)):
        raise ContractError(
            f"the '{what}' draw must be {count} rows out of {available}; got shape "
            f"{index.shape}" + (f" with values in [{index.min()}, {index.max()}]" if count else "")
        )
    if len(np.unique(index)) != count:
        raise ContractError(f"the '{what}' draw repeats rows; it must be without replacement")
    return index


# ---------------------------------------------------------------------- scene, once


def reference_views(num_view: int) -> np.ndarray:
    """The view lattice of the dense labels as the GraspNet API computes it.

    The API evaluates the Fibonacci lattice in single precision, including the angle
    ``2 * i * pi * golden`` that reaches 1160 rad, so its views differ from the exact
    lattice by up to 8e-5 in a component. The reference places label grasps with these
    values.

    Returns:
        ``(num_view, 3)`` float32 viewpoints; the approach direction of a view is its
        negative.
    """
    one, two = np.float32(1.0), np.float32(2.0)
    index = np.arange(num_view, dtype=np.float32)
    z = (two * index + one) / np.float32(num_view) - one
    radius = np.sqrt(one - z * z)
    angle = two * index * np.float32(math.pi) * np.float32((math.sqrt(5.0) - 1.0) / 2.0)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


def _norm32(vectors: np.ndarray) -> np.ndarray:
    """Single-precision length of ``(N, 3)`` vectors, summed in x, y, z order."""
    squares = vectors * vectors
    return np.sqrt((squares[:, 0] + squares[:, 1]) + squares[:, 2])[:, None]


def _fused32(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``fl32(fl32(a * b) + c * d)`` for float32 inputs: the second product is not rounded.

    This is what a single-precision matrix product gives on a machine with fused
    multiply-add. A product of two float32 values is exact in double precision, so the
    value is obtained here without relying on the BLAS of the machine.
    """
    first = (a.astype(np.float64) * b.astype(np.float64)).astype(np.float32)
    second = c.astype(np.float64) * d.astype(np.float64)
    return (first.astype(np.float64) + second).astype(np.float32)


def _label_frames(grid: LabelGrid) -> np.ndarray:
    """``(V, A, 3, 3)`` float32 gripper frame of every (view, angle) bin, object frame.

    The convention of :func:`grasppanda.ops.approach_to_rotation`: column 0 is the
    approach axis ``-view``, column 1 is horizontal and column 2 completes the frame,
    all rotated by the in-plane angle about the approach axis. The arithmetic is that of
    the GraspNet API, which builds the frames in single precision; a recorded reference
    run is reproduced bit for bit, which the suppression of step 5 needs (a pair of
    grasps at a threshold is decided by the last bit).
    """
    approach = -reference_views(grid.num_view)
    side = np.stack([-approach[:, 1], approach[:, 0], np.zeros_like(approach[:, 0])], axis=1)
    side[_norm32(side)[:, 0] == 0] = (0.0, 1.0, 0.0)  # a vertical approach has no horizontal
    approach = approach / _norm32(approach)
    side = side / _norm32(side)
    up = np.stack(
        [
            approach[:, 1] * side[:, 2] - approach[:, 2] * side[:, 1],
            approach[:, 2] * side[:, 0] - approach[:, 0] * side[:, 2],
            approach[:, 0] * side[:, 1] - approach[:, 1] * side[:, 0],
        ],
        axis=1,
    )
    angle = np.array(grid.angles, dtype=np.float32)
    cos, sin = np.cos(angle)[None, :, None], np.sin(angle)[None, :, None]
    side, up = side[:, None, :], up[:, None, :]
    shape = (grid.num_view, grid.num_angle, 3)
    columns = [
        np.broadcast_to(approach[:, None, :], shape),
        _fused32(side, cos, up, sin),
        _fused32(side, -sin, up, cos),
    ]
    return np.stack(columns, axis=-1)


@dataclass(frozen=True)
class SceneGrasps:
    """Candidate grasps of a scene in their object frames, in label order.

    Label order is instance by instance and, within one, the row-major order of the
    label grid (point, view, angle, depth). The reference sub-samples by position in
    this order, so it is part of the definition.

    Attributes:
        counts: ``(I,)`` candidates of each object instance; rows are grouped by instance.
        point: ``(K, 3)`` float32 grasp point.
        rotation: ``(K, 3, 3)`` float32 gripper frame.
        width, depth, score: ``(K,)`` float32.
    """

    counts: np.ndarray
    point: np.ndarray
    rotation: np.ndarray
    width: np.ndarray
    depth: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        return len(self.point)


def scene_grasps(labels: Sequence[StoredGraspLabels], config: ViewGraspConfig) -> SceneGrasps:
    """Step 1 for a whole scene: the candidates do not depend on the frame.

    Args:
        labels: Dense labels of every object instance, in the order of the scene's
            collision labels.
        config: Thresholds.
    """
    threshold = np.float32(config.friction_threshold)  # scores are float32; compare in kind
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("point", "rotation", "width", "depth", "score")
    }
    counts = []
    frames: dict[LabelGrid, np.ndarray] = {}
    for instance in labels:
        grid = instance.grid
        if grid not in frames:
            frames[grid] = _label_frames(grid)
        # The label files hold depths and widths in single precision; compare what they hold.
        depths = np.array(grid.depths, dtype=np.float32)
        bins = np.flatnonzero(
            (depths.astype(np.float64) > config.min_depth)
            & (depths.astype(np.float64) < config.max_depth)
        )
        found = 0
        for start in range(0, instance.num_points if len(bins) else 0, _LABEL_ROWS):
            rows = slice(start, start + _LABEL_ROWS)
            friction = np.asarray(instance.scores[rows])[..., bins]
            usable = (friction > 0) & (friction <= threshold) & ~instance.collision(rows)[..., bins]
            point, view, angle, depth = np.nonzero(usable)  # row-major: the label order
            if len(point) == 0:
                continue
            # Only the few usable entries of the width grid are read from the store.
            width = np.asarray(instance.widths[start + point, view, angle, bins[depth]])
            wide = width.astype(np.float64)
            keep = (wide > config.min_width) & (wide < config.max_width)
            point, view, angle, depth = point[keep], view[keep], angle[keep], depth[keep]
            parts["point"].append(np.asarray(instance.points[rows], dtype=np.float32)[point])
            parts["rotation"].append(frames[grid][view, angle])
            parts["width"].append(width[keep].astype(np.float32))
            parts["depth"].append(depths[bins[depth]])
            parts["score"].append(np.float32(_SCORE_OFFSET) - friction[usable][keep])
            found += len(point)
        counts.append(found)
    shapes = {"point": (0, 3), "rotation": (0, 3, 3)}
    merged = {
        name: (
            np.concatenate(chunks) if chunks else np.zeros(shapes.get(name, (0,)), dtype=np.float32)
        )
        for name, chunks in parts.items()
    }
    return SceneGrasps(counts=np.array(counts, dtype=np.int64), **merged)


@dataclass(frozen=True)
class SceneModels:
    """Vertices of the object models of a scene, instance by instance.

    Attributes:
        vertices: Per instance, ``(N, 3)`` float64 in the object frame.
    """

    vertices: Sequence[np.ndarray]

    def __len__(self) -> int:
        return sum(len(part) for part in self.vertices)

    def placed(self, index: np.ndarray, poses: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
        """Rows ``index`` of all vertices concatenated, in the camera frame.

        Args:
            index: ``(M,)`` rows of the concatenation of :attr:`vertices`.
            poses: ``(object_id, (4, 4) pose)`` of every instance, in the same order.

        Returns:
            ``(M, 3)`` float64, in the order of ``index``.
        """
        if len(poses) != len(self.vertices):
            raise ContractError(f"{len(poses)} object poses for {len(self.vertices)} object models")
        offsets = np.concatenate([[0], np.cumsum([len(part) for part in self.vertices])])
        owner = np.searchsorted(offsets, index, side="right") - 1
        out = np.empty((len(index), 3), dtype=np.float64)
        for instance, (part, (_, pose)) in enumerate(zip(self.vertices, poses)):
            rows = np.flatnonzero(owner == instance)
            if len(rows):
                out[rows] = transform_points(part[index[rows] - offsets[instance]], pose)
        return out


# --------------------------------------------------------------------------- per frame


@dataclass(frozen=True)
class GraspSet:
    """Grasps of one frame in the camera frame (GraspNet conventions), float64.

    Attributes:
        translation: ``(G, 3)`` metres.
        rotation: ``(G, 3, 3)``; column 0 is the approach axis, column 1 the closing
            axis, column 2 the height axis.
        width, depth, score: ``(G,)``.
        object_id: ``(G,)`` int64.
    """

    translation: np.ndarray
    rotation: np.ndarray
    width: np.ndarray
    depth: np.ndarray
    score: np.ndarray
    object_id: np.ndarray

    def __len__(self) -> int:
        return len(self.translation)

    def take(self, index: np.ndarray) -> GraspSet:
        """The grasps at ``index`` (rows or a boolean mask), in that order."""
        return GraspSet(**{name: value[index] for name, value in vars(self).items()})

    def moved(self, translation: np.ndarray) -> GraspSet:
        """The same grasps at other positions."""
        return GraspSet(**{**vars(self), "translation": translation})


def camera_grasps(
    scene: SceneGrasps, poses: Sequence[tuple[int, np.ndarray]], approach_threshold: float
) -> GraspSet:
    """Step 2: place a scene's candidates in a frame and keep those facing away.

    The placed grasps are rounded to single precision, as the GraspNet grasp array
    holds them; everything after this step is double precision on those values.

    Args:
        scene: Candidates in object frames.
        poses: ``(object_id, (4, 4) object-to-camera pose)`` of every instance, in the
            order of the scene's collision labels.
        approach_threshold: See :class:`ViewGraspConfig`.
    """
    if len(poses) != len(scene.counts):
        raise ContractError(
            f"{len(poses)} object poses for the {len(scene.counts)} labelled instances of the scene"
        )
    translation = np.empty((len(scene), 3), dtype=np.float32)
    rotation = np.empty((len(scene), 3, 3), dtype=np.float32)
    object_id = np.empty(len(scene), dtype=np.int64)
    start = 0
    for count, (identity, pose) in zip(scene.counts.tolist(), poses):
        rows = slice(start, start + count)
        pose = np.asarray(pose, dtype=np.float64)
        translation[rows] = transform_points(scene.point[rows], pose)
        for axis in range(3):  # each frame axis is a direction: rotate it like a point
            rotation[rows, :, axis] = transform_points(scene.rotation[rows, :, axis], pose[:3, :3])
        object_id[rows] = identity
        start += count
    facing = rotation[:, 2, 0].astype(np.float64) > approach_threshold
    return GraspSet(
        translation=translation[facing].astype(np.float64),
        rotation=rotation[facing].astype(np.float64),
        width=scene.width[facing].astype(np.float64),
        depth=scene.depth[facing].astype(np.float64),
        score=scene.score[facing].astype(np.float64),
        object_id=object_id[facing],
    )


def refine_centers(
    translation: np.ndarray,
    rotation: np.ndarray,
    width: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
    *,
    height: float,
    finger_length: float,
) -> np.ndarray:
    """Step 4: centre each grasp on the points between its fingers.

    A point ``p`` has the grasp-frame coordinates ``(p - centre) @ rotation``, each
    evaluated as ``(dx * r0 + dy * r1) + dz * r2``. It is between the fingers when
    ``depth - finger_length < x < depth``, ``|y| < width / 2`` and ``|z| < height / 2``
    (all strict). The centre moves along the closing axis by the mean of the smallest
    and the largest ``y`` of those points; without such a point it stays.

    The search is exhaustive in effect but not in cost. Grasps annotated at one label
    point share their centre, so they are handled together, and only the points inside
    the sphere around the largest of their regions are transformed. The sphere is wider
    than the region by more than any rounding: it prunes and never decides.

    Args:
        translation: ``(G, 3)`` float64 grasp centres.
        rotation: ``(G, 3, 3)`` float64 grasp frames.
        width, depth: ``(G,)`` float64.
        points: ``(N, 3)`` float64, same frame as the grasps.
        height: Finger thickness, metres.
        finger_length: See :class:`ViewGraspConfig`.

    Returns:
        ``(G, 3)`` float64 refined centres.
    """
    refined = np.array(translation, dtype=np.float64)
    if len(refined) == 0 or len(points) == 0:
        return refined
    back, front = depth - finger_length, depth
    reach = np.sqrt(
        np.maximum(np.abs(back), np.abs(front)) ** 2 + (width / 2) ** 2 + (height / 2) ** 2
    )
    reach = reach * (1.0 + 1e-9) + 1e-12
    # Sorted along the axis of largest extent, the candidates of a centre are one run; the
    # axis of second-largest extent thins the run before the sphere is tested. Coordinates
    # are kept as separate contiguous columns: the loop below selects from them often.
    other, axis = (int(k) for k in np.argsort(points.max(axis=0) - points.min(axis=0))[1:])
    order = np.argsort(points[:, axis], kind="stable")
    columns = [np.ascontiguousarray(points[order, k]) for k in range(3)]

    centres, group = np.unique(refined, axis=0, return_inverse=True)
    members = np.argsort(group.reshape(-1), kind="stable")
    bounds = np.searchsorted(group.reshape(-1)[members], np.arange(len(centres) + 1))
    for index, centre in enumerate(centres):
        rows = members[bounds[index] : bounds[index + 1]]
        radius = reach[rows].max()
        first = np.searchsorted(columns[axis], centre[axis] - radius, side="left")
        last = np.searchsorted(columns[axis], centre[axis] + radius, side="right")
        run = np.flatnonzero(np.abs(columns[other][first:last] - centre[other]) <= radius) + first
        dx, dy, dz = (columns[k][run] - centre[k] for k in range(3))
        near = (dx * dx + dy * dy) + dz * dz <= radius * radius
        if not near.any():
            continue
        dx, dy, dz = dx[near], dy[near], dz[near]
        frames = rotation[rows]
        x, y, z = (
            (dx * frames[:, 0, k, None] + dy * frames[:, 1, k, None]) + dz * frames[:, 2, k, None]
            for k in range(3)
        )  # each (grasps of the centre, candidate points)
        between = (
            (z > -height / 2)
            & (z < height / 2)
            & (x > back[rows, None])
            & (x < front[rows, None])
            & (y > -width[rows, None] / 2)
            & (y < width[rows, None] / 2)
        )
        found = between.any(axis=1)
        if not found.any():
            continue
        low = np.where(between, y, np.inf).min(axis=1)[found]
        high = np.where(between, y, -np.inf).max(axis=1)[found]
        low = (low - _REFERENCE_SHIFT) + _REFERENCE_SHIFT
        high = (high + _REFERENCE_SHIFT) - _REFERENCE_SHIFT
        owner = rows[found]
        refined[owner] = translation[owner] + ((low + high) / 2)[:, None] * _dual_closing_axis(
            rotation[owner]
        )
    return refined


def _dual_closing_axis(rotation: np.ndarray) -> np.ndarray:
    """``(G, 3)`` direction along which only the grasp-frame ``y`` of a point changes.

    For an orthonormal frame this is the closing axis itself. Label frames are
    orthonormal to single precision only, and the refined centre is defined by its
    grasp-frame coordinates ``(0, middle, 0)``: mapping them back needs the inverse of
    the frame's transpose, whose second column is ``(height axis x approach axis) / det``.
    """
    approach, closing, height = rotation[:, :, 0], rotation[:, :, 1], rotation[:, :, 2]
    dual = np.stack(
        [
            height[:, 1] * approach[:, 2] - height[:, 2] * approach[:, 1],
            height[:, 2] * approach[:, 0] - height[:, 0] * approach[:, 2],
            height[:, 0] * approach[:, 1] - height[:, 1] * approach[:, 0],
        ],
        axis=1,
    )
    volume = (dual[:, 0] * closing[:, 0] + dual[:, 1] * closing[:, 1]) + dual[:, 2] * closing[:, 2]
    return dual / volume[:, None]


def suppress(
    grasps: GraspSet, tie_order: np.ndarray, *, distance: float, angle: float
) -> np.ndarray:
    """Step 5: rows of the grasps that survive greedy suppression, in visiting order.

    Grasps are visited in descending score order; equal scores in the order in which
    they occur in ``tie_order``.

    Args:
        grasps: The grasps of a frame.
        tie_order: ``(G,)`` a permutation of the rows.
        distance: Metres, strict.
        angle: Radians, strict.
    """
    from grasppanda.ops import grasp_nms

    count = len(grasps)
    tie_order = np.asarray(tie_order, dtype=np.int64)
    if tie_order.shape != (count,) or not np.array_equal(np.sort(tie_order), np.arange(count)):
        raise ContractError(f"tie_order must be a permutation of the {count} grasps")
    visit = tie_order[np.argsort(-grasps.score[tie_order], kind="stable")]
    rank = np.empty(count, dtype=np.float64)
    rank[visit] = np.arange(count)
    keep = grasp_nms(
        torch.from_numpy(grasps.translation),
        torch.from_numpy(grasps.rotation),
        torch.from_numpy(-rank),  # distinct scores: the operator visits exactly `visit`
        translation_threshold=distance,
        rotation_threshold=angle,
    )
    return keep.numpy()


def grasp_rectangles(
    grasps: GraspSet, intrinsics: tuple[float, float, float, float], height: float
) -> dict[str, np.ndarray]:
    """Step 6: the image-space description of camera-frame grasps.

    With ``R`` the grasp frame and the grasp moved to ``(0, 0, z)``, the jaw point is
    ``o = (0, 0, z) - width / 2 * R[:, 1]`` and the two edge points are
    ``o ± height / 2 * R[:, 2]``. Then, with ``(u, v)`` projections::

        theta  = atan2(v_c - v_o, u_o - u_c)
        gamma  = asin((o_z - z) * 2 / width)              = asin(-R[2, 1])
        beta   = asin((e+_z - e-_z) / cos(gamma) / height) = asin(R[2, 2] / cos(gamma))
        width  = 2 * |(u_o, v_o) - (u_c, v_c)| / cos(gamma)
        height = |proj(e+) - proj(e-)| * 2 / (1 + sqrt(1 - (cos(gamma) * sin(beta))**2))

    The reference keeps the projected rectangle in single precision, so ``theta``,
    ``width`` and ``height`` are single-precision results here too.

    Args:
        grasps: Grasps in the camera frame.
        intrinsics: ``(fx, fy, cx, cy)`` of the full-resolution image.
        height: Finger thickness of the gripper, metres.

    Returns:
        ``centers (G, 2)``, ``depth``, ``theta``, ``gamma``, ``beta``, ``width`` and
        ``height`` as float64 arrays. A grasp at zero depth has non-finite entries.
    """
    fx, fy, cx, cy = intrinsics
    translation, rotation = grasps.translation, grasps.rotation
    z = translation[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        centers = np.stack(
            [translation[:, 0] / z * fx + cx, translation[:, 1] / z * fy + cy], axis=-1
        )
        on_axis = np.zeros_like(translation)
        on_axis[:, 2] = z
        # Rotated local points (0, -w/2, 0) and (0, -w/2, ±h/2), then the translation: the
        # order of a matrix product followed by a sum, which the angles below depend on.
        closing = -(grasps.width / 2)[:, None] * rotation[:, :, 1]
        rise = (height / 2) * rotation[:, :, 2]
        jaw = closing + on_axis
        upper = (closing + rise) + on_axis
        lower = (closing - rise) + on_axis

        def project(points: np.ndarray) -> np.ndarray:
            return np.stack(
                [points[:, 0] / points[:, 2] * fx + cx, points[:, 1] / points[:, 2] * fy + cy],
                axis=-1,
            )

        centre32 = np.array([cx, cy], dtype=np.float32)
        offset = centre32 - project(jaw).astype(np.float32)  # (du, dv), single precision
        theta = np.arctan2(offset[:, 1], -offset[:, 0])
        gamma = np.arcsin(np.clip((jaw[:, 2] - z) * 2 / grasps.width, -1, 1))
        beta = np.arcsin(np.clip((upper[:, 2] - lower[:, 2]) / np.cos(gamma) / height, -1, 1))
        width = np.sqrt(np.square(offset[:, 0]) + np.square(offset[:, 1])) * np.float32(2)
        width = (width / np.cos(gamma)).astype(np.float32)
        edge = np.linalg.norm(project(upper) - project(lower), axis=-1).astype(np.float32)
        correction = 2 / (1 + np.sqrt(1 - np.square(np.cos(gamma) * np.sin(beta))))
        edge = (edge * correction).astype(np.float32)
    return {
        "centers": centers,
        "depth": z.copy(),
        "theta": theta.astype(np.float64),
        "gamma": gamma,
        "beta": beta,
        "width": width.astype(np.float64),
        "height": edge.astype(np.float64),
    }


@dataclass(frozen=True)
class ViewGraspTrace:
    """Intermediate results of one frame, for shard records and parity tests.

    Attributes:
        candidates: Grasps of the frame after step 2.
        model_points: ``(model_points, 3)`` the drawn model vertices, camera frame.
        refined: The (sub-sampled) grasps with refined centres, after step 4.
        survivors: Rows of ``refined`` that survive suppression, in visiting order.
        inside: Per survivor, whether its centre pixel is inside the image.
        finite: Per survivor, whether its image-space description is finite.
        visible: Per survivor, whether it passes the visibility test; ``None`` when the
            test was not made.
    """

    candidates: GraspSet
    model_points: np.ndarray
    refined: GraspSet
    survivors: np.ndarray
    inside: np.ndarray
    finite: np.ndarray
    visible: np.ndarray | None


def frame_view_grasps(
    scene: SceneGrasps,
    models: SceneModels,
    poses: Sequence[tuple[int, np.ndarray]],
    config: ViewGraspConfig,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
    draws: Draws,
    depth_points: np.ndarray | None = None,
    gripper: Gripper | None = None,
) -> tuple[dict[str, np.ndarray], ViewGraspTrace]:
    """The labels of one frame; the definition in the module docstring, step by step.

    Args:
        scene: The scene's candidates (:func:`scene_grasps`).
        models: The scene's object models.
        poses: ``(object_id, pose)`` of every instance for this frame.
        config: Generator parameters.
        intrinsics: ``(fx, fy, cx, cy)`` to project with.
        image_size: ``(H, W)`` of the full-resolution image.
        draws: The arbitrary choices of this frame.
        depth_points: ``(N, 3)`` points of the frame's depth cloud; required by
            ``config.visibility_filter`` and ignored without it.
        gripper: Owner of the finger thickness; the benchmark gripper by default.

    Returns:
        The labels as ``{field: float64 or int64 array}`` (see :data:`FIELDS`) and the
        intermediate results.
    """
    # The GraspNet grasp array stores the finger thickness in single precision.
    height = float(np.float32((gripper or Gripper()).finger_height))
    candidates = camera_grasps(scene, poses, config.approach_threshold)
    grasps = candidates
    if len(grasps) > config.max_grasps:
        grasps = grasps.take(_checked_draw(draws, "grasps", len(grasps), config.max_grasps))

    if len(models) < config.model_points:
        raise AssetError(
            f"the object models of the scene have {len(models)} vertices in total, fewer than "
            f"the {config.model_points} that `view_grasps.model_points` draws per frame"
        )
    model_index = _checked_draw(draws, "model_points", len(models), config.model_points)
    model_points = models.placed(model_index, poses)
    grasps = grasps.moved(
        refine_centers(
            grasps.translation,
            grasps.rotation,
            grasps.width,
            grasps.depth,
            model_points,
            height=height,
            finger_length=config.finger_length,
        )
    )

    ties = _checked_draw(draws, "nms_ties", len(grasps), len(grasps))
    survivors = suppress(
        grasps, ties, distance=config.nms_distance, angle=math.radians(config.nms_angle)
    )
    kept = grasps.take(survivors)
    labels = grasp_rectangles(kept, intrinsics, height)
    labels["score"], labels["object_id"] = kept.score, kept.object_id
    centers = labels["centers"]
    finite = np.isfinite(_pack(labels, "float64", finite=False)).all(axis=1)
    with np.errstate(invalid="ignore"):
        inside = (
            (centers.min(axis=1) > 0)
            & (centers[:, 0] < image_size[1])
            & (centers[:, 1] < image_size[0])
        )
    visible = None
    if config.visibility_filter:
        if depth_points is None:
            raise ContractError(
                "`view_grasps.visibility_filter` needs the depth cloud of the frame"
            )
        visible = _visible(kept.translation, depth_points, config, draws)
    keep = finite & inside if visible is None else finite & inside & visible
    trace = ViewGraspTrace(candidates, model_points, grasps, survivors, inside, finite, visible)
    return {name: labels[name][keep] for name in FIELDS}, trace


def _visible(
    centers: np.ndarray, depth_points: np.ndarray, config: ViewGraspConfig, draws: Draws
) -> np.ndarray:
    """Whether each grasp centre is within the visibility distance of the drawn points."""
    from scipy.spatial import cKDTree

    if len(depth_points) < config.visibility_points:
        raise AssetError(
            f"the frame has {len(depth_points)} pixels with depth, fewer than the "
            f"{config.visibility_points} that `view_grasps.visibility_points` draws"
        )
    index = _checked_draw(draws, "depth_points", len(depth_points), config.visibility_points)
    if len(centers) == 0:
        return np.zeros(0, dtype=bool)
    nearest, _ = cKDTree(depth_points[index]).query(centers)
    return nearest < config.visibility_distance


# ----------------------------------------------------------------------------- storage


class ViewGrasps:
    """Per-view grasp labels in the asset store: one compressed archive per scene.

    Args:
        asset: The asset directory.
    """

    KIND = "view_grasps"
    GENERATOR = "grasppanda.view_grasps"
    VERSION = 1

    def __init__(self, asset: Asset) -> None:
        self.asset = asset

    @staticmethod
    def spec(dataset: str, camera: str, config: ViewGraspConfig, labels_hash: str) -> AssetSpec:
        """The asset a dataset configured this way reads."""
        about = {
            "layout": "scene_<id>.npz, one (L, 10) array per frame named by frame id; an "
            "empty array is a frame without labels",
            "columns": ", ".join(COLUMNS),
            "u, v, width, height": "pixels of the full-resolution image",
            "depth": "camera z of the grasp centre in metres (the HGGD converter stores "
            "millimetres as center_z_depths)",
            "theta, gamma, beta": "radians, as the HGGD converter stores them (theta in "
            "(-pi, pi], not folded)",
            "score": "1.1 - friction coefficient of the label grasp",
            "object_id": "dataset id of the object, stored as a float",
            "order": "descending score; equal scores in the random order they were visited "
            "by the suppression",
        }
        if config.intrinsics == "reference":
            constants = REFERENCE_INTRINSICS.get(camera)
            about["intrinsics"] = (
                f"fx, fy, cx, cy = {constants} for every frame"
                if constants
                else "no reference constants exist for this camera"
            )
        else:
            about["intrinsics"] = "the calibration of each frame"
        return AssetSpec(
            dataset=dataset,
            kind=ViewGrasps.KIND,
            generator=ViewGrasps.GENERATOR,
            version=ViewGrasps.VERSION,
            parameters={"camera": camera, **asdict(config)},
            inputs={"dataset": dataset, "grasp_labels": labels_hash},
            about=about,
        )

    def shard_file(self, scene: int) -> Path:
        return self.asset.path / f"{scene_shard(scene)}.npz"

    def write_scene(
        self, scene: int, frames: Iterable[tuple[int, dict[str, np.ndarray]]], count: int
    ) -> None:
        """Write a scene's archive atomically from ``(frame, labels)`` pairs.

        The shard is not marked complete: several processes may write scenes at once,
        and only the one that owns the manifest calls :meth:`mark_complete`.
        """
        dtype = self.asset.spec.parameters["dtype"]
        written = 0
        with (
            atomic_path(self.shard_file(scene)) as temporary,
            zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive,
        ):
            for frame, labels in frames:
                with archive.open(f"{frame:04d}.npy", "w") as member:
                    np.lib.format.write_array(member, _pack(labels, dtype), allow_pickle=False)
                written += 1
            if written != count:
                raise AssetError(f"scene {scene}: wrote {written} frames, expected {count}")

    def mark_complete(self, scene: int, record: dict[str, int]) -> None:
        """Record a written scene, with what its generation reported, in the manifest."""
        file = self.shard_file(scene)
        if not file.is_file():
            raise AssetError(f"cannot mark scene {scene} complete: {file} is missing")
        self.asset.mark_complete(scene_shard(scene), **record, bytes=file.stat().st_size)

    def read(self, key: SampleKey) -> dict[str, np.ndarray]:
        """The labels of one frame as ``{field: array}``; see :data:`FIELDS`.

        Float arrays have the asset's ``dtype``, ``object_id`` is int64. A frame without
        labels yields arrays of length zero; a frame that was never generated is an error.
        """
        spec = self.asset.spec
        if key.dataset != spec.dataset or key.camera != spec.parameters["camera"]:
            raise AssetError(
                f"{key} was asked from the view-grasp asset of {spec.dataset} / "
                f"{spec.parameters['camera']} at {self.asset.path}"
            )
        if not self.asset.is_complete(scene_shard(key.scene)):
            raise AssetError(
                f"view grasps of {key} are not in the asset at {self.asset.path}: scene "
                f"{key.scene} has not been generated"
            )
        file = self.shard_file(key.scene)
        with np.load(file) as archive:
            name = f"{key.frame:04d}"
            if name not in archive.files:
                raise AssetError(f"{file} has no labels for frame {key.frame}")
            packed = archive[name]
        if (
            packed.ndim != 2
            or packed.shape[1] != len(COLUMNS)
            or packed.dtype != spec.parameters["dtype"]
        ):
            raise AssetError(
                f"{file}: frame {key.frame} is a {packed.dtype} array of shape {packed.shape}; "
                f"expected (L, {len(COLUMNS)}) {spec.parameters['dtype']}. Rebuild the asset."
            )
        return _unpack(packed)

    def record(self) -> dict[str, Any]:
        return self.asset.record()


def _pack(labels: dict[str, np.ndarray], dtype: str, *, finite: bool = True) -> np.ndarray:
    """The ``(L, 10)`` array a frame is stored as; see :data:`COLUMNS`."""
    if set(labels) != set(FIELDS):
        raise ContractError(
            f"view grasps of a frame have exactly the fields {FIELDS}, got {sorted(labels)}"
        )
    centers = np.asarray(labels["centers"])
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ContractError(f"view-grasp field 'centers' must be (L, 2), got {centers.shape}")
    columns = [centers]
    for name in FIELDS[1:]:
        values = np.asarray(labels[name])
        if values.shape != (len(centers),):
            raise ContractError(
                f"view-grasp field '{name}' has shape {values.shape} for {len(centers)} grasps"
            )
        columns.append(values[:, None])
    packed = np.concatenate(columns, axis=1).astype(dtype)
    if finite and not np.isfinite(packed).all():
        raise ContractError("view grasps must be finite; drop degenerate labels before storing")
    return packed


def _unpack(packed: np.ndarray) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {"centers": np.ascontiguousarray(packed[:, :2])}
    for column, name in enumerate(FIELDS[1:], start=2):
        labels[name] = np.ascontiguousarray(packed[:, column])
    labels["object_id"] = np.rint(labels["object_id"]).astype(np.int64)
    return labels


# -------------------------------------------------------------------------- generation


class SceneReader(Protocol):
    """What the generator needs from a dataset."""

    name: str
    camera: str
    root: Path
    #: ``(H, W)`` of every image of the camera.
    image_size: tuple[int, int]
    pose_sources: tuple[str, ...]

    def scene_keys(self, scene: int) -> list[SampleKey]:
        """Every frame of a scene for the dataset's camera."""

    def frame(self, key: SampleKey) -> Frame:
        """The preprocessed frame; read only for frame intrinsics and visibility."""

    def object_poses(self, key: SampleKey, source: str) -> list[tuple[int, np.ndarray]]:
        """``(object_id, pose)`` of every instance, in the order of the collision labels."""

    def scene_labels(self, scene: int) -> list[StoredGraspLabels]:
        """Grasp labels of every instance of a scene, in the same order."""


def object_model_file(dataset: SceneReader, object_id: int) -> Path:
    """The vertex file of an object model.

    A reader that knows its own layout says so with an ``object_model_file`` method;
    otherwise the GraspNet-1Billion layout ``models/<id>/nontextured.ply`` is assumed.
    """
    locate = getattr(dataset, "object_model_file", None)
    if locate is not None:
        return Path(locate(object_id))
    return Path(dataset.root) / "models" / f"{object_id:03d}" / "nontextured.ply"


def scene_models(dataset: SceneReader, object_ids: Sequence[int]) -> SceneModels:
    """Read the model vertices of a scene's objects, in the given order."""
    # The evaluator owns the PLY reader; imported here because it is only needed to generate.
    from grasppanda.evaluation.graspnet1b.models import read_ply_vertices

    vertices = []
    for object_id in object_ids:
        file = object_model_file(dataset, object_id)
        if not file.is_file():
            raise FileNotFoundError(
                f"model of object {object_id} not found: {file}. The centre refinement of the "
                "view grasps needs the object models of the dataset."
            )
        vertices.append(read_ply_vertices(file))
    return SceneModels(vertices)


def _reference_intrinsics(dataset: SceneReader) -> tuple[float, float, float, float]:
    if dataset.camera not in REFERENCE_INTRINSICS:
        raise ContractError(
            "`view_grasps.intrinsics: reference` has constants for the cameras "
            f"{', '.join(REFERENCE_INTRINSICS)} only, not for '{dataset.camera}' of "
            f"{dataset.name}; use `intrinsics: frame`"
        )
    return REFERENCE_INTRINSICS[dataset.camera]


def generate_view_grasps(
    dataset: SceneReader,
    store: ViewGrasps,
    scene: int,
    config: ViewGraspConfig,
    progress: Callable[[SampleKey], None] | None = None,
) -> dict[str, int]:
    """Generate and write the labels of one scene; the object-level part runs once.

    Returns:
        The shard record (frames written, labels kept and what each step removed) for
        :meth:`ViewGrasps.mark_complete`, which the caller invokes.
    """
    if config.poses not in dataset.pose_sources:
        raise ContractError(
            f"`view_grasps.poses` must be one of {', '.join(dataset.pose_sources)} for "
            f"{dataset.name}, got {config.poses!r}"
        )
    constants = _reference_intrinsics(dataset) if config.intrinsics == "reference" else None
    candidates = scene_grasps(dataset.scene_labels(scene), config)
    keys = dataset.scene_keys(scene)
    record = {
        "frames": len(keys),
        "candidates": len(candidates),
        "grasps": 0,
        "outside_image": 0,
        "degenerate": 0,
        "invisible": 0,
    }
    models: SceneModels | None = None

    def frames() -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        nonlocal models
        for key in keys:
            poses = dataset.object_poses(key, config.poses)
            if models is None:
                models = scene_models(dataset, [object_id for object_id, _ in poses])
            intrinsics, depth_points = constants, None
            if config.visibility_filter or intrinsics is None:
                frame = dataset.frame(key)
                matrix = frame.intrinsics
                intrinsics = intrinsics or (
                    float(matrix[0, 0]),
                    float(matrix[1, 1]),
                    float(matrix[0, 2]),
                    float(matrix[1, 2]),
                )
                if config.visibility_filter:
                    cloud, _ = frame.require_cloud("the visibility filter of the view grasps")
                    depth_points = cloud[frame.depth > 0]
            found, trace = frame_view_grasps(
                candidates,
                models,
                poses,
                config,
                intrinsics,
                tuple(dataset.image_size),
                SeededDraws(config.seed, key),
                depth_points,
            )
            record["grasps"] += len(found["score"])
            record["degenerate"] += int(np.count_nonzero(~trace.finite))
            record["outside_image"] += int(np.count_nonzero(trace.finite & ~trace.inside))
            if trace.visible is not None:
                record["invisible"] += int(
                    np.count_nonzero(trace.finite & trace.inside & ~trace.visible)
                )
            yield key.frame, found
            if progress is not None:
                progress(key)

    store.write_scene(scene, frames(), len(keys))
    return record
