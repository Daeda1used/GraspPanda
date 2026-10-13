"""Per-view grasp labels (the label asset HGGD trains on).

The definition is checked step by step against oracles written here with plain loops,
on synthetic labels; the storage and the ``prepare`` hook run on the tiny dataset tree
of ``test_data_graspnet1b``, to which object models are added.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from grasppanda.cli.main import main
from grasppanda.datasets.assets import Asset, AssetError
from grasppanda.datasets.labels import LabelGrid, StoredGraspLabels
from grasppanda.datasets.view_grasps import (
    COLUMNS,
    FIELDS,
    REFERENCE_INTRINSICS,
    GraspSet,
    SceneModels,
    SeededDraws,
    ViewGraspConfig,
    ViewGrasps,
    camera_grasps,
    frame_view_grasps,
    generate_view_grasps,
    grasp_rectangles,
    reference_views,
    refine_centers,
    scene_grasps,
    scene_shard,
    suppress,
)
from grasppanda.ops import approach_to_rotation, fibonacci_views
from grasppanda.structures import ContractError, SampleKey

from .test_data_graspnet1b import FRAMES, POINTS, SCENES, SIZE, TEST, build, tree  # noqa: F401

GRID = LabelGrid(num_view=7, num_angle=4, num_depth=3)  # depths 0.01, 0.02, 0.03
FRICTION = np.array([-1.0, 0.1, 0.2, 0.3, 0.5, 1.0], dtype=np.float32)
HEIGHT = float(np.float32(0.02))
#: Where synthetic grasps and points are: half a metre in front of the camera.
AHEAD = np.array([0.0, 0.0, 0.5])
#: A configuration that leaves something to label on a few dozen synthetic grasps.
SMALL = ViewGraspConfig(max_grasps=12, model_points=80, visibility_points=30)
#: The `view_grasps` section used on the tiny dataset tree (24 x 32 images).
VIEW = {"intrinsics": "frame", "model_points": 60, "visibility_points": 100}


# ------------------------------------------------------------------------------ oracles


def random_rotations(rng: np.random.Generator, count: int) -> np.ndarray:
    """Proper rotations from QR decompositions of Gaussian matrices."""
    q, r = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[:, :, 2] *= np.linalg.det(q)[:, None]
    return q


def make_labels(rng: np.random.Generator, points: int, grid: LabelGrid = GRID):
    """Dense labels of one object instance and its collision grid, unpacked."""
    arrays = (
        rng.uniform(-0.03, 0.03, (points, 3)).astype(np.float32),
        rng.choice(FRICTION, (points, *grid.shape)),
        rng.uniform(0.0, 0.2, (points, *grid.shape)).astype(np.float32),
    )
    collision = rng.random((points, *grid.shape)) < 0.3
    packed = np.packbits(collision.reshape(points, -1), axis=1)
    return StoredGraspLabels(*arrays, packed, grid), collision


def make_scene(seed: int, sizes=(9, 6, 11)):
    rng = np.random.default_rng(seed)
    made = [make_labels(rng, count) for count in sizes]
    labels, collisions = [m[0] for m in made], [m[1] for m in made]
    poses = []
    for index in range(len(sizes)):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = random_rotations(rng, 1)[0]
        pose[:3, 3] = (0.08 * index - 0.08, 0.03 * index, 0.45 + 0.02 * index)
        poses.append((20 + index, pose))
    models = SceneModels([rng.uniform(-0.035, 0.035, (70 + 10 * i, 3)) for i in range(len(sizes))])
    return labels, collisions, poses, models


def frame_of(view: np.ndarray, angle: float) -> np.ndarray:
    """Textbook gripper frame of a label bin, in double precision."""
    approach = -view.astype(np.float64)
    approach /= np.linalg.norm(approach)
    side = np.array([-approach[1], approach[0], 0.0])
    side = np.array([0.0, 1.0, 0.0]) if np.linalg.norm(side) == 0 else side / np.linalg.norm(side)
    up = np.cross(approach, side)
    spin = np.array(
        [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
    )
    return np.stack([approach, side, up], axis=1) @ spin


def expected_candidates(labels, collisions, config: ViewGraspConfig) -> list[dict]:
    """Step 1 with loops, in label order."""
    rows = []
    for instance, (source, collision) in enumerate(zip(labels, collisions)):
        grid = source.grid
        views = reference_views(grid.num_view)
        for p in range(source.num_points):
            for v in range(grid.num_view):
                for a in range(grid.num_angle):
                    for d in range(grid.num_depth):
                        friction = source.scores[p, v, a, d]
                        width = float(source.widths[p, v, a, d])
                        depth = float(np.float32(grid.depths[d]))
                        if not 0 < friction <= np.float32(config.friction_threshold):
                            continue
                        if collision[p, v, a, d]:
                            continue
                        if not config.min_width < width < config.max_width:
                            continue
                        if not config.min_depth < depth < config.max_depth:
                            continue
                        rows.append(
                            {
                                "instance": instance,
                                "point": source.points[p],
                                "rotation": frame_of(views[v], grid.angles[a]),
                                "width": width,
                                "depth": depth,
                                "score": float(np.float32(1.1) - friction),
                            }
                        )
    return rows


def expected_camera(rows: list[dict], poses, threshold: float) -> list[dict]:
    """Step 2 with loops."""
    out = []
    for row in rows:
        object_id, pose = poses[row["instance"]]
        pose = pose.astype(np.float64)
        rotation = (pose[:3, :3] @ row["rotation"]).astype(np.float32).astype(np.float64)
        if rotation[2, 0] > threshold:
            translation = pose[:3, :3] @ row["point"].astype(np.float64) + pose[:3, 3]
            out.append(
                {
                    **row,
                    "object_id": object_id,
                    "translation": translation.astype(np.float32).astype(np.float64),
                    "camera_rotation": rotation,
                }
            )
    return out


def expected_refined(translation, rotation, width, depth, points, finger_length=0.04):
    """Step 4 with an exhaustive search and an explicit inverse."""
    out = translation.copy()
    for g in range(len(translation)):
        between = []
        for point in points:
            delta = point - translation[g]
            x, y, z = (
                (delta[0] * rotation[g, 0, k] + delta[1] * rotation[g, 1, k])
                + delta[2] * rotation[g, 2, k]
                for k in range(3)
            )
            if (
                -HEIGHT / 2 < z < HEIGHT / 2
                and depth[g] - finger_length < x < depth[g]
                and -width[g] / 2 < y < width[g] / 2
            ):
                between.append(y)
        if between:
            middle = (min(between) + max(between)) / 2
            # the point whose grasp-frame coordinates are (0, middle, 0)
            out[g] = translation[g] + np.linalg.solve(rotation[g].T, np.array([0, middle, 0]))
    return out


def expected_survivors(translation, rotation, score, tie_order, distance, angle) -> list[int]:
    """Step 5: greedy suppression with a double loop."""
    visit = sorted(range(len(score)), key=lambda g: (-score[g], list(tie_order).index(g)))
    kept: list[int] = []
    for g in visit:
        for k in kept:
            cosine = (np.trace(rotation[k].T @ rotation[g]) - 1) / 2
            turned = math.acos(min(1.0, max(-1.0, cosine)))
            if np.linalg.norm(translation[k] - translation[g]) < distance and turned < angle:
                break
        else:
            kept.append(g)
    return kept


def expected_rectangles(translation, rotation, width, intrinsics) -> dict[str, np.ndarray]:
    """Step 6 in double precision, from the matrix form of the definition."""
    fx, fy, cx, cy = intrinsics
    out = {name: [] for name in ("centers", "depth", "theta", "gamma", "beta", "width", "height")}
    for t, frame, w in zip(translation, rotation, width):
        out["centers"].append([t[0] / t[2] * fx + cx, t[1] / t[2] * fy + cy])
        out["depth"].append(t[2])
        local = np.array(
            [[0, 0, 0], [0, -w / 2, 0], [0, -w / 2, HEIGHT / 2], [0, -w / 2, -HEIGHT / 2]]
        )
        centre, jaw, upper, lower = local @ frame.T + np.array([0, 0, t[2]])
        pixel = [
            np.array([p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy]) for p in (jaw, upper, lower)
        ]
        gamma = math.asin(np.clip((jaw[2] - centre[2]) * 2 / w, -1, 1))
        beta = math.asin(np.clip((upper[2] - lower[2]) / math.cos(gamma) / HEIGHT, -1, 1))
        out["theta"].append(math.atan2(cy - pixel[0][1], pixel[0][0] - cx))
        out["gamma"].append(gamma)
        out["beta"].append(beta)
        out["width"].append(2 * np.linalg.norm(pixel[0] - [cx, cy]) / math.cos(gamma))
        fold = 2 / (1 + math.sqrt(1 - (math.cos(gamma) * math.sin(beta)) ** 2))
        out["height"].append(np.linalg.norm(pixel[1] - pixel[2]) * fold)
    shapes = {"centers": (len(translation), 2)}
    return {
        name: np.array(values, dtype=np.float64).reshape(shapes.get(name, (len(translation),)))
        for name, values in out.items()
    }


class Recorder:
    """Draws that remember what they returned, for an oracle to replay."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.log: dict[str, tuple[int, np.ndarray]] = {}

    def sample(self, what: str, available: int, count: int) -> np.ndarray:
        index = self.rng.permutation(available)[:count]
        self.log[what] = (available, index)
        return index


class Fixed:
    """Draws that return what they were given."""

    def __init__(self, **draws: np.ndarray) -> None:
        self.draws = draws

    def sample(self, what: str, available: int, count: int) -> np.ndarray:
        return self.draws[what]


def grasp_set(rng: np.random.Generator, count: int, spread: float = 0.04) -> GraspSet:
    """Random grasps around (0, 0, 0.5) whose approach axis points away from the camera."""
    rotation = random_rotations(rng, 4 * count + 8)
    rotation = rotation[rotation[:, 2, 0] > 0.05][:count]
    return GraspSet(
        translation=rng.uniform(-spread, spread, (count, 3)) + AHEAD,
        rotation=rotation,
        width=rng.uniform(0.03, 0.15, count),
        depth=np.full(count, float(np.float32(0.02))),
        score=rng.choice(np.float32([1.0, 0.9]).astype(np.float64), count),
        object_id=rng.integers(0, 88, count),
    )


# ----------------------------------------------------------------------- configuration


def test_defaults_are_the_reference_converter():
    assert asdict(ViewGraspConfig()) == {
        "friction_threshold": 0.2,
        "min_width": 0.03,
        "max_width": 0.15,
        "min_depth": 0.015,
        "max_depth": 0.025,
        "approach_threshold": 0.0,
        "max_grasps": 5000,
        "model_points": 40000,
        "finger_length": 0.04,
        "nms_distance": 0.02,
        "nms_angle": 60.0,
        "poses": "annotations",
        "intrinsics": "reference",
        "visibility_filter": False,
        "visibility_distance": 0.01,
        "visibility_points": 40000,
        "seed": 0,
        "dtype": "float64",
    }
    assert REFERENCE_INTRINSICS == {
        "realsense": (927.17, 927.37, 651.32, 349.62),
        "kinect": (631.55, 631.21, 638.43, 366.50),
    }


def test_every_parameter_the_camera_and_the_labels_decide_the_hash():
    base = ViewGrasps.spec("graspnet1b", "realsense", ViewGraspConfig(), "abc")
    assert base.kind == "view_grasps" and base.generator == "grasppanda.view_grasps"
    assert base.parameters == {"camera": "realsense", **asdict(ViewGraspConfig())}
    assert base.inputs == {"dataset": "graspnet1b", "grasp_labels": "abc"}
    hashes = {base.hash}
    changes = {
        "friction_threshold": 0.4,
        "min_width": 0.02,
        "max_width": 0.1,
        "min_depth": 0.005,
        "max_depth": 0.035,
        "approach_threshold": 0.1,
        "max_grasps": 4000,
        "model_points": 20000,
        "finger_length": 0.06,
        "nms_distance": 0.03,
        "nms_angle": 30.0,
        "poses": "meta",
        "intrinsics": "frame",
        "visibility_filter": True,
        "visibility_distance": 0.02,
        "visibility_points": 20000,
        "seed": 1,
        "dtype": "float32",
    }
    assert set(changes) == set(asdict(ViewGraspConfig()))
    for name, value in changes.items():
        hashes.add(
            ViewGrasps.spec("graspnet1b", "realsense", ViewGraspConfig(**{name: value}), "abc").hash
        )
    hashes.add(ViewGrasps.spec("graspnet1b", "kinect", ViewGraspConfig(), "abc").hash)
    hashes.add(ViewGrasps.spec("graspnet1b", "realsense", ViewGraspConfig(), "abc2").hash)
    hashes.add(ViewGrasps.spec("graspclutter6d", "realsense", ViewGraspConfig(), "abc").hash)
    assert len(hashes) == len(changes) + 4
    # the spelling of a number in a config file is not part of the identity
    spelled = ViewGraspConfig(friction_threshold=1, nms_angle=60, approach_threshold=0)
    assert isinstance(spelled.friction_threshold, float) and isinstance(spelled.nms_angle, float)
    plain = ViewGraspConfig(friction_threshold=1.0)
    assert (
        ViewGrasps.spec("d", "c", spelled, "h").hash == ViewGrasps.spec("d", "c", plain, "h").hash
    )


def test_manifest_states_units_and_the_intrinsics_used():
    about = ViewGrasps.spec("graspnet1b", "kinect", ViewGraspConfig(), "abc").about
    assert "metres" in about["depth"] and "millimetres" in about["depth"]
    assert "radians" in about["theta, gamma, beta"] and "not folded" in about["theta, gamma, beta"]
    assert about["columns"] == ", ".join(COLUMNS) and len(COLUMNS) == 10
    assert "(631.55, 631.21, 638.43, 366.5)" in about["intrinsics"]
    frame = ViewGraspConfig(intrinsics="frame")
    assert "each frame" in ViewGrasps.spec("graspnet1b", "kinect", frame, "abc").about["intrinsics"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"min_width": 0.15, "max_width": 0.15}, r"min_width \(0.15\) must be below max_width"),
        ({"min_depth": 0.03}, r"min_depth \(0.03\) must be below max_depth \(0.025\)"),
        ({"friction_threshold": 0.0}, "friction_threshold must be positive"),
        ({"max_grasps": 0}, "max_grasps must be at least 1"),
        ({"model_points": 0}, "model_points must be at least 1"),
        ({"finger_length": -0.04}, "finger_length must be positive"),
        ({"nms_distance": 0.0}, "nms_distance must be positive"),
        ({"nms_angle": 0.0}, r"nms_angle must be in \(0, 180\] degrees"),
        ({"nms_angle": 200.0}, r"nms_angle must be in \(0, 180\] degrees"),
        ({"intrinsics": "camK"}, "intrinsics must be 'reference' or 'frame', got 'camK'"),
        ({"dtype": "float16"}, "dtype must be 'float64' or 'float32', got 'float16'"),
        ({"max_grasps": 5000.0}, "view_grasps.max_grasps must be int, got 5000.0"),
        ({"seed": True}, "view_grasps.seed must be int, got True"),
        ({"friction_threshold": "0.2"}, "view_grasps.friction_threshold must be float, got '0.2'"),
        ({"visibility_filter": 1}, "view_grasps.visibility_filter must be bool, got 1"),
        ({"poses": 3}, "view_grasps.poses must be str, got 3"),
    ],
)
def test_invalid_parameters_are_rejected(change, message):
    with pytest.raises(ContractError, match=message):
        ViewGraspConfig(**change)


def test_draws_are_a_function_of_seed_frame_and_name():
    key = SampleKey("graspnet1b", 3, "realsense", 7)
    one = SeededDraws(0, key).sample("model_points", 1000, 50)
    assert len(set(one.tolist())) == 50 and one.min() >= 0 and one.max() < 1000
    assert np.array_equal(one, SeededDraws(0, key).sample("model_points", 1000, 50))
    assert not np.array_equal(one, np.sort(one))  # an order, not a set
    others = [
        SeededDraws(1, key).sample("model_points", 1000, 50),
        SeededDraws(0, replace(key, frame=8)).sample("model_points", 1000, 50),
        SeededDraws(0, replace(key, scene=4)).sample("model_points", 1000, 50),
        SeededDraws(0, key).sample("depth_points", 1000, 50),
    ]
    assert all(not np.array_equal(one, other) for other in others)
    ties = SeededDraws(0, key).sample("nms_ties", 40, 40)
    assert sorted(ties.tolist()) == list(range(40))
    assert SeededDraws(0, key).sample("grasps", 0, 0).shape == (0,)


# --------------------------------------------------------------------- the definition


def test_reference_views_are_the_single_precision_lattice():
    views = reference_views(300)
    assert views.dtype == np.float32 and views.shape == (300, 3)
    assert np.abs(np.linalg.norm(views.astype(np.float64), axis=1) - 1).max() < 1e-6
    exact = fibonacci_views(300, dtype=torch.float64).numpy()
    difference = np.abs(views - exact).max()
    # the single-precision angle 2 * i * pi * golden is off by up to 1e-4 rad at i = 299
    assert 1e-6 < difference < 2e-4
    assert np.abs(views[:, 2] - exact[:, 2]).max() < 1e-7
    # each step is single precision: recompute view 299 by hand
    f = np.float32
    z = (f(2) * f(299) + f(1)) / f(300) - f(1)
    angle = f(2) * f(299) * f(math.pi) * f((math.sqrt(5.0) - 1.0) / 2.0)
    assert views[299].tolist() == [
        np.sqrt(f(1) - z * z) * np.cos(angle),
        np.sqrt(f(1) - z * z) * np.sin(angle),
        z,
    ]


@pytest.mark.parametrize(
    "config",
    [
        ViewGraspConfig(),
        ViewGraspConfig(friction_threshold=0.5, min_depth=0.005, max_depth=0.035),
        ViewGraspConfig(min_width=0.1, max_width=0.12, min_depth=0.015, max_depth=0.035),
    ],
)
def test_scene_candidates_follow_the_definition_in_label_order(config):
    labels, collisions, _, _ = make_scene(0)
    expected = expected_candidates(labels, collisions, config)
    found = scene_grasps(labels, config)
    assert len(found) == len(expected) > 20
    counts = [sum(row["instance"] == i for row in expected) for i in range(len(labels))]
    assert found.counts.tolist() == counts
    for name in ("point", "rotation", "width", "depth", "score"):
        assert getattr(found, name).dtype == np.float32
    assert np.array_equal(found.point, [row["point"] for row in expected])
    assert np.array_equal(found.width, np.float32([row["width"] for row in expected]))
    assert np.array_equal(found.depth, np.float32([row["depth"] for row in expected]))
    assert np.array_equal(found.score, np.float32([row["score"] for row in expected]))
    assert np.abs(found.rotation - [row["rotation"] for row in expected]).max() < 5e-7
    assert set(np.unique(found.score).tolist()) <= {float(np.float32(1.1) - f) for f in FRICTION}


def nearest_float32(value: Fraction) -> np.float32:
    """The float32 closest to an exact rational."""
    guess = np.float32(float(value))
    options = (
        np.nextafter(guess, np.float32(-np.inf)),
        guess,
        np.nextafter(guess, np.float32(np.inf)),
    )
    return min(options, key=lambda option: abs(Fraction(float(option)) - value))


def fused(a: np.float32, b: np.float32, c: np.float32, d: np.float32) -> np.float32:
    """``fl32(fl32(a * b) + c * d)`` with exact rational arithmetic."""
    first = nearest_float32(Fraction(float(a)) * Fraction(float(b)))
    return nearest_float32(Fraction(float(first)) + Fraction(float(c)) * Fraction(float(d)))


def test_label_frames_are_the_single_precision_frames_of_the_reference_bit_for_bit():
    """The suppression is decided by last bits, so the arithmetic of the frames is pinned.

    Normalised approach, horizontal side and their cross product in float32; the two
    rotated axes as a float32 matrix product evaluated with a fused multiply-add.
    """
    labels, _, _, _ = make_scene(1)
    everything = ViewGraspConfig(friction_threshold=1.0, min_width=-1.0, max_width=1.0)
    found = scene_grasps(labels, everything).rotation
    f = np.float32
    expected = set()
    plain = set()
    for view in reference_views(GRID.num_view):
        approach = -view
        approach = approach / np.sqrt(
            (approach[0] * approach[0] + approach[1] * approach[1]) + approach[2] * approach[2]
        )
        side = np.array([-approach[1], approach[0], f(0)])
        side = side / np.sqrt((side[0] * side[0] + side[1] * side[1]) + side[2] * side[2])
        up = np.array(
            [
                approach[1] * side[2] - approach[2] * side[1],
                approach[2] * side[0] - approach[0] * side[2],
                approach[0] * side[1] - approach[1] * side[0],
            ]
        )
        assert approach.dtype == side.dtype == up.dtype == np.float32
        for angle in np.array(GRID.angles, dtype=np.float32):
            cos, sin = np.cos(angle), np.sin(angle)
            closing = [fused(side[k], cos, up[k], sin) for k in range(3)]
            height = [fused(side[k], -sin, up[k], cos) for k in range(3)]
            expected.add(np.stack([approach, closing, height], axis=1).astype(f).tobytes())
            plain.add(
                np.stack([approach, side * cos + up * sin, up * cos - side * sin], axis=1)
                .astype(f)
                .tobytes()
            )
    assert {frame.tobytes() for frame in found} == expected and len(
        expected
    ) == GRID.num_view * GRID.num_angle
    assert expected != plain  # rounding both products first gives other last bits


def test_label_frames_follow_the_toolbox_convention():
    """Column 0 is the approach axis; the frame is the one `approach_to_rotation` builds."""
    labels, _, _, _ = make_scene(1)
    everything = ViewGraspConfig(friction_threshold=1.0, min_width=-1.0, max_width=1.0)
    found = scene_grasps(labels, everything)
    rotation = found.rotation.astype(np.float64)
    assert np.abs(rotation.transpose(0, 2, 1) @ rotation - np.eye(3)).max() < 1e-6
    assert np.abs(np.linalg.det(rotation) - 1).max() < 1e-6
    views = torch.from_numpy(reference_views(GRID.num_view)).double()
    angles = torch.tensor(GRID.angles, dtype=torch.float64)
    table = approach_to_rotation(
        -views[:, None, :].expand(GRID.num_view, GRID.num_angle, 3),
        angles[None, :].expand(GRID.num_view, GRID.num_angle),
    ).numpy()
    # every (view, angle) frame of the table occurs among the candidates, and nothing else
    nearest = np.abs(rotation[:, None, None] - table[None]).max(axis=(-1, -2)).min(axis=(1, 2))
    assert nearest.max() < 5e-7


def test_no_depth_bin_in_range_means_no_candidates():
    labels, _, poses, _ = make_scene(2)
    found = scene_grasps(labels, ViewGraspConfig(min_depth=0.021, max_depth=0.029))
    assert len(found) == 0 and found.counts.tolist() == [0, 0, 0]
    assert found.point.shape == (0, 3) and found.rotation.shape == (0, 3, 3)
    assert len(camera_grasps(found, poses, 0.0)) == 0


@pytest.mark.parametrize("threshold", [0.0, 0.3, -2.0])
def test_camera_grasps_are_placed_rounded_and_filtered(threshold):
    labels, collisions, poses, _ = make_scene(3)
    config = ViewGraspConfig(min_depth=0.005, max_depth=0.035)
    expected = expected_camera(expected_candidates(labels, collisions, config), poses, threshold)
    found = camera_grasps(scene_grasps(labels, config), poses, threshold)
    assert len(found) == len(expected) > 10
    assert found.object_id.tolist() == [row["object_id"] for row in expected]
    assert found.object_id.dtype == np.int64
    assert np.array_equal(found.width, [row["width"] for row in expected])
    assert np.array_equal(found.depth, [row["depth"] for row in expected])
    assert np.array_equal(found.score, [row["score"] for row in expected])
    assert np.abs(found.translation - [row["translation"] for row in expected]).max() < 1e-7
    assert np.abs(found.rotation - [row["camera_rotation"] for row in expected]).max() < 5e-7
    for values in (found.translation, found.rotation, found.width):  # single-precision values
        assert values.dtype == np.float64
        assert np.array_equal(values, values.astype(np.float32).astype(np.float64))
    assert (found.rotation[:, 2, 0] > threshold).all()
    if threshold == -2.0:  # nothing is filtered: every candidate is placed
        assert len(found) == len(scene_grasps(labels, config))
    with pytest.raises(ContractError, match="2 object poses for the 3 labelled instances"):
        camera_grasps(scene_grasps(labels, config), poses[:2], 0.0)


def test_model_vertices_are_placed_with_their_own_pose():
    _, _, poses, models = make_scene(4)
    everything = np.concatenate(
        [
            part @ pose[:3, :3].astype(np.float64).T + pose[:3, 3]
            for part, (_, pose) in zip(models.vertices, poses)
        ]
    )
    assert len(models) == len(everything) == 70 + 80 + 90
    index = np.random.default_rng(0).permutation(len(models))[:100]
    assert np.allclose(models.placed(index, poses), everything[index], rtol=0, atol=1e-15)
    assert models.placed(index[:0], poses).shape == (0, 3)
    with pytest.raises(ContractError, match="2 object poses for 3 object models"):
        models.placed(index, poses[:2])


def test_refinement_equals_the_exhaustive_search():
    rng = np.random.default_rng(5)
    grasps = grasp_set(rng, 60)
    translation = grasps.translation.copy()
    translation[10:25] = translation[10]  # grasps annotated at one point share their centre
    translation[40] += 1.0  # a grasp far from every point
    points = rng.uniform(-0.09, 0.09, (700, 3)) + AHEAD
    arguments = (translation, grasps.rotation, grasps.width, grasps.depth, points)
    refined = refine_centers(*arguments, height=HEIGHT, finger_length=0.04)
    expected = expected_refined(*arguments)
    moved = np.abs(expected - translation).max(axis=1) > 0
    assert 20 < moved.sum() < 60 and not moved[40]
    assert np.array_equal(refined[~moved], translation[~moved])
    assert np.abs(refined - expected).max() < 1e-15
    # the centre moves along the closing axis only, and by less than half the width
    shift = np.einsum("gi,gij->gj", refined - translation, grasps.rotation)
    assert np.abs(shift[:, [0, 2]]).max() < 1e-15
    assert (np.abs(shift[:, 1]) < grasps.width / 2).all()
    # neither the order of the points nor that of the grasps matters
    order = rng.permutation(len(points))
    assert np.array_equal(
        refine_centers(
            translation,
            grasps.rotation,
            grasps.width,
            grasps.depth,
            points[order],
            height=HEIGHT,
            finger_length=0.04,
        ),
        refined,
    )
    rows = rng.permutation(60)
    assert np.array_equal(
        refine_centers(
            translation[rows],
            grasps.rotation[rows],
            grasps.width[rows],
            grasps.depth[rows],
            points,
            height=HEIGHT,
            finger_length=0.04,
        ),
        refined[rows],
    )
    assert refine_centers(
        translation[:0],
        grasps.rotation[:0],
        grasps.width[:0],
        grasps.depth[:0],
        points,
        height=HEIGHT,
        finger_length=0.04,
    ).shape == (0, 3)
    unchanged = refine_centers(*arguments[:4], points[:0], height=HEIGHT, finger_length=0.04)
    assert np.array_equal(unchanged, translation)


def test_refinement_bounds_are_strict_and_the_centre_is_the_middle_of_the_extremes():
    # An identity frame at the origin: points are their own grasp-frame coordinates, exactly.
    centre, frame = np.zeros((1, 3)), np.eye(3)[None]
    width, depth = np.array([0.08]), np.array([0.02])

    def refine(points):
        points = np.array(points, dtype=np.float64).reshape(-1, 3)
        return refine_centers(centre, frame, width, depth, points, height=0.02, finger_length=0.04)[
            0
        ]

    inside = [[0.0, -0.01, 0.0], [0.01, 0.03, 0.005], [-0.01, 0.0, -0.005]]
    assert np.allclose(refine(inside), [0, 0.01, 0], atol=1e-15)  # (-0.01 + 0.03) / 2
    assert np.allclose(refine(inside[:1]), [0, -0.01, 0], atol=1e-15)  # a single point
    on_the_boundary = [
        [0.0, 0.04, 0.0],  # y = width / 2
        [0.0, -0.04, 0.0],
        [0.0, 0.035, 0.01],  # z = height / 2
        [0.0, 0.035, -0.01],
        [0.02, 0.035, 0.0],  # x = depth
        [-0.02, 0.035, 0.0],  # x = depth - finger_length
    ]
    for point in on_the_boundary:  # each alone: a single inclusive bound would move the centre
        assert np.array_equal(refine(point), [0, 0, 0]), point
        nudged = np.nextafter(point, 0.0)  # one step towards the inside in every coordinate
        assert np.allclose(refine(nudged), [0, nudged[1], 0], atol=1e-15), point
    beyond = [[0.021, 0.0, 0.0], [0.0, 0.0, 0.011], [0.0, -0.05, 0.0]]
    assert np.array_equal(refine([*on_the_boundary, *beyond]), [0, 0, 0])
    assert np.allclose(refine(inside + on_the_boundary), [0, 0.01, 0], atol=1e-15)


def test_refined_centre_has_the_grasp_frame_coordinates_it_was_given():
    """Label frames are orthonormal to single precision only; the rule must not rely on more."""
    rng = np.random.default_rng(6)
    grasps = grasp_set(rng, 30)
    skewed = grasps.rotation + rng.normal(scale=1e-3, size=grasps.rotation.shape)
    points = rng.uniform(-0.07, 0.07, (900, 3)) + AHEAD
    arguments = (grasps.translation, skewed, grasps.width, grasps.depth, points)
    refined = refine_centers(*arguments, height=HEIGHT, finger_length=0.04)
    assert np.abs(refined - expected_refined(*arguments)).max() < 1e-14
    local = np.einsum("gi,gij->gj", refined - grasps.translation, skewed)
    assert np.abs(local[:, 1]).max() > 1e-3  # centres did move
    assert np.abs(local[:, [0, 2]]).max() < 1e-14  # and only along y of their own frame
    naive = grasps.translation + local[:, 1:2] * skewed[:, :, 1]
    assert (
        np.abs(np.einsum("gi,gij->gj", naive - grasps.translation, skewed)[:, [0, 2]]).max() > 1e-7
    )


def test_suppression_equals_the_greedy_double_loop():
    rng = np.random.default_rng(7)
    grasps = grasp_set(rng, 80, spread=0.03)
    orders = []
    for _ in range(3):
        ties = rng.permutation(80)
        kept = suppress(grasps, ties, distance=0.02, angle=math.radians(60))
        expected = expected_survivors(
            grasps.translation, grasps.rotation, grasps.score, ties, 0.02, math.radians(60)
        )
        assert kept.tolist() == expected and 5 < len(kept) < 80
        assert (np.diff(grasps.score[kept]) <= 0).all()  # descending score
        orders.append(kept.tolist())
    assert orders[0] != orders[1] != orders[2]  # the order among equal scores decides
    assert suppress(grasps, ties, distance=1e-9, angle=math.radians(60)).tolist() == sorted(
        range(80), key=lambda g: (-grasps.score[g], ties.tolist().index(g))
    )
    assert len(suppress(grasps, ties, distance=10.0, angle=math.pi + 1e-9)) == 1
    assert suppress(grasps.take(np.arange(0)), np.arange(0), distance=0.02, angle=1.0).shape == (0,)
    for wrong in (ties[:-1], np.zeros(80, dtype=np.int64), np.arange(1, 81)):
        with pytest.raises(ContractError, match="tie_order must be a permutation of the 80 grasps"):
            suppress(grasps, wrong, distance=0.02, angle=1.0)


def test_suppression_thresholds_are_strict():
    turn = np.array(
        [[0.5, -math.sqrt(0.75), 0.0], [math.sqrt(0.75), 0.5, 0.0], [0.0, 0.0, 1.0]]
    )  # 60 deg

    def pair(offset, rotation):
        return GraspSet(
            translation=np.array([[0.0, 0.0, 0.5], [offset, 0.0, 0.5]]),
            rotation=np.stack([np.eye(3), rotation]),
            width=np.full(2, 0.05),
            depth=np.full(2, 0.02),
            score=np.array([1.0, 0.9]),
            object_id=np.zeros(2, dtype=np.int64),
        )

    order = np.arange(2)
    assert suppress(pair(0.019, np.eye(3)), order, distance=0.02, angle=1.0).tolist() == [0]
    assert suppress(pair(0.02, np.eye(3)), order, distance=0.02, angle=1.0).tolist() == [0, 1]
    assert suppress(pair(0.0, turn), order, distance=0.02, angle=math.radians(61)).tolist() == [0]
    assert suppress(pair(0.0, turn), order, distance=0.02, angle=math.radians(59)).tolist() == [
        0,
        1,
    ]
    # the better grasp wins wherever it stands in the tie order
    assert suppress(pair(0.0, np.eye(3)), order[::-1], distance=0.02, angle=1.0).tolist() == [0]


def test_rectangles_equal_the_matrix_form_of_the_definition():
    rng = np.random.default_rng(8)
    grasps = grasp_set(rng, 200, spread=0.15)
    for camera, intrinsics in REFERENCE_INTRINSICS.items():
        found = grasp_rectangles(grasps, intrinsics, HEIGHT)
        expected = expected_rectangles(
            grasps.translation, grasps.rotation, grasps.width, intrinsics
        )
        assert set(found) == set(expected), camera
        for name in ("centers", "depth", "gamma", "beta"):  # double precision
            assert found[name].dtype == np.float64
            assert np.allclose(found[name], expected[name], rtol=1e-12, atol=1e-12), name
        for name in ("theta", "width", "height"):  # derived from the float32 rectangle
            assert np.array_equal(found[name], found[name].astype(np.float32).astype(np.float64))
            assert np.allclose(found[name], expected[name], rtol=2e-5, atol=2e-6), name
        assert np.array_equal(found["depth"], grasps.translation[:, 2])
        assert np.abs(found["theta"]).max() <= math.pi and np.abs(found["theta"]).max() > 2.0
        assert (
            np.abs(found["gamma"]).max() < math.pi / 2 and np.abs(found["beta"]).max() < math.pi / 2
        )


def test_angles_rebuild_the_grasp_frame():
    """theta, gamma and beta are the angles HGGD's decoder turns back into a rotation."""
    rng = np.random.default_rng(9)
    grasps = grasp_set(rng, 300)
    square = (900.0, 900.0, 640.0, 360.0)  # equal focal lengths: the image plane is isotropic
    found = grasp_rectangles(grasps, square, HEIGHT)
    theta, gamma, beta = found["theta"], found["gamma"], found["beta"]
    zero, one = np.zeros_like(theta), np.ones_like(theta)
    in_plane = np.stack(  # approach along the optical axis, jaw direction (-cos, sin) in the image
        [
            zero,
            -np.cos(theta),
            -np.sin(theta),
            zero,
            np.sin(theta),
            -np.cos(theta),
            one,
            zero,
            zero,
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    about_height = np.stack(
        [np.cos(gamma), -np.sin(gamma), zero, np.sin(gamma), np.cos(gamma), zero, zero, zero, one],
        axis=1,
    ).reshape(-1, 3, 3)
    about_closing = np.stack(
        [np.cos(beta), zero, np.sin(beta), zero, one, zero, -np.sin(beta), zero, np.cos(beta)],
        axis=1,
    ).reshape(-1, 3, 3)
    rebuilt = in_plane @ about_height @ about_closing
    assert np.abs(rebuilt - grasps.rotation).max() < 1e-5  # theta is single precision
    # width and height are the metric sizes at the depth of the jaw, in pixels
    jaw_depth = grasps.translation[:, 2] + grasps.width / 2 * np.sin(gamma)
    assert np.allclose(found["width"], 900.0 * grasps.width / jaw_depth, rtol=2e-6)
    # the angles describe the grasp moved onto the optical axis: they ignore where it is
    shifted = grasps.moved(grasps.translation + np.array([0.2, -0.1, 0.0]))
    moved = grasp_rectangles(shifted, square, HEIGHT)
    for name in ("theta", "gamma", "beta", "width", "height", "depth"):
        assert np.array_equal(moved[name], found[name]), name
    assert np.allclose(
        moved["centers"] - found["centers"], np.array([0.2, -0.1]) * 900.0 / found["depth"][:, None]
    )


def test_a_fronto_parallel_grasp_has_the_obvious_rectangle():
    for degrees in (0.0, 30.0, 135.0, -100.0):
        t = math.radians(degrees)
        frame = np.array(
            [[0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)], [1.0, 0.0, 0.0]]
        )
        grasp = GraspSet(
            translation=np.array([[0.05, -0.02, 0.5]]),
            rotation=frame[None],
            width=np.array([0.08]),
            depth=np.array([0.02]),
            score=np.array([1.0]),
            object_id=np.array([3]),
        )
        found = grasp_rectangles(grasp, (800.0, 800.0, 640.0, 360.0), 0.02)
        assert np.allclose(found["centers"], [[640 + 800 * 0.1, 360 - 800 * 0.04]])
        assert found["depth"].tolist() == [0.5]
        assert np.allclose([found["gamma"][0], found["beta"][0]], 0, atol=1e-12)
        # the jaw point is centre - width / 2 * closing axis; theta = atan2(-dv, du) to it
        expected = math.atan2(math.sin(t), -math.cos(t))
        assert abs(math.remainder(found["theta"][0] - expected, 2 * math.pi)) < 1e-6
        assert np.allclose(found["width"], 800 * 0.08 / 0.5, rtol=1e-6)
        assert np.allclose(found["height"], 800 * 0.02 / 0.5, rtol=1e-6)


def test_frame_labels_compose_the_steps_in_the_reference_order():
    labels, collisions, poses, models = make_scene(10, sizes=(40, 30, 45))
    # the principal point is near the left edge and the image is narrow: labels fall off
    # both sides
    intrinsics, image_size = (300.0, 310.0, 40.0, 120.0), (240, 80)
    crowded = replace(SMALL, max_grasps=60)
    for config in (crowded, replace(crowded, visibility_filter=True, max_grasps=1000)):
        rng = np.random.default_rng(11)
        # a "depth cloud" of 60 points near grasp centres: some refined centres stay within
        # a centimetre of one of the 30 that are drawn, others do not
        placed = camera_grasps(scene_grasps(labels, config), poses, 0.0).translation
        depth_points = placed[rng.permutation(len(placed))[:60]] + rng.normal(
            scale=0.002, size=(60, 3)
        )
        draws = Recorder(12)
        found, trace = frame_view_grasps(
            scene_grasps(labels, config),
            models,
            poses,
            config,
            intrinsics,
            image_size,
            draws,
            depth_points,
        )
        rows = expected_camera(expected_candidates(labels, collisions, config), poses, 0.0)
        assert len(trace.candidates) == len(rows) > 12
        sampled = "grasps" in draws.log
        assert sampled == (len(rows) > config.max_grasps)
        if sampled:  # the sub-sample comes first and keeps the drawn order
            assert draws.log["grasps"][0] == len(rows)
            rows = [rows[i] for i in draws.log["grasps"][1]]
        assert draws.log["model_points"][0] == len(models)
        placed = np.concatenate(
            [
                p @ pose[:3, :3].astype(np.float64).T + pose[:3, 3]
                for p, (_, pose) in zip(models.vertices, poses)
            ]
        )[draws.log["model_points"][1]]
        assert np.allclose(trace.model_points, placed, rtol=0, atol=1e-15)
        translation = np.array([row["translation"] for row in rows])
        rotation = np.array([row["camera_rotation"] for row in rows])
        width = np.array([row["width"] for row in rows])
        depth = np.array([row["depth"] for row in rows])
        score = np.array([row["score"] for row in rows])
        # the candidates agree with the oracle up to single-precision rounding; continue from
        # the values the function produced so that later steps compare exactly
        assert np.abs(trace.refined.rotation - rotation).max() < 5e-7
        rotation = trace.refined.rotation
        before = trace.candidates.take(draws.log["grasps"][1]) if sampled else trace.candidates
        assert np.abs(before.translation - translation).max() < 1e-7
        refined = expected_refined(before.translation, rotation, width, depth, trace.model_points)
        assert np.abs(trace.refined.translation - refined).max() < 1e-15
        assert (np.abs(refined - before.translation).max(axis=1) > 0).sum() > 3  # some did move
        ties = draws.log["nms_ties"]
        assert ties[0] == len(rows) and sorted(ties[1].tolist()) == list(range(len(rows)))
        survivors = expected_survivors(
            trace.refined.translation, rotation, score, ties[1], 0.02, math.radians(60)
        )
        assert trace.survivors.tolist() == survivors and 2 < len(survivors) < len(rows)
        kept = trace.refined.take(trace.survivors)
        rectangles = expected_rectangles(kept.translation, kept.rotation, kept.width, intrinsics)
        u, v = rectangles["centers"].T
        inside = (u > 0) & (v > 0) & (u < 80) & (v < 240)
        assert (u <= 0).any() and (u >= 80).any()
        assert np.array_equal(trace.inside, inside) and trace.finite.all()
        assert 0 < inside.sum() < len(inside)  # the scene is wider than the image
        keep = inside
        if config.visibility_filter:
            chosen = depth_points[draws.log["depth_points"][1]]
            assert draws.log["depth_points"][0] == 60 and len(chosen) == 30
            visible = np.array(
                [np.linalg.norm(chosen - c, axis=1).min() < 0.01 for c in kept.translation]
            )
            assert (
                np.array_equal(trace.visible, visible)
                and 0 < (inside & visible).sum() < inside.sum()
            )
            keep = inside & visible
        else:
            assert trace.visible is None and "depth_points" not in draws.log
        assert tuple(found) == FIELDS
        assert len(found["score"]) == keep.sum()
        assert np.allclose(found["centers"], rectangles["centers"][keep], rtol=1e-12)
        assert np.array_equal(found["depth"], kept.translation[keep, 2])
        for name in ("theta", "gamma", "beta", "width", "height"):
            assert np.allclose(found[name], rectangles[name][keep], rtol=2e-5, atol=2e-6), name
        assert np.array_equal(found["score"], kept.score[keep])
        assert np.array_equal(found["object_id"], kept.object_id[keep])
        assert found["object_id"].dtype == np.int64 and set(found["object_id"]) <= {20, 21, 22}
        assert (np.diff(found["score"]) <= 0).all()


def test_frame_labels_reject_misuse():
    labels, _, poses, models = make_scene(13)
    scene = scene_grasps(labels, SMALL)
    arguments = (scene, models, poses, SMALL, (300.0, 300.0, 160.0, 120.0), (240, 320))
    count = len(camera_grasps(scene, poses, 0.0))
    good = {
        "grasps": np.arange(12),
        "model_points": np.arange(80),
        "nms_ties": np.arange(12),
    }
    found, trace = frame_view_grasps(*arguments, Fixed(**good))
    assert count > 12 and len(trace.refined) == 12 and len(found["score"]) > 0
    with pytest.raises(ContractError, match="the 'grasps' draw must be 12 rows out of"):
        frame_view_grasps(*arguments, Fixed(**{**good, "grasps": np.arange(11)}))
    with pytest.raises(
        ContractError, match=r"'model_points' draw must be 80 rows out of 240.*\[161, 240\]"
    ):
        frame_view_grasps(*arguments, Fixed(**{**good, "model_points": np.arange(161, 241)}))
    with pytest.raises(ContractError, match="the 'grasps' draw repeats rows"):
        frame_view_grasps(*arguments, Fixed(**{**good, "grasps": np.zeros(12, dtype=np.int64)}))
    with pytest.raises(ContractError, match="tie_order must be a permutation|repeats rows"):
        frame_view_grasps(*arguments, Fixed(**{**good, "nms_ties": np.zeros(12, dtype=np.int64)}))
    greedy = replace(SMALL, model_points=241)
    with pytest.raises(AssetError, match="240 vertices in total, fewer than the 241"):
        frame_view_grasps(
            scene, models, poses, greedy, *arguments[4:], SeededDraws(0, SampleKey("d", 0, "c", 0))
        )
    blind = replace(SMALL, visibility_filter=True)
    with pytest.raises(ContractError, match="visibility_filter` needs the depth cloud"):
        frame_view_grasps(scene, models, poses, blind, *arguments[4:], Fixed(**good))
    with pytest.raises(AssetError, match="20 pixels with depth, fewer than the 30"):
        frame_view_grasps(
            scene, models, poses, blind, *arguments[4:], Fixed(**good), np.zeros((20, 3))
        )
    with pytest.raises(ContractError, match="2 object poses"):
        frame_view_grasps(scene, models, poses[:2], *arguments[3:], Fixed(**good))


def test_labels_without_a_pixel_are_dropped_and_edge_on_grasps_are_kept():
    def grasp(translation, columns):
        return GraspSet(
            translation=np.array([translation], dtype=np.float64),
            rotation=np.array(columns, dtype=np.float64).T[None],
            width=np.array([0.05]),
            depth=np.array([0.02]),
            score=np.array([1.0]),
            object_id=np.array([4]),
        )

    intrinsics = (300.0, 300.0, 160.0, 120.0)
    # a centre in the camera plane has no pixel: nothing finite can be stored for it
    flat = grasp_rectangles(grasp([0.1, 0.0, 0.0], np.eye(3)), intrinsics, HEIGHT)
    assert not np.isfinite(flat["centers"]).any()
    # closing axis along the optical axis: the reference keeps what its formulas give
    edge_on = grasp([0.0, 0.0, 0.5], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    assert np.allclose(np.linalg.det(edge_on.rotation), 1.0)
    found = grasp_rectangles(edge_on, intrinsics, HEIGHT)
    assert all(np.isfinite(values).all() for values in found.values())
    assert found["gamma"].tolist() == [-math.pi / 2] and found["width"].tolist() == [0.0]


# ------------------------------------------------------------------------------ storage


def some_labels(rng: np.random.Generator, count: int) -> dict[str, np.ndarray]:
    labels = {name: rng.normal(size=count) for name in FIELDS[1:-1]}
    labels["centers"] = rng.uniform(0, 1280, (count, 2))
    labels["object_id"] = rng.integers(0, 88, count)
    return labels


def store_in(tmp_path, **config) -> ViewGrasps:
    spec = ViewGrasps.spec("graspnet1b", "realsense", ViewGraspConfig(**config), "labelhash")
    return ViewGrasps(Asset(spec, tmp_path))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_scene_round_trips_including_an_empty_frame(tmp_path, dtype):
    rng = np.random.default_rng(0)
    store = store_in(tmp_path, dtype=dtype)
    written = {0: some_labels(rng, 5), 1: some_labels(rng, 0), 7: some_labels(rng, 131)}
    store.asset.create()
    store.write_scene(3, written.items(), 3)
    assert store.shard_file(3).name == "scene_0003.npz" == scene_shard(3) + ".npz"
    assert store.shard_file(3).parent == store.asset.path
    assert not store.asset.is_complete(scene_shard(3))  # written is not yet complete
    with pytest.raises(AssetError, match="scene 3 has not been generated"):
        store.read(SampleKey("graspnet1b", 3, "realsense", 0))
    store.mark_complete(3, {"frames": 3, "grasps": 136})
    record = store.asset.shards(refresh=True)["scene_0003"]
    assert record["frames"] == 3 and record["grasps"] == 136 and record["bytes"] > 0

    reader = store_in(tmp_path, dtype=dtype)  # a fresh object, as a dataset would build it
    for frame, labels in written.items():
        read = reader.read(SampleKey("graspnet1b", 3, "realsense", frame))
        assert tuple(read) == FIELDS
        count = len(labels["score"])
        assert read["centers"].shape == (count, 2)
        for name in FIELDS:
            expected_type = np.int64 if name == "object_id" else np.dtype(dtype)
            assert read[name].dtype == expected_type, name
            assert len(read[name]) == count
            assert np.array_equal(read[name], labels[name].astype(expected_type)), name
            assert read[name].flags.c_contiguous
    assert reader.record()["kind"] == "view_grasps" and reader.record()["provenance"] == "native"


def test_missing_shards_and_frames_are_errors_never_empty_labels(tmp_path):
    rng = np.random.default_rng(1)
    store = store_in(tmp_path)
    with pytest.raises(AssetError, match="No view_grasps asset of graspnet1b"):
        store.read(SampleKey("graspnet1b", 0, "realsense", 0))
    store.asset.create()
    store.write_scene(0, [(0, some_labels(rng, 2)), (2, some_labels(rng, 0))], 2)
    store.mark_complete(0, {"frames": 2})
    assert len(store.read(SampleKey("graspnet1b", 0, "realsense", 2))["score"]) == 0
    with pytest.raises(AssetError, match="scene_0000.npz has no labels for frame 1"):
        store.read(SampleKey("graspnet1b", 0, "realsense", 1))
    with pytest.raises(AssetError, match="scene 1 has not been generated"):
        store.read(SampleKey("graspnet1b", 1, "realsense", 0))
    with pytest.raises(
        AssetError, match="was asked from the view-grasp asset of graspnet1b / realsense"
    ):
        store.read(SampleKey("graspnet1b", 0, "kinect", 0))
    with pytest.raises(AssetError, match="was asked from the view-grasp asset of graspnet1b"):
        store.read(SampleKey("graspclutter6d", 0, "realsense", 0))
    with pytest.raises(AssetError, match="cannot mark scene 5 complete"):
        store.mark_complete(5, {"frames": 2})
    # an asset of another configuration is another directory
    assert store_in(tmp_path, seed=1).asset.path != store.asset.path
    with pytest.raises(AssetError, match="differs from the requested one"):
        store_in(tmp_path, seed=1).read(SampleKey("graspnet1b", 0, "realsense", 0))


def test_a_scene_that_fails_to_write_leaves_no_shard(tmp_path):
    rng = np.random.default_rng(2)
    store = store_in(tmp_path)
    store.asset.create()
    with pytest.raises(AssetError, match="scene 4: wrote 2 frames, expected 3"):
        store.write_scene(4, [(0, some_labels(rng, 2)), (1, some_labels(rng, 1))], 3)
    assert not store.shard_file(4).exists() and list(store.asset.path.glob("*.npz")) == []
    bad = some_labels(rng, 3)
    bad["width"][1] = np.inf
    with pytest.raises(ContractError, match="view grasps must be finite"):
        store.write_scene(4, [(0, bad)], 1)
    short = some_labels(rng, 3)
    short["theta"] = short["theta"][:2]
    with pytest.raises(ContractError, match=r"field 'theta' has shape \(2,\) for 3 grasps"):
        store.write_scene(4, [(0, short)], 1)
    extra = {**some_labels(rng, 3), "opens": np.zeros((3, 2))}
    with pytest.raises(ContractError, match="exactly the fields"):
        store.write_scene(4, [(0, extra)], 1)
    flat = {**some_labels(rng, 3), "centers": np.zeros(6)}
    with pytest.raises(ContractError, match=r"'centers' must be \(L, 2\)"):
        store.write_scene(4, [(0, flat)], 1)
    assert not store.shard_file(4).exists()


def test_a_shard_of_another_layout_is_refused(tmp_path):
    store = store_in(tmp_path)
    store.asset.create()
    np.savez(
        store.shard_file(0),
        **{"0000": np.zeros((4, 9)), "0001": np.zeros((4, 10), dtype=np.float32)},
    )
    store.mark_complete(0, {"frames": 2})
    for frame, found in (
        (0, r"float64 array of shape \(4, 9\)"),
        (1, r"float32 array of shape \(4, 10\)"),
    ):
        with pytest.raises(AssetError, match=found + r"; expected \(L, 10\) float64"):
            store.read(SampleKey("graspnet1b", 0, "realsense", frame))


# ------------------------------------------------------------- the tiny dataset tree


def write_models(root, vertices: int = 40) -> dict[int, np.ndarray]:
    """Add ``models/<id>/nontextured.ply`` (binary PLY, float32 vertices) to the tiny tree."""
    rng = np.random.default_rng(99)
    models = {}
    for object_id in POINTS:
        points = rng.uniform(-0.03, 0.03, (vertices + object_id, 3)).astype("<f4")
        header = (
            "ply\nformat binary_little_endian 1.0\ncomment synthetic\n"
            f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        file = root / "models" / f"{object_id:03d}" / "nontextured.ply"
        file.parent.mkdir(parents=True)
        file.write_bytes(header.encode("ascii") + points.tobytes())
        models[object_id] = points.astype(np.float64)
    return models


def generate(dataset, scenes=(0, 1)) -> dict[int, dict[str, int]]:
    store = dataset.view_grasps
    store.asset.create()
    records = {}
    for scene in scenes:
        records[scene] = generate_view_grasps(dataset, store, scene, dataset.view_grasp_config)
        store.mark_complete(scene, records[scene])
    return records


def test_generated_scene_holds_the_labels_of_every_frame(tree):  # noqa: F811
    models = write_models(tree["root"])
    dataset = build(tree, transforms=[], view_grasps=VIEW, check_assets=False)
    config = dataset.view_grasp_config
    assert config == ViewGraspConfig(**VIEW)
    records = generate(dataset)
    total = 0
    for scene in (0, 1):
        candidates = scene_grasps(dataset.scene_labels(scene), config)
        vertices = SceneModels([models[i] for i in SCENES[scene]])
        assert records[scene]["frames"] == FRAMES and records[scene]["candidates"] == len(
            candidates
        )
        for key in dataset.scene_keys(scene):
            frame = dataset.frame(key)
            intrinsics = tuple(
                float(frame.intrinsics[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2))
            )
            poses = dataset.object_poses(key, "annotations")
            expected, _ = frame_view_grasps(
                candidates, vertices, poses, config, intrinsics, SIZE, SeededDraws(0, key)
            )
            stored = dataset.view_grasps.read(key)
            assert tuple(stored) == FIELDS
            for name in FIELDS:
                assert np.array_equal(stored[name], expected[name]), (key, name)
            total += len(stored["score"])
            # sanity of the content: on the objects, inside the image, valid angles
            u, v = stored["centers"].T
            assert ((u > 0) & (u < SIZE[1]) & (v > 0) & (v < SIZE[0])).all()
            assert ((stored["depth"] > 0.40) & (stored["depth"] < 0.60)).all()
            assert set(stored["object_id"]) <= set(SCENES[scene])
            assert set(stored["score"]) <= {1.0, float(np.float32(1.1) - np.float32(0.2))}
            assert (np.abs(stored["gamma"]) < math.pi / 2).all() and (stored["width"] > 0).all()
        assert records[scene]["grasps"] <= total
    assert total == records[0]["grasps"] + records[1]["grasps"] > 4
    assert records[0]["invisible"] == 0 and records[0]["degenerate"] == 0
    shard = dataset.view_grasps.asset.shards()["scene_0001"]
    assert (
        shard["frames"] == FRAMES and shard["bytes"] > 0 and shard["grasps"] == records[1]["grasps"]
    )


def test_dataset_serves_the_stored_labels(tree):  # noqa: F811
    write_models(tree["root"])
    generate(build(tree, transforms=[], view_grasps=VIEW, check_assets=False))
    dataset = build(tree, transforms=TEST, view_grasps=VIEW, images=["depth"], object_labels=False)
    key = dataset.keys[4]
    stored = dataset.view_grasps.read(key)
    sample = dataset.sample(key)
    assert all(np.array_equal(sample.extras["view_grasps"][n], stored[n]) for n in FIELDS)
    batch = dataset[4]
    (served,) = batch.supervision.extras["view_grasps"]
    assert tuple(served) == FIELDS
    assert served["centers"].dtype == torch.float64 and served["object_id"].dtype == torch.int64
    assert torch.equal(served["centers"], torch.from_numpy(stored["centers"]))
    records = dataset.asset_records()
    assert [r["kind"] for r in records] == ["view_grasps"]  # the dense labels are not served
    assert (
        records[0]["generator"] == "grasppanda.view_grasps@1"
        and not records[0]["uses_ground_truth"]
    )
    manifest = json.loads((dataset.view_grasps.asset.path / "manifest.json").read_text())
    assert manifest["inputs"]["grasp_labels"] == dataset.label_store.asset.spec.hash
    assert manifest["parameters"]["camera"] == "realsense" and manifest["preprocessing"] is None
    assert sorted(manifest["shards"]) == ["scene_0000", "scene_0001"]
    with pytest.raises(
        AssetError, match=r"(?s)lacks 1 of the 1 shards.*--view-grasps --camera realsense"
    ):
        build(tree, transforms=TEST, view_grasps=VIEW, split="test")
    with pytest.raises(AssetError, match="No view_grasps asset of tinygraspnet"):
        build(tree, transforms=TEST, view_grasps={**VIEW, "seed": 5})
    for other in ({"camera": "kinect"}, {"view_grasps": {**VIEW, "nms_angle": 30}}):
        variant = build(tree, transforms=TEST, check_assets=False, **{"view_grasps": VIEW, **other})
        assert variant.view_grasps.asset.path != dataset.view_grasps.asset.path


def test_generation_is_reproducible_and_seeded(tree):  # noqa: F811
    write_models(tree["root"])
    first = build(tree, transforms=[], view_grasps=VIEW, check_assets=False)
    generate(first, [0])
    content = first.view_grasps.shard_file(0).read_bytes()
    first.view_grasps.shard_file(0).unlink()
    generate_view_grasps(first, first.view_grasps, 0, first.view_grasp_config)
    with np.load(first.view_grasps.shard_file(0)) as again:
        frames = {name: again[name] for name in again.files}
    assert sorted(frames) == [f"{frame:04d}" for frame in range(FRAMES)]
    with np.load(io.BytesIO(content)) as before:
        assert all(np.array_equal(before[name], frames[name]) for name in frames)
    other = build(tree, transforms=[], view_grasps={**VIEW, "seed": 3}, check_assets=False)
    generate(other, [0])
    with np.load(other.view_grasps.shard_file(0)) as reseeded:
        assert any(not np.array_equal(reseeded[name], frames[name]) for name in frames)


def test_visibility_filter_and_storage_type_are_options(tree):  # noqa: F811
    write_models(tree["root"])
    plain = build(tree, transforms=[], view_grasps=VIEW, check_assets=False)
    strict = build(
        tree,
        transforms=[],
        view_grasps={
            **VIEW,
            "visibility_filter": True,
            "visibility_distance": 0.004,
            "dtype": "float32",
        },
        check_assets=False,
    )
    kept, dropped = generate(plain, [0])[0], generate(strict, [0])[0]
    assert kept["invisible"] == 0 and dropped["invisible"] > 0
    assert dropped["grasps"] == kept["grasps"] - dropped["invisible"]
    for key in plain.scene_keys(0):
        everything, visible = plain.view_grasps.read(key), strict.view_grasps.read(key)
        assert visible["centers"].dtype == np.float32 and visible["object_id"].dtype == np.int64
        # the survivors are a subset, in the same order
        rows = [
            np.flatnonzero((everything["centers"].astype(np.float32) == c).all(axis=1))
            for c in visible["centers"]
        ]
        assert (
            all(len(r) == 1 for r in rows) and (np.diff(np.concatenate(rows)) > 0).all()
            if rows
            else True
        )


def test_reference_intrinsics_put_the_tiny_scene_outside_its_image(tree):  # noqa: F811
    """The constants of the real cameras on 24 x 32 images: every label is outside, none is stored."""
    write_models(tree["root"])
    dataset = build(
        tree, transforms=[], view_grasps={**VIEW, "intrinsics": "reference"}, check_assets=False
    )
    record = generate(dataset, [0])[0]
    assert record["grasps"] == 0 and record["outside_image"] > 0
    for key in dataset.scene_keys(0):
        stored = dataset.view_grasps.read(key)
        assert stored["centers"].shape == (0, 2) and stored["object_id"].dtype == np.int64


def test_generation_rejects_what_it_cannot_do(tree):  # noqa: F811
    dataset = build(tree, transforms=[], view_grasps=VIEW, check_assets=False)
    dataset.view_grasps.asset.create()
    with pytest.raises(
        FileNotFoundError, match=r"model of object 3 not found: .*models/003/nontextured.ply"
    ):
        generate_view_grasps(dataset, dataset.view_grasps, 0, dataset.view_grasp_config)
    assert not dataset.view_grasps.shard_file(0).exists()  # no partial shard is left behind
    write_models(tree["root"])
    wrong = replace(dataset.view_grasp_config, poses="scene_gt")
    with pytest.raises(
        ContractError, match="poses` must be one of meta, annotations for tinygraspnet"
    ):
        generate_view_grasps(dataset, dataset.view_grasps, 0, wrong)
    many = replace(dataset.view_grasp_config, model_points=10_000)
    with pytest.raises(AssetError, match="fewer than the 10000 that `view_grasps.model_points`"):
        generate_view_grasps(dataset, dataset.view_grasps, 0, many)
    dataset.camera = "zivid"
    reference = replace(dataset.view_grasp_config, intrinsics="reference")
    with pytest.raises(ContractError, match="constants for the cameras realsense, kinect only"):
        generate_view_grasps(dataset, dataset.view_grasps, 0, reference)
    assert not dataset.view_grasps.shard_file(0).exists()
    with pytest.raises(
        ContractError, match="`view_grasps` is a label and needs `load_labels: true`"
    ):
        build(tree, transforms=[], view_grasps=VIEW, load_labels=False)
    with pytest.raises(ContractError, match="max_grasps must be at least 1"):
        build(tree, transforms=[], view_grasps={**VIEW, "max_grasps": 0}, check_assets=False)


# ---------------------------------------------------------------------------- prepare


@pytest.fixture
def config(tree, tmp_path):  # noqa: F811
    """A dataset config with a `view_grasps` section for the synthetic tree."""
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "root": str(tree["root"]),
        "preprocess": [
            {"type": "DepthToCloud"},
            {
                "type": "SegmentationBoxWorkspace",
                "margin": 0.02,
                "invalid_depth_foreground": "include",
            },
        ],
        "transforms": TEST,
        "object_labels": False,
        "view_grasps": VIEW,
    }
    file = tmp_path / "tiny_views.yaml"
    document = {
        "custom_imports": ["tests.unit.test_data_graspnet1b"],
        "data": {"train": {"dataset": node}},
    }
    file.write_text(yaml.safe_dump(document))
    return str(file)


def test_prepare_generates_resumes_and_reports(tree, config, capsys):  # noqa: F811
    write_models(tree["root"])
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--check"]) == 1
    out = capsys.readouterr().out
    assert "grasp labels 0/5 shards" in out and "view grasps 0/2 scenes" in out
    assert "--labels / --graspness / --compact-labels / --view-grasps" in out

    assert main([*base, "--view-grasps", "--workers", "1"]) == 1  # labels come first
    assert "grasppanda prepare tinygraspnet --labels" in capsys.readouterr().err
    assert main([*base, "--labels"]) == 0
    capsys.readouterr()

    assert main([*base, "--view-grasps", "--workers", "1", "--scenes", "0"]) == 0
    out = capsys.readouterr().out
    assert "view-grasp asset:" in out and "0 of 1 scenes already present" in out
    assert "1 scenes written" in out and "grasp labels" in out
    assert main([*base, "--view-grasps", "--workers", "1"]) == 0  # resumes: scene 1 only
    out = capsys.readouterr().out
    assert "1 of 2 scenes already present" in out and "1 scenes written" in out
    assert main([*base, "--view-grasps", "--workers", "1"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert main([*base, "--check", "--camera", "realsense"]) == 0
    assert "view grasps 2/2 scenes" in capsys.readouterr().out

    # what the config describes can now be read, and equals a direct generation
    dataset = build(tree, transforms=TEST, view_grasps=VIEW, object_labels=False)
    served = dataset[2].supervision.extras["view_grasps"][0]
    assert tuple(served) == FIELDS and dataset[2].supervision.objects == [[]]
    direct = build(
        tree, transforms=[], view_grasps={**VIEW, "dtype": "float32"}, check_assets=False
    )
    generate(direct)
    for key in dataset.keys:
        one, two = dataset.view_grasps.read(key), direct.view_grasps.read(key)
        assert all(np.array_equal(one[n].astype(two[n].dtype), two[n]) for n in FIELDS)
    manifest = json.loads((dataset.view_grasps.asset.path / "manifest.json").read_text())
    assert sorted(manifest["shards"]) == ["scene_0000", "scene_0001"]
    assert "metres" in manifest["about"]["depth"] and manifest["uses_ground_truth"] is False


def test_prepare_needs_a_view_grasps_section_and_the_models(tree, config, tmp_path, capsys):  # noqa: F811
    base = ["prepare", "tinygraspnet", "--config", config]
    assert main([*base, "--labels"]) == 0
    assert main([*base, "--view-grasps", "--workers", "1"]) == 1  # no models in the tree
    assert "model of object" in capsys.readouterr().err
    assert main([*base, "--check", "--camera", "realsense"]) == 1  # nothing was marked complete
    assert "view grasps 0/2 scenes" in capsys.readouterr().out

    document = yaml.safe_load(Path(config).read_text())
    del document["data"]["train"]["dataset"]["view_grasps"]
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump(document))
    with pytest.raises(SystemExit, match="has no `view_grasps` section"):
        main(["prepare", "tinygraspnet", "--config", str(plain), "--view-grasps"])
    assert (
        main(
            ["prepare", "tinygraspnet", "--config", str(plain), "--check", "--camera", "realsense"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "view grasps" not in out and "grasp labels 5/5 shards" in out
    assert main(["prepare", "tinygraspnet", "--config", config]) == 2
    assert "--view-grasps" in capsys.readouterr().err


@pytest.mark.slow
def test_view_grasps_in_worker_processes_match_a_single_process(tree, config, capsys):  # noqa: F811
    write_models(tree["root"])
    base = ["prepare", "tinygraspnet", "--config", config, "--split", "all"]
    assert main([*base, "--labels"]) == 0
    assert main([*base, "--view-grasps", "--workers", "2"]) == 0
    assert "3 scenes written" in capsys.readouterr().out
    parallel = build(tree, transforms=[], view_grasps=VIEW, split="all")
    serial = build(
        tree,
        transforms=[],
        view_grasps={**VIEW, "dtype": "float32"},
        split="all",
        check_assets=False,
    )
    generate(serial, [0, 1, 2])
    for key in parallel.keys:
        one, two = parallel.view_grasps.read(key), serial.view_grasps.read(key)
        assert all(np.array_equal(one[n].astype(two[n].dtype), two[n]) for n in FIELDS)
    assert sorted(parallel.view_grasps.asset.shards()) == [f"scene_{s:04d}" for s in range(3)]
