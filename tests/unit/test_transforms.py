"""Data transforms: each checked against a definition written out in the test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError, Provenance, SampleKey
from grasppanda.transforms.base import PointTransform, Transform, fingerprint
from grasppanda.transforms.frame import (
    Frame,
    ObjectAnnotation,
    PointSample,
    align,
    to_batch,
    transform_columns,
    transform_points,
)
from grasppanda.transforms.image import (
    DepthToCloud,
    NoWorkspace,
    SegmentationBoxWorkspace,
    SegmentationRectWorkspace,
    TableBoxWorkspace,
)
from grasppanda.transforms.points import (
    ConstantFeatures,
    FilterObjects,
    RandomFlip,
    RandomRotation,
    RandomSample,
    SampleGraspLabels,
    rotation_matrix,
)

KEY = SampleKey("demo", 0, "cam", 0)


def make_frame(seed: int = 0, height: int = 6, width: int = 8) -> Frame:
    rng = np.random.default_rng(seed)
    depth = rng.integers(400, 900, (height, width)).astype(np.uint16)
    depth[rng.random((height, width)) < 0.2] = 0
    segmentation = np.zeros((height, width), dtype=np.uint8)
    segmentation[2:5, 3:6] = 4
    depth[2, 3] = 0  # a foreground pixel without depth
    table = np.eye(4)
    table[:3, :3] = rotation_matrix("x", 2.6) @ rotation_matrix("z", 0.4)
    table[:3, 3] = [0.05, -0.02, 0.7]
    return Frame(
        key=KEY,
        depth=depth,
        depth_scale=1000.0,
        intrinsics=np.array([[9.0, 0, 3.7], [0, 9.5, 2.9], [0, 0, 1]]),
        table_from_camera=table,
        segmentation=segmentation,
    )


class FakeLabels:
    """A grasp-label source holding ``count`` points on a 2 x 2 x 2 grid."""

    def __init__(self, count: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_points = count
        self.points = rng.random((count, 3)).astype(np.float32)
        self.scores = rng.random((count, 2, 2, 2)).astype(np.float32) + 0.1
        self.widths = rng.random((count, 2, 2, 2)).astype(np.float32)
        self.collides = rng.random((count, 2, 2, 2)) < 0.5

    def take(self, rows, *, collision):
        return (
            self.points[rows],
            self.scores[rows],
            self.widths[rows],
            self.collides[rows] if collision else None,
        )


def make_sample(count: int = 30, seed: int = 0, objects: bool = True) -> PointSample:
    rng = np.random.default_rng(seed)
    instance = rng.integers(0, 3, count)
    annotations = None
    if objects:
        annotations = []
        for instance_id, size in ((1, 20), (2, 5)):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation_matrix("y", 0.3 * instance_id)
            pose[:3, 3] = rng.random(3)
            annotations.append(
                ObjectAnnotation(instance_id + 10, instance_id, pose, FakeLabels(size, instance_id))
            )
    points = rng.random((count, 3))
    return PointSample(
        key=KEY,
        intrinsics=np.eye(3),
        points=points,
        pixels=rng.permutation(1000)[:count],
        image_size=(25, 40),
        full_points=points,
        point_labels={"instance": instance, "graspness": rng.random(count).astype(np.float32)},
        objects=annotations,
        rng=np.random.default_rng(seed + 1),
    )


# -------------------------------------------------------------------- image transforms


def test_depth_to_cloud_is_the_pinhole_model():
    frame = DepthToCloud()(make_frame())
    fx, fy, cx, cy = 9.0, 9.5, 3.7, 2.9
    for v in range(6):
        for u in range(8):
            z = float(frame.depth[v, u]) / 1000.0
            expected = [(u - cx) * z / fx, (v - cy) * z / fy, z]
            assert frame.cloud[v, u].tolist() == expected  # exact: same operation order
            assert frame.valid[v, u] == (frame.depth[v, u] > 0)
    assert frame.cloud.dtype == np.float64


def workspace_oracle(frame: Frame, margin: float, include_invalid: bool) -> np.ndarray:
    """The reference crop, point by point."""
    height, width = frame.depth.shape
    table = np.zeros((height, width, 3))
    for v in range(height):
        for u in range(width):
            table[v, u] = (frame.table_from_camera @ np.append(frame.cloud[v, u], 1.0))[:3]
    foreground = frame.segmentation > 0
    if not include_invalid:
        foreground &= frame.depth > 0
    lower = table[foreground].min(axis=0) - margin
    upper = table[foreground].max(axis=0) + margin
    inside = ((table > lower) & (table < upper)).all(axis=-1)
    return inside & (frame.depth > 0)


@pytest.mark.parametrize("mode", ["include", "exclude"])
def test_segmentation_box_workspace_matches_the_definition(mode):
    frame = SegmentationBoxWorkspace(0.02, mode)(DepthToCloud()(make_frame()))
    expected = workspace_oracle(frame, 0.02, mode == "include")
    assert np.array_equal(frame.valid, expected)
    assert 0 < frame.valid.sum() < (frame.depth > 0).sum()
    assert frame.provenance == {"workspace": Provenance.GROUND_TRUTH}
    assert SegmentationBoxWorkspace.uses_ground_truth


def test_foreground_without_depth_stretches_the_reference_box_to_the_camera():
    included = SegmentationBoxWorkspace(0.02, "include")(DepthToCloud()(make_frame()))
    excluded = SegmentationBoxWorkspace(0.02, "exclude")(DepthToCloud()(make_frame()))
    assert (excluded.valid <= included.valid).all()
    assert included.valid.sum() > excluded.valid.sum()


@pytest.mark.parametrize(
    ("margin", "box"), [(0.0, (2, 5, 3, 6)), (0.2, (1, 6, 2, 7)), (0.5, (0, 6, 0, 8))]
)
def test_segmentation_rect_workspace_is_the_padded_pixel_bounding_box(margin, box):
    """The 6 x 8 frame has foreground in rows 2-4 and columns 3-5."""
    frame = SegmentationRectWorkspace(margin)(DepthToCloud()(make_frame()))
    expected = np.zeros((6, 8), dtype=bool)
    for v in range(6):
        for u in range(8):
            # padded by int(6 * margin) rows and int(8 * margin) columns, clipped to the image
            inside_rows = 2 - int(6 * margin) <= v <= 4 + int(6 * margin)
            inside_columns = 3 - int(8 * margin) <= u <= 5 + int(8 * margin)
            expected[v, u] = inside_rows and inside_columns
    top, bottom, left, right = box
    assert expected[top:bottom, left:right].all() and expected.sum() == (bottom - top) * (
        right - left
    )
    assert np.array_equal(frame.valid, expected & (frame.depth > 0))
    assert frame.provenance == {"workspace": Provenance.GROUND_TRUTH}
    assert SegmentationRectWorkspace.uses_ground_truth


def test_workspace_providers_state_their_provenance():
    frame = DepthToCloud()(make_frame())
    table = frame.table_points("test")
    lower, upper = table.min(axis=0) + 0.05, table.max(axis=0) - 0.05
    boxed = TableBoxWorkspace(lower.tolist(), upper.tolist())(frame)
    expected = ((table > lower) & (table < upper)).all(axis=1).reshape(frame.depth.shape)
    assert np.array_equal(boxed.valid, expected & (frame.depth > 0))
    assert boxed.provenance == {"workspace": Provenance.CALIBRATION}
    assert not TableBoxWorkspace.uses_ground_truth

    untouched = NoWorkspace()(DepthToCloud()(make_frame()))
    assert np.array_equal(untouched.valid, untouched.depth > 0) and untouched.provenance == {}


def test_coordinates_one_at_a_time_are_the_interleaved_ones_bit_for_bit():
    """The box workspaces work on one table coordinate at a time; that must not move a
    single pixel in or out. The interleaved formula is written out here as the reference."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5000, 3)) * [0.4, 0.3, 0.9]
    points[::7] = 0.0  # pixels without depth sit on the camera centre
    for shape in ((3, 3), (3, 4), (4, 4)):
        matrix = rng.normal(size=shape)
        columns = transform_columns(points.astype(np.float32), matrix)
        reference = transform_points(points.astype(np.float32), matrix)
        assert all(column.flags.c_contiguous and column.dtype == np.float64 for column in columns)
        assert all(np.array_equal(columns[axis], reference[:, axis]) for axis in range(3))
    with pytest.raises(ValueError, match="expected a .* matrix"):
        transform_columns(points, np.eye(2))

    for seed, size in ((1, (6, 8)), (2, (48, 64)), (3, (120, 160))):
        frame = DepthToCloud()(make_frame(seed, *size))
        table = frame.table_points("test")
        assert all(
            np.array_equal(column, table[:, axis])
            for axis, column in enumerate(frame.table_coordinates("test"))
        )
        foreground = frame.segmentation.reshape(-1) > 0
        measured = foreground & (frame.depth.reshape(-1) > 0)
        for mode, chosen in (("include", foreground), ("exclude", measured)):
            lower = table[chosen].min(axis=0) - 0.02
            upper = table[chosen].max(axis=0) + 0.02
            expected = (table[:, 0] > lower[0]) & (table[:, 0] < upper[0])
            expected &= (table[:, 1] > lower[1]) & (table[:, 1] < upper[1])
            expected &= (table[:, 2] > lower[2]) & (table[:, 2] < upper[2])
            inside = SegmentationBoxWorkspace(0.02, mode).inside(frame)
            assert inside.dtype == np.bool_ and inside.shape == size
            assert np.array_equal(inside, expected.reshape(size)) and inside.sum() > 0
        lower, upper = np.quantile(table, 0.2, axis=0), np.quantile(table, 0.8, axis=0)
        expected = ((table > lower) & (table < upper)).all(axis=1).reshape(size)
        inside = TableBoxWorkspace(lower.tolist(), upper.tolist()).inside(frame)
        assert np.array_equal(inside, expected) and 0 < inside.sum() < inside.size


def test_workspace_errors_explain_themselves():
    with pytest.raises(ContractError, match="put DepthToCloud before it"):
        NoWorkspace()(make_frame())
    frame = NoWorkspace()(DepthToCloud()(make_frame()))
    with pytest.raises(ContractError, match=r"two workspace providers \(NoWorkspace and Table"):
        TableBoxWorkspace([0, 0, 0], [1, 1, 1])(frame)
    blind = DepthToCloud()(make_frame())
    blind.segmentation = None
    with pytest.raises(ContractError, match="needs the ground-truth segmentation"):
        SegmentationBoxWorkspace(0.02, "include")(blind)
    empty = DepthToCloud()(make_frame())
    empty.segmentation[:] = 0
    with pytest.raises(ContractError, match="no foreground pixel"):
        SegmentationBoxWorkspace(0.02, "include")(empty)
    empty.workspace = None
    with pytest.raises(ContractError, match="no foreground pixel"):
        SegmentationRectWorkspace(0.1)(empty)
    blind.workspace = None
    with pytest.raises(ContractError, match="needs the ground-truth segmentation"):
        SegmentationRectWorkspace(0.1)(blind)
    uncalibrated = DepthToCloud()(make_frame())
    uncalibrated.table_from_camera = None
    with pytest.raises(ContractError, match="no camera-to-table calibration"):
        TableBoxWorkspace([0, 0, 0], [1, 1, 1])(uncalibrated)
    with pytest.raises(ValueError, match="empty box"):
        TableBoxWorkspace([0, 0, 0], [1, 0, 1])


def test_fingerprint_follows_the_configuration_not_its_spelling():
    reference = [DepthToCloud(), SegmentationBoxWorkspace(0.02, "include")]
    assert fingerprint(reference) == fingerprint(
        [TRANSFORMS.build({"type": "DepthToCloud"}), SegmentationBoxWorkspace(2e-2, "include")]
    )
    assert TableBoxWorkspace([0, 0, 0], [1, 1, 1]).config() == {
        "type": "TableBoxWorkspace",
        "version": 1,
        "lower": [0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0],
    }
    different = [
        [DepthToCloud(), SegmentationBoxWorkspace(0.03, "include")],
        [DepthToCloud(), SegmentationBoxWorkspace(0.02, "exclude")],
        [DepthToCloud(), NoWorkspace()],
        [DepthToCloud()],
    ]
    prints = {fingerprint(chain) for chain in different} | {fingerprint(reference)}
    assert len(prints) == 5

    class Changed(SegmentationBoxWorkspace):
        version = 2

    Changed.__name__ = "SegmentationBoxWorkspace"
    assert fingerprint([DepthToCloud(), Changed(0.02, "include")]) != fingerprint(reference)
    with pytest.raises(TypeError, match="RandomSample is not an ImageTransform"):
        fingerprint([DepthToCloud(), RandomSample(4)])


def test_a_transform_must_keep_its_arguments_to_be_fingerprinted():
    class Forgetful(Transform):
        def __init__(self, strength: float) -> None:
            pass

    with pytest.raises(TypeError, match="must keep its constructor argument 'strength'"):
        Forgetful(1.0).config()


# ----------------------------------------------------------------------- the boundary


def test_align_orders_points_by_pixel_and_keys_them():
    frame = NoWorkspace()(DepthToCloud()(make_frame()))
    sample = align(frame)
    rows, columns = np.nonzero(frame.depth > 0)
    assert np.array_equal(sample.pixels, rows * 8 + columns)
    assert np.array_equal(sample.points, frame.cloud[rows, columns])
    assert sample.full_points is sample.points and sample.image_size == (6, 8)
    assert align(frame, keep_full_points=False).full_points is None
    frame.valid[:] = False
    with pytest.raises(ContractError, match="no pixel survives preprocessing"):
        align(frame)


# -------------------------------------------------------------------- point transforms


def test_random_sample_draws_without_replacement_when_it_can():
    sample = make_sample(30)
    before = {name: values.copy() for name, values in sample.point_labels.items()}
    points, pixels = sample.points.copy(), sample.pixels.copy()
    out = RandomSample(12)(sample)
    assert len(out) == 12 and len(set(out.index.tolist())) == 12
    assert np.array_equal(out.points, points[out.index])
    assert np.array_equal(out.pixels, pixels[out.index])
    for name, values in before.items():
        assert np.array_equal(out.point_labels[name], values[out.index])
    assert len(out.full_points) == 30  # the unsampled cloud is kept


def test_random_sample_pads_with_replacement_when_points_are_few():
    index = RandomSample(50).draw(30, np.random.default_rng(0))
    assert index[:30].tolist() == list(range(30)) and len(index) == 50
    assert ((index[30:] >= 0) & (index[30:] < 30)).all()
    sample = RandomSample(50)(make_sample(30))
    assert len(sample) == 50 and sample.point_labels["graspness"].shape == (50,)
    again = RandomSample(10).apply(sample, np.arange(10))
    assert again.index.tolist() == list(range(10))  # indices compose back to the masked cloud


def test_filter_objects_counts_points_of_each_instance():
    sample = make_sample(30)
    counts = {i: int((sample.point_labels["instance"] == i).sum()) for i in (1, 2)}
    threshold = max(counts.values())
    kept = FilterObjects(threshold)(sample).objects
    assert [a.instance_id for a in kept] == [i for i in (1, 2) if counts[i] >= threshold]
    with pytest.raises(ContractError, match="carries no supervision"):
        FilterObjects(1)(make_sample(objects=False))


def test_sample_grasp_labels_follows_the_quarter_with_minimum_rule():
    step = SampleGraspLabels(0.25, 300, "zero_scores")
    assert [step.count(n) for n in (100, 300, 1199, 1200, 1204, 9605)] == [
        100,
        300,
        300,
        300,
        301,
        2401,
    ]
    for count in (0, 1, 7, 299, 300, 1203, 9605):  # the reference: min(max(int(n / 4), 300), n)
        assert step.count(count) == min(max(int(count / 4), 300), count)


@pytest.mark.parametrize("mode", ["zero_scores", "attach", "ignore"])
def test_sample_grasp_labels_reads_rows_and_handles_collisions(mode):
    sample = make_sample()
    sources = [annotation.source for annotation in sample.objects]
    step = SampleGraspLabels(0.5, 3, mode)
    rows = step.draw(sample)
    assert [len(r) for r in rows] == [10, 3] and all(len(set(r.tolist())) == len(r) for r in rows)
    step.apply(sample, rows)
    for annotation, source, index in zip(sample.objects, sources, rows):
        assert annotation.source is None
        assert np.array_equal(annotation.points, source.points[index])
        assert np.array_equal(annotation.widths, source.widths[index])
        expected = source.scores[index].copy()
        if mode == "zero_scores":
            expected[source.collides[index]] = 0
            assert (annotation.scores == 0).any()
        assert np.array_equal(annotation.scores, expected)
        if mode == "attach":
            assert np.array_equal(annotation.collision, source.collides[index])
        else:
            assert annotation.collision is None
    assert not (sources[0].scores == 0).any()  # the store itself is never modified
    with pytest.raises(ContractError, match="has no grasp-label source"):
        step(sample)


def test_rotation_matrix_is_right_handed():
    quarter = math.pi / 2
    assert np.allclose(rotation_matrix("x", quarter) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(rotation_matrix("y", quarter) @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(rotation_matrix("z", quarter) @ [1, 0, 0], [0, 1, 0])
    for axis in "xyz":
        matrix = rotation_matrix(axis, 0.3)
        assert np.allclose(matrix @ matrix.T, np.eye(3)) and np.isclose(np.linalg.det(matrix), 1)


def test_transform_points_is_the_affine_map_in_a_fixed_order():
    rng = np.random.default_rng(0)
    points, matrix = rng.random((5, 3)), rng.random((4, 4))
    out = transform_points(points, matrix)
    for point, moved in zip(points, out):
        for row in range(3):
            m = matrix[row]
            assert moved[row] == ((m[0] * point[0] + m[1] * point[1]) + m[2] * point[2]) + m[3]
    assert np.allclose(transform_points(points, matrix[:3, :3]), points @ matrix[:3, :3].T)


def world_points(sample: PointSample) -> list[np.ndarray]:
    """Label points of each object placed in the camera frame by its pose."""
    return [
        transform_points(annotation.source.points, annotation.pose) for annotation in sample.objects
    ]


@pytest.mark.parametrize(
    ("step", "argument", "matrix"),
    [
        (RandomFlip("x", 0.5), True, np.diag([-1.0, 1.0, 1.0])),
        (RandomFlip("z", 0.5), True, np.diag([1.0, 1.0, -1.0])),
        (RandomFlip("x", 0.5), False, np.eye(3)),
        (RandomRotation("x", 30.0), 0.4, rotation_matrix("x", 0.4)),
        (RandomRotation("z", 30.0), -0.2, rotation_matrix("z", -0.2)),
    ],
)
def test_augmentation_moves_points_and_object_poses_together(step, argument, matrix):
    sample = make_sample()
    points = sample.points.copy()
    placed = world_points(sample)
    labels = {name: values.copy() for name, values in sample.point_labels.items()}
    step.apply(sample, argument)
    assert np.allclose(sample.points, points @ matrix.T, atol=1e-15)
    assert sample.full_points is sample.points
    for before, after in zip(placed, world_points(sample)):
        # A label point expressed in the object frame lands where the scene moved to.
        assert np.allclose(after, before @ matrix.T, atol=1e-6)
    for annotation in sample.objects:
        assert annotation.pose.dtype == np.float32 and annotation.pose[3].tolist() == [0, 0, 0, 1]
    for name, values in labels.items():
        assert np.array_equal(sample.point_labels[name], values)  # per-point labels do not move


def test_flip_is_exact_and_makes_poses_reflections():
    sample = make_sample()
    points, pose = sample.points.copy(), sample.objects[0].pose.copy()
    RandomFlip("x", 1.0)(sample)
    assert np.array_equal(sample.points, points * [-1, 1, 1])
    assert np.array_equal(sample.objects[0].pose[:3], pose[:3] * np.array([[-1], [1], [1]]))
    assert np.isclose(np.linalg.det(sample.objects[0].pose[:3, :3]), -1)


def test_augmentation_draws_are_within_their_ranges_and_seeded():
    rng = np.random.default_rng(0)
    flips = [RandomFlip("x", 0.5).draw(rng) for _ in range(2000)]
    assert 0.45 < np.mean(flips) < 0.55
    angles = np.array([RandomRotation("x", 30.0).draw(rng) for _ in range(2000)])
    assert angles.min() >= -math.pi / 6 and angles.max() < math.pi / 6 and angles.std() > 0.25
    assert not any(RandomFlip("x", 0.0).draw(rng) for _ in range(50))
    one, two = make_sample(seed=3), make_sample(seed=3)
    for step in (RandomSample(10), RandomFlip("x", 0.5), RandomRotation("x", 30.0)):
        one, two = step(one), step(two)
    assert np.array_equal(one.points, two.points)


def test_sampling_after_augmentation_keeps_full_points_consistent():
    sample = make_sample()
    RandomSample(8)(sample)
    RandomRotation("x", 30.0).apply(sample, 0.3)
    assert np.allclose(sample.points, sample.full_points[sample.index], atol=1e-15)


def test_constant_features_and_batch_conversion():
    sample = ConstantFeatures(3)(RandomSample(8)(make_sample(objects=False)))
    batch = to_batch(sample)
    assert batch.supervision is None
    assert (
        batch.observation.point_feats.shape == (8, 3) and (batch.observation.point_feats == 1).all()
    )
    assert batch.observation.points.coords.dtype.is_floating_point
    assert len(batch.observation.full_points) == 30
    labelled = make_sample()
    with pytest.raises(ContractError, match="were never loaded"):
        to_batch(labelled)
    SampleGraspLabels(1.0, 0, "attach")(labelled)
    supervision = to_batch(labelled).supervision
    assert [len(objects) for objects in supervision.objects] == [2]
    assert supervision.objects[0][0].scores.shape == (20, 2, 2, 2)
    assert supervision.point_labels["graspness"].shape == (30,)


def test_invalid_arguments_are_rejected():
    for build in (
        lambda: RandomSample(0),
        lambda: SampleGraspLabels(0.0, 1, "ignore"),
        lambda: RandomFlip("w", 0.5),
        lambda: RandomFlip("x", 1.5),
        lambda: RandomRotation("x", -1.0),
        lambda: ConstantFeatures(0),
        lambda: SegmentationBoxWorkspace(-0.1, "include"),
        lambda: SegmentationRectWorkspace(-0.1),
    ):
        with pytest.raises(ValueError):
            build()
    assert issubclass(RandomSample, PointTransform)
