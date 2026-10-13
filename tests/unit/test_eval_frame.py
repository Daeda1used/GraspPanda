"""One frame end to end on a toy scene, plus the selection and ranking steps in isolation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b.frame import (
    evaluate_frame,
    nearest_point,
    precision_table,
    prepare_prediction,
    select_grasps,
)
from grasppanda.evaluation.graspnet1b.protocol import GRASPNET1B_SV_V1 as PROTOCOL
from grasppanda.evaluation.graspnet1b.scene import table_lattice

from .eval_synthetic import (
    BOX_CENTRE,
    SPHERE_CENTRE,
    TABLE_DEPTH,
    box_asset,
    grasp_row,
    sphere_asset,
    top_grasp,
)

FRICTIONS = PROTOCOL.frictions


@pytest.fixture(scope="module")
def scene():
    """A sphere and a box on a table, seen from straight above."""
    objects = [sphere_asset(0), box_asset(1)]
    poses = []
    for centre in (SPHERE_CENTRE, BOX_CENTRE):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = centre
        poses.append(pose)
    table = table_lattice(PROTOCOL.table) * [1.0, -1.0, -1.0] + [0.0, 0.0, TABLE_DEPTH]
    return objects, poses, table


def score(scene, rows, protocol=PROTOCOL, dtype=np.float32):
    objects, poses, table = scene
    prediction = np.array(rows, dtype=dtype).reshape(-1, 17)
    return evaluate_frame(prediction, objects, poses, table if protocol.table else None, protocol)


GOOD_SPHERE = {"width": 0.08}
GOOD_BOX = {"width": 0.06, "closing": (1.0, 0.0, 0.0)}


def four_grasps():
    return [
        top_grasp(SPHERE_CENTRE, score=0.9, **GOOD_SPHERE),  # holds the sphere
        # too narrow: the fingers hit the sphere. Turned by 90 degrees, so not suppressed.
        top_grasp(SPHERE_CENTRE, score=0.8, width=0.04, closing=(1.0, 0.0, 0.0)),
        top_grasp(BOX_CENTRE, score=0.7, **GOOD_BOX),  # holds the box
        top_grasp(np.array([0.0, 0.25, 0.3]), score=0.6, width=0.08),  # in free space
    ]


def expected_table(hits):
    """``hits[i]`` says whether the grasp at rank i succeeds at every friction."""
    hits = np.array(list(hits) + [False] * (PROTOCOL.top_k - len(hits)))
    column = np.cumsum(hits) / np.arange(1, PROTOCOL.top_k + 1)
    return np.repeat(column[:, None], len(FRICTIONS), axis=1)


# ---------------------------------------------------------------------- end to end


def test_every_stage_on_the_toy_scene(scene):
    result = score(scene, four_grasps())
    assert result.selected.tolist() == [0, 1, 2, 3]  # nothing suppressed, best first
    assert result.owner.tolist() == [0, 0, 1, 0]  # the free-space grasp is nearest the sphere
    assert result.kept.tolist() == [0, 1, 3, 2]  # grouped by object
    assert result.kept_owner.tolist() == [0, 0, 0, 1]
    assert result.collision.tolist() == [False, True, True, False]
    assert result.empty.tolist() == [False, False, True, False]
    assert result.friction.tolist() == [0.2, -1.0, -1.0, 0.2]
    assert result.kept[result.ranking].tolist() == [0, 1, 2, 3]
    assert result.precision.shape == (50, 6)
    assert np.array_equal(result.precision, expected_table([True, False, True, False]))
    assert result.precision[49, 0] == 2 / 50  # missing ranks count as failures


def test_row_order_is_irrelevant_without_ties(scene):
    rows = four_grasps()
    reference = score(scene, rows)
    for permutation in ([3, 2, 1, 0], [2, 0, 3, 1]):
        result = score(scene, [rows[i] for i in permutation])
        assert np.array_equal(result.precision, reference.precision)
        assert [permutation[i] for i in result.selected] == [0, 1, 2, 3]


def test_tied_scores_are_ranked_by_row(scene):
    good, bad = four_grasps()[0], four_grasps()[1]
    good[0] = bad[0] = 0.5
    bad[13] += 0.06  # move it out of suppression range; it now misses the sphere entirely
    first = score(scene, [good, bad])
    second = score(scene, [bad, good])
    assert first.friction[first.ranking].tolist() == [0.2, -1.0]
    assert second.friction[second.ranking].tolist() == [-1.0, 0.2]
    assert first.precision[0, 0] == 1.0 and second.precision[0, 0] == 0.0
    assert np.array_equal(first.precision[1:], second.precision[1:])


def test_empty_prediction_scores_zero(scene):
    result = score(scene, np.zeros((0, 17)))
    assert np.array_equal(result.precision, np.zeros((50, 6)))
    assert len(result.selected) == len(result.kept) == len(result.friction) == 0


def test_widths_are_clipped_before_anything_else(scene):
    wide = top_grasp(SPHERE_CENTRE, score=0.9, width=0.35)  # clipped to 0.1: still a good grasp
    negative = top_grasp(BOX_CENTRE, score=0.8, width=-0.02, closing=(1.0, 0.0, 0.0))
    result = score(scene, [wide, negative])
    assert result.friction.tolist() == [0.2, -1.0]
    assert result.empty.tolist() == [False, True]  # zero width holds nothing


def test_suppression_happens_before_assignment_and_cuts(scene):
    rows = [top_grasp(SPHERE_CENTRE, score=0.9 - 0.01 * i, **GOOD_SPHERE) for i in range(5)]
    for i, row in enumerate(rows):
        row[13] += 0.001 * i  # 1 mm apart: all but the best are suppressed
    result = score(scene, rows)
    assert result.selected.tolist() == [0] and result.kept.tolist() == [0]
    assert np.array_equal(result.precision, expected_table([True]))


def test_per_object_cut_keeps_the_ten_best(scene):
    rows = []
    for i in range(14):  # 3.5 cm apart: no suppression; all on the sphere's side of the scene
        offset = [-0.035 * (i // 5), 0.035 * (i % 5) - 0.07, -0.02]
        rows.append(grasp_row(SPHERE_CENTRE + offset, score=0.5 + 0.01 * i, width=0.08))
    result = score(scene, rows)
    assert len(result.selected) == 14 and (result.owner == 0).all()
    assert sorted(result.kept.tolist()) == list(range(4, 14))


def test_a_bad_closing_axis_is_an_error_only_where_the_reference_fails(scene):
    skewed = top_grasp(SPHERE_CENTRE, score=0.9, **GOOD_SPHERE)
    skewed[4:13].reshape(3, 3)[:, 1] *= 1.001
    with pytest.raises(ValueError, match="closing axis"):
        score(scene, [skewed])
    slightly = top_grasp(SPHERE_CENTRE, score=0.9, **GOOD_SPHERE)
    slightly[4:13].reshape(3, 3)[:, 1] *= 1.000005  # inside the tolerance: normalised
    assert score(scene, [slightly], dtype=np.float64).friction.tolist() == [0.2]
    empty = top_grasp(np.array([0.0, 0.25, 0.3]), score=0.9, width=0.08)
    empty[4:13].reshape(3, 3)[:, 1] *= 1.5  # never reaches the force-closure stage
    assert score(scene, [empty]).friction.tolist() == [-1.0]


def test_table_is_part_of_the_collision_scene(scene):
    # long fingers around the sphere: they touch nothing but reach 1 cm into the table
    above = SPHERE_CENTRE - np.array([0.0, 0.0, 0.02])
    deep = grasp_row(above, score=0.9, width=0.08, depth=0.06)
    assert score(scene, [deep]).collision.tolist() == [True]
    no_table = replace(PROTOCOL, name="toy-no-table", table=None)
    assert score(scene, [deep], protocol=no_table).collision.tolist() == [False]


def test_pre_nms_distance_filter(scene):
    filtered = replace(PROTOCOL, name="toy-filter", pre_nms_max_distance=0.05, table=None)
    result = score(scene, four_grasps(), protocol=filtered)
    assert result.selected.tolist() == [0, 1, 2]  # the free-space grasp is dropped up front
    assert score(scene, four_grasps()[3:], protocol=filtered).precision.sum() == 0


def test_objects_and_poses_must_pair_up(scene):
    objects, poses, table = scene
    with pytest.raises(ValueError, match="2 objects but 1 poses"):
        evaluate_frame(np.zeros((1, 17), dtype=np.float32), objects, poses[:1], table, PROTOCOL)


# ------------------------------------------------------------------- preparation


def test_prepare_prediction_clips_in_the_file_precision():
    limit32 = np.float32(0.1)  # 0.100000001490116..., above the float64 value 0.1
    rows = np.zeros((4, 17), dtype=np.float32)
    rows[:, 1] = [-0.5, 0.05, limit32, np.nextafter(limit32, np.float32(1))]
    out = prepare_prediction(rows, PROTOCOL.gripper.max_width)
    assert out.dtype == np.float64
    assert out[:, 1].tolist() == [0.0, float(np.float32(0.05)), float(limit32), float(limit32)]
    rows64 = rows.astype(np.float64)
    assert prepare_prediction(rows64, 0.1)[:, 1].tolist() == [
        0.0,
        float(np.float32(0.05)),
        0.1,
        0.1,
    ]
    assert rows[0, 1] == np.float32(-0.5)  # the caller's array is not modified


@pytest.mark.parametrize(
    "array,message",
    [
        (np.zeros((3, 16), dtype=np.float32), r"shape \(G, 17\)"),
        (np.zeros(17, dtype=np.float32), r"shape \(G, 17\)"),
        (np.zeros((3, 17), dtype=np.int64), "float32 or float64"),
        (np.full((3, 17), np.nan, dtype=np.float32), r"3 grasp\(s\) contain NaN"),
    ],
)
def test_prepare_prediction_rejects_malformed_input(array, message):
    with pytest.raises(ValueError, match=message):
        prepare_prediction(array, 0.1)


def test_object_id_column_is_ignored():
    rows = np.zeros((2, 17), dtype=np.float32)
    rows[:, 16] = [np.nan, 7]
    assert prepare_prediction(rows, 0.1).shape == (2, 17)


# ------------------------------------------------------------------ assignment


def test_nearest_point_equals_brute_force():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, size=(500, 3))
    points[100] = points[7]  # duplicates: the lower index must win
    points[300:310] = points[250:260]
    queries = np.concatenate([rng.uniform(-1.2, 1.2, size=(200, 3)), points[[7, 100, 255, 305]]])
    index, distance = nearest_point(queries, points)
    gaps = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=-1)
    assert np.array_equal(index, np.argmin(gaps, axis=1))
    assert np.allclose(distance, gaps.min(axis=1), rtol=1e-14)
    assert index[-4:].tolist() == [7, 7, 255, 255]


def test_nearest_point_small_inputs():
    one = np.array([[1.0, 2.0, 3.0]])
    index, distance = nearest_point(np.zeros((3, 3)), one)
    assert index.tolist() == [0, 0, 0] and np.allclose(distance, np.sqrt(14.0))
    index, distance = nearest_point(np.zeros((0, 3)), one)
    assert index.shape == distance.shape == (0,)
    with pytest.raises(ValueError, match="no model points"):
        nearest_point(np.zeros((1, 3)), np.zeros((0, 3)))


# ----------------------------------------------------------------------- cuts


def grasps_with_scores(scores):
    grasps = np.zeros((len(scores), 17))
    grasps[:, 0] = scores
    return grasps


def test_select_grasps_per_object_then_global():
    protocol = replace(PROTOCOL, name="toy-cuts", per_object_top=3, top_k=4)
    #          rows:   0    1    2    3    4    5    6    7    8
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.75, 0.3, 0.2])
    owner = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])
    grasps = grasps_with_scores(scores)
    selected = np.argsort(-scores, kind="stable")
    kept, kept_owner = select_grasps(grasps, selected, owner[selected], 4, protocol)
    # per object: {0, 1, 2}, {5, 6}, {7, 8}, {}; fourth-best of those is 0.75
    assert kept.tolist() == [0, 1, 5, 6] and kept_owner.tolist() == [0, 0, 1, 1]


def test_global_cut_keeps_ties_with_the_last_admitted_score():
    protocol = replace(PROTOCOL, name="toy-ties", per_object_top=10, top_k=3)
    scores = np.array([0.9, 0.8, 0.7, 0.7, 0.7, 0.1])
    owner = np.array([0, 1, 0, 1, 0, 1])
    selected = np.argsort(-scores, kind="stable")
    kept, _ = select_grasps(grasps_with_scores(scores), selected, owner[selected], 2, protocol)
    assert sorted(kept.tolist()) == [0, 1, 2, 3, 4]


def test_global_cut_with_fewer_grasps_than_ranks():
    scores = np.array([0.3, 0.9, 0.5])
    selected = np.argsort(-scores, kind="stable")
    kept, kept_owner = select_grasps(
        grasps_with_scores(scores), selected, np.zeros(3, dtype=int), 1, PROTOCOL
    )
    assert kept.tolist() == [1, 2, 0] and kept_owner.tolist() == [0, 0, 0]
    kept, kept_owner = select_grasps(
        grasps_with_scores(scores), selected[:0], np.zeros(0, dtype=int), 3, PROTOCOL
    )
    assert kept.shape == kept_owner.shape == (0,)


# ------------------------------------------------------------------ precision


def precision_reference(ranked):
    """The definition, entry by entry."""
    table = np.zeros((PROTOCOL.top_k, len(FRICTIONS)))
    for column, friction in enumerate(FRICTIONS):
        for k in range(PROTOCOL.top_k):
            head = ranked[: k + 1]
            table[k, column] = sum(1 for value in head if 0 < value <= friction) / (k + 1)
    return table


@pytest.mark.parametrize("count", [0, 1, 7, 50, 63])
def test_precision_table_matches_the_definition(count):
    rng = np.random.default_rng(count)
    ranked = rng.choice([-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], size=count)
    table = precision_table(ranked, PROTOCOL)
    assert np.array_equal(table, precision_reference(ranked.tolist()))
    assert table.shape == (50, 6) and (np.diff(table, axis=1) >= 0).all()


def test_precision_ignores_ranks_beyond_top_k():
    ranked = np.array([-1.0] * 50 + [0.2] * 10)
    assert precision_table(ranked, PROTOCOL).sum() == 0
    ranked = np.array([0.2] * 50 + [-1.0] * 10)
    assert (precision_table(ranked, PROTOCOL) == 1).all()
