"""Deferred grasp labels on the synthetic tree: the reader against the stored arrays,
and proof that a sample, the labels matched to its seeds and the scale statistic are the
same, value for value, whether the grids travel with the sample or stay in the store."""

from __future__ import annotations

import dataclasses
import itertools
import os
import pickle
import threading
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from grasppanda.datasets import labels as label_module
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.labels import (
    READ_AHEAD_VARIABLE,
    DeferredGraspLabels,
    GridFiles,
    StoredGraspLabels,
    _map_budget,
    _mapped,
    _ReadAhead,
)
from grasppanda.models.assigners.nearest_label import NearestLabelAssigner
from grasppanda.models.assigners.scale_balance import (
    HISTOGRAM_KEY,
    ScaleBalancedAssigner,
    collect_seed_scales,
)
from grasppanda.models.views.template import ViewTemplate
from grasppanda.registry import TRANSFORMS
from grasppanda.structures import (
    ApproachSet,
    Batch,
    ContractError,
    Gripper,
    ObjectGraspLabels,
    SeedSet,
    Supervision,
    collate,
)
from grasppanda.structures.observation import gather_grids
from grasppanda.transforms.points import SampleGraspLabels

from .test_data_graspnet1b import (  # noqa: F401
    GRID,
    POINTS,
    PREPROCESS,
    SCENES,
    TinyGraspNet,
    build,
    tree,
)
from .test_transforms import make_sample

MODES = ("zero_scores", "attach", "ignore")
TOLERANCE = {"max_friction": 0.55, "min_good_ratio": 0.6, "radius_step": 0.01, "max_radius": 0.05}


def recipe(grids: str, collision: str = "zero_scores", visible_within: float | None = None):
    """The GraspNet-baseline data recipe at the size of the synthetic tree."""
    return [
        {"type": "RandomSample", "num_points": 64},
        {"type": "FilterObjects", "min_visible_points": 2},
        {
            "type": "SampleGraspLabels",
            "point_fraction": 0.5,
            "min_points": 4,
            "collision": collision,
            "visible_within": visible_within,
            "grids": grids,
        },
        {"type": "RandomFlip", "axis": "x", "prob": 0.5},
        {"type": "RandomRotation", "axis": "x", "max_angle": 30.0},
    ]


def datasets(tree, collision: str = "zero_scores", **options):  # noqa: F811
    """The same dataset twice: grids in the sample, and grids left in the store."""
    visible = options.pop("visible_within", None)
    pair = [
        build(tree, transforms=recipe(grids, collision, visible), **options)
        for grids in ("dense", "deferred")
    ]
    return pair[0], pair[1]


def write_tolerance(tree) -> None:  # noqa: F811
    """Generate the tolerance asset of every object of the synthetic tree."""
    probe = build(tree, tolerance=TOLERANCE, transforms=[], check_assets=False)
    probe.tolerance.asset.create()
    for object_id in POINTS:
        points, scores, _ = probe.label_store.object_arrays(object_id)
        probe.tolerance.mark_complete(object_id, probe.tolerance.write(object_id, points, scores))


def seeds_and_views(batch: Batch, per_frame: int, seed: int):
    """Seeds that are a subsample of the scene points, and a random view for each."""
    generator = torch.Generator().manual_seed(seed)
    points = batch.observation.points
    offsets = points.batch_offsets.tolist()
    keep = torch.cat(
        [
            torch.randperm(offsets[f + 1] - offsets[f], generator=generator)[:per_frame]
            + offsets[f]
            for f in range(len(offsets) - 1)
        ]
    )
    seeds = SeedSet(points.subset(keep, torch.arange(len(offsets)) * per_frame))
    template = ViewTemplate.fibonacci(GRID.num_view)
    view = torch.randint(0, GRID.num_view, (len(keep),), generator=generator)
    return seeds, ApproachSet(template.approach(view), template.rotation(view), view), template


def assert_same_matches(first, second) -> None:
    for field in dataclasses.fields(first):
        one, two = getattr(first, field.name), getattr(second, field.name)
        if one is None or two is None:
            assert one is None and two is None, field.name
        else:
            assert one.dtype == two.dtype and torch.equal(one, two), field.name


# --------------------------------------------------------------------------- reader


@pytest.mark.parametrize("mode", MODES)
def test_the_reader_returns_what_the_store_holds(tree, mode):  # noqa: F811
    write_tolerance(tree)
    dataset = build(tree, tolerance=TOLERANCE, transforms=[])
    rng = np.random.default_rng(0)
    for scene, object_ids in SCENES.items():
        for index, object_id in enumerate(object_ids):
            source = dataset._label_source(scene, index, object_id)
            chosen = rng.permutation(source.num_points)[: source.num_points - 2]
            _, scores, widths, collision = source.take(chosen, collision=True)
            tolerance = source.tolerance(chosen)
            if mode == "zero_scores":
                scores[collision], tolerance[collision] = 0, 0
            reader = source.defer(chosen, collision=mode)
            assert reader.shape == (len(chosen), *GRID.shape) and reader.has_tolerance
            assert reader.has_collision == (mode == "attach")
            reader = pickle.loads(pickle.dumps(reader))  # what a dataloader worker sends
            labels = ObjectGraspLabels(
                object_id, torch.eye(4), torch.zeros(len(chosen), 3), deferred=reader
            )
            # requests repeat and are not sorted
            rows = torch.from_numpy(rng.integers(0, len(chosen), 3 * len(chosen)))
            views = torch.from_numpy(rng.integers(0, GRID.num_view, len(rows)))
            read = labels.gather_scores(rows)
            assert read.dtype == torch.float32 and np.array_equal(read.numpy(), scores[rows])
            assert np.array_equal(labels.gather_widths(rows).numpy(), widths[rows])
            assert np.array_equal(labels.gather_widths(rows, views).numpy(), widths[rows, views])
            assert np.array_equal(labels.gather_tolerance(rows).numpy(), tolerance[rows])
            at_views = labels.gather_tolerance(rows, views)
            assert at_views.dtype == torch.float32
            assert np.array_equal(at_views.numpy(), tolerance[rows, views])
            mask = labels.gather_collision(rows)
            if mode == "attach":
                assert mask.dtype == torch.bool and np.array_equal(mask.numpy(), collision[rows])
            else:
                assert mask is None
            dense = labels.dense()
            assert dense.deferred is None and np.array_equal(dense.scores.numpy(), scores)
            assert np.array_equal(dense.tolerance.numpy(), tolerance)
            assert labels.gather_scores(rows[:0]).shape == (0, *GRID.shape)


def test_one_pass_reads_many_objects_and_mixed_storage(tree):  # noqa: F811
    dense_set, deferred_set = datasets(tree, collision="attach")
    dense, deferred = dense_set[0].supervision.objects[0], deferred_set[0].supervision.objects[0]
    assert len(dense) == 2 and all(obj.deferred is not None for obj in deferred)
    generator = torch.Generator().manual_seed(1)
    owner = torch.randint(0, 2, (40,), generator=generator)
    counts = torch.tensor([len(obj.points) for obj in dense])
    rows = (torch.rand(40, generator=generator) * counts[owner]).long()
    views = torch.randint(0, GRID.num_view, (40,), generator=generator)
    mixed = [dense[0], deferred[1]]
    for name, at in itertools.product(("scores", "widths", "collision"), (None, views)):
        if at is not None and name != "widths":
            continue
        expected = gather_grids(dense, name, owner, rows, at)
        for objects in (deferred, mixed):
            read = gather_grids(objects, name, owner, rows, at)
            assert read.dtype == expected.dtype and torch.equal(read, expected), name
    # requests that all go to the deferred objects of a mixed list, in mixed order
    shifted = gather_grids([dense[0], *deferred], "widths", owner + 1, rows, views)
    assert torch.equal(shifted, gather_grids(dense, "widths", owner, rows, views))
    with pytest.raises(ContractError, match="a request names object 2, but there are 2 objects"):
        gather_grids(mixed, "scores", owner + 1, rows)
    with pytest.raises(AssetError, match="a request names object 2, but 2 were given"):
        gather_grids(deferred, "scores", owner + 1, rows)
    for position in (0, 1):  # the same values as indexing the tensors object by object
        member = owner == position
        assert torch.equal(
            gather_grids(deferred, "scores", owner, rows)[member],
            dense[position].scores[rows[member]],
        )
    # a mask that only some objects carry reads as "no collision" for the others
    bare = ObjectGraspLabels(7, dense[1].pose, dense[1].points, dense[1].scores, dense[1].widths)
    partial = gather_grids([dense[0], bare], "collision", owner, rows)
    assert not partial[owner == 1].any()
    assert torch.equal(partial[owner == 0], dense[0].collision[rows[owner == 0]])
    assert gather_grids([bare], "collision", owner * 0, rows * 0) is None
    with pytest.raises(ContractError, match="carry no tolerance labels"):
        gather_grids(deferred, "tolerance", owner, rows)
    with pytest.raises(ValueError, match="unknown label grid 'graspness'"):
        gather_grids(deferred, "graspness", owner, rows)


def test_labels_hold_exactly_one_storage(tree):  # noqa: F811
    _, deferred_set = datasets(tree)
    labels = deferred_set[0].supervision.objects[0][0]
    assert labels.scores is None and labels.widths is None and labels.deferred is not None
    assert labels.shapes() == dict.fromkeys(("scores", "widths"), (len(labels.points), 5, 3, 2))
    moved = labels.to("cpu")
    assert moved is not labels and moved.deferred is labels.deferred
    pose, points, grid = labels.pose, labels.points, torch.zeros(len(labels.points), 5, 3, 2)
    with pytest.raises(ContractError, match="not both and not neither"):
        ObjectGraspLabels(3, pose, points)
    with pytest.raises(ContractError, match="not both and not neither"):
        ObjectGraspLabels(3, pose, points, scores=grid)
    with pytest.raises(ContractError, match="not both and not neither"):
        ObjectGraspLabels(3, pose, points, grid, grid, deferred=labels.deferred)
    dense = ObjectGraspLabels(3, pose, points, grid, grid, tolerance=grid)
    assert set(dense.shapes()) == {"scores", "widths", "tolerance"} and dense.dense() is dense


def test_only_labels_of_a_store_can_be_deferred(tree):  # noqa: F811
    dataset = build(tree, transforms=[])
    source = dataset._label_source(0, 0, 3)
    rows = np.arange(3)
    memory = StoredGraspLabels(*source.take(np.arange(7), collision=False)[:3], None, GRID)
    with pytest.raises(AssetError, match="held in memory, not in a label store"):
        memory.defer(rows, collision="ignore")
    source.widths = np.asarray(source.widths) + np.float32(0.1)  # no longer what the file holds
    assert np.array_equal(source.take(rows, collision=False)[2], source.widths[rows])
    with pytest.raises(AssetError, match="held in memory, not in a label store"):
        source.defer(rows, collision="ignore")
    with pytest.raises(ValueError, match="zero_scores, attach or ignore"):
        dataset._label_source(0, 0, 3).defer(rows, collision="zero")

    sample = make_sample()  # its label sources hold arrays and cannot defer
    with pytest.raises(ContractError, match=r"grids: deferred.*cannot leave its grids in place"):
        SampleGraspLabels(0.5, 3, "zero_scores", grids="deferred")(sample)
    with pytest.raises(ValueError, match="grids must be 'dense' or 'deferred'"):
        SampleGraspLabels(0.5, 3, "zero_scores", grids="lazy")

    stored = dataset._label_source(0, 0, 3).files
    files = GridFiles(stored.scores, stored.widths, stored.collision, collision_start=0)
    beyond = DeferredGraspLabels(files, np.array([0, 99]), GRID, "ignore")
    labels = ObjectGraspLabels(3, torch.eye(4), torch.zeros(2, 3), deferred=beyond)
    with pytest.raises(AssetError, match="rows 0..99 were asked of an array of 7 rows"):
        labels.gather_scores(torch.tensor([0, 1]))


def test_storage_is_not_part_of_the_identity_of_the_recipe(tree):  # noqa: F811
    dense, deferred = (TRANSFORMS.build(recipe(grids)[2]) for grids in ("dense", "deferred"))
    assert dense.grids == "dense" and deferred.grids == "deferred"
    assert dense.config() == deferred.config() and "grids" not in dense.config()
    assert SampleGraspLabels(0.5, 4, "zero_scores").grids == "dense"  # the reference behaviour
    first, second = datasets(tree)
    # assets derived from the data recipe (the scale statistic) are shared by both
    assert first._scale_statistic_inputs() == second._scale_statistic_inputs()
    node = {"preprocess": PREPROCESS, "transforms": recipe("deferred")}
    assert list(TinyGraspNet.check_config(node)) == []


def test_a_worker_that_defers_opens_no_grid_file(tree):  # noqa: F811
    _, deferred = datasets(tree, visible_within=0.05)
    for index in range(len(deferred)):
        deferred[index]
    assert len(deferred.label_store._objects) == 0 and len(deferred.label_store._scenes) == 0
    assert 64 <= _map_budget() <= 4096
    file = deferred[0].supervision.objects[0][0].deferred.files.scores
    assert _mapped(file) is _mapped(file)  # one map per file and process


def test_a_file_is_read_ahead_once_when_it_is_first_opened(tree, monkeypatch):  # noqa: F811
    _, deferred = datasets(tree)
    files = deferred[0].supervision.objects[0][0].deferred.files
    assert _ReadAhead.read(files.scores) == os.path.getsize(files.scores) > 0
    assert _ReadAhead.read(files.scores + ".missing") == 0  # best effort, never an error
    read = []
    monkeypatch.setattr(_ReadAhead, "read", classmethod(lambda cls, path: read.append(path)))
    fresh = _ReadAhead()
    monkeypatch.setattr(label_module, "_read_ahead", fresh)
    label_module._MAPS.clear()

    def wait() -> None:
        deadline = time.monotonic() + 10
        while not fresh.idle() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fresh.idle()

    for _ in range(3):  # opened once, then served from the cache of maps
        _mapped(files.scores)
        _mapped(files.collision)
    wait()
    assert sorted(read) == sorted([files.scores, files.collision])
    label_module._MAPS.clear()
    _mapped(files.scores)  # a map that fell out and is opened again is not read again
    wait()
    assert len(read) == 2
    monkeypatch.setenv(READ_AHEAD_VARIABLE, "0")
    _mapped(files.widths)
    wait()
    assert len(read) == 2
    # the threads end with their queue: nothing of this stays alive in the process
    for thread in threading.enumerate():
        if thread.name == "label-read-ahead":
            thread.join(timeout=10)
    assert not [thread for thread in threading.enumerate() if thread.name == "label-read-ahead"]


# ------------------------------------------------------------------------- identity


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("visible_within", [None, 0.05])
def test_samples_are_the_same_with_either_storage(tree, mode, visible_within):  # noqa: F811
    dense_set, deferred_set = datasets(tree, mode, visible_within=visible_within)
    assert len(dense_set) == len(deferred_set) == 6
    for index in range(len(dense_set)):
        dense, deferred = dense_set[index], deferred_set[index]
        assert torch.equal(dense.observation.points.coords, deferred.observation.points.coords)
        assert dense.supervision.point_labels.keys() == deferred.supervision.point_labels.keys()
        for name, values in dense.supervision.point_labels.items():
            assert torch.equal(values, deferred.supervision.point_labels[name]), name
        (first,), (second,) = dense.supervision.objects, deferred.supervision.objects
        assert [obj.object_id for obj in first] == [obj.object_id for obj in second]
        for one, two in zip(first, second):
            assert two.deferred is not None and one.shapes() == two.shapes()
            assert torch.equal(one.pose, two.pose) and torch.equal(one.points, two.points)
            read = two.dense()
            for name in ("scores", "widths", "collision", "tolerance"):
                held, got = getattr(one, name), getattr(read, name)
                if held is None:
                    assert got is None, name
                else:
                    assert held.dtype == got.dtype and torch.equal(held, got), name


def test_a_loader_with_workers_delivers_deferred_labels(tree):  # noqa: F811
    dense_set, deferred_set = datasets(tree)
    loader = DataLoader(deferred_set, batch_size=3, num_workers=2, collate_fn=collate)
    batches = list(loader)
    assert sum(len(batch) for batch in batches) == len(deferred_set)
    expected = collate([dense_set[index] for index in range(3)])
    assert torch.equal(batches[0].observation.points.coords, expected.observation.points.coords)
    for objects, reference in zip(batches[0].supervision.objects, expected.supervision.objects):
        for obj, dense in zip(objects, reference):
            rows = torch.arange(len(obj.points)).flip(0)
            assert torch.equal(obj.gather_scores(rows), dense.scores[rows])
            assert torch.equal(obj.gather_widths(rows), dense.widths[rows])


def _options():
    """Every combination of the assigner's policies."""
    for distance, centre, padded, supervise, widths in itertools.product(
        (None, 0.05), ("seed", "label_point"), ("count", "ignore"), ("all", "object_seeds"),
        (False, True),
    ):  # fmt: skip
        yield {
            "max_match_distance": distance,
            "crop_centre": centre,
            "padded_seeds": padded,
            "supervise": supervise,
            "view_widths": widths,
        }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tolerance", [False, True])
def test_matched_labels_are_identical_for_every_assigner_option(tree, mode, tolerance):  # noqa: F811
    options = {}
    if tolerance:
        write_tolerance(tree)
        options["tolerance"] = TOLERANCE
    dense_set, deferred_set = datasets(tree, mode, **options)
    dense = collate([dense_set[index] for index in range(len(dense_set))])
    deferred = collate([deferred_set[index] for index in range(len(deferred_set))])
    seeds, approach, template = seeds_and_views(dense, per_frame=24, seed=3)
    seeds.valid = torch.rand(len(seeds.points), generator=torch.Generator().manual_seed(4)) < 0.8
    compared = 0
    for policy in _options():
        assigner = NearestLabelAssigner(template=template, **policy)
        reference = assigner(seeds, approach, dense.supervision)
        assert_same_matches(reference, assigner(seeds, approach, deferred.supervision))
        assert (reference.tolerance is not None) == tolerance
        assert (reference.view_friction == 0).any() == (mode != "ignore")
        compared += 1
    assert compared == 32 and len(torch.unique(reference.object_index)) > 1


def test_scale_balanced_weights_and_the_scale_statistic_are_identical(tree):  # noqa: F811
    write_tolerance(tree)
    dense_set, deferred_set = datasets(tree, tolerance=TOLERANCE)
    dense = collate([dense_set[index] for index in range(len(dense_set))])
    deferred = collate([deferred_set[index] for index in range(len(deferred_set))])
    histogram = {
        "edges": torch.linspace(0.0, 0.1, 6, dtype=torch.float64),
        "counts": torch.tensor([3, 40, 11, 7, 1]),
    }
    for batch in (dense, deferred):
        batch.supervision.extras[HISTOGRAM_KEY] = [histogram] * len(batch)
    seeds, approach, template = seeds_and_views(dense, per_frame=24, seed=5)
    assigner = ScaleBalancedAssigner(template=template, gripper=Gripper())
    reference = assigner(seeds, approach, dense.supervision)
    assert_same_matches(reference, assigner(seeds, approach, deferred.supervision))
    assert reference.weight is not None and len(torch.unique(reference.weight)) > 1

    arguments = {
        "frame_step": 1,
        "num_seeds": 16,
        "min_graspable_views": 0,
        "graspable_score": 0.1,
        "max_width": 0.1,
    }
    widths = collect_seed_scales(dense_set, **arguments)
    assert len(widths) > 0 and torch.equal(widths, collect_seed_scales(deferred_set, **arguments))


def test_a_detector_trains_on_deferred_labels_as_on_dense_ones(tree):  # noqa: F811
    """The label matching of a whole batch, object lists of different lengths included."""
    dense_set, deferred_set = datasets(tree)
    for indices in ([0], [5, 1, 3], [2, 2]):
        dense = collate([dense_set[index] for index in indices])
        deferred = collate([deferred_set[index] for index in indices])
        dense.supervision.objects[0] = dense.supervision.objects[0][:1]
        deferred.supervision.objects[0] = deferred.supervision.objects[0][:1]
        seeds, approach, template = seeds_and_views(dense, per_frame=20, seed=len(indices))
        assigner = NearestLabelAssigner(template=template)
        matched = assigner(seeds, approach, deferred.supervision)
        assert_same_matches(assigner(seeds, approach, dense.supervision), matched)
        mixed = Supervision(
            objects=[dense.supervision.objects[0], *deferred.supervision.objects[1:]],
            point_labels=dense.supervision.point_labels,
        )
        assert_same_matches(assigner(seeds, approach, mixed), matched)
