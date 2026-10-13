"""The options of NearestLabelAssigner that the GraspNet-baseline family uses: seeds
supervised by their objectness label, widths of every view, tolerance labels, crops
centred at the label point. The matching itself is tested in ``test_gsnet_assigner.py``."""

from __future__ import annotations

import math

import pytest
import torch

from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import ASSIGNERS
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    ObjectGraspLabels,
    PointSet,
    SeedSet,
    Supervision,
)

VIEWS, ANGLES, DEPTHS, SEEDS, POINTS = 12, 3, 2, 10, 40


@pytest.fixture
def template() -> ViewTemplate:
    return ViewTemplate.fibonacci(VIEWS)


def make_object(generator, points: int = 15, tolerance: bool = True) -> ObjectGraspLabels:
    """Labels whose widths and tolerances are functions of the score, so that any
    mix-up of points or views between the three arrays is visible."""
    pose = torch.eye(4)
    pose[:3, :3] = approach_to_rotation(
        torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
    )
    pose[:3, 3] = torch.rand(3, generator=generator)
    shape = (points, VIEWS, ANGLES, DEPTHS)
    scores = torch.rand(shape, generator=generator)
    return ObjectGraspLabels(
        object_id=0,
        pose=pose,
        points=torch.rand(points, 3, generator=generator) * 0.2 - 0.1,
        scores=scores,
        widths=0.02 + 0.1 * scores,
        tolerance=0.05 * scores if tolerance else None,
    )


def make_batch(template, frames: int = 2, seed: int = 0):
    """Seeds that are a subsample of a scene, as a sub-sampling backbone returns them."""
    generator = torch.Generator().manual_seed(seed)
    scene = PointSet.from_dense(torch.rand(frames, POINTS, 3, generator=generator))
    keep = torch.cat(
        [torch.randperm(POINTS, generator=generator)[:SEEDS] + f * POINTS for f in range(frames)]
    )
    seeds = SeedSet(points=scene.subset(keep, torch.arange(frames + 1) * SEEDS))
    index = torch.randint(0, VIEWS, (frames * SEEDS,), generator=generator)
    approach = ApproachSet(template.approach(index), template.rotation(index), index)
    objects = [[make_object(generator) for _ in range(3)] for _ in range(frames)]
    objectness = torch.randint(
        0, 3, (frames * POINTS,), generator=generator
    )  # object ids, 0 = none
    supervision = Supervision(objects=objects, point_labels={"objectness": objectness})
    return seeds, approach, supervision, keep


def baseline(template, **options):
    config = {
        "type": "NearestLabelAssigner",
        "crop_centre": "label_point",
        "supervise": "object_seeds",
        "view_widths": True,
    }
    return ASSIGNERS.build({**config, **options}, template=template)


def test_only_seeds_on_an_object_are_supervised_and_their_label_is_returned(template):
    seeds, approach, supervision, keep = make_batch(template)
    matched = baseline(template)(seeds, approach, supervision)
    expected = supervision.point_labels["objectness"][keep] != 0
    assert 0 < expected.sum() < len(expected)
    assert torch.equal(matched.on_object, expected) and torch.equal(matched.valid, expected)
    assert (matched.object_index >= 0).all()  # every seed is still matched: crops need labels
    everything = baseline(template, supervise="all")(seeds, approach, supervision)
    assert everything.on_object is None and bool(everything.valid.all())
    close = baseline(template, max_match_distance=0.15)(seeds, approach, supervision)
    assert torch.equal(close.valid, expected & (close.distance <= 0.15))  # criteria combine
    assert torch.equal(close.on_object, expected)  # the label itself is not a matter of distance


def test_widths_of_every_view_and_tolerance_follow_the_same_points_and_views(template):
    seeds, approach, supervision, _ = make_batch(template)
    matched = baseline(template)(seeds, approach, supervision)
    count = len(seeds)
    assert matched.view_width.shape == matched.view_friction.shape == (count, VIEWS, ANGLES, DEPTHS)
    assert torch.allclose(matched.view_width, 0.02 + 0.1 * matched.view_friction, atol=1e-6)
    assert torch.allclose(matched.width, 0.02 + 0.1 * matched.friction, atol=1e-6)
    assert matched.tolerance.shape == (count, ANGLES, DEPTHS)
    assert torch.allclose(matched.tolerance, 0.05 * matched.friction, atol=1e-6)
    rows = torch.arange(count)
    assert torch.equal(matched.view_width[rows, approach.view_index], matched.width)
    without = baseline(template, view_widths=False)(seeds, approach, supervision)
    assert without.view_width is None and torch.equal(without.width, matched.width)


def test_training_crops_sit_at_the_label_point(template):
    seeds, approach, supervision, _ = make_batch(template)
    matched = baseline(template)(seeds, approach, supervision)
    assert torch.equal(matched.crop_centre, matched.label_point)
    assert not torch.allclose(matched.crop_centre, seeds.points.coords)
    at_seed = baseline(template, crop_centre="seed")(seeds, approach, supervision)
    assert torch.equal(at_seed.crop_centre, seeds.points.coords)
    assert torch.equal(at_seed.crop_rotation, matched.crop_rotation)


def test_objects_without_label_points_are_harmless(template):
    seeds, approach, supervision, _ = make_batch(template)
    generator = torch.Generator().manual_seed(5)
    supervision.objects[0].insert(1, make_object(generator, points=0))  # nothing visible of it
    supervision.objects[1] = [make_object(generator, points=0)]  # a frame without any label point
    matched = baseline(template)(seeds, approach, supervision)
    first, second = slice(0, SEEDS), slice(SEEDS, 2 * SEEDS)
    assert set(matched.object_index[first].tolist()) <= {0, 2, 3}  # the empty object owns no seed
    assert (matched.object_index[second] == -1).all() and not matched.valid[second].any()
    assert torch.equal(matched.crop_centre[second], seeds.points.coords[second])
    assert (matched.view_width[second] == 0).all() and (matched.tolerance[second] == 0).all()


def test_the_graspness_defaults_return_none_of_the_extra_labels(template):
    seeds, approach, supervision, _ = make_batch(template)
    for objects in supervision.objects:
        for obj in objects:
            obj.tolerance = None
    matched = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)(
        seeds,
        approach,
        Supervision(objects=supervision.objects),  # no point labels needed
    )
    assert matched.on_object is None and matched.view_width is None and matched.tolerance is None
    assert bool(matched.valid.all()) and torch.equal(matched.crop_centre, seeds.points.coords)


def test_missing_or_inconsistent_labels_are_explained(template):
    seeds, approach, supervision, _ = make_batch(template)
    assigner = baseline(template)
    with pytest.raises(
        ContractError, match=r"supervise: object_seeds.*available point labels: \[\]"
    ):
        assigner(seeds, approach, Supervision(objects=supervision.objects))
    loose = SeedSet(points=PointSet(seeds.points.coords, seeds.points.batch_offsets))
    with pytest.raises(ContractError, match="do not record which scene points they are"):
        assigner(loose, approach, supervision)
    short = Supervision(supervision.objects, {"objectness": torch.zeros(5, dtype=torch.long)})
    with pytest.raises(ContractError, match="does not cover the scene points"):
        assigner(seeds, approach, short)
    supervision.objects[0][0].tolerance = None
    with pytest.raises(ContractError, match="some objects of the batch carry tolerance labels"):
        assigner(seeds, approach, supervision)
    supervision.objects[0][0].tolerance = torch.zeros(15, VIEWS, ANGLES, DEPTHS + 1)
    with pytest.raises(ContractError, match="tolerance labels have shape"):
        assigner(seeds, approach, supervision)
    with pytest.raises(ValueError, match="unknown policy: supervise='objects'"):
        baseline(template, supervise="objects")
