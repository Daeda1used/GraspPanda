"""CompactLabelAssigner against a brute-force oracle written out in NumPy, its policies
and the inputs it must refuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.assigners.compact_label import CompactLabelAssigner
from grasppanda.models.assigners.matched import CompactMatchedLabels
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import ASSIGNERS, BuildError
from grasppanda.structures import (
    COMPACT_LABELS,
    ApproachSet,
    CompactGraspLabels,
    ContractError,
    ObjectGraspLabels,
    PointSet,
    SeedSet,
    Supervision,
)

VIEWS, SEEDS = 12, 40


@pytest.fixture
def template() -> ViewTemplate:
    return ViewTemplate.fibonacci(VIEWS)


def make_object(generator, *, mirrored: bool = False, points: int = 15, views: int = VIEWS):
    rotation = approach_to_rotation(
        torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
    )
    if mirrored:  # what a horizontal-flip augmentation does to an object pose
        rotation = torch.diag(torch.tensor([-1.0, 1.0, 1.0])) @ rotation
    pose = torch.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = torch.rand(3, generator=generator)
    shape = (points, views)

    def integers(high: int) -> torch.Tensor:
        return torch.randint(0, high, shape, generator=generator).to(torch.uint8)

    return CompactGraspLabels(
        object_id=int(torch.randint(0, 88, (), generator=generator)),
        pose=pose,
        points=torch.rand(points, 3, generator=generator) * 0.4 - 0.2,
        view_graspness=torch.rand(shape, generator=generator),
        angle=integers(12),
        depth=integers(4),
        score=integers(11),
        width=integers(150),
        view_max_friction=0.6,
    )


def make_batch(template, frames: int = 2, seed: int = 0, valid: torch.Tensor | None = None):
    generator = torch.Generator().manual_seed(seed)
    objects = [
        [make_object(generator, mirrored=(frame == 1 and n == 0)) for n in range(3)]
        for frame in range(frames)
    ]
    # Seeds next to label points: some within the matching tolerance, some beyond it.
    clouds = []
    for frame_objects in objects:
        placed = torch.cat([o.points @ o.pose[:3, :3].T + o.pose[:3, 3] for o in frame_objects])
        pick = torch.randint(0, len(placed), (SEEDS,), generator=generator)
        offset = torch.randn(SEEDS, 3, generator=generator)
        offset = offset / offset.norm(dim=-1, keepdim=True)
        clouds.append(placed[pick] + offset * torch.rand(SEEDS, 1, generator=generator) * 0.01)
    seeds = SeedSet(points=PointSet.from_dense(torch.stack(clouds)), valid=valid)
    index = torch.randint(0, VIEWS, (frames * SEEDS,), generator=generator)
    approach = ApproachSet(
        direction=template.approach(index), rotation=template.rotation(index), view_index=index
    )
    return seeds, approach, Supervision(extras={COMPACT_LABELS: objects})


def oracle(seed_xyz, view_index, objects, template, rule: str, limit: float | None):
    """Label matching of one frame, seed by seed, in float64."""
    views = template.views.double().numpy()
    frames = template.rotations.double().numpy()
    cloud, owner, view_maps = [], [], []
    for position, obj in enumerate(objects):
        rotation = obj.pose[:3, :3].double().numpy()
        cloud.append(obj.points.double().numpy() @ rotation.T + obj.pose[:3, 3].double().numpy())
        owner += [(position, p) for p in range(len(obj.points))]
        rotated = views @ rotation.T
        distance = ((views[:, None, :] - rotated[None, :, :]) ** 2).sum(-1)
        view_maps.append(distance.argmin(axis=1))  # template view → labelled view
    cloud = np.concatenate(cloud)
    results = []
    for xyz, view in zip(seed_xyz.double().numpy(), view_index.tolist()):
        distance = np.linalg.norm(cloud - xyz, axis=1)
        nearest = int(distance.argmin())
        position, point = owner[nearest]
        obj, view_map = objects[position], view_maps[position]
        sharing = np.flatnonzero(view_map == view_map[view])  # template views of one label
        labelled = {"lowest": view == sharing.min(), "highest": view == sharing.max()}.get(
            rule, True
        )
        column = view_map[view]
        results.append(
            {
                "label_point": cloud[nearest],
                "distance": distance[nearest],
                "object_index": position,
                "view_graspness": obj.view_graspness[point].numpy()[view_map],
                "crop_rotation": obj.pose[:3, :3].double().numpy() @ frames[column],
                "view_labelled": labelled,
                "valid": labelled and (limit is None or distance[nearest] < limit),
                "angle": int(obj.angle[point, column]) if labelled else -1,
                "depth": int(obj.depth[point, column]) if labelled else -1,
                "score": int(obj.score[point, column]) / 10 if labelled else 0.0,
                "width": int(obj.width[point, column]) / 1000 if labelled else 0.0,
                "shared": len(sharing) > 1,
            }
        )
    return results


@pytest.mark.parametrize("rule", ["lowest", "highest", "all"])
def test_matches_the_brute_force_definition(template, rule):
    assigner = ASSIGNERS.build(
        {"type": "CompactLabelAssigner", "shared_views": rule}, template=template
    )
    seeds, approach, supervision = make_batch(template)
    matched = assigner(seeds, approach, supervision)
    assert isinstance(matched, CompactMatchedLabels) and matched.view_max_friction == 0.6
    assert matched.view_graspness.shape == (2 * SEEDS, VIEWS)
    assert matched.angle.dtype == matched.depth.dtype == torch.long
    shared = 0
    for frame in range(2):
        rows = slice(frame * SEEDS, (frame + 1) * SEEDS)
        expected = oracle(
            seeds.points.coords[rows],
            approach.view_index[rows],
            supervision.extras[COMPACT_LABELS][frame],
            template,
            rule,
            0.005,
        )
        for seed, reference in zip(range(rows.start, rows.stop), expected):
            shared += reference.pop("shared")
            for name, value in reference.items():
                produced = getattr(matched, name)[seed].double().numpy()
                assert np.allclose(produced, value, atol=1e-5), (seed, name)
    assert bool(matched.view_valid.all())  # the view loss is not subject to the two rules
    assert torch.equal(matched.crop_centre, seeds.points.coords)
    assert 0 < matched.valid.sum() < len(matched.valid)
    assert 0 < (matched.distance < 0.005).sum() < 2 * SEEDS
    assert 0 < shared < 2 * SEEDS  # the batch exercises shared and unshared views
    if rule == "all":
        assert bool(matched.view_labelled.all())
        assert torch.equal(matched.valid, matched.distance < 0.005)
    else:
        assert 0 < (~matched.view_labelled).sum() <= shared
    # A mirrored pose yields a reflection as crop frame; nothing "repairs" it.
    determinant = torch.linalg.det(matched.crop_rotation)
    mirrored = (matched.object_index == 0) & (torch.arange(2 * SEEDS) >= SEEDS)
    assert mirrored.any() and (determinant[mirrored] < 0).all()
    assert (determinant[~mirrored] > 0).all()


def test_exactly_one_template_view_keeps_a_shared_label(template):
    rotation = approach_to_rotation(torch.tensor([0.3, -0.5, 0.8]), torch.tensor(0.7))
    rotated = template.views @ rotation.T
    view_map = ((template.views[:, None] - rotated[None]) ** 2).sum(-1).argmin(dim=1)
    groups = [torch.nonzero(view_map == label).flatten() for label in view_map.unique()]
    assert any(len(group) > 1 for group in groups) and len(groups) < VIEWS
    for rule, pick in (("lowest", min), ("highest", max)):
        kept = CompactLabelAssigner(shared_views=rule, template=template).labelled_views(view_map)
        for group in groups:
            assert kept[group].sum() == 1 and bool(kept[pick(group.tolist())])
    everything = CompactLabelAssigner(shared_views="all", template=template)
    assert bool(everything.labelled_views(view_map).all())


def test_the_distance_limit_is_an_option_and_does_not_touch_the_labels(template):
    seeds, approach, supervision = make_batch(template, seed=1)
    near = CompactLabelAssigner(template=template)(seeds, approach, supervision)
    free = CompactLabelAssigner(max_match_distance=None, template=template)(
        seeds, approach, supervision
    )
    assert near.valid.sum() < free.valid.sum() and torch.equal(free.valid, free.view_labelled)
    assert torch.equal(near.valid, free.valid & (near.distance < 0.005))
    for name in ("angle", "depth", "score", "width", "view_graspness", "crop_rotation"):
        assert torch.equal(getattr(near, name), getattr(free, name))


def test_stored_integers_become_bins_qualities_and_metres(template):
    seeds, approach, supervision = make_batch(template, seed=2)
    matched = CompactLabelAssigner(shared_views="all", template=template)(
        seeds, approach, supervision
    )
    assert matched.angle.min() >= 0 and matched.angle.max() <= 11 and matched.depth.max() <= 3
    levels = matched.score * 10
    assert torch.allclose(levels, levels.round(), atol=1e-5) and matched.score.max() <= 1.0
    millimetres = matched.width * 1000
    assert (
        torch.allclose(millimetres, millimetres.round(), atol=1e-4) and matched.width.max() < 0.15
    )
    unlabelled = CompactLabelAssigner(template=template)(seeds, approach, supervision)
    without = ~unlabelled.view_labelled
    assert without.any() and (unlabelled.angle[without] == -1).all()
    assert (unlabelled.depth[without] == -1).all() and (unlabelled.score[without] == 0).all()


def test_padded_seed_policy(template):
    valid = torch.ones(3 * SEEDS, dtype=torch.bool)
    valid[4:SEEDS] = False  # frame 0: most of its seeds are padding
    valid[2 * SEEDS :] = False  # frame 2: no real seed at all
    seeds, approach, supervision = make_batch(template, frames=3, seed=3, valid=valid)
    options = {"max_match_distance": None, "shared_views": "all", "template": template}
    counted = CompactLabelAssigner(**options)
    ignored = CompactLabelAssigner(padded_seeds="ignore", **options)
    assert counted.padded_seeds == "count"  # the reference averages over every seed slot
    expected = torch.ones(3 * SEEDS, dtype=torch.bool)
    expected[2 * SEEDS :] = False  # slots of a frame without real seeds never count
    for policy, mask in ((counted, expected), (ignored, valid)):
        matched = policy(seeds, approach, supervision)
        assert torch.equal(matched.valid, mask) and torch.equal(matched.view_valid, mask)


def test_frames_and_objects_without_label_points(template):
    assigner = CompactLabelAssigner(max_match_distance=None, template=template)
    seeds, approach, supervision = make_batch(template, seed=4)
    frames = supervision.extras[COMPACT_LABELS]
    empty = make_object(torch.Generator().manual_seed(9), points=0)
    full = assigner(seeds, approach, supervision)
    frames[1].insert(0, empty)  # an object whose points were all filtered out
    shifted = assigner(seeds, approach, supervision)
    assert torch.equal(shifted.object_index[SEEDS:], full.object_index[SEEDS:] + 1)
    assert torch.equal(shifted.angle, full.angle) and torch.equal(shifted.valid, full.valid)
    frames[0] = []  # a frame without objects is unsupervised but keeps the batch running
    matched = assigner(seeds, approach, supervision)
    assert not matched.valid[:SEEDS].any() and not matched.view_valid[:SEEDS].any()
    assert (matched.object_index[:SEEDS] == -1).all() and torch.isinf(
        matched.distance[:SEEDS]
    ).all()
    assert torch.equal(matched.crop_rotation[:SEEDS], approach.rotation[:SEEDS])  # template frame
    assert matched.view_valid[SEEDS:].all()
    nothing = assigner(seeds, approach, Supervision(extras={COMPACT_LABELS: [[empty], [empty]]}))
    assert not nothing.valid.any() and not nothing.view_valid.any()
    with pytest.raises(ContractError, match="without any labelled object"):
        assigner(seeds, approach, Supervision(extras={COMPACT_LABELS: [[], []]}))


def test_labels_that_do_not_fit_are_refused(template):
    assigner = CompactLabelAssigner(template=template)
    seeds, approach, supervision = make_batch(template, seed=5)
    frames = supervision.extras[COMPACT_LABELS]
    with pytest.raises(
        ContractError, match=r"compact labels for 1 frame\(s\), but the batch has 2"
    ):
        assigner(seeds, approach, Supervision(extras={COMPACT_LABELS: frames[:1]}))

    generator = torch.Generator().manual_seed(0)
    dense = ObjectGraspLabels(
        0, torch.eye(4), torch.zeros(4, 3), torch.zeros(4, VIEWS, 3, 2), torch.zeros(4, VIEWS, 3, 2)
    )
    with pytest.raises(ContractError, match="does not carry .* dense label objects: 2"):
        assigner(seeds, approach, Supervision(objects=[[dense], [dense]]))
    with pytest.raises(ContractError, match="expects CompactGraspLabels, got ObjectGraspLabels"):
        assigner(seeds, approach, Supervision(extras={COMPACT_LABELS: [[dense], [dense]]}))

    other = [[make_object(generator, views=8)], frames[1]]
    with pytest.raises(ContractError, match="labelled on 8 views, but the view template has 12"):
        assigner(seeds, approach, Supervision(extras={COMPACT_LABELS: other}))

    mixed = make_object(generator)
    mixed.view_max_friction = 0.8
    with pytest.raises(ContractError, match=r"different friction levels \[0.6, 0.8\]"):
        assigner(seeds, approach, Supervision(extras={COMPACT_LABELS: [[mixed], frames[1]]}))

    bare = ApproachSet(direction=approach.direction)
    with pytest.raises(ContractError, match="view indices and template frames"):
        assigner(seeds, bare, supervision)


def test_the_view_template_is_injected_and_options_are_checked(template):
    with pytest.raises(BuildError, match="needs template"):
        ASSIGNERS.build({"type": "CompactLabelAssigner"})
    with pytest.raises(BuildError, match="must be the ViewTemplate of the view stage"):
        ASSIGNERS.build({"type": "CompactLabelAssigner"}, template=torch.zeros(12, 3))
    with pytest.raises(ValueError, match="unknown policy"):
        CompactLabelAssigner(shared_views="first", template=template)
    with pytest.raises(ValueError, match="must be positive or null"):
        CompactLabelAssigner(max_match_distance=0.0, template=template)
    assigner = CompactLabelAssigner(template=template)
    assert assigner.max_match_distance == 0.005 and assigner.shared_views == "lowest"
    assert assigner.label_format == "compact"
    assert assigner.template is template  # shared with the view stage, never re-derived
