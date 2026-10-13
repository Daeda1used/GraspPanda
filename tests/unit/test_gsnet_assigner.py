"""NearestLabelAssigner against a brute-force oracle written out in NumPy, its policies
and the inputs it must refuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.assigners.nearest_label import NearestLabelAssigner
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import ASSIGNERS, BuildError
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    ObjectGraspLabels,
    PointSet,
    SeedSet,
    Supervision,
)

VIEWS, ANGLES, DEPTHS, SEEDS = 12, 3, 2, 10


@pytest.fixture
def template() -> ViewTemplate:
    return ViewTemplate.fibonacci(VIEWS)


def make_object(generator, *, mirrored: bool = False, collision: bool = False, points: int = 15):
    rotation = approach_to_rotation(
        torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
    )
    if mirrored:  # what a horizontal-flip augmentation does to an object pose
        rotation = torch.diag(torch.tensor([-1.0, 1.0, 1.0])) @ rotation
    pose = torch.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = torch.rand(3, generator=generator)
    levels = torch.tensor([-1.0, 0.1, 0.4, 0.8])
    shape = (points, VIEWS, ANGLES, DEPTHS)
    return ObjectGraspLabels(
        object_id=int(torch.randint(0, 88, (), generator=generator)),
        pose=pose,
        points=torch.rand(points, 3, generator=generator) * 0.2 - 0.1,
        scores=levels[torch.randint(0, 4, shape, generator=generator)],
        widths=torch.rand(shape, generator=generator) * 0.12,
        collision=(torch.rand(shape, generator=generator) < 0.3) if collision else None,
    )


def make_batch(template, frames: int = 2, seed: int = 0, valid: torch.Tensor | None = None):
    generator = torch.Generator().manual_seed(seed)
    points = PointSet.from_dense(torch.rand(frames, SEEDS, 3, generator=generator))
    seeds = SeedSet(points=points, valid=valid)
    index = torch.randint(0, VIEWS, (frames * SEEDS,), generator=generator)
    approach = ApproachSet(
        direction=template.approach(index), rotation=template.rotation(index), view_index=index
    )
    objects = [
        [
            make_object(generator, mirrored=(frame == 1 and n == 0), collision=(n == 1))
            for n in range(3)
        ]
        for frame in range(frames)
    ]
    return seeds, approach, Supervision(objects=objects)


def oracle(seed_xyz, view_index, objects, template):
    """Label matching of one frame, seed by seed, in float64."""
    views = template.views.double().numpy()
    frames = template.rotations.double().numpy()
    cloud, owner = [], []
    view_maps = []
    for position, obj in enumerate(objects):
        rotation = obj.pose[:3, :3].double().numpy()
        cloud.append(obj.points.double().numpy() @ rotation.T + obj.pose[:3, 3].double().numpy())
        owner += [(position, p) for p in range(len(obj.points))]
        rotated = views @ rotation.T
        distance = ((views[:, None, :] - rotated[None, :, :]) ** 2).sum(-1)
        view_maps.append(distance.argmin(axis=1))  # template view → object view
    cloud = np.concatenate(cloud)
    results = []
    for xyz, view in zip(seed_xyz.double().numpy(), view_index.tolist()):
        distance = np.linalg.norm(cloud - xyz, axis=1)
        nearest = int(distance.argmin())
        position, point = owner[nearest]
        obj, view_map = objects[position], view_maps[position]
        scores = obj.scores[point].numpy()[view_map].copy()
        if obj.collision is not None:
            scores[obj.collision[point].numpy()[view_map]] = 0.0
        results.append(
            {
                "label_point": cloud[nearest],
                "distance": distance[nearest],
                "object_index": position,
                "view_friction": scores,
                "friction": scores[view],
                "width": obj.widths[point].numpy()[view_map[view]],
                "crop_rotation": obj.pose[:3, :3].double().numpy() @ frames[view_map[view]],
            }
        )
    return results


def test_matches_the_brute_force_definition(template):
    assigner = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)
    seeds, approach, supervision = make_batch(template)
    matched = assigner(seeds, approach, supervision)
    assert matched.view_friction.shape == (2 * SEEDS, VIEWS, ANGLES, DEPTHS)
    assert matched.friction.shape == matched.width.shape == (2 * SEEDS, ANGLES, DEPTHS)
    for frame in range(2):
        rows = slice(frame * SEEDS, (frame + 1) * SEEDS)
        expected = oracle(
            seeds.points.coords[rows],
            approach.view_index[rows],
            supervision.objects[frame],
            template,
        )
        for seed, reference in zip(range(rows.start, rows.stop), expected):
            for name, value in reference.items():
                produced = getattr(matched, name)[seed].double().numpy()
                assert np.allclose(produced, value, atol=1e-5), (seed, name)
    assert bool(matched.valid.all())  # the reference accepts every match
    assert torch.equal(matched.crop_centre, seeds.points.coords)
    # A mirrored pose yields a reflection as crop frame; nothing "repairs" it.
    determinant = torch.linalg.det(matched.crop_rotation)
    mirrored = (matched.object_index == 0) & (torch.arange(2 * SEEDS) >= SEEDS)
    assert (
        mirrored.any() and (determinant[mirrored] < 0).all() and (determinant[~mirrored] > 0).all()
    )
    # Collision-masked and invalid cells are present, so both code paths were exercised.
    assert (matched.view_friction == 0).any() and (matched.view_friction < 0).any()


def test_a_distance_limit_excludes_far_seeds_from_the_losses(template):
    seeds, approach, supervision = make_batch(template, seed=1)
    free = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)
    distance = free(seeds, approach, supervision).distance
    limit = float(distance.median())
    limited = ASSIGNERS.build(
        {"type": "NearestLabelAssigner", "max_match_distance": limit}, template=template
    )
    matched = limited(seeds, approach, supervision)
    assert torch.equal(matched.valid, distance <= limit) and 0 < matched.valid.sum() < len(seeds)
    assert torch.equal(matched.friction, free(seeds, approach, supervision).friction)  # labels stay


def test_crop_centre_policy(template):
    seeds, approach, supervision = make_batch(template, seed=2)
    at_label = ASSIGNERS.build(
        {"type": "NearestLabelAssigner", "crop_centre": "label_point"}, template=template
    )
    matched = at_label(seeds, approach, supervision)
    assert torch.equal(matched.crop_centre, matched.label_point)
    assert not torch.allclose(matched.crop_centre, seeds.points.coords)


def test_padded_seed_policy(template):
    valid = torch.ones(3 * SEEDS, dtype=torch.bool)
    valid[4:SEEDS] = False  # frame 0: six of its seeds are padding
    valid[2 * SEEDS :] = False  # frame 2: no real seed at all
    seeds, approach, supervision = make_batch(template, frames=3, seed=3, valid=valid)
    counted = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)
    ignored = ASSIGNERS.build(
        {"type": "NearestLabelAssigner", "padded_seeds": "ignore"}, template=template
    )
    assert counted.padded_seeds == "count"  # the reference averages over every seed slot
    expected = torch.ones(3 * SEEDS, dtype=torch.bool)
    expected[2 * SEEDS :] = False  # slots of a frame without real seeds never count
    assert torch.equal(counted(seeds, approach, supervision).valid, expected)
    assert torch.equal(ignored(seeds, approach, supervision).valid, valid)


def test_a_frame_without_objects_is_unsupervised_but_keeps_the_batch_running(template):
    assigner = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)
    seeds, approach, supervision = make_batch(template, seed=4)
    supervision.objects[0] = []
    matched = assigner(seeds, approach, supervision)
    assert not matched.valid[:SEEDS].any() and matched.valid[SEEDS:].all()
    assert (matched.object_index[:SEEDS] == -1).all() and torch.isinf(
        matched.distance[:SEEDS]
    ).all()
    assert torch.equal(matched.crop_rotation[:SEEDS], approach.rotation[:SEEDS])  # template frame
    assert torch.equal(matched.crop_centre[:SEEDS], seeds.points.coords[:SEEDS])
    assert matched.view_friction[:SEEDS].abs().sum() == 0 and matched.width[:SEEDS].abs().sum() == 0
    supervision.objects[1] = []
    with pytest.raises(ContractError, match="without any labelled object"):
        assigner(seeds, approach, supervision)


def test_labels_that_do_not_fit_are_refused(template):
    assigner = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)
    seeds, approach, supervision = make_batch(template, seed=5)
    with pytest.raises(ContractError, match=r"objects for 1 frame\(s\), but the batch has 2"):
        assigner(seeds, approach, Supervision(objects=supervision.objects[:1]))

    other = supervision.objects[0][0]
    supervision.objects[0][0] = ObjectGraspLabels(
        other.object_id, other.pose, other.points, other.scores[:, :8], other.widths[:, :8]
    )
    with pytest.raises(ContractError, match="labelled on 8 views, but the view template has 12"):
        assigner(seeds, approach, supervision)

    supervision.objects[0][0] = ObjectGraspLabels(
        other.object_id, other.pose, other.points, other.scores[..., :1], other.widths[..., :1]
    )
    with pytest.raises(ContractError, match="different angle × depth grids"):
        assigner(seeds, approach, supervision)

    supervision.objects[0][0] = ObjectGraspLabels(
        other.object_id, other.pose, other.points, other.scores, other.widths[:5]
    )
    with pytest.raises(ContractError, match=r"scores and widths must both be \(P, V, A, D\)"):
        assigner(seeds, approach, supervision)

    bare = ApproachSet(direction=approach.direction)
    with pytest.raises(ContractError, match="view indices and template frames"):
        assigner(seeds, bare, make_batch(template, seed=5)[2])


def test_the_view_template_is_injected_and_options_are_checked(template):
    with pytest.raises(BuildError, match="needs template"):
        ASSIGNERS.build({"type": "NearestLabelAssigner"})
    with pytest.raises(BuildError, match="must be the ViewTemplate of the view stage"):
        ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=torch.zeros(12, 3))
    with pytest.raises(ValueError, match="unknown policy"):
        NearestLabelAssigner(crop_centre="object", template=template)
    with pytest.raises(ValueError, match="must be positive or null"):
        NearestLabelAssigner(max_match_distance=0.0, template=template)
    assigner = NearestLabelAssigner(template=template)
    assert assigner.max_match_distance is None and assigner.crop_centre == "seed"
    assert assigner.template is template  # shared with the view stage, never re-derived
