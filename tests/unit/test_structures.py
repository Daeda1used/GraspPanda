"""Data contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.ops import approach_to_rotation
from grasppanda.structures import (
    ContractError,
    GraspGroup,
    GroupFeatures,
    Mask,
    Observation,
    PointFeatures,
    PointSet,
    Provenance,
    SampleKey,
    SeedSet,
)


def random_grasps(count: int = 5, seed: int = 0) -> GraspGroup:
    generator = torch.Generator().manual_seed(seed)
    approach = torch.randn(count, 3, generator=generator)
    return GraspGroup(
        translation=torch.randn(count, 3, generator=generator),
        rotation=approach_to_rotation(approach, torch.rand(count, generator=generator) * math.pi),
        width=torch.rand(count, generator=generator) * 0.1,
        depth=torch.full((count,), 0.02),
        score=torch.rand(count, generator=generator),
    )


def test_point_set_dense_round_trip_and_identity():
    cloud = torch.rand(2, 5, 3)
    points = PointSet.from_dense(cloud)
    assert len(points) == 10 and points.batch_size == 2
    assert torch.equal(points.dense(), cloud)
    assert points.to("cpu").id == points.id
    assert PointSet.from_dense(cloud).id != points.id


def test_point_set_rejects_bad_shapes_and_ragged_dense():
    with pytest.raises(ContractError, match=r"must be \(N, 3\)"):
        PointSet(torch.rand(4, 2))
    ragged = PointSet(torch.rand(5, 3), torch.tensor([0, 2, 5]))
    with pytest.raises(ContractError, match="equally sized samples"):
        ragged.dense()


def test_features_are_bound_to_their_point_set():
    scene = PointSet.from_dense(torch.rand(1, 8, 3))
    seeds = scene.subset(torch.tensor([0, 3, 5]))
    assert seeds.parent_id == scene.id
    feats = PointFeatures(torch.rand(3, 4), seeds)
    feats.expect(seeds, user="test")
    with pytest.raises(ContractError, match="add an explicit propagation step"):
        feats.expect(scene, user="GraspnessFPS")
    with pytest.raises(ContractError, match=r"must be \(N, C\) with N = 8"):
        PointFeatures(torch.rand(3, 4), scene)


def test_group_features_name_their_axes():
    groups = GroupFeatures(torch.rand(6, 4, 16), ("seed", "depth", "channel"))
    assert groups.size("depth") == 4
    with pytest.raises(ContractError, match=r"expects features with axes \('seed', 'channel'\)"):
        groups.expect(("seed", "channel"), user="SWADHead")
    with pytest.raises(ContractError):
        GroupFeatures(torch.rand(6, 16), ("seed", "depth", "channel"))


def test_seed_set_checks_row_counts():
    points = PointSet(torch.rand(4, 3))
    SeedSet(points, scores={"graspness": torch.rand(4)})
    with pytest.raises(ContractError, match="SeedSet.graspness has 3 rows for 4 seeds"):
        SeedSet(points, scores={"graspness": torch.rand(3)})


def test_observation_reports_ground_truth_inputs():
    observation = Observation(
        keys=[SampleKey("graspnet1b", 100, "realsense", 0)],
        points=PointSet(torch.rand(4, 3)),
        masks={
            "workspace": Mask(torch.ones(4, dtype=torch.bool), Provenance.GROUND_TRUTH),
            "instance": Mask(torch.ones(4, dtype=torch.bool), Provenance.PREDICTED),
        },
    )
    assert observation.gt_assisted() == ["workspace"]
    assert observation.provides() == {"points", "workspace", "instance"}
    assert str(observation.keys[0]) == "graspnet1b/scene_0100/realsense/0000"


def test_grasp_group_graspnet_array_round_trip():
    grasps = random_grasps()
    array = grasps.to_graspnet_array()
    assert array.shape == (5, 17) and array.dtype == np.float32
    assert np.allclose(array[:, 0], grasps.score.numpy())
    assert np.allclose(array[:, 4:13].reshape(5, 3, 3), grasps.rotation.numpy())  # row-major
    assert np.allclose(array[:, 13:16], grasps.translation.numpy())
    assert (array[:, 2] == np.float32(0.02)).all() and (array[:, 16] == -1).all()
    back = GraspGroup.from_graspnet_array(array)
    for name in ("translation", "rotation", "width", "depth", "score", "height", "object_id"):
        assert torch.equal(getattr(back, name), getattr(grasps, name)), name


def test_grasp_group_container_operations():
    grasps = random_grasps(6)
    assert len(grasps[2]) == 1 and len(grasps[1:4]) == 3 and len(grasps[grasps.score > 2]) == 0
    assert len(GraspGroup.cat([grasps, grasps[:2]])) == 8
    assert len(GraspGroup.empty()) == 0 and GraspGroup.empty().to_graspnet_array().shape == (0, 17)
    with pytest.raises(ContractError, match="different frame"):
        GraspGroup.cat([grasps, grasps.transform(torch.eye(4), frame="world")])


def test_sort_is_stable_for_tied_scores():
    grasps = random_grasps(4)
    grasps.score = torch.tensor([0.5, 0.9, 0.5, 0.5])
    grasps.object_id = torch.arange(4)
    assert grasps.sort_by_score().object_id.tolist() == [1, 0, 2, 3]


def test_validate_rejects_unphysical_grasps():
    random_grasps().validate()
    GraspGroup.empty().validate()

    bad = random_grasps()
    bad.width[0] = -0.01
    with pytest.raises(ContractError, match="1 negative width"):
        bad.validate()

    bad = random_grasps()
    bad.rotation[1] *= 1.1
    with pytest.raises(ContractError, match="not orthonormal"):
        bad.validate()

    bad = random_grasps()
    bad.rotation[:, :, 2] *= -1
    with pytest.raises(ContractError, match="reflection"):
        bad.validate()

    bad = random_grasps()
    bad.score[2] = float("nan")
    with pytest.raises(ContractError, match="score contains NaN or Inf"):
        bad.validate()


def test_transform_moves_grasps_rigidly():
    grasps = random_grasps()
    matrix = torch.eye(4)
    matrix[:3, :3] = approach_to_rotation(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(0.7))
    matrix[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    moved = grasps.transform(matrix, frame="world").validate()
    assert moved.frame == "world" and grasps.frame == "camera"
    back = moved.transform(torch.linalg.inv(matrix), frame="camera")
    assert torch.allclose(back.translation, grasps.translation, atol=1e-5)
    assert torch.allclose(back.rotation, grasps.rotation, atol=1e-5)


def _frame(count: int, scene: int, with_labels: bool = True):
    from grasppanda.structures import Batch, ObjectGraspLabels, Supervision

    observation = Observation(
        keys=[SampleKey("demo", scene, "cam", 0)],
        intrinsics=torch.eye(3)[None],
        points=PointSet(torch.rand(count, 3)),
        point_feats=torch.ones(count, 3),
        provenance={"workspace": Provenance.GROUND_TRUTH},
    )
    labels = ObjectGraspLabels(
        1, torch.eye(4), torch.rand(5, 3), torch.rand(5, 2, 2, 2), torch.rand(5, 2, 2, 2)
    )
    supervision = Supervision(objects=[[labels]], point_labels={"graspness": torch.rand(count, 1)})
    return Batch(observation, supervision if with_labels else None)


def test_collate_merges_frames_of_different_sizes():
    from grasppanda.structures import collate

    batch = collate([_frame(4, 0), _frame(6, 1)])
    assert len(batch) == 2
    assert batch.observation.points.batch_offsets.tolist() == [0, 4, 10]
    assert batch.observation.points.batch_index.tolist() == [0] * 4 + [1] * 6
    assert batch.observation.point_feats.shape == (
        10,
        3,
    ) and batch.observation.intrinsics.shape == (2, 3, 3)
    assert batch.supervision.point_labels["graspness"].shape == (10, 1)
    assert [len(objects) for objects in batch.supervision.objects] == [1, 1]
    assert batch.observation.gt_assisted() == ["workspace"]
    assert batch.to("cpu").observation.points.id == batch.observation.points.id


def test_collate_refuses_inconsistent_frames():
    from grasppanda.structures import collate

    with pytest.raises(ContractError, match="with and without supervision"):
        collate([_frame(4, 0), _frame(4, 1, with_labels=False)])
    other = _frame(4, 1)
    other.observation.provenance = {}
    with pytest.raises(ContractError, match="different provenance"):
        collate([_frame(4, 0), other])


def test_predictions_round_trip_through_the_official_layout(tmp_path):
    from grasppanda.evaluation import load_prediction, prediction_path, save_prediction

    key = SampleKey("graspnet1b", 100, "realsense", 7)
    grasps = random_grasps(3)
    grasps.key = key
    path = save_prediction(tmp_path, grasps)
    assert (
        path == tmp_path / "scene_0100" / "realsense" / "0007.npy" == prediction_path(tmp_path, key)
    )
    loaded = load_prediction(tmp_path, key)
    assert loaded.key == key and torch.equal(loaded.rotation, grasps.rotation)
    empty = GraspGroup.empty(key=SampleKey("graspnet1b", 100, "realsense", 8))
    assert len(load_prediction(tmp_path, save_prediction(tmp_path, empty) and empty.key)) == 0
