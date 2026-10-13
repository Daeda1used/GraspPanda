"""The inference-time post-processors on analytic scenes: what each keeps, moves or refuses."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter
from grasppanda.models.postprocessors.scene import as_transform, fit_plane
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import (
    ContractError,
    GraspGroup,
    Mask,
    Observation,
    PointSet,
    Provenance,
    SampleKey,
)

KEY = SampleKey("unit", 0, "none", 0)
#: Camera looking along +z at a table 0.6 m away: gripper x (approach) = camera z.
DOWN = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T
#: Approaching along camera x: parallel to the table.
FLAT = torch.eye(3)


def build(name: str, **options):
    return POSTPROCESSORS.build({"type": name, **options})


def grasps_at(points, rotation=DOWN, width=0.06, depth=0.02, score=None, **meta) -> GraspGroup:
    points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
    count = len(points)
    rotation = torch.as_tensor(rotation, dtype=torch.float32)
    if rotation.ndim == 2:
        rotation = rotation.expand(count, 3, 3).clone()
    return GraspGroup(
        translation=points,
        rotation=rotation,
        width=torch.full((count,), width),
        depth=torch.full((count,), depth),
        score=torch.arange(count, 0, -1).float() if score is None else torch.as_tensor(score),
        key=KEY,
        **meta,
    )


def table_scene(seed: int = 0) -> torch.Tensor:
    """A table at z = 0.6 and a 4 cm wide, 5 cm tall bar on it, centred at x = y = 0."""
    generator = torch.Generator().manual_seed(seed)
    table = torch.rand(4000, 3, generator=generator) * torch.tensor([0.6, 0.6, 0.0])
    table += torch.tensor([-0.3, -0.3, 0.6])
    bar = torch.rand(1500, 3, generator=generator) * torch.tensor([0.10, 0.04, 0.05])
    bar += torch.tensor([-0.05, -0.02, 0.55])
    return torch.cat([table, bar])


def observe(points: torch.Tensor, **extra) -> Observation:
    return Observation(
        keys=[KEY], points=PointSet(points), full_points=PointSet(points.clone()), **extra
    )


EMPTY = observe(torch.zeros(4, 3))


def test_every_new_post_processor_is_registered_with_source_status_and_a_primary_option():
    names = [
        "TopK",
        "ScoreThreshold",
        "FurthestGraspSampling",
        "WorkspaceFilter",
        "ApproachAngleFilter",
        "TablePlaneFilter",
        "MaskFilter",
        "GripperLimits",
        "GripperCentering",
        "WidthFromCloud",
        "ToFrame",
        "GripperMeshCollisionFilter",
    ]
    for name in names:
        cls = POSTPROCESSORS.get(name)
        assert cls.source and cls.status in ("tested", "bridge"), name
        assert cls.__doc__.strip().splitlines()[0].endswith((".", ":")), name
        assert cls.primary in cls.__init__.__code__.co_varnames, name
    assert POSTPROCESSORS.get("GripperMeshCollisionFilter").extras == ("trimesh",)


def test_an_empty_prediction_passes_through_every_step():
    empty = GraspGroup.empty(key=KEY)
    steps = [
        build("TopK"),
        build("ScoreThreshold"),
        build("FurthestGraspSampling"),
        build("WorkspaceFilter", z=[0, 1]),
        build("ApproachAngleFilter", direction="camera"),
        build("TablePlaneFilter", plane=[0, 0, -1, 0.6]),
        build("GripperLimits"),
        build("GripperCentering"),
        build("WidthFromCloud"),
        build("ToFrame", transform=torch.eye(4).tolist()),
    ]
    for step in steps:
        assert len(step(empty, EMPTY)) == 0, type(step).__name__


# ----------------------------------------------------------------------- selection


def test_top_k_keeps_the_best_in_order_and_refuses_a_k_of_zero():
    grasps = grasps_at(torch.rand(6, 3), score=[0.1, 0.9, 0.5, 0.9, 0.3, 0.7])
    kept = build("TopK", k=3)(grasps, EMPTY)
    assert kept.score.tolist() == pytest.approx([0.9, 0.9, 0.7])
    assert torch.equal(kept.translation[0], grasps.translation[1])  # ties keep their order
    assert len(build("TopK", k=50)(grasps, EMPTY)) == 6
    with pytest.raises(ValueError, match="k must be at least 1"):
        build("TopK", k=0)


def test_score_threshold_in_its_three_modes_and_its_floor():
    grasps = grasps_at(torch.rand(5, 3), score=[-2.0, 0.0, 1.0, 3.0, 8.0])
    assert build("ScoreThreshold", threshold=1.0)(grasps, EMPTY).score.tolist() == [1.0, 3.0, 8.0]
    assert build("ScoreThreshold", threshold=0.5, mode="relative")(
        grasps, EMPTY
    ).score.tolist() == [3.0, 8.0]
    assert build("ScoreThreshold", threshold=0.75, mode="quantile")(
        grasps, EMPTY
    ).score.tolist() == [3.0, 8.0]
    floor = build("ScoreThreshold", threshold=100.0, min_keep=2)(grasps, EMPTY)
    assert sorted(floor.score.tolist()) == [3.0, 8.0]
    with pytest.raises(ValueError, match=r"share in \[0, 1\]"):
        build("ScoreThreshold", threshold=3.0, mode="quantile")


def test_furthest_grasp_sampling_spreads_over_clusters():
    generator = torch.Generator().manual_seed(0)
    centres = torch.tensor([[0.0, 0, 0.5], [0.2, 0, 0.5], [0, 0.2, 0.5], [0.2, 0.2, 0.5]])
    points = (centres[:, None] + torch.randn(4, 25, 3, generator=generator) * 0.003).reshape(-1, 3)
    score = torch.cat([torch.full((25,), 4.0 - index) for index in range(4)])
    grasps = grasps_at(points, score=score + torch.rand(100, generator=generator) * 0.1)
    top = build("TopK", k=4)(grasps, EMPTY)
    assert (top.translation - centres[0]).norm(dim=1).max() < 0.02  # all on one cluster
    spread = build("FurthestGraspSampling", num=4, candidates=None)(grasps, EMPTY)
    nearest = torch.cdist(spread.translation, centres).argmin(dim=1)
    assert sorted(nearest.tolist()) == [0, 1, 2, 3]
    assert spread.score[0] == grasps.score.max()  # the best grasp is always among them
    assert torch.all(spread.score[:-1] >= spread.score[1:])
    limited = build("FurthestGraspSampling", num=4, candidates=25)(grasps, EMPTY)
    assert (limited.translation - centres[0]).norm(dim=1).max() < 0.02
    with pytest.raises(ValueError, match="candidates at least num"):
        build("FurthestGraspSampling", num=10, candidates=5)


# -------------------------------------------------------------------------- region


def test_workspace_filter_in_the_grasp_frame_and_in_a_calibrated_frame():
    grasps = grasps_at([[0.0, 0.0, 0.5], [0.0, 0.0, 0.9], [0.4, 0.0, 0.5]])
    kept = build("WorkspaceFilter", x=[-0.2, 0.2], z=[0.3, 0.7])(grasps, EMPTY)
    assert kept.translation.tolist() == [[0.0, 0.0, 0.5]]
    # base frame: z up, the camera 1 m above the base looking down
    base_from_camera = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]
    lifted = build("WorkspaceFilter", z=[0.4, 0.6], transform=base_from_camera)(grasps, EMPTY)
    assert len(lifted) == 2 and lifted.frame == "camera"
    with pytest.raises(ValueError, match="at least one of x, y, z"):
        build("WorkspaceFilter")
    with pytest.raises(ValueError, match=r"\[min, max\]"):
        build("WorkspaceFilter", z=[0.7, 0.3])


def test_fit_plane_finds_the_table_with_its_normal_towards_the_camera():
    plane, share = fit_plane(table_scene())
    assert plane.tolist() == pytest.approx([0.0, 0.0, -1.0, 0.6], abs=1e-3)
    assert share > 0.7


def test_approach_angle_filter_rejects_flat_and_upward_grasps():
    up = DOWN @ torch.diag(torch.tensor([-1.0, 1.0, -1.0]))  # approaching from below
    tilt = math.radians(40)
    tilted = torch.tensor(
        [[math.sin(tilt), 0, math.cos(tilt)], [0, 1, 0], [-math.cos(tilt), 0, math.sin(tilt)]]
    ).T
    rotation = torch.stack([DOWN, FLAT, up, tilted])
    grasps = grasps_at(torch.tensor([[0.0, 0.0, 0.5]]).repeat(4, 1), rotation=rotation)
    scene = observe(table_scene())
    for direction in ("table_normal", "camera", [0.0, 0.0, 2.0]):
        kept = build("ApproachAngleFilter", direction=direction, max_angle=60)(grasps, scene)
        assert kept.score.tolist() == [4.0, 1.0], direction
    narrow = build("ApproachAngleFilter", direction="camera", max_angle=30)(grasps, scene)
    assert narrow.score.tolist() == [4.0]
    fallback = build("ApproachAngleFilter", direction=[0, 0, -1], max_angle=10, min_keep=1)
    assert fallback(grasps, scene).score.tolist() == [2.0]  # the best aligned: from below
    with pytest.raises(ContractError, match="probably not the table"):
        ball = torch.nn.functional.normalize(torch.randn(3000, 3), dim=1) * 0.2
        build("ApproachAngleFilter", min_inliers=0.5)(grasps, observe(ball))
    with pytest.raises(ValueError, match="'table_normal', 'camera' or a vector"):
        build("ApproachAngleFilter", direction="gravity")


def test_table_plane_filter_measures_the_lowest_gripper_corner():
    # fingertips at z = t_z + depth; the table is at z = 0.6
    heights = torch.tensor([0.50, 0.57, 0.59])
    grasps = grasps_at(torch.stack([torch.zeros(3), torch.zeros(3), heights], dim=1), depth=0.02)
    step = build("TablePlaneFilter")
    lowest = step.heights(grasps, torch.tensor([0.0, 0.0, -1.0, 0.6]))
    assert lowest.tolist() == pytest.approx([0.08, 0.01, -0.01], abs=1e-6)
    scene = observe(table_scene())
    assert len(step(grasps, scene)) == 2
    assert len(build("TablePlaneFilter", clearance=0.02)(grasps, scene)) == 1
    assert len(build("TablePlaneFilter", plane=[0, 0, -1, 0.55])(grasps, EMPTY)) == 1
    # a flat grasp low over the table: a finger corner dips below although its centre does not
    flat = grasps_at([[0.0, 0.0, 0.595]], rotation=FLAT, width=0.08)
    assert len(step(flat, scene)) == 0


def test_mask_filter_keeps_the_grasps_on_the_target_and_names_a_missing_mask():
    points = table_scene()
    target = torch.zeros(len(points), dtype=torch.bool)
    target[4000:] = True  # the bar
    scene = observe(points, masks={"target": Mask(target, Provenance.PREDICTED)})
    grasps = grasps_at([points[4100].tolist(), points[10].tolist(), [0.0, 0.0, 0.2]])
    kept = build("MaskFilter", mask="target", max_distance=0.005)(grasps, scene)
    assert kept.score.tolist() == [3.0]
    others = build("MaskFilter", mask="target", max_distance=0.005, invert=True)(grasps, scene)
    assert others.score.tolist() == [2.0, 1.0]
    with pytest.raises(ContractError, match=r"masks\['object'\], which this observation does not"):
        build("MaskFilter", mask="object")(grasps, scene)
    scene.masks["short"] = Mask(target[:10], Provenance.PREDICTED)
    with pytest.raises(ContractError, match="one value per point"):
        build("MaskFilter", mask="short")(grasps, scene)


# ------------------------------------------------------------------------- gripper


def test_gripper_limits_remove_or_clip():
    grasps = grasps_at(torch.rand(4, 3))
    grasps.width = torch.tensor([0.02, 0.07, 0.09, 0.10])
    grasps.depth = torch.tensor([0.01, 0.04, 0.02, 0.03])
    assert build("GripperLimits", max_width=0.08)(grasps, EMPTY).width.tolist() == pytest.approx(
        [0.02, 0.07]
    )
    assert len(build("GripperLimits", max_width=None)(grasps, EMPTY)) == 4  # the gripper's 0.10
    limited = build("GripperLimits", max_width=0.08, min_width=0.03, max_depth=0.03)
    assert len(limited(grasps, EMPTY)) == 0
    clipped = build("GripperLimits", max_width=0.08, max_depth=0.03, mode="clip")(grasps, EMPTY)
    assert clipped.width.tolist() == pytest.approx([0.02, 0.07, 0.08, 0.08])
    assert clipped.depth.tolist() == pytest.approx([0.01, 0.03, 0.02, 0.03])
    with pytest.raises(ValueError, match="min_width <= max_width"):
        build("GripperLimits", max_width=0.02, min_width=0.05)


def test_centering_and_width_follow_the_object_between_the_fingers():
    scene = observe(table_scene())  # the bar spans y in [-0.02, 0.02], top at z = 0.55
    # closing axis = camera y; the grasp is 1.5 cm off the middle of the bar
    grasps = grasps_at([[0.0, 0.015, 0.54], [0.0, 0.3, 0.30]], width=0.09, depth=0.03)
    centred = build("GripperCentering")(grasps, scene)
    assert centred.translation[0].tolist() == pytest.approx([0.0, 0.0, 0.54], abs=2e-3)
    exact = build("GripperCentering", voxel_size=0.0)(grasps, scene)
    assert exact.translation[0].tolist() == pytest.approx([0.0, 0.0, 0.54], abs=1e-3)
    assert torch.equal(centred.translation[1], grasps.translation[1])  # nothing between: as is
    assert torch.equal(centred.score, grasps.score) and torch.equal(centred.width, grasps.width)
    capped = build("GripperCentering", max_shift=0.005)(grasps, scene)
    assert capped.translation[0, 1].item() == pytest.approx(0.010, abs=1e-6)

    fitted = build("WidthFromCloud", clearance=0.01, voxel_size=0.0)(centred, scene)
    assert fitted.width.tolist() == pytest.approx([0.04 + 0.02, 0.09], abs=2e-3)
    coarse = build("WidthFromCloud", clearance=0.01)(centred, scene)  # 5 mm voxels: errs wide
    assert 0.06 - 2e-3 <= coarse.width[0].item() <= 0.06 + 5e-3
    narrow = grasps_at([[0.0, 0.0, 0.54]], width=0.03, depth=0.03)
    assert build("WidthFromCloud", margin=0.02, voxel_size=0.0)(
        narrow, scene
    ).width.item() == pytest.approx(0.06, abs=2e-3)  # the fingers would have landed on the bar
    assert build("WidthFromCloud", margin=0.02, mode="shrink")(narrow, scene).width.item() == (
        pytest.approx(0.03)
    )


def test_to_frame_moves_grasps_last_and_cloud_readers_refuse_the_new_frame(tmp_path):
    base_from_camera = [[1, 0, 0, 0.5], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]
    grasps = grasps_at([[0.1, 0.2, 0.6]], depth=0.03)
    moved = build("ToFrame", transform=base_from_camera, frame="base")(grasps, EMPTY)
    assert moved.frame == "base" and moved.key == KEY
    assert moved.translation.tolist() == [pytest.approx([0.6, -0.2, 0.4])]
    assert moved.rotation[0, :, 0].tolist() == pytest.approx([0, 0, -1])  # approaching downwards
    tips = build("ToFrame", transform=base_from_camera, origin="fingertips")(grasps, EMPTY)
    assert tips.translation.tolist() == [pytest.approx([0.6, -0.2, 0.37])]
    assert tips.depth.tolist() == [0.0]
    file = tmp_path / "calibration.txt"
    np.savetxt(file, np.asarray(base_from_camera, dtype=float))
    from_file = build("ToFrame", transform=str(file))(grasps, EMPTY)
    assert torch.allclose(from_file.translation, moved.translation)
    with pytest.raises(ContractError, match="grasps are in frame 'base' but the scene cloud"):
        ModelFreeCollisionFilter()(moved, observe(table_scene()))
    with pytest.raises(ContractError, match="grasps are in frame 'base' but the scene cloud"):
        build("GripperCentering")(moved, observe(table_scene()))
    sampled_only = Observation(keys=[KEY], points=PointSet(table_scene()))  # no full cloud
    with pytest.raises(ContractError, match=r"keep_full_points=true.*or.*`cloud: points`"):
        ModelFreeCollisionFilter()(grasps, sampled_only)
    assert len(build("GripperCentering")(grasps, sampled_only)) == 1  # falls back to points
    with pytest.raises(ValueError, match="4x4 homogeneous matrix"):
        build("ToFrame", transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="not rigid"):
        as_transform([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "ToFrame")


# ---------------------------------------------------------------- mesh collision (bridge)


def test_gripper_mesh_collision_filter_names_the_extra_when_trimesh_is_absent():
    try:
        import trimesh  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r'pip install "grasppanda\[trimesh\]"'):
            build("GripperMeshCollisionFilter")(grasps_at([[0, 0, 0.5]]), observe(table_scene()))


def test_gripper_mesh_collision_filter_with_the_built_in_gripper_and_with_a_mesh_file(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    scene = observe(table_scene())
    # 1: over the middle of the bar, fingers either side: free. 2: 4 cm to the side, a finger
    # in the bar. 3: fingertips 2 cm below the table.
    grasps = grasps_at([[0.0, 0.0, 0.53], [0.0, 0.05, 0.54], [0.2, 0.2, 0.60]], depth=0.02)
    step = build("GripperMeshCollisionFilter", opening=0.08, voxel_size=0.0)
    counts = step.collision_counts(grasps, scene.full_points.coords)
    assert counts[0] == 0 and counts[1] > 0 and counts[2] > 0
    assert step(grasps, scene).score.tolist() == [3.0]
    assert len(build("GripperMeshCollisionFilter", opening=0.08)(grasps, scene)) == 1
    sampled_only = Observation(keys=[KEY], points=PointSet(table_scene()))  # no full cloud
    assert len(build("GripperMeshCollisionFilter", opening=0.08)(grasps, sampled_only)) == 1

    # The same gripper as a file in millimetres with z as its approach axis.
    bodies = [body.copy() for body in step.gripper_bodies(grasps)]
    file_from_tool = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])
    hand = trimesh.util.concatenate(bodies)
    hand.apply_transform(file_from_tool)
    hand.apply_scale(1000.0)
    path = tmp_path / "hand.stl"
    hand.export(path)
    tool_from_file = np.linalg.inv(file_from_tool).tolist()
    for kind in ("exact", "convex"):
        from_file = build(
            "GripperMeshCollisionFilter",
            mesh=str(path),
            scale=0.001,
            mesh_transform=tool_from_file,
            bodies=kind,
            voxel_size=0.0,
        )
        assert torch.equal(from_file.collision_counts(grasps, scene.full_points.coords), counts)
    with pytest.raises(ValueError, match="a mesh in millimetres needs scale=0.001"):
        build("GripperMeshCollisionFilter", mesh=str(path))
    with pytest.raises(ValueError, match="transform file not found|4x4"):
        build("GripperMeshCollisionFilter", mesh=str(path), scale=0.001, mesh_transform=[[1.0]])
