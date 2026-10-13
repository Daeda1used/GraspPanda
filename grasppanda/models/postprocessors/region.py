"""Where a grasp may be and from where it may come: workspace box, approach cone,
table plane, target mask.

These are execution filters for a real robot. The benchmark evaluators reward none of
them, so a variant that uses one reports a lower AP whenever it removes a good grasp;
keep such a variant apart from the benchmark columns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from grasppanda.models.base import PostProcessor
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import ContractError, GraspGroup, Observation

from .scene import CHUNK, as_transform, fit_plane, frame_index, scene_points

__all__ = ["ApproachAngleFilter", "MaskFilter", "TablePlaneFilter", "WorkspaceFilter"]

_Range = Sequence[float] | None


def _range(name: str, value: _Range) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2 or value[0] > value[1]:
        raise ValueError(f"{name} must be [min, max] in metres with min <= max, got {value!r}")
    return float(value[0]), float(value[1])


def _table_plane(
    name: str,
    plane: Sequence[float] | None,
    grasps: GraspGroup,
    observation: Observation,
    cloud: Literal["full_points", "points"],
    threshold: float,
    min_inliers: float,
) -> Tensor:
    """``[a, b, c, d]`` of the table in the frame of the grasps, unit normal towards the camera."""
    if plane is not None:
        return grasps.translation.new_tensor(plane).double()
    scene = scene_points(name, grasps, observation, cloud, fallback=True)
    fitted, share = fit_plane(scene, threshold=threshold)
    if share < min_inliers:
        raise ContractError(
            f"{name}: the largest plane of the cloud holds {share:.0%} of its points, below "
            f"min_inliers ({min_inliers:.0%}); it is probably not the table. Give the plane "
            "(`plane: [a, b, c, d]`, unit normal towards the camera, metres) or lower "
            "min_inliers."
        )
    return fitted


@POSTPROCESSORS.register()
class WorkspaceFilter(PostProcessor):
    """Keep grasps whose grasp point lies inside an axis-aligned box.

    The box is given in the frame of the grasps (the camera frame, unless a ``ToFrame``
    step ran before), or in another frame when ``transform`` maps the grasp frame to it:
    with a camera-to-base calibration the limits are those of the robot's reachable
    region. A limit that is ``None`` is not tested. GPD (``workspace_grasps``) and the
    AnyGrasp SDK (``lims``) ship the same step.

    Args:
        x, y, z: ``[min, max]`` in metres, or ``None``.
        transform: ``(4, 4)`` rigid transform from the frame of the grasps to the frame
            of the box, as nested lists or the path of a ``.npy`` or text file. The
            grasps themselves stay in their frame.
    """

    status = "tested"
    source = (
        "Generic; the workspace limits of GPD (ten Pas et al., arXiv 1706.09911) and AnyGrasp SDK"
    )
    primary = "z"

    def __init__(
        self,
        x: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        z: Sequence[float] | None = None,
        transform: Sequence[Sequence[float]] | str | None = None,
    ) -> None:
        self.limits = [_range(axis, value) for axis, value in (("x", x), ("y", y), ("z", z))]
        if all(limit is None for limit in self.limits):
            raise ValueError(
                "WorkspaceFilter needs at least one of x, y, z as [min, max] in metres"
            )
        self.transform = None if transform is None else as_transform(transform, "WorkspaceFilter")

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        position = grasps.translation
        if self.transform is not None:
            matrix = self.transform.to(position)
            position = position @ matrix[:3, :3].T + matrix[:3, 3]
        keep = torch.ones(len(grasps), dtype=torch.bool, device=position.device)
        for axis, limit in enumerate(self.limits):
            if limit is not None:
                keep &= (position[:, axis] >= limit[0]) & (position[:, axis] <= limit[1])
        return grasps[keep]


@POSTPROCESSORS.register()
class ApproachAngleFilter(PostProcessor):
    """Keep grasps that approach within a cone about a reference direction.

    The approach axis is column 0 of a grasp rotation: the direction the gripper moves
    in to reach the object. With ``direction: table_normal`` the reference is "straight
    down onto the table", found from the cloud, and ``max_angle: 60`` rejects grasps
    that come in flatter than 30° above the table or from below it. With
    ``direction: camera`` it is the viewing direction (camera z), which is the same thing
    for a camera that looks straight down.

    Args:
        direction: ``table_normal``, ``camera``, or a vector ``[x, y, z]`` in the frame
            of the grasps (it is normalised).
        max_angle: Half-angle of the cone, degrees, in ``(0, 180]``.
        min_keep: When fewer grasps pass, keep the ``min_keep`` best-aligned ones
            instead (the fallback of ThinkGrasp's gravity cone). ``0`` may return none.
        plane: With ``table_normal``: the table as ``[a, b, c, d]`` (unit normal towards
            the camera, metres) instead of fitting it to the cloud.
        cloud: With ``table_normal``: the cloud the plane is fitted to.
        plane_threshold: With ``table_normal``: inlier distance of the fit, metres.
        min_inliers: With ``table_normal``: refuse a fitted plane that holds a smaller
            share of the cloud.
    """

    status = "tested"
    source = (
        "Generic; `filter_approach_direction` of GPD (ten Pas et al., arXiv 1706.09911), "
        "`approach_steering` of the AnyGrasp SDK, the gravity cone of ThinkGrasp (github.com/H-Freax/ThinkGrasp)"
    )
    primary = "max_angle"

    def __init__(
        self,
        direction: Sequence[float] | Literal["table_normal", "camera"] = "table_normal",
        max_angle: float = 60.0,
        min_keep: int = 0,
        plane: Sequence[float] | None = None,
        cloud: Literal["full_points", "points"] = "full_points",
        plane_threshold: float = 0.01,
        min_inliers: float = 0.1,
    ) -> None:
        if isinstance(direction, str):
            if direction not in ("table_normal", "camera"):
                raise ValueError(
                    "direction must be 'table_normal', 'camera' or a vector [x, y, z]; "
                    f"got {direction!r}"
                )
        elif len(direction) != 3 or not any(direction):
            raise ValueError(f"direction must be a non-zero vector [x, y, z], got {direction!r}")
        if not 0 < max_angle <= 180:
            raise ValueError(f"max_angle is a half-angle in (0, 180] degrees, got {max_angle}")
        if plane is not None and len(plane) != 4:
            raise ValueError(f"plane must be [a, b, c, d], got {plane!r}")
        self.direction = direction if isinstance(direction, str) else list(direction)
        self.max_angle = max_angle
        self.min_keep = min_keep
        self.plane = None if plane is None else list(plane)
        self.cloud = cloud
        self.plane_threshold = plane_threshold
        self.min_inliers = min_inliers

    def reference(self, grasps: GraspGroup, observation: Observation) -> Tensor:
        """The unit reference direction in the frame of the grasps."""
        if self.direction == "camera":
            if grasps.frame != "camera":
                raise ContractError(
                    f"ApproachAngleFilter: direction 'camera' is the z axis of the camera "
                    f"frame, but the grasps are in frame '{grasps.frame}'. Give the "
                    "direction as a vector in that frame."
                )
            vector = grasps.translation.new_tensor([0.0, 0.0, 1.0])
        elif self.direction == "table_normal":
            plane = _table_plane(
                "ApproachAngleFilter",
                self.plane,
                grasps,
                observation,
                self.cloud,
                self.plane_threshold,
                self.min_inliers,
            )
            vector = -plane[:3].to(grasps.translation)  # into the table
        else:
            vector = grasps.translation.new_tensor(self.direction)
        return vector / vector.norm()

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        cosine = grasps.rotation[:, :, 0] @ self.reference(grasps, observation)
        keep = cosine >= math.cos(math.radians(self.max_angle)) - 1e-7
        if self.min_keep and int(keep.sum()) < self.min_keep:
            keep = torch.zeros_like(keep)
            keep[torch.argsort(cosine, descending=True, stable=True)[: self.min_keep]] = True
        return grasps[keep]


@POSTPROCESSORS.register()
class TablePlaneFilter(PostProcessor):
    """Remove grasps whose gripper would reach into the table.

    The gripper is the box model of the collision filter: fingers of ``finger_length``
    that end ``depth`` past the grasp point, as wide as the grasp plus two finger
    widths. Its eight corners must stay ``clearance`` above the table plane. The plane
    is fitted to the cloud (RANSAC, the largest plane) or given. ZeroGrasp's detector
    tests the gripper's box vertices against ``z = 0`` of a calibrated table frame in
    the same way; here no calibration is needed.

    A cloud-based collision filter sees the table only where the table is in the cloud;
    a workspace crop or a shadow removes it. This filter extends it as a plane.

    Args:
        clearance: Required height of every gripper corner above the plane, metres.
        plane: The table as ``[a, b, c, d]`` with ``a·x + b·y + c·z + d`` the height of
            a point in metres (unit normal towards the camera), in the frame of the
            grasps. ``None`` fits it to the cloud of each frame.
        cloud: The cloud the plane is fitted to; ``points`` is used when ``full_points``
            is absent.
        plane_threshold: Inlier distance of the fit, metres.
        min_inliers: Refuse a fitted plane that holds a smaller share of the cloud.
        finger_length: Length of the gripper box behind the fingertips, metres.
    """

    status = "tested"
    source = (
        "Generic; the table test of the ZeroGrasp collision detector (Iwase et al., "
        "CVPR 2025) with a plane fitted to the cloud instead of a calibrated table frame"
    )
    primary = "clearance"

    def __init__(
        self,
        clearance: float = 0.0,
        plane: Sequence[float] | None = None,
        cloud: Literal["full_points", "points"] = "full_points",
        plane_threshold: float = 0.01,
        min_inliers: float = 0.1,
        finger_length: float = 0.06,
    ) -> None:
        if plane is not None and len(plane) != 4:
            raise ValueError(f"plane must be [a, b, c, d], got {plane!r}")
        if finger_length <= 0 or plane_threshold <= 0:
            raise ValueError("finger_length and plane_threshold must be positive")
        self.clearance = clearance
        self.plane = None if plane is None else list(plane)
        self.cloud = cloud
        self.plane_threshold = plane_threshold
        self.min_inliers = min_inliers
        self.finger_length = finger_length

    def heights(self, grasps: GraspGroup, plane: Tensor) -> Tensor:
        """``(G,)`` height of the lowest gripper corner above the plane, metres."""
        finger_width = grasps.gripper.finger_width
        ends = torch.stack([grasps.depth - self.finger_length - finger_width, grasps.depth], dim=1)
        half_y = grasps.width / 2 + finger_width
        half_z = grasps.height / 2
        corners = []
        for end in (0, 1):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(torch.stack([ends[:, end], sy * half_y, sz * half_z], dim=1))
        local = torch.stack(corners, dim=1)  # (G, 8, 3) in the gripper frame
        world = local @ grasps.rotation.transpose(1, 2) + grasps.translation[:, None]
        plane = plane.to(world)
        return (world @ plane[:3] + plane[3]).amin(dim=1)

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        plane = _table_plane(
            "TablePlaneFilter",
            self.plane,
            grasps,
            observation,
            self.cloud,
            self.plane_threshold,
            self.min_inliers,
        )
        return grasps[self.heights(grasps, plane) >= self.clearance]


@POSTPROCESSORS.register()
class MaskFilter(PostProcessor):
    """Keep the grasps on a target: those near a point of a named mask of the observation.

    ``observation.masks[<mask>]`` is a boolean mask over ``observation.points`` (an
    instance or a language-grounded segmentation projected onto the cloud). A grasp is
    kept when a masked point lies within ``max_distance`` of its grasp point. Seed-based
    detectors put the grasp point on a point of the cloud, so a small distance selects
    exactly the grasps seeded on the target; HGGD computes its centres, so the distance
    is a tolerance. Contact-GraspNet (``filter_grasps``) and the AnyGrasp SDK
    (``apply_object_mask``) select grasps this way.

    A mask made from ground truth makes the variant ground-truth assisted, and the
    result record says so (``Mask.provenance``).

    Args:
        mask: Name of the mask in ``observation.masks``.
        max_distance: Largest distance between a grasp point and a masked point, metres.
        invert: Keep the grasps that are *not* on the mask (avoid an object).
    """

    status = "tested"
    source = (
        "Generic; `filter_grasps` of Contact-GraspNet (Sundermeyer et al., ICRA 2021) "
        "and `apply_object_mask` of the AnyGrasp SDK"
    )
    primary = "mask"

    def __init__(
        self, mask: str = "target", max_distance: float = 0.01, invert: bool = False
    ) -> None:
        if max_distance < 0:
            raise ValueError(f"max_distance must not be negative, got {max_distance}")
        self.mask = mask
        self.max_distance = max_distance
        self.invert = invert

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if self.mask not in observation.masks:
            present = ", ".join(sorted(observation.masks)) or "none"
            raise ContractError(
                f"MaskFilter reads observation.masks['{self.mask}'], which this observation "
                f"does not provide (masks present: {present}). A mask is a boolean tensor "
                "over observation.points; put it there before post-processing: "
                "`observation.masks['" + self.mask + "'] = Mask(data, Provenance.PREDICTED)`."
            )
        if len(grasps) == 0:
            return grasps
        points = observation.points
        if points is None:
            raise ContractError("MaskFilter reads observation.points, which is absent")
        if points.frame != grasps.frame:
            raise ContractError(
                f"MaskFilter: grasps are in frame '{grasps.frame}' but the scene cloud is in "
                f"frame '{points.frame}'"
            )
        data = observation.masks[self.mask].data
        if data.shape != (len(points),):
            raise ContractError(
                f"MaskFilter: mask '{self.mask}' has shape {tuple(data.shape)}; it must be "
                f"({len(points)},), one value per point of observation.points"
            )
        frame = frame_index("MaskFilter", grasps, observation)
        coords, chosen = points.coords, data.bool()
        if points.batch_offsets is not None:
            start, stop = points.batch_offsets[frame], points.batch_offsets[frame + 1]
            coords, chosen = coords[start:stop], chosen[start:stop]
        target = coords[chosen].to(grasps.translation)
        if len(target) == 0:
            near = torch.zeros(len(grasps), dtype=torch.bool, device=grasps.translation.device)
        else:
            near = torch.cat(
                [
                    torch.cdist(grasps.translation[start : start + CHUNK], target).amin(dim=1)
                    <= self.max_distance
                    for start in range(0, len(grasps), CHUNK)
                ]
            )
        return grasps[~near if self.invert else near]
