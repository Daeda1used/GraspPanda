"""Fitting predictions to a real gripper and to a robot: opening limits, centring and
width from the local cloud, the robot's frame.

``GripperLimits`` and ``ToFrame`` read only the grasps. ``GripperCentering`` and
``WidthFromCloud`` read the sensor cloud between the fingers and change a pose or a
width; they never remove a grasp and never change a score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from grasppanda.models.base import PostProcessor
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation

from .collision import voxel_down_sample
from .scene import CHUNK, as_transform, gripper_frame_points, scene_points

__all__ = ["GripperCentering", "GripperLimits", "ToFrame", "WidthFromCloud"]


@POSTPROCESSORS.register()
class GripperLimits(PostProcessor):
    """Keep the grasps a real gripper can execute: opening and insertion depth in range.

    Detectors trained on GraspNet-1Billion predict openings up to 0.10 m (0.14 m on
    GraspClutter6D); a Franka hand opens 0.08 m, a Robotiq 2F-85 0.085 m. ``remove``
    drops grasps outside the range, as ``max_gripper_width`` of the AnyGrasp SDK and
    the aperture limits of GPD do. ``clip`` keeps them with the opening set to the
    limit, which is only right for a consumer that clips itself (the benchmark
    evaluators do): a gripper that cannot open as far as the object is wide cannot
    grasp it.

    Args:
        max_width: Largest opening, metres; ``None`` uses the gripper of the grasps.
        min_width: Smallest opening, metres (a gripper that cannot close fully).
        max_depth: Largest insertion depth, metres; ``None`` does not test it.
        mode: ``remove`` or ``clip``.
    """

    status = "tested"
    source = (
        "Generic; `max_gripper_width` of the AnyGrasp SDK, `min_aperture`/`max_aperture` "
        "of GPD (arXiv 1706.09911)"
    )
    primary = "max_width"

    def __init__(
        self,
        max_width: float | None = 0.08,
        min_width: float = 0.0,
        max_depth: float | None = None,
        mode: Literal["remove", "clip"] = "remove",
    ) -> None:
        if mode not in ("remove", "clip"):
            raise ValueError(f"mode must be 'remove' or 'clip', got {mode!r}")
        if min_width < 0 or (max_width is not None and max_width < min_width):
            raise ValueError(
                f"widths are metres with 0 <= min_width <= max_width; got {min_width}, {max_width}"
            )
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must not be negative, got {max_depth}")
        self.max_width = max_width
        self.min_width = min_width
        self.max_depth = max_depth
        self.mode = mode

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        widest = grasps.gripper.max_width if self.max_width is None else self.max_width
        if self.mode == "clip":
            depth = grasps.depth
            if self.max_depth is not None:
                depth = depth.clamp(max=self.max_depth)
            return grasps._with(width=grasps.width.clamp(self.min_width, widest), depth=depth)
        keep = (grasps.width >= self.min_width) & (grasps.width <= widest)
        if self.max_depth is not None:
            keep &= grasps.depth <= self.max_depth
        return grasps[keep]


def _between_fingers(
    grasps: GraspGroup, scene: Tensor, finger_length: float, half_width: Tensor
) -> tuple[Tensor, Tensor]:
    """``(y, inside)``, both ``(G, N)``: closing-axis coordinate of every scene point and
    whether it lies in the slab the fingers close through, up to ``half_width`` wide."""
    local = gripper_frame_points(grasps, scene)
    x, y, z = local.unbind(dim=-1)
    depth, height = grasps.depth[:, None], grasps.height[:, None]
    inside = (z.abs() < height / 2) & (x > depth - finger_length) & (x < depth)
    return y, inside & (y.abs() <= half_width[:, None])


@POSTPROCESSORS.register()
class GripperCentering(PostProcessor):
    """Shift each grasp along its closing axis so that both fingers travel equally far.

    The contact points are the outermost scene points between the fingers; the grasp
    point moves to the middle between them. A gripper that closes off-centre pushes the
    object with the finger that arrives first. This is the centring step of AnyGrasp
    (section IV-D: contact points are "the outermost points inside the gripper space").
    Poses change, so run it before a collision filter; nothing is removed and no score
    changes. Contacts hidden from the camera are not seen.

    Args:
        cloud: ``full_points`` (before sub-sampling; ``points`` when absent) or
            ``points``.
        finger_length: Length of the slab along the approach axis, metres.
        min_points: A grasp with fewer points between its fingers is left as it is.
        max_shift: Largest shift, metres; ``None`` allows up to half the width.
        voxel_size: The cloud is down-sampled to this edge length first, metres (a full
            cloud of half a million points against a thousand grasps is slow and adds
            nothing); ``0`` uses every point. ``min_points`` counts the points left.
    """

    status = "tested"
    source = (
        "Generic, written here from the description in AnyGrasp (Fang et al., "
        "arXiv 2212.08333, section IV-D)"
    )
    primary = "max_shift"

    def __init__(
        self,
        cloud: Literal["full_points", "points"] = "full_points",
        finger_length: float = 0.06,
        min_points: int = 5,
        max_shift: float | None = None,
        voxel_size: float = 0.005,
    ) -> None:
        if finger_length <= 0 or min_points < 1 or voxel_size < 0:
            raise ValueError(
                "finger_length must be positive, min_points at least 1, voxel_size not negative"
            )
        self.voxel_size = voxel_size
        self.cloud = cloud
        self.finger_length = finger_length
        self.min_points = min_points
        self.max_shift = max_shift

    @torch.no_grad()
    def shifts(self, grasps: GraspGroup, scene: Tensor) -> Tensor:
        """``(G,)`` shift along the closing axis, metres; zero where nothing is moved."""
        out = []
        for start in range(0, len(grasps), CHUNK):
            part = grasps[slice(start, start + CHUNK)]
            y, inside = _between_fingers(part, scene, self.finger_length, part.width / 2)
            low = torch.where(inside, y, torch.full_like(y, float("inf"))).amin(dim=1)
            high = torch.where(inside, y, torch.full_like(y, float("-inf"))).amax(dim=1)
            shift = (low + high) / 2
            shift = torch.where(inside.sum(dim=1) >= self.min_points, shift, 0.0)
            out.append(shift)
        shift = torch.cat(out)
        if self.max_shift is not None:
            shift = shift.clamp(-self.max_shift, self.max_shift)
        return shift.to(grasps.translation.dtype)

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        scene = scene_points("GripperCentering", grasps, observation, self.cloud, fallback=True)
        if self.voxel_size > 0:
            scene = voxel_down_sample(scene, self.voxel_size)
        shift = self.shifts(grasps, scene)
        moved = grasps.translation + shift[:, None] * grasps.rotation[:, :, 1]
        return grasps._with(translation=moved)


@POSTPROCESSORS.register()
class WidthFromCloud(PostProcessor):
    """Set each opening to the extent of the scene between the fingers plus a clearance.

    For every grasp the points between its fingers are searched up to ``margin`` beyond
    the predicted opening; the new opening is twice the largest distance of such a
    point from the grasp's centre plane, plus ``2 * clearance``, limited to the
    gripper. A predicted width is a regression (several heads also multiply it by a
    gain of 1.2); the cloud says how wide the thing between the fingers is in this
    frame. ZeroGrasp refines widths from contact distances with the same clearance
    range; this is the single-view form, without its reconstruction.

    Poses and scores do not change; run it before a collision filter.

    Args:
        clearance: Free space left on each side, metres.
        margin: How far beyond the predicted half-opening points are searched, metres.
        mode: ``fit`` sets the measured width; ``shrink`` only ever narrows.
        cloud: ``full_points`` (``points`` when absent) or ``points``.
        finger_length: Length of the slab along the approach axis, metres.
        min_points: A grasp with fewer points between its fingers keeps its width.
        min_width: Smallest opening that is written, metres.
        voxel_size: The cloud is down-sampled to this edge length first, metres; ``0``
            uses every point. ``min_points`` counts the points left, and half a voxel is
            added to the measured extent on each side, so the result errs wide.
    """

    status = "tested"
    source = (
        "Generic; the idea of the contact-based width refinement of ZeroGrasp (Iwase et "
        "al., arXiv 2504.10857), on the sensor cloud alone"
    )
    primary = "clearance"

    def __init__(
        self,
        clearance: float = 0.01,
        margin: float = 0.01,
        mode: Literal["fit", "shrink"] = "fit",
        cloud: Literal["full_points", "points"] = "full_points",
        finger_length: float = 0.06,
        min_points: int = 5,
        min_width: float = 0.01,
        voxel_size: float = 0.005,
    ) -> None:
        if voxel_size < 0:
            raise ValueError(f"voxel_size must not be negative, got {voxel_size}")
        self.voxel_size = voxel_size
        if mode not in ("fit", "shrink"):
            raise ValueError(f"mode must be 'fit' or 'shrink', got {mode!r}")
        if min(clearance, margin, min_width) < 0 or finger_length <= 0 or min_points < 1:
            raise ValueError(
                "clearance, margin and min_width must not be negative, finger_length must be "
                "positive and min_points at least 1"
            )
        self.clearance = clearance
        self.margin = margin
        self.mode = mode
        self.cloud = cloud
        self.finger_length = finger_length
        self.min_points = min_points
        self.min_width = min_width

    @torch.no_grad()
    def widths(self, grasps: GraspGroup, scene: Tensor) -> Tensor:
        """``(G,)`` refined openings, metres."""
        out = []
        for start in range(0, len(grasps), CHUNK):
            part = grasps[slice(start, start + CHUNK)]
            reach = part.width / 2 + self.margin
            y, inside = _between_fingers(part, scene, self.finger_length, reach)
            extent = torch.where(inside, y.abs(), torch.zeros_like(y)).amax(dim=1)
            # A voxel centroid lies up to half a voxel inside the true extent.
            extent = extent + self.voxel_size / 2
            measured = (2 * extent + 2 * self.clearance).to(part.width.dtype)
            measured = measured.clamp(self.min_width, part.gripper.max_width)
            if self.mode == "shrink":
                measured = torch.minimum(measured, part.width)
            out.append(torch.where(inside.sum(dim=1) >= self.min_points, measured, part.width))
        return torch.cat(out)

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        scene = scene_points("WidthFromCloud", grasps, observation, self.cloud, fallback=True)
        if self.voxel_size > 0:
            scene = voxel_down_sample(scene, self.voxel_size)
        return grasps._with(width=self.widths(grasps, scene))


@POSTPROCESSORS.register()
class ToFrame(PostProcessor):
    """Express the grasps in another frame, for example the robot base.

    ``transform`` is the pose of the current frame of the grasps in the target frame
    (for a camera: the hand-eye calibration ``T_base_camera``), so a point ``p`` becomes
    ``R p + t``. The result carries the new frame name. Put this step **last**: every
    step that reads the cloud refuses grasps in another frame than the cloud's.

    With ``origin: fingertips`` the grasp point is moved to the middle between the
    fingertips (``translation + depth * approach``) and the depth is set to zero, which
    is the tool centre point most robot drivers expect; the axes stay x approach,
    y closing, z finger height.

    Args:
        transform: ``(4, 4)`` rigid transform as nested lists, or the path of a
            ``.npy`` or whitespace-separated text file.
        frame: Name of the target frame.
        origin: ``grasp_point`` (unchanged convention) or ``fingertips``.
    """

    status = "tested"
    source = "Generic; GraspGroup.transform with a calibration from the config"
    primary = "transform"

    def __init__(
        self,
        transform: Sequence[Sequence[float]] | str,
        frame: str = "base",
        origin: Literal["grasp_point", "fingertips"] = "grasp_point",
    ) -> None:
        if origin not in ("grasp_point", "fingertips"):
            raise ValueError(f"origin must be 'grasp_point' or 'fingertips', got {origin!r}")
        if not frame:
            raise ValueError("frame must be a non-empty name")
        self.transform = as_transform(transform, "ToFrame")
        self.frame = frame
        self.origin = origin

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if self.origin == "fingertips" and len(grasps):
            tips = grasps.translation + grasps.depth[:, None] * grasps.rotation[:, :, 0]
            grasps = grasps._with(translation=tips, depth=torch.zeros_like(grasps.depth))
        return grasps.transform(self.transform, frame=self.frame)
