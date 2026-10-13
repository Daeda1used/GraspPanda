"""Model-free collision filter: drop grasps whose gripper would hit observed points.

This is the *model-side* filter of the GraspNet family. It needs no object models: it
voxel-down-samples the sensor cloud and counts the points inside a simple box model of
the gripper (two fingers, a palm, and the volume the gripper sweeps while approaching).
A grasp is rejected when the count, relative to the volume of those boxes measured in
voxels, exceeds a threshold.

It is unrelated to the collision *check* of the benchmark evaluator, which uses
ground-truth meshes and is part of scoring. Applying this filter changes reported
numbers, so results are always labelled with whether it ran.

The filter reads ``observation.full_points``, the cloud before the network's point
budget was applied. Its voxel size is its own parameter and has nothing to do with the
voxel size of a sparse backbone (``docs/design/ARCHITECTURE.md`` §8.6).

Two reference test scripts ship a filter of this kind, and they differ in more than
their constants. The defaults are the filter of the Graspness test script. HGGD's (its
detector in mode ``graspnet``) is the same four boxes with these options::

    cloud: points             # the 25 600 points the network saw, not the full cloud
    finger_length: 0.04
    height: 0.01              # one gripper for every grasp, whatever the grasp says
    depth: 0.02
    collision_threshold: 0.0  # a single point in a box is a collision
    min_points_between: 2     # and at most two points between the fingers is no grasp
    precision: float16        # the reference rounds cloud, grasps and limits to half floats
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from grasppanda.models.base import PostProcessor
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation

from .scene import scene_points

__all__ = ["ModelFreeCollisionFilter", "voxel_down_sample"]

#: Grasps tested per block; bounds the (grasps × scene points) work tensor.
_CHUNK = 128


def voxel_down_sample(
    points: Tensor, voxel_size: float, *, arithmetic: Literal["open3d", "cupoch"] = "open3d"
) -> Tensor:
    """Centroid of the points in each occupied voxel.

    The grid starts half a voxel below the minimum corner of the cloud; the order of
    the returned points is not specified. Points on a voxel face are common in clouds
    from a depth camera, so *how* the cell index is computed is part of the result:

    ``open3d``
        Double precision, ``floor((p - origin) / voxel_size)``; float64 centroids. The
        arithmetic of Open3D's ``voxel_down_sample``.
    ``cupoch``
        Single precision, ``floor((p - origin) * (1 / voxel_size))`` with the
        reciprocal rounded to float32; centroids summed in float32 in point order and
        returned as float32. The arithmetic of cupoch's GPU ``voxel_down_sample``.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {tuple(points.shape)}")
    if arithmetic not in ("open3d", "cupoch"):
        raise ValueError(f"arithmetic must be 'open3d' or 'cupoch', got {arithmetic!r}")
    if arithmetic == "cupoch":
        # On the CPU: ``index_add_`` adds in index order there, so the rounding of the
        # float32 sums is defined, and no fused multiply-add changes a cell index.
        single = points.to("cpu", torch.float32)
        if len(single) == 0:
            return single.to(points.device)
        size = single.new_tensor(voxel_size)
        origin = single.amin(dim=0) - size * 0.5
        cells = torch.floor((single - origin) * (1.0 / size)).long()
        _, inverse = torch.unique(cells, dim=0, return_inverse=True)
        count = torch.bincount(inverse)
        total = single.new_zeros(len(count), 3).index_add_(0, inverse, single)
        return (total / count[:, None].to(torch.float32)).to(points.device)
    cloud = points.double()
    if len(cloud) == 0:
        return cloud
    origin = cloud.amin(dim=0) - 0.5 * voxel_size
    # A true division (by a tensor): CUDA would multiply by the reciprocal of a Python
    # number, and depth-camera points sit exactly on voxel faces often enough for that
    # last bit to move them.
    cells = torch.floor((cloud - origin) / cloud.new_tensor(voxel_size)).long()
    _, inverse = torch.unique(cells, dim=0, return_inverse=True)
    count = torch.bincount(inverse)
    total = cloud.new_zeros(len(count), 3).index_add_(0, inverse, cloud)
    return total / count[:, None]


@POSTPROCESSORS.register()
class ModelFreeCollisionFilter(PostProcessor):
    """Remove grasps that collide with the observed cloud.

    In the gripper frame of a grasp (x: approach, y: closing, z: finger height), with
    ``d`` its depth, ``w`` its width and ``h`` its height, a scene point counts when
    ``|z| < h / 2`` and it lies in

    * a finger: ``d - finger_length < x < d`` and ``w/2 < |y| < w/2 + finger_width``;
    * the palm: ``d - finger_length - finger_width < x ≤ d - finger_length`` and
      ``|y| < w/2 + finger_width``;
    * the approach sweep: the next ``approach_distance`` behind the palm, same ``y``.

    The grasp collides when ``count / volume > collision_threshold``, with the volume
    of the four boxes expressed in voxels of ``voxel_size``. The finger width comes
    from the gripper of the grasps.

    Args:
        voxel_size: Edge length used to down-sample the scene cloud and to express the
            gripper volume, metres. Reference value of the Graspness test script: 0.01.
        approach_distance: Length of the swept volume behind the palm, metres; at least
            one finger width is always used.
        collision_threshold: Occupancy ratio above which a grasp is rejected. ``0``
            rejects a grasp for a single point inside a box.
        finger_length: Length of the finger boxes along the approach axis, metres.
        cloud: Which cloud of the observation is tested: ``full_points`` (before
            sub-sampling) or ``points`` (what the network saw).
        height: Finger thickness ``h`` used for every grasp, metres; ``None`` uses each
            grasp's own ``height``.
        depth: Insertion depth ``d`` used for every grasp, metres; ``None`` uses each
            grasp's own ``depth``.
        min_points_between: Also reject a grasp with at most this many scene points
            between its fingers (``|z| < h / 2``, ``d - finger_length < x < d``,
            ``|y| <= w / 2``): there is nothing to grasp. ``None`` disables the rule.
        precision: Arithmetic of the test. ``float64`` forms the box limits in the
            precision of the grasps and compares them with float64 points, on a cloud
            down-sampled with Open3D's arithmetic. ``float16`` is HGGD's filter: the
            cloud is down-sampled with cupoch's single-precision arithmetic (see
            :func:`voxel_down_sample`), then cloud, grasps and every box limit are
            rounded to half floats; the gripper-frame coordinates are the exact
            products rounded to half floats.
    """

    reads_full_points = True

    status = "tested"
    source = "Collision filter of the GSNet and HGGD test scripts"

    def __init__(
        self,
        voxel_size: float = 0.01,
        approach_distance: float = 0.05,
        collision_threshold: float = 0.01,
        finger_length: float = 0.06,
        cloud: Literal["full_points", "points"] = "full_points",
        height: float | None = None,
        depth: float | None = None,
        min_points_between: int | None = None,
        precision: Literal["float64", "float16"] = "float64",
    ) -> None:
        if voxel_size <= 0 or finger_length <= 0 or approach_distance < 0:
            raise ValueError(
                "voxel_size and finger_length must be positive and approach_distance "
                f"non-negative; got {voxel_size}, {finger_length}, {approach_distance}"
            )
        if cloud not in ("full_points", "points"):
            raise ValueError(f"cloud must be 'full_points' or 'points', got {cloud!r}")
        if precision not in ("float64", "float16"):
            raise ValueError(f"precision must be 'float64' or 'float16', got {precision!r}")
        if (height is not None and height <= 0) or (depth is not None and depth <= 0):
            raise ValueError(f"height and depth must be positive, got {height} and {depth}")
        if min_points_between is not None and min_points_between < 0:
            raise ValueError(f"min_points_between must not be negative, got {min_points_between}")
        self.voxel_size = voxel_size
        self.approach_distance = approach_distance
        self.collision_threshold = collision_threshold
        self.finger_length = finger_length
        self.cloud = cloud
        self.height = height
        self.depth = depth
        self.min_points_between = min_points_between
        self.precision = precision

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        scene = self.down_sample(self._scene_points(grasps, observation))
        return grasps[~self.collision_mask(grasps, scene)]

    def down_sample(self, points: Tensor) -> Tensor:
        """The voxel centroids the grasps are tested against."""
        arithmetic = "cupoch" if self.precision == "float16" else "open3d"
        return voxel_down_sample(points, self.voxel_size, arithmetic=arithmetic)

    def _scene_points(self, grasps: GraspGroup, observation: Observation) -> Tensor:
        return scene_points(type(self).__name__, grasps, observation, self.cloud)

    @torch.no_grad()
    def collision_mask(self, grasps: GraspGroup, scene: Tensor) -> Tensor:
        """``(G,)`` true where a grasp is rejected against the down-sampled ``scene`` points."""
        finger_width = grasps.gripper.finger_width
        approach = max(self.approach_distance, finger_width)
        half = self.precision == "float16"
        # float64: box limits are formed in the precision of the grasps and only then
        # compared with the float64 scene. float16: limits and coordinates are half
        # floats. Each reference detector does it its own way.
        scene = scene.to(torch.float16) if half else scene.double()
        cell = grasps.height.new_tensor(self.voxel_size**3)  # a tensor: true division on CUDA
        masks = []
        for start in range(0, len(grasps), _CHUNK):
            rows = slice(start, start + _CHUNK)
            height = grasps.height[rows, None]
            depth = grasps.depth[rows, None]
            width = grasps.width[rows, None]
            if self.height is not None:
                height = torch.full_like(height, self.height)
            if self.depth is not None:
                depth = torch.full_like(depth, self.depth)
            if half:
                height, depth, width = (v.to(torch.float16) for v in (height, depth, width))
                offset = scene[None] - grasps.translation[rows, None].to(torch.float16)
                rotation = grasps.rotation[rows].to(torch.float16)
                local = (offset.double() @ rotation.double()).to(torch.float16)
            else:
                offset = scene[None] - grasps.translation[rows, None].double()
                local = offset @ grasps.rotation[rows].double()
            x, y, z = local.unbind(dim=-1)
            finger_end = depth - self.finger_length
            palm_end = finger_end - finger_width
            outer = width / 2 + finger_width
            in_height = (z > -height / 2) & (z < height / 2)
            beside_fingers = (x > finger_end) & (x < depth)
            within_left, within_right = y > -outer, y < outer
            left = in_height & beside_fingers & within_left & (y < -width / 2)
            right = in_height & beside_fingers & within_right & (y > width / 2)
            across = in_height & within_left & within_right
            palm = across & (x <= finger_end) & (x > palm_end)
            sweep = across & (x <= palm_end) & (x > palm_end - approach)
            occupied = (left | right | palm | sweep).sum(dim=1)
            between = in_height & beside_fingers & (y >= -width / 2) & (y <= width / 2)

            height, width = height[:, 0].to(cell.dtype), width[:, 0].to(cell.dtype)
            finger_volume = height * self.finger_length * finger_width / cell
            palm_volume = height * (width + 2 * finger_width) * finger_width / cell
            sweep_volume = height * (width + 2 * finger_width) * approach / cell
            volume = finger_volume * 2 + palm_volume + sweep_volume
            ratio = occupied.double() / (volume + 1e-6).double()
            rejected = ratio > self.collision_threshold
            if self.min_points_between is not None:
                rejected |= between.sum(dim=1) <= self.min_points_between
            masks.append(rejected)
        return torch.cat(masks)
