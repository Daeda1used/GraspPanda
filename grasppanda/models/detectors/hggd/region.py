"""The region stage of HGGD: lift partial grasps to 3-D and cut a point region around each.

1. **Features on points.** The feature map of the image network lives on a coarse pixel
   grid. Each cell is back-projected with the sensor depth at its corner pixel; every
   scene point then takes, channel by channel, the maximum over its ``k`` nearest cells
   *in 3-D*.
2. **Lifting.** A partial grasp's centre is its pixel back-projected at the sensor depth
   plus the predicted depth offset. A pixel without depth borrows the median of the
   valid depths around it.
3. **Regions.** All scene points closer to the centre than the grasp's metric width form
   its region: an unrotated ball, since nothing is known yet about the out-of-plane
   rotation. Regions with too few points are dropped; small ones are re-sampled with
   replacement; all are jittered and reduced to a fixed size by furthest point sampling
   from a random start; coordinates are expressed relative to the centre.

Reference behaviours kept on purpose, each a named option:

* ``zero_depth_centres: append`` — the reference appends the centres of pixels without
  depth to the *end* of the centre list while the grasp attributes keep their order, so
  after such a pixel centres and attributes are paired off by one. ``in_place`` keeps
  every centre with its own grasp.
* the jitter and the random start are applied at inference too, so two runs of the
  reference on one frame differ; here the draws can be supplied (:class:`RegionDraws`).
"""

from __future__ import annotations

import math
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.ops import furthest_point_sample, knn
from grasppanda.structures import ContractError

from .options import RegionOptions, options_from
from .structures import PartialGraspSet, RegionDraws, RegionSet

__all__ = ["RegionStage", "attach_features", "cell_points"]


def cell_points(depth: Tensor, intrinsics: Tensor, shape: tuple[int, int]) -> Tensor:
    """``(U', V', 3)`` camera-frame points of the cells of a coarse ``(u, v)`` grid.

    Cell ``(i, j)`` is represented by the pixel ``(i W / U', j H / V')`` of the
    ``(H, W)`` depth image (nearest-neighbour reduction); a pixel without depth lands on
    the camera centre.
    """
    cells_u, cells_v = shape
    width = depth.shape[1]
    reduced = F.interpolate(depth[None, None].float(), size=(cells_v, cells_u), mode="nearest")
    z = reduced[0, 0].T  # (U', V'), metres
    factor = width / cells_u
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    ray_u = (torch.arange(cells_u, dtype=torch.float64) - cx / factor) / (fx / factor)
    ray_v = (torch.arange(cells_v, dtype=torch.float64) - cy / factor) / (fy / factor)
    x = ray_u.to(z)[:, None] * z
    y = ray_v.to(z)[None, :] * z
    return torch.stack([x, y, z], dim=-1)


def attach_features(points: Tensor, cells: Tensor, features: Tensor, neighbours: int) -> Tensor:
    """``(N, C)`` features of scene points: maximum over the nearest feature-map cells.

    Args:
        points: ``(N, 3)`` scene points.
        cells: ``(U', V', 3)`` points of the feature-map cells, see :func:`cell_points`.
        features: ``(C, U', V')`` feature map.
        neighbours: Cells pooled per point.
    """
    flat = cells.reshape(-1, 3)
    values = features.reshape(features.shape[0], -1).T  # (U' V', C), row-major like ``flat``
    _, index = knn(points[None], flat[None], neighbours)
    return values[index[0]].amax(dim=1)


class RegionStage(nn.Module):
    """Lifting, feature attachment and region grouping. Holds no parameters.

    Every option of :class:`~.options.RegionOptions` is an attribute of the stage.
    """

    def __init__(self, options: RegionOptions | None = None) -> None:
        super().__init__()
        self.options = options_from(RegionOptions, options)
        for field in fields(self.options):
            setattr(self, field.name, getattr(self.options, field.name))
        if self.num_points < 1 or self.min_points < 0 or self.feature_neighbours < 1:
            raise ValueError("num_points and feature_neighbours must be positive")
        if self.zero_depth_centres not in ("append", "in_place"):
            raise ValueError(f"unknown zero_depth_centres '{self.zero_depth_centres}'")

    # ---------------------------------------------------------------------- lifting

    def lift(
        self,
        grasps: PartialGraspSet,
        depth: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """``(P, 3)`` float64 centres of the partial grasps of ONE frame, camera frame.

        Args:
            grasps: Proposals of the frame; ``pixel`` is in the network image.
            depth: ``(H, W)`` sensor depth of the frame, metres.
            intrinsics: ``(3, 3)`` camera matrix of the sensor image.
            image_size: ``(U, V)`` of the network image.
        """
        height, width = depth.shape
        millimetres = depth.float() * 1000.0
        scale_u, scale_v = width / image_size[0], height / image_size[1]
        u = (grasps.pixel[:, 0] * scale_u).long()
        v = (grasps.pixel[:, 1] * scale_v).long()
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        observed = millimetres[v, u]
        offset = grasps.depth_offset.double() * 1000.0
        valid = observed != 0
        # The pixel offset from the principal point is formed in single precision.
        z = (observed.double() + offset) / 1000.0
        x = z / fx * (u.float() - cx)
        y = z / fy * (v.float() - cy)
        centre = torch.stack([x, y, z], dim=1)
        fixed = []
        half = self.zero_depth_window
        for row in torch.nonzero(~valid)[:, 0].tolist():
            pu, pv = int(u[row]), int(v[row])
            window = millimetres[
                max(0, pv - half) : min(height - 1, pv + half),
                max(0, pu - half) : min(width - 1, pu + half),
            ]
            present = window[window > 0]
            if len(present) == 0:
                # No depth anywhere near: the mean of the whole image, in single
                # precision and without the predicted offset (the reference's fallback).
                z0 = millimetres.mean() / 1000.0
                fixed.append(torch.stack([z0 / fx * (pu - cx), z0 / fy * (pv - cy), z0]).double())
            else:
                z0 = (present.median().double() + offset[row]) / 1000.0
                fixed.append(torch.stack([z0 / fx * (pu - cx), z0 / fy * (pv - cy), z0]))
        if not fixed:
            return centre
        if self.zero_depth_centres == "append":
            return torch.cat([centre[valid], torch.stack(fixed)])
        centre = centre.clone()
        centre[~valid] = torch.stack(fixed)
        return centre

    def radius(
        self,
        grasps: PartialGraspSet,
        centre: Tensor,
        intrinsics: Tensor,
        scale: float,
    ) -> Tensor:
        """``(P,)`` metric jaw opening of each partial grasp at its centre depth, float32.

        The opening is ``width`` pixels of the network image (``scale`` sensor pixels
        each) along the in-plane direction ``theta``, seen at the centre's depth.
        """
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        z = centre[:, 2].double() * 1000.0 * scale / 1000.0
        # The opening is measured from the principal point truncated to a whole pixel
        # (the reference keeps it in an integer array); only the rounding depends on it.
        base_u, base_v = float(int(cx)), float(int(cy))
        open_u = base_u - grasps.width.double() / 2 * torch.cos(grasps.theta.double())
        open_v = base_v + grasps.width.double() / 2 * torch.sin(grasps.theta.double())
        dx = z / fx * (open_u - cx) - z / fx * (base_u - cx)
        dy = z / fy * (open_v - cy) - z / fy * (base_v - cy)
        return (torch.sqrt(dx * dx + dy * dy) * 2).to(torch.float32)

    # ---------------------------------------------------------------------- regions

    def draw(self, support: Tensor, longest: int, device: torch.device) -> RegionDraws:
        """Draw the random choices for regions with ``support`` points each."""
        size = self.num_points
        resample = [
            torch.randint(int(count), (size,), device=device)
            for count in support.tolist()
            if self.min_points < count < size
        ]
        jitter = torch.randn(len(support), longest, 3, device=device)
        lengths = torch.where(support < size, torch.full_like(support, size), support)
        start = torch.stack([torch.randint(int(length), (1,))[0] for length in lengths.tolist()])
        return RegionDraws(resample=resample, jitter=jitter, start=start.to(device))

    def group(
        self,
        scene: Tensor,
        centre: Tensor,
        radius: Tensor,
        draws: RegionDraws | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cut, re-sample and sample the regions of ONE frame.

        Args:
            scene: ``(N, 3 + C)`` scene points with their attached features.
            centre: ``(P, 3)`` float64 region centres.
            radius: ``(P,)`` ball radii.
            draws: Random choices to replay; drawn when omitted.

        Returns:
            ``(regions, kept, support)``: ``(R, S, 3 + C)`` regions of the centres that
            have enough support, coordinates relative to their centre; the ``(P,)`` mask
            of those centres; and the ``(P,)`` number of scene points in each ball.
        """
        size = self.num_points
        count, channels = len(centre), scene.shape[1]
        if count == 0:
            return (
                scene.new_zeros(0, size, channels),
                torch.zeros(0, dtype=torch.bool, device=scene.device),
                torch.zeros(0, dtype=torch.long, device=scene.device),
            )
        # Distances in double precision through the expanded square, as the reference
        # forms them; rounding there decides membership of points on the boundary.
        p1, p2 = centre.double(), scene[:, :3].double()
        squared = -2 * (p1 @ p2.T) + p1.square().sum(1)[:, None] + p2.square().sum(1)[None, :]
        inside = squared.sqrt() < radius.double()[:, None]
        support = inside.sum(dim=1)
        longest = max(size, int(support.max()))
        if draws is None:
            draws = self.draw(support, longest, scene.device)
        if tuple(draws.jitter.shape) != (count, longest, 3) or len(draws.start) != count:
            raise ContractError(
                f"the supplied draws are for {tuple(draws.jitter.shape[:2])} region slots, "
                f"but this frame has {count} regions of up to {longest} points"
            )
        # Members of every ball in scene order: a stable sort puts them first.
        order = torch.argsort((~inside).to(torch.uint8), dim=1, stable=True)[:, :longest]
        if order.shape[1] < longest:  # a scene with fewer points than a region holds
            order = F.pad(order, (0, longest - order.shape[1]))
        slot = torch.arange(longest, device=scene.device)[None, :]
        slots = scene[order] * (slot < support[:, None])[..., None]
        lengths = support.clamp(min=size)
        kept = (support >= size) | (support > self.min_points)
        resample = iter(draws.resample)
        for row in torch.nonzero(support < size)[:, 0].tolist():
            slots[row] = 0.0
            if not kept[row]:
                continue
            try:
                picked = next(resample).to(scene.device)
            except StopIteration:
                raise ContractError(
                    "the supplied draws hold too few re-sampling index sets for this frame"
                ) from None
            slots[row, :size] = scene[order[row, : int(support[row])][picked]]
        slots[..., :3] += draws.jitter.to(slots) * self.jitter
        index = self.sample(slots[..., :3], lengths, draws.start.to(scene.device))
        picked = torch.gather(slots, 1, index[..., None].expand(-1, -1, channels))[kept]
        relative = picked[..., :3].double() - centre[kept][:, None, :]
        regions = torch.cat([relative.to(picked.dtype), picked[..., 3:]], dim=-1)
        return regions, kept, support

    def sample(self, points: Tensor, lengths: Tensor, start: Tensor) -> Tensor:
        """Furthest point sampling of ``num_points`` among the first ``lengths`` points.

        ``(P, L, 3)`` points → ``(P, num_points)`` indices, the first being ``start``.
        """
        count, longest, _ = points.shape
        rows = torch.arange(count, device=points.device)
        slot = torch.arange(longest, device=points.device)[None, :]
        first = points[rows, start]
        # Slots past a region's length must never be chosen: put them on the start point.
        work = torch.where((slot < lengths[:, None])[..., None], points, first[:, None, :])
        # The sampler starts at index 0: exchange it with the start point.
        work[rows, start] = work[:, 0].clone()
        work[:, 0] = first
        index = furthest_point_sample(work, self.num_points)
        at_zero, at_start = index == 0, index == start[:, None]
        index = torch.where(at_zero, start[:, None].expand_as(index), index)
        return torch.where(at_start & ~at_zero, torch.zeros_like(index), index)

    # ---------------------------------------------------------------------- forward

    def forward(
        self,
        grasps: PartialGraspSet,
        points: list[Tensor],
        features: Tensor,
        depth: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
        draws: list[RegionDraws | None] | None = None,
    ) -> RegionSet:
        """Regions of a batch.

        Args:
            grasps: Partial grasps of the batch, not yet lifted.
            points: Per frame, the ``(N, 3)`` scene cloud.
            features: ``(B, C, U', V')`` feature map of the image network.
            depth: ``(B, H, W)`` sensor depth, metres.
            intrinsics: ``(B, 3, 3)`` camera matrices of the sensor images.
            image_size: ``(U, V)`` of the network image.
            draws: Per frame, the random choices to replay.
        """
        batch = features.shape[0]
        if not (len(points) == depth.shape[0] == intrinsics.shape[0] == batch):
            raise ContractError(
                f"RegionStage: {len(points)} clouds, {depth.shape[0]} depth images and "
                f"{intrinsics.shape[0]} camera matrices for a batch of {batch}"
            )
        scale = depth.shape[-1] / image_size[0]
        if not math.isclose(scale, depth.shape[-2] / image_size[1]):
            raise ContractError(
                f"the network image {image_size} does not have the aspect ratio of the "
                f"{tuple(depth.shape[-2:])} depth image"
            )
        regions, kept, support, centres = [], [], [], []
        for frame in range(batch):
            mine = grasps.select(grasps.rows(frame))
            cells = cell_points(depth[frame], intrinsics[frame], tuple(features.shape[2:]))
            attached = attach_features(
                points[frame], cells, features[frame], self.feature_neighbours
            )
            scene = torch.cat([points[frame], attached.to(points[frame].dtype)], dim=1)
            centre = self.lift(mine, depth[frame], intrinsics[frame], image_size)
            radius = self.radius(mine, centre, intrinsics[frame], scale)
            out = self.group(scene, centre, radius, None if draws is None else draws[frame])
            regions.append(out[0])
            kept.append(out[1])
            support.append(out[2])
            centres.append(centre)
        kept_all = torch.cat(kept)
        lifted = grasps.with_centre(torch.cat(centres))
        return RegionSet(
            points=torch.cat(regions),
            grasps=lifted.select(kept_all),
            kept=kept_all,
            support=torch.cat(support),
        )
