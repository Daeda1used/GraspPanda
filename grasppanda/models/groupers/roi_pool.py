# Copyright (c) OpenPCDet authors (OpenMMLab). The licence file names no holder; the pooling
# kernels are signed "Written by Shaoshuai Shi".
# Adapted for GraspPanda from https://github.com/open-mmlab/OpenPCDet @ 233f849 (Apache-2.0):
# pcdet/models/roi_heads/pvrcnn_head.py (roi_grid_pool, get_dense_grid_points),
# pcdet/models/roi_heads/partA2_head.py and pcdet/ops/roiaware_pool3d (roiaware_pool),
# pcdet/models/backbones_3d/pfe/voxel_set_abstraction.py (raw-point source and fusion).
# Changes: the region is the gripper-aligned box of a seed instead of a yaw-only detection
# box; the CUDA extensions are replaced by grasppanda.ops and scatter operations of
# PyTorch; neighbour offsets are written in the crop frame; the sparse convolutions that
# follow RoI-aware pooling in Part-A2 are replaced by a linear layer per cell.
"""Region pooling of two-stage 3-D detectors, on a gripper-aligned region per seed.

Second-stage detectors pool the features of a scene into a fixed layout per proposal box.
A seed with its approach frame defines such a box (``hmin < x < hmax`` along the approach
axis, ``|y|, |z| < radius``), so their pooling operators are groupers of this skeleton:

* :class:`RoIGridGroup`: RoI-grid pooling of PV-RCNN. A regular grid of points is laid in
  the box; every grid point gathers the support points within a small ball and reduces
  them by a shared MLP and a maximum; the grid, in a fixed order, is flattened and reduced
  by fully connected layers.
* :class:`RoIAwareGroup`: RoI-aware pooling of Part-A². The box is divided into cells;
  every cell takes the maximum of the features and the mean position of its points; empty
  cells stay zero, so the layout of the crop is kept.
* :class:`VoxelSetAbstractionGroup`: the keypoint feature of PV-RCNN. Every seed gathers
  the raw scene points within balls of several radii, and the result is fused with the
  seed's own feature by a linear layer.

The first two group among the seeds, the third over the scene cloud; all return
``(seed, channel)`` features. They were written for LiDAR detection: no grasp detector
is known to use them, and nothing here says how accurate a grasp model with them is.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.ops import ball_query, box_query, group_points
from grasppanda.registry import GROUPERS, INJECTED
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures

from .cylinder import _LinearNormAct
from .local import _shared_mlp, _tokens, masked_pool

__all__ = ["RoIAwareGroup", "RoIGridGroup", "VoxelSetAbstractionGroup"]


def _seed_inputs(
    name: str, support: PointFeatures, centres: Tensor, rotation: Tensor, in_channels: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Checked per-frame views ``(B, S, 3)``, ``(B, S, 3)`` and ``(B, S, 3, 3)``."""
    count = len(support.points)
    if support.channels != in_channels:
        raise ContractError(
            f"{name} was built for {in_channels}-channel support features, got {support.channels}"
        )
    if tuple(centres.shape) != (count, 3) or tuple(rotation.shape) != (count, 3, 3):
        raise ContractError(
            f"{name} groups among the seeds (support = 'seeds'): it needs one centre (M, 3) and "
            f"one frame (M, 3, 3) per support point with M = {count}, got "
            f"{tuple(centres.shape)} and {tuple(rotation.shape)}"
        )
    coords = support.points.dense()
    batch, per_frame, _ = coords.shape
    return coords, centres.reshape(batch, per_frame, 3), rotation.reshape(batch, per_frame, 3, 3)


def _check_box(name: str, radius: float, hmin: float, hmax: float, grid_size: int) -> None:
    if radius <= 0 or hmax <= hmin or grid_size < 1:
        raise ValueError(
            f"{name} needs radius > 0, hmin < hmax and grid_size >= 1; got radius={radius}, "
            f"hmin={hmin}, hmax={hmax}, grid_size={grid_size}"
        )


@GROUPERS.register()
class RoIGridGroup(Grouper):
    """RoI-grid pooling (PV-RCNN) on the gripper-aligned box of every seed.

    ``grid_size³`` grid points are placed at the cell centres of the box. Every grid
    point gathers up to ``nsample`` seeds within ``pool_radius``; their features, with
    the offset to the grid point in the crop frame divided by ``pool_radius``, pass a
    shared MLP and a maximum. A grid point without neighbours yields zeros. The grid
    features of a seed, in x-y-z order of the crop frame, are flattened and reduced by
    ``fc_channels``.

    Args:
        radius: Half the side of the box across the approach axis, metres.
        hmin: Near end of the box along the approach axis, metres.
        hmax: Far end of the box along the approach axis, metres.
        grid_size: Grid points per side.
        pool_radius: Radius of the ball around a grid point, metres.
        nsample: Neighbours per grid point.
        mlp_channels: Widths of the shared MLP at the grid points.
        fc_channels: Widths of the layers after flattening; the last is ``out_channels``.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "tested"
    source = (
        "PV-RCNN (Shi et al., CVPR 2020), RoI-grid pooling; adapted from "
        "open-mmlab/OpenPCDet@233f849, Apache-2.0"
    )

    def __init__(
        self,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        grid_size: int = 3,
        pool_radius: float = 0.03,
        nsample: int = 8,
        mlp_channels: Sequence[int] = (64, 64),
        fc_channels: Sequence[int] = (256, 256),
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        _check_box(name, radius, hmin, hmax, grid_size)
        if pool_radius <= 0 or nsample < 1 or not mlp_channels or not fc_channels:
            raise ValueError(
                f"{name} needs pool_radius > 0, nsample >= 1 and at least one width in "
                f"mlp_channels and fc_channels; got pool_radius={pool_radius}, nsample={nsample}"
            )
        self.radius, self.hmin, self.hmax, self.grid_size = radius, hmin, hmax, grid_size
        self.pool_radius, self.nsample = pool_radius, nsample
        self.in_channels = in_channels
        self.out_channels = int(fc_channels[-1])
        self.mlp = _shared_mlp([3 + in_channels, *mlp_channels])
        self.fc = _shared_mlp([grid_size**3 * int(mlp_channels[-1]), *fc_channels])

    def grid_points(self, like: Tensor) -> Tensor:
        """``(grid_size³, 3)`` grid points in the crop frame, metres (cell centres)."""
        steps = (torch.arange(self.grid_size, device=like.device, dtype=like.dtype) + 0.5) / (
            self.grid_size
        )
        lower = like.new_tensor([self.hmin, -self.radius, -self.radius])
        upper = like.new_tensor([self.hmax, self.radius, self.radius])
        cells = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij"), dim=-1)
        return lower + cells.reshape(-1, 3) * (upper - lower)

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        coords, centre, frame = _seed_inputs(
            type(self).__name__, support, centres, rotation, self.in_channels
        )
        batch, seeds, _ = coords.shape
        local = self.grid_points(coords)  # (G, 3)
        cells = local.shape[0]
        # A point of the crop frame sits at c + R l in the scene.
        grid = centre[:, :, None, :] + torch.einsum("bsij,gj->bsgi", frame, local)
        grid = grid.reshape(batch, seeds * cells, 3)
        index, hits = ball_query(grid, coords, self.pool_radius, self.nsample)
        index = index.long()
        offset = group_points(coords, index) - grid[:, :, None, :]
        frames = frame[:, :, None].expand(-1, -1, cells, -1, -1).reshape(batch, -1, 3, 3)
        offset = torch.matmul(offset, frames) / self.pool_radius
        grouped = torch.cat([offset, group_points(support.dense(), index)], dim=-1)
        hidden = _tokens(self.mlp, grouped.flatten(0, 1)).amax(dim=1)
        hidden = hidden * (hits.reshape(-1, 1) > 0)  # an empty ball is zero, not point 0
        return GroupFeatures(self.fc(hidden.reshape(batch * seeds, -1)), self.out_axes)

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, hmin={self.hmin}, hmax={self.hmax}, "
            f"grid_size={self.grid_size}, pool_radius={self.pool_radius}, nsample={self.nsample}"
        )


@GROUPERS.register()
class RoIAwareGroup(Grouper):
    """RoI-aware pooling (Part-A²) on the gripper-aligned box of every seed.

    Up to ``nsample`` seeds inside the box are assigned to the ``grid_size³`` cells of
    the box by their position in the crop frame. A cell holds the maximum of the features
    and the mean position (relative to the box, in units of ``radius``) of its points; a
    cell without points is zero. Every cell passes a linear layer, and the cells, in x-y-z
    order, are flattened and reduced by ``fc_channels``. (Part-A² applies sparse
    convolutions to the cells; a linear layer per cell stands in for them here.)

    Args:
        radius: Half the side of the box across the approach axis, metres.
        hmin: Near end of the box along the approach axis, metres.
        hmax: Far end of the box along the approach axis, metres.
        grid_size: Cells per side.
        nsample: Points of a box that are pooled (the first in index order).
        cell_channels: Width of the linear layer of a cell.
        fc_channels: Widths of the layers after flattening; the last is ``out_channels``.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "tested"
    source = (
        "Part-A² (Shi et al., TPAMI 2020), RoI-aware point cloud pooling; adapted from "
        "open-mmlab/OpenPCDet@233f849, Apache-2.0"
    )

    def __init__(
        self,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        grid_size: int = 4,
        nsample: int = 32,
        cell_channels: int = 32,
        fc_channels: Sequence[int] = (256, 256),
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        _check_box(name, radius, hmin, hmax, grid_size)
        if nsample < 1 or cell_channels < 1 or not fc_channels:
            raise ValueError(
                f"{name} needs nsample >= 1, cell_channels >= 1 and at least one width in "
                f"fc_channels; got nsample={nsample}, cell_channels={cell_channels}"
            )
        self.radius, self.hmin, self.hmax, self.grid_size = radius, hmin, hmax, grid_size
        self.nsample, self.cell_channels = nsample, cell_channels
        self.in_channels = in_channels
        self.out_channels = int(fc_channels[-1])
        self.cell = _LinearNormAct(in_channels + 3, cell_channels)
        self.fc = _shared_mlp([grid_size**3 * cell_channels, *fc_channels])

    def pool(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> Tensor:
        """The pooled cells ``(M, grid_size³, in_channels + 3)``: maximum of the features,
        then the mean position."""
        coords, centre, frame = _seed_inputs(
            type(self).__name__, support, centres, rotation, self.in_channels
        )
        lower = (self.hmin, -self.radius, -self.radius)
        upper = (self.hmax, self.radius, self.radius)
        index, hits = box_query(centre, coords, frame, lower, upper, self.nsample)
        index = index.long()
        real = torch.arange(self.nsample, device=index.device) < hits[..., None]
        local = torch.matmul(group_points(coords, index) - centre[:, :, None, :], frame)
        low, high = local.new_tensor(lower), local.new_tensor(upper)
        unit = (local - low) / (high - low)  # in [0, 1) inside the box
        cell = (unit * self.grid_size).long().clamp(0, self.grid_size - 1)
        cell = (cell[..., 0] * self.grid_size + cell[..., 1]) * self.grid_size + cell[..., 2]
        crops, cells = index.shape[0] * index.shape[1], self.grid_size**3
        # Row of every slot in a (crops * cells + 1) table; padded slots go to the last row.
        row = torch.arange(crops, device=index.device).reshape(index.shape[:2])[..., None]
        row = torch.where(real, row * cells + cell, torch.full_like(cell, crops * cells))
        row = row.reshape(-1)
        feats = group_points(support.dense(), index).reshape(-1, self.in_channels)
        position = (local / self.radius).reshape(-1, 3)
        peak = feats.new_zeros(crops * cells + 1, self.in_channels)
        peak = peak.scatter_reduce(
            0, row[:, None].expand_as(feats), feats, reduce="amax", include_self=False
        )
        total = position.new_zeros(crops * cells + 1, 3).index_add(0, row, position)
        members = position.new_zeros(crops * cells + 1).index_add(
            0, row, torch.ones_like(position[:, 0])
        )
        mean = total / members.clamp(min=1)[:, None]
        return torch.cat([peak, mean], dim=-1)[:-1].reshape(crops, cells, -1)

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        pooled = self.pool(support, centres, rotation)
        crops, cells, channels = pooled.shape
        occupied = pooled.abs().sum(dim=-1, keepdim=True) > 0
        hidden = self.cell(pooled.reshape(crops * cells, channels)).reshape(crops, cells, -1)
        hidden = hidden * occupied  # an empty cell stays zero after the layer and its norm
        return GroupFeatures(self.fc(hidden.reshape(crops, -1)), self.out_axes)

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, hmin={self.hmin}, hmax={self.hmax}, "
            f"grid_size={self.grid_size}, nsample={self.nsample}"
        )


@GROUPERS.register()
class VoxelSetAbstractionGroup(Grouper):
    """The keypoint feature of PV-RCNN: scene points at several radii, fused with the seed.

    Every seed gathers up to ``nsample[i]`` scene points within ``radii[i]``; their
    offsets in the crop frame, divided by the radius (with the scene features appended
    when ``scene_features`` is set), pass a shared MLP per radius and a maximum; a ball
    without points is zero. The results are concatenated with the seed's own feature and
    fused by one linear layer. In PV-RCNN the sources of a keypoint are the raw points,
    four levels of the voxel encoder and a bird's-eye-view map; this slot is handed the
    scene and the seeds, which stand for the first and the last of those.

    Args:
        radii: Ball radii in metres.
        nsample: Scene points per ball, one entry per radius.
        mlp_channels: Widths of the shared MLP of every radius.
        scene_features: Read the features of the scene points too. Only a dense encoder
            returns them; under a sub-sampling encoder the scene is geometry only.
        out_channels: Width of the fused feature.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "scene"
    out_axes = ("seed", "channel")
    status = "tested"
    source = (
        "PV-RCNN (Shi et al., CVPR 2020), voxel set abstraction; adapted from "
        "open-mmlab/OpenPCDet@233f849, Apache-2.0"
    )

    def __init__(
        self,
        radii: Sequence[float] = (0.03, 0.06),
        nsample: Sequence[int] = (32, 32),
        mlp_channels: Sequence[int] = (32, 64),
        scene_features: bool = False,
        out_channels: int = 256,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        radii = tuple(float(value) for value in radii)
        nsample = tuple(int(value) for value in nsample)
        if not radii or min(radii) <= 0 or len(nsample) != len(radii) or min(nsample) < 1:
            raise ValueError(
                f"{name} needs positive radii and one nsample >= 1 per radius; got "
                f"radii={list(radii)}, nsample={list(nsample)}"
            )
        if not mlp_channels or out_channels < 1:
            raise ValueError(f"{name} needs at least one MLP width and out_channels >= 1")
        self.radii, self.nsample, self.scene_features = radii, nsample, scene_features
        self.in_channels, self.out_channels = in_channels, out_channels
        #: The seed feature is one of the sources, so ``forward`` takes it.
        self.uses_seed_features = True
        source = 3 + (in_channels if scene_features else 0)
        self.scales = nn.ModuleList(_shared_mlp([source, *mlp_channels]) for _ in radii)
        self.fuse = _LinearNormAct(in_channels + len(radii) * int(mlp_channels[-1]), out_channels)

    def forward(
        self,
        support: PointFeatures,
        centres: Tensor,
        rotation: Tensor,
        seed_feats: Tensor | None = None,
    ) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        name = type(self).__name__
        coords = support.points.dense()
        batch, count = coords.shape[0], centres.shape[0]
        if (
            tuple(centres.shape) != (count, 3)
            or tuple(rotation.shape) != (count, 3, 3)
            or (count % batch)
        ):
            raise ContractError(
                f"{name} groups over the scene (support = 'scene'): it needs one centre (M, 3) "
                f"and one frame (M, 3, 3) per seed, with the same number of seeds in each of the "
                f"{batch} frame(s), got {tuple(centres.shape)} and {tuple(rotation.shape)}"
            )
        if seed_feats is None or tuple(seed_feats.shape) != (count, self.in_channels):
            got = None if seed_feats is None else tuple(seed_feats.shape)
            raise ContractError(
                f"{name} fuses the crop with the ({count}, {self.in_channels}) features of the "
                f"seeds, got {got}"
            )
        if self.scene_features and support.channels != self.in_channels:
            raise ContractError(
                f"{name} is configured with scene_features: true, but the scene carries "
                f"{support.channels} feature channel(s) instead of {self.in_channels}. Only a "
                "dense encoder returns a feature per scene point; under a sub-sampling encoder "
                "set scene_features: false."
            )
        centre, frame = centres.reshape(batch, -1, 3), rotation.reshape(batch, -1, 3, 3)
        parts = [seed_feats]
        for radius, nsample, mlp in zip(self.radii, self.nsample, self.scales):
            index, hits = ball_query(centre, coords, radius, nsample)
            index = index.long()
            offset = group_points(coords, index) - centre[:, :, None, :]
            grouped = torch.matmul(offset, frame) / radius
            if self.scene_features:
                grouped = torch.cat([grouped, group_points(support.dense(), index)], dim=-1)
            real = torch.arange(nsample, device=index.device) < hits[..., None]
            hidden = _tokens(mlp, grouped.flatten(0, 1))
            parts.append(masked_pool(hidden, real.flatten(0, 1), "max", None))
        return GroupFeatures(self.fuse(torch.cat(parts, dim=-1)), self.out_axes)

    def extra_repr(self) -> str:
        return (
            f"radii={list(self.radii)}, nsample={list(self.nsample)}, "
            f"scene_features={self.scene_features}"
        )
