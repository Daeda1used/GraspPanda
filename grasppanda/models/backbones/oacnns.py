# Copyright (c) 2023 Pointcept
# Structure and parameter names follow https://github.com/Pointcept/Pointcept @ 1342eda (MIT),
# `pointcept/models/oacnns`, so that its released weights load; see THIRD_PARTY_NOTICES.md.
"""OA-CNNs: a sparse CNN whose receptive field adapts per voxel, without attention.

Paper: Peng et al., "OA-CNNs: Omni-Adaptive Sparse CNNs for 3D Semantic Segmentation",
CVPR 2024.

The network is a sparse U-Net: a stem of three 3³ submanifold convolutions, four
stages that halve the grid and run residual blocks, and four up-sampling stages that
fuse the encoder feature of the same resolution through an MLP. What sets it apart is
the block. Before its two 3³ convolutions, every voxel gathers context from several
*pooling grids*: the level is cut into cubes of ``point_grid_size`` cells, each voxel
votes with a learned, softmax-normalised weight inside its cube, and the weighted sum
of the cube is handed back to every voxel in it. A second softmax, predicted per voxel,
mixes the cubes of different sizes, so each voxel chooses how far it looks.

:class:`OACNNs` returns ``dense`` features. Points are quantised to a grid owned by the
backbone (``grid_size``) exactly as the official data pipeline does, the network runs
on the first point of every cell, and the feature of a cell is handed to every point in
it.

The official model treats a batch as one sparse tensor in three places, which makes the
result of a cloud depend on its batch neighbours. Here each is evaluated per cloud;
for a single cloud the two are the same computation.

* **Bounding box.** See ``crop_odd_extent``. Officially the box is that of the batch.
* **Softmax shift.** The voting weights are ``exp(logit - m)`` with ``m`` the largest
  logit, normalised with ``+ 1e-6`` in the denominator, so ``m`` is not just a
  numerical convenience: weights far below it are damped. Officially ``m`` is the
  maximum over the whole batch; here over the cloud.
* **Pooling grids.** The cubes are anchored at the lowest occupied cell of the level:
  officially that of the batch, here that of the cloud.

One behaviour of the official model is kept as an option: ``crop_odd_extent``, the way
it treats the border of its grid.

Constructor arguments of the official model that take no part in its computation are
not parameters here: ``groups`` (only checked to divide the width), ``enc_num_ref``,
``dec_depth`` (the decoder has no blocks) and ``down_ratio`` (the stride is fixed at 2).

Provenance. Written for this code base from the paper and from the behaviour of the
official implementation in Pointcept (``pointcept/models/oacnns/oacnns_v1m1_base.py``,
MIT licence). The module tree and parameter names (``stem.0``, ``enc.0.blocks.0.l_w``,
``dec.0.fuse`` ...) follow it on purpose, so that a state dict of the official model
loads without renaming. The cube assignment, which the official code takes from
torch_geometric and torch_cluster, is integer arithmetic here.

Needs spconv for the sparse convolutions; nothing else beyond PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops.sparse import check_cpu_convolution
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet
from grasppanda.utils.imports import require

from ._grid import grid_cells
from ._packed import check_features

__all__ = ["OACNNs"]


def _norm(channels: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


def _linear_norm_act(in_channels: int, out_channels: int, bias: bool = False) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels, bias=bias), _norm(out_channels), nn.ReLU()
    )


def _convolve(conv: nn.Module, tensor: Any) -> Any:
    """Apply a sparse convolution on whatever device the tensor lives on.

    A CUDA build of spconv defaults small kernels to an algorithm that only exists on
    the GPU, so on the CPU the layer is switched to the native one for the call (the
    weight layout is the same). The CPU path first asks the sparse facade whether the
    installed spconv can be trusted there at all.
    """
    if tensor.features.is_cuda:
        return conv(tensor)
    check_cpu_convolution()
    configured = conv.algo
    conv.algo = require("spconv.core", extra="spconv-cu118", purpose="OACNNs").ConvAlgo.Native
    try:
        return conv(tensor)
    finally:
        conv.algo = configured


def _run(layers: nn.Sequential, tensor: Any) -> Any:
    """Run a stack on a sparse tensor: norms and activations on its rows, the rest on it."""
    for layer in layers:
        if isinstance(layer, (nn.BatchNorm1d, nn.ReLU)):
            tensor = tensor.replace_feature(layer(tensor.features))
        else:
            tensor = _convolve(layer, tensor)
    return tensor


@dataclass(eq=False)
class _Pool:
    """One pooling grid of a level: the cube every voxel lies in."""

    cube: Tensor  # (n,) cube of every voxel
    count: Tensor  # (cubes, 1) voxels per cube

    @classmethod
    def build(cls, cells: Tensor, batch: Tensor, size: int) -> _Pool:
        cube = torch.div(cells, size, rounding_mode="floor")
        span = cube.amax(dim=0) + 1
        key = ((batch * span[0] + cube[:, 0]) * span[1] + cube[:, 1]) * span[2] + cube[:, 2]
        _, index = torch.unique(key, return_inverse=True)
        return cls(index, torch.bincount(index)[:, None])

    def sum(self, values: Tensor) -> Tensor:
        """``(cubes, C)`` sums of the ``(n, C)`` rows that share a cube."""
        out = values.new_zeros(len(self.count), values.shape[1])
        return out.scatter_add_(0, self.cube[:, None].expand_as(values), values)

    def mean(self, values: Tensor) -> Tensor:
        return self.sum(values) / self.count.to(values.dtype)


class _Block(nn.Module):
    """Adaptive aggregation over the pooling grids, then a residual pair of 3³ convolutions."""

    def __init__(self, spconv: Any, channels: int, grids: int, key: str) -> None:
        super().__init__()
        # One projection per pooling grid, and a last one for the voxel's own feature.
        self.proj = nn.ModuleList(_linear_norm_act(channels, channels) for _ in range(grids + 1))
        self.l_w = nn.ModuleList(_linear_norm_act(channels, channels) for _ in range(grids))
        self.weight = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(grids))
        self.adaptive = nn.Linear(channels, grids, bias=False)
        self.fuse = _linear_norm_act(2 * channels, channels)

        def conv() -> nn.Module:
            return spconv.SubMConv3d(
                channels, channels, kernel_size=3, padding=1, indice_key=key, bias=False
            )

        self.voxel_block = nn.Sequential(
            conv(), _norm(channels), nn.ReLU(), conv(), _norm(channels)
        )
        self.act = nn.ReLU()

    def forward(self, x: Any, pools: Sequence[_Pool], batch: Tensor, batch_size: int) -> Any:
        feat = x.features
        gathered = []
        for index, pool in enumerate(pools):
            logit = self.l_w[index](feat)
            logit = self.weight[index](logit - pool.mean(logit)[pool.cube])
            # A softmax over the voxels of a cube, per channel. The shift by the largest
            # logit of the cloud and the constant in the denominator are the official ones.
            top = logit.new_full((batch_size,), float("-inf"))
            top = top.scatter_reduce(0, batch, logit.amax(dim=1), reduce="amax")
            vote = torch.exp(logit - top[batch, None])
            vote = vote / (pool.sum(vote)[pool.cube] + 1e-6)
            gathered.append(pool.sum(self.proj[index](feat) * vote)[pool.cube])
        mix = torch.softmax(self.adaptive(feat), dim=1)
        context = torch.einsum("ng,ngc->nc", mix, torch.stack(gathered, dim=1))
        feat = self.fuse(torch.cat([self.proj[-1](feat), context], dim=1)) + feat
        out = _run(self.voxel_block, x.replace_feature(feat))
        return out.replace_feature(self.act(out.features + feat))


class _Down(nn.Module):
    """Halve the grid, then run the blocks of the stage on its pooling grids."""

    def __init__(
        self,
        spconv: Any,
        in_channels: int,
        channels: int,
        depth: int,
        grids: Sequence[int],
        stage: int,
    ) -> None:
        super().__init__()
        self.grids = tuple(grids)
        self.down = nn.Sequential(
            spconv.SparseConv3d(
                in_channels,
                channels,
                kernel_size=2,
                stride=2,
                indice_key=f"spconv{stage}",
                bias=False,
            ),
            _norm(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            _Block(spconv, channels, len(self.grids), f"subm{stage + 1}") for _ in range(depth)
        )

    def forward(self, x: Any, batch_size: int) -> Any:
        x = _run(self.down, x)
        batch, cells = x.indices[:, 0].long(), x.indices[:, 1:].long()
        index = batch[:, None].expand_as(cells)
        lowest = torch.zeros(batch_size, 3, dtype=torch.long, device=cells.device)
        lowest = lowest.scatter_reduce(0, index, cells, reduce="amin", include_self=False)
        cells = cells - lowest[batch]
        pools = [_Pool.build(cells, batch, size) for size in self.grids]
        for block in self.blocks:
            x = block(x, pools, batch, batch_size)
        return x


class _Up(nn.Module):
    """Restore the finer grid and fuse the encoder feature of that resolution."""

    def __init__(
        self, spconv: Any, in_channels: int, skip_channels: int, channels: int, stage: int
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            spconv.SparseInverseConv3d(
                in_channels, channels, kernel_size=2, indice_key=f"spconv{stage}", bias=False
            ),
            _norm(channels),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            *_linear_norm_act(skip_channels + channels, channels, bias=True),
            *_linear_norm_act(channels, channels, bias=True),
        )

    def forward(self, x: Any, skip: Any, rows: Tensor | None) -> Any:
        feat = _convolve(self.up[0], x).features
        if rows is not None:
            # Voxels the down-sampling left out receive nothing from the coarser level.
            feat = feat.new_zeros(len(skip.features), feat.shape[1]).index_copy(0, rows, feat)
        feat = self.up[2](self.up[1](feat))
        feat = self.fuse(torch.cat([feat, skip.features], dim=1)) + feat
        return skip.replace_feature(feat)


@BACKBONES.register()
class OACNNs(Backbone):
    """OA-CNNs encoder-decoder (Pointcept ``oacnns-v1m1``), one feature per input point.

    The defaults are the published ScanNet semantic-segmentation model
    (``semseg-oacnns-v1m1-0-base``, 51.5 M parameters): 0.02 m voxels, 9 input channels
    (coordinates, colour, normal), a 64-channel stem, four stages of 3-3-9-8 blocks with
    64-64-128-256 channels, a 256-channel decoder and a 20-channel output. For tabletop
    clouds a finer grid is usual (GSNet voxelises at 0.005 m); the pooling grids are
    counted in cells, so they shrink with it. At 0.02 m such a cloud is easily thinner
    than the 16 voxels that ``crop_odd_extent`` requires.

    Args:
        in_channels: Feature channels per point.
        out_channels: Channels of the returned features: the width of the official
            ``final`` 1×1 convolution, which the published model sets to the number of
            classes.
        grid_size: Edge of a voxel in the unit of the coordinates. Points sharing a
            voxel share one output feature.
        embed_channels: Width of the full-resolution stem.
        enc_channels: Width of every encoder stage; each stage halves the grid first.
        enc_depth: Blocks of every encoder stage.
        dec_channels: Width of every decoder stage, finest first.
        point_grid_size: For every encoder stage, the edge lengths of its pooling grids
            in cells of that stage (a stage-``s`` cell is ``2 ** (s + 1)`` voxels wide).
            A stage with a single grid has nothing to choose between: its ``adaptive``
            layers are then inert, as in the official model.
        crop_odd_extent: The official model declares the sparse grid as the exact
            bounding box of the voxels. A stride-2 convolution without padding cannot
            place the last cell of an axis with an odd number of cells, so at every
            level the voxels in that cell are left out of the down-sampling and receive
            zeros from the up-sampling; the box then shrinks to half, rounded down.
            ``True`` reproduces this, cloud by cloud, and needs every cloud to span
            ``2 ** stages`` voxels along every axis. ``False`` keeps every voxel.
    """

    output = "dense"
    status = "tested"
    source = "OA-CNNs (Peng et al., CVPR 2024), Pointcept oacnns-v1m1"
    extras = ("spconv",)
    #: Each stage halves the resolution.
    stage_stride = 2

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int:
        return options["out_channels"]

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 20,
        grid_size: float = 0.02,
        embed_channels: int = 64,
        enc_channels: Sequence[int] = (64, 64, 128, 256),
        enc_depth: Sequence[int] = (3, 3, 9, 8),
        dec_channels: Sequence[int] = (256, 256, 256, 256),
        point_grid_size: Sequence[Sequence[int]] = (
            (8, 12, 16, 16),
            (6, 9, 12, 12),
            (4, 6, 8, 8),
            (3, 4, 6, 6),
        ),
        crop_odd_extent: bool = True,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        spconv = require("spconv.pytorch", extra="spconv-cu118", purpose=name)
        stages = len(enc_channels)
        if in_channels < 1:
            raise ValueError(f"{name} needs at least one input feature channel per point")
        if grid_size <= 0:
            raise ValueError(f"{name}: grid_size must be positive, got {grid_size}")
        if stages == 0 or any(
            len(values) != stages for values in (enc_depth, dec_channels, point_grid_size)
        ):
            raise ValueError(
                f"{name}: enc_channels, enc_depth, dec_channels and point_grid_size need one "
                f"entry per stage, got {len(enc_channels)}, {len(enc_depth)}, "
                f"{len(dec_channels)} and {len(point_grid_size)}"
            )
        if any(len(grids) == 0 or min(grids) < 1 for grids in point_grid_size):
            raise ValueError(
                f"{name}: every stage needs at least one pooling grid of one cell or more, "
                f"got point_grid_size={[list(grids) for grids in point_grid_size]}"
            )
        self.in_channels, self.out_channels = in_channels, out_channels
        self.grid_size = grid_size
        self.crop_odd_extent = crop_odd_extent
        self.num_stages = stages
        self.total_stride = self.stage_stride**stages

        def conv(channels: int) -> nn.Module:
            return spconv.SubMConv3d(
                channels, embed_channels, kernel_size=3, padding=1, indice_key="stem", bias=False
            )

        self.stem = nn.Sequential(
            conv(in_channels),
            _norm(embed_channels),
            nn.ReLU(),
            conv(embed_channels),
            _norm(embed_channels),
            nn.ReLU(),
            conv(embed_channels),
            _norm(embed_channels),
            nn.ReLU(),
        )
        widths = [embed_channels, *enc_channels]  # width of level 0 (the stem), 1, 2, ...
        below = [*dec_channels[1:], enc_channels[-1]]  # what every decoder stage reads
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for stage in range(stages):
            self.enc.append(
                _Down(
                    spconv,
                    widths[stage],
                    widths[stage + 1],
                    enc_depth[stage],
                    point_grid_size[stage],
                    stage,
                )
            )
            self.dec.append(_Up(spconv, below[stage], widths[stage], dec_channels[stage], stage))
        self.final = spconv.SubMConv3d(dec_channels[0], out_channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise as the official model does.

        Linear layers and submanifold convolutions: truncated normal with a standard
        deviation of 0.02, zero bias. Batch norm: identity. The down- and up-sampling
        convolutions keep the initialisation of the sparse library.
        """
        submanifold = require("spconv.pytorch", extra="spconv-cu118", purpose="OACNNs").SubMConv3d
        for module in self.modules():
            if isinstance(module, (nn.Linear, submanifold)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode(self, cells: Tensor, batch: Tensor, extent: Tensor, feat: Tensor) -> Tensor:
        """Run the network on voxels and return one feature vector per voxel."""
        name = type(self).__name__
        spconv = require("spconv.pytorch", extra="spconv-cu118", purpose=name)
        batch_size = len(extent)
        shape = extent.amax(dim=0)
        if not self.crop_odd_extent:
            # A box that halves exactly at every level never loses a cell.
            shape = (shape + self.total_stride - 1) // self.total_stride * self.total_stride
        indices = torch.cat([batch[:, None], cells], dim=1).int().contiguous()
        x = _run(self.stem, spconv.SparseConvTensor(feat, indices, shape.tolist(), batch_size))
        skips, kept = [], []
        for level, stage in enumerate(self.enc):
            skips.append(x)
            rows = None
            if self.crop_odd_extent:
                extent = extent // self.stage_stride
                cloud = x.indices[:, 0].long()
                inside = (x.indices[:, 1:] // self.stage_stride < extent[cloud]).all(dim=1)
                left = torch.bincount(cloud[inside], minlength=batch_size)
                if int(left.min()) == 0:
                    empty = int(left.argmin())
                    raise ContractError(
                        f"{name}: cloud {empty} of the batch has no voxel left after "
                        f"{level + 1} of {self.num_stages} down-sampling steps. With "
                        "crop_odd_extent=True the sparse grid is the bounding box of the cloud "
                        "and loses the last cell of an odd axis at every level, so a cloud has "
                        f"to span at least {self.total_stride} voxels along every axis. Use a "
                        "smaller grid_size, or set crop_odd_extent=False to keep every voxel."
                    )
                if not bool(inside.all()):
                    rows = torch.nonzero(inside).flatten()
                    x = spconv.SparseConvTensor(
                        x.features[rows], x.indices[rows], x.spatial_shape, batch_size
                    )
            kept.append(rows)
            x = stage(x, batch_size)
        for stage, skip, rows in zip(reversed(self.dec), reversed(skips), reversed(kept)):
            x = stage(x, skip, rows)
        return _convolve(self.final, x).features

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        grid = grid_cells(points, self.grid_size, user=name)
        out = self._encode(grid.cells, grid.batch, grid.extent, feats[grid.first])
        return PointFeatures(out[grid.inverse], points)
