# Copyright (c) 2023 Matrix-ASC
# Adapted for GraspPanda from https://github.com/Matrix-ASC/DeLA @ 0f427c3 (MIT):
# ``S3DIS/delasemseg.py`` and the pyramid construction of ``S3DIS/s3dis.py``.
"""DeLA: decoupled local aggregation.

Paper: Chen, Xia, Zu, Cao, Zhu, Wang, Liu, Lu, "Decoupled Local Aggregation for Point
Cloud Learning", arXiv 2308.16532 (2023).

Most point networks recompute relative positions in every layer. DeLA encodes the
neighbour offsets *once per stage* (a small MLP and a max-pool over the ``k`` nearest
neighbours) and then alternates point-wise MLPs with a parameter-light "edge
max-pooling" ``max_j f_j - f_i`` over the same neighbours. Stages live on a grid
pyramid (each level keeps one point per cell of a coarser grid); every stage projects
its features to a common width, copies them back to the input points through the
nearest kept point, and the copies are summed. Features on every input point:
``output = "dense"``.

Differences from the upstream, each because the slot needs it:

* the upstream builds neighbour lists, grid subsampling and back-projection indices in
  its data pipeline with compiled code; here the model builds them, per cloud of the
  packed batch, with the operators of :mod:`grasppanda.ops`;
* a cell is represented by its first point in input order, in training too (the
  upstream picks a random point of the cell while training);
* the auxiliary loss of the upstream (regressing relative coordinates from feature
  differences) is not built: a backbone of this slot returns features only;
* stochastic depth is left out;
* the hidden unit of length of the upstream (coordinates multiplied by 40 for a 4 cm
  grid) is derived from ``grid_size[0]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import knn
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointFeatures, PointSet

from ._packed import check_features, cloud_sizes, knn_packed

__all__ = ["DeLA"]


@torch.no_grad()
def _knn_padded(coords: Tensor, sizes: Sequence[int], k: int) -> Tensor:
    """``(N, k)`` rows of the nearest points inside each cloud. A cloud with fewer than
    ``k`` points repeats its nearest neighbour (the point itself), which a max-pool over
    the neighbours does not notice."""
    found, start = [], 0
    for size in sizes:
        block = coords[start : start + size][None]
        index = knn(block, block, min(k, size))[1][0] + start
        if size < k:
            index = torch.cat([index, index[:, :1].expand(-1, k - size)], dim=1)
        found.append(index)
        start += size
    return torch.cat(found)


@torch.no_grad()
def _grid_subsample(coords: Tensor, sizes: Sequence[int], grid: float) -> tuple[Tensor, list[int]]:
    """Rows of the first point of every occupied cell of each cloud, in input order."""
    owner = torch.repeat_interleave(
        torch.arange(len(sizes), device=coords.device),
        torch.as_tensor(sizes, device=coords.device),
    )
    cells = torch.floor(coords / grid).long()
    keys = torch.cat([owner[:, None], cells - cells.amin(dim=0)], dim=1)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    rows = torch.arange(len(coords), device=coords.device)
    first = rows.new_full((int(inverse.max()) + 1,), len(coords)).scatter_reduce(
        0, inverse, rows, reduce="amin"
    )
    first = first.sort().values
    counts = torch.bincount(owner[first], minlength=len(sizes)).tolist()
    return first, counts


def _edge_maxpool(x: Tensor, neighbours: Tensor) -> Tensor:
    return x[neighbours].amax(dim=1) - x


class _LFP(nn.Module):
    """Local feature propagation: ``bn(max_j linear(f)_j - linear(f)_i)``."""

    def __init__(self, inputs: int, outputs: int, momentum: float, init: float | None) -> None:
        super().__init__()
        self.proj = nn.Linear(inputs, outputs, bias=False)
        self.bn = nn.BatchNorm1d(outputs, momentum=momentum)
        if init is not None:
            nn.init.constant_(self.bn.weight, init)

    def forward(self, x: Tensor, neighbours: Tensor) -> Tensor:
        return self.bn(_edge_maxpool(self.proj(x), neighbours))


class _Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float, momentum: float, init: float | None) -> None:
        super().__init__()
        hidden = round(dim * ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
            nn.BatchNorm1d(dim, momentum=momentum),
        )
        if init is not None:
            nn.init.constant_(self.mlp[-1].weight, init)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class _Block(nn.Module):
    def __init__(self, dim: int, depth: int, ratio: float, momentum: float, zero: bool) -> None:
        super().__init__()
        self.lfps = nn.ModuleList(
            _LFP(dim, dim, momentum, 0.0 if zero else None) for _ in range(depth)
        )
        self.mlp = _Mlp(dim, ratio, momentum, 0.2 if zero else None)
        self.mlps = nn.ModuleList(
            _Mlp(dim, ratio, momentum, 0.0 if zero else None) for _ in range(depth // 2)
        )

    def forward(self, x: Tensor, neighbours: Tensor) -> Tensor:
        x = x + self.mlp(x)
        for index, lfp in enumerate(self.lfps):
            x = x + lfp(x, neighbours)
            if index % 2 == 1:
                x = x + self.mlps[index // 2](x)
        return x


class _Stage(nn.Module):
    def __init__(
        self,
        level: int,
        in_channels: int,
        dims: Sequence[int],
        depth: int,
        nbr_dims: Sequence[int],
        head_dim: int,
        ratio: float,
        momentum: float,
        zero: bool,
    ) -> None:
        super().__init__()
        self.first = level == 0
        dim = dims[level]
        nbr_in = 3 + in_channels if self.first else 3
        hidden = nbr_dims[0] if self.first else nbr_dims[1] // 2
        nbr_out = dim if self.first else nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in, hidden // 2, bias=False),
            nn.BatchNorm1d(hidden // 2, momentum=momentum),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden, bias=False),
            nn.BatchNorm1d(hidden, momentum=momentum),
            nn.GELU(),
            nn.Linear(hidden, nbr_out, bias=False),
        )
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=momentum)
        self.nbr_proj = nn.Identity() if self.first else nn.Linear(nbr_out, dim, bias=False)
        if not self.first:
            self.lfp = _LFP(dims[level - 1], dim, momentum, 0.3 if zero else None)
            self.skip_proj = nn.Sequential(
                nn.Linear(dims[level - 1], dim, bias=False),
                nn.BatchNorm1d(dim, momentum=momentum),
            )
        self.blk = _Block(dim, depth, ratio, momentum, zero)
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=momentum), nn.Linear(dim, head_dim, bias=False)
        )
        if zero:
            nn.init.constant_(self.nbr_bn.weight, 0.8 if self.first else 0.2)
            nn.init.constant_(self.postproj[0].weight, (dims[0] / dim) ** 0.5)
            if not self.first:
                nn.init.constant_(self.skip_proj[1].weight, 0.3)

    def embed(self, scaled: Tensor, feats: Tensor | None, neighbours: Tensor) -> Tensor:
        """Encode the neighbour offsets (and, in the first stage, the input features)."""
        count, k = neighbours.shape
        nbr = scaled[neighbours] - scaled[:, None, :]
        if feats is not None:
            nbr = torch.cat([nbr, feats[neighbours]], dim=-1)
        nbr = self.nbr_embed(nbr.reshape(count * k, -1)).reshape(count, k, -1).amax(dim=1)
        return self.nbr_bn(self.nbr_proj(nbr))


@BACKBONES.register()
class DeLA(Backbone):
    """DeLA encoder with features on every input point.

    The defaults are the S3DIS model of the official repository: four stages on grids
    of 0.04 / 0.08 / 0.16 / 0.32 m, 24 neighbours, depths 4-4-8-4, widths 64 to 512,
    256 output channels.

    Args:
        in_channels: Feature channels per point besides the coordinates. They enter
            the first stage next to the neighbour offsets.
        grid_size: One entry per stage. The first is the point spacing of the input
            cloud and only sets the unit in which offsets are encoded (an offset of
            ``grid_size[0]`` becomes 1.6, as upstream); the others are the cell sizes
            the cloud is subsampled to for the later stages. Level sizes therefore
            depend on the frame.
        k: Neighbours of every stage.
        depths: Local aggregation layers of every stage.
        dims: Width of every stage.
        nbr_dims: Hidden widths of the offset embedding (first stage, later stages).
        head_dim: Width every stage is projected to before the copies are summed.
        mlp_ratio: Width multiplier of the point-wise MLPs.
        bn_momentum: Momentum of all batch norms.
        zero_init_residual: Start the residual branches near zero with the constants
            of the official code (batch-norm scales of 0, 0.2, 0.3 and 0.8). ``false``
            keeps PyTorch's defaults.
    """

    output = "dense"
    status = "tested"
    source = "DeLA (Chen et al., arXiv 2308.16532), adapted from Matrix-ASC/DeLA@0f427c3, MIT"

    def __init__(
        self,
        in_channels: int = 4,
        grid_size: Sequence[float] = (0.04, 0.08, 0.16, 0.32),
        k: Sequence[int] = (24, 24, 24, 24),
        depths: Sequence[int] = (4, 4, 8, 4),
        dims: Sequence[int] = (64, 128, 256, 512),
        nbr_dims: Sequence[int] = (32, 32),
        head_dim: int = 256,
        mlp_ratio: float = 2.0,
        bn_momentum: float = 0.02,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        stages = len(grid_size)
        if not (stages == len(k) == len(depths) == len(dims)) or stages == 0:
            raise ValueError(
                f"DeLA: grid_size, k, depths and dims need one entry per stage, got "
                f"{stages}, {len(k)}, {len(depths)}, {len(dims)}"
            )
        if any(b <= a for a, b in pairwise(grid_size)) or grid_size[0] <= 0:
            raise ValueError(
                f"DeLA: grid_size must be positive and grow from stage to stage (metres), "
                f"got {list(grid_size)}"
            )
        if len(nbr_dims) != 2:
            raise ValueError("DeLA: nbr_dims takes two widths (first stage, later stages)")
        self.in_channels, self.out_channels = in_channels, head_dim
        self.grid_size, self.k = tuple(grid_size), tuple(k)
        self.coord_scale = 1.6 / grid_size[0]
        self.stages = nn.ModuleList(
            _Stage(
                level,
                in_channels,
                dims,
                depths[level],
                nbr_dims,
                head_dim,
                mlp_ratio,
                bn_momentum,
                zero_init_residual,
            )
            for level in range(stages)
        )
        self.head = nn.Sequential(nn.BatchNorm1d(head_dim, momentum=bn_momentum), nn.GELU())
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="DeLA")
        sizes0 = cloud_sizes(points)
        coords0 = points.coords
        coords, sizes = coords0, sizes0
        neighbours = _knn_padded(coords, sizes, self.k[0])
        x = self.stages[0].embed(coords * self.coord_scale, feats, neighbours)
        x = self.stages[0].blk(x, neighbours)
        out = self.stages[0].postproj(x)
        for level in range(1, len(self.stages)):
            stage = self.stages[level]
            keep, counts = _grid_subsample(coords, sizes, self.grid_size[level])
            x = stage.skip_proj(x)[keep] + stage.lfp(x, neighbours)[keep]
            coords, sizes = coords[keep], counts
            neighbours = _knn_padded(coords, sizes, self.k[level])
            x = stage.embed(coords * self.coord_scale, None, neighbours) + x
            x = stage.blk(x, neighbours)
            back = knn_packed(coords0, sizes0, coords, sizes, 1)[1][:, 0]
            out = out + stage.postproj(x)[back]
        return PointFeatures(self.head(out), points)
