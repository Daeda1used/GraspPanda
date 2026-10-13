# Copyright (c) 2023 Pointcept
# Adapted for GraspPanda from https://github.com/Pointcept/Pointcept @ 1342eda (MIT)
"""Point Transformer V1 and V2: vector attention over k nearest neighbours.

Papers: Zhao, Jiang, Jia, Torr, Koltun, "Point Transformer", ICCV 2021; Wu, Lao, Jiang,
Liu, Zhao, "Point Transformer V2: Grouped Vector Attention and Partition-based
Pooling", NeurIPS 2022.

Both are U-Nets that return one feature per input point (``dense``). They differ in
how they change resolution and in the attention layer:

* :class:`PointTransformerV1` subsamples by furthest point sampling with a ratio, pools
  the k nearest neighbours of every sample, and goes back up by three-neighbour
  interpolation. Its attention predicts one weight per group of ``share_planes``
  channels.
* :class:`PointTransformerV2` pools all points of a grid cell (cell edges in metres,
  one per stage) and un-pools by sending a cell's feature back to its points. Its
  attention predicts one weight per group of channels from the difference of key and
  query plus an encoding of the offset.

Provenance. The module trees, parameter names and arithmetic are those of
``pointcept/models/point_transformer/point_transformer_seg.py`` and
``pointcept/models/point_transformer_v2/point_transformer_v2m2_base.py`` (the variant
the Pointcept README recommends), without their classification heads. What differs:
the compiled ``pointops`` queries, ``torch_geometric.voxel_grid`` and ``torch_scatter``
are replaced by the operators of :mod:`grasppanda.ops` and ``scatter_reduce``; a cloud
with fewer points than neighbours is handled by a mask in both models (upstream V1 has
no such case, upstream V2 masks after the soft-max, as here). Neither repository
released weights for these models; they start from random weights.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import knn
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import check_features, cloud_sizes, fps_packed

__all__ = ["PointTransformerV1", "PointTransformerV2"]


# --------------------------------------------------------------------------- queries


@torch.no_grad()
def _neighbours(
    query: Tensor, query_sizes: Sequence[int], support: Tensor, support_sizes: Sequence[int], k: int
) -> tuple[Tensor, Tensor, Tensor | None]:
    """``k`` nearest support points of every query point of the same cloud.

    Returns:
        ``(distance, index, valid)``: ``(M, k)`` each. A cloud with fewer than ``k``
        support points repeats its nearest neighbour in the missing columns and marks
        them ``False`` in ``valid``; ``valid`` is ``None`` when nothing is missing.
    """
    distances, indices, valids = [], [], []
    q_start = s_start = 0
    missing = False
    for q_size, s_size in zip(query_sizes, support_sizes):
        found = min(k, s_size)
        block = support[s_start : s_start + s_size][None]
        distance, index = knn(query[q_start : q_start + q_size][None], block, found)
        distance, index = distance[0], index[0] + s_start
        valid = torch.ones_like(index, dtype=torch.bool)
        if found < k:
            missing = True
            pad = k - found
            distance = torch.cat([distance, distance[:, :1].expand(-1, pad)], dim=1)
            index = torch.cat([index, index[:, :1].expand(-1, pad)], dim=1)
            valid = torch.cat([valid, valid.new_zeros(q_size, pad)], dim=1)
        distances.append(distance)
        indices.append(index)
        valids.append(valid)
        q_start += q_size
        s_start += s_size
    return torch.cat(distances), torch.cat(indices), torch.cat(valids) if missing else None


def _segment(values: Tensor, cluster: Tensor, count: int, reduce: str) -> Tensor:
    """Reduce the rows of ``values`` that share a cluster (``amax``, ``amin`` or ``mean``)."""
    out = values.new_zeros(count, values.shape[1])
    index = cluster[:, None].expand_as(values)
    return out.scatter_reduce(0, index, values, reduce=reduce, include_self=False)


def _batch_index(sizes: Sequence[int], device: torch.device) -> Tensor:
    counts = torch.as_tensor(sizes, device=device)
    return torch.repeat_interleave(torch.arange(len(sizes), device=device), counts)


class _PointBatchNorm(nn.Module):
    """Batch norm over the channels of ``(M, C)`` rows or ``(M, K, C)`` neighbourhoods."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class _NeighbourNorm(nn.BatchNorm1d):
    """Batch norm over the channels of ``(M, K, C)`` (named ``LayerNorm1d`` upstream)."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class _DropPath(nn.Module):
    """Stochastic depth, drawn per point."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        return x * x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep


# ------------------------------------------------------------------ Point Transformer V1


class _V1Attention(nn.Module):
    def __init__(self, planes: int, share_planes: int, nsample: int) -> None:
        super().__init__()
        self.planes, self.share_planes, self.nsample = planes, share_planes, nsample
        self.linear_q = nn.Linear(planes, planes)
        self.linear_k = nn.Linear(planes, planes)
        self.linear_v = nn.Linear(planes, planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3), _NeighbourNorm(3), nn.ReLU(inplace=True), nn.Linear(3, planes)
        )
        shared = planes // share_planes
        self.linear_w = nn.Sequential(
            _NeighbourNorm(planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, shared),
            _NeighbourNorm(shared),
            nn.ReLU(inplace=True),
            nn.Linear(shared, shared),
        )

    def forward(self, coords: Tensor, feats: Tensor, sizes: Sequence[int]) -> Tensor:
        _, index, valid = _neighbours(coords, sizes, coords, sizes, self.nsample)
        query, key, value = self.linear_q(feats), self.linear_k(feats), self.linear_v(feats)
        position = self.linear_p(coords[index] - coords[:, None, :])  # (M, K, C)
        weight = self.linear_w(key[index] - query[:, None, :] + position)  # (M, K, C / s)
        if valid is not None:
            weight = weight.masked_fill(~valid[:, :, None], float("-inf"))
        weight = torch.softmax(weight, dim=1)
        value = (value[index] + position).unflatten(-1, (self.share_planes, -1))
        return torch.einsum("ntsi,nti->nsi", value, weight).flatten(1)


class _V1Bottleneck(nn.Module):
    def __init__(self, planes: int, share_planes: int, nsample: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = _V1Attention(planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, coords: Tensor, feats: Tensor, sizes: Sequence[int]) -> Tensor:
        out = self.relu(self.bn1(self.linear1(feats)))
        out = self.relu(self.bn2(self.transformer(coords, out, sizes)))
        return self.relu(self.bn3(self.linear3(out)) + feats)


class _V1Down(nn.Module):
    """Stride 1: a linear embedding. Otherwise: furthest point sample, pool k neighbours."""

    def __init__(self, in_planes: int, out_planes: int, stride: int, nsample: int) -> None:
        super().__init__()
        self.stride, self.nsample = stride, nsample
        self.linear = nn.Linear(in_planes + (3 if stride != 1 else 0), out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, coords: Tensor, feats: Tensor, sizes: list[int], presampled: bool
    ) -> tuple[Tensor, Tensor, list[int], Tensor | None]:
        if self.stride == 1:
            return coords, self.relu(self.bn(self.linear(feats))), sizes, None
        counts = [size // self.stride for size in sizes]
        chosen = fps_packed(coords, sizes, counts, presampled=presampled)
        centres = coords[chosen]
        _, index, _ = _neighbours(centres, counts, coords, sizes, self.nsample)
        grouped = torch.cat([coords[index] - centres[:, None, :], feats[index]], dim=-1)
        out = self.relu(self.bn(self.linear(grouped).transpose(1, 2)))
        return centres, out.amax(dim=-1), counts, chosen


class _V1Up(nn.Module):
    """The coarsest level mixes in the mean of its cloud; the others interpolate."""

    def __init__(self, in_planes: int, out_planes: int | None) -> None:
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def head(self, feats: Tensor, sizes: Sequence[int]) -> Tensor:
        batch = _batch_index(sizes, feats.device)
        mean = _segment(feats, batch, len(sizes), "mean")
        return self.linear1(torch.cat([feats, self.linear2(mean)[batch]], dim=1))

    def forward(
        self,
        coords: Tensor,
        feats: Tensor,
        sizes: Sequence[int],
        coarse_coords: Tensor,
        coarse_feats: Tensor,
        coarse_sizes: Sequence[int],
    ) -> Tensor:
        distance, index, valid = _neighbours(coords, sizes, coarse_coords, coarse_sizes, 3)
        weight = 1.0 / (distance + 1e-8)
        if valid is not None:
            weight = weight * valid
        weight = weight / weight.sum(dim=1, keepdim=True)
        lifted = (self.linear2(coarse_feats)[index] * weight[:, :, None]).sum(dim=1)
        return self.linear1(feats) + lifted


@BACKBONES.register()
class PointTransformerV1(Backbone):
    """Point Transformer (V1) U-Net with features on every input point.

    The defaults are ``PointTransformer-Seg26`` of Pointcept: five stages of 32 to 512
    channels, one attention block each, sub-sampling by 4 between stages. ``blocks``
    gives the larger published models: ``[1, 2, 2, 2, 2]`` is Seg38 and
    ``[1, 2, 3, 5, 2]`` Seg50 (the model of the paper). Neighbourhoods are counted in
    points, so the encoder has no unit of length to re-tune; its receptive field
    follows the density of the cloud.

    Args:
        in_channels: Feature channels per point, at least one.
        out_channels: ``None`` returns the ``planes[0]`` channels of the last decoder
            stage; a number adds the first half of the official segmentation head
            (linear, batch norm, ReLU) with that width.
        planes: Channels of every stage.
        blocks: Attention blocks per stage.
        strides: Sub-sampling ratio of every stage; the first must be 1.
        nsamples: Neighbours of the attention (and of the pooling) of every stage.
        share_planes: Channels that share one attention weight.
    """

    output = "dense"
    status = "tested"
    source = (
        "Point Transformer (Zhao et al., ICCV 2021); adapted from Pointcept/Pointcept@1342eda, MIT"
    )

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int | None = None,
        planes: Sequence[int] = (32, 64, 128, 256, 512),
        blocks: Sequence[int] = (1, 1, 1, 1, 1),
        strides: Sequence[int] = (1, 4, 4, 4, 4),
        nsamples: Sequence[int] = (8, 16, 16, 16, 16),
        share_planes: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("PointTransformerV1 needs at least one input feature channel")
        if not len(planes) == len(blocks) == len(strides) == len(nsamples) >= 2:
            raise ValueError(
                "PointTransformerV1: planes, blocks, strides and nsamples need one entry per "
                f"stage (at least two), got {len(planes)}, {len(blocks)}, {len(strides)}, "
                f"{len(nsamples)}"
            )
        if strides[0] != 1 or any(stride < 2 for stride in strides[1:]):
            raise ValueError(
                "PointTransformerV1: the first stride must be 1 and the others at least 2, got "
                f"{list(strides)}"
            )
        if any(width % share_planes for width in planes):
            raise ValueError(
                f"PointTransformerV1: every entry of planes {list(planes)} must be a multiple "
                f"of share_planes={share_planes}"
            )
        self.in_channels = in_channels
        self.out_channels = planes[0] if out_channels is None else out_channels
        self.cls: nn.Module = (
            nn.Identity()
            if out_channels is None
            else nn.Sequential(
                nn.Linear(planes[0], out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        self.strides = tuple(strides)
        stages = range(len(planes))
        self.down = nn.ModuleList(
            _V1Down(in_channels if i == 0 else planes[i - 1], planes[i], strides[i], nsamples[i])
            for i in stages
        )
        self.enc = nn.ModuleList(
            nn.ModuleList(
                _V1Bottleneck(planes[i], share_planes, nsamples[i]) for _ in range(blocks[i])
            )
            for i in stages
        )
        last = len(planes) - 1
        self.up = nn.ModuleList(
            _V1Up(planes[i], None) if i == last else _V1Up(planes[i + 1], planes[i]) for i in stages
        )
        self.dec = nn.ModuleList(
            _V1Bottleneck(planes[i], share_planes, nsamples[i]) for i in stages
        )

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="PointTransformerV1")
        sizes = cloud_sizes(points)
        needed = math.prod(self.strides)
        if min(sizes) < needed:
            raise ContractError(
                f"PointTransformerV1 subsamples by {'x'.join(map(str, self.strides[1:]))}, so "
                f"every cloud needs at least {needed} points; cloud {sizes.index(min(sizes))} of "
                f"the batch has {min(sizes)}."
            )
        coords = points.coords
        levels: list[tuple[Tensor, Tensor, list[int]]] = []
        for index, (down, blocks) in enumerate(zip(self.down, self.enc)):
            # From the second sub-sampling on, the input is a furthest point sample already.
            coords, feats, sizes, _ = down(coords, feats, sizes, presampled=index > 1)
            for block in blocks:
                feats = block(coords, feats, sizes)
            levels.append((coords, feats, sizes))
        coords, feats, sizes = levels[-1]
        feats = self.dec[-1](coords, self.up[-1].head(feats, sizes), sizes)
        for index in reversed(range(len(levels) - 1)):
            coarse = (coords, feats, sizes)
            coords, skip, sizes = levels[index]
            feats = self.dec[index](coords, self.up[index](coords, skip, sizes, *coarse), sizes)
        return PointFeatures(self.cls(feats), points)


# ------------------------------------------------------------------ Point Transformer V2


class _GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        attn_drop_rate: float,
        qkv_bias: bool,
        pe_multiplier: bool,
        pe_bias: bool,
    ) -> None:
        super().__init__()
        self.groups = groups

        def embed(inputs: int, outputs: int, bias: bool = True) -> list[nn.Module]:
            layers = [nn.Linear(inputs, outputs, bias=bias), _PointBatchNorm(outputs)]
            return [*layers, nn.ReLU(inplace=True)]

        self.linear_q = nn.Sequential(*embed(channels, channels, qkv_bias))
        self.linear_k = nn.Sequential(*embed(channels, channels, qkv_bias))
        self.linear_v = nn.Linear(channels, channels, bias=qkv_bias)
        if pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                *embed(3, channels), nn.Linear(channels, channels)
            )
        if pe_bias:
            self.linear_p_bias = nn.Sequential(*embed(3, channels), nn.Linear(channels, channels))
        self.pe_multiplier, self.pe_bias = pe_multiplier, pe_bias
        self.weight_encoding = nn.Sequential(*embed(channels, groups), nn.Linear(groups, groups))
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feats: Tensor, coords: Tensor, index: Tensor, valid: Tensor | None) -> Tensor:
        query, key, value = self.linear_q(feats), self.linear_k(feats), self.linear_v(feats)
        offset = coords[index] - coords[:, None, :]
        relation = key[index] - query[:, None, :]
        value = value[index]
        if self.pe_multiplier:
            relation = relation * self.linear_p_multiplier(offset)
        if self.pe_bias:
            bias = self.linear_p_bias(offset)
            relation = relation + bias
            value = value + bias
        weight = self.attn_drop(torch.softmax(self.weight_encoding(relation), dim=1))
        if valid is not None:
            weight = weight * valid[:, :, None]
        value = value.unflatten(-1, (self.groups, -1))
        return torch.einsum("nsgi,nsg->ngi", value, weight).flatten(1)


class _V2Block(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        qkv_bias: bool,
        pe_multiplier: bool,
        pe_bias: bool,
        attn_drop_rate: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.attn = _GroupedVectorAttention(
            channels, groups, attn_drop_rate, qkv_bias, pe_multiplier, pe_bias
        )
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc3 = nn.Linear(channels, channels, bias=False)
        self.norm1 = _PointBatchNorm(channels)
        self.norm2 = _PointBatchNorm(channels)
        self.norm3 = _PointBatchNorm(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = _DropPath(drop_path_rate)

    def forward(self, coords: Tensor, feats: Tensor, index: Tensor, valid: Tensor | None) -> Tensor:
        out = self.act(self.norm1(self.fc1(feats)))
        out = self.act(self.norm2(self.attn(out, coords, index, valid)))
        out = self.norm3(self.fc3(out))
        return self.act(feats + self.drop_path(out))


class _V2Blocks(nn.Module):
    """Blocks of one resolution; they share one neighbour query."""

    def __init__(
        self,
        depth: int,
        channels: int,
        groups: int,
        neighbours: int,
        drop_path_rates: Sequence[float],
        qkv_bias: bool,
        pe_multiplier: bool,
        pe_bias: bool,
        attn_drop_rate: float,
    ) -> None:
        super().__init__()
        self.neighbours = neighbours
        self.blocks = nn.ModuleList(
            _V2Block(
                channels,
                groups,
                qkv_bias,
                pe_multiplier,
                pe_bias,
                attn_drop_rate,
                drop_path_rates[i],
            )
            for i in range(depth)
        )

    def forward(self, coords: Tensor, feats: Tensor, sizes: Sequence[int]) -> Tensor:
        _, index, valid = _neighbours(coords, sizes, coords, sizes, self.neighbours)
        for block in self.blocks:
            feats = block(coords, feats, index, valid)
        return feats


class _GridPool(nn.Module):
    """Pool the points of every grid cell: mean coordinate, maximum feature."""

    def __init__(self, in_channels: int, out_channels: int, grid_size: float) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = _PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self, coords: Tensor, feats: Tensor, sizes: Sequence[int]
    ) -> tuple[Tensor, Tensor, list[int], Tensor]:
        batch = _batch_index(sizes, coords.device)
        feats = self.act(self.norm(self.fc(feats)))
        with torch.no_grad():
            start = _segment(coords, batch, len(sizes), "amin")  # the grid starts at each cloud
            cells = torch.floor((coords - start[batch]) / coords.new_tensor(self.grid_size))
            keys = torch.cat([batch[:, None], cells.long()], dim=1)
            unique, cluster = torch.unique(keys, dim=0, return_inverse=True)
            counts = torch.bincount(unique[:, 0], minlength=len(sizes)).tolist()
        coords = _segment(coords, cluster, len(unique), "mean")
        feats = _segment(feats, cluster, len(unique), "amax")
        return coords, feats, counts, cluster


class _Unpool(nn.Module):
    """Send every cell's feature back to its points and add the projected skip."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            _PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels),
            _PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: Tensor, skip: Tensor, cluster: Tensor) -> Tensor:
        return self.proj(feats)[cluster] + self.proj_skip(skip)


class _V2Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: int, grid_size: float, blocks: _V2Blocks):
        super().__init__()
        self.down = _GridPool(in_channels, channels, grid_size)
        self.blocks = blocks


class _V2Decoder(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, channels: int, blocks: _V2Blocks):
        super().__init__()
        self.up = _Unpool(in_channels, skip_channels, channels)
        self.blocks = blocks


class _V2PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: _V2Blocks) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, channels, bias=False),
            _PointBatchNorm(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = blocks


@BACKBONES.register()
class PointTransformerV2(Backbone):
    """Point Transformer V2 U-Net with features on every input point.

    The defaults are ``PT-v2m2`` of Pointcept with the stage sizes of its ScanNet
    config: grouped vector attention over 16 nearest neighbours, four grid poolings,
    un-pooling through the stored cell of every point. ``grid_sizes`` are metres: the
    defaults suit rooms sampled at 2 cm, a table-top cloud needs smaller cells.

    Args:
        in_channels: Feature channels per point, at least one.
        out_channels: Width of the returned features. ``None`` returns the last decoder
            stage as it is (``dec_channels[0]``); a number adds the first half of the
            official segmentation head (linear, batch norm, ReLU) with that width.
        patch_embed_depth: Attention blocks at full resolution before the first pooling.
        patch_embed_channels: Their width.
        patch_embed_groups: Their attention groups.
        patch_embed_neighbours: Their neighbours.
        enc_depths: Attention blocks of every encoder stage.
        enc_channels: Width of every encoder stage.
        enc_groups: Attention groups of every encoder stage; each divides the width.
        enc_neighbours: Neighbours of every encoder stage.
        dec_depths: Attention blocks of every decoder stage, finest first.
        dec_channels: Width of every decoder stage.
        dec_groups: Attention groups of every decoder stage.
        dec_neighbours: Neighbours of every decoder stage.
        grid_sizes: Cell edge of every pooling, in the unit of the coordinates.
        attn_qkv_bias: Bias in the query, key and value projections.
        pe_multiplier: Multiply the relation by an encoding of the offset (``PT-v2m1``).
        pe_bias: Add an encoding of the offset to the relation and the value.
        attn_drop_rate: Dropout on the attention weights.
        drop_path_rate: Largest stochastic-depth rate; it grows linearly with depth.
    """

    output = "dense"
    status = "tested"
    source = (
        "Point Transformer V2 (Wu et al., NeurIPS 2022); adapted from "
        "Pointcept/Pointcept@1342eda, MIT"
    )

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int | None = None,
        patch_embed_depth: int = 1,
        patch_embed_channels: int = 48,
        patch_embed_groups: int = 6,
        patch_embed_neighbours: int = 8,
        enc_depths: Sequence[int] = (2, 2, 6, 2),
        enc_channels: Sequence[int] = (96, 192, 384, 512),
        enc_groups: Sequence[int] = (12, 24, 48, 64),
        enc_neighbours: Sequence[int] = (16, 16, 16, 16),
        dec_depths: Sequence[int] = (1, 1, 1, 1),
        dec_channels: Sequence[int] = (48, 96, 192, 384),
        dec_groups: Sequence[int] = (6, 12, 24, 48),
        dec_neighbours: Sequence[int] = (16, 16, 16, 16),
        grid_sizes: Sequence[float] = (0.06, 0.15, 0.375, 0.9375),
        attn_qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("PointTransformerV2 needs at least one input feature channel")
        per_stage = {
            "enc_depths": enc_depths,
            "enc_channels": enc_channels,
            "enc_groups": enc_groups,
            "enc_neighbours": enc_neighbours,
            "dec_depths": dec_depths,
            "dec_channels": dec_channels,
            "dec_groups": dec_groups,
            "dec_neighbours": dec_neighbours,
            "grid_sizes": grid_sizes,
        }
        stages = len(enc_depths)
        wrong = {name: len(value) for name, value in per_stage.items() if len(value) != stages}
        if wrong or stages < 1:
            raise ValueError(
                f"PointTransformerV2: every per-stage option needs {stages} entries (the length "
                f"of enc_depths), got {wrong}"
            )
        widths = [
            (patch_embed_channels, patch_embed_groups),
            *zip(enc_channels, enc_groups),
            *zip(dec_channels, dec_groups),
        ]
        for width, groups in widths:
            if width % groups:
                raise ValueError(
                    f"PointTransformerV2: a width of {width} cannot be split into {groups} "
                    "attention groups; every width must be a multiple of its groups"
                )
        self.in_channels = in_channels
        shared = (attn_qkv_bias, pe_multiplier, pe_bias, attn_drop_rate)
        self.patch_embed = _V2PatchEmbed(
            in_channels,
            patch_embed_channels,
            _V2Blocks(
                patch_embed_depth,
                patch_embed_channels,
                patch_embed_groups,
                patch_embed_neighbours,
                [0.0] * patch_embed_depth,
                *shared,
            ),
        )
        enc_rates = torch.linspace(0, drop_path_rate, sum(enc_depths)).tolist()
        dec_rates = torch.linspace(0, drop_path_rate, sum(dec_depths)).tolist()
        enc_widths = [patch_embed_channels, *enc_channels]
        dec_widths = [*dec_channels, enc_widths[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(stages):
            rates = enc_rates[sum(enc_depths[:i]) : sum(enc_depths[: i + 1])]
            blocks = _V2Blocks(
                enc_depths[i], enc_widths[i + 1], enc_groups[i], enc_neighbours[i], rates, *shared
            )
            self.enc_stages.append(
                _V2Encoder(enc_widths[i], enc_widths[i + 1], grid_sizes[i], blocks)
            )
            rates = dec_rates[sum(dec_depths[:i]) : sum(dec_depths[: i + 1])]
            blocks = _V2Blocks(
                dec_depths[i], dec_widths[i], dec_groups[i], dec_neighbours[i], rates, *shared
            )
            self.dec_stages.append(
                _V2Decoder(dec_widths[i + 1], enc_widths[i], dec_widths[i], blocks)
            )
        if out_channels is None:
            self.seg_head: nn.Module = nn.Identity()
            self.out_channels = dec_widths[0]
        else:
            self.seg_head = nn.Sequential(
                nn.Linear(dec_widths[0], out_channels),
                _PointBatchNorm(out_channels),
                nn.ReLU(inplace=True),
            )
            self.out_channels = out_channels

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="PointTransformerV2")
        sizes = cloud_sizes(points)
        if min(sizes) < 1:
            raise ContractError(
                "PointTransformerV2 received an empty cloud; every cloud needs points"
            )
        coords = points.coords
        feats = self.patch_embed.blocks(coords, self.patch_embed.proj(feats), sizes)
        skips: list[tuple[Tensor, Tensor, list[int], Tensor]] = []
        for stage in self.enc_stages:
            pooled_coords, pooled, pooled_sizes, cluster = stage.down(coords, feats, sizes)
            skips.append((coords, feats, sizes, cluster))
            coords, sizes = pooled_coords, pooled_sizes
            feats = stage.blocks(coords, pooled, sizes)
        for stage in reversed(self.dec_stages):
            coords, skip, sizes, cluster = skips.pop()
            feats = stage.blocks(coords, stage.up(feats, skip, cluster), sizes)
        return PointFeatures(self.seg_head(feats), points)
