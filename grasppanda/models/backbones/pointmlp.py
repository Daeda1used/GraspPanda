# Copyright the authors of pointMLP-pytorch (Xu Ma et al.). The upstream licence file is
# the unmodified Apache-2.0 text and names no copyright holder.
# Adapted for GraspPanda from https://github.com/ma-xu/pointMLP-pytorch @ 1ab7f04
# (Apache-2.0): ``part_segmentation/model/pointMLP.py``. Changes made here: packed
# batches of clouds of any size, anchors by ratio, the operators of grasppanda.ops, no
# object category token, no classifier, no batch norm on the per-cloud context vector.
"""PointMLP, dense (segmentation) form.

Paper: Ma, Qin, You, Ran, Fu, "Rethinking Network Design and Local Geometry in Point
Cloud: A Simple Residual MLP Framework", ICLR 2022.

A hierarchy of residual point-wise MLPs without any hand-made local operator. Each
stage samples anchors with furthest point sampling, gathers the ``k`` nearest
neighbours of every anchor, normalises the group with the *geometric affine* module
(subtract the anchor, divide by one standard deviation taken over the whole cloud, then
a learned scale and shift per channel), runs residual MLPs before and after a max-pool.
A propagation decoder with residual MLPs returns to the input points, and a max-pooled
context vector of every level is appended, as in the part-segmentation network of the
official repository, the only dense form the authors published.

The authors published it for single objects of 2048 points. On a scene it is an
experimental encoder: FlexLoG reports a PointMLP-style network as a *region* encoder for
grasp detection; no scene-level grasp result is known.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import check_features, cloud_sizes, fps_packed, knn_packed

__all__ = ["PointMLP"]


class _Layer(nn.Module):
    """Linear layer and batch norm on the last axis, optional ReLU (a 1x1 convolution)."""

    def __init__(self, inputs: int, outputs: int, *, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Linear(inputs, outputs)
        self.norm = nn.BatchNorm1d(outputs)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(self.conv(x).flatten(0, -2)).reshape(*x.shape[:-1], -1)
        return self.act(out)


class _Residual(nn.Module):
    """``ConvBNReLURes1D``: two layers and a skip connection."""

    def __init__(self, channels: int, expansion: float) -> None:
        super().__init__()
        self.net1 = _Layer(channels, int(channels * expansion))
        self.net2 = _Layer(int(channels * expansion), channels, act=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.net2(self.net1(x)) + x)


class _Stage(nn.Module):
    """Sample anchors, group neighbours with the geometric affine, residual MLPs, max-pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reducer: int,
        k: int,
        pre_blocks: int,
        pos_blocks: int,
        expansion: float,
    ) -> None:
        super().__init__()
        self.reducer, self.k = reducer, k
        self.affine_alpha = nn.Parameter(torch.ones(in_channels + 3))
        self.affine_beta = nn.Parameter(torch.zeros(in_channels + 3))
        self.transfer = _Layer(2 * in_channels + 3, out_channels)
        self.pre = nn.Sequential(*[_Residual(out_channels, expansion) for _ in range(pre_blocks)])
        self.pos = nn.Sequential(*[_Residual(out_channels, expansion) for _ in range(pos_blocks)])

    def forward(
        self, coords: Tensor, feats: Tensor, sizes: Sequence[int], presampled: bool
    ) -> tuple[Tensor, Tensor, list[int]]:
        counts = [size // self.reducer for size in sizes]
        index = fps_packed(coords, sizes, counts, presampled=presampled)
        anchors, anchor_feats = coords[index], feats[index]
        neighbours = knn_packed(anchors, counts, coords, sizes, self.k)[1]
        grouped = torch.cat([feats[neighbours], coords[neighbours]], dim=-1)
        grouped = grouped - torch.cat([anchor_feats, anchors], dim=-1)[:, None, :]
        # One standard deviation per cloud, over all its groups and channels.
        scale, start = [], 0
        for count in counts:
            std = grouped[start : start + count].std()
            scale.append(std.expand(count))
            start += count
        grouped = grouped / (torch.cat(scale)[:, None, None] + 1e-5)
        grouped = self.affine_alpha * grouped + self.affine_beta
        repeated = anchor_feats[:, None, :].expand(-1, self.k, -1)
        out = self.pre(self.transfer(torch.cat([grouped, repeated], dim=-1))).amax(dim=1)
        return anchors, self.pos(out), counts


class _Propagation(nn.Module):
    """Three-neighbour interpolation (inverse squared distance), fuse, residual MLPs."""

    def __init__(self, inputs: int, outputs: int, blocks: int, expansion: float) -> None:
        super().__init__()
        self.fuse = _Layer(inputs, outputs)
        self.extraction = nn.Sequential(*[_Residual(outputs, expansion) for _ in range(blocks)])

    def forward(
        self,
        coords: Tensor,
        sizes: Sequence[int],
        skip: Tensor,
        known_coords: Tensor,
        known_sizes: Sequence[int],
        known_feats: Tensor,
    ) -> Tensor:
        distance, index = knn_packed(coords, sizes, known_coords, known_sizes, 3)
        weight = 1.0 / (distance.pow(2) + 1e-8)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        lifted = (known_feats[index] * weight[..., None]).sum(dim=1)
        return self.extraction(self.fuse(torch.cat([skip, lifted], dim=-1)))


@BACKBONES.register()
class PointMLP(Backbone):
    """PointMLP encoder-decoder with features on every input point.

    The defaults are the part-segmentation network of the official repository
    (``pointMLP``): 64-channel embedding, four stages that keep a quarter of the points
    and double the width, 32 neighbours, two residual blocks before and after each
    pooling, decoder widths 512 / 256 / 128 / 128 with four blocks each, a 64-channel
    context vector, 128 output channels.

    Args:
        in_channels: Feature channels per point besides the coordinates, which the
            network always reads as its first three input channels (the official model
            reads coordinates and normals).
        embed_dim: Channels of the point-wise embedding.
        dim_expansion: Width multiplier of every stage.
        pre_blocks: Residual blocks before the max-pool of every stage.
        pos_blocks: Residual blocks after the max-pool of every stage.
        k_neighbors: Neighbours per anchor of every stage.
        reducers: Subsampling ratio of every stage (the official code fixes anchor
            counts for 2048-point objects; a ratio serves clouds of any size).
        de_dims: Width of every decoder level, coarsest first.
        de_blocks: Residual blocks of every decoder level.
        res_expansion: Width multiplier inside a residual block.
        gmp_dim: Channels of the context vector: every level is max-pooled over the
            cloud, mapped to this width, and the fused vector is appended to all points
            of the cloud. It ties features to the workspace crop; 0 leaves it out.
        out_channels: Channels of the returned features.
    """

    output = "dense"
    status = "tested"
    source = (
        "PointMLP (Ma et al., ICLR 2022), dense form, adapted from "
        "ma-xu/pointMLP-pytorch@1ab7f04, Apache-2.0"
    )

    def __init__(
        self,
        in_channels: int = 0,
        embed_dim: int = 64,
        dim_expansion: Sequence[int] = (2, 2, 2, 2),
        pre_blocks: Sequence[int] = (2, 2, 2, 2),
        pos_blocks: Sequence[int] = (2, 2, 2, 2),
        k_neighbors: Sequence[int] = (32, 32, 32, 32),
        reducers: Sequence[int] = (4, 4, 4, 4),
        de_dims: Sequence[int] = (512, 256, 128, 128),
        de_blocks: Sequence[int] = (4, 4, 4, 4),
        res_expansion: float = 1.0,
        gmp_dim: int = 64,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        stages = len(reducers)
        lengths = {
            len(dim_expansion), len(pre_blocks), len(pos_blocks), len(k_neighbors),
            len(de_dims), len(de_blocks),
        }  # fmt: skip
        if lengths != {stages} or stages == 0:
            raise ValueError(
                "PointMLP: dim_expansion, pre_blocks, pos_blocks, k_neighbors, reducers, "
                f"de_dims and de_blocks need one entry per stage; reducers has {stages}"
            )
        if any(reducer < 2 for reducer in reducers):
            raise ValueError(f"PointMLP: every reducer must be at least 2, got {list(reducers)}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.reducers, self.k_neighbors, self.gmp_dim = tuple(reducers), tuple(k_neighbors), gmp_dim
        self.embedding = _Layer(3 + in_channels, embed_dim)
        widths = [embed_dim]
        self.stages = nn.ModuleList()
        for index in range(stages):
            widths.append(widths[-1] * dim_expansion[index])
            self.stages.append(
                _Stage(
                    widths[-2],
                    widths[-1],
                    reducers[index],
                    k_neighbors[index],
                    pre_blocks[index],
                    pos_blocks[index],
                    res_expansion,
                )
            )
        self.decode_list = nn.ModuleList()
        carried = widths[-1]
        for index in range(stages):
            skip = widths[stages - 1 - index]
            self.decode_list.append(
                _Propagation(carried + skip, de_dims[index], de_blocks[index], res_expansion)
            )
            carried = de_dims[index]
        if gmp_dim:
            self.gmp_map_list = nn.ModuleList(_Layer(width, gmp_dim) for width in widths)
            # One row per cloud: no batch norm, so that a batch of one frame can train.
            self.gmp_map_end = nn.Sequential(
                nn.Linear(gmp_dim * len(widths), gmp_dim), nn.ReLU(inplace=True)
            )
        self.head = _Layer(carried + gmp_dim, out_channels)

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="PointMLP")
        sizes = cloud_sizes(points)
        needed = 3 * math.prod(self.reducers)
        for index, k in enumerate(self.k_neighbors):
            needed = max(needed, k * math.prod(self.reducers[:index]))
        if min(sizes) < needed:
            raise ContractError(
                f"PointMLP subsamples by {'x'.join(map(str, self.reducers))}, gathers "
                f"{max(self.k_neighbors)} neighbours and interpolates over 3 points at the "
                f"coarsest level, so every cloud needs at least {needed} points; cloud "
                f"{sizes.index(min(sizes))} of the batch has {min(sizes)}."
            )
        coords = points.coords
        x = coords if feats is None else torch.cat([coords, feats], dim=-1)
        x = self.embedding(x)
        level_coords, level_feats, level_sizes = [coords], [x], [sizes]
        for index, stage in enumerate(self.stages):
            coords, x, sizes = stage(coords, x, sizes, presampled=index > 0)
            level_coords.append(coords)
            level_feats.append(x)
            level_sizes.append(sizes)
        for index, layer in enumerate(self.decode_list):
            target = len(self.stages) - 1 - index
            x = layer(
                level_coords[target],
                level_sizes[target],
                level_feats[target],
                level_coords[target + 1],
                level_sizes[target + 1],
                x,
            )
        if self.gmp_dim:
            batch = len(level_sizes[0])
            pooled = []
            for mapper, values, counts in zip(self.gmp_map_list, level_feats, level_sizes):
                mapped = mapper(values)
                owner = torch.repeat_interleave(
                    torch.arange(batch, device=mapped.device),
                    torch.as_tensor(counts, device=mapped.device),
                )
                pooled.append(
                    mapped.new_zeros(batch, self.gmp_dim).scatter_reduce(
                        0, owner[:, None].expand_as(mapped), mapped, "amax", include_self=False
                    )
                )
            context = self.gmp_map_end(torch.cat(pooled, dim=-1))
            x = torch.cat([x, context[points.batch_index]], dim=-1)
        return PointFeatures(self.head(x), points)
