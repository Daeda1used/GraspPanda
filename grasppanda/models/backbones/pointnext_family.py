# Copyright (c) Guocheng Qian.
# Adapted for GraspPanda from https://github.com/linhaojia13/PointMetaBase @ c364a67 (MIT),
# ``openpoints/models/backbone/pointmetabase.py``, and from
# https://github.com/guochengqian/openpoints @ bc778ec (MIT),
# ``models/backbone/pointvector.py`` ("Official implementation of PointVector",
# Xin Deng, WenYu Zhang, Qing Ding, XinMing Zhang).
"""PointMetaBase and PointVector: two encoders of the PointNeXt family.

Papers: Lin et al., "Meta Architecture for Point Cloud Analysis", CVPR 2023
(PointMetaBase); Deng et al., "PointVector: A Vector Representation In Point Cloud
Analysis", CVPR 2023.

Both keep the PointNeXt skeleton of :mod:`.pointnext` (a point-wise stem, stages that
subsample by a ratio with furthest point sampling and group a ball, a three-neighbour
propagation decoder, features on every input point) and change what happens inside a
stage:

* **PointMetaBase** embeds the neighbour offsets once per stage with a small MLP and
  *adds* that embedding to the neighbour features before the max-pool, where PointNeXt
  concatenates raw offsets in every block. Its blocks are a point-wise layer, the pooled
  sum, and a point-wise MLP of expansion 1: much cheaper than PointNeXt at equal depth.
* **PointVector** replaces the max-pool of a block by a vector representation: every
  channel of a neighbour predicts two angles and a length, which make a 3-vector; the
  vectors are summed over 8 neighbours and a grouped convolution turns each channel's
  3-vector back into a scalar. This is the network the official code builds for scene
  segmentation (its ``flag: 1``).

The compiled samplers and groupers of the upstream are replaced by the reference
operators of :mod:`grasppanda.ops`; one ball query per stage serves all blocks of the
stage, which see the same neighbourhoods upstream too.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import ball_query_packed, check_features, cloud_sizes, fps_packed
from .pointnext import _block1d, _block2d, _Propagation, _rows, _SetAbstraction, _Stem

__all__ = ["PointMetaBase", "PointVector"]


def _conv_rows(module: nn.Module, grouped: Tensor) -> Tensor:
    """Apply a 2-D module to ``(M, K, C)`` neighbourhoods and return ``(M, K, C_out)``."""
    return module(grouped.permute(2, 0, 1)[None])[0].permute(1, 2, 0)


class _MetaSetAbstraction(nn.Module):
    """Point-wise layer, then subsample, group a ball, add the offset embedding, max-pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        radius: float,
        nsample: int,
        normalize_dp: bool,
    ) -> None:
        super().__init__()
        self.stride, self.radius, self.nsample = stride, radius, nsample
        self.normalize_dp = normalize_dp
        self.convs1 = nn.Sequential(_block1d(in_channels, out_channels))
        self.convs2 = nn.Sequential(_block2d(3, out_channels))

    def forward(
        self, coords: Tensor, feats: Tensor, sizes: Sequence[int], presampled: bool = False
    ) -> tuple[Tensor, Tensor, list[int]]:
        counts = [size // self.stride for size in sizes]
        index = fps_packed(coords, sizes, counts, presampled=presampled)
        centres = coords[index]
        feats = _rows(self.convs1, feats)
        neighbours = ball_query_packed(centres, counts, coords, sizes, self.radius, self.nsample)
        offset = coords[neighbours] - centres[:, None, :]
        if self.normalize_dp:
            offset = offset / self.radius
        out = (_conv_rows(self.convs2, offset) + feats[neighbours]).amax(dim=1)
        return centres, out, counts


class _MetaBlock(nn.Module):
    """PointMetaBase block: point-wise layer, pooled sum with the stage embedding, MLP."""

    def __init__(self, channels: int, expansion: int) -> None:
        super().__init__()
        self.convs = nn.Module()
        self.convs.convs1 = nn.Sequential(_block1d(channels, channels))
        hidden = int(channels * expansion)
        self.pwconv = nn.Sequential(
            _block1d(channels, hidden), _block1d(hidden, channels, act=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, feats: Tensor, neighbours: Tensor, embedding: Tensor) -> Tensor:
        out = (_rows(self.convs.convs1, feats)[neighbours] + embedding).amax(dim=1)
        return self.act(_rows(self.pwconv, out) + feats)


class _VPSA(nn.Module):
    """PointVector block ("vector-oriented point set abstraction"), scene form."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.theta_x_alpha = _block2d(channels, channels)
        self.theta_x_beta = _block2d(channels, channels)
        self.z_x = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.tf_zx = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.result = _block1d(channels, channels, act=False)
        self.pos_x = _block2d(3, channels, act=False)
        self.residual = _block1d(channels, channels, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats: Tensor, neighbours: Tensor, offset: Tensor) -> Tensor:
        # (1, C, M, K): the layout the 2-D layers and their batch norms expect.
        relative = (feats[neighbours] - feats[:, None, :]).permute(2, 0, 1)[None]
        position = self.pos_x(offset.permute(2, 0, 1)[None])
        y = self.relu(relative + position)
        alpha, beta, length = self.theta_x_alpha(y), self.theta_x_beta(y), self.z_x(y)
        sin_beta = torch.sin(beta)
        vectors = torch.stack(
            [
                length * sin_beta * torch.cos(alpha),
                length * sin_beta * torch.sin(alpha),
                length * torch.cos(beta),
            ],
            dim=2,
        )  # (1, C, 3, M, K): the three components of a channel are neighbours for the groups
        summed = vectors.sum(dim=-1).reshape(1, 3 * self.channels, -1)
        out = self.result(self.tf_zx(summed))[0].T
        return self.relu(out + _rows(self.residual, feats))


class _Family(Backbone):
    """The shared skeleton: stem, subsampling stages with blocks, propagation decoder."""

    output = "dense"

    def _check(
        self, in_channels: int, blocks: Sequence[int], strides: Sequence[int], decoder_layers: int
    ) -> None:
        name = type(self).__name__
        if in_channels < 1:
            raise ValueError(f"{name} needs at least one input feature channel per point")
        if len(blocks) != len(strides) or len(blocks) < 2:
            raise ValueError(
                f"{name}: blocks and strides need one entry per stage (at least two), got "
                f"{len(blocks)} and {len(strides)}"
            )
        if strides[0] != 1 or any(stride < 2 for stride in strides[1:]):
            raise ValueError(
                f"{name}: the first stride must be 1 (the stem keeps all points) and the "
                f"others at least 2, got {list(strides)}"
            )
        if blocks[0] != 1 or any(count < 1 for count in blocks) or decoder_layers < 1:
            raise ValueError(
                f"{name}: the stem stage has exactly one block and every other stage at "
                f"least one, got blocks {list(blocks)}"
            )

    def _decoder(self, channels: Sequence[int], decoder_layers: int) -> None:
        self.decoder = nn.Module()
        levels: list[nn.Module] = []
        carried = channels[-1]
        for index in reversed(range(len(channels) - 1)):
            layer = _Propagation(channels[index] + carried, channels[index], decoder_layers)
            levels.insert(0, nn.Sequential(layer))
            carried = channels[index]
        self.decoder.decoder = nn.Sequential(*levels)
        self.out_channels = channels[0]

    def _blocks(
        self, index: int, stage: nn.Sequential, coords: Tensor, feats: Tensor, sizes: Sequence[int]
    ) -> Tensor:
        raise NotImplementedError

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        sizes = cloud_sizes(points)
        needed = 3 * math.prod(self.strides)
        if min(sizes) < needed:
            raise ContractError(
                f"{name} subsamples by {'x'.join(map(str, self.strides[1:]))} and interpolates "
                f"over 3 points at the coarsest level, so every cloud needs at least {needed} "
                f"points; cloud {sizes.index(min(sizes))} of the batch has {min(sizes)}."
            )
        level_coords, level_feats, level_sizes = [], [], []
        coords = points.coords
        for index, stage in enumerate(self.encoder.encoder):
            if index == 0:
                feats = stage[0](feats)
            else:
                coords, feats, sizes = stage[0](coords, feats, sizes, presampled=index > 1)
                if len(stage) > 1:
                    feats = self._blocks(index, stage, coords, feats, sizes)
            level_coords.append(coords)
            level_feats.append(feats)
            level_sizes.append(sizes)
        feats = level_feats[-1]
        for index in reversed(range(len(self.decoder.decoder))):
            feats = self.decoder.decoder[index][0](
                level_coords[index],
                level_sizes[index],
                level_feats[index],
                level_coords[index + 1],
                level_sizes[index + 1],
                feats,
            )
        return PointFeatures(feats, points)


@BACKBONES.register()
class PointMetaBase(_Family):
    """PointMetaBase encoder-decoder with features on every input point.

    The defaults are PointMetaBase-L as published for S3DIS: 4 input channels, width 32
    doubling at each of four stride-4 stages, blocks 1-3-5-3-3 with expansion 1, radius
    0.1 m doubling per stage, 32 neighbours, 32 output channels (2.7 M parameters with
    the segmentation head). The published XL is ``blocks: [1, 4, 7, 4, 4], width: 64``,
    XXL is ``blocks: [1, 5, 9, 5, 5], width: 64`` (15.3 M and 19.7 M parameters).

    The radius is tuned to rooms voxelised at 4 cm; tabletop clouds need a smaller one
    (preset ``pointmetabase``).

    Args:
        in_channels: Feature channels per point, at least one.
        width: Channels of the stem; doubled by every stage.
        blocks: Blocks per stage, counting the stem or set abstraction as the first.
        strides: Subsampling ratio of every stage; the first must be 1.
        expansion: Width multiplier inside a block's point-wise MLP.
        radius: Ball radius of the first set abstraction. Each stage multiplies it by
            ``radius_scaling``; the blocks of a stage use the radius of the next stage.
        radius_scaling: Growth of the radius per stage.
        nsample: Neighbours gathered per ball.
        normalize_dp: Divide grouped offsets by the ball radius.
        decoder_layers: Layers of each propagation level's MLP.
    """

    status = "tested"
    source = (
        "PointMetaBase (Lin et al., CVPR 2023), adapted from linhaojia13/PointMetaBase@c364a67, MIT"
    )

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        blocks: Sequence[int] = (1, 3, 5, 3, 3),
        strides: Sequence[int] = (1, 4, 4, 4, 4),
        expansion: int = 1,
        radius: float = 0.1,
        radius_scaling: float = 2.0,
        nsample: int = 32,
        normalize_dp: bool = True,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self._check(in_channels, blocks, strides, decoder_layers)
        self.in_channels, self.strides = in_channels, tuple(strides)
        self.nsample, self.normalize_dp = nsample, normalize_dp
        channels = [width * 2**index for index in range(len(strides))]
        self.encoder = nn.Module()
        stages: list[nn.Module] = [nn.Sequential(_Stem(in_channels, channels[0]))]
        self.pe_encoder = nn.ModuleList([nn.Identity()])
        self.block_radii = [radius]
        for index in range(1, len(strides)):
            first = _MetaSetAbstraction(
                channels[index - 1], channels[index], strides[index], radius, nsample, normalize_dp
            )
            radius = radius * radius_scaling
            self.block_radii.append(radius)
            rest = [_MetaBlock(channels[index], expansion) for _ in range(blocks[index] - 1)]
            stages.append(nn.Sequential(first, *rest))
            self.pe_encoder.append(
                nn.Sequential(_block2d(3, channels[index])) if rest else nn.Identity()
            )
        self.encoder.encoder = nn.Sequential(*stages)
        self._decoder(channels, decoder_layers)

    def _blocks(
        self, index: int, stage: nn.Sequential, coords: Tensor, feats: Tensor, sizes: Sequence[int]
    ) -> Tensor:
        radius = self.block_radii[index]
        neighbours = ball_query_packed(coords, sizes, coords, sizes, radius, self.nsample)
        offset = coords[neighbours] - coords[:, None, :]
        if self.normalize_dp:
            offset = offset / radius
        embedding = _conv_rows(self.pe_encoder[index], offset)
        for block in stage[1:]:
            feats = block(feats, neighbours, embedding)
        return feats


@BACKBONES.register()
class PointVector(_Family):
    """PointVector encoder-decoder (scene form) with features on every input point.

    The defaults are PointVector-L as published for S3DIS: 4 input channels, width 32
    doubling at each of four stride-4 stages, blocks 1-3-5-3-3, radius 0.1 m doubling
    per stage, 32 neighbours in the set abstractions and 8 in the vector blocks, 32
    output channels. The published XL is ``blocks: [1, 4, 7, 4, 4], width: 64``.

    The official code builds a different network per dataset (``flag``); this is the
    one for scene segmentation: sum over neighbours, ReLU, no extra batch norm.

    Args:
        in_channels: Feature channels per point, at least one.
        width: Channels of the stem; doubled by every stage.
        blocks: Blocks per stage, counting the stem or set abstraction as the first;
            further blocks are vector blocks.
        strides: Subsampling ratio of every stage; the first must be 1.
        radius: Ball radius of the first set abstraction. Each stage multiplies it by
            ``radius_scaling``; the blocks of a stage use the radius of the next stage.
        radius_scaling: Growth of the radius per stage.
        nsample: Neighbours gathered per ball by the set abstractions.
        block_nsample: Neighbours of a vector block (the official code fixes 8).
        normalize_dp: Divide grouped offsets by the ball radius.
        decoder_layers: Layers of each propagation level's MLP.
    """

    status = "tested"
    source = (
        "PointVector (Deng et al., CVPR 2023), adapted from guochengqian/openpoints@bc778ec, MIT"
    )

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        blocks: Sequence[int] = (1, 3, 5, 3, 3),
        strides: Sequence[int] = (1, 4, 4, 4, 4),
        radius: float = 0.1,
        radius_scaling: float = 2.0,
        nsample: int = 32,
        block_nsample: int = 8,
        normalize_dp: bool = True,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self._check(in_channels, blocks, strides, decoder_layers)
        self.in_channels, self.strides = in_channels, tuple(strides)
        self.block_nsample, self.normalize_dp = block_nsample, normalize_dp
        channels = [width * 2**index for index in range(len(strides))]
        self.encoder = nn.Module()
        stages: list[nn.Module] = [nn.Sequential(_Stem(in_channels, channels[0]))]
        self.block_radii = [radius]
        for index in range(1, len(strides)):
            first = _SetAbstraction(
                channels[index - 1],
                channels[index],
                1,
                strides[index],
                radius,
                nsample,
                False,
                normalize_dp,
            )
            radius = radius * radius_scaling
            self.block_radii.append(radius)
            rest = [_VPSA(channels[index]) for _ in range(blocks[index] - 1)]
            stages.append(nn.Sequential(first, *rest))
        self.encoder.encoder = nn.Sequential(*stages)
        self._decoder(channels, decoder_layers)

    def _blocks(
        self, index: int, stage: nn.Sequential, coords: Tensor, feats: Tensor, sizes: Sequence[int]
    ) -> Tensor:
        radius = self.block_radii[index]
        neighbours = ball_query_packed(coords, sizes, coords, sizes, radius, self.block_nsample)
        offset = coords[neighbours] - coords[:, None, :]
        if self.normalize_dp:
            offset = offset / radius
        for block in stage[1:]:
            feats = block(feats, neighbours, offset)
        return feats
