"""PointNeXt.

Paper: Qian, Li, Peng, Mai, Hammoud, Elhoseiny, Ghanem, "PointNeXt: Revisiting
PointNet++ with Improved Training and Scaling Strategies", NeurIPS 2022.

PointNeXt keeps the PointNet++ recipe (furthest point sampling, ball grouping, shared
MLPs, max pooling, three-neighbour propagation) and changes what the paper found to
matter: a point-wise stem, offsets normalised by the ball radius, a residual connection
in the smallest model's set abstraction, and, for the larger models, inverted residual
MLP blocks after each abstraction. This module is the encoder-decoder of the
segmentation models and returns ``dense`` features on the input points.

Sampling is by ratio: stage ``i`` keeps ``floor(n / stride)`` points of each cloud, so
clouds of different sizes stay different in size at every level. Each cloud must be
large enough to leave three points at the coarsest level for the interpolation.

Provenance. Written for this code base on the reference operators of
:mod:`grasppanda.ops`, from the paper and the behaviour of the official implementation
(``openpoints``, Guocheng Qian, MIT licence). The module tree and parameter names
(``encoder.encoder.1.0.convs.0.0``, ``decoder.decoder.0.0.convs.0.0`` ...) follow the
official ``BaseSeg`` model on purpose, so that its state dicts load once the
``head.*`` entries are dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import inverse_distance_weights, three_interpolate
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import ball_query_packed, check_features, cloud_sizes, fps_packed, knn_packed

__all__ = ["PointNeXt"]


def _block2d(inputs: int, outputs: int, *, act: bool = True) -> nn.Sequential:
    """1×1 convolution over ``(points, neighbours)`` with batch norm and optional ReLU."""
    layers: list[nn.Module] = [
        nn.Conv2d(inputs, outputs, kernel_size=1, bias=False),
        nn.BatchNorm2d(outputs),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _block1d(inputs: int, outputs: int, *, norm: bool = True, act: bool = True) -> nn.Sequential:
    """Point-wise convolution; batch norm replaces the bias when present."""
    layers: list[nn.Module] = [nn.Conv1d(inputs, outputs, kernel_size=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm1d(outputs))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _rows(module: nn.Module, feats: Tensor) -> Tensor:
    """Apply a point-wise 1-D module to ``(M, C)`` rows as one batch-norm population."""
    return module(feats.T[None])[0].T


def _groups(module: nn.Module, feats: Tensor) -> Tensor:
    """Apply a 2-D module to ``(M, K, C)`` neighbourhoods and max-pool over neighbours."""
    return module(feats.permute(2, 0, 1)[None])[0].amax(dim=-1).T


def _grouped(
    centres: Tensor,
    centre_sizes: Sequence[int],
    coords: Tensor,
    feats: Tensor,
    sizes: Sequence[int],
    radius: float,
    nsample: int,
    normalize_dp: bool,
) -> Tensor:
    """``(M, K, 3 + C)``: offsets to the neighbours, then the neighbours' features."""
    index = ball_query_packed(centres, centre_sizes, coords, sizes, radius, nsample)
    offset = coords[index] - centres[:, None, :]
    if normalize_dp:
        offset = offset / radius
    return torch.cat([offset, feats[index]], dim=-1)


class _Stem(nn.Module):
    """The first stage: a linear embedding of every point's features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(_block1d(in_channels, out_channels, norm=False, act=False))

    def forward(self, feats: Tensor) -> Tensor:
        return _rows(self.convs, feats)


class _SetAbstraction(nn.Module):
    """Subsample by a ratio, group a ball around each sample, shared MLP, max-pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        stride: int,
        radius: float,
        nsample: int,
        use_res: bool,
        normalize_dp: bool,
    ) -> None:
        super().__init__()
        self.stride, self.radius, self.nsample = stride, radius, nsample
        self.use_res, self.normalize_dp = use_res, normalize_dp
        widths = [in_channels + 3, *[out_channels // 2] * (layers - 1), out_channels]
        self.convs = nn.Sequential(
            *[
                _block2d(widths[i], widths[i + 1], act=not (use_res and i == len(widths) - 2))
                for i in range(len(widths) - 1)
            ]
        )
        if use_res:
            same = in_channels == out_channels
            self.skipconv = (
                nn.Identity()
                if same
                else _block1d(in_channels, out_channels, norm=False, act=False)
            )
            self.act = nn.ReLU(inplace=True)

    def forward(
        self, coords: Tensor, feats: Tensor, sizes: Sequence[int], presampled: bool = False
    ) -> tuple[Tensor, Tensor, list[int]]:
        """``presampled``: the clouds are furthest point samples already, see ``fps_packed``."""
        counts = [size // self.stride for size in sizes]
        index = fps_packed(coords, sizes, counts, presampled=presampled)
        centres = coords[index]
        grouped = _grouped(
            centres, counts, coords, feats, sizes, self.radius, self.nsample, self.normalize_dp
        )
        out = _groups(self.convs, grouped)
        if self.use_res:
            out = self.act(out + _rows(self.skipconv, feats[index]))
        return centres, out, counts


class _LocalAggregation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(_block2d(channels + 3, channels))


class _InvResMLP(nn.Module):
    """Local aggregation, then an inverted bottleneck MLP, with a residual connection."""

    def __init__(
        self, channels: int, expansion: int, radius: float, nsample: int, normalize_dp: bool
    ) -> None:
        super().__init__()
        self.radius, self.nsample, self.normalize_dp = radius, nsample, normalize_dp
        self.convs = _LocalAggregation(channels)
        hidden = int(channels * expansion)
        self.pwconv = nn.Sequential(
            _block1d(channels, hidden), _block1d(hidden, channels, act=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, coords: Tensor, feats: Tensor, sizes: Sequence[int]) -> Tensor:
        grouped = _grouped(
            coords, sizes, coords, feats, sizes, self.radius, self.nsample, self.normalize_dp
        )
        out = _rows(self.pwconv, _groups(self.convs.convs, grouped))
        return self.act(out + feats)


class _Propagation(nn.Module):
    """Concatenate a level's own features with those interpolated from the coarser level."""

    def __init__(self, in_channels: int, out_channels: int, layers: int) -> None:
        super().__init__()
        widths = [in_channels, *[out_channels] * layers]
        self.convs = nn.Sequential(*[_block1d(a, b) for a, b in pairwise(widths)])

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
        weight = inverse_distance_weights(distance, power=1.0)
        lifted = three_interpolate(known_feats[None], index[None], weight[None])[0]
        return _rows(self.convs, torch.cat([skip, lifted], dim=-1))


@BACKBONES.register()
class PointNeXt(Backbone):
    """PointNeXt encoder-decoder with features on every input point.

    The defaults are PointNeXt-S as published for S3DIS: 4 input channels (colour and
    height), width 32 doubling at each of four stride-4 stages, one block per stage,
    two-layer set abstraction with a residual connection, radius 0.1 m doubling per
    stage, 32 neighbours, 32 output channels. The larger published models change three
    arguments:

    ========  =================  ==========  =============
    model     ``blocks``         ``width``   ``sa_layers`` / ``sa_use_res``
    ========  =================  ==========  =============
    S         1, 1, 1, 1, 1      32          2 / true
    B         1, 2, 3, 2, 2      32          1 / false
    L         1, 3, 5, 3, 3      32          1 / false
    XL        1, 4, 7, 4, 4      64          1 / false
    ========  =================  ==========  =============

    The radius is tuned to rooms voxelised at 4 cm; tabletop clouds need a smaller one.

    Args:
        in_channels: Feature channels per point, at least one.
        width: Channels of the stem; doubled by every stage that subsamples.
        blocks: Blocks per stage, counting the stem or set abstraction as the first;
            further blocks are inverted residual MLPs.
        strides: Subsampling ratio of every stage. The first must be 1 (the stem), the
            others greater than 1.
        sa_layers: Layers of each set abstraction's shared MLP.
        sa_use_res: Residual connection around each set abstraction.
        expansion: Width multiplier inside the inverted residual MLP.
        radius: Ball radius of the first set abstraction. Each stage multiplies it by
            ``radius_scaling``; inverted residual blocks of a subsampled stage use the
            radius of the next stage.
        radius_scaling: Growth of the radius per stage.
        nsample: Neighbours gathered per ball.
        normalize_dp: Divide grouped offsets by the ball radius.
        decoder_layers: Layers of each propagation level's MLP.
    """

    output = "dense"
    status = "tested"
    source = "PointNeXt (Qian et al., NeurIPS 2022)"

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        blocks: Sequence[int] = (1, 1, 1, 1, 1),
        strides: Sequence[int] = (1, 4, 4, 4, 4),
        sa_layers: int = 2,
        sa_use_res: bool = True,
        expansion: int = 4,
        radius: float = 0.1,
        radius_scaling: float = 2.0,
        nsample: int = 32,
        normalize_dp: bool = True,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("PointNeXt needs at least one input feature channel per point")
        if len(blocks) != len(strides) or len(blocks) < 2:
            raise ValueError(
                f"PointNeXt: blocks and strides need one entry per stage (at least two), got "
                f"{len(blocks)} and {len(strides)}"
            )
        if strides[0] != 1 or any(stride < 2 for stride in strides[1:]):
            raise ValueError(
                f"PointNeXt: the first stride must be 1 (the stem keeps all points) and the "
                f"others at least 2, got {list(strides)}"
            )
        if any(count < 1 for count in blocks) or sa_layers < 1 or decoder_layers < 1:
            raise ValueError("PointNeXt: blocks, sa_layers and decoder_layers must be positive")
        self.in_channels = in_channels
        self.strides = tuple(strides)

        channels = [width * 2**index for index in range(len(strides))]
        self.encoder = nn.Module()
        stages = []
        for index, (count, stride) in enumerate(zip(blocks, strides)):
            if index == 0:
                first: nn.Module = _Stem(in_channels, channels[0])
                block_radius = radius
            else:
                first = _SetAbstraction(
                    channels[index - 1],
                    channels[index],
                    sa_layers,
                    stride,
                    radius,
                    nsample,
                    sa_use_res,
                    normalize_dp,
                )
                radius = block_radius = radius * radius_scaling
            rest = [
                _InvResMLP(channels[index], expansion, block_radius, nsample, normalize_dp)
                for _ in range(count - 1)
            ]
            stages.append(nn.Sequential(first, *rest))
        self.encoder.encoder = nn.Sequential(*stages)

        self.decoder = nn.Module()
        levels: list[nn.Module] = []
        carried = channels[-1]
        for index in reversed(range(len(strides) - 1)):
            layer = _Propagation(channels[index] + carried, channels[index], decoder_layers)
            levels.insert(0, nn.Sequential(layer))
            carried = channels[index]
        self.decoder.decoder = nn.Sequential(*levels)
        self.out_channels = channels[0]

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="PointNeXt")
        sizes = cloud_sizes(points)
        needed = 3 * math.prod(self.strides)
        if min(sizes) < needed:
            raise ContractError(
                f"PointNeXt subsamples by {'x'.join(map(str, self.strides[1:]))} and interpolates "
                f"over 3 points at the coarsest level, so every cloud needs at least {needed} "
                f"points; cloud {sizes.index(min(sizes))} of the batch has {min(sizes)}."
            )
        level_coords, level_feats, level_sizes = [], [], []
        coords = points.coords
        for index, stage in enumerate(self.encoder.encoder):
            if isinstance(stage[0], _Stem):
                feats = stage[0](feats)
            else:  # from the second abstraction on, the input is a furthest point sample
                coords, feats, sizes = stage[0](coords, feats, sizes, presampled=index > 1)
            for block in stage[1:]:
                feats = block(coords, feats, sizes)
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
