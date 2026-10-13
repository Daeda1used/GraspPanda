# Copyright (c) Guocheng Qian.
# Adapted for GraspPanda from https://github.com/guochengqian/openpoints @ bc778ec (MIT):
# the ``ASSA`` layer of ``models/layers/local_aggregation.py`` and the encoder and decoder
# of ``models/backbone/pointnetv2.py``, with the layout of ``cfgs/s3dis/assanet.yaml`` of
# https://github.com/guochengqian/PointNeXt @ b7ec10f (MIT).
"""ASSANet: PointNet++ with anisotropic separable set abstraction.

Paper: Qian, Hammoud, Li, Thabet, Ghanem, "ASSANet: An Anisotropic Separable Set
Abstraction for Efficient Point Cloud Representation Learning", NeurIPS 2021
(spotlight).

PointNet++ runs its shared MLP on every neighbour of every centre. ASSANet separates
the two: point-wise layers run once per point *before* grouping, the neighbours are
then reduced, and the remaining layers run once per centre. To keep the reduction
sensitive to direction, each neighbour feature is multiplied by its three offset
coordinates before the mean (the "anisotropic" part), so a channel becomes three. A
residual connection goes around each abstraction, and the two scales of a level run
one after the other (the second on the points the first kept).

Encoder and decoder, features on every input point: ``output = "dense"``. The compiled
samplers and groupers of the upstream are replaced by :mod:`grasppanda.ops`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import ball_query_packed, check_features, cloud_sizes, fps_packed
from .pointnext import _block1d, _Propagation, _rows

__all__ = ["ASSANet"]


class _ASSA(nn.Module):
    """Point-wise layers, anisotropic reduction over a ball, point-wise layers, residual."""

    def __init__(
        self,
        channels: Sequence[int],
        radius: float,
        nsample: int,
        normalize_dp: bool,
        reduction: str,
    ) -> None:
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp, self.reduction = normalize_dp, reduction
        widths = list(channels)
        before = math.ceil((len(widths) - 1) / 2)
        widths[before] = math.ceil(widths[before] / 3)
        self.pre = nn.Sequential(*[_block1d(widths[i], widths[i + 1]) for i in range(before)])
        skip = widths[before]
        widths[before] = 3 * skip
        last = len(widths) - 2
        self.post = nn.Sequential(
            *[
                _block1d(widths[i], widths[i + 1], act=i != last)
                for i in range(before, len(widths) - 1)
            ]
        )
        self.skip_layer = (
            nn.Identity() if skip == widths[-1] else nn.Conv1d(skip, widths[-1], 1, bias=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        centres: Tensor,
        centre_sizes: Sequence[int],
        coords: Tensor,
        feats: Tensor,
        sizes: Sequence[int],
        centre_rows: Tensor | None,
    ) -> Tensor:
        feats = _rows(self.pre, feats)
        neighbours = ball_query_packed(
            centres, centre_sizes, coords, sizes, self.radius, self.nsample
        )
        offset = coords[neighbours] - centres[:, None, :]
        if self.normalize_dp:
            offset = offset / self.radius
        # (M, K, 3, C): every neighbour feature times each of its offset coordinates.
        grouped = offset[..., None] * feats[neighbours][:, :, None, :]
        if self.reduction == "mean":
            reduced = grouped.mean(dim=1)
        elif self.reduction == "sum":
            reduced = grouped.sum(dim=1)
        else:
            reduced = grouped.amax(dim=1)
        out = _rows(self.post, reduced.flatten(1))
        identity = feats if centre_rows is None else feats[centre_rows]
        return self.act(out + _rows(self.skip_layer, identity))


class _Level(nn.Module):
    """Sample once, then one ASSA per scale, each on the points the previous one kept."""

    def __init__(
        self,
        in_channels: int,
        stride: int,
        radii: Sequence[float],
        nsamples: Sequence[int],
        mlps: Sequence[Sequence[int]],
        normalize_dp: bool,
        reduction: str,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.local_aggregations = nn.ModuleList()
        self.out_channels = 0
        for radius, nsample, mlp in zip(radii, nsamples, mlps):
            layer = _ASSA([in_channels, *mlp], radius, nsample, normalize_dp, reduction)
            self.local_aggregations.append(layer)
            in_channels = mlp[-1]
            self.out_channels += mlp[-1]

    def forward(
        self, coords: Tensor, feats: Tensor, sizes: Sequence[int], presampled: bool
    ) -> tuple[Tensor, Tensor, list[int]]:
        counts = [size // self.stride for size in sizes]
        index = fps_packed(coords, sizes, counts, presampled=presampled)
        centres = coords[index]
        outputs = []
        support, support_feats, support_sizes, rows = coords, feats, sizes, index
        for layer in self.local_aggregations:
            out = layer(centres, counts, support, support_feats, support_sizes, rows)
            outputs.append(out)
            support, support_feats, support_sizes, rows = centres, out, counts, None
        return centres, torch.cat(outputs, dim=-1), counts


@BACKBONES.register()
class ASSANet(Backbone):
    """ASSANet encoder-decoder with features on every input point.

    The defaults are ASSANet as published for S3DIS in the PointNeXt model zoo: 4 input
    channels, a 16-channel stem with one aggregation at full resolution, four stride-4
    levels with two scales each (radii 0.1 / 0.2 m doubling per level, 16 and 32
    neighbours), a four-level propagation decoder, 64 output channels; 2.4 M parameters
    with the segmentation head.

    Args:
        in_channels: Feature channels per point, at least one.
        width: Channels of the stem.
        stem_layers: Layers of the aggregation at full resolution that follows the
            stem; 0 leaves it out.
        strides: Subsampling ratio of every level.
        radii: Ball radii of every level, one per scale, in the unit of the coordinates.
        nsamples: Neighbours gathered per ball, per level and scale. With ``mean`` or
            ``sum`` a ball that holds fewer points counts its first hit several times
            (the padding rule of the ball query, as upstream).
        mlps: Layer widths per level and scale.
        fp_mlps: Layer widths of every propagation level, finest first (the order of
            the upstream config); the widths of a level must be equal.
        normalize_dp: Divide grouped offsets by the ball radius.
        reduction: How the anisotropic neighbour features are reduced.
    """

    output = "dense"
    status = "tested"
    source = (
        "ASSANet (Qian et al., NeurIPS 2021), adapted from guochengqian/openpoints@bc778ec, MIT"
    )

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 16,
        stem_layers: int = 3,
        strides: Sequence[int] = (4, 4, 4, 4),
        radii: Sequence[Sequence[float]] = ((0.1, 0.2), (0.2, 0.4), (0.4, 0.8), (0.8, 1.6)),
        nsamples: Sequence[Sequence[int]] = ((16, 32), (16, 32), (16, 32), (16, 32)),
        mlps: Sequence[Sequence[Sequence[int]]] = (
            ((16, 16, 32), (32, 32, 64)),
            ((64, 64, 128), (64, 96, 128)),
            ((128, 196, 256), (128, 196, 256)),
            ((256, 256, 512), (256, 384, 512)),
        ),
        fp_mlps: Sequence[Sequence[int]] = ((64, 64), (128, 128), (256, 256), (512, 512)),
        normalize_dp: bool = True,
        reduction: Literal["mean", "max", "sum"] = "mean",
    ) -> None:
        super().__init__()
        levels = len(strides)
        if in_channels < 1:
            raise ValueError("ASSANet needs at least one input feature channel per point")
        if not (levels == len(radii) == len(nsamples) == len(mlps) == len(fp_mlps)) or not levels:
            raise ValueError(
                f"ASSANet: strides, radii, nsamples, mlps and fp_mlps need one entry per level, "
                f"got {levels}, {len(radii)}, {len(nsamples)}, {len(mlps)}, {len(fp_mlps)}"
            )
        if any(stride < 2 for stride in strides):
            raise ValueError(f"ASSANet: every stride must be at least 2, got {list(strides)}")
        if any(len(set(mlp)) != 1 for mlp in fp_mlps):
            raise ValueError(f"ASSANet: the widths of a propagation level must be equal: {fp_mlps}")
        if reduction not in ("mean", "max", "sum"):
            raise ValueError(f"ASSANet: unknown reduction '{reduction}'")
        self.in_channels, self.strides = in_channels, tuple(strides)
        self.conv1 = _block1d(in_channels, width, norm=False, act=False)
        self.stem = (
            _ASSA([width] * (stem_layers + 1), radii[0][0], nsamples[0][0], normalize_dp, reduction)
            if stem_layers
            else None
        )
        self.SA_modules = nn.ModuleList()
        skips = [width]
        for level in range(levels):
            layer = _Level(
                skips[-1],
                strides[level],
                radii[level],
                nsamples[level],
                mlps[level],
                normalize_dp,
                reduction,
            )
            self.SA_modules.append(layer)
            skips.append(layer.out_channels)
        self.FP_modules = nn.ModuleList()
        for level in range(levels):
            carried = fp_mlps[level + 1][-1] if level + 1 < levels else skips[-1]
            self.FP_modules.append(
                _Propagation(carried + skips[level], fp_mlps[level][0], len(fp_mlps[level]))
            )
        self.out_channels = fp_mlps[0][-1]

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="ASSANet")
        sizes = cloud_sizes(points)
        needed = 3 * math.prod(self.strides)
        if min(sizes) < needed:
            raise ContractError(
                f"ASSANet subsamples by {'x'.join(map(str, self.strides))} and interpolates "
                f"over 3 points at the coarsest level, so every cloud needs at least {needed} "
                f"points; cloud {sizes.index(min(sizes))} of the batch has {min(sizes)}."
            )
        coords = points.coords
        x = _rows(self.conv1, feats)
        if self.stem is not None:
            x = self.stem(coords, sizes, coords, x, sizes, None)
        level_coords, level_feats, level_sizes = [coords], [x], [sizes]
        for index, layer in enumerate(self.SA_modules):
            coords, x, sizes = layer(coords, x, sizes, presampled=index > 0)
            level_coords.append(coords)
            level_feats.append(x)
            level_sizes.append(sizes)
        for level in reversed(range(len(self.FP_modules))):
            level_feats[level] = self.FP_modules[level](
                level_coords[level],
                level_sizes[level],
                level_feats[level],
                level_coords[level + 1],
                level_sizes[level + 1],
                level_feats[level + 1],
            )
        return PointFeatures(level_feats[0], points)
