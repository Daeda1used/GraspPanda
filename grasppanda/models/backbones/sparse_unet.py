"""Sparse residual U-Net: the scene encoder of the Graspness family.

``SparseUNet14`` voxelises the cloud, runs a four-level encoder-decoder with skip
connections and hands every *input point* the feature of its voxel. Its default widths
make it the architecture the Graspness reference calls ``MinkUNet14D`` (and
EconomicGrasp ``TDUnet``): a 5³ stem, four stride-2 stages of one residual block each
(32-64-128-256), four up-sampling stages (192 each) that concatenate the encoder
feature of the same resolution, and a linear projection to ``out_channels``.

It is written against :mod:`grasppanda.ops.sparse`, not against a sparse library.
``tools/convert_checkpoint.py`` maps a MinkowskiEngine checkpoint of the reference
network onto this module; ``tests/integration/test_gsnet_parity.py`` compares the two
level by level.

The voxel size lives here and nowhere else (``docs/design/ARCHITECTURE.md`` §8.6).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import sparse, voxelize
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

__all__ = ["SparseUNet14", "VoxelMap"]


@dataclass(eq=False)
class VoxelMap:
    """The voxelisation of one point set, and the only way back from it.

    Attributes:
        voxels: ``(V, 4)`` occupied ``[batch, x, y, z]`` voxels.
        inverse: ``(N,)`` voxel row of every point.
        first: ``(V,)`` first point of every voxel.
        voxel_size: Edge length in metres.
        source: The point set that was voxelised.
    """

    voxels: Tensor
    inverse: Tensor
    first: Tensor
    voxel_size: float
    source: PointSet

    @classmethod
    def build(cls, points: PointSet, voxel_size: float) -> VoxelMap:
        grid = voxelize(points.coords, voxel_size, points.batch_index)
        return cls(grid.voxels, grid.inverse, grid.first, voxel_size, points)

    def reduce(self, feats: Tensor, rule: Literal["first", "mean"]) -> Tensor:
        """``(V, C)`` voxel features from ``(N, C)`` point features."""
        if rule == "first":
            return feats[self.first]
        total = feats.new_zeros(len(self.voxels), feats.shape[1]).index_add_(0, self.inverse, feats)
        count = torch.bincount(self.inverse, minlength=len(self.voxels)).to(feats.dtype)
        return total / count[:, None]

    def devoxelize(self, voxel_feats: Tensor, points: PointSet) -> PointFeatures:
        """Per-point features on ``points``, which must be the set that was voxelised."""
        if points.id != self.source.id:
            raise ContractError(
                f"this voxel map was built from point set #{self.source.id} and cannot map "
                f"features onto point set #{points.id}; voxelise that set instead"
            )
        if voxel_feats.shape[0] != len(self.voxels):
            raise ContractError(
                f"expected one feature row per voxel ({len(self.voxels)}), got {voxel_feats.shape[0]}"
            )
        return PointFeatures(voxel_feats[self.inverse], points)


class _ConvNormAct(nn.Module):
    """A sparse convolution followed by batch norm and ReLU."""

    def __init__(self, conv: nn.Module, channels: int) -> None:
        super().__init__()
        self.conv = conv
        self.norm = sparse.BatchNorm(channels)
        self.act = sparse.ReLU()

    def forward(self, x: Any) -> Any:
        return self.act(self.norm(self.conv(x)))


class _Shortcut(nn.Module):
    """Projection of the residual branch when the channel count changes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = sparse.Linear(in_channels, out_channels, bias=False)
        self.norm = sparse.BatchNorm(out_channels)

    def forward(self, x: Any) -> Any:
        return self.norm(self.conv(x))


class _BasicBlock(nn.Module):
    """Two 3³ submanifold convolutions with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, key: str) -> None:
        super().__init__()
        self.conv1 = sparse.SubMConv3d(in_channels, out_channels, 3, key=key)
        self.norm1 = sparse.BatchNorm(out_channels)
        self.conv2 = sparse.SubMConv3d(out_channels, out_channels, 3, key=key)
        self.norm2 = sparse.BatchNorm(out_channels)
        self.act = sparse.ReLU()
        self.shortcut = (
            _Shortcut(in_channels, out_channels) if in_channels != out_channels else None
        )

    def forward(self, x: Any) -> Any:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        residual = x if self.shortcut is None else self.shortcut(x)
        summed = sparse.features(out) + sparse.features(residual)
        return self.act(sparse.replace_features(out, summed))


@BACKBONES.register()
class SparseUNet14(Backbone):
    """Sparse residual U-Net returning one feature per input point.

    Args:
        in_channels: Channels of the per-point input features.
        out_channels: Channels of the returned per-point features.
        voxel_size: Edge length of a voxel in metres. Points are quantised by
            ``floor(xyz / voxel_size)``.
        stem_channels: Width of the full-resolution stem.
        encoder_channels: Width after each of the four down-sampling stages.
        decoder_channels: Width after each of the four up-sampling stages.
        stem_kernel: Edge length of the stem kernel.
        voxel_features: How points sharing a voxel are reduced to one input feature:
            ``first`` takes the first point's (what the reference collate does; any rule
            is equivalent for constant features), ``mean`` averages.
    """

    output = "dense"
    status = "tested"
    source = "MinkUNet14D (Choy et al., CVPR 2019), as GSNet, EconomicGrasp and FineGrasp use it"
    extras = ("spconv",)
    #: Each stage halves the resolution.
    stage_stride = 2

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int:
        return options["out_channels"]

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        voxel_size: float = 0.005,
        stem_channels: int = 32,
        encoder_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        decoder_channels: tuple[int, int, int, int] = (192, 192, 192, 192),
        stem_kernel: int = 5,
        voxel_features: Literal["first", "mean"] = "first",
    ) -> None:
        super().__init__()
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if len(encoder_channels) != 4 or len(decoder_channels) != 4:
            raise ValueError(
                "SparseUNet14 has four down-sampling and four up-sampling stages; got "
                f"encoder_channels={list(encoder_channels)}, decoder_channels={list(decoder_channels)}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.voxel_features = voxel_features
        self.total_stride = self.stage_stride ** len(encoder_channels)

        self.stem = _ConvNormAct(
            sparse.SubMConv3d(in_channels, stem_channels, stem_kernel, key="stem"), stem_channels
        )
        # Level 0 is full resolution; down-sampling stage ``level`` goes from level - 1 to
        # level. Blocks of one level share a key: they see the same voxels, in the encoder
        # and again in the decoder once the up-sampling has restored them.
        skips = [stem_channels, *encoder_channels[:-1]]
        self.down = nn.ModuleList()
        self.encoder = nn.ModuleList()
        width = stem_channels
        for level, channels in enumerate(encoder_channels, start=1):
            conv = sparse.SparseConv3d(width, width, 2, self.stage_stride, key=f"down{level}")
            self.down.append(_ConvNormAct(conv, width))
            self.encoder.append(_BasicBlock(width, channels, key=f"level{level}"))
            width = channels
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for stage, channels in enumerate(decoder_channels):
            level = len(encoder_channels) - stage  # the level this stage leaves
            conv = sparse.SparseInverseConv3d(width, channels, 2, key=f"down{level}")
            self.up.append(_ConvNormAct(conv, channels))
            self.decoder.append(
                _BasicBlock(channels + skips[level - 1], channels, key=f"level{level - 1}")
            )
            width = channels
        self.project = sparse.Linear(width, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise like the reference network, layer type by layer type.

        Convolutions: normal, ``std = sqrt(2 / (out_channels · kernel_volume))``.
        Up-sampling convolutions: uniform within ``±1 / sqrt(out_channels · kernel_volume)``.
        Linear layers (the 1×1 projections): normal with ``std = sqrt(2 / in_channels)``;
        the reference initialiser reads the fan of a volume-one kernel from its input
        side. Biases: uniform within ``±1 / sqrt(in_channels)``. Batch norm: identity.
        """
        for module in self.modules():
            if isinstance(module, (sparse.SubMConv3d, sparse.SparseConv3d)):
                out_channels, volume = module.weight.shape[0], module.weight[0, ..., 0].numel()
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / (out_channels * volume)))
            elif isinstance(module, sparse.SparseInverseConv3d):
                out_channels, volume = module.weight.shape[0], module.weight[0, ..., 0].numel()
                bound = 1.0 / math.sqrt(out_channels * volume)
                nn.init.uniform_(module.weight, -bound, bound)
            elif isinstance(module, sparse.Linear):
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / module.in_features))
                if module.bias is not None:
                    bound = 1.0 / math.sqrt(module.in_features)
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, sparse.BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        name = type(self).__name__
        if len(points) == 0:
            raise ContractError(f"{name} received an empty point set")
        if feats.ndim != 2 or tuple(feats.shape) != (len(points), self.in_channels):
            raise ContractError(
                f"{name} expects input features of shape ({len(points)}, {self.in_channels}) "
                f"(one row per point, in_channels={self.in_channels}), got {tuple(feats.shape)}"
            )
        voxel_map = VoxelMap.build(points, self.voxel_size)
        x, _ = sparse.make_tensor(
            voxel_map.reduce(feats, self.voxel_features),
            voxel_map.voxels,
            points.batch_size,
            align=self.total_stride,
        )
        x = self.stem(x)
        skips = [x]
        for down, block in zip(self.down, self.encoder):
            x = block(down(x))
            skips.append(x)
        skips.pop()  # the coarsest level is the bottleneck, not a skip
        for up, block in zip(self.up, self.decoder):
            x = block(sparse.cat([up(x), skips.pop()]))
        return voxel_map.devoxelize(sparse.features(self.project(x)), points)
