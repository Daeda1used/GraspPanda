# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
# Copyright (c) 2020 NVIDIA CORPORATION.
# Adapted for GraspPanda from https://github.com/NVIDIA/MinkowskiEngine @ 02fc608
# (MIT): examples/minkunet.py, examples/resnet.py, MinkowskiEngine/modules/resnet_block.py.
"""MinkUNet: the residual sparse U-Net family of MinkowskiEngine, at any depth and width.

Paper: Choy et al., "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
Networks", CVPR 2019.

The plan is the one of ``examples/minkunet.py`` (and of ``Res16UNet`` in
SpatioTemporalSegmentation, which adds a choice of stem kernel): a stem at full
resolution, four stages that halve the grid with a kernel-2 stride-2 convolution *keeping
the width* and then run ``layers[i]`` residual blocks to ``planes[i]``, and four stages
that double the grid with a transposed convolution, concatenate the encoder feature of
that resolution and run ``layers[4 + i]`` blocks. ``layers`` and ``planes`` give every
published member:

==========  ==========================  ===================================
name        ``layers``                  ``planes`` (decoder half)
==========  ==========================  ===================================
14A / 18A   1 x 8 / 2 x 8               128, 128, 96, 96
14B / 18B   1 x 8 / 2 x 8               128, 128, 128, 128
14C         1 x 8                       192, 192, 128, 128
14D / 18D   1 x 8 / 2 x 8               384, 384, 384, 384
34A         2, 3, 4, 6, 2, 2, 2, 2      256, 128, 64, 64
34B         2, 3, 4, 6, 2, 2, 2, 2      256, 128, 64, 32
34C         2, 3, 4, 6, 2, 2, 2, 2      256, 128, 96, 96
50 / 101    34's / 2, 3, 4, 23, 2 x 4   256, 128, 96, 96, ``bottleneck``
==========  ==========================  ===================================

The encoder half of ``planes`` is 32, 64, 128, 256 in all of them.
:class:`~grasppanda.models.backbones.sparse_unet.SparseUNet14` is the 14-layer member
with the widths of the Graspness reference and keeps its own parameter names;
:class:`~grasppanda.models.backbones.spunet.SpUNet` is Pointcept's network with the 34C
sizes, whose stride-2 convolutions change the width (a checkpoint of one does not fit
the other).

Differences from the upstream code: the layers are those of
:mod:`grasppanda.ops.sparse` (spconv underneath) instead of MinkowskiEngine's, the
network is a :class:`~grasppanda.models.base.Backbone` that voxelises a
:class:`~grasppanda.structures.PointSet` and returns one feature per input point, and
the four stages are built in a loop. Module names are upstream's (``conv0p1s1``,
``bn0``, ``block1.0.conv1``, ``convtr4p16s2``, ``final``) so that
:func:`convert_minkowski_state` only has to re-lay kernels. The equivalence of the
facade's layers with MinkowskiEngine's is tested for ``SparseUNet14``
(``tests/integration/test_gsnet_parity.py``); this class was not compared with
MinkowskiEngine activations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import sparse
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._pretrained import load_pretrained
from .sparse_unet import VoxelMap

__all__ = ["MinkUNet", "convert_minkowski_state"]

#: Edge of a cubic MinkowskiEngine kernel by its number of cells.
_KERNEL_EDGE = {1: 1, 8: 2, 27: 3, 125: 5, 343: 7}

_DOWN = ("conv1p1s2", "conv2p2s2", "conv3p4s2", "conv4p8s2")
_UP = ("convtr4p16s2", "convtr5p8s2", "convtr6p4s2", "convtr7p2s2")


def convert_minkowski_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Re-lay the state dict of a MinkowskiEngine MinkUNet / Res16UNet for :class:`MinkUNet`.

    ``<layer>.kernel`` becomes ``<layer>.weight`` in the facade's layout
    (:func:`grasppanda.ops.sparse.convert_minkowski_kernel`), the ``(1, C)`` bias of the
    final layer becomes ``(C,)``, and the ``.bn.`` level MinkowskiEngine puts around a
    batch norm is removed. No value is changed.
    """
    converted = {}
    for key, value in state.items():
        key = key.replace(".bn.", ".")
        if key.endswith(".kernel"):
            volume = 1 if value.ndim == 2 else value.shape[0]
            if volume not in _KERNEL_EDGE:
                raise ValueError(f"`{key}` has an unsupported kernel volume {volume}")
            key = key[: -len("kernel")] + "weight"
            value = sparse.convert_minkowski_kernel(value, _KERNEL_EDGE[volume])
        elif key.endswith(".bias") and value.ndim == 2 and value.shape[0] == 1:
            value = value.reshape(-1).clone()
        converted[key] = value
    return converted


class _BasicBlock(nn.Module):
    """Two 3³ convolutions with a residual connection (``BasicBlock`` upstream)."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, key: str, momentum: float) -> None:
        super().__init__()
        self.conv1 = sparse.SubMConv3d(inplanes, planes, 3, key=key)
        self.norm1 = sparse.BatchNorm(planes, momentum=momentum)
        self.conv2 = sparse.SubMConv3d(planes, planes, 3, key=key)
        self.norm2 = sparse.BatchNorm(planes, momentum=momentum)
        self.relu = sparse.ReLU()
        self.downsample = _projection(inplanes, planes, momentum)

    def forward(self, x: Any) -> Any:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return _add_residual(self, out, x)


class _Bottleneck(nn.Module):
    """1×1, 3³, 1×1 with four times the width at the output (``Bottleneck`` upstream)."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, key: str, momentum: float) -> None:
        super().__init__()
        self.conv1 = sparse.Linear(inplanes, planes, bias=False)
        self.norm1 = sparse.BatchNorm(planes, momentum=momentum)
        self.conv2 = sparse.SubMConv3d(planes, planes, 3, key=key)
        self.norm2 = sparse.BatchNorm(planes, momentum=momentum)
        self.conv3 = sparse.Linear(planes, planes * 4, bias=False)
        self.norm3 = sparse.BatchNorm(planes * 4, momentum=momentum)
        self.relu = sparse.ReLU()
        self.downsample = _projection(inplanes, planes * 4, momentum)

    def forward(self, x: Any) -> Any:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return _add_residual(self, out, x)


def _projection(inplanes: int, planes: int, momentum: float) -> nn.Sequential | None:
    if inplanes == planes:
        return None
    return nn.Sequential(
        sparse.Linear(inplanes, planes, bias=False), sparse.BatchNorm(planes, momentum=momentum)
    )


def _add_residual(block: nn.Module, out: Any, x: Any) -> Any:
    residual = x if block.downsample is None else block.downsample(x)
    summed = sparse.features(out) + sparse.features(residual)
    return block.relu(sparse.replace_features(out, summed))


@BACKBONES.register()
class MinkUNet(Backbone):
    """Residual sparse U-Net of MinkowskiEngine (MinkUNet 14 to 101, Res16UNet), per-point output.

    The defaults are MinkUNet34C / Res16UNet34C with the 3³ stem of the PointContrast
    and Contrastive Scene Contexts checkpoints, on a 2 cm grid. For tabletop clouds a
    finer grid is usual (GSNet, EconomicGrasp and FineGrasp voxelise at 0.005 m).

    Args:
        in_channels: Feature channels per point.
        out_channels: Width of the final 1×1 layer with bias. ``None`` leaves the layer
            out and returns the features of the last decoder stage
            (``planes[7]``, times four with bottleneck blocks).
        voxel_size: Edge length of a voxel in metres. Points are quantised by
            ``floor(xyz / voxel_size)``; points sharing a voxel share one feature.
        layers: Residual blocks of the four encoder and the four decoder stages.
        planes: Width of the eight stages (see the table in the module docstring).
        block: ``basic`` (14, 18, 34) or ``bottleneck`` (50, 101).
        init_dim: Width of the full-resolution stem.
        stem_kernel: Edge of the stem kernel: 5 in MinkUNet, 3 or 5 in Res16UNet
            checkpoints (``conv1_kernel_size``).
        bn_momentum: Momentum of every batch norm (0.1 upstream; PointContrast trains
            with 0.05).
        voxel_features: How points sharing a voxel are reduced to one input feature.
        final_bias: Bias of the final layer (upstream: yes; the distilled networks of
            OpenScene have none).
        pretrained: Weights to start from: a name of
            ``grasppanda.models.backbones._pretrained.RELEASED`` or a file path. A
            MinkowskiEngine state dict is re-laid by :func:`convert_minkowski_state`.
        pretrained_mismatch: ``error`` refuses when a tensor of the encoder is absent
            from the file or has another shape there; ``skip`` keeps the initialisation
            of such tensors (the stem, when the number of input channels differs).
    """

    output = "dense"
    status = "tested"
    source = "MinkUNet (Choy et al., CVPR 2019), adapted from NVIDIA/MinkowskiEngine@02fc608, MIT"
    extras = ("spconv",)
    #: Each stage halves the resolution.
    stage_stride = 2

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int | None:
        """``out_channels``, or the width of the last decoder stage."""
        width = options.get("out_channels")
        return (
            width
            if isinstance(width, int)
            else options["planes"][7] * (1 if options["block"] == "basic" else 4)
        )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        voxel_size: float = 0.02,
        layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        planes: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        block: Literal["basic", "bottleneck"] = "basic",
        init_dim: int = 32,
        stem_kernel: int = 3,
        bn_momentum: float = 0.1,
        voxel_features: Literal["first", "mean"] = "first",
        final_bias: bool = True,
        pretrained: str | None = None,
        pretrained_mismatch: Literal["error", "skip"] = "error",
    ) -> None:
        super().__init__()
        if voxel_size <= 0:
            raise ValueError(f"MinkUNet: voxel_size must be positive, got {voxel_size}")
        if len(layers) != 8 or len(planes) != 8 or min(layers) < 1:
            raise ValueError(
                "MinkUNet has four encoder and four decoder stages: `layers` and `planes` "
                f"need eight entries each and at least one block per stage; got "
                f"layers={list(layers)}, planes={list(planes)}"
            )
        if block not in ("basic", "bottleneck"):
            raise ValueError(f"MinkUNet: block must be `basic` or `bottleneck`, got {block!r}")
        if stem_kernel % 2 == 0:
            raise ValueError(f"MinkUNet: stem_kernel must be odd, got {stem_kernel}")
        kind = _BasicBlock if block == "basic" else _Bottleneck
        self.in_channels = in_channels
        self.voxel_size = voxel_size
        self.voxel_features = voxel_features
        self.total_stride = self.stage_stride**4

        def stage(inplanes: int, width: int, count: int, key: str) -> nn.Sequential:
            widths = [inplanes, *[width * kind.expansion] * count]
            return nn.Sequential(
                *[kind(widths[index], width, key, bn_momentum) for index in range(count)]
            )

        self.conv0p1s1 = sparse.SubMConv3d(in_channels, init_dim, stem_kernel, key="stem")
        self.bn0 = sparse.BatchNorm(init_dim, momentum=bn_momentum)
        # Level 0 is the full resolution; encoder stage ``i`` goes from level i to i + 1.
        # Blocks of one level share a key, in the encoder and again in the decoder.
        skips = [init_dim]
        width = init_dim
        for index, name in enumerate(_DOWN):
            conv = sparse.SparseConv3d(width, width, 2, self.stage_stride, key=f"down{index + 1}")
            setattr(self, name, conv)
            setattr(self, f"bn{index + 1}", sparse.BatchNorm(width, momentum=bn_momentum))
            blocks = stage(width, planes[index], layers[index], f"level{index + 1}")
            setattr(self, f"block{index + 1}", blocks)
            width = planes[index] * kind.expansion
            skips.append(width)
        skips.pop()  # the coarsest level is the bottleneck, not a skip
        for index, name in enumerate(_UP, start=4):
            level = 8 - index  # the level this stage leaves
            conv = sparse.SparseInverseConv3d(width, planes[index], 2, key=f"down{level}")
            setattr(self, name, conv)
            setattr(self, f"bntr{index}", sparse.BatchNorm(planes[index], momentum=bn_momentum))
            blocks = stage(
                planes[index] + skips.pop(), planes[index], layers[index], f"level{level - 1}"
            )
            setattr(self, f"block{index + 1}", blocks)
            width = planes[index] * kind.expansion
        self.final = (
            None if out_channels is None else sparse.Linear(width, out_channels, bias=final_bias)
        )
        self.out_channels = width if out_channels is None else out_channels
        self.relu = sparse.ReLU()
        self.reset_parameters()
        self.pretrained_report = (
            None
            if pretrained is None
            else load_pretrained(
                self, pretrained, mismatch=pretrained_mismatch, convert=convert_minkowski_state
            )
        )

    def reset_parameters(self) -> None:
        """Convolutions as upstream (``kaiming_normal_`` with ``fan_out``): normal with
        ``std = sqrt(2 / (out_channels · kernel_volume))``; batch norm: identity. The 1×1
        layers and the transposed convolutions keep the initialisation of their layer."""
        for module in self.modules():
            if isinstance(module, (sparse.SubMConv3d, sparse.SparseConv3d)):
                out_channels, volume = module.weight.shape[0], module.weight[0, ..., 0].numel()
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / (out_channels * volume)))
            elif isinstance(module, sparse.BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        if len(points) == 0:
            raise ContractError("MinkUNet received an empty point set")
        if feats.ndim != 2 or tuple(feats.shape) != (len(points), self.in_channels):
            raise ContractError(
                f"MinkUNet expects input features of shape ({len(points)}, {self.in_channels}) "
                f"(one row per point, in_channels={self.in_channels}), got {tuple(feats.shape)}"
            )
        voxel_map = VoxelMap.build(points, self.voxel_size)
        x, _ = sparse.make_tensor(
            voxel_map.reduce(feats, self.voxel_features),
            voxel_map.voxels,
            points.batch_size,
            align=self.total_stride,
        )
        x = self.relu(self.bn0(self.conv0p1s1(x)))
        skips = [x]
        for index, name in enumerate(_DOWN, start=1):
            x = self.relu(getattr(self, f"bn{index}")(getattr(self, name)(x)))
            x = getattr(self, f"block{index}")(x)
            skips.append(x)
        skips.pop()
        for index, name in enumerate(_UP, start=4):
            x = self.relu(getattr(self, f"bntr{index}")(getattr(self, name)(x)))
            x = getattr(self, f"block{index + 1}")(sparse.cat([x, skips.pop()]))
        if self.final is not None:
            x = self.final(x)
        return voxel_map.devoxelize(sparse.features(x), points)
