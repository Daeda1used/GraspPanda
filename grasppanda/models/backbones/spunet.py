"""SpUNet: the deep sparse residual U-Net of Pointcept, in the MinkUNet-34C layout.

Paper of the layout: Choy et al., "4D Spatio-Temporal ConvNets: Minkowski Convolutional
Neural Networks", CVPR 2019 (the residual U-Net and its 34C sizes).

A 5³ stem, four stages that halve the grid and run 2-3-4-6 residual blocks at
32-64-128-256 channels, and four up-sampling stages of two blocks each (256-128-96-96)
that concatenate the encoder feature of the same resolution. In Pointcept this network
is the backbone of the sparse-convolution baselines and of the methods built on them
(the published configs include PointGroup, MSC, PPT and CAC).

It is Pointcept's network, not a re-labelled MinkUNet34C: the block counts and widths
are those of 34C, but here the stride-2 convolution of a stage already changes the
width and every encoder block keeps it, whereas in MinkUNet the stride-2 convolution
keeps the width and the first block of the stage changes it through a projected
shortcut. :class:`~grasppanda.models.backbones.sparse_unet.SparseUNet14` is the
MinkUNet form (one block per stage), as the Graspness reference uses it.

:class:`SpUNet` returns ``dense`` features. Points are quantised to a grid owned by the
backbone (``grid_size``) as the official data pipeline does it, the network runs on the
first point of every cell (officially a random one while training), and the feature of
a cell is handed to every point in it. The official encoder-only mode (``enc_mode``,
one pooled vector per cloud) is not a scene encoder and is not implemented.

Written against :mod:`grasppanda.ops.sparse`, not against a sparse library. The module
tree and parameter names (``conv_input.0``, ``enc.0.block0.conv1``, ``dec.0.block0.proj``
...) follow the official model on purpose, and every kernel has the facade's layout,
which is spconv's. The one exception is the 1×1 convolution, which the facade defines
as a linear layer: :func:`convert_pointcept_state` re-lays those kernels and nothing
else, after which a state dict of the official model loads strictly.

Provenance. Written for this code base from the papers and from the behaviour of the
official implementation in Pointcept
(``pointcept/models/sparse_unet/spconv_unet_v1m1_base.py``, MIT licence).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import sparse
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointFeatures, PointSet

from ._grid import grid_cells
from ._packed import check_features
from ._pretrained import load_pretrained

__all__ = ["SpUNet", "convert_pointcept_state"]


def convert_pointcept_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Re-lay the state dict of the official SpUNet for :class:`SpUNet`.

    Keys are kept. Kernels of 5³, 3³ and 2³ cells are kept as well: the facade's layout
    ``(out, kx, ky, kz, in)`` is the one spconv stores. A 1×1 convolution is a
    :class:`grasppanda.ops.sparse.Linear` here, and spconv does not read its
    ``(out, 1, 1, 1, in)`` parameter as that shape suggests: it multiplies the features
    by the same memory viewed as ``(in, out)``. The equivalent linear weight is therefore
    the transpose of that view, not a reshape to ``(out, in)``. No value is changed.

    The equivalence is established against spconv itself in
    ``tests/unit/test_backbone_spunet.py`` and against the official model in
    ``tests/integration/test_backbone_sparse_cnn_oracle.py``.

    Args:
        state: State dict of Pointcept's ``SpUNet-v1m1`` (the backbone alone, without a
            ``backbone.`` prefix).
    """
    converted = {}
    for key, value in state.items():
        if value.ndim == 5 and tuple(value.shape[1:4]) == (1, 1, 1):
            out_channels, in_channels = value.shape[0], value.shape[4]
            value = value.reshape(in_channels, out_channels).t().contiguous()
        converted[key] = value
    return converted


def _norm(channels: int) -> sparse.BatchNorm:
    return sparse.BatchNorm(channels, eps=1e-3, momentum=0.01)


def _conv_norm_act(conv: nn.Module, channels: int) -> nn.Sequential:
    return nn.Sequential(conv, _norm(channels), sparse.ReLU())


class _BasicBlock(nn.Module):
    """Two 3³ submanifold convolutions with a residual connection, projected when the
    width changes."""

    def __init__(self, in_channels: int, channels: int, key: str) -> None:
        super().__init__()
        self.proj = nn.Sequential()
        if in_channels != channels:
            self.proj.append(sparse.Linear(in_channels, channels, bias=False))
            self.proj.append(_norm(channels))
        self.conv1 = sparse.SubMConv3d(in_channels, channels, 3, key=key)
        self.bn1 = _norm(channels)
        self.relu = sparse.ReLU()
        self.conv2 = sparse.SubMConv3d(channels, channels, 3, key=key)
        self.bn2 = _norm(channels)

    def forward(self, x: Any) -> Any:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        summed = sparse.features(out) + sparse.features(self.proj(x))
        return self.relu(sparse.replace_features(out, summed))


def _blocks(in_channels: int, channels: int, count: int, key: str) -> nn.Sequential:
    """``count`` blocks named ``block0``, ``block1``, ...; only the first changes the width."""
    widths = [in_channels, *[channels] * count]
    return nn.Sequential(
        OrderedDict(
            (f"block{index}", _BasicBlock(widths[index], channels, key)) for index in range(count)
        )
    )


@BACKBONES.register()
class SpUNet(Backbone):
    """Sparse residual U-Net of Pointcept (``SpUNet-v1m1``), one feature per input point.

    The defaults are the published ScanNet semantic-segmentation model
    (``semseg-spunet-v1m1-0-base``, 39.2 M parameters): 0.02 m voxels, 6 input channels
    (colour and normal), four stages down and four up, and a 20-channel output. For
    tabletop clouds a finer grid is usual (GSNet voxelises at 0.005 m).

    Args:
        in_channels: Feature channels per point.
        out_channels: Channels of the returned features: the width of the official
            ``final`` 1×1 convolution, which the published segmentation model sets to
            the number of classes. ``None`` leaves that layer out and returns the
            features of the last decoder stage (``channels[-1]``), which is what the
            official ``num_classes=0`` does for the methods that put their own head on
            the backbone.
        grid_size: Edge of a voxel in the unit of the coordinates. Points sharing a
            voxel share one output feature.
        base_channels: Width of the full-resolution 5³ stem.
        channels: Width of every stage: the encoder stages, each of which halves the
            grid first, then as many decoder stages, coarsest first.
        layers: Residual blocks of every stage, in the order of ``channels``.
        pretrained: Released weights to start from: a name of
            ``grasppanda.models.backbones._pretrained.RELEASED`` or a file path. The
            encoder part of the file is loaded at construction and reported in the log; the 1×1
            kernels of a Pointcept file are re-laid by :func:`convert_pointcept_state`.
        pretrained_mismatch: ``error`` refuses when a tensor of the encoder is absent
            from the file or has another shape there; ``skip`` keeps the initialisation
            of such tensors (the stem, when the number of input channels differs).
    """

    output = "dense"
    status = "tested"
    source = "SpUNet of Pointcept, MinkUNet-34C layout (Choy et al., CVPR 2019)"
    extras = ("spconv",)
    #: Each stage halves the resolution.
    stage_stride = 2

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int | None = 20,
        grid_size: float = 0.02,
        base_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        pretrained: str | None = None,
        pretrained_mismatch: Literal["error", "skip"] = "error",
    ) -> None:
        super().__init__()
        name = type(self).__name__
        if in_channels < 1:
            raise ValueError(f"{name} needs at least one input feature channel per point")
        if grid_size <= 0:
            raise ValueError(f"{name}: grid_size must be positive, got {grid_size}")
        if len(channels) == 0 or len(channels) % 2 or len(layers) != len(channels):
            raise ValueError(
                f"{name}: channels and layers need one entry per stage, as many encoder as "
                f"decoder stages; got {len(channels)} and {len(layers)} entries"
            )
        if min(layers) < 1:
            raise ValueError(f"{name}: every stage needs at least one block, got {list(layers)}")
        stages = len(channels) // 2
        self.in_channels = in_channels
        self.out_channels = channels[-1] if out_channels is None else out_channels
        self.grid_size = grid_size
        self.num_stages = stages
        self.total_stride = self.stage_stride**stages

        self.conv_input = _conv_norm_act(
            sparse.SubMConv3d(in_channels, base_channels, 5, key="stem"), base_channels
        )
        # Level 0 is the full resolution, level ``s + 1`` what encoder stage ``s`` produces.
        # Decoder stage ``s`` undoes encoder stage ``s``: it returns to level ``s`` and
        # joins the encoder feature of that level. Blocks of one level share a key.
        widths = [base_channels, *channels[:stages]]  # of the encoder feature at every level
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for stage in range(stages):
            mirror = len(channels) - 1 - stage  # this stage's entry on the decoder side
            pair = f"spconv{stage + 1}"
            conv = sparse.SparseConv3d(
                widths[stage], widths[stage + 1], 2, self.stage_stride, key=pair
            )
            self.down.append(_conv_norm_act(conv, widths[stage + 1]))
            self.enc.append(
                _blocks(widths[stage + 1], widths[stage + 1], layers[stage], f"subm{stage + 1}")
            )
            conv = sparse.SparseInverseConv3d(channels[mirror - 1], channels[mirror], 2, key=pair)
            self.up.append(_conv_norm_act(conv, channels[mirror]))
            self.dec.append(
                _blocks(
                    channels[mirror] + widths[stage],
                    channels[mirror],
                    layers[mirror],
                    f"subm{stage}",
                )
            )
        self.final = None if out_channels is None else sparse.Linear(channels[-1], out_channels)
        self.reset_parameters()
        self.pretrained_report = (
            None
            if pretrained is None
            else load_pretrained(
                self, pretrained, mismatch=pretrained_mismatch, convert=convert_pointcept_state
            )
        )

    def reset_parameters(self) -> None:
        """Initialise as the official model does.

        Submanifold convolutions and the 1×1 layers: truncated normal with a standard
        deviation of 0.02, zero bias. Batch norm: identity. The down- and up-sampling
        convolutions keep the initialisation of the sparse backend.
        """
        for module in self.modules():
            if isinstance(module, (sparse.SubMConv3d, sparse.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, sparse.BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        grid = grid_cells(points, self.grid_size, user=name)
        voxels = torch.cat([grid.batch[:, None], grid.cells], dim=1)
        x, _ = sparse.make_tensor(
            feats[grid.first], voxels, points.batch_size, align=self.total_stride
        )
        x = self.conv_input(x)
        skips = []
        for down, blocks in zip(self.down, self.enc):
            skips.append(x)
            x = blocks(down(x))
        for up, blocks in zip(reversed(self.up), reversed(self.dec)):
            x = blocks(sparse.cat([up(x), skips.pop()]))
        if self.final is not None:
            x = self.final(x)
        return PointFeatures(sparse.features(x)[grid.inverse], points)
