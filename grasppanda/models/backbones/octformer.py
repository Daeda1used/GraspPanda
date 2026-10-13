# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Adapted for GraspPanda from https://github.com/octree-nn/octformer @ c2975ab (MIT)
"""OctFormer: attention over fixed-size groups of octree nodes.

Paper: Wang, "OctFormer: Octree-based Transformers for 3D Point Clouds", ACM
Transactions on Graphics 42(4) (SIGGRAPH 2023).

The scene is stored as an octree whose leaves are the occupied voxels. The nodes of a
level are sorted along the z-order curve and cut into groups of ``patch_size`` nodes;
attention runs inside a group, and every second block attends over a dilated group
(every ``dilation``-th node) to widen the receptive field. A group always holds the
same number of nodes, whatever shape it covers, which is what makes the attention cheap
to batch. A convolutional stem reduces the leaves by four per axis, four stages follow,
and a feature-pyramid head brings the features back to the leaves and to the points.

Provenance. Blocks, stages, stem and head follow ``models/octformer.py`` and
``models/octformerseg.py`` of the official repository, parameter names included; the
octree, its convolutions and the interpolation are those of the optional ``ocnn``
library (pure PyTorch, the same author). What differs: the patch tables are kept in a
helper next to the octree in place of a subclass of it; the position encoding uses
``ocnn``'s own depth-wise octree convolution where the reference imports a separate
CUDA extension (``dwconv``); the 3x3x3 convolutions run on ``ocnn``'s pure-PyTorch
``block_gemm`` path unless ``conv_method: triton`` is asked for. Loading the released
checkpoints (OneDrive links in the official README) was not tried: this module starts
from random weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet
from grasppanda.utils.imports import require

from ._packed import check_features, cloud_sizes

__all__ = ["OctFormer"]

_MASKED = -1e3  # added to the attention logits of nodes of different clouds


# Every call into ocnn below passes ``True`` for the library's non-empty flag (after the
# stride of a convolution, after the depth of a query): only occupied octree nodes carry
# features, as in the reference configuration.
def _ocnn() -> Any:
    return require("ocnn", extra="ocnn", purpose="OctFormer (octree and octree convolutions)")


class _Patches:
    """The nodes of every level cut into groups of ``patch_size``, plain and dilated."""

    def __init__(
        self, octree: Any, patch_size: int, dilation: int, max_depth: int, start_depth: int
    ) -> None:
        ocnn = _ocnn()
        self.octree = octree
        block = patch_size * dilation
        self.count = octree.nnum_nempty
        self.padded = ((self.count / block).ceil() * block).int()
        self.patch_mask: dict[int, Tensor] = {}
        self.dilate_mask: dict[int, Tensor] = {}
        self.rel_pos: dict[int, Tensor] = {}
        self.dilate_pos: dict[int, Tensor] = {}
        for depth in range(start_depth, max_depth + 1):
            batch = self.partition(octree.batch_id(depth, True), depth, octree.batch_size)
            self.patch_mask[depth] = self._mask(batch.view(-1, patch_size))
            dilated = batch.view(-1, patch_size, dilation).transpose(1, 2)
            self.dilate_mask[depth] = self._mask(dilated.reshape(-1, patch_size))
            key = self.partition(octree.key(depth, True), depth)
            x, y, z, _ = ocnn.octree.key2xyz(key, depth)
            xyz = torch.stack([x, y, z], dim=1).view(-1, patch_size, 3)
            self.rel_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)
            xyz = xyz.view(-1, patch_size, dilation, 3).transpose(1, 2).reshape(-1, patch_size, 3)
            self.dilate_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    @staticmethod
    def _mask(batch: Tensor) -> Tensor:
        mask = batch.unsqueeze(2) - batch.unsqueeze(1)
        return mask.masked_fill(mask != 0, _MASKED)

    def partition(self, data: Tensor, depth: int, fill_value: int = 0) -> Tensor:
        num = int(self.padded[depth] - self.count[depth])
        return torch.cat([data, data.new_full((num, *data.shape[1:]), fill_value)], dim=0)

    def reverse(self, data: Tensor, depth: int) -> Tensor:
        return data[: int(self.count[depth])]


class _RPE(nn.Module):
    """Relative position bias, one table per axis, summed."""

    def __init__(self, patch_size: int, num_heads: int, dilation: int) -> None:
        super().__init__()
        self.pos_bnd = int(0.8 * patch_size * dilation**0.5)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, xyz: Tensor) -> Tensor:
        mul = torch.arange(3, device=xyz.device) * self.rpe_num
        idx = xyz.clamp(-self.pos_bnd, self.pos_bnd) + (self.pos_bnd + mul)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        return out.view(*idx.shape, -1).sum(3).permute(0, 3, 1, 2)


class _OctreeAttention(nn.Module):
    def __init__(
        self, dim: int, patch_size: int, num_heads: int, dilation: int, use_rpe: bool
    ) -> None:
        super().__init__()
        self.dim, self.patch_size, self.num_heads, self.dilation = (
            dim,
            patch_size,
            num_heads,
            dilation,
        )
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rpe = _RPE(patch_size, num_heads, dilation) if use_rpe else None

    def forward(self, data: Tensor, patches: _Patches, depth: int) -> Tensor:
        heads, size, dim, dilation = self.num_heads, self.patch_size, self.dim, self.dilation
        data = patches.partition(data, depth)
        if dilation > 1:
            rel_pos, mask = patches.dilate_pos[depth], patches.dilate_mask[depth]
            data = data.view(-1, size, dilation, dim).transpose(1, 2).reshape(-1, dim)
        else:
            rel_pos, mask = patches.rel_pos[depth], patches.patch_mask[depth]
        data = data.view(-1, size, dim)
        qkv = self.qkv(data).reshape(-1, size, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = query @ key.transpose(-2, -1)
        if self.rpe is not None:
            attn = attn + self.rpe(rel_pos)
        attn = torch.softmax(attn + mask.unsqueeze(1), dim=-1)
        data = (attn @ value).transpose(1, 2).reshape(-1, dim)
        if dilation > 1:
            data = data.view(-1, dilation, size, dim).transpose(1, 2).reshape(-1, dim)
        return self.proj(patches.reverse(data, depth))


class _CPE(nn.Module):
    """Convolutional position encoding: a depth-wise 3x3x3 octree convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _ocnn().nn.OctreeDWConv(channels, [3], 1, True, False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, data: Tensor, octree: Any, depth: int) -> Tensor:
        return self.bn(self.conv(data, octree, depth))


class _MLP(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, data: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(data)))


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: int,
        dilation: int,
        mlp_ratio: float,
        drop_path: float,
        use_rpe: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _OctreeAttention(dim, patch_size, num_heads, dilation, use_rpe)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))
        self.drop_path = _ocnn().nn.OctreeDropPath(drop_path, True)
        self.cpe = _CPE(dim)

    def forward(self, data: Tensor, patches: _Patches, depth: int) -> Tensor:
        octree = patches.octree
        data = self.cpe(data, octree, depth) + data
        attn = self.attention(self.norm1(data), patches, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data + self.drop_path(self.mlp(self.norm2(data)), octree, depth)


class _Stage(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)


class _PatchEmbed(nn.Module):
    """The stem: per halving a 3x3x3 convolution and a stride-2 convolution."""

    def __init__(self, in_channels: int, dim: int, num_down: int) -> None:
        super().__init__()
        conv = _ocnn().modules.OctreeConvBnRelu
        self.num_stages = num_down
        channels = [int(dim * 2**i) for i in range(-num_down, 1)]
        self.convs = nn.ModuleList(
            conv(in_channels if i == 0 else channels[i], channels[i], [3], 1, True)
            for i in range(num_down)
        )
        self.downsamples = nn.ModuleList(
            conv(channels[i], channels[i + 1], [2], 2, True) for i in range(num_down)
        )
        self.proj = conv(channels[-1], dim, [3], 1, True)

    def forward(self, data: Tensor, octree: Any, depth: int) -> Tensor:
        for i in range(self.num_stages):
            data = self.convs[i](data, octree, depth - i)
            data = self.downsamples[i](data, octree, depth - i)
        return self.proj(data, octree, depth - self.num_stages)


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channels)
        self.conv = _ocnn().nn.OctreeConv(in_channels, out_channels, [2], 2, True, use_bias=True)

    def forward(self, data: Tensor, octree: Any, depth: int) -> Tensor:
        return self.norm(self.conv(data, octree, depth))


class _Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        patch_size: int,
        dilation: int,
        mlp_ratio: float,
        drop_path: float,
        stem_down: int,
        use_rpe: bool,
    ) -> None:
        super().__init__()
        rates = torch.linspace(0, drop_path, sum(num_blocks)).tolist()
        self.patch_embed = _PatchEmbed(in_channels, channels[0], stem_down)
        self.layers = nn.ModuleList()
        for i, count in enumerate(num_blocks):
            stage_rates = rates[sum(num_blocks[:i]) : sum(num_blocks[: i + 1])]
            self.layers.append(
                _Stage(
                    [
                        _Block(
                            channels[i],
                            num_heads[i],
                            patch_size,
                            1 if j % 2 == 0 else dilation,
                            mlp_ratio,
                            stage_rates[j],
                            use_rpe,
                        )
                        for j in range(count)
                    ]
                )
            )
        self.downsamples = nn.ModuleList(
            _Downsample(channels[i], channels[i + 1]) for i in range(len(num_blocks) - 1)
        )


class _Head(nn.Module):
    """Feature pyramid over the stages, up to the leaves, read out at the points."""

    def __init__(
        self, out_channels: int, channels: Sequence[int], fpn_channel: int, num_up: int
    ) -> None:
        super().__init__()
        ocnn = _ocnn()
        stages = len(channels)
        self.conv1x1 = nn.ModuleList(
            nn.Linear(channels[i], fpn_channel) for i in range(stages - 1, -1, -1)
        )
        self.upsample = ocnn.nn.OctreeUpsample("nearest", True)
        self.conv3x3 = nn.ModuleList(
            ocnn.modules.OctreeConvBnRelu(fpn_channel, fpn_channel, [3], 1, True)
            for _ in range(stages)
        )
        self.up_conv = nn.ModuleList(
            ocnn.modules.OctreeDeconvBnRelu(fpn_channel, fpn_channel, [3], 2, True)
            for _ in range(num_up)
        )
        self.interp = ocnn.nn.OctreeInterp("nearest", True)
        self.classifier = nn.Sequential(
            nn.Linear(fpn_channel, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: dict[int, Tensor], octree: Any, query: Tensor) -> Tensor:
        depth, depth_max = min(features), max(features)
        feature = self.conv1x1[0](features[depth])
        out = self.upsample(self.conv3x3[0](feature, octree, depth), octree, depth, depth_max)
        for i in range(1, len(features)):
            feature = self.upsample(feature, octree, depth + i - 1)
            feature = self.conv1x1[i](features[depth + i]) + feature
            conv_out = self.conv3x3[i](feature, octree, depth + i)
            out = out + self.upsample(conv_out, octree, depth + i, depth_max)
        for i, up_conv in enumerate(self.up_conv):
            out = up_conv(out, octree, depth_max + i)
        return self.classifier(self.interp(out, octree, depth_max + len(self.up_conv), query))


@BACKBONES.register()
class OctFormer(Backbone):
    """OctFormer encoder with a feature-pyramid head: features on every input point.

    The defaults are the ScanNet segmentation model of the official repository
    (channels 96-192-384-384, 2-2-18-2 blocks, groups of 26 nodes, dilation 4, FPN width
    168) on leaves of 1 cm. ``voxel_size`` is the unit of length: a leaf of the octree.
    Every cloud is centred on the middle of its bounding box and must fit a cube of
    ``voxel_size * 2**octree_depth`` (10.24 m at the defaults).

    The groups of ``patch_size`` nodes are cut from the z-ordered nodes of the whole
    batch, as in the reference: nodes of different clouds do not attend to each other,
    but where a group starts inside a cloud depends on the clouds before it, so the
    features of a cloud are not exactly those of the cloud run alone.

    Args:
        in_channels: Feature channels per point, averaged per leaf.
        out_channels: Width of the returned features (linear, batch norm, ReLU after
            the pyramid).
        voxel_size: Edge of a leaf, in the unit of the coordinates.
        octree_depth: Depth of the leaves; at least ``stem_down`` plus the number of
            stages plus 1.
        local_coords: Append the position of the points inside their leaf (3 channels,
            the ``L`` input feature of ``ocnn``) to the features.
        channels: Width of every stage.
        num_blocks: Blocks of every stage.
        num_heads: Attention heads of every stage.
        patch_size: Nodes per attention group.
        dilation: Dilation of every second block.
        mlp_ratio: Hidden width of a block's MLP over its width.
        drop_path: Largest stochastic-depth rate; it grows linearly with depth.
        stem_down: Halvings in the stem.
        fpn_channel: Width of the feature pyramid.
        use_rpe: Relative position bias in the attention (the released code enables it;
            the paper's experiments did not have it).
        conv_method: Implementation of ``ocnn``'s 3x3x3 convolutions: ``block_gemm`` and
            ``explicit_gemm`` are pure PyTorch; ``triton`` compiles GPU kernels with
            Triton at the first call (ocnn's own default on CUDA).
    """

    output = "dense"
    status = "bridge"
    extras = ("ocnn",)
    source = (
        "OctFormer (Wang, ACM TOG / SIGGRAPH 2023); adapted from octree-nn/octformer@c2975ab, "
        "MIT; octree operators via ocnn"
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 168,
        voxel_size: float = 0.01,
        octree_depth: int = 11,
        local_coords: bool = True,
        channels: Sequence[int] = (96, 192, 384, 384),
        num_blocks: Sequence[int] = (2, 2, 18, 2),
        num_heads: Sequence[int] = (6, 12, 24, 24),
        patch_size: int = 26,
        dilation: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.5,
        stem_down: int = 2,
        fpn_channel: int = 168,
        use_rpe: bool = True,
        conv_method: str = "block_gemm",
    ) -> None:
        super().__init__()
        if conv_method not in ("block_gemm", "explicit_gemm", "triton"):
            raise ValueError(
                f"OctFormer: unknown conv_method '{conv_method}'; choose block_gemm, "
                "explicit_gemm or triton"
            )
        stages = len(channels)
        if not stages == len(num_blocks) == len(num_heads) >= 1:
            raise ValueError(
                "OctFormer: channels, num_blocks and num_heads need one entry per stage, got "
                f"{len(channels)}, {len(num_blocks)}, {len(num_heads)}"
            )
        if any(width % heads for width, heads in zip(channels, num_heads)):
            raise ValueError("OctFormer: every stage width must be a multiple of its heads")
        if in_channels < 1 and not local_coords:
            raise ValueError("OctFormer needs input: in_channels >= 1 or local_coords: true")
        if octree_depth - stem_down - stages + 1 < 2:
            raise ValueError(
                f"OctFormer: octree_depth={octree_depth} is too shallow for a stem of "
                f"{stem_down} halvings and {stages} stages; it must be at least "
                f"{stem_down + stages + 1}"
            )
        if voxel_size <= 0:
            raise ValueError("OctFormer: voxel_size must be positive")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.voxel_size, self.octree_depth, self.local_coords = (
            voxel_size,
            octree_depth,
            local_coords,
        )
        self.patch_size, self.dilation, self.stem_down = patch_size, dilation, stem_down
        self.backbone = _Encoder(
            in_channels + (3 if local_coords else 0),
            channels,
            num_blocks,
            num_heads,
            patch_size,
            dilation,
            mlp_ratio,
            drop_path,
            stem_down,
            use_rpe,
        )
        self.head = _Head(out_channels, channels, fpn_channel, stem_down)
        for module in self.modules():
            if getattr(module, "method", None) == "triton" and conv_method != "triton":
                module.method = conv_method  # an ocnn convolution on its Triton default
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _octree(self, points: PointSet, feats: Tensor | None) -> tuple[Any, Tensor]:
        """The merged octree of the batch and the ``(N, 4)`` query ``[x, y, z, cloud]``."""
        ocnn = _ocnn()
        coords, batch = points.coords, points.batch_index
        index = batch[:, None].expand_as(coords)
        low = coords.new_zeros(points.batch_size, 3).scatter_reduce(
            0, index, coords, reduce="amin", include_self=False
        )
        high = coords.new_zeros(points.batch_size, 3).scatter_reduce(
            0, index, coords, reduce="amax", include_self=False
        )
        half = self.voxel_size * 2 ** (self.octree_depth - 1)
        xyz = (coords - ((low + high) / 2)[batch]) / half
        if float(xyz.abs().max()) >= 1.0:
            extent = float((high - low).max())
            raise ContractError(
                f"OctFormer: a cloud spans {extent:.2f} m, more than the {2 * half:.2f} m cube "
                f"of an octree of depth {self.octree_depth} with voxel_size={self.voxel_size}. "
                "Raise octree_depth or voxel_size."
            )
        octrees = []
        start = 0
        for size in cloud_sizes(points):
            rows = slice(start, start + size)
            cloud = ocnn.octree.Points(xyz[rows], features=None if feats is None else feats[rows])
            octree = ocnn.octree.Octree(self.octree_depth, 2, device=xyz.device)
            octree.build_octree(cloud)
            octrees.append(octree)
            start += size
        merged = ocnn.octree.merge_octrees(octrees)
        merged.construct_all_neigh()
        return merged, torch.cat([xyz, batch[:, None].to(xyz.dtype)], dim=1)

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="OctFormer")
        if min(cloud_sizes(points)) < 1:
            raise ContractError("OctFormer received an empty cloud; every cloud needs points")
        octree, query = self._octree(points, feats)
        wanted = ("F" if feats is not None else "") + ("L" if self.local_coords else "")
        data = octree.get_input_feature(wanted, True)
        encoder = self.backbone
        data = encoder.patch_embed(data, octree, self.octree_depth)
        depth = self.octree_depth - self.stem_down
        stages = len(encoder.layers)
        patches = _Patches(octree, self.patch_size, self.dilation, depth, depth - stages + 1)
        features: dict[int, Tensor] = {}
        for i, stage in enumerate(encoder.layers):
            for block in stage.blocks:
                data = block(data, patches, depth - i)
            features[depth - i] = data
            if i < stages - 1:
                data = encoder.downsamples[i](data, octree, depth - i)
        return PointFeatures(self.head(features, octree, query), points)
