"""LitePT: sparse convolutions at fine resolution, attention at coarse resolution.

Paper: Yue et al., "LitePT: Lighter Yet Stronger Point Transformer", CVPR 2026
(arXiv 2512.13689).

LitePT keeps the skeleton of Point Transformer V3 (a voxel hierarchy with grid pooling,
attention inside patches of a serialised order) and changes what runs where. Every stage
is a stack of blocks, and a stage switches two things on or off:

* ``conv``: a 3×3×3 sparse convolution followed by a linear layer and a layer norm,
  added to the features. In PTv3 this is the positional encoding of every block; here
  it is the whole block of the three finest stages, which therefore need no
  serialisation and no attention.
* ``attn``: patch attention and an MLP, as in PTv3. Positions enter through *PointROPE*,
  a rotary embedding without parameters: the channels of a head are three chunks, one per
  axis, and each chunk is rotated by the voxel's integer grid coordinate along that axis.
  A stage that attends without convolving normalises its input first instead.

The published models convolve in stages 0 to 2 and attend in stages 3 and 4, and their
decoder is the unpooling alone (LitePT-S, -B, -L); the instance-segmentation variant
LitePT-S* adds two blocks to every decoder stage. :class:`LitePT` returns ``dense``
features: points are quantised to a voxel grid owned by the backbone (``grid_size``), the
network runs on one representative point per voxel (the first one) and voxel features are
mapped back to exactly the points that produced them.

Deliberate differences from the official code, each for the backbone contract:

* **No flash-attention.** The official attention casts queries, keys and values to half
  precision and calls flash-attention; it has no other path. Here attention is PyTorch's
  ``scaled_dot_product_attention`` in the precision of the input, on the same patches.
  With the released weights the official output moves by 6e-4 to 2e-3 (on features of
  magnitude 5 to 7) when its attention runs in float32 instead; that is the only
  numerical difference to the official model (``docs/model_zoo_backbones.md``).
* **Batch independence.** The serialisation depth is taken per cloud. Officially it is
  the depth of the largest cloud in the batch, and the Hilbert curve depends on it. For
  a single cloud the two agree.
* **Determinism.** ``shuffle_orders`` permutes the curves among the blocks in training
  mode only. Officially the pooling layers shuffle on every forward pass, in evaluation
  mode too, whatever the model's ``shuffle_orders`` says (the argument only reaches the
  serialisation of the input level).
* **One patch size per level.** Officially a level's patch layout is computed once and
  reused, so a decoder stage that attends would silently run with the encoder's patch
  size. Here the two sizes have to be equal.
* **PointROPE in PyTorch.** The official repository ships a CUDA extension and a PyTorch
  fallback that looks the rotations up in a table. Both compute what is computed here;
  the comparisons were recorded with each of them.

The official ``enc_mode`` (encoder without decoder) is not implemented: no released
checkpoint uses it and the model then returns its coarsest level, not dense features.

Provenance. Written for this code base from the paper and from the behaviour of the
official implementation (``prs-eth/LitePT`` at commit ``436d048``, MIT licence; a copy
lives in Pointcept).
The module tree and parameter names (``embedding.stem``, ``enc.enc3.block0.attn.qkv``
...) follow it on purpose, so that official state dicts load without renaming. The
released weights are MIT as well. The serialisation and patch helpers are the ones of
:mod:`grasppanda.models.backbones.ptv3`.

Needs spconv for the sparse convolutions; nothing else beyond PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import SerializationOrder, voxelize
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet
from grasppanda.utils.imports import require

from ._packed import check_features, cloud_sizes
from ._pretrained import load_pretrained
from .ptv3 import (
    _MLP,
    _cloud_depths,
    _codes,
    _DropPath,
    _Layout,
    _patch_layout,
    _segment,
    _sort,
    _sparse_conv,
    _sparse_tensor,
)

__all__ = ["LitePT"]


@dataclass(eq=False)
class _Level:
    """One resolution of the hierarchy: its voxels, their features and, if a stage
    attends here, their order along each curve."""

    feat: Tensor  # (n, C)
    grid: Tensor  # (n, 3) non-negative integer voxel coordinates
    batch: Tensor  # (n,) cloud of every voxel, non-decreasing
    sizes: list[int]  # voxels per cloud
    sparse: Any  # spconv tensor over these voxels; caches the convolution index pairs
    parent: _Level | None = None
    cluster: Tensor | None = None  # (n_parent,) row of this level every parent row fell into
    order: Tensor | None = None  # (K, n) rows in curve order
    inverse: Tensor | None = None  # (K, n) position of every row in curve order
    layouts: dict[int, _Layout] = field(default_factory=dict)

    def serialize(self, orders: Sequence[SerializationOrder], shuffle: bool) -> None:
        depths = _cloud_depths(self.grid, self.batch, len(self.sizes), spare=1)
        code = _codes(self.grid, self.batch, depths, orders)
        _, self.order, self.inverse = _sort(code, shuffle)

    def layout(self, patch_size: int) -> _Layout:
        if patch_size not in self.layouts:
            self.layouts[patch_size] = _patch_layout(self.sizes, patch_size, self.feat.device)
        return self.layouts[patch_size]


def _batch_norm(channels: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


def _point_rope(tokens: Tensor, position: Tensor, base: float) -> Tensor:
    """Rotate ``(n, H, D)`` tokens by their ``(n, 3)`` integer grid positions.

    The ``D`` channels of a head are three chunks, for x, y and z. A chunk is two halves
    ``u`` and ``v``; the pair ``(u[i], v[i])`` is turned by the angle
    ``position * base ** (-i / len(u))``. The dot product of two rotated tokens then
    depends on the offset between their voxels, not on where the voxels are.

    The angles are float32 whatever the precision of the tokens, and half-precision
    tokens are rotated in float32, as in the official implementation.
    """
    chunk = tokens.shape[-1] // 3
    steps = torch.arange(0, chunk, 2, device=tokens.device).float() / chunk
    angle = position.float()[:, :, None] * (1.0 / base**steps)  # (n, 3, D / 6)
    precise = tokens.to(torch.promote_types(tokens.dtype, torch.float32))
    angle = torch.cat([angle, angle], dim=-1).flatten(1)[:, None, :].to(precise.dtype)  # (n, 1, D)
    first, second = precise.unflatten(-1, (3, chunk)).chunk(2, dim=-1)
    turned = torch.cat([-second, first], dim=-1).flatten(-2)
    return (precise * angle.cos() + turned * angle.sin()).to(tokens.dtype)


class _PointROPEAttention(nn.Module):
    """Multi-head self-attention inside patches of one serialised order, with PointROPE."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        rope_freq: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        order_index: int,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"{channels} channels cannot be split into {num_heads} heads")
        if (channels // num_heads) % 6:
            raise ValueError(
                f"PointROPE needs a head width divisible by 6 (three axes, channel pairs), got "
                f"{channels // num_heads} ({channels} channels in {num_heads} heads)"
            )
        self.channels, self.num_heads, self.patch_size = channels, num_heads, patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.rope_freq = rope_freq
        self.order_index = order_index
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat: Tensor, level: _Level) -> Tensor:
        heads, width = self.num_heads, self.channels // self.num_heads
        pad, unpad, runs = level.layout(self.patch_size)
        order = level.order[self.order_index][pad]  # row held by every slot of the padded sequence
        slot = unpad[level.inverse[self.order_index]]  # slot every row reads its result from
        q, k, v = self.qkv(feat)[order].reshape(-1, 3, heads, width).unbind(dim=1)
        position = level.grid[order]
        q, k = _point_rope(q, position, self.rope_freq), _point_rope(k, position, self.rope_freq)
        out, start = [], 0
        for slots, length in runs:
            shape = (slots // length, length, heads, width)
            queries, keys, values = (
                part[start : start + slots].reshape(shape).transpose(1, 2) for part in (q, k, v)
            )
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.attn_drop if self.training else 0.0,
                scale=self.scale,
            )
            out.append(attended.transpose(1, 2).reshape(slots, self.channels))
            start += slots
        return self.proj_drop(self.proj(torch.cat(out)[slot]))


class _Block(nn.Module):
    """A sparse convolution with a residual, or a layer norm; then, if the stage attends,
    attention and an MLP, each with a residual."""

    def __init__(
        self,
        spconv: Any,
        channels: int,
        num_heads: int,
        patch_size: int,
        rope_freq: float,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        pre_norm: bool,
        order_index: int,
        indice_key: str,
        enable_conv: bool,
        enable_attn: bool,
    ) -> None:
        super().__init__()
        self.pre_norm, self.enable_conv, self.enable_attn = pre_norm, enable_conv, enable_attn
        if enable_conv:
            self.conv = nn.Sequential(
                spconv.SubMConv3d(
                    channels, channels, kernel_size=3, bias=True, indice_key=indice_key
                ),
                nn.Linear(channels, channels),
                nn.LayerNorm(channels),
            )
        else:
            self.norm0 = nn.Sequential(nn.LayerNorm(channels))
        if enable_attn:
            self.norm1 = nn.Sequential(nn.LayerNorm(channels))
            self.attn = _PointROPEAttention(
                channels,
                num_heads,
                patch_size,
                rope_freq,
                qkv_bias,
                qk_scale,
                attn_drop,
                proj_drop,
                order_index,
            )
            self.norm2 = nn.Sequential(nn.LayerNorm(channels))
            self.mlp = nn.Sequential(_MLP(channels, int(channels * mlp_ratio), proj_drop))
            self.drop_path = _DropPath(drop_path)

    def forward(self, level: _Level) -> None:
        feat = level.feat
        if self.enable_conv:
            level.sparse = _sparse_conv(self.conv[0], level.sparse.replace_feature(feat))
            feat = feat + self.conv[2](self.conv[1](level.sparse.features))
        else:
            feat = self.norm0(feat)
        if self.enable_attn:
            branch = self.norm1(feat) if self.pre_norm else feat
            feat = feat + self.drop_path(self.attn(branch, level))
            if not self.pre_norm:
                feat = self.norm1(feat)

            branch = self.norm2(feat) if self.pre_norm else feat
            feat = feat + self.drop_path(self.mlp(branch))
            if not self.pre_norm:
                feat = self.norm2(feat)
        level.feat = feat


class _Pooling(nn.Module):
    """Project, then max-pool the voxels that fall into the same coarser cell."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.Sequential(_batch_norm(out_channels))
        self.act = nn.Sequential(nn.GELU())

    def forward(self, spconv: Any, level: _Level) -> _Level:
        batch_size = len(level.sizes)
        keys = torch.cat([level.batch[:, None], level.grid // self.stride], dim=1)
        cells, cluster = torch.unique(keys, dim=0, return_inverse=True)
        batch, grid = cells[:, 0], cells[:, 1:]
        feat = self.act(self.norm(_segment(self.proj(level.feat), cluster, len(cells), "amax")))
        return _Level(
            feat=feat,
            grid=grid,
            batch=batch,
            sizes=torch.bincount(batch, minlength=batch_size).tolist(),
            sparse=_sparse_tensor(spconv, feat, grid, batch, batch_size),
            parent=level,
            cluster=cluster,
        )


class _Unpooling(nn.Module):
    """Project the coarse and the skip features and add them through the pooling map."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels), _batch_norm(out_channels), nn.GELU()
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels), _batch_norm(out_channels), nn.GELU()
        )

    def forward(self, level: _Level) -> _Level:
        parent = level.parent
        parent.feat = self.proj_skip(parent.feat) + self.proj(level.feat)[level.cluster]
        return parent


def _voxels(points: PointSet, grid_size: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantise per cloud. Returns grid, batch, representative row and inverse map."""
    # Quantise in double precision, like the NumPy grid sampling of the official
    # pipeline: in float32 a point that sits on a cell boundary can land on either side.
    cells = voxelize(points.coords.double(), grid_size, points.batch_index)
    batch, grid = cells.voxels[:, 0], cells.voxels[:, 1:]
    # Shift every cloud to start at cell 0, as the official grid sampling does, so that
    # a cloud's curves and pooling cells do not depend on where the other clouds sit.
    lowest = torch.zeros(points.batch_size, 3, dtype=torch.long, device=grid.device)
    index = batch[:, None].expand_as(grid)
    lowest = lowest.scatter_reduce(0, index, grid, reduce="amin", include_self=False)
    return grid - lowest[batch], batch, cells.first, cells.inverse


@BACKBONES.register()
class LitePT(Backbone):
    """LitePT encoder-decoder (official ``LitePT``, Pointcept ``LitePT-v1``).

    The defaults are the published LitePT-S for ScanNet semantic segmentation
    (``semseg-litept-small-v1m1``): five encoder stages of 2-2-2-6-2 blocks with 36 to
    504 channels, convolution in stages 0 to 2 and attention in stages 3 and 4, a decoder
    of four unpooling steps without blocks, patches of 1024 voxels, all four curves,
    0.02 m voxels, 6 input channels (colour and normal) and 72 output channels;
    12.7 M parameters. ``docs/model_zoo_backbones.md`` lists the other published sizes.
    For tabletop clouds a finer grid is usual (GSNet voxelises at 0.005 m).

    The strides and the grid size decide what attention sees: with the defaults, voxels
    of ``8 * grid_size`` and ``16 * grid_size``. PointROPE reads positions in units of
    those voxels, so ``enc_rope_freq`` sets wavelengths in coarse voxels, not in metres,
    and a scene with fewer coarse voxels than ``enc_patch_size`` is attended as a whole.

    Args:
        in_channels: Feature channels per point.
        grid_size: Edge of a voxel in the unit of the coordinates. Points sharing a
            voxel share one output feature.
        order: Space-filling curves; block ``i`` of an attending stage uses curve
            ``i mod len``.
        stride: Pooling stride between consecutive stages (any positive integer).
        enc_depths, enc_channels: Blocks and width of every encoder stage.
        enc_conv, enc_attn: Per encoder stage, whether its blocks convolve and whether
            they attend. A stage may do both (the PTv3 block) or neither (a layer norm).
        enc_num_head, enc_patch_size, enc_rope_freq: Heads, attention patch size and
            PointROPE base of every encoder stage; read by attending stages only. The
            head width must be divisible by 6. A head of width ``6 * k`` has ``k``
            frequencies per axis, from 1 down to ``base ** -((k - 1) / k)`` radians per
            voxel, so the base does nothing for a width of 6.
        dec_depths, dec_channels, dec_conv, dec_attn, dec_num_head, dec_patch_size,
        dec_rope_freq: The same for the decoder stages, finest first. A decoder stage can
            attend only where the encoder stage of that resolution does (the level is
            serialised by its encoder stage), with the same patch size.
        mlp_ratio: Hidden width of the block MLP relative to the stage width.
        qkv_bias, qk_scale: Bias of the query/key/value projection; attention scale
            (``None`` means ``head_width ** -0.5``).
        attn_drop, proj_drop, drop_path: Dropout on attention weights, on projections,
            and the maximum stochastic-depth rate (linear over the blocks; it only acts
            in attending blocks).
        pre_norm: Normalise before attention and MLP (``True``) or after the residual.
        shuffle_orders: Permute the curves among the blocks at every training step.
        pretrained: Released weights to start from: a name of
            ``grasppanda.models.backbones._pretrained.RELEASED`` or a file path. The
            encoder part of the file is loaded at construction and reported in the log.
        pretrained_mismatch: ``error`` refuses when a tensor of the encoder is absent
            from the file or has another shape there; ``skip`` keeps the initialisation
            of such tensors (the stem, when the number of input channels differs).
    """

    output = "dense"
    status = "weights-verified"
    source = "LitePT (Yue et al., CVPR 2026)"
    extras = ("spconv",)

    def __init__(
        self,
        in_channels: int = 6,
        grid_size: float = 0.02,
        order: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (36, 72, 144, 252, 504),
        enc_num_head: Sequence[int] = (2, 4, 8, 14, 28),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        enc_conv: Sequence[bool] = (True, True, True, False, False),
        enc_attn: Sequence[bool] = (False, False, False, True, True),
        enc_rope_freq: Sequence[float] = (100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths: Sequence[int] = (0, 0, 0, 0),
        dec_channels: Sequence[int] = (72, 72, 144, 252),
        dec_num_head: Sequence[int] = (4, 4, 8, 14),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        dec_conv: Sequence[bool] = (False, False, False, False),
        dec_attn: Sequence[bool] = (False, False, False, False),
        dec_rope_freq: Sequence[float] = (100.0, 100.0, 100.0, 100.0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        pretrained: str | None = None,
        pretrained_mismatch: Literal["error", "skip"] = "error",
    ) -> None:
        super().__init__()
        spconv = require("spconv.pytorch", extra="spconv-cu118", purpose="LitePT")
        stages = len(enc_depths)
        if in_channels < 1:
            raise ValueError("LitePT needs at least one input feature channel per point")
        if grid_size <= 0:
            raise ValueError(f"LitePT: grid_size must be positive, got {grid_size}")
        if len(order) == 0:
            raise ValueError("LitePT: order needs at least one space-filling curve")
        encoder = (enc_channels, enc_num_head, enc_patch_size, enc_conv, enc_attn, enc_rope_freq)
        if any(len(values) != stages for values in encoder) or len(stride) != stages - 1:
            raise ValueError(
                "LitePT: enc_depths, enc_channels, enc_num_head, enc_patch_size, enc_conv, "
                f"enc_attn and enc_rope_freq need one entry per stage ({stages}) and stride "
                f"one per transition ({stages - 1})"
            )
        decoder = (dec_depths, dec_channels, dec_num_head, dec_patch_size)
        decoder += (dec_conv, dec_attn, dec_rope_freq)
        if any(len(values) != stages - 1 for values in decoder):
            raise ValueError(
                "LitePT: dec_depths, dec_channels, dec_num_head, dec_patch_size, dec_conv, "
                f"dec_attn and dec_rope_freq need {stages - 1} entries, one per decoder stage"
            )
        if min(stride, default=1) < 1:
            raise ValueError(f"LitePT: stride must be positive, got {tuple(stride)}")
        for stage in range(stages - 1):
            if not dec_attn[stage]:
                continue
            if not enc_attn[stage]:
                raise ValueError(
                    f"LitePT: decoder stage {stage} attends but encoder stage {stage} does "
                    "not; a level is serialised for attention by its encoder stage only "
                    f"(set enc_attn[{stage}], or clear dec_attn[{stage}])"
                )
            if dec_patch_size[stage] != enc_patch_size[stage]:
                raise ValueError(
                    f"LitePT: stage {stage} has patch size {enc_patch_size[stage]} in the "
                    f"encoder and {dec_patch_size[stage]} in the decoder; a level has one "
                    "patch layout (the official model would silently use the encoder's)"
                )
        self.in_channels, self.grid_size = in_channels, grid_size
        self.order = tuple(order)
        self.shuffle_orders = shuffle_orders
        self.enc_attn = tuple(bool(flag) for flag in enc_attn)

        def blocks(
            modules: nn.ModuleDict,
            stage: int,
            depth: int,
            channels: int,
            heads: int,
            patch: int,
            freq: float,
            conv: bool,
            attn: bool,
            rates: Sequence[float],
        ) -> None:
            for index in range(depth):
                modules[f"block{index}"] = _Block(
                    spconv,
                    channels,
                    heads,
                    patch,
                    freq,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    attn_drop,
                    proj_drop,
                    rates[index],
                    pre_norm,
                    index % len(self.order),
                    f"stage{stage}",
                    bool(conv),
                    bool(attn),
                )

        self.embedding = nn.Module()
        stem = nn.Sequential()
        stem.add_module(
            "conv",
            spconv.SubMConv3d(
                in_channels,
                enc_channels[0],
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
        )
        stem.add_module("norm", _batch_norm(enc_channels[0]))
        stem.add_module("act", nn.GELU())
        self.embedding.stem = stem

        rates = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.enc = nn.ModuleDict()
        for stage in range(stages):
            modules = nn.ModuleDict()
            if stage > 0:
                modules["down"] = _Pooling(
                    enc_channels[stage - 1], enc_channels[stage], stride[stage - 1]
                )
            blocks(
                modules,
                stage,
                enc_depths[stage],
                enc_channels[stage],
                enc_num_head[stage],
                enc_patch_size[stage],
                enc_rope_freq[stage],
                enc_conv[stage],
                enc_attn[stage],
                rates[sum(enc_depths[:stage]) : sum(enc_depths[: stage + 1])],
            )
            self.enc[f"enc{stage}"] = modules

        rates = torch.linspace(0, drop_path, sum(dec_depths)).tolist()
        widths = [*dec_channels, enc_channels[-1]]
        self.dec = nn.ModuleDict()
        for stage in reversed(range(stages - 1)):
            modules = nn.ModuleDict()
            modules["up"] = _Unpooling(widths[stage + 1], enc_channels[stage], widths[stage])
            blocks(
                modules,
                stage,
                dec_depths[stage],
                widths[stage],
                dec_num_head[stage],
                dec_patch_size[stage],
                dec_rope_freq[stage],
                dec_conv[stage],
                dec_attn[stage],
                rates[sum(dec_depths[:stage]) : sum(dec_depths[: stage + 1])][::-1],
            )
            self.dec[f"dec{stage}"] = modules
        self.out_channels = widths[0]
        self.pretrained_report = (
            None
            if pretrained is None
            else load_pretrained(self, pretrained, mismatch=pretrained_mismatch)
        )

    # --------------------------------------------------------------------- forward

    def _encode(self, grid: Tensor, batch: Tensor, feat: Tensor) -> Tensor:
        """Run the network on voxels and return one feature vector per voxel."""
        # Looked up on use instead of stored: a module object as an attribute would make
        # the model impossible to deep-copy or pickle.
        spconv = require("spconv.pytorch", extra="spconv-cu118", purpose="LitePT")
        batch_size = int(batch[-1]) + 1
        shuffle = self.shuffle_orders and self.training
        stem = self.embedding.stem
        sparse = _sparse_conv(stem.conv, _sparse_tensor(spconv, feat, grid, batch, batch_size))
        level = _Level(
            feat=stem.act(stem.norm(sparse.features)),
            grid=grid,
            batch=batch,
            sizes=torch.bincount(batch, minlength=batch_size).tolist(),
            sparse=sparse,
        )
        for attends, modules in zip(self.enc_attn, self.enc.values()):
            if "down" in modules:
                level = modules["down"](spconv, level)
            if attends:
                level.serialize(self.order, shuffle)
            for key, module in modules.items():
                if key != "down":
                    module(level)
        for modules in self.dec.values():
            level = modules["up"](level)
            for key, module in modules.items():
                if key != "up":
                    module(level)
        return level.feat

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="LitePT")
        if min(cloud_sizes(points)) == 0:
            raise ContractError("LitePT received an empty cloud; every cloud needs points")
        grid, batch, first, inverse = _voxels(points, self.grid_size)
        out = self._encode(grid, batch, feats[first])
        return PointFeatures(out[inverse], points)
