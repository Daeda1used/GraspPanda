"""Point Transformer V3 and its self-supervised descendants.

Papers:

* Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024.
* Wu et al., "Sonata: Self-Supervised Learning of Reliable Point Representations",
  CVPR 2025.
* Zhang et al., "Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial
  Representations", NeurIPS 2025.
* Zhang et al., "Utonia: Toward One Encoder for All Point Clouds", ICML 2026.

PTv3 replaces neighbourhood search by *serialisation*: voxels are sorted along
space-filling curves (:mod:`grasppanda.ops.serialization`) and attention runs inside
consecutive patches of the sorted sequence. A sparse convolution before every block
supplies the positional encoding; pooling halves the grid, unpooling adds the coarse
features back through the pooling map. Three registered backbones share that engine:

* :class:`PTv3`: the CVPR 2024 encoder-decoder (Pointcept ``PT-v3m1``).
* :class:`PTv3Sonata`: the variant behind the Sonata and Concerto weights
  (``PT-v3m2``): linear stem, layer norm throughout, grid pooling, optional layer scale.
  The released checkpoints are encoder-only; dense features are then the encoder
  stages concatenated through the pooling maps, as the authors do for linear probing.
* :class:`PTv3Utonia`: ``PT-v3m3``, the Sonata variant plus rotary position embedding
  of the point coordinates.

All three return ``dense`` features. Points are quantised to a voxel grid owned by the
backbone (``grid_size``), the network runs on one representative point per voxel (the
first one, where the official data pipeline picks one at random), and voxel features
are mapped back to exactly the points that produced them.

Deliberate differences from the official code, each for the backbone contract:

* **No flash-attention.** Attention uses PyTorch's ``scaled_dot_product_attention`` in
  the precision of the input, with the patch layout of the official flash path (fixed
  patch size, a cloud smaller than a patch forms one shorter patch). The official
  fallback without flash-attention instead shrinks the patch size to the smallest
  cloud of the batch, which makes a cloud's result depend on its batch neighbours.
* **Batch independence.** The serialisation depth is taken per cloud. Officially it is
  the depth of the largest cloud in the batch, and the Hilbert curve depends on it.
  For a single cloud the two agree.
* **Determinism.** ``shuffle_orders`` permutes the curves among the blocks in training
  mode only. Officially the shuffle also runs in evaluation mode.

One quirk is kept because released weights were trained with it: in ``PT-v3m1`` the
first positional-encoding convolution of every decoder stage reads the projected skip
features only, not their sum with the unpooled features (``fuse_before_decoder_cpe``).
Sonata fixed this, so :class:`PTv3Sonata` always fuses first.

Provenance. Written for this code base from the papers and from the behaviour of the
official implementations: Pointcept and PointTransformerV3 (MIT licence), and the
``sonata`` / ``concerto`` / ``utonia`` packages (Apache-2.0 code). The module tree and
parameter names (``embedding.stem``, ``enc.enc0.block0.cpe.0`` ...) follow them on
purpose, so that official state dicts load without renaming. The Sonata, Concerto and
Utonia *weights* are released under CC-BY-NC 4.0: loading them is the user's decision
and makes the resulting model non-commercial.

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
from grasppanda.ops import SerializationOrder, serialization_code, voxelize
from grasppanda.ops.sparse import check_cpu_convolution
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet
from grasppanda.utils.imports import require

from ._packed import check_features, cloud_sizes

__all__ = ["PTv3", "PTv3Sonata", "PTv3Utonia"]

#: Margin the official code adds to the extent of the sparse tensor.
_SPARSE_PAD = 96

#: ``(pad, unpad, runs)`` of one level for one patch size, see :func:`_patch_layout`.
_Layout = tuple[Tensor, Tensor, list[tuple[int, int]]]


@dataclass(eq=False)
class _Level:
    """One resolution of the hierarchy: voxels, their order along each curve, features."""

    feat: Tensor  # (n, C)
    coord: Tensor  # (n, 3) representative (level 0) or mean (pooled) coordinates
    grid: Tensor  # (n, 3) non-negative integer voxel coordinates
    batch: Tensor  # (n,) cloud of every voxel, non-decreasing
    sizes: list[int]  # voxels per cloud
    depth: int  # bits per axis of the serialisation codes
    code: Tensor  # (K, n) sort key along each of the K curves
    order: Tensor  # (K, n) rows in curve order
    inverse: Tensor  # (K, n) position of every row in curve order
    sparse: Any  # spconv tensor over these voxels; caches the convolution index pairs
    parent: _Level | None = None
    cluster: Tensor | None = None  # (n_parent,) row of this level every parent row fell into
    conv_feat: Tensor | None = None  # what the next positional convolution reads, if not feat
    layouts: dict[int, _Layout] = field(default_factory=dict)

    def layout(self, patch_size: int) -> _Layout:
        if patch_size not in self.layouts:
            self.layouts[patch_size] = _patch_layout(self.sizes, patch_size, self.feat.device)
        return self.layouts[patch_size]


def _patch_layout(sizes: Sequence[int], patch_size: int, device: torch.device) -> _Layout:
    """Split every cloud's sequence into attention patches.

    A cloud with at most ``patch_size`` voxels is one patch. A larger cloud is cut into
    patches of exactly ``patch_size``; when the last one is short it is completed with
    the voxels just before it, whose results from that patch are then discarded.

    Returns:
        ``pad``: for every slot of the padded sequence, the sequence position it holds;
        ``unpad``: for every sequence position, its own slot; ``runs``: consecutive
        slots that share a patch length, as ``(slots, patch length)``.
    """
    pad, unpad, runs = [], [], []
    start = slots = 0
    for size in sizes:
        positions = torch.arange(start, start + size, device=device)
        unpad.append(torch.arange(slots, slots + size, device=device))
        remainder = size % patch_size
        if size > patch_size and remainder:
            positions = torch.cat([positions, positions[size - patch_size : size - remainder]])
        length = patch_size if size > patch_size else size
        if runs and runs[-1][1] == length:
            runs[-1] = (runs[-1][0] + len(positions), length)
        else:
            runs.append((len(positions), length))
        pad.append(positions)
        start += size
        slots += len(positions)
    return torch.cat(pad), torch.cat(unpad), runs


def _cloud_depths(grid: Tensor, batch: Tensor, batch_size: int, spare: int) -> Tensor:
    """Bits per axis of every cloud's curve: the bit length of ``largest coordinate + spare``.

    ``spare`` is 0 in the original PTv3 release, which the PT-v3m1 weights were trained
    with, and 1 since Sonata. The two differ only when the largest coordinate plus one
    is a power of two; the Hilbert curve then runs one level deeper.
    """
    largest = torch.zeros(batch_size, dtype=torch.long, device=grid.device)
    largest = largest.scatter_reduce(0, batch, grid.amax(dim=1), reduce="amax", include_self=True)
    depths = [(int(value) + spare).bit_length() for value in largest]
    return torch.tensor(depths, dtype=torch.long, device=grid.device)


def _codes(
    grid: Tensor, batch: Tensor, depths: Tensor, orders: Sequence[SerializationOrder]
) -> Tensor:
    """``(K, n)`` sort keys: each cloud on a curve of its own depth, clouds kept apart."""
    row_depth = depths[batch]
    code = torch.empty(len(orders), len(grid), dtype=torch.long, device=grid.device)
    for depth in torch.unique(depths).tolist():
        rows = torch.nonzero(row_depth == depth).flatten()
        for index, order in enumerate(orders):
            code[index, rows] = serialization_code(grid[rows], max(depth, 1), order)
    return code | (batch << (3 * int(depths.max())))


def _sort(code: Tensor, shuffle: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Order and inverse order along every curve, optionally permuting the curves."""
    if shuffle:
        code = code[torch.randperm(code.shape[0], device=code.device)]
    order = torch.argsort(code, dim=1)
    rows = torch.arange(code.shape[1], device=code.device).expand_as(order)
    inverse = torch.zeros_like(order).scatter_(1, order, rows)
    return code, order, inverse


def _segment(values: Tensor, cluster: Tensor, count: int, reduce: str) -> Tensor:
    """Reduce the rows of ``values`` that share a cluster (``amax`` or ``mean``)."""
    out = values.new_zeros(count, values.shape[1])
    index = cluster[:, None].expand_as(values)
    return out.scatter_reduce(0, index, values, reduce=reduce, include_self=False)


def _first_rows(cluster: Tensor, count: int) -> Tensor:
    """Lowest row of every cluster."""
    rows = torch.arange(len(cluster), device=cluster.device)
    first = torch.full((count,), len(cluster), dtype=torch.long, device=cluster.device)
    return first.scatter_reduce(0, cluster, rows, reduce="amin", include_self=True)


class _DropPath(nn.Module):
    """Stochastic depth, drawn per voxel as in the official implementation."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        return x * x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep


class _LayerScale(nn.Module):
    def __init__(self, channels: int, init_value: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _RPE(nn.Module):
    """Learned bias per head, looked up from the clamped grid offset along each axis."""

    def __init__(self, patch_size: int, num_heads: int) -> None:
        super().__init__()
        self.bound = int((4 * patch_size) ** (1 / 3) * 2)
        self.count = 2 * self.bound + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.count, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, offset: Tensor) -> Tensor:
        """``(P, L, L, 3)`` integer offsets to a ``(P, H, L, L)`` attention bias."""
        axis = torch.arange(3, device=offset.device) * self.count
        index = offset.clamp(-self.bound, self.bound) + self.bound + axis
        bias = self.rpe_table[index.reshape(-1)].reshape(*index.shape, -1).sum(dim=3)
        return bias.permute(0, 3, 1, 2)


class _RoPE(nn.Module):
    """Rotary position embedding of metric coordinates (Utonia).

    A head's channels are split into three chunks, one per axis; within a chunk, pairs
    of channels are rotated by ``coordinate × frequency``.
    """

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        if head_dim % 6:
            raise ValueError(
                f"rotary embedding needs a head width divisible by 6 (three axes, channel "
                f"pairs), got {head_dim}"
            )
        self.chunk = head_dim // 3
        steps = torch.arange(0, self.chunk, 2).float() / self.chunk
        self.register_buffer("inv_freq", 1.0 / (base**steps))

    def forward(self, q: Tensor, k: Tensor, coord: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate ``(n, H, D)`` queries and keys by their ``(n, 3)`` coordinates."""
        angle = coord[:, :, None] * self.inv_freq  # (n, 3, D / 6)
        angle = torch.cat([angle, angle], dim=-1).flatten(1)[:, None, :]  # (n, 1, D)
        cos, sin = angle.cos(), angle.sin()
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        chunks = x.unflatten(-1, (3, self.chunk))
        first, second = chunks.chunk(2, dim=-1)
        turned = torch.cat([-second, first], dim=-1).flatten(-2)
        return x * cos + turned * sin


class _SerializedAttention(nn.Module):
    """Multi-head self-attention inside patches of one serialised order."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        order_index: int,
        enable_rpe: bool,
        rope_base: float | None,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"{channels} channels cannot be split into {num_heads} heads")
        self.channels, self.num_heads, self.patch_size = channels, num_heads, patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe = _RPE(patch_size, num_heads) if enable_rpe else None
        self.rope = _RoPE(channels // num_heads, rope_base) if rope_base is not None else None

    def forward(self, feat: Tensor, level: _Level) -> Tensor:
        heads, width = self.num_heads, self.channels // self.num_heads
        pad, unpad, runs = level.layout(self.patch_size)
        index = self.order_index % level.order.shape[0]
        order = level.order[index][pad]  # row held by every slot of the padded sequence
        slot = unpad[level.inverse[index]]  # slot every row reads its result from
        q, k, v = self.qkv(feat)[order].reshape(-1, 3, heads, width).unbind(dim=1)
        if self.rope is not None:
            q, k = self.rope(q, k, level.coord[order])
        grid = level.grid[order] if self.rpe is not None else None
        out, start = [], 0
        for slots, length in runs:
            shape = (slots // length, length, heads, width)
            queries, keys, values = (
                part[start : start + slots].reshape(shape).transpose(1, 2) for part in (q, k, v)
            )
            bias = None
            if grid is not None:
                cell = grid[start : start + slots].reshape(-1, length, 3)
                bias = self.rpe(cell[:, :, None, :] - cell[:, None, :, :]).to(queries.dtype)
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=bias,
                dropout_p=self.attn_drop if self.training else 0.0,
                scale=self.scale,
            )
            out.append(attended.transpose(1, 2).reshape(slots, self.channels))
            start += slots
        return self.proj_drop(self.proj(torch.cat(out)[slot]))


class _MLP(nn.Module):
    def __init__(self, channels: int, hidden: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Block(nn.Module):
    """Positional sparse convolution, then attention and an MLP, each with a residual."""

    def __init__(
        self,
        spconv: Any,
        channels: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        layer_scale: float | None,
        pre_norm: bool,
        order_index: int,
        indice_key: str,
        enable_rpe: bool,
        rope_base: float | None,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = nn.Sequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=indice_key),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.norm1 = nn.Sequential(nn.LayerNorm(channels))
        self.attn = _SerializedAttention(
            channels,
            num_heads,
            patch_size,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            order_index,
            enable_rpe,
            rope_base,
        )
        self.norm2 = nn.Sequential(nn.LayerNorm(channels))
        self.mlp = nn.Sequential(_MLP(channels, int(channels * mlp_ratio), proj_drop))
        scaled = layer_scale is not None
        self.ls1 = nn.Sequential(_LayerScale(channels, layer_scale) if scaled else nn.Identity())
        self.ls2 = nn.Sequential(_LayerScale(channels, layer_scale) if scaled else nn.Identity())
        self.drop_path = _DropPath(drop_path)

    def forward(self, level: _Level) -> None:
        feat = level.feat
        source = feat if level.conv_feat is None else level.conv_feat
        level.conv_feat = None
        level.sparse = _sparse_conv(self.cpe[0], level.sparse.replace_feature(source))
        feat = feat + self.cpe[2](self.cpe[1](level.sparse.features))

        branch = self.norm1(feat) if self.pre_norm else feat
        feat = feat + self.drop_path(self.ls1(self.attn(branch, level)))
        if not self.pre_norm:
            feat = self.norm1(feat)

        branch = self.norm2(feat) if self.pre_norm else feat
        feat = feat + self.drop_path(self.ls2(self.mlp(branch)))
        if not self.pre_norm:
            feat = self.norm2(feat)
        level.feat = feat


class _Pooling(nn.Module):
    """Project, then max-pool the voxels that fall into the same coarser cell.

    ``mode="serialized"`` (PT-v3m1) merges voxels whose codes agree after dropping the
    low bits and keeps the shifted codes; ``mode="grid"`` (PT-v3m2) divides the grid
    coordinates by the stride and serialises the coarse grid again.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm: nn.Module,
        mode: Literal["serialized", "grid"],
    ) -> None:
        super().__init__()
        if mode == "serialized" and (stride < 2 or stride & (stride - 1)):
            raise ValueError(f"serialised pooling needs a power-of-two stride, got {stride}")
        if stride < 1:
            raise ValueError(f"stride must be positive, got {stride}")
        self.stride, self.mode = stride, mode
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.Sequential(norm)
        self.act = nn.Sequential(nn.GELU())

    def forward(
        self, spconv: Any, level: _Level, orders: Sequence[SerializationOrder], shuffle: bool
    ) -> _Level:
        batch_size = len(level.sizes)
        if self.mode == "serialized":
            bits = (self.stride - 1).bit_length()
            if bits > level.depth:
                bits = 0
            code = level.code >> (3 * bits)
            _, cluster = torch.unique(code[0], sorted=True, return_inverse=True)
            count = int(cluster.max()) + 1
            first = _first_rows(cluster, count)
            grid, batch, depth = level.grid[first] >> bits, level.batch[first], level.depth - bits
            code = code[:, first]
        else:
            keys = torch.cat([level.batch[:, None], level.grid // self.stride], dim=1)
            cells, cluster = torch.unique(keys, dim=0, return_inverse=True)
            count = len(cells)
            batch, grid = cells[:, 0], cells[:, 1:]
            depths = _cloud_depths(grid, batch, batch_size, spare=1)
            depth, code = int(depths.max()), _codes(grid, batch, depths, orders)
        code, order, inverse = _sort(code, shuffle)
        feat = self.act(self.norm(_segment(self.proj(level.feat), cluster, count, "amax")))
        return _Level(
            feat=feat,
            coord=_segment(level.coord, cluster, count, "mean"),
            grid=grid,
            batch=batch,
            sizes=torch.bincount(batch, minlength=batch_size).tolist(),
            depth=depth,
            code=code,
            order=order,
            inverse=inverse,
            sparse=_sparse_tensor(spconv, feat, grid, batch, batch_size),
            parent=level,
            cluster=cluster,
        )


class _Unpooling(nn.Module):
    """Project the coarse and the skip features and add them through the pooling map."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, norms: tuple[nn.Module, ...]
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels), norms[0], nn.GELU())
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels), norms[1], nn.GELU())

    def forward(self, level: _Level, fuse_before_cpe: bool) -> _Level:
        parent = level.parent
        skip = self.proj_skip(parent.feat)
        parent.feat = skip + self.proj(level.feat)[level.cluster]
        parent.conv_feat = None if fuse_before_cpe else skip
        return parent


def _sparse_tensor(spconv: Any, feat: Tensor, grid: Tensor, batch: Tensor, batch_size: int) -> Any:
    indices = torch.cat([batch[:, None], grid], dim=1).int().contiguous()
    shape = (grid.amax(dim=0) + _SPARSE_PAD).tolist()
    return spconv.SparseConvTensor(feat, indices, shape, batch_size)


def _sparse_conv(conv: nn.Module, tensor: Any) -> Any:
    """Apply a sparse convolution on whatever device the tensor lives on.

    On the GPU this is a plain call. On the CPU spconv 2.3 needs two settings changed
    for the duration of the call: a CUDA build defaults small kernels to an algorithm
    that only exists on the GPU, so the layer is switched to the native one (the weight
    layout is the same for both); and the native CPU kernel cannot fuse the bias, which
    spconv only attempts in evaluation mode, so the layer is put in training mode, where
    the bias is added afterwards. Neither changes the result.

    The CPU path first asks the sparse facade whether the installed spconv can be trusted
    on the CPU at all (one comparison per process; the CPU-only wheel is wrong with more
    than one OpenMP thread).
    """
    if tensor.features.is_cuda:
        return conv(tensor)
    check_cpu_convolution()
    native = require("spconv.core", extra="spconv-cu118", purpose="PTv3").ConvAlgo.Native
    configured = conv.algo, conv.training
    conv.algo, conv.training = native, True
    try:
        return conv(tensor)
    finally:
        conv.algo, conv.training = configured


class _PTv3(Backbone):
    """Shared engine of the PTv3 family; see the module docstring."""

    output = "dense"

    def __init__(
        self,
        *,
        variant: Literal["m1", "m2"],
        in_channels: int,
        grid_size: float,
        order: Sequence[SerializationOrder],
        stride: Sequence[int],
        enc_depths: Sequence[int],
        enc_channels: Sequence[int],
        enc_num_head: Sequence[int],
        enc_patch_size: Sequence[int],
        dec_depths: Sequence[int],
        dec_channels: Sequence[int],
        dec_num_head: Sequence[int],
        dec_patch_size: Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        layer_scale: float | None,
        pre_norm: bool,
        shuffle_orders: bool,
        enable_rpe: bool,
        rope_base: float | None,
        mask_token: bool,
        enc_mode: bool,
        concat_stages: int | None,
        fuse_before_decoder_cpe: bool,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        spconv = require("spconv.pytorch", extra="spconv-cu118", purpose=name)
        stages = len(enc_depths)
        if in_channels < 1:
            raise ValueError(f"{name} needs at least one input feature channel per point")
        if grid_size <= 0:
            raise ValueError(f"{name}: grid_size must be positive, got {grid_size}")
        if len(order) == 0:
            raise ValueError(f"{name}: order needs at least one space-filling curve")
        encoder = (enc_channels, enc_num_head, enc_patch_size)
        if any(len(values) != stages for values in encoder) or len(stride) != stages - 1:
            raise ValueError(
                f"{name}: enc_depths, enc_channels, enc_num_head and enc_patch_size need one "
                f"entry per stage ({stages}) and stride one per transition ({stages - 1})"
            )
        decoder = (dec_depths, dec_channels, dec_num_head, dec_patch_size)
        if not enc_mode and any(len(values) != stages - 1 for values in decoder):
            raise ValueError(
                f"{name}: dec_depths, dec_channels, dec_num_head and dec_patch_size need "
                f"{stages - 1} entries, one per decoder stage"
            )
        if concat_stages is None:
            concat_stages = stages
        if enc_mode and not 1 <= concat_stages <= stages:
            raise ValueError(f"{name}: concat_stages must be in 1..{stages}, got {concat_stages}")
        self.in_channels, self.grid_size = in_channels, grid_size
        self.order = tuple(order)
        self.shuffle_orders, self.enc_mode = shuffle_orders, enc_mode
        self.concat_stages = concat_stages
        self.fuse_before_decoder_cpe = fuse_before_decoder_cpe
        self.num_stages = stages
        legacy = variant == "m1"

        def norm(channels: int) -> nn.Module:
            if legacy:
                return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
            return nn.LayerNorm(channels)

        def block(
            stage: int, index: int, channels: int, heads: int, patch: int, rate: float
        ) -> _Block:
            return _Block(
                spconv,
                channels,
                heads,
                patch,
                mlp_ratio,
                qkv_bias,
                qk_scale,
                attn_drop,
                proj_drop,
                rate,
                layer_scale,
                pre_norm,
                index % len(self.order),
                f"stage{stage}",
                enable_rpe,
                rope_base,
            )

        self.embedding = nn.Module()
        stem = nn.Sequential()
        if legacy:
            conv = spconv.SubMConv3d(
                in_channels,
                enc_channels[0],
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
            stem.add_module("conv", conv)
        else:
            stem.add_module("linear", nn.Linear(in_channels, enc_channels[0]))
        stem.add_module("norm", norm(enc_channels[0]))
        stem.add_module("act", nn.GELU())
        self.embedding.stem = stem
        if mask_token:
            # Used only by the masked pre-training objective; kept so that released
            # encoder checkpoints load without an unexpected key.
            self.embedding.mask_token = nn.Parameter(torch.zeros(1, enc_channels[0]))

        rates = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.enc = nn.ModuleDict()
        for stage in range(stages):
            modules = nn.ModuleDict()
            if stage > 0:
                modules["down"] = _Pooling(
                    enc_channels[stage - 1],
                    enc_channels[stage],
                    stride[stage - 1],
                    norm(enc_channels[stage]),
                    "serialized" if legacy else "grid",
                )
            stage_rates = rates[sum(enc_depths[:stage]) : sum(enc_depths[: stage + 1])]
            for index in range(enc_depths[stage]):
                modules[f"block{index}"] = block(
                    stage,
                    index,
                    enc_channels[stage],
                    enc_num_head[stage],
                    enc_patch_size[stage],
                    stage_rates[index],
                )
            self.enc[f"enc{stage}"] = modules

        if enc_mode:
            self.out_channels = sum(enc_channels[stages - concat_stages :])
        else:
            rates = torch.linspace(0, drop_path, sum(dec_depths)).tolist()
            widths = [*dec_channels, enc_channels[-1]]
            self.dec = nn.ModuleDict()
            for stage in reversed(range(stages - 1)):
                modules = nn.ModuleDict()
                modules["up"] = _Unpooling(
                    widths[stage + 1],
                    enc_channels[stage],
                    widths[stage],
                    (norm(widths[stage]), norm(widths[stage])),
                )
                stage_rates = rates[sum(dec_depths[:stage]) : sum(dec_depths[: stage + 1])][::-1]
                for index in range(dec_depths[stage]):
                    modules[f"block{index}"] = block(
                        stage,
                        index,
                        widths[stage],
                        dec_num_head[stage],
                        dec_patch_size[stage],
                        stage_rates[index],
                    )
                self.dec[f"dec{stage}"] = modules
            self.out_channels = widths[0]
        if not legacy:
            self.apply(self._init_weights)

    @property
    def _spconv(self) -> Any:
        # Looked up on use instead of stored: a module object as an attribute would make
        # the model impossible to deep-copy or pickle.
        return require("spconv.pytorch", extra="spconv-cu118", purpose=type(self).__name__)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, self._spconv.SubMConv3d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- forward

    def _voxels(self, points: PointSet) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Quantise per cloud. Returns grid, batch, representative row and inverse map."""
        # Quantise in double precision, like the NumPy grid sampling of the official
        # pipeline: in float32 a point that sits on a cell boundary can land on either side.
        cells = voxelize(points.coords.double(), self.grid_size, points.batch_index)
        batch, grid = cells.voxels[:, 0], cells.voxels[:, 1:]
        # Shift every cloud to start at cell 0, as the official grid sampling does, so
        # that a cloud's codes do not depend on where the other clouds sit.
        lowest = torch.zeros(points.batch_size, 3, dtype=torch.long, device=grid.device)
        index = batch[:, None].expand_as(grid)
        lowest = lowest.scatter_reduce(0, index, grid, reduce="amin", include_self=False)
        return grid - lowest[batch], batch, cells.first, cells.inverse

    def _encode(self, coord: Tensor, grid: Tensor, batch: Tensor, feat: Tensor) -> Tensor:
        """Run the network on voxels and return one feature vector per voxel."""
        spconv = self._spconv
        batch_size = int(batch[-1]) + 1
        shuffle = self.shuffle_orders and self.training
        stem = self.embedding.stem
        linear_stem = isinstance(stem[0], nn.Linear)
        depths = _cloud_depths(grid, batch, batch_size, spare=int(linear_stem))
        code, order, inverse = _sort(_codes(grid, batch, depths, self.order), shuffle)
        if linear_stem:
            feat = stem(feat)
        sparse = _sparse_tensor(spconv, feat, grid, batch, batch_size)
        if not linear_stem:
            sparse = _sparse_conv(stem.conv, sparse)
            feat = stem.act(stem.norm(sparse.features))
        level = _Level(
            feat=feat,
            coord=coord,
            grid=grid,
            batch=batch,
            sizes=torch.bincount(batch, minlength=batch_size).tolist(),
            depth=int(depths.max()),
            code=code,
            order=order,
            inverse=inverse,
            sparse=sparse,
        )
        for modules in self.enc.values():
            if "down" in modules:
                level = modules["down"](spconv, level, self.order, shuffle)
            for key, module in modules.items():
                if key != "down":
                    module(level)
        if self.enc_mode:
            return self._upcast(level)
        for modules in self.dec.values():
            level = modules["up"](level, self.fuse_before_decoder_cpe)
            for key, module in modules.items():
                if key != "up":
                    module(level)
        return level.feat

    def _upcast(self, level: _Level) -> Tensor:
        """Concatenate the deepest stages through the pooling maps, then copy down."""
        feat = level.feat
        for step in range(self.num_stages - 1):
            feat = feat[level.cluster]
            level = level.parent
            if step < self.concat_stages - 1:
                feat = torch.cat([level.feat, feat], dim=-1)
        return feat

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        if min(cloud_sizes(points)) == 0:
            raise ContractError(f"{name} received an empty cloud; every cloud needs points")
        grid, batch, first, inverse = self._voxels(points)
        out = self._encode(points.coords[first], grid, batch, feats[first])
        return PointFeatures(out[inverse], points)


@BACKBONES.register()
class PTv3(_PTv3):
    """Point Transformer V3 encoder-decoder (Pointcept ``PT-v3m1``).

    The defaults are the published ScanNet semantic-segmentation model
    (``semseg-pt-v3m1-0-base``): five encoder stages of 2-2-2-6-2 blocks with 32 to 512
    channels, four decoder stages, patches of 1024 voxels, all four curves, 0.02 m
    voxels, 6 input channels (colour and normal) and 64 output channels. For tabletop
    clouds a finer grid is usual (GSNet voxelises at 0.005 m).

    Args:
        in_channels: Feature channels per point.
        grid_size: Edge of a voxel in the unit of the coordinates. Points sharing a
            voxel share one output feature.
        order: Space-filling curves; block ``i`` of a stage uses curve ``i mod len``.
        stride: Pooling stride between consecutive stages (powers of two).
        enc_depths, enc_channels, enc_num_head, enc_patch_size: Blocks, width, heads and
            attention patch size of every encoder stage.
        dec_depths, dec_channels, dec_num_head, dec_patch_size: The same for the decoder
            stages, finest first.
        mlp_ratio: Hidden width of the block MLP relative to the stage width.
        qkv_bias, qk_scale: Bias of the query/key/value projection; attention scale
            (``None`` means ``head_width ** -0.5``).
        attn_drop, proj_drop, drop_path: Dropout on attention weights, on projections,
            and the maximum stochastic-depth rate (linear over the blocks).
        pre_norm: Normalise before attention and MLP (``True``) or after the residual.
        shuffle_orders: Permute the curves among the blocks at every training step.
        enable_rpe: Add a learned relative-position bias to the attention logits.
        fuse_before_decoder_cpe: ``False`` reproduces the official model, whose first
            positional convolution of a decoder stage reads the skip features only.
    """

    status = "weights-verified"
    source = "Point Transformer V3 (Wu et al., CVPR 2024)"
    extras = ("spconv",)

    def __init__(
        self,
        in_channels: int = 6,
        grid_size: float = 0.02,
        order: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (64, 64, 128, 256),
        dec_num_head: Sequence[int] = (4, 4, 8, 16),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        fuse_before_decoder_cpe: bool = False,
    ) -> None:
        super().__init__(
            variant="m1",
            in_channels=in_channels,
            grid_size=grid_size,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale=None,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            rope_base=None,
            mask_token=False,
            enc_mode=False,
            concat_stages=1,
            fuse_before_decoder_cpe=fuse_before_decoder_cpe,
        )


@BACKBONES.register()
class PTv3Sonata(_PTv3):
    """The PTv3 variant of Sonata and Concerto (Pointcept ``PT-v3m2``).

    The defaults are the encoder stored in the released ``sonata`` and
    ``concerto_base`` checkpoints: encoder only, 3-3-3-12-3 blocks with 48 to 512
    channels, 9 input channels (coordinates, colour, normal: set the ones you do not
    have to zero, as the authors advise), 0.02 m voxels. The dense output concatenates
    all five encoder stages through the pooling maps (1232 channels), which is the
    authors' linear-probing protocol. Other released sizes differ in ``enc_channels``,
    ``enc_depths`` and ``enc_num_head`` only; ``docs/model_zoo_backbones.md`` lists them.

    Args:
        in_channels, grid_size, order, stride, enc_depths, enc_channels, enc_num_head,
        enc_patch_size, mlp_ratio, qkv_bias, qk_scale, attn_drop, proj_drop, drop_path,
        pre_norm, shuffle_orders, enable_rpe: As in :class:`PTv3`; ``stride`` may be any
            integer here.
        dec_depths, dec_channels, dec_num_head, dec_patch_size: Decoder stages, used
            when ``enc_mode`` is off. The defaults are the authors' fine-tuning decoder.
        layer_scale: Initial value of a learned per-channel scale on both residual
            branches; ``None`` disables it.
        mask_token: Keep the mask token of the pre-training objective as a parameter.
            It is never used here; released checkpoints contain it.
        enc_mode: Encoder only. Off adds the decoder, whose weights no released
            self-supervised checkpoint provides.
        concat_stages: With ``enc_mode``, how many of the deepest stages are
            concatenated; shallower ones only pass the result down to the input voxels.
            ``None`` concatenates all of them.
    """

    status = "weights-verified"
    source = "Sonata (Wu et al., CVPR 2025), Concerto (Zhang et al., NeurIPS 2025)"
    extras = ("spconv",)

    def __init__(
        self,
        in_channels: int = 9,
        grid_size: float = 0.02,
        order: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (3, 3, 3, 12, 3),
        enc_channels: Sequence[int] = (48, 96, 192, 384, 512),
        enc_num_head: Sequence[int] = (3, 6, 12, 24, 32),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (64, 96, 192, 384),
        dec_num_head: Sequence[int] = (4, 6, 12, 24),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        layer_scale: float | None = None,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        mask_token: bool = True,
        enc_mode: bool = True,
        concat_stages: int | None = None,
    ) -> None:
        super().__init__(
            variant="m2",
            in_channels=in_channels,
            grid_size=grid_size,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale=layer_scale,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            rope_base=None,
            mask_token=mask_token,
            enc_mode=enc_mode,
            concat_stages=concat_stages,
            fuse_before_decoder_cpe=True,
        )


@BACKBONES.register()
class PTv3Utonia(_PTv3):
    """The PTv3 variant of Utonia (Pointcept ``PT-v3m3``): :class:`PTv3Sonata` plus RoPE.

    The defaults are the encoder stored in the released ``utonia`` checkpoint: 54 to 576
    channels, rotary embedding with base 10 of the point coordinates in every attention
    layer, 0.01 m voxels as in the authors' inference pipeline, 9 input channels, all
    five stages concatenated (1386 channels). Rotary embedding makes the features depend
    on the scale and origin of the coordinates: the authors centre each cloud and expose
    a scale factor in their data pipeline, which is the caller's job here.

    Args:
        rope_base: Base of the rotary frequencies. Every head width must be divisible
            by 6.
        Others: As in :class:`PTv3Sonata`.
    """

    status = "weights-verified"
    source = "Utonia (Zhang et al., ICML 2026)"
    extras = ("spconv",)

    def __init__(
        self,
        in_channels: int = 9,
        grid_size: float = 0.01,
        order: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (3, 3, 3, 12, 3),
        enc_channels: Sequence[int] = (54, 108, 216, 432, 576),
        enc_num_head: Sequence[int] = (3, 6, 12, 24, 32),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (54, 108, 216, 432),
        dec_num_head: Sequence[int] = (3, 6, 12, 24),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        layer_scale: float | None = None,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        rope_base: float = 10.0,
        mask_token: bool = True,
        enc_mode: bool = True,
        concat_stages: int | None = None,
    ) -> None:
        super().__init__(
            variant="m2",
            in_channels=in_channels,
            grid_size=grid_size,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale=layer_scale,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            rope_base=rope_base,
            mask_token=mask_token,
            enc_mode=enc_mode,
            concat_stages=concat_stages,
            fuse_before_decoder_cpe=True,
        )
