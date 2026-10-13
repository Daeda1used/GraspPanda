# Copyright (c) 2026 Kadir Yilmaz
# Adapted for GraspPanda from https://github.com/YilmazKadir/Volt @ 8dcd608 (MIT)
"""Volt: a plain vision transformer over volumetric patches.

Paper: Yilmaz et al., "Volume Transformer" (Volt), ECCV 2026 according to its arXiv
record (arXiv:2604.19609).

The cloud is voxelised; every block of ``kernel_size``³ voxels that holds a point
becomes one token (a linear layer over the block's voxel slots, empty ones zero); all
tokens of a cloud attend to each other with a rotary position embedding of the block
index along x, y and z; one linear layer per slot turns a token back into its voxels
(a transposed convolution with stride = kernel). No hierarchy, no windows, no sparse
convolution. The output is one feature per input point (``dense``), read from the
point's voxel.

Provenance. Tokeniser, blocks, rotary embedding and decoder follow
``pointcept/models/volt/volt_base.py`` and ``decoder.py`` of the official repository,
parameter names included, so that the released checkpoints fit
(:func:`convert_volt_checkpoint`). What differs: the reference attends through
``flash_attn`` on half-precision tensors, here each cloud goes through
``torch.nn.functional.scaled_dot_product_attention`` in the precision of the model; the
rotary angles are computed for the indices at hand in place of a table with a maximum
grid size; ``timm``'s ``Mlp``, ``DropPath`` and ``LayerScale`` are written out. The
authors report that Volt overfits without their recipe (strong augmentation,
distillation from a sparse U-Net, EMA); a run from random weights tests that claim, not
the architecture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointFeatures, PointSet

from ._grid import grid_cells
from ._packed import check_features

__all__ = ["Volt", "convert_volt_checkpoint"]


class _Tokenizer(nn.Module):
    """One token per occupied block of ``kernel_size``³ voxels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = nn.Linear(kernel_size**3 * in_channels, out_channels)

    def forward(self, feats: Tensor, batch: Tensor, cells: Tensor) -> tuple[Tensor, ...]:
        size = self.kernel_size
        blocks = torch.cat([batch[:, None], cells // size], dim=1)
        blocks, inverse = torch.unique(blocks, dim=0, sorted=True, return_inverse=True)
        offset = cells % size
        slot = offset[:, 0] * size * size + offset[:, 1] * size + offset[:, 2]
        patches = feats.new_zeros(len(blocks), size**3, feats.shape[1])
        patches[inverse, slot] = feats
        return self.proj(patches.flatten(1)), blocks, inverse, slot


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm: nn.Module = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    @staticmethod
    def _rotate(x: Tensor, rotation: Tensor) -> Tensor:
        pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(pairs * rotation).flatten(2).type_as(x)

    def forward(self, x: Tensor, rotation: Tensor, sizes: Sequence[int]) -> Tensor:
        count, channels = x.shape
        qkv = self.qkv(x).view(count, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        query, key, value = qkv.unbind(dim=0)  # (heads, tokens, head_dim) each
        query = self._rotate(self.q_norm(query), rotation)
        key = self._rotate(self.k_norm(key), rotation)
        out, start = [], 0
        for size in sizes:  # global attention inside every cloud
            rows = slice(start, start + size)
            out.append(F.scaled_dot_product_attention(query[:, rows], key[:, rows], value[:, rows]))
            start += size
        return self.proj(torch.cat(out, dim=1).transpose(0, 1).reshape(count, channels))


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _DropPath(nn.Module):
    """Stochastic depth, drawn per token as in the reference."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        return x * x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        init_values: float | None,
        qk_norm: bool,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, qk_norm)
        self.ls1: nn.Module = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2: nn.Module = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x: Tensor, rotation: Tensor, sizes: Sequence[int]) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rotation, sizes)))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class _Detokenizer(nn.Module):
    """A token to its voxels: one linear read-out per slot of the block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size, self.out_channels = kernel_size, out_channels
        self.proj = nn.Linear(in_channels, kernel_size**3 * out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, tokens: Tensor, inverse: Tensor, slot: Tensor) -> Tensor:
        # The reference computes every slot of every token and indexes; here only the
        # slots that hold a voxel are computed, slot by slot, with the same weight rows.
        weight = self.proj.weight.view(self.kernel_size**3, self.out_channels, -1)
        out = tokens.new_empty(len(inverse), self.out_channels)
        for index in slot.unique().tolist():
            rows = (slot == index).nonzero(as_tuple=True)[0]
            out[rows] = tokens[inverse[rows]] @ weight[index].T
        return out + self.bias


class _Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        def norm(channels: int) -> nn.Module:
            return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

        self.pre = nn.Sequential(
            norm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels, bias=False),
            norm(out_channels),
            nn.GELU(),
        )
        self.unembed = _Detokenizer(out_channels, out_channels, kernel_size)
        self.post = nn.Sequential(norm(out_channels), nn.GELU())

    def forward(self, tokens: Tensor, inverse: Tensor, slot: Tensor) -> Tensor:
        return self.post(self.unembed(self.pre(tokens), inverse, slot))


@BACKBONES.register()
class Volt(Backbone):
    """Volt: global attention over voxel-patch tokens, features on every input point.

    The defaults are Volt-B of the official repository (768 channels, 12 blocks, 12
    heads) on a 2 cm grid with patches of 5 voxels, i.e. tokens of 10 cm. Volt-S is
    ``embed_dim: 384, num_heads: 6, qk_norm: true, up_mlp_dim: 128``. ``grid_size``
    times ``kernel_size`` is the edge of a token: the smallest thing attention can
    relate, and what decides how many tokens a frame has (attention is quadratic in
    it). A voxel takes the features of the first point that falls into it.

    Args:
        in_channels: Feature channels per point, at least one.
        out_channels: ``None`` returns the ``up_mlp_dim`` channels of the decoder; a
            number adds a linear layer with batch norm and GELU to that width.
        up_mlp_dim: Width of the decoder (128 in the released Volt-S, 256 in Volt-B).
        grid_size: Edge of a voxel, in the unit of the coordinates.
        kernel_size: Voxels per token along each axis.
        embed_dim: Width of the transformer.
        depth: Transformer blocks.
        num_heads: Attention heads.
        mlp_ratio: Hidden width of a block's MLP over ``embed_dim``.
        init_values: Initial layer scale; ``None`` for none.
        qk_norm: Layer norm on queries and keys.
        drop_path: Largest stochastic-depth rate.
        increase_drop_path: Grow the rate linearly with depth (otherwise constant).
        rope_theta: Base of the rotary frequencies.
        rope_split: Complex pairs of a head given to x, y and z; twice their sum must
            equal ``embed_dim / num_heads`` (64 in the released models).
    """

    output = "dense"
    status = "tested"
    source = (
        "Volt (Yilmaz et al., ECCV 2026, arXiv:2604.19609); adapted from "
        "YilmazKadir/Volt@8dcd608, MIT"
    )

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int | None = None,
        up_mlp_dim: int = 256,
        grid_size: float = 0.02,
        kernel_size: int = 5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
        drop_path: float = 0.3,
        increase_drop_path: bool = True,
        rope_theta: float = 100.0,
        rope_split: Sequence[int] = (12, 12, 8),
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("Volt needs at least one input feature channel per point")
        if grid_size <= 0 or kernel_size < 1:
            raise ValueError("Volt: grid_size must be positive and kernel_size at least 1")
        if embed_dim % num_heads:
            raise ValueError(f"Volt: embed_dim={embed_dim} is not a multiple of {num_heads} heads")
        head_dim = embed_dim // num_heads
        if len(rope_split) != 3 or 2 * sum(rope_split) != head_dim:
            raise ValueError(
                f"Volt: rope_split {list(rope_split)} rotates {2 * sum(rope_split)} channels "
                f"per head, but a head has embed_dim / num_heads = {head_dim}. Choose "
                "embed_dim = 64 * num_heads, or three numbers that sum to half the head width."
            )
        self.in_channels = in_channels
        self.out_channels = up_mlp_dim if out_channels is None else out_channels
        self.grid_size = grid_size
        self.tokenizer = _Tokenizer(in_channels, embed_dim, kernel_size)
        rates = (
            torch.linspace(0, drop_path, depth).tolist()
            if increase_drop_path
            else [drop_path] * depth
        )
        self.blocks = nn.Sequential(
            *[_Block(embed_dim, num_heads, mlp_ratio, init_values, qk_norm, rate) for rate in rates]
        )
        for axis, pairs in zip("xyz", rope_split):
            freqs = 1.0 / rope_theta ** torch.linspace(0, 1, pairs)
            self.register_buffer(f"freqs_{axis}", freqs, persistent=False)
        self.decoder = _Decoder(embed_dim, up_mlp_dim, kernel_size)
        self.head: nn.Module = (
            nn.Identity()
            if out_channels is None
            else nn.Sequential(
                nn.Linear(up_mlp_dim, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.GELU(),
            )
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _rotation(self, blocks: Tensor) -> Tensor:
        """``(1, T, head_dim / 2)`` complex rotations of the block indices."""
        angles = [
            torch.outer(blocks[:, axis].float(), getattr(self, f"freqs_{name}"))
            for axis, name in enumerate("xyz")
        ]
        angle = torch.cat(angles, dim=-1)
        return torch.polar(torch.ones_like(angle), angle).unsqueeze(0)

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="Volt")
        grid = grid_cells(points, self.grid_size, user="Volt")
        tokens, blocks, inverse, slot = self.tokenizer(feats[grid.first], grid.batch, grid.cells)
        sizes = torch.bincount(blocks[:, 0], minlength=points.batch_size).tolist()
        rotation = self._rotation(blocks[:, 1:])
        for block in self.blocks:
            tokens = block(tokens, rotation, sizes)
        voxels = self.head(self.decoder(tokens, inverse, slot))
        return PointFeatures(voxels[grid.inverse], points)


def convert_volt_checkpoint(
    checkpoint: Mapping[str, Any], model: nn.Module, *, prefix: str = "backbone."
) -> tuple[dict[str, Tensor], dict[str, list[str]]]:
    """Pick the tensors of a released Volt checkpoint that fit ``model``.

    The files of ``huggingface.co/KadirYilmaz/Volt`` are Pointcept checkpoints: the
    encoder under ``module.backbone.`` next to segmentation heads. A tensor is kept
    when the model has a parameter of that name and shape.

    Returns:
        The state dict with ``prefix`` in front of every key, and a report with the
        keys ``loaded``, the model keys left ``missing``, the file keys ``dropped`` and
        the ones of a different shape (``mismatched``).
    """
    state = checkpoint
    for key in ("state_dict", "model"):
        if isinstance(state.get(key), Mapping):
            state = state[key]
            break
    own = model.state_dict()
    report: dict[str, list[str]] = {"loaded": [], "missing": [], "dropped": [], "mismatched": []}
    out: dict[str, Tensor] = {}
    for name, value in state.items():
        key = name.removeprefix("module.")
        if not key.startswith("backbone.") or key.removeprefix("backbone.") not in own:
            report["dropped"].append(name)
            continue
        key = key.removeprefix("backbone.")
        if tuple(own[key].shape) != tuple(value.shape):
            report["mismatched"].append(
                f"{name}: file {tuple(value.shape)}, model {tuple(own[key].shape)}"
            )
        else:
            out[prefix + key] = value
            report["loaded"].append(key)
    report["missing"] = [key for key in own if prefix + key not in out]
    return out, report
