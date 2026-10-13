# Copyright (c) 2022 PANG-Yatian, YUAN-Li
# Adapted for GraspPanda from https://github.com/Pang-Yatian/Point-MAE @ 7445a68 (MIT)
# Copyright (c) 2023 BAAI-Vision
# Adapted for GraspPanda from https://github.com/baaivision/Uni3D @ 64e03c3 (MIT)
# Adapted for GraspPanda from https://github.com/LMD0311/PointMamba @ 80858e0 (Apache-2.0;
# its licence file names no copyright holder)
"""Encoders over point-patch tokens: a plain ViT, a Mamba state-space model, Uni3D.

Papers: Yu et al., "Point-BERT", CVPR 2022; Pang et al., "Masked Autoencoders for Point
Cloud Self-supervised Learning" (Point-MAE), ECCV 2022; Liang et al., "PointMamba: A
Simple State Space Model for Point Cloud Analysis", NeurIPS 2024; Zhou et al., "Uni3D:
Exploring Unified 3D Representation at Scale", ICLR 2024.

The family shares one tokeniser. ``num_group`` centres are drawn from every cloud by
furthest point sampling; the ``group_size`` nearest points of a centre, relative to it,
go through a small PointNet and become one token; an MLP of the centre coordinate is
the position embedding. A sequence model then mixes the tokens of one cloud:

* :class:`PointViT`: standard pre-norm transformer blocks with global attention, the
  encoder of Point-BERT, Point-MAE and the methods built on them;
* :class:`PointMamba`: Mamba blocks over the tokens ordered along a Hilbert curve and
  along its transposed curve;
* :class:`Uni3D`: the blocks of an EVA vision transformer from ``timm``, as in Uni3D,
  whose checkpoints were aligned with CLIP on a million coloured 3-D objects.

Features live on the centres, so both are ``subsampled`` encoders: the place of
PointNet++ in GraspNet-baseline. ``num_group`` is the number of seeds such a method
gets.

Provenance. Tokeniser and transformer follow ``models/Point_MAE.py`` of the Point-MAE
repository, parameter names included (``encoder.first_conv.0``, ``pos_embed.0``,
``blocks.blocks.3.attn.qkv`` ...), so that released checkpoints of the family load
through :func:`convert_point_token_checkpoint`. The compiled ``KNN_CUDA`` and
``pointnet2_ops`` of the reference are replaced by :mod:`grasppanda.ops`. The token
order, the order scaling and the block layout of :class:`PointMamba` follow
``models/point_mamba_scan.py`` of LMD0311/PointMamba @ 80858e0 (Apache-2.0); its mixer
is the ``Mamba`` module of the optional ``mamba-ssm`` library. The layout of
:class:`Uni3D` (a class token with its own position, a linear layer between patch
encoder and transformer) follows ``models/point_encoder.py`` of baaivision/Uni3D @
64e03c3 (MIT); its transformer blocks are built by the optional ``timm`` library. The
released weights of all three were trained on single objects scaled to the unit sphere
(ShapeNet; Objaverse and others for Uni3D): on a metric scene they are an
initialisation, not a feature extractor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import serialization_code
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet
from grasppanda.utils.imports import require

from ._packed import check_features, cloud_sizes, fps_packed, knn_packed, offsets_of

__all__ = ["PointMamba", "PointViT", "Uni3D", "convert_point_token_checkpoint"]


class _PatchEncoder(nn.Module):
    """The mini-PointNet that turns a group of points into one token."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, out_channels, 1),
        )

    def forward(self, groups: Tensor) -> Tensor:
        """``(M, K, C_in)`` groups to ``(M, C_out)`` tokens."""
        feature = self.first_conv(groups.transpose(1, 2))
        pooled = feature.amax(dim=2, keepdim=True)
        feature = torch.cat([pooled.expand(-1, -1, feature.shape[2]), feature], dim=1)
        return self.second_conv(feature).amax(dim=2)


class _DropPath(nn.Module):
    """Stochastic depth, drawn per cloud."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        return x * x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep


class _Mlp(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, channels: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(query, key, value)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, channels))


class _Block(nn.Module):
    def __init__(
        self, channels: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, drop_path: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = _Attention(channels, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = _Mlp(channels, int(channels * mlp_ratio))
        self.drop_path = _DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _Blocks(nn.Module):
    """Holds the blocks under the name the reference checkpoints use (``blocks.blocks``)."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)


class _PointTokenEncoder(Backbone):
    """Tokeniser, position embedding and output projection shared by the family."""

    output = "subsampled"

    @classmethod
    def output_points(cls, options: Mapping[str, Any]) -> int | None:
        """``num_group``: one returned point per token."""
        return int(options["num_group"])

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None,
        num_group: int,
        group_size: int,
        embed_dim: int,
        token_dim: int,
        coord_scale: float,
        centre_clouds: bool,
    ) -> None:
        super().__init__()
        if num_group < 1 or group_size < 1:
            raise ValueError(f"{type(self).__name__}: num_group and group_size must be positive")
        if coord_scale <= 0:
            raise ValueError(f"{type(self).__name__}: coord_scale must be positive")
        self.in_channels = in_channels
        self.num_group, self.group_size = num_group, group_size
        self.coord_scale, self.centre_clouds = coord_scale, centre_clouds
        self.encoder = _PatchEncoder(3 + in_channels, token_dim)
        self.reduce_dim: nn.Module = (
            nn.Identity() if token_dim == embed_dim else nn.Linear(token_dim, embed_dim)
        )
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.proj: nn.Module = (
            nn.Identity() if out_channels is None else nn.Linear(embed_dim, out_channels)
        )
        self.out_channels = embed_dim if out_channels is None else out_channels

    def _tokens(self, points: PointSet, feats: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """``(B, G, C)`` tokens, ``(B, G, 3)`` scaled centres and the ``(B * G,)`` rows kept."""
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        sizes = cloud_sizes(points)
        needed = max(self.num_group + 1, self.group_size)
        if min(sizes) < needed:
            raise ContractError(
                f"{name} keeps num_group={self.num_group} centres with group_size="
                f"{self.group_size} neighbours each, so every cloud needs at least {needed} "
                f"points; cloud {sizes.index(min(sizes))} of the batch has {min(sizes)}."
            )
        batch = len(sizes)
        coords = points.coords
        rows = fps_packed(coords, sizes, self.num_group)
        centres = coords[rows]
        counts = [self.num_group] * batch
        _, index = knn_packed(centres, counts, coords, sizes, self.group_size)
        groups = (coords[index] - centres[:, None, :]) * self.coord_scale
        if feats is not None:
            groups = torch.cat([groups, feats[index]], dim=-1)
        tokens = self.reduce_dim(self.encoder(groups)).reshape(batch, self.num_group, -1)
        centres = centres.reshape(batch, self.num_group, 3)
        if self.centre_clouds:
            centres = centres - centres.mean(dim=1, keepdim=True)
        return tokens, centres * self.coord_scale, rows

    def _features(self, points: PointSet, tokens: Tensor, rows: Tensor) -> PointFeatures:
        offsets = offsets_of([self.num_group] * points.batch_size, tokens.device)
        return PointFeatures(self.proj(tokens.flatten(0, 1)), points.subset(rows, offsets))


@BACKBONES.register()
class PointViT(_PointTokenEncoder):
    """Plain transformer over point-patch tokens (Point-BERT / Point-MAE encoder).

    The defaults are the ViT-S of Point-MAE (384 channels, 12 blocks, 6 heads, groups of
    32 points) with 1024 groups in place of the 64 used for single objects, so that a
    scene yields 1024 seed features. Attention is global over the tokens of a cloud.
    The released checkpoints were trained on objects scaled to the unit sphere:
    ``coord_scale`` multiplies metric coordinates before they enter the network.

    Args:
        in_channels: Extra feature channels per point, concatenated to the offsets of
            a group. 0 (the default) is coordinates only, as in the released models.
        out_channels: ``None`` returns the ``embed_dim`` channels of the last norm; a
            number adds a linear projection to that width.
        num_group: Centres (tokens, and returned points) per cloud.
        group_size: Nearest points gathered around a centre.
        embed_dim: Width of the transformer.
        depth: Transformer blocks.
        num_heads: Attention heads.
        token_dim: Width of the patch encoder's output; when it differs from
            ``embed_dim`` a linear layer joins them (Point-BERT: 256 to 384).
        mlp_ratio: Hidden width of a block's MLP over ``embed_dim``.
        qkv_bias: Bias in the query, key and value projection.
        drop_path_rate: Largest stochastic-depth rate; it grows linearly with depth.
        coord_scale: Factor applied to offsets and centres before the network.
        centre_clouds: Subtract the mean centre of every cloud before the position
            embedding (the reference models see objects centred at the origin).
    """

    status = "tested"
    source = (
        "Point-BERT (Yu et al., CVPR 2022) and Point-MAE (Pang et al., ECCV 2022) encoder; "
        "adapted from Pang-Yatian/Point-MAE@7445a68, MIT"
    )

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int | None:
        """``out_channels``, or the width of the transformer."""
        width = options.get("out_channels")
        return width if isinstance(width, int) else options["embed_dim"]

    def __init__(
        self,
        in_channels: int = 0,
        out_channels: int | None = None,
        num_group: int = 1024,
        group_size: int = 32,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        token_dim: int = 384,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.1,
        coord_scale: float = 1.0,
        centre_clouds: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            num_group,
            group_size,
            embed_dim,
            token_dim,
            coord_scale,
            centre_clouds,
        )
        if embed_dim % num_heads:
            raise ValueError(
                f"PointViT: embed_dim={embed_dim} is not a multiple of num_heads={num_heads}"
            )
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = _Blocks(
            [_Block(embed_dim, num_heads, mlp_ratio, qkv_bias, rate) for rate in rates]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        tokens, centres, rows = self._tokens(points, feats)
        position = self.pos_embed(centres)
        for block in self.blocks.blocks:  # the reference adds the position before every block
            tokens = block(tokens + position)
        return self._features(points, self.norm(tokens), rows)


class _MambaBlock(nn.Module):
    def __init__(self, channels: int, mixer: nn.Module, drop_path: float) -> None:
        super().__init__()
        self.mixer = mixer
        self.norm = nn.LayerNorm(channels, eps=1e-5)
        self.drop_path = _DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))


class _Mixer(nn.Module):
    """Holds the Mamba blocks under the names of the reference (``blocks.layers``)."""

    def __init__(self, channels: int, layers: Sequence[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm_f = nn.LayerNorm(channels, eps=1e-5)


@BACKBONES.register()
class PointMamba(_PointTokenEncoder):
    """Mamba state-space model over point-patch tokens in Hilbert order (PointMamba).

    The tokens of a cloud are laid out twice, along a Hilbert curve through the centres
    and along the curve with x and y swapped; each copy gets its own learned scale and
    shift, and the doubled sequence runs through ``depth`` Mamba blocks. The feature of
    a centre is the mean of its two copies (the reference pools the whole sequence for
    classification; this read-out is ours). Cost grows linearly with the number of
    tokens, where :class:`PointViT` grows quadratically.

    Needs the ``mamba-ssm`` library, whose selective scan is a CUDA kernel: this encoder
    does not run on the CPU.

    Args:
        in_channels: Extra feature channels per point; 0 is coordinates only.
        out_channels: ``None`` returns ``embed_dim`` channels; a number adds a linear
            projection to that width.
        num_group: Centres (tokens, and returned points) per cloud.
        group_size: Nearest points gathered around a centre.
        embed_dim: Width of the model.
        depth: Mamba blocks.
        token_dim: Width of the patch encoder's output.
        drop_path_rate: Largest stochastic-depth rate; it grows linearly with depth.
        serialization_grid: Cell edge of the grid the Hilbert curve runs on, in the
            unit of the scaled coordinates.
        coord_scale: Factor applied to offsets and centres before the network.
        centre_clouds: Subtract the mean centre of every cloud before the position
            embedding.
        ssm_cfg: Arguments passed to ``mamba_ssm.Mamba`` as they are (``d_state``,
            ``d_conv``, ``expand`` ...); ``None`` keeps the library defaults.
    """

    status = "bridge"
    extras = ("mamba-ssm",)
    source = (
        "PointMamba (Liang et al., NeurIPS 2024), layout of LMD0311/PointMamba@80858e0 "
        "(Apache-2.0); mixer via mamba-ssm"
    )

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int | None:
        """``out_channels``, or the width of the model."""
        width = options.get("out_channels")
        return width if isinstance(width, int) else options["embed_dim"]

    def __init__(
        self,
        in_channels: int = 0,
        out_channels: int | None = None,
        num_group: int = 1024,
        group_size: int = 32,
        embed_dim: int = 384,
        depth: int = 12,
        token_dim: int = 384,
        drop_path_rate: float = 0.1,
        serialization_grid: float = 0.02,
        coord_scale: float = 1.0,
        centre_clouds: bool = True,
        ssm_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            num_group,
            group_size,
            embed_dim,
            token_dim,
            coord_scale,
            centre_clouds,
        )
        mamba = require(
            "mamba_ssm", extra="mamba-ssm", purpose="PointMamba (the Mamba selective scan)"
        )
        if serialization_grid <= 0:
            raise ValueError("PointMamba: serialization_grid must be positive")
        self.serialization_grid = serialization_grid
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = _Mixer(
            embed_dim,
            [
                _MambaBlock(embed_dim, mamba.Mamba(embed_dim, **dict(ssm_cfg or {})), rate)
                for rate in rates
            ],
        )
        for name in ("OrderScale_gamma_1", "OrderScale_gamma_2"):
            setattr(self, name, nn.Parameter(torch.ones(embed_dim).normal_(mean=1.0, std=0.02)))
        for name in ("OrderScale_beta_1", "OrderScale_beta_2"):
            setattr(self, name, nn.Parameter(torch.zeros(embed_dim).normal_(std=0.02)))

    @torch.no_grad()
    def _orders(self, centres: Tensor) -> list[Tensor]:
        """``(B, G)`` token order along the Hilbert curve and along its transposed curve."""
        grid = torch.floor(centres / self.serialization_grid).long()
        grid = grid - grid.amin(dim=1, keepdim=True)
        depth = max(int(grid.max()).bit_length(), 1)
        flat = grid.flatten(0, 1)
        return [
            serialization_code(flat, depth, order).reshape(grid.shape[:2]).argsort(dim=1)
            for order in ("hilbert", "hilbert-trans")
        ]

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        tokens, centres, rows = self._tokens(points, feats)
        if not tokens.is_cuda:
            raise ContractError(
                "PointMamba runs on CUDA only: the selective scan of mamba-ssm has no CPU "
                "kernel. Move the model and the points to a GPU, or use PointViT."
            )
        position = self.pos_embed(centres)
        forward, backward = self._orders(centres)
        scales = (
            (self.OrderScale_gamma_1, self.OrderScale_beta_1),
            (self.OrderScale_gamma_2, self.OrderScale_beta_2),
        )
        sequence, positions = [], []
        for order, (gamma, beta) in zip((forward, backward), scales):
            index = order[:, :, None].expand_as(tokens)
            sequence.append(tokens.gather(1, index) * gamma + beta)
            positions.append(position.gather(1, index))
        hidden = torch.cat(sequence, dim=1) + torch.cat(positions, dim=1)
        for layer in self.blocks.layers:
            hidden = layer(hidden)
        hidden = self.blocks.norm_f(hidden)
        out = torch.zeros_like(tokens)
        for order, half in zip((forward, backward), hidden.chunk(2, dim=1)):
            out = out.scatter_add(1, order[:, :, None].expand_as(half), half)
        return self._features(points, out / 2, rows)


@BACKBONES.register()
class Uni3D(_PointTokenEncoder):
    """Uni3D point encoder: point-patch tokens through the blocks of a ``timm`` EVA ViT.

    The tokeniser of the family with colour next to the offsets (``in_channels: 3``), a
    512-wide patch encoder, a class token, and the transformer blocks of the named
    ``timm`` model; the feature of a centre is its token after the final norms (Uni3D
    itself normalises the class token only and projects it into CLIP space; that
    projection is not part of this encoder). The released models and their ``timm_model``:

    ==========  ===========================  =========
    checkpoint  ``timm_model``               channels
    ==========  ===========================  =========
    uni3d-ti    ``eva02_tiny_patch14_224``   192
    uni3d-s     ``eva02_small_patch14_224``  384
    uni3d-b     ``eva02_base_patch14_448``   768
    uni3d-l     ``eva02_large_patch14_448``  1024
    uni3d-g     ``eva_giant_patch14_560``    1408
    ==========  ===========================  =========

    Args:
        in_channels: Extra feature channels per point, concatenated to the offsets of a
            group; the released models read three (colour in 0..1).
        out_channels: ``None`` returns the width of the ``timm`` model; a number adds a
            linear projection to that width.
        num_group: Centres (tokens, and returned points) per cloud; Uni3D uses 512.
        group_size: Nearest points gathered around a centre; Uni3D uses 64.
        timm_model: Name of the ``timm`` model whose blocks and final norm are used.
        token_dim: Width of the patch encoder's output.
        drop_path_rate: Stochastic-depth rate handed to ``timm``.
        coord_scale: Factor applied to offsets and centres before the network.
        centre_clouds: Subtract the mean centre of every cloud before the position
            embedding.
        timm_kwargs: Further arguments of ``timm.create_model`` as they are; ``None``
            for none.
    """

    status = "bridge"
    extras = ("timm",)
    source = (
        "Uni3D (Zhou et al., ICLR 2024), layout of baaivision/Uni3D@64e03c3 (MIT); "
        "transformer blocks via timm"
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        num_group: int = 1024,
        group_size: int = 64,
        timm_model: str = "eva02_small_patch14_224",
        token_dim: int = 512,
        drop_path_rate: float = 0.0,
        coord_scale: float = 1.0,
        centre_clouds: bool = True,
        timm_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        timm = require("timm", extra="timm", purpose="Uni3D (the EVA transformer blocks)")
        vit = timm.create_model(
            timm_model, pretrained=False, drop_path_rate=drop_path_rate, **dict(timm_kwargs or {})
        )
        if not hasattr(vit, "blocks") or not hasattr(vit, "norm"):
            raise ValueError(
                f"Uni3D: the timm model '{timm_model}' has no `blocks` and `norm`; choose a "
                "vision transformer (the released checkpoints use EVA-02 models)"
            )
        embed_dim = int(vit.embed_dim)
        super().__init__(
            in_channels,
            out_channels,
            num_group,
            group_size,
            embed_dim,
            token_dim,
            coord_scale,
            centre_clouds,
        )
        self.visual = nn.Module()
        self.visual.blocks = vit.blocks
        self.visual.norm = vit.norm
        # EVA-02 models normalise in `fc_norm` and leave `norm` an identity.
        fc_norm = getattr(vit, "fc_norm", None)
        self.visual.fc_norm = fc_norm if isinstance(fc_norm, nn.Module) else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        tokens, centres, rows = self._tokens(points, feats)
        batch = tokens.shape[0]
        hidden = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
        position = torch.cat([self.cls_pos.expand(batch, -1, -1), self.pos_embed(centres)], dim=1)
        hidden = hidden + position
        for block in self.visual.blocks:
            hidden = block(hidden)
        hidden = self.visual.fc_norm(self.visual.norm(hidden))
        return self._features(points, hidden[:, 1:], rows)


#: Prefixes under which released checkpoints of the family store the encoder.
_ENCODER_PREFIXES = (
    "point_encoder.",
    "MAE_encoder.",
    "transformer_q.",
    "GPT_Transformer.",
    "base_model.",
)


def convert_point_token_checkpoint(
    checkpoint: Mapping[str, Any], model: nn.Module, *, prefix: str = "backbone."
) -> tuple[dict[str, Tensor], dict[str, list[str]]]:
    """Pick the tensors of a released point-token checkpoint that fit ``model``.

    Works for files laid out like Point-MAE's ``pretrain.pth`` and PointMamba's
    ``pretrain.pth`` (``base_model`` → ``MAE_encoder.*``), Point-BERT's
    ``Point-BERT.pth`` (``transformer_q.*``) and Uni3D's ``model.pt`` (``module`` →
    ``point_encoder.*``, whose ``encoder2trans`` is ``reduce_dim`` here): the wrapper
    prefix is stripped and a tensor is kept when the model has a parameter of that name
    and shape.

    Args:
        checkpoint: The loaded file.
        model: A :class:`PointViT`, :class:`PointMamba` or :class:`Uni3D` built with the
            sizes of the file.
        prefix: Put in front of every kept key; ``backbone.`` gives a file that
            ``runtime.load_from`` accepts with ``runtime.allow_weight_mismatch=true``.

    Returns:
        The state dict, and a report with the keys that were ``loaded``, the model keys
        left ``missing``, the file keys ``dropped`` (decoder, class token, heads) and the
        keys of a different shape (``mismatched``).
    """
    state = checkpoint
    for key in ("base_model", "module", "model", "state_dict"):
        if isinstance(state.get(key), Mapping):
            state = state[key]
            break
    own = model.state_dict()
    report: dict[str, list[str]] = {"loaded": [], "missing": [], "dropped": [], "mismatched": []}
    out: dict[str, Tensor] = {}
    for name, value in state.items():
        key = name.removeprefix("module.")
        for wrapper in _ENCODER_PREFIXES:
            key = key.removeprefix(wrapper)
        if key.startswith("encoder2trans."):
            key = "reduce_dim." + key.removeprefix("encoder2trans.")
        if key not in own:
            report["dropped"].append(name)
        elif tuple(own[key].shape) != tuple(value.shape):
            report["mismatched"].append(
                f"{name}: file {tuple(value.shape)}, model {tuple(own[key].shape)}"
            )
        else:
            out[prefix + key] = value
            report["loaded"].append(key)
    report["missing"] = [key for key in own if prefix + key not in out]
    return out, report
