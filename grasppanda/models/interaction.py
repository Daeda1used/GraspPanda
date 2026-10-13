"""Token interaction: one self-attention block over a short sequence.

EconomicGrasp lets the members of a small set exchange information twice: the
neighbours of a cylinder crop before they are pooled, and the four task tokens of its
grasp head before each predicts its quantity. Both use the same block,

    y = LayerNorm(x + W_o · Attention(W_q x, W_k x, W_v x))

with scaled dot-product attention, dropout on the attention weights, and neither a
feed-forward sub-layer nor a second normalisation. It is written out here instead of
wrapping ``nn.MultiheadAttention`` so that the order of the floating-point operations,
and with it the result, is fixed by this file.

The block is also the attention half of a standard post-norm transformer encoder layer,
which additionally drops part of the attention output before the residual sum
(``residual_dropout``); FineGrasp fuses its crop scales with such layers
(:mod:`grasppanda.models.groupers.multi_scale`).
"""

from __future__ import annotations

import math

from torch import Tensor, nn

__all__ = ["TokenInteraction"]


class TokenInteraction(nn.Module):
    """Self-attention with a residual connection and layer norm.

    Args:
        channels: Width of the tokens; kept by the block.
        num_heads: Attention heads; must divide ``channels``.
        dropout: Probability of dropping an attention weight during training. The
            weights are *not* renormalised afterwards.
        residual_dropout: Probability of dropping an element of the attention output
            before it is added to the tokens, during training. EconomicGrasp has none
            (0); a transformer encoder layer uses the same rate as ``dropout``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError(
                f"num_heads must divide channels, got channels={channels}, num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError(f"residual_dropout must be in [0, 1), got {residual_dropout}")
        self.channels = channels
        self.num_heads = num_heads
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        #: Acts on the ``(N, heads, T, T)`` attention weights.
        self.dropout = nn.Dropout(dropout)
        #: Acts on the ``(N, T, channels)`` attention output. Without a rate it is the
        #: identity, so that no random number is drawn for it.
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout else nn.Identity()
        self.norm = nn.LayerNorm(channels)

    def attention(self, tokens: Tensor) -> Tensor:
        """``(N, heads, T, T)`` attention weights (before dropout); each row sums to one."""
        query, key = self._split(self.query(tokens)), self._split(self.key(tokens))
        scale = math.sqrt(self.channels // self.num_heads)
        return (query @ key.transpose(-1, -2) / scale).softmax(dim=-1)

    def forward(self, tokens: Tensor) -> Tensor:
        """``(N, T, channels)`` → ``(N, T, channels)``; tokens interact within a row only."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.channels:
            raise ValueError(
                f"expected tokens of shape (N, T, {self.channels}), got {tuple(tokens.shape)}"
            )
        weights = self.dropout(self.attention(tokens))
        mixed = (weights @ self._split(self.value(tokens))).transpose(1, 2)
        update = self.residual_dropout(self.output(mixed.reshape(tokens.shape)))
        return self.norm(tokens + update)

    def _split(self, projected: Tensor) -> Tensor:
        """``(N, T, C)`` → ``(N, heads, T, C / heads)``."""
        count, length, _ = projected.shape
        return projected.reshape(count, length, self.num_heads, -1).transpose(1, 2)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, num_heads={self.num_heads}"
