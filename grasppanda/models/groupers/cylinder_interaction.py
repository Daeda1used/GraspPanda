"""Cylinder crop among the seeds whose neighbours interact before they are pooled.

The crop of EconomicGrasp. It starts as :class:`CylinderGroup` does (the same cylinder
query among the seeds and the same shared MLP per neighbour) and then, instead of
pooling at once,

1. appends the coordinates of the crop centre to every neighbour, in the frame of the
   scene cloud (so the crop feature knows where in the camera frame it sits, which the
   centred and rotated offsets alone do not say);
2. lets the neighbours of a crop attend to each other
   (:class:`~grasppanda.models.interaction.TokenInteraction`);
3. applies a second shared MLP and takes the maximum over the neighbours.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor, nn

from grasppanda.models.interaction import TokenInteraction
from grasppanda.registry import GROUPERS, INJECTED
from grasppanda.structures import GroupFeatures, PointFeatures

from .cylinder import CylinderGroup, _LinearNormAct

__all__ = ["CylinderInteractionGroup"]


@GROUPERS.register()
class CylinderInteractionGroup(CylinderGroup):
    """:class:`CylinderGroup` with self-attention among the neighbours of each crop.

    Args:
        radius: Cylinder radius in metres.
        hmin: Lower end of the cylinder along the approach axis, metres.
        hmax: Upper end of the cylinder along the approach axis, metres.
        nsample: Neighbours per crop; they are the tokens of the attention.
        mlp_channels: Widths of the shared MLP before the interaction.
        centre_coordinates: Append the crop centre's coordinates (scene frame, metres)
            to every neighbour before the interaction, as the reference does. ``False``
            keeps the crop feature independent of where the crop sits.
        attention_heads: Heads of the interaction block.
        attention_dropout: Dropout on its attention weights during training.
        post_mlp_channels: Widths of the shared MLP after the interaction; the last
            one is ``out_channels``.
        in_channels: Channels of the support features; injected by the detector.
    """

    status = "tested"
    source = "EconomicGrasp (Wu et al., ECCV 2024)"

    def __init__(
        self,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        nsample: int = 16,
        mlp_channels: tuple[int, ...] = (256, 256),
        centre_coordinates: bool = True,
        attention_heads: int = 1,
        attention_dropout: float = 0.05,
        post_mlp_channels: tuple[int, ...] = (256, 256),
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__(radius, hmin, hmax, nsample, mlp_channels, in_channels)
        if not post_mlp_channels:
            raise ValueError("CylinderInteractionGroup needs at least one post-interaction width")
        self.centre_coordinates = centre_coordinates
        #: Token layout of the interaction: [shared-MLP feature | centre xyz].
        self.token_channels = mlp_channels[-1] + (3 if centre_coordinates else 0)
        self.interaction = TokenInteraction(self.token_channels, attention_heads, attention_dropout)
        widths = [self.token_channels, *post_mlp_channels]
        self.post_mlp = nn.Sequential(*(_LinearNormAct(a, b) for a, b in pairwise(widths)))
        self.out_channels = post_mlp_channels[-1]

    def tokens(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> Tensor:
        """Neighbour tokens before the interaction: ``(M, nsample, token_channels)``."""
        grouped = self.group(support, centres, rotation)
        seeds, neighbours, channels = grouped.shape
        hidden = self.mlp(grouped.reshape(seeds * neighbours, channels))
        hidden = hidden.reshape(seeds, neighbours, -1)
        if not self.centre_coordinates:
            return hidden
        where = centres[:, None, :].expand(-1, neighbours, -1).to(hidden.dtype)
        return torch.cat([hidden, where], dim=-1)

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        mixed = self.interaction(self.tokens(support, centres, rotation))
        seeds, neighbours, channels = mixed.shape
        hidden = self.post_mlp(mixed.reshape(seeds * neighbours, channels))
        pooled = hidden.reshape(seeds, neighbours, self.out_channels).amax(dim=1)
        return GroupFeatures(pooled, self.out_axes)
