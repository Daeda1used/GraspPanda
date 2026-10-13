"""Cylinder crop among the seeds.

For every seed, the grouper gathers the seeds that fall inside a cylinder aligned with
the chosen approach direction, expresses their offsets in the crop frame, concatenates
those with the neighbours' features, and reduces the neighbourhood with a shared MLP
and a max over neighbours.

The support of the query is the *seed set*, not the scene cloud: a crop sees at most
the other seeds of its frame. This is what the Graspness reference does; it makes the
stage cheap and ties the crop density to the seed count. The cylinder constants live
here and nowhere else (``docs/design/ARCHITECTURE.md`` §8.6).
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.ops import cylinder_query, group_points
from grasppanda.registry import GROUPERS, INJECTED
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures

__all__ = ["CylinderGroup"]


class _LinearNormAct(nn.Sequential):
    """Shared-MLP layer: the same affine map for every neighbour, batch norm, ReLU.

    The weight starts from a normal distribution with ``std = sqrt(2 / in_channels)``,
    as the shared MLP of the reference does. Batch norm hides the scale in the forward
    pass, but the scale sets how fast these weights move under the optimiser.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.linear.weight)


@GROUPERS.register()
class CylinderGroup(Grouper):
    """One approach-aligned cylinder per seed, grouping among the seeds.

    A support point ``p`` belongs to the crop of a seed with centre ``c`` and crop
    frame ``R`` when ``d = Rᵀ (p - c)`` satisfies ``hmin < d.x < hmax`` and
    ``d.y² + d.z² < radius²``; column 0 of ``R`` is the approach axis. The first
    ``nsample`` members in index order are used and missing slots repeat the first.
    The geometric part of a neighbour's feature is ``Rᵀ (p - c) / radius``.

    Args:
        radius: Cylinder radius in metres.
        hmin: Lower end of the cylinder along the approach axis, metres (negative =
            behind the crop centre).
        hmax: Upper end of the cylinder along the approach axis, metres.
        nsample: Neighbours per crop.
        mlp_channels: Widths of the shared MLP; the last one is ``out_channels``.
        in_channels: Channels of the support features; injected by the detector.
    """

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "tested"
    source = "GSNet (Wang et al., ICCV 2021)"

    def __init__(
        self,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        nsample: int = 16,
        mlp_channels: tuple[int, ...] = (256, 256),
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if radius <= 0 or hmax <= hmin or nsample < 1 or not mlp_channels:
            raise ValueError(
                "CylinderGroup needs radius > 0, hmin < hmax, nsample >= 1 and at least one "
                f"MLP width; got radius={radius}, hmin={hmin}, hmax={hmax}, nsample={nsample}, "
                f"mlp_channels={list(mlp_channels)}"
            )
        self.radius, self.hmin, self.hmax, self.nsample = radius, hmin, hmax, nsample
        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        widths = [3 + in_channels, *mlp_channels]
        self.mlp = nn.Sequential(*(_LinearNormAct(a, b) for a, b in pairwise(widths)))

    def group(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> Tensor:
        """Gather the neighbourhoods: ``(M, nsample, 3 + in_channels)``.

        Args:
            support: Features of the seed set.
            centres: ``(M, 3)`` crop centre of every seed.
            rotation: ``(M, 3, 3)`` crop frame of every seed.
        """
        name = type(self).__name__
        count = len(support.points)
        if support.channels != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel support features, got "
                f"{support.channels}"
            )
        if tuple(centres.shape) != (count, 3) or tuple(rotation.shape) != (count, 3, 3):
            raise ContractError(
                f"{name} groups among the seeds (support = '{self.support}'): it needs one "
                f"centre (M, 3) and one frame (M, 3, 3) per support point with M = {count}, got "
                f"{tuple(centres.shape)} and {tuple(rotation.shape)}"
            )
        coords = support.points.dense()
        batch, per_frame, _ = coords.shape
        centre = centres.reshape(batch, per_frame, 3)
        frame = rotation.reshape(batch, per_frame, 3, 3)
        index, _ = cylinder_query(
            centre, coords, frame, self.radius, self.hmin, self.hmax, self.nsample
        )
        offset = (group_points(coords, index) - centre[:, :, None, :]) / self.radius
        local = torch.matmul(offset, frame)  # row vectors: Rᵀ (p - c) for every neighbour
        grouped = torch.cat([local, group_points(support.dense(), index)], dim=-1)
        return grouped.flatten(0, 1)

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        grouped = self.group(support, centres, rotation)
        seeds, neighbours, channels = grouped.shape
        hidden = self.mlp(grouped.reshape(seeds * neighbours, channels))
        pooled = hidden.reshape(seeds, neighbours, self.out_channels).amax(dim=1)
        return GroupFeatures(pooled, self.out_axes)
