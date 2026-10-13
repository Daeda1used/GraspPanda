"""Set abstraction with a point-transformer layer after every level.

The encoder of Scale-Balanced-Grasp (Ma and Huang, CoRL 2022): two PointNet++
set-abstraction levels, each followed by one vector self-attention layer in the style of
Point Transformer (Zhao et al., ICCV 2021) over the points the level kept. There is no
feature propagation: the features live on the points of the last level, a furthest-point
subsample of the scene (``output = "subsampled"``).

The attention layer, for a point ``i`` with feature ``f_i`` and its ``k`` nearest
points ``j`` among the points of its level (itself included):

    x = W_e f                                     embedding
    δ_ij = θ(p_i − p_j)                           position encoding, a two-layer MLP
    a_ij = softmax_j(γ(W_q x_i − W_k x_j + δ_ij) / sqrt(C))        per channel
    y_i = W_o Σ_j a_ij ⊙ (W_v x_j + δ_ij) + f_i   residual

It has no normalisation layer.

Provenance. Written for this code base on the reference operators of
:mod:`grasppanda.ops`; the set-abstraction levels are those of
:mod:`grasppanda.models.backbones.pointnet2`. Scale-Balanced-Grasp is under a
non-commercial licence: only its published hyper-parameters and the behaviour of its
network are used here, and its released weights are an evaluation aid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import check_features, cloud_sizes, knn_packed, offsets_of
from .pointnet2 import _SetAbstraction

__all__ = ["PointNet2Transformer"]


class _VectorAttention(nn.Module):
    """One vector self-attention layer over the ``neighbours`` nearest points."""

    def __init__(self, channels: int, neighbours: int) -> None:
        super().__init__()
        self.channels, self.num_neighbours = channels, neighbours
        self.embed = nn.Linear(channels, channels)
        self.project = nn.Linear(channels, channels)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.attention = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)

    @torch.no_grad()
    def neighbours(self, coords: Tensor, sizes: Sequence[int]) -> Tensor:
        """``(N, k)`` rows of the nearest points of every point within its own cloud.

        Squared distances are sums of squared coordinate differences and equal
        distances are ordered by index.
        """
        return knn_packed(coords, sizes, coords, sizes, self.num_neighbours)[1]

    def forward(self, coords: Tensor, feats: Tensor, sizes: Sequence[int]) -> Tensor:
        index = self.neighbours(coords, sizes)
        embedded = self.embed(feats)
        position = self.position(coords[:, None, :] - coords[index])  # (N, k, C)
        weight = self.attention(
            self.query(embedded)[:, None, :] - self.key(embedded)[index] + position
        )
        weight = torch.softmax(weight / math.sqrt(self.channels), dim=1)  # over the neighbours
        mixed = (weight * (self.value(embedded)[index] + position)).sum(dim=1)
        return self.project(mixed) + feats


@BACKBONES.register()
class PointNet2Transformer(Backbone):
    """PointNet++ set abstraction interleaved with point-transformer layers.

    The defaults are the backbone of Scale-Balanced-Grasp: abstraction to 2048 and 1024
    points with radii 0.04 / 0.1 m and 64 / 32 neighbours (the first two levels of the
    GraspNet-baseline backbone), an attention layer over 16 nearest points after each,
    256 output channels on the 1024-point set.

    Every cloud must hold at least ``npoints[0]`` points; clouds may differ in size.
    The returned set is ``points.subset(index)`` with ``index`` in sampling order.

    Args:
        in_channels: Feature channels per point besides the coordinates.
        npoints: Points kept by each set-abstraction level.
        radii: Ball radius of each level, in the unit of the coordinates.
        nsamples: Neighbours gathered per ball; a ball with fewer points repeats its
            first hit.
        sa_mlps: Layer widths of each level's shared MLP. The attention layer of a
            level works on the last width.
        attention_neighbours: Nearest points every attention layer attends to. Each
            level must keep at least this many points.
        normalize_xyz: Divide grouped offsets by the ball radius.
        fps_skip_near_origin: Reproduce the sampling kernel of the PyTorch PointNet++
            lineage, which never selects a point with squared norm ≤ 1e-3 after the
            first one.
        fps_tie_break: Which point is sampled when several are equally far from the
            points chosen so far: ``pointnet2_cuda`` (reference) follows the order of
            that lineage's sampling kernel, ``lowest_index`` takes the first. The order
            of the sampled points matters here: a level keeps the first points of the
            level before it, and the attention layers index neighbours by position.
    """

    output = "subsampled"
    status = "weights-verified"
    source = "Scale-Balanced-Grasp (Ma and Huang, CoRL 2022)"

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int:
        return options["sa_mlps"][-1][-1]

    def __init__(
        self,
        in_channels: int = 0,
        npoints: Sequence[int] = (2048, 1024),
        radii: Sequence[float] = (0.04, 0.1),
        nsamples: Sequence[int] = (64, 32),
        sa_mlps: Sequence[Sequence[int]] = ((64, 64, 128), (128, 128, 256)),
        attention_neighbours: int = 16,
        normalize_xyz: bool = True,
        fps_skip_near_origin: bool = True,
        fps_tie_break: Literal["lowest_index", "pointnet2_cuda"] = "pointnet2_cuda",
    ) -> None:
        super().__init__()
        if fps_tie_break not in ("lowest_index", "pointnet2_cuda"):
            raise ValueError(f"PointNet2Transformer: unknown fps_tie_break '{fps_tie_break}'")
        levels = len(npoints)
        if not (levels == len(radii) == len(nsamples) == len(sa_mlps)) or levels == 0:
            raise ValueError(
                "PointNet2Transformer: npoints, radii, nsamples and sa_mlps need one entry per "
                f"level, got {levels}, {len(radii)}, {len(nsamples)}, {len(sa_mlps)}"
            )
        if any(later > earlier for earlier, later in pairwise(npoints)):
            raise ValueError(
                f"PointNet2Transformer: npoints must not increase with depth, got {list(npoints)}"
            )
        if any(len(mlp) == 0 for mlp in sa_mlps):
            raise ValueError("PointNet2Transformer: every MLP needs at least one layer")
        if not 1 <= attention_neighbours <= min(npoints):
            raise ValueError(
                f"PointNet2Transformer: attention_neighbours must be in 1..{min(npoints)} (the "
                f"smallest level), got {attention_neighbours}"
            )
        self.in_channels = in_channels
        self.npoints = tuple(npoints)
        width = in_channels
        for index in range(levels):
            layer = _SetAbstraction(
                width,
                npoints[index],
                radii[index],
                nsamples[index],
                sa_mlps[index],
                normalize_xyz=normalize_xyz,
                skip_near_origin=fps_skip_near_origin,
                tie_break=fps_tie_break,
            )
            width = sa_mlps[index][-1]
            self.add_module(f"sa{index + 1}", layer)
            self.add_module(
                f"transformer{index + 1}", _VectorAttention(width, attention_neighbours)
            )
        self.num_levels = levels
        self.out_channels = width

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        sizes = cloud_sizes(points)
        if min(sizes) < self.npoints[0]:
            raise ContractError(
                f"{name} samples {self.npoints[0]} points per cloud at its first level, but "
                f"cloud {sizes.index(min(sizes))} of the batch has only {min(sizes)}. Feed "
                "larger clouds or configure smaller npoints; points are never duplicated "
                "silently."
            )
        coords, rows = points.coords, None
        for index in range(self.num_levels):
            abstraction: _SetAbstraction = getattr(self, f"sa{index + 1}")
            chosen, feats = abstraction(coords, feats, sizes, presampled=index > 0)
            rows = chosen if rows is None else rows[chosen]
            coords = coords[chosen]
            sizes = [abstraction.npoint] * len(sizes)
            feats = getattr(self, f"transformer{index + 1}")(coords, feats, sizes)
        return PointFeatures(feats, points.subset(rows, offsets_of(sizes, feats.device)))
