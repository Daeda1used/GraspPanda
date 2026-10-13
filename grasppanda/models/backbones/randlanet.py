# Copyright (c) 2020 www.open3d.org
# Adapted for GraspPanda from https://github.com/isl-org/Open3D-ML @ a3d0f7a (MIT):
# ``ml3d/torch/models/randlanet.py`` (the PyTorch RandLA-Net of Open3D-ML).
"""RandLA-Net.

Paper: Hu, Yang, Xie, Rosa, Guo, Wang, Trigoni, Markham, "RandLA-Net: Efficient
Semantic Segmentation of Large-Scale Point Clouds", CVPR 2020 (oral).

An encoder-decoder that subsamples by *random* sampling, which costs nothing, and makes
up for the points it drops with a local feature aggregation module: every point encodes
the positions of its ``k`` nearest neighbours (distance, offset, both absolute
positions), pools them with learned attention weights, and does so twice, which widens
its view before the next random cut. The decoder copies features back from the nearest
kept point. Features come back on every input point: ``output = "dense"``.

Differences from the upstream, each because the slot needs it:

* the upstream shuffles a cloud and computes neighbour, pooling and interpolation
  indices in its data pipeline, on the CPU; here the model does it, per cloud of the
  packed batch, with :func:`grasppanda.ops.knn`;
* random sampling is random in training and a fixed, seeded permutation in evaluation
  (a function of the cloud size only), so that inference is repeatable and a frame gives
  the same result alone or in a batch;
* the classifier is replaced by a projection to ``out_channels``.

Upstream weights are not loadable as they are (1x1 convolutions are linear layers here).
The official TensorFlow repository of the authors is CC BY-NC-SA 4.0 and was not used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import check_features, cloud_sizes, knn_packed

__all__ = ["RandLANet"]


class _Shared(nn.Module):
    """Linear layer with batch norm and activation on the last axis (a 1x1 convolution)."""

    def __init__(self, inputs: int, outputs: int, *, slope: float | None = 0.2) -> None:
        super().__init__()
        self.conv = nn.Linear(inputs, outputs)
        self.batch_norm = nn.BatchNorm1d(outputs, eps=1e-6, momentum=0.01)
        self.activation = nn.LeakyReLU(slope) if slope is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.batch_norm(self.conv(x).flatten(0, -2)).reshape(*x.shape[:-1], -1)
        return self.activation(out)


class _AttentivePooling(nn.Module):
    """Softmax scores over the neighbours of a point, weighted sum, shared layer."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.score_fn = nn.Linear(inputs, inputs)
        self.mlp = _Shared(inputs, outputs)

    def forward(self, x: Tensor) -> Tensor:
        scores = torch.softmax(self.score_fn(x), dim=1)  # over the K neighbours
        return self.mlp((scores * x).sum(dim=1))


class _LocalFeatureAggregation(nn.Module):
    """Two rounds of local spatial encoding and attentive pooling with a shortcut."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.mlp1 = _Shared(d_in, d_out // 2)
        self.lse1 = _Shared(10, d_out // 2)
        self.pool1 = _AttentivePooling(d_out, d_out // 2)
        self.lse2 = _Shared(d_out // 2, d_out // 2)
        self.pool2 = _AttentivePooling(d_out, d_out)
        self.mlp2 = _Shared(d_out, 2 * d_out, slope=None)
        self.shortcut = _Shared(d_in, 2 * d_out, slope=None)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords: Tensor, feats: Tensor, neighbours: Tensor) -> Tensor:
        x = self.mlp1(feats)
        near = coords[neighbours]
        centre = coords[:, None, :].expand_as(near)
        offset = centre - near
        distance = offset.pow(2).sum(dim=-1, keepdim=True).sqrt()
        relative = self.lse1(torch.cat([distance, offset, centre, near], dim=-1))
        x = self.pool1(torch.cat([x[neighbours], relative], dim=-1))
        relative = self.lse2(relative)
        x = self.pool2(torch.cat([x[neighbours], relative], dim=-1))
        return self.lrelu(self.mlp2(x) + self.shortcut(feats))


def _prefix_rows(sizes: Sequence[int], counts: Sequence[int], device: torch.device) -> Tensor:
    """Rows of the first ``counts[b]`` points of every cloud of a packed batch."""
    rows, start = [], 0
    for size, count in zip(sizes, counts):
        rows.append(torch.arange(start, start + count, device=device))
        start += size
    return torch.cat(rows)


@BACKBONES.register()
class RandLANet(Backbone):
    """RandLA-Net encoder-decoder with features on every input point.

    The defaults are the published network: four levels that each keep a quarter of the
    points, 16 neighbours, widths 16 / 64 / 128 / 256 (doubled at each level's output),
    and a 32-channel output.

    Args:
        in_channels: Feature channels per point besides the coordinates, which the
            network always reads as its first three input channels.
        num_neighbors: Neighbours of the local feature aggregation.
        sub_sampling_ratio: Ratio of random subsampling of every level.
        dim_features: Channels of the point-wise stem.
        dim_output: Width of every level; a level returns twice its width.
        out_channels: Channels of the returned features.
        seed: Seed of the permutation used in evaluation mode. In training mode every
            forward pass draws a new one from the global random state.
    """

    output = "dense"
    status = "tested"
    source = "RandLA-Net (Hu et al., CVPR 2020), adapted from isl-org/Open3D-ML@a3d0f7a, MIT"

    def __init__(
        self,
        in_channels: int = 0,
        num_neighbors: int = 16,
        sub_sampling_ratio: Sequence[int] = (4, 4, 4, 4),
        dim_features: int = 8,
        dim_output: Sequence[int] = (16, 64, 128, 256),
        out_channels: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(sub_sampling_ratio) != len(dim_output) or not dim_output:
            raise ValueError(
                f"RandLANet: sub_sampling_ratio and dim_output need one entry per level, got "
                f"{len(sub_sampling_ratio)} and {len(dim_output)}"
            )
        if any(ratio < 1 for ratio in sub_sampling_ratio) or any(d % 2 for d in dim_output):
            raise ValueError("RandLANet: ratios must be at least 1 and dim_output even")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_neighbors, self.ratios, self.seed = num_neighbors, tuple(sub_sampling_ratio), seed
        self.fc0 = _Shared(3 + in_channels, dim_features)
        self.encoder = nn.ModuleList()
        width, skips = dim_features, []
        for index, level in enumerate(dim_output):
            self.encoder.append(_LocalFeatureAggregation(width, level))
            width = 2 * level
            if index == 0:
                skips.append(width)
            skips.append(width)
        self.mlp = _Shared(width, width)
        self.decoder = nn.ModuleList()
        for index in range(len(dim_output)):
            self.decoder.append(_Shared(skips[-index - 2] + width, skips[-index - 2]))
            width = skips[-index - 2]
        self.fc1 = nn.Sequential(_Shared(width, 64), _Shared(64, out_channels))

    def _order(self, sizes: Sequence[int], device: torch.device) -> Tensor:
        """A permutation of every cloud: random in training, seeded in evaluation."""
        rows, start = [], 0
        for size in sizes:
            generator = None if self.training else torch.Generator().manual_seed(self.seed)
            rows.append(torch.randperm(size, generator=generator) + start)
            start += size
        return torch.cat(rows).to(device)

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="RandLANet")
        sizes = cloud_sizes(points)
        needed = self.num_neighbors * math.prod(self.ratios[:-1])
        if min(sizes) < max(needed, math.prod(self.ratios)):
            raise ContractError(
                f"RandLANet keeps 1/{math.prod(self.ratios[:-1])} of a cloud at its coarsest "
                f"level and needs {self.num_neighbors} neighbours there, so every cloud needs "
                f"at least {max(needed, math.prod(self.ratios))} points; cloud "
                f"{sizes.index(min(sizes))} of the batch has {min(sizes)}."
            )
        order = self._order(sizes, points.coords.device)
        coords = points.coords[order]
        x = coords if feats is None else torch.cat([coords, feats[order]], dim=-1)
        x = self.fc0(x)
        skips, lifts = [], []
        for index, layer in enumerate(self.encoder):
            neighbours = knn_packed(coords, sizes, coords, sizes, self.num_neighbors)[1]
            encoded = layer(coords, x, neighbours)
            counts = [size // self.ratios[index] for size in sizes]
            keep = _prefix_rows(sizes, counts, coords.device)
            x = encoded[neighbours[keep]].amax(dim=1)
            if index == 0:
                skips.append(encoded)
            skips.append(x)
            kept = coords[keep]
            lifts.append(knn_packed(coords, sizes, kept, counts, 1)[1][:, 0])
            coords, sizes = kept, counts
        x = self.mlp(x)
        for index, layer in enumerate(self.decoder):
            x = layer(torch.cat([skips[-index - 2], x[lifts[-index - 1]]], dim=-1))
        x = self.fc1(x)
        out = torch.empty_like(x)
        out[order] = x
        return PointFeatures(out, points)
