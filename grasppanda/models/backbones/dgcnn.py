# Copyright (c) 2020 An Tao
# Adapted for GraspPanda from https://github.com/antao97/dgcnn.pytorch @ 07d534c (MIT):
# ``DGCNN_semseg_s3dis`` of ``model.py`` (author Yue Wang; modified by An Tao and Ziyi Wu).
"""DGCNN (Dynamic Graph CNN), scene-segmentation form.

Paper: Wang, Sun, Liu, Sarma, Bronstein, Solomon, "Dynamic Graph CNN for Learning on
Point Clouds", ACM Transactions on Graphics 2019.

Three EdgeConv layers at full resolution. Each connects every point to its ``k``
nearest neighbours, runs a shared MLP on the edge ``(x_j - x_i, x_i)`` and max-pools
over the edges. The first graph is built in space; the later ones in *feature* space,
recomputed in every forward pass (hence "dynamic"), so that points far apart but alike
become neighbours. The three results and a max-pooled vector of the whole cloud are
fused into one feature per input point. No subsampling: ``output = "dense"``.

Differences from the upstream, each because the slot needs it:

* the neighbour search runs per cloud of the packed batch and in blocks of queries, so
  a 20 000-point frame never builds its 20 000 x 20 000 distance matrix at once;
* the coordinates are the first input channels (``coords_as_features``); the upstream
  S3DIS model reads them from a 9-channel input (position, colour, room-normalised
  position) and searches its first neighbours in the last three;
* the classifier and its dropout are left to the grasp stages.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import check_features, cloud_sizes

__all__ = ["DGCNN"]


@torch.no_grad()
def _knn_rows(values: Tensor, sizes: Sequence[int], k: int, chunk: int) -> Tensor:
    """``(N, k)`` rows of the ``k`` nearest points of every point inside its own cloud,
    by Euclidean distance between the rows of ``values`` (the point itself included)."""
    found, start = [], 0
    for size in sizes:
        block = values[start : start + size]
        for first in range(0, size, chunk):
            distance = torch.cdist(block[first : first + chunk], block)
            found.append(distance.topk(k, dim=-1, largest=False).indices + start)
        start += size
    return torch.cat(found)


def _edge_block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.LeakyReLU(negative_slope=0.2),
    )


def _point_block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(inputs, outputs, kernel_size=1, bias=False),
        nn.BatchNorm1d(outputs),
        nn.LeakyReLU(negative_slope=0.2),
    )


@BACKBONES.register()
class DGCNN(Backbone):
    """Dynamic Graph CNN with features on every input point.

    The defaults are the S3DIS segmentation network of the PyTorch reference: three
    EdgeConv layers of 64 channels with 20 neighbours, a 1024-channel global vector,
    256 output channels.

    Args:
        in_channels: Feature channels per point besides the coordinates.
        k: Neighbours per point. Every cloud needs more than ``k`` points.
        width: Channels of the EdgeConv layers.
        emb_dims: Channels of the vector that is max-pooled over each cloud and given
            back to all its points. It makes every feature depend on the whole cloud,
            hence on the workspace crop. 0 leaves it out.
        out_channels: Channels of the returned features.
        dynamic: Rebuild the graph in feature space for the second and third layer
            (the paper's method). ``false`` keeps the spatial graph of the first layer,
            which is cheaper and local.
        coords_as_features: Put the coordinates in front of the input features. Needed
            whenever the data pipeline feeds constants (``point_features: ones``),
            because an EdgeConv sees geometry only through its input channels.
        knn_chunk: Queries per block of the neighbour search; trades memory for speed.
    """

    output = "dense"
    status = "tested"
    source = "DGCNN (Wang et al., ACM TOG 2019), adapted from antao97/dgcnn.pytorch@07d534c, MIT"

    def __init__(
        self,
        in_channels: int = 0,
        k: int = 20,
        width: int = 64,
        emb_dims: int = 1024,
        out_channels: int = 256,
        dynamic: bool = True,
        coords_as_features: bool = True,
        knn_chunk: int = 2048,
    ) -> None:
        super().__init__()
        first = in_channels + (3 if coords_as_features else 0)
        if first == 0:
            raise ValueError(
                "DGCNN reads geometry through its input channels: with in_channels=0 it "
                "needs coords_as_features=true."
            )
        if k < 1 or knn_chunk < 1 or emb_dims < 0:
            raise ValueError("DGCNN: k and knn_chunk must be positive, emb_dims non-negative")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.k, self.dynamic, self.coords_as_features = k, dynamic, coords_as_features
        self.knn_chunk, self.emb_dims = knn_chunk, emb_dims
        self.conv1 = _edge_block(2 * first, width)
        self.conv2 = _edge_block(width, width)
        self.conv3 = _edge_block(2 * width, width)
        self.conv4 = _edge_block(width, width)
        self.conv5 = _edge_block(2 * width, width)
        if emb_dims:
            self.conv6 = _point_block(3 * width, emb_dims)
        self.conv7 = _point_block(emb_dims + 3 * width, 512)
        self.conv8 = _point_block(512, out_channels)

    def _edges(self, feats: Tensor, neighbours: Tensor, *layers: nn.Module) -> Tensor:
        """EdgeConv: shared MLP on ``(x_j - x_i, x_i)``, max over the ``k`` edges."""
        centre = feats[:, None, :].expand(-1, neighbours.shape[1], -1)
        edge = torch.cat([feats[neighbours] - centre, centre], dim=-1).permute(2, 0, 1)[None]
        for layer in layers:
            edge = layer(edge)
        return edge[0].amax(dim=-1).T

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="DGCNN")
        sizes = cloud_sizes(points)
        if min(sizes) <= self.k:
            raise ContractError(
                f"DGCNN connects every point to its {self.k} nearest neighbours, so every "
                f"cloud needs more than {self.k} points; cloud {sizes.index(min(sizes))} of "
                f"the batch has {min(sizes)}. Lower k or feed larger clouds."
            )
        coords = points.coords
        x = coords if feats is None else feats
        if self.coords_as_features and feats is not None:
            x = torch.cat([coords, feats], dim=-1)
        graph = _knn_rows(coords, sizes, self.k, self.knn_chunk)
        x1 = self._edges(x, graph, self.conv1, self.conv2)
        if self.dynamic:
            graph = _knn_rows(x1, sizes, self.k, self.knn_chunk)
        x2 = self._edges(x1, graph, self.conv3, self.conv4)
        if self.dynamic:
            graph = _knn_rows(x2, sizes, self.k, self.knn_chunk)
        x3 = self._edges(x2, graph, self.conv5)
        local = torch.cat([x1, x2, x3], dim=-1)
        fused = local
        if self.emb_dims:
            embedded = self.conv6(local.T[None])[0].T
            index = points.batch_index[:, None].expand(-1, self.emb_dims)
            pooled = embedded.new_zeros(len(sizes), self.emb_dims).scatter_reduce(
                0, index, embedded, reduce="amax", include_self=False
            )
            fused = torch.cat([pooled[points.batch_index], local], dim=-1)
        out = self.conv8(self.conv7(fused.T[None]))[0].T
        return PointFeatures(out, points)
