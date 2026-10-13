"""Bridge to the point-cloud convolutions of PyTorch Geometric.

PyTorch Geometric (Fey and Lenssen, "Fast Graph Representation Learning with PyTorch
Geometric", ICLR 2019 workshop; MIT licence) implements the message-passing layers of
many point-cloud papers. :class:`PyGPointEncoder` stacks one of them at full resolution
on a k-nearest-neighbour graph and returns a feature per input point:

* ``conv: pointnet``: ``PointNetConv``, the layer of PointNet++ (NeurIPS 2017);
* ``conv: edge``: ``EdgeConv``, the layer of DGCNN (ACM TOG 2019);
* ``conv: point_transformer``: ``PointTransformerConv``, the vector attention of Point
  Transformer (ICCV 2021);
* ``conv: pointgnn``: ``PointGNNConv``, the layer of Point-GNN (CVPR 2020).

The graph is built here, with a blocked neighbour search per cloud, so the bridge needs
``torch_geometric`` alone (its own ``knn_graph`` asks for further compiled packages).
Only the layers come from the library: the stack around them (stem, batch norm,
residual, output projection) is this module's and is not a published network.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet
from grasppanda.utils.imports import require

from ._packed import check_features, cloud_sizes
from .dgcnn import _knn_rows

__all__ = ["PyGPointEncoder"]


def _mlp(*widths: int) -> nn.Sequential:
    """Linear layers with batch norm and ReLU between them (none after the last)."""
    layers: list[nn.Module] = []
    for index in range(len(widths) - 1):
        layers.append(nn.Linear(widths[index], widths[index + 1]))
        if index < len(widths) - 2:
            layers += [nn.BatchNorm1d(widths[index + 1]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


@BACKBONES.register()
class PyGPointEncoder(Backbone):
    """A stack of PyTorch Geometric point convolutions with features on every input point.

    ``stem -> [conv -> batch norm -> ReLU (+ residual when widths agree)] x L -> linear``
    on one spatial k-nearest-neighbour graph per frame. No subsampling, so the receptive
    field is ``L`` hops of ``k`` neighbours: local features, comparable to a shallow
    sparse CNN, not to a U-Net.

    Args:
        in_channels: Feature channels per point besides the coordinates.
        conv: Which layer to stack: ``pointnet`` (max over an MLP of neighbour feature
            and offset), ``edge`` (max over an MLP of ``(x_i, x_j - x_i)``),
            ``point_transformer`` (vector attention with a learned position encoding),
            ``pointgnn`` (Point-GNN's update with a learned alignment offset; all
            ``channels`` must be equal).
        channels: Output width of every layer.
        out_channels: Channels of the returned features.
        k: Neighbours per point (the point itself included). Every cloud needs at least
            ``k`` points.
        coords_as_features: Put the coordinates in front of the input features.
            ``edge`` sees geometry only this way; the other layers also receive the
            neighbour offsets from the library.
        conv_args: Further keyword arguments of the library layer, e.g.
            ``{aggr: mean}``.
        knn_chunk: Queries per block of the neighbour search.
    """

    output = "dense"
    status = "bridge"
    source = (
        "PointNetConv / EdgeConv / PointTransformerConv / PointGNNConv via "
        "torch_geometric (Fey and Lenssen, ICLR-W 2019)"
    )
    extras = ("torch_geometric",)

    def __init__(
        self,
        in_channels: int = 0,
        conv: Literal["pointnet", "edge", "point_transformer", "pointgnn"] = "pointnet",
        channels: Sequence[int] = (64, 64, 128),
        out_channels: int = 256,
        k: int = 16,
        coords_as_features: bool = True,
        conv_args: dict[str, Any] | None = None,
        knn_chunk: int = 2048,
    ) -> None:
        super().__init__()
        pyg = require(
            "torch_geometric.nn", extra="torch_geometric", purpose="The backbone PyGPointEncoder"
        )
        first = in_channels + (3 if coords_as_features else 0)
        if first == 0:
            raise ValueError(
                "PyGPointEncoder: with in_channels=0 the layers need coords_as_features=true."
            )
        if not channels or k < 1:
            raise ValueError("PyGPointEncoder: channels needs one width per layer and k >= 1")
        if conv == "pointgnn" and len(set(channels)) != 1:
            raise ValueError(
                f"PyGPointEncoder: PointGNNConv keeps the width of its input, so all channels "
                f"must be equal for conv=pointgnn, got {list(channels)}"
            )
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv, self.k, self.knn_chunk = conv, k, knn_chunk
        self.coords_as_features = coords_as_features
        extra = dict(conv_args or {})
        self.stem = _mlp(first, channels[0])
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        widths = [channels[0], *channels]
        for inputs, outputs in pairwise(widths):
            if conv == "pointnet":
                layer = pyg.PointNetConv(
                    _mlp(inputs + 3, outputs, outputs), add_self_loops=False, **extra
                )
            elif conv == "edge":
                layer = pyg.EdgeConv(_mlp(2 * inputs, outputs, outputs), **extra)
            elif conv == "point_transformer":
                layer = pyg.PointTransformerConv(inputs, outputs, add_self_loops=False, **extra)
            elif conv == "pointgnn":
                layer = pyg.PointGNNConv(
                    _mlp(inputs, inputs, 3),
                    _mlp(inputs + 3, outputs, outputs),
                    _mlp(outputs, outputs, outputs),
                    **extra,
                )
            else:
                raise ValueError(
                    f"PyGPointEncoder: unknown conv '{conv}'; choose pointnet, edge, "
                    "point_transformer or pointgnn"
                )
            self.convs.append(layer)
            self.norms.append(nn.BatchNorm1d(outputs))
        self.head = nn.Linear(channels[-1], out_channels)

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        feats = check_features(points, feats, self.in_channels, user="PyGPointEncoder")
        sizes = cloud_sizes(points)
        if min(sizes) < self.k:
            raise ContractError(
                f"PyGPointEncoder connects every point to {self.k} neighbours, so every cloud "
                f"needs at least {self.k} points; cloud {sizes.index(min(sizes))} of the batch "
                f"has {min(sizes)}. Lower k or feed larger clouds."
            )
        coords = points.coords
        x = coords if feats is None else feats
        if self.coords_as_features and feats is not None:
            x = torch.cat([coords, feats], dim=-1)
        rows = _knn_rows(coords, sizes, self.k, self.knn_chunk)
        target = torch.arange(len(coords), device=coords.device).repeat_interleave(self.k)
        edges = torch.stack([rows.reshape(-1), target])  # messages flow neighbour -> point
        x = self.stem(x)
        for layer, norm in zip(self.convs, self.norms):
            out = layer(x, edges) if self.conv == "edge" else layer(x, coords, edges)
            out = torch.relu(norm(out))
            x = out + x if out.shape == x.shape else out
        return PointFeatures(self.head(x), points)
