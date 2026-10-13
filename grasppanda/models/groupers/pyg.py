"""Message-passing layers of PyTorch Geometric as the encoder of a crop.

PyTorch Geometric ships the point operators of many papers as layers over a bipartite
graph. A crop is such a graph: its members are the source nodes, its seed is the target
node. :class:`PyGConvGroup` builds that graph in the *crop frame* (every member sits at
its offset ``Rᵀ (p - c) / radius`` and the seed at the origin), so that a layer which
computes ``pos_j - pos_i`` reads positions aligned with the approach direction, and hands
it to one of the layers below. Members of different seeds are different source nodes,
even when they are the same seed of the scene.

The library is optional: ``pip install -e ".[torch_geometric]"``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.ops import group_points
from grasppanda.registry import GROUPERS, INJECTED
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures
from grasppanda.utils.imports import require

from .local import SHAPES, crop_geometry, crop_members

__all__ = ["PYG_CONVS", "PyGConvGroup"]


def _pointnet(pyg: Any, channels: int, hidden: int, width: int, args: dict[str, Any]) -> Any:
    local = pyg.MLP([channels + 3, hidden, width], norm="batch_norm")
    return pyg.PointNetConv(local_nn=local, **{"add_self_loops": False, **args})


def _point_transformer(
    pyg: Any, channels: int, hidden: int, width: int, args: dict[str, Any]
) -> Any:
    options = {"add_self_loops": False, **args}
    return pyg.PointTransformerConv((channels, channels), width, **options)


def _edge(pyg: Any, channels: int, hidden: int, width: int, args: dict[str, Any]) -> Any:
    return pyg.EdgeConv(pyg.MLP([2 * (channels + 3), hidden, width], norm="batch_norm"), **args)


def _gatv2(pyg: Any, channels: int, hidden: int, width: int, args: dict[str, Any]) -> Any:
    options = {"heads": 4, "concat": False, "edge_dim": 3, "add_self_loops": False, **args}
    return pyg.GATv2Conv((channels, channels), width, **options)


def _transformer(pyg: Any, channels: int, hidden: int, width: int, args: dict[str, Any]) -> Any:
    options = {"heads": 4, "concat": False, "edge_dim": 3, **args}
    return pyg.TransformerConv((channels, channels), width, **options)


#: Layer name → (constructor, how the crop geometry reaches the layer).
#: ``pos``: as node positions; ``x``: appended to the node features (the seed gets zeros);
#: ``edge``: as edge attributes.
PYG_CONVS: dict[str, tuple[Callable[..., Any], str]] = {
    "PointNetConv": (_pointnet, "pos"),
    "PointTransformerConv": (_point_transformer, "pos"),
    "EdgeConv": (_edge, "x"),
    "GATv2Conv": (_gatv2, "edge"),
    "TransformerConv": (_transformer, "edge"),
}


@GROUPERS.register()
class PyGConvGroup(Grouper):
    """A crop among the seeds, encoded by a message-passing layer of PyTorch Geometric.

    Args:
        conv: The layer: ``PointNetConv`` (PointNet++), ``PointTransformerConv`` (Point
            Transformer), ``EdgeConv`` (DGCNN), ``GATv2Conv`` (GATv2, the offset as edge
            attribute) or ``TransformerConv`` (a graph transformer, likewise).
        conv_args: Further keyword arguments of the layer, as PyTorch Geometric names
            them (``{heads: 8}``); they override the ones set here.
        shape: ``ball``, ``knn``, ``box`` or ``cylinder``; see
            :func:`~grasppanda.models.groupers.local.crop_members`.
        radius: Ball or cylinder radius, half the side of the box section, metres.
        hmin: Near end of a box or cylinder along the approach axis, metres.
        hmax: Far end of a box or cylinder along the approach axis, metres.
        nsample: Members per crop at most; a crop with fewer has fewer edges (nothing is
            padded).
        hidden_channels: Hidden width of the MLP of ``PointNetConv`` and ``EdgeConv``.
        out_channels: Width of the crop feature.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "bridge"
    extras = ("torch_geometric",)
    source = (
        "PyTorch Geometric (Fey and Lenssen, 2019), MIT: PointNetConv, PointTransformerConv, "
        "EdgeConv, GATv2Conv, TransformerConv; via torch_geometric"
    )

    def __init__(
        self,
        conv: Literal[
            "PointNetConv", "PointTransformerConv", "EdgeConv", "GATv2Conv", "TransformerConv"
        ] = "PointNetConv",
        conv_args: Mapping[str, Any] | None = None,
        shape: Literal["ball", "knn", "box", "cylinder"] = "cylinder",
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        nsample: int = 16,
        hidden_channels: int = 256,
        out_channels: int = 256,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        if conv not in PYG_CONVS:
            raise ValueError(f"{name}: unknown conv '{conv}'; choose one of {list(PYG_CONVS)}")
        if shape not in SHAPES:
            raise ValueError(f"{name}: unknown shape '{shape}'; choose one of {list(SHAPES)}")
        if radius <= 0 or hmax <= hmin or nsample < 1 or out_channels < 1:
            raise ValueError(
                f"{name} needs radius > 0, hmin < hmax, nsample >= 1 and out_channels >= 1; got "
                f"radius={radius}, hmin={hmin}, hmax={hmax}, nsample={nsample}"
            )
        pyg = require("torch_geometric.nn", extra="torch_geometric", purpose=name)
        self.conv_name, self.conv_args = conv, dict(conv_args or {})
        self.shape, self.radius, self.hmin, self.hmax = shape, radius, hmin, hmax
        self.nsample, self.hidden_channels = nsample, hidden_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        build, self.geometry_as = PYG_CONVS[conv]
        self.conv = build(pyg, in_channels, hidden_channels, out_channels, self.conv_args)
        self.norm = nn.BatchNorm1d(out_channels)

    def graph(
        self, support: PointFeatures, centres: Tensor, rotation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The crops as one bipartite graph.

        Returns:
            ``(feats, offsets, edge_index)``: ``(E, in_channels)`` features and ``(E, 3)``
            crop-frame offsets of the members (one source node each) and the ``(2, E)``
            edges from member ``e`` to its seed.
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
        index, mask = crop_members(
            self.shape, centre, coords, frame, self.radius, self.hmin, self.hmax, self.nsample
        )
        offsets, _ = crop_geometry(coords, index, centre, frame, self.radius, "offset")
        mask = mask.reshape(count, self.nsample)
        seed = torch.arange(count, device=mask.device)[:, None].expand_as(mask)[mask]
        feats = group_points(support.dense(), index).reshape(count, self.nsample, -1)[mask]
        offsets = offsets.reshape(count, self.nsample, 3)[mask]
        edge_index = torch.stack([torch.arange(len(seed), device=seed.device), seed])
        return feats, offsets, edge_index

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        feats, offsets, edge_index = self.graph(support, centres, rotation)
        own = support.feats
        origin = offsets.new_zeros(len(own), 3)
        if self.geometry_as == "pos":
            out = self.conv((feats, own), (offsets, origin), edge_index)
        elif self.geometry_as == "x":
            source = torch.cat([offsets, feats], dim=-1)
            out = self.conv((source, torch.cat([origin, own], dim=-1)), edge_index)
        else:
            out = self.conv((feats, own), edge_index, edge_attr=offsets)
        return GroupFeatures(torch.relu(self.norm(out)), self.out_axes)

    def extra_repr(self) -> str:
        return (
            f"conv={self.conv_name!r}, shape={self.shape!r}, radius={self.radius}, "
            f"hmin={self.hmin}, hmax={self.hmax}, nsample={self.nsample}"
        )
