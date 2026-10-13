"""HGGD's region network: a PointNet over one region plus what the image stage knows.

A region is ``S`` points with relative coordinates and the image features attached to
them. The coordinates pass a learned 3×3 alignment (the spatial transformer of
PointNet), are embedded, concatenated with the image features, embedded further and
max-pooled into one vector. That vector is joined with an embedding of the partial
grasp ``(theta, width, depth offset)`` and read out twice: one logit per rotation
anchor, and one 3-D offset per rotation anchor.

The side input enters in the units the reference feeds it: radians, pixels of the
network image and **millimetres**; trained weights depend on that scale.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from grasppanda.structures import ContractError

__all__ = ["LocalNet"]


class _PointLayer(nn.Module):
    """Shared linear layer over points, batch norm, optional ReLU: ``(R, S, C)`` in and out."""

    def __init__(self, in_channels: int, out_channels: int, act: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.linear(x).transpose(1, 2)).transpose(1, 2)
        return torch.relu(y) if self.act else y


class _VectorLayer(nn.Module):
    """Linear layer on vectors, batch norm, ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.norm(self.linear(x)))


class _Alignment(nn.Module):
    """Predicts a 3×3 matrix (identity plus a learned residual) from the coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.points = nn.Sequential(
            _PointLayer(3, 64), _PointLayer(64, 128), _PointLayer(128, 1024)
        )
        self.vector = nn.Sequential(_VectorLayer(1024, 512), _VectorLayer(512, 256))
        self.out = nn.Linear(256, 9)

    def forward(self, xyz: Tensor) -> Tensor:
        pooled = self.points(xyz).amax(dim=1)
        residual = self.out(self.vector(pooled)).reshape(-1, 3, 3)
        return residual + torch.eye(3, dtype=xyz.dtype, device=xyz.device)


class _Readout(nn.Module):
    """Linear, layer norm, ReLU, linear."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_channels, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(torch.relu(self.norm(self.hidden(x))))


class LocalNet(nn.Module):
    """``(R, S, 3 + C)`` regions and ``(R, 3)`` partial-grasp info → anchor logits and offsets.

    Args:
        feature_channels: Image-feature channels attached to every region point.
        num_anchors: Rotation anchors (the product of the two anchor counts).
        info_channels: Size of the side input.
        info_embedding: Width of its embedding.
        hidden: Width of the two read-outs.
        dropout: Dropout probability on the region vector.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        num_anchors: int = 49,
        info_channels: int = 3,
        info_embedding: int = 32,
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.num_anchors = num_anchors
        self.align = _Alignment()
        self.embed = _PointLayer(3, 64)
        self.fuse = _PointLayer(64 + feature_channels, 128)
        self.lift = _PointLayer(128, 1024, act=False)
        self.reduce = nn.Linear(1024, 512)
        self.reduce_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(dropout)
        self.info = nn.Linear(info_channels, info_embedding)
        self.anchor = _Readout(512 + info_embedding, hidden, num_anchors)
        self.offset = _Readout(512 + info_embedding, hidden, num_anchors * 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """He initialisation of the layers outside the point branch; biases zero there."""
        for module in (
            self.reduce,
            self.info,
            self.anchor.hidden,
            self.anchor.out,
            self.offset.hidden,
            self.offset.out,
            self.align.vector[0].linear,
            self.align.vector[1].linear,
            self.align.out,
        ):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def region_vector(self, regions: Tensor) -> Tensor:
        """``(R, 1024)`` max-pooled descriptor of each region."""
        if regions.ndim != 3 or regions.shape[-1] != 3 + self.feature_channels:
            raise ContractError(
                f"LocalNet expects regions (R, S, {3 + self.feature_channels}), got "
                f"{tuple(regions.shape)}"
            )
        xyz, feats = regions[..., :3], regions[..., 3:]
        aligned = xyz @ self.align(xyz)
        x = torch.cat([self.embed(aligned), feats], dim=-1)
        return self.lift(self.fuse(x)).amax(dim=1)

    def forward(self, regions: Tensor, info: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(R, A)`` anchor logits and ``(R, A, 3)`` raw offsets."""
        if info.ndim != 2 or info.shape[0] != regions.shape[0]:
            raise ContractError(
                f"LocalNet got {tuple(info.shape)} side inputs for {regions.shape[0]} regions"
            )
        vector = torch.relu(
            self.dropout(self.reduce_norm(self.reduce(self.region_vector(regions))))
        )
        joint = torch.cat([vector, self.info(info)], dim=1)
        return self.anchor(joint), self.offset(joint).reshape(-1, self.num_anchors, 3)
