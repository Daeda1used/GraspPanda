"""Cylinder crops over the scene cloud, one per insertion depth.

For every seed the grouper gathers, from the *scene points*, the neighbours inside a
cylinder aligned with the chosen approach direction, once for each depth bin: all
cylinders of a seed share the centre, the radius and the near end, and differ in how
far they reach along the approach axis. The offsets of the neighbours, expressed in the
crop frame, are reduced by a shared MLP and a max over neighbours. The result keeps the
depth axis: ``(seed, depth, channel)``, one feature vector per (seed, depth bin).

This is the crop of GraspNet-baseline. Two things set it apart from
:class:`~grasppanda.models.groupers.cylinder.CylinderGroup`, and neither is a matter of
resolution: the support is the raw scene geometry (thousands of points, no learned
features) instead of the seeds with their features, and the output has an explicit
depth axis that the head bundle must be written for.

Scale-Balanced-Grasp crops at several radii (``radius_scales``), each with its own MLP,
fuses the results with one linear layer, and adds the seed features back through a
learned gate (``seed_residual``). Both are options here because the support, the
geometry of a crop and the output axes are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.ops import cylinder_query, group_points
from grasppanda.registry import GROUPERS, INJECTED
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures

from .cylinder import _LinearNormAct

__all__ = ["DepthCylinderGroup", "DepthCylinderProjectionGroup"]


@GROUPERS.register()
class DepthCylinderGroup(Grouper):
    """Approach-aligned cylinders of increasing reach over the scene cloud.

    A scene point ``p`` belongs to the crop of depth bin ``d`` of a seed with centre
    ``c`` and crop frame ``R`` when ``o = Rᵀ (p - c)`` satisfies ``hmin < o.x < hmax[d]``
    and ``o.y² + o.z² < r²``; column 0 of ``R`` is the approach axis and ``r`` is
    ``radius`` times the scale of the crop. The first ``nsample`` members in index order
    are used and missing slots repeat the first. A cylinder without any member is filled
    with scene point 0 of its frame, as the reference kernel does, so its feature
    describes one far-away point rather than being zero. The feature of a neighbour is
    ``o`` in metres (``o / r`` with ``normalize_offsets``).

    Args:
        radius: Cylinder radius in metres.
        hmin: Near end of every cylinder along the approach axis, metres (negative =
            behind the crop centre).
        hmax: Far end of the cylinder of each depth bin, metres, increasing. Its length
            is the size of the depth axis; the reference uses the insertion depths of
            the label grid, 0.01 to 0.04.
        nsample: Neighbours per cylinder.
        mlp_channels: Widths of the shared MLP; the last one is ``out_channels``.
        normalize_offsets: Divide the offsets by the radius of their crop. Reference
            value: false.
        radius_scales: One crop per entry, with radius ``radius * scale`` and an MLP of
            its own. With more than one, the features of all crops of a (seed, depth)
            are concatenated in this order and fused by a linear layer. GraspNet-baseline
            uses ``(1.0,)``; Scale-Balanced-Grasp ``(0.25, 0.5, 0.75, 1.0)`` of 0.08 m.
        seed_residual: ``gated`` adds ``sigmoid(W f) * f`` to the crop features of every
            depth bin, with ``f`` the feature of the seed (Scale-Balanced-Grasp).
            ``none`` uses the scene geometry only.
        in_channels: Channels of the seed features; injected by the detector. Only the
            gated seed residual reads it, and it must then equal ``out_channels``.
    """

    support = "scene"
    out_axes = ("seed", "depth", "channel")
    status = "tested"
    source = "GraspNet-baseline (Fang et al., CVPR 2020)"
    #: The grouper to use instead when the encoder returns another width than the gated
    #: seed residual needs; named by the seed-width rule and applied by ``compose``.
    projected_variant: ClassVar[str | None] = "DepthCylinderProjectionGroup"

    def __init__(
        self,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: Sequence[float] = (0.01, 0.02, 0.03, 0.04),
        nsample: int = 64,
        mlp_channels: Sequence[int] = (64, 128, 256),
        normalize_offsets: bool = False,
        radius_scales: Sequence[float] = (1.0,),
        seed_residual: Literal["none", "gated"] = "none",
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        hmax = tuple(float(value) for value in hmax)
        scales = tuple(float(value) for value in radius_scales)
        if (
            radius <= 0
            or nsample < 1
            or not mlp_channels
            or not hmax
            or hmax[0] <= hmin
            or any(later <= earlier for earlier, later in pairwise(hmax))
            or not scales
            or min(scales) <= 0
        ):
            raise ValueError(
                "DepthCylinderGroup needs radius > 0, nsample >= 1, at least one MLP width, "
                f"positive radius_scales and hmin < hmax[0] < hmax[1] < ...; got radius={radius}, "
                f"hmin={hmin}, hmax={list(hmax)}, nsample={nsample}, "
                f"mlp_channels={list(mlp_channels)}, radius_scales={list(scales)}"
            )
        if seed_residual not in ("none", "gated"):
            raise ValueError(f"unknown seed_residual '{seed_residual}'")
        self.radius, self.hmin, self.hmax, self.nsample = radius, hmin, hmax, nsample
        self.normalize_offsets = normalize_offsets
        self.radius_scales = scales
        self.seed_residual = seed_residual
        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        #: Whether ``forward`` takes the seed features as a fourth argument.
        self.uses_seed_features = seed_residual != "none"
        if self.uses_seed_features and in_channels != self.out_channels:
            raise ValueError(
                f"DepthCylinderGroup: the gated seed residual adds {in_channels}-channel seed "
                f"features to {self.out_channels}-channel crop features; the two must be equal "
                "(in_channels is injected by the detector: it is the width of the scene "
                f"encoder). Use DepthCylinderProjectionGroup (a linear layer in front of the "
                f"gate), or end mlp_channels on {in_channels}, or use an encoder that returns "
                f"{self.out_channels} channels, or set seed_residual: none."
            )
        widths = [3, *mlp_channels]
        self.scales = nn.ModuleList(
            nn.Sequential(*(_LinearNormAct(a, b) for a, b in pairwise(widths))) for _ in scales
        )
        self.fuse = (
            nn.Linear(len(scales) * self.out_channels, self.out_channels)
            if len(scales) > 1
            else None
        )
        self.gate = (
            nn.Linear(self.out_channels, self.out_channels) if self.uses_seed_features else None
        )

    @classmethod
    def required_seed_channels(cls, options: Mapping[str, Any]) -> int | None:
        """Width the seed features must have under ``options``, or ``None`` for any.

        The gated seed residual adds the seed features to the crop features, so the
        encoder has to return the last width of ``mlp_channels``. The contract rules
        compare this with the width of the encoder.
        """
        if options.get("seed_residual") != "gated":
            return None
        widths = options.get("mlp_channels") or ()
        return int(widths[-1]) if widths else None

    @property
    def num_depth(self) -> int:
        """Size of the depth axis."""
        return len(self.hmax)

    @property
    def radii(self) -> tuple[float, ...]:
        """Radius of every crop, metres."""
        return tuple(self.radius * scale for scale in self.radius_scales)

    def neighbours(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> Tensor:
        """Scene points of every crop: ``(B, S, scales, num_depth, nsample)`` indices into
        each frame.

        Args:
            support: The scene set (any features on it are not used: the crop is
                geometry only).
            centres: ``(M, 3)`` crop centre of every seed, ``M = B * S``.
            rotation: ``(M, 3, 3)`` crop frame of every seed.
        """
        coords, centre, frame = self._dense(support, centres, rotation)
        return torch.stack(
            [
                torch.stack(
                    [
                        cylinder_query(
                            centre, coords, frame, radius, self.hmin, hmax, self.nsample
                        )[0]
                        for hmax in self.hmax
                    ],
                    dim=2,
                )
                for radius in self.radii
            ],
            dim=2,
        )

    def group(
        self,
        support: PointFeatures,
        centres: Tensor,
        rotation: Tensor,
        neighbours: Tensor | None = None,
    ) -> Tensor:
        """Gather the neighbourhoods: ``(M, scales, num_depth, nsample, 3)`` offsets in the
        crop frame.

        ``neighbours`` replaces the cylinder query by recorded indices (the result of
        :meth:`neighbours`, possibly from a reference implementation); membership of a
        point on a cylinder boundary is decided by the last bit of a float32 product.
        """
        coords, centre, frame = self._dense(support, centres, rotation)
        batch, seeds = centre.shape[:2]
        shape = (batch, seeds, len(self.radii), self.num_depth, self.nsample)
        if neighbours is None:
            neighbours = self.neighbours(support, centres, rotation)
        elif tuple(neighbours.shape) != shape:
            raise ContractError(f"neighbours must be {shape}, got {tuple(neighbours.shape)}")
        flat = neighbours.to(coords.device).long().reshape(batch, seeds, -1)
        offset = group_points(coords, flat) - centre[:, :, None, :]
        local = torch.matmul(offset, frame)  # row vectors: Rᵀ (p - c) for every neighbour
        local = local.reshape(batch * seeds, *shape[2:], 3)
        if self.normalize_offsets:
            local = local / local.new_tensor(self.radii)[None, :, None, None, None]
        return local

    def forward(
        self,
        support: PointFeatures,
        centres: Tensor,
        rotation: Tensor,
        seed_feats: Tensor | None = None,
        neighbours: Tensor | None = None,
    ) -> GroupFeatures:
        """Crop features ``(M, num_depth, out_channels)`` with axes ``(seed, depth, channel)``.

        Args:
            support, centres, rotation: As in :meth:`neighbours`.
            seed_feats: ``(M, in_channels)`` features of the seeds; required with a seed
                residual and refused without one.
            neighbours: Recorded neighbour indices, see :meth:`group`.
        """
        name = type(self).__name__
        grouped = self.group(support, centres, rotation, neighbours)
        seeds, _, depths, samples, _ = grouped.shape
        pooled = [
            mlp(grouped[:, index].reshape(-1, 3))
            .reshape(seeds, depths, samples, self.out_channels)
            .amax(dim=2)
            for index, mlp in enumerate(self.scales)
        ]
        feats = pooled[0] if self.fuse is None else self.fuse(torch.cat(pooled, dim=-1))
        if self.gate is None:
            if seed_feats is not None:
                raise ContractError(
                    f"{name} was handed seed features, but it is configured with "
                    "seed_residual: none and would ignore them"
                )
        else:
            if seed_feats is None or tuple(seed_feats.shape) != (seeds, self.out_channels):
                got = None if seed_feats is None else tuple(seed_feats.shape)
                raise ContractError(
                    f"{name} with seed_residual: {self.seed_residual} needs the "
                    f"({seeds}, {self.out_channels}) features of the seeds, got {got}"
                )
            feats = feats + (torch.sigmoid(self.gate(seed_feats)) * seed_feats)[:, None, :]
        return GroupFeatures(feats, self.out_axes)

    def _dense(
        self, support: PointFeatures, centres: Tensor, rotation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-frame views ``(B, N, 3)``, ``(B, S, 3)`` and ``(B, S, 3, 3)`` of the inputs."""
        name = type(self).__name__
        coords = support.points.dense()
        batch = coords.shape[0]
        count = centres.shape[0]
        if (
            centres.ndim != 2
            or centres.shape[1] != 3
            or tuple(rotation.shape) != (count, 3, 3)
            or count % batch
        ):
            raise ContractError(
                f"{name} groups over the scene (support = '{self.support}'): it needs one "
                f"centre (M, 3) and one frame (M, 3, 3) per seed, with the same number of seeds "
                f"in each of the {batch} frame(s), got {tuple(centres.shape)} and "
                f"{tuple(rotation.shape)}"
            )
        return coords, centres.reshape(batch, -1, 3), rotation.reshape(batch, -1, 3, 3)

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, radius_scales={list(self.radius_scales)}, hmin={self.hmin}, "
            f"hmax={list(self.hmax)}, nsample={self.nsample}, seed_residual={self.seed_residual!r}"
        )


@GROUPERS.register()
class DepthCylinderProjectionGroup(DepthCylinderGroup):
    """:class:`DepthCylinderGroup` with a gated seed residual for an encoder of any width.

    The crop of Scale-Balanced-Grasp adds ``sigmoid(W f) * f`` of the seed feature ``f`` to
    its crop features, so its encoder must return the width of the crops. Here a linear
    layer ``in_channels -> out_channels`` maps the seed feature first: ``g = P f``, then
    ``sigmoid(W g) * g`` is added. The reference model has no such layer; this grouper is
    what ``grasppanda compose`` puts in its place under an encoder of another width.

    Args:
        radius: Cylinder radius in metres.
        hmin: Near end of every cylinder along the approach axis, metres.
        hmax: Far end of the cylinder of each depth bin, metres, increasing.
        nsample: Neighbours per cylinder.
        mlp_channels: Widths of the shared MLP; the last one is ``out_channels``.
        normalize_offsets: Divide the offsets by the radius of their crop.
        radius_scales: One crop per entry, with radius ``radius * scale``.
        in_channels: Channels of the seed features; injected by the detector. Any width.
    """

    status = "tested"
    source = (
        "Scale-Balanced-Grasp (Ma and Huang, CoRL 2022) crop with a projection of the seed "
        "feature added here"
    )
    projected_variant = None

    def __init__(
        self,
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: Sequence[float] = (0.01, 0.02, 0.03, 0.04),
        nsample: int = 64,
        mlp_channels: Sequence[int] = (64, 128, 256),
        normalize_offsets: bool = False,
        radius_scales: Sequence[float] = (1.0,),
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__(
            radius,
            hmin,
            hmax,
            nsample,
            mlp_channels,
            normalize_offsets,
            radius_scales,
            "gated",
            int(mlp_channels[-1]) if mlp_channels else in_channels,
        )
        self.in_channels = in_channels
        self.project = nn.Linear(in_channels, self.out_channels)

    @classmethod
    def required_seed_channels(cls, options: Mapping[str, Any]) -> int | None:
        """Any width: the projection maps it."""
        return None

    def forward(
        self,
        support: PointFeatures,
        centres: Tensor,
        rotation: Tensor,
        seed_feats: Tensor | None = None,
        neighbours: Tensor | None = None,
    ) -> GroupFeatures:
        """Crop features ``(M, num_depth, out_channels)`` with axes ``(seed, depth, channel)``."""
        if seed_feats is None or seed_feats.ndim != 2 or seed_feats.shape[1] != self.in_channels:
            got = None if seed_feats is None else tuple(seed_feats.shape)
            raise ContractError(
                f"{type(self).__name__} needs the (M, {self.in_channels}) features of the "
                f"seeds, got {got}"
            )
        return super().forward(support, centres, rotation, self.project(seed_feats), neighbours)
