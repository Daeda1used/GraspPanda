"""Local crops with interchangeable parts: the query shape, the encoder, the pooling.

Every crop of the seed-based family is the same five steps; the method ports register
them as one class each, with the parts fixed. The two groupers of this module leave the
parts open, so that the well-known alternatives can be tried by changing one option:

``shape``
    what is gathered around a seed, in the frame of its approach direction:
    ``ball`` (PointNet++), ``knn`` (the k nearest points, DGCNN), ``box`` (an oriented
    box, the RoI point pooling of PointRCNN) or ``cylinder`` (the grasp methods);
``padding``
    what fills the slots of a crop with fewer than ``nsample`` members: ``first``
    repeats the first member (the PointNet++ kernels) and marks the copies, so that a
    mean or an attention does not count them; ``cyclic`` repeats the members in turn
    and counts every slot (PointRCNN);
``geometry``
    how a neighbour's position is written: its offset in the crop frame divided by the
    radius, optionally with its length (the relative point position encoding of
    RandLA-Net, in the crop frame);
``encoder``
    ``mlp`` (a shared MLP over ``[geometry, f_j]``, PointNet++), ``edge`` (over
    ``[geometry, f_j - f_i, f_i]``, the edge feature of DGCNN) or ``invres`` (the shared
    MLP, then the inverted residual MLP of PointNeXt with the seed feature as identity);
``pool``
    ``max``, ``mean``, ``attentive`` (learned soft-max weights per channel, RandLA-Net)
    or ``vector_attention`` (the layer of Point Transformer, the seed as query).

:class:`LocalGroup` crops among the seeds and returns ``(seed, channel)`` features, like
the crops of GSNet, EconomicGrasp and FineGrasp. :class:`DepthLocalGroup` crops the
scene cloud and returns ``(seed, depth, channel)``, like the crop of GraspNet-baseline:
one feature per insertion depth, pooled over the members in front of that depth.

Nothing here was copied: the parts are a few lines each on the operators of
:mod:`grasppanda.ops`, written from the papers named in ``source``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.ops import ball_query, box_query, cylinder_query, group_points, knn
from grasppanda.registry import GROUPERS, INJECTED
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures

from .cylinder import _LinearNormAct

__all__ = ["DepthLocalGroup", "LocalGroup", "crop_members", "masked_pool"]

SHAPES = ("ball", "knn", "box", "cylinder")
PADDINGS = ("first", "cyclic")
GEOMETRIES = ("offset", "offset+distance")


def crop_members(
    shape: str,
    centre: Tensor,
    coords: Tensor,
    frame: Tensor,
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    padding: str = "first",
) -> tuple[Tensor, Tensor]:
    """Members of every crop: ``(B, M, nsample)`` indices and a mask of the real ones.

    Args:
        shape: ``ball`` (within ``radius`` of the centre), ``knn`` (the ``nsample``
            nearest), ``box`` (``hmin < x < hmax`` along the approach axis and
            ``|y|, |z| < radius`` in the crop frame) or ``cylinder`` (``hmin < x < hmax``
            and ``y² + z² < radius²``).
        centre: ``(B, M, 3)`` crop centres.
        coords: ``(B, N, 3)`` points to search.
        frame: ``(B, M, 3, 3)`` crop frames; column 0 is the approach axis.
        padding: ``first`` repeats the first member in empty slots and masks them;
            ``cyclic`` repeats the members in turn and masks nothing.

    A crop without any member holds point 0 of its frame in every slot, as the
    PointNet++ kernels do, and its first slot counts as real, so that no pooling ever
    reduces an empty set.
    """
    if shape == "ball":
        index, hits = ball_query(centre, coords, radius, nsample)
    elif shape == "knn":
        nearest = min(nsample, coords.shape[1])
        index = knn(centre, coords, nearest)[1]
        if nearest < nsample:
            index = torch.cat([index, index[..., :1].expand(-1, -1, nsample - nearest)], dim=-1)
        hits = index.new_full(index.shape[:2], nearest)
    elif shape == "box":
        index, hits = box_query(
            centre, coords, frame, (hmin, -radius, -radius), (hmax, radius, radius), nsample
        )
    elif shape == "cylinder":
        index, hits = cylinder_query(centre, coords, frame, radius, hmin, hmax, nsample)
    else:
        raise ValueError(f"unknown crop shape '{shape}'; choose one of {list(SHAPES)}")
    index = index.long()
    real = hits.clamp(min=1)[..., None]
    slot = torch.arange(nsample, device=index.device).expand_as(index)
    if padding == "cyclic":
        return torch.gather(index, -1, slot % real), torch.ones_like(index, dtype=torch.bool)
    return index, slot < real


def crop_geometry(
    coords: Tensor, index: Tensor, centre: Tensor, frame: Tensor, scale: float, geometry: str
) -> tuple[Tensor, Tensor]:
    """Neighbour positions in the crop frame.

    Returns:
        ``(encoded, local)``: ``(B, M, K, 3 or 4)`` offsets ``Rᵀ (p - c) / scale``, with
        their length appended for ``offset+distance``, and the offsets in metres.
    """
    offset = group_points(coords, index) - centre[:, :, None, :]
    local = torch.matmul(offset, frame)  # row vectors: Rᵀ (p - c) for every neighbour
    encoded = local / scale
    if geometry == "offset+distance":
        encoded = torch.cat([encoded, encoded.norm(dim=-1, keepdim=True)], dim=-1)
    return encoded, local


def masked_pool(hidden: Tensor, mask: Tensor, pool: str, score: nn.Module | None) -> Tensor:
    """Reduce ``(M, K, C)`` neighbour features to ``(M, C)`` over the slots in ``mask``.

    ``max`` and ``mean`` see the masked slots only; a row without any yields zeros.
    ``attentive`` weights every channel of a neighbour by a soft-max over the neighbours
    of a learned score (``score``: a linear map ``C -> C``).
    """
    keep = mask[..., None]
    if pool == "max":
        pooled = hidden.masked_fill(~keep, float("-inf")).amax(dim=1)
        return torch.where(mask.any(dim=1)[:, None], pooled, torch.zeros_like(pooled))
    if pool == "mean":
        return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
    if pool == "attentive":
        assert score is not None
        weight = torch.softmax(score(hidden).masked_fill(~keep, -1e4), dim=1) * keep
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (weight * hidden).sum(dim=1)
    raise ValueError(f"unknown pooling '{pool}'")


def _shared_mlp(widths: Sequence[int]) -> nn.Sequential:
    return nn.Sequential(*(_LinearNormAct(a, b) for a, b in pairwise(widths)))


def _tokens(mlp: nn.Module, grouped: Tensor) -> Tensor:
    """A shared MLP over ``(M, K, C)`` neighbours (batch norm over all of them)."""
    seeds, neighbours, channels = grouped.shape
    return mlp(grouped.reshape(seeds * neighbours, channels)).reshape(seeds, neighbours, -1)


def _check_common(name: str, shape: str, padding: str, geometry: str) -> None:
    for value, allowed, what in (
        (shape, SHAPES, "shape"),
        (padding, PADDINGS, "padding"),
        (geometry, GEOMETRIES, "geometry"),
    ):
        if value not in allowed:
            raise ValueError(f"{name}: unknown {what} '{value}'; choose one of {list(allowed)}")


@GROUPERS.register()
class LocalGroup(Grouper):
    """A crop among the seeds whose shape, encoder and pooling are options.

    The crop of a seed is expressed in its crop frame (column 0 of the given rotation is
    the approach axis), so the feature is aligned with the approach direction whatever
    the shape. With the defaults it is a PointNet++ ball: a shared MLP over
    ``[offset / radius, f_j]`` and a maximum.

    Args:
        shape: ``ball``, ``knn``, ``box`` or ``cylinder``; see :func:`crop_members`.
        radius: Ball or cylinder radius, half the side of the box section, metres. For
            ``knn`` only the unit the offsets are divided by.
        hmin: Near end of a box or cylinder along the approach axis, metres.
        hmax: Far end of a box or cylinder along the approach axis, metres.
        nsample: Neighbours per crop (``k`` of ``knn``).
        padding: ``first`` or ``cyclic``; see :func:`crop_members`.
        geometry: ``offset`` or ``offset+distance``.
        encoder: ``mlp``, ``edge`` or ``invres`` (see the module text).
        pool: ``max``, ``mean``, ``attentive`` or ``vector_attention``.
        mlp_channels: Widths of the shared MLP; the last one is ``out_channels``.
        expansion: Width factor of the hidden layer of ``invres``.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "tested"
    source = (
        "Parts from PointNet++ (Qi et al., NeurIPS 2017), DGCNN (Wang et al., TOG 2019), "
        "PointRCNN (Shi et al., CVPR 2019), RandLA-Net (Hu et al., CVPR 2020), Point "
        "Transformer (Zhao et al., ICCV 2021) and PointNeXt (Qian et al., NeurIPS 2022), "
        "written here on grasppanda.ops"
    )

    def __init__(
        self,
        shape: Literal["ball", "knn", "box", "cylinder"] = "ball",
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: float = 0.04,
        nsample: int = 16,
        padding: Literal["first", "cyclic"] = "first",
        geometry: Literal["offset", "offset+distance"] = "offset",
        encoder: Literal["mlp", "edge", "invres"] = "mlp",
        pool: Literal["max", "mean", "attentive", "vector_attention"] = "max",
        mlp_channels: Sequence[int] = (256, 256),
        expansion: int = 4,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        _check_common(name, shape, padding, geometry)
        if encoder not in ("mlp", "edge", "invres"):
            raise ValueError(f"{name}: unknown encoder '{encoder}'")
        if pool not in ("max", "mean", "attentive", "vector_attention"):
            raise ValueError(f"{name}: unknown pool '{pool}'")
        if radius <= 0 or hmax <= hmin or nsample < 1 or not mlp_channels or expansion < 1:
            raise ValueError(
                f"{name} needs radius > 0, hmin < hmax, nsample >= 1, expansion >= 1 and at "
                f"least one MLP width; got radius={radius}, hmin={hmin}, hmax={hmax}, "
                f"nsample={nsample}, expansion={expansion}, mlp_channels={list(mlp_channels)}"
            )
        self.shape, self.radius, self.hmin, self.hmax = shape, radius, hmin, hmax
        self.nsample, self.padding, self.geometry = nsample, padding, geometry
        self.encoder, self.pool, self.expansion = encoder, pool, expansion
        self.in_channels = in_channels
        self.out_channels = width = int(mlp_channels[-1])
        self.geometry_channels = 4 if geometry == "offset+distance" else 3
        feature_channels = 2 * in_channels if encoder == "edge" else in_channels
        self.mlp = _shared_mlp([self.geometry_channels + feature_channels, *mlp_channels])
        self.score = nn.Linear(width, width, bias=False) if pool == "attentive" else None
        self.after_pool = _LinearNormAct(width, width) if pool == "attentive" else None
        if pool == "vector_attention":
            self.query = nn.Linear(in_channels, width, bias=False)
            self.key = nn.Linear(width, width, bias=False)
            self.value = nn.Linear(width, width, bias=False)
            self.position = nn.Sequential(
                nn.Linear(self.geometry_channels, width), nn.ReLU(), nn.Linear(width, width)
            )
            self.attention = nn.Sequential(
                nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
            )
        if encoder == "invres":
            hidden = width * expansion
            self.expand = _LinearNormAct(width, hidden)
            self.contract = nn.Sequential(
                nn.Linear(hidden, width, bias=False), nn.BatchNorm1d(width)
            )
            self.skip = nn.Identity() if in_channels == width else nn.Linear(in_channels, width)

    def group(
        self, support: PointFeatures, centres: Tensor, rotation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gather the crops.

        Returns:
            ``(geometry, feats, mask, own)``: ``(M, K, 3 or 4)`` encoded positions,
            ``(M, K, in_channels)`` features of the neighbours, ``(M, K)`` mask of the
            real members and ``(M, in_channels)`` feature of every seed itself.
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
            self.shape,
            centre,
            coords,
            frame,
            self.radius,
            self.hmin,
            self.hmax,
            self.nsample,
            self.padding,
        )
        geometry, _ = crop_geometry(coords, index, centre, frame, self.radius, self.geometry)
        feats = group_points(support.dense(), index)
        return geometry.flatten(0, 1), feats.flatten(0, 1), mask.flatten(0, 1), support.feats

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        geometry, feats, mask, own = self.group(support, centres, rotation)
        if self.encoder == "edge":
            centre_feats = own[:, None, :].expand_as(feats)
            feats = torch.cat([feats - centre_feats, centre_feats], dim=-1)
        hidden = _tokens(self.mlp, torch.cat([geometry, feats], dim=-1))
        if self.pool == "vector_attention":
            position = self.position(geometry)
            weight = self.attention(self.query(own)[:, None, :] - self.key(hidden) + position)
            weight = torch.softmax(weight.masked_fill(~mask[..., None], -1e4), dim=1)
            pooled = (weight * (self.value(hidden) + position)).sum(dim=1)
        else:
            pooled = masked_pool(hidden, mask, self.pool, self.score)
            if self.after_pool is not None:
                pooled = self.after_pool(pooled)
        if self.encoder == "invres":
            pooled = torch.relu(self.contract(self.expand(pooled)) + self.skip(own))
        return GroupFeatures(pooled, self.out_axes)

    def extra_repr(self) -> str:
        return (
            f"shape={self.shape!r}, radius={self.radius}, hmin={self.hmin}, hmax={self.hmax}, "
            f"nsample={self.nsample}, encoder={self.encoder!r}, pool={self.pool!r}"
        )


@GROUPERS.register()
class DepthLocalGroup(Grouper):
    """A crop of the scene cloud, pooled once per insertion depth.

    The members of a crop are gathered once, to the farthest reach ``hmax[-1]``, and
    encoded by a shared MLP over their positions in the crop frame. The feature of depth
    bin ``d`` is the pooling over the members with ``hmin < x < hmax[d]`` along the
    approach axis; a bin without members is zero. The scene is geometry only, as for
    :class:`~.depth_cylinder.DepthCylinderGroup`, which this grouper differs from in two
    ways: the shape and the pooling are options, and the ``nsample`` members are those
    of the deepest crop, sliced, instead of ``nsample`` per depth.

    Args:
        shape: ``ball``, ``knn``, ``box`` or ``cylinder``; see :func:`crop_members`.
        radius: Ball or cylinder radius, half the side of the box section, metres.
        hmin: Near end along the approach axis, metres.
        hmax: Far end of each depth bin, metres, increasing; its length is the size of
            the depth axis.
        nsample: Members per crop.
        padding: ``first`` or ``cyclic``.
        geometry: ``offset`` (metres, as the baseline) or ``offset+distance``.
        normalize_offsets: Divide the offsets by ``radius``.
        pool: ``max``, ``mean`` or ``attentive``.
        mlp_channels: Widths of the shared MLP; the last one is ``out_channels``.
        seed_residual: ``projected`` adds a linear map of the seed feature to every
            depth bin, whatever the width of the encoder; ``none`` uses geometry only.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "scene"
    out_axes = ("seed", "depth", "channel")
    status = "tested"
    source = (
        "The depth-sliced crop of GraspNet-baseline (Fang et al., CVPR 2020) with the "
        "shapes and poolings of LocalGroup, written here on grasppanda.ops"
    )

    def __init__(
        self,
        shape: Literal["ball", "knn", "box", "cylinder"] = "ball",
        radius: float = 0.05,
        hmin: float = -0.02,
        hmax: Sequence[float] = (0.01, 0.02, 0.03, 0.04),
        nsample: int = 64,
        padding: Literal["first", "cyclic"] = "first",
        geometry: Literal["offset", "offset+distance"] = "offset",
        normalize_offsets: bool = False,
        pool: Literal["max", "mean", "attentive"] = "max",
        mlp_channels: Sequence[int] = (64, 128, 256),
        seed_residual: Literal["none", "projected"] = "none",
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        _check_common(name, shape, padding, geometry)
        hmax = tuple(float(value) for value in hmax)
        if pool not in ("max", "mean", "attentive"):
            raise ValueError(
                f"{name}: unknown pool '{pool}'; a depth-sliced crop pools with max, mean or "
                "attentive"
            )
        if seed_residual not in ("none", "projected"):
            raise ValueError(f"{name}: unknown seed_residual '{seed_residual}'")
        if (
            radius <= 0
            or nsample < 1
            or not mlp_channels
            or not hmax
            or hmax[0] <= hmin
            or any(later <= earlier for earlier, later in pairwise(hmax))
        ):
            raise ValueError(
                f"{name} needs radius > 0, nsample >= 1, at least one MLP width and "
                f"hmin < hmax[0] < hmax[1] < ...; got radius={radius}, hmin={hmin}, "
                f"hmax={list(hmax)}, nsample={nsample}, mlp_channels={list(mlp_channels)}"
            )
        self.shape, self.radius, self.hmin, self.hmax = shape, radius, hmin, hmax
        self.nsample, self.padding, self.geometry = nsample, padding, geometry
        self.normalize_offsets, self.pool, self.seed_residual = (
            normalize_offsets,
            pool,
            seed_residual,
        )
        self.in_channels = in_channels
        self.out_channels = width = int(mlp_channels[-1])
        #: Whether ``forward`` takes the seed features as a fourth argument.
        self.uses_seed_features = seed_residual != "none"
        channels = 4 if geometry == "offset+distance" else 3
        self.mlp = _shared_mlp([channels, *mlp_channels])
        self.score = nn.Linear(width, width, bias=False) if pool == "attentive" else None
        self.project = nn.Linear(in_channels, width) if self.uses_seed_features else None

    @property
    def num_depth(self) -> int:
        """Size of the depth axis."""
        return len(self.hmax)

    def forward(
        self,
        support: PointFeatures,
        centres: Tensor,
        rotation: Tensor,
        seed_feats: Tensor | None = None,
    ) -> GroupFeatures:
        """Crop features ``(M, num_depth, out_channels)`` with axes ``(seed, depth, channel)``."""
        name = type(self).__name__
        coords = support.points.dense()
        batch, count = coords.shape[0], centres.shape[0]
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
        centre, frame = centres.reshape(batch, -1, 3), rotation.reshape(batch, -1, 3, 3)
        index, mask = crop_members(
            self.shape,
            centre,
            coords,
            frame,
            self.radius,
            self.hmin,
            self.hmax[-1],
            self.nsample,
            self.padding,
        )
        scale = self.radius if self.normalize_offsets else 1.0
        geometry, local = crop_geometry(coords, index, centre, frame, scale, self.geometry)
        hidden = _tokens(self.mlp, geometry.flatten(0, 1))
        axial, mask = local[..., 0].flatten(0, 1), mask.flatten(0, 1)
        inside = mask & (axial > self.hmin)
        feats = torch.stack(
            [
                masked_pool(hidden, inside & (axial < far), self.pool, self.score)
                for far in self.hmax
            ],
            dim=1,
        )
        if self.project is None:
            if seed_feats is not None:
                raise ContractError(
                    f"{name} was handed seed features, but it is configured with "
                    "seed_residual: none and would ignore them"
                )
        else:
            if seed_feats is None or tuple(seed_feats.shape) != (count, self.in_channels):
                got = None if seed_feats is None else tuple(seed_feats.shape)
                raise ContractError(
                    f"{name} with seed_residual: {self.seed_residual} needs the "
                    f"({count}, {self.in_channels}) features of the seeds, got {got}"
                )
            feats = feats + self.project(seed_feats)[:, None, :]
        return GroupFeatures(feats, self.out_axes)

    def extra_repr(self) -> str:
        return (
            f"shape={self.shape!r}, radius={self.radius}, hmin={self.hmin}, "
            f"hmax={list(self.hmax)}, nsample={self.nsample}, pool={self.pool!r}, "
            f"seed_residual={self.seed_residual!r}"
        )
