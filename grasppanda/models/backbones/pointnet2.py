"""PointNet++ with single-scale grouping.

Paper: Qi, Yi, Su, Guibas, "PointNet++: Deep Hierarchical Feature Learning on Point
Sets in a Metric Space", NeurIPS 2017.

Two registered forms share one engine:

* :class:`PointNet2` stops propagating early and returns features on a subsample of
  the scene (``output = "subsampled"``). Its defaults are the backbone of
  GraspNet-baseline (Fang et al., CVPR 2020), which is also the VoteNet backbone with
  metric radii for tabletop scenes: four set-abstraction levels, two feature-propagation
  levels, 1024 seeds with 256 channels.
* :class:`PointNet2Dense` propagates all the way back to the input points
  (``output = "dense"``). Its defaults are the semantic-segmentation network of the
  paper as published in the authors' repository.

Why the details are spelled out as constructor arguments: PointNet++ exists in two
lineages that differ in ways a checkpoint notices. The authors' TensorFlow code groups
raw offsets and interpolates with inverse *squared* distance (the paper's ``p = 2``).
The PyTorch lineage (Pointnet2_PyTorch, then VoteNet, then GraspNet-baseline) can divide
the grouped offsets by the ball radius, interpolates with inverse distance (``p = 1``),
and its sampling kernel never picks a point closer than about 3 cm to the origin. Each
difference is a named argument whose default is what the published configuration does.

Provenance. Written for this code base on the reference operators of
:mod:`grasppanda.ops`. The module tree and parameter names (``sa1.mlp_module.layer0.conv``,
``fp1.mlp.layer0.bn.bn`` ...) deliberately follow VoteNet's ``Pointnet2Backbone``
(Facebook Research, MIT licence), which builds on Erik Wijmans' Pointnet2_PyTorch (public
domain), so that state dicts of that lineage load without renaming. GraspNet-baseline
itself is under a non-commercial licence: only its published hyper-parameters are used
here, and its released weights are an evaluation aid, not part of this project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.ops import inverse_distance_weights, three_interpolate
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import (
    ball_query_packed,
    check_features,
    cloud_sizes,
    fps_packed,
    knn_packed,
    offsets_of,
)

__all__ = ["PointNet2", "PointNet2Dense"]


class _SharedMLP(nn.Sequential):
    """1×1 convolution, batch norm and ReLU per layer, shared by all points."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        for index, (inputs, outputs) in enumerate(pairwise(channels)):
            layer = nn.Sequential()
            layer.add_module("conv", nn.Conv2d(inputs, outputs, kernel_size=1, bias=False))
            norm = nn.Sequential()
            norm.add_module("bn", nn.BatchNorm2d(outputs))
            layer.add_module("bn", norm)
            layer.add_module("activation", nn.ReLU(inplace=True))
            nn.init.kaiming_normal_(layer.conv.weight)
            self.add_module(f"layer{index}", layer)

    def apply_rows(self, feats: Tensor) -> Tensor:
        """Run on ``(M, K, C)`` rows and return ``(M, K, C_out)``.

        All rows share one batch-norm population, exactly as ``(B, C, npoint, nsample)``
        does in the layout the parameter shapes come from.
        """
        return self(feats.permute(2, 0, 1)[None])[0].permute(1, 2, 0)


class _SetAbstraction(nn.Module):
    """Sample centres, group a ball around each, run a shared MLP and max-pool."""

    def __init__(
        self,
        in_channels: int,
        npoint: int,
        radius: float,
        nsample: int,
        mlp: Sequence[int],
        *,
        normalize_xyz: bool,
        skip_near_origin: bool,
        tie_break: str = "lowest_index",
    ) -> None:
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.normalize_xyz, self.skip_near_origin = normalize_xyz, skip_near_origin
        self.tie_break = tie_break
        self.mlp_module = _SharedMLP([in_channels + 3, *mlp])

    def forward(
        self, coords: Tensor, feats: Tensor | None, sizes: Sequence[int], presampled: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Return the sampled rows of ``coords`` and one feature vector per sample.

        ``presampled`` says the clouds already are furthest point samples in sampling
        order, as every level below the first is; see :func:`fps_packed`.
        """
        index = fps_packed(
            coords,
            sizes,
            self.npoint,
            skip_near_origin=self.skip_near_origin,
            presampled=presampled,
            tie_break=self.tie_break,
        )
        centres = coords[index]
        neighbours = ball_query_packed(
            centres, [self.npoint] * len(sizes), coords, sizes, self.radius, self.nsample
        )
        grouped = coords[neighbours] - centres[:, None, :]
        if self.normalize_xyz:
            grouped = grouped / self.radius
        if feats is not None:
            grouped = torch.cat([grouped, feats[neighbours]], dim=-1)
        return index, self.mlp_module.apply_rows(grouped).amax(dim=1)


class _Propagation(nn.Module):
    """Interpolate features onto a denser set, concatenate its own, run a shared MLP."""

    def __init__(
        self, in_channels: int, mlp: Sequence[int], *, power: float, neighbours: int = 3
    ) -> None:
        super().__init__()
        self.power, self.neighbours = power, neighbours
        self.mlp = _SharedMLP([in_channels, *mlp])

    def forward(
        self,
        coords: Tensor,
        sizes: Sequence[int],
        skip: Tensor | None,
        known_coords: Tensor,
        known_sizes: Sequence[int],
        known_feats: Tensor,
    ) -> Tensor:
        distance, index = knn_packed(coords, sizes, known_coords, known_sizes, self.neighbours)
        weight = inverse_distance_weights(distance, power=self.power)
        feats = three_interpolate(known_feats[None], index[None], weight[None])[0]
        if skip is not None:
            feats = torch.cat([feats, skip], dim=-1)
        return self.mlp.apply_rows(feats[:, None, :])[:, 0, :]


class _PointNet2(Backbone):
    """Set abstraction down, feature propagation part or all of the way back up."""

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int:
        """The last width of the last propagation level, or of the last abstraction."""
        last = options["fp_mlps"][-1] if options["fp_mlps"] else options["sa_mlps"][-1]
        return last[-1]

    def __init__(
        self,
        in_channels: int,
        npoints: Sequence[int],
        radii: Sequence[float],
        nsamples: Sequence[int],
        sa_mlps: Sequence[Sequence[int]],
        fp_mlps: Sequence[Sequence[int]],
        normalize_xyz: bool,
        interpolation_power: float,
        fps_skip_near_origin: bool,
        fps_tie_break: str = "lowest_index",
    ) -> None:
        super().__init__()
        name = type(self).__name__
        if fps_tie_break not in ("lowest_index", "pointnet2_cuda"):
            raise ValueError(f"{name}: unknown fps_tie_break '{fps_tie_break}'")
        levels = len(npoints)
        if not (levels == len(radii) == len(nsamples) == len(sa_mlps)) or levels == 0:
            raise ValueError(
                f"{name}: npoints, radii, nsamples and sa_mlps need one entry per "
                f"set-abstraction level, got {levels}, {len(radii)}, {len(nsamples)}, {len(sa_mlps)}"
            )
        if any(later > earlier for earlier, later in pairwise(npoints)):
            raise ValueError(f"{name}: npoints must not increase with depth, got {list(npoints)}")
        if npoints[-1] < 3 and fp_mlps:
            raise ValueError(f"{name}: feature propagation interpolates over 3 points")
        if len(fp_mlps) > levels:
            raise ValueError(
                f"{name}: {len(fp_mlps)} propagation levels for {levels} abstraction levels"
            )
        if any(len(mlp) == 0 for mlp in (*sa_mlps, *fp_mlps)):
            raise ValueError(f"{name}: every MLP needs at least one layer")
        self.in_channels = in_channels
        self.npoints = tuple(npoints)

        #: Channels of level 0 (the input) to level L (the coarsest set).
        widths = [in_channels]
        for index in range(levels):
            layer = _SetAbstraction(
                widths[-1],
                npoints[index],
                radii[index],
                nsamples[index],
                sa_mlps[index],
                normalize_xyz=normalize_xyz,
                skip_near_origin=fps_skip_near_origin,
                tie_break=fps_tie_break,
            )
            self.add_module(f"sa{index + 1}", layer)
            widths.append(sa_mlps[index][-1])
        carried = widths[-1]
        for index, mlp in enumerate(fp_mlps):
            target = levels - 1 - index
            layer = _Propagation(carried + widths[target], mlp, power=interpolation_power)
            self.add_module(f"fp{index + 1}", layer)
            carried = mlp[-1]
        self.num_sa, self.num_fp = levels, len(fp_mlps)
        self.out_channels = carried

    def _run(self, points: PointSet, feats: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Return the features of the output level and its rows of ``points`` (``None`` = all)."""
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
        batch = len(sizes)
        level_coords, level_feats, level_sizes = [points.coords], [feats], [sizes]
        #: Rows of the input cloud that make up each level; ``None`` means all of them.
        level_rows: list[Tensor | None] = [None]
        for index in range(self.num_sa):
            layer: _SetAbstraction = getattr(self, f"sa{index + 1}")
            chosen, new_feats = layer(
                level_coords[-1], level_feats[-1], level_sizes[-1], presampled=index > 0
            )
            rows = level_rows[-1]
            level_rows.append(chosen if rows is None else rows[chosen])
            level_coords.append(level_coords[-1][chosen])
            level_feats.append(new_feats)
            level_sizes.append([layer.npoint] * batch)
        carried = level_feats[-1]
        for index in range(self.num_fp):
            target = self.num_sa - 1 - index
            carried = getattr(self, f"fp{index + 1}")(
                level_coords[target],
                level_sizes[target],
                level_feats[target],
                level_coords[target + 1],
                level_sizes[target + 1],
                carried,
            )
        return carried, level_rows[self.num_sa - self.num_fp]


@BACKBONES.register()
class PointNet2(_PointNet2):
    """PointNet++ (SSG) returning features on a furthest-point subsample of the scene.

    The defaults are the GraspNet-baseline backbone: abstraction to 2048 / 1024 / 512 /
    256 points with radii 0.04 / 0.1 / 0.2 / 0.3 m and 64 / 32 / 16 / 16 neighbours,
    then two propagation levels back to the 1024-point set, 256 channels. Coordinates
    are the only input (``in_channels = 0``); extra per-point features are concatenated
    after the grouped offsets when ``in_channels > 0``.

    Every cloud must hold at least ``npoints[0]`` points; clouds may differ in size.
    The returned set is ``points.subset(index)`` with ``index`` in sampling order, the
    same for every cloud size, so a batch of ``B`` clouds yields ``B * npoints[k]`` rows.

    Args:
        in_channels: Feature channels per point besides the coordinates.
        npoints: Points kept by each set-abstraction level.
        radii: Ball radius of each level, in the unit of the coordinates.
        nsamples: Neighbours gathered per ball; a ball with fewer points repeats its
            first hit.
        sa_mlps: Layer widths of each level's shared MLP.
        fp_mlps: Layer widths of each propagation level, coarsest first. With ``k``
            entries the output lives on level ``len(npoints) - k``; it must stay above
            the input level (use :class:`PointNet2Dense` to reach the input).
        normalize_xyz: Divide grouped offsets by the ball radius.
        interpolation_power: Exponent of the inverse-distance weights of propagation.
        fps_skip_near_origin: Reproduce the sampling kernel of the PyTorch lineage,
            which never selects a point with squared norm ≤ 1e-3 after the first one.
            It only matters for clouds that pass within 3.2 cm of the frame origin.
        fps_tie_break: Which point is sampled when several are equally far from the
            points chosen so far: ``lowest_index`` takes the first, ``pointnet2_cuda``
            follows the order of that lineage's sampling kernel. Clouds from a depth
            camera lie on a lattice, so exact ties occur; a tie decides the order of the
            sampled points and can decide which points are sampled at all. Reproducing a
            checkpoint of the lineage point for point needs ``pointnet2_cuda``.
    """

    output = "subsampled"
    status = "weights-verified"
    source = "PointNet++ (Qi et al., NeurIPS 2017), GraspNet-baseline layout"

    @classmethod
    def output_points(cls, options: Mapping[str, Any]) -> int | None:
        """Points per cloud of the level the last propagation ends on."""
        npoints, levels = options["npoints"], len(options["fp_mlps"])
        return int(npoints[len(npoints) - levels - 1])

    def __init__(
        self,
        in_channels: int = 0,
        npoints: Sequence[int] = (2048, 1024, 512, 256),
        radii: Sequence[float] = (0.04, 0.1, 0.2, 0.3),
        nsamples: Sequence[int] = (64, 32, 16, 16),
        sa_mlps: Sequence[Sequence[int]] = (
            (64, 64, 128),
            (128, 128, 256),
            (128, 128, 256),
            (128, 128, 256),
        ),
        fp_mlps: Sequence[Sequence[int]] = ((256, 256), (256, 256)),
        normalize_xyz: bool = True,
        interpolation_power: float = 1.0,
        fps_skip_near_origin: bool = True,
        fps_tie_break: Literal["lowest_index", "pointnet2_cuda"] = "lowest_index",
    ) -> None:
        if len(fp_mlps) >= len(npoints):
            raise ValueError(
                f"PointNet2 returns subsampled features, so it needs fewer propagation levels "
                f"({len(fp_mlps)}) than abstraction levels ({len(npoints)}). Use PointNet2Dense "
                "to propagate back to the input points."
            )
        super().__init__(
            in_channels,
            npoints,
            radii,
            nsamples,
            sa_mlps,
            fp_mlps,
            normalize_xyz,
            interpolation_power,
            fps_skip_near_origin,
            fps_tie_break,
        )

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        out, rows = self._run(points, feats)
        count = self.npoints[self.num_sa - self.num_fp - 1]
        offsets = offsets_of([count] * points.batch_size, out.device)
        return PointFeatures(out, points.subset(rows, offsets))


@BACKBONES.register()
class PointNet2Dense(_PointNet2):
    """PointNet++ (SSG) propagated back to every input point.

    The defaults are the semantic-segmentation network of the paper
    (``pointnet2_sem_seg`` in the authors' repository): abstraction to 1024 / 256 / 64 /
    16 points with radii 0.1 / 0.2 / 0.4 / 0.8 m and 32 neighbours, four propagation
    levels, 128 output channels, raw grouped offsets and inverse squared distance
    interpolation. Those radii suit room-scale scans; for tabletop clouds in metres
    override ``npoints`` and ``radii`` (the :class:`PointNet2` values are a sensible
    start).

    Args:
        in_channels: Feature channels per point besides the coordinates. They also act
            as the skip connection of the last propagation level.
        npoints, radii, nsamples, sa_mlps, normalize_xyz, fps_skip_near_origin: As in
            :class:`PointNet2`.
        fp_mlps: Layer widths of each propagation level, coarsest first; exactly one
            per abstraction level.
        interpolation_power: Exponent of the inverse-distance weights (2 in the paper).
    """

    output = "dense"
    status = "tested"
    source = "PointNet++ (Qi et al., NeurIPS 2017), segmentation layout"

    def __init__(
        self,
        in_channels: int = 0,
        npoints: Sequence[int] = (1024, 256, 64, 16),
        radii: Sequence[float] = (0.1, 0.2, 0.4, 0.8),
        nsamples: Sequence[int] = (32, 32, 32, 32),
        sa_mlps: Sequence[Sequence[int]] = (
            (32, 32, 64),
            (64, 64, 128),
            (128, 128, 256),
            (256, 256, 512),
        ),
        fp_mlps: Sequence[Sequence[int]] = ((256, 256), (256, 256), (256, 128), (128, 128, 128)),
        normalize_xyz: bool = False,
        interpolation_power: float = 2.0,
        fps_skip_near_origin: bool = False,
    ) -> None:
        if len(fp_mlps) != len(npoints):
            raise ValueError(
                f"PointNet2Dense propagates back to the input, so it needs one propagation "
                f"level per abstraction level, got {len(fp_mlps)} for {len(npoints)}. Use "
                "PointNet2 for features on a subsample."
            )
        super().__init__(
            in_channels,
            npoints,
            radii,
            nsamples,
            sa_mlps,
            fp_mlps,
            normalize_xyz,
            interpolation_power,
            fps_skip_near_origin,
        )

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        out, _ = self._run(points, feats)
        return PointFeatures(out, points)
