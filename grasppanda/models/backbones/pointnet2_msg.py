# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for GraspPanda from https://github.com/facebookresearch/votenet @ 2f6d6d3 (MIT):
# the multi-scale set abstraction ``PointnetSAModuleMSG`` of ``pointnet2/pointnet2_modules.py``,
# itself modified from https://github.com/erikwijmans/Pointnet2_PyTorch @ b5ceb6d (Unlicense),
# whose ``pointnet2/models/pointnet2_msg_sem.py`` gives the segmentation layout.
"""PointNet++ with multi-scale grouping (MSG).

Paper: Qi, Yi, Su, Guibas, "PointNet++: Deep Hierarchical Feature Learning on Point
Sets in a Metric Space", NeurIPS 2017, section 3.3.

Every set-abstraction level samples its centres once and then looks at each centre
through several balls of different radius. Each ball has its own shared MLP and
max-pool; the results are concatenated. A depth camera gives dense points near the
camera and sparse points far away and at object borders, which is the case multi-scale
grouping was designed for. Contact-GraspNet (Sundermeyer et al., ICRA 2021) and GPR use
an encoder of this kind on scene clouds.

Two registered forms, as for single-scale grouping in :mod:`.pointnet2`:

* :class:`PointNet2MSG` stops propagating early (``output = "subsampled"``);
* :class:`PointNet2MSGDense` propagates back to every input point (``output = "dense"``).

The compiled sampling, grouping and interpolation of the upstream are replaced by the
reference operators of :mod:`grasppanda.ops`. The module tree (``SA_modules.0.mlps.1``,
``FP_modules.0.mlp``) follows the upstream segmentation model, finest propagation level
first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import ball_query_packed, check_features, cloud_sizes, fps_packed, offsets_of
from .pointnet2 import _Propagation, _SharedMLP

__all__ = ["PointNet2MSG", "PointNet2MSGDense"]


class _SetAbstractionMSG(nn.Module):
    """Sample centres once; one ball query, shared MLP and max-pool per radius."""

    def __init__(
        self,
        in_channels: int,
        npoint: int,
        radii: Sequence[float],
        nsamples: Sequence[int],
        mlps: Sequence[Sequence[int]],
        normalize_xyz: bool,
    ) -> None:
        super().__init__()
        self.npoint, self.radii, self.nsamples = npoint, tuple(radii), tuple(nsamples)
        self.normalize_xyz = normalize_xyz
        self.mlps = nn.ModuleList(_SharedMLP([in_channels + 3, *mlp]) for mlp in mlps)
        self.out_channels = sum(mlp[-1] for mlp in mlps)

    def forward(
        self, coords: Tensor, feats: Tensor | None, sizes: Sequence[int], presampled: bool
    ) -> tuple[Tensor, Tensor]:
        index = fps_packed(coords, sizes, self.npoint, presampled=presampled)
        centres = coords[index]
        pooled = []
        for radius, nsample, mlp in zip(self.radii, self.nsamples, self.mlps):
            neighbours = ball_query_packed(
                centres, [self.npoint] * len(sizes), coords, sizes, radius, nsample
            )
            grouped = coords[neighbours] - centres[:, None, :]
            if self.normalize_xyz:
                grouped = grouped / radius
            if feats is not None:
                grouped = torch.cat([grouped, feats[neighbours]], dim=-1)
            pooled.append(mlp.apply_rows(grouped).amax(dim=1))
        return index, torch.cat(pooled, dim=-1)


class _PointNet2MSG(Backbone):
    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int | None:
        """The last propagation width; without propagation, the scales of the last level joined."""
        if options["fp_mlps"]:
            return options["fp_mlps"][-1][-1]
        return sum(mlp[-1] for mlp in options["sa_mlps"][-1])

    def __init__(
        self,
        in_channels: int,
        npoints: Sequence[int],
        radii: Sequence[Sequence[float]],
        nsamples: Sequence[Sequence[int]],
        sa_mlps: Sequence[Sequence[Sequence[int]]],
        fp_mlps: Sequence[Sequence[int]],
        normalize_xyz: bool,
        interpolation_power: float,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        levels = len(npoints)
        if not (levels == len(radii) == len(nsamples) == len(sa_mlps)) or levels == 0:
            raise ValueError(
                f"{name}: npoints, radii, nsamples and sa_mlps need one entry per "
                f"set-abstraction level, got {levels}, {len(radii)}, {len(nsamples)}, {len(sa_mlps)}"
            )
        for level in range(levels):
            scales = (len(radii[level]), len(nsamples[level]), len(sa_mlps[level]))
            if len(set(scales)) != 1 or scales[0] == 0:
                raise ValueError(
                    f"{name}: level {level} needs one radius, one neighbour count and one MLP "
                    f"per scale, got {scales[0]} radii, {scales[1]} nsamples, {scales[2]} MLPs. "
                    "Multi-scale grouping takes a list per level, e.g. radii: [[0.02, 0.04], ...]"
                )
        if any(later > earlier for earlier, later in pairwise(npoints)):
            raise ValueError(f"{name}: npoints must not increase with depth, got {list(npoints)}")
        if npoints[-1] < 3 and fp_mlps:
            raise ValueError(f"{name}: feature propagation interpolates over 3 points")
        self.in_channels = in_channels
        self.npoints = tuple(npoints)

        widths = [in_channels]
        self.SA_modules = nn.ModuleList()
        for level in range(levels):
            layer = _SetAbstractionMSG(
                widths[-1],
                npoints[level],
                radii[level],
                nsamples[level],
                sa_mlps[level],
                normalize_xyz,
            )
            self.SA_modules.append(layer)
            widths.append(layer.out_channels)
        # Propagation level ``k`` writes onto level ``k``; ``fp_mlps`` lists the coarsest first.
        self.first_fp = levels - len(fp_mlps)
        layers: list[nn.Module] = []
        carried = widths[-1]
        for index, mlp in enumerate(fp_mlps):
            target = levels - 1 - index
            layers.insert(0, _Propagation(carried + widths[target], mlp, power=interpolation_power))
            carried = mlp[-1]
        self.FP_modules = nn.ModuleList(layers)
        self.out_channels = carried

    def _run(self, points: PointSet, feats: Tensor | None) -> tuple[Tensor, Tensor | None]:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        sizes = cloud_sizes(points)
        if min(sizes) < self.npoints[0]:
            raise ContractError(
                f"{name} samples {self.npoints[0]} points per cloud at its first level, but "
                f"cloud {sizes.index(min(sizes))} of the batch has only {min(sizes)}. Feed "
                "larger clouds or configure smaller npoints."
            )
        batch = len(sizes)
        level_coords, level_feats, level_sizes = [points.coords], [feats], [sizes]
        level_rows: list[Tensor | None] = [None]
        for index, layer in enumerate(self.SA_modules):
            chosen, new_feats = layer(
                level_coords[-1], level_feats[-1], level_sizes[-1], presampled=index > 0
            )
            rows = level_rows[-1]
            level_rows.append(chosen if rows is None else rows[chosen])
            level_coords.append(level_coords[-1][chosen])
            level_feats.append(new_feats)
            level_sizes.append([layer.npoint] * batch)
        carried = level_feats[-1]
        for offset in reversed(range(len(self.FP_modules))):
            target = self.first_fp + offset
            carried = self.FP_modules[offset](
                level_coords[target],
                level_sizes[target],
                level_feats[target],
                level_coords[target + 1],
                level_sizes[target + 1],
                carried,
            )
        return carried, level_rows[self.first_fp]


@BACKBONES.register()
class PointNet2MSG(_PointNet2MSG):
    """PointNet++ with multi-scale grouping, features on a furthest-point subsample.

    The defaults are a table-top layout: the level sizes of the GraspNet-baseline
    encoder (2048 / 1024 / 512 / 256 points) with two balls per level, from 2 cm to
    30 cm, and two propagation levels back to the 1024-point set, 256 channels. So it is
    a drop-in for :class:`~grasppanda.models.backbones.pointnet2.PointNet2`: the same
    seeds, the same width, twice the neighbourhoods.

    Args:
        in_channels: Feature channels per point besides the coordinates.
        npoints: Points kept by each set-abstraction level.
        radii: Ball radii of each level, one list per level, in the unit of the
            coordinates.
        nsamples: Neighbours gathered per ball, one list per level.
        sa_mlps: For each level, the layer widths of the shared MLP of each scale. A
            level's output width is the sum of the last widths of its scales.
        fp_mlps: Layer widths of each propagation level, coarsest first. With ``k``
            entries the output lives on level ``len(npoints) - k`` (the last level
            itself for an empty list).
        normalize_xyz: Divide grouped offsets by the ball radius.
        interpolation_power: Exponent of the inverse-distance weights of propagation.
    """

    output = "subsampled"
    status = "tested"
    source = (
        "PointNet++ MSG (Qi et al., NeurIPS 2017), adapted from "
        "facebookresearch/votenet@2f6d6d3, MIT"
    )

    @classmethod
    def output_points(cls, options: Mapping[str, Any]) -> int | None:
        """Points per cloud of the level the last propagation ends on."""
        npoints, levels = options["npoints"], len(options["fp_mlps"])
        return int(npoints[len(npoints) - levels - 1])

    def __init__(
        self,
        in_channels: int = 0,
        npoints: Sequence[int] = (2048, 1024, 512, 256),
        radii: Sequence[Sequence[float]] = ((0.02, 0.04), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)),
        nsamples: Sequence[Sequence[int]] = ((32, 64), (16, 32), (16, 16), (16, 16)),
        sa_mlps: Sequence[Sequence[Sequence[int]]] = (
            ((32, 32, 64), (64, 64, 128)),
            ((64, 64, 128), (128, 128, 256)),
            ((128, 128, 256), (128, 128, 256)),
            ((128, 128, 256), (128, 128, 256)),
        ),
        fp_mlps: Sequence[Sequence[int]] = ((256, 256), (256, 256)),
        normalize_xyz: bool = True,
        interpolation_power: float = 1.0,
    ) -> None:
        if len(fp_mlps) >= len(npoints):
            raise ValueError(
                f"PointNet2MSG returns subsampled features, so it needs fewer propagation "
                f"levels ({len(fp_mlps)}) than abstraction levels ({len(npoints)}). Use "
                "PointNet2MSGDense to propagate back to the input points."
            )
        super().__init__(
            in_channels, npoints, radii, nsamples, sa_mlps, fp_mlps, normalize_xyz,
            interpolation_power,
        )  # fmt: skip

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        out, rows = self._run(points, feats)
        count = self.npoints[self.first_fp - 1]
        offsets = offsets_of([count] * points.batch_size, out.device)
        return PointFeatures(out, points.subset(rows, offsets))


@BACKBONES.register()
class PointNet2MSGDense(_PointNet2MSG):
    """PointNet++ with multi-scale grouping, propagated back to every input point.

    The defaults are the multi-scale segmentation model of Pointnet2_PyTorch
    (``PointNet2SemSegMSG``): 1024 / 256 / 64 / 16 points, balls of (0.05, 0.1) to
    (0.4, 0.8) m with 16 and 32 neighbours, four propagation levels, 128 output
    channels. Those radii are for rooms; the ``pointnet2-msg-dense`` preset has
    table-top values.

    Args:
        in_channels: Feature channels per point besides the coordinates. They are also
            the skip connection of the last propagation level.
        npoints, radii, nsamples, sa_mlps, normalize_xyz, interpolation_power: As in
            :class:`PointNet2MSG`.
        fp_mlps: Layer widths of each propagation level, coarsest first; exactly one
            per abstraction level.
    """

    output = "dense"
    status = "tested"
    source = (
        "PointNet++ MSG (Qi et al., NeurIPS 2017), segmentation layout, adapted from "
        "facebookresearch/votenet@2f6d6d3, MIT"
    )

    def __init__(
        self,
        in_channels: int = 0,
        npoints: Sequence[int] = (1024, 256, 64, 16),
        radii: Sequence[Sequence[float]] = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
        nsamples: Sequence[Sequence[int]] = ((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_mlps: Sequence[Sequence[Sequence[int]]] = (
            ((16, 16, 32), (32, 32, 64)),
            ((64, 64, 128), (64, 96, 128)),
            ((128, 196, 256), (128, 196, 256)),
            ((256, 256, 512), (256, 384, 512)),
        ),
        fp_mlps: Sequence[Sequence[int]] = ((512, 512), (512, 512), (256, 256), (128, 128)),
        normalize_xyz: bool = False,
        interpolation_power: float = 1.0,
    ) -> None:
        if len(fp_mlps) != len(npoints):
            raise ValueError(
                f"PointNet2MSGDense propagates back to the input, so it needs one propagation "
                f"level per abstraction level, got {len(fp_mlps)} for {len(npoints)}. Use "
                "PointNet2MSG for features on a subsample."
            )
        super().__init__(
            in_channels, npoints, radii, nsamples, sa_mlps, fp_mlps, normalize_xyz,
            interpolation_power,
        )  # fmt: skip

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        out, _ = self._run(points, feats)
        return PointFeatures(out, points)
