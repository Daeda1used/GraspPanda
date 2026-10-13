"""Samplers on PyTorch alone.

All are written here from the definitions in the papers named in ``source``; the lines
of released code that were read to pin a convention are cited in the docstrings. None is
a copy of a compiled kernel: the furthest-point variants run on
:func:`grasppanda.ops.furthest_point_sample` and
:func:`grasppanda.ops.generalized_furthest_point_sample`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from grasppanda.ops import furthest_point_sample, generalized_furthest_point_sample, voxelize
from grasppanda.registry import SAMPLERS

from .base import Sampler, check_inputs

__all__ = [
    "FPS",
    "FPSThenTopK",
    "FeatureFPS",
    "FusionFPS",
    "RandomSampler",
    "ScoreNMS",
    "ScoreWeightedFPS",
    "TopK",
    "VoxelGridSampler",
    "distinct",
]


def _generator(seed: int | None) -> torch.Generator | None:
    return None if seed is None else torch.Generator().manual_seed(seed)


def distinct(picks: Tensor) -> Tensor:
    """The picks without repeats, in the order they were made."""
    order = torch.argsort(picks, stable=True)
    ranked = picks[order]
    first = torch.ones_like(ranked, dtype=torch.bool)
    first[1:] = ranked[1:] != ranked[:-1]
    keep = torch.zeros_like(first)
    keep[order[first]] = True
    return picks[keep]


@SAMPLERS.register()
class FPS(Sampler):
    """Furthest point sampling on coordinates (D-FPS): seeds spread evenly over the candidates.

    Args:
        start: The first pick. ``first``: candidate 0, as the PointNet++ kernel and
            every ported method do. ``random``: a random candidate, as PyTorch3D's
            ``random_start_point`` and ``torch_cluster.fps`` do by default.
        skip_near_origin: Never pick points whose squared norm is ≤ 1e-3, a quirk of the
            PointNet++ CUDA kernel (parity with checkpoints trained against it).
        tie_break: ``lowest_index`` or ``pointnet2_cuda`` (the order of that kernel).
        seed: Seed of the generator behind ``start: random``; ``None`` uses the global one.
    """

    status = "tested"
    source = "PointNet++ (Qi et al., NeurIPS 2017); grasppanda.ops.furthest_point_sample"

    def __init__(
        self,
        start: Literal["first", "random"] = "first",
        skip_near_origin: bool = False,
        tie_break: Literal["lowest_index", "pointnet2_cuda"] = "lowest_index",
        seed: int | None = 0,
    ) -> None:
        if start not in ("first", "random"):
            raise ValueError(f"FPS: unknown start '{start}'; choose 'first' or 'random'")
        if tie_break not in ("lowest_index", "pointnet2_cuda"):
            raise ValueError(f"FPS: unknown tie_break '{tie_break}'")
        self.start = start
        self.skip_near_origin = skip_near_origin
        self.tie_break = tie_break
        self.seed = seed
        self._generator = _generator(seed)

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        count = len(coords)
        num_samples = min(num_samples, count)
        order = None
        if self.start == "random" and count > 1:
            first = int(torch.randint(count, (1,), generator=self._generator))
            order = torch.arange(count, device=coords.device)
            order[0], order[first] = first, 0
            coords = coords[order]
        picks = furthest_point_sample(
            coords, num_samples, skip_near_origin=self.skip_near_origin, tie_break=self.tie_break
        )
        picks = picks if order is None else order[picks]
        # Once every distinct position is taken the kernel repeats a point.
        return distinct(picks)


@SAMPLERS.register()
class FeatureFPS(Sampler):
    """Furthest point sampling in feature space (F-FPS): seeds that differ in what the encoder sees.

    3DSSD introduced it to keep points of rare foreground instances that coordinate-space
    sampling loses among background. Two distances go by this name; both are offered.

    Args:
        metric: ``concat``: squared L2 distance of ``[coord_weight · xyz, features]``,
            what the released 3DSSD code computes (``lib/utils/layers_util.py`` L101-L104
            with ``calc_square_dist(norm=False)``). ``sum``:
            ``coord_weight · |Δxyz| + |Δfeatures|``, Eq. 1 of the paper (its λ is
            ``coord_weight``); SASA's ``calc_dist_matrix_for_sampling`` is this form
            with the weight on the features.
        coord_weight: Weight of the coordinates (metres) against the features. With
            hundreds of feature channels and 1.0 the coordinates hardly count.
        normalize: L2-normalise every feature vector first, so that the distance does
            not follow the feature magnitude.
    """

    needs = ("feats",)
    status = "tested"
    source = (
        "3DSSD (Yang et al., CVPR 2020); written from the definition, conventions read in "
        "github.com/dvlab-research/3DSSD@8bc7605 (MIT)"
    )

    def __init__(
        self,
        metric: Literal["concat", "sum"] = "concat",
        coord_weight: float = 1.0,
        normalize: bool = False,
    ) -> None:
        if metric not in ("concat", "sum"):
            raise ValueError(f"FeatureFPS: unknown metric '{metric}'; choose 'concat' or 'sum'")
        if coord_weight < 0:
            raise ValueError("FeatureFPS: coord_weight must not be negative")
        self.metric = metric
        self.coord_weight = coord_weight
        self.normalize = normalize

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        check_inputs(self, scores, feats)
        if self.normalize:
            feats = torch.nn.functional.normalize(feats.float(), dim=1)
        return generalized_furthest_point_sample(
            coords,
            min(num_samples, len(coords)),
            feats=feats,
            metric=self.metric,
            coord_weight=self.coord_weight,
        )


@SAMPLERS.register()
class FusionFPS(Sampler):
    """Fusion sampling (FS): part of the seeds by F-FPS, the rest by D-FPS.

    3DSSD's fusion sampling returns ``2 M`` points, ``M`` of each kind
    (``layers_util.py`` L93-L98). A proposal has a fixed seed count, so here the two
    kinds share it: ``feature_fraction`` of the seeds by feature distance, then D-FPS
    over the points not yet taken.

    Args:
        feature_fraction: Share of the seeds picked by F-FPS, in ``[0, 1]``.
        metric: See ``FeatureFPS``.
        coord_weight: See ``FeatureFPS``.
        normalize: See ``FeatureFPS``.
    """

    needs = ("feats",)
    status = "tested"
    source = (
        "3DSSD (Yang et al., CVPR 2020); written from the definition, conventions read in "
        "github.com/dvlab-research/3DSSD@8bc7605 (MIT)"
    )

    def __init__(
        self,
        feature_fraction: float = 0.5,
        metric: Literal["concat", "sum"] = "concat",
        coord_weight: float = 1.0,
        normalize: bool = False,
    ) -> None:
        if not 0.0 <= feature_fraction <= 1.0:
            raise ValueError("FusionFPS: feature_fraction must lie in [0, 1]")
        self.feature_fraction = feature_fraction
        self.metric = metric
        self.coord_weight = coord_weight
        self.normalize = normalize
        self._by_feature = FeatureFPS(metric, coord_weight, normalize)

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        check_inputs(self, scores, feats)
        count = len(coords)
        num_samples = min(num_samples, count)
        by_feature = round(num_samples * self.feature_fraction)
        first = torch.zeros(0, dtype=torch.long, device=coords.device)
        if by_feature:
            first = self._by_feature.sample(coords, by_feature, feats=feats)
        if by_feature == num_samples:
            return first
        free = torch.ones(count, dtype=torch.bool, device=coords.device)
        free[first] = False
        rest = torch.nonzero(free).flatten()
        second = generalized_furthest_point_sample(coords[rest], num_samples - by_feature)
        return torch.cat([first, rest[second]])


@SAMPLERS.register()
class ScoreWeightedFPS(Sampler):
    """Semantics-guided furthest point sampling (S-FPS): spread seeds, prefer high scores.

    The first pick is the best-scored candidate; every later pick maximises
    ``score^gamma × distance to the picks so far``. ``gamma = 0`` is plain furthest point
    sampling started at the best point; a large ``gamma`` approaches top-k.

    Args:
        gamma: Exponent of the score. Reference value of SASA: 1.
        distance: ``squared``: the weight multiplies the squared distance, as the
            released kernel does (``sampling_gpu.cu`` L414-L463: ``d * max(w, 1e-12)``
            on squared ``d``). ``euclidean``: it multiplies the distance, as Eq. 3 of
            the paper is written. ``squared`` with ``gamma`` equals ``euclidean`` with
            ``gamma / 2``.
    """

    needs = ("scores",)
    status = "tested"
    source = (
        "SASA (Chen et al., AAAI 2022); written from the definition, conventions read in "
        "github.com/blakechen97/SASA@cd79f60 (Apache-2.0)"
    )

    def __init__(
        self, gamma: float = 1.0, distance: Literal["squared", "euclidean"] = "squared"
    ) -> None:
        if gamma < 0:
            raise ValueError("ScoreWeightedFPS: gamma must not be negative")
        if distance not in ("squared", "euclidean"):
            raise ValueError(f"ScoreWeightedFPS: unknown distance '{distance}'")
        self.gamma = gamma
        self.distance = distance

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        check_inputs(self, scores, feats)
        exponent = self.gamma * (2.0 if self.distance == "euclidean" else 1.0)
        weights = scores.to(coords.dtype).clamp(min=1e-12) ** exponent
        return generalized_furthest_point_sample(
            coords, min(num_samples, len(coords)), weights=weights, start=int(scores.argmax())
        )


@SAMPLERS.register()
class TopK(Sampler):
    """The best-scored candidates, wherever they are (top-k): no spatial spread at all.

    IA-SSD selects the points of its later layers this way (``torch.topk`` of the
    sigmoid of the class logit, ``pointnet2_modules.py`` L276-L280) and GraNet its 512
    grasp points. Equal scores are ordered by index.
    """

    needs = ("scores",)
    status = "tested"
    source = "IA-SSD (Zhang et al., CVPR 2022) and GraNet (Wang et al., IROS 2023); torch.sort"

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        check_inputs(self, scores, feats)
        order = torch.sort(scores, descending=True, stable=True).indices
        return order[: min(num_samples, len(coords))]


@SAMPLERS.register()
class RandomSampler(Sampler):
    """Uniformly random candidates without replacement: the baseline of every sampling ablation.

    Seed density follows the sensor: near and large surfaces get most seeds. GSNet's
    sampling ablation (random against furthest point sampling) and RandLA-Net use it.

    Args:
        seed: Seed of the sampler's own generator, so that two evaluations pick the same
            seeds; ``None`` uses PyTorch's global generator.
    """

    status = "tested"
    source = "GSNet (Wang et al., ICCV 2021), sampling ablation; RandLA-Net (Hu et al., CVPR 2020)"

    def __init__(self, seed: int | None = 0) -> None:
        self.seed = seed
        self._generator = _generator(seed)

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        count = len(coords)
        order = torch.randperm(count, generator=self._generator)
        return order[: min(num_samples, count)].to(coords.device)


@SAMPLERS.register()
class VoxelGridSampler(Sampler):
    """One candidate per occupied voxel: seeds on a regular grid, their number set by the scene.

    The candidates are quantised to cubes of ``voxel_size``; every occupied cube
    contributes its best-scored candidate (its first one without scores). More occupied
    cubes than seeds are reduced by furthest point sampling over the representatives;
    fewer are returned as they are and the proposal fills the frame up.

    Args:
        voxel_size: Edge of a cube in metres. Choose it so that the candidates occupy
            about ``num_seeds`` cubes; ``metric/real_seeds`` in the log shows the result.
    """

    status = "tested"
    source = "Grid subsampling as in KPConv (Thomas et al., ICCV 2019); grasppanda.ops.voxelize"

    def __init__(self, voxel_size: float = 0.01) -> None:
        if voxel_size <= 0:
            raise ValueError("VoxelGridSampler: voxel_size must be positive")
        self.voxel_size = voxel_size

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        count = len(coords)
        grid = voxelize(coords, self.voxel_size)
        picks = grid.first
        if scores is not None:
            values = scores.float()
            best = torch.full((len(grid.voxels),), -float("inf"), device=coords.device)
            best = best.scatter_reduce(0, grid.inverse, values, reduce="amax")
            position = torch.arange(count, device=coords.device)
            position = torch.where(values == best[grid.inverse], position, count)
            picks = torch.full_like(grid.first, count).scatter_reduce(
                0, grid.inverse, position, reduce="amin"
            )
        if len(picks) > num_samples:
            picks = picks[generalized_furthest_point_sample(coords[picks], num_samples)]
        return picks


@SAMPLERS.register()
class FPSThenTopK(Sampler):
    """Furthest point sampling for coverage, then the best-scored of the rest.

    The selection rule of Contact-GraspNet's inference (furthest points among the
    confident contacts up to a maximum, the remainder filled in order of confidence),
    written here from that description; the reference code is under the NVIDIA Source
    Code License and was not copied.

    Args:
        fps_fraction: Share of the seeds picked by furthest point sampling, in
            ``[0, 1]``; the others are the best-scored remaining candidates.
    """

    needs = ("scores",)
    status = "tested"
    source = "Contact-GraspNet (Sundermeyer et al., ICRA 2021), selection rule; written from it"

    def __init__(self, fps_fraction: float = 0.5) -> None:
        if not 0.0 <= fps_fraction <= 1.0:
            raise ValueError("FPSThenTopK: fps_fraction must lie in [0, 1]")
        self.fps_fraction = fps_fraction

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        check_inputs(self, scores, feats)
        num_samples = min(num_samples, len(coords))
        spread = round(num_samples * self.fps_fraction)
        first = torch.zeros(0, dtype=torch.long, device=coords.device)
        if spread:
            first = generalized_furthest_point_sample(coords, spread)
        if spread == num_samples:
            return first
        remaining = scores.clone().float()
        remaining[first] = -float("inf")
        order = torch.sort(remaining, descending=True, stable=True).indices
        return torch.cat([first, order[: num_samples - spread]])


@SAMPLERS.register()
class ScoreNMS(Sampler):
    """Greedy suppression on points: best score first, no two seeds closer than a radius.

    Candidates are visited in order of score; one is accepted when no accepted
    candidate lies within ``min_distance``. S4G selects its grasps this way (Algorithm 1
    of the paper, without its collision test). The number of seeds depends on the scene:
    fewer than ``num_samples`` are returned when the candidates run out.

    Args:
        min_distance: Minimum spacing of two seeds in metres.
    """

    needs = ("scores",)
    status = "tested"
    source = "S4G (Qin et al., CoRL 2019), Algorithm 1; written from the paper"

    def __init__(self, min_distance: float = 0.01) -> None:
        if min_distance < 0:
            raise ValueError("ScoreNMS: min_distance must not be negative")
        self.min_distance = min_distance

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        check_inputs(self, scores, feats)
        # A fixed number of steps without reading a value back from the device; a step
        # after the candidates ran out picks a suppressed point and is dropped at the end.
        remaining = scores.clone().float()
        limit = self.min_distance**2
        steps = min(num_samples, len(coords))
        picks = torch.empty(steps, dtype=torch.long, device=coords.device)
        real = torch.empty(steps, dtype=torch.bool, device=coords.device)
        for step in range(steps):
            best = remaining.argmax()
            picks[step] = best
            real[step] = remaining[best] > -float("inf")
            near = (coords - coords[best]).square().sum(dim=1) <= limit
            remaining[near] = -float("inf")
            remaining[best] = -float("inf")
        return picks[real]
