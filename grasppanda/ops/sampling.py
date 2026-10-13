"""Point sampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from . import backends

if TYPE_CHECKING:
    from grasppanda._ops.library import Kernels

__all__ = ["furthest_point_sample", "generalized_furthest_point_sample"]

#: Squared-norm threshold below which the PointNet++ CUDA kernel ignores a point.
_NEAR_ORIGIN = 1e-3
#: Threads of the PointNet++ CUDA kernel; thread ``t`` scans the indices ``k ≡ t (mod 512)``.
_KERNEL_THREADS = 512


def _squared_norm(vectors: Tensor) -> Tensor:
    """``x² + y² + z²``, added left to right.

    Not ``pow(2).sum(-1)``: CUDA reduces the three terms in another order than the CPU,
    which changes the last bit of the result and with it the winner of a near-tie.
    """
    squares = vectors * vectors
    return squares[..., 0] + squares[..., 1] + squares[..., 2]


def _kernel_threads(count: int) -> int:
    """Threads the PointNet++ CUDA kernel runs on ``count`` points."""
    return min(1 << (count.bit_length() - 1), _KERNEL_THREADS)


def _kernel_priority(count: int, device: torch.device) -> Tensor:
    """Rank of every index in the tie order of the PointNet++ CUDA kernel; lower wins.

    Each thread keeps the first of its own maxima (``k`` ascending) and the threads are
    then reduced in a binary tree that keeps the lower slot on a tie. Two threads meet
    at the level of their lowest differing bit, where the one with a zero there sits
    in the lower slot: thread ids are effectively compared after reversing their bits.
    """
    threads = _kernel_threads(count)
    bits = threads.bit_length() - 1
    index = torch.arange(count, device=device)
    thread = index % threads
    reversed_bits = torch.zeros_like(thread)
    for bit in range(bits):
        reversed_bits |= ((thread >> bit) & 1) << (bits - 1 - bit)
    return reversed_bits * count + index


def furthest_point_sample(
    points: Tensor,
    num_samples: int,
    *,
    skip_near_origin: bool = False,
    tie_break: Literal["lowest_index", "pointnet2_cuda"] = "lowest_index",
) -> Tensor:
    """Iterative furthest point sampling.

    The first sample is always index 0 and ties are resolved by a fixed rule, so the
    result is deterministic and the same on every device. Which points become seeds
    decides whether a ported checkpoint reproduces, so faster backends are held to
    exactly this behaviour (:mod:`grasppanda.ops.backends`).

    Args:
        points: ``(B, N, 3)`` or ``(N, 3)``.
        num_samples: Number of points to pick, at most ``N``.
        skip_near_origin: Never pick points whose squared norm is ≤ 1e-3 (apart from
            index 0), as the widely used PointNet++ CUDA kernel does. Needed only for
            parity with checkpoints trained against that kernel.
        tie_break: Which point wins when several are equally far from the selected
            set. ``lowest_index`` takes the first. ``pointnet2_cuda`` reproduces the
            order of the PointNet++ CUDA kernel (bit-reversed thread id, then index;
            the kernel runs ``min(512, 2^floor(log2 N))`` threads). Exact ties are not
            exotic: clouds from a depth camera lie on a lattice, and 3 of 64 sampled
            GraspNet-1Billion frames had one among their first 1024 samples.

    Returns:
        ``(B, num_samples)`` or ``(num_samples,)`` indices into ``points``.
    """
    single = points.ndim == 2
    if single:
        points = points[None]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"expected (B, N, 3) or (N, 3), got {tuple(points.shape)}")
    if tie_break not in ("lowest_index", "pointnet2_cuda"):
        raise ValueError(f"unknown tie_break '{tie_break}'")
    count = points.shape[1]
    if not 0 < num_samples <= count:
        raise ValueError(
            f"cannot sample {num_samples} points from {count}; pad or mask explicitly in the caller"
        )
    kernels = backends.kernels_for("fps", points)
    if kernels is None:
        index = _sample(points, num_samples, skip_near_origin, tie_break)
    else:
        index = _sample_compiled(kernels, points, num_samples, skip_near_origin, tie_break)
    return index[0] if single else index


@torch.no_grad()
def _sample(points: Tensor, num_samples: int, skip_near_origin: bool, tie_break: str) -> Tensor:
    """The reference: ``(B, num_samples)`` picks from ``(B, N, 3)``.

    Each step measures every point against the last pick, keeps the smaller of that
    and its distance so far, and picks the point whose distance is largest. Almost all
    the time goes into per-step overhead (a handful of tiny GPU kernels, or passes over
    ``N`` numbers on a CPU), so every buffer and every view is created once and each
    operation writes in place: a pick goes straight into its column of the result.
    """
    batch, count, _ = points.shape
    device = points.device
    index = torch.zeros(batch, num_samples, dtype=torch.long, device=device)  # sample 0 is point 0
    picks = index.unbind(1)
    #: Squared distance of every point to the picks so far.
    distance = points.new_full((batch, count), float("inf"))
    ignore = _squared_norm(points) <= _NEAR_ORIGIN if skip_near_origin else None
    measure = (_measure_interleaved if points.is_cuda else _measure_planar)(points, index)
    if tie_break == "pointnet2_cuda":
        priority = _kernel_priority(count, device).expand(batch, count)
        last = torch.full_like(priority, torch.iinfo(torch.long).max)
        rank = torch.empty_like(last)
        tied = torch.empty(batch, count, dtype=torch.bool, device=device)
        furthest = points.new_empty(batch, 1)
    for step in range(1, num_samples):
        torch.minimum(distance, measure(step - 1), out=distance)
        if ignore is not None:
            # Overwriting the running distance is the same as masking a copy of it: a
            # masked point is masked again after every update.
            distance.masked_fill_(ignore, -1.0)
        if tie_break == "lowest_index":
            torch.argmax(distance, dim=1, out=picks[step])
        else:
            torch.amax(distance, dim=1, keepdim=True, out=furthest)
            torch.eq(distance, furthest, out=tied)
            torch.where(tied, priority, last, out=rank)
            torch.argmin(rank, dim=1, out=picks[step])
    return index


def _measure_interleaved(points: Tensor, index: Tensor) -> Callable[[int], Tensor]:
    """``measure(step)``: squared distance of every point to pick ``index[:, step]``, per cloud.

    Works on the ``(B, N, 3)`` layout as it is, which takes the fewest operations:
    what a GPU step costs. The returned tensor is reused by the next call.
    """
    batch, count, _ = points.shape
    picks = index[:, :, None, None].expand(batch, index.shape[1], 1, 3).unbind(1)
    centre = points.new_empty(batch, 1, 3)
    offset = torch.empty_like(points)
    x, y, z = offset.unbind(-1)
    out = points.new_empty(batch, count)

    def measure(step: int) -> Tensor:
        torch.gather(points, 1, picks[step], out=centre)
        torch.sub(points, centre, out=offset)
        offset.mul_(offset)
        torch.add(x, y, out=out)  # x² + y² + z², left to right as in _squared_norm
        return out.add_(z)

    return measure


def _measure_planar(points: Tensor, index: Tensor) -> Callable[[int], Tensor]:
    """As :func:`_measure_interleaved`, on one contiguous array per axis.

    More operations per step, but each is a vectorised pass over contiguous memory,
    which is what a CPU step costs. The arithmetic per point is the same.
    """
    batch, count, _ = points.shape
    picks = index[:, :, None].unbind(1)
    axes = [points[..., axis].contiguous() for axis in range(3)]
    centre = points.new_empty(batch, 1)
    square = points.new_empty(batch, count)
    out = points.new_empty(batch, count)

    def measure(step: int) -> Tensor:
        for axis, values in enumerate(axes):
            target = square if axis else out
            torch.gather(values, 1, picks[step], out=centre)
            torch.sub(values, centre, out=target)
            target.mul_(target)
            if axis:
                out.add_(square)  # x² + y² + z², left to right as in _squared_norm
        return out

    return measure


def _sample_compiled(
    kernels: Kernels, points: Tensor, num_samples: int, skip_near_origin: bool, tie_break: str
) -> Tensor:
    """:func:`_sample` on the compiled kernels."""
    return kernels.fps(
        points,
        num_samples,
        skip_near_origin=skip_near_origin,
        near_origin=_NEAR_ORIGIN,
        tie_break=tie_break,
        priority_threads=_kernel_threads(points.shape[1]),
    )


@torch.no_grad()
def generalized_furthest_point_sample(
    coords: Tensor,
    num_samples: int,
    *,
    feats: Tensor | None = None,
    metric: Literal["concat", "sum"] = "concat",
    coord_weight: float = 1.0,
    weights: Tensor | None = None,
    start: int | None = None,
) -> Tensor:
    """Furthest point sampling of one cloud under a feature distance and with point weights.

    The generalisation the detection literature uses to make the sampler *choose* instead
    of only *spread*: a distance that includes features (F-FPS of 3DSSD) and a weight on
    the running distance of every point (S-FPS of SASA). Without features and weights it
    is :func:`furthest_point_sample` with ``tie_break="lowest_index"``. A picked point is
    never picked again, so the indices are distinct even on duplicated coordinates.

    Args:
        coords: ``(N, 3)`` coordinates.
        num_samples: Points to pick, at most ``N``.
        feats: ``(N, C)`` features that enter the distance; ``None`` for coordinates only.
        metric: With features. ``concat``: squared L2 distance of the concatenation
            ``[coord_weight * xyz, feats]`` (what the released 3DSSD code computes).
            ``sum``: ``coord_weight * |Δxyz| + |Δfeats|`` with L2 norms (Eq. 1 of the
            3DSSD paper, its λ being ``coord_weight``).
        coord_weight: Weight of the coordinates against the features.
        weights: ``(N,)`` non-negative weights; the next pick maximises
            ``weight × running distance``. The running distance is the *squared* one
            unless ``metric="sum"`` (the convention of the released SASA kernel).
        start: Index of the first pick. ``None``: the heaviest point when weights are
            given (SASA), otherwise index 0.

    Returns:
        ``(num_samples,)`` distinct indices into ``coords``, in the order they were picked.
    """
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) coordinates, got {tuple(coords.shape)}")
    count = coords.shape[0]
    if not 0 < num_samples <= count:
        raise ValueError(
            f"cannot sample {num_samples} points from {count}; pad or mask explicitly in the caller"
        )
    if metric not in ("concat", "sum"):
        raise ValueError(f"unknown metric '{metric}'")
    if feats is not None and (feats.ndim != 2 or feats.shape[0] != count):
        raise ValueError(f"expected ({count}, C) features, got {tuple(feats.shape)}")
    if weights is not None and tuple(weights.shape) != (count,):
        raise ValueError(f"expected ({count},) weights, got {tuple(weights.shape)}")
    spaces = [coords * coord_weight]
    if feats is not None:
        if metric == "concat":
            spaces = [torch.cat([coords * coord_weight, feats.to(coords.dtype)], dim=1)]
        else:
            spaces.append(feats.to(coords.dtype))
    squared = feats is None or metric == "concat"
    if start is None:
        start = int(weights.argmax()) if weights is not None else 0
    if not 0 <= start < count:
        raise ValueError(f"start index {start} is outside the cloud of {count} points")
    # No value leaves the device inside the loop: a pick stays a tensor. Reading it as a
    # Python number would wait for the GPU once per pick, which costs more than the
    # arithmetic of the step (and far more on a GPU that is shared).
    index = torch.empty(num_samples, dtype=torch.long, device=coords.device)
    distance = coords.new_full((count,), float("inf"))
    pick = torch.tensor(start, dtype=torch.long, device=coords.device)
    norms = [space.square().sum(dim=1) if space.shape[1] > 3 else None for space in spaces]
    for step in range(num_samples):
        index[step] = pick
        if step + 1 == num_samples:
            break
        current = None
        for space, norm in zip(spaces, norms):
            if norm is None:  # three columns: subtract and square, exactly
                part = (space - space[pick]).square_().sum(dim=1)
            else:  # many columns: |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, one product per pick
                part = torch.mv(space, space[pick]).mul_(-2.0).add_(norm).add_(norm[pick])
                part.clamp_(min=0.0)
            if not squared:
                part = part.sqrt_()
            current = part if current is None else current.add_(part)
        distance = torch.minimum(distance, current)
        distance[pick] = -1.0  # taken: below every remaining point, whatever its weight
        gain = distance
        if weights is not None:
            gain = torch.where(distance < 0, distance, distance * weights)
        pick = gain.argmax()
    return index
