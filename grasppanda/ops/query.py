"""Neighbourhood queries and gathering.

Conventions shared by ``ball_query`` and ``cylinder_query`` (the PointNet++ ones):
the first ``nsample`` hits in index order are returned; missing slots repeat the first
hit; a query without any hit returns index 0 in every slot. Unlike the classic
kernels, the number of real hits is returned as well, so an empty neighbourhood can
be told apart from a neighbourhood made of point 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import backends

if TYPE_CHECKING:
    from grasppanda._ops.library import Kernels

__all__ = ["ball_query", "box_query", "cylinder_query", "gather_points", "group_points", "knn"]

#: Largest ``k`` the compiled kNN kernel is used for. It keeps the neighbours of a query
#: in a sorted list (linear insertion), which beats sorting the whole row only for few.
_COMPILED_KNN_MAX_K = 64
#: Queries the reference re-runs per call for the compiled cylinder query.
_RERUN_CHUNK = 256


def _check(query: Tensor, support: Tensor) -> None:
    if query.ndim != 3 or support.ndim != 3 or query.shape[-1] != 3 or support.shape[-1] != 3:
        raise ValueError(
            f"expected query (B, M, 3) and support (B, N, 3), got {tuple(query.shape)} and {tuple(support.shape)}"
        )
    if query.shape[0] != support.shape[0]:
        raise ValueError("query and support must have the same batch size")


def _first_hits(mask: Tensor, nsample: int) -> tuple[Tensor, Tensor]:
    """Indices of the first ``nsample`` true entries along the last axis, padded."""
    count = mask.shape[-1]
    # A hit at index i scores count - i, a miss scores 0: top-k returns the earliest
    # hits, already in ascending index order.
    rank = torch.arange(count, 0, -1, device=mask.device, dtype=torch.int32)
    score, index = (mask * rank).topk(min(nsample, count), dim=-1)
    hit = score > 0
    hits = hit.sum(-1)
    index = torch.where(hit, index, index[..., :1])
    index = torch.where(hits[..., None] > 0, index, torch.zeros_like(index))
    if nsample > count:
        pad = index[..., :1].expand(*index.shape[:-1], nsample - count)
        index = torch.cat([index, pad], dim=-1)
    return index, hits


def knn(query: Tensor, support: Tensor, k: int, *, chunk: int = 1024) -> tuple[Tensor, Tensor]:
    """k nearest support points of every query point.

    Indices are 0-based and equidistant neighbours are ordered by index.

    Returns:
        ``(distance, index)``, both ``(B, M, k)``; distances are Euclidean.
    """
    _check(query, support)
    if not 0 < k <= support.shape[1]:
        raise ValueError(f"k must be in 1..{support.shape[1]}, got {k}")
    # The reference returns distances that carry gradients; the kernel does not.
    differentiable = torch.is_grad_enabled() and (query.requires_grad or support.requires_grad)
    if k <= _COMPILED_KNN_MAX_K and not differentiable:
        kernels = backends.kernels_for("knn", query, support)
        if kernels is not None:
            return _knn_compiled(kernels, query, support, k)
    return _knn(query, support, k, chunk)


def _knn(query: Tensor, support: Tensor, k: int, chunk: int) -> tuple[Tensor, Tensor]:
    """The reference: all pairwise distances of a block, sorted stably."""
    distances, indices = [], []
    for start in range(0, query.shape[1], chunk):
        block = query[:, start : start + chunk]
        squared = (block[:, :, None, :] - support[:, None, :, :]).pow(2).sum(-1)
        value, index = torch.sort(squared, dim=-1, stable=True)
        distances.append(value[..., :k].sqrt())
        indices.append(index[..., :k])
    return torch.cat(distances, dim=1), torch.cat(indices, dim=1)


def _knn_compiled(
    kernels: Kernels, query: Tensor, support: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """:func:`_knn` on the compiled kernels.

    The kernel returns squared distances and the root is taken here, by PyTorch, as in
    the reference: PyTorch's float32 square root on the CPU is within one unit in the
    last place but not correctly rounded, so a kernel's own root would differ from it.
    """
    squared, index = kernels.knn(query, support, k)
    return squared.sqrt(), index


def ball_query(
    query: Tensor, support: Tensor, radius: float, nsample: int, *, chunk: int = 256
) -> tuple[Tensor, Tensor]:
    """Support points strictly within ``radius`` of every query point.

    Returns:
        ``(index, hits)``: ``(B, M, nsample)`` indices and ``(B, M)`` hit counts
        (capped at ``nsample``).
    """
    _check(query, support)
    kernels = backends.kernels_for("ball_query", query, support)
    if kernels is not None:
        return _ball_query_compiled(kernels, query, support, radius, nsample)
    return _ball_query(query, support, radius, nsample, chunk)


def _ball_query(
    query: Tensor, support: Tensor, radius: float, nsample: int, chunk: int
) -> tuple[Tensor, Tensor]:
    """The reference: a pairwise test per block of queries."""
    indices, counts = [], []
    for start in range(0, query.shape[1], chunk):
        block = query[:, start : start + chunk]
        squared = (support[:, None, :, :] - block[:, :, None, :]).pow(2).sum(-1)
        index, hits = _first_hits(squared < radius * radius, nsample)
        indices.append(index)
        counts.append(hits)
    return torch.cat(indices, dim=1), torch.cat(counts, dim=1)


def _ball_query_compiled(
    kernels: Kernels, query: Tensor, support: Tensor, radius: float, nsample: int
) -> tuple[Tensor, Tensor]:
    """:func:`_ball_query` on the compiled kernels.

    The reference compares float32 distances with ``radius * radius`` rounded to
    float32, which is what the kernel does with the double it is handed.
    """
    return kernels.ball_query(query, support, radius * radius, nsample)


def cylinder_query(
    query: Tensor,
    support: Tensor,
    rotation: Tensor,
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    *,
    chunk: int = 256,
) -> tuple[Tensor, Tensor]:
    """Support points inside an oriented cylinder at every query point.

    The cylinder axis is column 0 of ``rotation`` (the approach axis). A support point
    ``p`` is inside when, with ``d = Rᵀ (p - q)``, ``hmin < d.x < hmax`` and
    ``d.y² + d.z² < radius²`` (all strict).

    Args:
        query: ``(B, M, 3)`` cylinder origins.
        support: ``(B, N, 3)`` points to search.
        rotation: ``(B, M, 3, 3)`` frame per query.

    Returns:
        ``(index, hits)``: ``(B, M, nsample)`` indices and ``(B, M)`` hit counts
        (capped at ``nsample``).
    """
    _check(query, support)
    if tuple(rotation.shape) != (*query.shape[:2], 3, 3):
        raise ValueError(
            f"rotation must be {(*query.shape[:2], 3, 3)}, got {tuple(rotation.shape)}"
        )
    if _rotation_is_float32(query):
        kernels = backends.kernels_for("cylinder_query", query, support, rotation)
        if kernels is not None:
            return _cylinder_query_compiled(
                kernels, query, support, rotation, radius, hmin, hmax, nsample, chunk
            )
    return _cylinder_query(query, support, rotation, radius, hmin, hmax, nsample, chunk)


def _cylinder_query(
    query: Tensor,
    support: Tensor,
    rotation: Tensor,
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    chunk: int,
) -> tuple[Tensor, Tensor]:
    """The reference: offsets rotated by a batched matrix product, then a pairwise test."""
    indices, counts = [], []
    for start in range(0, query.shape[1], chunk):
        block = query[:, start : start + chunk]
        offset = support[:, None, :, :] - block[:, :, None, :]
        local = torch.einsum("bmnj,bmjk->bmnk", offset, rotation[:, start : start + chunk])
        axial = local[..., 0]
        radial = local[..., 1].pow(2) + local[..., 2].pow(2)
        inside = (radial < radius * radius) & (axial > hmin) & (axial < hmax)
        index, hits = _first_hits(inside, nsample)
        indices.append(index)
        counts.append(hits)
    return torch.cat(indices, dim=1), torch.cat(counts, dim=1)


def box_query(
    query: Tensor,
    support: Tensor,
    rotation: Tensor,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    nsample: int,
    *,
    chunk: int = 256,
) -> tuple[Tensor, Tensor]:
    """Support points inside an oriented box at every query point.

    The box is axis-aligned in the frame of its query: a support point ``p`` is inside
    when ``d = Rᵀ (p - q)`` satisfies ``lower < d < upper`` on all three axes (strict).
    Column 0 of ``rotation`` is the approach axis, as for :func:`cylinder_query`, and
    the conventions for the returned members are the same (first hits in index order,
    missing slots repeat the first, no hit returns index 0).

    Args:
        query: ``(B, M, 3)`` box origins.
        support: ``(B, N, 3)`` points to search.
        rotation: ``(B, M, 3, 3)`` frame per query.
        lower: Lower corner of the box in its frame, metres.
        upper: Upper corner of the box in its frame, metres.

    Returns:
        ``(index, hits)``: ``(B, M, nsample)`` indices and ``(B, M)`` hit counts
        (capped at ``nsample``).
    """
    _check(query, support)
    if tuple(rotation.shape) != (*query.shape[:2], 3, 3):
        raise ValueError(
            f"rotation must be {(*query.shape[:2], 3, 3)}, got {tuple(rotation.shape)}"
        )
    low = support.new_tensor(lower)
    high = support.new_tensor(upper)
    indices, counts = [], []
    for start in range(0, query.shape[1], chunk):
        block = query[:, start : start + chunk]
        offset = support[:, None, :, :] - block[:, :, None, :]
        local = torch.einsum("bmnj,bmjk->bmnk", offset, rotation[:, start : start + chunk])
        inside = ((local > low) & (local < high)).all(-1)
        index, hits = _first_hits(inside, nsample)
        indices.append(index)
        counts.append(hits)
    return torch.cat(indices, dim=1), torch.cat(counts, dim=1)


def _rotation_is_float32(like: Tensor) -> bool:
    """Whether the matrix product of :func:`_cylinder_query` runs in plain float32.

    Under autocast it runs in half precision, and with TF32 enabled a CUDA product keeps
    10 bits of mantissa. The membership test of the reference then has that precision
    and a kernel computing in float32 would disagree with it, so it is not used.
    """
    kind = like.device.type
    try:
        autocast = torch.is_autocast_enabled(kind)
    except TypeError:  # torch < 2.4 has one query per device type
        autocast = torch.is_autocast_enabled() if like.is_cuda else torch.is_autocast_cpu_enabled()
    return not (autocast or (like.is_cuda and torch.backends.cuda.matmul.allow_tf32))


def _cylinder_query_compiled(
    kernels: Kernels,
    query: Tensor,
    support: Tensor,
    rotation: Tensor,
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    chunk: int,
) -> tuple[Tensor, Tensor]:
    """:func:`_cylinder_query` on the compiled kernels.

    The reference rotates through a BLAS whose rounding no kernel can reproduce, so the
    kernel only answers the queries it can decide whatever that rounding is, and marks
    the others (``hits = -1``): those with a support point within round-off of the
    cylinder surface (``gp_cylinder_test`` in ``grasppanda/_ops/src/common.h`` has the
    bound). The marked queries, a few in ten thousand on real clouds, are answered here
    by the reference itself.
    """
    index, hits = kernels.cylinder_query(
        query, support, rotation, radius * radius, hmin, hmax, nsample
    )
    marked = torch.nonzero(hits < 0)
    for start in range(0, len(marked), _RERUN_CHUNK):
        frame, row = marked[start : start + _RERUN_CHUNK].unbind(1)
        # One query per "batch" entry, each with the support cloud of its own frame.
        again_index, again_hits = _cylinder_query(
            query[frame, row][:, None],
            support[frame],
            rotation[frame, row][:, None],
            radius,
            hmin,
            hmax,
            nsample,
            chunk,
        )
        index[frame, row] = again_index[:, 0]
        hits[frame, row] = again_hits[:, 0]
    return index, hits


def gather_points(values: Tensor, index: Tensor) -> Tensor:
    """Select ``(B, M, C)`` rows of ``values (B, N, C)`` with ``index (B, M)``."""
    return torch.gather(values, 1, index[..., None].expand(-1, -1, values.shape[-1]))


def group_points(values: Tensor, index: Tensor) -> Tensor:
    """Select ``(B, M, K, C)`` neighbourhoods of ``values (B, N, C)`` with ``index (B, M, K)``."""
    batch, queries, neighbours = index.shape
    flat = gather_points(values, index.reshape(batch, queries * neighbours))
    return flat.reshape(batch, queries, neighbours, values.shape[-1])
