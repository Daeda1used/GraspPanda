"""Per-cloud geometric queries on packed batches.

The reference operators in :mod:`grasppanda.ops` take equally sized clouds as
``(B, N, 3)``. A :class:`~grasppanda.structures.PointSet` instead concatenates clouds of
any size and marks their boundaries with batch offsets. The helpers here bridge the
two: when all clouds have the same size the operator runs once on the whole batch,
otherwise once per cloud. Either way a cloud never sees points of another cloud, and
every returned index is a row of the packed tensor, so results can be used to index
packed coordinates and features directly.

Private to the backbones until a second user appears.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor

from grasppanda.ops import ball_query, furthest_point_sample, knn
from grasppanda.structures import ContractError, PointSet

__all__ = [
    "ball_query_packed",
    "check_features",
    "cloud_sizes",
    "fps_packed",
    "knn_packed",
    "offsets_of",
]


def cloud_sizes(points: PointSet) -> list[int]:
    """Number of points of every cloud in the batch."""
    return [int(size) for size in points.sizes.tolist()]


def offsets_of(sizes: Sequence[int], device: torch.device) -> Tensor:
    """``(B + 1,)`` batch offsets for clouds of the given sizes."""
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.as_tensor(sizes, dtype=torch.long, device=device).cumsum(0)
    return offsets


def check_features(
    points: PointSet, feats: Tensor | None, in_channels: int, *, user: str
) -> Tensor | None:
    """Validate the input features of a backbone and return them (``None`` if it takes none)."""
    if in_channels == 0:
        if feats is not None and feats.shape[-1] != 0:
            raise ContractError(
                f"{user} was built with in_channels=0 (coordinates only) but received "
                f"{feats.shape[-1]} feature channels. Set in_channels to match the features, "
                "or pass feats=None."
            )
        return None
    if feats is None:
        raise ContractError(
            f"{user} was built with in_channels={in_channels} but received no features."
        )
    if feats.ndim != 2 or feats.shape[0] != len(points) or feats.shape[1] != in_channels:
        raise ContractError(
            f"{user} expects features of shape ({len(points)}, {in_channels}) for this point "
            f"set, got {tuple(feats.shape)}. Set in_channels to the number of feature channels "
            "the dataset provides."
        )
    return feats


def _dense_groups(
    *size_lists: Sequence[int],
) -> Iterator[tuple[int, int]]:
    """Ranges of clouds ``[first, last)`` that can go through an operator in one call."""
    batch = len(size_lists[0])
    if all(len(set(sizes)) == 1 for sizes in size_lists):
        yield 0, batch
    else:
        for index in range(batch):
            yield index, index + 1


def _dense(values: Tensor, sizes: Sequence[int], first: int, last: int) -> tuple[Tensor, Tensor]:
    """``(last - first, n, C)`` view of some clouds and the packed row each one starts at."""
    start = sum(sizes[:first])
    count = sizes[first]
    block = values[start : start + count * (last - first)]
    starts = start + count * torch.arange(last - first, device=values.device)
    return block.reshape(last - first, count, values.shape[-1]), starts


@torch.no_grad()
def fps_packed(
    coords: Tensor,
    sizes: Sequence[int],
    count: int | Sequence[int],
    *,
    skip_near_origin: bool = False,
    presampled: bool = False,
    tie_break: str = "lowest_index",
) -> Tensor:
    """Furthest point sampling inside every cloud.

    Args:
        count: Points to keep per cloud: one number for all clouds, or one per cloud.
        tie_break: Which of several equally far candidates is picked, see
            :func:`grasppanda.ops.furthest_point_sample`.
        presampled: The clouds are themselves the output of this function, with the
            same ``skip_near_origin``. Furthest point samples are nested (the first
            ``k`` samples do not depend on how many more are drawn), so sampling such a
            cloud again returns its first ``count`` rows, which is what is done instead
            of repeating the search. Hierarchical encoders spend most of their sampling
            time on exactly this.

    Returns:
        Rows of ``coords``, the samples of cloud 0 first, each cloud in sampling order.
    """
    counts = [count] * len(sizes) if isinstance(count, int) else list(count)
    if presampled:
        if any(wanted > size for wanted, size in zip(counts, sizes)):
            raise ValueError(f"cannot sample {counts} points from clouds of {list(sizes)}")
        starts = [0, *torch.tensor(sizes).cumsum(0).tolist()]
        rows = [torch.arange(start, start + wanted) for start, wanted in zip(starts, counts)]
        return torch.cat(rows).to(coords.device)
    chosen = []
    for first, last in _dense_groups(sizes, counts):
        block, starts = _dense(coords, sizes, first, last)
        index = furthest_point_sample(
            block, counts[first], skip_near_origin=skip_near_origin, tie_break=tie_break
        )
        chosen.append((index + starts[:, None]).reshape(-1))
    return torch.cat(chosen)


@torch.no_grad()
def ball_query_packed(
    centres: Tensor,
    centre_sizes: Sequence[int],
    support: Tensor,
    support_sizes: Sequence[int],
    radius: float,
    nsample: int,
) -> Tensor:
    """Ball query of every centre against the support points of its own cloud.

    Returns:
        ``(M, nsample)`` rows of ``support``, with the padding rule of
        :func:`grasppanda.ops.ball_query`.
    """
    found = []
    for first, last in _dense_groups(centre_sizes, support_sizes):
        query, _ = _dense(centres, centre_sizes, first, last)
        block, starts = _dense(support, support_sizes, first, last)
        index, _ = ball_query(query, block, radius, nsample)
        found.append((index + starts[:, None, None]).reshape(-1, nsample))
    return torch.cat(found)


@torch.no_grad()
def knn_packed(
    query: Tensor,
    query_sizes: Sequence[int],
    support: Tensor,
    support_sizes: Sequence[int],
    k: int,
) -> tuple[Tensor, Tensor]:
    """``k`` nearest support points of every query point within its own cloud.

    Returns:
        ``(distance, index)``, both ``(M, k)``; ``index`` holds rows of ``support``.
    """
    distances, indices = [], []
    for first, last in _dense_groups(query_sizes, support_sizes):
        block_query, _ = _dense(query, query_sizes, first, last)
        block_support, starts = _dense(support, support_sizes, first, last)
        distance, index = knn(block_query, block_support, k)
        distances.append(distance.reshape(-1, k))
        indices.append((index + starts[:, None, None]).reshape(-1, k))
    return torch.cat(distances), torch.cat(indices)
