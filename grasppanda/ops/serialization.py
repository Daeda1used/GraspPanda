"""Space-filling-curve codes for point cloud serialisation.

Point Transformer V3 (Wu et al., CVPR 2024) turns an unordered set of voxels into a
sequence by sorting them along a space-filling curve, then runs attention inside
consecutive patches of that sequence. Two curves are used, each also with x and y
swapped, so that different blocks see different neighbourhoods:

* ``z``: Morton order. Bit ``i`` of x, y and z goes to bits ``3i + 2``, ``3i + 1`` and
  ``3i`` of the code (x is the most significant axis), the layout PTv3 inherits from
  O-CNN.
* ``hilbert``: the Hilbert curve, computed with Skilling's transpose algorithm
  ("Programming the Hilbert curve", AIP Conf. Proc. 707, 2004). Consecutive codes are
  always face-adjacent cells, which Morton order does not guarantee.

Both functions are written from those definitions on integer tensors; they return the
same codes as the reference implementation (checked in the tests against values
recorded from it), which is what lets official PTv3-family checkpoints reproduce.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch
from torch import Tensor

__all__ = ["SerializationOrder", "hilbert_code", "serialization_code", "z_order_code"]

SerializationOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]

#: A code is stored in a signed 64-bit integer.
_CODE_BITS = 63


def _check_grid(grid: Tensor, depth: int) -> Tensor:
    if grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError(f"expected integer grid coordinates (N, 3), got {tuple(grid.shape)}")
    if grid.is_floating_point():
        raise ValueError("grid coordinates must be integers; quantise the points first")
    if not 0 < 3 * depth <= _CODE_BITS:
        raise ValueError(f"depth must be in 1..{_CODE_BITS // 3}, got {depth}")
    grid = grid.long()
    if len(grid) and (grid.min() < 0 or grid.max() >= 1 << depth):
        raise ValueError(
            f"grid coordinates must lie in [0, 2**{depth}) for depth {depth}, got "
            f"[{int(grid.min())}, {int(grid.max())}]"
        )
    return grid


def z_order_code(grid: Tensor, depth: int) -> Tensor:
    """Morton code of non-negative integer coordinates.

    Args:
        grid: ``(N, 3)`` integer coordinates in ``[0, 2**depth)``.
        depth: Number of bits per axis.

    Returns:
        ``(N,)`` int64 codes in ``[0, 8**depth)``.
    """
    x, y, z = _check_grid(grid, depth).unbind(-1)
    code = torch.zeros_like(x)
    for bit in range(depth):
        code |= ((x >> bit) & 1) << (3 * bit + 2)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit)
    return code


def hilbert_code(grid: Tensor, depth: int) -> Tensor:
    """Hilbert-curve index of non-negative integer coordinates.

    Args:
        grid: ``(N, 3)`` integer coordinates in ``[0, 2**depth)``.
        depth: Number of bits per axis. The curve depends on it: the same cell has a
            different index on a deeper curve.

    Returns:
        ``(N,)`` int64 indices in ``[0, 8**depth)``.
    """
    axes = list(_check_grid(grid, depth).unbind(-1))
    # Skilling's "transpose" form: walk from the most significant bit down. Where an
    # axis has the bit set, invert the lower bits of axis 0; where it does not,
    # exchange the lower bits of that axis with those of axis 0.
    for bit in range(depth - 1, 0, -1):
        low = (1 << bit) - 1
        for dim in range(3):
            on = ((axes[dim] >> bit) & 1).bool()
            exchange = (axes[0] ^ axes[dim]) & low
            first = torch.where(on, axes[0] ^ low, axes[0] ^ exchange)
            if dim:
                axes[dim] = torch.where(on, axes[dim], axes[dim] ^ exchange)
            axes[0] = first
    # Interleave the axes (axis 0 most significant) into one Gray-coded integer ...
    code = torch.zeros_like(axes[0])
    for bit in range(depth):
        for dim in range(3):
            code |= ((axes[dim] >> bit) & 1) << (3 * bit + 2 - dim)
    # ... and decode Gray to binary: every bit is the XOR of all bits above it.
    shift = 1
    while shift < 3 * depth:
        code = code ^ (code >> shift)
        shift *= 2
    return code


def serialization_code(
    grid: Tensor,
    depth: int,
    order: SerializationOrder = "z",
    batch_index: Tensor | None = None,
) -> Tensor:
    """Sort key of every voxel along one space-filling curve.

    Args:
        grid: ``(N, 3)`` integer coordinates in ``[0, 2**depth)``.
        depth: Number of bits per axis.
        order: Curve. The ``-trans`` variants swap x and y before encoding.
        batch_index: ``(N,)`` sample index of every voxel. When given it occupies the
            bits above the curve code, so sorting the codes keeps samples contiguous.

    Returns:
        ``(N,)`` int64 codes.
    """
    if order not in get_args(SerializationOrder):
        raise ValueError(
            f"unknown serialisation order {order!r}; choose from {get_args(SerializationOrder)}"
        )
    if order.endswith("-trans"):
        grid = grid[:, [1, 0, 2]]
    code = hilbert_code(grid, depth) if order.startswith("hilbert") else z_order_code(grid, depth)
    if batch_index is not None:
        batch_index = batch_index.long()
        if len(batch_index) and 3 * depth + int(batch_index.max()).bit_length() > _CODE_BITS:
            raise ValueError(
                f"{int(batch_index.max()) + 1} samples at depth {depth} do not fit a 64-bit code"
            )
        code = batch_index << (3 * depth) | code
    return code
