"""Voxelisation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = ["VoxelGrid", "voxelize"]


class VoxelGrid(NamedTuple):
    """Result of :func:`voxelize`.

    Attributes:
        voxels: ``(V, 4)`` integer ``[batch, x, y, z]`` of the occupied voxels, sorted.
        inverse: ``(N,)`` voxel row of every input point; ``voxel_feats[inverse]`` maps
            voxel features back to exactly the points that were voxelised.
        first: ``(V,)`` index of the first input point falling in each voxel.
    """

    voxels: Tensor
    inverse: Tensor
    first: Tensor


def voxelize(coords: Tensor, voxel_size: float, batch_index: Tensor | None = None) -> VoxelGrid:
    """Quantise points to a regular grid by flooring ``coords / voxel_size``."""
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"expected (N, 3), got {tuple(coords.shape)}")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    # Divide by a tensor, not by a Python number: on CUDA, PyTorch turns division by a
    # scalar into multiplication by its reciprocal, which puts points next to a voxel
    # boundary into a different voxel than the same code does on the CPU.
    cells = torch.floor(coords / coords.new_tensor(voxel_size)).long()
    if batch_index is None:
        batch_index = torch.zeros(len(coords), dtype=torch.long, device=coords.device)
    keys = torch.cat([batch_index.long()[:, None], cells], dim=1)
    voxels, inverse = torch.unique(keys, dim=0, return_inverse=True)
    position = torch.arange(len(coords), device=coords.device)
    first = torch.full((len(voxels),), len(coords), dtype=torch.long, device=coords.device)
    first = first.scatter_reduce(0, inverse, position, reduce="amin")
    return VoxelGrid(voxels, inverse, first)
