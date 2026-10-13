"""Grid sampling as the Pointcept data pipeline does it.

The sparse encoders ported from Pointcept (OA-CNNs, the SpUNet layout) do not see
points: the official data pipeline quantises every cloud to a grid first, keeps one
point per occupied cell and hands the network integer cell coordinates that start at
zero in every cloud. Where a cloud starts therefore decides which cells merge when the
network halves the grid, and a backbone that wants to reproduce the official numbers
has to quantise the same way. This module is that step, for backbones whose contract is
"points in, one feature per point out".

Private to the backbones until a second user outside this package appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from grasppanda.ops import voxelize
from grasppanda.structures import ContractError, PointSet

__all__ = ["GridCells", "grid_cells"]


@dataclass(eq=False)
class GridCells:
    """The occupied cells of a batch of clouds, and the way back to the points.

    Attributes:
        cells: ``(V, 3)`` integer cell of every voxel, counted from the lowest occupied
            cell of its own cloud, so every cloud has a voxel at 0 on every axis.
        batch: ``(V,)`` cloud of every voxel, non-decreasing.
        first: ``(V,)`` the first point that fell into every voxel: its representative.
        inverse: ``(N,)`` voxel of every point.
        extent: ``(B, 3)`` number of cells every cloud spans along each axis.
    """

    cells: Tensor
    batch: Tensor
    first: Tensor
    inverse: Tensor
    extent: Tensor


def grid_cells(points: PointSet, grid_size: float, *, user: str) -> GridCells:
    """Quantise every cloud of a batch on its own grid.

    A point belongs to cell ``floor(xyz / grid_size)``, evaluated in double precision as
    the NumPy code of the official pipeline does (in float32 a point on a cell boundary
    can land on either side); the lowest occupied cell of the cloud is then subtracted.

    Args:
        points: The batch of clouds.
        grid_size: Edge of a cell in the unit of the coordinates.
        user: Name of the calling backbone, for error messages.

    Raises:
        ContractError: A cloud of the batch has no points.
    """
    if len(points) == 0 or int(points.sizes.min()) == 0:
        raise ContractError(f"{user} received an empty cloud; every cloud needs points")
    grid = voxelize(points.coords.double(), grid_size, points.batch_index)
    batch, cells = grid.voxels[:, 0], grid.voxels[:, 1:]
    index = batch[:, None].expand_as(cells)
    lowest = torch.zeros(points.batch_size, 3, dtype=torch.long, device=cells.device)
    lowest = lowest.scatter_reduce(0, index, cells, reduce="amin", include_self=False)
    cells = cells - lowest[batch]
    extent = torch.zeros_like(lowest).scatter_reduce(0, index, cells, reduce="amax") + 1
    return GridCells(cells, batch, grid.first, grid.inverse, extent)
