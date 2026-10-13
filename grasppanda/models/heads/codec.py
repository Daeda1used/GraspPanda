"""Grasp parameterisations shared by the four parts of a head bundle.

A codec is the single owner of the discretisation a bundle predicts on: the grids, the
order in which grid cells are flattened into network channels, the units of the
regressed quantities, and the mapping back to metric grasp parameters. The head, its
target encoder, its loss and its decoder all go through the same codec object, so they
cannot disagree (``docs/design/ARCHITECTURE.md`` §8.4, §8.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor

from grasppanda.structures import ContractError

__all__ = ["AngleDepthCodec"]


@dataclass(frozen=True)
class AngleDepthCodec:
    """In-plane angle × insertion depth grid of the GraspNet-1Billion grasp labels.

    Angle bin ``a`` stands for an in-plane rotation of ``a · π / num_angle`` about the
    approach axis (a parallel-jaw gripper is symmetric under a half turn, so the bins
    cover ``[0, π)``). Depth bin ``d`` stands for an insertion depth of
    ``(d + 1) · depth_step``. Grids are laid out ``(..., num_angle, num_depth)`` and
    flattened angle-major: cell ``a · num_depth + d``.

    Widths are regressed in units of ``1 / width_scale`` metres. Decoding multiplies
    the prediction by ``width_decode_gain`` before converting back; the reference
    implementations open the gripper 20 % wider than regressed, in the decoder only.

    Attributes:
        num_angle: Number of in-plane angle bins.
        num_depth: Number of depth bins.
        depth_step: Metres between consecutive depth bins.
        width_scale: Network width units per metre.
        width_decode_gain: Factor applied to predicted widths when decoding.
    """

    num_angle: int = 12
    num_depth: int = 4
    depth_step: float = 0.01
    width_scale: float = 10.0
    width_decode_gain: float = 1.2

    def __post_init__(self) -> None:
        if self.num_angle < 1 or self.num_depth < 1:
            raise ValueError("num_angle and num_depth must be positive")
        if self.depth_step <= 0 or self.width_scale <= 0 or self.width_decode_gain <= 0:
            raise ValueError("depth_step, width_scale and width_decode_gain must be positive")

    @property
    def grid(self) -> tuple[int, int]:
        """``(num_angle, num_depth)``."""
        return (self.num_angle, self.num_depth)

    @property
    def num_cells(self) -> int:
        return self.num_angle * self.num_depth

    def expect_grid(self, tensor: Tensor, *, what: str, user: str) -> None:
        """Raise unless the last two axes of ``tensor`` are this codec's grid."""
        if tuple(tensor.shape[-2:]) != self.grid:
            raise ContractError(
                f"{user}: {what} are laid out on an angle × depth grid of "
                f"{tuple(tensor.shape[-2:])}, but the head is configured for {self.grid} "
                "(num_angle × num_depth). The label set and the head bundle must use the same grid."
            )

    # ------------------------------------------------------------------- cell order

    def flatten(self, grid: Tensor) -> Tensor:
        """``(..., num_angle, num_depth)`` → ``(..., num_cells)``."""
        return grid.flatten(-2)

    def unflatten(self, flat: Tensor) -> Tensor:
        """``(..., num_cells)`` → ``(..., num_angle, num_depth)``."""
        return flat.unflatten(-1, self.grid)

    def cell(self, angle_index: Tensor, depth_index: Tensor) -> Tensor:
        """Flat cell of an (angle bin, depth bin) pair."""
        return angle_index * self.num_depth + depth_index

    def split_cell(self, cell: Tensor) -> tuple[Tensor, Tensor]:
        """(angle bin, depth bin) of a flat cell."""
        return cell // self.num_depth, cell % self.num_depth

    # ---------------------------------------------------------------- metric values

    def angle(self, angle_index: Tensor) -> Tensor:
        """In-plane rotation in radians of an angle bin."""
        return (angle_index * math.pi) / self.num_angle

    def depth(self, depth_index: Tensor) -> Tensor:
        """Insertion depth in metres of a depth bin."""
        return (depth_index + 1) * self.depth_step

    def encode_width(self, width: Tensor) -> Tensor:
        """Metres → regression units."""
        return width * self.width_scale

    def decode_width(self, raw: Tensor, max_width: float) -> Tensor:
        """Regression units → metres, with the decode gain, clamped to the gripper."""
        return (self.width_decode_gain * raw / self.width_scale).clamp(min=0.0, max=max_width)
