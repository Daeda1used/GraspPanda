"""Approach-view templates and grasp frames (GraspNet-1Billion conventions)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["approach_to_rotation", "fibonacci_views", "rotation_angle"]


def fibonacci_views(num_views: int = 300, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Viewpoints on the unit sphere from a Fibonacci lattice.

    This is the view grid of the GraspNet-1Billion grasp labels: view ``i`` of the
    label tensors is row ``i`` here. The approach direction of a view is the vector
    from the viewpoint to the origin, i.e. ``-views[i]``.

    Returns:
        ``(num_views, 3)`` unit vectors.
    """
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    index = torch.arange(num_views, dtype=torch.float64)
    z = (2.0 * index + 1.0) / num_views - 1.0
    radius = torch.sqrt(1.0 - z * z)
    angle = 2.0 * math.pi * index * golden
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle), z], dim=-1).to(dtype)


def approach_to_rotation(approach: Tensor, angle: Tensor | None = None) -> Tensor:
    """Gripper frame from an approach direction and an in-plane rotation.

    Column 0 of the result is the normalised approach axis. Column 1 is horizontal
    (``[-a_y, a_x, 0]`` normalised, or ``[0, 1, 0]`` when the approach is vertical),
    column 2 completes the right-handed frame, and the frame is then rotated by
    ``angle`` about the approach axis.

    Args:
        approach: ``(..., 3)`` direction from the gripper towards the object.
        angle: ``(...)`` in-plane rotation in radians; zero when omitted.

    Returns:
        ``(..., 3, 3)`` rotation matrices.
    """
    axis_x = approach / approach.norm(dim=-1, keepdim=True)
    zeros = torch.zeros_like(axis_x[..., 0])
    axis_y = torch.stack([-axis_x[..., 1], axis_x[..., 0], zeros], dim=-1)
    degenerate = axis_y.norm(dim=-1, keepdim=True) == 0
    fallback = torch.tensor([0.0, 1.0, 0.0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.where(degenerate, fallback.expand_as(axis_y), axis_y)
    axis_y = axis_y / axis_y.norm(dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y, dim=-1)
    frame = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    if angle is None:
        return frame
    cos, sin = torch.cos(angle), torch.sin(angle)
    ones = torch.ones_like(cos)
    zeros = torch.zeros_like(cos)
    spin = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1).reshape(
        *angle.shape, 3, 3
    )
    return frame @ spin.to(frame.dtype)


def rotation_angle(first: Tensor, second: Tensor) -> Tensor:
    """Angle in radians of the rotation taking ``first`` to ``second`` (``(..., 3, 3)``)."""
    trace = (first * second).sum(dim=(-2, -1))
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
