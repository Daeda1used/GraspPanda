"""Approach-view templates and grasp frames (GraspNet-1Billion conventions)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "approach_to_rotation",
    "fibonacci_views",
    "healpix_views",
    "icosphere_views",
    "nearest_view",
    "rotation_angle",
    "two_vector_rotation",
]


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
    forward = approach / torch.linalg.vector_norm(approach, dim=-1, keepdim=True)
    # The second column lies in the horizontal plane: the world's vertical crossed with
    # the approach, which vanishes exactly when the approach is vertical.
    vertical = forward.new_tensor([0.0, 0.0, 1.0]).expand_as(forward)
    side = torch.linalg.cross(vertical, forward, dim=-1)
    length = torch.linalg.vector_norm(side, dim=-1, keepdim=True)
    undefined = length == 0
    side = torch.where(
        undefined,
        forward.new_tensor([0.0, 1.0, 0.0]).expand_as(forward),
        side / length.masked_fill(undefined, 1.0),
    )
    third = torch.linalg.cross(forward, side, dim=-1)
    if angle is not None:
        # Turning the frame about its own first column mixes the other two.
        cosine = torch.cos(angle).to(forward.dtype).unsqueeze(-1)
        sine = torch.sin(angle).to(forward.dtype).unsqueeze(-1)
        side, third = cosine * side + sine * third, cosine * third - sine * side
    return torch.stack([forward, side, third], dim=-1)


def rotation_angle(first: Tensor, second: Tensor) -> Tensor:
    """Angle in radians of the rotation taking ``first`` to ``second`` (``(..., 3, 3)``)."""
    trace = (first * second).sum(dim=(-2, -1))
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


def icosphere_views(subdivisions: int = 2, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Viewpoints at the vertices of a subdivided icosahedron (through ``trimesh``).

    ``subdivisions`` 0, 1, 2, 3, 4 give 12, 42, 162, 642, 2562 views; the count is
    ``10 · 4**subdivisions + 2``. The vertices are almost evenly spaced and, unlike a
    Fibonacci lattice, come with a mesh (every view knows its neighbours).

    Returns:
        ``(V, 3)`` unit vectors.
    """
    from grasppanda.utils.imports import require

    if subdivisions < 0 or subdivisions > 5:
        raise ValueError(f"subdivisions must be between 0 and 5, got {subdivisions}")
    creation = require("trimesh.creation", extra="trimesh", purpose="An icosphere view lattice")
    views = torch.as_tensor(creation.icosphere(subdivisions=subdivisions).vertices, dtype=dtype)
    return views / views.norm(dim=-1, keepdim=True)


def healpix_views(nside: int = 4, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Viewpoints at the pixel centres of a HEALPix grid (through ``healpix``).

    HEALPix cuts the sphere into ``12 · nside**2`` pixels of exactly equal area on
    rings of constant latitude: ``nside`` 2, 4, 5, 8 give 48, 192, 300, 768 views.
    Pixels are in RING order.

    Returns:
        ``(V, 3)`` unit vectors.
    """
    from grasppanda.utils.imports import require

    if nside < 1:
        raise ValueError(f"nside must be at least 1, got {nside}")
    healpix = require("healpix", extra="healpix", purpose="A HEALPix view lattice")
    x, y, z = healpix.pix2vec(nside, list(range(12 * nside * nside)))
    views = torch.stack([torch.as_tensor(axis, dtype=torch.float64) for axis in (x, y, z)], dim=-1)
    return (views / views.norm(dim=-1, keepdim=True)).to(dtype)


def nearest_view(directions: Tensor, views: Tensor) -> Tensor:
    """``(...)`` index of the row of ``views`` closest in angle to each direction.

    Args:
        directions: ``(..., 3)`` vectors (any length above zero).
        views: ``(V, 3)`` unit vectors.
    """
    return (directions @ views.T).argmax(dim=-1)


def two_vector_rotation(first: Tensor, second: Tensor) -> Tensor:
    """Rotation whose first column is ``first`` and whose second lies towards ``second``.

    Gram-Schmidt on two free 3-vectors: ``first`` is normalised, ``second`` loses its
    component along it and is normalised, the third column is their cross product. This
    is the mapping of the 6-D rotation representation (Zhou et al., CVPR 2019), and,
    with the roles of the two vectors exchanged, how Contact-GraspNet builds a grasp
    frame from a predicted baseline and approach.

    Returns:
        ``(..., 3, 3)`` rotation matrices with columns ``first, second', first × second'``.
    """
    eps = 1e-8
    axis_x = first / first.norm(dim=-1, keepdim=True).clamp(min=eps)
    rest = second - (second * axis_x).sum(dim=-1, keepdim=True) * axis_x
    axis_y = rest / rest.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.stack([axis_x, axis_y, torch.cross(axis_x, axis_y, dim=-1)], dim=-1)
