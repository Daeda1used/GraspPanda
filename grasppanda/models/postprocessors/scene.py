"""What several post-processors need from the observation: the cloud of one frame,
grasps expressed in their gripper frames, a rigid transform given in a config.

Nothing here is a component.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from grasppanda.structures import ContractError, GraspGroup, Observation

__all__ = ["as_transform", "fit_plane", "frame_index", "gripper_frame_points", "scene_points"]

#: Grasps handled per block; bounds the (grasps × scene points) work tensor.
CHUNK = 128


def frame_index(name: str, grasps: GraspGroup, observation: Observation) -> int:
    """Which frame of the observation ``grasps`` belong to."""
    if grasps.key is None:
        if len(observation) != 1:
            raise ContractError(
                f"{name} cannot tell which of the {len(observation)} frames these grasps "
                "belong to: the GraspGroup has no key"
            )
        return 0
    if grasps.key in observation.keys:
        return observation.keys.index(grasps.key)
    raise ContractError(f"{name}: the observation does not contain frame {grasps.key}")


def scene_points(
    name: str,
    grasps: GraspGroup,
    observation: Observation,
    cloud: Literal["full_points", "points"] = "full_points",
    *,
    fallback: bool = False,
) -> Tensor:
    """``(N, 3)`` points of the frame of ``grasps``, on the device of the grasps.

    Args:
        name: The component that asks, for the messages.
        grasps: Their ``key`` selects the frame and their ``frame`` must be the cloud's.
        observation: The batch the grasps were predicted from.
        cloud: ``full_points`` (before sub-sampling) or ``points`` (what the network saw).
        fallback: Read ``points`` when ``full_points`` was asked for and is absent.

    Raises:
        ContractError: The cloud is absent, or in another frame than the grasps (a step
            that changes the frame, such as ``ToFrame``, must come last).
    """
    if cloud not in ("full_points", "points"):
        raise ValueError(f"cloud must be 'full_points' or 'points', got {cloud!r}")
    full = observation.full_points if cloud == "full_points" else observation.points
    if full is None and fallback:
        full = observation.points
    if full is None:
        what = (
            "observation.full_points (the cloud before sub-sampling)"
            if cloud == "full_points"
            else "observation.points"
        )
        hint = (
            ". Keep it with `keep_full_points: true` in the dataset section of the config "
            "(`--opts data.test.dataset.keep_full_points=true`), or test the cloud the network "
            "saw with `cloud: points`"
            if cloud == "full_points"
            else ""
        )
        raise ContractError(f"{name} reads {what}, which this observation does not provide{hint}")
    if full.frame != grasps.frame:
        raise ContractError(
            f"{name}: grasps are in frame '{grasps.frame}' but the scene cloud is in "
            f"frame '{full.frame}'"
        )
    if grasps.key is None and full.batch_size != 1:
        raise ContractError(
            f"{name} cannot tell which of the {full.batch_size} frames these grasps "
            "belong to: the GraspGroup has no key"
        )
    frame = frame_index(name, grasps, observation)
    if full.batch_offsets is None:
        return full.coords.to(grasps.translation.device)
    start, stop = full.batch_offsets[frame], full.batch_offsets[frame + 1]
    return full.coords[start:stop].to(grasps.translation.device)


def gripper_frame_points(grasps: GraspGroup, scene: Tensor) -> Tensor:
    """``(G, N, 3)`` scene points in the gripper frame of every grasp.

    x is the approach axis, y the closing axis, z the finger-height axis; the origin is
    the grasp point. Call it on a slice of at most :data:`CHUNK` grasps.
    """
    offset = scene[None].to(grasps.translation) - grasps.translation[:, None]
    return offset @ grasps.rotation


def as_transform(value: Sequence[Sequence[float]] | str, name: str) -> Tensor:
    """A rigid ``(4, 4)`` float64 transform from a nested list or a ``.npy``/text file."""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"{name}: transform file not found: {path}")
        array = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    else:
        array = np.asarray(value, dtype=np.float64)
    matrix = torch.as_tensor(np.asarray(array, dtype=np.float64))
    if matrix.shape != (4, 4):
        raise ValueError(
            f"{name}: a transform is a 4x4 homogeneous matrix (rows of [R | t], then "
            f"[0, 0, 0, 1]); got shape {tuple(matrix.shape)}"
        )
    rotation = matrix[:3, :3]
    error = float((rotation.T @ rotation - torch.eye(3, dtype=torch.float64)).abs().max())
    last = matrix[3] - torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    if error > 1e-4 or float(torch.linalg.det(rotation)) < 0 or float(last.abs().max()) > 1e-6:
        raise ValueError(
            f"{name}: the transform is not rigid (rotation part deviates from an orthonormal "
            f"matrix by {error:.2e}, or the last row is not [0, 0, 0, 1]). Lengths are metres."
        )
    return matrix


def fit_plane(
    points: Tensor, *, threshold: float = 0.01, iterations: int = 200, seed: int = 0
) -> tuple[Tensor, float]:
    """The dominant plane of a cloud by RANSAC, refined by least squares on its inliers.

    Returns:
        ``(plane, inlier_fraction)`` with ``plane = [a, b, c, d]`` such that
        ``a·x + b·y + c·z + d`` is the signed distance of a point in metres, positive on
        the side of the origin (the camera, for a cloud in the camera frame).
    """
    cloud = points.double()
    if len(cloud) < 3:
        raise ContractError(f"a plane needs at least 3 points, the cloud has {len(cloud)}")
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randint(len(cloud), (iterations, 3), generator=generator).to(cloud.device)
    a, b, c = cloud[picks[:, 0]], cloud[picks[:, 1]], cloud[picks[:, 2]]
    normal = torch.linalg.cross(b - a, c - a)
    length = normal.norm(dim=1, keepdim=True)
    valid = length[:, 0] > 1e-12
    normal = normal / length.clamp_min(1e-12)
    offset = -(normal * a).sum(dim=1)
    # At most ~20 000 points decide; the refinement below uses every point.
    stride = max(1, len(cloud) // 20000)
    distance = (cloud[::stride] @ normal.T + offset).abs()
    votes = (distance < threshold).sum(dim=0)
    votes[~valid] = -1
    best = int(votes.argmax())
    inliers = (cloud @ normal[best] + offset[best]).abs() < threshold
    chosen = cloud[inliers]
    centre = chosen.mean(dim=0)
    _, _, vh = torch.linalg.svd(chosen - centre, full_matrices=False)
    refined = vh[-1]
    d = -float(refined @ centre)
    if d < 0:  # the origin on the positive side
        refined, d = -refined, -d
    plane = torch.cat([refined, refined.new_tensor([d])])
    return plane, float(inliers.double().mean())
