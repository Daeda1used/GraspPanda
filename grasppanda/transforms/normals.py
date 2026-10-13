"""Surface normals from the depth image, and coordinates, as per-point input features.

FineGrasp feeds its encoder ``[x, y, z, n_x, n_y, n_z]`` per point. The normal of a
pixel is the cross product of the two tangents of the organised cloud ``p``,

    n(v, u) = (p(v, u + s) - p(v, u - s)) × (p(v + s, u) - p(v - s, u)),   normalised,

computed on the whole image before any workspace crop, so that a point at the edge of
the crop still has its four neighbours. On a surface that faces the camera this vector
points away from it (``n_z > 0``); ``towards_camera`` negates it.

The authors' training set stores the normals as files that their loader divides by 255,
and neither the files nor their generator are published. :class:`DepthNormals` is
therefore a reconstruction: the estimator is equation 5 of the paper, the orientation
and the encoding were identified with the released weights (see ``configs/finegrasp``):
the weights expect camera-facing normals in the value range of an 8-bit normal image,
``(n + 1) / 2`` in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError

from .base import PointTransform
from .frame import PointSample

__all__ = ["CoordinateFeatures", "DepthNormals", "depth_normal_map"]


def depth_normal_map(
    depth: np.ndarray, intrinsics: np.ndarray, step: int = 1, towards_camera: bool = True
) -> np.ndarray:
    """Unit normals ``(H, W, 3)`` of a depth image, float64; zero where undefined.

    ``depth`` is in metres with zero for missing pixels. The cloud is back-projected in
    double precision (``x = (u - cx) z / fx``, ``y = (v - cy) z / fy``); the image is
    extended by ``step`` pixels of edge replication, so a border pixel uses a one-sided
    difference. A missing pixel sits on the camera centre and takes part in the
    differences of its neighbours like any other point, which is what a generator
    without a validity mask produces. Where the cross product vanishes the normal is
    zero.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    height, width = depth.shape
    z = depth.astype(np.float64)
    cloud = np.empty((height, width, 3), dtype=np.float64)
    cloud[..., 0] = (np.arange(width) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    cloud[..., 1] = (np.arange(height) - intrinsics[1, 2])[:, None] * z / intrinsics[1, 1]
    cloud[..., 2] = z
    padded = np.pad(cloud, ((step, step), (step, step), (0, 0)), mode="edge")
    across = padded[step:-step, 2 * step :] - padded[step:-step, : -2 * step]
    down = padded[2 * step :, step:-step] - padded[: -2 * step, step:-step]
    normal = np.cross(down, across) if towards_camera else np.cross(across, down)
    length = np.sqrt((normal * normal).sum(axis=-1, keepdims=True))
    return np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)


@TRANSFORMS.register()
class DepthNormals(PointTransform):
    """Give every point the surface normal of its pixel as input features.

    Reads the depth image of the sample (the dataset must be configured with
    ``images: [depth]``) and writes ``(N, 3)`` float32 features. It is placed before
    sampling and augmentation: later steps gather the features with the points.
    A later rigid map of the cloud moves the points and not these features, which is
    what the FineGrasp reference does (its flip augmentation mirrors the cloud and
    leaves the normals as they were).

    Args:
        step: Distance in pixels between a pixel and the neighbours its tangents are
            taken from.
        towards_camera: Orient the normals towards the camera (``n_z < 0`` on a surface
            that faces it). The released FineGrasp weights expect ``true``.
        encoding: ``image8`` stores the normal as an 8-bit image does and rescales it to
            ``[0, 1]``: ``floor((n + 1) / 2 * 255) / 255`` (what the released FineGrasp
            weights expect). ``unit`` keeps the unit vector, components in ``[-1, 1]``.
        keep_depth: Leave the depth image on the sample. By default it is removed once
            the normals exist, because a sample that carries an image refuses the
            point-space augmentations that would move the cloud away from it.
    """

    status = "tested"
    source = "FineGrasp (arXiv 2507.05978)"

    def __init__(
        self,
        step: int = 1,
        towards_camera: bool = True,
        encoding: Literal["image8", "unit"] = "image8",
        keep_depth: bool = False,
    ) -> None:
        if step < 1:
            raise ValueError(f"step must be at least 1, got {step}")
        if encoding not in ("image8", "unit"):
            raise ValueError(f"encoding must be 'image8' or 'unit', got {encoding!r}")
        self.step = int(step)
        self.towards_camera = bool(towards_camera)
        self.encoding = encoding
        self.keep_depth = bool(keep_depth)

    def encode(self, normal: np.ndarray) -> np.ndarray:
        """Unit normals ``(..., 3)`` as the float32 features the encoding stands for."""
        if self.encoding == "unit":
            return normal.astype(np.float32)
        return (((normal + 1.0) / 2.0 * 255.0).astype(np.uint8) / 255.0).astype(np.float32)

    def __call__(self, sample: PointSample) -> PointSample:
        name = type(self).__name__
        if sample.depth is None:
            raise ContractError(
                f"{name} needs the depth image of {sample.key}; configure the dataset with "
                "`images: [depth]`"
            )
        if sample.index is not None:
            raise ContractError(
                f"{name} must run before the cloud of {sample.key} is sampled: put it first in "
                "`transforms`"
            )
        if tuple(sample.depth.shape) != tuple(sample.image_size):
            raise ContractError(
                f"{name}: the depth image of {sample.key} is {sample.depth.shape}, the cloud "
                f"was taken from an image of {sample.image_size}"
            )
        normals = depth_normal_map(sample.depth, sample.intrinsics, self.step, self.towards_camera)
        sample.feats = self.encode(normals.reshape(-1, 3)[sample.pixels])
        if not self.keep_depth:
            sample.depth = None
        return sample


@TRANSFORMS.register()
class CoordinateFeatures(PointTransform):
    """Put the coordinates of every point in front of its input features.

    ``feats ← [x, y, z | feats]`` in float32, or the coordinates alone when the sample
    has no features yet. Placed after the augmentations, the coordinates are those of
    the cloud the model sees.
    """

    status = "tested"
    source = "FineGrasp (arXiv 2507.05978)"

    def __init__(self) -> None:
        pass

    def __call__(self, sample: PointSample) -> PointSample:
        coords = sample.points.astype(np.float32)
        if sample.feats is None:
            sample.feats = coords
        else:
            if len(sample.feats) != len(coords):
                raise ContractError(
                    f"{type(self).__name__}: {sample.key} has {len(sample.feats)} feature rows "
                    f"for {len(coords)} points"
                )
            sample.feats = np.concatenate([coords, sample.feats.astype(np.float32)], axis=1)
        return sample
