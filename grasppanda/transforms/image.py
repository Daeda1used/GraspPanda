"""Image-space preprocessing: back-projection and workspace providers.

These steps run before the label-alignment boundary and decide which pixels become
points. Their configuration is the preprocessing fingerprint.

A *workspace provider* restricts the observation to the region where objects are. The
GraspNet-1Billion reference pipelines and the official GraspClutter6D tool derive that
region from the **ground-truth** segmentation, at test time as well, so numbers measured
that way are measured on inputs cropped with knowledge of where the objects are. Here
the provider is a named component that stamps the origin of its information on the
observation:

==================================  ===========================  ==========================
provider                            information used             ``provenance["workspace"]``
==================================  ===========================  ==========================
:class:`SegmentationBoxWorkspace`   ground-truth segmentation    ``ground_truth``
:class:`SegmentationRectWorkspace`  ground-truth segmentation    ``ground_truth``
:class:`TableBoxWorkspace`          camera-to-table calibration  ``calibration``
:class:`NoWorkspace`                nothing                      no entry
==================================  ===========================  ==========================
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError, Provenance

from .base import ImageTransform
from .frame import Frame

__all__ = [
    "DepthToCloud",
    "NoWorkspace",
    "SegmentationBoxWorkspace",
    "SegmentationRectWorkspace",
    "TableBoxWorkspace",
    "WorkspaceProvider",
]


@TRANSFORMS.register()
class DepthToCloud(ImageTransform):
    """Back-project the depth image into an organised cloud; valid where depth > 0.

    ``z = depth / depth_scale``, ``x = (u - cx) * z / fx``, ``y = (v - cy) * z / fy`` in
    double precision, evaluated in this order. Invalid pixels land on the camera centre
    and stay in the organised cloud; only ``valid`` says they are not points.
    """

    source = "GraspPanda"

    def __init__(self) -> None:
        pass

    def __call__(self, frame: Frame) -> Frame:
        if frame.depth.ndim != 2:
            raise ContractError(f"{frame.key}: depth must be (H, W), got {frame.depth.shape}")
        height, width = frame.depth.shape
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
        # Per-column and per-row offsets broadcast against z: the same operations per
        # pixel as on full coordinate grids, without building them.
        columns = np.arange(width) - cx
        rows = (np.arange(height) - cy)[:, None]
        cloud = np.empty((height, width, 3), dtype=np.float64)
        z = cloud[..., 2]
        np.divide(frame.depth, np.float64(frame.depth_scale), out=z)
        np.divide(columns * z, fx, out=cloud[..., 0])
        np.divide(rows * z, fy, out=cloud[..., 1])
        frame.cloud = cloud
        frame.valid = frame.depth > 0
        return frame


class WorkspaceProvider(ImageTransform):
    """Restricts the valid pixels to a workspace. A pipeline holds exactly one.

    Subclasses implement :meth:`inside` and declare where their information comes from.
    """

    #: Recorded as ``observation.provenance["workspace"]``; ``None`` records nothing.
    provenance: ClassVar[Provenance | None] = None

    def inside(self, frame: Frame) -> np.ndarray | None:
        """``(H, W)`` mask of the pixels inside the workspace, or ``None`` for all."""
        raise NotImplementedError

    def __call__(self, frame: Frame) -> Frame:
        name = type(self).__name__
        _, valid = frame.require_cloud(name)
        if frame.workspace is not None:
            raise ContractError(
                f"{frame.key}: `preprocess` holds two workspace providers ({frame.workspace} "
                f"and {name}); keep exactly one"
            )
        inside = self.inside(frame)
        if inside is not None:
            frame.valid = valid & inside
        frame.workspace = name
        if self.provenance is not None:
            frame.provenance["workspace"] = self.provenance
        return frame


def _inside_box(
    columns: tuple[np.ndarray, np.ndarray, np.ndarray], lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Strictly inside an axis-aligned box, per point; ``columns`` are its x, y and z."""
    inside = (columns[0] > lower[0]) & (columns[0] < upper[0])
    for axis in (1, 2):
        inside &= columns[axis] > lower[axis]
        inside &= columns[axis] < upper[axis]
    return inside


@TRANSFORMS.register()
class SegmentationBoxWorkspace(WorkspaceProvider):
    """Crop to the bounding box of the ground-truth foreground. **Uses ground truth.**

    The box is the axis-aligned extent, in the table frame, of every pixel whose
    segmentation id is positive, enlarged by ``margin`` on each side; a point is kept
    when it lies strictly inside. This is the crop of the GraspNet-1Billion reference
    pipelines (graspnet-baseline, Graspness, EconomicGrasp, Scale-Balanced-Grasp), which
    apply it during testing too.

    Args:
        margin: Enlargement of the box in metres (reference: 0.02).
        invalid_depth_foreground: What to do with foreground pixels that have no depth.
            Such a pixel back-projects to the camera centre. ``"include"`` reproduces
            the references, which take the extent over *all* foreground pixels: when one
            of them lacks depth, the box stretches to the camera position.
            ``"exclude"`` builds the box from measured foreground points only; it is a
            controlled variant with its own fingerprint.
    """

    source = "GraspNet-baseline (Fang et al., CVPR 2020)"

    uses_ground_truth = True
    provenance = Provenance.GROUND_TRUTH

    def __init__(
        self, margin: float, invalid_depth_foreground: Literal["include", "exclude"]
    ) -> None:
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.margin = float(margin)
        self.invalid_depth_foreground = invalid_depth_foreground

    def inside(self, frame: Frame) -> np.ndarray:
        name = type(self).__name__
        if frame.segmentation is None:
            raise ContractError(
                f"{name} needs the ground-truth segmentation of {frame.key}, which the dataset "
                "did not provide"
            )
        # One coordinate at a time, each in contiguous memory: the numbers are those of
        # frame.table_points(), the extent and the box test run several times faster.
        columns = frame.table_coordinates(name)
        foreground = frame.segmentation.reshape(-1) > 0
        if self.invalid_depth_foreground == "exclude":
            foreground = foreground & (frame.depth.reshape(-1) > 0)
        if not foreground.any():
            raise ContractError(
                f"{frame.key}: the segmentation has no foreground pixel, so {name} cannot "
                "place a workspace"
            )
        extent = [column[foreground] for column in columns]
        lower = np.array([values.min() for values in extent]) - self.margin
        upper = np.array([values.max() for values in extent]) + self.margin
        return _inside_box(columns, lower, upper).reshape(frame.depth.shape)


@TRANSFORMS.register()
class SegmentationRectWorkspace(WorkspaceProvider):
    """Crop to the image rectangle around the ground-truth foreground. **Uses ground truth.**

    The rectangle is the pixel bounding box of every pixel whose segmentation id is
    positive, enlarged to the left and right by ``margin`` times the image width and to
    the top and bottom by ``margin`` times the image height (rounded down), and clipped
    to the image. This is the crop of the official GraspClutter6D tool, which uses a
    margin of 0.1.

    Args:
        margin: Enlargement on each side as a fraction of the image size.
    """

    source = "graspclutter6dAPI"

    uses_ground_truth = True
    provenance = Provenance.GROUND_TRUTH

    def __init__(self, margin: float) -> None:
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.margin = float(margin)

    def inside(self, frame: Frame) -> np.ndarray:
        name = type(self).__name__
        if frame.segmentation is None:
            raise ContractError(
                f"{name} needs the ground-truth segmentation of {frame.key}, which the dataset "
                "did not provide"
            )
        foreground = frame.segmentation > 0
        columns = np.flatnonzero(foreground.any(axis=0))
        rows = np.flatnonzero(foreground.any(axis=1))
        if len(columns) == 0:
            raise ContractError(
                f"{frame.key}: the segmentation has no foreground pixel, so {name} cannot "
                "place a workspace"
            )
        height, width = foreground.shape
        pad_x, pad_y = int(width * self.margin), int(height * self.margin)
        inside = np.zeros(foreground.shape, dtype=bool)
        inside[
            max(0, rows[0] - pad_y) : min(height, rows[-1] + 1 + pad_y),
            max(0, columns[0] - pad_x) : min(width, columns[-1] + 1 + pad_x),
        ] = True
        return inside


@TRANSFORMS.register()
class TableBoxWorkspace(WorkspaceProvider):
    """Crop to a fixed box in the table frame. Uses calibration only, no ground truth.

    Args:
        lower: ``[x, y, z]`` minimum corner in the table frame, metres.
        upper: ``[x, y, z]`` maximum corner.
    """

    source = "GraspPanda"

    provenance = Provenance.CALIBRATION

    def __init__(self, lower: list[float], upper: list[float]) -> None:
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("lower and upper must each be three coordinates [x, y, z]")
        self.lower = [float(value) for value in lower]
        self.upper = [float(value) for value in upper]
        if any(low >= high for low, high in zip(self.lower, self.upper)):
            raise ValueError(f"empty box: lower {self.lower} must be below upper {self.upper}")

    def inside(self, frame: Frame) -> np.ndarray:
        columns = frame.table_coordinates(type(self).__name__)
        inside = _inside_box(columns, np.array(self.lower), np.array(self.upper))
        return inside.reshape(frame.depth.shape)


@TRANSFORMS.register()
class NoWorkspace(WorkspaceProvider):
    """Keep every pixel with valid depth. States the absence of a crop explicitly."""

    source = "GraspPanda"

    def __init__(self) -> None:
        pass

    def inside(self, frame: Frame) -> None:
        return None
