"""What travels between the stages of HGGD.

The image stage does not propose points: it proposes *partial grasps*. A partial grasp
fixes the grasp centre in the image, the rotation about the optical axis and the jaw
opening, and leaves the two out-of-plane angles and a small translation to the region
stage. That is a different thing from a seed of the seed-based family (a scene point
with a graspness score), so it has a structure of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from grasppanda.structures import ContractError

__all__ = ["ImagePredictions", "PartialGraspSet", "RegionDraws", "RegionSet"]


@dataclass(eq=False)
class ImagePredictions:
    """Raw outputs of the image network, in its ``(batch, channel, u, v)`` layout.

    ``u`` runs along the image width and ``v`` along its height; maps at the proposal
    stride are ``stride`` times smaller in both.

    Attributes:
        heatmap: ``(B, U, V)`` logits of "a grasp centre is here", full resolution.
        angle: ``(B, K, U/s, V/s)`` logits of the ``K`` in-plane angle bins.
        angle_offset: ``(B, K, U/s, V/s)`` offset inside the bin, in bins.
        depth_offset: ``(B, K, U/s, V/s)`` centre depth relative to the surface, in units
            of twice the depth anchor.
        width_offset: ``(B, K, U/s, V/s)`` log of the width relative to the width anchor.
        features: ``(B, C, U/s, V/s)`` feature map handed to the region stage.
    """

    heatmap: Tensor
    angle: Tensor
    angle_offset: Tensor
    depth_offset: Tensor
    width_offset: Tensor
    features: Tensor


@dataclass(eq=False)
class PartialGraspSet:
    """Partial grasps of a batch of frames, in descending heat order per frame.

    Attributes:
        frame: ``(P,)`` index of the frame each proposal belongs to, non-decreasing.
        pixel: ``(P, 2)`` integer ``(u, v)`` of the grasp centre in the network image.
        theta: ``(P,)`` in-plane angle in ``[-pi/2, pi/2]``, radians.
        width: ``(P,)`` jaw opening in pixels of the network image.
        depth_offset: ``(P,)`` grasp centre minus observed surface along the camera z,
            metres.
        score: ``(P,)`` heat value at the centre pixel, in ``(0, 1)``.
        centre: ``(P, 3)`` grasp centre in the camera frame, metres; ``None`` until the
            proposals are lifted.
        num_frames: Size of the batch; frames without proposals are simply absent
            from ``frame``.
    """

    frame: Tensor
    pixel: Tensor
    theta: Tensor
    width: Tensor
    depth_offset: Tensor
    score: Tensor
    centre: Tensor | None = None
    num_frames: int = 1

    _PER_GRASP = ("frame", "pixel", "theta", "width", "depth_offset", "score", "centre")

    def __post_init__(self) -> None:
        count = self.frame.shape[0]
        shapes = {
            "frame": (count,),
            "pixel": (count, 2),
            "theta": (count,),
            "width": (count,),
            "depth_offset": (count,),
            "score": (count,),
            "centre": (count, 3),
        }
        for name, expected in shapes.items():
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != expected:
                raise ContractError(
                    f"PartialGraspSet.{name} must have shape {expected}, got {tuple(value.shape)}"
                )
        if self.pixel.is_floating_point():
            raise ContractError("PartialGraspSet.pixel holds integer pixel coordinates")
        if count and (
            (self.frame[1:] < self.frame[:-1]).any()
            or self.frame[0] < 0
            or self.frame[-1] >= self.num_frames
        ):
            raise ContractError(
                "PartialGraspSet.frame must be sorted and index one of the "
                f"{self.num_frames} frame(s) of the batch"
            )

    def __len__(self) -> int:
        return self.frame.shape[0]

    def select(self, index: Tensor) -> PartialGraspSet:
        """The proposals ``index`` (a mask or sorted indices), still grouped by frame."""
        picked = {
            name: None if getattr(self, name) is None else getattr(self, name)[index]
            for name in self._PER_GRASP
        }
        return PartialGraspSet(**picked, num_frames=self.num_frames)

    def with_centre(self, centre: Tensor) -> PartialGraspSet:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["centre"] = centre
        return PartialGraspSet(**values)

    def rows(self, frame: int) -> Tensor:
        """Indices of the proposals of one frame."""
        return torch.nonzero(self.frame == frame)[:, 0]

    @classmethod
    def cat(cls, sets: list[PartialGraspSet], num_frames: int) -> PartialGraspSet:
        """Concatenate per-frame sets (already in frame order)."""
        if not sets:
            raise ValueError("cat() needs at least one PartialGraspSet")
        lifted = [s.centre is not None for s in sets]
        if any(lifted) and not all(lifted):
            raise ContractError("cannot concatenate lifted and unlifted partial grasps")
        merged = {
            name: None
            if getattr(sets[0], name) is None
            else torch.cat([getattr(s, name) for s in sets])
            for name in cls._PER_GRASP
        }
        return cls(**merged, num_frames=num_frames)


@dataclass(eq=False)
class RegionDraws:
    """The random choices of the region stage for one frame.

    Drawn from PyTorch's generators in normal use. A test that compares the stage with a
    reference run passes the reference's draws instead: sampling with a random start
    and a random jitter cannot be reproduced any other way.

    Attributes:
        resample: Per region with fewer points than the region size (in centre order),
            ``(num_points,)`` indices into its points, drawn with replacement.
        jitter: ``(P, L, 3)`` offsets added to the coordinates of every region slot.
        start: ``(P,)`` index of the first sample of each region.
    """

    resample: list[Tensor]
    jitter: Tensor
    start: Tensor


@dataclass(eq=False)
class RegionSet:
    """Point regions around the lifted centres that have enough support.

    Attributes:
        points: ``(R, S, 3 + C)`` per region ``S`` points: coordinates relative to the
            centre (metres) followed by the image features attached to each point.
        grasps: The ``R`` partial grasps the regions belong to, lifted.
        kept: ``(P,)`` mask over the proposals that were lifted: which of them got a region.
        support: ``(P,)`` number of scene points inside each proposal's ball.
    """

    points: Tensor
    grasps: PartialGraspSet
    kept: Tensor
    support: Tensor

    def __post_init__(self) -> None:
        if self.points.ndim != 3 or self.points.shape[0] != len(self.grasps):
            raise ContractError(
                f"RegionSet.points must be (R, S, 3 + C) with R = {len(self.grasps)}, got "
                f"{tuple(self.points.shape)}"
            )
        if self.grasps.centre is None:
            raise ContractError("RegionSet.grasps must be lifted (have a 3-D centre)")

    def __len__(self) -> int:
        return self.points.shape[0]
