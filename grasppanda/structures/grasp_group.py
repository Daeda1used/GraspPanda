"""The single output type of every detector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .common import ContractError, Gripper, SampleKey

__all__ = ["GRASPNET_ARRAY_WIDTH", "GraspGroup"]

#: ``[score, width, height, depth, R(9, row-major), t(3), object_id]``
GRASPNET_ARRAY_WIDTH = 17


@dataclass(eq=False)
class GraspGroup:
    """A set of 6-DoF parallel-jaw grasps for one sample.

    Geometry follows the GraspNet-1Billion convention. All lengths are metres.

    Attributes:
        translation: ``(G, 3)`` grasp point in ``frame``.
        rotation: ``(G, 3, 3)`` proper rotations. Column 0 is the approach axis,
            column 1 the closing axis, column 2 the finger-height axis.
        width: ``(G,)`` jaw opening along the closing axis.
        depth: ``(G,)`` how far the fingers reach past the grasp point along the
            approach axis. This is an insertion depth, not a camera depth.
        score: ``(G,)`` ranking score, higher is better. ``score_kind`` names what it
            means; it is not assumed to be a calibrated probability.
        height: ``(G,)`` finger thickness; defaults to the gripper's.
        object_id: ``(G,)`` instance association, ``-1`` when unknown.
        frame: Reference frame of ``translation`` and ``rotation``.
        gripper: Gripper geometry the widths and depths refer to.
        score_kind: Semantics of ``score``, declared by the head bundle.
        key: Which sample these grasps belong to.
    """

    translation: Tensor
    rotation: Tensor
    width: Tensor
    depth: Tensor
    score: Tensor
    height: Tensor | None = None
    object_id: Tensor | None = None
    frame: str = "camera"
    gripper: Gripper = field(default_factory=Gripper)
    score_kind: str = "unspecified"
    key: SampleKey | None = None

    def __post_init__(self) -> None:
        count = self.translation.shape[0]
        if self.height is None:
            self.height = self.width.new_full((count,), self.gripper.finger_height)
        if self.object_id is None:
            self.object_id = torch.full(
                (count,), -1, dtype=torch.long, device=self.translation.device
            )
        shapes = {
            "translation": (count, 3),
            "rotation": (count, 3, 3),
            "width": (count,),
            "depth": (count,),
            "score": (count,),
            "height": (count,),
            "object_id": (count,),
        }
        for name, expected in shapes.items():
            actual = tuple(getattr(self, name).shape)
            if actual != expected:
                raise ContractError(f"GraspGroup.{name} must have shape {expected}, got {actual}")

    # ---------------------------------------------------------------- container API

    _TENSORS = ("translation", "rotation", "width", "depth", "score", "height", "object_id")

    def __len__(self) -> int:
        return self.translation.shape[0]

    def _with(self, **tensors: Tensor) -> GraspGroup:
        return replace(self, **tensors)

    def __getitem__(self, index: Any) -> GraspGroup:
        if isinstance(index, int):
            index = [index]
        return self._with(**{name: getattr(self, name)[index] for name in self._TENSORS})

    @classmethod
    def empty(cls, *, device: torch.device | str = "cpu", **meta: Any) -> GraspGroup:
        """A valid group with no grasps. An empty prediction is a result, not an error."""
        zeros = torch.zeros(0, device=device)
        return cls(
            translation=torch.zeros(0, 3, device=device),
            rotation=torch.zeros(0, 3, 3, device=device),
            width=zeros,
            depth=zeros.clone(),
            score=zeros.clone(),
            **meta,
        )

    @classmethod
    def cat(cls, groups: Sequence[GraspGroup]) -> GraspGroup:
        """Concatenate groups that share frame, gripper and score semantics."""
        if not groups:
            raise ValueError("cat() needs at least one GraspGroup")
        first = groups[0]
        for other in groups[1:]:
            if (other.frame, other.gripper, other.score_kind) != (
                first.frame,
                first.gripper,
                first.score_kind,
            ):
                raise ContractError(
                    "cannot concatenate grasps with different frame, gripper or score_kind"
                )
        return first._with(
            **{name: torch.cat([getattr(g, name) for g in groups]) for name in cls._TENSORS}
        )

    def to(self, device: torch.device | str) -> GraspGroup:
        return self._with(**{name: getattr(self, name).to(device) for name in self._TENSORS})

    def sort_by_score(self, *, descending: bool = True) -> GraspGroup:
        """Stable sort, so equal scores keep their current order on every platform."""
        order = torch.argsort(self.score, descending=descending, stable=True)
        return self[order]

    # ------------------------------------------------------------------- validation

    def validate(self, *, atol: float = 1e-4) -> GraspGroup:
        """Raise :class:`ContractError` unless every grasp is physically meaningful."""
        problems = []
        for name in ("translation", "rotation", "width", "depth", "score", "height"):
            if not torch.isfinite(getattr(self, name)).all():
                problems.append(f"{name} contains NaN or Inf")
        if len(self) and not problems:
            rotation = self.rotation.double()
            identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
            error = (rotation.transpose(1, 2) @ rotation - identity).abs().amax(dim=(1, 2))
            if (error > atol).any():
                problems.append(
                    f"{int((error > atol).sum())} rotation(s) are not orthonormal "
                    f"(max deviation {float(error.max()):.2e})"
                )
            elif (torch.linalg.det(rotation) < 0).any():
                problems.append("rotation(s) with determinant -1 (a reflection, not a rotation)")
            if (self.width < 0).any():
                problems.append(f"{int((self.width < 0).sum())} negative width(s)")
        if problems:
            where = f" for {self.key}" if self.key else ""
            raise ContractError(f"invalid GraspGroup{where}: " + "; ".join(problems))
        return self

    # --------------------------------------------------------------------- geometry

    def transform(self, matrix: Tensor, *, frame: str) -> GraspGroup:
        """Apply a rigid ``(4, 4)`` transform and relabel the reference frame."""
        matrix = matrix.to(self.translation)
        rot, shift = matrix[:3, :3], matrix[:3, 3]
        out = self._with(translation=self.translation @ rot.T + shift, rotation=rot @ self.rotation)
        out.frame = frame
        return out

    # ------------------------------------------------------- GraspNet 17-D boundary

    def to_graspnet_array(self, dtype: Any = np.float32) -> np.ndarray:
        """Export to the GraspNet ``(G, 17)`` layout. Use only at the I/O boundary."""
        columns = [
            self.score[:, None],
            self.width[:, None],
            self.height[:, None],
            self.depth[:, None],
            self.rotation.reshape(len(self), 9),
            self.translation,
            self.object_id[:, None].to(self.score.dtype),
        ]
        return torch.cat([c.detach().cpu().double() for c in columns], dim=1).numpy().astype(dtype)

    @classmethod
    def from_graspnet_array(cls, array: Any, **meta: Any) -> GraspGroup:
        """Import from the GraspNet ``(G, 17)`` layout. Use only at the I/O boundary."""
        data = torch.as_tensor(np.asarray(array), dtype=torch.float32)
        if data.ndim != 2 or data.shape[1] != GRASPNET_ARRAY_WIDTH:
            raise ContractError(
                f"expected a (G, {GRASPNET_ARRAY_WIDTH}) array, got {tuple(data.shape)}"
            )
        return cls(
            score=data[:, 0],
            width=data[:, 1],
            height=data[:, 2],
            depth=data[:, 3],
            rotation=data[:, 4:13].reshape(-1, 3, 3),
            translation=data[:, 13:16],
            object_id=data[:, 16].long(),
            **meta,
        )
