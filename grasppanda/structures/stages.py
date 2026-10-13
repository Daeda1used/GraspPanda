"""Outputs passed between the stages of a detector."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from .common import ContractError
from .points import PointSet

__all__ = ["ApproachSet", "GroupFeatures", "SeedSet"]


@dataclass(eq=False)
class SeedSet:
    """Candidate grasp locations.

    Attributes:
        points: The seeds, usually a child of the scene point set.
        feats: ``(M, C)`` features, if the stage produced any.
        scores: Named per-seed scores with declared meaning, e.g. ``graspness`` or
            ``objectness``. No score is mandatory: a method without graspness leaves it out.
        valid: ``(M,)`` mask. Padding added to reach a fixed seed count is marked
            invalid rather than passed on as a real candidate.
        source: How the seeds were chosen, for records and debugging.
    """

    points: PointSet
    feats: Tensor | None = None
    scores: dict[str, Tensor] = field(default_factory=dict)
    valid: Tensor | None = None
    source: str = ""

    def __post_init__(self) -> None:
        count = len(self.points)
        named = {"feats": self.feats, "valid": self.valid, **self.scores}
        for name, value in named.items():
            if value is not None and value.shape[0] != count:
                raise ContractError(f"SeedSet.{name} has {value.shape[0]} rows for {count} seeds")

    def __len__(self) -> int:
        return len(self.points)


@dataclass(eq=False)
class ApproachSet:
    """One approach direction per seed.

    Attributes:
        direction: ``(M, 3)`` unit vectors pointing from the gripper towards the object.
        rotation: ``(M, 3, 3)`` crop frame whose first column is the approach axis.
        view_index: ``(M,)`` index into a view template, when directions are discrete.
        logits: Raw view scores, kept for the loss.
    """

    direction: Tensor
    rotation: Tensor | None = None
    view_index: Tensor | None = None
    logits: Tensor | None = None

    def __post_init__(self) -> None:
        if self.direction.ndim != 2 or self.direction.shape[-1] != 3:
            raise ContractError(
                f"ApproachSet.direction must be (M, 3), got {tuple(self.direction.shape)}"
            )
        count = self.direction.shape[0]
        if self.rotation is not None and tuple(self.rotation.shape) != (count, 3, 3):
            raise ContractError(
                f"ApproachSet.rotation must be ({count}, 3, 3), got {tuple(self.rotation.shape)}"
            )

    def __len__(self) -> int:
        return self.direction.shape[0]


@dataclass(eq=False)
class GroupFeatures:
    """Local features per seed, with named axes.

    ``axes`` names every dimension of ``feats``, e.g. ``("seed", "channel")`` or
    ``("seed", "depth", "channel")``. A head states the axes it expects, so a depth
    axis is never squeezed or invented implicitly.
    """

    feats: Tensor
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.feats.ndim != len(self.axes) or len(set(self.axes)) != len(self.axes):
            raise ContractError(
                f"GroupFeatures.axes {self.axes} must name each of the {self.feats.ndim} "
                f"dimensions of feats {tuple(self.feats.shape)} exactly once"
            )

    def size(self, axis: str) -> int:
        return self.feats.shape[self.axes.index(axis)]

    def expect(self, axes: tuple[str, ...], *, user: str) -> None:
        """Assert the axis layout a consumer was written for."""
        if self.axes != tuple(axes):
            raise ContractError(f"{user} expects features with axes {tuple(axes)}, got {self.axes}")
