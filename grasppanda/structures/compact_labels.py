"""Compact grasp annotations: one grasp per annotated point and view."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
from torch import Tensor

__all__ = ["COMPACT_LABELS", "CompactGraspLabels"]

#: Key of ``Supervision.extras`` under which compact labels travel: per frame, a list
#: with one :class:`CompactGraspLabels` per object instance.
COMPACT_LABELS = "compact_labels"


@dataclass(eq=False)
class CompactGraspLabels:
    """The best grasp of every (annotated point, view) of one object instance in a frame.

    The dense label set holds a grid of in-plane angles and depths per point and view
    (:class:`~grasppanda.structures.ObjectGraspLabels`). Here that grid is collapsed to
    its best cell, which leaves four small numbers per point and view; this is the
    "economic" supervision of EconomicGrasp. Points without any good grasp are absent.

    Values are stored the way the label asset stores them, as small integers, and are
    turned into metric quantities by :meth:`quality` and :meth:`metres`. The integers
    travel through the data loader; only the rows a batch matches are ever converted.

    Attributes:
        object_id: Dataset id of the object model.
        pose: ``(4, 4)`` object-to-camera transform for this frame.
        points: ``(P, 3)`` annotated grasp points in the object frame.
        view_graspness: ``(P, V)`` target view scores of every point, already
            normalised over its views; the views are in the order of the label lattice.
        angle: ``(P, V)`` in-plane angle bin of the kept grasp.
        depth: ``(P, V)`` depth bin of the kept grasp.
        score: ``(P, V)`` quality level of the kept grasp in ``0 … SCORE_LEVELS``. Zero
            means the view has no usable grasp at this point; otherwise higher is
            better (the label asset derives it from the friction coefficient).
        width: ``(P, V)`` gripper width of the kept grasp in ``1 / WIDTH_UNITS`` metres.
        view_max_friction: Friction coefficient up to which a grasp counted towards
            ``view_graspness`` when the asset was generated.
    """

    #: ``score / SCORE_LEVELS`` is the quality in ``[0, 1]``.
    SCORE_LEVELS: ClassVar[int] = 10
    #: ``width / WIDTH_UNITS`` is the width in metres.
    WIDTH_UNITS: ClassVar[int] = 1000

    object_id: int
    pose: Tensor
    points: Tensor
    view_graspness: Tensor
    angle: Tensor
    depth: Tensor
    score: Tensor
    width: Tensor
    view_max_friction: float

    def __post_init__(self) -> None:
        shape = tuple(self.view_graspness.shape)
        if len(shape) != 2 or shape[0] != len(self.points):
            raise ValueError(
                f"object {self.object_id}: view_graspness must be (P, V) with P = "
                f"{len(self.points)} label points, got {shape}"
            )
        for name in ("angle", "depth", "score", "width"):
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.is_floating_point():
                raise ValueError(
                    f"object {self.object_id}: {name} must be an integer tensor of shape "
                    f"{shape}, got {value.dtype} {tuple(value.shape)}"
                )

    @property
    def num_views(self) -> int:
        return self.view_graspness.shape[1]

    @classmethod
    def quality(cls, score: Tensor) -> Tensor:
        """Stored score levels → quality in ``[0, 1]`` (float32)."""
        return _divide(score, cls.SCORE_LEVELS)

    @classmethod
    def metres(cls, width: Tensor) -> Tensor:
        """Stored widths → metres (float32)."""
        return _divide(width, cls.WIDTH_UNITS)

    def to(self, device: torch.device | str) -> CompactGraspLabels:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return CompactGraspLabels(
            **{k: v.to(device) if isinstance(v, Tensor) else v for k, v in values.items()}
        )


def _divide(stored: Tensor, units: int) -> Tensor:
    """``stored / units`` in float32, correctly rounded on every device.

    The divisor is a tensor on purpose: CUDA divides by a Python number by multiplying
    with its reciprocal, which is off by one unit in the last place for some values.
    """
    return stored.to(torch.float32) / torch.tensor(units, dtype=torch.float32, device=stored.device)
