"""The single owner of every number the GraspNet-style evaluators use.

A benchmark protocol is a set of constants plus an order of operations. The order
lives in :mod:`.frame`; the constants live here, once. Nothing else in the evaluator
types a threshold: components receive a :class:`Protocol` (or one value taken from
it), so two parts of the pipeline cannot disagree about, say, the finger width.

Changing any value below produces a different benchmark, which is why none of them is
a constructor argument of the registered evaluators. A new benchmark gets a new
:class:`Protocol` instance under a new identifier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from grasppanda.structures.common import Gripper

__all__ = ["GRASPNET1B_SV_V1", "Protocol", "TableSlab"]


@dataclass(frozen=True)
class TableSlab:
    """Synthetic table added to the collision scene, in the table frame, in metres.

    Points lie on a regular lattice: along each axis ``floor(size / grid_size)`` samples
    spread evenly over ``[offset, offset + size]``, both ends included.
    """

    size: tuple[float, float, float] = (1.0, 1.0, 0.05)
    offset: tuple[float, float, float] = (-0.5, -0.5, -0.05)
    grid_size: float = 0.008


@dataclass(frozen=True)
class Protocol:
    """Constants of one pinned evaluation protocol.

    Attributes:
        name: Protocol identifier stamped on every result.
        gripper: Gripper used for the collision and emptiness test. Its ``max_width``
            is the upper clip of predicted widths.
        top_k: Number of ranks in the precision table.
        frictions: Friction coefficients of the table columns, ascending.
        reported: Extra metrics, each the mean of one friction column.
        nms_translation: A kept grasp suppresses lower-scored grasps closer than this
            (metres, strict) that are also within ``nms_rotation_deg``.
        nms_rotation_deg: Rotation threshold of the suppression, degrees, strict.
        model_voxel_size: Voxel edge used to down-sample object models, metres.
        per_object_top: Grasps kept per object before the global cut.
        crop_margin: Margin added to the assigned object's bounding box when cropping
            the collision scene, metres.
        empty_threshold: Fewer points than this between the fingers means "empty".
        axis_tolerance: Largest deviation from unit norm of a closing axis that the
            force-closure stage accepts.
        normalize_closing_axis: Rescale the closing axis to unit length before the
            force-closure stage instead of rejecting axes outside ``axis_tolerance``.
        table: Synthetic table of the collision scene, or ``None`` for no table.
        pre_nms_max_distance: When set, grasps further than this from every model
            point are dropped before suppression (metres, strict).
        frames_per_scene: Frames of one camera in one scene.
    """

    name: str
    gripper: Gripper = field(default_factory=Gripper)
    top_k: int = 50
    frictions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    reported: tuple[tuple[str, float], ...] = (("AP0.8", 0.8), ("AP0.4", 0.4))
    nms_translation: float = 0.03
    nms_rotation_deg: float = 30.0
    model_voxel_size: float = 0.008
    per_object_top: int = 10
    crop_margin: float = 0.05
    empty_threshold: int = 10
    axis_tolerance: float = 1e-5
    normalize_closing_axis: bool = False
    table: TableSlab | None = field(default_factory=TableSlab)
    pre_nms_max_distance: float | None = None
    frames_per_scene: int = 256

    def __post_init__(self) -> None:
        if list(self.frictions) != sorted(set(self.frictions)):
            raise ValueError(f"frictions must be strictly ascending, got {self.frictions}")
        for metric, friction in self.reported:
            if friction not in self.frictions:
                raise ValueError(
                    f"metric {metric} needs friction {friction}, which is not a column of "
                    f"the precision table {self.frictions}"
                )

    @property
    def nms_rotation(self) -> float:
        """Rotation threshold in radians, computed the way the reference tool does."""
        return self.nms_rotation_deg / 180 * math.pi

    def friction_column(self, friction: float) -> int:
        """Column of the precision table that belongs to ``friction``."""
        return self.frictions.index(friction)

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable view for result records."""
        return asdict(self)


#: GraspNet-1Billion, single view: graspnetAPI evaluation path as of v1.2.10 / v1.2.11.
GRASPNET1B_SV_V1 = Protocol(name="graspnet1b-sv-v1")
