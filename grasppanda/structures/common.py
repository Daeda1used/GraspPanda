"""Small value types shared by all structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch import Tensor

__all__ = ["ContractError", "Gripper", "Mask", "Provenance", "SampleKey"]


class ContractError(ValueError):
    """Data handed between stages violates a declared contract."""


@dataclass(frozen=True)
class SampleKey:
    """Identity of one evaluation unit."""

    dataset: str
    scene: int
    camera: str
    frame: int

    def __str__(self) -> str:
        return f"{self.dataset}/scene_{self.scene:04d}/{self.camera}/{self.frame:04d}"


class Provenance(str, Enum):
    """Where a piece of inference-time input came from."""

    SENSOR = "sensor"
    CALIBRATION = "calibration"
    PREDICTED = "predicted"
    GROUND_TRUTH = "ground_truth"


@dataclass(eq=False)
class Mask:
    """A boolean mask together with its provenance."""

    data: Tensor
    provenance: Provenance


@dataclass(frozen=True)
class Gripper:
    """Parallel-jaw gripper geometry in metres.

    Defaults are the GraspNet-1Billion benchmark gripper, the same constants its
    evaluator uses for collision checking.
    """

    name: str = "graspnet-parallel-jaw"
    max_width: float = 0.1
    finger_width: float = 0.01
    finger_height: float = 0.02
    depth_base: float = 0.02
