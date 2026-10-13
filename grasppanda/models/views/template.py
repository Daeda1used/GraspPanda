"""The set of candidate approach views a view stage chooses from.

The template is the single owner of the view grid: the stage that scores views, the
assigner that re-indexes labels to it and anything else that needs "view ``i``" receive
this object instead of re-deriving the lattice (``docs/design/ARCHITECTURE.md`` §8.6).
"""

from __future__ import annotations

from torch import Tensor, nn

from grasppanda.ops import approach_to_rotation, fibonacci_views

__all__ = ["ViewTemplate"]


class ViewTemplate(nn.Module):
    """Viewpoints on the unit sphere and the gripper frame each one implies.

    A *view* is a viewpoint; the gripper approaches from it towards the origin, so the
    approach direction of view ``i`` is ``-views[i]``. ``rotations[i]`` is the gripper
    frame for that approach with zero in-plane rotation (column 0 = approach axis).

    The tensors are buffers, so they follow the owning module across devices; they are
    not saved in checkpoints because they are derived from ``num_views`` alone.
    """

    views: Tensor
    rotations: Tensor

    def __init__(self, views: Tensor) -> None:
        super().__init__()
        if views.ndim != 2 or views.shape[1] != 3:
            raise ValueError(f"views must be (V, 3), got {tuple(views.shape)}")
        self.register_buffer("views", views, persistent=False)
        self.register_buffer("rotations", approach_to_rotation(-views), persistent=False)

    @classmethod
    def fibonacci(cls, num_views: int) -> ViewTemplate:
        """The Fibonacci lattice of the GraspNet-1Billion grasp labels."""
        return cls(fibonacci_views(num_views))

    @property
    def num_views(self) -> int:
        return self.views.shape[0]

    def __len__(self) -> int:
        return self.num_views

    def approach(self, index: Tensor) -> Tensor:
        """``(..., 3)`` unit approach directions of the views ``index``."""
        return -self.views[index]

    def rotation(self, index: Tensor) -> Tensor:
        """``(..., 3, 3)`` zero-angle gripper frames of the views ``index``."""
        return self.rotations[index]

    def extra_repr(self) -> str:
        return f"num_views={self.num_views}"
