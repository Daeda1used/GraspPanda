"""Base class of training hooks.

The training loop does the arithmetic of one step and nothing else; whatever is optional
(logging, checkpoints, evaluation, schedules of anything but the learning rate) is a hook.
A hook is a registered component: annotate every constructor parameter and add it to a
run with ``hooks: [{type: MyHook, ...}]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from grasppanda.registry import HOOKS

if TYPE_CHECKING:
    from grasppanda.engine.state import TrainState

__all__ = ["Hook"]


@HOOKS.base
class Hook:
    """Callbacks around the training loop; every method is optional.

    Under ``torchrun`` hooks run on every process. A hook that writes or prints checks
    ``state.run.is_main`` itself.
    """

    #: Hooks run in ascending priority; equal priorities keep their config order.
    #: Evaluation (70) precedes checkpointing (80), which precedes the loggers (90).
    priority: ClassVar[int] = 50

    def __init__(self) -> None:
        # Declared so that a hook without settings can be registered: the inherited
        # ``object.__init__`` takes ``**kwargs``, which the registry refuses.
        pass

    def before_run(self, state: TrainState) -> None:
        """Once, after the state is built (and restored, when resuming)."""

    def before_epoch(self, state: TrainState) -> None:
        """``state.epoch`` is the 0-based index of the epoch about to run."""

    def after_iter(self, state: TrainState) -> None:
        """After every batch; ``state.outputs`` holds its loss terms and metrics."""

    def after_epoch(self, state: TrainState) -> None:
        """``state.epoch`` already counts the epoch that just ended."""

    def after_run(self, state: TrainState) -> None:
        """Once, after the last epoch."""
