"""Tells a detector where training stands."""

from __future__ import annotations

from typing import ClassVar

from grasppanda.engine.errors import EngineError
from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS

from .base import Hook

__all__ = ["EpochSignalHook"]


@HOOKS.register()
class EpochSignalHook(Hook):
    """Calls ``model.on_train_epoch(epoch, max_epochs)`` before every epoch.

    ``loss(observation, supervision)`` does not know the epoch. A detector whose
    training behaviour depends on it defines ``on_train_epoch`` and asks for this hook
    in its config. HGGD does: its rotation anchors follow the labels during the first
    epochs only, and its label pool starts empty in each epoch.

    ``epoch`` is 0-based. The signal depends on the epoch alone, so a resumed run gets
    the right one without any saved state. The hook fails when the model does not
    listen: a schedule that silently never happens is worse than an error.
    """

    source = "GraspPanda"

    #: Before the hooks that act on an epoch with the model already informed.
    priority: ClassVar[int] = 20

    def before_epoch(self, state: TrainState) -> None:
        listener = getattr(state.model, "on_train_epoch", None)
        if not callable(listener):
            raise EngineError(
                f"EpochSignalHook is configured, but {type(state.model).__name__} defines no "
                "on_train_epoch(epoch, max_epochs). Remove the hook from `hooks`."
            )
        listener(state.epoch, state.max_epochs)
