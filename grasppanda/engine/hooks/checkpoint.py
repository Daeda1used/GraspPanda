"""Periodic checkpoints, a bounded history and the best checkpoint by a monitored metric."""

from __future__ import annotations

from grasppanda.engine import distributed as dist
from grasppanda.engine.checkpoint import (
    checkpoint_name,
    link_or_copy,
    prune_checkpoints,
    save_checkpoint,
)
from grasppanda.engine.errors import EngineError
from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS

from .base import Hook

__all__ = ["CheckpointHook"]


@HOOKS.register()
class CheckpointHook(Hook):
    """Writes ``checkpoints/epoch_NNNN.pth`` and ``checkpoints/best.pth``.

    Built by default from ``runtime.checkpoint`` and ``evaluation.save_best``.

    Args:
        interval: Save every ``interval`` epochs, and always after the last one. ``0``
            saves after the last epoch only.
        keep_last: Keep this many of the newest epoch checkpoints; ``0`` keeps all.
        save_best: ``<variant>/<metric>`` of the in-training evaluation to monitor; the
            checkpoint of the epoch with the highest value is kept as ``best.pth``.
            A partial evaluation has no such metric and never becomes the best.
    """

    source = "GraspPanda"

    priority = 80

    def __init__(self, interval: int = 1, keep_last: int = 3, save_best: str | None = None) -> None:
        self.interval, self.keep_last, self.save_best = interval, keep_last, save_best

    def _is_best(self, state: TrainState) -> bool:
        if self.save_best is None or state.metrics_epoch != state.epoch:
            return False
        if self.save_best not in state.metrics:
            variant = self.save_best.partition("/")[0]
            if not any(name.startswith(f"{variant}/") for name in state.metrics):
                state.run.log.warning(
                    "evaluation of variant '%s' is partial at epoch %d: no '%s' to compare, "
                    "best checkpoint unchanged",
                    variant,
                    state.epoch,
                    self.save_best,
                )
                return False
            raise EngineError(
                f"evaluation.save_best monitors '{self.save_best}', but the evaluation "
                f"reported: {', '.join(sorted(state.metrics))}."
            )
        value = float(state.metrics[self.save_best])
        if state.best is not None and value <= state.best["value"]:
            return False
        state.best = {"metric": self.save_best, "value": value, "epoch": state.epoch}
        return True

    def after_epoch(self, state: TrainState) -> None:
        directory = state.run.checkpoints
        periodic = (self.interval > 0 and state.epoch % self.interval == 0) or state.is_last_epoch
        best = self._is_best(state)  # before saving: the checkpoint records the new best
        path = directory / checkpoint_name(state.epoch)
        if periodic:
            save_checkpoint(state, path)
        elif best:
            save_checkpoint(state, directory / "best.pth")
        if not dist.is_main():
            return
        if best:
            if periodic:
                link_or_copy(path, directory / "best.pth")
            state.run.log.info(
                "best checkpoint: %s = %.4f at epoch %d",
                self.save_best,
                state.best["value"],
                state.epoch,
            )
        prune_checkpoints(directory, self.keep_last)
