"""Loggers: text (console and ``log.txt``), TensorBoard and Weights & Biases.

All three report the same scalars every ``interval`` iterations: the running mean of
each loss term and metric, the learning rate, data and iteration time, peak GPU memory
and the estimated time left. TensorBoard and W&B are optional extras, imported only
when their logger is built; a missing package fails before training starts, with the
install command.
"""

from __future__ import annotations

import datetime
from typing import Any, Literal

import torch

from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS
from grasppanda.utils.imports import require

from .base import Hook

__all__ = ["LOGGER_HOOKS", "TensorBoardLoggerHook", "TextLoggerHook", "WandbLoggerHook", "scalars"]


def scalars(state: TrainState) -> dict[str, float]:
    """The numbers every logger reports for the current iteration."""
    values = state.meters.means()
    values.update(state.learning_rates())
    if state.device.type == "cuda":
        values["memory_mb"] = torch.cuda.max_memory_allocated(state.device) / 2**20
    if "time/iter" in values:
        remaining = (state.max_epochs - state.epoch) * state.epoch_length - state.inner
        values["eta_seconds"] = max(remaining, 0) * values["time/iter"]
    return values


def _format(name: str, value: float) -> str:
    if name == "eta_seconds":
        return f"eta {datetime.timedelta(seconds=int(value))}"
    if name == "memory_mb":
        return f"mem {value:.0f}M"
    if name.startswith("lr"):
        return f"{name} {value:.3e}"
    if name.startswith("time/"):
        return f"{name.removeprefix('time/')} {value:.3f}s"
    return f"{name} {value:.4f}"


def format_line(state: TrainState, values: dict[str, float]) -> str:
    """One line: position, total loss, loss terms, metrics, then rates and timings."""
    tail = [
        *(n for n in values if n.startswith("lr")),
        "grad_norm",
        "time/data",
        "time/iter",
        *(n for n in values if n.startswith("time/forward.")),
        "memory_mb",
        "eta_seconds",
    ]
    head = ["loss", *sorted(n for n in values if n.startswith("loss/"))]
    middle = [n for n in values if n not in head and n not in tail]
    width = len(str(state.epoch_length))
    parts = [
        f"epoch {state.epoch + 1}/{state.max_epochs}",
        f"iter {state.inner:>{width}}/{state.epoch_length}",
    ]
    parts += [_format(name, values[name]) for name in (*head, *middle, *tail) if name in values]
    return "  ".join(parts)


class _IntervalLogger(Hook):
    """Shared timing: every ``interval`` iterations and at the last one of each epoch."""

    priority = 90

    def __init__(self, interval: int = 50) -> None:
        self.interval = max(interval, 1)

    def _due(self, state: TrainState) -> bool:
        return state.run.is_main and (
            state.inner % self.interval == 0 or state.inner == state.epoch_length
        )


@HOOKS.register()
class TextLoggerHook(_IntervalLogger):
    """Prints one line per interval to the console and to ``log.txt``."""

    source = "GraspPanda"

    def after_iter(self, state: TrainState) -> None:
        if self._due(state):
            state.run.log.info(format_line(state, scalars(state)))


@HOOKS.register()
class TensorBoardLoggerHook(_IntervalLogger):
    """Writes scalars to ``<run dir>/tensorboard``: ``train/*`` by iteration, ``eval/*`` by epoch."""

    source = "GraspPanda"
    extras = ("tensorboard",)

    def __init__(self, interval: int = 50) -> None:
        super().__init__(interval)
        self.writer: Any = None

    def before_run(self, state: TrainState) -> None:
        if not state.run.is_main:
            return
        tensorboard = require(
            "torch.utils.tensorboard", extra="tensorboard", purpose="The TensorBoard logger"
        )
        self.writer = tensorboard.SummaryWriter(log_dir=str(state.run.dir / "tensorboard"))

    def after_iter(self, state: TrainState) -> None:
        if self.writer is not None and self._due(state):
            for name, value in scalars(state).items():
                self.writer.add_scalar(f"train/{name}", value, state.iteration)

    def after_epoch(self, state: TrainState) -> None:
        if self.writer is None:
            return
        if state.metrics_epoch == state.epoch:
            for name, value in state.metrics.items():
                self.writer.add_scalar(f"eval/{name}", value, state.epoch)
        self.writer.flush()

    def after_run(self, state: TrainState) -> None:
        if self.writer is not None:
            self.writer.close()


@HOOKS.register()
class WandbLoggerHook(_IntervalLogger):
    """Logs scalars to Weights & Biases; the run is named after the run directory.

    Args:
        interval: Iterations between log calls.
        project: W&B project.
        entity: W&B team or user; ``None`` uses the account default.
        mode: ``online``, ``offline`` (sync later) or ``disabled``.
    """

    source = "GraspPanda"
    extras = ("wandb",)

    def __init__(
        self,
        interval: int = 50,
        project: str = "grasppanda",
        entity: str | None = None,
        mode: Literal["online", "offline", "disabled"] = "online",
    ) -> None:
        super().__init__(interval)
        self.project, self.entity, self.mode = project, entity, mode
        self.wandb: Any = None

    def before_run(self, state: TrainState) -> None:
        if not state.run.is_main:
            return
        self.wandb = require("wandb", extra="wandb", purpose="The Weights & Biases logger")
        self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=state.run.id,
            id=state.run.id,  # resuming the run directory resumes the same W&B run
            resume="allow",
            dir=str(state.run.dir),
            config=state.cfg,
            mode=self.mode,
        )

    def after_iter(self, state: TrainState) -> None:
        if self.wandb is not None and self._due(state):
            values = {f"train/{name}": value for name, value in scalars(state).items()}
            self.wandb.log(values, step=state.iteration)

    def after_epoch(self, state: TrainState) -> None:
        if self.wandb is not None and state.metrics_epoch == state.epoch:
            values = {f"eval/{name}": value for name, value in state.metrics.items()}
            self.wandb.log({**values, "epoch": state.epoch}, step=state.iteration)

    def after_run(self, state: TrainState) -> None:
        if self.wandb is not None:
            self.wandb.finish()


#: ``runtime.log.backends`` entry → hook class.
LOGGER_HOOKS: dict[str, type[_IntervalLogger]] = {
    "text": TextLoggerHook,
    "tensorboard": TensorBoardLoggerHook,
    "wandb": WandbLoggerHook,
}
