"""Where a training step spends its time, per stage of the detector."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS

from .base import Hook

__all__ = ["StageTimerHook"]


@HOOKS.register()
class StageTimerHook(Hook):
    """Times the forward pass of each stage (direct child module) of the detector.

    Seconds per iteration appear in the logs as ``time/forward.<stage>`` next to
    ``time/data`` and ``time/iter``; a stage called several times in one iteration is
    summed. Work the detector does outside its child modules is not attributed. Without
    ``sync`` the numbers on a GPU show where work is *launched*, not where it is done,
    because CUDA runs asynchronously. ``sync`` waits for the device around every stage,
    which makes the numbers honest and the training slower: a diagnostic, not a default.

    Args:
        sync: Synchronise CUDA before and after each stage.
        depth: 1 times the detector's children, 2 their children as well
            (``backbone.encoder``), and so on.
    """

    source = "GraspPanda"

    def __init__(self, sync: bool = False, depth: int = 1) -> None:
        if depth < 1:
            raise ValueError(f"StageTimerHook.depth must be at least 1, got {depth}")
        self.sync, self.depth = sync, depth
        self._started: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}
        self._handles: list[Any] = []

    def _clock(self, device: torch.device) -> float:
        if self.sync and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _stages(self, module: nn.Module, prefix: str = "", level: int = 1):
        for name, child in module.named_children():
            yield prefix + name, child
            if level < self.depth:
                yield from self._stages(child, f"{prefix}{name}.", level + 1)

    def before_run(self, state: TrainState) -> None:
        device = state.device
        for name, stage in self._stages(state.model):

            def start(module: nn.Module, args: Any, name: str = name) -> None:
                self._started[name] = self._clock(device)

            def stop(module: nn.Module, args: Any, output: Any, name: str = name) -> None:
                elapsed = self._clock(device) - self._started.pop(name)
                self._elapsed[name] = self._elapsed.get(name, 0.0) + elapsed

            self._handles.append(stage.register_forward_pre_hook(start))
            self._handles.append(stage.register_forward_hook(stop))

    def before_epoch(self, state: TrainState) -> None:
        self._elapsed.clear()  # drop what an evaluation between two epochs accumulated

    def after_iter(self, state: TrainState) -> None:
        state.meters.update(
            {f"time/forward.{name}": value for name, value in self._elapsed.items()}
        )
        self._elapsed.clear()

    def after_run(self, state: TrainState) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
