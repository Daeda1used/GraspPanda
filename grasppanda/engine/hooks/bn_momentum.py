"""Step schedule for the momentum of batch-norm layers."""

from __future__ import annotations

from torch import nn

from grasppanda.engine.errors import EngineError
from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS

from .base import Hook

__all__ = ["BNMomentumHook"]


@HOOKS.register()
class BNMomentumHook(Hook):
    """Decays the momentum of every batch-norm layer in steps of epochs.

    Before epoch ``e`` (0-based) the momentum is set to::

        max(initial * decay ** (max(e - delay, 0) // step), minimum)

    The defaults give ``0.5 * 0.5 ** (e // 2)`` floored at ``0.001``, the schedule of the
    PointNet++-based reference methods. Momentum follows the PyTorch convention: the
    weight of the *new* batch statistic. The value depends on the epoch only, so a
    resumed run continues the schedule without any saved state.

    Args:
        initial: Momentum of the first ``step`` epochs.
        decay: Factor applied every ``step`` epochs.
        step: Epochs between two decays.
        minimum: Floor of the momentum.
        delay: Epochs by which the schedule lags. HGGD's scheduler steps *after* each
            epoch from its initial state, so epoch ``e`` trains with the value of
            ``e - 1``: ``delay: 1``.
        module: Dotted name of the sub-module whose batch-norm layers are scheduled
            (e.g. ``image_net``); ``None`` schedules the whole model. Layers outside it
            keep the momentum they were built with.
    """

    source = "BN momentum schedule of the PointNet++-based methods"

    def __init__(
        self,
        initial: float = 0.5,
        decay: float = 0.5,
        step: int = 2,
        minimum: float = 0.001,
        delay: int = 0,
        module: str | None = None,
    ) -> None:
        if step < 1:
            raise ValueError(f"BNMomentumHook.step must be at least 1, got {step}")
        if delay < 0:
            raise ValueError(f"BNMomentumHook.delay must not be negative, got {delay}")
        self.initial, self.decay, self.step, self.minimum = initial, decay, step, minimum
        self.delay, self.module = delay, module

    def momentum(self, epoch: int) -> float:
        steps = max(epoch - self.delay, 0) // self.step
        return max(self.initial * self.decay**steps, self.minimum)

    def before_epoch(self, state: TrainState) -> None:
        momentum = self.momentum(state.epoch)
        root: nn.Module = state.model
        if self.module is not None:
            try:
                root = state.model.get_submodule(self.module)
            except AttributeError:
                children = ", ".join(name for name, _ in state.model.named_children())
                raise EngineError(
                    f"BNMomentumHook.module is '{self.module}', but "
                    f"{type(state.model).__name__} has no such sub-module (it has: {children})."
                ) from None
        layers = [m for m in root.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for layer in layers:
            layer.momentum = momentum
        changed = state.epoch == 0 or momentum != self.momentum(state.epoch - 1)
        if changed and state.run.is_main:
            state.run.log.info(
                "batch-norm momentum %.6g on %d layers from epoch %d",
                momentum,
                len(layers),
                state.epoch + 1,
            )
