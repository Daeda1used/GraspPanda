"""Hooks of the training recipe: weight averages, freezing, schedule-free mode, adaptive clipping.

Each is one line of a config::

    hooks:
      - {type: EMAHook, decay: 0.999}
      - {type: SWAHook, start_epoch: 7}
      - {type: FreezeHook, modules: [backbone], epochs: 2}
      - {type: ScheduleFreeHook}
      - {type: AdaptiveGradClipHook, clip_factor: 0.01}

The averages are ``torch.optim.swa_utils.AveragedModel`` and take the place of
``train.ema``: ``evaluation.weights: ema`` scores them and checkpoints carry them under
``ema``. What the plain train keys already do needs no hook: ``train.grad_clip``
(global norm or value), ``train.accumulate`` (gradient accumulation), ``train.amp``
(mixed precision), ``train.ema`` (a fixed-decay average of the whole state).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Literal

import torch
from torch import Tensor, nn

from grasppanda.engine.errors import EngineError
from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS
from grasppanda.utils.imports import require

from .base import Hook

__all__ = ["AdaptiveGradClipHook", "EMAHook", "FreezeHook", "SWAHook", "ScheduleFreeHook"]


def _submodule(model: nn.Module, name: str, owner: str) -> nn.Module:
    try:
        return model.get_submodule(name)
    except AttributeError:
        children = ", ".join(child for child, _ in model.named_children())
        raise EngineError(
            f"{owner} names the sub-module '{name}', but {type(model).__name__} has no such "
            f"sub-module (it has: {children})."
        ) from None


# ---------------------------------------------------------------------- weight averages


class WeightAverage:
    """A ``torch.optim.swa_utils.AveragedModel`` behind the interface of ``train.ema``.

    The trainer calls :meth:`update` after every optimizer step, the evaluation hook
    scores the model inside :meth:`applied`, and checkpoints store :meth:`state_dict`
    (the keys of the model, so a weights-only load of ``ema`` works).
    """

    def __init__(
        self,
        model: nn.Module,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None,
        *,
        buffers: str,
        interval: int = 1,
    ) -> None:
        if buffers not in ("copy", "average"):
            raise EngineError(f"buffers must be 'copy' or 'average', got '{buffers}'.")
        self.averaged = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=avg_fn, use_buffers=buffers == "average"
        )
        self.interval, self.steps = interval, 0

    @property
    def count(self) -> int:
        return int(self.averaged.n_averaged)

    def average(self, model: nn.Module) -> None:
        self.averaged.update_parameters(model)

    def update(self, model: nn.Module) -> None:
        self.steps += 1
        if self.interval > 0 and self.steps % self.interval == 0:
            self.average(model)

    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.averaged.module.state_dict())

    def load_state_dict(self, state: Mapping[str, Tensor], count: int = 1, steps: int = 0) -> None:
        """Restore the average; ``count`` updates went into it during ``steps`` optimizer steps."""
        self.averaged.module.load_state_dict(state)
        self.averaged.n_averaged.fill_(count)
        self.steps = steps

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Temporarily replace the model's weights by their average (once there is one)."""
        if self.count == 0:
            yield
            return
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.averaged.module.state_dict())
        try:
            yield
        finally:
            model.load_state_dict(backup)


def _install(state: TrainState, average: WeightAverage, owner: str, count: int) -> None:
    """Make ``average`` the weight average of the run; on resume restore it from the checkpoint.

    A checkpoint stores the averaged weights under the keys of the model and nothing
    else, so ``count``, the number of updates that went into them, is derived by the
    caller from the epoch the run resumes at.
    """
    if state.ema is not None:
        raise EngineError(
            f"{owner} and another weight average (train.ema, EMAHook or SWAHook) are both "
            "set: a run keeps one average. Remove one."
        )
    state.ema = average  # type: ignore[assignment]  # same interface as ModelEMA
    source = state.run.resume_from
    if source is None:
        return
    payload = torch.load(source, map_location="cpu", weights_only=True)
    saved = payload.get("ema") if isinstance(payload, Mapping) else None
    if isinstance(saved, Mapping):
        average.load_state_dict(saved, count, _steps(state))
    elif state.run.is_main:
        state.run.log.warning(
            "%s: %s holds no averaged weights; the average starts from the resumed weights",
            owner,
            source,
        )


def _steps(state: TrainState) -> int:
    """Optimizer steps of the epochs completed so far."""
    accumulate = max(state.cfg["train"]["accumulate"], 1)
    return state.epoch * math.ceil(state.epoch_length / accumulate)


@HOOKS.register()
class EMAHook(Hook):
    """Exponential moving average of the weights, with warm-up of the decay.

    After every ``interval`` optimizer steps ``average = d · average + (1 - d) · weights``
    with ``d = min(decay, (1 + n) / (10 + n))`` for the ``n``-th update when ``warmup`` is
    on (the rule of TensorFlow and of ``timm.utils.ModelEmaV3``), so a short run is not
    dominated by the initial weights as it is with ``train.ema``. ``evaluation.weights:
    ema`` scores the average, and checkpoints store it under ``ema``.

    Args:
        decay: Decay ``d`` once warmed up; 0.999 averages over about 1000 updates.
        warmup: Ramp the decay up from 0.1.
        interval: Optimizer steps between two updates.
        buffers: ``copy`` keeps the buffers (batch-norm statistics, learned anchors) equal
            to those of the trained model; ``average`` averages floating-point buffers
            like the weights.
    """

    priority = 20
    status = "tested"
    source = "Tarvainen and Valpola, NeurIPS 2017 (arXiv 1703.01780), via torch.optim.swa_utils"

    def __init__(
        self,
        decay: float = 0.999,
        warmup: bool = True,
        interval: int = 1,
        buffers: Literal["copy", "average"] = "copy",
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMAHook.decay must be in (0, 1), got {decay}")
        if interval < 1:
            raise ValueError(f"EMAHook.interval must be at least 1, got {interval}")
        self.decay, self.warmup, self.interval, self.buffers = decay, warmup, interval, buffers

    def factor(self, count: int) -> float:
        """Decay of the update after ``count`` earlier ones."""
        return min(self.decay, (1.0 + count) / (10.0 + count)) if self.warmup else self.decay

    def before_run(self, state: TrainState) -> None:
        def avg_fn(average: Tensor, new: Tensor, count: Tensor | int) -> Tensor:
            if not average.is_floating_point():
                return new
            decay = self.factor(int(count))
            return decay * average + (1.0 - decay) * new

        average = WeightAverage(state.model, avg_fn, buffers=self.buffers, interval=self.interval)
        _install(state, average, "EMAHook", _steps(state) // self.interval)


@HOOKS.register()
class SWAHook(Hook):
    """Stochastic weight averaging: the equal-weight mean of the weights after late epochs.

    From ``start_epoch`` on, the weights at the end of every epoch enter a running mean.
    ``evaluation.weights: ema`` scores the mean and checkpoints store it under ``ema``.
    Batch-norm statistics are those of the last epoch (``buffers: copy``) or their mean
    (``average``); they are not re-estimated for the averaged weights, which the paper
    does with one more pass over the data.

    Args:
        start_epoch: Completed epochs before averaging starts (7 of 10: the last three
            epochs are averaged). Keep the learning rate constant or cyclic from there.
        buffers: ``copy`` or ``average``, as for ``EMAHook``.
    """

    priority = 20
    status = "tested"
    source = "Izmailov et al., UAI 2018 (arXiv 1803.05407), via torch.optim.swa_utils"

    def __init__(self, start_epoch: int, buffers: Literal["copy", "average"] = "copy") -> None:
        if start_epoch < 0:
            raise ValueError(f"SWAHook.start_epoch must not be negative, got {start_epoch}")
        self.start_epoch, self.buffers = start_epoch, buffers
        self.average: WeightAverage | None = None

    def before_run(self, state: TrainState) -> None:
        if self.start_epoch >= state.max_epochs:
            raise EngineError(
                f"SWAHook.start_epoch is {self.start_epoch}, but the run has "
                f"{state.max_epochs} epochs: nothing would be averaged."
            )
        self.average = WeightAverage(state.model, None, buffers=self.buffers, interval=0)
        _install(state, self.average, "SWAHook", max(state.epoch - self.start_epoch, 0))

    def after_epoch(self, state: TrainState) -> None:
        if self.average is not None and state.epoch > self.start_epoch:
            self.average.average(state.model)


# ------------------------------------------------------------------------------ freezing


@HOOKS.register()
class FreezeHook(Hook):
    """Freeze sub-modules for the first epochs (or for the whole run), then release them.

    While frozen, the parameters of the modules take no gradient, so no optimizer moves
    them (an optimizer skips a parameter without a gradient, weight decay included), and
    with ``eval_mode`` their batch-norm layers keep their statistics and drop-out is off.
    From epoch ``epochs`` on they train like the rest. This is how a pretrained encoder
    is protected while a freshly initialised head settles.

    Args:
        modules: Dotted names of the sub-modules (``backbone``, ``backbone.enc``).
        epochs: Number of epochs they stay frozen; ``None`` for the whole run.
        eval_mode: Put the frozen modules in evaluation mode.
    """

    priority = 30
    status = "tested"
    source = "Howard and Ruder, ACL 2018 (arXiv 1801.06146): gradual unfreezing"

    def __init__(
        self, modules: list[str], epochs: int | None = None, eval_mode: bool = True
    ) -> None:
        if not modules:
            raise ValueError("FreezeHook.modules is empty: name at least one sub-module")
        if epochs is not None and epochs < 0:
            raise ValueError(f"FreezeHook.epochs must not be negative, got {epochs}")
        self.modules, self.epochs, self.eval_mode = list(modules), epochs, eval_mode
        self.frozen: bool | None = None

    def before_run(self, state: TrainState) -> None:
        for name in self.modules:
            _submodule(state.model, name, "FreezeHook.modules")

    def before_epoch(self, state: TrainState) -> None:
        frozen = self.epochs is None or state.epoch < self.epochs
        count = 0
        for name in self.modules:
            module = _submodule(state.model, name, "FreezeHook.modules")
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)
                count += parameter.numel()
            if frozen and self.eval_mode:
                module.eval()  # the trainer has just called model.train()
        if frozen != self.frozen and state.run.is_main:
            state.run.log.info(
                "%s %s (%d parameters) from epoch %d",
                "froze" if frozen else "released",
                ", ".join(self.modules),
                count,
                state.epoch + 1,
            )
        self.frozen = frozen


# --------------------------------------------------------------------- optimizer helpers


@HOOKS.register()
class ScheduleFreeHook(Hook):
    """Switch a schedule-free optimizer between its training point and its average.

    A schedule-free optimizer (``ScheduleFree``, or ``PytorchOptimizer`` with a
    ``schedulefree*`` name) holds the averaged iterate only after ``optimizer.eval()``.
    This hook calls ``train()`` before every epoch and ``eval()`` after it, before the
    in-training evaluation and the checkpoint, so both see the averaged weights.
    Batch-norm statistics are those collected at the training point.
    """

    priority = 10
    status = "tested"
    source = "Defazio et al., The Road Less Scheduled (arXiv 2405.15682), via schedulefree"

    def _switch(self, state: TrainState, mode: str) -> None:
        for optimizer in state.optimizers.values():
            getattr(optimizer, mode)()

    def before_run(self, state: TrainState) -> None:
        plain = [
            f"{name} ({type(optimizer).__name__})"
            for name, optimizer in state.optimizers.items()
            if not (hasattr(optimizer, "train") and hasattr(optimizer, "eval"))
        ]
        if plain:
            raise EngineError(
                f"ScheduleFreeHook: the optimizer {', '.join(plain)} has no train()/eval() "
                "modes, so it is not schedule-free. Use `optimizer: {type: ScheduleFree, ...}` "
                "or remove the hook."
            )

    def before_epoch(self, state: TrainState) -> None:
        self._switch(state, "train")

    def after_epoch(self, state: TrainState) -> None:
        self._switch(state, "eval")


@HOOKS.register()
class AdaptiveGradClipHook(Hook):
    """Adaptive gradient clipping: bound each unit's gradient by a fraction of its weight norm.

    Before every optimizer step the gradient of each output unit is rescaled so that its
    norm is at most ``clip_factor`` times the norm of the unit's weights. Unlike
    ``train.grad_clip`` the bound follows the scale of each layer. It replaces
    ``train.grad_clip``: set only one of the two.

    Args:
        clip_factor: Largest ratio of gradient norm to weight norm (the paper: 0.01 to
            0.16, smaller for larger batches).
        eps: Floor of the weight norm, so that zero-initialised weights can move.
        exclude: Dotted names of sub-modules whose parameters are not clipped (the paper
            leaves the final linear layer out).
    """

    priority = 40
    extras = ("optim",)
    status = "bridge"
    source = "Brock et al. (arXiv 2102.06171), via timm.utils.adaptive_clip_grad"

    def __init__(
        self, clip_factor: float = 0.01, eps: float = 1e-3, exclude: list[str] | None = None
    ) -> None:
        if clip_factor <= 0:
            raise ValueError(
                f"AdaptiveGradClipHook.clip_factor must be positive, got {clip_factor}"
            )
        self.clip_factor, self.eps, self.exclude = clip_factor, eps, list(exclude or [])
        self.parameters: list[nn.Parameter] = []
        self.clip: Callable[..., None] | None = None

    def before_run(self, state: TrainState) -> None:
        if state.cfg["train"]["grad_clip"] is not None:
            raise EngineError(
                "AdaptiveGradClipHook and train.grad_clip are both set: choose one way of "
                "clipping (train.grad_clip=null keeps the hook)."
            )
        utils = require("timm.utils", extra="optim", purpose="AdaptiveGradClipHook")
        self.clip = utils.adaptive_clip_grad
        skipped = {
            id(parameter)
            for name in self.exclude
            for parameter in _submodule(
                state.model, name, "AdaptiveGradClipHook.exclude"
            ).parameters()
        }
        self.parameters = [p for p in state.model.parameters() if id(p) not in skipped]

    def before_step(self, state: TrainState) -> None:
        if self.clip is None:
            return
        for optimizer in state.optimizers.values():
            state.scaler.unscale_(optimizer)  # clip real gradients, not scaled ones
        self.clip(self.parameters, clip_factor=self.clip_factor, eps=self.eps)
