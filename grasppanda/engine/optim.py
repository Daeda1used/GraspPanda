"""Optimizers, schedulers, gradient clipping, the weight average and the AMP scaler.

The trainer never assumes a single optimizer. By default there is one, over every
trainable parameter. A detector that needs more defines::

    def configure_optimizers(self, optimizer: dict) -> ...

and returns any of

* an iterable of parameters or of parameter groups (``{"params": ..., "lr": ...}``):
  one optimizer of the configured type over them;
* a mapping ``name -> parameters | parameter groups | torch.optim.Optimizer``: one
  optimizer per entry. Entries that are not optimizers yet are built from the
  ``optimizer`` node of the config, which is passed in so that learning rates can be
  derived from it instead of being repeated in code.

Every optimizer steps on the same total loss, and each gets its own scheduler built from
the ``scheduler`` node.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from grasppanda.registry import OPTIMIZERS, SCHEDULERS

from .errors import EngineError

__all__ = [
    "ModelEMA",
    "build_optimizers",
    "build_schedulers",
    "clip_gradients",
    "make_scaler",
    "scheduler_steps",
]

#: Schedulers whose shape depends on the length of the run, and the argument that says
#: it. The engine fills it in when the config leaves it out.
TOTAL_STEPS_ARGUMENT = {
    "OneCycleLR": "total_steps",
    "CosineAnnealingLR": "T_max",
    "PolynomialLR": "total_iters",
}
_UNSUPPORTED = {
    "ReduceLROnPlateau": "it steps on a monitored metric, which the trainer does not provide",
    "LambdaLR": "it takes a Python function, which a config cannot express",
    "MultiplicativeLR": "it takes a Python function, which a config cannot express",
    "SequentialLR": "it takes scheduler objects, which a config cannot express",
    "ChainedScheduler": "it takes scheduler objects, which a config cannot express",
}


# ----------------------------------------------------------------------- optimizers


def _parameters(spec: Any, name: str) -> list[Any]:
    params = list(spec)
    if not params:
        raise EngineError(f"optimizer '{name}' has no parameters to optimise.")
    return params


def build_optimizers(model: nn.Module, node: Mapping[str, Any]) -> dict[str, Optimizer]:
    """Build the optimizer(s) of ``model`` from the ``optimizer`` node of a config."""
    configure = getattr(model, "configure_optimizers", None)
    if callable(configure):
        spec = configure(dict(node))
    else:
        spec = [param for param in model.parameters() if param.requires_grad]
    if isinstance(spec, Optimizer):
        return {"main": spec}
    if not isinstance(spec, Mapping):
        spec = {"main": spec}
    if not spec:
        raise EngineError("configure_optimizers() returned no optimizer.")
    return {
        name: value
        if isinstance(value, Optimizer)
        else OPTIMIZERS.build(node, params=_parameters(value, name))
        for name, value in spec.items()
    }


# ----------------------------------------------------------------------- schedulers


def scheduler_steps(unit: str, epochs: int, epoch_length: int, accumulate: int) -> int:
    """How often the scheduler steps over a whole run."""
    if unit == "epoch":
        return epochs
    return epochs * math.ceil(epoch_length / accumulate)


def build_schedulers(
    node: Mapping[str, Any] | None, optimizers: Mapping[str, Optimizer], *, total_steps: int
) -> dict[str, LRScheduler]:
    """Build one scheduler per optimizer from the ``scheduler`` node (``None``: constant rate).

    ``total_steps`` is the number of times the trainer will call ``step()``. Schedulers
    that need the length of the run receive it unless the config states it.
    """
    if node is None:
        return {}
    node = dict(node)
    kind = node.get("type")
    if kind in _UNSUPPORTED:
        raise EngineError(f"scheduler {kind} is not supported: {_UNSUPPORTED[kind]}.")
    argument = TOTAL_STEPS_ARGUMENT.get(kind, "")
    per_epoch = node.get("epochs") is not None and node.get("steps_per_epoch") is not None
    if argument and node.get(argument) is None and not (kind == "OneCycleLR" and per_epoch):
        node[argument] = total_steps
    return {name: SCHEDULERS.build(node, optimizer=opt) for name, opt in optimizers.items()}


# ------------------------------------------------------------------------ gradients


def clip_gradients(
    parameters: Iterable[nn.Parameter], limit: float | None, mode: str = "norm"
) -> Tensor | None:
    """Clip gradients in place.

    ``mode="norm"`` rescales all gradients so that their global L2 norm is at most
    ``limit`` and returns the norm before clipping; ``mode="value"`` clamps every
    element to ``[-limit, limit]`` and returns ``None``.
    """
    if limit is None:
        return None
    if mode == "norm":
        return nn.utils.clip_grad_norm_(parameters, limit)
    if mode == "value":
        nn.utils.clip_grad_value_(parameters, limit)
        return None
    raise EngineError(f"train.grad_clip_mode must be 'norm' or 'value', got '{mode}'.")


def make_scaler(enabled: bool) -> Any:
    """A gradient scaler for mixed precision; a no-op object when ``enabled`` is false."""
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------- weight average


class ModelEMA:
    """Exponential moving average of a model's state.

    After every optimizer step ``average = decay * average + (1 - decay) * weights``
    for floating-point entries; integer buffers (step counters) are copied. There is no
    warm-up: early in training the average is dominated by the initial weights.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise EngineError(f"train.ema must be a decay in (0, 1), got {decay}.")
        self.decay = decay
        self.average = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            average = self.average[name]
            if average.is_floating_point():
                average.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                average.copy_(value)

    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.average)

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != set(self.average):
            raise EngineError("the averaged weights in the checkpoint do not match the model.")
        for name, value in state.items():
            self.average[name].copy_(value)

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Temporarily replace the model's weights by their average."""
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.average)
        try:
            yield
        finally:
            model.load_state_dict(backup)
