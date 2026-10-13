"""The state of a training run: what hooks read, and what a checkpoint stores."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from grasppanda.config import Config
from grasppanda.models import BaseGraspDetector

if TYPE_CHECKING:
    from .optim import ModelEMA
    from .run import Run

__all__ = ["Meters", "TrainState"]


class Meters:
    """Running means over the last ``window`` values of named scalars.

    Values may be 0-dim tensors; they are only moved to the CPU when a mean is read, so
    recording them every iteration does not synchronise with the GPU.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = max(int(window), 1)
        self._values: dict[str, deque[Any]] = {}

    def update(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            self._values.setdefault(name, deque(maxlen=self.window)).append(value)

    def means(self) -> dict[str, float]:
        out = {}
        for name, values in self._values.items():
            items = [
                v.detach().float() if isinstance(v, Tensor) else torch.tensor(float(v))
                for v in values
            ]
            out[name] = float(torch.stack([item.cpu() for item in items]).mean())
        return out


@dataclass
class TrainState:
    """Everything a hook may read or change.

    Attributes:
        config: The validated config; ``cfg`` is its resolved form with every default.
        run: Run directory, logger, device.
        model: The bare detector, never a ``DistributedDataParallel`` wrapper.
        optimizers: By name; ``main`` unless the detector configures several.
        schedulers: One per optimizer; empty for a constant learning rate.
        scaler: Gradient scaler of mixed precision (a no-op when ``train.amp`` is off).
        ema: Moving average of the weights, when ``train.ema`` is set.
        epoch_length: Batches per epoch on this process.
        max_epochs: ``train.epochs``.
        epoch: Completed epochs. Inside an epoch this is its 0-based index; in
            ``after_epoch`` it already counts the epoch that just ended.
        iteration: Batches seen since the start of training, across resumes.
        inner: Batches seen in the running epoch.
        outputs: The last iteration's loss terms and metrics plus ``loss``, their total.
        meters: Running means that the loggers print; hooks may add their own scalars.
        metrics: The latest evaluation as ``<variant>/<metric>``, from epoch ``metrics_epoch``.
        best: ``{"metric", "value", "epoch"}`` of the best checkpoint so far.
    """

    config: Config
    run: Run
    model: BaseGraspDetector
    optimizers: dict[str, Optimizer]
    schedulers: dict[str, LRScheduler]
    scaler: Any
    ema: ModelEMA | None
    epoch_length: int
    max_epochs: int
    epoch: int = 0
    iteration: int = 0
    inner: int = 0
    outputs: dict[str, Tensor] = field(default_factory=dict)
    meters: Meters = field(default_factory=Meters)
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_epoch: int = -1
    best: dict[str, Any] | None = None

    @property
    def cfg(self) -> dict[str, Any]:
        return self.run.cfg

    @property
    def device(self) -> torch.device:
        return self.run.device

    @property
    def is_last_epoch(self) -> bool:
        return self.epoch >= self.max_epochs

    def learning_rates(self) -> dict[str, float]:
        """Current learning rate of each optimizer (its first parameter group).

        Read after an iteration, this is the rate the next step will use.
        """
        rates = {name: opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()}
        if list(rates) == ["main"]:
            return {"lr": rates["main"]}
        return {f"lr/{name}": rate for name, rate in rates.items()}
