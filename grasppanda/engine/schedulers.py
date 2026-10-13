"""Learning-rate schedules that ``torch.optim.lr_scheduler`` cannot express in a config.

Every scheduler of PyTorch is available by name (see :mod:`grasppanda.registry`), but a
warm-up followed by another shape needs ``SequentialLR`` or ``LambdaLR``, which take
objects and functions. The schedules here are closed-form classes with plain arguments.
They are registered together with PyTorch's when the scheduler registry is first used, so
a config names them like any other scheduler.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from grasppanda.registry import SCHEDULERS

__all__ = ["WarmupCosineLR"]


@SCHEDULERS.register()
class WarmupCosineLR(LRScheduler):
    """Linear warm-up, then a cosine: the schedule of the FineGrasp reference.

    With ``t`` the number of scheduler steps taken so far and ``m = min_factor``, the
    learning rate is the optimizer's rate times

        m + (1 - m) · t / warmup_steps                                    for t < warmup_steps,
        m + (1 - m) · (1 + cos(π · (t - warmup_steps) / cosine_steps)) / 2   afterwards.

    Args:
        optimizer: The optimizer whose rates are scheduled.
        warmup_steps: Steps of the linear ramp from ``m`` to 1.
        cosine_steps: Steps in which the cosine would fall from 1 to ``m``, counted from
            the end of the warm-up. The reference sets it to the length of the *whole*
            run, so its cosine is cut off ``warmup_steps`` early: after 10 epochs with
            one epoch of warm-up the factor is 0.0245, not ``m``.
        min_factor: Smallest factor, at step 0 and at the end of a complete cosine.
        last_epoch: Index of the last step when resuming; ``-1`` starts afresh.
    """

    status = "tested"
    source = "FineGrasp (arXiv 2507.05978)"

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        cosine_steps: int,
        min_factor: float = 1e-5,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0 or cosine_steps < 1:
            raise ValueError(
                "WarmupCosineLR needs warmup_steps >= 0 and cosine_steps >= 1; got "
                f"warmup_steps={warmup_steps}, cosine_steps={cosine_steps}"
            )
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError(f"min_factor must be in [0, 1], got {min_factor}")
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.min_factor = min_factor
        super().__init__(optimizer, last_epoch)

    def factor(self, step: int) -> float:
        """Multiplier of the base learning rate after ``step`` scheduler steps."""
        span = 1.0 - self.min_factor
        if step < self.warmup_steps:
            return span * step / self.warmup_steps + self.min_factor
        ratio = (step - self.warmup_steps) / self.cosine_steps
        return span * 0.5 * (1.0 + math.cos(math.pi * ratio)) + self.min_factor

    def get_lr(self) -> list[float]:
        factor = self.factor(self.last_epoch)
        return [base * factor for base in self.base_lrs]
