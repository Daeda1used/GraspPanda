"""The training loop.

Read :meth:`Trainer.train` top to bottom: epochs of batches; a batch is forward and
backward; every ``train.accumulate`` batches the gradients are clipped and applied, the
weight average and the scheduler follow. Nothing else happens in the loop. Logging,
checkpoints, evaluation and every other schedule are hooks (:mod:`grasppanda.engine.hooks`),
and the mechanics live in small modules next to this one:

==================  ===========================================================
``builders``        model, datasets, loaders with reproducible order, hooks
``optim``           optimizers, schedulers, clipping, weight average, AMP scaler
``checkpoint``      save, resume, weights-only load with a key report
``evaluation``      predict and score a split (shared with ``grasppanda test``)
``distributed``     rank detection and collectives for ``torchrun``
``run``             run directory, snapshot, log, device
``seeding``         seeds and random states
==================  ===========================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

from grasppanda.structures import Observation, SampleKey, Supervision

from . import distributed as dist
from .builders import build_dataset, build_hooks, build_model, build_train_loader, set_epoch
from .checkpoint import load_weights, resume
from .errors import EngineError, NonFiniteLossError
from .optim import (
    ModelEMA,
    build_optimizers,
    build_schedulers,
    clip_gradients,
    make_scaler,
    scheduler_steps,
)
from .run import Run
from .seeding import seed_everything
from .state import Meters, TrainState

__all__ = ["Trainer", "total_loss"]

LossFunction = Callable[[Observation, Supervision], Mapping[str, Tensor]]


def total_loss(outputs: Mapping[str, Tensor], where: str, keys: Sequence[SampleKey] = ()) -> Tensor:
    """Sum the ``loss/*`` entries of ``outputs``; name the term that is NaN or infinite.

    Other entries are metrics: they are logged, never summed and never checked.
    """
    terms = {name: value for name, value in outputs.items() if name.startswith("loss/")}
    if not terms:
        raise EngineError(
            f"loss() returned {sorted(outputs) or 'nothing'}: no entry starts with 'loss/', "
            "so there is nothing to optimise."
        )
    for name, value in terms.items():
        if not isinstance(value, Tensor) or value.numel() != 1:
            shape = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
            raise EngineError(f"{name} must be a scalar tensor, got {shape}. Reduce it in loss().")
    total = torch.stack([value.reshape(()) for value in terms.values()]).sum()
    if not torch.isfinite(total):
        bad = {n: float(v) for n, v in terms.items() if not torch.isfinite(v).all()}
        named = ", ".join(f"{name} = {value}" for name, value in bad.items())
        frames = ", ".join(str(key) for key in keys[:8]) + (" ..." if len(keys) > 8 else "")
        raise NonFiniteLossError(
            f"non-finite loss at {where}: {named or 'every term is finite but their sum is not'}"
            + (f". Frames in the batch: {frames}" if frames else "")
        )
    return total


class Trainer:
    """Builds the training state of a run and runs the loop."""

    def __init__(self, run: Run) -> None:
        cfg, config, device = run.cfg, run.config, run.device
        train, runtime = cfg["train"], cfg["runtime"]
        self.run = run
        self.amp = train["amp"]
        self.accumulate = max(train["accumulate"], 1)
        self.grad_clip, self.clip_mode = train["grad_clip"], train["grad_clip_mode"]
        self.scheduler_unit = train["scheduler_unit"]

        # Every process builds the model from the same seed: replicas must start equal.
        seed_everything(runtime["seed"], deterministic=runtime["deterministic"])
        model = build_model(cfg)
        if train["sync_bn"]:
            model = dist.sync_batchnorm(model, device)
        model.to(device)
        if runtime["load_from"] and run.resume_from is None:
            report = load_weights(
                model, runtime["load_from"], allow_mismatch=runtime["allow_weight_mismatch"]
            )
            run.log.info(report.render())
        # From here on each process draws its own random numbers (sampling, augmentation).
        seed_everything(runtime["seed"] + dist.rank(), deterministic=runtime["deterministic"])

        self.loader = build_train_loader(cfg, build_dataset(cfg, "train"))
        optimizers = build_optimizers(model, config["optimizer"])
        steps = scheduler_steps(
            self.scheduler_unit, train["epochs"], len(self.loader), self.accumulate
        )
        self.state = TrainState(
            config=config,
            run=run,
            model=model,
            optimizers=optimizers,
            schedulers=build_schedulers(config.get("scheduler"), optimizers, total_steps=steps),
            scaler=make_scaler(self.amp and device.type == "cuda"),
            ema=ModelEMA(model, train["ema"]) if train["ema"] is not None else None,
            epoch_length=len(self.loader),
            max_epochs=train["epochs"],
            meters=Meters(runtime["log"]["interval"]),
        )
        self.hooks = build_hooks(cfg)
        replicated = dist.parallel(model, device)  # averages gradients across processes
        self.compute_loss: LossFunction = model.loss if replicated is None else replicated
        if run.resume_from is not None:
            resume(
                self.state,
                run.resume_from,
                allow_config_mismatch=runtime["allow_config_mismatch"],
            )
            run.log.info("resumed %s at epoch %d", run.resume_from, self.state.epoch)

    def _call(self, event: str) -> None:
        for hook in self.hooks:
            getattr(hook, event)(self.state)

    # ------------------------------------------------------------------ the loop

    def train(self) -> TrainState:
        state, device = self.state, self.run.device
        self._call("before_run")
        for epoch in range(state.epoch, state.max_epochs):
            state.inner = 0
            set_epoch(self.loader, epoch)
            state.model.train()
            self._call("before_epoch")
            mark = time.perf_counter()
            for batch in self.loader:
                data_time = time.perf_counter() - mark
                batch = batch.to(device)
                if batch.supervision is None:
                    raise EngineError("data.train yields frames without supervision.")
                with torch.autocast(device.type, enabled=self.amp):
                    outputs = self.compute_loss(batch.observation, batch.supervision)
                    where = f"epoch {epoch + 1}, iteration {state.inner + 1}"
                    total = total_loss(outputs, where, batch.observation.keys)
                state.scaler.scale(total / self.accumulate).backward()
                state.inner += 1
                state.iteration += 1
                if state.inner % self.accumulate == 0 or state.inner == state.epoch_length:
                    self.step()
                state.outputs = {n: torch.as_tensor(v).detach() for n, v in outputs.items()}
                state.outputs["loss"] = total.detach()
                timing = {"time/data": data_time, "time/iter": time.perf_counter() - mark}
                state.meters.update({**state.outputs, **timing})
                self._call("after_iter")
                mark = time.perf_counter()
            if self.scheduler_unit == "epoch":
                for scheduler in state.schedulers.values():
                    scheduler.step()
            state.epoch = epoch + 1
            self._call("after_epoch")
        self._call("after_run")
        return state

    def step(self) -> None:
        """Apply the accumulated gradients: clip, update, average, schedule."""
        state = self.state
        optimizers = list(state.optimizers.values())
        for hook in self.hooks:  # optional: an object with the other callbacks is a hook too
            before_step = getattr(hook, "before_step", None)
            if before_step is not None:
                before_step(state)
        if self.grad_clip is not None:
            for optimizer in optimizers:
                state.scaler.unscale_(optimizer)  # clip real gradients, not scaled ones
            norm = clip_gradients(state.model.parameters(), self.grad_clip, self.clip_mode)
            if norm is not None:
                state.meters.update({"grad_norm": norm})
        for optimizer in optimizers:
            state.scaler.step(optimizer)
        state.scaler.update()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if state.ema is not None:
            state.ema.update(state.model)
        if self.scheduler_unit == "iteration":
            for scheduler in state.schedulers.values():
                scheduler.step()
