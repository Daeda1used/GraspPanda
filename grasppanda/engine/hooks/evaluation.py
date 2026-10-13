"""In-training evaluation on the validation split."""

from __future__ import annotations

from contextlib import nullcontext

from grasppanda.engine.errors import EngineError
from grasppanda.engine.evaluation import EvalJob, Evaluation, build_job, evaluate
from grasppanda.engine.state import TrainState
from grasppanda.registry import HOOKS

from .base import Hook

__all__ = ["EvaluationHook"]


@HOOKS.register()
class EvaluationHook(Hook):
    """Predicts on ``data.val`` and scores every variant of ``evaluation.variants``.

    Uses the same routine as ``grasppanda test``. The metrics of complete variants land
    in ``state.metrics`` as ``<variant>/<metric>``, where the checkpoint hook and the
    loggers pick them up; result records go to ``results/epoch_NNNN/``. Model selection
    belongs on a validation split: this hook never reads ``data.test``.

    Args:
        interval: Evaluate every ``interval`` epochs, and always after the last one.
    """

    source = "GraspPanda"

    priority = 70

    def __init__(self, interval: int = 1) -> None:
        self.interval = interval
        self.job: EvalJob | None = None
        self.last: Evaluation | None = None

    def before_run(self, state: TrainState) -> None:
        if state.cfg["evaluation"]["weights"] == "ema" and state.ema is None:
            raise EngineError(
                "evaluation.weights is 'ema', but train.ema is not set: there is no weight "
                "average to evaluate."
            )
        self.job = build_job(state.cfg, "val")  # built up front: a broken evaluator fails now

    def after_epoch(self, state: TrainState) -> None:
        due = self.interval > 0 and state.epoch % self.interval == 0
        if self.job is None or not (due or state.is_last_epoch):
            return
        kind = state.cfg["evaluation"]["weights"]
        averaged = state.ema.applied(state.model) if kind == "ema" else nullcontext()
        with averaged:
            self.last = evaluate(
                state.model,
                self.job,
                state.run,
                results_dir=state.run.results / f"epoch_{state.epoch:04d}",
                weights={"source": "in-training", "kind": kind, "epoch": state.epoch},
            )
        state.metrics, state.metrics_epoch = self.last.metrics, state.epoch
        if state.run.is_main:
            state.run.log.info(
                "evaluation after epoch %d (%d frames, %.1f s)\n%s",
                state.epoch,
                self.last.frames,
                self.last.seconds,
                self.last.render(),
            )
