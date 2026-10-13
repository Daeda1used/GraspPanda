"""What ``grasppanda train``, ``test`` and ``infer`` do, callable from Python.

Each takes a validated :class:`~grasppanda.config.Config` and nothing else: every setting
of a run is in the config, so the command line only loads it (after turning its few
convenience flags into overrides) and calls one of these.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch import nn

from grasppanda import __version__
from grasppanda.config import Config
from grasppanda.registry import BuildError
from grasppanda.structures import ContractError
from grasppanda.utils.console import banner, style
from grasppanda.utils.imports import MissingDependencyError

from . import distributed as dist
from .builders import build_model
from .checkpoint import checkpoint_name, file_sha256, load_weights
from .errors import EngineError
from .evaluation import Dump, EvalJob, Evaluation, build_job, evaluate, predict
from .run import Run, start_run
from .seeding import seed_everything
from .state import TrainState
from .trainer import Trainer

__all__ = ["infer", "launch", "test", "train"]

#: Failures with a message the user can act on; anything else is a bug and keeps its traceback.
_EXPECTED = (EngineError, BuildError, ContractError, MissingDependencyError)


def launch(command: Callable[[Config], Any], config: Config) -> int:
    """Run a command for the CLI: 0 on success, 1 with a one-paragraph reason on failure."""
    try:
        command(config)
    except _EXPECTED as error:
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        print(f"{mark} {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


# -------------------------------------------------------------------------- summary


def _header(run: Run, model: nn.Module, lines: list[tuple[str, str]]) -> str:
    config, runtime = run.config, run.cfg["runtime"]
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    processes = dist.world_size()
    rows = [
        ("config", f"{config.path or '<dict>'}  (hash {config.digest()[:12]})"),
        ("run", str(run.dir)),
        (
            "device",
            f"{run.device} · {processes} process{'es' if processes > 1 else ''} · seed "
            f"{runtime['seed']} · deterministic {'on' if runtime['deterministic'] else 'off'}",
        ),
        ("model", f"{type(model).__name__} · {parameters:,} parameters ({trainable:,} trainable)"),
        *lines,
    ]
    width = max(len(name) for name, _ in rows)
    body = [f"  {name:<{width}}  {value}" for name, value in rows]
    return "\n".join([banner(__version__, enabled=False), f"  {run.command}", *body])


def _component(node: Any) -> str:
    if not node:
        return "none"
    arguments = ", ".join(f"{key}={value}" for key, value in node.items() if key != "type")
    return f"{node['type']}({arguments})"


def _prepare(run: Run, *, with_evaluator: bool) -> tuple[nn.Module, EvalJob, dict[str, Any]]:
    """Build the model and the job of ``test`` or ``infer``, print the header, load the weights.

    Returns the model, the job and what a result record says about the weights.
    """
    cfg = run.cfg
    runtime, section, kind = (
        cfg["runtime"],
        cfg["evaluation"]["split"],
        cfg["evaluation"]["weights"],
    )
    path = runtime["load_from"]
    if not path:
        raise EngineError(
            f"`grasppanda {run.command}` needs weights: pass --weights or set runtime.load_from. "
            "An untrained model is never scored."
        )
    seed_everything(runtime["seed"], deterministic=runtime["deterministic"])
    model = build_model(cfg).to(run.device)
    job = build_job(cfg, section, with_evaluator=with_evaluator)
    lines = [("data", f"data.{section}: {job.dataset.name}, {len(job.dataset)} frames")]
    if job.evaluator is not None:
        scorer = f"{type(job.evaluator).__name__} · protocol {job.evaluator.protocol}"
        lines.append(("evaluator", scorer))
    lines.append(("variants", ", ".join(job.chains)))
    run.log.info(_header(run, model, lines))
    report = load_weights(model, path, kind=kind, allow_mismatch=runtime["allow_weight_mismatch"])
    run.log.info(report.render())
    weights = {"path": str(path), "kind": kind, "clean_load": report.clean, **report.meta}
    if run.is_main:
        weights["sha256"] = file_sha256(path)
        run.log.info("  sha256 %s", weights["sha256"])
    return model, job, weights


def _evaluation_plan(cfg: dict[str, Any]) -> str:
    """One line saying whether, how often and on what a training run evaluates itself."""
    evaluation = cfg["evaluation"]
    needs = {
        "data.val": (cfg.get("data") or {}).get("val") is not None,
        "evaluation.evaluator": evaluation["evaluator"] is not None,
        "evaluation.interval > 0": evaluation["interval"] > 0,
    }
    if not all(needs.values()):
        return "off: " + ", ".join(name for name, met in needs.items() if not met) + " not set"
    return (
        f"data.val every {evaluation['interval']} epoch(s) · variants "
        f"{', '.join(evaluation['variants'])} · best by {evaluation['save_best'] or 'nothing'}"
    )


# ----------------------------------------------------------------------------- train


def train(config: Config) -> TrainState:
    """Train as ``config`` says; returns the final state (``state.run.dir`` is the run)."""
    with start_run(config, "train") as run:
        trainer = Trainer(run)
        state, cfg = trainer.state, run.cfg
        loader = cfg["data"]["train"]
        if run.is_main:
            lines = [
                (
                    "data",
                    f"{len(trainer.loader.dataset)} frames · {state.epoch_length} batches of "
                    f"{loader['batch_size']} per epoch"
                    + (" per process" if dist.is_distributed() else ""),
                ),
                (
                    "schedule",
                    f"{state.max_epochs} epochs · {_component(config['optimizer'])} · "
                    f"{_component(config.get('scheduler'))} per {cfg['train']['scheduler_unit']}",
                ),
                ("evaluation", _evaluation_plan(cfg)),
                ("hooks", ", ".join(type(hook).__name__ for hook in trainer.hooks)),
            ]
            run.log.info(_header(run, state.model, lines))
        started = time.perf_counter()
        first = state.epoch
        trainer.train()
        if run.is_main:
            final = run.checkpoints / checkpoint_name(state.epoch)
            summary = [
                f"✓ trained epochs {first + 1}-{state.epoch} in "
                f"{time.perf_counter() - started:.1f} s"
                if state.epoch > first
                else f"✓ nothing to do: {state.epoch} of {state.max_epochs} epochs were done",
                f"  checkpoint  {final if final.is_file() else 'none written'}",
            ]
            if state.best is not None:
                summary.append(
                    f"  best        {state.best['metric']} = {state.best['value']:.4f} at epoch "
                    f"{state.best['epoch']}  ({run.checkpoints / 'best.pth'})"
                )
            run.log.info("\n".join(summary))
    return state


# ------------------------------------------------------------------------------ test


def test(config: Config) -> Evaluation:
    """Score ``runtime.load_from`` on ``data.<evaluation.split>`` and write result records."""
    with start_run(config, "test") as run:
        model, job, weights = _prepare(run, with_evaluator=True)
        evaluation = evaluate(model, job, run, weights=weights)
        if run.is_main:
            records = _listing(run.results, "*.json")
            run.log.info("%s\nresult records: %s", evaluation.render(), records)
    return evaluation


# ----------------------------------------------------------------------------- infer


def infer(config: Config) -> Dump:
    """Predict on ``data.<evaluation.split>`` and write the grasps; nothing is scored."""
    with start_run(config, "infer") as run:
        model, job, weights = _prepare(run, with_evaluator=False)
        interval = run.cfg["runtime"]["log"]["interval"]
        dump = predict(model, job, run.predictions, device=run.device, log_interval=interval)
        directories = sorted({str(job.directory(dump.root, name)) for name in job.chains})
        run.stamp(
            gt_assisted=dump.gt_assisted,
            frames=len(dump.keys),
            weights=weights,
            predictions=directories,
        )
        if run.is_main:
            run.log.info(
                "✓ %d frames in %.1f s · ground-truth-assisted inputs: %s\n"
                "  one GraspNet (G, 17) array per frame, scene_NNNN/<camera>/NNNN.npy, below\n  %s",
                len(dump.keys),
                dump.seconds,
                ", ".join(dump.gt_assisted) or "none",
                "\n  ".join(directories),
            )
    return dump


def _listing(directory: Path, pattern: str) -> str:
    return ", ".join(str(path) for path in sorted(directory.glob(pattern))) or "none"
