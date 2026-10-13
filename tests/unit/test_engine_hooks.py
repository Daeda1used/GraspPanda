"""Hooks: defaults and ordering, BN momentum, stage timer, optional logging backends."""

from __future__ import annotations

import sys
import types

import pytest
from torch import nn

from grasppanda.config import ConfigError
from grasppanda.engine.builders import build_hooks
from grasppanda.engine.hooks import Hook
from grasppanda.engine.hooks.bn_momentum import BNMomentumHook
from grasppanda.engine.hooks.checkpoint import CheckpointHook
from grasppanda.engine.hooks.evaluation import EvaluationHook
from grasppanda.engine.hooks.logger import TensorBoardLoggerHook, TextLoggerHook, WandbLoggerHook
from grasppanda.engine.hooks.stage_timer import StageTimerHook
from grasppanda.engine.state import Meters
from grasppanda.registry import HOOKS
from grasppanda.utils.imports import MissingDependencyError

from .test_engine_tiny import (
    run_training,
    single_thread,  # noqa: F401  (autouse fixture)
    tiny_config,
)

# ------------------------------------------------------------------ registration


def test_hooks_are_registered_by_auto_import():
    assert {
        "BNMomentumHook",
        "CheckpointHook",
        "EvaluationHook",
        "StageTimerHook",
        "TensorBoardLoggerHook",
        "TextLoggerHook",
        "WandbLoggerHook",
    } <= set(HOOKS.names())


def test_default_hooks_follow_the_config_and_run_in_priority_order(tmp_path):
    hooks = build_hooks(tiny_config(tmp_path).resolved())
    assert [type(hook) for hook in hooks] == [EvaluationHook, CheckpointHook, TextLoggerHook]
    assert [hook.priority for hook in hooks] == [70, 80, 90]
    extra = {"hooks": [{"type": "BNMomentumHook"}, {"type": "StageTimerHook", "sync": True}]}
    hooks = build_hooks(tiny_config(tmp_path, extra).resolved())
    assert [type(hook).__name__ for hook in hooks][:2] == ["BNMomentumHook", "StageTimerHook"]
    no_val = build_hooks(tiny_config(tmp_path, {"data.val": None}).resolved())
    assert EvaluationHook not in {type(hook) for hook in no_val}
    off = build_hooks(tiny_config(tmp_path, {"evaluation.interval": 0}).resolved())
    assert EvaluationHook not in {type(hook) for hook in off}


def test_a_listed_hook_replaces_the_default_of_its_class(tmp_path):
    listed = {"hooks": [{"type": "CheckpointHook", "interval": 5, "keep_last": 9}]}
    hooks = build_hooks(tiny_config(tmp_path, listed).resolved())
    (checkpoint,) = [hook for hook in hooks if isinstance(hook, CheckpointHook)]
    assert (checkpoint.interval, checkpoint.keep_last) == (5, 9)


def test_hook_configs_are_validated_like_any_component(tmp_path):
    with pytest.raises(ConfigError, match="BNMomentumHook has no parameter 'decay_rate'"):
        tiny_config(tmp_path, {"hooks": [{"type": "BNMomentumHook", "decay_rate": 0.5}]})
    with pytest.raises(ConfigError, match="expected one of 'online', 'offline', 'disabled'"):
        tiny_config(tmp_path, {"hooks": [{"type": "WandbLoggerHook", "mode": "sometimes"}]})


# ------------------------------------------------------------------- BN momentum


def test_bn_momentum_schedule_matches_the_reference_formula():
    hook = BNMomentumHook()
    expected = [max(0.5 * 0.5 ** (epoch // 2), 0.001) for epoch in range(24)]
    assert [hook.momentum(epoch) for epoch in range(24)] == expected
    assert expected[:5] == [0.5, 0.5, 0.25, 0.25, 0.125] and expected[-1] == 0.001  # the floor
    custom = BNMomentumHook(initial=0.1, decay=0.1, step=3, minimum=0.01)
    assert [custom.momentum(e) for e in (0, 2, 3, 6)] == pytest.approx([0.1, 0.1, 0.01, 0.01])
    with pytest.raises(ValueError, match="at least 1"):
        BNMomentumHook(step=0)


def test_bn_momentum_is_applied_to_every_batch_norm_layer_each_epoch(tmp_path):
    seen = []

    class Spy(Hook):
        priority = 60  # after BNMomentumHook

        def before_epoch(self, state) -> None:
            layers = [m for m in state.model.modules() if isinstance(m, nn.BatchNorm1d)]
            seen.append([layer.momentum for layer in layers])

    changes = {
        "model.bn": True,
        "train.epochs": 5,
        "evaluation.interval": 0,
        "hooks": [
            {"type": "BNMomentumHook", "initial": 0.4, "decay": 0.5, "step": 2, "minimum": 0.15}
        ],
    }
    state = run_training(tiny_config(tmp_path, changes), Spy())
    assert seen == [[0.4], [0.4], [0.2], [0.2], [0.15]]
    log = (state.run.dir / "log.txt").read_text()
    assert log.count("batch-norm momentum") == 3  # logged when it changes, not every epoch
    assert "batch-norm momentum 0.15 on 1 layers from epoch 5" in log


def test_bn_momentum_needs_no_state_to_resume(tmp_path):
    from .test_engine_tiny import Interrupt, Stop

    changes = {
        "model.bn": True,
        "train.epochs": 4,
        "evaluation.interval": 0,
        "hooks": [{"type": "BNMomentumHook"}],
    }
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path, changes), Interrupt(3))
    resumed = run_training(tiny_config(tmp_path, {**changes, "runtime.resume": True}))
    (layer,) = [m for m in resumed.model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert layer.momentum == 0.25  # epoch index 3: 0.5 * 0.5 ** (3 // 2)


# ------------------------------------------------------------------- stage timer


def test_stage_timer_reports_seconds_per_stage(tmp_path):
    changes = {"hooks": [{"type": "StageTimerHook"}], "train.epochs": 1}
    state = run_training(tiny_config(tmp_path, changes))
    means = state.meters.means()
    stages = {name for name in means if name.startswith("time/forward.")}
    assert stages == {"time/forward.encoder", "time/forward.drop", "time/forward.head"}
    assert all(0.0 < means[name] < means["time/iter"] for name in stages)
    assert "forward.encoder " in (state.run.dir / "log.txt").read_text()
    assert not state.model.encoder._forward_hooks  # removed after the run


def test_stage_timer_can_descend(tmp_path):
    changes = {
        "hooks": [{"type": "StageTimerHook", "depth": 2}],
        "train.epochs": 1,
        "evaluation.interval": 0,
    }
    means = run_training(tiny_config(tmp_path, changes)).meters.means()
    assert "time/forward.encoder.0" in means and "time/forward.encoder" in means
    with pytest.raises(ValueError, match="at least 1"):
        StageTimerHook(depth=0)


def test_meters_average_a_sliding_window():
    import torch

    meters = Meters(window=2)
    for value in (1.0, 2.0, torch.tensor(4.0)):
        meters.update({"x": value})
    meters.update({"y": 3})
    assert meters.means() == {"x": 3.0, "y": 3.0}


# ------------------------------------------------------------- optional backends


def test_missing_logging_backends_fail_before_training_with_the_install_command(tmp_path):
    for backend, package in (("tensorboard", "tensorboard"), ("wandb", "wandb")):
        if package in sys.modules or _installed(package):
            pytest.skip(f"{package} is installed here")
        config = tiny_config(tmp_path / backend, {"runtime.log.backends": ["text", backend]})
        with pytest.raises(MissingDependencyError, match=rf'pip install -e "\.\[{backend}\]"'):
            run_training(config)
        assert not list(
            (tmp_path / backend / "work").glob("*/checkpoints/*")
        )  # nothing was trained


def _installed(package: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(package) is not None


def test_tensorboard_logger_writes_train_and_eval_scalars(tmp_path, monkeypatch):
    """Against a stand-in for ``SummaryWriter``: checks what is sent, not TensorBoard itself."""
    calls: list[tuple] = []

    class Writer:
        def __init__(self, log_dir: str) -> None:
            calls.append(("open", log_dir))

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            calls.append(("scalar", tag, float(value), step))

        def flush(self) -> None:
            calls.append(("flush",))

        def close(self) -> None:
            calls.append(("close",))

    fake = types.ModuleType("torch.utils.tensorboard")
    fake.SummaryWriter = Writer
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake)
    state = run_training(tiny_config(tmp_path, {"runtime.log.backends": ["tensorboard"]}))
    assert calls[0] == ("open", str(state.run.dir / "tensorboard")) and calls[-1] == ("close",)
    scalars = [call for call in calls if call[0] == "scalar"]
    assert {step for _, tag, _, step in scalars if tag == "train/loss"} == {2, 4, 6, 8}
    assert {"train/loss/fit", "train/lr", "train/time/iter", "train/eta_seconds"} <= {
        c[1] for c in scalars
    }
    assert [(tag, value, step) for _, tag, value, step in scalars if tag.startswith("eval/")] == [
        ("eval/raw/AP", 1.0, 1),
        ("eval/raw/AP", 1.0, 2),
    ]
    assert not isinstance(build_hooks(state.cfg)[-1], TextLoggerHook)
    assert isinstance(build_hooks(state.cfg)[-1], TensorBoardLoggerHook)


def test_wandb_logger_names_the_run_and_logs_by_iteration(tmp_path, monkeypatch):
    """Against a stand-in for the ``wandb`` module: checks what is sent, not the service."""
    calls: list[tuple] = []
    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: calls.append(("init", kwargs))
    fake.log = lambda values, step: calls.append(("log", values, step))
    fake.finish = lambda: calls.append(("finish",))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    changes = {
        "runtime.log.backends": ["text"],
        "runtime.run_id": "named",
        "hooks": [{"type": "WandbLoggerHook", "interval": 4, "project": "p", "mode": "offline"}],
    }
    state = run_training(tiny_config(tmp_path, changes))
    kind, init = calls[0]
    assert kind == "init" and (init["project"], init["name"], init["id"], init["mode"]) == (
        "p",
        "named",
        "named",
        "offline",
    )
    assert (
        init["resume"] == "allow"
        and init["config"]["train"]["epochs"] == 2
        and init["dir"] == str(state.run.dir)
    )
    logs = [call for call in calls if call[0] == "log"]
    assert [step for _, values, step in logs if "train/loss" in values] == [4, 8]
    assert [values for _, values, _ in logs if "eval/raw/AP" in values] == [
        {"eval/raw/AP": 1.0, "epoch": 1},
        {"eval/raw/AP": 1.0, "epoch": 2},
    ]
    assert calls[-1] == ("finish",)
    assert isinstance(WandbLoggerHook(), Hook)
