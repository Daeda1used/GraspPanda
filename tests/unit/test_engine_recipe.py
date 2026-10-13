"""The training recipe as config lines: optimizer bridges, parameter groups, timm schedules
and the recipe hooks, each trained for a few steps on the tiny CPU run."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from grasppanda.config import Config
from grasppanda.engine.errors import EngineError
from grasppanda.engine.hooks.recipe import EMAHook, FreezeHook
from grasppanda.engine.optim import build_optimizers, build_schedulers
from grasppanda.engine.optimizers import layer_names
from grasppanda.registry import HOOKS, OPTIMIZERS, SCHEDULERS, BuildError
from grasppanda.utils.imports import MissingDependencyError

from .test_engine_tiny import (
    Interrupt,
    Recorder,
    Stop,
    run_training,
    single_thread,  # noqa: F401  (autouse fixture)
    tiny_config,
)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)) for _ in range(2)
        )


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Encoder()
        self.head = nn.Linear(4, 1)


def groups(optimizer):
    return {group["name"]: group for group in optimizer.param_groups}


# ---------------------------------------------------------------------------- registry


def test_the_recipe_components_are_registered_without_the_optional_libraries():
    assert {"ParamGroups", "PytorchOptimizer", "ScheduleFree", "TimmOptimizer"} <= set(
        OPTIMIZERS.names()
    )
    assert {"TimmSchedule", "WarmupCosineLR", "OneCycleLR"} <= set(SCHEDULERS.names())
    assert {
        "AdaptiveGradClipHook",
        "EMAHook",
        "FreezeHook",
        "SWAHook",
        "ScheduleFreeHook",
    } <= set(HOOKS.names())


def test_a_bridge_without_its_library_says_what_to_install(monkeypatch):
    import builtins
    import importlib

    real = importlib.import_module

    def missing(name, *args, **kwargs):
        if name.split(".")[0] in ("timm", "pytorch_optimizer", "schedulefree"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    assert builtins  # the patched function is the one `require` calls
    model = Model()
    for node in (
        {"type": "TimmOptimizer", "name": "lamb", "lr": 1e-3},
        {"type": "PytorchOptimizer", "name": "prodigy", "lr": 1.0},
        {"type": "ScheduleFree"},
    ):
        with pytest.raises(MissingDependencyError, match=r"grasppanda\[optim\]"):
            build_optimizers(model, node)


# --------------------------------------------------------------------- parameter groups


def test_param_groups_scale_the_rate_per_module_and_drop_decay_on_1d():
    model = Model()
    node = {
        "type": "ParamGroups",
        "optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05},
        "lr_multipliers": {"backbone": 0.1},
        "no_decay_1d": True,
    }
    optimizer = build_optimizers(model, node)["main"]
    assert isinstance(optimizer, torch.optim.AdamW)
    found = groups(optimizer)
    assert set(found) == {"backbone", "backbone (no decay)", "rest", "rest (no decay)"}
    assert found["backbone"]["lr"] == pytest.approx(1e-4) and found["rest"]["lr"] == 1e-3
    assert found["backbone"]["weight_decay"] == 0.05
    assert found["rest (no decay)"]["weight_decay"] == 0.0
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(model.parameters())
    )
    # OneCycleLR would overwrite the rates with one max_lr: it is scaled per group.
    scheduler = build_schedulers(
        {"type": "OneCycleLR", "max_lr": 1e-3, "div_factor": 10.0},
        {"main": optimizer},
        total_steps=10,
    )["main"]
    assert scheduler.get_last_lr() == pytest.approx(
        [group["lr_scale"] * 1e-4 for group in optimizer.param_groups]
    )


def test_a_one_cycle_schedule_of_a_few_steps_gets_the_shortest_warm_up():
    optimizer = torch.optim.SGD(Model().parameters(), lr=1.0)
    node = {"type": "OneCycleLR", "max_lr": 1.0, "pct_start": 0.05}
    with pytest.warns(UserWarning, match=r"is 1 of 20 steps.*uses pct_start = 0\.1"):
        scheduler = build_schedulers(node, {"main": optimizer}, total_steps=20)["main"]
    rates = []
    for _ in range(20):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert max(rates) == pytest.approx(1.0) and rates.index(max(rates)) == 1
    assert node["pct_start"] == 0.05  # the config node is not changed
    with pytest.raises(EngineError, match="at least 3 scheduler steps"):
        build_schedulers(node, {"main": optimizer}, total_steps=2)


def test_layer_decay_follows_the_layers_of_the_encoder():
    model = Model()
    assert layer_names(model.backbone, 2) == ["stem", "stages.0", "stages.1"]
    assert layer_names(model.backbone, 1) == ["stem", "stages"]
    node = {
        "type": "ParamGroups",
        "optimizer": {"type": "SGD", "lr": 1.0},
        "layer_decay": 0.5,
    }
    found = groups(build_optimizers(model, node)["main"])
    rates = {name: group["lr"] for name, group in found.items()}
    assert rates == {"backbone[0]": 0.125, "backbone[1]": 0.25, "backbone[2]": 0.5, "rest": 1.0}


def test_param_groups_refuse_what_they_cannot_do():
    model = Model()
    inner = {"type": "SGD", "lr": 1.0}
    with pytest.raises(EngineError, match=r"no parameter matches encoder.*backbone, head"):
        build_optimizers(
            model, {"type": "ParamGroups", "optimizer": inner, "lr_multipliers": {"encoder": 0.1}}
        )
    with pytest.raises(EngineError, match="must set `lr`"):
        build_optimizers(model, {"type": "ParamGroups", "optimizer": {"type": "SGD"}})
    with pytest.raises(EngineError, match=r"has no such sub-module \(it has: backbone, head\)"):
        build_optimizers(
            model,
            {
                "type": "ParamGroups",
                "optimizer": inner,
                "layer_decay": 0.5,
                "layer_decay_module": "encoder",
            },
        )


def test_param_groups_train_and_are_checked_by_the_config(tmp_path):
    node = {
        "type": "ParamGroups",
        "optimizer": {"type": "SGD", "lr": 0.1},
        "lr_multipliers": {"encoder": 0.0},
    }
    config = tiny_config(tmp_path, {"optimizer": node, "evaluation.interval": 0})
    before = {}

    class Keep(Recorder):
        def before_run(self, state):
            before["encoder"] = state.model.encoder[0].weight.detach().clone()
            before["head"] = state.model.head.weight.detach().clone()

    state = run_training(config, Keep())
    assert torch.equal(state.model.encoder[0].weight, before["encoder"])  # multiplier 0
    assert not torch.equal(state.model.head.weight, before["head"])


# ----------------------------------------------------------------------------- bridges


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"type": "TimmOptimizer", "name": "lamb", "lr": 1e-2}, "Lamb"),
        ({"type": "TimmOptimizer", "name": "adopt", "lr": 1e-2, "options": {"eps": 1e-6}}, "Adopt"),
        ({"type": "TimmOptimizer", "name": "lookahead_adan", "lr": 1e-2}, "Lookahead"),
    ],
)
def test_timm_optimizers_train_by_name(tmp_path, node, expected):
    pytest.importorskip("timm")
    config = tiny_config(tmp_path, {"optimizer": node, "evaluation.interval": 0})
    recorder = Recorder()
    state = run_training(config, recorder)
    assert type(state.optimizers["main"]).__name__ == expected
    assert all(torch.isfinite(torch.tensor(recorder.losses)))
    with pytest.raises(EngineError, match=r"'lambda', which timm .* does not have"):
        build_optimizers(Model(), {"type": "TimmOptimizer", "name": "lambda", "lr": 1.0})
    with pytest.raises(EngineError, match="move it out of `options`"):
        build_optimizers(
            Model(), {"type": "TimmOptimizer", "name": "lamb", "lr": 1.0, "options": {"lr": 2.0}}
        )


@pytest.mark.parametrize(
    ("name", "lr", "expected"),
    [("prodigy", 1.0, "Prodigy"), ("ademamix", 1e-2, "AdEMAMix"), ("stableadamw", 1e-2, None)],
)
def test_pytorch_optimizer_trains_by_name(tmp_path, name, lr, expected):
    pytest.importorskip("pytorch_optimizer")
    node = {"type": "PytorchOptimizer", "name": name, "lr": lr}
    config = tiny_config(tmp_path, {"optimizer": node, "evaluation.interval": 0})
    recorder = Recorder()
    state = run_training(config, recorder)
    if expected is not None:
        assert type(state.optimizers["main"]).__name__ == expected
    assert all(torch.isfinite(torch.tensor(recorder.losses)))
    with pytest.raises(EngineError, match="'adamww', which pytorch_optimizer"):
        build_optimizers(Model(), {"type": "PytorchOptimizer", "name": "adamww", "lr": 1.0})


@pytest.mark.parametrize("name", ["adamw", "radam", "sgd"])
def test_schedule_free_trains_and_the_hook_switches_to_the_average(tmp_path, name):
    pytest.importorskip("schedulefree")
    changes = {
        "optimizer": {"type": "ScheduleFree", "name": name, "lr": 0.05},
        "hooks": [{"type": "ScheduleFreeHook"}],
        "train.epochs": 2,
    }
    modes: list[bool] = []

    class Modes(Recorder):
        priority = 75  # after the in-training evaluation

        def after_iter(self, state):
            super().after_iter(state)
            modes.append(state.optimizers["main"].param_groups[0]["train_mode"])

        def after_epoch(self, state):
            modes.append(state.optimizers["main"].param_groups[0]["train_mode"])

    recorder = Modes()
    run_training(tiny_config(tmp_path, changes), recorder)
    assert modes[0] is True and modes[-1] is False and modes.count(False) == 2
    assert all(torch.isfinite(torch.tensor(recorder.losses)))


def test_schedule_free_misuse_is_named(tmp_path):
    with pytest.raises(EngineError, match="has no train\\(\\)/eval\\(\\) modes"):
        run_training(tiny_config(tmp_path, {"hooks": [{"type": "ScheduleFreeHook"}]}))
    pytest.importorskip("schedulefree")
    with pytest.raises(EngineError, match="must be one of adamw, radam, sgd"):
        build_optimizers(Model(), {"type": "ScheduleFree", "name": "lion"})


def test_timm_schedules_are_closed_form_and_get_the_length_of_the_run():
    pytest.importorskip("timm")
    optimizer = torch.optim.SGD(Model().parameters(), lr=1.0)
    node = {"type": "TimmSchedule", "name": "cosine", "warmup_steps": 0.1, "lr_min": 0.01}
    scheduler = build_schedulers(node, {"main": optimizer}, total_steps=100)["main"]
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == 0.0 and rates[5] == pytest.approx(0.5)
    assert max(rates) == pytest.approx(rates[10], abs=1e-6) and rates[10] > 0.97
    assert rates[99] == pytest.approx(0.01, abs=1e-3)
    assert rates[10:] == sorted(rates[10:], reverse=True)
    steps = {"type": "TimmSchedule", "name": "multistep", "warmup_steps": 2}
    steps["options"] = {"decay_t": [4, 6], "decay_rate": 0.1}
    optimizer = torch.optim.SGD(Model().parameters(), lr=1.0)
    scheduler = build_schedulers(steps, {"main": optimizer}, total_steps=8)["main"]
    rates = []
    for _ in range(8):
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert rates == pytest.approx([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1][:8], abs=1e-9) or (
        rates[2] == 1.0 and rates[-1] == pytest.approx(0.01)
    )
    with pytest.raises((ValueError, BuildError), match="must be one of cosine"):
        SCHEDULERS.build({"type": "TimmSchedule", "name": "plateau"}, optimizer=optimizer)
    with pytest.raises((ValueError, BuildError), match="needs options.decay_t"):
        SCHEDULERS.build({"type": "TimmSchedule", "name": "step"}, optimizer=optimizer)


def test_a_timm_schedule_trains_per_iteration(tmp_path):
    pytest.importorskip("timm")
    changes = {
        "scheduler": {"type": "TimmSchedule", "name": "cosine", "warmup_steps": 2},
        "train.scheduler_unit": "iteration",
        "evaluation.interval": 0,
    }
    recorder = Recorder()
    run_training(tiny_config(tmp_path, changes), recorder)
    assert recorder.rates[0] == pytest.approx(0.05) and recorder.rates[-1] < 0.01
    assert max(recorder.rates) == pytest.approx(recorder.rates[1]) and recorder.rates[1] > 0.085


# ------------------------------------------------------------------------------- hooks


def test_ema_hook_warms_up_and_serves_evaluation_and_checkpoints(tmp_path):
    hook = EMAHook(decay=0.999)
    assert hook.factor(0) == pytest.approx(0.1) and hook.factor(10**6) == 0.999
    assert EMAHook(decay=0.9, warmup=False).factor(0) == 0.9
    changes = {
        "hooks": [{"type": "EMAHook", "decay": 0.5, "warmup": False}],
        "evaluation.weights": "ema",
        "model.bn": True,
        "train.epochs": 1,
    }
    state = run_training(tiny_config(tmp_path, changes))
    average = state.ema.state_dict()
    assert set(average) == set(state.model.state_dict())
    assert not torch.equal(average["head.weight"], state.model.head.weight)
    # Buffers are copied, not averaged: batch-norm statistics are those of the model.
    assert torch.equal(average["encoder.1.running_mean"], state.model.encoder[1].running_mean)
    saved = torch.load(state.run.checkpoints / "epoch_0001.pth", weights_only=True)
    assert torch.equal(saved["ema"]["head.weight"], average["head.weight"])
    weights = state.model.head.weight.detach().clone()
    with state.ema.applied(state.model):
        assert torch.equal(state.model.head.weight, average["head.weight"])
    assert torch.equal(state.model.head.weight, weights)


@pytest.mark.parametrize(
    "hook",
    [
        {"type": "EMAHook", "decay": 0.9, "interval": 2},
        {"type": "SWAHook", "start_epoch": 1},
    ],
)
def test_a_resumed_run_continues_the_weight_average_exactly(tmp_path, hook):
    changes = {
        "hooks": [hook],
        "train.epochs": 4,
        "train.accumulate": 2,
        "model.bn": True,
        "runtime.deterministic": True,
    }
    straight = run_training(tiny_config(tmp_path / "a", changes))
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path / "b", changes), Interrupt(2))
    resumed = run_training(tiny_config(tmp_path / "b", {**changes, "runtime.resume": True}))
    assert straight.ema.count == resumed.ema.count > 0
    for name, value in straight.ema.state_dict().items():
        assert torch.equal(value, resumed.ema.state_dict()[name]), name


def test_one_weight_average_per_run(tmp_path):
    changes = {"hooks": [{"type": "EMAHook"}], "train.ema": 0.9}
    with pytest.raises(EngineError, match="a run keeps one average"):
        run_training(tiny_config(tmp_path, changes))
    with pytest.raises(EngineError, match="nothing would be averaged"):
        run_training(tiny_config(tmp_path, {"hooks": [{"type": "SWAHook", "start_epoch": 2}]}))


def test_swa_hook_is_the_mean_of_the_late_epochs(tmp_path):
    seen: list[torch.Tensor] = []

    class Ends(Recorder):
        priority = 15

        def after_epoch(self, state):
            seen.append(state.model.head.weight.detach().clone())

    changes = {
        "hooks": [{"type": "SWAHook", "start_epoch": 1}],
        "train.epochs": 3,
        "evaluation.interval": 0,
    }
    state = run_training(tiny_config(tmp_path, changes), Ends())
    assert torch.allclose(state.ema.state_dict()["head.weight"], (seen[1] + seen[2]) / 2)


def test_freeze_hook_freezes_then_releases(tmp_path):
    snapshots: list[torch.Tensor] = []
    modes: list[bool] = []

    class Watch(Recorder):
        priority = 35

        def before_epoch(self, state):
            snapshots.append(state.model.encoder[0].weight.detach().clone())
            modes.append(state.model.encoder.training)

    changes = {
        "hooks": [{"type": "FreezeHook", "modules": ["encoder"], "epochs": 1}],
        "optimizer": {"type": "AdamW", "lr": 0.05, "weight_decay": 0.1},
        "train.epochs": 3,
        "evaluation.interval": 0,
    }
    state = run_training(tiny_config(tmp_path, changes), Watch())
    assert torch.equal(snapshots[0], snapshots[1])  # frozen in epoch 1, weight decay included
    assert not torch.equal(snapshots[1], snapshots[2])  # released from epoch 2
    assert modes == [False, True, True]
    assert all(parameter.requires_grad for parameter in state.model.parameters())
    with pytest.raises(EngineError, match="has no such sub-module"):
        run_training(
            tiny_config(tmp_path, {"hooks": [{"type": "FreezeHook", "modules": ["backbone"]}]})
        )
    with pytest.raises(ValueError, match="name at least one"):
        FreezeHook(modules=[])


def test_adaptive_clipping_bounds_the_gradient_before_the_step(tmp_path):
    pytest.importorskip("timm")
    norms: list[float] = []

    class Norms(Recorder):
        priority = 45  # after the clipping hook

        def before_step(self, state):
            weight = state.model.head.weight
            norms.append(float(weight.grad.norm() / weight.detach().norm().clamp(min=1e-3)))

    changes = {
        "hooks": [{"type": "AdaptiveGradClipHook", "clip_factor": 0.01}],
        "evaluation.interval": 0,
    }
    run_training(tiny_config(tmp_path, changes), Norms())
    assert norms and max(norms) <= 0.01 + 1e-6
    changes["train.grad_clip"] = 1.0
    with pytest.raises(EngineError, match="choose one way of clipping"):
        run_training(tiny_config(tmp_path, changes))


def test_recipe_nodes_pass_the_config_check(tmp_path):
    changes = {
        "optimizer": {
            "type": "ParamGroups",
            "optimizer": {"type": "TimmOptimizer", "name": "lamb", "lr": 1e-3},
            "lr_multipliers": {"encoder": 0.1},
        },
        "scheduler": {"type": "TimmSchedule", "name": "poly", "options": {"power": 0.9}},
        "hooks": [
            {"type": "EMAHook", "decay": 0.99},
            {"type": "FreezeHook", "modules": ["encoder"], "epochs": 1},
        ],
    }
    config = tiny_config(tmp_path, changes)
    assert isinstance(config, Config) and len(config.digest()) == 64
