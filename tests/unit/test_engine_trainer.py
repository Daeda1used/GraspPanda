"""The training loop: optimisation, accumulation, clipping, schedules, EMA, NaN guard, logs."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from grasppanda.config import ConfigError
from grasppanda.engine.errors import EngineError, NonFiniteLossError
from grasppanda.engine.optim import (
    ModelEMA,
    build_optimizers,
    build_schedulers,
    clip_gradients,
    scheduler_steps,
)
from grasppanda.engine.run import start_run
from grasppanda.engine.trainer import Trainer, total_loss
from grasppanda.structures import SampleKey

from .test_engine_tiny import (
    Recorder,
    TinyDetector,
    run_training,
    single_thread,  # noqa: F401  (autouse fixture)
    tiny_config,
)


def weights(state) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in state.model.state_dict().items()}


# ------------------------------------------------------------------ optimisation


def test_loss_goes_down(tmp_path):
    recorder = Recorder()
    config = tiny_config(tmp_path, {"train.epochs": 15, "evaluation.interval": 0})
    state = run_training(config, recorder)
    first, last = recorder.losses[:4], recorder.losses[-4:]
    assert len(recorder.losses) == 15 * 4 == state.iteration
    assert sum(last) / 4 < 0.2 * sum(first) / 4, (first, last)
    assert state.epoch == 15 and state.is_last_epoch


def test_training_is_repeatable_and_depends_on_the_seed(tmp_path):
    changes = {"model.dropout": 0.3, "data.train.dataset.jitter": 0.05, "evaluation.interval": 0}
    first = weights(run_training(tiny_config(tmp_path / "a", changes)))
    again = weights(run_training(tiny_config(tmp_path / "b", changes)))
    other = weights(run_training(tiny_config(tmp_path / "c", {**changes, "runtime.seed": 1})))
    assert all(torch.equal(first[name], again[name]) for name in first)
    assert any(not torch.equal(first[name], other[name]) for name in first)


def test_gradient_accumulation_equals_a_larger_batch(tmp_path):
    """Two batches of 2 with ``accumulate: 2`` take the same steps as one batch of 4."""
    base = {"evaluation.interval": 0, "train.epochs": 3, "optimizer.lr": 0.2}
    small = run_training(
        tiny_config(tmp_path / "a", {**base, "data.train.batch_size": 2, "train.accumulate": 2})
    )
    large = run_training(
        tiny_config(tmp_path / "b", {**base, "data.train.batch_size": 4, "train.accumulate": 1})
    )
    plain = run_training(tiny_config(tmp_path / "c", {**base, "data.train.batch_size": 2}))
    assert small.iteration == 12 and large.iteration == 6
    a, b, c = weights(small), weights(large), weights(plain)
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-6), name
    assert any(not torch.allclose(a[name], c[name], atol=1e-4) for name in a)


def test_accumulation_steps_on_the_last_batch_of_an_epoch(tmp_path):
    """4 batches with ``accumulate: 3``: a step after batch 3 and one for the remainder."""
    config = tiny_config(tmp_path, {"train.accumulate": 3, "evaluation.interval": 0})
    with start_run(config, "train") as run:
        trainer = Trainer(run)
        steps = []
        optimizer = trainer.state.optimizers["main"]
        optimizer.register_step_post_hook(lambda *_: steps.append(trainer.state.inner))
        trainer.train()
    assert steps == [3, 4, 3, 4]
    assert all(p.grad is None for p in trainer.state.model.parameters())


def one_step_update(tmp_path, name: str, **train) -> torch.Tensor:
    """The parameter change of a single SGD step with learning rate 1: minus the applied gradient."""
    changes = {
        "data.train": {
            "dataset": {"type": "TinyScenes", "frames": 2},
            "batch_size": 2,
            "num_workers": 0,
        },
        "train.epochs": 1,
        "optimizer.lr": 1.0,
        "evaluation.interval": 0,
        **{f"train.{key}": value for key, value in train.items()},
    }
    config = tiny_config(tmp_path / name, changes)
    with start_run(config, "train") as run:
        trainer = Trainer(run)
        before = torch.cat([p.detach().flatten().clone() for p in trainer.state.model.parameters()])
        trainer.train()
        after = torch.cat([p.detach().flatten() for p in trainer.state.model.parameters()])
    return after - before


def test_gradient_clipping_by_norm_and_by_value(tmp_path):
    free = one_step_update(tmp_path, "free")
    # The unclipped gradient is large enough for both limits to bite.
    assert free.norm() > 0.2 and free.abs().max() > 0.1
    by_norm = one_step_update(tmp_path, "norm", grad_clip=0.1, grad_clip_mode="norm")
    assert by_norm.norm().item() == pytest.approx(0.1, rel=1e-4)
    assert torch.allclose(by_norm, free * (0.1 / free.norm()), atol=1e-6)
    by_value = one_step_update(tmp_path, "value", grad_clip=0.05, grad_clip_mode="value")
    assert torch.allclose(by_value, free.clamp(-0.05, 0.05), atol=1e-6)
    assert by_value.abs().max().item() == pytest.approx(0.05, rel=1e-5)
    loose = one_step_update(tmp_path, "loose", grad_clip=1e6)
    assert torch.allclose(loose, free, atol=1e-7)


def test_clip_gradients_reports_the_norm_before_clipping():
    layer = nn.Linear(2, 1)
    layer.weight.grad = torch.tensor([[3.0, 4.0]])
    layer.bias.grad = torch.zeros(1)
    assert clip_gradients(layer.parameters(), 1.0, "norm").item() == pytest.approx(5.0)
    assert layer.weight.grad.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert clip_gradients(layer.parameters(), None) is None
    with pytest.raises(EngineError, match="grad_clip_mode"):
        clip_gradients(layer.parameters(), 1.0, "bogus")


# --------------------------------------------------------------------- NaN guard


def test_nan_guard_names_the_term_the_iteration_and_the_frames(tmp_path):
    config = tiny_config(tmp_path, {"model.nan_at": 3, "evaluation.interval": 0})
    with pytest.raises(NonFiniteLossError) as failure:
        run_training(config)
    message = str(failure.value)
    assert "loss/aux = nan" in message and "loss/fit" not in message
    assert "epoch 1, iteration 3" in message and "tiny/scene_0000/cam/" in message


def test_total_loss_sums_loss_terms_only():
    outputs = {
        "loss/a": torch.tensor(1.0),
        "loss/b": torch.tensor(2.5),
        "metric": torch.tensor(9.0),
    }
    assert total_loss(outputs, "here").item() == pytest.approx(3.5)
    outputs["metric"] = torch.tensor(float("nan"))  # a metric is logged, never checked
    assert total_loss(outputs, "here").item() == pytest.approx(3.5)
    with pytest.raises(EngineError, match="no entry starts with 'loss/'"):
        total_loss({"metric": torch.tensor(1.0)}, "here")
    outputs["loss/b"] = torch.tensor(float("inf"))
    with pytest.raises(NonFiniteLossError, match=r"loss/b = inf.*tiny/scene_0001/cam/0002"):
        total_loss(outputs, "here", [SampleKey("tiny", 1, "cam", 2)])


# -------------------------------------------------------------------- schedulers


def test_scheduler_steps_per_epoch_by_default(tmp_path):
    recorder = Recorder()
    changes = {"scheduler": {"type": "ExponentialLR", "gamma": 0.5}, "train.epochs": 3}
    run_training(tiny_config(tmp_path, {**changes, "evaluation.interval": 0}), recorder)
    assert recorder.rates == pytest.approx([0.1] * 4 + [0.05] * 4 + [0.025] * 4)


def test_scheduler_can_step_per_iteration_and_gets_the_total_injected(tmp_path):
    recorder = Recorder()
    changes = {
        "scheduler": {"type": "OneCycleLR", "max_lr": 0.5},  # no total_steps in the config
        "train.scheduler_unit": "iteration",
        "train.epochs": 3,
        "evaluation.interval": 0,
    }
    state = run_training(tiny_config(tmp_path, changes), recorder)
    scheduler = state.schedulers["main"]
    assert scheduler.total_steps == 12 and scheduler.last_epoch == 12
    # The recorder reads the rate after each step, i.e. the one the next batch will use.
    assert max(recorder.rates) == pytest.approx(0.5, rel=2e-2)
    assert recorder.rates[0] < recorder.rates[2]  # warm-up
    assert recorder.rates[-2] == pytest.approx(0.5 / 25 / 1e4)  # used by the last batch


def test_total_steps_count_optimizer_steps_under_accumulation(tmp_path):
    assert scheduler_steps("epoch", 10, 7, 3) == 10
    assert scheduler_steps("iteration", 10, 7, 3) == 30  # ceil(7 / 3) steps per epoch
    changes = {
        "scheduler": {"type": "CosineAnnealingLR"},  # T_max is required by torch, injected here
        "train.scheduler_unit": "iteration",
        "train.accumulate": 3,
        "evaluation.interval": 0,
    }
    state = run_training(tiny_config(tmp_path, changes))
    assert state.schedulers["main"].T_max == 2 * math.ceil(4 / 3) == 4
    assert state.learning_rates()["lr"] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_arguments_stated_in_the_config_are_kept():
    model = nn.Linear(2, 1)
    optimizers = build_optimizers(model, {"type": "SGD", "lr": 0.1})
    stated = build_schedulers({"type": "CosineAnnealingLR", "T_max": 7}, optimizers, total_steps=99)
    assert stated["main"].T_max == 7
    injected = build_schedulers({"type": "PolynomialLR", "power": 2.0}, optimizers, total_steps=99)
    assert injected["main"].total_iters == 99
    assert build_schedulers(None, optimizers, total_steps=99) == {}
    with pytest.raises(EngineError, match="ReduceLROnPlateau is not supported"):
        build_schedulers({"type": "ReduceLROnPlateau"}, optimizers, total_steps=99)


def test_a_missing_required_scheduler_argument_is_still_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="StepLR requires 'step_size'"):
        tiny_config(tmp_path, {"scheduler": {"type": "StepLR"}})


# -------------------------------------------------------------------- optimizers


def test_detector_can_configure_parameter_groups_and_several_optimizers():
    class Grouped(TinyDetector):
        def configure_optimizers(self, optimizer):
            head = {"params": self.head.parameters(), "lr": optimizer["lr"] * 10}
            return [{"params": self.encoder.parameters()}, head]

    class Split(TinyDetector):
        def configure_optimizers(self, optimizer):
            return {
                "encoder": self.encoder.parameters(),
                "head": torch.optim.SGD(self.head.parameters(), lr=0.5),
            }

    node = {"type": "Adam", "lr": 0.01}
    default = build_optimizers(TinyDetector(), node)
    assert list(default) == ["main"] and len(default["main"].param_groups) == 1
    grouped = build_optimizers(Grouped(), node)["main"]
    assert [group["lr"] for group in grouped.param_groups] == pytest.approx([0.01, 0.1])
    split = build_optimizers(Split(), node)
    assert isinstance(split["encoder"], torch.optim.Adam) and isinstance(
        split["head"], torch.optim.SGD
    )
    frozen = TinyDetector()
    frozen.requires_grad_(False)
    with pytest.raises(EngineError, match="no parameters"):
        build_optimizers(frozen, node)


def test_the_loop_steps_every_optimizer(tmp_path, monkeypatch):
    def configure(self, optimizer):
        return {"encoder": self.encoder.parameters(), "head": self.head.parameters()}

    monkeypatch.setattr(TinyDetector, "configure_optimizers", configure, raising=False)
    changes = {"scheduler": {"type": "ExponentialLR", "gamma": 0.5}, "evaluation.interval": 0}
    config = tiny_config(tmp_path, changes)
    with start_run(config, "train") as run:
        trainer = Trainer(run)
        before = weights(trainer.state)
        state = trainer.train()
    after = weights(state)
    assert set(state.optimizers) == set(state.schedulers) == {"encoder", "head"}
    assert state.learning_rates() == pytest.approx({"lr/encoder": 0.025, "lr/head": 0.025})
    assert all(not torch.equal(before[name], after[name]) for name in before)


# --------------------------------------------------------------------------- EMA


def test_ema_follows_the_documented_update_and_can_be_swapped_in():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.9)
    start = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[1].num_batches_tracked.add_(5)
    ema.update(model)
    assert torch.allclose(ema.average["0.weight"], start["0.weight"] + 0.1)
    assert ema.average["1.num_batches_tracked"].item() == 5  # integer buffers are copied
    live = model[0].weight.clone()
    with ema.applied(model):
        assert torch.allclose(model[0].weight, start["0.weight"] + 0.1)
    assert torch.equal(model[0].weight, live)
    with pytest.raises(EngineError, match="decay in"):
        ModelEMA(model, decay=1.0)


def test_ema_is_updated_once_per_optimizer_step(tmp_path):
    changes = {"train.ema": 0.5, "train.accumulate": 2, "train.epochs": 1, "evaluation.interval": 0}
    config = tiny_config(tmp_path, changes)
    with start_run(config, "train") as run:
        trainer = Trainer(run)
        expected = trainer.state.ema.average["head.weight"].clone()
        snapshots = []
        trainer.state.optimizers["main"].register_step_post_hook(
            lambda *_: snapshots.append(trainer.state.model.head.weight.detach().clone())
        )
        state = trainer.train()
    assert len(snapshots) == 2
    for snapshot in snapshots:
        expected = 0.5 * expected + 0.5 * snapshot
    assert torch.allclose(state.ema.average["head.weight"], expected, atol=1e-7)
    assert not torch.equal(state.ema.average["head.weight"], state.model.head.weight)


# ------------------------------------------------------------------- switches


def test_amp_and_sync_bn_switches_run_on_one_cpu_process(tmp_path):
    """On a CPU autocast uses bfloat16 and the scaler is off; sync-BN needs several processes."""
    changes = {"train.amp": True, "train.sync_bn": True, "model.bn": True, "evaluation.interval": 0}
    recorder = Recorder()
    state = run_training(tiny_config(tmp_path, changes), recorder)
    assert not state.scaler.is_enabled()
    assert not any(isinstance(m, nn.SyncBatchNorm) for m in state.model.modules())
    assert all(math.isfinite(loss) for loss in recorder.losses)
    # A detector that does not declare `amp_ready` is refused before anything runs.
    from grasppanda.registry import DETECTORS

    tiny = DETECTORS.get("TinyDetector")
    try:
        tiny.amp_ready = False
        with pytest.raises(ConfigError, match="mixed precision is not supported by TinyDetector"):
            run_training(tiny_config(tmp_path / "refused", changes), Recorder())
    finally:
        tiny.amp_ready = True


def test_deterministic_switch_is_applied_and_restored(tmp_path):
    assert not torch.are_deterministic_algorithms_enabled()
    config = tiny_config(tmp_path, {"runtime.deterministic": True, "evaluation.interval": 0})
    with start_run(config, "train") as run:
        Trainer(run)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
    assert not torch.are_deterministic_algorithms_enabled()


def test_training_data_must_carry_supervision(tmp_path, monkeypatch):
    from grasppanda.structures import Batch

    from .test_engine_tiny import TinyScenes

    original = TinyScenes.__getitem__
    monkeypatch.setattr(
        TinyScenes, "__getitem__", lambda self, i: Batch(original(self, i).observation)
    )
    with pytest.raises(EngineError, match="without supervision"):
        run_training(tiny_config(tmp_path, {"evaluation.interval": 0}))


# ----------------------------------------------------------------------- logging


def test_log_reports_losses_metrics_rate_timings_and_eta(tmp_path):
    state = run_training(tiny_config(tmp_path, {"train.grad_clip": 10.0, "evaluation.interval": 0}))
    lines = (state.run.dir / "log.txt").read_text().splitlines()
    progress = [line for line in lines if "epoch 1/2  iter 2/4" in line]
    assert len(progress) == 1, lines
    for token in (
        "loss ",
        "loss/aux ",
        "loss/fit ",
        "abs_error ",
        "lr 1.000e-01",
        "grad_norm ",
        "data ",
        "iter ",
        "eta 0:00:",
    ):
        assert token in progress[0], (token, progress[0])
    assert sum("iter 4/4" in line for line in lines) == 2  # the last batch of every epoch
    assert "mem " not in progress[0]  # GPU memory is reported on CUDA only
