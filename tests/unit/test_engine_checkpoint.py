"""Checkpoints: exact resume, refusal rules, weights-only loading, retention and best selection."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from grasppanda import __version__
from grasppanda.engine.checkpoint import atomic_save, file_sha256, load_weights, read_checkpoint
from grasppanda.engine.errors import EngineError, ResumeError, WeightMismatchError
from grasppanda.engine.hooks import Hook
from grasppanda.engine.seeding import preserved_rng, rng_state, set_rng_state

from .test_engine_tiny import (
    Interrupt,
    Recorder,
    Stop,
    TinyDetector,
    run_training,
    single_thread,  # noqa: F401  (autouse fixture)
    tiny_config,
)

#: A run whose every stateful part matters: dropout and jitter consume random numbers, Adam
#: and batch norm carry buffers, the one-cycle schedule depends on the step count, and the
#: weight average, clipping and accumulation are on.
STATEFUL = {
    "model.dropout": 0.2,
    "model.bn": True,
    "data.train.dataset.jitter": 0.02,
    "optimizer": {"type": "Adam", "lr": 0.01},
    "scheduler": {"type": "OneCycleLR", "max_lr": 0.05},
    "train.scheduler_unit": "iteration",
    "train.epochs": 4,
    "train.ema": 0.9,
    "train.grad_clip": 1.0,
    "train.accumulate": 2,
    "evaluation.save_best": "raw/AP",
    "runtime.deterministic": True,
}


def assert_same_state(first, second) -> None:
    for name, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name]), name
    for name, value in first.ema.average.items():
        assert torch.equal(value, second.ema.average[name]), f"ema {name}"
    one, two = first.optimizers["main"].state_dict(), second.optimizers["main"].state_dict()
    for left, right in zip(one["state"].values(), two["state"].values()):
        for key in left:
            assert torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])), key
    assert one["param_groups"][0]["lr"] == two["param_groups"][0]["lr"]
    assert first.schedulers["main"].last_epoch == second.schedulers["main"].last_epoch
    assert (first.epoch, first.iteration, first.best) == (
        second.epoch,
        second.iteration,
        second.best,
    )


class Reseed(Hook):
    """Throws away the restored random state, as a resume without it would."""

    def before_run(self, state) -> None:
        torch.manual_seed(12345)


# ------------------------------------------------------------------------ resume


def test_resume_gives_the_same_weights_as_an_uninterrupted_run(tmp_path):
    """4 epochs straight == 2 epochs, killed, resumed for 2 more: bit for bit, on CPU."""
    straight_losses, resumed_losses = Recorder(), Recorder()
    straight = run_training(tiny_config(tmp_path / "a", STATEFUL), straight_losses)

    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path / "b", STATEFUL), Interrupt(2))
    resumed = run_training(
        tiny_config(tmp_path / "b", {**STATEFUL, "runtime.resume": True}), resumed_losses
    )

    assert straight.epoch == resumed.epoch == 4 and straight.iteration == resumed.iteration == 16
    assert_same_state(straight, resumed)
    assert resumed_losses.losses == straight_losses.losses[8:]  # epochs 3 and 4, exactly
    assert len(list((tmp_path / "b/work").iterdir())) == 1  # continued in the same run directory
    log = (resumed.run.dir / "log.txt").read_text()
    assert "resumed" in log and "epoch 1/4" in log and "epoch 4/4" in log
    assert len(list((resumed.run.dir / "resumes").iterdir())) == 1  # its snapshot, kept beside
    assert json.loads((resumed.run.dir / "run.json").read_text())["last_resume"]["checkpoint"]


def test_resume_is_only_exact_because_the_random_state_is_restored(tmp_path):
    """Negative control for the test above: reseeding after the restore changes the result."""
    straight = run_training(tiny_config(tmp_path / "a", STATEFUL))
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path / "b", STATEFUL), Interrupt(2))
    reseeded = run_training(
        tiny_config(tmp_path / "b", {**STATEFUL, "runtime.resume": True}), Reseed()
    )
    left, right = straight.model.state_dict(), reseeded.model.state_dict()
    assert any(not torch.equal(left[name], right[name]) for name in left)


@pytest.mark.slow
def test_resume_is_exact_with_worker_processes(tmp_path):
    """Augmentation drawn inside data-loader workers is seeded per (seed, rank, epoch)."""
    changes = {**STATEFUL, "data.train.num_workers": 2, "data.train.dataset.jitter": 0.05}
    straight = run_training(tiny_config(tmp_path / "a", changes))
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path / "b", changes), Interrupt(2))
    resumed = run_training(tiny_config(tmp_path / "b", {**changes, "runtime.resume": True}))
    assert_same_state(straight, resumed)
    in_process = run_training(tiny_config(tmp_path / "c", {**changes, "data.train.num_workers": 0}))
    left, right = straight.model.state_dict(), in_process.model.state_dict()
    assert any(not torch.equal(left[name], right[name]) for name in left)  # the jitter is real


def test_checkpoint_holds_everything_resume_needs_and_no_pickled_objects(tmp_path):
    state = run_training(tiny_config(tmp_path, STATEFUL))
    path = state.run.checkpoints / "epoch_0004.pth"
    payload = torch.load(
        path, map_location="cpu", weights_only=True
    )  # tensors and plain values only
    assert set(payload) == {
        "model", "optimizers", "schedulers", "scaler", "ema", "rng", "epoch", "iteration",
        "best", "config_hash", "version",
    }  # fmt: skip
    assert payload["epoch"] == 4 and payload["iteration"] == 16
    assert payload["config_hash"] == state.config.digest() and payload["version"] == __version__
    assert set(payload["optimizers"]) == set(payload["schedulers"]) == {"main"}
    assert (
        payload["schedulers"]["main"]["last_epoch"] == 8
        and payload["ema"].keys() == payload["model"].keys()
    )
    assert len(payload["rng"]) == 1 and set(payload["rng"][0]) == {
        "python",
        "numpy",
        "torch",
        "cuda",
    }
    assert payload["best"] == state.best and payload["best"]["metric"] == "raw/AP"
    assert not list(state.run.checkpoints.glob("*.tmp"))


def test_resume_refuses_another_config_unless_allowed(tmp_path):
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path, {"train.epochs": 3}), Interrupt(1))
    changed = {"train.epochs": 3, "optimizer.lr": 0.05, "runtime.resume": True}
    with pytest.raises(ResumeError, match="different experiment") as refusal:
        run_training(tiny_config(tmp_path, changed))
    assert "runtime.allow_config_mismatch=true" in str(refusal.value)
    state = run_training(tiny_config(tmp_path, {**changed, "runtime.allow_config_mismatch": True}))
    assert state.epoch == 3
    assert (
        "Continuing because runtime.allow_config_mismatch is set"
        in (state.run.dir / "log.txt").read_text()
    )


def test_allowing_a_mismatch_is_not_itself_a_mismatch(tmp_path):
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path), Interrupt(1))
    allowed = {"runtime.resume": True, "runtime.allow_config_mismatch": True}
    state = run_training(tiny_config(tmp_path, allowed))
    assert (
        state.epoch == 2 and "different experiment" not in (state.run.dir / "log.txt").read_text()
    )


def files_of(directory):
    """Every file below ``directory`` with its bytes and modification time."""
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_resume_into_another_work_dir_starts_a_new_run_and_leaves_the_original_alone(tmp_path):
    """Moving the work directory is not a different experiment, but it is another directory."""
    straight_losses, resumed_losses = Recorder(), Recorder()
    straight = run_training(tiny_config(tmp_path / "a", STATEFUL), straight_losses)
    with pytest.raises(Stop):
        run_training(
            tiny_config(tmp_path / "b", {**STATEFUL, "runtime.run_id": "one"}), Interrupt(2)
        )
    original = tmp_path / "b/work/one"
    checkpoint = original / "checkpoints/epoch_0002.pth"
    before = files_of(original)

    moved = {
        **STATEFUL,
        "runtime.work_dir": str(tmp_path / "elsewhere"),
        "runtime.resume": str(checkpoint),
    }
    resumed = run_training(tiny_config(tmp_path / "b", moved), resumed_losses)

    assert files_of(original) == before and not (original / "resumes").exists()
    (new,) = (tmp_path / "elsewhere").iterdir()
    assert resumed.run.dir == new and resumed.epoch == straight.epoch == 4
    assert resumed_losses.losses == straight_losses.losses[8:]  # epochs 3 and 4, exactly
    assert_same_state(straight, resumed)
    facts = json.loads((new / "run.json").read_text())
    assert facts["resumed_from"] == str(checkpoint)
    assert facts["config_hash"] == straight.config.digest()
    assert {"epoch_0003.pth", "epoch_0004.pth"} <= {p.name for p in (new / "checkpoints").iterdir()}
    log = (new / "log.txt").read_text()
    assert f"resuming {checkpoint} in a new run directory" in log and str(original) in log


def test_resume_with_a_run_id_of_its_own_starts_that_run(tmp_path):
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path, {"runtime.run_id": "one"}), Interrupt(1))
    before = files_of(tmp_path / "work/one")
    checkpoint = tmp_path / "work/one/checkpoints/epoch_0001.pth"
    state = run_training(
        tiny_config(tmp_path, {"runtime.run_id": "two", "runtime.resume": str(checkpoint)})
    )
    assert state.epoch == 2 and state.run.dir == tmp_path / "work/two"
    assert files_of(tmp_path / "work/one") == before


def test_a_run_continued_under_another_config_never_writes_into_the_original(tmp_path):
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path, {"train.epochs": 3}), Interrupt(1))
    (original,) = (tmp_path / "work").iterdir()
    before = files_of(original)
    changed = {"train.epochs": 3, "optimizer.lr": 0.05, "runtime.resume": True}

    with pytest.raises(ResumeError, match="different experiment"):
        run_training(tiny_config(tmp_path, changed))
    assert list((tmp_path / "work").iterdir()) == [original]  # refused before writing anything
    allowed = {**changed, "runtime.allow_config_mismatch": True}
    with pytest.raises(ResumeError, match=f"runtime.run_id '{original.name}' names the directory"):
        run_training(tiny_config(tmp_path, {**allowed, "runtime.run_id": original.name}))
    state = run_training(tiny_config(tmp_path, allowed))

    assert state.epoch == 3 and state.run.dir.parent == original.parent != state.run.dir
    assert files_of(original) == before and not (original / "resumes").exists()
    assert json.loads((state.run.dir / "run.json").read_text())["resumed_from"] == str(
        original / "checkpoints/epoch_0001.pth"
    )
    log = (state.run.dir / "log.txt").read_text()
    assert "because the config differs from the one the checkpoint was written under" in log


def test_resume_from_a_file_outside_a_run_starts_a_new_run_directory(tmp_path):
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path), Interrupt(1))
    (source,) = (tmp_path / "work").glob("*/checkpoints/epoch_0001.pth")
    loose = tmp_path / "loose.pth"
    loose.write_bytes(source.read_bytes())
    state = run_training(
        tiny_config(tmp_path, {"runtime.resume": str(loose), "runtime.run_id": "new"})
    )
    assert state.epoch == 2 and state.run.dir == tmp_path / "work/new"
    assert json.loads((state.run.dir / "run.json").read_text())["resumed_from"] == str(loose)


def test_resume_fails_loudly_when_there_is_nothing_to_resume(tmp_path):
    with pytest.raises(ResumeError, match="no checkpoint below"):
        run_training(tiny_config(tmp_path, {"runtime.resume": True}))
    with pytest.raises(ResumeError, match="no such checkpoint"):
        run_training(tiny_config(tmp_path, {"runtime.resume": str(tmp_path / "nope.pth")}))
    weights_only = atomic_save({"model": TinyDetector().state_dict()}, tmp_path / "weights.pth")
    with pytest.raises(ResumeError, match="not a training checkpoint"):
        run_training(tiny_config(tmp_path, {"runtime.resume": str(weights_only)}))


def test_resuming_a_finished_run_does_nothing(tmp_path):
    done = run_training(tiny_config(tmp_path))
    again = run_training(tiny_config(tmp_path, {"runtime.resume": True}), recorder := Recorder())
    assert again.epoch == 2 and recorder.losses == []
    assert all(
        torch.equal(v, again.model.state_dict()[k]) for k, v in done.model.state_dict().items()
    )


def test_a_run_directory_is_never_reused_by_a_fresh_run(tmp_path):
    run_training(tiny_config(tmp_path, {"runtime.run_id": "mine"}))
    with pytest.raises(EngineError, match="already holds a run"):
        run_training(tiny_config(tmp_path, {"runtime.run_id": "mine"}))
    first = run_training(tiny_config(tmp_path))  # generated ids never collide, even within a second
    second = run_training(tiny_config(tmp_path))
    assert first.run.dir != second.run.dir and first.run.dir.parent == second.run.dir.parent


def draws() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_rng_state_round_trip():
    state = rng_state()
    first = draws()
    set_rng_state(state)
    assert draws() == first


def test_preserved_rng_runs_a_block_under_its_own_seed_and_restores_the_state():
    state = rng_state()
    expected = draws()
    set_rng_state(state)
    with preserved_rng(7):
        inside = draws()
    assert draws() == expected  # as if the block had never drawn anything
    with preserved_rng(7):
        assert draws() == inside  # and the block itself is repeatable
    with preserved_rng(8):
        assert draws() != inside


# --------------------------------------------------------- retention and "best"


def test_interval_and_keep_last(tmp_path):
    changes = {
        "train.epochs": 7,
        "runtime.checkpoint": {"interval": 2, "keep_last": 2},
        "evaluation.interval": 0,
    }
    state = run_training(tiny_config(tmp_path, changes))
    # saved after epochs 2, 4, 6 and after the last one; the two newest survive
    assert sorted(p.name for p in state.run.checkpoints.iterdir()) == [
        "epoch_0006.pth",
        "epoch_0007.pth",
    ]
    everything = run_training(
        tiny_config(
            tmp_path / "all",
            {"train.epochs": 3, "runtime.checkpoint.keep_last": 0, "evaluation.interval": 0},
        )
    )
    assert len(list(everything.run.checkpoints.iterdir())) == 3
    final_only = run_training(
        tiny_config(
            tmp_path / "final",
            {"train.epochs": 3, "runtime.checkpoint.interval": 0, "evaluation.interval": 0},
        )
    )
    assert [p.name for p in final_only.run.checkpoints.iterdir()] == ["epoch_0003.pth"]


def test_best_checkpoint_follows_the_configured_metric(tmp_path):
    changes = {
        "train.epochs": 5,
        "evaluation.evaluator.script": [0.2, 0.5, 0.3, 0.5, 0.4],  # AP after epochs 1 .. 5
        "evaluation.save_best": "raw/AP",
        "runtime.checkpoint": {"interval": 1, "keep_last": 1},
    }
    state = run_training(tiny_config(tmp_path, changes))
    assert state.best == {"metric": "raw/AP", "value": 0.5, "epoch": 2}  # a tie does not replace it
    best = read_checkpoint(state.run.checkpoints / "best.pth")
    assert best["epoch"] == 2 and best["best"]["value"] == 0.5
    # epoch_0002.pth was pruned by keep_last; best.pth is its own file and survives
    assert sorted(p.name for p in state.run.checkpoints.iterdir()) == ["best.pth", "epoch_0005.pth"]
    log = (state.run.dir / "log.txt").read_text()
    assert log.count("best checkpoint:") == 2 and "raw/AP = 0.5000 at epoch 2" in log


def test_best_is_saved_even_when_the_epoch_is_not_a_checkpoint_epoch(tmp_path):
    changes = {
        "train.epochs": 4,
        "evaluation.evaluator.script": [0.1, 0.9, 0.2, 0.3],
        "evaluation.save_best": "raw/AP",
        "runtime.checkpoint.interval": 0,
    }
    state = run_training(tiny_config(tmp_path, changes))
    assert read_checkpoint(state.run.checkpoints / "best.pth")["epoch"] == 2
    assert sorted(p.name for p in state.run.checkpoints.iterdir()) == ["best.pth", "epoch_0004.pth"]


def test_best_is_restored_on_resume(tmp_path):
    changes = {
        "train.epochs": 4,
        "evaluation.save_best": "raw/AP",
        "evaluation.evaluator.script": [0.9, 0.1, 0.2, 0.3],
    }
    with pytest.raises(Stop):
        run_training(tiny_config(tmp_path, changes), Interrupt(2))
    # the resumed evaluator starts its script again: epochs 3 and 4 score 0.9 and 0.1
    resumed = run_training(tiny_config(tmp_path, {**changes, "runtime.resume": True}))
    assert resumed.best == {"metric": "raw/AP", "value": 0.9, "epoch": 1}  # 0.9 again is a tie


def test_save_best_must_name_a_reported_metric(tmp_path):
    with pytest.raises(EngineError, match=r"monitors 'raw/mAP'.*reported: raw/AP"):
        run_training(tiny_config(tmp_path / "a", {"evaluation.save_best": "raw/mAP"}))
    with pytest.raises(EngineError, match="no in-training evaluation"):
        run_training(
            tiny_config(
                tmp_path / "b", {"evaluation.save_best": "raw/AP", "evaluation.interval": 0}
            )
        )


# ------------------------------------------------------------------ weights only


def test_load_weights_reports_a_clean_load(tmp_path):
    source = TinyDetector(bn=True)
    path = atomic_save(
        {"model": source.state_dict(), "epoch": 3, "config_hash": "abc"}, tmp_path / "w.pth"
    )
    target = TinyDetector(bn=True)
    report = load_weights(target, path)
    assert report.clean and len(report.matched) == len(source.state_dict())
    assert report.meta == {"epoch": 3, "config_hash": "abc"}
    assert (
        f"{len(report.matched)} matched, 0 missing, 0 unexpected, 0 with a different shape"
        in report.render()
    )
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in source.state_dict().items())


def test_load_weights_refuses_mismatched_weights_and_says_which_keys(tmp_path):
    path = atomic_save({"model": TinyDetector(hidden=8, bn=True).state_dict()}, tmp_path / "w.pth")
    target = TinyDetector(hidden=16)  # other width, and no batch norm
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(WeightMismatchError) as failure:
        load_weights(target, path)
    message = str(failure.value)
    assert (
        "unexpected (in the file, not in the model)" in message
        and "encoder.1.running_mean" in message
    )
    assert "different shape" in message and "head.weight: file (1, 8), model (1, 16)" in message
    assert "Nothing was loaded" in message and "runtime.allow_weight_mismatch=true" in message
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())


def test_load_weights_lists_missing_keys(tmp_path):
    partial = {k: v for k, v in TinyDetector().state_dict().items() if not k.startswith("head.")}
    path = atomic_save(partial, tmp_path / "bare.pth")  # a bare state dict
    with pytest.raises(
        WeightMismatchError, match=r"missing \(in the model, not in the file\):\n    head.weight"
    ):
        load_weights(TinyDetector(), path)
    target = TinyDetector()
    head = target.head.weight.clone()
    report = load_weights(target, path, allow_mismatch=True)
    assert not report.clean and report.missing == ["head.weight", "head.bias"]
    assert torch.equal(target.head.weight, head)  # untouched
    assert torch.equal(target.encoder[0].weight, partial["encoder.0.weight"])  # loaded


def test_load_weights_understands_foreign_layouts_and_says_so(tmp_path):
    source = TinyDetector()
    wrapped = {f"module.{key}": value for key, value in source.state_dict().items()}
    path = atomic_save({"state_dict": wrapped, "epoch": 7}, tmp_path / "foreign.pth")
    target = TinyDetector()
    report = load_weights(target, path)
    assert report.clean and len(report.notes) == 2  # which entry was used; the stripped prefix
    assert torch.equal(target.head.weight, source.head.weight)
    garbage = atomic_save({"args": [1, 2], "stuff": "x"}, tmp_path / "garbage.pth")
    with pytest.raises(EngineError, match="cannot find weights.*top-level keys are args, stuff"):
        load_weights(TinyDetector(), garbage)
    with pytest.raises(EngineError, match="no such checkpoint"):
        load_weights(TinyDetector(), tmp_path / "missing.pth")


def test_load_weights_never_unpickles_arbitrary_objects(tmp_path):
    path = tmp_path / "unsafe.pth"
    torch.save({"model": TinyDetector().state_dict(), "args": Recorder()}, path)
    with pytest.raises(EngineError, match="weights-only"):
        load_weights(TinyDetector(), path)


def test_load_weights_can_select_the_averaged_weights(tmp_path):
    state = run_training(tiny_config(tmp_path, {"train.ema": 0.5, "evaluation.interval": 0}))
    path = state.run.checkpoints / "epoch_0002.pth"
    plain, averaged = TinyDetector(), TinyDetector()
    load_weights(plain, path)
    report = load_weights(averaged, path, kind="ema")
    assert "averaged weights" in report.render()
    assert torch.equal(plain.head.weight, state.model.head.weight)
    assert torch.equal(averaged.head.weight, state.ema.average["head.weight"])
    assert not torch.equal(plain.head.weight, averaged.head.weight)
    without = run_training(tiny_config(tmp_path / "b", {"evaluation.interval": 0}))
    with pytest.raises(EngineError, match="holds no averaged weights"):
        load_weights(TinyDetector(), without.run.checkpoints / "epoch_0002.pth", kind="ema")


def test_load_from_starts_a_new_training_from_those_weights(tmp_path):
    donor = run_training(tiny_config(tmp_path / "a", {"evaluation.interval": 0}))
    path = donor.run.checkpoints / "epoch_0002.pth"
    changes = {
        "runtime.load_from": str(path),
        "optimizer.lr": 0.0,
        "train.epochs": 1,
        "evaluation.interval": 0,
    }
    state = run_training(tiny_config(tmp_path / "b", changes))
    assert state.epoch == 1 and state.iteration == 4  # counters restart: weights only
    assert torch.equal(state.model.head.weight, donor.model.head.weight)  # lr 0 keeps them
    assert "8 matched, 0 missing" not in (state.run.dir / "log.txt").read_text()
    assert "4 matched, 0 missing, 0 unexpected" in (state.run.dir / "log.txt").read_text()
    mismatch = {**changes, "model.hidden": 16}
    with pytest.raises(WeightMismatchError):
        run_training(tiny_config(tmp_path / "c", mismatch))


# ------------------------------------------------------------------------- files


def test_atomic_save_keeps_the_old_file_when_writing_fails(tmp_path, monkeypatch):
    path = atomic_save({"value": 1}, tmp_path / "file.pth")
    digest = file_sha256(path)

    def failing_save(payload, target):
        with open(target, "wb") as stream:
            stream.write(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        atomic_save({"value": 2}, path)
    assert file_sha256(path) == digest and read_checkpoint(path) == {"value": 1}
