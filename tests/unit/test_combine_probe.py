"""``grasppanda probe``: one verdict per config, and never more than was run.

The real path is exercised on the example project (a toy detector and a toy dataset that
train on a CPU in a second); the wiring check on tiny seed-based models and on a tiny
HGGD. What a probe says about real GraspNet-1Billion frames is the subject of
``tests/integration/test_combine_grid.py``.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import yaml

from grasppanda.cli.main import main
from grasppanda.combine.probe import (
    EXIT_STATUS,
    JUDGE_FIELDS,
    SCHEDULES,
    TREND_FLAGS,
    VERDICTS,
    ProbeReport,
    ProbeRequestError,
    _component,
    _fail,
    _judge,
    _phase,
    dead_parts,
    loss_trend,
    out_of_memory,
    peak_rates,
    probe,
)
from grasppanda.combine.synthetic import HISTOGRAM_KEY, random_batch, scene_batch
from grasppanda.config import Config
from grasppanda.models.assigners import scale_balance
from grasppanda.structures import COMPACT_LABELS

from . import test_swap_matrix as matrix
from .conftest import REPO

TOY = str(REPO / "projects/example_project/configs/toy_train.yaml")
needs_spconv = pytest.mark.skipif(
    importlib.util.find_spec("spconv") is None, reason="needs the spconv extra"
)


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def write(tmp_path: Path, name: str, content: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return str(path)


def tiny(detector: str, backbone: str, **replace: str) -> dict:
    """A tiny seed-based model on CPU: the swap matrix's stages around one encoder."""
    return {
        "model": matrix.method(detector, matrix.encoder(backbone), **replace),
        "optimizer": {"type": "Adam", "lr": 1e-3},
        "runtime": {"device": "cpu"},
    }


# ----------------------------------------------------------------------- the vocabulary


def test_the_verdicts_are_a_fixed_documented_vocabulary():
    assert list(VERDICTS) == ["runs", "runs-empty", "runs-unstable", "fails", "refused", "wired"]
    assert set(EXIT_STATUS) == set(VERDICTS)
    assert all(len(meaning) > 60 and meaning.endswith(".") for meaning in VERDICTS.values())
    assert EXIT_STATUS["runs"] == 0 and all(
        EXIT_STATUS[name] != 0 for name in ("runs-unstable", "fails", "refused")
    )
    # A run that predicted nothing still ran: a script that trains after a probe goes on.
    assert EXIT_STATUS["runs-empty"] == 0
    # What judge 2 adds is defined field by field, and every field is in the report.
    fields = set(ProbeReport().to_dict())
    assert set(JUDGE_FIELDS) <= fields and ProbeReport().judge == 2
    assert list(TREND_FLAGS) == ["flat", "rising", "zero_at_end", "constant"]
    # `wired` is success only when the wiring check is what was asked for.
    report = ProbeReport(verdict="wired")
    assert report.exit_status() == 3 and report.exit_status(synthetic_requested=True) == 0
    page = (REPO / "docs/user_guides/combinations.md").read_text(encoding="utf-8")
    # The guide defines every verdict in the words of the code, so the two cannot drift.
    for name, meaning in VERDICTS.items():
        assert f"| `{name}` | {meaning} |" in page, name


def recorded(**fields) -> ProbeReport:
    """A report as a clean run of twenty iterations leaves it, with ``fields`` changed."""
    report = ProbeReport(
        iterations_requested=20,
        iterations=20,
        real_seeds=[1.0] * 20,
        supervised_seeds=[0.5] * 20,
        grasps_per_frame=[12, 0],
    )
    for name, value in fields.items():
        setattr(report, name, value)
    return report


def test_the_verdict_follows_from_what_was_recorded():
    """Every branch of the judgement, on reports written by hand."""

    def judged(good: str = "runs", frames: int = 2, **fields) -> ProbeReport:
        report = recorded(**fields)
        _judge(report, good, "8 frames of the dataset", frames)
        return report

    clean = judged()
    assert clean.verdict == "runs"
    assert clean.detail == (
        "trained 20 iterations on 8 frames of the dataset and predicted on 2 frame(s), with "
        "finite losses and real seeds in every iteration."
    )
    # An iteration in which no point passed the proposal trained the proposal only.
    seedless = judged(real_seeds=[1.0, 0.0, 0.3, 0.0, *[1.0] * 16])
    assert seedless.verdict == "runs-unstable"
    assert "2 of 20 iterations had no real seed (iteration 2, 4)" in seedless.detail
    many = judged(real_seeds=[0.0] * 20)
    assert "(iteration 1, 2, 3, 4, 5, 6 ...)" in many.detail
    # Seeds, but none of them ever took part in the grasp losses.
    assert "no iteration supervised a seed" in judged(supervised_seeds=[0.0] * 20).detail
    assert judged(supervised_seeds=[0.0] * 19 + [0.01]).verdict == "runs"
    # A non-finite loss ends the training: said once, not also as "too few iterations".
    stopped = judged(iterations=7, nonfinite=["the loss is nan at iteration 8"])
    assert stopped.verdict == "runs-unstable"
    assert "the loss is nan at iteration 8" in stopped.detail and "only 7" not in stopped.detail
    assert "a weight is not finite" in judged(nonfinite=["a weight is not finite"]).detail
    # Less was run than was asked for: an empty loader, a dataset with a single frame.
    assert "only 3 of 20 iterations ran" in judged(iterations=3).detail
    assert judged(iterations=0, real_seeds=[], supervised_seeds=[]).verdict == "runs-unstable"
    one = judged(grasps_per_frame=[5])
    assert (
        one.verdict == "runs-unstable" and "only 1 of 2 frame(s) could be predicted" in one.detail
    )
    assert judged(grasps_per_frame=[]).verdict == "runs-unstable"
    # An empty prediction ran every stage: a verdict of its own, not `runs`.
    empty = judged(grasps_per_frame=[0, 0])
    assert empty.verdict == "runs-empty"
    assert "no grasp was predicted on any probed frame" in empty.detail
    assert "No grasp was predicted" not in clean.detail and "no grasp" not in clean.detail
    assert judged(grasps_per_frame=[0, 3]).verdict == "runs"  # one frame with a grasp
    # Unstable and empty: the instability is the verdict, the emptiness is said.
    both = judged(grasps_per_frame=[0, 0], real_seeds=[0.0, *[1.0] * 19])
    assert both.verdict == "runs-unstable" and "No grasp was predicted" in both.detail
    # A detector that reports no seed metrics is judged on what it does report.
    plain = judged(real_seeds=[], supervised_seeds=[])
    assert plain.verdict == "runs" and plain.detail.endswith("with finite losses.")
    # A part with trainable weights that never received a gradient is a dead stage.
    dead = judged(no_gradient=["backbone"])
    assert dead.verdict == "runs-unstable"
    assert "no gradient reached backbone in any iteration" in dead.detail
    # A sub-module without a gradient in a part that learned is reported, not judged.
    assert dead_parts(["backbone", "head.score_net", "view.bias"]) == ["backbone"]
    inner = judged(no_gradient=["head.score_net"])
    assert inner.verdict == "runs" and "no gradient" not in inner.detail
    # The wiring check has no unstable verdict: what would be unstable there fails; and
    # it has no empty one: on noise labels a prediction means nothing.
    assert judged("wired").verdict == "wired"
    assert (
        judged("wired", nonfinite=["predicted width of a frame is not finite"]).verdict == "fails"
    )
    assert judged("wired", grasps_per_frame=[0, 0]).verdict == "wired"


def test_the_trend_of_a_loss_term_counts_the_iterations_in_which_it_was_active():
    """Every flag on a series written by hand; see TREND_FLAGS for the definitions."""
    falling = loss_trend([4.0, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0])
    assert falling == {
        "active": 8,
        "first": 4.0,
        "last_quarter": 1.0,  # the last ceil(8 / 4) = 2 active values
        "fall": 0.75,
        "flags": [],
    }
    # A grasp term that is 0.0 until a seed is supervised starts where it is active.
    late = loss_trend([0.0, 0.0, None, 2.0, 1.0, 0.5])
    assert late["active"] == 3 and late["first"] == 2.0 and late["last_quarter"] == 0.5
    assert late["fall"] == 0.75 and late["flags"] == []
    # Moving less than 5 % of its first active value: flat, whatever happened between.
    assert loss_trend([1.0, 1.5, 0.5, 0.98])["flags"] == ["flat"]
    assert loss_trend([1.0, 1.5, 0.5, 0.94])["flags"] == []
    # The last half more than 10 % above the first half: rising.
    climbing = loss_trend([1.0, 1.2, 1.5, 2.0])
    assert climbing["flags"] == ["rising"] and climbing["fall"] == -1.0
    assert "rising" not in loss_trend([1.0, 1.0, 1.08, 1.08])["flags"]  # 8 % is not
    # Back where it started after a climb: flat and rising at once.
    detour = loss_trend([1.0, 0.6, 0.6, 1.0, 1.4, 1.4, 1.0, 1.0])
    assert detour["flags"] == ["flat", "rising"] and detour["fall"] == 0.0
    # Exactly 0.0, or not reported, at the end: the term no longer trains anything.
    stopped = loss_trend([0.8, 0.4, 0.0])
    assert stopped["flags"] == ["zero_at_end"] and stopped["active"] == 2
    assert loss_trend([0.8, 0.4, None])["flags"] == ["zero_at_end"]
    # Zero throughout: never active, no trend, and constant.
    off = loss_trend([0.0] * 5)
    assert off == {
        "active": 0,
        "first": None,
        "last_quarter": None,
        "fall": None,
        "flags": ["zero_at_end", "constant"],
    }
    # Identical values throughout: constant and flat; a negative term falls downwards.
    assert loss_trend([0.25] * 4)["flags"] == ["flat", "constant"]
    assert loss_trend([-1.0, -2.0, -3.0, -4.0])["fall"] == 3.0
    # One active iteration has no span: no trend flag, and nothing reported is no flag.
    assert loss_trend([0.0, 0.7])["flags"] == []
    assert loss_trend([None, None])["flags"] == [] and loss_trend([])["active"] == 0


def test_out_of_memory_is_told_apart_from_a_failure_of_the_config():
    report = ProbeReport()
    try:
        raise RuntimeError("CUDA out of memory. Tried to allocate 134.00 MiB.")
    except RuntimeError as error:
        _fail(report, error, "loss", "iteration 1")
    assert report.verdict == "fails" and out_of_memory(report)
    assert any("not about the combination" in note for note in report.notes)
    other = ProbeReport()
    try:
        raise ValueError("radius must be positive")
    except ValueError as error:
        _fail(other, error, "build")
    assert not out_of_memory(other) and other.notes == []


def test_a_request_that_cannot_be_run_is_an_error_not_a_verdict(monkeypatch, capsys):
    for wrong in ({"iters": 0}, {"iters": -5}, {"frames": 0}, {"predict_frames": 0}):
        with pytest.raises(ProbeRequestError, match="must be at least 1"):
            probe(TOY, **wrong)
    with pytest.raises(ProbeRequestError, match="workers must be 0 or more"):
        probe(TOY, workers=-1)
    with pytest.raises(ProbeRequestError, match="schedule must be one of recipe, constant"):
        probe(TOY, schedule="cosine")
    for arguments in (
        ["--iters", "0"],
        ["--predict-frames", "0"],
        ["--frames", "x"],
        ["--schedule", "cosine"],
    ):
        with pytest.raises(SystemExit) as stop:
            main(["probe", TOY, *arguments])
        assert stop.value.code == 2
    assert "must be at least 1, got 0" in capsys.readouterr().err

    # A split the dataset does not have is a wrong request too, not a reason to fall back
    # to the synthetic check.
    def no_such_split(cfg, section):
        raise ValueError("'bogus' is not a split of toy. Splits: train, sample.")

    monkeypatch.setattr("grasppanda.engine.builders.build_dataset", no_such_split)
    with pytest.raises(ProbeRequestError, match="'bogus' is not a split of toy"):
        probe(TOY, split="bogus")
    assert main(["probe", TOY, "--split", "bogus"]) == 2
    assert "Choose the frames with `--split`" in capsys.readouterr().err


# ---------------------------------------------------------------------------- real path


def test_a_launch_failure_of_the_device_is_told_apart_too():
    report = ProbeReport()
    try:
        raise RuntimeError("cuda failed with error 719 unspecified launch failure")
    except RuntimeError as error:
        _fail(report, error, "backward", "iteration 1")
    assert report.verdict == "fails"
    assert any("probe again" in note and "launch failure" in note for note in report.notes)


def test_the_real_probe_notes_a_part_that_received_no_gradient(monkeypatch):
    """A stage whose output does not depend on its weights trains "without error";
    the probe on real frames says which part learned nothing."""
    from grasppanda.registry import DETECTORS

    toy = DETECTORS.get("ToyDetector")
    original = toy.__init__

    def with_idle_part(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.idle = torch.nn.Linear(2, 2)  # trainable, and never used

    monkeypatch.setattr(toy, "__init__", with_idle_part)
    report = probe(TOY, iters=3, predict_frames=1)
    assert report.no_gradient == ["idle"]
    assert report.verdict == "runs-unstable" and "no gradient reached idle" in report.detail


# ----------------------------------------------------------- judge 2: does it learn?


def test_a_module_whose_output_is_constant_is_flagged(monkeypatch):
    """An encoder whose output depends neither on its input nor on its weights trains
    "without error" and the head behind it learns; the probe names the encoder as reached
    by no gradient and unchanged."""
    from grasppanda.registry import BACKBONES
    from grasppanda.structures import PointFeatures

    def constant(self, points, feats):
        return PointFeatures(self.mlp(feats) * 0.0 + 1.0, points)

    monkeypatch.setattr(BACKBONES.get("ToyBackbone"), "forward", constant)
    report = probe(TOY, iters=4, predict_frames=1)
    assert report.no_gradient == ["backbone"] and report.unchanged == ["backbone"]
    assert report.units_checked == 3 and report.parts_checked == 2  # backbone.mlp, head.*
    assert report.verdict == "runs-unstable" and "no gradient reached backbone" in report.detail


def test_a_sub_module_without_a_gradient_is_named_at_depth_two(monkeypatch, capsys):
    """One layer of the head is cut off from the loss; the rest of the head learns. The
    probe names the layer, not the head, and reports it without changing the verdict."""
    from grasppanda.registry import HEADS

    toy = HEADS.get("ToyHead")
    original = toy.forward

    def detached_width(self, groups):
        raw = original(self, groups)
        return {**raw, "width": raw["width"].detach()}

    monkeypatch.setattr(toy, "forward", detached_width)
    report = probe(TOY, iters=4, predict_frames=1)
    assert report.no_gradient == ["head.width"] and report.unchanged == []
    assert report.verdict == "runs"
    assert main(["probe", TOY, "--iters", "4", "--predict-frames", "1"]) == 0
    assert "gradient    none reached head.width (of 3 units)" in capsys.readouterr().out


def test_a_head_term_that_does_not_move_is_flagged_flat(monkeypatch, capsys):
    """Every iteration's losses are kept, and every term gets its trend: a term that cannot
    move is flat (and constant), a term that is 0.0 throughout is never active."""
    from grasppanda.registry import HEADS

    toy = HEADS.get("ToyHead")
    original = toy.loss

    def with_dead_terms(self, raw, cell, width):
        terms = original(self, raw, cell, width)
        terms["loss/frozen"] = self.width.bias.sum() * 0.0 + 0.25  # cannot move
        terms["loss/off"] = raw["width"].sum() * 0.0  # 0.0 in every iteration
        return terms

    monkeypatch.setattr(toy, "loss", with_dead_terms)
    report = probe(TOY, iters=4, predict_frames=1)
    assert report.verdict == "runs"  # reported, not judged
    series = report.losses_per_iteration
    assert set(series) == {"loss/score", "loss/width", "loss/frozen", "loss/off", "loss"}
    assert all(len(values) == 4 for values in series.values())
    assert series["loss/frozen"] == [0.25] * 4 and series["loss/off"] == [0.0] * 4
    for index in range(4):  # the total is the sum of the terms, to the digits kept
        terms = sum(series[name][index] for name in series if name != "loss")
        assert series["loss"][index] == pytest.approx(terms, rel=1e-3)
    frozen, off = report.loss_trends["loss/frozen"], report.loss_trends["loss/off"]
    assert frozen == {
        "active": 4,
        "first": 0.25,
        "last_quarter": 0.25,
        "fall": 0.0,
        "flags": ["flat", "constant"],
    }
    assert off["active"] == 0 and off["fall"] is None
    assert off["flags"] == ["zero_at_end", "constant"]
    assert report.losses["loss/frozen"] == [0.25, 0.25]  # first and last, as before
    assert main(["probe", TOY, "--iters", "4", "--predict-frames", "1"]) == 0
    out = capsys.readouterr().out
    assert "fell 0 % · flat · constant" in out and "never active · zero at end · constant" in out


def test_an_empty_prediction_is_runs_empty(monkeypatch, capsys):
    from grasppanda.registry import DETECTORS

    toy = DETECTORS.get("ToyDetector")
    original = toy.predict

    def nothing(self, observation):
        return [group[:0] for group in original(self, observation)]

    monkeypatch.setattr(toy, "predict", nothing)
    report = probe(TOY, iters=3, predict_frames=2)
    assert report.verdict == "runs-empty" and report.grasps_per_frame == [0, 0]
    assert report.score_quantiles == [None, None]
    assert "no grasp was predicted on any probed frame" in report.detail
    assert main(["probe", TOY, "--iters", "3"]) == 0  # it ran: a script goes on
    assert "∅ runs-empty: trained 3 iterations" in capsys.readouterr().out


def test_scores_and_the_decay_of_the_learning_rate_are_reported(capsys):
    """The toy's recipe multiplies the rate by 0.95 per epoch, and a probe of twelve
    iterations of six per epoch is two epochs: the last step runs at 0.95 of the first."""
    report = probe(TOY, iters=12, predict_frames=2)
    assert report.verdict == "runs" and report.probe_overrides[3] == "train.epochs=2"
    assert report.learning_rate == [0.005, 0.00475] and report.lr_factor == 0.95
    assert len(report.score_quantiles) == 2
    for low, median, high in report.score_quantiles:
        assert low <= median <= high
    # Stepped per iteration, the schedule decays within one epoch.
    fast = probe(TOY, ["train.scheduler_unit=iteration", "scheduler.gamma=0.5"], iters=6)
    assert fast.lr_factor == 0.03125  # 0.5 ** 5: six steps, five decays before the last
    assert report.schedule == fast.schedule == "recipe" and report.rate is None


def test_the_schedule_constant_trains_every_step_at_the_recipe_peak(capsys):
    """`constant` removes the scheduler and holds the recipe's peak rate; the config, its
    hash, the optimizer and the frames stay those of the recipe."""
    held = probe(TOY, iters=12, predict_frames=2, schedule="constant")
    assert held.verdict == "runs" and held.schedule == "constant" and held.rate == 0.005
    assert held.learning_rate == [0.005, 0.005] and held.lr_factor == 1.0
    # One override more than the recipe's probe, the one `grasppanda train` takes for it.
    recipe = probe(TOY, iters=12, predict_frames=2)
    assert held.probe_overrides == [
        *recipe.probe_overrides[:3],
        "scheduler=null",
        *recipe.probe_overrides[3:],
    ]
    # The same experiment by its hash: the hash is that of the config as given.
    assert held.config_hash == recipe.config_hash == Config.load(TOY).digest()
    assert held.frames == recipe.frames and held.parameters == recipe.parameters
    assert held.losses_per_iteration["loss"][0] == pytest.approx(
        recipe.losses_per_iteration["loss"][0], rel=1e-3
    )
    # A schedule that sets its own peak: the probe trains at it and says so.
    onecycle = ["scheduler.type=OneCycleLR", "scheduler.max_lr=0.02"]
    onecycle.append("train.scheduler_unit=iteration")
    raised = probe(TOY, onecycle, iters=4, predict_frames=1, schedule="constant")
    assert raised.verdict == "runs" and raised.rate == 0.02
    assert raised.learning_rate == [0.02, 0.02]
    assert any("OneCycleLR of the recipe peaks at 0.02" in note for note in raised.notes)
    # Batches smaller than the recipe's: the peak is scaled by their ratio (the 32 frames
    # of the toy where the recipe asks for 64), and the probe says so.
    wide = ["data.train.batch_size=64"]
    small = probe(TOY, wide, iters=2, predict_frames=1, schedule="constant")
    assert small.verdict == "runs" and small.batch_size == 32 and small.rate == 0.0025
    assert small.learning_rate == [0.0025, 0.0025]
    assert any("scaled by 32/64 (the linear scaling rule)" in note for note in small.notes)
    assert not any("scaled by" in note for note in held.notes)
    # The command states it in place of the decay.
    assert main(["probe", TOY, "--iters", "2", "--schedule", "constant"]) == 0
    out = capsys.readouterr().out
    assert "constant: 0.005, the recipe's peak rate, at every step" in out
    assert "scheduler=null" in out and "×" not in out
    assert main(["probe", TOY, "--iters", "2", "--schedule", "constant", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schedule"] == "constant" and report["rate"] == 0.005
    with pytest.raises(SystemExit):
        main(["probe", "--help"])
    help_text = capsys.readouterr().out
    assert "{recipe,constant}" in help_text and set(SCHEDULES) == {"recipe", "constant"}


def test_the_schedule_constant_holds_a_decaying_batch_norm_momentum(monkeypatch):
    """A batch-norm momentum that the recipe decays per epoch (BNMomentumHook) is a
    schedule too: decayed over the probe's short epochs it reaches its floor within the
    probe, and the running statistics that prediction uses stop following the weights
    (Scale-Balanced-Grasp then predicted no grasp after 250 iterations). `constant` holds
    it at its first epoch's value, as it holds the rate at its peak."""
    from grasppanda.engine.hooks.bn_momentum import BNMomentumHook

    momenta: list[float] = []
    original = BNMomentumHook.before_epoch

    def spy(self, state):
        momenta.append(self.momentum(state.epoch))
        original(self, state)

    monkeypatch.setattr(BNMomentumHook, "before_epoch", spy)
    hooks = ["hooks=[{type: BNMomentumHook, initial: 0.5, decay: 0.5, step: 1, minimum: 0.1}]"]
    # The toy's epoch is eight iterations: the recipe's momentum halves every eight.
    recipe = probe(TOY, hooks, iters=32, predict_frames=1)
    assert recipe.verdict == "runs" and momenta == [0.5, 0.25, 0.125, 0.1]
    assert not any("decay" in override for override in recipe.probe_overrides)
    momenta.clear()
    held = probe(TOY, hooks, iters=32, predict_frames=1, schedule="constant")
    assert held.verdict == "runs" and momenta == [0.5] * 4
    assert "hooks.0.decay=1" in held.probe_overrides
    assert held.config_hash == recipe.config_hash  # the experiment as given
    note = next(note for note in held.notes if "batch-norm momentum" in note)
    assert "held at 0.5" in note and "to 0.1, which an epoch of 8 iteration(s) reaches at " in note
    assert "iteration 25;" in note
    # A momentum that does not decay is left alone.
    flat = ["hooks=[{type: BNMomentumHook, initial: 0.1, decay: 1.0}]"]
    steady = probe(TOY, flat, iters=2, predict_frames=1, schedule="constant")
    assert not any("hooks." in override for override in steady.probe_overrides)
    assert not any("batch-norm momentum" in note for note in steady.notes)


def test_the_peak_rate_of_every_schedule_is_read_without_touching_the_optimizer():
    weights = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    groups = [
        {"params": [weights[0]], "lr": 1e-3, "lr_scale": 1.0},
        {"params": [weights[1]], "lr": 1e-4, "lr_scale": 0.1},  # ParamGroups' multiplier
    ]
    optimizer = torch.optim.Adam(groups, lr=1e-3)
    before = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
    assert peak_rates(None, optimizer) == [1e-3, 1e-4]
    for scheduler in (
        {"type": "ExponentialLR", "gamma": 0.95},
        {"type": "CosineAnnealingLR", "eta_min": 0.0},
        {"type": "MultiStepLR", "milestones": [8, 12], "gamma": 0.1},
        {"type": "StepLR", "step_size": 5, "gamma": 0.1},
        # FineGrasp's: the factor reaches 1 at the end of the warm-up of 800 steps.
        {"type": "WarmupCosineLR", "warmup_steps": 800, "cosine_steps": 8000},
    ):
        assert peak_rates(scheduler, optimizer) == [1e-3, 1e-4], scheduler
    # OneCycleLR replaces the rate by max_lr, scaled per group as the engine scales it.
    onecycle = {"type": "OneCycleLR", "max_lr": 2e-3, "cycle_momentum": True}
    assert peak_rates(onecycle, optimizer) == pytest.approx([2e-3, 2e-4])
    # (PyTorch 2.1 cycles the momentum only of an optimizer with `momentum`, not of Adam's
    # betas; the peak rate does not depend on it.)
    cyclic = {"type": "CyclicLR", "base_lr": 1e-5, "max_lr": 5e-3, "cycle_momentum": False}
    assert peak_rates(cyclic, optimizer) == [5e-3, 5e-3]
    linear = {"type": "LinearLR", "start_factor": 0.5, "end_factor": 0.25}
    assert peak_rates(linear, optimizer) == pytest.approx([5e-4, 5e-5])
    # Neither the rates nor the momentum of the optimizer were touched.
    after = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
    assert after == before and "initial_lr" not in optimizer.param_groups[0]


def test_a_config_that_trains_is_probed_with_the_real_trainer(capsys):
    report = probe(TOY, iters=6, predict_frames=3)
    assert report.verdict == "runs" and report.data == "real" and report.fallback is None
    assert report.iterations == report.iterations_requested == 6 and report.batch_size == 4
    assert report.detector == "ToyDetector" and report.parameters == 7601
    assert report.device == "cpu" and report.peak_gpu_mib is None
    assert set(report.losses) == {"loss/score", "loss/width", "loss"}
    assert all(len(pair) == 2 for pair in report.losses.values())
    first, last = report.losses["loss"]
    assert first == pytest.approx(sum(report.losses[k][0] for k in ("loss/score", "loss/width")))
    assert last < first  # six steps of Adam on a learnable toy problem
    assert len(report.frames) == 24 and report.frames[0].startswith("toy/")
    assert len(report.grasps_per_frame) == len(report.predicted_frames) == 3
    assert report.nonfinite == [] and report.failure is None and report.refusal == []
    assert report.no_gradient == []  # every part of the toy received a gradient
    # Judge 2: every iteration's losses and their trends, the units reached, the scores.
    assert report.judge == 2 and set(report.losses_per_iteration) == set(report.losses)
    assert all(len(values) == 6 for values in report.losses_per_iteration.values())
    assert report.losses_per_iteration["loss"][0] == pytest.approx(first, rel=1e-3)
    assert set(report.loss_trends) == set(report.losses)
    assert report.loss_trends["loss"]["active"] == 6
    assert report.units_checked == 3 and report.parts_checked == 2 and report.unchanged == []
    assert report.lr_factor == 1.0  # one epoch: the per-epoch schedule has not stepped
    assert len(report.score_quantiles) == 3 and all(report.score_quantiles)
    assert report.seconds_per_iteration is not None and report.seconds_per_iteration >= 0
    # What the probe changed is spelled as overrides of `grasppanda train` ...
    assert report.probe_overrides == [
        "data.train.num_workers=0",
        "evaluation.interval=0",
        "evaluation.save_best=null",
        "train.epochs=1",
        "data.val.num_workers=0",
    ]
    assert any("has no `split` option" in note for note in report.notes)
    # ... and it is the same experiment: the hash is that of the config as given.
    assert report.config_hash == Config.load(TOY).digest()
    assert json.loads(json.dumps(report.to_dict()))["verdict"] == "runs"


def test_the_probe_is_the_trainer_of_the_config_not_a_loop_of_its_own(monkeypatch):
    """Same weights after the same number of iterations as ``Trainer.train``: recipe,
    seed, data order, hooks."""
    from grasppanda.engine import trainer as engine

    seen: dict[str, torch.Tensor] = {}
    original = engine.Trainer.train

    def recording(self):
        try:
            return original(self)
        finally:
            seen.update({k: v.detach().clone() for k, v in self.state.model.state_dict().items()})

    monkeypatch.setattr(engine.Trainer, "train", recording)
    report = probe(TOY, iters=8)
    assert report.verdict == "runs"
    probed = dict(seen)
    seen.clear()

    from grasppanda.engine.run import Run, resolve_device

    config = Config.load(TOY, [*report.probe_overrides], command="train")
    cfg = config.resolved()
    run = Run(Path("unused"), "plain", "train", config, cfg, resolve_device("cpu"))
    plain = engine.Trainer(run)
    plain.hooks = []
    plain.train()  # one epoch of eight batches: exactly the eight iterations of the probe
    assert plain.state.iteration == 8 and probed.keys() == seen.keys()
    assert all(torch.equal(probed[name], seen[name]) for name in probed)


def test_a_refused_config_is_refused_before_anything_is_built(capsys):
    report = probe(TOY, ["model.head.num_angles=6"])
    assert report.verdict == "refused" and report.parameters is None and report.iterations == 0
    assert any("Did you mean 'num_angle'" in message for message in report.refusal)
    assert main(["probe", TOY, "--opts", "model.grouper.type=ToyDepthGrouper"]) == 2
    out = capsys.readouterr().out
    assert "✗ refused: the config check reports 1 problem(s)." in out
    assert "ToyHead expects ('seed', 'channel')" in out
    # A config `train` cannot run is refused by the same check as `print-config --for train`.
    model_only = str(REPO / "projects/example_project/configs/toy_detector.yaml")
    assert probe(model_only).refusal == [
        "data.train: `grasppanda train` needs this section, but it is not set."
    ]


def test_a_failure_names_the_phase_and_the_component(tmp_path):
    """Passes the config check, fails when built: the attention of EconomicGrasp's crop
    needs a head count that divides its token width, which only the built grouper knows."""
    config = tiny("EconomicGrasp", "PointNet2Dense")
    config["model"]["grouper"]["attention_heads"] = 7  # tokens are 8 + 3 channels wide
    path = write(tmp_path, "heads.yaml", config)
    report = probe(path, synthetic=True)
    assert report.verdict == "fails" and report.iterations == 0
    assert report.failure["phase"] == "build" and report.failure["error"] == "ValueError"
    assert report.failure["component"] == "grouper CylinderInteractionGroup"
    assert "num_heads must divide channels" in report.failure["message"]
    assert report.detail.startswith(
        "at build, in the grouper CylinderInteractionGroup: ValueError:"
    )
    assert main(["probe", path, "--synthetic"]) == 1


def test_a_first_loss_that_is_not_finite_is_a_failure_not_an_unstable_run(monkeypatch):
    """`runs-unstable` means that training ran to the end; a loss that is not finite in the
    first iteration ends the run before anything was trained."""
    from grasppanda.engine import trainer as engine

    def broken(self):
        raise engine.NonFiniteLossError("non-finite loss at epoch 1, iteration 1: loss/score = nan")

    monkeypatch.setattr(engine.Trainer, "train", broken)
    report = probe(TOY, iters=4)
    assert report.verdict == "fails" and report.iterations == 0 and report.nonfinite == []
    assert report.failure["when"] == "iteration 1"
    assert report.failure["error"] == "NonFiniteLossError"
    assert "loss/score = nan" in report.failure["message"]


def test_a_seed_width_mismatch_is_refused_before_anything_is_built(tmp_path):
    """The gated seed residual of Scale-Balanced-Grasp needs the encoder's width; the
    grouper states it, so the config check refuses the pair with the ways out."""
    config = tiny("ScaleBalancedGrasp", "PointNet2")
    config["model"]["grouper"]["mlp_channels"] = [8, 8]  # the encoder returns 12 channels
    report = probe(write(tmp_path, "narrow.yaml", config), synthetic=True)
    assert report.verdict == "refused"
    assert any("DepthCylinderProjectionGroup" in message for message in report.refusal)
    del config["model"]["grouper"]["seed_residual"]
    config["model"]["grouper"]["type"] = "DepthCylinderProjectionGroup"
    assert probe(write(tmp_path, "projected.yaml", config), synthetic=True).verdict == "wired"


def test_phase_and_component_are_read_from_the_traceback():
    from grasppanda.engine.builders import build_model
    from grasppanda.registry import BuildError

    with pytest.raises(BuildError) as caught:
        build_model({"model": {"type": "GSNet", "backbone": {"type": "PointNet2"}}})
    assert _phase(caught.value, "loss") == "build"

    def backward():
        raise RuntimeError("in a detector's loss")

    with pytest.raises(RuntimeError) as plain:
        backward()
    assert _phase(plain.value, "loss") == "loss" and _component(plain.value) == ""


# ------------------------------------------------------------------ the wiring check


def test_without_the_dataset_the_probe_says_so_and_only_claims_wired(tmp_path, capsys):
    config = tiny("GSNet", "PointNet2Dense")
    # The grid of the labels GraspNet-1Billion serves; a model on another one is refused.
    config["model"]["view"]["num_views"] = 300
    config["model"]["head"].update(num_angle=12, num_depth=4)
    config["data"] = {
        "train": {
            "dataset": {
                "type": "GraspNet1B",
                "camera": "realsense",
                "split": "train",
                "root": str(tmp_path / "no_dataset_here"),
                "preprocess": [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}],
                # The labels GSNet's stages read: point-wise graspness and the dense grids.
                "graspness": {
                    "source": "native",
                    "friction_threshold": 0.8,
                    "poses": "annotations",
                },
                "transforms": [
                    {"type": "RandomSample", "num_points": 600},
                    {
                        "type": "SampleGraspLabels",
                        "point_fraction": 0.25,
                        "min_points": 300,
                        "collision": "zero_scores",
                        "grids": "deferred",
                    },
                ],
            },
            "batch_size": 2,
        }
    }
    path = write(tmp_path, "tiny_gsnet.yaml", config)
    report = probe(path, iters=20)
    assert report.verdict == "wired" and report.data == "synthetic"
    assert "the dataset is not on this machine" in report.fallback
    assert "no_dataset_here" in report.fallback and "GRASPPANDA_GRASPNET1B_ROOT" in report.fallback
    assert report.probe_overrides == [] and all(f.startswith("synthetic/") for f in report.frames)
    assert any("graspness_threshold was lowered" in note for note in report.notes)
    assert report.iterations == report.iterations_requested == 2  # a wiring check is short
    assert "synthetic scene with random labels" in report.detail
    assert set(report.losses) >= {"loss/objectness", "loss/graspness", "loss/view", "loss"}
    assert report.grasps_per_frame and report.real_seeds == []
    # As a fallback, `wired` is not success: exit status 3, and the output says why.
    assert main(["probe", path]) == 3
    out = capsys.readouterr().out
    assert "NOT READ: the dataset is not on this machine" in out
    assert "SYNTHETIC scene with random labels" in out and "✓ wired:" in out
    assert "Real frames were not probed" in out
    # Asked for, it is: the check that was requested passed.
    assert main(["probe", path, "--synthetic", "--json"]) == 0
    asked = json.loads(capsys.readouterr().out)
    assert asked["verdict"] == "wired" and asked["fallback"] is None


@pytest.mark.parametrize(
    ("detector", "backbone"),
    [
        ("GSNet", "PointNet2Dense"),
        ("EconomicGrasp", "PointNeXt"),
        ("GraspNetBaseline", "PointNet2"),
        ("ScaleBalancedGrasp", "PointNet2Transformer"),
    ],
)
def test_every_seed_based_method_passes_the_wiring_check_in_small(detector, backbone, tmp_path):
    report = probe(write(tmp_path, "tiny.yaml", tiny(detector, backbone)), synthetic=True)
    assert report.verdict == "wired", report.detail
    assert report.iterations == 2 and report.nonfinite == [] and len(report.grasps_per_frame) == 2
    assert all(torch.isfinite(torch.tensor(pair)).all() for pair in report.losses.values())


def test_the_wiring_check_knows_an_image_based_detector(tmp_path):
    from .test_hggd_detector import TINY

    config = {"model": TINY, "optimizer": {"type": "AdamW", "lr": 1e-3}}
    config["runtime"] = {"device": "cpu"}
    report = probe(write(tmp_path, "tiny_hggd.yaml", config), synthetic=True)
    assert report.verdict == "wired", report.detail
    assert set(report.losses) >= {"loss/heatmap", "loss/anchor", "loss"}


def test_the_scene_batch_has_every_label_at_the_sizes_asked_for():
    observation, supervision = scene_batch(
        frames=2, points=500, views=7, angles=3, depths=2, label_points=9, feature_channels=6
    )
    assert observation.points.coords.shape == (1000, 3) and observation.point_feats.shape == (
        1000,
        6,
    )
    assert torch.equal(observation.point_feats[:, :3], observation.points.coords)
    assert observation.rgb is None and len(observation.keys) == 2
    depth = observation.points.coords[:, 2]
    assert float(depth.min()) > 0.2 and float(depth.max()) < 1.5  # metres, a table top
    assert set(supervision.point_labels) == {"objectness", "graspness"}
    assert 0 < int(supervision.point_labels["objectness"].sum()) < 1000
    for frame in range(2):
        dense, compact = supervision.objects[frame], supervision.extras[COMPACT_LABELS][frame]
        assert dense and len(dense) == len(compact)
        assert dense[0].scores.shape == dense[0].tolerance.shape == (9, 7, 3, 2)
        assert compact[0].view_graspness.shape == (9, 7)
        # Label points lie on the cloud: a seed can be within any matching distance.
        cloud = observation.points.coords[frame * 500 : (frame + 1) * 500]
        placed = dense[0].points @ dense[0].pose[:3, :3].T + dense[0].pose[:3, 3]
        assert float(torch.cdist(placed, cloud).min(dim=1).values.max()) < 0.01
    assert HISTOGRAM_KEY == scale_balance.HISTOGRAM_KEY and HISTOGRAM_KEY in supervision.extras
    again, _ = scene_batch(frames=2, points=500, views=7, angles=3, depths=2, label_points=9)
    assert torch.equal(again.points.coords, observation.points.coords) and again.point_feats is None
    pictures, labels = scene_batch(frames=1, points=300, images=True)
    assert pictures.rgb.shape == (1, 3, 720, 1280) and pictures.depth.shape == (1, 720, 1280)
    assert pictures.intrinsics.shape == (1, 3, 3)
    assert labels.extras["view_grasps"][0]["centers"].shape == (60, 2)


def test_the_random_batch_is_the_swap_matrix_batch():
    """`random_batch` is the batch of the swap matrix and of the contract tests; the
    wiring check of `grasppanda probe` uses `scene_batch`."""
    observation, supervision = random_batch(seed=3)
    theirs, _ = matrix.batch(seed=3)
    assert torch.equal(observation.points.coords, theirs.points.coords)
    assert supervision.objects[0][0].scores.shape == (20, 10, 3, 2)


def test_the_command_prints_the_numbers_and_one_verdict(capsys):
    assert main(["probe", TOY, "--iters", "4", "--predict-frames", "1"]) == 0
    out = capsys.readouterr().out
    for fragment in (
        "ToyDetector · 7,601 parameters",
        "frames of the config's dataset · 4 per iteration",
        "as `grasppanda train` with data.train.num_workers=0",
        "trained     4 of 4 iterations",
        "measured on this machine now",
        "loss/score",
        "predicted   1 frame(s)",
        "✓ runs: trained 4 iterations on 16 frames of the dataset and predicted on 1 frame(s)",
    ):
        assert fragment in out, fragment
    assert main(["probe", TOY, "--iters", "4", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "runs" and report["iterations"] == 4
    with pytest.raises(SystemExit) as stop:
        main(["probe", "--help"])
    text = capsys.readouterr().out
    assert stop.value.code == 0 and "--synthetic" in text
    # The help defines every verdict with its exit status and says what a probe needs.
    for name in VERDICTS:
        assert f"\n  {name} " in text, name
    for fragment in ("exit status", "what it needs", "a probe is one run", "--split sample"):
        assert fragment in text, fragment


def test_a_device_fault_that_surfaces_after_training_still_gives_a_report(monkeypatch):
    """A CUDA launch failure often surfaces at the next read of the device, which is the
    summary of the weights and gradients; the probe reports it instead of dying."""
    from grasppanda.combine import probe as module

    def faulty(report, recorder):
        raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setattr(module, "_summarise", faulty)
    report = probe(TOY, iters=2, predict_frames=1)
    assert report.verdict == "fails" and report.failure["error"] == "RuntimeError"
    assert "unspecified launch failure" in report.failure["message"]
    assert "after training" in json.dumps(report.failure)
