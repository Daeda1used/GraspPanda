"""The evaluation routine: variants, empty and missing predictions, records, shards."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from grasppanda.engine.builders import ShardSampler
from grasppanda.engine.errors import EngineError
from grasppanda.engine.evaluation import RAW, build_job, evaluate, merge_keys, predict, score
from grasppanda.engine.run import start_run
from grasppanda.evaluation import load_prediction, prediction_path
from grasppanda.evaluation.records import ResultRecord, summarize
from grasppanda.structures import ContractError, SampleKey

from .test_engine_tiny import (
    TinyDetector,
    TinyScenes,
    run_training,
    single_thread,  # noqa: F401  (autouse fixture)
    tiny_config,
)

VARIANTS = {"evaluation.variants": {"raw": [], "best1": [{"type": "TinyKeepBest", "k": 1}]}}


def run_test(tmp_path, changes=None, model=None):
    """What ``grasppanda test`` does after loading weights, on the tiny components."""
    config = tiny_config(tmp_path, {**VARIANTS, **(changes or {})}, command="test")
    with start_run(config, "test") as run:
        job = build_job(run.cfg, "test")
        model = model or TinyDetector(**{k: v for k, v in run.cfg["model"].items() if k != "type"})
        evaluation = evaluate(model, job, run, weights={"path": "memory", "sha256": "0" * 64})
    return evaluation, run, job


# ----------------------------------------------------------------- the routine


def test_every_variant_is_predicted_written_and_scored(tmp_path):
    evaluation, run, job = run_test(tmp_path)
    assert list(evaluation.records) == ["raw", "best1"] and evaluation.partial == []
    assert evaluation.metrics == {
        "raw/AP": 1.0,
        "best1/AP": 0.5,
    }  # 2 of 2 and 1 of 2 expected grasps
    assert evaluation.frames == 6 and evaluation.run_dir == run.dir
    for key in job.dataset.keys:
        assert len(load_prediction(run.predictions / "raw", key)) == 4
        assert len(load_prediction(run.predictions / "best1", key)) == 1
    assert sorted(p.name for p in run.predictions.iterdir()) == ["best1", "raw"]
    assert sorted(p.name for p in run.results.iterdir()) == ["best1.json", "raw.json"]


def test_post_processors_see_the_frame_they_filter(tmp_path):
    """``TinyKeepBest`` raises unless it gets the single-frame observation of its grasps."""
    _, run, job = run_test(tmp_path, {"data.test.batch_size": 4})
    key = job.dataset.keys[5]
    raw, best = (
        load_prediction(run.predictions / "raw", key),
        load_prediction(run.predictions / "best1", key),
    )
    assert best.score.item() == pytest.approx(raw.score.max().item())


def test_result_record_says_what_was_measured(tmp_path):
    changes = {"data.test.dataset.gt_workspace": True, "runtime.seed": 7}
    evaluation, run, _ = run_test(tmp_path, changes)
    record = json.loads((run.results / "best1.json").read_text())
    assert record["schema"] == 1 and record["variant"] == "best1"
    assert (
        record["protocol"] == "tiny-v0"
        and record["dataset"] == "tiny"
        and record["split"] == "test"
    )
    assert record["cameras"] == ["cam"]
    assert (record["requested"], record["scored"], record["coverage"]) == (6, 6, 1.0)
    assert record["missing"] == [] and record["partial"] is False
    assert record["postprocessing"] == [{"type": "TinyKeepBest", "k": 1}]
    assert record["gt_assisted"] == ["workspace"]
    assert record["evaluator"] == {
        "type": "TinyEvaluator",
        "config": {"type": "TinyEvaluator", "script": None},
        "info": {"evaluator": "tiny"},
    }
    assert record["config_hash"] == run.config.digest() and record["seed"] == 7
    assert set(record["code"]) == {"commit", "dirty", "version"}
    assert record["weights"] == {"path": "memory", "sha256": "0" * 64}
    assert record["metrics"] == {"AP": 0.5} and record["partial_metrics"] is None
    assert record["per_scene"] == {
        "cam/scene_0020": {"frames": 6, "mean_precision": 0.5, "per_column": [0.5] * 3}
    }
    stamped = json.loads((run.dir / "run.json").read_text())
    assert stamped["gt_assisted"] == ["workspace"] and stamped["protocol"] == "tiny-v0"
    assert "ground-truth-assisted inputs: workspace" in evaluation.render()
    assert ResultRecord.read(run.results / "best1.json") == evaluation.records["best1"]


def test_an_empty_prediction_is_written_and_scored(tmp_path):
    evaluation, run, _ = run_test(tmp_path, {"model.empty_frame": 2})
    key = SampleKey("tiny", 20, "cam", 2)
    assert np.load(prediction_path(run.predictions / "raw", key)).shape == (0, 17)
    assert np.load(prediction_path(run.predictions / "best1", key)).shape == (0, 17)
    record = evaluation.records["raw"]
    assert not record.partial and record.scored == 6 and record.missing == []
    assert evaluation.metrics["raw/AP"] == pytest.approx(5 / 6)  # the empty frame scores zero
    assert f"{5 / 6:.2f}" in evaluation.render()


def test_a_missing_prediction_makes_the_result_partial_and_withholds_the_headline(tmp_path):
    config = tiny_config(tmp_path, VARIANTS, command="test")
    with start_run(config, "test") as run:
        job = build_job(run.cfg, "test")
        dump = predict(TinyDetector(), job, run.predictions, device=run.device)
        lost = job.dataset.keys[4]
        prediction_path(run.predictions / "best1", lost).unlink()
        records = score(job, dump, run.results)
    partial, complete = records["best1"], records["raw"]
    assert partial.partial and partial.scored == 5 and partial.requested == 6
    assert partial.missing == ["tiny/scene_0020/cam/0004"] and partial.headline() is None
    assert not complete.partial and complete.headline() == {"AP": 1.0}
    written = json.loads((run.results / "best1.json").read_text())
    assert written["partial"] is True and written["metrics"] is None
    assert written["partial_metrics"] == {"AP": 0.5} and written["coverage"] == pytest.approx(5 / 6)
    text = summarize(records)
    row = next(line for line in text.splitlines() if line.startswith("best1 "))
    assert "5/6" in row and "withheld (partial)" in row and "0.50" not in row
    assert "best1: PARTIAL, 5 of 6 frames scored (83.3 %), 1 missing" in text
    assert "cam/scene_0020  frames 5  0.5000" in text  # per-scene numbers instead
    assert ResultRecord.read(run.results / "best1.json").partial


def test_partial_variants_contribute_no_metric(tmp_path, monkeypatch):
    from .test_engine_tiny import TinyEvaluator

    original = TinyEvaluator.evaluate

    def lossy(self, predictions, keys, *, num_workers=0):
        result = original(self, predictions, keys, num_workers=num_workers)
        if str(predictions).endswith("best1"):
            result.missing.append(keys[0])
            del result.precision[keys[0]]
        return result

    monkeypatch.setattr(TinyEvaluator, "evaluate", lossy)
    evaluation, _, _ = run_test(tmp_path)
    assert evaluation.partial == ["best1"] and evaluation.metrics == {"raw/AP": 1.0}


def test_stale_predictions_never_stand_in_for_new_ones(tmp_path):
    """Whatever an earlier evaluation left behind is removed before predicting."""
    config = tiny_config(tmp_path, VARIANTS, command="test")
    with start_run(config, "test") as run:
        job = build_job(run.cfg, "test")
        predict(TinyDetector(), job, run.predictions, device=run.device)
        stale = prediction_path(run.predictions / RAW, SampleKey("tiny", 99, "cam", 0))
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"left over")
        predict(TinyDetector(), job, run.predictions, device=run.device)
        assert not stale.exists()


def test_contract_violations_of_a_detector_fail_loudly(tmp_path):
    class Short(TinyDetector):
        def predict(self, observation):
            return super().predict(observation)[:-1]

    class Mislabelled(TinyDetector):
        def predict(self, observation):
            groups = super().predict(observation)
            groups[0].key = SampleKey("tiny", 0, "cam", 999)
            return groups

    with pytest.raises(ContractError, match="returned 1 GraspGroups for a batch of 2 frames"):
        run_test(tmp_path / "a", model=Short())
    with pytest.raises(ContractError, match="returned grasps of tiny/scene_0000/cam/0999"):
        run_test(tmp_path / "b", model=Mislabelled())


def test_variant_names_are_checked(tmp_path):
    reserved = {"evaluation.variants": {"raw": [{"type": "TinyKeepBest"}]}}
    with (
        start_run(tiny_config(tmp_path / "a", reserved), "test") as run,
        pytest.raises(EngineError, match="must be an empty chain"),
    ):
        build_job(run.cfg, "test")
    slash = {"evaluation.variants": {"a/b": []}}
    with (
        start_run(tiny_config(tmp_path / "b", slash), "test") as run,
        pytest.raises(EngineError, match="cannot be used as a directory name"),
    ):
        build_job(run.cfg, "test")


def test_an_alias_of_raw_is_scored_from_the_raw_directory(tmp_path):
    changes = {"evaluation.variants": {"plain": [], "best1": [{"type": "TinyKeepBest"}]}}
    evaluation, run, _ = run_test(tmp_path, changes)
    assert evaluation.metrics == {"plain/AP": 1.0, "best1/AP": 0.5}
    assert sorted(p.name for p in run.predictions.iterdir()) == ["best1", "raw"]  # written once


# ------------------------------------------------------------------ in training


def test_in_training_evaluation_writes_records_per_epoch(tmp_path):
    state = run_training(
        tiny_config(tmp_path, {**VARIANTS, "train.epochs": 3, "evaluation.interval": 2})
    )
    assert sorted(p.name for p in state.run.results.iterdir()) == ["epoch_0002", "epoch_0003"]
    record = json.loads((state.run.results / "epoch_0003/raw.json").read_text())
    assert record["split"] == "val" and record["requested"] == 4
    assert record["weights"] == {"source": "in-training", "kind": "model", "epoch": 3}
    assert state.metrics == {"raw/AP": 1.0, "best1/AP": 0.5} and state.metrics_epoch == 3
    assert state.model.training  # evaluation leaves the model as it found it
    log = (state.run.dir / "log.txt").read_text()
    assert "evaluation after epoch 2 (4 frames" in log and "evaluation after epoch 1" not in log


def test_in_training_evaluation_can_score_the_averaged_weights(tmp_path):
    seen = []

    class Spy(TinyDetector):
        def predict(self, observation):
            seen.append(self.head.weight.detach().clone())
            return super().predict(observation)

    changes = {
        "train.ema": 0.5,
        "train.epochs": 1,
        "evaluation.weights": "ema",
        "data.val.batch_size": 4,
    }
    config = tiny_config(tmp_path, changes)
    with start_run(config, "train") as run:
        from grasppanda.engine.trainer import Trainer

        trainer = Trainer(run)
        trainer.state.model.__class__ = Spy
        state = trainer.train()
    assert len(seen) == 1 and torch.equal(seen[0], state.ema.average["head.weight"])
    assert not torch.equal(seen[0], state.model.head.weight)  # the live weights are back
    record = json.loads((state.run.results / "epoch_0001/raw.json").read_text())
    assert record["weights"]["kind"] == "ema"
    with pytest.raises(EngineError, match="train.ema is not set"):
        run_training(tiny_config(tmp_path / "b", {"evaluation.weights": "ema"}))


def test_evaluating_during_training_does_not_change_what_is_trained(tmp_path):
    """Prediction draws random numbers (noisy validation data), yet training is unaffected."""
    changes = {
        "model.dropout": 0.3,
        "data.train.dataset.jitter": 0.05,
        "data.val.dataset.jitter": 0.05,
        "train.epochs": 3,
    }
    with_evaluation = run_training(
        tiny_config(tmp_path / "a", {**changes, "evaluation.interval": 1})
    )
    without = run_training(tiny_config(tmp_path / "b", {**changes, "evaluation.interval": 0}))
    left, right = with_evaluation.model.state_dict(), without.model.state_dict()
    assert all(torch.equal(left[name], right[name]) for name in left)
    assert with_evaluation.metrics and not without.metrics


def test_the_same_weights_predict_the_same_in_training_and_in_test(tmp_path):
    """Noisy validation data again: in-training evaluation and ``test`` must still agree."""
    from grasppanda.engine import commands

    changes = {"data.val.dataset.jitter": 0.05, "runtime.run_id": "train", "train.epochs": 2}
    state = run_training(tiny_config(tmp_path, changes))
    checkpoint = state.run.checkpoints / "epoch_0002.pth"
    later = {**changes, "runtime.run_id": "later", "evaluation.split": "val"}
    later["runtime.load_from"] = str(checkpoint)
    evaluation = commands.test(tiny_config(tmp_path, later, command="test"))
    for key in TinyScenes(frames=4, first_scene=10).keys:
        during = np.load(prediction_path(state.run.predictions / "raw", key))
        after = np.load(prediction_path(evaluation.run_dir / "predictions/raw", key))
        assert during.shape == (4, 17) and np.array_equal(during, after), key
    other_seed = commands.test(
        tiny_config(
            tmp_path, {**later, "runtime.run_id": "seed", "runtime.seed": 1}, command="test"
        )
    )
    key = SampleKey("tiny", 10, "cam", 0)
    assert not np.array_equal(
        np.load(prediction_path(other_seed.run_dir / "predictions/raw", key)),
        np.load(prediction_path(evaluation.run_dir / "predictions/raw", key)),
    )  # the noise is real, and it follows runtime.seed


# ------------------------------------------------------------------ distributed


@pytest.mark.parametrize(("size", "world"), [(6, 2), (7, 2), (5, 3), (2, 4)])
def test_prediction_shards_cover_every_frame_exactly_once(size, world):
    shards = [list(ShardSampler(size, rank, world)) for rank in range(world)]
    assert sorted(index for shard in shards for index in shard) == list(range(size))
    assert [len(ShardSampler(size, rank, world)) for rank in range(world)] == [
        len(s) for s in shards
    ]


def test_frames_predicted_by_two_processes_are_counted_once(tmp_path):
    """Two processes of one evaluation, run here one after the other.

    First with the engine's own shards (disjoint), then with a padding sampler as used
    for training, where the frame added to equalise the shards is predicted twice.
    """
    config = tiny_config(
        tmp_path, {"data.test.dataset.frames": 5, "data.test.batch_size": 2}, command="test"
    )
    with start_run(config, "test") as run:
        jobs = [build_job(run.cfg, "test", rank=rank, world_size=2) for rank in range(2)]
        dumps = [
            predict(TinyDetector(), job, run.dir / f"rank{i}", device=run.device)
            for i, job in enumerate(jobs)
        ]
        assert [len(d.keys) for d in dumps] == [3, 2] and not set(dumps[0].keys) & set(
            dumps[1].keys
        )
        assert merge_keys([d.keys for d in dumps]) == [*dumps[0].keys, *dumps[1].keys]

        dataset = jobs[0].dataset
        padded = []
        for rank, job in enumerate(jobs):
            job.loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                sampler=DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=False),
                collate_fn=job.loader.collate_fn,
            )
            padded.append(
                predict(TinyDetector(), job, run.dir / f"padded{rank}", device=run.device).keys
            )
    assert [len(keys) for keys in padded] == [3, 3]  # 5 frames padded to 6
    merged = merge_keys(padded)
    assert len(merged) == 5 and set(merged) == set(dataset.keys)


def test_a_frame_repeated_within_one_process_is_predicted_once(tmp_path):
    config = tiny_config(tmp_path, command="test")
    with start_run(config, "test") as run:
        job = build_job(run.cfg, "test")
        job.loader = torch.utils.data.DataLoader(
            job.dataset, batch_size=3, sampler=[0, 1, 0, 2, 1], collate_fn=job.loader.collate_fn
        )
        dump = predict(TinyDetector(), job, run.predictions, device=run.device)
    assert dump.keys == list(job.dataset.keys[:3])


def test_prediction_loader_refuses_to_shuffle(tmp_path):
    with (
        start_run(tiny_config(tmp_path, {"data.test.shuffle": True}), "test") as run,
        pytest.raises(EngineError, match="prediction reads frames in order"),
    ):
        build_job(run.cfg, "test")
    assert len(TinyScenes()) == 8
