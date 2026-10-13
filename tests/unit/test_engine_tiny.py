"""Tiny registered components the engine tests train and score, and tests of those components.

Everything is synthetic and small enough for a CPU: a dataset whose labels are a linear
function of the points, a two-layer detector, a post-processor and an evaluator whose
numbers can be scripted. They depend on the public interfaces only (``BaseDataset``,
``BaseGraspDetector``, ``PostProcessor``, ``Evaluator``), as the engine does.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from grasppanda.config import Config
from grasppanda.datasets import BaseDataset
from grasppanda.engine import distributed as dist
from grasppanda.engine.hooks import Hook
from grasppanda.engine.run import start_run, write_json
from grasppanda.engine.state import TrainState
from grasppanda.engine.trainer import Trainer
from grasppanda.evaluation import EvaluationResult, Evaluator, load_prediction, prediction_path
from grasppanda.models import BaseGraspDetector, PostProcessor
from grasppanda.registry import DATASETS, DETECTORS, EVALUATORS, HOOKS, POSTPROCESSORS
from grasppanda.structures import (
    Batch,
    ContractError,
    GraspGroup,
    Observation,
    PointSet,
    Provenance,
    SampleKey,
    Supervision,
)

_WEIGHTS = torch.tensor([1.0, -2.0, 0.5])


@DATASETS.register()
class TinyScenes(BaseDataset):
    """``frames`` clouds per scene; the label of a point is a linear function of it.

    ``jitter`` adds noise drawn from the *global* generator, as an augmentation would, so
    that training depends on the random state a checkpoint has to restore.
    """

    name = "tiny"

    def __init__(
        self,
        frames: int = 8,
        points: int = 16,
        scenes: int = 1,
        first_scene: int = 0,
        jitter: float = 0.0,
        gt_workspace: bool = False,
    ) -> None:
        self.points, self.jitter, self.gt_workspace = points, jitter, gt_workspace
        self._keys = [
            SampleKey(self.name, scene, "cam", frame)
            for scene in range(first_scene, first_scene + scenes)
            for frame in range(frames)
        ]

    @property
    def keys(self) -> Sequence[SampleKey]:
        return self._keys

    def __getitem__(self, index: int) -> Batch:
        key = self._keys[index]
        generator = torch.Generator().manual_seed(1000 * key.scene + key.frame)
        coords = torch.rand(self.points, 3, generator=generator)
        if self.jitter:
            coords = coords + self.jitter * torch.randn(self.points, 3)
        provenance = {"workspace": Provenance.GROUND_TRUTH} if self.gt_workspace else {}
        observation = Observation(keys=[key], points=PointSet(coords), provenance=provenance)
        return Batch(observation, Supervision(point_labels={"target": coords @ _WEIGHTS + 0.3}))


@DETECTORS.register()
class TinyDetector(BaseGraspDetector):
    """Regresses the point label; predicts ``grasps`` grasps per frame scored by that regression.

    Args:
        hidden: Width of the hidden layer.
        dropout: Dropout probability (a second consumer of the global random state).
        bn: Insert a batch-norm layer.
        nan_at: 1-based call of ``loss`` at which ``loss/aux`` becomes NaN; 0 never.
        grasps: Grasps predicted per frame.
        empty_frame: Frame index for which the prediction is empty; -1 for none.
    """

    def __init__(
        self,
        hidden: int = 8,
        dropout: float = 0.0,
        bn: bool = False,
        nan_at: int = 0,
        grasps: int = 4,
        empty_frame: int = -1,
    ) -> None:
        super().__init__()
        self.nan_at, self.grasps, self.empty_frame = nan_at, grasps, empty_frame
        self.calls = 0
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden), nn.BatchNorm1d(hidden) if bn else nn.Identity(), nn.ReLU()
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def _regress(self, coords: Tensor) -> Tensor:
        return self.head(self.drop(self.encoder(coords))).squeeze(-1)

    def loss(self, observation: Observation, supervision: Supervision) -> dict[str, Tensor]:
        self.calls += 1
        prediction = self._regress(observation.points.coords)
        target = supervision.point_labels["target"]
        aux = 0.01 * prediction.pow(2).mean()
        if self.calls == self.nan_at:
            aux = aux * float("nan")
        return {
            "loss/fit": (prediction - target).pow(2).mean(),
            "loss/aux": aux,
            "abs_error": (prediction - target).abs().mean(),
        }

    @torch.no_grad()
    def predict(self, observation: Observation) -> list[GraspGroup]:
        scores = self._regress(observation.points.coords)
        offsets = observation.points.batch_offsets.tolist()
        groups = []
        for key, start in zip(observation.keys, offsets[:-1]):
            count = 0 if key.frame == self.empty_frame else self.grasps
            groups.append(
                GraspGroup(
                    translation=observation.points.coords[start : start + count],
                    rotation=torch.eye(3, device=scores.device).expand(count, 3, 3).clone(),
                    width=torch.full((count,), 0.05, device=scores.device),
                    depth=torch.full((count,), 0.02, device=scores.device),
                    score=scores[start : start + count],
                )
            )
        return groups


@POSTPROCESSORS.register()
class TinyKeepBest(PostProcessor):
    """Keeps the ``k`` best grasps, and insists on receiving the observation of that very frame."""

    def __init__(self, k: int = 1) -> None:
        self.k = k

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(observation) != 1 or observation.keys[0] != grasps.key:
            raise ContractError(f"expected the observation of {grasps.key}, got {observation.keys}")
        return grasps.sort_by_score()[: self.k]


@EVALUATORS.register()
class TinyEvaluator(Evaluator):
    """A stand-in evaluator.

    Each frame's ``(2, 3)`` precision table is filled with one value: 0 for an empty
    prediction, else ``script[n]`` on the ``n``-th evaluation of that prediction
    directory when a script is given, else the fraction of the two expected grasps that
    are present. ``AP`` is the mean of the tables. A frame without a file is missing.
    """

    protocol = "tiny-v0"

    def __init__(self, script: list[float] | None = None) -> None:
        self.script = list(script or [])
        self.calls: dict[str, int] = {}

    def evaluate(
        self, predictions: str | Path, keys: Sequence[SampleKey], *, num_workers: int = 0
    ) -> EvaluationResult:
        name = Path(predictions).name
        call = self.calls.get(name, 0)
        self.calls[name] = call + 1
        precision, missing = {}, []
        for key in keys:
            if not prediction_path(predictions, key).is_file():
                missing.append(key)
                continue
            count = len(load_prediction(predictions, key))
            value = self.script[call] if self.script else min(count, 2) / 2
            precision[key] = np.full((2, 3), value if count else 0.0)
        metrics = {"AP": float(np.mean(list(precision.values())))} if precision else {}
        return EvaluationResult(precision, metrics, len(keys), missing, {"evaluator": "tiny"})


@HOOKS.register()
class ReplicaProbe(Hook):
    """After a run, writes every process's parameter checksum to ``replicas.json``.

    Under ``torchrun`` all processes must hold the same weights; this makes it checkable.
    """

    def after_run(self, state: TrainState) -> None:
        checksum = float(sum(p.detach().double().abs().sum() for p in state.model.parameters()))
        replicas = dist.gather(
            {"rank": dist.rank(), "checksum": checksum, "iterations": state.iteration}
        )
        if state.run.is_main:
            write_json(state.run.dir / "replicas.json", replicas)


# ----------------------------------------------------------------------- helpers


def _loader(frames: int, first_scene: int = 0) -> dict[str, Any]:
    dataset = {"type": "TinyScenes", "frames": frames, "first_scene": first_scene}
    return {"dataset": dataset, "batch_size": 2, "num_workers": 0}


def tiny_config(tmp_path: Path, changes: dict[str, Any] | None = None, **kwargs: Any) -> Config:
    """A complete CPU config; ``changes`` maps dotted paths to values (``None`` removes a key)."""
    data: dict[str, Any] = {
        "model": {"type": "TinyDetector"},
        "data": {"train": _loader(8), "val": _loader(4, 10), "test": _loader(6, 20)},
        "optimizer": {"type": "SGD", "lr": 0.1},
        "train": {"epochs": 2},
        "evaluation": {"interval": 1, "evaluator": {"type": "TinyEvaluator"}},
        "runtime": {"work_dir": str(tmp_path / "work"), "device": "cpu", "log": {"interval": 2}},
    }
    for path, value in (changes or {}).items():
        node = data
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = copy.deepcopy(value)
    return Config.from_dict(data, **kwargs)


class Recorder(Hook):
    """Remembers the total loss of every iteration and the learning rate after it."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.rates: list[float] = []

    def after_iter(self, state: TrainState) -> None:
        self.losses.append(float(state.outputs["loss"]))
        self.rates.append(state.learning_rates()["lr"])


class Stop(Exception):
    """Raised by :class:`Interrupt` to end a run the way a killed job ends."""


@HOOKS.register()
class Interrupt(Hook):
    """Kills the run after ``epoch`` epochs, once the checkpoint of that epoch is on disk."""

    priority = 85  # after CheckpointHook

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch

    def after_epoch(self, state: TrainState) -> None:
        if state.epoch == self.epoch:
            raise Stop


def run_training(config: Config, *hooks: Hook) -> TrainState:
    """Train ``config`` with extra hooks attached; returns the final state."""
    with start_run(config, "train") as run:
        trainer = Trainer(run)
        trainer.hooks = sorted([*trainer.hooks, *hooks], key=lambda hook: hook.priority)
        return trainer.train()


@pytest.fixture(autouse=True, scope="module")
def single_thread() -> Iterator[None]:
    """Tiny tensors gain nothing from threads and lose a lot on a busy machine.

    Module-scoped so that it is in force before module-scoped fixtures that train.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------------- the fixtures themselves


def test_tiny_dataset_is_deterministic_and_collates():
    dataset = TinyScenes(frames=3, points=5, scenes=2, first_scene=7)
    assert len(dataset) == 6 and dataset.keys[3] == SampleKey("tiny", 8, "cam", 0)
    assert torch.equal(dataset[4].observation.points.coords, dataset[4].observation.points.coords)
    batch = dataset.collate([dataset[0], dataset[1]])
    assert len(batch) == 2 and batch.supervision.point_labels["target"].shape == (10,)
    assert batch.observation.gt_assisted() == []
    assert TinyScenes(gt_workspace=True)[0].observation.gt_assisted() == ["workspace"]


def test_tiny_detector_predicts_one_group_per_frame_and_can_be_empty():
    dataset = TinyScenes(frames=3)
    batch = dataset.collate([dataset[i] for i in range(3)])
    groups = TinyDetector(grasps=4, empty_frame=1).eval().predict(batch.observation)
    assert [len(group) for group in groups] == [4, 0, 4]
    losses = TinyDetector().loss(batch.observation, batch.supervision)
    assert set(losses) == {"loss/fit", "loss/aux", "abs_error"}


def test_engine_package_exports_resolve_to_objects_not_modules():
    import types

    import grasppanda.engine as engine
    import grasppanda.engine.commands
    import grasppanda.engine.evaluation

    for name in engine.__all__:
        assert not isinstance(getattr(engine, name), types.ModuleType), name
    assert engine.evaluate is grasppanda.engine.evaluation.evaluate
    assert engine.train is grasppanda.engine.commands.train
    with pytest.raises(AttributeError):
        _ = engine.no_such_name


def test_tiny_config_is_valid_for_every_command(tmp_path):
    for command in ("train", "test", "infer"):
        tiny_config(tmp_path, command=command)
