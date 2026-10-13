"""A synthetic dataset and its evaluator, so the example trains and scores without data.

A "scene" is a random point cloud in a box in front of the camera. Its labels are simple
functions of the point coordinates, which makes them learnable and lets the evaluator
recompute the truth from a grasp's position alone. Nothing here resembles a real
benchmark; it only has the same *shape*: a dataset addressed by ``SampleKey``, labels
that reach the model through ``Supervision`` only, and an evaluator that reads dumped
predictions and returns per-frame precision tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from grasppanda.datasets import BaseDataset
from grasppanda.evaluation import EvaluationResult, Evaluator, load_prediction, prediction_path
from grasppanda.registry import DATASETS, EVALUATORS
from grasppanda.structures import Batch, Observation, PointSet, SampleKey, Supervision

#: The box the points live in: x and y in ±0.2 m, z from 0.3 to 0.7 m.
_LOW = torch.tensor([-0.2, -0.2, 0.3])
_SIZE = 0.4


def _unit(coords: Tensor) -> Tensor:
    """Coordinates mapped to ``[0, 1)`` along each axis of the box."""
    return ((coords - _LOW.to(coords)) / _SIZE).clamp(0.0, 1.0 - 1e-6)


def toy_width(coords: Tensor) -> Tensor:
    """Label width in metres: grows with x from 0.02 to 0.08."""
    return 0.02 + 0.06 * _unit(coords)[..., 0]


def toy_bins(coords: Tensor, num_angle: int, num_depth: int) -> tuple[Tensor, Tensor]:
    """Label angle bin (from y) and depth bin (from z)."""
    unit = _unit(coords)
    return (unit[..., 1] * num_angle).long(), (unit[..., 2] * num_depth).long()


@DATASETS.register()
class ToyScenes(BaseDataset):
    """Random clouds with analytic labels. Frame ``(scene, frame)`` is always the same cloud.

    ``point_labels`` holds, per point, ``cell`` (``angle_bin * num_depth + depth_bin``) and
    ``width`` (metres). The label grid (``num_angle``, ``num_depth``) must be the head's.
    Use disjoint ``first_scene`` ranges for training, validation and test.
    """

    name = "toy"
    status = "example"

    def __init__(
        self,
        num_scenes: int = 2,
        frames_per_scene: int = 8,
        first_scene: int = 0,
        num_points: int = 256,
        num_angle: int = 12,
        num_depth: int = 4,
        labels: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_points, self.num_angle, self.num_depth = num_points, num_angle, num_depth
        self.labels, self.seed = labels, seed
        self._keys = [
            SampleKey(self.name, scene, "toy", frame)
            for scene in range(first_scene, first_scene + num_scenes)
            for frame in range(frames_per_scene)
        ]

    @property
    def keys(self) -> Sequence[SampleKey]:
        return self._keys

    def __getitem__(self, index: int) -> Batch:
        key = self._keys[index]
        generator = torch.Generator().manual_seed(
            1_000_003 * self.seed + 1_009 * key.scene + key.frame
        )
        coords = _LOW + _SIZE * torch.rand(self.num_points, 3, generator=generator)
        observation = Observation(keys=[key], points=PointSet(coords))
        if not self.labels:
            return Batch(observation)
        angle, depth = toy_bins(coords, self.num_angle, self.num_depth)
        labels = {"cell": angle * self.num_depth + depth, "width": toy_width(coords)}
        return Batch(observation, Supervision(point_labels=labels))


@EVALUATORS.register()
class ToyEvaluator(Evaluator):
    """Precision of the ``top_k`` best-scored grasps of each frame.

    A grasp is correct at tolerance ``t`` when its depth equals the label depth at its
    position and its width is within ``t`` metres of the label width. The table of a
    frame is ``(top_k, len(tolerances))``; ``AP`` is 100 times the mean over all tables.
    An empty prediction scores zero; a missing file is reported, not scored.
    """

    protocol = "toy-synthetic-v0"
    status = "example"

    def __init__(
        self,
        top_k: int = 10,
        tolerances: tuple[float, ...] = (0.0025, 0.005, 0.01),
        num_depth: int = 4,
        depth_step: float = 0.01,
    ) -> None:
        self.top_k, self.tolerances = top_k, tuple(tolerances)
        self.num_depth, self.depth_step = num_depth, depth_step

    def _table(self, predictions: Path, key: SampleKey) -> np.ndarray:
        grasps = load_prediction(predictions, key).sort_by_score()[: self.top_k]
        table = np.zeros((self.top_k, len(self.tolerances)))
        if len(grasps) == 0:
            return table
        _, depth_bin = toy_bins(grasps.translation, 1, self.num_depth)
        depth_ok = (grasps.depth - (depth_bin + 1) * self.depth_step).abs() < 1e-4
        error = (grasps.width - toy_width(grasps.translation)).abs()
        ranks = torch.arange(1, self.top_k + 1)
        for column, tolerance in enumerate(self.tolerances):
            hits = torch.zeros(self.top_k)  # grasps that do not exist count as failures
            hits[: len(grasps)] = ((error <= tolerance) & depth_ok).float()
            table[:, column] = (hits.cumsum(0) / ranks).numpy()
        return table

    def evaluate(
        self, predictions: str | Path, keys: Sequence[SampleKey], *, num_workers: int = 0
    ) -> EvaluationResult:
        precision, missing = {}, []
        for key in keys:
            if prediction_path(predictions, key).is_file():
                precision[key] = self._table(Path(predictions), key)
            else:
                missing.append(key)
        metrics = {}
        if precision:
            tables = np.stack(list(precision.values()))
            metrics = {"AP": 100.0 * tables.mean(), "AP_tight": 100.0 * tables[..., 0].mean()}
        info = {"evaluator": "toy", "top_k": self.top_k, "tolerances": list(self.tolerances)}
        return EvaluationResult(precision, metrics, len(keys), missing, info)
