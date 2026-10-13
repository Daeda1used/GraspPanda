"""Evaluator interface and the on-disk prediction layout it reads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grasppanda.registry import EVALUATORS
from grasppanda.structures import GraspGroup, SampleKey

__all__ = ["EvaluationResult", "Evaluator", "load_prediction", "prediction_path", "save_prediction"]


def prediction_path(root: str | Path, key: SampleKey) -> Path:
    """``<root>/scene_0100/realsense/0000.npy``: the layout of the official GraspNet tools."""
    return Path(root) / f"scene_{key.scene:04d}" / key.camera / f"{key.frame:04d}.npy"


def save_prediction(root: str | Path, grasps: GraspGroup) -> Path:
    """Write one frame's grasps as a ``(G, 17)`` float32 array. An empty group is written too."""
    if grasps.key is None:
        raise ValueError("cannot save a GraspGroup without a SampleKey")
    path = prediction_path(root, grasps.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, grasps.to_graspnet_array())
    temporary.replace(path)  # atomic: a reader never sees a half-written file
    return path


def load_prediction(root: str | Path, key: SampleKey) -> GraspGroup:
    """Read one frame's grasps. A missing file raises; an empty file is a valid result."""
    return GraspGroup.from_graspnet_array(np.load(prediction_path(root, key)), key=key)


@dataclass
class EvaluationResult:
    """Outcome of scoring a set of frames.

    Attributes:
        precision: Per frame, the precision-at-k table ``(top_k, num_friction)``.
        metrics: Aggregates over the frames that were scored, e.g. ``AP``, ``AP0.8``, ``AP0.4``.
        requested: Number of frames that were asked for.
        missing: Frames without a prediction file. Missing is an error state, not a zero.
        info: Evaluator name and version, constants, timing.
        breakdown: Optional split of the metrics by scene and by object, for evaluators
            that can attribute them (:mod:`grasppanda.evaluation.breakdown`). Empty for
            an evaluator that cannot; nothing depends on it being present.
    """

    precision: dict[SampleKey, np.ndarray]
    metrics: dict[str, float]
    requested: int
    missing: list[SampleKey] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of the requested frames that were scored."""
        return len(self.precision) / self.requested if self.requested else 0.0

    @property
    def complete(self) -> bool:
        """Only a complete result may be reported as a headline number."""
        return self.requested > 0 and len(self.precision) == self.requested


@EVALUATORS.base
class Evaluator:
    """Scores dumped predictions under one pinned protocol.

    An empty prediction is valid and is scored; a missing prediction is recorded in
    ``EvaluationResult.missing`` and lowers coverage. Evaluators read the dataset in
    place and never write into it.
    """

    #: Identifier of the pinned protocol, e.g. ``graspnet1b-sv-v1``.
    protocol: str = "unspecified"

    def evaluate(
        self, predictions: str | Path, keys: Sequence[SampleKey], *, num_workers: int = 0
    ) -> EvaluationResult:
        """Score the frames ``keys`` whose predictions live under ``predictions``."""
        raise NotImplementedError
