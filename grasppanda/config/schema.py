"""The closed top-level schema of a GraspPanda config.

Everything about a run lives here; no behaviour is reachable only through a CLI flag.
Adding a field to one of these dataclasses is all it takes to make it a valid,
validated, documented config key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from grasppanda.datasets.base import BaseDataset
from grasppanda.engine.hooks.base import Hook
from grasppanda.evaluation.base import Evaluator
from grasppanda.models.base import BaseGraspDetector, PostProcessor
from grasppanda.ops.backends import BackendChoice
from grasppanda.registry import Cfg

__all__ = [
    "CheckpointConfig",
    "DataConfig",
    "EvaluationConfig",
    "LoaderConfig",
    "LogConfig",
    "RootConfig",
    "RuntimeConfig",
    "TrainConfig",
]


@dataclass
class LoaderConfig:
    dataset: Cfg[BaseDataset]
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool | None = None
    drop_last: bool = False


@dataclass
class DataConfig:
    train: LoaderConfig | None = None
    val: LoaderConfig | None = None
    test: LoaderConfig | None = None


@dataclass
class TrainConfig:
    epochs: int = 10
    amp: bool = False
    #: Batches whose gradients are summed before one optimizer step.
    accumulate: int = 1
    grad_clip: float | None = None
    grad_clip_mode: Literal["norm", "value"] = "norm"
    #: Decay of the exponential moving average of the weights; ``null`` disables it.
    ema: float | None = None
    sync_bn: bool = False
    #: When the scheduler steps: after every epoch or after every optimizer step. The
    #: total number of steps is injected into schedulers that need it (``OneCycleLR``,
    #: ``CosineAnnealingLR``, ``PolynomialLR``), so it is never repeated in a config.
    scheduler_unit: Literal["epoch", "iteration"] = "epoch"


@dataclass
class EvaluationConfig:
    #: Evaluate every N epochs during training; 0 disables in-training evaluation.
    interval: int = 1
    evaluator: Cfg[Evaluator] | None = None
    #: Named post-processing chains. Each is applied to the same raw predictions and
    #: scored separately, so results with and without a filter are always both reported.
    variants: dict[str, list[Cfg[PostProcessor]]] = field(default_factory=lambda: {"raw": []})
    #: ``<variant>/<metric>`` monitored for the best checkpoint, e.g. ``raw/AP``.
    save_best: str | None = None
    #: The ``data`` section that ``grasppanda test`` and ``grasppanda infer`` read.
    #: In-training evaluation always reads ``data.val``.
    split: Literal["train", "val", "test"] = "test"
    #: Which weights are scored: the trained ones, or their moving average (``train.ema``).
    weights: Literal["model", "ema"] = "model"
    #: Worker processes of the evaluator.
    num_workers: int = 0


@dataclass
class CheckpointConfig:
    interval: int = 1
    keep_last: int = 3


@dataclass
class LogConfig:
    interval: int = 50
    backends: list[Literal["text", "tensorboard", "wandb"]] = field(
        default_factory=lambda: ["text"]
    )


@dataclass
class RuntimeConfig:
    work_dir: str = "work_dirs/${config_name}"
    #: Name of the run directory below ``work_dir``; ``null`` generates one from the
    #: start time. Like ``work_dir`` it says where a run lives and is not hashed.
    run_id: str | None = None
    #: ``auto`` (CUDA when available), ``cpu``, ``cuda`` or ``cuda:N``. Under ``torchrun``
    #: each process uses the GPU of its local rank.
    device: str = "auto"
    seed: int = 0
    deterministic: bool = False
    #: ``true`` resumes from the latest checkpoint in ``work_dir``; a path resumes from that file.
    resume: bool | str = False
    #: Resuming a checkpoint written under a different config hash is refused unless this is set.
    allow_config_mismatch: bool = False
    #: Weights to load before training or testing (no optimizer state).
    load_from: str | None = None
    #: ``load_from`` fails on missing, unexpected or mis-shaped keys unless this is set.
    #: The key report is printed either way.
    allow_weight_mismatch: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: LogConfig = field(default_factory=LogConfig)
    #: Backend of each accelerated operator (``fps``, ``ball_query``, ``cylinder_query``,
    #: ``knn``), e.g. ``{fps: reference}``; operators left out use ``auto``. Backends
    #: return identical results, see ``docs/advanced_guides/operators.md``.
    ops: dict[str, BackendChoice] = field(default_factory=dict)


@dataclass
class RootConfig:
    #: Modules imported before validation, for out-of-tree components.
    custom_imports: list[str] = field(default_factory=list)
    model: Cfg[BaseGraspDetector] | None = None
    data: DataConfig | None = None
    optimizer: Cfg[Optimizer] | None = None
    scheduler: Cfg[LRScheduler] | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hooks: list[Cfg[Hook]] = field(default_factory=list)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
