"""Turn config sections into live objects: model, datasets, loaders and hooks.

Loaders are built so that a run can be repeated and resumed exactly:

* the order of an epoch is a function of ``(runtime.seed, epoch)`` only, and the seeds
  of worker processes a function of ``(runtime.seed, rank, epoch)``, so neither depends
  on how many epochs ran in this process (:func:`set_epoch`);
* under ``torchrun`` the training sampler gives each process a disjoint share of one
  common shuffle;
* prediction loaders never shuffle and never pad: process ``r`` of ``R`` reads frames
  ``r, r + R, ...``, so no frame is predicted twice;
* a loader with workers is refused where its workers could not deliver a batch
  (:func:`worker_socket_problem`), instead of waiting for the first batch forever.
"""

from __future__ import annotations

import contextlib
import multiprocessing.util
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from grasppanda.datasets import BaseDataset
from grasppanda.models import BaseGraspDetector
from grasppanda.registry import DATASETS, DETECTORS, HOOKS
from grasppanda.structures import Batch, Observation

from . import distributed as dist
from .errors import EngineError
from .hooks.base import Hook

__all__ = [
    "PairedCollate",
    "ShardSampler",
    "build_dataset",
    "build_hooks",
    "build_model",
    "build_predict_loader",
    "build_train_loader",
    "set_epoch",
    "worker_socket_problem",
]


def build_model(cfg: Mapping[str, Any]) -> BaseGraspDetector:
    return DETECTORS.build(cfg["model"])


def build_dataset(cfg: Mapping[str, Any], section: str) -> BaseDataset:
    """Build the dataset of ``data.<section>``."""
    node = (cfg.get("data") or {}).get(section)
    if node is None:
        raise EngineError(f"data.{section} is not set in the config.")
    dataset = DATASETS.build(node["dataset"])
    if len(dataset) == 0:
        raise EngineError(f"data.{section}: the dataset is empty.")
    return dataset


# -------------------------------------------------------------------------- workers

#: Bytes of the path of a Unix socket, the terminating zero included (``sun_path``).
_SOCKET_PATH_BYTES = 108 if sys.platform.startswith("linux") else 104
#: What multiprocessing appends to the temporary directory for the socket of a worker.
_SOCKET_SUFFIX = len("/pymp-XXXXXXXX/listener-XXXXXXXX")


def _socket_directory() -> str:
    """The directory below which multiprocessing creates the sockets of this process."""
    # Python releases that fall back to /tmp themselves when TMPDIR is too deep for a
    # socket have this function; it returns the directory they will use.
    base = getattr(multiprocessing.util, "_get_base_temp_dir", None)
    if base is not None:
        with contextlib.suppress(Exception):
            return str(base(tempfile))
    return tempfile.gettempdir()


def worker_socket_problem() -> str | None:
    """Why dataloader workers could not deliver a batch to this process, or ``None``.

    A worker hands the tensors of a batch over as shared memory, announced through a Unix
    socket that it creates below the temporary directory. The path of a socket may not
    exceed 107 bytes (103 outside Linux). Below a deeper ``TMPDIR`` the hand-over fails in
    the worker, which prints the error and drops the batch, while the loader goes on
    waiting for that batch: the run hangs without an error of its own.
    """
    if os.name == "nt":  # workers there do not use Unix sockets
        return None
    directory = _socket_directory()
    length = len(os.fsencode(directory)) + _SOCKET_SUFFIX
    if length < _SOCKET_PATH_BYTES:
        return None
    return (
        f"loader workers hand every batch over through a Unix socket below the temporary "
        f"directory {directory!r} (TMPDIR), and its path would be {length} bytes long, more "
        f"than the {_SOCKET_PATH_BYTES - 1} the system allows: no batch would arrive and the "
        f"run would wait for the first one forever. Point TMPDIR to a shorter directory (for "
        f"example TMPDIR=/tmp), or set num_workers to 0."
    )


def _check_workers(node: Mapping[str, Any], section: str) -> None:
    if node["num_workers"] > 0 and (problem := worker_socket_problem()) is not None:
        raise EngineError(f"data.{section}.num_workers is {node['num_workers']}, but {problem}")


# ------------------------------------------------------------------------- training


def build_train_loader(
    cfg: Mapping[str, Any],
    dataset: BaseDataset,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    """Loader of ``data.train``. ``batch_size`` is per process."""
    node, seed = cfg["data"]["train"], cfg["runtime"]["seed"]
    _check_workers(node, "train")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.world_size() if world_size is None else world_size,
        rank=dist.rank() if rank is None else rank,
        shuffle=node["shuffle"] is not False,  # null means: shuffle when training
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=node["batch_size"],
        sampler=sampler,
        num_workers=node["num_workers"],
        collate_fn=dataset.collate,
        drop_last=node["drop_last"],
        generator=torch.Generator(),
    )
    if len(loader) == 0:
        raise EngineError(
            f"data.train yields no batch: {len(sampler)} frames on this process, batch_size "
            f"{node['batch_size']}, drop_last {node['drop_last']}."
        )
    return loader


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Fix the shuffle and the worker seeds of ``epoch``, independent of earlier epochs."""
    sampler = loader.sampler
    sampler.set_epoch(epoch)  # the shuffle of this epoch: seed + epoch, the same on every process
    loader.generator.manual_seed(sampler.seed + 1_000_003 * (epoch + 1) + 7_919 * sampler.rank)


# ----------------------------------------------------------------------- prediction


class ShardSampler(Sampler[int]):
    """Process ``rank`` of ``world_size`` reads indices ``rank, rank + world_size, ...`` in order."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class PairedCollate:
    """Collate frames into a batch and keep each frame's own observation beside it.

    The model predicts on the batch; a post-processor then receives the grasps of one
    frame together with the observation of that frame.
    """

    def __init__(self, dataset: BaseDataset) -> None:
        self.collate = dataset.collate

    def __call__(self, items: Sequence[Batch]) -> tuple[Batch, list[Observation]]:
        return self.collate(items), [item.observation for item in items]


def build_predict_loader(
    cfg: Mapping[str, Any],
    section: str,
    dataset: BaseDataset,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    """Loader of ``data.<section>`` for prediction: fixed order, no padding, every frame once."""
    node = cfg["data"][section]
    if node["shuffle"] and section != "train":
        raise EngineError(
            f"data.{section}.shuffle is true, but prediction reads frames in order. Remove it."
        )
    _check_workers(node, section)
    return DataLoader(
        dataset,
        batch_size=node["batch_size"],
        sampler=ShardSampler(
            len(dataset),
            dist.rank() if rank is None else rank,
            dist.world_size() if world_size is None else world_size,
        ),
        num_workers=node["num_workers"],
        collate_fn=PairedCollate(dataset),
        generator=torch.Generator(),  # seeded by predict(); never the global generator
    )


# ---------------------------------------------------------------------------- hooks


def build_hooks(cfg: Mapping[str, Any]) -> list[Hook]:
    """The default hooks implied by the config plus those listed under ``hooks``.

    Defaults: one logger per ``runtime.log.backends`` entry, in-training evaluation when
    ``data.val`` and ``evaluation.evaluator`` are set, and checkpointing. A hook listed
    under ``hooks`` replaces the default of the same class. Hooks run in ``priority``
    order, so evaluation precedes checkpointing and loggers come last.
    """
    from .hooks.checkpoint import CheckpointHook
    from .hooks.evaluation import EvaluationHook
    from .hooks.logger import LOGGER_HOOKS

    hooks: list[Hook] = [HOOKS.build(node) for node in cfg["hooks"]]
    listed = {type(hook) for hook in hooks}
    evaluation, runtime = cfg["evaluation"], cfg["runtime"]
    defaults: list[Hook] = [
        LOGGER_HOOKS[backend](interval=runtime["log"]["interval"])
        for backend in dict.fromkeys(runtime["log"]["backends"])
    ]
    evaluates = (
        evaluation["interval"] > 0
        and (cfg.get("data") or {}).get("val") is not None
        and evaluation["evaluator"] is not None
    )
    if evaluates:
        defaults.append(EvaluationHook(interval=evaluation["interval"]))
    elif evaluation["save_best"] and EvaluationHook not in listed:
        raise EngineError(
            f"evaluation.save_best is '{evaluation['save_best']}', but there is no in-training "
            "evaluation: set data.val, evaluation.evaluator and evaluation.interval > 0."
        )
    defaults.append(
        CheckpointHook(
            interval=runtime["checkpoint"]["interval"],
            keep_last=runtime["checkpoint"]["keep_last"],
            save_best=evaluation["save_best"],
        )
    )
    hooks += [hook for hook in defaults if type(hook) not in listed]
    return sorted(hooks, key=lambda hook: hook.priority)
