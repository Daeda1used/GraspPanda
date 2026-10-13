"""Distributed helpers: who am I, and the three collectives the engine needs.

A distributed run is started with plain ``torchrun ... -m grasppanda.cli.main train ...``;
nothing in the config says "distributed". ``torchrun`` exports ``RANK``, ``LOCAL_RANK``
and ``WORLD_SIZE``, and that is the only signal read here. Without them every function
degrades to a single process, so the engine has one code path.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch import nn

from .errors import EngineError

__all__ = [
    "barrier",
    "broadcast",
    "gather",
    "init",
    "is_distributed",
    "is_main",
    "local_rank",
    "parallel",
    "rank",
    "shutdown",
    "sync_batchnorm",
    "world_size",
]

T = TypeVar("T")

#: How long a process may wait inside a collective before PyTorch aborts it. While the
#: main process scores an evaluation on the CPU, the others wait for its result; the
#: defaults (10 minutes for NCCL, 30 for gloo) are shorter than a benchmark evaluation.
#: A dead process is still noticed at once: the launcher stops the others.
COLLECTIVE_TIMEOUT = timedelta(hours=24)


def _env(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value.strip().lstrip("-").isdigit() else default


def world_size() -> int:
    """Number of processes of this run: the initialised group, else what the launcher says."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return max(_env("WORLD_SIZE", 1), 1) if "LOCAL_RANK" in os.environ else 1


def rank() -> int:
    """Global rank of this process; 0 in a single-process run."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env("RANK", 0) if "LOCAL_RANK" in os.environ else 0


def local_rank() -> int:
    """Rank within this node, which is also the index of the GPU this process uses."""
    return _env("LOCAL_RANK", 0)


def is_distributed() -> bool:
    return world_size() > 1


def is_main() -> bool:
    """True on the one process that writes files and prints progress."""
    return rank() == 0


def init(device: torch.device) -> None:
    """Join the process group when a launcher started more than one process."""
    if not is_distributed() or dist.is_initialized():
        return
    # NCCL moves CUDA tensors between GPUs; gloo covers CPU runs (and is what CI can test).
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend, timeout=COLLECTIVE_TIMEOUT)


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_batchnorm(model: nn.Module, device: torch.device) -> nn.Module:
    """Convert batch-norm layers so that their statistics span all processes.

    With one process plain batch norm already sees the whole batch, so the model is
    returned unchanged. PyTorch implements the synchronised layer for CUDA only.
    """
    if not is_distributed():
        return model
    if device.type != "cuda":
        raise EngineError(
            "train.sync_bn is set, but this run uses several CPU processes and PyTorch "
            "synchronises batch norm on CUDA only. Unset train.sync_bn or run on GPUs."
        )
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def parallel(model: nn.Module, device: torch.device) -> nn.Module | None:
    """Wrap ``model`` so that gradients are averaged over all processes.

    Returns ``None`` in a single-process run. Calling the wrapper with
    ``(observation, supervision)`` reaches ``model.loss`` through ``forward``.
    """
    if not is_distributed():
        return None
    from torch.nn.parallel import DistributedDataParallel

    device_ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)


def barrier() -> None:
    """Wait until every process reaches this point."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast(value: T | None) -> T:
    """Return the main process's ``value`` on every process (others pass anything)."""
    if not (dist.is_available() and dist.is_initialized()):
        return value  # type: ignore[return-value]
    box: list[Any] = [value if is_main() else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def gather(value: T) -> list[T]:
    """Return every process's ``value``, ordered by rank, on every process."""
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    out: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return out
