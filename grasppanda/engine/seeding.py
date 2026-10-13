"""Random seeds and random states.

Three rules keep runs repeatable:

* a run seeds Python, NumPy and PyTorch from ``runtime.seed`` (plus the rank, once the
  model is built, so that processes draw different numbers);
* a checkpoint stores the random state of every process and a resume restores it;
* prediction runs under its own seed and puts the previous state back afterwards
  (:func:`preserved_rng`), so that evaluating during training neither changes what is
  trained nor differs from evaluating the same weights later with ``grasppanda test``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

__all__ = [
    "determinism_restored",
    "preserved_rng",
    "rng_state",
    "seed_everything",
    "set_rng_state",
]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch; optionally ask PyTorch for deterministic kernels.

    ``deterministic`` trades speed for repeatability. Operators without a deterministic
    implementation warn instead of failing, because some point-cloud operators have none.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


@contextmanager
def determinism_restored() -> Iterator[None]:
    """Leave PyTorch's process-wide determinism switches as they were found."""
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def rng_state() -> dict[str, Any]:
    """Random states of this process, as tensors and plain values (safe to checkpoint)."""
    version, internal, gauss = random.getstate()
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "python": [version, list(internal), gauss],
        "numpy": [name, torch.from_numpy(keys.astype(np.int64)), position, has_gauss, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore what :func:`rng_state` captured."""
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda") or []
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


@contextmanager
def preserved_rng(seed: int) -> Iterator[None]:
    """Run a block under ``seed`` and restore the random states it found afterwards."""
    saved = rng_state()
    seed_everything(seed)
    try:
        yield
    finally:
        set_rng_state(saved)
