"""Trainer, hooks and run utilities.

Names are resolved on first use rather than imported here: the config schema imports
``grasppanda.engine.hooks.base`` to type the ``hooks`` list, and that must not drag in
the trainer (which itself imports the config package).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EngineError": "errors",
    "NonFiniteLossError": "errors",
    "ResumeError": "errors",
    "WeightMismatchError": "errors",
    "Run": "run",
    "start_run": "run",
    "TrainState": "state",
    "Trainer": "trainer",
    "KeyReport": "checkpoint",
    "load_weights": "checkpoint",
    "save_checkpoint": "checkpoint",
    "Evaluation": "evaluation",
    "build_job": "evaluation",
    "evaluate": "evaluation",
    "predict": "evaluation",
    "infer": "commands",
    "test": "commands",
    "train": "commands",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
