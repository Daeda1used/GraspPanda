"""Failures of a run that the user can act on.

Each is raised with a message that says what happened and what to change; the command
line prints it without a traceback and exits non-zero.
"""

from __future__ import annotations

__all__ = ["EngineError", "NonFiniteLossError", "ResumeError", "WeightMismatchError"]


class EngineError(RuntimeError):
    """A run cannot start or continue as configured."""


class NonFiniteLossError(EngineError):
    """A loss term became NaN or infinite."""


class ResumeError(EngineError):
    """A checkpoint cannot be resumed under the current config."""


class WeightMismatchError(EngineError):
    """Weights do not fit the model and the config does not allow a partial load."""
