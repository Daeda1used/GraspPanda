"""GraspPanda: a toolbox for 6-DoF grasp detection in cluttered scenes."""

from typing import Any

try:
    from ._version import __version__
except ImportError:  # running from a source tree that was never built or installed
    __version__ = "0.0.0+unknown"

__all__ = ["GraspInferencer", "__version__"]


def __getattr__(name: str) -> Any:
    # Resolved on first use, so that `import grasppanda` stays as light as it was.
    if name == "GraspInferencer":
        from .apis.inferencer import GraspInferencer

        return GraspInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
