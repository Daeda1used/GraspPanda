"""Optional-dependency handling."""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["MissingDependencyError", "require"]


class MissingDependencyError(ImportError):
    """An optional dependency is needed for the requested component."""


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import an optional dependency or explain exactly how to install it.

    Call this inside the component that needs the dependency, never at module import
    time, so that ``import grasppanda`` works with the core install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingDependencyError(
            f"{purpose} needs '{module}', which is not installed. "
            f'Install it with: pip install "grasppanda[{extra}]"'
        ) from error
