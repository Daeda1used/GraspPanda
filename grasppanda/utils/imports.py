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
            f'Install it from the repository root with: pip install -e ".[{extra}]"'
        ) from error


#: Names a component may declare in ``extras`` that stand for several extras of
#: ``pyproject.toml``: one per build of the library, of which the user installs one.
EXTRA_BUILDS: dict[str, tuple[str, ...]] = {
    "spconv": ("spconv-cu118", "spconv-cu121", "spconv-cu124", "spconv-cu126", "spconv-cpu"),
}


def installable(extra: str) -> str:
    """The extra to name in an install line for a declared ``extra``: itself, or for a
    library with one extra per build the choice (``spconv-cu118 | spconv-cu121 | ...``)."""
    return " | ".join(EXTRA_BUILDS.get(extra, (extra,)))
