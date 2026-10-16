"""Optional-dependency handling."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

__all__ = ["MissingDependencyError", "install_hint", "installable", "require"]


class MissingDependencyError(ImportError):
    """An optional dependency is needed for the requested component."""


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import an optional dependency or explain exactly how to install it.

    Call this inside the component that needs the dependency, never at module import
    time, so that ``import grasppanda`` works with the core install. ``extra`` is an extra
    of ``pyproject.toml``, or a library with one extra per build (``spconv``), for which
    the hint names the build that matches the installed PyTorch.
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingDependencyError(
            f"{purpose} needs '{module}', which is not installed. {install_hint(extra)}"
        ) from error


#: Names a component may declare in ``extras`` that stand for several extras of
#: ``pyproject.toml``: one per build of the library, of which the user installs one.
EXTRA_BUILDS: dict[str, tuple[str, ...]] = {
    "spconv": ("spconv-cu118", "spconv-cu121", "spconv-cu124", "spconv-cu126", "spconv-cpu"),
}


def _family(extra: str) -> str:
    """The library an extra installs a build of (``spconv-cu118`` → ``spconv``), or itself."""
    return next((name for name, builds in EXTRA_BUILDS.items() if extra in builds), extra)


def _torch_cuda() -> str | None:
    """The CUDA version the installed PyTorch was built for; ``None`` for a CPU-only build."""
    try:
        import torch
    except ImportError:
        return None
    return torch.version.cuda


def installable(extra: str, *, this_machine: bool = False) -> str:
    """The extra to name in an install line for a declared ``extra``: itself, or for a
    library with one extra per build the choice (``spconv-cu118 | spconv-cu121 | ...``).

    With ``this_machine``, the build that matches the CUDA version of the installed PyTorch
    (``spconv-cpu`` for a CPU-only PyTorch) when there is one.
    """
    extra = _family(extra)
    builds = EXTRA_BUILDS.get(extra, (extra,))
    if this_machine and extra in EXTRA_BUILDS:
        cuda = _torch_cuda()
        build = f"{extra}-cu{cuda.replace('.', '')}" if cuda else f"{extra}-cpu"
        if build in builds:
            return build
    return " | ".join(builds)


def _checkout() -> bool:
    """Whether the package runs from a source checkout (editable install or ``PYTHONPATH``)."""
    root = Path(__file__).resolve().parents[2]
    return (root / "pyproject.toml").is_file() and (root / "grasppanda" / "__init__.py").is_file()


def install_hint(extra: str) -> str:
    """The sentence that says how to install an extra: ``pip install "grasppanda[...]"``
    from a wheel, ``pip install -e ".[...]"`` in a checkout."""
    name = installable(extra, this_machine=True)
    builds = EXTRA_BUILDS.get(_family(extra), ())
    cuda = _torch_cuda() if builds else None
    choice = ""
    if " | " in name:  # one extra per build, none of them for this PyTorch
        choice = f". Choose <build> to match your PyTorch (CUDA {cuda or 'none'}): {name}."
        name = "<build>"
    elif builds:
        others = " | ".join(build for build in builds if build != name)
        built = f"CUDA {cuda}" if cuda else "CPU only"
        choice = f". That is the build for your PyTorch ({built}); the others: {others}."
    if _checkout():
        line = f'Install it from the repository root with: pip install -e ".[{name}]"'
    else:
        line = f'Install it with: pip install "grasppanda[{name}]"'
    return line + choice
