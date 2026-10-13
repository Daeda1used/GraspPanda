"""Environment report for run snapshots and bug reports."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

__all__ = ["collect_env"]

#: Optional backends, matched by distribution-name prefix (``spconv-cu118`` → ``spconv``).
_OPTIONAL = ("spconv", "warpconvnet", "MinkowskiEngine", "torch-cluster", "pytorch3d", "open3d")


def _operator_backends() -> dict[str, Any]:
    """Backend each accelerated operator uses in this process; see ``grasppanda.ops.backends``."""
    try:
        from grasppanda.ops import backends

        return backends.report()
    except Exception as error:  # the report describes the environment, it must not break it
        return {"error": f"{type(error).__name__}: {error}"}


def collect_env(*, packages: bool = False) -> dict[str, Any]:
    """Describe the interpreter, torch build, GPUs, optional and operator backends.

    ``ops`` records which implementation each accelerated operator uses on CPU and on
    CUDA. Finding out settles the choice for the process, which may compile the optional
    kernels (once per machine) and runs their self-checks.

    With ``packages`` the full list of installed distributions is included, which is
    what a run snapshot stores.
    """
    import numpy
    import torch

    from grasppanda import __version__

    installed = {
        (dist.metadata["Name"] or "").lower(): dist.version for dist in metadata.distributions()
    }
    env: dict[str, Any] = {
        "grasppanda": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": [],
        "optional": {
            name: next((v for n, v in installed.items() if n.startswith(name.lower())), None)
            for name in _OPTIONAL
        },
    }
    if torch.cuda.is_available():
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    env["ops"] = _operator_backends()
    if packages:
        env["packages"] = sorted(
            f"{name}=={version}" for name, version in installed.items() if name
        )
    return env
