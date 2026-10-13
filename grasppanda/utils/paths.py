"""Where datasets and caches live on this machine.

Paths never enter the config hash: an experiment is the same experiment wherever its
data is mounted.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["cache_root", "dataset_root"]


def dataset_root(name: str, explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the root directory of a dataset.

    Order: the explicit config value, ``$GRASPPANDA_<NAME>_ROOT``, ``$GRASPPANDA_DATA/<name>``,
    then ``./data/<name>``. ``name`` is the dataset's short name, e.g. ``graspnet1b``.
    """
    if explicit:
        return Path(explicit).expanduser()
    specific = os.environ.get(f"GRASPPANDA_{name.upper()}_ROOT")
    if specific:
        return Path(specific).expanduser()
    shared = os.environ.get("GRASPPANDA_DATA")
    return (Path(shared).expanduser() if shared else Path("data")) / name


def cache_root() -> Path:
    """Root of derived assets and caches: ``$GRASPPANDA_CACHE`` or ``~/.cache/grasppanda``.

    Nothing derived is ever written into a dataset tree.
    """
    configured = os.environ.get("GRASPPANDA_CACHE")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "grasppanda"
