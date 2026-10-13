"""Run snapshots: everything needed to say exactly what a run was.

The snapshot records code and config; a run never executes from it.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grasppanda.utils.env import collect_env

from .config import Config

__all__ = ["git_info", "write_snapshot"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKED = ("grasppanda", "configs")


def _git(*args: str, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def git_info(root: Path = _REPO_ROOT) -> dict[str, Any]:
    """Commit, branch and dirty flag of the checkout, or ``{"available": False}``."""
    if _git("rev-parse", "--is-inside-work-tree", cwd=root) is None:
        return {"available": False}
    commit = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", cwd=root) or ""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    return {
        "available": True,
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": bool(status.strip()),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _archive_code(target: Path, root: Path) -> bool:
    listing = _git("ls-files", "-z", "--", *_TRACKED, cwd=root)
    if not listing:
        return False
    with tarfile.open(target, "w:gz") as archive:
        for name in filter(None, listing.split("\0")):
            file = root / name
            if file.is_file():
                archive.add(file, arcname=name)
    return True


def write_snapshot(
    config: Config, run_dir: str | Path, *, command: Sequence[str] | None = None
) -> dict[str, Any]:
    """Write the run snapshot into ``run_dir`` and return the content of ``run.json``."""
    from grasppanda import __version__

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.resolved.yaml").write_text(config.dump(defaults=True), encoding="utf-8")

    sources = run_dir / "config.sources"
    if sources.exists():
        shutil.rmtree(sources)
    sources.mkdir()
    for index, source in enumerate(config.sources):
        shutil.copyfile(source, sources / f"{index:02d}_{source.name}")
    (sources / "overrides.txt").write_text(
        "".join(f"{item}\n" for item in config.overrides), encoding="utf-8"
    )

    git = git_info()
    _write_json(run_dir / "git.json", git)
    if git.get("available"):
        diff = _git("diff", "HEAD", cwd=_REPO_ROOT) if git.get("commit") else None
        if diff:
            (run_dir / "diff.patch").write_text(diff, encoding="utf-8")
        _archive_code(run_dir / "code.tar.gz", _REPO_ROOT)

    _write_json(run_dir / "env.json", collect_env(packages=True))

    run = {
        "command": list(command) if command is not None else sys.argv,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "grasppanda": __version__,
        "config": str(config.path) if config.path else None,
        "config_hash": config.digest(),
        "seed": config.select("runtime.seed"),
    }
    _write_json(run_dir / "run.json", run)
    return run
