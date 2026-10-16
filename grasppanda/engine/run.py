"""A run: its directory, its log and its device.

Layout of ``<runtime.work_dir>/<run id>/``::

    config.resolved.yaml  config.sources/  run.json  git.json  env.json  code.tar.gz
    log.txt
    checkpoints/     epoch_0001.pth ... and best.pth
    predictions/     <variant>/scene_XXXX/<camera>/NNNN.npy
    results/         <variant>.json (test), epoch_NNNN/<variant>.json (during training)
    resumes/<time>/  the snapshot of each later invocation that resumed this run

One directory is one run. ``train`` without ``runtime.resume`` and every ``test`` and
``infer`` start a new one and refuse to write into a directory that already holds a
run. Resuming continues in the directory that owns the checkpoint, and keeps the
original snapshot next to the new one, only when it is the same run: the checkpoint was
written under the current config hash and ``runtime.work_dir`` (and ``runtime.run_id``,
when set) point at that directory. Otherwise (another work directory or run id, a config
continued with ``runtime.allow_config_mismatch``, a checkpoint outside a run directory)
the resumed run starts a new directory below ``runtime.work_dir``, its ``run.json``
records ``resumed_from``, and nothing is written into the checkpoint's directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from grasppanda.config import Config, git_info, write_snapshot
from grasppanda.ops import backends

from . import distributed as dist
from .checkpoint import MISMATCH_ADVICE, checkpoint_config_hash, config_mismatch
from .errors import EngineError, ResumeError
from .seeding import determinism_restored

__all__ = [
    "LOGGER_NAME",
    "Run",
    "find_latest_checkpoint",
    "resolve_device",
    "start_run",
    "write_json",
]

#: Name of the logger every run configures; library code logs through it.
LOGGER_NAME = "grasppanda"
_ID_SUFFIX = {"train": "", "test": "-test", "infer": "-infer"}


@dataclass
class Run:
    """Where a run lives and how it reports.

    Attributes:
        dir: The run directory.
        id: Its name below ``runtime.work_dir``.
        command: ``train``, ``test`` or ``infer``.
        config: The validated config of this invocation.
        cfg: ``config.resolved()``: every default written out. Components are built from it.
        device: Device of this process.
        git: Commit, branch and dirty flag of the code that runs.
        resume_from: Checkpoint this invocation continues from, if any.
        log: Logger writing to the console and, on the main process, to ``log.txt``.
    """

    dir: Path
    id: str
    command: str
    config: Config
    cfg: dict[str, Any]
    device: torch.device
    git: dict[str, Any] = field(default_factory=dict)
    resume_from: Path | None = None
    log: logging.Logger = field(default_factory=lambda: logging.getLogger(LOGGER_NAME))

    @property
    def is_main(self) -> bool:
        return dist.is_main()

    @property
    def checkpoints(self) -> Path:
        return self.dir / "checkpoints"

    @property
    def predictions(self) -> Path:
        return self.dir / "predictions"

    @property
    def results(self) -> Path:
        return self.dir / "results"

    def stamp(self, **fields: Any) -> None:
        """Merge facts learnt while running (protocol id, ground-truth-assisted inputs) into ``run.json``."""
        if not self.is_main:
            return
        path = self.dir / "run.json"
        info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        info.update(fields)
        write_json(path, info)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


# --------------------------------------------------------------------------- device


def resolve_device(name: str) -> torch.device:
    """Turn ``runtime.device`` into the device of this process."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except RuntimeError as error:
        raise EngineError(f"runtime.device: '{name}' is not a device ({error}).") from None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise EngineError(
                f"runtime.device is '{name}', but CUDA is not available. Use 'cpu' or 'auto'."
            )
        if device.index is None and dist.is_distributed():
            device = torch.device("cuda", dist.local_rank())
    return device


# ------------------------------------------------------------------- run directory


def find_latest_checkpoint(work_dir: str | Path, run_id: str | None = None) -> Path | None:
    """Newest ``epoch_*.pth`` below ``work_dir`` (within ``run_id`` when given)."""
    pattern = f"{run_id or '*'}/checkpoints/epoch_*.pth"
    found = sorted(Path(work_dir).glob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    return found[-1] if found else None


def _new_run_dir(work_dir: Path, run_id: str | None, command: str) -> Path:
    if run_id is not None:
        run_dir = work_dir / run_id
        if (run_dir / "run.json").exists():
            raise EngineError(
                f"{run_dir} already holds a run. Choose another runtime.run_id, or set "
                "runtime.resume to continue a training run."
            )
        return run_dir
    stem = time.strftime("%Y%m%d-%H%M%S") + _ID_SUFFIX.get(command, f"-{command}")
    run_dir, attempt = work_dir / stem, 1
    while run_dir.exists():  # two runs started within one second
        attempt += 1
        run_dir = work_dir / f"{stem}-{attempt}"
    return run_dir


def _same_run(owner: Path, work_dir: Path, run_id: str | None) -> bool:
    """Whether ``runtime.work_dir`` and ``runtime.run_id`` point at the run directory ``owner``."""
    return owner.parent.resolve() == work_dir.resolve() and run_id in (None, owner.name)


def _locate(
    cfg: dict[str, Any], command: str, digest: str
) -> tuple[Path, Path | None, bool, str | None]:
    """Return the run directory, the checkpoint to resume, whether the directory is new and,
    for a resumed run that starts a new directory, why it does."""
    runtime = cfg["runtime"]
    work_dir = Path(runtime["work_dir"]).expanduser()
    run_id = runtime["run_id"]
    resume = runtime["resume"] if command == "train" else False
    checkpoint: Path | None = None
    note = None
    if resume:
        if resume is True:
            checkpoint = find_latest_checkpoint(work_dir, run_id)
            if checkpoint is None:
                where = work_dir / run_id if run_id else work_dir
                raise ResumeError(
                    f"runtime.resume is true, but there is no checkpoint below {where}. "
                    "Start without resume, or point runtime.resume at a checkpoint file."
                )
        else:
            checkpoint = Path(resume).expanduser()
            if not checkpoint.is_file():
                raise ResumeError(f"runtime.resume: no such checkpoint: {checkpoint}")
        theirs = checkpoint_config_hash(checkpoint)
        other_config = theirs is not None and theirs != digest
        if other_config and not runtime["allow_config_mismatch"]:
            raise ResumeError(config_mismatch(checkpoint, theirs, digest) + MISMATCH_ADVICE)
        owner = checkpoint.parent.parent
        if checkpoint.parent.name == "checkpoints" and (owner / "run.json").is_file():
            here = _same_run(owner, work_dir, run_id)
            if here and not other_config:
                return owner, checkpoint, False, None
            if here and run_id is not None:  # the run id names the checkpoint's own run
                raise ResumeError(
                    f"{checkpoint} was written under another config, so with "
                    "runtime.allow_config_mismatch the run continues in a new directory, but "
                    f"runtime.run_id '{run_id}' names the directory of the checkpoint, "
                    f"{owner}. Choose another runtime.run_id, or leave it unset."
                )
            if other_config:
                reason = "the config differs from the one the checkpoint was written under"
            else:
                target = work_dir / run_id if run_id else work_dir
                reason = f"runtime.work_dir and runtime.run_id ask for {target}"
            note = (
                f"resuming {checkpoint} in a new run directory, because {reason}; the run "
                f"that owns the checkpoint, {owner}, is left as it is"
            )
    return _new_run_dir(work_dir, run_id, command), checkpoint, True, note


@contextmanager
def _logging(run_dir: Path) -> Iterator[None]:
    """Send the ``grasppanda`` logger to the console and, on the main process, to ``log.txt``."""
    logger = logging.getLogger(LOGGER_NAME)
    before = (logger.level, logger.propagate)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(logging.INFO if dist.is_main() else logging.WARNING)
    handlers: list[logging.Handler] = [console]
    if dist.is_main():
        file = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        handlers.append(file)
    for handler in handlers:
        logger.addHandler(handler)
    try:
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(before[0])
        logger.propagate = before[1]


@contextmanager
def start_run(config: Config, command: str) -> Iterator[Run]:
    """Create (or re-enter) the run directory, write the snapshot and open the log.

    Under ``torchrun`` the main process decides where the run lives and the others
    follow; only the main process writes.
    """
    cfg = config.resolved()
    device = resolve_device(cfg["runtime"]["device"])
    backends.configure(cfg["runtime"]["ops"])
    if device.type == "cuda" and device.index is not None:
        # The snapshot records which operator backend this run uses, which is settled per
        # device; it has to look at the device of the run, not at GPU 0.
        torch.cuda.set_device(device)
    joined = dist.is_distributed() and not torch.distributed.is_initialized()
    dist.init(device)
    try:
        located = _locate(cfg, command, config.digest()) if dist.is_main() else None
        run_dir, checkpoint, fresh, note = dist.broadcast(located)
        run = Run(
            dir=run_dir,
            id=run_dir.name,
            command=command,
            config=config,
            cfg=cfg,
            device=device,
            git=git_info(),
            resume_from=checkpoint,
        )
        if run.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            facts = {"run_id": run.id, "command_name": command, "world_size": dist.world_size()}
            if fresh:
                write_snapshot(config, run_dir)
                if checkpoint is not None:
                    facts["resumed_from"] = str(checkpoint)
                run.stamp(**facts, device=str(device))
            else:  # keep the original snapshot; record this invocation beside it
                stamp = time.strftime("%Y%m%d-%H%M%S")
                write_snapshot(config, run_dir / "resumes" / stamp)
                run.stamp(last_resume={"time": stamp, "checkpoint": str(checkpoint)})
        dist.barrier()
        with _logging(run_dir), determinism_restored():
            if note:
                run.log.info(note)
            yield run
    finally:
        if joined:
            dist.shutdown()
