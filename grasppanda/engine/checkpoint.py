"""Checkpoints: what is saved, how it is written, and how weights are loaded.

Two different operations, kept apart on purpose:

* **Resume** (``runtime.resume``) restores a training run exactly: model, optimizer(s),
  scheduler(s), gradient scaler, weight average, random-number states of every process,
  epoch, iteration and best metric. It refuses a checkpoint written under a different
  config hash unless ``runtime.allow_config_mismatch`` is set; such a run then lives in a
  new run directory (``run.py``).
* **Load** (``runtime.load_from``) reads weights only. It always prints which keys
  matched, which are missing and which are unexpected, and raises on any difference
  unless ``runtime.allow_weight_mismatch`` is set. There is no silent ``strict=False``.

Files are written to a temporary name and renamed, so an interrupted job never leaves a
truncated checkpoint. They contain only tensors and plain Python values and are read
with ``weights_only=True``: loading a checkpoint never executes code from it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from grasppanda import __version__

from . import distributed as dist
from .errors import EngineError, ResumeError, WeightMismatchError
from .seeding import rng_state, set_rng_state
from .state import TrainState

__all__ = [
    "MISMATCH_ADVICE",
    "KeyReport",
    "atomic_save",
    "checkpoint_config_hash",
    "checkpoint_name",
    "config_mismatch",
    "file_sha256",
    "link_or_copy",
    "load_weights",
    "prune_checkpoints",
    "read_checkpoint",
    "resume",
    "save_checkpoint",
]

#: What to do about a checkpoint written under another config hash.
MISMATCH_ADVICE = (
    " Resume with the original config (config.resolved.yaml in the run directory), or set "
    "runtime.allow_config_mismatch=true to continue it as a new run."
)

#: Keys under which other code bases store a model's weights.
_FOREIGN_WEIGHT_KEYS = ("state_dict", "model_state_dict")


def checkpoint_name(epoch: int) -> str:
    return f"epoch_{epoch:04d}.pth"


# -------------------------------------------------------------------------- files


def atomic_save(payload: Any, path: str | Path) -> Path:
    """``torch.save`` to a temporary file, then rename over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def link_or_copy(source: Path, target: Path) -> None:
    """Make ``target`` the same file as ``source`` without a moment where it is half written."""
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:  # file systems without hard links
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint or weight file onto the CPU without executing pickled code."""
    path = Path(path)
    if not path.is_file():
        raise EngineError(f"no such checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise EngineError(
            f"cannot read {path} as a weights-only file: {error}\n"
            "GraspPanda loads tensors and plain values only. Convert files that pickle "
            "other objects (for example argparse namespaces) to a plain state dict first."
        ) from error
    if not isinstance(payload, Mapping):
        raise EngineError(f"{path} holds a {type(payload).__name__}, not a mapping of weights.")
    return dict(payload)


def checkpoint_config_hash(path: str | Path) -> str | None:
    """The config hash a training checkpoint was written under; ``None`` for a weights file.

    Tensors are memory-mapped where PyTorch can, so this costs little more than opening the
    file.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # not a zip archive (or no mmap in this PyTorch): read it plainly
        payload = read_checkpoint(path)
    value = payload.get("config_hash") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else None


def config_mismatch(path: str | Path, theirs: str, ours: str) -> str:
    """The sentence that says a checkpoint belongs to another experiment."""
    return (
        f"{path} was written under config hash {theirs[:12]}, but the current config "
        f"hashes to {ours[:12]}: this is a different experiment."
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prune_checkpoints(directory: Path, keep_last: int) -> list[Path]:
    """Delete all but the ``keep_last`` newest ``epoch_*.pth``; ``keep_last <= 0`` keeps all."""
    if keep_last <= 0:
        return []
    stale = sorted(directory.glob("epoch_*.pth"))[:-keep_last]
    for path in stale:
        path.unlink()
    return stale


# ----------------------------------------------------------------- save and resume


def _plain(value: Any) -> bool:
    """True for values a weights-only load accepts."""
    if value is None or isinstance(value, (bool, int, float, str, Tensor)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_plain(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, (str, int)) and _plain(item) for key, item in value.items())
    return False


def save_checkpoint(state: TrainState, path: str | Path) -> Path:
    """Write the full training state. Call on every process; the main one writes."""
    rng = dist.gather(rng_state())  # every process continues its own random stream
    path = Path(path)
    if not dist.is_main():
        return path
    payload = {
        "model": state.model.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in state.optimizers.items()},
        # Scheduler states are their attribute dicts; keep what a weights-only load accepts.
        # The rest (bound methods in old PyTorch versions) is rebuilt from the config.
        "schedulers": {
            name: {key: value for key, value in scheduler.state_dict().items() if _plain(value)}
            for name, scheduler in state.schedulers.items()
        },
        "scaler": state.scaler.state_dict(),
        "ema": state.ema.state_dict() if state.ema is not None else None,
        "rng": rng,
        "epoch": state.epoch,
        "iteration": state.iteration,
        "best": state.best,
        "config_hash": state.config.digest(),
        "version": __version__,
    }
    return atomic_save(payload, path)


def resume(state: TrainState, path: str | Path, *, allow_config_mismatch: bool = False) -> None:
    """Restore ``state`` from a checkpoint written by :func:`save_checkpoint`."""
    payload = read_checkpoint(path)
    missing = [key for key in ("model", "optimizers", "epoch", "config_hash") if key not in payload]
    if missing:
        raise ResumeError(
            f"{path} is not a training checkpoint (no {', '.join(missing)}). Use "
            "runtime.load_from to load weights only."
        )
    theirs, ours = payload["config_hash"], state.config.digest()
    if theirs != ours:
        message = config_mismatch(path, theirs, ours)
        if not allow_config_mismatch:
            raise ResumeError(message + MISMATCH_ADVICE)
        state.run.log.warning(
            "%s Continuing because runtime.allow_config_mismatch is set.", message
        )
    if payload.get("version") != __version__:
        state.run.log.warning(
            "checkpoint written by grasppanda %s, running %s", payload.get("version"), __version__
        )
    try:
        state.model.load_state_dict(payload["model"])
        if set(payload["optimizers"]) != set(state.optimizers):
            raise ValueError(
                f"optimizers {sorted(payload['optimizers'])} in the checkpoint, "
                f"{sorted(state.optimizers)} in the run"
            )
        for name, optimizer in state.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        for name, scheduler in state.schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        state.scaler.load_state_dict(payload.get("scaler") or {})
        if state.ema is not None:
            if payload.get("ema") is None:
                raise ValueError("train.ema is set, but the checkpoint has no averaged weights")
            state.ema.load_state_dict(payload["ema"])
    except (KeyError, ValueError, RuntimeError, EngineError) as error:
        raise ResumeError(f"{path} does not fit this run: {error}") from error
    rng = payload.get("rng") or []
    if len(rng) == dist.world_size():
        set_rng_state(rng[dist.rank()])
    else:
        state.run.log.warning(
            "checkpoint holds random states of %d process(es), this run has %d: random "
            "streams restart from the seed and the continuation is not bit-exact",
            len(rng),
            dist.world_size(),
        )
    state.epoch, state.iteration, state.best = (
        payload["epoch"],
        payload["iteration"],
        payload["best"],
    )


# ------------------------------------------------------------------ weights only


@dataclass
class KeyReport:
    """What happened to every key when weights were loaded into a model."""

    source: str
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (self.missing or self.unexpected or self.mismatched)

    def render(self, limit: int = 20) -> str:
        lines = [
            f"weights from {self.source}: {len(self.matched)} matched, "
            f"{len(self.missing)} missing, {len(self.unexpected)} unexpected, "
            f"{len(self.mismatched)} with a different shape"
        ]
        lines += [f"  note: {note}" for note in self.notes]
        groups = (
            ("missing (in the model, not in the file)", self.missing),
            ("unexpected (in the file, not in the model)", self.unexpected),
            ("different shape", self.mismatched),
        )
        for title, keys in groups:
            if keys:
                lines.append(f"  {title}:")
                lines += [f"    {key}" for key in keys[:limit]]
                if len(keys) > limit:
                    lines.append(f"    ... and {len(keys) - limit} more")
        return "\n".join(lines)


def _select_weights(
    payload: Mapping[str, Any], kind: str, report: KeyReport
) -> Mapping[str, Tensor]:
    """Find the state dict inside a checkpoint, a foreign checkpoint or a bare state dict."""
    if isinstance(payload.get("model"), Mapping):  # one of ours
        report.meta = {
            key: payload[key] for key in ("epoch", "config_hash", "version") if key in payload
        }
        if kind == "ema":
            if not isinstance(payload.get("ema"), Mapping):
                raise EngineError(
                    f"evaluation.weights is 'ema', but {report.source} holds no averaged "
                    "weights (it was trained without train.ema)."
                )
            report.notes.append("using the averaged weights ('ema')")
            return payload["ema"]
        return payload["model"]
    if kind == "ema":
        raise EngineError(
            f"evaluation.weights is 'ema', but {report.source} is not a GraspPanda checkpoint."
        )
    for key in _FOREIGN_WEIGHT_KEYS:
        if isinstance(payload.get(key), Mapping):
            report.notes.append(f"using the '{key}' entry of the file")
            return payload[key]
    if payload and all(isinstance(value, Tensor) for value in payload.values()):
        return payload
    raise EngineError(
        f"cannot find weights in {report.source}: top-level keys are "
        f"{', '.join(map(str, list(payload)[:10]))}. Expected 'model', 'state_dict' or a bare "
        "state dict."
    )


def load_weights(
    model: nn.Module, path: str | Path, *, kind: str = "model", allow_mismatch: bool = False
) -> KeyReport:
    """Load weights (and nothing else) into ``model`` and report on every key.

    Raises :class:`WeightMismatchError` when any key is missing, unexpected or of a
    different shape, unless ``allow_mismatch``; then the keys that fit are loaded.
    The caller prints ``report.render()`` either way.
    """
    report = KeyReport(source=str(path))
    weights = dict(_select_weights(read_checkpoint(path), kind, report))
    if weights and all(key.startswith("module.") for key in weights):
        weights = {key[len("module.") :]: value for key, value in weights.items()}
        report.notes.append("stripped the 'module.' prefix of a DataParallel checkpoint")
    own = model.state_dict()
    usable = {}
    for key, value in weights.items():
        if key not in own:
            report.unexpected.append(key)
        elif tuple(own[key].shape) != tuple(value.shape):
            report.mismatched.append(
                f"{key}: file {tuple(value.shape)}, model {tuple(own[key].shape)}"
            )
        else:
            report.matched.append(key)
            usable[key] = value
    report.missing = [key for key in own if key not in weights]
    if not report.clean and not allow_mismatch:
        raise WeightMismatchError(
            report.render() + "\nNothing was loaded. Fix the model config or the file, or set "
            "runtime.allow_weight_mismatch=true to load the keys that match."
        )
    model.load_state_dict(usable, strict=False)  # every skipped key is listed in the report
    return report
