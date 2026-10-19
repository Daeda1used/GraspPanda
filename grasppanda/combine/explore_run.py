"""The Run panel of the live explorer: one frame in, the picture of ``grasppanda demo`` out.

The page served by ``grasppanda explore`` names a ported method (or the selection composed
on the page), a weights file and an input: the synthetic example scene, or a depth image
with its intrinsics, an optional colour image and an optional workspace box.
:class:`Runner` checks the request, runs :class:`~grasppanda.apis.GraspInferencer` on that
one frame in a worker thread, and keeps what ``grasppanda demo`` prints and draws: the
picture, the best grasps, the counts through the post-processing chain and the
``(G, 17)`` array.

The rules are as strict as those of the rest of the server:

* Weights are read only from files the command line named: a file under ``--weights-dir``
  (its resolved path lies inside the resolved directory) or a file given exactly with
  ``--weights NAME=PATH``. What the page sends is looked up among those, never opened as
  a path of its own.
* An upload is limited in bytes (:data:`UPLOAD_LIMIT`) and in pixels (:data:`MAX_SIDE`),
  checked against the format its name claims, and decoded into a fresh temporary
  directory, which is removed when the run ends. A run writes nowhere else and downloads
  nothing: a composed selection is built with ``pretrained`` switched off, since the
  weights file supplies every parameter.
* One run at a time: a request while one runs is answered with :class:`Busy`.
* A run predicts and never trains: the model is in evaluation mode, under ``no_grad``.
"""

from __future__ import annotations

import base64
import binascii
import gc
import io
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = [
    "BODY_LIMIT",
    "MAX_SIDE",
    "STAGES",
    "UPLOAD_LIMIT",
    "Busy",
    "RunError",
    "Runner",
    "WeightsSource",
]

#: Bytes of one uploaded file, after base64 decoding.
UPLOAD_LIMIT = 20 * 1024 * 1024
#: Bytes of the JSON body of a run: two uploads in base64 and the fields around them.
BODY_LIMIT = 2 * (UPLOAD_LIMIT * 4 // 3 + 4) + 65536
#: Pixels along either side of an uploaded image.
MAX_SIDE = 4096
#: Files the panel lists as weights: what ``torch.load`` reads.
WEIGHT_SUFFIXES = (".pth", ".pt", ".tar", ".ckpt")
#: The steps of a run, in order, as the page shows them.
STAGES = (
    "building the model and loading the weights",
    "reading the input and preparing the cloud",
    "predicting",
    "post-processing",
    "drawing the picture",
)
_LIST_DEPTH = 4
_LIST_LIMIT = 500
_WALK_LIMIT = 2000  # folders looked into for the listing
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TOP_K = 30  # grasps drawn, as `grasppanda demo` draws by default
_ROWS = 10  # grasps in the table, as `grasppanda demo` prints them
_DEPTH_SUFFIXES = (".png", ".npy", ".tif", ".tiff")
_RGB_SUFFIXES = (".png", ".jpg", ".jpeg")
_MAGIC = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".npy": (b"\x93NUMPY",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}
#: Words in a file name that suggest which method the weights are for: those of the files
#: the `convert` commands of the metafiles write (`baseline_realsense.pth`, ...). A word
#: that several methods could carry (`graspness`) suggests none.
_HINTS = {
    "GraspNet-baseline": ("baseline",),
    "GSNet": ("gsnet",),
    "Scale-Balanced-Grasp": ("sbg", "scale_balanced", "scale-balanced"),
    "EconomicGrasp": ("economic",),
    "FineGrasp": ("finegrasp",),
    "HGGD": ("hggd",),
}
_INTRINSICS = (
    "a depth image needs the intrinsics of the camera that took it: fx, fy, cx, cy in pixels, "
    "for example 925, 925, 640, 360. Without them a depth image cannot become a point cloud. "
    "Your camera's SDK reports them (RealSense: rs-enumerate-devices -c)."
)


class RunError(ValueError):
    """A request the Run panel cannot accept; the message says what to change.

    ``kind`` tells the page what went wrong: ``request`` (the page sent something
    malformed), ``weights``, ``input`` or ``too-large`` (an upload over the limit).
    """

    def __init__(self, message: str, kind: str = "request") -> None:
        super().__init__(message)
        self.kind = kind


class Busy(RuntimeError):
    """Another run is in progress; the server runs one at a time."""


def _inside(path: Path, root: Path) -> bool:
    return root in path.parents


def _megabytes(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB"


# ------------------------------------------------------------------------ weights


@dataclass(frozen=True)
class WeightsSource:
    """The weights files a server may read: under ``directory`` and the ``named`` files.

    Both are resolved when the server starts. Build it with :meth:`from_arguments`.
    """

    directory: Path | None = None
    named: Mapping[str, Path] = field(default_factory=dict)

    @classmethod
    def from_arguments(
        cls,
        directory: str | Path | None = None,
        named: Sequence[str] | Mapping[str, str | Path] = (),
    ) -> WeightsSource:
        """From ``--weights-dir DIR`` and ``--weights NAME=PATH`` (or ``--weights PATH``).

        Raises:
            ValueError: A directory or a file does not exist, or a name is not a plain word.
        """
        root = None
        if directory is not None:
            root = Path(directory).expanduser().resolve()
            if not root.is_dir():
                raise ValueError(f"--weights-dir {directory}: no such directory")
        pairs = list(named.items()) if isinstance(named, Mapping) else [_pair(i) for i in named]
        files: dict[str, Path] = {}
        for name, path in pairs:
            if not _NAME.match(name):
                raise ValueError(
                    f"--weights {name}=...: a name is a plain word of letters, digits, '.', '_' "
                    "and '-', for example hggd=hggd_realsense.pth"
                )
            if name in files:
                raise ValueError(f"--weights {name}=... is given twice")
            resolved = Path(path).expanduser().resolve()
            if not resolved.is_file():
                raise ValueError(f"--weights {name}={path}: no such file")
            files[name] = resolved
        return cls(root, files)

    @property
    def empty(self) -> bool:
        return self.directory is None and not self.named

    def _listed(self) -> tuple[list[Path], bool]:
        """Weights files under the directory, a few levels deep; and whether the list was cut."""
        root = self.directory
        if root is None:
            return [], False
        found: list[Path] = []
        for visited, (current, folders, files) in enumerate(os.walk(root)):
            if visited >= _WALK_LIMIT:  # a home directory, say: the page stays quick
                return found, True
            depth = len(Path(current).relative_to(root).parts)
            folders[:] = (
                sorted(f for f in folders if not f.startswith(".")) if depth < _LIST_DEPTH else []
            )
            for name in sorted(files):
                if name.startswith(".") or Path(name).suffix.lower() not in WEIGHT_SUFFIXES:
                    continue
                path = Path(current) / name
                resolved = path.resolve()
                if not _inside(resolved, root) or not resolved.is_file():
                    continue  # a link that leaves the directory is not offered
                found.append(path)
                if len(found) >= _LIST_LIMIT:
                    return found, True
        return found, False

    def entries(self) -> tuple[list[dict[str, Any]], bool]:
        """What the page offers: one entry per file, and whether the listing was cut short."""
        out = [
            {
                "id": name,
                "label": f"{name} · {path.name}",
                "path": str(path),
                "origin": "command line",
                "bytes": path.stat().st_size,
            }
            for name, path in self.named.items()
        ]
        listed, cut = self._listed()
        for path in listed:
            relative = path.relative_to(self.directory).as_posix()  # type: ignore[arg-type]
            out.append(
                {
                    "id": relative,
                    "label": relative,
                    "path": str(path),
                    "origin": "directory",
                    "bytes": path.stat().st_size,
                }
            )
        return out, cut

    def resolve(self, text: Any) -> Path:
        """The file a name or a path from the page stands for, if this server may read it.

        A name given with ``--weights``, or a path that resolves to a weights file inside
        ``--weights-dir`` (relative to it, or absolute), or exactly to a file named with
        ``--weights``. Anything else is refused without saying whether it exists.
        """
        if not isinstance(text, str) or not text.strip():
            raise RunError("choose a weights file", "weights")
        text = text.strip()
        if len(text) > 1024 or "\x00" in text:
            raise RunError("that is no weights file", "weights")
        if text in self.named:
            return self.named[text]
        candidate = Path(text)
        root = self.directory
        if candidate.is_absolute():
            resolved = candidate.resolve()
        elif root is not None:
            resolved = (root / candidate).resolve()
        else:
            resolved = None
        if resolved is not None and resolved in self.named.values():
            return resolved
        if root is not None and resolved is not None and _inside(resolved, root):
            if resolved.suffix.lower() not in WEIGHT_SUFFIXES:
                raise RunError(
                    f"'{text}' is not a weights file: the panel reads "
                    f"{', '.join(WEIGHT_SUFFIXES)} files",
                    "weights",
                )
            if not resolved.is_file():
                raise RunError(f"there is no weights file '{text}' in {root}", "weights")
            return resolved
        where = [f"the files under {root} (--weights-dir)"] if root is not None else []
        if self.named:
            where.append(f"the files named {', '.join(self.named)} (--weights NAME=PATH)")
        raise RunError(
            f"this server does not read '{text}'. It reads "
            + (" and ".join(where) or "no weights at all")
            + ". To use another file, start it again with --weights NAME=FILE or "
            "--weights-dir DIR.",
            "weights",
        )


def _pair(item: str) -> tuple[str, str]:
    """``NAME=PATH``, or a bare ``PATH`` named after its file."""
    name, sep, path = item.partition("=")
    if sep and name and path:
        return name, path
    if sep:
        raise ValueError(f"--weights {item}: write NAME=PATH, for example sbg=sbg_realsense.pth")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(item).stem).lstrip("._-") or "weights"
    return stem[:64], item


def suggest(method: str, config: str, entries: Sequence[Mapping[str, Any]]) -> str:
    """The entry whose name suggests it holds weights of ``method``; ``""`` when none does.

    A name only suggests: the run itself says whether the weights fit the config. Files
    named on the command line come first, then files of the config's camera.
    """
    hints = _HINTS.get(method, (method.lower(),))
    camera = next((name for name in ("realsense", "kinect") if name in config), "")
    best, score = "", 0
    for entry in entries:
        label = str(entry["label"]).lower()
        if not any(hint in label for hint in hints):
            continue
        points = 4 + (2 if entry["origin"] == "command line" else 0)
        points += (
            1 if camera and (camera in label or (camera == "realsense" and "_rs" in label)) else 0
        )
        if points > score:
            best, score = str(entry["id"]), points
    return best


# -------------------------------------------------------------------------- input


def _numbers(value: Any, count: int, message: str) -> list[float] | None:
    """``count`` finite numbers from a list or from text; ``None`` when nothing is given."""
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        tokens: list[Any] = value.replace(",", " ").split()
    elif isinstance(value, list):
        tokens = value
    else:
        raise RunError(message, "input")
    if all(token in ("", None) for token in tokens):
        return None
    try:
        if any(isinstance(token, bool) for token in tokens):
            raise ValueError
        numbers = [float(token) for token in tokens]
    except (TypeError, ValueError):
        raise RunError(message, "input") from None
    if len(numbers) != count or not all(math.isfinite(number) for number in numbers):
        raise RunError(message, "input")
    return numbers


def _store(
    upload: Any, role: str, what: str, suffixes: Sequence[str], work: Path
) -> tuple[Path, str]:
    """Decode an upload ``{"name", "data" (base64)}`` into ``work``; returns (file, shown name).

    The file is stored under a fixed name; the name the page sent only gives the format,
    which the first bytes must confirm.
    """
    if not isinstance(upload, Mapping) or not isinstance(upload.get("data"), str):
        raise RunError(f"the {what} must be sent as {{name, data}}, the data in base64")
    shown = Path(str(upload.get("name") or role).replace("\\", "/")).name[:120] or role
    suffix = Path(shown).suffix.lower()
    if suffix not in suffixes:
        raise RunError(f"{shown}: the {what} is read from {', '.join(suffixes)} files", "input")
    data = upload["data"]
    if len(data) > UPLOAD_LIMIT * 4 // 3 + 4:
        raise RunError(
            f"{shown} is larger than {_megabytes(UPLOAD_LIMIT)}, the limit of an upload",
            "too-large",
        )
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise RunError(f"{shown} did not arrive as base64") from None
    if len(raw) > UPLOAD_LIMIT:
        raise RunError(
            f"{shown} is {_megabytes(len(raw))}, larger than {_megabytes(UPLOAD_LIMIT)}, the "
            "limit of an upload",
            "too-large",
        )
    if not raw:
        raise RunError(f"{shown} is empty", "input")
    if not raw.startswith(_MAGIC[suffix]):
        raise RunError(
            f"{shown} is not a {suffix[1:].upper()} file: its first bytes are those of "
            "something else",
            "input",
        )
    target = work / f"{role}{suffix}"
    target.write_bytes(raw)
    return target, shown


def _check_size(path: Path, shown: str) -> None:
    """Refuse an image over :data:`MAX_SIDE` pixels before it is decoded."""
    if path.suffix == ".npy":
        try:
            shape = np.load(path, mmap_mode="r").shape
        except Exception as error:  # a pickled object array, a broken header
            raise RunError(f"{shown} cannot be read as a NumPy array: {error}", "input") from None
    else:
        from PIL import Image

        try:
            with Image.open(path) as image:
                shape = image.size
        except Exception as error:
            raise RunError(f"{shown} cannot be read as an image: {error}", "input") from None
    if any(side > MAX_SIDE for side in shape[:2]):
        raise RunError(
            f"{shown} is {' x '.join(map(str, shape))}; the panel takes images up to "
            f"{MAX_SIDE} pixels a side",
            "input",
        )


@dataclass
class _Plan:
    """A checked request: what to build, with which weights, on which frame."""

    method: str
    config: str
    sets: list[str]
    weights: Path
    kind: str
    collision_filter: bool
    work: Path
    depth: np.ndarray | None = None
    depth_name: str = ""
    rgb: np.ndarray | None = None
    rgb_name: str = ""
    intrinsics: list[float] | None = None
    workspace: list[float] | None = None

    @property
    def target(self) -> str:
        return self.method + (f" with {', '.join(self.sets)}" if self.sets else "")

    @property
    def input(self) -> str:
        return "the example scene" if self.kind == "example" else self.depth_name


# ---------------------------------------------------------------------------- jobs


class _Job:
    """One run: its steps with their times, then a result or an error."""

    def __init__(self, plan: _Plan) -> None:
        self.id = secrets.token_urlsafe(9)
        self.target = plan.target
        self.input = plan.input
        self.started = time.monotonic()
        self.state = "running"
        self.stage = -1
        self.entered = self.started
        self.times: list[float | None] = [None] * len(STAGES)
        self.notes: dict[int, str] = {}
        self.ended: float | None = None
        self.result: dict[str, Any] | None = None
        self.error: dict[str, str] | None = None
        self._lock = threading.Lock()

    def _close(self, now: float) -> None:
        if self.stage >= 0 and self.times[self.stage] is None:
            self.times[self.stage] = now - self.entered

    def enter(self, index: int, note: str = "") -> None:
        now = time.monotonic()
        with self._lock:
            self._close(now)
            self.stage, self.entered = index, now
            if note:
                self.notes[index] = note

    def finish(self, result: dict[str, Any]) -> None:
        now = time.monotonic()
        with self._lock:
            self._close(now)
            self.result, self.state, self.ended = result, "done", now

    def fail(self, error: dict[str, str]) -> None:
        now = time.monotonic()
        with self._lock:
            self.error, self.state, self.ended = error, "failed", now

    def busy(self) -> str:
        with self._lock:
            step = STAGES[self.stage] if self.stage >= 0 else "starting"
            elapsed = time.monotonic() - self.started
        return (
            f"another run is in progress ({self.target} on {self.input}: {step}, "
            f"{elapsed:.0f} s so far). This server runs one at a time; wait for it to finish."
        )

    def status(self, full: bool = True) -> dict[str, Any]:
        with self._lock:
            now = self.ended or time.monotonic()
            stages = []
            for index, name in enumerate(STAGES):
                seconds = self.times[index]
                if seconds is not None and not (self.state == "failed" and index == self.stage):
                    state = "done"
                elif index == self.stage:
                    state = "failed" if self.state == "failed" else "running"
                    seconds = now - self.entered
                else:
                    state = "waiting"
                stages.append(
                    {
                        "name": name,
                        "state": state,
                        "seconds": None if seconds is None else round(seconds, 2),
                        "note": self.notes.get(index, ""),
                    }
                )
            out: dict[str, Any] = {
                "id": self.id,
                "state": self.state,
                "target": self.target,
                "input": self.input,
                "elapsed": round(now - self.started, 2),
                "stages": stages,
            }
            if self.error is not None:
                out["error"] = self.error
            if full and self.result is not None:
                out["result"] = self.result
            return out


def _explain(error: BaseException) -> dict[str, str]:
    """The kind, a title and the message of a failed run, in the words of ``grasppanda demo``."""
    from grasppanda.apis import InputError
    from grasppanda.combine.compose import ComposeError
    from grasppanda.engine.errors import EngineError, WeightMismatchError
    from grasppanda.registry import BuildError
    from grasppanda.structures import ContractError
    from grasppanda.utils.imports import MissingDependencyError

    text = str(error).strip()
    message = f"{type(error).__name__}: {text}"
    if isinstance(error, RunError):
        return {"kind": error.kind, "title": "The request was refused", "message": text}
    if isinstance(error, WeightMismatchError):
        kind, title = "weights", "These weights do not fit the model of this config"
    elif isinstance(error, EngineError):
        kind, title = "weights", "The weights cannot be used"
    elif isinstance(error, MissingDependencyError):
        kind, title = "dependency", "An optional library is missing"
    elif isinstance(error, InputError):
        kind, title = "input", "The input was refused"
    elif isinstance(error, (ComposeError, BuildError, ContractError)):
        kind, title = "config", "The config cannot be built"
    elif isinstance(error, RuntimeError) and (
        "out of memory" in text or "unable to find an engine" in text
    ):
        kind, title = "memory", "The GPU has no free memory"
        message = (
            f"{text.splitlines()[0]}\nOne frame needs little memory, so the GPU is probably in "
            "use by another process: wait for it, or start grasppanda explore with --device cpu."
        )
    else:  # a bug: the terminal keeps the traceback
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        kind, title = "bug", "Unexpected error"
        message += "\nThis is a bug. The terminal that runs grasppanda explore shows the traceback."
    return {"kind": kind, "title": title, "message": message}


# ---------------------------------------------------------------------------- runner


def _shipped_methods() -> dict[str, str]:
    from .grid import FAMILIES, METHODS

    return {**METHODS, **FAMILIES}


def _no_downloads(node: Any) -> tuple[Any, bool]:
    """``node`` with every ``pretrained: true`` switched off, and whether one was."""
    if isinstance(node, Mapping):
        changed, out = False, {}
        for key, value in node.items():
            if key == "pretrained" and value is True:
                out[key], changed = False, True
            else:
                out[key], inner = _no_downloads(value)
                changed |= inner
        return out, changed
    if isinstance(node, list):
        pairs = [_no_downloads(item) for item in node]
        return [item for item, _ in pairs], any(changed for _, changed in pairs)
    return node, False


class Runner:
    """Runs one method on one frame for the Run panel, one run at a time.

    Args:
        weights: The files the server may read.
        methods: Method name → config; default the six ported methods.
        device: ``auto``, ``cpu``, ``cuda`` or ``cuda:N``; ``None`` takes the config's.
        picture_size: Size of the picture, as ``grasppanda demo --size``.
    """

    def __init__(
        self,
        weights: WeightsSource | None = None,
        *,
        methods: Mapping[str, str] | None = None,
        device: str | None = None,
        picture_size: tuple[int, int] = (960, 720),
    ) -> None:
        self.weights = weights or WeightsSource()
        self.methods = dict(methods) if methods is not None else _shipped_methods()
        self.device = device
        self.picture_size = (int(picture_size[0]), int(picture_size[1]))
        self._lock = threading.Lock()
        self._job: _Job | None = None
        self._kept: tuple[tuple[Any, ...], Any, dict[str, Any]] | None = None
        self._facts: dict[str, dict[str, Any]] = {}
        self._scene: Any = None
        self._scene_lock = threading.Lock()

    # ------------------------------------------------------------------ what it offers

    def _method(self, name: str) -> dict[str, Any]:
        if name not in self._facts:
            from grasppanda.config import Config

            config = self.methods[name]
            facts: dict[str, Any] = {"name": name, "config": config, "needs_rgb": False}
            try:
                cfg = Config.load(config).resolved()
                section = cfg["evaluation"]["split"]
                dataset = ((cfg.get("data") or {}).get(section) or {}).get("dataset") or {}
                facts["needs_rgb"] = "rgb" in (dataset.get("images") or ())
            except Exception as error:  # the run will say it again, with the whole message
                facts["problem"] = f"{type(error).__name__}: {error}"
            self._facts[name] = facts
        return self._facts[name]

    def setup(self) -> dict[str, Any]:
        """What the panel shows before a run: methods, weights, limits, a run in progress."""
        entries, cut = self.weights.entries()
        methods = []
        for name, config in self.methods.items():
            methods.append({**self._method(name), "suggested": suggest(name, config, entries)})
        with self._lock:
            job = self._job
        running = job.status(full=False) if job is not None and job.state == "running" else None
        return {
            "methods": methods,
            "weights": entries,
            "weights_dir": str(self.weights.directory or ""),
            "truncated": cut,
            "limits": {"upload_bytes": UPLOAD_LIMIT, "max_side": MAX_SIDE},
            "device": self.device or "",
            "stages": list(STAGES),
            "running": running,
        }

    def status(self, job_id: str) -> dict[str, Any] | None:
        """The state of a run; ``None`` for a run this server does not know (it keeps the last)."""
        with self._lock:
            job = self._job
        return job.status() if job is not None and job.id == job_id else None

    # ---------------------------------------------------------------------- a request

    def _plan(self, request: Mapping[str, Any]) -> _Plan:
        method = request.get("method")
        if not isinstance(method, str) or method not in self.methods:
            raise RunError(f"unknown method {method!r}. Methods: {', '.join(self.methods)}.")
        sets = request.get("sets") or []
        if not isinstance(sets, list) or not all(isinstance(item, str) for item in sets):
            raise RunError("`sets` must be a list of `slot=option` strings")
        if len(sets) > 32 or any(len(item) > 200 or "=" not in item for item in sets):
            raise RunError("`sets` holds something that is no `slot=option`")
        if self.weights.empty:
            raise RunError(
                "this server reads no weights. Stop it and start it again with "
                "grasppanda explore --weights-dir DIR (or --weights NAME=FILE).",
                "weights",
            )
        weights = self.weights.resolve(request.get("weights"))
        kind = request.get("input", "example")
        if kind not in ("example", "depth"):
            raise RunError("`input` is `example` or `depth`")
        collision = request.get("collision_filter", False)
        if not isinstance(collision, bool):
            raise RunError("`collision_filter` is true or false")
        given = [
            name
            for name in ("depth", "rgb", "intrinsics", "workspace")
            if request.get(name) not in (None, "", [])
        ]
        if kind == "example":
            if given:
                raise RunError(
                    "the example scene brings its own depth image, colour image and intrinsics; "
                    f"{', '.join(given)} cannot be combined with it",
                    "input",
                )
        elif request.get("depth") in (None, ""):
            raise RunError(
                "choose a depth image: a 16-bit PNG in millimetres or a .npy in metres", "input"
            )
        work = Path(tempfile.mkdtemp(prefix="grasppanda-run-"))
        plan = _Plan(
            method=method,
            config=self.methods[method],
            sets=list(sets),
            weights=weights,
            kind=kind,
            collision_filter=collision,
            work=work,
        )
        try:
            if kind == "depth":
                self._read(plan, request)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return plan

    def _read(self, plan: _Plan, request: Mapping[str, Any]) -> None:
        from grasppanda.apis import InputError
        from grasppanda.apis import io as readers
        from grasppanda.apis.inferencer import _box

        plan.intrinsics = _numbers(request.get("intrinsics"), 4, _INTRINSICS)
        if plan.intrinsics is None:
            raise RunError(_INTRINSICS, "input")
        workspace = _numbers(
            request.get("workspace"),
            6,
            "the workspace box is six numbers, the corners (xmin, ymin, zmin, xmax, ymax, "
            "zmax) of a box in the camera frame in metres; leave all six empty for none",
        )
        stored: dict[str, str] = {}
        try:
            if workspace is not None:
                _box(workspace)  # the inferencer's own check, with its message
            plan.workspace = workspace
            path, plan.depth_name = _store(
                request["depth"], "depth", "depth image", _DEPTH_SUFFIXES, plan.work
            )
            stored[str(path)] = plan.depth_name
            _check_size(path, plan.depth_name)
            plan.depth = readers.read_depth(path)
            if request.get("rgb") not in (None, ""):
                path, plan.rgb_name = _store(
                    request["rgb"], "rgb", "colour image", _RGB_SUFFIXES, plan.work
                )
                stored[str(path)] = plan.rgb_name
                _check_size(path, plan.rgb_name)
                plan.rgb = readers.read_rgb(path)
        except InputError as error:
            # The readers name the temporary file; the page knows the file by its own name.
            text = str(error)
            for path_text, shown in stored.items():
                text = text.replace(path_text, shown)
            raise RunError(text, "input") from None

    def start(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Check a request and start its run in the background; returns the run's status.

        Raises:
            RunError: The request names something unknown, or an upload is refused.
            Busy: Another run is in progress.
        """
        with self._lock:
            if self._job is not None and self._job.state == "running":
                raise Busy(self._job.busy())
        plan = self._plan(request)
        with self._lock:
            if self._job is not None and self._job.state == "running":
                shutil.rmtree(plan.work, ignore_errors=True)
                raise Busy(self._job.busy())
            job = self._job = _Job(plan)
        threading.Thread(
            target=self._run, args=(job, plan), name="grasppanda-run", daemon=True
        ).start()
        return job.status()

    # ------------------------------------------------------------------------ the run

    def _inferencer(self, plan: _Plan, job: _Job) -> tuple[Any, dict[str, Any]]:
        """The inferencer of the plan: the one of the last run when nothing changed."""
        from grasppanda.apis import GraspInferencer
        from grasppanda.engine.checkpoint import file_sha256

        stat = plan.weights.stat()
        key = (
            plan.method,
            tuple(plan.sets),
            str(plan.weights),
            stat.st_mtime_ns,
            stat.st_size,
            self.device,
        )
        if self._kept is not None and self._kept[0] == key:
            job.enter(0, "kept from the last run")
            return self._kept[1], self._kept[2]
        job.enter(0)
        if self._kept is not None:  # one model in memory at a time
            self._kept = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        about: dict[str, Any] = {"post": (), "actions": [], "compose": ""}
        config: Any = plan.config
        if plan.sets:
            config = self._compose(plan, about)
        inferencer = GraspInferencer(config, plan.weights, device=self.device)
        about["sha256"] = file_sha256(plan.weights)
        self._kept = (key, inferencer, about)
        return inferencer, about

    def _compose(self, plan: _Plan, about: dict[str, Any]) -> Path:
        from .compose import ComposeError, compose
        from .explore import _LOCK

        with _LOCK:
            composition = compose(plan.config, plan.sets)
        if not composition.accepted:
            raise ComposeError(
                "compose refuses this selection, so there is no config to run:\n"
                + "\n".join(composition.refusal())
            )
        body, switched = _no_downloads(composition.body)
        if switched:
            about["actions"].append(
                "pretrained: false wherever the selection's presets download released weights: "
                "the weights file supplies every parameter, so nothing is downloaded"
            )
            composition = replace(composition, body=body)
        path = plan.work / "selection.yaml"
        path.write_text(composition.text(path), encoding="utf-8")
        about["post"] = tuple(composition.post)
        about["compose"] = composition.command
        return path

    def example(self) -> Any:
        """The example scene, computed once per server (it takes a few seconds)."""
        with self._scene_lock:
            if self._scene is None:
                from grasppanda.apis import example_scene

                self._scene = example_scene()
            return self._scene

    def _call(self, plan: _Plan, inferencer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """The arguments of the inferencer call and what the picture needs, as demo makes them.

        The example scene is passed exactly as ``grasppanda demo --example`` passes it
        (``_call_arguments`` of :mod:`grasppanda.cli.demo`; a test compares the two).
        """
        if plan.kind == "example":
            scene = self.example()
            call: dict[str, Any] = {
                "depth": scene.depth,
                "intrinsics": scene.intrinsics,
                "rgb": scene.rgb,
                "depth_scale": scene.depth_scale,
                "table_from_camera": scene.table_from_camera,
            }
            crop = ""
            if any(step.uses_ground_truth for step in inferencer.preprocess):
                call["mask"] = scene.mask  # the config crops around object pixels
                crop = "; workspace from the scene's own object mask"
            height, width = scene.depth.shape
            return call, {
                "rgb": scene.rgb,
                "intrinsics": scene.intrinsics,
                "colors": None,
                "caption": "synthetic example scene",
                "line": f"synthetic example scene: {width} x {height}, {len(scene.objects)} "
                f"primitives on a table, no sensor noise (not a camera image){crop}",
            }
        depth = plan.depth
        assert depth is not None and plan.intrinsics is not None
        scale = 1000.0 if depth.dtype.kind in "ui" else 1.0
        unit = "millimetres" if scale == 1000.0 else "metres"
        call = {
            "depth": depth,
            "intrinsics": np.asarray(plan.intrinsics, dtype=np.float64),
            "depth_scale": scale,
        }
        line = (
            f"{plan.depth_name}: {depth.shape[1]} x {depth.shape[0]} {depth.dtype}, read as {unit}"
        )
        if plan.rgb is not None:
            call["rgb"] = plan.rgb
            line += f"; colour {plan.rgb_name}"
        if plan.workspace is not None:
            call["workspace"] = plan.workspace
            line += "; workspace " + " ".join(f"{value:g}" for value in plan.workspace)
        info = {
            "rgb": plan.rgb,
            "intrinsics": call["intrinsics"],
            "colors": None,
            "caption": None,
            "line": line,
        }
        return call, info

    def _run(self, job: _Job, plan: _Plan) -> None:
        try:
            try:
                result = self._predict(job, plan)
            finally:  # gone before the page hears that the run ended
                shutil.rmtree(plan.work, ignore_errors=True)
        except Exception as error:
            job.fail(_explain(error))
        else:
            job.finish(result)

    def _predict(self, job: _Job, plan: _Plan) -> dict[str, Any]:
        from grasppanda.apis import InferenceWarning
        from grasppanda.cli.demo import _draw, _picture_cloud, _shown

        started = time.perf_counter()
        inferencer, about = self._inferencer(plan, job)
        job.enter(1)
        call, info = self._call(plan, inferencer)
        inferencer._warned = set()  # the notes are said on every run, not once per kept model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InferenceWarning)
            observation = inferencer.observe(**call)
        notes = [str(w.message) for w in caught if issubclass(w.category, InferenceWarning)]
        chain = inferencer.chain(collision_filter=plan.collision_filter, post=about["post"] or None)

        job.enter(2)
        clock = time.perf_counter()
        raw = inferencer.predict(observation)
        seconds = time.perf_counter() - clock

        job.enter(3)
        counts, grasps, drawn = [], raw, raw
        for step in chain:  # one step at a time, to say what each one left (as demo does)
            grasps = inferencer.postprocess(grasps, observation, [step])
            counts.append({"step": type(step).__name__, "count": len(grasps)})
            if grasps.frame == raw.frame:
                drawn = grasps
        grasps, drawn = grasps.sort_by_score().to("cpu"), drawn.sort_by_score().to("cpu")

        job.enter(4)
        points, colors = _picture_cloud(observation, info)
        options = SimpleNamespace(size=self.picture_size, top_k=_TOP_K, gif_frames=0)
        _draw(points, colors, drawn, options, plan.work, info["caption"])
        picture = (plan.work / "grasps.png").read_bytes()
        array = grasps.to_graspnet_array()
        buffer = io.BytesIO()
        np.save(buffer, array)

        report = inferencer.weights_report
        config = inferencer.config
        shown_config = (
            f"composed: {plan.target}" if plan.sets else _shown(config.path) or str(plan.config)
        )
        lines = [
            ("config", f"{shown_config}  (hash {config.digest()[:12]})"),
            (
                "weights",
                f"{_shown(plan.weights)}  (sha256 {about['sha256'][:12]}; {len(report.matched)} matched, "
                f"{len(report.missing)} missing, {len(report.unexpected)} unexpected)",
            ),
            *(tuple(line.split(None, 1)) for line in inferencer.describe().splitlines()),
            ("input", info["line"]),
        ]
        kept = len(observation.full_points) if observation.full_points is not None else None
        cloud = f"{len(observation.points):,} points to the model"
        if kept is not None and kept != len(observation.points):
            cloud += f", sampled from {kept:,} in the workspace"
        lines.append(("cloud", cloud))
        lines += [("compose", action) for action in about["actions"]]
        rows = [
            {
                "rank": rank + 1,
                "score": round(float(row[0]), 4),
                "width": round(float(row[1]), 4),
                "depth": round(float(row[3]), 4),
                "position": [round(float(value), 4) for value in row[13:16]],
                "approach": [round(float(row[index]), 3) for index in (4, 7, 10)],
            }
            for rank, row in enumerate(array[:_ROWS])
        ]
        stem = re.sub(r"[^a-z0-9]+", "-", plan.method.lower()).strip("-") or "grasps"
        return {
            "target": plan.target,
            "input": plan.input,
            "picture": base64.b64encode(picture).decode("ascii"),
            "picture_size": list(self.picture_size),
            "drawn": min(_TOP_K, len(drawn)),
            "drawn_frame": drawn.frame,
            "frame": grasps.frame,
            "predicted": len(raw),
            "total": len(grasps),
            "counts": counts,
            "grasps": rows,
            "array": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "array_name": f"{stem}_grasps.npy",
            "array_shape": list(array.shape),
            "seconds": {
                "total": round(time.perf_counter() - started, 2),
                "predict": round(seconds, 2),
            },
            "reused": job.notes.get(0, "") != "",
            "lines": [list(line) for line in lines],
            "notes": notes,
            "command": self._command(plan, inferencer, about),
        }

    def _command(self, plan: _Plan, inferencer: Any, about: Mapping[str, Any]) -> str:
        """The commands that do the same in a terminal, in the server's working directory."""
        import shlex

        from grasppanda.cli.demo import _shown

        lines = []
        config = plan.config
        if plan.sets:
            lines.append(f"{about['compose']} -o selection.yaml")
            config = "selection.yaml"
        parts = [
            "grasppanda demo",
            shlex.quote(config),
            "--weights",
            shlex.quote(_shown(plan.weights)),
        ]
        if plan.kind == "example":
            parts.append("--example")
        else:
            assert plan.intrinsics is not None
            parts += ["--depth", shlex.quote(plan.depth_name)]
            parts += ["--intrinsics", ",".join(f"{value:g}" for value in plan.intrinsics)]
            if plan.rgb_name:
                parts += ["--rgb", shlex.quote(plan.rgb_name)]
            if plan.workspace is not None:
                parts += ["--workspace", *(f"{value:g}" for value in plan.workspace)]
        if plan.collision_filter:
            parts.append("--collision-filter")
        if about["post"]:
            parts += ["--post", *(shlex.quote(text) for text in about["post"])]
        if self.device:
            parts += ["--device", self.device]
        lines.append(" ".join(parts))
        return "\n".join(lines)
