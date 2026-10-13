"""Which implementation runs each accelerated operator.

Four operators have a second, compiled implementation (:mod:`grasppanda._ops`):

    fps             furthest_point_sample
    ball_query      ball_query
    cylinder_query  cylinder_query
    knn             knn

Each can be set to

* ``reference``: the pure-PyTorch implementation, which defines the result;
* ``compiled``: the C++/CUDA kernels; an error if they cannot be used;
* ``auto`` (default): the kernels when they are available *and* reproduce the
  reference on this machine, otherwise the reference.

The selection comes from ``runtime.ops`` of the config (:func:`configure`) and can be
overridden per process with the environment variable ``GRASPPANDA_OPS``, either one
backend for every operator (``GRASPPANDA_OPS=reference``) or a list
(``GRASPPANDA_OPS=fps=reference,knn=compiled``).

An operator settles on a backend the first time it runs on a device and keeps it for
the life of the process. Before the kernels are used on a device they pass a self-check
there: on small random and lattice inputs with ties, duplicates and empty
neighbourhoods the kernel has to return exactly what the reference returns, and its
squared distances have to equal PyTorch's bit for bit. :func:`report` says what was
chosen and why; it is part of ``grasppanda env`` and of every run snapshot.

The kernels take float32. Other floating types, and kNN when gradients are requested or
``k`` is large, always go through the reference.
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from grasppanda._ops.library import Kernels

__all__ = [
    "ENV",
    "OPERATORS",
    "BackendChoice",
    "OperatorBackendError",
    "configure",
    "kernels_for",
    "override",
    "report",
    "requested",
]

#: Operators with more than one implementation; the keys of ``runtime.ops``.
OPERATORS = ("fps", "ball_query", "cylinder_query", "knn")
#: Values of ``runtime.ops``.
BackendChoice = Literal["auto", "reference", "compiled"]
_CHOICES = ("auto", "reference", "compiled")
#: Environment variable that overrides ``runtime.ops``.
ENV = "GRASPPANDA_OPS"
#: Largest extent the CUDA kernels index with 32-bit arithmetic.
_MAX_EXTENT = (2**31 - 1) // 3

_log = logging.getLogger(__name__)
_lock = threading.RLock()
_configured: dict[str, str] = {}
_resolved: dict[tuple[str, torch.device], _Resolution] = {}
#: Outcome of the self-check per (operator, device), with the kernels it was run on.
_checked: dict[tuple[str, torch.device], tuple[Kernels, str | None]] = {}
_warned: set[str] = set()


class OperatorBackendError(RuntimeError):
    """A backend was requested explicitly and cannot be used."""


@dataclass(frozen=True)
class _Resolution:
    backend: str
    kernels: Kernels | None = None
    #: Why ``auto`` ended on the reference; ``None`` when the backend is what was asked for.
    note: str | None = None


# ------------------------------------------------------------------------ selection


def _validated(selection: Mapping[str, str], source: str) -> dict[str, str]:
    checked = {}
    for operator, backend in selection.items():
        if operator not in OPERATORS:
            close = difflib.get_close_matches(str(operator), OPERATORS, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(
                f"{source}: unknown operator '{operator}'.{hint} Operators with selectable "
                f"backends: {', '.join(OPERATORS)}."
            )
        if backend not in _CHOICES:
            raise ValueError(
                f"{source}.{operator}: unknown backend '{backend}'; expected one of "
                f"{', '.join(_CHOICES)}."
            )
        checked[operator] = backend
    return checked


def configure(selection: Mapping[str, str] | None = None) -> None:
    """Set the backend of each operator for this process, as ``runtime.ops`` gives it.

    Operators that are not mentioned use ``auto``. Calling this again replaces the
    selection and makes every operator settle anew on its next use.

    Raises:
        ValueError: An unknown operator or backend name.
    """
    checked = _validated(selection or {}, "runtime.ops")
    with _lock:
        _configured.clear()
        _configured.update(checked)
        _resolved.clear()


def _from_environment() -> dict[str, str]:
    text = os.environ.get(ENV, "").strip()
    if not text:
        return {}
    if "=" not in text:
        return _validated(dict.fromkeys(OPERATORS, text), ENV)
    pairs = [item.split("=", 1) for item in text.split(",") if item.strip()]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(
            f"{ENV}='{text}': expected one backend, or a list such as 'fps=reference,knn=auto'."
        )
    return _validated({key.strip(): value.strip() for key, value in pairs}, ENV)


def requested(operator: str) -> str:
    """The backend asked for: ``$GRASPPANDA_OPS`` first, then :func:`configure`, then ``auto``."""
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator '{operator}'; expected one of {', '.join(OPERATORS)}")
    return _from_environment().get(operator) or _configured.get(operator, "auto")


@contextlib.contextmanager
def override(**selection: str) -> Iterator[None]:
    """Temporarily select backends, e.g. ``with override(fps="reference"):``.

    For tests and benchmarks that compare backends within one process. Not thread-safe:
    the selection is process-wide. ``$GRASPPANDA_OPS`` still takes precedence.
    """
    checked = _validated(selection, "override")
    with _lock:
        saved_configured, saved_resolved = dict(_configured), dict(_resolved)
        _configured.update(checked)
        _resolved.clear()
    try:
        yield
    finally:
        with _lock:
            _configured.clear()
            _configured.update(saved_configured)
            _resolved.clear()
            _resolved.update(saved_resolved)


# ----------------------------------------------------------------------- resolution


def kernels_for(operator: str, *tensors: Tensor) -> Kernels | None:
    """The compiled kernels if ``operator`` should use them for these tensors, else ``None``.

    ``None`` means "run the reference". That is the answer when the operator is set to
    the reference or settled on it, and for inputs the kernels do not take: anything
    but float32 on one device.

    Raises:
        OperatorBackendError: The operator is set to ``compiled`` and the kernels
            cannot be used on this device.
    """
    first = tensors[0]
    resolution = _resolved.get((operator, first.device))
    if resolution is None:
        resolution = _resolve(operator, first.device)
    if resolution.kernels is None:
        return None
    for tensor in tensors:
        if tensor.dtype != torch.float32 or tensor.device != first.device:
            return None
        if max(tensor.shape, default=0) > _MAX_EXTENT:
            return None
    return resolution.kernels


def _warn_once(key: str, message: str, *arguments: object) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(message, *arguments)


def _resolve(operator: str, device: torch.device) -> _Resolution:
    kind = device.type
    with _lock:
        if (operator, device) in _resolved:
            return _resolved[operator, device]
        want = requested(operator)
        if want == "reference":
            resolution = _Resolution("reference")
        else:
            from grasppanda._ops import library

            kernels, detail = library.kernels()
            if kernels is None:
                problem = f"the compiled kernels are unavailable: {detail}"
                warning = ("library", "Operators use the pure-PyTorch reference (%s).", problem)
            elif kind not in ("cpu", "cuda") or (kind == "cuda" and not kernels.cuda):
                problem = f"the compiled library has no kernels for '{kind}' tensors: {detail}"
                warning = (
                    kind,
                    "Operators on %s use the pure-PyTorch reference (%s).",
                    kind,
                    problem,
                )
            else:
                # Once per process: changing the selection does not change the kernels.
                if _checked.get((operator, device), (None,))[0] is not kernels:
                    _checked[operator, device] = (kernels, _self_check(operator, kernels, device))
                failure = _checked[operator, device][1]
                problem = None if failure is None else f"its self-check failed: {failure}"
                warning = (
                    f"{operator}/{device}",
                    "%s on %s uses the pure-PyTorch reference (the compiled kernel was rejected: %s).",
                    operator,
                    device,
                    failure,
                )
            if problem is None:
                resolution = _Resolution("compiled", kernels)
            elif want == "compiled":
                raise OperatorBackendError(
                    f"operator '{operator}' is set to the 'compiled' backend, but {problem}. "
                    "Build the kernels with `python -m grasppanda._ops.build`, or select 'auto' "
                    "to fall back to the pure-PyTorch reference."
                )
            else:
                _warn_once(*warning)
                resolution = _Resolution("reference", note=problem)
        _resolved[operator, device] = resolution
        return resolution


def report() -> dict[str, Any]:
    """Backend of every operator on the CPU and on the current CUDA device; never raises.

    Settles every operator (which may build the kernels and runs their self-checks), so
    that what is reported is what the process uses. ``cuda`` is the current device of
    PyTorch (``torch.cuda.current_device()``), absent without one; another GPU of the
    same machine runs its own self-check when an operator first uses it::

        {"fps": {"requested": "auto", "cpu": "compiled", "cuda": "reference",
                 "note": {"cuda": "the compiled library has no kernels for ..."}},
         ...,
         "library": {"path": ".../grasppanda_ops_....so", "cuda": false, "detail": "..."}}
    """
    devices = {"cpu": torch.device("cpu")}
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            devices["cuda"] = torch.device("cuda", torch.cuda.current_device())
    out: dict[str, Any] = {}
    loaded = False
    for operator in OPERATORS:
        try:
            entry: dict[str, Any] = {"requested": requested(operator)}
        except ValueError as error:  # a malformed $GRASPPANDA_OPS
            out[operator] = {"requested": f"invalid ({error})"}
            continue
        notes = {}
        for kind, device in devices.items():
            try:
                resolution = _resolve(operator, device)
            except Exception as error:
                entry[kind] = "unavailable"
                notes[kind] = str(error)
                continue
            entry[kind] = resolution.backend
            if resolution.note:
                notes[kind] = resolution.note
        if notes:
            entry["note"] = notes
        loaded = loaded or entry["requested"] != "reference"
        out[operator] = entry
    library_entry: dict[str, Any] = {"path": None, "cuda": False, "detail": "not needed"}
    if loaded:
        with contextlib.suppress(Exception):
            from grasppanda._ops import library

            kernels, detail = library.kernels()
            library_entry = {
                "path": str(kernels.path) if kernels else None,
                "cuda": bool(kernels and kernels.cuda),
                "detail": detail,
            }
    out["library"] = library_entry
    return out


# ----------------------------------------------------------------------- self-check


def _self_check(operator: str, kernels: Kernels, device: torch.device) -> str | None:
    """Run the kernel of ``operator`` against the reference on ``device``; ``None`` when identical."""
    try:
        with torch.no_grad(), _full_precision(device.type):
            failure = _probe_mismatch(kernels, device) or _CHECKS[operator](kernels, device)
    except Exception as error:  # a kernel that cannot run is a kernel that failed
        return f"{type(error).__name__}: {error}"
    return failure


@contextlib.contextmanager
def _full_precision(kind: str) -> Iterator[None]:
    """Plain float32 arithmetic, whatever mode the first caller of an operator is in.

    Under autocast or with TF32 matrix products the reference itself computes in
    reduced precision (see ``_rotation_is_float32`` in ``query.py``); the kernels are
    judged against the full-precision reference.
    """
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=kind, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _cloud(count: int, seed: int, device: torch.device, *, lattice: bool = False) -> Tensor:
    """A reproducible cloud; on a lattice it is full of exact ties and repeated points."""
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(2, count, 3, generator=generator)
    if lattice:
        points = torch.round(points * 4) / 8  # 5 values per axis, spacing 1/8
    return points.to(device)


def _probe_mismatch(kernels: Kernels, device: torch.device) -> str | None:
    from .sampling import _squared_norm

    generator = torch.Generator().manual_seed(0)
    # Components of very different size: the order of the three-term sum then shows.
    scale = torch.tensor([1.0, 1e-3, 1e3])
    first = (torch.randn(4096, 3, generator=generator) * scale.flip(0)).to(device)
    second = (torch.randn(4096, 3, generator=generator) * scale).to(device)
    norm, distance = kernels.probe(first, second)
    if not torch.equal(norm, _squared_norm(first - second)):
        return "its squared norm differs from PyTorch's (a fused multiply-add?)"
    if not torch.equal(distance, (first[None, None] - second[None, None]).pow(2).sum(-1)[0, 0]):
        return "its sum of squares is not ordered like PyTorch's sum(-1) on this device"
    return None


def _check_fps(kernels: Kernels, device: torch.device) -> str | None:
    from . import sampling

    for seed, (count, lattice) in enumerate([(257, False), (300, True), (1200, True), (5, False)]):
        points = _cloud(count, seed, device, lattice=lattice)
        points[:, min(3, count - 1)] = 0.01  # a point the near-origin rule skips
        for skip in (False, True):
            for tie_break in ("lowest_index", "pointnet2_cuda"):
                wanted = min(count, 96)
                expected = sampling._sample(points, wanted, skip, tie_break)
                got = sampling._sample_compiled(kernels, points, wanted, skip, tie_break)
                if not torch.equal(got, expected):
                    return f"picks differ ({count} points, skip={skip}, {tie_break})"
    return None


def _check_ball_query(kernels: Kernels, device: torch.device) -> str | None:
    from . import query

    for seed, lattice in enumerate([False, True]):
        support = _cloud(400, seed, device, lattice=lattice)
        centres = torch.cat(
            [support[:, :60], support[:, :4] + 5.0], dim=1
        )  # the last four are empty
        for radius, nsample in ((0.125, 8), (0.25, 40), (0.3, 500)):
            expected = query._ball_query(centres, support, radius, nsample, 256)
            got = query._ball_query_compiled(kernels, centres, support, radius, nsample)
            if not all(torch.equal(a, b) for a, b in zip(got, expected)):
                return f"neighbourhoods differ (radius {radius}, nsample {nsample})"
    return None


def _check_cylinder_query(kernels: Kernels, device: torch.device) -> str | None:
    from . import query
    from .views import approach_to_rotation

    for seed, lattice in enumerate([False, True]):
        support = _cloud(400, seed, device, lattice=lattice)
        centres = torch.cat([support[:, :60], support[:, :4] + 5.0], dim=1)
        generator = torch.Generator().manual_seed(seed)
        approach = torch.randn(2, 64, 3, generator=generator)
        approach[:, ::3] = torch.tensor([0.0, 0.0, 1.0])  # axis-aligned: exact boundary hits
        rotation = approach_to_rotation(approach).to(device)
        # Queries the kernel cannot decide are re-run one per batch entry; the rotated
        # offsets of the reference must not depend on that regrouping.
        offset = support[:, None] - centres[:, :, None]
        together = torch.einsum("bmnj,bmjk->bmnk", offset, rotation)
        apart = torch.einsum(
            "bmnj,bmjk->bmnk", offset.flatten(0, 1)[:, None], rotation.flatten(0, 1)[:, None]
        )
        if not torch.equal(together.flatten(0, 1), apart[:, 0]):
            return "the reference's rotation depends on how the queries are batched"
        for radius, hmin, hmax, nsample in ((0.125, -0.125, 0.25, 8), (0.3, 0.0, 0.5, 40)):
            arguments = (centres, support, rotation, radius, hmin, hmax, nsample)
            expected = query._cylinder_query(*arguments, 256)
            got = query._cylinder_query_compiled(kernels, *arguments, 256)
            if not all(torch.equal(a, b) for a, b in zip(got, expected)):
                return f"neighbourhoods differ (radius {radius}, nsample {nsample})"
    return None


def _check_knn(kernels: Kernels, device: torch.device) -> str | None:
    from . import query

    for seed, lattice in enumerate([False, True]):
        support = _cloud(300, seed, device, lattice=lattice)
        centres = _cloud(70, seed + 10, device, lattice=lattice)
        for k in (1, 3, 16):
            expected = query._knn(centres, support, k, 1024)
            got = query._knn_compiled(kernels, centres, support, k)
            if not all(torch.equal(a, b) for a, b in zip(got, expected)):
                return f"neighbours or distances differ (k={k})"
    return None


_CHECKS = {
    "fps": _check_fps,
    "ball_query": _check_ball_query,
    "cylinder_query": _check_cylinder_query,
    "knn": _check_knn,
}
