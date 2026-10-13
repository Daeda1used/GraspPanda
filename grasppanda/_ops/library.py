"""The compiled kernels as functions on tensors.

:func:`kernels` loads the library once per process (building it on first use, see
:mod:`grasppanda._ops.build`) and returns a :class:`Kernels` object, or ``None`` and the
reason. The methods of :class:`Kernels` are thin: they hand buffers to the kernels and
allocate the results. Argument checking, the choice between this backend and the
reference, and everything that defines an operator live in :mod:`grasppanda.ops`.

All methods take float32 tensors on one device (CPU, or CUDA when :attr:`Kernels.cuda`).
CUDA kernels are queued on the current stream like any PyTorch operation and are not
waited for.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

import torch
from torch import Tensor

from . import build

__all__ = ["Kernels", "kernels", "reset"]

#: Must equal GP_ABI_VERSION of src/common.h.
_ABI_VERSION = 1
_TIE_BREAKS = {"lowest_index": 0, "pointnet2_cuda": 1}

_int, _long, _double, _pointer = ctypes.c_int, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
_SHAPE = (_long, _long, _long)  # batch, queries, support points
_CUDA = (_int, _pointer)  # device index, stream
_PROTOTYPES = {
    "gp_fps_cpu": (_pointer, _long, _long, _long, _int, _double, _int, _int, _pointer, _int),
    "gp_ball_query_cpu": (_pointer, _pointer, *_SHAPE, _double, _long, _pointer, _pointer, _int),
    "gp_cylinder_query_cpu": (
        *(_pointer, _pointer, _pointer, *_SHAPE),
        *(_double, _double, _double, _long, _pointer, _pointer, _int),
    ),
    "gp_knn_cpu": (_pointer, _pointer, *_SHAPE, _long, _pointer, _pointer, _int),
    "gp_probe_cpu": (_pointer, _pointer, _long, _pointer, _pointer),
}
_CUDA_PROTOTYPES = {
    "gp_fps_cuda": (
        *(_pointer, _pointer, _long, _long, _long, _int, _double, _int, _int, _pointer),
        *_CUDA,
    ),
    "gp_ball_query_cuda": (
        *(_pointer, _pointer, *_SHAPE, _double, _long, _pointer, _pointer),
        *_CUDA,
    ),
    "gp_cylinder_query_cuda": (
        *(_pointer, _pointer, _pointer, *_SHAPE),
        *(_double, _double, _double, _long, _pointer, _pointer),
        *_CUDA,
    ),
    "gp_knn_cuda": (_pointer, _pointer, *_SHAPE, _long, _pointer, _pointer, *_CUDA),
    "gp_probe_cuda": (_pointer, _pointer, _long, _pointer, _pointer, *_CUDA),
}


def _address(tensor: Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


class Kernels:
    """A loaded kernel library.

    Attributes:
        path: The shared library.
        cuda: Whether it has CUDA kernels.
        openmp: Whether its CPU kernels are multi-threaded.
        detail: What the library contains, in a sentence.
    """

    def __init__(self, path: Path, detail: str) -> None:
        library = ctypes.CDLL(str(path))
        version = library.gp_abi_version()
        if version != _ABI_VERSION:
            raise RuntimeError(f"{path} has interface version {version}, expected {_ABI_VERSION}")
        self.path = path
        self.detail = detail
        self.cuda = bool(library.gp_has_cuda())
        self.openmp = bool(library.gp_has_openmp())
        prototypes = {**_PROTOTYPES, **(_CUDA_PROTOTYPES if self.cuda else {})}
        for name, arguments in prototypes.items():
            function = getattr(library, name)
            function.argtypes = arguments
            function.restype = _int
        if self.cuda:
            library.gp_cuda_error.argtypes = (_int,)
            library.gp_cuda_error.restype = ctypes.c_char_p
        self._library = library
        self._process = os.getpid()

    # ------------------------------------------------------------------- plumbing

    def _run(self, name: str, on: Tensor, *arguments: object) -> None:
        """Call kernel ``name`` on the device of ``on`` and raise if it reports an error."""
        if on.is_cuda:
            device = on.device.index
            if device is None:
                device = torch.cuda.current_device()
            stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
            status = getattr(self._library, f"{name}_cuda")(*arguments, device, stream)
            if status:
                message = self._library.gp_cuda_error(status).decode()
                raise RuntimeError(f"compiled operator {name} failed on {on.device}: {message}")
        else:
            # A forked child (a data-loader worker, say) inherits this library but not
            # the threads of its OpenMP pool; GNU OpenMP can hang there, so it runs serially.
            threads = torch.get_num_threads() if os.getpid() == self._process else 1
            status = getattr(self._library, f"{name}_cpu")(*arguments, threads)
            if status:
                raise RuntimeError(f"compiled operator {name} failed: out of memory")

    # -------------------------------------------------------------------- kernels

    def fps(
        self,
        points: Tensor,
        num_samples: int,
        *,
        skip_near_origin: bool,
        near_origin: float,
        tie_break: str,
        priority_threads: int,
    ) -> Tensor:
        """``(B, num_samples)`` indices; ``points`` is ``(B, N, 3)`` with ``N >= 1``."""
        points = points.contiguous()
        batch, count, _ = points.shape
        index = torch.empty(batch, num_samples, dtype=torch.long, device=points.device)
        if batch == 0:
            return index
        # Working memory of the CUDA kernel (x, y, z and the running distance of every
        # point); the CPU kernel allocates its own. Like any temporary of a queued
        # PyTorch operation it may be released before the kernel has run: the allocator
        # only hands it out again to work queued behind it on the same stream.
        scratch = (points.new_empty(batch, count, 4),) if points.is_cuda else ()
        self._run(
            "gp_fps",
            points,
            _address(points),
            *(_address(buffer) for buffer in scratch),
            *(batch, count, num_samples, int(skip_near_origin), near_origin),
            *(_TIE_BREAKS[tie_break], priority_threads, _address(index)),
        )
        return index

    def ball_query(
        self, query: Tensor, support: Tensor, radius_squared: float, nsample: int
    ) -> tuple[Tensor, Tensor]:
        """``(B, M, nsample)`` indices and ``(B, M)`` hit counts."""
        query, support = query.contiguous(), support.contiguous()
        batch, queries, _ = query.shape
        index = torch.empty(batch, queries, nsample, dtype=torch.long, device=query.device)
        hits = torch.empty(batch, queries, dtype=torch.long, device=query.device)
        if index.numel():
            self._run(
                "gp_ball_query",
                query,
                *(_address(query), _address(support), batch, queries, support.shape[1]),
                *(radius_squared, nsample, _address(index), _address(hits)),
            )
        return index, hits

    def cylinder_query(
        self,
        query: Tensor,
        support: Tensor,
        rotation: Tensor,
        radius_squared: float,
        hmin: float,
        hmax: float,
        nsample: int,
    ) -> tuple[Tensor, Tensor]:
        """As :meth:`ball_query`; ``hits`` is -1 for a query the kernel could not decide.

        See ``gp_cylinder_test`` in ``src/common.h``: such a query has a support point
        within floating-point round-off of the cylinder surface and has to be answered
        by the reference.
        """
        query, support, rotation = query.contiguous(), support.contiguous(), rotation.contiguous()
        batch, queries, _ = query.shape
        index = torch.empty(batch, queries, nsample, dtype=torch.long, device=query.device)
        hits = torch.empty(batch, queries, dtype=torch.long, device=query.device)
        if index.numel():
            self._run(
                "gp_cylinder_query",
                query,
                *(_address(query), _address(support), _address(rotation)),
                *(batch, queries, support.shape[1], radius_squared, hmin, hmax, nsample),
                *(_address(index), _address(hits)),
            )
        return index, hits

    def knn(self, query: Tensor, support: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """``(B, M, k)`` *squared* distances, ascending, and indices; needs ``1 <= k <= N``."""
        query, support = query.contiguous(), support.contiguous()
        batch, queries, _ = query.shape
        distance = query.new_empty(batch, queries, k)
        index = torch.empty(batch, queries, k, dtype=torch.long, device=query.device)
        if index.numel():
            self._run(
                "gp_knn",
                query,
                *(_address(query), _address(support), batch, queries, support.shape[1]),
                *(k, _address(distance), _address(index)),
            )
        return distance, index

    def probe(self, first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
        """The kernels' two squared distances between rows of two ``(N, 3)`` tensors.

        First the left-to-right sum of furthest point sampling, then the device-ordered
        sum of the queries. The self-check compares both with PyTorch bit for bit.
        """
        first, second = first.contiguous(), second.contiguous()
        norm, distance = first.new_empty(len(first)), first.new_empty(len(first))
        if len(first):
            self._run(
                "gp_probe",
                first,
                *(_address(first), _address(second), len(first)),
                *(_address(norm), _address(distance)),
            )
        return norm, distance


# ------------------------------------------------------------------ process state

_lock = threading.Lock()
_loaded: tuple[Kernels | None, str] | None = None


def kernels() -> tuple[Kernels | None, str]:
    """The kernels of this process and a sentence describing them, or ``None`` and why.

    The first call locates the library, building it if necessary; the result (also a
    failure) is kept for the life of the process.
    """
    global _loaded
    with _lock:
        if _loaded is None:
            found = build.ensure()
            if found.path is None:
                _loaded = (None, found.detail)
            else:
                try:
                    _loaded = (Kernels(found.path, found.detail), found.detail)
                except (OSError, AttributeError, RuntimeError) as error:
                    _loaded = (None, f"cannot load {found.path}: {error}")
        return _loaded


def reset() -> None:
    """Forget the loaded library, so that the next :func:`kernels` call looks again."""
    global _loaded
    with _lock:
        _loaded = None
