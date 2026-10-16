"""spconv 2.x backend of the sparse facade.

This module is imported the first time a sparse layer or tensor is created, never at
``import grasppanda`` time. spconv's native weight layout ``(out, kx, ky, kz, in)`` and
its kernel-offset convention are the facade's, so the layers below only pin down
arguments; they add no computation.

Two spconv behaviours are handled here so that networks do not have to know them:

* Indices must be non-negative ``int32`` inside a declared spatial shape, so
  :func:`make_tensor` shifts the grid and returns the shift.
* The default GPU algorithm (implicit GEMM) does not run on CPU tensors. Each layer
  picks the algorithm from the device of its input, which keeps one model usable on
  both.

Two defects of spconv 2.3 on machines without a GPU are handled here as well:

* Its backward functions ask PyTorch for the current CUDA stream before they look at the
  device of their input, so without a usable CUDA nothing could be trained on the CPU.
  The stream is not used for CPU tensors; see :func:`_allow_cpu_backward`.
* The CPU-only wheel (``pip install spconv``) returns wrong, non-repeatable features and
  input gradients when PyTorch runs more than one intra-op thread (measured with 2.3.8:
  errors of 0.1 to 3 on values of order 1; exact with one thread; the CUDA wheels are
  exact on CPU tensors with any number of threads). Its convolution and its backward on
  CPU tensors therefore run under ``torch.set_num_threads(1)``, which is restored
  afterwards, so the rest of the model keeps every thread; see :func:`one_thread`. The
  first convolution of a CPU tensor is still preceded by a comparison with a dense
  convolution, and a failure is an error, not a wrong feature map; see
  :func:`check_cpu_convolution`.
"""

from __future__ import annotations

import functools
import importlib
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from grasppanda.utils.imports import require

from . import api

sp = require("spconv.pytorch", extra="spconv", purpose="Sparse convolution (backend 'spconv')")
ConvAlgo = require("spconv.core", extra="spconv", purpose="Sparse convolution").ConvAlgo

__all__ = [
    "SparseConv3d",
    "SparseInverseConv3d",
    "SubMConv3d",
    "check_cpu_convolution",
    "cpu_thread_note",
    "features",
    "indices",
    "make_tensor",
    "one_thread",
    "replace_features",
]

_log = logging.getLogger(__name__)


def make_tensor(feats: Tensor, voxels: Tensor, batch_size: int, align: int) -> tuple[Any, Tensor]:
    cells = voxels[:, 1:].long()
    origin = torch.div(cells.amin(dim=0), align, rounding_mode="floor") * align
    extent = cells.amax(dim=0) - origin + 1
    # Rounding the shape up to a multiple of ``align`` makes every level halve exactly.
    shape = torch.div(extent + align - 1, align, rounding_mode="floor") * align
    index = torch.cat([voxels[:, :1].long(), cells - origin], dim=1).int()
    tensor = sp.SparseConvTensor(feats.contiguous(), index, shape.tolist(), batch_size)
    return tensor, origin


def features(tensor: Any) -> Tensor:
    return tensor.features


def indices(tensor: Any) -> Tensor:
    return tensor.indices


def replace_features(tensor: Any, feats: Tensor) -> Any:
    return tensor.replace_feature(feats)


def _allow_cpu_backward() -> None:
    """Let spconv back-propagate through CPU tensors on a machine without CUDA.

    Only where PyTorch has no usable CUDA, and the unpatched lookup could do nothing but
    raise, it is replaced by one that returns a null stream. Code paths for CUDA tensors
    cannot be reached there, and the CPU paths never read the value.
    """
    if torch.cuda.is_available():
        return
    for name in ("spconv.pytorch.ops", "spconv.pytorch.cppcore"):
        module = require(name, extra="spconv", purpose="Sparse convolution")
        module.get_current_stream = lambda: 0


_allow_cpu_backward()


@contextmanager
def one_thread() -> Iterator[None]:
    """Run the enclosed PyTorch work on one intra-op thread, then restore the count."""
    threads = torch.get_num_threads()
    if threads == 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


#: The thread count PyTorch had when a convolution of a CPU tensor first ran on one thread
#: in this process; 0 before.
_limited_from = 0


def _on_one_thread(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function(features, ...)`` of spconv, run under :func:`one_thread` for CPU tensors."""

    @functools.wraps(function)
    def wrapper(features: Tensor, *args: Any, **kwargs: Any) -> Any:
        global _limited_from
        if features.is_cuda:
            return function(features, *args, **kwargs)
        if not _limited_from and torch.get_num_threads() > 1:
            _limited_from = torch.get_num_threads()
            _log.info(cpu_thread_note())
        with one_thread():
            return function(features, *args, **kwargs)

    wrapper.grasppanda_one_thread = True  # type: ignore[attr-defined]
    return wrapper


def _cpu_only_build() -> bool:
    try:
        return bool(importlib.import_module("spconv.cppconstants").CPU_ONLY_BUILD)
    except (ImportError, AttributeError):
        return False


def _run_cpu_convolutions_on_one_thread() -> bool:
    """For the CPU-only wheel, wrap the two functions every spconv convolution layer calls
    (forward and backward of the native algorithm). True when they are wrapped."""
    if not _cpu_only_build():
        return False
    ops = require("spconv.pytorch.ops", extra="spconv", purpose="Sparse convolution")
    names = ("indice_conv", "indice_conv_backward")
    if not all(callable(getattr(ops, name, None)) for name in names):
        return False  # another spconv layout: the self-check below still guards
    for name in names:
        function = getattr(ops, name)
        if getattr(function, "grasppanda_one_thread", False):  # this module imported again
            function = function.__wrapped__
        setattr(ops, name, _on_one_thread(function))
    return True


#: Whether convolutions of CPU tensors run on one thread in this process.
ONE_THREAD = _run_cpu_convolutions_on_one_thread()


def cpu_thread_note() -> str | None:
    """One sentence saying that sparse convolutions of CPU tensors ran on one thread, or
    ``None`` when none did."""
    if not _limited_from:
        return None
    return (
        "sparse convolution of CPU tensors runs on one thread (the CPU-only spconv wheel "
        f"computes wrong features with more); the other operators use {_limited_from}"
    )


#: Result of the CPU self-check of this process: ``None`` before it ran, then the failure
#: text, which is empty when the check passed.
_cpu_check: str | None = None


def _dense_mismatch(points: int = 2000, size: int = 16, trials: int = 3) -> float:
    """Largest difference between spconv and a dense convolution on a random grid, in the
    features and in their gradient."""
    # Inference code may convolve first, but the comparison needs autograd.
    with torch.inference_mode(False), torch.enable_grad():
        return _compare_with_dense(points, size, trials)


def _compare_with_dense(points: int, size: int, trials: int) -> float:
    generator = torch.Generator().manual_seed(20261012)
    cells = torch.randperm(size**3, generator=generator)[:points]
    cells = torch.stack([cells // size**2, cells // size % size, cells % size], dim=1)
    index = torch.cat([torch.zeros(points, 1, dtype=torch.long), cells], dim=1).int()
    flat = (cells[:, 0] * size + cells[:, 1]) * size + cells[:, 2]
    feats = torch.randn(points, 8, generator=generator)
    target = torch.randn(points, 8, generator=generator)
    conv = sp.SubMConv3d(8, 8, 3, padding=1, bias=False, algo=ConvAlgo.Native)
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.2)
    # The facade's layout is (out, kx, ky, kz, in); conv3d takes (out, in, kx, ky, kz).
    weight = conv.weight.detach().permute(0, 4, 1, 2, 3)
    dense_feats = feats.clone().requires_grad_(True)
    grid = torch.zeros(size**3, 8).index_add(0, flat, dense_feats)
    expected = F.conv3d(grid.T.reshape(1, 8, size, size, size), weight, padding=1)
    expected = expected[0].reshape(8, -1).T[flat]
    (expected * target).sum().backward()
    worst = 0.0
    for _ in range(trials):  # the defect is a race, so one clean pass proves little
        sparse_feats = feats.clone().requires_grad_(True)
        out = conv(sp.SparseConvTensor(sparse_feats, index, [size] * 3, 1)).features
        (out * target).sum().backward()
        worst = max(
            worst,
            float((out.detach() - expected.detach()).abs().max()),
            float((sparse_feats.grad - dense_feats.grad).abs().max()),
        )
    return worst


def check_cpu_convolution(*, force: bool = False) -> None:
    """Raise unless spconv convolves CPU tensors correctly in this process.

    The comparison of a convolution and its input gradient with dense ones runs once per
    process (again with ``force``) and costs a few milliseconds. It exists for the CPU-only
    wheel, whose convolution is wrong with more than one thread unless it runs under
    :func:`one_thread`; any other cause of a wrong result is caught the same way.

    Raises:
        SparseBackendError: The convolution of a random grid differs from the dense one.
    """
    global _cpu_check
    if _cpu_check is None or force:
        mismatch = _dense_mismatch()
        _cpu_check = ""
        if not mismatch < 1e-4:
            if ONE_THREAD:
                cause = (
                    "Its CPU convolutions already run on one thread, which makes the CPU-only "
                    "wheel exact, so the cause is another one."
                )
            else:
                cause = (
                    "The CPU-only wheel (`pip install spconv`) does this whenever PyTorch runs "
                    f"more than one thread (here {torch.get_num_threads()}); starting Python "
                    "with OMP_NUM_THREADS=1 avoids it."
                )
            _cpu_check = (
                "spconv returns wrong results for CPU tensors in this process: a 3×3×3 "
                f"convolution or its gradient differs from the dense one by {mismatch:.2g} on "
                f"values of order 1. {cause} A CUDA build of spconv (extras spconv-cu118, "
                "spconv-cu121, …) computes CPU tensors correctly with any number of threads "
                "and needs no GPU to run."
            )
    if _cpu_check:
        raise api.SparseBackendError(_cpu_check)


class _DeviceAlgorithm:
    """Choose spconv's algorithm per call: native on CPU, the layer's default on CUDA."""

    def forward(self, tensor: Any) -> Any:
        if tensor.features.is_cuda:
            self.algo = self._cuda_algo
        else:
            check_cpu_convolution()
            self.algo = ConvAlgo.Native
        return super().forward(tensor)


class SubMConv3d(_DeviceAlgorithm, api.SubMConv3d, sp.SubMConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = False,
        key: str | None = None,
    ) -> None:
        if kernel_size % 2 != 1:
            raise ValueError(f"SubMConv3d needs an odd kernel size, got {kernel_size}")
        sp.SubMConv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            indice_key=key,
        )
        self._cuda_algo = self.algo


class SparseConv3d(_DeviceAlgorithm, api.SparseConv3d, sp.SparseConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        bias: bool = False,
        key: str | None = None,
    ) -> None:
        if kernel_size != stride:
            raise ValueError(
                "SparseConv3d is defined for non-overlapping down-sampling only "
                f"(kernel_size == stride), got kernel_size={kernel_size}, stride={stride}"
            )
        sp.SparseConv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
            indice_key=key,
        )
        self._cuda_algo = self.algo


class SparseInverseConv3d(_DeviceAlgorithm, api.SparseInverseConv3d, sp.SparseInverseConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        key: str,
        bias: bool = False,
    ) -> None:
        sp.SparseInverseConv3d.__init__(
            self, in_channels, out_channels, kernel_size, indice_key=key, bias=bias
        )
        self._cuda_algo = self.algo
