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
* The CPU-only wheel (``pip install spconv``) returns wrong, non-repeatable features
  when OpenMP runs more than one thread (measured with 2.3.8: errors of 0.1 to 0.5 on
  features of order 1; exact with one thread; the CUDA wheels are exact on CPU tensors
  with any number of threads). The first convolution of a CPU tensor is therefore
  preceded by a comparison with a dense convolution, and a failure is an error, not a
  wrong feature map; see :func:`check_cpu_convolution`.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from grasppanda.utils.imports import require

from . import api

sp = require(
    "spconv.pytorch",
    extra="spconv-cu118",
    purpose=(
        "Sparse convolution (backend 'spconv'; choose the spconv-* extra that matches your "
        "CUDA build, or spconv-cpu)"
    ),
)
ConvAlgo = require("spconv.core", extra="spconv-cu118", purpose="Sparse convolution").ConvAlgo

__all__ = [
    "SparseConv3d",
    "SparseInverseConv3d",
    "SubMConv3d",
    "check_cpu_convolution",
    "features",
    "indices",
    "make_tensor",
    "replace_features",
]


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
        module = require(name, extra="spconv-cpu", purpose="Sparse convolution")
        module.get_current_stream = lambda: 0


_allow_cpu_backward()

#: Result of the CPU self-check of this process: ``None`` before it ran, then the failure
#: text, which is empty when the check passed.
_cpu_check: str | None = None


def _dense_mismatch(points: int = 2000, size: int = 16, trials: int = 3) -> float:
    """Largest difference between spconv and a dense convolution on a random grid."""
    generator = torch.Generator().manual_seed(20261012)
    cells = torch.randperm(size**3, generator=generator)[:points]
    cells = torch.stack([cells // size**2, cells // size % size, cells % size], dim=1)
    index = torch.cat([torch.zeros(points, 1, dtype=torch.long), cells], dim=1).int()
    feats = torch.randn(points, 8, generator=generator)
    conv = sp.SubMConv3d(8, 8, 3, padding=1, bias=False, algo=ConvAlgo.Native)
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.2)
        dense = torch.zeros(1, 8, size, size, size)
        dense[0, :, cells[:, 0], cells[:, 1], cells[:, 2]] = feats.T
        # The facade's layout is (out, kx, ky, kz, in); conv3d takes (out, in, kx, ky, kz).
        expected = F.conv3d(dense, conv.weight.permute(0, 4, 1, 2, 3), padding=1)
        expected = expected[0, :, cells[:, 0], cells[:, 1], cells[:, 2]].T
        worst = 0.0
        for _ in range(trials):  # the defect is a race, so one clean pass proves little
            out = conv(sp.SparseConvTensor(feats, index, [size] * 3, 1)).features
            worst = max(worst, float((out - expected).abs().max()))
    return worst


def check_cpu_convolution(*, force: bool = False) -> None:
    """Raise unless spconv convolves CPU tensors correctly in this process.

    The comparison runs once per process (again with ``force``) and costs a few
    milliseconds. It exists for the CPU-only wheel, whose convolution is wrong with more
    than one OpenMP thread; any other cause of a wrong result is caught the same way.

    Raises:
        SparseBackendError: The convolution of a random grid differs from the dense one.
    """
    global _cpu_check
    if _cpu_check is None or force:
        mismatch = _dense_mismatch()
        _cpu_check = ""
        if not mismatch < 1e-4:
            threads = os.environ.get("OMP_NUM_THREADS", "unset")
            _cpu_check = (
                "spconv returns wrong features for CPU tensors in this process: a 3×3×3 "
                f"convolution differs from the dense one by {mismatch:.2g} on values of order "
                "1. The CPU-only wheel (`pip install spconv`) does this whenever OpenMP runs "
                f"more than one thread (OMP_NUM_THREADS is {threads}). Either start Python "
                "with OMP_NUM_THREADS=1, or install a CUDA build of spconv (spconv-cu118, "
                "spconv-cu121, …): their CPU path is correct with any number of threads and "
                "needs no GPU to run."
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
