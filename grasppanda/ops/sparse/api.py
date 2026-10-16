"""The backend-independent half of the sparse facade: names, contracts, dispatch."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "BACKENDS",
    "BatchNorm",
    "Linear",
    "ReLU",
    "SparseBackendError",
    "SparseConv3d",
    "SparseInverseConv3d",
    "SubMConv3d",
    "cat",
    "check_cpu_convolution",
    "convert_minkowski_kernel",
    "cpu_thread_note",
    "features",
    "features_at",
    "get_backend",
    "indices",
    "make_tensor",
    "replace_features",
    "set_backend",
]

#: Backend name → module that implements the facade.
BACKENDS: dict[str, str] = {"spconv": "grasppanda.ops.sparse._spconv"}


class SparseBackendError(RuntimeError):
    """The installed sparse-convolution library cannot be trusted in this process."""


_active = "spconv"


def set_backend(name: str) -> None:
    """Select the backend used by layers and tensors created from now on."""
    global _active
    if name not in BACKENDS:
        raise ValueError(
            f"unknown sparse backend '{name}'. Available: {', '.join(sorted(BACKENDS))}."
        )
    _active = name


def get_backend() -> str:
    """Name of the active backend."""
    return _active


def _backend() -> ModuleType:
    """Import the active backend. This is where a missing optional dependency surfaces."""
    return importlib.import_module(BACKENDS[_active])


def check_cpu_convolution() -> None:
    """Raise :class:`SparseBackendError` unless the backend convolves CPU tensors correctly.

    The layers of the facade call this themselves. Code that drives the backend library
    directly (PTv3 keeps the library's own layers so that official checkpoints load by
    name) calls it before it convolves a CPU tensor. The comparison runs once per process.
    """
    _backend().check_cpu_convolution()


def cpu_thread_note() -> str | None:
    """A sentence for reports when the backend ran convolutions of CPU tensors on fewer
    threads than PyTorch has (the CPU-only spconv wheel), else ``None``. Never imports the
    backend."""
    module = sys.modules.get(BACKENDS[_active])
    note = getattr(module, "cpu_thread_note", None)
    return note() if note is not None else None


# ----------------------------------------------------------------------------- tensors


def make_tensor(
    feats: Tensor, voxels: Tensor, batch_size: int, *, align: int = 1
) -> tuple[Any, Tensor]:
    """Build a sparse tensor from occupied voxels.

    Args:
        feats: ``(V, C)`` one feature row per voxel.
        voxels: ``(V, 4)`` integer ``[batch, x, y, z]``, unique rows. Coordinates may be
            negative.
        batch_size: Number of samples in the batch.
        align: The grid is shifted by a multiple of ``align`` voxels only. A network
            that down-samples by a total factor ``s`` passes ``align=s``, so that which
            voxels merge at each level depends on their absolute coordinates and not on
            where the cloud happens to start.

    Returns:
        ``(tensor, origin)``. ``origin`` is the ``(3,)`` voxel coordinate that the
        backend's index ``(0, 0, 0)`` stands for.
    """
    if voxels.ndim != 2 or voxels.shape[1] != 4 or voxels.is_floating_point():
        raise ValueError(
            f"voxels must be integer [batch, x, y, z] rows, got {tuple(voxels.shape)} {voxels.dtype}"
        )
    if feats.ndim != 2 or feats.shape[0] != voxels.shape[0]:
        raise ValueError(
            f"need one feature row per voxel: {tuple(feats.shape)} features for {len(voxels)} voxels"
        )
    if len(voxels) == 0:
        raise ValueError("cannot build a sparse tensor without voxels")
    if align < 1:
        raise ValueError("align must be a positive integer")
    return _backend().make_tensor(feats, voxels, batch_size, align)


def features(tensor: Any) -> Tensor:
    """``(V, C)`` feature rows of a sparse tensor."""
    return _backend().features(tensor)


def indices(tensor: Any) -> Tensor:
    """``(V, 4)`` ``[batch, i, j, k]`` rows of a sparse tensor, relative to its origin
    and in units of its own (possibly down-sampled) grid."""
    return _backend().indices(tensor)


def replace_features(tensor: Any, feats: Tensor) -> Any:
    """The same voxels carrying new features."""
    return _backend().replace_features(tensor, feats)


def features_at(tensor: Any, inverse: Tensor) -> Tensor:
    """Per-point features: row ``inverse[n]`` of the tensor for every point ``n``."""
    return features(tensor)[inverse]


def cat(tensors: Sequence[Any]) -> Any:
    """Concatenate the channels of tensors that live on the same voxels, in order."""
    first = tensors[0]
    for other in tensors[1:]:
        if not torch.equal(indices(first), indices(other)):
            raise ValueError(
                "cannot concatenate sparse tensors that live on different voxel sets; a skip "
                "connection must join a tensor with the up-sampled tensor of the same level"
            )
    return replace_features(first, torch.cat([features(t) for t in tensors], dim=1))


# ------------------------------------------------------------------------------ layers


class _BackendLayer(nn.Module):
    """A layer whose implementation comes from the active backend.

    Instantiating the facade class returns the backend's subclass of it, so
    ``isinstance(layer, sparse.SubMConv3d)`` holds, parameters keep plain names
    (``weight``, ``bias``) and no wrapper level appears in a state dict. Facade layers
    are implemented by backends only; they are not meant to be subclassed elsewhere.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> _BackendLayer:
        if cls.__module__ == __name__:  # the abstract facade class itself
            cls = getattr(_backend(), cls.__name__)
        return super().__new__(cls)


class SubMConv3d(_BackendLayer):
    """Submanifold convolution: outputs live on exactly the input voxels.

    ``out[v] = Σ_{i,j,k} weight[:, i, j, k, :] @ in[v + (i - r, j - r, k - r)]`` with
    ``r = kernel_size // 2``, summing over the occupied neighbours only.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Odd edge length of the cubic kernel.
        bias: Add a learned bias.
        key: Layers with the same key and kernel size on the same voxel set share the
            neighbourhood search.
    """

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = False,
        key: str | None = None,
    ) -> None:  # pragma: no cover - the backend subclass is what gets constructed
        raise NotImplementedError


class SparseConv3d(_BackendLayer):
    """Down-sampling convolution over non-overlapping blocks of ``stride`` voxels.

    Output voxels are ``unique(floor(v / stride))``, and
    ``out[u] = Σ_{i,j,k} weight[:, i, j, k, :] @ in[stride * u + (i, j, k)]``.
    Only ``kernel_size == stride`` is defined by the facade so far.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Edge length of the cubic kernel.
        stride: Down-sampling factor; must equal ``kernel_size``.
        bias: Add a learned bias.
        key: Names this down-sampling step so that a :class:`SparseInverseConv3d` can
            undo it.
    """

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        bias: bool = False,
        key: str | None = None,
    ) -> None:  # pragma: no cover
        raise NotImplementedError


class SparseInverseConv3d(_BackendLayer):
    """Up-sampling that restores the voxels a :class:`SparseConv3d` consumed.

    With ``v = stride * u + (i, j, k)`` the fine voxel in cell ``(i, j, k)`` of coarse
    voxel ``u``: ``out[v] = weight[:, i, j, k, :] @ in[u]``. Output rows are exactly the
    input rows of the paired down-sampling layer, in the same order, so the result can
    be concatenated with a skip tensor of that level.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size of the paired :class:`SparseConv3d`.
        key: Key of the paired :class:`SparseConv3d`.
        bias: Add a learned bias.
    """

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        key: str,
        bias: bool = False,
    ) -> None:  # pragma: no cover
        raise NotImplementedError


class BatchNorm(nn.BatchNorm1d):
    """Batch normalisation over the voxels of a sparse tensor."""

    def forward(self, tensor: Any) -> Any:
        return replace_features(tensor, super().forward(features(tensor)))


class ReLU(nn.ReLU):
    """ReLU on the features of a sparse tensor."""

    def forward(self, tensor: Any) -> Any:
        return replace_features(tensor, super().forward(features(tensor)))


class Linear(nn.Linear):
    """Per-voxel affine map: the facade's 1×1 convolution."""

    def forward(self, tensor: Any) -> Any:
        return replace_features(tensor, super().forward(features(tensor)))


# ------------------------------------------------------------------ foreign checkpoints


def convert_minkowski_kernel(kernel: Tensor, kernel_size: int) -> Tensor:
    """Re-lay a MinkowskiEngine kernel in the facade's weight layout.

    MinkowskiEngine stores a kernel as ``(volume, in_channels, out_channels)`` and
    enumerates the cells of a cubic kernel with the *first* spatial axis varying
    fastest: cell ``n`` is ``(i, j, k)`` with ``n = i + ks * j + ks² * k``. Its cells
    mean the same offsets as the facade's (centred for a stride-1 kernel, ``0 … ks-1``
    for a down-sampling kernel and for its transposed counterpart), so no flip is
    involved, only a change of memory layout. A kernel of volume one is stored as
    ``(in_channels, out_channels)`` and becomes the ``(out_channels, in_channels)``
    weight of :class:`Linear`.

    The equivalence is established against recorded MinkowskiEngine 0.5.4 activations
    (``tests/unit/test_sparse_facade.py``, ``tests/integration/test_gsnet_parity.py``).
    """
    if kernel_size == 1:
        if kernel.ndim == 3 and kernel.shape[0] == 1:
            kernel = kernel[0]
        if kernel.ndim != 2:
            raise ValueError(f"a 1x1 kernel must be (in, out), got {tuple(kernel.shape)}")
        return kernel.t().contiguous()
    volume = kernel_size**3
    if kernel.ndim != 3 or kernel.shape[0] != volume:
        raise ValueError(
            f"a {kernel_size}^3 MinkowskiEngine kernel must be ({volume}, in, out), "
            f"got {tuple(kernel.shape)}"
        )
    _, in_channels, out_channels = kernel.shape
    cells = kernel.reshape(kernel_size, kernel_size, kernel_size, in_channels, out_channels)
    return cells.permute(4, 2, 1, 0, 3).contiguous()  # (k, j, i, in, out) → (out, i, j, k, in)
