"""Sparse-convolution facade.

Networks are written against the handful of names exported here, never against a
sparse-convolution library. The facade fixes the *meaning* of every layer, so that a
checkpoint means the same thing whichever backend executes it:

* A sparse tensor is a set of occupied voxels ``[batch, x, y, z]`` with one feature row
  each. Voxel coordinates are signed integers; :func:`make_tensor` hides whatever
  shifting a backend needs.
* Kernel weights have the layout ``(out_channels, kx, ky, kz, in_channels)``; the three
  kernel axes follow the voxel axes. The offset each kernel cell stands for is part of
  the contract and is spelled out on :class:`SubMConv3d`, :class:`SparseConv3d` and
  :class:`SparseInverseConv3d`.
* A 1×1 convolution is :class:`Linear`: a per-voxel affine map with an ordinary
  ``(out_channels, in_channels)`` weight. (Backends disagree on how a 1×1 kernel is laid
  out in memory; spconv 2.3.8 reinterprets its ``(out, 1, 1, 1, in)`` parameter as
  ``(in, out)`` without transposing. A linear layer has no such ambiguity.)

spconv is the default backend and the only one implemented. It is imported on first
use, so ``import grasppanda`` works without it; using a layer without it raises an
error that names the extra to install. Another backend is a module that defines the
same names (see ``_spconv.py``) plus one entry in :data:`BACKENDS`.
"""

from __future__ import annotations

from .api import (
    BACKENDS,
    BatchNorm,
    Linear,
    ReLU,
    SparseBackendError,
    SparseConv3d,
    SparseInverseConv3d,
    SubMConv3d,
    cat,
    check_cpu_convolution,
    convert_minkowski_kernel,
    features,
    features_at,
    get_backend,
    indices,
    make_tensor,
    replace_features,
    set_backend,
)

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
    "features",
    "features_at",
    "get_backend",
    "indices",
    "make_tensor",
    "replace_features",
    "set_backend",
]
