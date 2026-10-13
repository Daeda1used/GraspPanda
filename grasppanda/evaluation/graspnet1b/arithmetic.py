"""Small vector products with a pinned order of floating-point operations.

Why not ``np.dot`` / ``@``: the evaluator makes thousands of threshold decisions per
frame (is a point between the fingers, is a contact inside the friction cone), and a
BLAS library is free to evaluate ``a·b`` in any order, with or without fused
multiply-add. Two NumPy builds therefore disagree in the last bit, and occasionally a
last bit decides a threshold. Measured on the development machine: the BLAS of
NumPy 2.5 fuses, the BLAS of the reference environment (NumPy 1.24) does not, and
about a third of all 3-vector products differ in the last bit.

Every product here is spelled out as element-wise IEEE operations,
``((a0*b0 + a1*b1) + a2*b2)``, which is what the reference environment computes and is
the same on every platform and NumPy version.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cross3", "dot3", "matmul3", "norm3", "rotate_rows", "transform_points"]


def dot3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inner product over the last axis (length 3), summed left to right."""
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]


def norm3(a: np.ndarray) -> np.ndarray:
    """Euclidean norm over the last axis (length 3)."""
    return np.sqrt(dot3(a, a))


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product over the last axis."""
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def matmul3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` for ``(..., 3, 3)`` operands; each entry is summed over k = 0, 1, 2 in order."""
    left = a[..., :, :, None]  # (..., i, k, 1)
    right = b[..., None, :, :]  # (..., 1, k, j)
    return (left[..., 0, :] * right[..., 0, :] + left[..., 1, :] * right[..., 1, :]) + (
        left[..., 2, :] * right[..., 2, :]
    )


def rotate_rows(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """``points @ rotation`` for ``(..., N, 3)`` row vectors and ``(..., 3, 3)`` matrices.

    With an orthonormal ``rotation`` this expresses the points in the frame whose axes
    are the matrix columns.
    """
    x, y, z = points[..., 0:1], points[..., 1:2], points[..., 2:3]
    rows = rotation[..., None, :, :]  # (..., 1, k, j)
    return (x * rows[..., 0, :] + y * rows[..., 1, :]) + z * rows[..., 2, :]


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a ``(4, 4)`` homogeneous transform to ``(N, 3)`` points; returns float64.

    The matrix keeps its own precision until each product is formed, so a float32 pose
    contributes its float32 values, widened, exactly as mixed-precision ``matmul`` does.
    """
    matrix = np.asarray(matrix)
    points = np.asarray(points, dtype=np.float64)
    m = matrix[:3].astype(np.float64)
    x, y, z = points[:, 0:1], points[:, 1:2], points[:, 2:3]
    return ((x * m[:, 0] + y * m[:, 1]) + z * m[:, 2]) + m[:, 3]
