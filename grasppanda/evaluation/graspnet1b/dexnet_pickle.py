"""Read the signed-distance data out of ``dex_models/NNN.pkl`` without running upstream code.

Why the pickles are needed at all: when a query point leaves the 100³ grid, the
reference force-closure code measures the distance to the nearest of 1000 surface
voxels. Those 1000 voxels were drawn with an unseeded ``np.random.choice`` when each
object was first built and survive only inside the official pickles. Rebuilding an
object from ``textured.sdf`` draws a different sample and therefore different scores.

GraspClutter6D ships pickles of the same shape (its toolkit vendors the same classes
under ``graspclutter6dAPI``); they are read by the same code.

Why not ``pickle.load``: the files pickle instances of ``graspnetAPI``, ``autolab_core``
and ``trimesh`` classes. Loading them normally imports those packages (and Dex-Net code
under a research-only licence) and executes whatever their ``__reduce__`` hooks ask
for. :class:`InertUnpickler` instead resolves every pickled class to an attribute bag
that does nothing, rebuilds NumPy arrays itself, and refuses any other global. No
upstream module is imported and no upstream function is called.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

try:  # NumPy >= 2.0
    from numpy._core import multiarray as _multiarray
except ImportError:  # NumPy 1.26
    from numpy.core import multiarray as _multiarray

__all__ = ["DexNetSdf", "InertUnpickler", "file_sha256", "read_dexnet_sdf"]

#: Benchmark toolkits that ship the Dex-Net classes under their own package name.
_TOOLKITS = ("graspnetAPI", "graspclutter6dAPI")
#: Packages whose pickled classes become inert attribute bags.
_STUB_PACKAGES = (*_TOOLKITS, "autolab_core", "trimesh")
_NUMPY_MODULES = ("numpy.core.multiarray", "numpy._core.multiarray")
_OBJECT_CLASS = "utils.dexnet.grasping.graspable_object.GraspableObject3D"
_SDF_CLASS = "utils.dexnet.grasping.meshpy.sdf.Sdf3D"
_SIMILARITY_CLASS = "autolab_core.rigid_transformations.SimilarityTransform"


class _Inert:
    """Stands in for a pickled upstream class: keeps the state, runs nothing."""

    _origin = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = {**(state[0] or {}), **state[1]}  # (dict state, slot state)
        if not isinstance(state, dict):
            state = {"_state": state}
        self.__dict__.update(state)


def _inert_call(*args: Any, **kwargs: Any) -> None:
    """Replaces pickled function calls such as ``getattr(obj, '__hash__')``."""
    return None


def _plain_array(cls: Any, shape: tuple[int, ...], typecode: bytes) -> np.ndarray:
    """``numpy``'s array reconstructor, minus the pickled subclass (e.g. a trimesh array)."""
    return np.ndarray.__new__(np.ndarray, shape, np.dtype(typecode))


class InertUnpickler(pickle.Unpickler):
    """An unpickler for the official ``dex_models`` files that executes no foreign code."""

    def __init__(self, stream: BinaryIO) -> None:
        super().__init__(stream)
        self._stubs: dict[tuple[str, str], type[_Inert]] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module in _NUMPY_MODULES and name == "_reconstruct":
            return _plain_array
        if module in _NUMPY_MODULES and name == "scalar":
            return _multiarray.scalar
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) == ("builtins", "getattr"):
            return _inert_call
        if module.split(".")[0] in _STUB_PACKAGES:
            key = (module, name)
            if key not in self._stubs:
                self._stubs[key] = type(name, (_Inert,), {"_origin": f"{module}.{name}"})
            return self._stubs[key]
        raise pickle.UnpicklingError(
            f"refusing to load global '{module}.{name}': it is not part of an official "
            "dex_models pickle (expected NumPy arrays and classes of "
            f"{', '.join(_STUB_PACKAGES)}). The file may be corrupt or not what it claims to be."
        )


def file_sha256(path: str | Path) -> tuple[str, int]:
    """SHA-256 and size of a file, read in chunks."""
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 23):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True)
class DexNetSdf:
    """What scoring needs from one official Dex-Net object.

    Attributes:
        sdf: ``(nx, ny, nz)`` float64 signed distances in metres, negative inside.
        origin: ``(3,)`` object-frame position of grid index ``(0, 0, 0)``.
        resolution: Voxel edge in metres.
        surface_threshold: ``|sdf|`` below this counts as "on the surface".
        grid_to_object: ``(scale, translation)`` of ``object = scale * grid + translation``.
        object_to_grid: ``(scale, translation)`` of the stored inverse transform.
        surface_sample: ``(S, 3)`` int64 grid indices: the frozen random surface sample.
        sha256: Fingerprint of the pickle the data came from.
        size: Size of that file in bytes.
    """

    sdf: np.ndarray
    origin: np.ndarray
    resolution: float
    surface_threshold: float
    grid_to_object: tuple[float, np.ndarray]
    object_to_grid: tuple[float, np.ndarray]
    surface_sample: np.ndarray
    sha256: str
    size: int


def _field(holder: Any, name: str, path: Path) -> Any:
    try:
        return vars(holder)[name]
    except (KeyError, TypeError):
        origin = getattr(holder, "_origin", type(holder).__name__)
        raise ValueError(f"{path}: pickled {origin} has no attribute '{name}'") from None


def _expect_class(holder: Any, origin: str, what: str, path: Path) -> None:
    actual = getattr(holder, "_origin", type(holder).__name__)
    if actual != origin:
        raise ValueError(f"{path}: {what} should be a pickled {origin}, found {actual}")


def _expect_toolkit_class(holder: Any, origin: str, what: str, path: Path) -> None:
    """Like :func:`_expect_class` for a class every toolkit vendors under its own package."""
    actual = getattr(holder, "_origin", type(holder).__name__)
    if actual not in [f"{toolkit}.{origin}" for toolkit in _TOOLKITS]:
        raise ValueError(
            f"{path}: {what} should be a pickled <toolkit>.{origin} "
            f"(toolkit: {' or '.join(_TOOLKITS)}), found {actual}"
        )


def _similarity(holder: Any, what: str, path: Path) -> tuple[float, np.ndarray]:
    """Scale and translation of a pickled similarity transform; rotation must be identity.

    All 88 official objects satisfy this. A rotated grid would need the reference
    library's matrix product, whose rounding this evaluator does not model.
    """
    _expect_class(holder, _SIMILARITY_CLASS, what, path)
    rotation = np.asarray(_field(holder, "_rotation", path))
    if rotation.shape != (3, 3) or not np.array_equal(rotation, np.eye(3)):
        raise ValueError(
            f"{path}: {what} has a non-identity rotation; this evaluator only supports the "
            "axis-aligned signed-distance grids of the official release"
        )
    translation = np.asarray(_field(holder, "_translation", path), dtype=np.float64).reshape(3)
    return float(_field(holder, "_scale", path)), translation


def read_dexnet_sdf(path: str | Path) -> DexNetSdf:
    """Extract the signed-distance grid and its frozen surface sample from one pickle."""
    path = Path(path)
    sha256, size = file_sha256(path)
    with path.open("rb") as stream:
        graspable = InertUnpickler(stream).load()
    _expect_toolkit_class(graspable, _OBJECT_CLASS, "the root object", path)
    field = _field(graspable, "sdf_", path)
    _expect_toolkit_class(field, _SDF_CLASS, "its signed-distance field", path)

    sdf = np.ascontiguousarray(_field(field, "data_", path), dtype=np.float64)
    sample = np.ascontiguousarray(_field(field, "surface_for_signed_val", path), dtype=np.int64)
    if sdf.ndim != 3:
        raise ValueError(f"{path}: the signed-distance grid has shape {sdf.shape}, expected 3-D")
    if bool(_field(field, "use_abs_", path)):
        raise ValueError(f"{path}: unsigned distance fields (use_abs) are not supported")
    if not np.isfinite(sdf).all():
        raise ValueError(f"{path}: the signed-distance grid contains NaN or Inf")
    if sample.ndim != 2 or sample.shape[1] != 3 or len(sample) == 0:
        raise ValueError(f"{path}: the surface sample has shape {sample.shape}, expected (S, 3)")
    if (sample < 0).any() or (sample >= np.array(sdf.shape)).any():
        raise ValueError(f"{path}: the surface sample indexes outside the {sdf.shape} grid")

    return DexNetSdf(
        sdf=sdf,
        origin=np.asarray(_field(field, "origin_", path), dtype=np.float64).reshape(3),
        resolution=float(_field(field, "resolution_", path)),
        surface_threshold=float(_field(field, "surface_thresh_", path)),
        grid_to_object=_similarity(_field(field, "T_grid_world_", path), "grid-to-object", path),
        object_to_grid=_similarity(_field(field, "T_world_grid_", path), "object-to-grid", path),
        surface_sample=sample,
        sha256=sha256,
        size=size,
    )
