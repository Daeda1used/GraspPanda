"""Per-object evaluation cache: everything scoring needs, and nothing else.

The reference evaluator re-reads, for every scene and every worker, each object's full
model (up to 2.5 million vertices) and its Dex-Net pickle (58 MB to 1.1 GB, mostly a
mesh the force-closure test never touches). Scoring only needs, per object:

* the 100³ signed-distance grid with its two frame transforms and surface band;
* the frozen surface sample used outside the grid (see :mod:`.dexnet_pickle`);
* the model vertices after voxel down-sampling.

These are extracted once into ``<cache root>/<dataset>/eval/<hash>/objects/NNN.npz``
(about 7 MB each, against 58 MB to 1.1 GB per pickle). The store follows the asset rules of the architecture document:

* the directory name is a hash of generator, version, parameters and a dataset
  fingerprint, so a changed parameter or dataset can never be served stale data;
* ``manifest.json`` records what the directory holds; every object has a sidecar
  ``NNN.json`` with the SHA-256 of its two source files;
* an object is complete exactly when its sidecar exists; data and sidecar are written
  to temporary names and renamed, so interrupted or concurrent builds are harmless
  and a build resumes where it stopped;
* objects are built lazily, one at a time; a missing object is an error when building
  is disabled, never an empty result;
* nothing is ever written into the dataset tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import grasppanda
from grasppanda.utils.paths import cache_root

from .dexnet_pickle import file_sha256, read_dexnet_sdf
from .models import read_ply_vertices, voxel_downsample
from .sdf import SdfGrid

__all__ = [
    "CACHE_VERSION",
    "GRASPNET1B_SOURCES",
    "CacheError",
    "ObjectAsset",
    "ObjectCache",
    "ObjectSources",
]

#: Version of the extraction code. Bump it whenever stored values could change.
CACHE_VERSION = 1


@dataclass(frozen=True)
class ObjectSources:
    """Where a dataset keeps the two source files of an object.

    Attributes:
        dataset: Short dataset name; also names the cache sub-tree and its generator.
        dex_model: Path of the Dex-Net pickle below the dataset root; ``{}`` takes the id.
        model: Path of the model PLY, same convention.
        shard: Name of an object's files inside the cache; ``{}`` takes the id.
    """

    dataset: str
    dex_model: str
    model: str
    shard: str = "{:03d}"

    @property
    def generator(self) -> str:
        """Identity of the extraction code in manifests and hashes."""
        return f"{self.dataset}-eval-cache"

    def paths(self, data_root: Path, object_id: int) -> dict[str, Path]:
        """The two source files of one object."""
        return {
            "dex_model": data_root / self.dex_model.format(object_id),
            "model": data_root / self.model.format(object_id),
        }

    def listing(self, data_root: Path) -> list[Path]:
        """Source files of all objects, sorted (the ``{}`` fields act as wildcards)."""
        found: list[Path] = []
        for pattern in (self.dex_model, self.model):
            wildcard = "*".join(part.split("}")[-1] for part in pattern.split("{"))
            found += sorted(data_root.glob(wildcard))
        return found


GRASPNET1B_SOURCES = ObjectSources(
    dataset="graspnet1b",
    dex_model="dex_models/{:03d}.pkl",
    model="models/{:03d}/nontextured.ply",
)


class CacheError(RuntimeError):
    """The evaluation cache is missing, incomplete or inconsistent with its sources."""


@dataclass(frozen=True, eq=False)
class ObjectAsset:
    """Scoring data of one object.

    Attributes:
        object_id: Dataset object index.
        sdf: Signed-distance grid in the object frame.
        model_points: ``(M, 3)`` float64 down-sampled model vertices, object frame.
    """

    object_id: int
    sdf: SdfGrid
    model_points: np.ndarray


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


class ObjectCache:
    """Lazily built, read-through store of :class:`ObjectAsset`.

    Args:
        data_root: Dataset root holding the Dex-Net pickles and the models.
        model_voxel_size: Voxel edge of the model down-sampling, metres.
        sources: Layout of the object files in the dataset.
        root: Cache root; defaults to :func:`grasppanda.utils.paths.cache_root`.
        build: Whether a missing object may be extracted from the dataset.
        directory: Open exactly this cache directory, read-only, without fingerprinting
            the dataset again. Workers use it with the directory their parent has just
            validated and filled; listing a dataset on a network file system once per
            worker would cost seconds.
        max_loaded: Objects kept in memory (least recently used are dropped); one
            object takes about 8 MB.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        model_voxel_size: float,
        sources: ObjectSources = GRASPNET1B_SOURCES,
        root: str | Path | None = None,
        build: bool = True,
        directory: str | Path | None = None,
        max_loaded: int = 16,
    ) -> None:
        self.data_root = Path(data_root)
        self.sources = sources
        self.model_voxel_size = float(model_voxel_size)
        self.build = build and directory is None
        self.max_loaded = max_loaded
        self._loaded: OrderedDict[int, ObjectAsset] = OrderedDict()

        identity: dict[str, Any] = {
            "generator": sources.generator,
            "version": CACHE_VERSION,
            "parameters": {"model_voxel_size": self.model_voxel_size},
        }
        if directory is not None:
            self.directory = Path(directory)
            self.hash = self.directory.name
        else:
            fingerprint = self._dataset_fingerprint()
            identity["dataset"] = {"name": sources.dataset, "fingerprint": fingerprint}
            self.hash = hashlib.sha256(_canonical(identity)).hexdigest()[:16]
            base = Path(root) if root is not None else cache_root()
            self.directory = base / sources.dataset / "eval" / self.hash
        resolved = self.directory.resolve()
        if resolved.is_relative_to(self.data_root.resolve()):
            raise CacheError(
                f"the cache directory {resolved} lies inside the dataset {self.data_root}; "
                "derived data is never written into a dataset tree. Set GRASPPANDA_CACHE."
            )
        self._identity = identity
        self._open_manifest()

    # ---------------------------------------------------------------- identity

    def _sources(self, object_id: int) -> dict[str, Path]:
        return self.sources.paths(self.data_root, object_id)

    def _dataset_fingerprint(self) -> str:
        """Hash of the names and sizes of all source files (their content is hashed per object).

        Listing and ``stat`` are cheap; reading the 21 GB of pickles just to name the
        cache directory would not be.
        """
        entries = []
        for path in self.sources.listing(self.data_root):
            relative = path.relative_to(self.data_root).as_posix()
            entries.append(f"{relative}:{path.stat().st_size}")
        if not entries:
            raise CacheError(
                f"no object sources below {self.data_root}: expected "
                f"{self.sources.dex_model.format(0)} and {self.sources.model.format(0)}. The "
                "dex_models pickles are part of the official dataset release and cannot be "
                "regenerated (they hold a random surface sample the benchmark scores depend on)."
            )
        return hashlib.sha256("\n".join(entries).encode()).hexdigest()

    def _open_manifest(self) -> None:
        path = self.directory / "manifest.json"
        if path.is_file():
            stored = json.loads(path.read_text())
            for key, expected in self._identity.items():
                if stored.get(key) != expected:
                    raise CacheError(
                        f"{path} describes a different cache ({key}: {stored.get(key)!r}, "
                        f"expected {expected!r}). Delete {self.directory} and rebuild."
                    )
            return
        if not self.build:
            raise CacheError(
                f"no evaluation cache at {self.directory} and building is disabled; "
                "open it once with build=True (the evaluator does this before starting workers)"
            )
        (self.directory / "objects").mkdir(parents=True, exist_ok=True)
        manifest = {
            **self._identity,
            "code_version": grasppanda.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "units": "metres",
            "frame": "object model frame",
            "uses_ground_truth": True,
            "shards": "objects/<id>.npz, complete when objects/<id>.json exists",
            "arrays": {
                "sdf": "signed distances, float64, negative inside",
                "grid_to_object": "[scale, tx, ty, tz] of object = scale * grid + t",
                "object_to_grid": "[scale, tx, ty, tz] of grid = scale * object + t",
                "surface_threshold": "|sdf| below this is on the surface",
                "surface_sample": "frozen random surface voxels of the official object",
                "model_points": "model vertices after voxel down-sampling",
            },
        }
        _write_atomic(path, json.dumps(manifest, indent=2, sort_keys=True).encode())

    # ----------------------------------------------------------------- reading

    def _paths(self, object_id: int) -> tuple[Path, Path]:
        stem = self.directory / "objects" / self.sources.shard.format(object_id)
        return stem.with_name(stem.name + ".npz"), stem.with_name(stem.name + ".json")

    def is_complete(self, object_id: int) -> bool:
        """Whether the object has been extracted."""
        data, sidecar = self._paths(object_id)
        return sidecar.is_file() and data.is_file()

    def load(self, object_id: int) -> ObjectAsset:
        """Return one object's scoring data, extracting it first if allowed and needed."""
        if object_id in self._loaded:
            self._loaded.move_to_end(object_id)
            return self._loaded[object_id]
        if not self.is_complete(object_id):
            if not self.build:
                raise CacheError(
                    f"object {object_id:03d} is missing from the evaluation cache "
                    f"{self.directory} and building is disabled"
                )
            self._extract(object_id)
        data_path, _ = self._paths(object_id)
        with np.load(data_path) as stored:
            grid_to_object = stored["grid_to_object"]
            object_to_grid = stored["object_to_grid"]
            asset = ObjectAsset(
                object_id=object_id,
                sdf=SdfGrid(
                    values=stored["sdf"],
                    grid_to_object_scale=float(grid_to_object[0]),
                    grid_to_object_translation=grid_to_object[1:],
                    object_to_grid_scale=float(object_to_grid[0]),
                    object_to_grid_translation=object_to_grid[1:],
                    surface_threshold=float(stored["surface_threshold"]),
                    surface_sample=stored["surface_sample"],
                ),
                model_points=stored["model_points"],
            )
        self._loaded[object_id] = asset
        while len(self._loaded) > self.max_loaded:
            self._loaded.popitem(last=False)
        return asset

    def ensure(self, object_ids: Iterable[int]) -> list[int]:
        """Extract the objects that are not cached yet; returns the ids that were built."""
        built = []
        for object_id in sorted(set(object_ids)):
            if not self.is_complete(object_id):
                if not self.build:
                    raise CacheError(
                        f"object {object_id:03d} is missing from the evaluation cache "
                        f"{self.directory} and building is disabled"
                    )
                self._extract(object_id)
                built.append(object_id)
        return built

    def verify(self, object_id: int) -> None:
        """Re-hash the object's source files and compare with what the cache was built from."""
        _, sidecar = self._paths(object_id)
        if not sidecar.is_file():
            raise CacheError(f"object {object_id:03d} is not in the cache {self.directory}")
        recorded = json.loads(sidecar.read_text())["sources"]
        for name, path in self._sources(object_id).items():
            sha256, size = file_sha256(path)
            if (sha256, size) != (recorded[name]["sha256"], recorded[name]["size"]):
                raise CacheError(
                    f"{path} changed since object {object_id:03d} was cached "
                    f"(sha256 {sha256[:12]}… vs {recorded[name]['sha256'][:12]}…). "
                    f"Delete {self.directory} and rebuild."
                )

    # ---------------------------------------------------------------- building

    def _extract(self, object_id: int) -> None:
        sources = self._sources(object_id)
        for name, path in sources.items():
            if not path.is_file():
                raise CacheError(
                    f"cannot build the evaluation cache for object {object_id:03d}: "
                    f"{name} source {path} does not exist"
                )
        started = time.perf_counter()
        dexnet = read_dexnet_sdf(sources["dex_model"])
        model_sha256, model_size = file_sha256(sources["model"])
        vertices = read_ply_vertices(sources["model"])
        model_points = voxel_downsample(vertices, self.model_voxel_size)

        data_path, sidecar_path = self._paths(object_id)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                sdf=dexnet.sdf,
                grid_to_object=np.concatenate(
                    [[dexnet.grid_to_object[0]], dexnet.grid_to_object[1]]
                ),
                object_to_grid=np.concatenate(
                    [[dexnet.object_to_grid[0]], dexnet.object_to_grid[1]]
                ),
                surface_threshold=np.float64(dexnet.surface_threshold),
                surface_sample=dexnet.surface_sample,
                model_points=model_points,
            )
        os.replace(temporary, data_path)
        sidecar = {
            "object_id": object_id,
            "generator": self.sources.generator,
            "version": CACHE_VERSION,
            "code_version": grasppanda.__version__,
            "numpy_version": np.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.perf_counter() - started, 3),
            "sources": {
                "dex_model": {
                    "path": sources["dex_model"].relative_to(self.data_root).as_posix(),
                    "size": dexnet.size,
                    "sha256": dexnet.sha256,
                },
                "model": {
                    "path": sources["model"].relative_to(self.data_root).as_posix(),
                    "size": model_size,
                    "sha256": model_sha256,
                },
            },
            "sdf_shape": list(dexnet.sdf.shape),
            "resolution": dexnet.resolution,
            "surface_sample_size": len(dexnet.surface_sample),
            "model_vertices": len(vertices),
            "model_points": len(model_points),
        }
        _write_atomic(sidecar_path, json.dumps(sidecar, indent=2, sort_keys=True).encode())
