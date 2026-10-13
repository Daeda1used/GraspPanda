"""Derived-asset store: labels computed from a dataset, kept apart from it.

An asset is a directory ``<cache>/<dataset>/<kind>/<hash>/`` with a ``manifest.json``.
The hash covers everything that decides the content: the generator and its version, its
parameters, the dataset inputs, and the preprocessing chain the content is aligned to.
A dataset configured differently therefore looks for a different directory, and the
error names what differs instead of serving misaligned labels.

Guarantees:

* nothing is written into a dataset tree;
* every file appears atomically (write to a temporary name, then rename);
* generation is resumable shard by shard, and completion is recorded per shard;
* a missing shard is an error, never an empty or all-zero label.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grasppanda.transforms.base import config_fingerprint
from grasppanda.utils.paths import cache_root

__all__ = ["Asset", "AssetError", "AssetSpec", "atomic_path", "save_array", "save_arrays"]

FORMAT = 1
MANIFEST = "manifest.json"


class AssetError(RuntimeError):
    """A derived asset is missing, incomplete or does not match the configuration."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextlib.contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling of ``path`` that replaces it when the block succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def save_array(path: Path, array: np.ndarray) -> None:
    """Atomically write one array as an uncompressed ``.npy`` (memory-mappable)."""
    with atomic_path(path) as temporary, open(temporary, "wb") as handle:
        np.save(handle, array, allow_pickle=False)


def save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write named arrays as a compressed ``.npz``."""
    with atomic_path(path) as temporary, open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)


@dataclass(frozen=True)
class AssetSpec:
    """Everything that determines the content of an asset.

    Attributes:
        dataset: Dataset name, e.g. ``graspnet1b``.
        kind: What the asset holds, e.g. ``graspness``.
        generator: Name of the code that produces it.
        version: Version of that code; bumped when its output changes.
        parameters: Generator parameters.
        inputs: What identifies the dataset content the asset was computed from.
        preprocessing: Configs of the transforms before the label-alignment boundary,
            for assets aligned to the preprocessed frame; ``None`` otherwise.
        uses_ground_truth: Whether ground truth beyond the labels themselves shaped the
            asset (for example a workspace crop from ground-truth segmentation).
        provenance: ``native`` for assets generated here, ``legacy-import`` for content
            imported from files whose generator is unknown.
        about: Free-form description of units and frames. Not part of the hash.
    """

    dataset: str
    kind: str
    generator: str
    version: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: Sequence[Mapping[str, Any]] | None = None
    uses_ground_truth: bool = False
    provenance: str = "native"
    about: Mapping[str, str] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        """The hashed content as plain data."""
        return json.loads(
            _canonical(
                {
                    "dataset": self.dataset,
                    "kind": self.kind,
                    "generator": {"name": self.generator, "version": self.version},
                    "parameters": dict(self.parameters),
                    "inputs": dict(self.inputs),
                    "preprocessing": (
                        None if self.preprocessing is None else list(self.preprocessing)
                    ),
                    "uses_ground_truth": self.uses_ground_truth,
                    "provenance": self.provenance,
                }
            )
        )

    @property
    def hash(self) -> str:
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()[:16]

    @property
    def preprocessing_fingerprint(self) -> str | None:
        """Fingerprint of the preprocessing chain the content is aligned to, if any."""
        if self.preprocessing is None:
            return None
        return config_fingerprint(self.preprocessing)


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Human-readable list of where two plain trees differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right)):
            where = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                found.append(
                    f"{where}: {left.get(key, '<absent>')!r} vs {right.get(key, '<absent>')!r}"
                )
            else:
                found += _differences(left[key], right[key], where)
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        found = []
        for index, (a, b) in enumerate(zip(left, right)):
            found += _differences(a, b, f"{path}[{index}]")
        return found
    return [] if left == right else [f"{path}: {left!r} vs {right!r}"]


class Asset:
    """One asset directory: its manifest, its shards and atomic writes into it."""

    def __init__(self, spec: AssetSpec, root: str | os.PathLike[str] | None = None) -> None:
        self.spec = spec
        self.root = Path(root) if root is not None else cache_root()
        self.path = self.root / spec.dataset / spec.kind / spec.hash
        self._manifest: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Asset({self.spec.dataset}/{self.spec.kind}/{self.spec.hash})"

    # ---------------------------------------------------------------------- manifest

    def exists(self) -> bool:
        return (self.path / MANIFEST).is_file()

    def manifest(self, *, refresh: bool = False) -> dict[str, Any]:
        """Read and verify the manifest; raises :class:`AssetError` if there is none."""
        if self._manifest is not None and not refresh:
            return self._manifest
        file = self.path / MANIFEST
        if not file.is_file():
            raise AssetError(self._missing_message())
        manifest = json.loads(file.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise AssetError(
                f"{file} has asset format {manifest.get('format')!r}; this version reads format "
                f"{FORMAT}. Rebuild the asset."
            )
        stored = {key: manifest.get(key) for key in self.spec.identity()}
        differences = _differences(stored, self.spec.identity())
        if differences:
            raise AssetError(
                f"{file} does not describe the asset its directory name promises:\n  "
                + "\n  ".join(differences)
                + "\nThe manifest was edited or the directory was renamed; rebuild the asset."
            )
        self._manifest = manifest
        return manifest

    def _missing_message(self) -> str:
        spec = self.spec
        lines = [
            f"No {spec.kind} asset of {spec.dataset} for this configuration: {self.path} "
            "does not exist."
        ]
        siblings = sorted(p for p in self.path.parent.glob(f"*/{MANIFEST}") if p.is_file())
        for sibling in siblings[:3]:
            try:
                other = json.loads(sibling.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            stored = {key: other.get(key) for key in spec.identity()}
            differences = _differences(stored, spec.identity())
            lines.append(
                f"An asset built for another configuration exists ({sibling.parent.name}); "
                "it (left) differs from the requested one (right) in:\n    "
                + "\n    ".join(differences[:8])
            )
        return "\n".join(lines)

    def create(self) -> dict[str, Any]:
        """Create the directory and manifest if needed; return the manifest."""
        if self.exists():
            return self.manifest(refresh=True)
        from grasppanda import __version__

        manifest = {
            "format": FORMAT,
            **self.spec.identity(),
            "preprocessing_fingerprint": self.spec.preprocessing_fingerprint,
            "about": dict(self.spec.about),
            "hash": self.spec.hash,
            "created": _now(),
            "grasppanda": __version__,
            "shards": {},
        }
        self._write(manifest)
        return manifest

    def _write(self, manifest: dict[str, Any]) -> None:
        with atomic_path(self.path / MANIFEST) as temporary:
            temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        self._manifest = manifest

    # ------------------------------------------------------------------------ shards

    def shards(self, *, refresh: bool = False) -> dict[str, dict[str, Any]]:
        """Records of the shards that are complete."""
        return self.manifest(refresh=refresh)["shards"]

    def is_complete(self, shard: str) -> bool:
        return shard in self.shards()

    def mark_complete(self, shard: str, **record: Any) -> None:
        """Record that a shard is fully written. Re-reads the manifest before updating."""
        manifest = self.manifest(refresh=True)
        manifest["shards"][shard] = {**record, "finished": _now()}
        self._write(manifest)

    def missing(self, shards: Iterable[str]) -> list[str]:
        complete = self.shards()
        return [shard for shard in shards if shard not in complete]

    def require(self, shards: Iterable[str], *, build: str) -> None:
        """Raise :class:`AssetError` unless every listed shard is complete.

        Args:
            shards: Shards the caller is going to read.
            build: Command that generates the asset, shown in the error.
        """
        try:
            wanted = list(shards)
            missing = self.missing(wanted)
        except AssetError as error:
            raise AssetError(f"{error}\nBuild it with: {build}") from None
        if missing:
            shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
            raise AssetError(
                f"The {self.spec.kind} asset at {self.path} lacks {len(missing)} of the "
                f"{len(wanted)} shards this dataset reads ({shown}).\nBuild them with: {build}"
            )

    def record(self) -> dict[str, Any]:
        """Summary for run snapshots and result records."""
        return {
            "kind": self.spec.kind,
            "hash": self.spec.hash,
            "provenance": self.spec.provenance,
            "uses_ground_truth": self.spec.uses_ground_truth,
            "generator": f"{self.spec.generator}@{self.spec.version}",
            "path": str(self.path),
        }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
