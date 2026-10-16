"""Fetching model weights with mandatory checksum verification."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from grasppanda.utils.paths import cache_root

__all__ = ["ChecksumError", "fetch", "sha256_of"]


class ChecksumError(RuntimeError):
    """A downloaded or cached file does not have the expected sha256."""


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str | None, sha256: str, *, cache_dir: str | Path | None = None) -> Path:
    """Return a local, verified copy of ``url``.

    ``url`` may be ``http(s)://``, ``file://`` or a plain path. The file is stored under
    its digest, so a cached copy is reused only when its content still matches; a
    mismatch raises :class:`ChecksumError` and never returns a path.
    """
    if not url:  # a zoo entry of weights trained here and not published yet has no url
        raise ValueError(
            f"there is no url to fetch the weights with sha256 {sha256} from: they are not "
            "published"
        )
    sha256 = sha256.lower()
    directory = Path(cache_dir) if cache_dir else cache_root() / "weights"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{sha256[:16]}-{Path(url.split('?')[0]).name}"
    if target.is_file() and sha256_of(target) == sha256:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    source = url[7:] if url.startswith("file://") else url
    if "://" in source:
        with urllib.request.urlopen(source) as response, open(temporary, "wb") as handle:
            shutil.copyfileobj(response, handle)
    else:
        shutil.copyfile(source, temporary)
    actual = sha256_of(temporary)
    if actual != sha256:
        temporary.unlink()
        raise ChecksumError(
            f"sha256 mismatch for {url}: expected {sha256}, got {actual}. "
            "The file changed or the download was corrupted; it was not kept."
        )
    temporary.replace(target)
    return target
