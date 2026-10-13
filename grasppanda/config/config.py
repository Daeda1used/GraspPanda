"""The :class:`Config` object: a validated, fully traceable config tree."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError, Issue
from .loader import (
    Dropped,
    KeyPath,
    LoadedTree,
    apply_overrides,
    load_tree,
    resolve_interpolations,
)
from .validate import fill_defaults, validate_config

__all__ = ["Config", "available_configs", "configs_root", "resolve_config_path"]

#: Keys that say where a run lives or how it is continued, not what it is; excluded from
#: the hash.
_VOLATILE: tuple[KeyPath, ...] = (
    ("runtime", "work_dir"),
    ("runtime", "run_id"),
    ("runtime", "resume"),
    ("runtime", "allow_config_mismatch"),
)


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def configs_root() -> Path:
    """Directory of the shipped configs: packaged in a wheel, repo root in a checkout."""
    package = Path(__file__).resolve().parents[1]
    packaged = package / "_configs"
    return packaged if packaged.is_dir() else package.parent / "configs"


def available_configs() -> list[str]:
    """Names of shipped runnable configs, e.g. ``gsnet/gsnet_sunet14_g1b-realsense_10e``."""
    root = configs_root()
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).with_suffix("").as_posix()
        for path in root.rglob("*.yaml")
        if "_base_" not in path.parts and path.name != "metafile.yaml"
    )


def resolve_config_path(name: str | Path) -> Path:
    """Resolve a file path or the name of a shipped config."""
    path = Path(name)
    if path.is_file():
        return path
    root = configs_root()
    for candidate in (root / path, root / f"{name}.yaml"):
        if candidate.is_file():
            return candidate
    names = available_configs()
    close = difflib.get_close_matches(str(name), names, n=3, cutoff=0.5)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    raise ConfigError(
        [Issue("", f"'{name}' is neither a file nor a shipped config name.{hint}", str(name))]
    )


class Config(Mapping[str, Any]):
    """A merged config tree with provenance.

    Use :meth:`load` for files and :meth:`from_dict` for in-memory configs. Both
    validate by default and raise :class:`ConfigError` listing every problem.
    """

    def __init__(
        self, tree: LoadedTree, *, path: Path | None = None, overrides: Sequence[str] = ()
    ) -> None:
        self._tree = tree
        self.path = path
        self.overrides = tuple(overrides)

    # ----------------------------------------------------------------- constructors

    @classmethod
    def load(
        cls,
        path: str | Path,
        overrides: Sequence[str] = (),
        *,
        validate: bool = True,
        command: str | None = None,
    ) -> Config:
        """Load a config file (or shipped config name), apply overrides and validate."""
        resolved = resolve_config_path(path)
        tree = load_tree(resolved)
        apply_overrides(tree, overrides)
        config = cls(tree, path=resolved, overrides=overrides)
        config._interpolate()
        if validate:
            config.validate(command)
        return config

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, validate: bool = True, command: str | None = None
    ) -> Config:
        """Build a config from a dictionary (Python API, tests)."""
        config = cls(LoadedTree(data=copy.deepcopy(dict(data))))
        config._interpolate()
        if validate:
            config.validate(command)
        return config

    def _builtins(self) -> dict[str, Any]:
        return {"config_name": self.path.stem if self.path else "config"}

    def _interpolate(self) -> None:
        issues = resolve_interpolations(self._tree.data, self._builtins())
        if issues:
            raise ConfigError(issues, self._title())

    def _title(self) -> str | None:
        return str(self.path) if self.path else None

    # ------------------------------------------------------------------- validation

    def validate(self, command: str | None = None) -> None:
        """Run all three validation layers; raise :class:`ConfigError` on any problem."""
        issues = validate_config(self._tree.data, self._tree.origins, command=command)
        if issues:
            raise ConfigError(issues, self._title())

    # ---------------------------------------------------------------------- mapping

    def __getitem__(self, key: str) -> Any:
        return self._tree.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tree.data)

    def __len__(self) -> int:
        return len(self._tree.data)

    def __repr__(self) -> str:
        return f"Config({self.path or 'dict'}, keys={list(self._tree.data)})"

    def select(self, dotted_path: str, default: Any = None) -> Any:
        """Read ``a.b.0.c`` from the resolved tree (defaults included)."""
        node: Any = self.resolved()
        for token in dotted_path.split("."):
            if isinstance(node, Mapping) and token in node:
                node = node[token]
            elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
                node = node[int(token)]
            else:
                return default
        return node

    # ------------------------------------------------------------------- provenance

    @property
    def sources(self) -> list[Path]:
        """Every file read, leaf first."""
        return list(self._tree.sources)

    @property
    def dropped(self) -> list[Dropped]:
        """Keys discarded because a node changed its ``type``."""
        return list(self._tree.dropped)

    def source_of(self, dotted_path: str) -> str | None:
        """File (or ``<command line>``) that set a value."""
        path = tuple(int(t) if t.isdigit() else t for t in dotted_path.split("."))
        return self._tree.source_of(path)

    # ----------------------------------------------------------------------- output

    def to_dict(self) -> dict[str, Any]:
        """The merged tree as written, without defaults."""
        return copy.deepcopy(self._tree.data)

    def resolved(self) -> dict[str, Any]:
        """The merged tree with schema and component defaults filled in."""
        data = fill_defaults(self._tree.data)
        resolve_interpolations(data, self._builtins())
        return data

    def dump(self, *, defaults: bool = False) -> str:
        """Serialise to YAML; with ``defaults`` every implicit value is written out."""
        data = self.resolved() if defaults else self.to_dict()
        return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)

    def digest(self) -> str:
        """Hash of what the experiment *is*: resolved config minus run-location keys."""
        data = self.resolved()
        for path in _VOLATILE:
            node: Any = data
            for key in path[:-1]:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                node.pop(path[-1], None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()
