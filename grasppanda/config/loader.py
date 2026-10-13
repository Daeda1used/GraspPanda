"""YAML loading: inheritance, type-aware merge, command-line overrides, interpolation.

Rules implemented here (``docs/design/ARCHITECTURE.md`` §5.2, §5.4):

1. ``_base_`` chains are at most two hops deep.
2. Sibling bases may not define the same value.
3. Dict onto dict merges deeply, unless the child changes ``type``: then the parent
   node is replaced, so kwargs of the old component never leak into the new one.
4. ``_delete_: true`` replaces an untyped dict explicitly.
5. Lists replace wholesale.
6. ``${a.b.c}`` is resolved after all merges and overrides.
"""

from __future__ import annotations

import copy
import difflib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError, Issue

__all__ = [
    "BASE_KEY",
    "CLI_SOURCE",
    "DELETE_KEY",
    "MAX_BASE_HOPS",
    "TYPE_KEY",
    "Dropped",
    "LoadedTree",
    "apply_overrides",
    "dotted",
    "load_tree",
    "parse_yaml",
    "resolve_interpolations",
]

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
TYPE_KEY = "type"
MAX_BASE_HOPS = 2
CLI_SOURCE = "<command line>"

KeyPath = tuple[Any, ...]


# --------------------------------------------------------------------------- YAML


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 floats and duplicate-key detection."""


# PyYAML follows YAML 1.1, where `1e-3` is a string. Learning rates are written that
# way all the time, so accept YAML 1.2 floats.
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def _construct_mapping(loader: _Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_yaml(text: str) -> Any:
    """Parse YAML text with the same rules as config files."""
    return yaml.load(text, Loader=_Loader)


def _read(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError([Issue("", f"cannot read config file: {error}", str(path))]) from None
    try:
        data = parse_yaml(text)
    except yaml.YAMLError as error:
        raise ConfigError([Issue("", f"invalid YAML: {error}", str(path))]) from None
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ConfigError([Issue("", "the top level of a config must be a mapping", str(path))])
    return data


# ------------------------------------------------------------------- tree + origins


@dataclass(frozen=True)
class Dropped:
    """Keys discarded because a node changed its ``type``."""

    path: str
    old_type: str
    new_type: str
    keys: tuple[str, ...]
    source: str


@dataclass
class LoadedTree:
    """A merged config tree and where each of its values came from."""

    data: dict[str, Any]
    origins: dict[KeyPath, str] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)

    def source_of(self, path: KeyPath) -> str | None:
        """File that set ``path``, or the first value below it."""
        if path in self.origins:
            return self.origins[path]
        for key, source in self.origins.items():
            if key[: len(path)] == path:
                return source
        return None


def dotted(path: KeyPath) -> str:
    return ".".join(str(part) for part in path)


def _leaves(value: Any, path: KeyPath) -> Iterator[KeyPath]:
    if isinstance(value, dict) and value:
        for key, sub in value.items():
            yield from _leaves(sub, (*path, key))
    elif isinstance(value, list) and value:
        for index, sub in enumerate(value):
            yield from _leaves(sub, (*path, index))
    else:
        yield path


def _record(origins: dict[KeyPath, str], path: KeyPath, value: Any, source: str) -> None:
    for key in [key for key in origins if key[: len(path)] == path]:
        del origins[key]
    for leaf in _leaves(value, path):
        origins[leaf] = source


def _strip_delete(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_delete(sub) for key, sub in value.items() if key != DELETE_KEY}
    if isinstance(value, list):
        return [_strip_delete(sub) for sub in value]
    return copy.deepcopy(value)


def _merge(
    base: dict[str, Any],
    over: dict[str, Any],
    path: KeyPath,
    source: str,
    origins: dict[KeyPath, str],
    dropped: list[Dropped],
) -> dict[str, Any]:
    """Merge ``over`` onto ``base`` with the type-aware rules."""
    if over.get(DELETE_KEY, False):
        replacement = _strip_delete(over)
        _record(origins, path, replacement, source)
        return replacement
    if TYPE_KEY in base and TYPE_KEY in over and base[TYPE_KEY] != over[TYPE_KEY]:
        stale = tuple(sorted(key for key in base if key != TYPE_KEY and key not in over))
        if stale:
            dropped.append(
                Dropped(dotted(path), str(base[TYPE_KEY]), str(over[TYPE_KEY]), stale, source)
            )
        replacement = _strip_delete(over)
        _record(origins, path, replacement, source)
        return replacement
    out = dict(base)
    for key, value in over.items():
        if key == DELETE_KEY:
            continue
        sub = (*path, key)
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            out[key] = _merge(base[key], value, sub, source, origins, dropped)
        else:
            out[key] = _strip_delete(value)
            _record(origins, sub, out[key], source)
    return out


def _union(
    left: dict[str, Any],
    right: dict[str, Any],
    path: KeyPath,
    left_origins: dict[KeyPath, str],
    right_origins: dict[KeyPath, str],
    issues: list[Issue],
) -> dict[str, Any]:
    """Combine two sibling bases; the same value defined twice is an error."""
    out = dict(left)
    for key, value in right.items():
        sub = (*path, key)
        if key not in left:
            out[key] = value
            for origin_path, source in right_origins.items():
                if origin_path[: len(sub)] == sub:
                    left_origins[origin_path] = source
        elif isinstance(left[key], dict) and isinstance(value, dict):
            out[key] = _union(left[key], value, sub, left_origins, right_origins, issues)
        else:
            first = next((s for p, s in left_origins.items() if p[: len(sub)] == sub), "?")
            second = next((s for p, s in right_origins.items() if p[: len(sub)] == sub), "?")
            issues.append(
                Issue(
                    dotted(sub),
                    "defined by two files of the same `_base_` list:\n"
                    f"  {first}\n  {second}\n"
                    "A value may come from only one sibling base; override it in the child instead.",
                )
            )
    return out


def _load_file(
    path: Path, hops: int, stack: tuple[Path, ...], tree: LoadedTree
) -> tuple[dict[str, Any], dict[KeyPath, str]]:
    if path in stack:
        chain = " -> ".join(p.name for p in (*stack, path))
        raise ConfigError([Issue(BASE_KEY, f"circular inheritance: {chain}", str(path))])
    raw = _read(path)
    tree.sources.append(path)
    bases = raw.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) for base in bases):
        raise ConfigError([Issue(BASE_KEY, "must be a path or a list of paths", str(path))])
    if bases and hops >= MAX_BASE_HOPS:
        chain = " -> ".join(p.name for p in (*stack, path))
        raise ConfigError(
            [
                Issue(
                    BASE_KEY,
                    f"inheritance is limited to {MAX_BASE_HOPS} levels, but this file is already "
                    f"{hops} levels below the config being loaded ({chain}) and has bases of its "
                    "own. Flatten the chain.",
                    str(path),
                )
            ]
        )
    merged: dict[str, Any] = {}
    origins: dict[KeyPath, str] = {}
    for relative in bases:
        base_data, base_origins = _load_file(
            (path.parent / relative).resolve(), hops + 1, (*stack, path), tree
        )
        issues: list[Issue] = []
        merged = _union(merged, base_data, (), origins, base_origins, issues)
        if issues:
            raise ConfigError(issues, str(path))
    return _merge(merged, raw, (), str(path), origins, tree.dropped), origins


def load_tree(path: str | Path) -> LoadedTree:
    """Load a config file and everything it inherits from."""
    tree = LoadedTree(data={})
    tree.data, tree.origins = _load_file(Path(path).resolve(), 0, (), tree)
    return tree


# ------------------------------------------------------------------------ overrides


def apply_overrides(tree: LoadedTree, overrides: Sequence[str]) -> None:
    """Apply ``key.path=value`` overrides in place.

    Values are parsed as YAML, so they mean the same as in a file. Overriding ``type``
    follows the merge rule: kwargs of the previous component are dropped.
    """
    for item in overrides:
        key, separator, raw = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ConfigError(
                [Issue("", f"override {item!r} must look like `key.path=value`", CLI_SOURCE)]
            )
        try:
            value = parse_yaml(raw)
        except yaml.YAMLError as error:
            raise ConfigError(
                [Issue(key, f"cannot parse value {raw!r}: {error}", CLI_SOURCE)]
            ) from None
        _assign(tree, key.split("."), value)


def _index(container: list, token: str, path: KeyPath) -> int:
    if not token.lstrip("-").isdigit() or not -len(container) <= int(token) < len(container):
        raise ConfigError(
            [
                Issue(
                    dotted((*path, token)),
                    f"'{dotted(path)}' is a list of length {len(container)}; "
                    f"'{token}' is not a valid index",
                    CLI_SOURCE,
                )
            ]
        )
    return int(token) % len(container)


def _assign(tree: LoadedTree, tokens: list[str], value: Any) -> None:
    node: Any = tree.data
    path: KeyPath = ()
    for token in tokens[:-1]:
        if isinstance(node, list):
            index = _index(node, token, path)
            node, path = node[index], (*path, index)
        elif isinstance(node, dict):
            if node.get(token) is None:
                node[token] = {}
            node, path = node[token], (*path, token)
        if not isinstance(node, (dict, list)):
            raise ConfigError(
                [
                    Issue(
                        ".".join(tokens),
                        f"'{dotted(path)}' is a {type(node).__name__}, not a mapping or list",
                        CLI_SOURCE,
                    )
                ]
            )
    last = tokens[-1]
    if isinstance(node, list):
        index = _index(node, last, path)
        node[index] = value
        _record(tree.origins, (*path, index), value, CLI_SOURCE)
        return
    sub = (*path, last)
    if last == TYPE_KEY and TYPE_KEY in node and node[TYPE_KEY] != value:
        stale = tuple(sorted(key for key in node if key != TYPE_KEY))
        if stale:
            tree.dropped.append(
                Dropped(dotted(path), str(node[TYPE_KEY]), str(value), stale, CLI_SOURCE)
            )
        for key in stale:
            del node[key]
        _record(tree.origins, path, {}, CLI_SOURCE)
    if isinstance(value, dict) and isinstance(node.get(last), dict):
        node[last] = _merge(node[last], value, sub, CLI_SOURCE, tree.origins, tree.dropped)
    else:
        node[last] = value
        _record(tree.origins, sub, value, CLI_SOURCE)


# -------------------------------------------------------------------- interpolation

_REFERENCE = re.compile(r"\$\{([^{}]+)\}")
_MISSING = object()


class _Interpolator:
    def __init__(self, root: dict[str, Any], builtins: dict[str, Any]) -> None:
        self.root = root
        self.builtins = builtins
        self.issues: list[Issue] = []
        self.active: list[KeyPath] = []

    def walk(self, node: Any, path: KeyPath) -> Any:
        if isinstance(node, dict):
            for key in list(node):
                node[key] = self.walk(node[key], (*path, key))
        elif isinstance(node, list):
            for index in range(len(node)):
                node[index] = self.walk(node[index], (*path, index))
        elif isinstance(node, str) and "${" in node:
            return self.resolve(node, path)
        return node

    def resolve(self, text: str, path: KeyPath) -> Any:
        whole = _REFERENCE.fullmatch(text.strip())
        if whole:
            value = self.lookup(whole.group(1).strip(), path)
            return text if value is _MISSING else copy.deepcopy(value)

        def substitute(match: re.Match[str]) -> str:
            value = self.lookup(match.group(1).strip(), path)
            if value is _MISSING:
                return match.group(0)
            if isinstance(value, (dict, list)):
                self.issues.append(
                    Issue(
                        dotted(path),
                        f"'{match.group(0)}' is a mapping or list and cannot be embedded in a string",
                    )
                )
                return match.group(0)
            return str(value)

        return _REFERENCE.sub(substitute, text)

    def lookup(self, reference: str, at: KeyPath) -> Any:
        if reference in self.builtins:
            return self.builtins[reference]
        node: Any = self.root
        target: KeyPath = ()
        for token in reference.split("."):
            if isinstance(node, dict) and token in node:
                node, target = node[token], (*target, token)
            elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
                node, target = node[int(token)], (*target, int(token))
            else:
                known = [dotted(leaf) for leaf in _leaves(self.root, ())] + list(self.builtins)
                close = difflib.get_close_matches(reference, known, n=1, cutoff=0.6)
                hint = f" Did you mean '${{{close[0]}}}'?" if close else ""
                self.issues.append(
                    Issue(dotted(at), f"'${{{reference}}}' does not refer to a config value.{hint}")
                )
                return _MISSING
        if target in self.active:
            cycle = " -> ".join(dotted(p) for p in (*self.active, target))
            self.issues.append(Issue(dotted(at), f"circular interpolation: {cycle}"))
            return _MISSING
        self.active.append(target)
        try:
            return self.walk(copy.deepcopy(node), target)
        finally:
            self.active.pop()


def resolve_interpolations(
    data: dict[str, Any], builtins: dict[str, Any] | None = None
) -> list[Issue]:
    """Resolve ``${path}`` references in place and return the problems found."""
    interpolator = _Interpolator(data, builtins or {})
    interpolator.walk(data, ())
    return interpolator.issues
