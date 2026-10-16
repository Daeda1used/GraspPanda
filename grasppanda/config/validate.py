"""Strict validation, run before any dataset is opened or any module is built.

Three layers (``docs/design/ARCHITECTURE.md`` §5.3):

1. schema - top-level and fixed-structure keys;
2. component signatures - every ``{type: X, ...}`` node against ``X.__init__``;
3. cross-field rules - see :mod:`grasppanda.config.checker`.

All problems of one pass are collected and reported together.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import difflib
import hashlib
import importlib
import importlib.util
import inspect
import sys
import types
import typing
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, Union

from grasppanda.registry import (
    INJECTED,
    Cfg,
    Registry,
    RegistryError,
    component_parameters,
    slot_registry,
)

from .errors import Issue, kind_of
from .loader import TYPE_KEY, KeyPath, dotted

__all__ = ["fill_defaults", "import_custom", "iter_components", "validate_config"]

_SEQUENCE_ORIGINS = (list, tuple, collections.abc.Sequence)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping)
_NONE = type(None)


class _Context:
    def __init__(self, origins: Mapping[KeyPath, str] | None = None) -> None:
        self.origins = origins or {}
        self.issues: list[Issue] = []

    def fork(self) -> _Context:
        return _Context(self.origins)

    def add(self, path: KeyPath, message: str, kind: str | None = "not-a-combination") -> None:
        """Record a problem. A message that carries its kind (a ``Refusal``) keeps it; a
        plain one gets ``kind``: problems of a config's own statements (a missing option,
        an unknown key, a wrong type) are not about a combination."""
        source = self.origins.get(path)
        if source is None:
            source = next((s for p, s in self.origins.items() if p[: len(path)] == path), None)
        self.issues.append(Issue(dotted(path), str(message), source, kind_of(message, kind)))


def _suggest(name: str, candidates: Any) -> str:
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=1, cutoff=0.6)
    return f" Did you mean '{close[0]}'?" if close else ""


def _describe(annotation: Any) -> str:
    if annotation is _NONE:
        return "null"
    origin = typing.get_origin(annotation)
    if origin is Cfg:
        registry = slot_registry(annotation)
        return f"a {registry.name} component" if registry else "a component"
    if origin in (Union, types.UnionType):
        return " | ".join(_describe(arg) for arg in typing.get_args(annotation))
    if origin is Literal:
        return "one of " + ", ".join(repr(arg) for arg in typing.get_args(annotation))
    if origin in _SEQUENCE_ORIGINS:
        args = [a for a in typing.get_args(annotation) if a is not Ellipsis]
        return f"list of {_describe(args[0])}" if args else "list"
    if origin in _MAPPING_ORIGINS:
        return "mapping"
    if dataclasses.is_dataclass(annotation):
        return "mapping"
    return getattr(annotation, "__name__", str(annotation))


def _show(value: Any) -> str:
    text = repr(value)
    return f"{type(value).__name__} {text if len(text) <= 60 else text[:57] + '...'}"


def _is_opaque_class(annotation: Any) -> bool:
    return (
        typing.get_origin(annotation) is None  # on Python 3.10 `list[int]` passes isclass()
        and inspect.isclass(annotation)
        and annotation not in (bool, int, float, str, list, tuple, dict, _NONE)
        and not dataclasses.is_dataclass(annotation)
    )


def _has_slot(annotation: Any) -> bool:
    if typing.get_origin(annotation) is Cfg:
        return True
    return any(_has_slot(arg) for arg in typing.get_args(annotation))


def _hints(obj: Any) -> dict[str, Any]:
    target = obj.__init__ if inspect.isclass(obj) and not dataclasses.is_dataclass(obj) else obj
    return typing.get_type_hints(target)


# ------------------------------------------------------------------ generic checker


def _check(value: Any, annotation: Any, path: KeyPath, ctx: _Context) -> None:
    if annotation is Any or annotation is inspect.Parameter.empty:
        return
    origin = typing.get_origin(annotation)
    if origin is Cfg:
        _check_component(value, slot_registry(annotation), path, ctx)
    elif dataclasses.is_dataclass(annotation):
        _check_dataclass(value, annotation, path, ctx)
    elif origin in (Union, types.UnionType):
        _check_union(value, annotation, path, ctx)
    elif origin is Literal:
        if value not in typing.get_args(annotation):
            ctx.add(path, f"expected {_describe(annotation)}, got {_show(value)}.")
    elif origin in _SEQUENCE_ORIGINS or annotation in (list, tuple):
        if not isinstance(value, (list, tuple)):
            ctx.add(path, f"expected a list, got {_show(value)}.")
            return
        args = typing.get_args(annotation)
        if origin is tuple and args and Ellipsis not in args:
            if len(value) != len(args):
                ctx.add(path, f"expected {len(args)} items, got {len(value)}.")
                return
            for index, (item, arg) in enumerate(zip(value, args)):
                _check(item, arg, (*path, index), ctx)
        elif args:
            for index, item in enumerate(value):
                _check(item, args[0], (*path, index), ctx)
    elif origin in _MAPPING_ORIGINS or annotation is dict:
        if not isinstance(value, Mapping):
            ctx.add(path, f"expected a mapping, got {_show(value)}.")
            return
        args = typing.get_args(annotation)
        if len(args) == 2:
            for key, item in value.items():
                _check(item, args[1], (*path, key), ctx)
    elif annotation is bool:
        if not isinstance(value, bool):
            ctx.add(path, f"expected true or false, got {_show(value)}.")
    elif annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            ctx.add(path, f"expected an integer, got {_show(value)}.")
    elif annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            ctx.add(path, f"expected a number, got {_show(value)}.")
    elif annotation is str:
        if not isinstance(value, str):
            ctx.add(path, f"expected a string, got {_show(value)}.")
    elif annotation is _NONE and value is not None:
        ctx.add(path, f"expected null, got {_show(value)}.")
    # Other annotations (tensors, arbitrary classes) cannot be checked statically.


def _check_union(value: Any, annotation: Any, path: KeyPath, ctx: _Context) -> None:
    args = typing.get_args(annotation)
    if value is None and _NONE in args:
        return
    candidates = [arg for arg in args if arg is not _NONE]
    trials: list[tuple[Any, _Context]] = []
    for candidate in candidates:
        trial = ctx.fork()
        if _is_opaque_class(candidate):
            # A class we cannot check structurally (a tensor, say) only matches instances.
            if isinstance(value, candidate):
                return
            trial.add(path, f"expected {_describe(annotation)}, got {_show(value)}.")
        else:
            _check(value, candidate, path, trial)
        if not trial.issues:
            return
        trials.append((candidate, trial))
    # Report the candidate whose shape matches the value; its messages are specific.
    structural = [
        trial
        for candidate, trial in trials
        if isinstance(value, Mapping)
        and (typing.get_origin(candidate) is Cfg or dataclasses.is_dataclass(candidate))
    ]
    if structural:
        ctx.issues.extend(structural[0].issues)
    elif len(trials) == 1:
        ctx.issues.extend(trials[0][1].issues)
    else:
        ctx.add(path, f"expected {_describe(annotation)}, got {_show(value)}.")


def _check_dataclass(value: Any, cls: Any, path: KeyPath, ctx: _Context) -> None:
    if not isinstance(value, Mapping):
        ctx.add(path, f"expected a mapping, got {_show(value)}.")
        return
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = _hints(cls)
    kind = "top-level key" if not path else "key"
    for key in value:
        if key not in fields:
            ctx.add(
                (*path, key),
                f"unknown {kind} '{key}'.{_suggest(key, fields)} Valid keys: {', '.join(fields)}.",
            )
    for name, spec in fields.items():
        if name in value:
            _check(value[name], hints[name], (*path, name), ctx)
        elif spec.default is dataclasses.MISSING and spec.default_factory is dataclasses.MISSING:
            ctx.add((*path, name), f"missing required key '{name}'.")


def _is_component_node(value: Any) -> bool:
    return isinstance(value, Mapping) and TYPE_KEY in value


def _check_component(value: Any, registry: Registry | None, path: KeyPath, ctx: _Context) -> None:
    if registry is None:
        return
    if not isinstance(value, Mapping):
        ctx.add(path, f"expected a {registry.name} component `{{type: ...}}`, got {_show(value)}.")
        return
    if TYPE_KEY not in value:
        ctx.add(
            path,
            f"missing 'type'. Choose one of {registry.name}: {', '.join(registry.names()) or '(none yet)'}.",
        )
        return
    try:
        cls = registry.get(value[TYPE_KEY])
    except RegistryError as error:
        ctx.add((*path, TYPE_KEY), str(error))
        return
    params, var_kwargs = component_parameters(cls)
    problem: Exception | None = None
    try:
        hints = _hints(cls)
    except Exception as error:
        hints = {}
        if not registry.third_party:  # for e.g. torch.optim: check names, skip types
            problem = error
    if problem is not None:  # a developer error, surfaced where the user will see it
        ctx.add(
            path,
            f"cannot resolve the type hints of {cls.__name__}: {problem}. Import annotation "
            "names at runtime (not under TYPE_CHECKING) so configs can be validated.",
        )
        return
    before = len(ctx.issues)
    for key, sub in value.items():
        if key == TYPE_KEY:
            continue
        if key in registry.provided:
            ctx.add(
                (*path, key), f"'{key}' is supplied by the engine and cannot be set in a config."
            )
        elif key not in params:
            if var_kwargs:
                continue
            names = [name for name in params if name not in registry.provided]
            ctx.add(
                (*path, key),
                f"{cls.__name__} has no parameter '{key}'.{_suggest(key, names)} "
                f"Parameters: {', '.join(names) or '(none)'}.",
            )
        else:
            annotation = hints.get(key, Any)
            if _is_component_node(sub) and not _has_slot(annotation):
                ctx.add(
                    (*path, key),
                    f"{cls.__name__}.{key} is not a component slot, but a `{{type: ...}}` node "
                    "was given.",
                )
            else:
                _check(sub, annotation, (*path, key), ctx)
    for name, param in params.items():
        if name in value or name in registry.provided or name in registry.injectable:
            continue
        if param.default is inspect.Parameter.empty:
            ctx.add((*path, name), f"{cls.__name__} requires '{name}'.")
    if len(ctx.issues) == before:
        check = getattr(cls, "check_config", None)
        if callable(check):
            # A subclass may state a rule its base already checks: report a problem once.
            for message in dict.fromkeys(check(dict(value)) or ()):
                ctx.add(path, message)


# ------------------------------------------------------------------------- defaults


def _serialisable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_serialisable(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _serialisable(item) for key, item in value.items())
    return False


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _fill(value: Any, annotation: Any, library: bool = True) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        if value is None:
            return None
        for candidate in typing.get_args(annotation):
            structural = typing.get_origin(candidate) is Cfg or dataclasses.is_dataclass(candidate)
            if isinstance(value, Mapping) and structural:
                return _fill(value, candidate, library)
        return value
    if dataclasses.is_dataclass(annotation) and isinstance(value, Mapping):
        hints = _hints(annotation)
        out = dict(value)
        for spec in dataclasses.fields(annotation):
            if spec.name in out:
                out[spec.name] = _fill(out[spec.name], hints[spec.name], library)
            elif spec.default is not dataclasses.MISSING:
                out[spec.name] = _plain(spec.default)
            elif spec.default_factory is not dataclasses.MISSING:
                default = spec.default_factory()
                if dataclasses.is_dataclass(default):
                    default = dataclasses.asdict(default)
                out[spec.name] = _plain(default)
        return out
    if origin is Cfg and _is_component_node(value):
        registry = slot_registry(annotation)
        try:
            cls = registry.get(value[TYPE_KEY]) if registry else None
            hints = _hints(cls) if cls else {}
        except Exception:
            return value
        params, _ = component_parameters(cls)
        out = dict(value)
        # A class of an installed library (an optimizer or a scheduler of PyTorch named by
        # class) has the defaults of whatever version is installed.
        foreign = registry.third_party and cls.__module__.split(".")[0] != "grasppanda"
        own = library or not foreign
        for name, param in params.items():
            if name in out:
                out[name] = _fill(out[name], hints.get(name, Any), library)
            elif (
                own
                and name not in registry.provided
                and param.default is not inspect.Parameter.empty
                and param.default is not INJECTED
                and _serialisable(param.default)
            ):
                out[name] = _plain(param.default)
        return out
    if origin in _SEQUENCE_ORIGINS and isinstance(value, list):
        args = [a for a in typing.get_args(annotation) if a is not Ellipsis]
        return [_fill(item, args[0], library) for item in value] if args else value
    if origin in _MAPPING_ORIGINS and isinstance(value, Mapping):
        args = typing.get_args(annotation)
        if len(args) != 2:
            return value
        return {k: _fill(v, args[1], library) for k, v in value.items()}
    return value


def fill_defaults(data: Mapping[str, Any], *, library: bool = True) -> dict[str, Any]:
    """Return a copy of ``data`` with schema and component defaults written out.

    With ``library=False`` the defaults of classes that belong to an installed library
    (PyTorch's optimizers and schedulers, named by class) are left out: they change with
    the installed version, while everything GraspPanda defines does not.
    """
    from .schema import RootConfig

    return _fill(dict(data), RootConfig, library)


# ----------------------------------------------------------------- component nodes


def _walk(value: Any, annotation: Any, path: KeyPath) -> Iterator[tuple[KeyPath, Registry, str]]:
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        for candidate in typing.get_args(annotation):
            kind = typing.get_origin(candidate)
            structural = kind is Cfg or dataclasses.is_dataclass(candidate)
            if (isinstance(value, Mapping) and (structural or kind in _MAPPING_ORIGINS)) or (
                isinstance(value, (list, tuple)) and kind in _SEQUENCE_ORIGINS
            ):
                yield from _walk(value, candidate, path)
                return
    elif dataclasses.is_dataclass(annotation):
        if isinstance(value, Mapping):
            hints = _hints(annotation)
            for name in value:
                if name in hints:
                    yield from _walk(value[name], hints[name], (*path, name))
    elif origin is Cfg:
        registry = slot_registry(annotation)
        if registry is None or not _is_component_node(value):
            return
        try:
            hints = _hints(registry.get(value[TYPE_KEY]))
        except Exception:  # unknown type or unresolvable hints: validation reports those
            return
        yield path, registry, str(value[TYPE_KEY])
        for name, sub in value.items():
            if name in hints:
                yield from _walk(sub, hints[name], (*path, name))
    elif origin in _SEQUENCE_ORIGINS:
        args = [a for a in typing.get_args(annotation) if a is not Ellipsis]
        if args and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from _walk(item, args[0], (*path, index))
    elif origin in _MAPPING_ORIGINS:
        args = typing.get_args(annotation)
        if len(args) == 2 and isinstance(value, Mapping):
            for key, item in value.items():
                yield from _walk(item, args[1], (*path, key))


def iter_components(data: Mapping[str, Any]) -> Iterator[tuple[KeyPath, Registry, str]]:
    """Yield ``(path, registry, name)`` for every component node of a config tree.

    The registry of a node is the one that owns the slot it sits in, decided by the
    annotations exactly as in validation, so a name that exists in two registries is
    attributed correctly. Nodes validation would reject are skipped, not reported.
    """
    from .schema import RootConfig

    yield from _walk(data, RootConfig, ())


# ------------------------------------------------------------------- custom imports


def import_custom(entry: str, declared_in: str | None = None) -> None:
    """Import an out-of-tree module named in ``custom_imports``.

    ``entry`` is a dotted module name, or a ``.py`` path resolved against the config
    file that declares it. A path is loaded as a standalone module, which keeps a
    single-file plugin next to its config working without touching ``sys.path``.
    """
    if not entry.endswith(".py"):
        importlib.import_module(entry)
        return
    base = Path(declared_in).parent if declared_in and Path(declared_in).is_file() else Path.cwd()
    path = (base / entry).resolve()
    name = f"grasppanda_custom_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise


# ---------------------------------------------------------------------- entry point


def validate_config(
    data: Mapping[str, Any],
    origins: Mapping[KeyPath, str] | None = None,
    *,
    command: str | None = None,
) -> list[Issue]:
    """Validate a merged config tree and return every problem found."""
    from .checker import run_rules
    from .schema import RootConfig

    ctx = _Context(origins)
    imports = data.get("custom_imports") or []
    if isinstance(imports, list):
        for index, entry in enumerate(imports):
            if not isinstance(entry, str):
                continue
            declared_in = ctx.origins.get(("custom_imports", index))
            try:
                import_custom(entry, declared_in)
            except Exception as error:
                ctx.add(
                    ("custom_imports", index),
                    f"cannot import '{entry}': {type(error).__name__}: {error}. Use a module "
                    "name importable from this environment, or a `.py` path relative to the "
                    "config file.",
                )
    _check_dataclass(data, RootConfig, (), ctx)
    for path, message in run_rules(data, command):
        ctx.add(path, message, kind=None)  # a rule of a project may not say
    return ctx.issues
