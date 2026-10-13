"""Component registries.

Every swappable piece of GraspPanda is a class registered in exactly one flat
:class:`Registry`. A config node ``{type: Name, **kwargs}`` is resolved against the
registry that owns the slot it sits in; the owner is decided by the parent's type
annotation (``Cfg[Backbone]``), never by searching every registry for a name.

Design notes (see ``docs/design/ARCHITECTURE.md`` §6):

* Lookup is lazy: the first ``get`` imports every module below the registry's
  package, so adding a file is enough to register a component.
* Duplicate names raise. A failed lookup lists close matches and says whether the
  name lives in a different registry.
* Registered classes may not take ``**kwargs``: every config key must be checkable
  against a signature before anything is constructed.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import pkgutil
import typing
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

__all__ = [
    "ASSIGNERS",
    "BACKBONES",
    "DATASETS",
    "DETECTORS",
    "EVALUATORS",
    "GROUPERS",
    "HEADS",
    "HOOKS",
    "IMAGE_ENCODERS",
    "INJECTED",
    "LOSSES",
    "OPTIMIZERS",
    "POSTPROCESSORS",
    "PROPOSALS",
    "SCHEDULERS",
    "TRANSFORMS",
    "VIEWS",
    "BuildError",
    "Cfg",
    "Registry",
    "RegistryError",
    "all_registries",
    "component_parameters",
    "registry_for",
    "slot_registry",
]

T = TypeVar("T")
C = TypeVar("C", bound=Callable[..., Any])


class RegistryError(LookupError):
    """A component name could not be resolved or registered."""


class BuildError(RuntimeError):
    """A component could not be constructed from its config node."""


class _Injected:
    """Sentinel default: the parent component supplies this argument at build time."""

    _instance: _Injected | None = None

    def __new__(cls) -> _Injected:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "INJECTED"

    def __bool__(self) -> bool:
        return False


INJECTED: Any = _Injected()


class Cfg(Generic[T]):
    """Annotation for a component slot.

    ``backbone: Cfg[Backbone]`` means: either an already built ``Backbone`` or a
    config node ``{type: ...}`` that builds one. The static validator uses the type
    argument to find the registry that owns the slot.
    """

    __slots__ = ()


_REGISTRIES: dict[str, Registry] = {}
_BASES: dict[type, Registry] = {}
_plugins_loaded = False


def _load_plugins() -> None:
    """Import pip-installed plugins that declare a ``grasppanda.plugins`` entry point."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    from importlib.metadata import entry_points

    for entry_point in entry_points(group="grasppanda.plugins"):
        entry_point.load()


def component_parameters(obj: Callable[..., Any]) -> tuple[dict[str, inspect.Parameter], bool]:
    """Return the keyword parameters of a component and whether it takes ``**kwargs``."""
    if inspect.isclass(obj) and obj.__init__ is object.__init__:
        return (
            {},
            False,
        )  # no constructor of its own: takes nothing (object.__init__ is *args, **kwargs)
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # builtins without a signature
        return {}, True
    params: dict[str, inspect.Parameter] = {}
    var_kwargs = False
    for index, param in enumerate(signature.parameters.values()):
        if inspect.isclass(obj) and index == 0:  # self
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            var_kwargs = True
        elif param.kind is not inspect.Parameter.VAR_POSITIONAL:
            params[param.name] = param
    return params, var_kwargs


class Registry:
    """A flat name → component table with lazy auto-import.

    Args:
        name: Display name, upper case by convention (``"BACKBONES"``).
        location: Dotted package whose modules are imported on first lookup.
        provided: Constructor arguments the engine supplies itself (for example
            ``params`` of an optimizer); the validator does not expect them in a config.
        injectable: Required constructor arguments the engine fills in when a config
            leaves them out (for example ``T_max`` of a scheduler, which is the length
            of the run); a config may still set them.
        populate: Callback that fills the registry on first lookup, used to expose
            third-party classes such as ``torch.optim`` optimizers.
    """

    def __init__(
        self,
        name: str,
        *,
        location: str | None = None,
        provided: tuple[str, ...] = (),
        injectable: tuple[str, ...] = (),
        populate: Callable[[Registry], None] | None = None,
    ) -> None:
        if name in _REGISTRIES:
            raise ValueError(f"registry {name!r} already exists")
        self.name = name
        self.provided = frozenset(provided)
        self.injectable = frozenset(injectable)
        self._location = location
        self._populate = populate
        self._modules: dict[str, Any] = {}
        self._base: type | None = None
        self._imported = False
        self._import_errors: dict[str, BaseException] = {}
        _REGISTRIES[name] = self

    def __repr__(self) -> str:
        return f"Registry({self.name}, {len(self._modules)} registered)"

    # ------------------------------------------------------------------ registration

    def base(self, cls: type[T]) -> type[T]:
        """Class decorator declaring the base type of this registry's components."""
        if self._base is not None and self._base is not cls:
            raise ValueError(f"{self.name} already has base {self._base.__name__}")
        self._base = cls
        _BASES[cls] = self
        return cls

    @property
    def base_class(self) -> type | None:
        return self._base

    @property
    def third_party(self) -> bool:
        """True when the registry exposes classes this project does not define."""
        return self._populate is not None

    def register(
        self, name: str | None = None, *, force: bool = False, allow_var_kwargs: bool = False
    ) -> Callable[[C], C]:
        """Decorator form of :meth:`add`."""

        def decorator(obj: C) -> C:
            self.add(obj, name, force=force, allow_var_kwargs=allow_var_kwargs)
            return obj

        return decorator

    def add(
        self,
        obj: Callable[..., Any],
        name: str | None = None,
        *,
        force: bool = False,
        allow_var_kwargs: bool = False,
    ) -> None:
        """Register ``obj`` under ``name`` (default: its ``__name__``)."""
        if not callable(obj):
            raise TypeError(
                f"only classes and functions can be registered, got {type(obj).__name__}"
            )
        key = name or obj.__name__
        if not force and key in self._modules:
            existing = self._modules[key]
            raise RegistryError(
                f"'{key}' is already registered in {self.name} by "
                f"{existing.__module__}.{existing.__qualname__}. "
                "Choose another name or pass force=True to replace it."
            )
        if self._base is not None and inspect.isclass(obj) and not issubclass(obj, self._base):
            raise TypeError(
                f"{obj.__name__} must subclass {self._base.__name__} to be registered in {self.name}"
            )
        _, var_kwargs = component_parameters(obj)
        if var_kwargs and not allow_var_kwargs:
            raise TypeError(
                f"{obj.__name__} accepts **kwargs, so its config keys cannot be validated. "
                "Declare the parameters explicitly."
            )
        self._modules[key] = obj

    # ------------------------------------------------------------------------ lookup

    def _ensure_imported(self) -> None:
        if self._imported:
            return
        self._imported = True  # set first: modules may look things up while importing
        _load_plugins()
        if self._populate is not None:
            self._populate(self)
        if self._location is None:
            return
        package = importlib.import_module(self._location)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{self._location}."):
            try:
                importlib.import_module(info.name)
            except ImportError as error:  # keep going; reported on a failed lookup
                self._import_errors[info.name] = error

    def names(self) -> list[str]:
        """Sorted names of all registered components."""
        self._ensure_imported()
        return sorted(self._modules)

    def __contains__(self, name: object) -> bool:
        self._ensure_imported()
        return name in self._modules

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        self._ensure_imported()
        return len(self._modules)

    def get(self, name: str) -> Any:
        """Return the component registered as ``name`` or raise :class:`RegistryError`."""
        self._ensure_imported()
        try:
            return self._modules[name]
        except KeyError:
            raise RegistryError(self._not_found_message(name)) from None

    def _not_found_message(self, name: str) -> str:
        lines = [f"'{name}' is not registered in {self.name}."]
        close = difflib.get_close_matches(str(name), list(self._modules), n=3, cutoff=0.6)
        if close:
            lines.append("Did you mean " + " or ".join(f"'{c}'" for c in close) + "?")
        elsewhere = []
        for other in _REGISTRIES.values():
            if other is self:
                continue
            try:
                if name in other:
                    elsewhere.append(other.name)
            except ImportError:
                continue
        if elsewhere:
            lines.append(
                f"'{name}' is registered in {', '.join(elsewhere)}, which owns a different slot."
            )
        available = sorted(self._modules)
        if available:
            shown = ", ".join(available[:12])
            more = f" … ({len(available) - 12} more)" if len(available) > 12 else ""
            lines.append(f"Registered: {shown}{more}.")
        else:
            lines.append(f"{self.name} has no registered components yet.")
        if self._import_errors:
            module, error = next(iter(self._import_errors.items()))
            lines.append(
                f"{len(self._import_errors)} module(s) failed to import, e.g. {module}: {error}"
            )
        lines.append("Out-of-tree components are loaded with `custom_imports: [your.module]`.")
        return "\n".join(lines)

    # ------------------------------------------------------------------------- build

    def build(self, cfg: Any, /, **injected: Any) -> Any:
        """Construct a component from a config node.

        ``cfg`` is ``{type: Name, **kwargs}`` or an already built object, which is
        returned unchanged. ``injected`` values fill parameters the node leaves unset;
        a node that sets the same parameter to a different value is an error rather
        than a silent override.
        """
        if not isinstance(cfg, Mapping):
            if self._base is not None and not isinstance(cfg, self._base):
                raise BuildError(
                    f"{self.name} expected a config node or a {self._base.__name__}, "
                    f"got {type(cfg).__name__}"
                )
            return cfg
        node = dict(cfg)
        if "type" not in node:
            raise BuildError(f"a {self.name} node needs a 'type' key, got keys {sorted(node)}")
        cls = self.get(node.pop("type"))
        params, _ = component_parameters(cls)
        for key, value in injected.items():
            if key not in params:
                continue
            if key in node and not _same(node[key], value):
                raise BuildError(
                    f"{cls.__name__}.{key} is {node[key]!r} in the config, but the parent "
                    f"component provides {value!r}. Remove it from the config or make them agree."
                )
            node[key] = value
        unfilled = [p.name for p in params.values() if p.default is INJECTED and p.name not in node]
        if unfilled:
            raise BuildError(
                f"{cls.__name__} needs {', '.join(unfilled)}: set it in the config or build the "
                "component through a parent that injects it."
            )
        try:
            return cls(**node)
        except TypeError as error:
            raise BuildError(f"cannot build {cls.__name__} from {node}: {error}") from error


def _same(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


def all_registries() -> dict[str, Registry]:
    """All registries by name."""
    return dict(_REGISTRIES)


def registry_for(tp: Any) -> Registry | None:
    """Return the registry whose base class ``tp`` is or derives from."""
    if not inspect.isclass(tp):
        return None
    for registry in _REGISTRIES.values():
        if registry._populate is not None:
            registry._ensure_imported()
    for cls in inspect.getmro(tp):
        if cls in _BASES:
            return _BASES[cls]
    return None


def slot_registry(annotation: Any) -> Registry | None:
    """Return the registry named by a ``Cfg[Base]`` annotation, else ``None``."""
    if typing.get_origin(annotation) is Cfg:
        (base,) = typing.get_args(annotation)
        registry = registry_for(base)
        if registry is None:
            raise RegistryError(
                f"Cfg[{getattr(base, '__name__', base)}] does not name a registry base"
            )
        return registry
    return None


def _populate_optimizers(registry: Registry) -> None:
    import torch

    registry.base(torch.optim.Optimizer)
    for name, obj in vars(torch.optim).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, torch.optim.Optimizer)
            and obj is not torch.optim.Optimizer
        ):
            registry.add(obj, name, allow_var_kwargs=True)


def _populate_schedulers(registry: Registry) -> None:
    from torch.optim import lr_scheduler

    registry.base(lr_scheduler.LRScheduler)
    for name, obj in vars(lr_scheduler).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, lr_scheduler.LRScheduler)
            and obj is not lr_scheduler.LRScheduler
            and not name.startswith("_")
        ):
            registry.add(obj, name, allow_var_kwargs=True)
    # The schedules GraspPanda adds register themselves when their module is imported.
    importlib.import_module("grasppanda.engine.schedulers")


DATASETS = Registry("DATASETS", location="grasppanda.datasets")
TRANSFORMS = Registry("TRANSFORMS", location="grasppanda.transforms")
BACKBONES = Registry("BACKBONES", location="grasppanda.models.backbones")
IMAGE_ENCODERS = Registry("IMAGE_ENCODERS", location="grasppanda.models.image_encoders")
PROPOSALS = Registry("PROPOSALS", location="grasppanda.models.proposals")
VIEWS = Registry("VIEWS", location="grasppanda.models.views")
GROUPERS = Registry("GROUPERS", location="grasppanda.models.groupers")
ASSIGNERS = Registry("ASSIGNERS", location="grasppanda.models.assigners")
HEADS = Registry("HEADS", location="grasppanda.models.heads")
LOSSES = Registry("LOSSES", location="grasppanda.models.losses")
DETECTORS = Registry("DETECTORS", location="grasppanda.models.detectors")
POSTPROCESSORS = Registry("POSTPROCESSORS", location="grasppanda.models.postprocessors")
EVALUATORS = Registry("EVALUATORS", location="grasppanda.evaluation")
HOOKS = Registry("HOOKS", location="grasppanda.engine.hooks")
OPTIMIZERS = Registry("OPTIMIZERS", provided=("params",), populate=_populate_optimizers)
SCHEDULERS = Registry(
    "SCHEDULERS", provided=("optimizer",), injectable=("T_max",), populate=_populate_schedulers
)
