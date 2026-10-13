"""Optimizers beyond ``torch.optim``: bridges to optimizer libraries, and parameter groups.

Every class of ``torch.optim`` is nameable in a config already. The entries here are
registered next to them, so a config names them the same way::

    optimizer: {type: TimmOptimizer, name: lamb, lr: 2e-3, weight_decay: 0.02}
    optimizer: {type: PytorchOptimizer, name: prodigy, lr: 1.0}
    optimizer: {type: ScheduleFree, name: adamw, lr: 2.5e-3, warmup_steps: 500}

A bridge takes the options every optimizer has (``lr``, ``weight_decay``) as arguments of
its own and everything else as one mapping, ``options``, handed to the library unchanged.
The libraries are optional: ``pip install "grasppanda[optim]"``.

:func:`ParamGroups` wraps any of them (or a class of ``torch.optim``) and splits the
parameters into groups with their own learning rate: a multiplier per sub-module (a
pretrained encoder at a tenth of the rate), layer-wise decay through an encoder, and no
weight decay for biases and normalisation weights.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping
from typing import Any

from torch import nn
from torch.optim import Optimizer

from grasppanda.registry import OPTIMIZERS, Cfg
from grasppanda.utils.imports import require

from .errors import EngineError

__all__ = ["ParamGroups", "PytorchOptimizer", "ScheduleFree", "TimmOptimizer", "layer_names"]

_SCHEDULE_FREE = {
    "adamw": "AdamWScheduleFree",
    "radam": "RAdamScheduleFree",
    "sgd": "SGDScheduleFree",
}


class _Factory(Optimizer):
    """An optimizer named in a config that *is* an optimizer of another class.

    The constructor arguments are those of ``__init__`` (which is what the config check
    reads); constructing the class returns what ``build`` makes of them, an instance of
    the library's optimizer, so checkpoints and schedulers see the real thing.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(cls.__init__).bind(None, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        return cls.build(**arguments)

    @staticmethod
    def build(**arguments: Any) -> Optimizer:
        raise NotImplementedError


def _options(options: Mapping[str, Any] | None, owner: str, reserved: tuple[str, ...]) -> dict:
    options = dict(options or {})
    clash = sorted(set(options) & set(reserved))
    if clash:
        raise EngineError(
            f"{owner}.options sets {', '.join(clash)}, which {owner} takes as an argument of "
            "its own: move it out of `options`."
        )
    return options


@OPTIMIZERS.register()
class TimmOptimizer(_Factory):
    """Any optimizer of timm by its name (``lamb``, ``adan``, ``adopt``, ``lion``, ``mars``, …).

    ``timm.optim.list_optimizers()`` lists the names (98 in timm 1.0.30); a name starting
    with ``lookahead_`` wraps the optimizer in Lookahead. The parameters are passed as
    they are: timm's own filter that removes weight decay from biases and normalisation
    weights works on a model, not on parameters, so use :func:`ParamGroups` for that.

    Args:
        params: Parameters or parameter groups (supplied by the engine).
        name: Optimizer name of ``timm.optim.create_optimizer_v2``.
        lr: Learning rate.
        weight_decay: Weight decay; decoupled or not as the named optimizer defines it.
        options: Further keyword arguments of the optimizer (``betas``, ``eps``,
            ``momentum``, ``caution``, …), passed unchanged.
    """

    source = "Wightman, PyTorch Image Models (timm.optim), via timm"
    status = "bridge"
    extras = ("optim",)

    def __init__(
        self,
        params: Iterable[Any],
        name: str,
        lr: float,
        weight_decay: float = 0.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        raise AssertionError("unreachable: __new__ returns the built optimizer")

    @staticmethod
    def build(params, name, lr, weight_decay, options) -> Optimizer:  # type: ignore[override]
        optim = require("timm.optim", extra="optim", purpose="TimmOptimizer")
        known = optim.list_optimizers()
        bare = name.removeprefix("lookahead_")
        if bare not in known:
            raise EngineError(
                f"TimmOptimizer.name is '{name}', which timm {_version('timm')} does not have. "
                f"Its optimizers: {', '.join(known)}."
            )
        extra = _options(options, "TimmOptimizer", ("lr", "weight_decay", "opt", "params"))
        return optim.create_optimizer_v2(
            list(params), opt=name, lr=lr, weight_decay=weight_decay, **extra
        )


@OPTIMIZERS.register()
class PytorchOptimizer(_Factory):
    """Any optimizer of pytorch_optimizer by its name (``prodigy``, ``ademamix``, ``soap``, …).

    ``pytorch_optimizer.get_supported_optimizers()`` lists the names (127 in version 4.0.0).

    Args:
        params: Parameters or parameter groups (supplied by the engine).
        name: Optimizer name of ``pytorch_optimizer.load_optimizer``.
        lr: Learning rate. The learning-rate-free optimizers (``prodigy``,
            ``dadaptadam``, …) expect 1.0: they estimate the step size themselves.
        weight_decay: Weight decay.
        options: Further keyword arguments of the optimizer, passed unchanged.
    """

    source = "Kim, pytorch_optimizer (kozistr/pytorch_optimizer), via the library"
    status = "bridge"
    extras = ("optim",)

    def __init__(
        self,
        params: Iterable[Any],
        name: str,
        lr: float,
        weight_decay: float = 0.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        raise AssertionError("unreachable: __new__ returns the built optimizer")

    @staticmethod
    def build(params, name, lr, weight_decay, options) -> Optimizer:  # type: ignore[override]
        library = require("pytorch_optimizer", extra="optim", purpose="PytorchOptimizer")
        try:
            cls = library.load_optimizer(name)
        except NotImplementedError:
            known = sorted(
                getattr(entry, "__name__", str(entry)).lower()
                for entry in library.get_supported_optimizers()
            )
            raise EngineError(
                f"PytorchOptimizer.name is '{name}', which pytorch_optimizer "
                f"{_version('pytorch_optimizer')} does not have. Its optimizers: {', '.join(known)}."
            ) from None
        extra = _options(options, "PytorchOptimizer", ("lr", "weight_decay", "params"))
        return cls(list(params), lr=lr, weight_decay=weight_decay, **extra)


@OPTIMIZERS.register()
class ScheduleFree(_Factory):
    """A schedule-free optimizer: no learning-rate schedule, the iterates are averaged instead.

    The optimizer keeps two sequences: the point gradients are taken at, and their
    average, which is the result. It must be told which one the weights should hold, so
    a run with it lists ``ScheduleFreeHook``, which switches to the average for
    evaluation and checkpoints and back for training. Use it with ``scheduler: null``.

    Args:
        params: Parameters or parameter groups (supplied by the engine).
        name: ``adamw``, ``radam`` or ``sgd``.
        lr: Learning rate; the authors use values above those of the scheduled optimizer
            (their AdamW default is 2.5e-3).
        weight_decay: Weight decay.
        warmup_steps: Optimizer steps of linear warm-up, done inside the optimizer.
            ``radam`` needs none and does not take the option.
        options: Further keyword arguments (``betas``, ``momentum``, ``r``,
            ``weight_lr_power``), passed unchanged.
    """

    source = "Defazio et al., The Road Less Scheduled (arXiv 2405.15682), via schedulefree"
    status = "bridge"
    extras = ("optim",)

    def __init__(
        self,
        params: Iterable[Any],
        name: str = "adamw",
        lr: float = 2.5e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        options: dict[str, Any] | None = None,
    ) -> None:
        raise AssertionError("unreachable: __new__ returns the built optimizer")

    @staticmethod
    def build(params, name, lr, weight_decay, warmup_steps, options) -> Optimizer:  # type: ignore[override]
        library = require("schedulefree", extra="optim", purpose="ScheduleFree")
        if name not in _SCHEDULE_FREE:
            raise EngineError(
                f"ScheduleFree.name must be one of {', '.join(_SCHEDULE_FREE)}, got '{name}'."
            )
        extra = _options(options, "ScheduleFree", ("lr", "weight_decay", "warmup_steps", "params"))
        if name == "radam":
            if warmup_steps:
                raise EngineError("ScheduleFree: 'radam' has no warm-up; leave warmup_steps at 0.")
        else:
            extra["warmup_steps"] = warmup_steps
        cls = getattr(library, _SCHEDULE_FREE[name])
        return cls(list(params), lr=lr, weight_decay=weight_decay, **extra)


def _version(module: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(module)
    except PackageNotFoundError:
        return ""


# ------------------------------------------------------------------- parameter groups


def _matches(pattern: str, name: str) -> bool:
    """A pattern is a dotted prefix of the parameter name, or ``re:<regular expression>``."""
    if pattern.startswith("re:"):
        return re.search(pattern[3:], name) is not None
    return name == pattern or name.startswith(pattern + ".")


def layer_names(module: nn.Module, depth: int = 2) -> list[str]:
    """Dotted names of the layers of an encoder, in the order they were registered.

    A layer is a sub-module ``depth`` levels below ``module``, or a shallower one that
    has no children. For the encoders shipped here the order of registration is the
    order of the data: stem, stages, blocks.
    """
    found: list[str] = []

    def visit(current: nn.Module, prefix: str, level: int) -> None:
        children = list(current.named_children())
        if level == depth or not children:
            if any(True for _ in current.parameters()):
                found.append(prefix)
            return
        for name, child in children:
            visit(child, f"{prefix}.{name}" if prefix else name, level + 1)

    visit(module, "", 0)
    return [name for name in found if name]


@OPTIMIZERS.register()
class ParamGroups(_Factory):
    """Parameter groups for any optimizer: per-module rates, layer-wise decay, no decay on 1-D.

    The learning rate of a parameter is the ``lr`` of the wrapped optimizer times

    * the multiplier of the first pattern of ``lr_multipliers`` that matches its name
      (a pretrained encoder at a tenth of the rate: ``{backbone: 0.1}``), times
    * ``layer_decay ** (L - i)`` when it lies in layer ``i`` of the ``L`` layers of
      ``layer_decay_module``: the last layer trains at ``layer_decay`` times the rate,
      the first at ``layer_decay ** L`` (BEiT, as in ``timm``). The rest of the model
      (view stage, grouper, head) keeps the full rate.

    Every group carries ``name`` and ``lr_scale``, which ``OneCycleLR`` reads to scale
    its ``max_lr`` per group. A detector that already returns parameter groups from
    ``configure_optimizers`` is refused: its groups would be lost.

    Args:
        params: Parameters (supplied by the engine).
        model: The detector (supplied by the engine); names the parameters.
        optimizer: Node of the optimizer to build over the groups; it must set ``lr``.
        lr_multipliers: Pattern → multiplier. A pattern is a dotted prefix of the
            parameter name (``backbone``, ``backbone.enc``) or ``re:<regular expression>``.
        layer_decay: Factor per layer, e.g. 0.75; ``None`` for no layer-wise decay.
        layer_decay_module: Dotted name of the sub-module whose layers are counted.
        layer_decay_depth: How many levels below that module a layer is (2: the blocks
            of its stages for most encoders). ``grasppanda.engine.optimizers.layer_names``
            lists what a value gives.
        no_decay_1d: Set ``weight_decay`` to 0 for biases and other 1-D parameters
            (normalisation weights), the usual rule for AdamW.
    """

    source = "Bao et al., BEiT (arXiv 2106.08254): layer-wise decay as in timm.optim"
    status = "tested"

    def __init__(
        self,
        params: Iterable[Any],
        model: nn.Module,
        optimizer: Cfg[Optimizer],
        lr_multipliers: dict[str, float] | None = None,
        layer_decay: float | None = None,
        layer_decay_module: str = "backbone",
        layer_decay_depth: int = 2,
        no_decay_1d: bool = False,
    ) -> None:
        raise AssertionError("unreachable: __new__ returns the built optimizer")

    @staticmethod
    def build(
        params,
        model,
        optimizer,
        lr_multipliers,
        layer_decay,
        layer_decay_module,
        layer_decay_depth,
        no_decay_1d,
    ) -> Optimizer:  # type: ignore[override]
        node = dict(optimizer)
        if "lr" not in node:
            raise EngineError("ParamGroups.optimizer must set `lr`: the multipliers scale it.")
        base = float(node["lr"])
        names = {id(param): name for name, param in model.named_parameters()}
        members = list(params)
        if any(isinstance(param, Mapping) for param in members):
            raise EngineError(
                f"{type(model).__name__}.configure_optimizers already returns parameter groups; "
                "ParamGroups would discard them. Use the optimizer without ParamGroups."
            )
        multipliers = dict(lr_multipliers or {})
        unused = set(multipliers)
        layers: list[str] = []
        if layer_decay is not None:
            if not 0.0 < layer_decay <= 1.0:
                raise EngineError(f"ParamGroups.layer_decay must be in (0, 1], got {layer_decay}.")
            try:
                root = model.get_submodule(layer_decay_module)
            except AttributeError:
                children = ", ".join(name for name, _ in model.named_children())
                raise EngineError(
                    f"ParamGroups.layer_decay_module is '{layer_decay_module}', but "
                    f"{type(model).__name__} has no such sub-module (it has: {children})."
                ) from None
            layers = [
                f"{layer_decay_module}.{name}" for name in layer_names(root, layer_decay_depth)
            ]
            if not layers:
                raise EngineError(f"'{layer_decay_module}' has no layer with parameters.")
        groups: dict[tuple[str, float, bool], list[Any]] = {}
        for param in members:
            name = names.get(id(param), "")
            scale, label = 1.0, "rest"
            for pattern, multiplier in multipliers.items():
                if _matches(pattern, name):
                    scale, label = float(multiplier), pattern
                    unused.discard(pattern)
                    break
            if layers and _matches(layer_decay_module, name):
                index = next((i for i, layer in enumerate(layers) if _matches(layer, name)), 0)
                scale *= layer_decay ** (len(layers) - index)
                label = f"{layer_decay_module}[{index}]"
            plain = no_decay_1d and (param.ndim <= 1 or name.endswith(".bias"))
            groups.setdefault((label, scale, plain), []).append(param)
        if unused:
            raise EngineError(
                f"ParamGroups.lr_multipliers: no parameter matches {', '.join(sorted(unused))}. "
                f"Top-level modules of {type(model).__name__}: "
                + ", ".join(name for name, _ in model.named_children())
                + "."
            )
        built = []
        for (label, scale, plain), tensors in groups.items():
            group: dict[str, Any] = {
                "params": tensors,
                "lr": base * scale,
                "lr_scale": scale,
                "name": label + (" (no decay)" if plain else ""),
            }
            if plain:
                group["weight_decay"] = 0.0
            built.append(group)
        return OPTIMIZERS.build(node, params=built, model=model)
