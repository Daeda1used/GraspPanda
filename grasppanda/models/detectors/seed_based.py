"""The stage skeleton of seed-based grasp detectors.

Methods of the GraspNet family share one flow (``docs/design/ARCHITECTURE.md`` §8.5):

    backbone → proposal → view → (assigner, with labels only) → grouper → head bundle

``SeedGraspDetector`` holds that flow and nothing method-specific: which stages run is
decided by the configured components, and their compatibility is decided by the
contracts they declare. It is deliberately small. Graspness (:class:`GSNet`) is its
first user; the split between this base and a concrete method is to be re-examined
when the second seed-based method lands.

Ground truth enters in exactly one place, the assigner, and only through the
``supervision`` argument of :meth:`SeedGraspDetector.loss`.
"""

from __future__ import annotations

import functools
import json
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from grasppanda.config.errors import REFUSAL_KINDS, Refusal
from grasppanda.models.base import (
    Assigner,
    Backbone,
    BaseGraspDetector,
    Grouper,
    HeadBundle,
    Loss,
    Proposal,
    ViewStage,
    configured_width,
)
from grasppanda.models.heads.friction import SCORE_TARGETS
from grasppanda.models.losses.terms import (
    TERM_KINDS,
    accepted_terms,
    applies_weights,
    fitting_losses,
)
from grasppanda.registry import (
    ASSIGNERS,
    BACKBONES,
    GROUPERS,
    HEADS,
    INJECTED,
    LOSSES,
    PROPOSALS,
    VIEWS,
    Cfg,
    Registry,
    RegistryError,
    component_parameters,
)
from grasppanda.structures import (
    ApproachSet,
    ContractError,
    GraspGroup,
    Gripper,
    GroupFeatures,
    Observation,
    PointFeatures,
    SeedSet,
    Supervision,
)

__all__ = [
    "CONTRACT_RULES",
    "STAGE_REGISTRIES",
    "SeedGraspDetector",
    "StageOutputs",
    "axis_adapters",
    "stage_contract_problems",
    "stage_contract_warnings",
]


@dataclass(eq=False)
class StageOutputs:
    """Everything the stages of one forward pass produced.

    Attributes:
        feats: Scene features from the backbone.
        point_scores: Per-point predictions of the proposal stage (for its loss).
        seeds: The seeds, with the features the grouper consumes (after the view stage).
        approach: Chosen approach per seed.
        matched: Labels matched by the assigner; ``None`` without supervision.
        groups: Crop features per seed.
        raw: Raw predictions of the head bundle.
    """

    feats: PointFeatures
    point_scores: Any
    seeds: SeedSet
    approach: ApproachSet
    matched: Any
    groups: GroupFeatures
    raw: Any


#: Slots of the skeleton and the registry that owns each.
STAGE_REGISTRIES: dict[str, Registry] = {
    "backbone": BACKBONES,
    "proposal": PROPOSALS,
    "view": VIEWS,
    "assigner": ASSIGNERS,
    "grouper": GROUPERS,
    "head": HEADS,
}

_UNKNOWN: Any = object()


class _Stage:
    """One stage as the contract rules see it: its class and its options.

    Built from a config node (class from the registry, options from the constructor
    defaults overlaid with the node), from a bare class (defaults only), or from a built
    component (options read from the attributes named like its constructor parameters).
    The rules therefore give the same verdict before and after construction.
    """

    def __init__(self, slot: str, component: Any) -> None:
        self.built: Any = None
        #: The config node, when the stage was given as one.
        self.node: Mapping[str, Any] | None = None
        if isinstance(component, Mapping):
            self.node = component
            self.cls = STAGE_REGISTRIES[slot].get(component["type"])
            self._options: Mapping[str, Any] | None = {**_defaults(self.cls), **component}
        elif isinstance(component, type):
            self.cls, self._options = component, _defaults(component)
        else:
            self.cls, self._options, self.built = type(component), None, component

    @property
    def name(self) -> str:
        return self.cls.__name__

    def contract(self, attribute: str, default: Any = None) -> Any:
        return getattr(self.cls, attribute, default)

    def option(self, name: str) -> Any:
        """Value of a constructor option, or ``_UNKNOWN`` when it cannot be read."""
        if self._options is not None:
            return self._options.get(name, _UNKNOWN)
        return getattr(self.built, name, _UNKNOWN)

    @property
    def options(self) -> Mapping[str, Any]:
        if self._options is not None:
            return self._options
        params, _ = component_parameters(self.cls)
        return {name: getattr(self.built, name) for name in params if hasattr(self.built, name)}


def _defaults(cls: type) -> dict[str, Any]:
    params, _ = component_parameters(cls)
    return {
        name: param.default
        for name, param in params.items()
        if param.default is not param.empty and param.default is not INJECTED
    }


def _registered(registry: Registry, attribute: str, accept: Callable[[Any], bool]) -> str:
    """Names of the registered components whose contract attribute is acceptable."""
    names = [
        name for name in registry.names() if accept(getattr(registry.get(name), attribute, None))
    ]
    return ", ".join(names) if names else "none registered"


def _fields(names: Sequence[str]) -> str:
    return " and ".join(f"'{name}'" for name in names)


def _yaml(value: Any) -> str:
    return {True: "true", False: "false", None: "null"}.get(value, str(value))


Rule = Callable[[Mapping[str, "_Stage"]], Iterator[str]]


def _rule(kind: str) -> Callable[[Rule], Rule]:
    """Declare what a contract rule's refusals are (:data:`REFUSAL_KINDS`).

    A message the rule yields as a plain string gets this kind; a rule whose verdict
    depends on the case yields a :class:`Refusal` with the kind of that case.
    """
    if kind not in REFUSAL_KINDS:
        raise ValueError(f"unknown refusal kind '{kind}'")

    def declare(function: Rule) -> Rule:
        function.kind = kind  # type: ignore[attr-defined]
        return function

    return declare


def _seed_scores(stage: _Stage) -> tuple[str, ...]:
    """The seed scores a stage predicts as configured (``seed_scores_for``, else the
    class attribute ``seed_scores``)."""
    reader = getattr(stage.cls, "seed_scores_for", None)
    if callable(reader):
        try:
            return tuple(reader(stage.options))
        except (LookupError, TypeError, ValueError):
            pass
    return tuple(stage.contract("seed_scores", ()) or ())


def _without_score(stage: _Stage, score: str) -> Mapping[str, Any] | str | None:
    """What the stage declares about leaving ``score`` to another stage
    (``without_seed_score``): the node that neither predicts nor reads it, the reason it
    cannot as configured, or ``None`` when it has no such option."""
    reader = getattr(stage.cls, "without_seed_score", None)
    if not callable(reader):
        return None
    try:
        return reader(stage.options, score)
    except (LookupError, TypeError, ValueError):
        return None


def _switch(stage: _Stage, node: Mapping[str, Any]) -> str:
    """The change from a stage to ``node``, as the YAML a user would write."""
    changed = [f"`type: {node['type']}`"] if node.get("type") != stage.name else []
    changed += [
        f"`{key}: {_yaml(value)}`"
        for key, value in node.items()
        if key != "type" and stage.option(key) != value
    ]
    return " with ".join([*changed[:1], " and ".join(changed[1:])]) if changed[1:] else changed[0]


def axis_adapters(emitted: Any, expected: Any, support: Any) -> list[str]:
    """Registered axis adapters that turn crops with axes ``emitted`` among ``support``
    into crops with axes ``expected``: groupers that wrap another grouper and declare the
    axes they read (``inner_axes``), their own (``out_axes``) and the support they share
    with what they wrap (``support``)."""
    return [
        name
        for name in GROUPERS.names()
        if getattr(GROUPERS.get(name), "inner_axes", None) == emitted
        and GROUPERS.get(name).out_axes == expected
        and GROUPERS.get(name).support == support
    ]


def _adapter_preset(adapter: str) -> str:
    """`` (preset `name`)`` for the grouper preset of an axis adapter, if there is one."""
    try:
        from grasppanda.combine.presets import presets

        names = [p.name for p in presets().get("grouper", {}).values() if p.type == adapter]
    except Exception:  # presets unreadable: the message names the component alone
        names = []
    return f" (preset `{names[0]}`)" if names else ""


@_rule("fixed-by-compose")
def _encoder_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Where the scene features live: backbone ``output`` against proposal ``requires``."""
    backbone, proposal = stages.get("backbone"), stages.get("proposal")
    if backbone is None or proposal is None:
        return
    produced, required = backbone.contract("output"), proposal.contract("requires", "any")
    if required == "any" or produced == required:
        return
    message = (
        f"{proposal.name} requires '{required}' scene features, but {backbone.name} returns "
        f"'{produced}' ones"
    )
    if produced == "subsampled":
        message += (
            " (features on a sub-sampled point set). Wrap the backbone in the explicit "
            "adapter `{type: FeaturePropagation, backbone: {...}}` or choose a proposal "
            "that works on sub-sampled features."
        )
    elif produced == "dense":
        candidates = _registered(PROPOSALS, "requires", lambda value: value in ("dense", "any"))
        message += (
            " (one feature per scene point). Choose a proposal that selects seeds from dense "
            f"features ({candidates}), or wrap the backbone in the explicit adapter "
            "`{type: FurthestPointSubsample, backbone: {...}}`, which keeps the features of "
            "a furthest-point subsample."
        )
    else:
        yield Refusal(message + ".", "structural")
        return
    yield message


@_rule("fixed-by-compose")
def _axes_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Named feature axes: grouper ``out_axes`` against head ``in_axes``.

    An axis is never added or dropped implicitly. Where a registered axis adapter bridges
    the two (repeat a crop over the depth bins, pool over them), the refusal names it and
    ``grasppanda compose`` inserts it. A grouper that is itself an axis adapter under the
    head of the other family has no adapter: that pair contradicts itself.
    """
    grouper, head = stages.get("grouper"), stages.get("head")
    if grouper is not None and head is not None:
        emitted, expected = grouper.contract("out_axes"), head.contract("in_axes")
        if emitted != expected:
            first = f"{grouper.name} emits features with axes {emitted}, but {head.name} expects {expected}."
            fitting = (
                f" Groupers that emit {expected}: "
                f"{_registered(GROUPERS, 'out_axes', lambda value: value == expected)}. Head "
                f"bundles that expect {emitted}: "
                f"{_registered(HEADS, 'in_axes', lambda value: value == emitted)}."
            )
            support = grouper.contract("support")
            adapters = axis_adapters(emitted, expected, support)
            if grouper.contract("inner_axes") is not None:
                yield Refusal(
                    f"{first} {grouper.name} is itself an axis adapter, made to change the axes "
                    f"of the crop it wraps to {emitted}: under {head.name} use that crop "
                    f"without the adapter, or another grouper.{fitting}",
                    "structural",
                )
            elif adapters:
                adapter = adapters[0]
                options = ", num_depth: <the head's num_depth>" if "depth" in expected else ""
                yield (
                    f"{first} An axis is never added or squeezed implicitly: wrap the grouper "
                    f"in the axis adapter {adapter}{_adapter_preset(adapter)}, "
                    f"`{{type: {adapter}{options}, grouper: {{...}}}}`, which `grasppanda "
                    "compose` inserts, or use a grouper and a head bundle that agree."
                    f"{fitting}"
                )
            else:
                wrappers = [
                    f"{name} wraps {GROUPERS.get(name).support}-support crops"
                    for name in GROUPERS.names()
                    if getattr(GROUPERS.get(name), "inner_axes", None) == emitted
                    and GROUPERS.get(name).out_axes == expected
                ]
                said = f" ({'; '.join(wrappers)})" if wrappers else ""
                yield Refusal(
                    f"{first} No axis adapter turns a {support}-support crop with axes "
                    f"{emitted} into {expected}{said}: such an adapter is not implemented. Use "
                    f"a grouper and a head bundle that agree.{fitting}",
                    "not-implemented",
                )
    if grouper is not None:
        support = grouper.contract("support")
        if support not in ("seeds", "scene"):
            yield Refusal(
                f"{grouper.name} declares support = {support!r}; it must be 'seeds' or 'scene'.",
                "structural",
            )


@_rule("structural")
def _label_format_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """The kind of matched labels: assigner against head bundle and view stage.

    Dense grids (every grasp of a point and view) and compact labels (one) are different
    supervision; a dense grid cannot be recovered from compact labels."""
    assigner, head, view = stages.get("assigner"), stages.get("head"), stages.get("view")
    if assigner is None:
        return
    produced = assigner.contract("label_format", "dense")
    if head is not None:
        consumed = head.contract("label_format", "dense")
        if produced != consumed:
            yield (
                f"{assigner.name} returns '{produced}' matched labels, but {head.name} is "
                f"supervised with '{consumed}' ones. A head bundle and the assigner that feeds it "
                "must agree on the label set; the dataset must serve that label set too."
            )
    if view is not None:
        accepted = view.contract("label_formats", ("dense",))
        if produced not in accepted:
            able = _registered(VIEWS, "label_formats", lambda value: produced in (value or ()))
            yield (
                f"{view.name} computes its targets from {' or '.join(map(repr, accepted))} "
                f"matched labels, but {assigner.name} returns '{produced}' ones. View stages "
                f"that read '{produced}' labels: {able}."
            )


@_rule("fixed-by-compose")
def _matched_fields_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Optional fields of the matched labels: what a stage reads against what the
    assigner returns. A field behind an option of the assigner is ``fixed-by-compose``
    (compose sets the option); a field the assigner never returns is ``structural``."""
    assigner = stages.get("assigner")
    if assigner is None:
        return
    outputs = assigner.contract("optional_outputs", {})
    produced = assigner.contract("label_format", "dense")
    for slot in ("view", "head"):
        stage = stages.get(slot)
        if stage is None:
            continue
        readable = stage.contract("label_formats") or (stage.contract("label_format", "dense"),)
        if produced not in readable:
            continue  # refused by the label-format rule; its fields are beside the point
        missing, switches = [], []
        for field in stage.contract("needs_matched", ()):
            if field not in outputs:
                missing.append(field)
            elif outputs[field] is not None:
                option, value = outputs[field]
                actual = assigner.option(option)
                if actual is not _UNKNOWN and actual != value:
                    switches.append((field, option, value))
        if missing:
            able = _registered(
                ASSIGNERS,
                "optional_outputs",
                lambda value, missing=missing: all(field in (value or {}) for field in missing),
            )
            yield Refusal(
                f"{stage.name} reads {_fields(missing)} of the matched labels, which "
                f"{assigner.name} never returns. Assigners that can: {able}.",
                "structural",
            )
        if switches:
            fix = " and ".join(f"`{option}: {_yaml(value)}`" for _, option, value in switches)
            yield (
                f"{stage.name} reads {_fields([field for field, _, _ in switches])} of the "
                f"matched labels, which {assigner.name} returns only with {fix}. Set "
                f"{'that option' if len(switches) == 1 else 'those options'} on the assigner "
                "(`grasppanda compose` does that)."
            )


def _seed_weights_warning(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """A per-seed loss weight next to a stage that does not apply one: that stage trains
    unweighted. Not a refusal: the model is sound, the weighting is partial."""
    assigner = stages.get("assigner")
    if assigner is None:
        return
    outputs = assigner.contract("optional_outputs", {})
    if "weight" not in outputs:
        return
    switch = outputs["weight"]
    if switch is not None and assigner.option(switch[0]) not in (_UNKNOWN, switch[1]):
        return
    produced = assigner.contract("label_format", "dense")
    ignoring = []
    for slot in ("view", "head"):
        stage = stages.get(slot)
        if stage is None or stage.contract("applies_seed_weights", False):
            continue
        readable = stage.contract("label_formats") or (stage.contract("label_format", "dense"),)
        if produced in readable:  # otherwise refused by the label-format rule
            ignoring.append(stage.name)
    if ignoring:
        yield (
            f"{assigner.name} returns a loss weight per seed, but {' and '.join(ignoring)} "
            f"{'does' if len(ignoring) == 1 else 'do'} not apply seed weights: "
            f"{'its loss is' if len(ignoring) == 1 else 'their losses are'} unweighted, so the "
            "model is weighted only where the other stages apply the weight. Stages that apply "
            f"it: view stages {_registered(VIEWS, 'applies_seed_weights', bool)}; head bundles "
            f"{_registered(HEADS, 'applies_seed_weights', bool)}."
        )


@_rule("fixed-by-compose")
def _seed_scores_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Scores on the seeds: one owner per score, and every score the decoder reads.

    Where the stage that would give up the score has an option to do so (a proposal that
    predicts graspness only, a decode option of the head that does not read it), the
    refusal names it and compose sets it; where it has none, the refusal is
    ``structural``."""
    proposal, view, head = stages.get("proposal"), stages.get("view"), stages.get("head")
    predicted: dict[str, _Stage] = {}
    for stage in (proposal, view):
        if stage is None:
            continue
        for score in _seed_scores(stage):
            if score not in predicted:
                predicted[score] = stage
                continue
            owner = predicted[score]

            def without(value: Any, score: str = score) -> bool:
                return score not in (value or ())

            first = (
                f"{owner.name} and {stage.name} both predict and supervise the '{score}' of a "
                "seed; a detector needs exactly one owner for it."
            )
            others = (
                f"{owner.name} with a view stage that does not predict it "
                f"({_registered(VIEWS, 'seed_scores', without)}), or keep {stage.name} with a "
                f"proposal that does not ({_registered(PROPOSALS, 'seed_scores', without)})."
            )
            leave = _without_score(owner, score)
            if isinstance(leave, Mapping):
                yield (
                    f"{first} Let {stage.name} own it and {owner.name} leave it: "
                    f"{_switch(owner, leave)} (`grasppanda compose` does that); or keep {others}"
                )
            elif isinstance(leave, str):
                yield Refusal(f"{first} {leave} Keep {others}", "structural")
            else:
                yield Refusal(f"{first} Keep {others}", "structural")
    if head is None or proposal is None or view is None:
        return
    reader = getattr(head.cls, "required_seed_scores", None)
    absent = [score for score in (reader(head.options) if reader else ()) if score not in predicted]
    if absent:

        def predicts(value: Any) -> bool:
            return all(score in (value or ()) for score in absent)

        first = (
            f"{head.name} as configured reads the seed score {_fields(absent)} when it decodes, "
            f"but neither {proposal.name} nor {view.name} predicts it."
        )
        stages_that_do = (
            f"use a stage that does (proposals: {_registered(PROPOSALS, 'seed_scores', predicts)}; "
            f"view stages: {_registered(VIEWS, 'seed_scores', predicts)})"
        )
        decode = _without_score(head, absent[0]) if len(absent) == 1 else None
        if isinstance(decode, Mapping):
            yield (
                f"{first} Set {_switch(head, decode)}, the decode option of the head that does "
                f"not read it (`grasppanda compose` does that), or {stages_that_do}."
            )
        else:
            yield Refusal(
                f"{first} {stages_that_do[0].upper()}{stages_that_do[1:]}; the head has no decode "
                "option without it.",
                "structural",
            )


@functools.lru_cache(maxsize=64)
def _built_width(node: str) -> int | None:
    """Output width of an encoder node (JSON text), found by building it once.

    Released weights are left out of that build: the width belongs to the architecture,
    and the verdict must not depend on which files are on the machine that checks.
    """

    def bare(value: Any) -> Any:  # an adapter holds the encoder as a nested node
        if isinstance(value, dict):
            return {k: None if k == "pretrained" else bare(v) for k, v in value.items()}
        return value

    options = json.loads(node)
    for attempt in (bare(options), options):
        try:
            return int(BACKBONES.build(attempt).out_channels)
        except Exception:  # an optional dependency is missing, or it cannot be built alone
            continue
    return None


def _encoder_width(backbone: _Stage) -> int | None:
    """Channels of the features an encoder returns, or ``None`` when that cannot be told.

    A built encoder says it. A config node says it through what its class declares
    (``Backbone.output_channels``); where that is "known only once built", the node is
    built once, without released weights.
    """
    if backbone.built is not None:
        width = getattr(backbone.built, "out_channels", None)
        return int(width) if isinstance(width, int) else None
    if backbone.node is None:
        return None
    declared = configured_width(backbone.node)
    if declared is not None:
        return declared
    try:
        node = json.dumps(dict(backbone.node), sort_keys=True)
    except TypeError:
        return None
    return _built_width(node)


@_rule("not-a-combination")
def _seed_width_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Width of the seed features: what the encoder returns against what a grouper that
    adds them to its crops was configured for (``Grouper.required_seed_channels``)."""
    backbone, grouper = stages.get("backbone"), stages.get("grouper")
    if backbone is None or grouper is None:
        return
    reader = getattr(grouper.cls, "required_seed_channels", None)
    required = reader(grouper.options) if reader is not None else None
    if required is None:
        return
    width = _encoder_width(backbone)
    if width is None or width == required:
        return
    variant = getattr(grouper.cls, "projected_variant", None)
    projection = (
        f"Use `type: {variant}` for the grouper (the same crop with a linear layer in front of "
        "the residual; `grasppanda compose` does that), or "
        if variant
        else "Either "
    )
    message = (
        f"{grouper.name} as configured adds the seed features to its {required}-channel crop "
        f"features, but {backbone.name} returns {width}-channel features. {projection}give the "
        f"encoder {required} output channels where it has that option, or end the grouper's "
        f"`mlp_channels` on {width}, or set `seed_residual: none`."
    )
    # With a projected variant compose puts it in; without, two options disagree.
    yield Refusal(message, "fixed-by-compose") if variant else message


@_rule("not-a-combination")
def _depth_bins_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Size of the depth axis: one crop of the grouper per depth bin of the head."""
    grouper, head = stages.get("grouper"), stages.get("head")
    if grouper is None or head is None or "depth" not in (grouper.contract("out_axes") or ()):
        return
    reach, bins = grouper.option("hmax"), head.option("num_depth")
    if bins is _UNKNOWN:  # a built head keeps its grid in the codec
        bins = getattr(getattr(head.built, "codec", None), "num_depth", _UNKNOWN)
    stated = grouper.option("num_depth")  # a grouper whose depth axis is a plain number
    if not isinstance(reach, (list, tuple)) and isinstance(stated, int) and isinstance(bins, int):
        # Under a head without a depth axis the axes rule has refused the pair already.
        if stated != bins and "depth" in (head.contract("in_axes") or ()):
            yield (
                f"{grouper.name} has num_depth = {stated}, but the head bundle has num_depth = "
                f"{bins}. Set the grouper's `num_depth` to {bins}."
            )
        return
    if not (isinstance(reach, (list, tuple)) and isinstance(bins, int)) or len(reach) == bins:
        return
    if grouper.built is None:
        yield (
            f"the grouper lists {len(reach)} far end(s) (hmax), but the head bundle has "
            f"num_depth = {bins}. The grouper needs one crop per depth bin of the head."
        )
    else:
        yield (
            f"{grouper.name} makes {len(reach)} crop(s) per seed, but {head.name} is configured "
            f"for num_depth = {bins}. The grouper needs one far end (hmax) per depth bin of the "
            "head."
        )


@_rule("not-a-combination")
def _shared_score_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """One score definition for two stages: a view stage and a head bundle that both
    derive their targets from :mod:`grasppanda.models.heads.friction` must agree on it."""
    view, head = stages.get("view"), stages.get("head")
    if view is None or head is None:
        return
    if not all(stage.option("score_target") in SCORE_TARGETS for stage in (view, head)):
        return
    for option in ("score_target", "max_friction"):
        left, right = view.option(option), head.option(option)
        if _UNKNOWN not in (left, right) and left != right:
            yield (
                f"the view stage has {option} = {left!r} and the head bundle has {option} = "
                f"{right!r}. The view target is the best grasp score of a view, so both must "
                "encode scores the same way."
            )


@_rule("not-a-combination")
def _angle_bins_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Size of the in-plane angle axis: a view stage that turns the labelled angle bin
    into a rotation target against the angle grid of the head bundle."""
    view, head = stages.get("view"), stages.get("head")
    if view is None or head is None:
        return
    mine, bins = view.option("num_angle"), head.option("num_angle")
    if bins is _UNKNOWN:  # a built head keeps its grid in the codec
        bins = getattr(getattr(head.built, "codec", None), "num_angle", _UNKNOWN)
    if isinstance(mine, int) and isinstance(bins, int) and mine != bins:
        yield (
            f"{view.name} has num_angle = {mine} and {head.name} has num_angle = {bins}. The "
            "view stage reads the labelled in-plane angle bin as a rotation, so both must use "
            "the angle grid of the label set."
        )


@_rule("not-a-combination")
def _seed_count_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Number of seeds: a proposal that takes the encoder's sub-sampled points as its
    seeds (``seeds_from_backbone``) against the points the encoder returns per cloud."""
    backbone, proposal = stages.get("backbone"), stages.get("proposal")
    if backbone is None or proposal is None or not proposal.contract("seeds_from_backbone"):
        return
    wanted = proposal.option("num_seeds")
    declare = backbone.contract("output_points")
    if not isinstance(wanted, int) or not callable(declare):
        return
    try:
        returned = declare(backbone.options)
    except (LookupError, TypeError, ValueError):
        return
    if isinstance(returned, int) and returned != wanted:
        yield (
            f"{proposal.name} is configured for num_seeds = {wanted}, but {backbone.name} as "
            f"configured returns {returned} points per cloud, and those points are the "
            f"seeds. Set the proposal's `num_seeds` to {returned}, or change the encoder's "
            "output level (`npoints`, `npoint` or `num_group`) to match."
        )


@_rule("not-a-combination")
def _option_range_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Option values a stage can judge without being built: its ``option_problems``."""
    for stage in stages.values():
        declare = stage.contract("option_problems")
        if stage.built is not None or not callable(declare):
            continue
        try:
            yield from declare(stage.options)
        except (LookupError, TypeError, ValueError):
            continue  # options the signature check will report


@_rule("structural")
def _loss_terms_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Loss terms: the kind of term a stage hands to a primitive against the kinds the
    primitive fits, and a weight per seed against a primitive that cannot apply one.

    A stage that applies seed weights passes them to every loss term it has, unless it
    names the terms that receive them in ``seed_weighted_terms``."""
    assigner = stages.get("assigner")
    weighted = False
    if assigner is not None:
        outputs = assigner.contract("optional_outputs", {})
        switch = outputs.get("weight", _UNKNOWN)
        weighted = switch is None or (
            switch is not _UNKNOWN and assigner.option(switch[0]) in (_UNKNOWN, switch[1])
        )
    for slot in ("proposal", "view", "head"):
        stage = stages.get(slot)
        if stage is None:
            continue
        reader = getattr(stage.cls, "loss_terms_for", None)
        terms = reader(stage.options) if reader else stage.contract("loss_terms", {})
        for option, kind in (terms or {}).items():
            loss = stage.option(option)
            if stage.built is not None and not isinstance(loss, Loss):
                loss = getattr(stage.built, f"{option}_fn", None)  # `loss` is also a method
            try:
                found = accepted_terms(loss)
            except RegistryError:
                continue  # an unknown loss: the config check names it
            if found is None:
                continue  # left to the stage: it builds the loss of its reference
            cls, accepted = found
            where = f"{stage.name}.{option}"
            if kind not in accepted:
                fits = " or ".join(f"`{name}`" for name in accepted) or "no"
                yield (
                    f"{where} is a `{kind}` term ({TERM_KINDS.get(kind, 'undescribed')}), but "
                    f"{cls.__name__} fits {fits} terms. Losses for a `{kind}` term: "
                    f"{', '.join(fitting_losses(kind)) or 'none registered'}."
                )
            elif (
                weighted
                and slot != "proposal"
                and stage.contract("applies_seed_weights", False)
                and option in stage.contract("seed_weighted_terms", tuple(terms))
                and not applies_weights(loss)
            ):
                able = [
                    name
                    for name in fitting_losses(kind)
                    if getattr(LOSSES.get(name), "applies_weights", True)
                ]
                yield (
                    f"{assigner.name} returns a loss weight per seed and {stage.name} passes it "
                    f"to {option}, but {cls.__name__} scores the supervised set as a whole and "
                    f"cannot weight single elements. Losses for a `{kind}` term that apply "
                    f"weights: {', '.join(able)}."
                )


#: The contract rules, each with the kind of what it refuses (``rule.kind``, one of
#: :data:`~grasppanda.config.errors.REFUSAL_KINDS`; a rule whose verdict depends on the
#: case states the kind of each message).
CONTRACT_RULES: tuple[Rule, ...] = (
    _encoder_rule,
    _axes_rule,
    _label_format_rule,
    _matched_fields_rule,
    _seed_scores_rule,
    _seed_width_rule,
    _depth_bins_rule,
    _shared_score_rule,
    _angle_bins_rule,
    _seed_count_rule,
    _option_range_rule,
    _loss_terms_rule,
)
_RULES = CONTRACT_RULES
#: Findings that do not refuse a combination but change what it trains.
_WARNINGS: tuple[Rule, ...] = (_seed_weights_warning,)


def _wrapped(stages: Mapping[str, Any]) -> dict[str, _Stage]:
    unknown = sorted(set(stages) - set(STAGE_REGISTRIES))
    if unknown:
        raise ValueError(f"unknown slot(s) {unknown}; slots are {sorted(STAGE_REGISTRIES)}")
    return {slot: _Stage(slot, component) for slot, component in stages.items()}


def stage_contract_warnings(**stages: Any) -> list[str]:
    """What the given stages would train differently from what their names promise,
    without being refused: a per-seed loss weight that one of them does not apply.

    Arguments as for :func:`stage_contract_problems`. ``grasppanda compose`` reports these
    beside its actions, and the detector warns when it is built.
    """
    wrapped = _wrapped(stages)
    return [message for rule in _WARNINGS for message in rule(wrapped)]


def stage_contract_problems(**stages: Any) -> list[str]:
    """Contract problems among the given stages of a seed-based detector.

    Every argument is named after a slot (``backbone``, ``proposal``, ``view``,
    ``assigner``, ``grouper``, ``head``) and is a config node ``{type: ...}``, a
    component class or a built component. Rules whose stages are not all given are
    skipped, so two stages can be compared on their own::

        stage_contract_problems(grouper={"type": "CylinderGroup"}, head={"type": "OperationHead"})

    Returns one message per problem, each naming the conflict and a way out; an empty
    list means the contracts allow the combination. Every message is a
    :class:`~grasppanda.config.errors.Refusal`: a string that also carries ``kind`` (why it
    is refused) and ``rule``. This is what the config checker runs
    (:meth:`SeedGraspDetector.check_config`), what the detector runs again on its built
    stages, and what the compatibility tables of the documentation are generated from.
    """
    wrapped = _wrapped(stages)
    found: list[str] = []
    for rule in CONTRACT_RULES:
        for message in rule(wrapped):
            kind = message.kind if isinstance(message, Refusal) else rule.kind  # type: ignore[attr-defined]
            found.append(Refusal(str(message), kind, rule.__name__.strip("_")))
    return found


class SeedGraspDetector(BaseGraspDetector):
    """backbone → proposal → view → (assigner) → grouper → head bundle.

    Children are built here, so each receives what its parent already knows (channel
    counts, the view template, the gripper) instead of repeating it in the config.

    What the skeleton calls on its stages, i.e. what a new stage has to provide:

    * ``backbone(points, feats) -> PointFeatures``; attributes ``in_channels``,
      ``out_channels`` and the contract ``output``. A backbone with
      ``uses_observation = True`` is called as ``backbone(points, feats, observation)``
      and lists the observation fields it reads in ``required_inputs``.
    * ``proposal(feats, observation) -> (SeedSet, scores)`` and
      ``proposal.loss(scores, supervision) -> dict``. Every frame yields the same
      number of seeds; padding is marked in ``SeedSet.valid``.
    * ``view(seeds) -> (ApproachSet, SeedSet)`` with a frame per seed in
      ``ApproachSet.rotation``, ``view.loss(approach, matched) -> dict`` and the
      attribute ``template``.
    * ``assigner(seeds, approach, supervision)`` returning an object with ``valid``,
      ``crop_centre`` and ``crop_rotation`` plus whatever the view stage and the head
      bundle of the method read from it.
    * ``grouper(support, centres, rotation) -> GroupFeatures``; ``support`` is the seed
      features or the scene features according to the contract ``grouper.support``.
      Under a backbone that returns sub-sampled features the scene has none: a
      scene-support grouper then receives the scene points with zero feature channels.
      A grouper with ``uses_seed_features`` is called with the seed features as well:
      ``grouper(support, centres, rotation, seeds.feats)``.
    * ``head(groups) -> raw``, ``head.loss(raw, matched) -> dict`` and
      ``head.decode(raw, seeds, approach) -> GraspGroup`` with one row per seed.

    Args:
        backbone: Scene encoder.
        proposal: Turns scene features into seeds.
        view: Chooses one approach per seed.
        grouper: Gathers a crop per seed.
        head: Head bundle that predicts, encodes targets, computes its loss and decodes.
        assigner: Matches seeds to grasp labels during training.
        gripper: Gripper the predictions refer to; ``None`` is the benchmark gripper.
            It is the single owner of the maximum width and the finger height.
        point_features: Input features of the backbone. ``ones`` feeds a constant 1 in
            every channel (the Graspness reference: geometry only); ``observation``
            feeds ``observation.point_feats``.
    """

    def __init__(
        self,
        backbone: Cfg[Backbone],
        proposal: Cfg[Proposal],
        view: Cfg[ViewStage],
        grouper: Cfg[Grouper],
        head: Cfg[HeadBundle],
        assigner: Cfg[Assigner],
        gripper: Gripper | None = None,
        point_features: Literal["ones", "observation"] = "ones",
    ) -> None:
        super().__init__()
        if point_features not in ("ones", "observation"):
            raise ValueError(f"unknown point_features '{point_features}'")
        if isinstance(gripper, Mapping):  # a config node arrives as a plain mapping
            gripper = Gripper(**gripper)
        self.gripper = gripper if gripper is not None else Gripper()
        self.point_features = point_features
        self.required_inputs = frozenset(
            {"points"} | ({"point_feats"} if point_features == "observation" else set())
        )
        self.backbone = BACKBONES.build(backbone)
        # A scene encoder that reads more than the cloud (point painting reads the colour
        # image) says so; it is then handed the observation as a third argument.
        self.required_inputs |= frozenset(getattr(self.backbone, "required_inputs", ()))
        channels = self.backbone.out_channels
        self.proposal = PROPOSALS.build(proposal, in_channels=channels)
        self.view = VIEWS.build(view, in_channels=channels, gripper=self.gripper)
        self.grouper = GROUPERS.build(grouper, in_channels=channels)
        self.head = HEADS.build(head, in_channels=self.grouper.out_channels, gripper=self.gripper)
        self.assigner = ASSIGNERS.build(assigner, template=self.view.template, gripper=self.gripper)
        problems = stage_contract_problems(
            backbone=self.backbone,
            proposal=self.proposal,
            view=self.view,
            assigner=self.assigner,
            grouper=self.grouper,
            head=self.head,
        )
        if problems:
            raise ContractError(f"{type(self).__name__}: " + " ".join(problems))
        for message in stage_contract_warnings(
            view=self.view, assigner=self.assigner, head=self.head
        ):
            warnings.warn(f"{type(self).__name__}: {message}", UserWarning, stacklevel=2)

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Compare the contracts of the configured stages before anything is built."""
        stages = {}
        for slot in STAGE_REGISTRIES:
            child = node.get(slot)
            if not (isinstance(child, Mapping) and "type" in child):
                return
            stages[slot] = child
        yield from stage_contract_problems(**stages)

    @classmethod
    def supervision_needs(cls, node: Mapping[str, Any]) -> list[tuple[str, str]] | None:
        """What the configured stages read from the training supervision, with the stage
        that reads each field (``needs_supervision`` of every stage, with its options)."""
        needs = []
        for slot in STAGE_REGISTRIES:
            child = node.get(slot)
            if not (isinstance(child, Mapping) and child.get("type") in STAGE_REGISTRIES[slot]):
                return None  # a stage is missing or unknown: the signature check says so
            stage = _Stage(slot, child)
            for field, switch in stage.contract("needs_supervision", {}).items():
                if switch is None or stage.option(switch[0]) == switch[1]:
                    needs.append((field, stage.name))
        return needs

    # ---------------------------------------------------------------------- stages

    def _input_features(self, observation: Observation) -> Tensor:
        points = observation.points
        if self.point_features == "ones":
            return points.coords.new_ones(len(points), self.backbone.in_channels)
        if observation.point_feats is None:
            raise ContractError(
                f"{type(self).__name__} is configured with point_features: observation, but "
                "observation.point_feats is missing"
            )
        return observation.point_feats

    def run_stages(
        self,
        observation: Observation,
        supervision: Supervision | None = None,
        *,
        seed_index: Tensor | None = None,
        view_index: Tensor | None = None,
    ) -> StageOutputs:
        """Run the stage flow once and return every intermediate result.

        :meth:`loss` and :meth:`predict` are thin wrappers around this method. With
        ``supervision`` the assigner runs and the crop uses its centre and frame;
        without it the crop is centred at the seed and uses the template frame of the
        chosen view.

        ``seed_index`` and ``view_index`` replace the seed selection and the view
        choice by recorded ones. They exist to replay a run whose seeds or sampled
        views are known, which is how the stages after a threshold or a random draw are
        compared with a reference implementation.
        """
        name = type(self).__name__
        if observation.points is None:
            raise ContractError(f"{name} needs observation.points")
        if len(observation.keys) != observation.points.batch_size:
            raise ContractError(
                f"{name}: the observation lists {len(observation.keys)} sample key(s) for a "
                f"batch of {observation.points.batch_size} point cloud(s)"
            )
        reads = (observation,) if getattr(self.backbone, "uses_observation", False) else ()
        feats = self.backbone(observation.points, self._input_features(observation), *reads)
        replay = {} if seed_index is None else {"seed_index": seed_index}
        seeds, point_scores = self.proposal(feats, observation, **replay)
        replay = {} if view_index is None else {"view_index": view_index}
        approach, seeds = self.view(seeds, **replay)
        if approach.rotation is None:
            raise ContractError(f"{type(self.view).__name__} returned an approach without frames")
        matched = None
        centres, rotation = seeds.points.coords, approach.rotation
        if supervision is not None:
            matched = self.assigner(seeds, approach, supervision)
            centres, rotation = matched.crop_centre, matched.crop_rotation
        if self.grouper.support == "seeds":
            support = PointFeatures(seeds.feats, seeds.points)
        elif feats.points.id == observation.points.id:
            support = feats
        else:  # features on a subsample: the scene itself is geometry only
            scene = observation.points
            support = PointFeatures(scene.coords.new_zeros(len(scene), 0), scene)
        if getattr(self.grouper, "uses_seed_features", False):
            groups = self.grouper(support, centres, rotation, seeds.feats)
        else:
            groups = self.grouper(support, centres, rotation)
        raw = self.head(groups)
        return StageOutputs(feats, point_scores, seeds, approach, matched, groups, raw)

    # ------------------------------------------------------------------------ loss

    def loss(self, observation: Observation, supervision: Supervision) -> dict[str, Tensor]:
        """Stage losses of one training step (``loss/…``) plus a few logged metrics."""
        out = self.run_stages(observation, supervision)
        losses: dict[str, Tensor] = {}
        for stage, part in (
            (self.proposal, self.proposal.loss(out.point_scores, supervision)),
            (self.view, self.view.loss(out.approach, out.matched)),
            (self.head, self.head.loss(out.raw, out.matched)),
        ):
            clash = losses.keys() & part.keys()
            if clash:
                raise ContractError(
                    f"{type(stage).__name__} reports {sorted(clash)}, which another stage of "
                    f"{type(self).__name__} already reported"
                )
            losses.update(part)
        if out.seeds.valid is not None:
            losses["metric/real_seeds"] = out.seeds.valid.float().mean()
        losses["metric/supervised_seeds"] = out.matched.valid.float().mean()
        return losses

    # --------------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, observation: Observation) -> list[GraspGroup]:
        """One :class:`GraspGroup` per frame, in the frame's seed order.

        Padded or otherwise invalid seeds are dropped, so a frame may yield fewer
        grasps than seeds, down to an empty group; a zero-coordinate placeholder is
        never emitted. Call ``model.eval()`` first: view selection and normalisation
        layers follow ``nn.Module.training``.
        """
        return self.decode(self.run_stages(observation), observation)

    @torch.no_grad()
    def decode(self, out: StageOutputs, observation: Observation) -> list[GraspGroup]:
        """Turn the stage outputs of a forward pass into per-frame grasps."""
        grasps = self.head.decode(out.raw, out.seeds, out.approach)
        points = out.seeds.points
        offsets = (
            [0, len(points)] if points.batch_offsets is None else points.batch_offsets.tolist()
        )
        results = []
        for frame, key in enumerate(observation.keys):
            rows = torch.arange(offsets[frame], offsets[frame + 1], device=points.coords.device)
            if out.seeds.valid is not None:
                rows = rows[out.seeds.valid[rows]]
            group = grasps[rows]
            group.key = key
            results.append(group.validate())
        return results
