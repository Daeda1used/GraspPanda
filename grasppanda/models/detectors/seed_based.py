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

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from grasppanda.models.base import (
    Assigner,
    Backbone,
    BaseGraspDetector,
    Grouper,
    HeadBundle,
    Proposal,
    ViewStage,
)
from grasppanda.models.heads.friction import SCORE_TARGETS
from grasppanda.registry import (
    ASSIGNERS,
    BACKBONES,
    GROUPERS,
    HEADS,
    INJECTED,
    PROPOSALS,
    VIEWS,
    Cfg,
    Registry,
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

__all__ = ["STAGE_REGISTRIES", "SeedGraspDetector", "StageOutputs", "stage_contract_problems"]


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
        if isinstance(component, Mapping):
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
        message += "."
    yield message


def _axes_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Named feature axes: grouper ``out_axes`` against head ``in_axes``."""
    grouper, head = stages.get("grouper"), stages.get("head")
    if grouper is not None and head is not None:
        emitted, expected = grouper.contract("out_axes"), head.contract("in_axes")
        if emitted != expected:
            yield (
                f"{grouper.name} emits features with axes {emitted}, but {head.name} expects "
                f"{expected}. Use a grouper and a head bundle that agree; an axis is never added "
                f"or squeezed implicitly. Groupers that emit {expected}: "
                f"{_registered(GROUPERS, 'out_axes', lambda value: value == expected)}. Head "
                f"bundles that expect {emitted}: "
                f"{_registered(HEADS, 'in_axes', lambda value: value == emitted)}."
            )
    if grouper is not None:
        support = grouper.contract("support")
        if support not in ("seeds", "scene"):
            yield f"{grouper.name} declares support = {support!r}; it must be 'seeds' or 'scene'."


def _label_format_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """The kind of matched labels: assigner against head bundle and view stage."""
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


def _matched_fields_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Optional fields of the matched labels: what a stage reads against what the
    assigner returns, including the per-seed loss weight."""
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
            yield (
                f"{stage.name} reads {_fields(missing)} of the matched labels, which "
                f"{assigner.name} never returns. Assigners that can: {able}."
            )
        if switches:
            fix = " and ".join(f"`{option}: {_yaml(value)}`" for _, option, value in switches)
            yield (
                f"{stage.name} reads {_fields([field for field, _, _ in switches])} of the "
                f"matched labels, which {assigner.name} returns only with {fix}. Set "
                f"{'that option' if len(switches) == 1 else 'those options'} on the assigner."
            )
    if "weight" not in outputs:
        return
    switch = outputs["weight"]
    if switch is not None and assigner.option(switch[0]) not in (_UNKNOWN, switch[1]):
        return
    ignoring = [
        stages[slot].name
        for slot in ("view", "head")
        if slot in stages and not stages[slot].contract("applies_seed_weights", False)
    ]
    if ignoring:
        yield (
            f"{assigner.name} returns a loss weight per seed, but {' and '.join(ignoring)} "
            f"{'does' if len(ignoring) == 1 else 'do'} not apply seed weights, so the weighting "
            "would silently do nothing there. Use stages that apply it (view stages: "
            f"{_registered(VIEWS, 'applies_seed_weights', bool)}; head bundles: "
            f"{_registered(HEADS, 'applies_seed_weights', bool)}) or an assigner without weights."
        )


def _seed_scores_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Scores on the seeds: one owner per score, and every score the decoder reads."""
    proposal, view, head = stages.get("proposal"), stages.get("view"), stages.get("head")
    predicted: dict[str, str] = {}
    for stage in (proposal, view):
        if stage is None:
            continue
        for score in stage.contract("seed_scores", ()):
            if score not in predicted:
                predicted[score] = stage.name
                continue

            def without(value: Any, score: str = score) -> bool:
                return score not in (value or ())

            yield (
                f"{predicted[score]} and {stage.name} both predict and supervise the '{score}' "
                "of a seed; a detector needs exactly one owner for it. Keep "
                f"{predicted[score]} with a view stage that does not predict it "
                f"({_registered(VIEWS, 'seed_scores', without)}), or keep {stage.name} with a "
                f"proposal that does not ({_registered(PROPOSALS, 'seed_scores', without)})."
            )
    if head is None or proposal is None or view is None:
        return
    reader = getattr(head.cls, "required_seed_scores", None)
    absent = [score for score in (reader(head.options) if reader else ()) if score not in predicted]
    if absent:

        def predicts(value: Any) -> bool:
            return all(score in (value or ()) for score in absent)

        yield (
            f"{head.name} as configured reads the seed score {_fields(absent)} when it decodes, "
            f"but neither {proposal.name} nor {view.name} predicts it. Use a stage that does "
            f"(proposals: {_registered(PROPOSALS, 'seed_scores', predicts)}; view stages: "
            f"{_registered(VIEWS, 'seed_scores', predicts)}) or another decode option of the head."
        )


def _depth_bins_rule(stages: Mapping[str, _Stage]) -> Iterator[str]:
    """Size of the depth axis: one crop of the grouper per depth bin of the head."""
    grouper, head = stages.get("grouper"), stages.get("head")
    if grouper is None or head is None or "depth" not in (grouper.contract("out_axes") or ()):
        return
    reach, bins = grouper.option("hmax"), head.option("num_depth")
    if bins is _UNKNOWN:  # a built head keeps its grid in the codec
        bins = getattr(getattr(head.built, "codec", None), "num_depth", _UNKNOWN)
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


_RULES: tuple[Callable[[Mapping[str, _Stage]], Iterator[str]], ...] = (
    _encoder_rule,
    _axes_rule,
    _label_format_rule,
    _matched_fields_rule,
    _seed_scores_rule,
    _depth_bins_rule,
    _shared_score_rule,
)


def stage_contract_problems(**stages: Any) -> list[str]:
    """Contract problems among the given stages of a seed-based detector.

    Every argument is named after a slot (``backbone``, ``proposal``, ``view``,
    ``assigner``, ``grouper``, ``head``) and is a config node ``{type: ...}``, a
    component class or a built component. Rules whose stages are not all given are
    skipped, so two stages can be compared on their own::

        stage_contract_problems(grouper={"type": "CylinderGroup"}, head={"type": "OperationHead"})

    Returns one message per problem, each naming the conflict and a way out; an empty
    list means the contracts allow the combination. This is what the config checker
    runs (:meth:`SeedGraspDetector.check_config`), what the detector runs again on its
    built stages, and what the compatibility tables of the documentation are generated
    from.
    """
    unknown = sorted(set(stages) - set(STAGE_REGISTRIES))
    if unknown:
        raise ValueError(f"unknown slot(s) {unknown}; slots are {sorted(STAGE_REGISTRIES)}")
    wrapped = {slot: _Stage(slot, component) for slot, component in stages.items()}
    return [message for rule in _RULES for message in rule(wrapped)]


class SeedGraspDetector(BaseGraspDetector):
    """backbone → proposal → view → (assigner) → grouper → head bundle.

    Children are built here, so each receives what its parent already knows (channel
    counts, the view template, the gripper) instead of repeating it in the config.

    What the skeleton calls on its stages, i.e. what a new stage has to provide:

    * ``backbone(points, feats) -> PointFeatures``; attributes ``in_channels``,
      ``out_channels`` and the contract ``output``.
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
        feats = self.backbone(observation.points, self._input_features(observation))
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
