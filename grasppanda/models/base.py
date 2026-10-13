"""Base classes of the model slots.

Each base is bound to one registry and states the contract its subclasses declare.
The contracts are what the config checker compares before anything is constructed;
see ``docs/design/ARCHITECTURE.md`` §8.

Besides its contract, every registered component describes itself with three class
attributes that ``grasppanda components`` and ``docs/components.md`` are generated from:

* ``status``: what has been proven about it (``tested``, ``weights-verified``, or
  ``bridge`` for an adapter around an optional library whose own code runs);
* ``source``: the paper, method or library it implements, as a short citation;
* ``extras``: optional dependencies it needs (``("spconv",)``), empty when none.

The one-line summary shown next to them is the first sentence of the class docstring.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

from torch import Tensor, nn

from grasppanda.registry import (
    ASSIGNERS,
    BACKBONES,
    DETECTORS,
    GROUPERS,
    HEADS,
    LOSSES,
    POSTPROCESSORS,
    PROPOSALS,
    VIEWS,
)
from grasppanda.structures import GraspGroup, Observation, PointFeatures, PointSet, Supervision

__all__ = [
    "Assigner",
    "Backbone",
    "BaseGraspDetector",
    "Grouper",
    "HeadBundle",
    "Loss",
    "PostProcessor",
    "Proposal",
    "ViewStage",
]


@DETECTORS.base
class BaseGraspDetector(nn.Module):
    """A complete grasp detection method.

    The outer contract every detector family shares. ``predict`` receives an
    :class:`Observation` only; ground truth is reachable solely through the
    ``supervision`` argument of ``loss``. Train and eval behaviour follow
    ``nn.Module.training``.

    A detector that needs parameter groups or several optimizers may define
    ``configure_optimizers(self, optimizer: dict)``; see :mod:`grasppanda.engine.optim`.
    Without it the trainer builds one optimizer over all trainable parameters.
    """

    #: Observation fields this detector reads, e.g. ``{"points"}`` or ``{"rgb", "depth"}``.
    required_inputs: ClassVar[frozenset[str]] = frozenset({"points"})

    def loss(self, observation: Observation, supervision: Supervision) -> dict[str, Tensor]:
        """Return named scalars for one training step.

        Entries whose key starts with ``loss/`` are already weighted and are summed by
        the trainer; every other entry is a metric that is only logged.
        """
        raise NotImplementedError

    def predict(self, observation: Observation) -> list[GraspGroup]:
        """Return one :class:`GraspGroup` per sample in the batch."""
        raise NotImplementedError

    def forward(self, observation: Observation, supervision: Supervision | None = None) -> Any:
        if supervision is None:
            return self.predict(observation)
        return self.loss(observation, supervision)

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterable[str]:
        """Yield messages for cross-slot problems visible in the config node alone."""
        return ()


@BACKBONES.base
class Backbone(nn.Module):
    """Scene encoder.

    ``forward(points, feats)`` takes the scene :class:`PointSet` and its ``(N, C_in)``
    input features and returns :class:`PointFeatures`.

    ``output`` states where the returned features live: ``"dense"`` means on the input
    point set itself (the result is bound to ``points``); ``"subsampled"`` means on a
    child set created with ``points.subset(...)`` (PointNet++-style abstraction). A
    consumer that needs one kind refuses the other; converting is an explicit module.
    """

    output: ClassVar[Literal["dense", "subsampled"]]
    in_channels: int
    out_channels: int

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        raise NotImplementedError


@PROPOSALS.base
class Proposal(nn.Module):
    """Turns scene features into seeds. Owns its targets and its loss."""

    requires: ClassVar[Literal["dense", "subsampled", "any"]] = "any"
    #: Per-seed scores the stage predicts, supervises and stores in ``SeedSet.scores``.
    #: Two stages of one detector may not predict the same score.
    seed_scores: ClassVar[tuple[str, ...]] = ()
    #: Loss option → kind of term (see :class:`Loss`); an undeclared option is not checked.
    loss_terms: ClassVar[Mapping[str, str]] = {}


@VIEWS.base
class ViewStage(nn.Module):
    """Chooses one approach direction per seed. Owns its targets and its loss.

    ``label_formats`` lists the kinds of matched labels the stage can encode its targets
    from (the assigner's ``label_format`` must be one of them); ``needs_matched`` the
    optional fields of the matched labels it cannot work without; ``seed_scores`` what
    it adds to ``SeedSet.scores``; ``applies_seed_weights`` whether its loss honours a
    per-seed weight returned by the assigner; ``loss_terms`` the kind of every loss
    option it has (see :class:`Loss`).
    """

    seed_scores: ClassVar[tuple[str, ...]] = ()
    label_formats: ClassVar[tuple[str, ...]] = ("dense",)
    needs_matched: ClassVar[tuple[str, ...]] = ()
    applies_seed_weights: ClassVar[bool] = False
    loss_terms: ClassVar[Mapping[str, str]] = {}


@GROUPERS.base
class Grouper(nn.Module):
    """Gathers a local neighbourhood per seed.

    ``support`` says which points are grouped; ``out_axes`` names the axes of the
    returned features so that a head never has to guess (or squeeze) them. A grouper
    that sets ``uses_seed_features`` is also handed the features of the seeds, as a
    fourth argument of ``forward`` (a scene-support crop that adds them back).
    """

    support: ClassVar[Literal["seeds", "scene"]]
    out_axes: ClassVar[tuple[str, ...]]
    out_channels: int
    uses_seed_features: bool = False


@ASSIGNERS.base
class Assigner:
    """Training-only matching of seeds and chosen views to grasp labels.

    ``label_format`` names the kind of matched labels returned: ``dense`` (angle × depth
    grids, :class:`~grasppanda.models.assigners.matched.MatchedLabels`) or ``compact``
    (one grasp per point and view). ``optional_outputs`` maps every optional field of
    the matched labels this assigner can return to the option that switches it on, as
    ``(option, value)``, or to ``None`` when the field is always returned. A stage that
    lists such a field in ``needs_matched`` is refused when the option is off; an
    assigner that returns ``weight`` is refused next to a stage that would ignore it.
    """

    label_format: ClassVar[str] = "dense"
    optional_outputs: ClassVar[Mapping[str, tuple[str, Any] | None]] = {}


@HEADS.base
class HeadBundle(nn.Module):
    """Head + target encoder + loss + decoder, swapped as one unit.

    The four parts share one grasp parameterization, so they cannot be mixed across
    bundles. ``score_kind`` names the meaning of the scores the decoder emits.
    ``label_format`` is the kind of matched labels the bundle is supervised with (the
    assigner must return the same); ``needs_matched``, ``applies_seed_weights`` and
    ``loss_terms`` mean what they mean on :class:`ViewStage`. A bundle whose terms depend
    on its options defines ``loss_terms_for(options)`` beside the attribute.
    """

    in_axes: ClassVar[tuple[str, ...]]
    score_kind: ClassVar[str] = "unspecified"
    label_format: ClassVar[str] = "dense"
    needs_matched: ClassVar[tuple[str, ...]] = ()
    applies_seed_weights: ClassVar[bool] = False
    loss_terms: ClassVar[Mapping[str, str]] = {}

    @classmethod
    def required_seed_scores(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        """Keys of ``SeedSet.scores`` the decoder reads when configured with ``options``.

        A proposal or view stage of the detector must predict each of them
        (``seed_scores``); otherwise the config is refused instead of failing at the
        first evaluation.
        """
        return ()


@LOSSES.base
class Loss(nn.Module):
    """A primitive loss used inside a stage.

    A stage names the kind of every loss term it owns in ``loss_terms`` (option → kind),
    and a primitive lists in ``accepts`` the kinds it can be put into; the config checker
    refuses any other pairing and names the primitives that fit. The kinds are

    * ``class``: ``(N, C)`` logits against ``(N,)`` class indices;
    * ``unit``: prediction and target of equal shape, the target a quality in ``[0, 1]``;
    * ``value``: prediction and target of equal shape, the target an unbounded quantity
      (a width, a tolerance, a log-friction score);
    * ``bins``: ``(N, C)`` logits against ``(N,)`` fractional bin positions in
      ``[0, C - 1]``.

    ``prediction`` says how the primitive reads the output of a ``unit`` term: as the
    ``value`` itself or as a ``logit``. A stage hands that output on through
    :meth:`as_value`, so a threshold or a decoded score sees a quality either way.
    ``applies_weights`` is false for a primitive that cannot weight single elements; it
    is then refused where the assigner returns a weight per seed.
    """

    accepts: ClassVar[tuple[str, ...]] = ()
    prediction: ClassVar[Literal["value", "logit"]] = "value"
    applies_weights: ClassVar[bool] = True

    @classmethod
    def accepted_terms(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        """Kinds of loss term the primitive fits when configured with ``options``."""
        return cls.accepts

    def as_value(self, output: Tensor) -> Tensor:
        """What a consumer reads from the raw output of a ``unit`` term this loss trains."""
        return output.sigmoid() if self.prediction == "logit" else output


@POSTPROCESSORS.base
class PostProcessor:
    """Maps predictions to predictions at inference time.

    May read the observation (a collision filter needs the sensor cloud) but never
    ground truth.
    """

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        raise NotImplementedError
