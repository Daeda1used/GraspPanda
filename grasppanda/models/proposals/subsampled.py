"""Seeds of a sub-sampling backbone: every point the encoder kept is a seed.

A PointNet++-style encoder ends on a furthest-point subsample of the scene with one
feature vector per kept point. GraspNet-baseline and the methods built on it use that
subsample as the seed set directly: nothing is scored and nothing is selected before
the view stage. Whether a seed lies on an object is predicted *with* the view scores,
by the same head, and used to mask losses and to drop background seeds when decoding;
that prediction therefore belongs to the view stage
(:class:`~grasppanda.models.views.objectness_view.ObjectnessViewStage`), not here.

This proposal has no parameters and no loss. It exists so that the contract is explicit
(``requires = "subsampled"``: a dense encoder is refused instead of being indexed) and
so that the seed count has one owner that is checked against what the backbone returns.
A method that re-samples the seeds before the view stage (object-balanced sampling)
replaces this component.
"""

from __future__ import annotations

from torch import Tensor

from grasppanda.models.base import Proposal
from grasppanda.registry import PROPOSALS
from grasppanda.structures import (
    ContractError,
    Observation,
    PointFeatures,
    SeedSet,
    Supervision,
)

__all__ = ["SubsampledSeeds"]


@PROPOSALS.register()
class SubsampledSeeds(Proposal):
    """The backbone's subsample is the seed set; all seeds are real.

    Args:
        num_seeds: Seeds per frame. The backbone decides how many points it keeps; this
            value states what the rest of the model was configured for and a backbone
            that returns another number is an error. Reference value: 1024.
    """

    requires = "subsampled"
    #: The seeds are the points the encoder returns: the config check compares
    #: ``num_seeds`` with the encoder's ``output_points``.
    seeds_from_backbone = True
    status = "tested"
    source = "GraspNet-baseline (Fang et al., CVPR 2020)"

    def __init__(self, num_seeds: int = 1024) -> None:
        super().__init__()
        if num_seeds < 1:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}")
        self.num_seeds = num_seeds

    def forward(self, feats: PointFeatures, observation: Observation) -> tuple[SeedSet, None]:
        """Wrap the backbone output as seeds.

        Args:
            feats: Features on a child of ``observation.points`` holding ``num_seeds``
                points per frame.
            observation: The batch being processed.

        Returns:
            The seeds, and ``None``: there are no per-point scores to supervise.
        """
        name = type(self).__name__
        scene, points = observation.points, feats.points
        if scene is None:
            raise ContractError(f"{name} needs observation.points")
        if points.parent_id != scene.id or points.parent_index is None:
            relation = (
                " The features live on the scene points themselves; use a proposal that "
                "selects seeds from dense features."
                if points.id == scene.id
                else ""
            )
            raise ContractError(
                f"{name} (requires '{self.requires}' scene features) needs features on a "
                f"subsample of point set #{scene.id}, got features on point set #{points.id} "
                f"with parent #{points.parent_id}.{relation}"
            )
        sizes = points.sizes
        if points.batch_size != scene.batch_size or bool((sizes != self.num_seeds).any()):
            raise ContractError(
                f"{name} is configured for {self.num_seeds} seeds per frame, but the backbone "
                f"returned {sizes.tolist()} points for a batch of {scene.batch_size} frame(s). "
                "Make num_seeds and the backbone's output level agree."
            )
        valid = feats.feats.new_ones(len(points), dtype=bool)
        return SeedSet(points, feats.feats, valid=valid, source="backbone_subsample"), None

    def loss(self, scores: None, supervision: Supervision) -> dict[str, Tensor]:
        """Nothing is predicted here, so nothing is supervised."""
        return {}

    def extra_repr(self) -> str:
        return f"num_seeds={self.num_seeds}"
