"""View targets shared by the view stages that read the labels of every view."""

from __future__ import annotations

import math

from torch import Tensor

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.structures import ContractError

__all__ = ["view_graspness"]


def view_graspness(
    matched: MatchedLabels | CompactMatchedLabels,
    max_friction: float,
    num_views: int,
    owner: str,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Graspness of every labelled view of every seed, from either label format.

    Dense labels: the fraction of the angle × depth cells of a view that hold a grasp
    with friction in ``(0, max_friction]`` (the view-wise graspness of GSNet, before
    its normalisation). Compact labels carry that quantity already encoded (min-max
    normalised per seed) for one friction level, which must be ``max_friction``.

    Returns:
        ``(M, V)`` graspness in label-view order, the ``(M,)`` seeds that supervise a
        view loss, and the ``(M,)`` per-seed loss weights (``None`` when unweighted).
    """
    compact = isinstance(matched, CompactMatchedLabels)
    labels = matched.view_graspness if compact else matched.view_friction
    if labels.shape[1] != num_views:
        raise ContractError(
            f"the matched labels describe {labels.shape[1]} views, but {owner} was built "
            f"for label sets with {num_views} (its `label_views`)"
        )
    if compact:
        if not math.isclose(matched.view_max_friction, max_friction, rel_tol=1e-9):
            raise ContractError(
                f"the compact labels encode view scores with max_friction = "
                f"{matched.view_max_friction}, but {owner} is configured with {max_friction}. "
                "Generate the label asset with the value of the view stage, or set the view "
                "stage to the value of the asset."
            )
        return labels, matched.view_valid, None
    good = (labels > 0) & (labels <= max_friction)
    cells = labels.shape[-2] * labels.shape[-1]
    graspness = good.flatten(-2).sum(dim=-1).to(labels.dtype) / cells
    return graspness, matched.valid, matched.weight
