"""Score targets of the GraspNet-baseline family, derived from friction labels.

A label is the minimum friction coefficient ``μ`` under which a grasp is force-closed:
lower is better and values ≤ 0 mean "no grasp". The baseline family regresses

    s = log(μ_worst / μ)        for usable grasps,        s = 0        otherwise,

where a grasp is usable when ``μ > 0`` and its width fits the gripper. Two stages of a
detector read this quantity: the view stage (the best ``s`` of every view is the view
target) and the head bundle (``s`` on the angle × depth grid of the chosen view). Both
call the functions below, so the definition exists once.

``μ_worst`` is a reference behaviour with a name (``score_target``):

* ``log_friction_batch_max`` (reference): the largest label value among *all* matched
  labels of the current batch, over every seed and every view, usable or not. With the
  GraspNet-1Billion labels this is 1.0 in practically every batch, but a batch whose
  matched labels lack the worst level is scaled differently.
* ``log_friction_fixed_max``: the constant ``max_friction``; the controlled variant.
"""

from __future__ import annotations

import torch
from torch import Tensor

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.structures import ContractError

__all__ = ["SCORE_TARGETS", "check_score_target", "log_friction_score", "worst_friction"]

SCORE_TARGETS = ("log_friction_batch_max", "log_friction_fixed_max")


def check_score_target(score_target: str, max_friction: float) -> None:
    """Validate the two options every user of this module exposes."""
    if score_target not in SCORE_TARGETS:
        raise ValueError(f"unknown score_target '{score_target}'; choose from {SCORE_TARGETS}")
    if max_friction <= 0:
        raise ValueError(f"max_friction must be positive, got {max_friction}")


def worst_friction(matched: MatchedLabels, score_target: str, max_friction: float) -> Tensor:
    """The friction level that maps to score 0, as a 0-d tensor."""
    if score_target == "log_friction_fixed_max":
        return matched.view_friction.new_tensor(max_friction)
    if matched.view_friction.numel() == 0:
        raise ContractError("cannot derive score targets from an empty batch of matched labels")
    return matched.view_friction.max()


def log_friction_score(friction: Tensor, width: Tensor, worst: Tensor, max_width: float) -> Tensor:
    """``log(worst / μ)`` where the grasp is usable, 0 elsewhere; same shape as ``friction``."""
    if friction.shape != width.shape:
        raise ContractError(
            f"friction labels {tuple(friction.shape)} and width labels {tuple(width.shape)} "
            "must have the same shape"
        )
    usable = (friction > 0) & (width <= max_width)
    safe = torch.where(usable, friction, torch.ones_like(friction))
    return torch.where(usable, torch.log(worst / safe), torch.zeros_like(friction))
