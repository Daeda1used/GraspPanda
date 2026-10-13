"""Grasp non-maximum suppression."""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["grasp_nms"]


def grasp_nms(
    translation: Tensor,
    rotation: Tensor,
    score: Tensor,
    *,
    translation_threshold: float = 0.03,
    rotation_threshold: float = math.radians(30.0),
) -> Tensor:
    """Greedy suppression of near-duplicate grasps.

    Grasps are visited in descending score order (stable, so ties keep input order). A
    kept grasp suppresses every later grasp whose translation distance is strictly
    below ``translation_threshold`` *and* whose rotation angle is strictly below
    ``rotation_threshold``.

    This operator is a model post-processor. Benchmark evaluators keep their own
    pinned copy of the rule, so changing this function can never change a reported AP.

    Args:
        translation: ``(G, 3)`` metres.
        rotation: ``(G, 3, 3)``.
        score: ``(G,)``.

    Returns:
        Indices of the kept grasps, in descending score order.
    """
    count = translation.shape[0]
    if count == 0:
        return torch.zeros(0, dtype=torch.long, device=translation.device)
    order = torch.argsort(score, descending=True, stable=True)
    centre = translation[order].double()
    frame = rotation[order].double()
    cos_threshold = math.cos(rotation_threshold)
    alive = torch.ones(count, dtype=torch.bool, device=translation.device)
    keep = []
    for i in range(count):
        if not alive[i]:
            continue
        keep.append(i)
        near = (centre[i + 1 :] - centre[i]).norm(dim=-1) < translation_threshold
        # trace(Rᵢ Rⱼᵀ) = 1 + 2 cos(angle); angle < threshold  ⇔  cos(angle) > cos(threshold)
        cosine = ((frame[i + 1 :] * frame[i]).sum(dim=(-2, -1)).clamp(-1.0, 3.0) - 1.0) / 2.0
        alive[i + 1 :] &= ~(near & (cosine > cos_threshold))
    return order[torch.tensor(keep, dtype=torch.long, device=order.device)]
