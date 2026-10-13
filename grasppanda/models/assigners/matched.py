"""What an assigner hands to the stages that compute losses."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = ["CompactMatchedLabels", "MatchedLabels"]


@dataclass(eq=False)
class MatchedLabels:
    """Grasp labels matched to the seeds of a batch, at the views the network chose.

    All tensors have one row per seed (``M`` rows, frames concatenated). ``V`` is the
    number of template views; ``A × D`` is the in-plane-angle × depth grid of the label
    set. Label values are raw: ``friction`` is the minimum friction coefficient for
    force closure, lower is better, and values ≤ 0 mean "no grasp" (invalid or in
    collision). Turning them into regression targets is the job of the stage that owns
    the corresponding prediction.

    Attributes:
        valid: ``(M,)`` seeds that take part in the losses.
        label_point: ``(M, 3)`` matched label point, in the frame of the seeds.
        distance: ``(M,)`` distance from each seed to its label point.
        object_index: ``(M,)`` position of the matched object in the frame's object
            list, ``-1`` without a match.
        crop_centre: ``(M, 3)`` where the grouper centres the crop of each seed.
        crop_rotation: ``(M, 3, 3)`` crop frame of each seed during training: the
            object pose applied to the template frame of the chosen view. For a
            mirrored pose (flip augmentation) this is a reflection, as in the reference.
        view_friction: ``(M, V, A, D)`` friction grids of every view, views in template
            order.
        friction: ``(M, A, D)`` friction grid of the chosen view.
        width: ``(M, A, D)`` gripper widths of the chosen view, metres.
        on_object: ``(M,)`` whether the scene point of the seed is labelled as object.
            ``None`` unless the assigner restricts supervision to such seeds.
        view_width: ``(M, V, A, D)`` gripper widths of every view, views in template
            order. ``None`` unless the assigner is asked for them.
        tolerance: ``(M, A, D)`` grasp tolerance of the chosen view, metres. ``None``
            when the label set carries no tolerance.
        weight: ``(M,)`` weight of every seed in the view and grasp losses. ``None``
            means that seeds are not weighted.
    """

    valid: Tensor
    label_point: Tensor
    distance: Tensor
    object_index: Tensor
    crop_centre: Tensor
    crop_rotation: Tensor
    view_friction: Tensor
    friction: Tensor
    width: Tensor
    on_object: Tensor | None = None
    view_width: Tensor | None = None
    tolerance: Tensor | None = None
    weight: Tensor | None = None


@dataclass(eq=False)
class CompactMatchedLabels:
    """Compact grasp labels matched to the seeds of a batch, at the views the network chose.

    The counterpart of :class:`MatchedLabels` for label sets that keep one grasp per
    annotated point and view (:class:`~grasppanda.structures.CompactGraspLabels`). There
    is no angle × depth grid: a seed receives the bins, the quality and the width of the
    single grasp its label point holds at the chosen view, and the view-score targets
    arrive already encoded because the grids they were computed from no longer exist.

    All tensors have one row per seed (``M`` rows, frames concatenated); ``V`` is the
    number of template views.

    Attributes:
        valid: ``(M,)`` seeds that take part in the grasp losses: matched within the
            distance limit, at a view that carries a label.
        view_valid: ``(M,)`` seeds that take part in the view loss.
        label_point: ``(M, 3)`` matched label point, in the frame of the seeds.
        distance: ``(M,)`` distance from each seed to its label point.
        object_index: ``(M,)`` position of the matched object in the frame's object
            list, ``-1`` without a match.
        crop_centre: ``(M, 3)`` where the grouper centres the crop of each seed.
        crop_rotation: ``(M, 3, 3)`` crop frame of each seed during training: the
            object pose applied to the template frame of the chosen view.
        view_graspness: ``(M, V)`` target view scores, views in template order.
        view_max_friction: Friction level the view targets were encoded with.
        view_labelled: ``(M,)`` whether the chosen view carries a label for the seed.
        angle: ``(M,)`` in-plane angle bin of the labelled grasp, ``-1`` without a label.
        depth: ``(M,)`` depth bin of the labelled grasp, ``-1`` without a label.
        score: ``(M,)`` quality of the labelled grasp in ``[0, 1]``; higher is better
            and 0 means the view holds no usable grasp.
        width: ``(M,)`` gripper width of the labelled grasp, metres.
    """

    valid: Tensor
    view_valid: Tensor
    label_point: Tensor
    distance: Tensor
    object_index: Tensor
    crop_centre: Tensor
    crop_rotation: Tensor
    view_graspness: Tensor
    view_max_friction: float
    view_labelled: Tensor
    angle: Tensor
    depth: Tensor
    score: Tensor
    width: Tensor
