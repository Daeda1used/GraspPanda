"""Scale-balanced seed weights (Scale-Balanced-Grasp, CoRL 2022).

Labelled grasps are not spread evenly over gripper widths: seeds on large objects are
many, seeds whose best grasp is narrow are few. The method re-weights the view and grasp
losses of a seed by how rare the *scale* of its label is, with the scale read from a
histogram of grasp widths over the training labels:

    weight(bin) = 1 − ln(count(bin) / max count)

The scale of a seed is the label width of the best grasp of its matched label point over
*all* views, angles and depths, where "best" is the highest score target of
:mod:`grasppanda.models.heads.friction` and the first cell wins a tie.

Reference behaviours, kept:

* A width that lies in no bin, i.e. on an edge or outside the histogram's range, gets
  the weight of bin 0. The comparisons are strict on both sides, in single precision.
* A seed whose label point has no usable grasp has no best grasp; the first cell of its
  grid is then taken, whatever its width.

The histogram is a statistic of the training labels. It is not a constant of the model:
the dataset serves it (``Supervision.extras['scale_histogram']``, see
:mod:`grasppanda.datasets.scale_statistic`), so a different label set or seed recipe
comes with its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor

from grasppanda.models.heads.friction import log_friction_score
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import ASSIGNERS, INJECTED
from grasppanda.structures import ApproachSet, ContractError, Gripper, SeedSet, Supervision

from .matched import MatchedLabels
from .nearest_label import NearestLabelAssigner

__all__ = [
    "HISTOGRAM_KEY",
    "ScaleBalancedAssigner",
    "best_grasp_width",
    "bin_weights",
    "collect_seed_scales",
    "read_histogram",
    "scale_balanced_weights",
]

#: Entry of ``Supervision.extras`` that holds, per frame, ``{"edges": (K + 1,), "counts": (K,)}``.
HISTOGRAM_KEY = "scale_histogram"


def bin_weights(counts: Tensor) -> Tensor:
    """``1 − ln(count / max count)`` per bin, in single precision as the reference."""
    counts = counts.to(torch.float32)
    if counts.ndim != 1 or len(counts) == 0 or not bool((counts > 0).all()):
        raise ContractError(
            "a scale histogram needs a positive count in every bin; an empty bin has no "
            f"finite weight (counts: {counts.tolist()})"
        )
    return 1.0 - torch.log(counts / counts.max())


def read_histogram(supervision: Supervision, *, user: str) -> tuple[Tensor, Tensor]:
    """``(edges, counts)`` of the scale histogram the dataset attached to the batch."""
    entries: list[Any] = supervision.extras.get(HISTOGRAM_KEY) or []
    if not entries:
        raise ContractError(
            f"{user} weights seeds by a histogram of grasp widths and reads it from "
            f"supervision.extras['{HISTOGRAM_KEY}']; configure the dataset with a "
            "`scale_statistic:` section (build the asset with `grasppanda prepare <dataset> "
            "--scale-statistic`)"
        )
    edges, counts = entries[0]["edges"], entries[0]["counts"]
    if edges.ndim != 1 or len(edges) != len(counts) + 1 or bool((edges[1:] <= edges[:-1]).any()):
        raise ContractError(
            f"{user}: a scale histogram needs K + 1 increasing edges for K counts, got "
            f"{tuple(edges.shape)} edges and {tuple(counts.shape)} counts"
        )
    for other in entries[1:]:
        if not (torch.equal(other["edges"], edges) and torch.equal(other["counts"], counts)):
            raise ContractError(f"{user}: the frames of a batch carry different scale histograms")
    return edges, counts


@torch.no_grad()
def best_grasp_width(view_friction: Tensor, view_width: Tensor, max_width: float) -> Tensor:
    """``(M,)`` label width of the best grasp of every seed's label point over all views.

    The best grasp has the highest score target; the first cell wins a tie, and a label
    point without a usable grasp yields the width of its first cell.
    """
    seeds = view_friction.shape[0]
    worst = view_friction.max() if view_friction.numel() else view_friction.new_tensor(1.0)
    score = log_friction_score(view_friction, view_width, worst, max_width).reshape(seeds, -1)
    best = score.argmax(dim=1)
    return view_width.reshape(seeds, -1).gather(1, best[:, None])[:, 0]


@torch.no_grad()
def scale_balanced_weights(
    view_friction: Tensor,
    view_width: Tensor,
    max_width: float,
    edges: Tensor,
    counts: Tensor,
) -> Tensor:
    """Loss weight ``(M,)`` of every seed from the scale of its matched label.

    Args:
        view_friction: ``(M, V, A, D)`` friction grids of every view of the matched point.
        view_width: ``(M, V, A, D)`` label widths of the same grasps, metres.
        max_width: A grasp wider than this is not usable.
        edges: ``(K + 1,)`` bin edges of the width histogram, metres.
        counts: ``(K,)`` labels per bin.
    """
    width = best_grasp_width(view_friction, view_width, max_width)
    edges = edges.to(width)  # the comparison is made in the precision of the widths
    inside = (width[:, None] > edges[None, :-1]) & (width[:, None] < edges[None, 1:])
    index = inside.to(torch.uint8).argmax(dim=1)  # bins are disjoint; no bin → bin 0
    # The logarithm is taken on the CPU, as the reference does once at import: the last
    # bit of a float32 logarithm is not the same on every device.
    return bin_weights(counts.cpu()).to(width.device)[index]


@torch.no_grad()
def collect_seed_scales(
    dataset: Any,
    *,
    frame_step: int,
    num_seeds: int,
    min_graspable_views: int,
    graspable_score: float,
    max_width: float,
    device: torch.device | str = "cpu",
    progress: Callable[[int], None] | None = None,
) -> Tensor:
    """Best-grasp widths of the graspable seeds of a dataset's frames, for its histogram.

    For every ``frame_step``-th sample of ``dataset`` (which must serve labels): the
    seeds are the first ``num_seeds`` furthest point samples of the input cloud, taken
    as the PointNet++ lineage does (never a point next to the origin, ties in the order
    of its kernel), i.e. the seeds the reference backbones keep. Each seed is matched to
    its nearest label point; it counts when it lies on an object and more than
    ``min_graspable_views`` of its views have a score target above ``graspable_score``.

    Returns:
        ``(K,)`` widths in metres, on the CPU.
    """
    template = ViewTemplate.fibonacci(dataset.label_grid.num_view).to(device)
    assigner = NearestLabelAssigner(supervise="object_seeds", view_widths=True, template=template)
    collected = []
    for done, index in enumerate(range(0, len(dataset), frame_step), start=1):
        batch = dataset[index].to(device)
        points = batch.observation.points
        if len(points) < num_seeds:
            raise ContractError(
                f"{batch.observation.keys[0]} has {len(points)} points, fewer than the "
                f"{num_seeds} seeds the scale statistic is configured for"
            )
        rows = furthest_point_sample(
            points.coords, num_seeds, skip_near_origin=True, tie_break="pointnet2_cuda"
        )
        offsets = torch.tensor([0, num_seeds], device=rows.device)
        seeds = SeedSet(points.subset(rows, offsets))
        view = torch.zeros(num_seeds, dtype=torch.long, device=rows.device)
        approach = ApproachSet(template.approach(view), template.rotation(view), view)
        matched = assigner(seeds, approach, batch.supervision)
        worst = matched.view_friction.max()
        score = log_friction_score(matched.view_friction, matched.view_width, worst, max_width)
        views = (score.flatten(-2).amax(dim=-1) > graspable_score).sum(dim=-1)
        graspable = matched.on_object & (views > min_graspable_views)
        width = best_grasp_width(matched.view_friction, matched.view_width, max_width)
        collected.append(width[graspable].cpu())
        if progress is not None:
            progress(done)
    return torch.cat(collected) if collected else torch.zeros(0)


@ASSIGNERS.register()
class ScaleBalancedAssigner(NearestLabelAssigner):
    """Nearest-label matching plus a scale-balanced loss weight per seed.

    Everything :class:`NearestLabelAssigner` returns, with the widths of every view, and
    in addition ``MatchedLabels.weight``: the weight of the width bin of every seed's
    best grasp, read from the histogram in ``supervision.extras['scale_histogram']``.

    Args:
        max_match_distance, crop_centre, padded_seeds, supervise: As in
            :class:`NearestLabelAssigner`; the defaults here are those of the
            Scale-Balanced-Grasp reference.
        template: The view template of the view stage; injected by the detector.
        gripper: Gripper the labels are read for; injected by the detector. A grasp
            wider than its ``max_width`` cannot be the best grasp of a label point.
    """

    status = "tested"
    source = "Scale-Balanced-Grasp (Ma and Huang, CoRL 2022)"
    optional_outputs: ClassVar[dict[str, tuple[str, Any] | None]] = {
        "on_object": ("supervise", "object_seeds"),
        "view_width": None,
        "weight": None,
    }
    needs_supervision: ClassVar[dict[str, tuple[str, Any] | None]] = {
        **NearestLabelAssigner.needs_supervision,
        f"extras.{HISTOGRAM_KEY}": None,
    }

    def __init__(
        self,
        max_match_distance: float | None = None,
        crop_centre: Literal["seed", "label_point"] = "label_point",
        padded_seeds: Literal["count", "ignore"] = "count",
        supervise: Literal["all", "object_seeds"] = "object_seeds",
        template: ViewTemplate = INJECTED,
        gripper: Gripper = INJECTED,
    ) -> None:
        if not isinstance(gripper, Gripper):
            raise TypeError(f"gripper must be a Gripper, got {type(gripper).__name__}")
        super().__init__(
            max_match_distance=max_match_distance,
            crop_centre=crop_centre,
            padded_seeds=padded_seeds,
            supervise=supervise,
            view_widths=True,  # the scale of a seed is read from the widths of all views
            template=template,
        )
        self.gripper = gripper

    @torch.no_grad()
    def __call__(
        self, seeds: SeedSet, approach: ApproachSet, supervision: Supervision
    ) -> MatchedLabels:
        histogram = read_histogram(supervision, user=type(self).__name__)
        matched = super().__call__(seeds, approach, supervision)
        matched.weight = scale_balanced_weights(
            matched.view_friction, matched.view_width, self.gripper.max_width, *histogram
        )
        return matched
