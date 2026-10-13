"""Match seeds and chosen views to compact grasp labels (training only).

The assigner of EconomicGrasp. Its first half is that of
:class:`~grasppanda.models.assigners.nearest_label.NearestLabelAssigner`: object poses
place the annotated points in the camera frame, each object's view axis is re-indexed
to the camera-frame template, every seed takes the nearest annotated point of its
frame, and the crop frame is ``R_object · R_template[view]``.

The second half differs because the labels do
(:class:`~grasppanda.structures.CompactGraspLabels`): a label point holds one grasp per
view instead of an angle × depth grid, so a seed receives scalars, and three rules of
the reference decide whether those scalars are used:

* **Distance.** A seed counts only when it lies within ``max_match_distance`` of its
  label point. The view loss is not subject to this rule.
* **Shared views.** Re-indexing maps every template view to the labelled view that the
  object pose rotates nearest to it. The map is not one-to-one: of 300 template views
  some share their labelled view with a neighbour (44 to 64 of them for six random
  poses), and as many labelled views are used by no template view. The reference hands
  the labels of a shared view to *one* of the template views and leaves the other
  without a label, see ``shared_views``.
* **Empty views.** A labelled view without any usable grasp is still a label: its seed
  is supervised with quality 0 and the bins of the first grid cell.

Compact labels reach the assigner through ``supervision.extras[COMPACT_LABELS]``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import torch
from torch import Tensor

from grasppanda.models.base import Assigner
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import knn
from grasppanda.registry import ASSIGNERS, INJECTED
from grasppanda.structures import (
    COMPACT_LABELS,
    ApproachSet,
    CompactGraspLabels,
    ContractError,
    SeedSet,
    Supervision,
)

from .matched import CompactMatchedLabels

__all__ = ["CompactLabelAssigner"]

#: Seeds processed per nearest-neighbour block; bounds the pairwise-distance tensor.
_CHUNK = 256


@ASSIGNERS.register()
class CompactLabelAssigner(Assigner):
    """Nearest annotated point per seed; its one grasp at the chosen view, if usable.

    Args:
        max_match_distance: A seed takes part in the grasp losses only when it is
            closer than this to its label point, metres. Reference value 0.005; the
            number is a matching tolerance and unrelated to any voxel size. ``None``
            accepts every match.
        shared_views: Which template view keeps the labels when the object pose maps
            several template views onto the same labelled view. The reference decides
            this with an indexed assignment that has repeated indices, whose result is
            undefined on a GPU: ``lowest`` is what that assignment yields there for
            about 95 % of the affected (label point, view) cells (measured with
            torch 1.9.1 on an RTX A6000; the rest go to the higher view, and some of
            those change from run to run), ``highest`` is what the same statement
            yields on a CPU, and
            ``all`` labels every template view, which is the controlled variant
            without the ambiguity.
        padded_seeds: Whether seeds that merely repeat another seed of their frame
            (padding of a frame with too few candidates) are counted in the losses.
            ``count`` reproduces the reference; ``ignore`` supervises real seeds only.
            Slots of a frame without any real seed are never counted.
        template: The view template of the view stage; injected by the detector.
    """

    #: Kind of matched labels the assigner returns; the head bundle must consume the same.
    label_format = "compact"
    needs_supervision: ClassVar[dict[str, tuple[str, Any] | None]] = {
        f"extras.{COMPACT_LABELS}": None
    }
    status = "tested"
    source = "EconomicGrasp (Wu et al., ECCV 2024)"

    def __init__(
        self,
        max_match_distance: float | None = 0.005,
        shared_views: Literal["lowest", "highest", "all"] = "lowest",
        padded_seeds: Literal["count", "ignore"] = "count",
        template: ViewTemplate = INJECTED,
    ) -> None:
        if not isinstance(template, ViewTemplate):
            raise TypeError(
                f"template must be the ViewTemplate of the view stage, got {type(template).__name__}"
            )
        if shared_views not in ("lowest", "highest", "all") or padded_seeds not in (
            "count",
            "ignore",
        ):
            raise ValueError(
                f"unknown policy: shared_views='{shared_views}', padded_seeds='{padded_seeds}'"
            )
        if max_match_distance is not None and max_match_distance <= 0:
            raise ValueError("max_match_distance must be positive or null")
        self.max_match_distance = max_match_distance
        self.shared_views = shared_views
        self.padded_seeds = padded_seeds
        self.template = template

    def labelled_views(self, view_map: Tensor) -> Tensor:
        """``(V,)`` template views that keep the labels of the view they map to.

        Args:
            view_map: ``(V,)`` labelled view of every template view.
        """
        if self.shared_views == "all":
            return torch.ones_like(view_map, dtype=torch.bool)
        index = torch.arange(len(view_map), device=view_map.device)
        same = view_map[:, None] == view_map[None, :]
        if self.shared_views == "lowest":
            owner = torch.where(same, index[None, :], len(view_map)).amin(dim=1)
        else:
            owner = torch.where(same, index[None, :], -1).amax(dim=1)
        return owner == index

    @torch.no_grad()
    def __call__(
        self, seeds: SeedSet, approach: ApproachSet, supervision: Supervision
    ) -> CompactMatchedLabels:
        name = type(self).__name__
        points = seeds.points
        if approach.view_index is None or approach.rotation is None:
            raise ContractError(f"{name} needs an approach with view indices and template frames")
        frames = self._frames(supervision, points.batch_size)
        friction = self._check(frames)
        coords = points.coords
        count, views = len(points), self.template.num_views
        device = coords.device
        valid = torch.zeros(count, dtype=torch.bool, device=device)
        view_valid = torch.zeros(count, dtype=torch.bool, device=device)
        labelled = torch.zeros(count, dtype=torch.bool, device=device)
        label_point = coords.clone()
        distance = coords.new_full((count,), float("inf"))
        object_index = torch.full((count,), -1, dtype=torch.long, device=device)
        crop_rotation = approach.rotation.clone()
        view_graspness = coords.new_zeros(count, views)
        angle = torch.full((count,), -1, dtype=torch.long, device=device)
        depth = torch.full((count,), -1, dtype=torch.long, device=device)
        score = coords.new_zeros(count)
        width = coords.new_zeros(count)

        offsets = [0, count] if points.batch_offsets is None else points.batch_offsets.tolist()
        real = seeds.valid if seeds.valid is not None else torch.ones_like(valid)
        for frame, objects in enumerate(frames):
            rows = slice(offsets[frame], offsets[frame + 1])
            if rows.stop == rows.start or not any(len(obj.points) for obj in objects):
                continue
            seed_xyz = coords[rows]
            chosen = approach.view_index[rows]
            cloud, owner, local, view_maps, rotations, keeps = [], [], [], [], [], []
            for position, obj in enumerate(objects):
                rotation = obj.pose[:3, :3].to(coords)
                grasp_points = obj.points.to(coords)
                cloud.append(grasp_points @ rotation.T + obj.pose[:3, 3].to(coords))
                owner.append(torch.full((len(grasp_points),), position, device=device))
                local.append(torch.arange(len(grasp_points), device=device))
                # Template view i ← the labelled view that the pose rotates nearest to it.
                rotated = self.template.views @ rotation.T
                view_map = knn(self.template.views[None], rotated[None], 1)[1][0, :, 0]
                view_maps.append(view_map)
                rotations.append((rotation @ self.template.rotations)[view_map])
                keeps.append(self.labelled_views(view_map))
            cloud_all = torch.cat(cloud)
            nearest = knn(seed_xyz[None], cloud_all[None], 1, chunk=_CHUNK)[1][0, :, 0]
            owner_all, local_all = torch.cat(owner)[nearest], torch.cat(local)[nearest]
            label_point[rows] = cloud_all[nearest]
            distance[rows] = (seed_xyz - cloud_all[nearest]).norm(dim=-1)
            object_index[rows] = owner_all
            for position, obj in enumerate(objects):
                member = torch.nonzero(owner_all == position).flatten()
                if len(member) == 0:
                    continue
                target = member + rows.start
                picked = chosen[member]
                view_map = view_maps[position]
                # Gather on the device the labels live on (they may stay on the CPU: only
                # the matched rows are ever needed) and move the result.
                store = obj.view_graspness.device
                point, column = local_all[member].to(store), view_map[picked].to(store)
                view_graspness[target] = obj.view_graspness[point][:, view_map.to(store)].to(coords)
                angle[target] = obj.angle[point, column].to(device).long()
                depth[target] = obj.depth[point, column].to(device).long()
                score[target] = obj.quality(obj.score[point, column]).to(coords)
                width[target] = obj.metres(obj.width[point, column]).to(coords)
                crop_rotation[target] = rotations[position][picked]
                labelled[target] = keeps[position][picked]
            usable = torch.ones_like(real[rows])
            if self.padded_seeds == "ignore":
                usable = usable & real[rows]
            view_valid[rows] = usable & real[rows].any()
            near = torch.ones_like(usable)
            if self.max_match_distance is not None:
                near = distance[rows] < self.max_match_distance
            valid[rows] = view_valid[rows] & near & labelled[rows]

        unlabelled = ~labelled
        angle[unlabelled], depth[unlabelled] = -1, -1
        score[unlabelled], width[unlabelled] = 0.0, 0.0
        return CompactMatchedLabels(
            valid=valid,
            view_valid=view_valid,
            label_point=label_point,
            distance=distance,
            object_index=object_index,
            crop_centre=coords,
            crop_rotation=crop_rotation,
            view_graspness=view_graspness,
            view_max_friction=friction,
            view_labelled=labelled,
            angle=angle,
            depth=depth,
            score=score,
            width=width,
        )

    def _frames(self, supervision: Supervision, batch: int) -> list[list[CompactGraspLabels]]:
        name = type(self).__name__
        if COMPACT_LABELS not in supervision.extras:
            dense = sum(len(objects) for objects in supervision.objects)
            raise ContractError(
                f"{name} reads compact grasp labels from supervision.extras['{COMPACT_LABELS}'], "
                f"which this batch does not carry (extras: {sorted(supervision.extras)}; dense "
                f"label objects: {dense}). Configure `compact_labels` on the dataset, or use an "
                "assigner for the dense label grids."
            )
        frames = supervision.extras[COMPACT_LABELS]
        if len(frames) != batch:
            raise ContractError(
                f"{name}: supervision lists compact labels for {len(frames)} frame(s), but the "
                f"batch has {batch}"
            )
        return frames

    def _check(self, frames: list[list[CompactGraspLabels]]) -> float:
        """Check the labels against the template; return the friction their view scores use."""
        frictions = set()
        for objects in frames:
            for obj in objects:
                if not isinstance(obj, CompactGraspLabels):
                    raise ContractError(
                        f"{type(self).__name__} expects CompactGraspLabels, got "
                        f"{type(obj).__name__}"
                    )
                if obj.num_views != self.template.num_views:
                    raise ContractError(
                        f"object {obj.object_id} is labelled on {obj.num_views} views, but the "
                        f"view template has {self.template.num_views}. The label set and the "
                        "view stage must use the same view lattice."
                    )
                frictions.add(float(obj.view_max_friction))
        if not frictions:
            raise ContractError(
                f"{type(self).__name__} received a batch without any labelled object; the "
                "data pipeline must not hand such a batch to loss()"
            )
        if len(frictions) > 1:
            raise ContractError(
                f"the compact labels of one batch encode view scores with different friction "
                f"levels {sorted(frictions)}; they come from different label assets"
            )
        return frictions.pop()
