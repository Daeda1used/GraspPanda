"""Match seeds and chosen views to dense grasp labels (training only).

Targets of a seed-based detector cannot be computed before the forward pass: they
depend on which seeds and which view the network chose. The assigner runs in the
middle of the training forward and is the only place where ground truth enters it.

For every frame it

1. moves each object's annotated grasp points into the camera frame with the object's
   ground-truth pose;
2. re-indexes the object's view axis to the camera-frame template: template view ``i``
   takes the labels of the object-frame view whose rotated viewpoint is nearest to it;
3. gives each seed the nearest annotated point over all objects of the frame;
4. returns that point's friction and width grids at the chosen view, the friction
   grids of all views (the view stage derives its target from them), and the crop
   frame ``R_object · R_template[view]``.

Step 4 is why training and inference crops differ in the reference: the training crop
frame comes from the labels, the inference crop frame from the camera-frame template.

What differs between the methods that share this matching is a parameter here: whether
a match has a distance limit, where the training crop is centred, which seeds are
supervised, and whether the widths of all views are returned as well. A label set that
carries grasp tolerance has it sliced like the widths.

The grids are read through :func:`~grasppanda.structures.observation.gather_grids`, never
by indexing ``ObjectGraspLabels.scores`` directly: of the label points a frame carries,
the seeds of a batch are matched to a few per cent, and labels that were left in their
store (``SampleGraspLabels(grids="deferred")``) are read for those rows only.
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
    ApproachSet,
    ContractError,
    ObjectGraspLabels,
    SeedSet,
    Supervision,
)
from grasppanda.structures.observation import gather_grids

from .matched import MatchedLabels

__all__ = ["NearestLabelAssigner"]

#: Seeds processed per nearest-neighbour block; bounds the pairwise-distance tensor.
_CHUNK = 256
#: Seeds whose label grids are read in one pass; bounds the memory of the gathers
#: (4096 seeds of the GraspNet-1Billion grid are 236 MiB per grid).
_GATHER = 4096


@ASSIGNERS.register()
class NearestLabelAssigner(Assigner):
    """Nearest annotated grasp point per seed, labels sliced at the chosen view.

    Args:
        max_match_distance: A seed further than this from its label point (metres) is
            excluded from the losses. ``None`` (the Graspness reference) accepts every
            match, so a seed on the table still receives a full label grid.
        crop_centre: Where the training crop of a seed is centred: at the ``seed``
            itself (Graspness reference) or at the matched ``label_point``.
        padded_seeds: Whether seeds that merely repeat another seed of their frame
            (padding of a frame with too few candidates) are counted in the losses.
            ``count`` reproduces the reference, whose loss means run over all seed
            slots; ``ignore`` supervises real seeds only. Slots of a frame without any
            real seed are never counted.
        supervise: Which seeds take part in the view and grasp losses. ``all`` (the
            Graspness reference) supervises every seed. ``object_seeds`` (the
            GraspNet-baseline reference) supervises a seed only if its scene point is
            labelled as object in ``supervision.point_labels['objectness']``; the label
            of every seed is then returned as ``MatchedLabels.on_object``.
        view_widths: Also return the widths of every view (``MatchedLabels.view_width``).
            A view target that discards grasps wider than the gripper needs them.
        template: The view template of the view stage; injected by the detector.
    """

    status = "tested"
    source = "Label matching of GraspNet-baseline and GSNet"
    optional_outputs: ClassVar[dict[str, tuple[str, Any] | None]] = {
        "on_object": ("supervise", "object_seeds"),
        "view_width": ("view_widths", True),
    }
    needs_supervision: ClassVar[dict[str, tuple[str, Any] | None]] = {
        "objects": None,
        "point_labels.objectness": ("supervise", "object_seeds"),
    }

    def __init__(
        self,
        max_match_distance: float | None = None,
        crop_centre: Literal["seed", "label_point"] = "seed",
        padded_seeds: Literal["count", "ignore"] = "count",
        supervise: Literal["all", "object_seeds"] = "all",
        view_widths: bool = False,
        template: ViewTemplate = INJECTED,
    ) -> None:
        if not isinstance(template, ViewTemplate):
            raise TypeError(
                f"template must be the ViewTemplate of the view stage, got {type(template).__name__}"
            )
        if crop_centre not in ("seed", "label_point") or padded_seeds not in ("count", "ignore"):
            raise ValueError(
                f"unknown policy: crop_centre='{crop_centre}', padded_seeds='{padded_seeds}'"
            )
        if supervise not in ("all", "object_seeds"):
            raise ValueError(f"unknown policy: supervise='{supervise}'")
        if max_match_distance is not None and max_match_distance <= 0:
            raise ValueError("max_match_distance must be positive or null")
        self.max_match_distance = max_match_distance
        self.crop_centre = crop_centre
        self.padded_seeds = padded_seeds
        self.supervise = supervise
        self.view_widths = view_widths
        self.template = template

    @torch.no_grad()
    def __call__(
        self, seeds: SeedSet, approach: ApproachSet, supervision: Supervision
    ) -> MatchedLabels:
        name = type(self).__name__
        points = seeds.points
        if approach.view_index is None or approach.rotation is None:
            raise ContractError(f"{name} needs an approach with view indices and template frames")
        if len(supervision.objects) != points.batch_size:
            raise ContractError(
                f"{name}: supervision lists objects for {len(supervision.objects)} frame(s), "
                f"but the batch has {points.batch_size}"
            )
        coords = points.coords
        count = len(points)
        grid = self._label_grid(supervision)
        views = self.template.num_views
        on_object = self._on_object(seeds, supervision) if self.supervise != "all" else None
        with_tolerance = self._has_tolerance(supervision)
        valid = torch.zeros(count, dtype=torch.bool, device=coords.device)
        label_point = coords.clone()
        distance = coords.new_full((count,), float("inf"))
        object_index = torch.full((count,), -1, dtype=torch.long, device=coords.device)
        crop_rotation = approach.rotation.clone()
        view_friction = coords.new_zeros(count, views, *grid)
        friction = coords.new_zeros(count, *grid)
        width = coords.new_zeros(count, *grid)
        view_width = coords.new_zeros(count, views, *grid) if self.view_widths else None
        tolerance = coords.new_zeros(count, *grid) if with_tolerance else None

        offsets = [0, count] if points.batch_offsets is None else points.batch_offsets.tolist()
        real = seeds.valid if seeds.valid is not None else torch.ones_like(valid)
        # Geometry first, frame by frame: which object and which of its label points every
        # seed is matched to. `labelled` lists the objects of all frames; `view_maps` and
        # `crops` hold, per object, the object-frame view and the crop frame of every
        # template view.
        labelled: list[ObjectGraspLabels] = []
        view_maps: list[Tensor] = []
        crops: list[Tensor] = []
        matched_object = torch.full((count,), -1, dtype=torch.long, device=coords.device)
        matched_point = torch.zeros(count, dtype=torch.long, device=coords.device)
        for frame, objects in enumerate(supervision.objects):
            rows = slice(offsets[frame], offsets[frame + 1])
            if not objects or rows.stop == rows.start:
                continue
            seed_xyz = coords[rows]
            cloud, owner, local, maps, frames = [], [], [], [], []
            for position, obj in enumerate(objects):
                rotation = obj.pose[:3, :3].to(coords)
                grasp_points = obj.points.to(coords)
                cloud.append(grasp_points @ rotation.T + obj.pose[:3, 3].to(coords))
                owner.append(torch.full((len(grasp_points),), position, device=coords.device))
                local.append(torch.arange(len(grasp_points), device=coords.device))
                # Template view i ← the object-frame view that the pose rotates nearest to it.
                rotated = self.template.views @ rotation.T
                view_map = knn(self.template.views[None], rotated[None], 1)[1][0, :, 0]
                maps.append(view_map)
                frames.append((rotation @ self.template.rotations)[view_map])
            cloud_all = torch.cat(cloud)
            if len(cloud_all) == 0:  # objects without a single annotated point left
                continue
            nearest_distance, nearest = knn(seed_xyz[None], cloud_all[None], 1, chunk=_CHUNK)
            nearest = nearest[0, :, 0]
            owner_all, local_all = torch.cat(owner)[nearest], torch.cat(local)[nearest]
            label_point[rows] = cloud_all[nearest]
            distance[rows] = nearest_distance[0, :, 0]
            object_index[rows] = owner_all
            matched_object[rows] = owner_all + len(labelled)
            matched_point[rows] = local_all
            labelled += objects
            view_maps += maps
            crops += frames
            usable = torch.ones_like(real[rows])
            if self.max_match_distance is not None:
                usable = distance[rows] <= self.max_match_distance
            if self.padded_seeds == "ignore":
                usable = usable & real[rows]
            if on_object is not None:
                usable = usable & on_object[rows]
            valid[rows] = usable & real[rows].any()

        # Then the labels, for all frames at once. Only the grids of the matched label
        # points are read, through `gather_grids`: they may be tensors on any device or
        # still lie in the label store (deferred labels); the values are the same.
        matched = torch.nonzero(matched_object >= 0).flatten()
        if labelled:
            view_map_of, crop_of = torch.stack(view_maps), torch.stack(crops)
        for chunk in matched.split(_GATHER) if labelled else ():
            who, point = matched_object[chunk], matched_point[chunk]
            picked = approach.view_index[chunk]
            order = view_map_of[who]  # (K, V): the object-frame view of every template view
            steps = torch.arange(len(chunk), device=coords.device)
            scores = gather_grids(labelled, "scores", who, point)[steps[:, None], order]
            collides = gather_grids(labelled, "collision", who, point)
            if collides is not None:
                scores = scores.masked_fill(collides[steps[:, None], order], 0.0)
            scores = scores.to(coords)
            view_friction[chunk] = scores
            friction[chunk] = scores[steps, picked]
            chosen_view = order[steps, picked]
            if view_width is not None:
                widths = gather_grids(labelled, "widths", who, point)[steps[:, None], order]
                widths = widths.to(coords)
                view_width[chunk] = widths
                width[chunk] = widths[steps, picked]
            else:
                width[chunk] = gather_grids(labelled, "widths", who, point, chosen_view).to(coords)
            if tolerance is not None:
                read = gather_grids(labelled, "tolerance", who, point, chosen_view)
                tolerance[chunk] = read.to(coords)
            crop_rotation[chunk] = crop_of[who, picked]

        centre = coords
        if self.crop_centre == "label_point":
            centre = torch.where(object_index[:, None] >= 0, label_point, coords)
        return MatchedLabels(
            valid=valid,
            label_point=label_point,
            distance=distance,
            object_index=object_index,
            crop_centre=centre,
            crop_rotation=crop_rotation,
            view_friction=view_friction,
            friction=friction,
            width=width,
            on_object=on_object,
            view_width=view_width,
            tolerance=tolerance,
        )

    def _on_object(self, seeds: SeedSet, supervision: Supervision) -> Tensor:
        """``(M,)`` objectness label of the scene point every seed was taken from."""
        name = type(self).__name__
        index = seeds.points.parent_index
        labels = supervision.point_labels.get("objectness")
        if labels is None or index is None:
            raise ContractError(
                f"{name} is configured with supervise: object_seeds, which reads "
                "supervision.point_labels['objectness'] at the scene point of every seed; "
                + (
                    "the seeds do not record which scene points they are"
                    if labels is not None
                    else f"available point labels: {sorted(supervision.point_labels)}"
                )
            )
        labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] != 1 or (len(index) and int(index.max()) >= labels.shape[0]):
            raise ContractError(
                f"{name}: point label 'objectness' has shape {tuple(labels.shape)}, which does "
                "not cover the scene points the seeds refer to; labels must be aligned with "
                "observation.points"
            )
        return labels[:, 0].to(index.device)[index] != 0

    def _has_tolerance(self, supervision: Supervision) -> bool:
        """Whether the label set carries tolerance; it must do so for every object or none."""
        shapes = [obj.shapes() for objects in supervision.objects for obj in objects]
        present = ["tolerance" in shape for shape in shapes]
        if any(present) and not all(present):
            raise ContractError(
                f"{type(self).__name__}: some objects of the batch carry tolerance labels and "
                "others do not"
            )
        for objects in supervision.objects:
            for obj in objects:
                shape = obj.shapes()
                if "tolerance" in shape and shape["tolerance"] != shape["scores"]:
                    raise ContractError(
                        f"object {obj.object_id}: tolerance labels have shape "
                        f"{shape['tolerance']}, but the scores have {shape['scores']}"
                    )
        return bool(present) and all(present)

    def _label_grid(self, supervision: Supervision) -> tuple[int, int]:
        """Angle × depth size of the label grids, after checking them against the template."""
        grid: tuple[int, int] | None = None
        for objects in supervision.objects:
            for obj in objects:
                shapes = obj.shapes()
                shape = shapes["scores"]
                if len(shape) != 4 or shape != shapes["widths"] or shape[0] != len(obj.points):
                    raise ContractError(
                        f"object {obj.object_id}: scores and widths must both be (P, V, A, D) "
                        f"with P = {len(obj.points)} label points, got {shape} and "
                        f"{shapes['widths']}"
                    )
                if shape[1] != self.template.num_views:
                    raise ContractError(
                        f"object {obj.object_id} is labelled on {shape[1]} views, but the view "
                        f"template has {self.template.num_views}. The label set and the view "
                        "stage must use the same view lattice."
                    )
                if grid is not None and shape[2:] != grid:
                    raise ContractError(
                        f"objects are labelled on different angle × depth grids: {grid} and "
                        f"{shape[2:]}"
                    )
                grid = shape[2:]
        if grid is None:
            raise ContractError(
                f"{type(self).__name__} received a batch without any labelled object; the "
                "data pipeline must not hand such a batch to loss()"
            )
        return grid
