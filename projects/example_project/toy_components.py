"""A minimal out-of-tree detector: the template for adding your own method.

It is wired exactly like an in-tree method (registries, slot contracts, injection, a
config check, a loss, a post-processor) and trains end to end on the synthetic scenes
of ``toy_data.py``. What it learns is a toy task. Copy the structure, not the numbers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models import Backbone, BaseGraspDetector, Grouper, HeadBundle, PostProcessor
from grasppanda.ops import (
    approach_to_rotation,
    ball_query,
    fibonacci_views,
    furthest_point_sample,
    gather_points,
    group_points,
)
from grasppanda.registry import (
    BACKBONES,
    DETECTORS,
    GROUPERS,
    HEADS,
    INJECTED,
    POSTPROCESSORS,
    Cfg,
)
from grasppanda.structures import (
    ContractError,
    GraspGroup,
    GroupFeatures,
    Observation,
    PointFeatures,
    PointSet,
    Supervision,
)


@BACKBONES.register()
class ToyBackbone(Backbone):
    """Per-point MLP. ``output = "dense"``: features live on the input point set."""

    output = "dense"
    status = "example"

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)
        )

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        return PointFeatures(self.mlp(feats), points)


@GROUPERS.register()
class ToyGrouper(Grouper):
    """Max-pools features in a ball around each seed. Emits ``(seed, channel)``."""

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "example"

    def __init__(
        self, radius: float = 0.05, nsample: int = 16, in_channels: int = INJECTED
    ) -> None:
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.out_channels = in_channels

    def forward(self, seeds: Tensor, feats: Tensor) -> GroupFeatures:
        index, _ = ball_query(seeds, seeds, self.radius, self.nsample)
        pooled = group_points(feats, index).amax(dim=2)
        return GroupFeatures(pooled.flatten(0, 1), self.out_axes)


@GROUPERS.register()
class ToyDepthGrouper(ToyGrouper):
    """Same pooling repeated per depth bin. Emits ``(seed, depth, channel)``."""

    out_axes = ("seed", "depth", "channel")

    def __init__(
        self,
        radius: float = 0.05,
        nsample: int = 16,
        num_depth: int = 4,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__(radius, nsample, in_channels)
        self.num_depth = num_depth

    def forward(self, seeds: Tensor, feats: Tensor) -> GroupFeatures:
        flat = super().forward(seeds, feats).feats
        return GroupFeatures(flat[:, None, :].expand(-1, self.num_depth, -1), self.out_axes)


@HEADS.register()
class ToyHead(HeadBundle):
    """Scores an angle × depth grid per seed and decodes the best cell into a grasp.

    Head, loss and decoder live in one class because they share one parameterization:
    the cell index is ``angle_bin * num_depth + depth_bin`` in all three.
    """

    in_axes = ("seed", "channel")
    score_kind = "cell_logit"
    status = "example"

    def __init__(
        self,
        num_angle: int = 12,
        num_depth: int = 4,
        depth_step: float = 0.01,
        max_width: float = 0.1,
        width_weight: float = 10.0,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        self.num_angle, self.num_depth = num_angle, num_depth
        self.depth_step, self.max_width, self.width_weight = depth_step, max_width, width_weight
        self.score = nn.Linear(in_channels, num_angle * num_depth)
        self.width = nn.Linear(in_channels, 1)

    def forward(self, groups: GroupFeatures) -> dict[str, Tensor]:
        groups.expect(self.in_axes, user=type(self).__name__)
        return {"score": self.score(groups.feats), "width": self.width(groups.feats).squeeze(-1)}

    def loss(self, raw: dict[str, Tensor], cell: Tensor, width: Tensor) -> dict[str, Tensor]:
        """Classify the labelled cell and regress the labelled width of every seed.

        Keys starting with ``loss/`` are summed by the trainer; the rest is only logged.
        """
        if cell.numel() and int(cell.max()) >= self.num_angle * self.num_depth:
            raise ContractError(
                f"label cell {int(cell.max())} does not exist on a {self.num_angle} × "
                f"{self.num_depth} grid: the dataset and {type(self).__name__} must agree on it"
            )
        predicted = raw["width"].sigmoid() * self.max_width
        return {
            "loss/score": F.cross_entropy(raw["score"], cell),
            "loss/width": self.width_weight * F.smooth_l1_loss(predicted, width, beta=0.01),
            "cell_accuracy": (raw["score"].argmax(dim=-1) == cell).float().mean(),
        }

    def decode(self, raw: dict[str, Tensor], seeds: Tensor, approach: Tensor) -> GraspGroup:
        score, cell = raw["score"].max(dim=-1)
        angle = (cell // self.num_depth) * (torch.pi / self.num_angle)
        depth = (cell % self.num_depth + 1) * self.depth_step
        return GraspGroup(
            translation=seeds,
            rotation=approach_to_rotation(approach, angle),
            width=raw["width"].sigmoid() * self.max_width,
            depth=depth,
            score=score,
            score_kind=self.score_kind,
        )


@DETECTORS.register()
class ToyDetector(BaseGraspDetector):
    """backbone → FPS seeds → grouper → head, with a fixed approach per seed."""

    required_inputs = frozenset({"points"})
    status = "example"

    def __init__(
        self,
        backbone: Cfg[Backbone],
        grouper: Cfg[Grouper],
        head: Cfg[HeadBundle],
        num_seeds: int = 64,
    ) -> None:
        super().__init__()
        self.num_seeds = num_seeds
        # Children receive what the parent already knows instead of repeating it in YAML.
        self.backbone = BACKBONES.build(backbone)
        self.grouper = GROUPERS.build(grouper, in_channels=self.backbone.out_channels)
        self.head = HEADS.build(head, in_channels=self.grouper.out_channels)

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Compare slot contracts before anything is built."""
        grouper, head = node.get("grouper"), node.get("head")
        if not (isinstance(grouper, dict) and isinstance(head, dict)):
            return
        grouper_cls, head_cls = GROUPERS.get(grouper["type"]), HEADS.get(head["type"])
        if grouper_cls.out_axes != head_cls.in_axes:
            yield (
                f"{grouper_cls.__name__} emits features with axes {grouper_cls.out_axes}, but "
                f"{head_cls.__name__} expects {head_cls.in_axes}. Use a grouper and a head that "
                "agree; a depth axis is never added or squeezed implicitly."
            )

    def _seed_predictions(
        self, observation: Observation
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Raw head outputs, the ``(B, M, 3)`` seeds and their ``(B, M)`` indices into the cloud."""
        feats = self.backbone(observation.points, observation.points.coords)
        cloud, dense = observation.points.dense(), feats.dense()
        index = furthest_point_sample(cloud, self.num_seeds)
        seeds, seed_feats = gather_points(cloud, index), gather_points(dense, index)
        return self.head(self.grouper(seeds, seed_feats)), seeds, index

    def loss(self, observation: Observation, supervision: Supervision) -> dict[str, Tensor]:
        raw, _, index = self._seed_predictions(observation)
        labels = supervision.point_labels  # aligned with observation.points; pick the seeds
        cell = torch.gather(labels["cell"].reshape(len(index), -1), 1, index)
        width = torch.gather(labels["width"].reshape(len(index), -1), 1, index)
        return self.head.loss(raw, cell.flatten(), width.flatten())

    @torch.no_grad()
    def predict(self, observation: Observation) -> list[GraspGroup]:
        raw, seeds, _ = self._seed_predictions(observation)
        # A real method predicts the approach; the toy takes template views in order.
        views = fibonacci_views(self.num_seeds).to(seeds)
        approach = (-views).repeat(seeds.shape[0], 1)
        grasps = self.head.decode(raw, seeds.flatten(0, 1), approach)
        return [
            replace(grasps[b * self.num_seeds : (b + 1) * self.num_seeds].validate(), key=key)
            for b, key in enumerate(observation.keys)
        ]


@POSTPROCESSORS.register()
class ToyCentreFilter(PostProcessor):
    """Keeps the grasps within ``radius`` of the centre of the observed cloud.

    A post-processor may read the observation of its frame (here: the points), never
    ground truth. List it in ``evaluation.variants`` to score predictions with and
    without it side by side.
    """

    status = "example"

    def __init__(self, radius: float = 0.15) -> None:
        self.radius = radius

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        centre = observation.points.coords.mean(dim=0)
        return grasps[(grasps.translation - centre).norm(dim=-1) <= self.radius]
