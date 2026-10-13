"""Feature propagation as an explicit adapter.

A ``subsampled`` backbone (PointNet++ set abstraction) describes a few hundred points;
a stage that scores every scene point needs ``dense`` features. Bridging the two means
inventing features for points the encoder never looked at, which changes what the
model can express. In this code base that step is never implicit: a consumer that
receives features on the wrong set raises, and the one sanctioned bridge is this
component, written into the config::

    backbone:
      type: FeaturePropagation
      backbone: {type: PointNet2}
      mlp: [256, 256]

It is the feature-propagation level of PointNet++ (Qi et al., NeurIPS 2017, §3.4)
applied once, from the wrapped backbone's output set straight to the input points:
inverse-distance interpolation over the three nearest known points, concatenation with
the input features as a skip connection, then a shared MLP.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from torch import Tensor

from grasppanda.models.base import Backbone
from grasppanda.registry import BACKBONES, Cfg
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import check_features, cloud_sizes
from .pointnet2 import _Propagation

__all__ = ["FeaturePropagation"]


@BACKBONES.register()
class FeaturePropagation(Backbone):
    """Turn a ``subsampled`` backbone into a ``dense`` one by learned interpolation.

    The defaults are one PointNet++ propagation level as configured in
    GraspNet-baseline: three neighbours, inverse-distance weights (``p = 1``), a
    two-layer shared MLP of width 256 with batch norm and ReLU.

    Args:
        backbone: The wrapped encoder. It must declare ``output = "subsampled"`` and
            return features on a child of the set it was given.
        mlp: Layer widths of the shared MLP applied after interpolation; the last one
            is ``out_channels``.
        neighbours: Number of nearest known points interpolated per input point. Every
            cloud's subsample must hold at least this many points.
        interpolation_power: Exponent of the inverse-distance weights: 1 in the PyTorch
            PointNet++ lineage, 2 in the paper.
        skip_input: Concatenate the input features to the interpolated ones before the
            MLP, as PointNet++ does at its last propagation level. Has no effect when
            the wrapped backbone takes coordinates only.
        in_channels: Leave unset to take it from the wrapped backbone. A parent that
            injects it has it passed down, and the wrapped backbone's config must agree.
    """

    output = "dense"
    status = "tested"
    source = "PointNet++ feature propagation (Qi et al., NeurIPS 2017)"

    def __init__(
        self,
        backbone: Cfg[Backbone],
        mlp: Sequence[int] = (256, 256),
        neighbours: int = 3,
        interpolation_power: float = 1.0,
        skip_input: bool = True,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        if neighbours < 1:
            raise ValueError(f"FeaturePropagation: neighbours must be positive, got {neighbours}")
        if len(mlp) == 0:
            raise ValueError("FeaturePropagation: mlp needs at least one layer")
        injected = {} if in_channels is None else {"in_channels": in_channels}
        self.backbone = BACKBONES.build(backbone, **injected)
        inner = type(self.backbone).__name__
        if getattr(self.backbone, "output", None) != "subsampled":
            raise ValueError(
                f"FeaturePropagation wraps backbones that return subsampled features, but "
                f"{inner} declares output={getattr(self.backbone, 'output', None)!r}. Use "
                f"{inner} directly."
            )
        if in_channels is not None and self.backbone.in_channels != in_channels:
            raise ValueError(
                f"FeaturePropagation was given in_channels={in_channels}, but the wrapped "
                f"{inner} has in_channels={self.backbone.in_channels}."
            )
        self.in_channels = self.backbone.in_channels
        self.neighbours = neighbours
        self.skip_input = skip_input and self.in_channels > 0
        width = self.backbone.out_channels + (self.in_channels if self.skip_input else 0)
        self.fp = _Propagation(width, mlp, power=interpolation_power, neighbours=neighbours)
        self.out_channels = mlp[-1]

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Reject a dense inner backbone before anything is built."""
        inner = node.get("backbone")
        if not (isinstance(inner, dict) and "type" in inner):
            return
        inner_cls = BACKBONES.get(inner["type"])
        if getattr(inner_cls, "output", None) != "subsampled":
            yield (
                f"FeaturePropagation wraps backbones that return subsampled features, but "
                f"{inner_cls.__name__} declares output="
                f"{getattr(inner_cls, 'output', None)!r}. Use {inner_cls.__name__} directly."
            )

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        name = type(self).__name__
        feats = check_features(points, feats, self.in_channels, user=name)
        coarse = self.backbone(points, feats)
        inner = type(self.backbone).__name__
        if coarse.points.parent_id != points.id:
            raise ContractError(
                f"{name} expects {inner} to return features on a subsample of the point set "
                f"it was given (#{points.id}), got features on set #{coarse.points.id} with "
                f"parent #{coarse.points.parent_id}."
            )
        sizes, coarse_sizes = cloud_sizes(points), cloud_sizes(coarse.points)
        if len(sizes) != len(coarse_sizes):
            raise ContractError(
                f"{name}: the input holds {len(sizes)} clouds but the subsample returned by "
                f"{inner} describes {len(coarse_sizes)}. A subsample of a batch must carry "
                "batch offsets."
            )
        if min(coarse_sizes) < self.neighbours:
            raise ContractError(
                f"{name} interpolates over {self.neighbours} neighbours, but {inner} returned "
                f"only {min(coarse_sizes)} points for cloud "
                f"{coarse_sizes.index(min(coarse_sizes))}."
            )
        dense = self.fp(
            points.coords,
            sizes,
            feats if self.skip_input else None,
            coarse.points.coords,
            coarse_sizes,
            coarse.feats,
        )
        return PointFeatures(dense, points)
