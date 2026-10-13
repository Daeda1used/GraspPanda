"""Furthest-point subsampling as an explicit adapter.

The mirror image of :class:`~.feature_propagation.FeaturePropagation`. A ``dense``
backbone (a sparse U-Net, a point transformer with its decoder) describes every scene
point; a stage written for a PointNet++-style encoder expects features on a few hundred
points that *are* the seeds. Going from the first to the second means choosing which
points survive, which decides where grasps can be predicted at all. In this code base
that choice is never implicit: a consumer that receives features on the wrong set
raises, and the one sanctioned bridge is this component, written into the config::

    backbone:
      type: FurthestPointSubsample
      backbone: {type: SparseUNet14}
      npoint: 1024

It keeps ``npoint`` points of every cloud by furthest point sampling over the *scene
coordinates* and returns the wrapped backbone's features at those points. Nothing is
learned and nothing is interpolated.

The sample is the one a PointNet++ encoder would have ended on. Furthest point samples
are nested (the first ``k`` do not depend on how many more are drawn), so with the same
``fps_skip_near_origin`` and ``fps_tie_break`` the points kept here are exactly the
points :class:`~.pointnet2.PointNet2` returns for ``npoints[k] = npoint``, in the same
order. Swapping ``PointNet2`` for a wrapped dense encoder therefore changes the features
of the seeds and leaves the seeds where they were.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

from torch import Tensor

from grasppanda.models.base import Backbone, configured_width
from grasppanda.registry import BACKBONES, Cfg
from grasppanda.structures import ContractError, PointFeatures, PointSet

from ._packed import cloud_sizes, fps_packed, offsets_of

__all__ = ["FurthestPointSubsample"]


@BACKBONES.register()
class FurthestPointSubsample(Backbone):
    """Turn a ``dense`` backbone into a ``subsampled`` one by furthest point sampling.

    The defaults give the 1024 seeds of GraspNet-baseline, sampled with the rules of
    :class:`~.pointnet2.PointNet2`.

    Args:
        backbone: The wrapped encoder. It must declare ``output = "dense"`` and return
            features on the set it was given.
        npoint: Points kept per cloud. Every cloud must hold at least this many.
        fps_skip_near_origin: Reproduce the sampling kernel of the PyTorch PointNet++
            lineage, which never selects a point with squared norm ≤ 1e-3 after the
            first one. It only matters for clouds that pass within 3.2 cm of the frame
            origin.
        fps_tie_break: Which point is sampled when several are equally far from the
            points chosen so far: ``lowest_index`` takes the first, ``pointnet2_cuda``
            follows the order of that lineage's sampling kernel (what the
            GraspNet-baseline reference config uses).
        in_channels: Leave unset to take it from the wrapped backbone. A parent that
            injects it has it passed down, and the wrapped backbone's config must agree.
    """

    output = "subsampled"
    status = "tested"
    source = "PointNet++ furthest point sampling (Qi et al., NeurIPS 2017)"

    @classmethod
    def output_channels(cls, options: Mapping[str, Any]) -> int | None:
        """The width of the wrapped backbone: its features are kept as they are."""
        return configured_width(options["backbone"])

    def __init__(
        self,
        backbone: Cfg[Backbone],
        npoint: int = 1024,
        fps_skip_near_origin: bool = True,
        fps_tie_break: Literal["lowest_index", "pointnet2_cuda"] = "lowest_index",
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        if npoint < 1:
            raise ValueError(f"FurthestPointSubsample: npoint must be positive, got {npoint}")
        if fps_tie_break not in ("lowest_index", "pointnet2_cuda"):
            raise ValueError(f"unknown fps_tie_break '{fps_tie_break}'")
        injected = {} if in_channels is None else {"in_channels": in_channels}
        self.backbone = BACKBONES.build(backbone, **injected)
        inner = type(self.backbone).__name__
        if getattr(self.backbone, "output", None) != "dense":
            raise ValueError(
                f"FurthestPointSubsample wraps backbones that return dense features, but "
                f"{inner} declares output={getattr(self.backbone, 'output', None)!r}. Use "
                f"{inner} directly."
            )
        if in_channels is not None and self.backbone.in_channels != in_channels:
            raise ValueError(
                f"FurthestPointSubsample was given in_channels={in_channels}, but the wrapped "
                f"{inner} has in_channels={self.backbone.in_channels}."
            )
        self.in_channels = self.backbone.in_channels
        self.out_channels = self.backbone.out_channels
        self.npoint = npoint
        self.fps_skip_near_origin = fps_skip_near_origin
        self.fps_tie_break = fps_tie_break

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Reject a sub-sampling inner backbone before anything is built."""
        inner = node.get("backbone")
        if not (isinstance(inner, dict) and "type" in inner):
            return
        inner_cls = BACKBONES.get(inner["type"])
        if getattr(inner_cls, "output", None) != "dense":
            yield (
                f"FurthestPointSubsample wraps backbones that return dense features, but "
                f"{inner_cls.__name__} declares output="
                f"{getattr(inner_cls, 'output', None)!r}. Use {inner_cls.__name__} directly."
            )

    def forward(self, points: PointSet, feats: Tensor | None = None) -> PointFeatures:
        name = type(self).__name__
        sizes = cloud_sizes(points)
        if len(points) == 0 or min(sizes) < self.npoint:
            smallest = min(sizes) if sizes else 0
            raise ContractError(
                f"{name} keeps {self.npoint} points per cloud, but cloud "
                f"{sizes.index(smallest) if sizes else 0} of the batch has only {smallest}. "
                "Feed larger clouds or configure a smaller npoint; points are never duplicated "
                "silently."
            )
        dense = self.backbone(points, feats)
        dense.expect(points, user=f"{name} (wraps a backbone with output = 'dense')")
        rows = fps_packed(
            points.coords,
            sizes,
            self.npoint,
            skip_near_origin=self.fps_skip_near_origin,
            tie_break=self.fps_tie_break,
        )
        offsets = offsets_of([self.npoint] * len(sizes), rows.device)
        return PointFeatures(dense.feats[rows], points.subset(rows, offsets))

    def extra_repr(self) -> str:
        return f"npoint={self.npoint}, fps_tie_break={self.fps_tie_break!r}"
