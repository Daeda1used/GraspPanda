"""Adapters between crops with and without a depth axis.

A head bundle states the axes it reads: the heads of GSNet, EconomicGrasp and FineGrasp
take ``(seed, channel)``, the head of GraspNet-baseline and Scale-Balanced-Grasp takes
``(seed, depth, channel)``. The contract rules never add or drop an axis on their own.
The two groupers here do it explicitly, as a wrapper around another grouper, so that the
crop of one family can be tried under the head of the other:

* :class:`DepthEmbedGroup` gives a ``(seed, channel)`` crop a depth axis: the crop is
  repeated for every depth bin, a learned embedding of the bin is added, and a shared
  linear layer follows. The crop itself does not depend on the depth; only the embedding
  tells the bins apart.
* :class:`DepthPoolGroup` removes the depth axis of a ``(seed, depth, channel)`` crop by
  a maximum or a mean over the bins.

No published model is such an adapter; they exist to make swaps possible.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.registry import GROUPERS, INJECTED, Cfg
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures

__all__ = ["DepthEmbedGroup", "DepthPoolGroup"]


def _inner_problems(wrapper: type, inner: Any) -> Iterator[str]:
    """What the wrapped grouper must declare; works on classes and on built groupers."""
    name = inner.__name__ if isinstance(inner, type) else type(inner).__name__
    if getattr(inner, "support", None) != wrapper.support:
        yield (
            f"{wrapper.__name__} wraps a grouper with support = '{wrapper.support}', but "
            f"{name} declares support = {getattr(inner, 'support', None)!r}."
        )
    if getattr(inner, "out_axes", None) != wrapper.inner_axes:
        yield (
            f"{wrapper.__name__} wraps a grouper that emits {wrapper.inner_axes}, but {name} "
            f"emits {getattr(inner, 'out_axes', None)}."
        )


def _node(wrapper: type, grouper: Any) -> dict[str, Any]:
    if not (isinstance(grouper, Mapping) and "type" in grouper):
        raise TypeError(
            f"{wrapper.__name__} builds the grouper it wraps and needs it as a config node "
            f"with a `type`, got {type(grouper).__name__}"
        )
    problems = list(_inner_problems(wrapper, GROUPERS.get(grouper["type"])))
    if problems:
        raise ContractError(" ".join(problems))
    return dict(grouper)


class _Wrapper(Grouper):
    inner_axes: tuple[str, ...]

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Reject a wrapped grouper of the wrong kind before anything is built."""
        inner = node.get("grouper")
        if isinstance(inner, Mapping) and "type" in inner:
            yield from _inner_problems(cls, GROUPERS.get(inner["type"]))

    @classmethod
    def required_seed_channels(cls, options: Mapping[str, Any]) -> int | None:
        """The seed width the wrapped grouper needs: the wrapper hands the seeds on."""
        inner = options.get("grouper")
        if not (isinstance(inner, Mapping) and inner.get("type") in GROUPERS):
            return None
        return GROUPERS.get(inner["type"]).required_seed_channels(inner)


@GROUPERS.register()
class DepthEmbedGroup(_Wrapper):
    """Give a ``(seed, channel)`` crop among the seeds a depth axis by a learned embedding.

    ``out[s, d] = ReLU(W (crop[s] + e[d]) + b)`` with one learned vector ``e[d]`` per
    depth bin.

    Args:
        grouper: The wrapped grouper: support ``seeds``, axes ``(seed, channel)``.
        num_depth: Size of the depth axis; the ``num_depth`` of the head bundle.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "seeds"
    inner_axes = ("seed", "channel")
    out_axes = ("seed", "depth", "channel")
    status = "tested"
    source = "GraspPanda (axis adapter; no published model)"

    def __init__(
        self, grouper: Cfg[Grouper], num_depth: int = 4, in_channels: int = INJECTED
    ) -> None:
        super().__init__()
        if num_depth < 1:
            raise ValueError(f"DepthEmbedGroup needs num_depth >= 1, got {num_depth}")
        self.grouper = GROUPERS.build(_node(type(self), grouper), in_channels=in_channels)
        if getattr(self.grouper, "uses_seed_features", False):
            raise ContractError(
                f"DepthEmbedGroup cannot wrap {type(self.grouper).__name__}, which reads the "
                "seed features as a separate argument"
            )
        self.num_depth, self.in_channels = num_depth, in_channels
        self.out_channels = self.grouper.out_channels
        self.embedding = nn.Parameter(torch.zeros(num_depth, self.out_channels))
        nn.init.normal_(self.embedding, std=0.02)
        self.mix = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, num_depth, out_channels)`` with axes ``(seed, depth, channel)``."""
        crop = self.grouper(support, centres, rotation)
        crop.expect(self.inner_axes, user=type(self).__name__)
        feats = torch.relu(self.mix(crop.feats[:, None, :] + self.embedding[None]))
        return GroupFeatures(feats, self.out_axes)


@GROUPERS.register()
class DepthPoolGroup(_Wrapper):
    """Reduce the depth axis of a ``(seed, depth, channel)`` crop of the scene.

    Args:
        grouper: The wrapped grouper: support ``scene``, axes ``(seed, depth, channel)``.
        pool: ``max`` or ``mean`` over the depth bins.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "scene"
    inner_axes = ("seed", "depth", "channel")
    out_axes = ("seed", "channel")
    status = "tested"
    source = "GraspPanda (axis adapter; no published model)"

    def __init__(
        self,
        grouper: Cfg[Grouper],
        pool: Literal["max", "mean"] = "max",
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if pool not in ("max", "mean"):
            raise ValueError(f"DepthPoolGroup: unknown pool '{pool}'")
        self.grouper = GROUPERS.build(_node(type(self), grouper), in_channels=in_channels)
        self.pool, self.in_channels = pool, in_channels
        self.out_channels = self.grouper.out_channels
        #: Whether ``forward`` takes the seed features: as the wrapped grouper does.
        self.uses_seed_features = bool(getattr(self.grouper, "uses_seed_features", False))

    def forward(
        self,
        support: PointFeatures,
        centres: Tensor,
        rotation: Tensor,
        seed_feats: Tensor | None = None,
    ) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        if self.uses_seed_features:
            crop = self.grouper(support, centres, rotation, seed_feats)
        else:
            crop = self.grouper(support, centres, rotation)
        crop.expect(self.inner_axes, user=type(self).__name__)
        feats = crop.feats.amax(dim=1) if self.pool == "max" else crop.feats.mean(dim=1)
        return GroupFeatures(feats, self.out_axes)
