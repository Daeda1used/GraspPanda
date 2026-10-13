"""Crops at several radii among the seeds, fused per seed by attention over the scales.

The grouping stage of FineGrasp ("multi-range attention"). Every seed is cropped once
per radius by a complete grouper of its own, the *branch*, so each scale yields one
feature vector per seed. The vectors of a seed form a short sequence, one token per
scale, that passes through a few post-norm transformer encoder layers; a learned
soft-max weight per token then reduces the sequence to one vector:

    F = Encoder([f_1, …, f_S]),    out = Σ_s softmax_s(w · F_s) F_s .

What is new here is the fusion. The branches are ordinary registered groupers: FineGrasp
uses EconomicGrasp's :class:`~.cylinder_interaction.CylinderInteractionGroup` unchanged,
at radii 0.25, 0.5, 0.75 and 1 times 0.07 m, and its encoder layers start with the same
attention block as that grouper
(:class:`~grasppanda.models.interaction.TokenInteraction`). Any grouper that crops among
the seeds and emits ``(seed, channel)`` features can be a branch, and the result can go
to any head bundle that takes those axes.

Scale-Balanced-Grasp also crops at several radii, but over the scene cloud, with one
shared MLP per radius and a single linear layer as fusion; that is the ``radius_scales``
option of :class:`~.depth_cylinder.DepthCylinderGroup`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Grouper
from grasppanda.models.interaction import TokenInteraction
from grasppanda.registry import GROUPERS, INJECTED, Cfg, component_parameters
from grasppanda.structures import ContractError, GroupFeatures, PointFeatures

__all__ = ["MultiRadiusGroup", "MultiScaleGroup", "ScaleAttentionFusion"]


def _branch_problems(branch: Any) -> Iterator[str]:
    """Contracts a branch must declare; works on grouper classes and on built groupers."""
    name = branch.__name__ if isinstance(branch, type) else type(branch).__name__
    if getattr(branch, "support", None) != MultiScaleGroup.support:
        yield (
            f"MultiScaleGroup crops among the seeds, but its branch {name} declares support = "
            f"{getattr(branch, 'support', None)!r}. Scales over the scene cloud are the "
            "`radius_scales` option of DepthCylinderGroup."
        )
    if getattr(branch, "out_axes", None) != MultiScaleGroup.out_axes:
        yield (
            f"MultiScaleGroup fuses one feature vector per seed and scale, but its branch {name} "
            f"emits axes {getattr(branch, 'out_axes', None)}."
        )
    if (
        "radius"
        not in component_parameters(branch if isinstance(branch, type) else type(branch))[0]
    ):
        yield f"the branch {name} has no `radius` for MultiScaleGroup to set."


class _EncoderLayer(nn.Module):
    """A post-norm transformer encoder layer over ``(N, T, channels)`` tokens.

    ``x ← LayerNorm(x + Dropout(Attention(x)))`` is a :class:`TokenInteraction` with a
    residual dropout; then ``x ← LayerNorm(x + Dropout(W₂ Dropout(ReLU(W₁ x))))``.
    """

    def __init__(
        self, channels: int, num_heads: int, feedforward_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.interaction = TokenInteraction(channels, num_heads, dropout, residual_dropout=dropout)
        self.expand = nn.Linear(channels, feedforward_channels)
        self.contract = nn.Linear(feedforward_channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the attention as the reference layer does (``nn.MultiheadAttention``).

        Its query, key and value maps are one ``(3C, C)`` matrix drawn uniformly within
        ``±sqrt(6 / (C + 3C))``, and all attention biases start at zero. The remaining
        layers keep the default of ``nn.Linear``.
        """
        attention = self.interaction
        bound = math.sqrt(6.0 / (4 * attention.channels))
        for projection in (attention.query, attention.key, attention.value):
            nn.init.uniform_(projection.weight, -bound, bound)
            nn.init.zeros_(projection.bias)
        nn.init.zeros_(attention.output.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.interaction(tokens)
        hidden = self.contract(self.dropout(torch.relu(self.expand(tokens))))
        return self.norm(tokens + self.residual_dropout(hidden))


class ScaleAttentionFusion(nn.Module):
    """Reduce ``(N, S, channels)`` scale tokens to ``(N, channels)``.

    The tokens of a row attend to each other in ``num_layers`` encoder layers; the result
    is their sum weighted by a soft-max over a learned score per token.

    Args:
        channels: Width of the tokens; kept.
        num_layers: Encoder layers.
        num_heads: Attention heads of every layer; must divide ``channels``.
        feedforward_channels: Hidden width of the feed-forward sub-layer.
        dropout: Dropout rate of the encoder layers (attention weights, both residual
            branches and inside the feed-forward sub-layer) during training.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_channels: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 0 or feedforward_channels < 1:
            raise ValueError(
                "ScaleAttentionFusion needs num_layers >= 0 and feedforward_channels >= 1; got "
                f"num_layers={num_layers}, feedforward_channels={feedforward_channels}"
            )
        self.channels = channels
        self.layers = nn.ModuleList(
            _EncoderLayer(channels, num_heads, feedforward_channels, dropout)
            for _ in range(num_layers)
        )
        self.score = nn.Linear(channels, 1)

    def encode(self, tokens: Tensor) -> Tensor:
        """The tokens after the encoder layers: ``(N, S, channels)``."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.channels:
            raise ValueError(
                f"expected tokens of shape (N, S, {self.channels}), got {tuple(tokens.shape)}"
            )
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens

    def weights(self, encoded: Tensor) -> Tensor:
        """``(N, S)`` weight of every scale; each row sums to one."""
        return self.score(encoded)[..., 0].softmax(dim=1)

    def forward(self, tokens: Tensor) -> Tensor:
        encoded = self.encode(tokens)
        return (encoded * self.weights(encoded)[..., None]).sum(dim=1)


@GROUPERS.register()
class MultiScaleGroup(Grouper):
    """One crop per radius among the seeds, fused by :class:`ScaleAttentionFusion`.

    Args:
        branch: The grouper that crops at one radius; one copy with its own weights is
            built per scale. It must crop among the seeds, emit ``(seed, channel)``
            features and take a ``radius``, which is set here and must not be repeated
            in its config. FineGrasp: ``CylinderInteractionGroup`` with its defaults.
        radius: Largest crop radius in metres.
        radius_scales: One crop per entry, with radius ``radius * scale``, in this order
            (the order of the tokens).
        fusion_layers: Transformer encoder layers of the fusion.
        fusion_heads: Attention heads of each.
        fusion_feedforward_channels: Hidden width of their feed-forward sub-layers.
        fusion_dropout: Dropout rate inside them during training.
        in_channels: Channels of the seed features; injected by the detector.
    """

    support = "seeds"
    out_axes = ("seed", "channel")
    status = "tested"
    source = "FineGrasp (arXiv 2507.05978)"

    def __init__(
        self,
        branch: Cfg[Grouper],
        radius: float = 0.07,
        radius_scales: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_feedforward_channels: int = 1024,
        fusion_dropout: float = 0.1,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        scales = tuple(float(scale) for scale in radius_scales)
        if radius <= 0 or not scales or min(scales) <= 0:
            raise ValueError(
                "MultiScaleGroup needs radius > 0 and positive radius_scales; got "
                f"radius={radius}, radius_scales={list(scales)}"
            )
        if not (isinstance(branch, Mapping) and "type" in branch):
            raise TypeError(
                "MultiScaleGroup builds one branch per scale and needs the branch as a config "
                f"node with a `type`, got {type(branch).__name__}"
            )
        node = dict(branch)
        branch_class = GROUPERS.get(node["type"])
        problems = list(_branch_problems(branch_class))
        if problems:
            raise ContractError(" ".join(problems))
        # A resolved config carries the default radius of the branch; any other value
        # would be a second owner of the crop radius (`check_config` reports that for
        # configs, where a default written out by hand can still be told apart).
        default = component_parameters(branch_class)[0]["radius"].default
        if node.pop("radius", default) != default:
            raise ContractError(
                "the radius of every branch is `radius * radius_scales[i]` of MultiScaleGroup; "
                f"remove `radius: {branch['radius']}` from the branch."
            )
        self.radius = float(radius)
        self.radius_scales = scales
        self.in_channels = in_channels
        self.branches = nn.ModuleList(
            GROUPERS.build(node, in_channels=in_channels, radius=crop_radius)
            for crop_radius in self.radii
        )
        self.out_channels = self.branches[0].out_channels
        self.fusion = ScaleAttentionFusion(
            self.out_channels,
            fusion_layers,
            fusion_heads,
            fusion_feedforward_channels,
            fusion_dropout,
        )

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Reject a branch of another kind before anything is built."""
        branch = node.get("branch")
        if not (isinstance(branch, Mapping) and "type" in branch):
            return
        yield from _branch_problems(GROUPERS.get(branch["type"]))
        if "radius" in branch:
            yield (
                "the radius of every branch is `radius * radius_scales[i]` of MultiScaleGroup; "
                "remove `radius` from the branch."
            )

    @property
    def radii(self) -> tuple[float, ...]:
        """Radius of every crop, metres."""
        return tuple(self.radius * scale for scale in self.radius_scales)

    def scale_features(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> Tensor:
        """The crop of every scale before the fusion: ``(M, scales, out_channels)``."""
        crops = []
        for branch in self.branches:
            crop = branch(support, centres, rotation)
            crop.expect(self.out_axes, user=type(self).__name__)
            crops.append(crop.feats)
        return torch.stack(crops, dim=1)

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        fused = self.fusion(self.scale_features(support, centres, rotation))
        return GroupFeatures(fused, self.out_axes)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, radius_scales={list(self.radius_scales)}"


@GROUPERS.register()
class MultiRadiusGroup(MultiScaleGroup):
    """One crop per radius among the seeds, fused without attention.

    The branches are those of :class:`MultiScaleGroup`; the scale vectors of a seed are
    concatenated and reduced by one linear layer (``linear``, the multi-scale grouping of
    PointNet++), or reduced by a maximum or a mean over the scales.

    Args:
        branch: The grouper that crops at one radius; one copy with its own weights is
            built per scale. It must crop among the seeds, emit ``(seed, channel)``
            features and take a ``radius``, which is set here.
        radius: Largest crop radius in metres.
        radius_scales: One crop per entry, with radius ``radius * scale``.
        fusion: ``linear``, ``max`` or ``mean``.
        in_channels: Channels of the seed features; injected by the detector.
    """

    status = "tested"
    source = (
        "Multi-scale grouping of PointNet++ (Qi et al., NeurIPS 2017) on the seed crops of "
        "the grasp methods, written here"
    )

    def __init__(
        self,
        branch: Cfg[Grouper],
        radius: float = 0.07,
        radius_scales: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        fusion: Literal["linear", "max", "mean"] = "linear",
        in_channels: int = INJECTED,
    ) -> None:
        if fusion not in ("linear", "max", "mean"):
            raise ValueError(
                f"MultiRadiusGroup: unknown fusion '{fusion}'; choose linear, max or mean "
                "(attention over the scales is MultiScaleGroup)"
            )
        super().__init__(branch, radius, radius_scales, fusion_layers=0, in_channels=in_channels)
        self.fusion_kind = fusion
        self.fusion = (
            nn.Linear(len(self.radius_scales) * self.out_channels, self.out_channels)
            if fusion == "linear"
            else None
        )

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        """Crop features ``(M, out_channels)`` with axes ``(seed, channel)``."""
        scales = self.scale_features(support, centres, rotation)
        if self.fusion is not None:
            fused = self.fusion(scales.flatten(1))
        elif self.fusion_kind == "max":
            fused = scales.amax(dim=1)
        else:
            fused = scales.mean(dim=1)
        return GroupFeatures(fused, self.out_axes)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, fusion={self.fusion_kind!r}"
