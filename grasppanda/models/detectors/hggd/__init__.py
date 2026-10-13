"""HGGD and the stages that only it uses.

Only the detector is a registered component. Its stages stay private to this package
until a second method needs one of them (the heat-map network is the likely first:
RNGNet reuses it), at which point that stage becomes a registered slot.
"""

from .anchor_head import AnchorHead
from .detector import HGGD, StageOutputs
from .heatmap import HeatmapStage, HeatmapTargets
from .image_net import ImageNet
from .local_net import LocalNet
from .options import (
    AnchorOptions,
    EncoderOptions,
    ImageOptions,
    LocalOptions,
    ProposalOptions,
    RegionOptions,
)
from .region import RegionStage
from .structures import ImagePredictions, PartialGraspSet, RegionDraws, RegionSet

__all__ = [
    "HGGD",
    "AnchorHead",
    "AnchorOptions",
    "EncoderOptions",
    "HeatmapStage",
    "HeatmapTargets",
    "ImageNet",
    "ImageOptions",
    "ImagePredictions",
    "LocalNet",
    "LocalOptions",
    "PartialGraspSet",
    "ProposalOptions",
    "RegionDraws",
    "RegionOptions",
    "RegionSet",
    "RegionStage",
    "StageOutputs",
]
