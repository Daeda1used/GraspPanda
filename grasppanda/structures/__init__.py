"""Typed data contracts. Plain dataclasses holding tensors; no dependency on any model."""

from .common import ContractError, Gripper, Mask, Provenance, SampleKey
from .compact_labels import COMPACT_LABELS, CompactGraspLabels
from .grasp_group import GRASPNET_ARRAY_WIDTH, GraspGroup
from .observation import Batch, ObjectGraspLabels, Observation, Supervision, collate
from .points import PointFeatures, PointSet
from .stages import ApproachSet, GroupFeatures, SeedSet

__all__ = [
    "COMPACT_LABELS",
    "GRASPNET_ARRAY_WIDTH",
    "ApproachSet",
    "Batch",
    "CompactGraspLabels",
    "ContractError",
    "GraspGroup",
    "Gripper",
    "GroupFeatures",
    "Mask",
    "ObjectGraspLabels",
    "Observation",
    "PointFeatures",
    "PointSet",
    "Provenance",
    "SampleKey",
    "SeedSet",
    "Supervision",
    "collate",
]
