"""Clean-room GraspClutter6D evaluator (protocol ``graspclutter6d-sv-v1``).

All scoring stages live in :mod:`grasppanda.evaluation.graspnet1b`; this package adds
the benchmark's constants, its ground-truth reader and the official tool's dump layout.
"""

from .evaluator import GRASPCLUTTER6D_SOURCES, GRASPCLUTTER6D_SV_V1, GraspClutter6DEvaluator
from .scene import (
    CAMERAS,
    ClutterGroundTruth,
    export_official_layout,
    image_id,
    prediction_file,
)

__all__ = [
    "CAMERAS",
    "GRASPCLUTTER6D_SOURCES",
    "GRASPCLUTTER6D_SV_V1",
    "ClutterGroundTruth",
    "GraspClutter6DEvaluator",
    "export_official_layout",
    "image_id",
    "prediction_file",
]
