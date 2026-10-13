"""Grasp non-maximum suppression as an inference-time post-processor.

Some reference test scripts thin their predictions before dumping them (HGGD: 0.03 m
and 30°, after its collision filter). The benchmark evaluator applies the same rule
again on whatever it is given, so on its own this step does not change a score; it
matters for what a robot receives, and for comparing a dump with a reference dump.

This is :func:`grasppanda.ops.grasp_nms`. The evaluator keeps its own pinned copy of
the rule, so nothing configured here can change how predictions are scored.
"""

from __future__ import annotations

import math

from grasppanda.models.base import PostProcessor
from grasppanda.ops import grasp_nms
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation

__all__ = ["GraspNMS"]


@POSTPROCESSORS.register()
class GraspNMS(PostProcessor):
    """Keep the best grasp of every cluster of near-duplicates.

    Grasps are visited in descending score order (ties keep their input order). A kept
    grasp removes every later grasp that is closer than ``translation_threshold``
    **and** rotated by less than ``rotation_threshold``. The result is sorted by
    descending score.

    Args:
        translation_threshold: Distance between grasp points, metres.
        rotation_threshold: Angle between grasp rotations, degrees.
    """

    status = "tested"
    source = "Grasp NMS rule of graspnetAPI"

    def __init__(
        self, translation_threshold: float = 0.03, rotation_threshold: float = 30.0
    ) -> None:
        if translation_threshold < 0 or not 0 <= rotation_threshold <= 180:
            raise ValueError(
                "translation_threshold must not be negative and rotation_threshold must be "
                f"in [0, 180] degrees; got {translation_threshold} and {rotation_threshold}"
            )
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        keep = grasp_nms(
            grasps.translation,
            grasps.rotation,
            grasps.score,
            translation_threshold=self.translation_threshold,
            rotation_threshold=math.radians(self.rotation_threshold),
        )
        return grasps[keep]
