"""The registered GraspClutter6D evaluator (protocol ``graspclutter6d-sv-v1``).

It shares every stage with the GraspNet-1Billion evaluator and differs exactly where
the official ``graspclutter6dAPI`` differs from ``graspnetAPI``:

==========================  =========================  ==============================
                            GraspNet-1Billion          GraspClutter6D
==========================  =========================  ==============================
frames per camera, scene    256                        13 (``image_id = 4·frame + c``)
width clip                  ``[0, 0.10]`` m            ``[0, 0.14]`` m
before NMS                  nothing                    drop grasps ``>= 0.05`` m from
                                                       every model point
collision scene             models + synthetic table   models only
object poses                float32 via Euler angles   float64 from ``scene_gt.json``
closing axis                must be unit (else error)  normalised first
==========================  =========================  ==============================

**Dump layout.** By default the evaluator reads what ``grasppanda test`` writes, the
layout of every dump of this toolbox: ``<root>/scene_0100/<camera>/0003.npy`` (scene,
camera, view). The official tool names its files differently,
``<root>/000100/<camera>/000014.npy`` (scene, camera, image id). ``layout: official``
reads that, so a dump written for ``graspclutter6dAPI`` can be scored as it is, and
:func:`~.scene.export_official_layout` links a GraspPanda dump into the official
layout, so the official tool can score it.

One deliberate difference from the official tool remains: there, a frame without a
dump file silently scores zero. Here it is reported in ``EvaluationResult.missing``
and lowers coverage, as for every evaluator of this toolbox.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from grasppanda.evaluation.base import prediction_path
from grasppanda.evaluation.graspnet1b.cache import ObjectCache, ObjectSources
from grasppanda.evaluation.graspnet1b.evaluator import SceneEvaluator
from grasppanda.evaluation.graspnet1b.protocol import Protocol
from grasppanda.registry import EVALUATORS
from grasppanda.structures.common import Gripper, SampleKey

from .scene import ClutterGroundTruth, prediction_file

__all__ = ["GRASPCLUTTER6D_SOURCES", "GRASPCLUTTER6D_SV_V1", "GraspClutter6DEvaluator"]

#: graspclutter6dAPI evaluation path as of commit ``ddb2ee7``.
GRASPCLUTTER6D_SV_V1 = Protocol(
    name="graspclutter6d-sv-v1",
    gripper=Gripper(name="graspclutter6d-parallel-jaw", max_width=0.14),
    table=None,
    pre_nms_max_distance=0.05,
    normalize_closing_axis=True,
    frames_per_scene=13,
)

GRASPCLUTTER6D_SOURCES = ObjectSources(
    dataset="graspclutter6d",
    dex_model="dex_models/obj_{:06d}.pkl",
    model="models_m/obj_{:06d}.ply",
    shard="{:06d}",
)

_LAYOUTS = {"grasppanda": prediction_path, "official": prediction_file}


@EVALUATORS.register()
class GraspClutter6DEvaluator(SceneEvaluator):
    """GraspClutter6D single-view evaluation; see the module docstring for the protocol.

    Keys are ``SampleKey("graspclutter6d", scene, camera, frame)`` with ``camera`` one of
    ``realsense-d415``, ``realsense-d435``, ``azure-kinect``, ``zivid`` and ``frame`` in
    ``0 .. 12``.

    Args:
        data_root: Dataset root. Default: ``$GRASPPANDA_GRASPCLUTTER6D_ROOT`` and the
            other rules of :func:`grasppanda.utils.paths.dataset_root`.
        cache_dir: Root of derived data. Default: ``$GRASPPANDA_CACHE``.
        scene_timeout: Seconds a worker may spend on one scene before the run fails.
        start_method: How worker processes are created.
        layout: Naming of the prediction files. ``grasppanda`` is what ``grasppanda
            test`` and ``infer`` write (``scene_0100/<camera>/0003.npy``); ``official``
            is the layout of ``graspclutter6dAPI`` (``000100/<camera>/000014.npy``).
            The scores do not depend on it.
    """

    source = "graspclutter6dAPI"

    protocol = GRASPCLUTTER6D_SV_V1.name
    constants = GRASPCLUTTER6D_SV_V1
    sources = GRASPCLUTTER6D_SOURCES
    truth = ClutterGroundTruth
    prediction_file = staticmethod(prediction_path)

    def __init__(
        self,
        data_root: str | None = None,
        cache_dir: str | None = None,
        scene_timeout: float = 3600.0,
        start_method: Literal["forkserver", "spawn", "fork"] = "forkserver",
        layout: Literal["grasppanda", "official"] = "grasppanda",
    ) -> None:
        if layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(_LAYOUTS)}, got {layout!r}")
        super().__init__(data_root, cache_dir, scene_timeout, start_method)
        self.layout = layout

    def _tasks(
        self, predictions: str | Path, keys: Sequence[SampleKey], cache: ObjectCache
    ) -> list[Any]:
        locate = _LAYOUTS[self.layout]
        return [
            replace(task, prediction_file=locate)
            for task in super()._tasks(predictions, keys, cache)
        ]
