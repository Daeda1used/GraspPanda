"""Clean-room GraspNet-1Billion evaluator (protocol ``graspnet1b-sv-v1``).

Reads top to bottom:

* :mod:`.protocol` owns every constant;
* :mod:`.frame` is the benchmark's order of operations for one frame;
* :mod:`.nms`, :mod:`.collision`, :mod:`.force_closure` and :mod:`.sdf` are its stages;
* :mod:`.scene`, :mod:`.poses`, :mod:`.models`, :mod:`.dexnet_pickle` and :mod:`.cache`
  turn the dataset into what the stages consume;
* :mod:`.evaluator` is the registered component: files in, :class:`EvaluationResult` out.

The scoring code depends on NumPy and SciPy only and runs on NumPy 1.26 and 2.x.
"""

from .cache import GRASPNET1B_SOURCES, CacheError, ObjectAsset, ObjectCache, ObjectSources
from .evaluator import EVALUATOR_VERSION, EvaluationError, GraspNet1BEvaluator, SceneEvaluator
from .frame import FrameResult, evaluate_frame
from .protocol import GRASPNET1B_SV_V1, Protocol, TableSlab

__all__ = [
    "EVALUATOR_VERSION",
    "GRASPNET1B_SOURCES",
    "GRASPNET1B_SV_V1",
    "CacheError",
    "EvaluationError",
    "FrameResult",
    "GraspNet1BEvaluator",
    "ObjectAsset",
    "ObjectCache",
    "ObjectSources",
    "Protocol",
    "SceneEvaluator",
    "TableSlab",
    "evaluate_frame",
]
