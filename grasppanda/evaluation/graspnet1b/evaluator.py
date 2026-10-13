"""The registered GraspNet-1Billion evaluator: prediction files in, ``EvaluationResult`` out.

Work is split by ``(scene, camera)``: the ground truth of a scene (object data, poses,
table) is loaded once and reused for its frames. With ``num_workers > 0`` every scene
runs in its own short-lived process (see :mod:`.parallel`); with ``num_workers = 0``
scenes run in the calling process, which is what tests and debuggers want.

What a result means:

* ``precision[key]`` is the ``(50, 6)`` precision-at-k table of one frame.
* ``AP`` is the mean of all entries of all scored tables; ``AP0.8`` and ``AP0.4`` are
  the means of the friction-0.8 and friction-0.4 columns. Values are fractions in
  ``[0, 1]``, exactly what the reference tool returns; papers print them times 100.
* A prediction file without rows is scored (all zeros). A missing file is *not*
  scored: its key goes to ``missing`` and coverage drops below one.
* Any other problem (unreadable or malformed prediction, inconsistent ground truth,
  a worker that died or timed out) raises :class:`EvaluationError` naming the frame.

:class:`SceneEvaluator` holds everything that does not depend on the dataset. A
benchmark supplies four things: its :class:`~.protocol.Protocol`, where its object
files live, a ground-truth class and the layout of its prediction dumps.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import scipy

from grasppanda.evaluation.base import EvaluationResult, Evaluator, prediction_path
from grasppanda.evaluation.breakdown import Tally, frame_tallies
from grasppanda.registry import EVALUATORS
from grasppanda.structures.common import SampleKey
from grasppanda.utils.paths import cache_root, dataset_root

from .cache import CACHE_VERSION, GRASPNET1B_SOURCES, ObjectCache, ObjectSources
from .frame import evaluate_frame
from .parallel import WorkerError, run_in_processes
from .protocol import GRASPNET1B_SV_V1, Protocol
from .scene import SceneGroundTruth

__all__ = ["EVALUATOR_VERSION", "EvaluationError", "GraspNet1BEvaluator", "SceneEvaluator"]

#: Version of the scoring code. Bump when any score could change.
EVALUATOR_VERSION = "1.0.0"

_TIE_RULE = "equal scores are ranked by row index in the prediction file, earlier row first"
_EMPTY_RULE = "a prediction without rows scores an all-zero table (the reference tool raises)"


class EvaluationError(RuntimeError):
    """Scoring failed; the message starts with the frame or scene it failed on."""


@dataclass(frozen=True)
class _SceneTask:
    """Everything a worker needs to score the requested frames of one scene."""

    data_root: str
    cache_directory: str
    predictions: str
    protocol: Protocol
    sources: ObjectSources
    truth: Callable[..., Any]
    prediction_file: Callable[[str, SampleKey], Path]
    scene: int
    camera: str
    frames: tuple[int, ...]

    def key(self, frame: int) -> SampleKey:
        return SampleKey(self.sources.dataset, self.scene, self.camera, frame)

    @property
    def label(self) -> str:
        return f"{self.sources.dataset}/scene_{self.scene:04d}/{self.camera}"


@dataclass
class _SceneOutcome:
    precision: dict[int, np.ndarray] = field(default_factory=dict)
    missing: list[int] = field(default_factory=list)
    empty: int = 0
    grasps: int = 0
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    start_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    #: AP shares of the scored frames, in total and per object id (see ``breakdown``).
    tally: Tally = field(default_factory=Tally)
    objects: dict[int, Tally] = field(default_factory=dict)


def _peak_rss_mb() -> float:
    """High-water mark of this process's resident memory (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _score_scene(task: _SceneTask) -> _SceneOutcome:
    """Score one scene. Runs in a worker process or, with ``num_workers = 0``, in the caller."""
    started, cpu_started = time.perf_counter(), time.process_time()
    outcome = _SceneOutcome(start_rss_mb=_peak_rss_mb())
    try:
        cache = ObjectCache(
            task.data_root,
            model_voxel_size=task.protocol.model_voxel_size,
            sources=task.sources,
            directory=task.cache_directory,
        )
        truth = task.truth(task.data_root, task.scene, task.camera, cache, task.protocol)
    except Exception as error:
        raise EvaluationError(f"{task.label}: cannot load ground truth: {error}") from error
    for frame in task.frames:
        key = task.key(frame)
        path = task.prediction_file(task.predictions, key)
        if not path.is_file():
            outcome.missing.append(frame)
            continue
        try:
            prediction = np.load(path)
            poses, table = truth.frame(frame)
            result = evaluate_frame(prediction, truth.objects, poses, table, task.protocol)
        except Exception as error:
            raise EvaluationError(f"{key}: {error}") from error
        outcome.precision[frame] = result.precision
        outcome.empty += len(prediction) == 0
        outcome.grasps += len(prediction)
        object_ids = [asset.object_id for asset in truth.objects]
        whole, parts = frame_tallies(
            owner=result.kept_owner,
            collision=result.collision,
            empty=result.empty,
            friction=result.friction,
            ranking=result.ranking,
            object_ids=object_ids,
            top_k=task.protocol.top_k,
            frictions=task.protocol.frictions,
        )
        outcome.tally.add(whole)
        for object_id in set(object_ids):  # present in the frame, whether grasped or not
            outcome.objects.setdefault(object_id, Tally()).frames += 1
        for object_id, part in parts.items():
            outcome.objects[object_id].add(part)
    outcome.seconds = time.perf_counter() - started
    outcome.cpu_seconds = time.process_time() - cpu_started
    outcome.peak_rss_mb = _peak_rss_mb()
    return outcome


def _sort_key(key: SampleKey) -> tuple[int, str, int]:
    return key.scene, key.camera, key.frame


class SceneEvaluator(Evaluator):
    """Dataset-independent part of the GraspNet-style evaluators.

    Subclasses set the class attributes and stay free of logic.

    Attributes:
        constants: The pinned protocol.
        sources: Where the dataset keeps each object's Dex-Net pickle and model.
        truth: Ground truth of one scene: ``truth(data_root, scene, camera, cache,
            protocol)`` with ``objects``, ``frame(index) -> (poses, table)`` and a static
            ``list_object_ids(data_root, scene, camera)``.
        prediction_file: Maps a prediction root and a key to the dump file of a frame.
    """

    constants: ClassVar[Protocol]
    sources: ClassVar[ObjectSources]
    truth: ClassVar[Any]
    prediction_file: ClassVar[Callable[[str, SampleKey], Path]]

    def __init__(
        self,
        data_root: str | None = None,
        cache_dir: str | None = None,
        scene_timeout: float = 3600.0,
        start_method: Literal["forkserver", "spawn", "fork"] = "forkserver",
    ) -> None:
        if scene_timeout <= 0:
            raise ValueError(f"scene_timeout must be positive, got {scene_timeout}")
        self.data_root = dataset_root(self.sources.dataset, data_root)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else cache_root()
        self.scene_timeout = float(scene_timeout)
        self.start_method = start_method

    def _tasks(
        self, predictions: str | Path, keys: Sequence[SampleKey], cache: ObjectCache
    ) -> list[_SceneTask]:
        if len(set(keys)) != len(keys):
            raise EvaluationError("the same frame was requested more than once")
        frames: dict[tuple[int, str], list[int]] = {}
        for key in keys:
            if key.dataset != self.sources.dataset:
                raise EvaluationError(
                    f"{key}: {type(self).__name__} scores '{self.sources.dataset}' frames, "
                    f"not '{key.dataset}'"
                )
            frames.setdefault((key.scene, key.camera), []).append(key.frame)
        return [
            _SceneTask(
                data_root=str(self.data_root),
                cache_directory=str(cache.directory),
                predictions=str(predictions),
                protocol=self.constants,
                sources=self.sources,
                truth=type(self).truth,
                prediction_file=type(self).prediction_file,
                scene=scene,
                camera=camera,
                frames=tuple(sorted(scene_frames)),
            )
            for (scene, camera), scene_frames in sorted(frames.items())
        ]

    def _open_cache(self) -> ObjectCache:
        if not self.data_root.is_dir():
            variable = f"GRASPPANDA_{self.sources.dataset.upper()}_ROOT"
            raise EvaluationError(
                f"dataset '{self.sources.dataset}' not found at {self.data_root}; "
                f"pass data_root or set {variable}"
            )
        return ObjectCache(
            self.data_root,
            model_voxel_size=self.constants.model_voxel_size,
            sources=self.sources,
            root=self.cache_dir,
        )

    def _fill_cache(self, cache: ObjectCache, tasks: Sequence[_SceneTask]) -> None:
        """Extract missing objects once, in the parent, so workers only ever read."""
        needed: set[int] = set()
        for task in tasks:
            try:
                needed.update(task.truth.list_object_ids(self.data_root, task.scene, task.camera))
            except (OSError, ValueError, KeyError) as error:
                raise EvaluationError(
                    f"{task.label}: cannot read the object list: {error}"
                ) from error
        cache.ensure(needed)

    def evaluate(
        self, predictions: str | Path, keys: Sequence[SampleKey], *, num_workers: int = 0
    ) -> EvaluationResult:
        """Score the frames ``keys`` whose predictions live under ``predictions``."""
        started = time.perf_counter()
        keys = list(keys)
        cache = self._open_cache()
        tasks = self._tasks(predictions, keys, cache)
        self._fill_cache(cache, tasks)
        cache_seconds = time.perf_counter() - started

        outcomes: dict[int, _SceneOutcome] = {}
        if num_workers <= 0:
            for index, task in enumerate(tasks):
                outcomes[index] = _score_scene(task)
        else:
            try:
                for index, outcome in run_in_processes(
                    _score_scene,
                    tasks,
                    num_workers=num_workers,
                    timeout=self.scene_timeout,
                    start_method=self.start_method,
                    preload=(type(self).__module__,),
                ):
                    outcomes[index] = outcome
            except WorkerError as error:
                task = tasks[error.task_index]
                message = str(error).removeprefix(f"{EvaluationError.__name__}: ")
                if not message.startswith(self.sources.dataset):  # died, timed out, failed early
                    message = f"{task.label}: {message}"
                if error.remote_traceback:
                    message += f"\n--- worker traceback ---\n{error.remote_traceback}"
                raise EvaluationError(message) from error

        scored: dict[SampleKey, np.ndarray] = {}
        missing: set[SampleKey] = set()
        for index, task in enumerate(tasks):
            outcome = outcomes[index]
            scored.update({task.key(f): table for f, table in outcome.precision.items()})
            missing.update(task.key(frame) for frame in outcome.missing)
        precision = {key: scored[key] for key in keys if key in scored}

        constants = self.constants
        metrics = dict.fromkeys(["AP", *(name for name, _ in constants.reported)], float("nan"))
        if scored:
            stack = np.stack([scored[key] for key in sorted(scored, key=_sort_key)])
            metrics["AP"] = float(stack.mean())
            for name, friction in constants.reported:
                metrics[name] = float(stack[:, :, constants.friction_column(friction)].mean())

        info = {
            "evaluator": type(self).__name__,
            "version": EVALUATOR_VERSION,
            "protocol": self.protocol,
            "constants": constants.as_dict(),
            "metric_unit": "fraction in [0, 1]; multiply by 100 for the figures papers print",
            "tie_rule": _TIE_RULE,
            "empty_prediction_rule": _EMPTY_RULE,
            "frames_scored": len(precision),
            "frames_missing": len(missing),
            "empty_predictions": int(sum(outcome.empty for outcome in outcomes.values())),
            "grasps_read": int(sum(outcome.grasps for outcome in outcomes.values())),
            "num_workers": max(num_workers, 0),
            "seconds": round(time.perf_counter() - started, 3),
            "seconds_cache": round(cache_seconds, 3),
            "seconds_per_scene": {
                f"scene_{task.scene:04d}/{task.camera}": round(outcomes[index].seconds, 3)
                for index, task in enumerate(tasks)
            },
            "cpu_seconds_scoring": round(sum(o.cpu_seconds for o in outcomes.values()), 3),
            # Largest resident memory of any process that scored a scene, and what such a
            # process already held when it started (interpreter and imported libraries).
            "peak_rss_mb": round(
                max((outcome.peak_rss_mb for outcome in outcomes.values()), default=0.0), 1
            ),
            "start_rss_mb": round(
                max((outcome.start_rss_mb for outcome in outcomes.values()), default=0.0), 1
            ),
            "cache": {
                "generator": self.sources.generator,
                "version": CACHE_VERSION,
                "hash": cache.hash,
                "directory": str(cache.directory),
            },
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        per_object: dict[int, Tally] = {}
        for outcome in outcomes.values():
            for object_id, part in outcome.objects.items():
                per_object.setdefault(object_id, Tally()).add(part)
        breakdown = {
            "top_k": constants.top_k,
            "frictions": list(constants.frictions),
            "per_scene": {
                f"{task.camera}/scene_{task.scene:04d}": outcomes[index].tally.to_dict(
                    len(constants.frictions), with_unfilled=True
                )
                for index, task in enumerate(tasks)
                if outcomes[index].tally.frames
            },
            "per_object": {
                str(object_id): per_object[object_id].to_dict(
                    len(constants.frictions), with_unfilled=False
                )
                for object_id in sorted(per_object)
            },
        }
        return EvaluationResult(
            precision=precision,
            metrics=metrics,
            requested=len(keys),
            missing=[key for key in keys if key in missing],
            info=info,
            breakdown=breakdown,
        )


@EVALUATORS.register()
class GraspNet1BEvaluator(SceneEvaluator):
    """GraspNet-1Billion single-view evaluation, value-compatible with ``graspnetAPI`` 1.2.x.

    The constants of the protocol are not configurable; see
    :data:`~grasppanda.evaluation.graspnet1b.protocol.GRASPNET1B_SV_V1`.

    Args:
        data_root: Dataset root. Default: ``$GRASPPANDA_GRASPNET1B_ROOT`` and the other
            rules of :func:`grasppanda.utils.paths.dataset_root`.
        cache_dir: Root of derived data. Default: ``$GRASPPANDA_CACHE``. The per-object
            evaluation cache is created below it on first use.
        scene_timeout: Seconds a worker may spend on one scene before the run fails.
        start_method: How worker processes are created.
    """

    source = "graspnetAPI 1.2.x (Fang et al., CVPR 2020)"

    protocol = GRASPNET1B_SV_V1.name
    constants = GRASPNET1B_SV_V1
    sources = GRASPNET1B_SOURCES
    truth = SceneGroundTruth
    prediction_file = staticmethod(prediction_path)

    def __init__(
        self,
        data_root: str | None = None,
        cache_dir: str | None = None,
        scene_timeout: float = 3600.0,
        start_method: Literal["forkserver", "spawn", "fork"] = "forkserver",
    ) -> None:
        super().__init__(data_root, cache_dir, scene_timeout, start_method)
