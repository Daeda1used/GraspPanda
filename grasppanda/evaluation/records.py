"""Result records: one JSON file per scored post-processing variant.

A number without its conditions is not a result. A record therefore carries, next to the
metrics, everything needed to say what was measured: protocol, dataset, split and
cameras, how many frames were scored, the post-processing chain with its parameters,
which inputs were derived from ground truth, the evaluator, the config hash, the code
commit, the sha256 of the weights and the seed.

**Later additions.** ``grasp_counts`` and ``breakdown`` were added after the first
records were written. Both are optional: a record without them reads as before, with
both empty, and nothing but ``grasppanda analyze`` looks at them.

**Partial results.** A record whose coverage is below 100 % is still written, but it is
marked ``partial`` and its aggregate metrics are stored under ``partial_metrics`` while
``metrics`` stays ``null``. Anything that reads ``record["metrics"]`` (a model-zoo table,
a sweep) can therefore never mistake a partial number for a headline number, and the
console summary prints per-scene numbers and the coverage instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from grasppanda.structures import SampleKey
from grasppanda.utils.console import table

from .base import EvaluationResult

__all__ = ["RECORD_SCHEMA", "ResultRecord", "per_scene_metrics", "scene_label", "summarize"]

#: Version of the JSON layout written by :meth:`ResultRecord.write`.
RECORD_SCHEMA = 1


def scene_label(key: SampleKey) -> str:
    """``<camera>/scene_NNNN``: the name of a frame's scene in every per-scene table."""
    return f"{key.camera}/scene_{key.scene:04d}"


def per_scene_metrics(precision: Mapping[SampleKey, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Aggregate per-frame precision tables by scene and camera.

    Returns ``{"<camera>/scene_NNNN": {"frames", "mean_precision", "per_column"}}``.
    ``mean_precision`` is the mean of the scene's ``(frames, top_k, columns)`` tables, a
    fraction in ``[0, 1]``; ``per_column`` is the same mean for each column (friction
    level) separately. Scenes, not frames, are the independent unit for statistics.
    """
    groups: dict[str, list[np.ndarray]] = {}
    for key, frame in precision.items():
        groups.setdefault(scene_label(key), []).append(np.asarray(frame))
    out = {}
    for name in sorted(groups):
        tables = np.stack(groups[name]).astype(np.float64)  # (frames, top_k, columns)
        columns = tables.reshape(-1, tables.shape[-1]) if tables.ndim >= 2 else tables[:, None]
        out[name] = {
            "frames": len(tables),
            "mean_precision": float(tables.mean()),
            "per_column": [float(value) for value in columns.mean(axis=0)],
        }
    return out


@dataclass
class ResultRecord:
    """The outcome of scoring one variant of one run on one split.

    Attributes:
        variant: Name of the post-processing variant (``raw``, ``cd``, ...).
        protocol: Identifier of the pinned protocol the evaluator implements.
        dataset: Dataset name, as used in :class:`SampleKey`.
        split: The dataset split (or, when the reader has none, the ``data`` section).
        cameras: Cameras of the requested frames.
        requested: Frames the evaluator was asked to score.
        scored: Frames it scored.
        missing: Frames without a prediction, as strings.
        postprocessing: The variant's chain, each step with all its parameters.
        gt_assisted: Names of inference inputs that were derived from ground truth.
        evaluator: Evaluator class, its config node and the ``info`` it reported.
        config_hash: Hash of the config of the run that produced the predictions.
        code: Commit, dirty flag and package version of the code that ran.
        weights: Where the weights came from: path, sha256, kind (``model`` or ``ema``), epoch.
        seed: ``runtime.seed`` of the run.
        metrics: Aggregates reported by the evaluator; headline numbers only when complete.
        per_scene: See :func:`per_scene_metrics`.
        grasp_counts: How many grasps each step of the post-processing chain left:
            ``{"steps": ["predicted", <step type>, ...], "per_scene": {<scene>: [n, ...]}}``
            with one count per step, summed over the scene's frames. Empty when the
            run did not record it.
        breakdown: The evaluator's split of AP by scene and by object, see
            :mod:`grasppanda.evaluation.breakdown`. Empty when the evaluator has none.
        created: UTC time the record was made.
    """

    variant: str
    protocol: str
    dataset: str
    split: str
    cameras: list[str]
    requested: int
    scored: int
    missing: list[str] = field(default_factory=list)
    postprocessing: list[dict[str, Any]] = field(default_factory=list)
    gt_assisted: list[str] = field(default_factory=list)
    evaluator: dict[str, Any] = field(default_factory=dict)
    config_hash: str | None = None
    code: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    per_scene: dict[str, dict[str, Any]] = field(default_factory=dict)
    grasp_counts: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # ------------------------------------------------------------------ coverage

    @property
    def coverage(self) -> float:
        """Fraction of the requested frames that were scored."""
        return self.scored / self.requested if self.requested else 0.0

    @property
    def partial(self) -> bool:
        """True unless every requested frame was scored. A partial result has no headline."""
        return self.requested <= 0 or self.scored != self.requested

    def headline(self) -> dict[str, float] | None:
        """The aggregate metrics, or ``None`` when the result is partial."""
        return None if self.partial else dict(self.metrics)

    # -------------------------------------------------------------- construction

    @classmethod
    def from_result(
        cls,
        result: EvaluationResult,
        *,
        variant: str,
        keys: Sequence[SampleKey],
        protocol: str,
        dataset: str,
        split: str,
        **provenance: Any,
    ) -> ResultRecord:
        """Build a record from what an evaluator returned for ``keys``.

        ``provenance`` takes the remaining fields: ``postprocessing``, ``gt_assisted``,
        ``evaluator``, ``config_hash``, ``code``, ``weights``, ``seed``, ``grasp_counts``.
        """
        evaluator = dict(provenance.pop("evaluator", {}) or {})
        evaluator["info"] = dict(result.info)
        return cls(
            variant=variant,
            protocol=protocol,
            dataset=dataset,
            split=split,
            cameras=sorted({key.camera for key in keys}),
            requested=result.requested,
            scored=len(result.precision),
            missing=[str(key) for key in result.missing],
            evaluator=evaluator,
            metrics={name: float(value) for name, value in result.metrics.items()},
            per_scene=per_scene_metrics(result.precision),
            breakdown=dict(result.breakdown),
            **provenance,
        )

    # ------------------------------------------------------------------------ I/O

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        metrics = data.pop("metrics")
        return {
            "schema": RECORD_SCHEMA,
            **data,
            "coverage": self.coverage,
            "partial": self.partial,
            "metrics": None if self.partial else metrics,
            "partial_metrics": metrics if self.partial else None,
        }

    def write(self, path: str | Path) -> Path:
        """Write the record as JSON, atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> ResultRecord:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"{path}: unsupported result record schema {data.get('schema')!r}")
        metrics = data.pop("metrics") or data.get("partial_metrics") or {}
        for derived in ("schema", "coverage", "partial", "partial_metrics"):
            data.pop(derived, None)
        return cls(metrics=metrics, **data)


def summarize(records: Mapping[str, ResultRecord]) -> str:
    """Console summary of the variants of one evaluation.

    Complete variants show their metrics. A partial variant shows its coverage and, below
    the table, per-scene numbers; its aggregate metrics are withheld.
    """
    if not records:
        return "no variant was scored"
    first = next(iter(records.values()))
    complete = [record for record in records.values() if not record.partial]
    columns = list(dict.fromkeys(name for record in complete for name in record.metrics))
    columns = columns or ["metrics"]
    rows = []
    for name, record in records.items():
        if record.partial:
            cells = ["withheld (partial)"] + [""] * (len(columns) - 1)
        else:
            cells = [f"{record.metrics[c]:.2f}" if c in record.metrics else "-" for c in columns]
        rows.append([name, f"{record.scored}/{record.requested}", *cells])
    assisted = ", ".join(first.gt_assisted) or "none"
    lines = [
        f"protocol {first.protocol} · dataset {first.dataset} · split {first.split} · "
        f"cameras {', '.join(first.cameras) or '-'} · ground-truth-assisted inputs: {assisted}",
        table(["variant", "frames", *columns], rows, enabled=False),
    ]
    for record in records.values():
        if not record.partial:
            continue
        lines.append(
            f"{record.variant}: PARTIAL, {record.scored} of {record.requested} frames scored "
            f"({100 * record.coverage:.1f} %), {len(record.missing)} missing. "
            "Per-scene mean precision only:"
        )
        lines += [
            f"  {scene}  frames {numbers['frames']}  {numbers['mean_precision']:.4f}"
            for scene, numbers in record.per_scene.items()
        ]
    return "\n".join(lines)
