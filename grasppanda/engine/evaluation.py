"""Predict on a split and score it: one routine for in-training evaluation, ``test`` and ``infer``.

1. ``predict`` runs the detector over the split and writes the raw predictions once, to
   ``<root>/raw/``. While the observation of a frame is still in hand, every variant of
   ``evaluation.variants`` with a non-empty post-processing chain is applied to that
   frame's raw grasps and written to ``<root>/<variant>/``. A variant with an empty chain
   *is* the raw predictions and is scored straight from ``<root>/raw/``.
2. ``score`` hands each variant's directory to the configured evaluator together with
   every key of the split, so a frame that was never predicted is reported as missing,
   and writes one result record per variant.

An empty prediction is a valid result and is written and scored like any other. A missing
one lowers coverage, and a variant with incomplete coverage yields no headline metric.

Under ``torchrun`` each process predicts its own share of the frames (shares do not
overlap); the keys are merged and de-duplicated by ``SampleKey`` and the main process
scores. Prediction always runs in full precision, and under its own random seed: the
same weights give the same predictions whether they are evaluated during training or
later, and evaluating during training does not change what is trained.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from grasppanda import __version__
from grasppanda.datasets import BaseDataset
from grasppanda.evaluation import Evaluator, save_prediction
from grasppanda.evaluation.records import ResultRecord, scene_label, summarize
from grasppanda.models import BaseGraspDetector, PostProcessor
from grasppanda.registry import EVALUATORS, POSTPROCESSORS
from grasppanda.structures import ContractError, GraspGroup, SampleKey

from . import distributed as dist
from .builders import build_dataset, build_predict_loader
from .errors import EngineError
from .run import LOGGER_NAME, Run
from .seeding import preserved_rng

__all__ = ["RAW", "Dump", "EvalJob", "Evaluation", "build_job", "evaluate", "predict", "score"]

#: Directory of the unprocessed predictions, and the reserved name of the variant without
#: post-processing.
RAW = "raw"

_LOG = logging.getLogger(LOGGER_NAME)


@dataclass
class EvalJob:
    """A split to predict on and, unless only predictions are wanted, the means to score it.

    Attributes:
        section: The ``data`` section (``val``, ``test``, ...).
        split: Name of the split for result records: the reader's own ``split`` when it has one.
        dataset: The dataset of that section.
        loader: Yields ``(batch, [observation of each frame])``.
        chains: Per variant, the built post-processing chain (empty: raw predictions).
        chain_nodes: Per variant, the config of that chain with every parameter.
        evaluator: The scorer, or ``None`` for prediction only.
        evaluator_node: Its config.
        num_workers: Worker processes the evaluator may use.
        seed: ``runtime.seed``; prediction is seeded with it (plus the rank).
    """

    section: str
    split: str
    dataset: BaseDataset
    loader: DataLoader
    chains: dict[str, list[PostProcessor]]
    chain_nodes: dict[str, list[dict[str, Any]]]
    evaluator: Evaluator | None = None
    evaluator_node: dict[str, Any] | None = None
    num_workers: int = 0
    seed: int = 0

    def directory(self, root: Path, variant: str) -> Path:
        """Where the predictions of ``variant`` live below ``root``."""
        return root / (variant if self.chains[variant] else RAW)


def build_job(
    cfg: Mapping[str, Any],
    section: str,
    *,
    with_evaluator: bool = True,
    rank: int | None = None,
    world_size: int | None = None,
) -> EvalJob:
    """Build dataset, loader, post-processing chains and evaluator for ``data.<section>``."""
    evaluation = cfg["evaluation"]
    nodes = {name: list(chain) for name, chain in evaluation["variants"].items()}
    for name in nodes:
        if not name or "/" in name or name.startswith("."):
            raise EngineError(f"evaluation.variants: '{name}' cannot be used as a directory name.")
    if nodes.get(RAW):
        raise EngineError(
            f"evaluation.variants.{RAW} names the unprocessed predictions and must be an empty "
            "chain. Give the post-processed variant another name."
        )
    evaluator = None
    if with_evaluator:
        if evaluation["evaluator"] is None:
            raise EngineError("evaluation.evaluator is not set, so nothing can be scored.")
        evaluator = EVALUATORS.build(evaluation["evaluator"])
    dataset = build_dataset(cfg, section)
    dataset_node = cfg["data"][section]["dataset"]
    return EvalJob(
        section=section,
        split=str(dataset_node.get("split") or getattr(dataset, "split", None) or section),
        dataset=dataset,
        loader=build_predict_loader(cfg, section, dataset, rank=rank, world_size=world_size),
        chains={name: [POSTPROCESSORS.build(n) for n in chain] for name, chain in nodes.items()},
        chain_nodes=nodes,
        evaluator=evaluator,
        evaluator_node=evaluation["evaluator"] if with_evaluator else None,
        num_workers=evaluation["num_workers"],
        seed=cfg["runtime"]["seed"],
    )


# ----------------------------------------------------------------------- prediction


@dataclass
class Dump:
    """Predictions of one split on disk.

    Attributes:
        root: Directory holding ``raw/`` and one directory per post-processed variant.
        keys: The frames that were predicted, each exactly once.
        gt_assisted: Names of the inputs that were derived from ground truth, over all frames.
        seconds: Wall time of the prediction loop on this process.
        counts: Per variant and scene, the number of grasps the model emitted followed by
            the number each step of the variant's chain left, summed over the frames.
    """

    root: Path
    keys: list[SampleKey] = field(default_factory=list)
    gt_assisted: list[str] = field(default_factory=list)
    seconds: float = 0.0
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)


def _keyed(grasps: GraspGroup, key: SampleKey, who: str) -> GraspGroup:
    if not isinstance(grasps, GraspGroup):
        raise ContractError(f"{who} returned {type(grasps).__name__} for {key}, not a GraspGroup")
    if grasps.key is None:
        return replace(grasps, key=key)
    if grasps.key != key:
        raise ContractError(f"{who} returned grasps of {grasps.key} where {key} was expected")
    return grasps


def merge_keys(shards: list[list[SampleKey]]) -> list[SampleKey]:
    """Merge the frames of all processes; a frame predicted by several counts once."""
    return list(dict.fromkeys(key for shard in shards for key in shard))


def _count(table: dict[str, list[int]], key: SampleKey, sizes: list[int]) -> None:
    """Add the grasp counts of one frame to the row of its scene."""
    row = table.setdefault(scene_label(key), [0] * len(sizes))
    for step, size in enumerate(sizes):
        row[step] += size


def merge_counts(
    shards: list[dict[str, dict[str, list[int]]]],
) -> dict[str, dict[str, list[int]]]:
    """Sum the per-scene grasp counts of all processes (their frames do not overlap)."""
    merged: dict[str, dict[str, list[int]]] = {}
    for shard in shards:
        for variant, scenes in shard.items():
            for scene, sizes in scenes.items():
                row = merged.setdefault(variant, {}).setdefault(scene, [0] * len(sizes))
                for step, size in enumerate(sizes):
                    row[step] += size
    return {name: dict(sorted(scenes.items())) for name, scenes in merged.items()}


@torch.no_grad()
def predict(
    model: BaseGraspDetector,
    job: EvalJob,
    root: str | Path,
    *,
    device: torch.device,
    log_interval: int = 0,
) -> Dump:
    """Run ``model.predict`` over the split of ``job`` and write every variant below ``root``.

    Whatever ``root`` held for these variants is removed first, so a stale file can never
    stand in for a prediction that was not made. Call on every process; progress is
    logged every ``log_interval`` batches.
    """
    root = Path(root)
    if dist.is_main():
        for name in {RAW, *job.chains}:
            shutil.rmtree(root / name, ignore_errors=True)
    dist.barrier()
    was_training = model.training
    model.eval()
    processed_variants = [name for name, chain in job.chains.items() if chain]
    keys: dict[SampleKey, None] = {}
    assisted: set[str] = set()
    counts: dict[str, dict[str, list[int]]] = {name: {} for name in job.chains}
    start = time.perf_counter()
    seed = job.seed + dist.rank()
    if job.loader.generator is not None:
        job.loader.generator.manual_seed(seed)  # seeds of the loader's worker processes
    with preserved_rng(seed):
        for index, (batch, frames) in enumerate(job.loader, start=1):
            observation = batch.observation.to(device)
            groups = model.predict(observation)
            if len(groups) != len(observation):
                raise ContractError(
                    f"{type(model).__name__}.predict returned {len(groups)} GraspGroups for a "
                    f"batch of {len(observation)} frames"
                )
            assisted.update(observation.gt_assisted())
            for key, grasps, frame in zip(observation.keys, groups, frames):
                if key in keys:  # a sampler or dataset that repeats a frame
                    continue
                keys[key] = None
                grasps = _keyed(grasps, key, f"{type(model).__name__}.predict")
                save_prediction(root / RAW, grasps)
                for name in job.chains:
                    if name not in processed_variants:
                        _count(counts[name], key, [len(grasps)])
                if not processed_variants:
                    continue
                frame = frame.to(device)  # the observation of this frame alone
                for name in processed_variants:
                    processed = grasps
                    sizes = [len(grasps)]
                    for step in job.chains[name]:
                        processed = _keyed(step(processed, frame), key, type(step).__name__)
                        sizes.append(len(processed))
                    _count(counts[name], key, sizes)
                    save_prediction(root / name, processed)
            if log_interval and index % log_interval == 0 and dist.is_main():
                rate = len(keys) / max(time.perf_counter() - start, 1e-9)
                _LOG.info(
                    "predict %s  batch %d/%d  %.1f frames/s",
                    job.section,
                    index,
                    len(job.loader),
                    rate,
                )
    model.train(was_training)
    shards = dist.gather((list(keys), sorted(assisted), counts))
    return Dump(
        root=root,
        keys=merge_keys([shard for shard, _, _ in shards]),
        gt_assisted=sorted({name for _, names, _ in shards for name in names}),
        seconds=time.perf_counter() - start,
        counts=merge_counts([tally for _, _, tally in shards]),
    )


# -------------------------------------------------------------------------- scoring


def score(
    job: EvalJob, dump: Dump, results_dir: str | Path, **provenance: Any
) -> dict[str, ResultRecord]:
    """Score every variant of ``dump`` and write ``<results_dir>/<variant>.json``.

    The evaluator is asked for every frame of the split, not for the frames that happen
    to have a file. ``provenance`` is stored in each record: ``config_hash``, ``code``,
    ``weights``, ``seed``.
    """
    if job.evaluator is None:
        raise EngineError("this job was built without an evaluator")
    requested = list(job.dataset.keys)
    records = {}
    for name in job.chains:
        result = job.evaluator.evaluate(
            job.directory(dump.root, name), requested, num_workers=job.num_workers
        )
        record = ResultRecord.from_result(
            result,
            variant=name,
            keys=requested,
            protocol=job.evaluator.protocol,
            dataset=job.dataset.name,
            split=job.split,
            postprocessing=job.chain_nodes[name],
            grasp_counts=(
                {
                    "steps": ["predicted", *(node["type"] for node in job.chain_nodes[name])],
                    "per_scene": dump.counts[name],
                }
                if name in dump.counts
                else {}
            ),
            gt_assisted=dump.gt_assisted,
            evaluator={"type": type(job.evaluator).__name__, "config": job.evaluator_node},
            **provenance,
        )
        record.write(Path(results_dir) / f"{name}.json")
        records[name] = record
    return records


@dataclass
class Evaluation:
    """Result of :func:`evaluate`: one record per variant."""

    records: dict[str, ResultRecord]
    gt_assisted: list[str] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0
    run_dir: Path | None = None

    @property
    def metrics(self) -> dict[str, float]:
        """``<variant>/<metric>`` of every *complete* variant. Partial variants have none."""
        return {
            f"{name}/{metric}": value
            for name, record in self.records.items()
            for metric, value in (record.headline() or {}).items()
        }

    @property
    def partial(self) -> list[str]:
        """Names of the variants whose coverage is incomplete."""
        return [name for name, record in self.records.items() if record.partial]

    def render(self) -> str:
        return summarize(self.records)


def evaluate(
    model: BaseGraspDetector,
    job: EvalJob,
    run: Run,
    *,
    results_dir: str | Path | None = None,
    weights: Mapping[str, Any] | None = None,
) -> Evaluation:
    """Predict on the split of ``job``, score every variant and stamp the run.

    Call on every process; all of them return the same :class:`Evaluation`.
    """
    if job.evaluator is None:
        raise EngineError("this job was built without an evaluator")
    interval = run.cfg["runtime"]["log"]["interval"]
    dump = predict(model, job, run.predictions, device=run.device, log_interval=interval)
    records = None
    if run.is_main:
        run.stamp(gt_assisted=dump.gt_assisted, protocol=job.evaluator.protocol)
        code = {
            "commit": run.git.get("commit"),
            "dirty": run.git.get("dirty"),
            "version": __version__,
        }
        with preserved_rng(job.seed):  # an evaluator that draws numbers must not disturb training
            records = score(
                job,
                dump,
                results_dir or run.results,
                config_hash=run.config.digest(),
                code=code,
                weights=dict(weights or {}),
                seed=job.seed,
            )
    return Evaluation(
        records=dist.broadcast(records),
        gt_assisted=dump.gt_assisted,
        frames=len(dump.keys),
        seconds=dump.seconds,
        run_dir=run.dir,
    )
