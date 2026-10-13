"""Time the data path of a training config: loader, transfer, label matching, iteration.

For a config with a ``data.train`` section this measures, on the real dataset:

``loader``
    Frames per second the training loader delivers when nothing consumes them, the size
    of a batch as it leaves a worker, and the time until the first batch arrives.
``stages``
    One training iteration split into its parts, with the device synchronised around
    each: waiting for the loader, ``batch.to(device)``, the forward pass (and, inside
    it, the label assigner), the backward pass and the optimizer step.
``iteration``
    Seconds per training iteration as the trainer runs it, without any added
    synchronisation: the number that decides how long an epoch takes.

Every measure is repeated ``--repeats`` times; minimum and median over the repeats are
reported, the median being the number to trust on a machine that is busy with other
work. With ``--grids dense deferred`` every repeat runs both label storages back to
back on the same frames, with the same sampling and the same initial weights, so the
two columns differ in the label storage only (see ``docs/advanced_guides/data_loading.md``).

    python -m tools.bench_data gsnet/gsnet_sunet14_g1b-realsense_10e
    python -m tools.bench_data gsnet/gsnet_sunet14_g1b-realsense_10e --grids dense deferred
    python -m tools.bench_data graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
        --measure loader stages --scenes 8 --iterations 30 --markdown

A dataloader worker opens the files of a scene the first time it meets the scene, which
is slow on a network file system and is paid once per worker and epoch. ``--scenes N``
restricts the run to the first ``N`` scenes of the split so that a short run reaches the
steady state; ``--warmup`` batches are discarded in any case, and the time to the first
batch is reported on its own.

Nothing is written; the model starts from its initial weights and is discarded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from grasppanda.config import Config
from grasppanda.engine.builders import build_dataset, build_model, build_train_loader, set_epoch
from grasppanda.engine.optim import build_optimizers, clip_gradients
from grasppanda.engine.run import resolve_device
from grasppanda.engine.seeding import seed_everything
from grasppanda.engine.trainer import total_loss
from grasppanda.structures import Batch
from grasppanda.utils.console import table

MEASURES = ("loader", "stages", "iteration")


# ------------------------------------------------------------------------- config


def label_transforms(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The dense-label transforms of ``data.train``, whose ``grids`` option is switched."""
    nodes = cfg["data"]["train"]["dataset"].get("transforms") or []
    return [node for node in nodes if node.get("type") == "SampleGraspLabels"]


def load_config(args: argparse.Namespace, grids: str | None) -> tuple[Config, dict[str, Any]]:
    """The config of a trial: the file, ``--opts``, then the switches of this tool."""
    # The loader alone needs no model: only the measures that train are held to what
    # `grasppanda train` requires of a config.
    trains = set(getattr(args, "measure", MEASURES)) - {"loader"}
    config = Config.load(args.config, args.opts, command="train" if trains else None)
    cfg = config.resolved()
    train = (cfg.get("data") or {}).get("train")
    if train is None:
        raise SystemExit(f"{args.config}: there is no data.train section to measure")
    if args.workers is not None:
        train["num_workers"] = args.workers
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if grids is not None:
        nodes = label_transforms(cfg)
        if not nodes:
            raise SystemExit(
                f"{args.config}: data.train has no SampleGraspLabels transform, so there are no "
                "dense label grids to switch with --grids"
            )
        for node in nodes:
            node["grids"] = grids
    cfg["runtime"]["seed"] = args.seed
    return config, cfg


class SceneSubset(Dataset):
    """The frames of the first ``count`` scenes of a dataset, in its order."""

    def __init__(self, dataset: Any, count: int) -> None:
        self.dataset = dataset
        scenes = sorted({key.scene for key in dataset.keys})[:count]
        self.indices = [i for i, key in enumerate(dataset.keys) if key.scene in scenes]
        self.collate = dataset.collate

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Batch:
        return self.dataset[self.indices[index]]


def make_loader(cfg: Mapping[str, Any], scenes: int | None) -> DataLoader:
    dataset = build_dataset(cfg, "train")
    if scenes is not None:
        dataset = SceneSubset(dataset, scenes)
    loader = build_train_loader(cfg, dataset, rank=0, world_size=1)
    set_epoch(loader, 0)
    return loader


# ------------------------------------------------------------------------ payload


def payload_bytes(value: Any) -> int:
    """Bytes of the arrays, tensors and strings an object holds, however nested."""
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (str, bytes)):
        return len(value)
    if value is None or isinstance(value, (bool, int, float)):
        return 0
    if isinstance(value, Mapping):
        return sum(payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(payload_bytes(item) for item in value)
    if dataclasses.is_dataclass(value):
        return sum(payload_bytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    slots = getattr(type(value), "__slots__", ())
    names = list(vars(value)) if hasattr(value, "__dict__") else list(slots)
    return sum(payload_bytes(getattr(value, name, None)) for name in names)


# ------------------------------------------------------------------------ measures


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TimedAssigner:
    """Calls an assigner between two device synchronisations and keeps the durations."""

    def __init__(self, inner: Callable[..., Any], device: torch.device) -> None:
        self.inner, self.device = inner, device
        self.seconds: list[float] = []

    def __call__(self, *args: Any) -> Any:
        _synchronise(self.device)
        start = time.perf_counter()
        result = self.inner(*args)
        _synchronise(self.device)
        self.seconds.append(time.perf_counter() - start)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def measure_loader(cfg: Mapping[str, Any], args: argparse.Namespace) -> dict[str, float]:
    """Throughput of the bare loader and the size of what it delivers."""
    start = time.perf_counter()
    loader = make_loader(cfg, args.scenes)
    batches = iter(loader)
    sizes, frames = [], 0
    first = mark = 0.0
    for index in range(args.warmup + args.iterations):
        batch = next(batches)
        if index == 0:
            first = time.perf_counter() - start
        if index == args.warmup - 1 or (args.warmup == 0 and index == 0):
            mark = time.perf_counter()
            continue
        if index >= args.warmup:
            frames += len(batch)
            sizes.append(payload_bytes(batch))
    elapsed = time.perf_counter() - mark
    del batches
    return {
        "loader frames/s": frames / elapsed,
        "batch payload MiB": statistics.median(sizes) / 2**20,
        "first batch s": first,
    }


class Training:
    """Model, optimizer and loader of one trial, and the steps of the training loop."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        config: Config,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        seed_everything(args.seed)
        self.model = build_model(cfg).to(device)
        self.model.train()
        self.optimizers = build_optimizers(self.model, config["optimizer"])
        self.grad_clip = cfg["train"]["grad_clip"]
        self.clip_mode = cfg["train"]["grad_clip_mode"]
        self.batches = iter(make_loader(cfg, args.scenes))

    def fetch(self) -> Batch:
        return next(self.batches)

    def forward(self, batch: Batch) -> Tensor:
        if batch.supervision is None:
            raise SystemExit("data.train yields frames without supervision")
        outputs = self.model.loss(batch.observation, batch.supervision)
        return total_loss(outputs, "the benchmark", batch.observation.keys)

    def update(self) -> None:
        if self.grad_clip is not None:
            clip_gradients(self.model.parameters(), self.grad_clip, self.clip_mode)
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def measure_stages(
    cfg: Mapping[str, Any], config: Config, args: argparse.Namespace, device: torch.device
) -> dict[str, float]:
    """A training iteration part by part, the device synchronised around each part."""
    training = Training(cfg, config, args, device)
    model, assigner = training.model, None
    if getattr(model, "assigner", None) is not None:
        assigner = model.assigner = TimedAssigner(model.assigner, device)
    names = ("data wait", "to(device)", "forward", "backward", "step")
    parts: dict[str, list[float]] = {name: [] for name in names}

    def clock(name: str, call: Callable[[], Any]) -> Any:
        _synchronise(device)
        start = time.perf_counter()
        result = call()
        _synchronise(device)
        parts[name].append(time.perf_counter() - start)
        return result

    for _ in range(args.warmup + args.iterations):
        batch = clock("data wait", training.fetch)
        batch = clock("to(device)", lambda b=batch: b.to(device))
        total = clock("forward", lambda b=batch: training.forward(b))
        clock("backward", total.backward)
        clock("step", training.update)
    # Means, not medians: the wait for the loader is zero while batches are buffered and
    # long once they are used up, and an epoch pays the average.
    kept = {name: values[args.warmup :] for name, values in parts.items()}
    result = {f"{name} ms": 1e3 * statistics.fmean(values) for name, values in kept.items()}
    if assigner is not None:
        result["of which assigner ms"] = 1e3 * statistics.fmean(assigner.seconds[args.warmup :])
    result["sum of parts s"] = sum(statistics.fmean(values) for values in kept.values())
    return result


def measure_iteration(
    cfg: Mapping[str, Any], config: Config, args: argparse.Namespace, device: torch.device
) -> dict[str, float]:
    """Seconds per iteration of the plain training loop, synchronised only at both ends."""
    training = Training(cfg, config, args, device)
    waits: list[float] = []
    mark = 0.0
    for index in range(args.warmup + args.iterations):
        if index == args.warmup:
            _synchronise(device)
            mark = time.perf_counter()
        start = time.perf_counter()
        batch = training.fetch()
        waits.append(time.perf_counter() - start)
        total = training.forward(batch.to(device))
        total.backward()
        training.update()
    _synchronise(device)
    elapsed = time.perf_counter() - mark
    return {
        "iteration s": elapsed / args.iterations,
        "of which data wait s": statistics.fmean(waits[args.warmup :]),
    }


# --------------------------------------------------------------------------- main


def environment(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "device": str(device),
        "cpus": os.cpu_count(),
        "load average": [round(value, 1) for value in os.getloadavg()],
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def summarise(trials: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """``{measure: {min, median, max}}`` over the repeats of one configuration."""
    names = list(dict.fromkeys(name for trial in trials for name in trial))
    summary = {}
    for name in names:
        values = [trial[name] for trial in trials if name in trial]
        summary[name] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return summary


def _cell(stats: Mapping[str, float] | None, name: str) -> str:
    if stats is None:
        return "-"
    digits = 0 if stats["median"] >= 100 else 1 if stats["median"] >= 10 else 2
    # A rate is best at its maximum, a duration or a size at its minimum.
    best = stats["max"] if name.endswith("/s") else stats["min"]
    return f"{best:.{digits}f} / {stats['median']:.{digits}f}"


def render(results: Mapping[str, Mapping[str, Mapping[str, float]]], markdown: bool) -> str:
    modes = list(results)
    names = list(dict.fromkeys(name for mode in modes for name in results[mode]))
    headers = ["measure (best / median)", *modes]
    rows = [[name, *(_cell(results[mode].get(name), name) for mode in modes)] for name in names]
    if len(modes) == 2:
        headers.append("ratio of medians")
        for row, name in zip(rows, names):
            first, second = (results[mode].get(name) for mode in modes)
            usable = first and second and min(first["median"], second["median"]) > 0
            row.append(f"{first['median'] / second['median']:.2f}" if usable else "-")
    if not markdown:
        return table(headers, rows)
    lines = ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="a config file or the name of a shipped config")
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument(
        "--grids",
        nargs="+",
        choices=("dense", "deferred"),
        help="label storage(s) to run, back to back; default: as the config says",
    )
    parser.add_argument("--measure", nargs="+", choices=MEASURES, default=list(MEASURES))
    parser.add_argument("--iterations", type=int, default=30, help="timed batches per measure")
    parser.add_argument(
        "--warmup",
        type=int,
        default=16,
        help="batches discarded before them; more than the loader buffers (2 per worker)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenes", type=int, help="use the first N scenes of the split only")
    parser.add_argument("--workers", type=int, help="override data.train.num_workers")
    parser.add_argument("--batch-size", type=int, help="override data.train.batch_size")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--json", help="also write every trial to this file")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("--iterations and --repeats must be positive, --warmup non-negative")

    device = resolve_device(args.device)
    modes: list[str | None] = list(args.grids) if args.grids else [None]
    trials: dict[str, list[dict[str, float]]] = {mode or "as configured": [] for mode in modes}
    info = environment(device)
    print(f"# {args.config}", file=sys.stderr)
    print(f"# {info}", file=sys.stderr)
    for repeat in range(args.repeats):
        for mode in modes:
            config, cfg = load_config(args, mode)
            trial: dict[str, float] = {}
            if "loader" in args.measure:
                trial.update(measure_loader(cfg, args))
            if "stages" in args.measure:
                trial.update(measure_stages(cfg, config, args, device))
            if "iteration" in args.measure:
                trial.update(measure_iteration(cfg, config, args, device))
            trials[mode or "as configured"].append(trial)
            shown = ", ".join(f"{name} {value:.3g}" for name, value in trial.items())
            print(f"# repeat {repeat + 1}, {mode or 'as configured'}: {shown}", file=sys.stderr)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    results = {mode: summarise(runs) for mode, runs in trials.items()}
    train = load_config(args, modes[0])[1]["data"]["train"]
    print(
        f"{args.config}: batch {train['batch_size']}, {train['num_workers']} workers, "
        f"{args.iterations} batches after {args.warmup} of warm-up, {args.repeats} repeats"
        + (f", first {args.scenes} scenes" if args.scenes else "")
    )
    print(render(results, args.markdown))
    if args.json:
        payload = {"arguments": vars(args), "environment": info, "trials": trials}
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
