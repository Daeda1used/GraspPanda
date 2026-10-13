"""Distributed runs: rank detection, data sharding, and a real two-process run on CPU.

What cannot be tested here is stated in ``docs/user_guides/train_test.md``: several GPUs,
the NCCL backend and synchronised batch norm.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

from grasppanda.engine import distributed as dist
from grasppanda.engine.builders import build_dataset, build_train_loader, set_epoch
from grasppanda.engine.errors import EngineError
from grasppanda.engine.run import resolve_device

from .conftest import REPO
from .test_engine_tiny import (
    single_thread,  # noqa: F401  (autouse fixture)
    tiny_config,
)


def test_a_plain_run_is_a_single_main_process(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert (dist.world_size(), dist.rank(), dist.is_distributed(), dist.is_main()) == (
        1,
        0,
        False,
        True,
    )
    assert dist.gather("x") == ["x"] and dist.broadcast({"a": 1}) == {"a": 1}
    dist.barrier()
    model = nn.Linear(2, 2)
    assert dist.parallel(model, torch.device("cpu")) is None
    assert dist.sync_batchnorm(model, torch.device("cpu")) is model


def test_rank_and_world_size_come_from_the_launcher_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert (dist.world_size(), dist.rank(), dist.local_rank()) == (4, 2, 1)
    assert dist.is_distributed() and not dist.is_main()
    with pytest.raises(EngineError, match="synchronises batch norm on CUDA only"):
        dist.sync_batchnorm(nn.BatchNorm1d(2), torch.device("cpu"))
    if torch.cuda.is_available():  # each process takes the GPU of its local rank
        assert resolve_device("cuda") == torch.device("cuda", 1)
    monkeypatch.setenv("WORLD_SIZE", "1")  # `torchrun --nproc_per_node=1`
    assert not dist.is_distributed()
    monkeypatch.delenv("LOCAL_RANK")  # WORLD_SIZE alone (some schedulers set it) is not a launch
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert dist.world_size() == 1 and dist.rank() == 0


def test_device_resolution():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(EngineError, match="is not a device"):
        resolve_device("gpu0")


def epoch_order(loader, epoch: int) -> list[int]:
    set_epoch(loader, epoch)
    return list(loader.sampler)


def test_training_shards_split_one_common_shuffle(tmp_path):
    cfg = tiny_config(tmp_path, {"data.train.dataset.frames": 9, "runtime.seed": 3}).resolved()
    dataset = build_dataset(cfg, "train")
    loaders = [build_train_loader(cfg, dataset, rank=rank, world_size=2) for rank in range(2)]
    first, second = (epoch_order(loader, 0) for loader in loaders)
    assert len(first) == len(second) == 5  # 9 frames padded to 10 so that both step equally often
    assert set(first) | set(second) == set(range(9))
    assert len(set(first) & set(second)) == 1  # the padding frame
    single = epoch_order(build_train_loader(cfg, dataset, rank=0, world_size=1), 0)
    assert (
        single[0::2] == first and single[1::2][:4] == second[:4]
    )  # the same permutation, dealt out


def test_epoch_order_depends_on_seed_and_epoch_only(tmp_path):
    cfg = tiny_config(tmp_path, {"data.train.dataset.frames": 16}).resolved()
    dataset = build_dataset(cfg, "train")
    loader = build_train_loader(cfg, dataset)
    orders = [epoch_order(loader, epoch) for epoch in range(3)]
    assert orders[0] != orders[1] != orders[2] and sorted(orders[0]) == list(range(16))
    fresh = build_train_loader(cfg, dataset)  # a resumed process starts directly at epoch 2
    assert epoch_order(fresh, 2) == orders[2]
    reseeded = build_train_loader({**cfg, "runtime": {**cfg["runtime"], "seed": 1}}, dataset)
    assert epoch_order(reseeded, 2) != orders[2]
    unshuffled = {**cfg, "data": {"train": {**cfg["data"]["train"], "shuffle": False}}}
    assert epoch_order(build_train_loader(unshuffled, dataset), 5) == list(range(16))


def test_worker_seeds_depend_on_seed_rank_and_epoch(tmp_path):
    cfg = tiny_config(tmp_path).resolved()
    dataset = build_dataset(cfg, "train")
    seeds = set()
    for rank in range(2):
        loader = build_train_loader(cfg, dataset, rank=rank, world_size=2)
        for epoch in range(3):
            set_epoch(loader, epoch)
            seeds.add(loader.generator.initial_seed())
    assert len(seeds) == 6


def test_an_empty_training_split_is_an_error(tmp_path):
    cfg = tiny_config(
        tmp_path, {"data.train.dataset.frames": 1, "data.train.drop_last": True}
    ).resolved()
    with pytest.raises(EngineError, match="yields no batch"):
        build_train_loader(cfg, build_dataset(cfg, "train"))
    with pytest.raises(EngineError, match="dataset is empty"):
        build_dataset(tiny_config(tmp_path, {"data.train.dataset.frames": 0}).resolved(), "train")


CONFIG = """
    custom_imports: [tests.unit.test_engine_tiny]
    model: {type: TinyDetector, bn: true, dropout: 0.2}
    data:
      train: {dataset: {type: TinyScenes, frames: 8, jitter: 0.02}, batch_size: 2, num_workers: 0}
      val: {dataset: {type: TinyScenes, frames: 5, first_scene: 10}, batch_size: 2, num_workers: 0}
      test: {dataset: {type: TinyScenes, frames: 7, first_scene: 20}, batch_size: 2, num_workers: 0}
    optimizer: {type: SGD, lr: 0.1}
    train: {epochs: 2, ema: 0.9}
    evaluation:
      interval: 1
      evaluator: {type: TinyEvaluator}
      save_best: raw/AP
      variants: {raw: [], best1: [{type: TinyKeepBest}]}
    hooks: [{type: ReplicaProbe}]
    runtime: {device: cpu, deterministic: true, log: {interval: 1}}
"""


def _free_port() -> int:
    """A port nobody listens on: torchrun's default (29500) is taken when two suites run on
    one machine at the same time."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def torchrun(*arguments: str) -> subprocess.CompletedProcess:
    """``torchrun --nproc_per_node=2 -m grasppanda.cli.main ...`` with the CPU (gloo) backend."""
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2"]
    command.append(f"--master_port={_free_port()}")
    command += ["-m", "grasppanda.cli.main", *arguments]
    environment = {**os.environ, "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1"}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        environment.pop(name, None)
    return subprocess.run(
        command, cwd=REPO, env=environment, capture_output=True, text=True, timeout=300, check=False
    )


@pytest.mark.slow
def test_two_processes_train_and_test_with_plain_torchrun(tmp_path):
    """Rank detection, gradient averaging, sharded evaluation and main-only writing, for real."""
    config = tmp_path / "ddp.yaml"
    config.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
    work = tmp_path / "work"

    done = torchrun("train", str(config), "--work-dir", str(work), "--opts", "runtime.run_id=ddp")
    assert done.returncode == 0, done.stdout + done.stderr
    run = work / "ddp"
    assert json.loads((run / "run.json").read_text())["world_size"] == 2
    assert "2 processes" in done.stdout and "per process" in done.stdout
    # 8 frames on 2 processes: 2 batches of 2 each per epoch; only the main process reports
    assert done.stdout.count("epoch 1/2  iter 2/2") == 1 and done.stdout.count("GraspPanda") == 1
    replicas = json.loads((run / "replicas.json").read_text())
    assert [r["rank"] for r in replicas] == [0, 1] and [r["iterations"] for r in replicas] == [4, 4]
    assert replicas[0]["checksum"] == replicas[1]["checksum"]  # replicas hold identical weights
    # 5 validation frames split 3 + 2: every frame predicted once, scored by the main process
    record = json.loads((run / "results/epoch_0002/raw.json").read_text())
    assert (record["requested"], record["scored"], record["partial"]) == (5, 5, False)
    assert len(list((run / "predictions/raw").rglob("*.npy"))) == 5
    checkpoint = torch.load(run / "checkpoints/epoch_0002.pth", weights_only=True)
    assert len(checkpoint["rng"]) == 2  # one random state per process
    assert not torch.equal(checkpoint["rng"][0]["torch"], checkpoint["rng"][1]["torch"])

    weights = run / "checkpoints/best.pth"
    done = torchrun(
        "test",
        str(config),
        "--weights",
        str(weights),
        "--work-dir",
        str(work),
        "--opts",
        "runtime.run_id=ddp-test",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads((work / "ddp-test/results/best1.json").read_text())
    assert (record["requested"], record["scored"], record["partial"]) == (7, 7, False)
    assert record["weights"]["sha256"] and record["split"] == "test"
    assert done.stdout.count("7/7") == 2 and done.stdout.count("matched, 0 missing") == 1
