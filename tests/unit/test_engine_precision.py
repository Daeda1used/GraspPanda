"""``runtime.tf32``: a run states whether convolutions may round to TF32, and float32 is
the default, so that the same weights give the same numbers on every GPU."""

from __future__ import annotations

import json

import pytest
import torch

from grasppanda.engine.hooks import Hook
from grasppanda.engine.run import start_run
from grasppanda.engine.seeding import determinism_restored, set_precision
from grasppanda.engine.state import TrainState
from grasppanda.utils.env import collect_env

from .test_engine_tiny import run_training, single_thread, tiny_config  # noqa: F401


def switches() -> tuple[bool, bool]:
    return torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32


@pytest.fixture
def pytorch_defaults():
    """PyTorch's own defaults (TF32 for convolutions only), put back afterwards."""
    with determinism_restored():
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = True, False
        yield


def test_set_precision_sets_both_switches_and_the_context_restores_them(pytorch_defaults):
    with determinism_restored():
        set_precision(False)
        assert switches() == (False, False)
        set_precision(True)
        assert switches() == (True, True)
    assert switches() == (True, False)


class Seen(Hook):
    def __init__(self) -> None:
        self.during: list[tuple[bool, bool]] = []

    def after_iter(self, state: TrainState) -> None:
        self.during.append(switches())


@pytest.mark.parametrize("tf32", [False, True])
def test_a_run_computes_in_the_precision_of_its_config(tmp_path, pytorch_defaults, tf32):
    config = tiny_config(tmp_path, {"runtime.tf32": tf32})
    seen = Seen()
    state = run_training(config, seen)
    assert seen.during and set(seen.during) == {(tf32, tf32)}
    assert switches() == (True, False), "a run leaves the process as it found it"
    # the snapshot records what the run computed in
    env = json.loads((state.run.dir / "env.json").read_text())
    assert env["tf32"] == {"convolutions": tf32, "matmul": tf32}


@pytest.mark.parametrize("tf32", [False, True])
def test_the_header_of_a_run_says_when_tf32_is_on(tmp_path, pytorch_defaults, tf32):
    from grasppanda.engine import commands

    state = commands.train(tiny_config(tmp_path, {"runtime.tf32": tf32, "train.epochs": 1}))
    header = (state.run.dir / "log.txt").read_text().split("epoch 1/1")[0]
    assert "deterministic off" in header and ("· TF32 on" in header) == tf32


def test_float32_is_the_default_and_leaves_every_hash_as_it_was(tmp_path):
    plain = tiny_config(tmp_path)
    assert plain.resolved()["runtime"]["tf32"] is False
    stated = tiny_config(tmp_path, {"runtime.tf32": False})
    faster = tiny_config(tmp_path, {"runtime.tf32": True})
    assert plain.digest() == stated.digest(), "the default adds nothing to the hash"
    assert faster.digest() != plain.digest(), "TF32 is another experiment"


def test_the_environment_report_says_what_the_process_computes_in(pytorch_defaults):
    assert collect_env()["tf32"] == {"convolutions": True, "matmul": False}
    with start_run_free_precision(False):
        assert collect_env()["tf32"] == {"convolutions": False, "matmul": False}


class start_run_free_precision:
    """``set_precision`` inside the restoring context, as a run does it."""

    def __init__(self, tf32: bool) -> None:
        self.tf32, self.context = tf32, determinism_restored()

    def __enter__(self) -> None:
        self.context.__enter__()
        set_precision(self.tf32)

    def __exit__(self, *error: object) -> None:
        self.context.__exit__(*error)


def test_start_run_restores_the_switches_when_the_run_fails(tmp_path, pytorch_defaults):
    config = tiny_config(tmp_path)
    with pytest.raises(RuntimeError, match="stop"), start_run(config, "train"):
        assert switches() == (False, False)
        raise RuntimeError("stop")
    assert switches() == (True, False)


def test_a_memory_limit_is_checked_and_does_nothing_on_the_cpu(tmp_path):
    from grasppanda.engine.errors import EngineError
    from grasppanda.engine.run import limit_gpu_memory

    limit_gpu_memory(None, torch.device("cpu"))
    limit_gpu_memory(0.5, torch.device("cpu"))  # a share of a GPU: nothing to do here
    for wrong in (0.0, -0.2, 1.5):
        with pytest.raises(EngineError, match="share of the GPU's memory"):
            limit_gpu_memory(wrong, torch.device("cpu"))
    config = tiny_config(tmp_path, {"runtime.gpu_memory_fraction": 0.5})
    assert config.digest() == tiny_config(tmp_path).digest()  # where a run lives, not what it is
    with start_run(config, "train") as run:
        assert run.cfg["runtime"]["gpu_memory_fraction"] == 0.5
