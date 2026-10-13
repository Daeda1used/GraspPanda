"""Backend selection of the accelerated operators, and the optional build behind it.

What is asserted here is the behaviour around the kernels, not the kernels themselves
(``test_ops_equivalence.py``): how a backend is selected and reported, that a machine
without a toolchain keeps working with a single log line, that an explicit request
fails loudly, and that a kernel which disagrees with the reference is never used.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from grasppanda._ops import build, library
from grasppanda.config import Config, ConfigError
from grasppanda.ops import backends, ball_query, cylinder_query, furthest_point_sample, knn, query
from grasppanda.utils.env import collect_env

from .conftest import REPO

ALL = dict.fromkeys(backends.OPERATORS, "compiled")


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    """Every test starts unconfigured and leaves no selection behind."""
    monkeypatch.delenv(backends.ENV, raising=False)
    monkeypatch.setattr(backends, "_warned", set())
    backends.configure()
    yield
    backends.configure()


@pytest.fixture
def kernels():
    loaded, detail = library.kernels()
    if loaded is None:
        pytest.skip(f"no compiled kernels on this machine: {detail}")
    return loaded


@pytest.fixture
def no_kernels(monkeypatch):
    """A machine on which the kernels could not be built."""
    reason = "no C++ compiler found (set CXX or install g++)"
    monkeypatch.setattr(library, "kernels", lambda: (None, reason))
    return reason


def cloud(count: int = 64, seed: int = 0) -> torch.Tensor:
    return torch.rand(2, count, 3, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------ selection


def test_default_is_auto_and_configure_selects_per_operator():
    assert {backends.requested(name) for name in backends.OPERATORS} == {"auto"}
    backends.configure({"fps": "reference"})
    assert backends.requested("fps") == "reference" and backends.requested("knn") == "auto"
    backends.configure({})
    assert backends.requested("fps") == "auto"


def test_configure_rejects_unknown_names_with_a_hint():
    with pytest.raises(ValueError, match=r"unknown operator 'fp'. Did you mean 'fps'\?"):
        backends.configure({"fp": "reference"})
    with pytest.raises(
        ValueError, match="runtime.ops.fps: unknown backend 'fast'; expected one of"
    ):
        backends.configure({"fps": "fast"})
    assert backends.requested("fps") == "auto"  # a rejected selection changes nothing


def test_environment_overrides_the_configuration(monkeypatch):
    backends.configure({"fps": "compiled", "knn": "compiled"})
    monkeypatch.setenv(backends.ENV, "reference")
    assert {backends.requested(name) for name in backends.OPERATORS} == {"reference"}
    monkeypatch.setenv(backends.ENV, "fps=reference, ball_query=auto")
    assert backends.requested("fps") == "reference"
    assert backends.requested("ball_query") == "auto"
    assert backends.requested("knn") == "compiled"  # not mentioned: the configuration stands
    monkeypatch.setenv(backends.ENV, "fps=quick")
    with pytest.raises(ValueError, match="GRASPPANDA_OPS.fps: unknown backend 'quick'"):
        backends.requested("fps")


def test_override_is_temporary():
    backends.configure({"fps": "reference"})
    with backends.override(fps="auto", knn="reference"):
        assert backends.requested("fps") == "auto" and backends.requested("knn") == "reference"
    assert backends.requested("fps") == "reference" and backends.requested("knn") == "auto"
    with pytest.raises(ValueError, match="unknown operator 'nms'"), backends.override(nms="auto"):
        pass


def test_runtime_ops_is_validated_by_the_config_schema():
    config = Config.from_dict({"runtime": {"ops": {"fps": "reference", "knn": "auto"}}})
    assert config.resolved()["runtime"]["ops"] == {"fps": "reference", "knn": "auto"}
    # Nothing is filled in by default, so configs that do not mention it keep their hash.
    assert Config.from_dict({}).resolved()["runtime"]["ops"] == {}
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"runtime": {"ops": {"fps": "fastest"}}})
    (issue,) = error.value.issues
    assert issue.path == "runtime.ops.fps"
    assert "'auto', 'reference', 'compiled'" in issue.message


# --------------------------------------------------------- without a compiled library


def test_auto_falls_back_to_the_reference_with_one_log_line(no_kernels, caplog):
    points = cloud()
    with caplog.at_level(logging.INFO, logger="grasppanda"):
        expected = furthest_point_sample(points, 8)
        index, hits = ball_query(points[:, :5], points, 0.3, 4)
        knn(points[:, :5], points, 3)
        furthest_point_sample(points, 8)
    assert index.shape == (2, 5, 4) and hits.shape == (2, 5)
    assert all(backends.kernels_for(name, points) is None for name in backends.OPERATORS)
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1  # one line for the process, whatever the number of operators
    assert no_kernels in warnings[0].getMessage()
    assert "pure-PyTorch reference" in warnings[0].getMessage()
    with backends.override(fps="reference"):
        assert torch.equal(furthest_point_sample(points, 8), expected)


def test_an_explicit_backend_that_cannot_be_used_is_an_error(no_kernels):
    backends.configure({"fps": "compiled"})
    with pytest.raises(backends.OperatorBackendError) as error:
        furthest_point_sample(cloud(), 8)
    message = str(error.value)
    assert "operator 'fps' is set to the 'compiled' backend" in message and no_kernels in message
    assert "python -m grasppanda._ops.build" in message and "'auto'" in message
    knn(cloud(), cloud(), 1)  # other operators are on auto and keep working


def test_report_and_environment_record_what_runs(no_kernels):
    backends.configure({"knn": "reference", "fps": "compiled"})
    report = backends.report()
    assert set(report) == {*backends.OPERATORS, "library"}
    assert report["knn"] == {"requested": "reference", "cpu": "reference"} or (
        report["knn"]["cuda"] == "reference"
    )
    assert (
        report["ball_query"]["requested"] == "auto" and report["ball_query"]["cpu"] == "reference"
    )
    assert no_kernels in report["ball_query"]["note"]["cpu"]
    assert report["fps"]["cpu"] == "unavailable"  # asked for explicitly, not there
    assert report["library"] == {"path": None, "cuda": False, "detail": no_kernels}
    assert collect_env()["ops"] == report
    json.dumps(report)  # goes into env.json


def test_report_survives_a_malformed_environment_variable(monkeypatch):
    monkeypatch.setenv(backends.ENV, "fps=warp-speed")
    report = collect_env()["ops"]
    assert report["fps"]["requested"].startswith("invalid (GRASPPANDA_OPS.fps: unknown backend")


# ------------------------------------------------------------ with a compiled library


def test_kernels_take_float32_on_one_device_and_nothing_else(kernels):
    points = cloud()
    with backends.override(**ALL):
        assert backends.kernels_for("fps", points) is kernels
        assert backends.kernels_for("knn", points, points.double()) is None
        assert backends.kernels_for("ball_query", points.double(), points.double()) is None
        # Other types run the reference, also when the kernels were asked for explicitly.
        assert furthest_point_sample(points.double(), 8).shape == (2, 8)
    with backends.override(fps="reference"):
        assert backends.kernels_for("fps", points) is None
    report = backends.report()
    assert report["fps"]["cpu"] == "compiled" and report["library"]["path"] == str(kernels.path)


def test_a_kernel_that_disagrees_with_the_reference_is_rejected(kernels, monkeypatch, caplog):
    class Disagreeing:
        """The real kernels, except that furthest point sampling picks other points."""

        def __getattr__(self, name):
            return getattr(kernels, name)

        def fps(self, points, num_samples, **options):
            return kernels.fps(points, num_samples, **options).flip(-1)

    monkeypatch.setattr(library, "kernels", lambda: (Disagreeing(), "CPU kernels only"))
    monkeypatch.setattr(backends, "_checked", {})
    points = cloud()
    with caplog.at_level(logging.WARNING, logger="grasppanda"):
        assert backends.kernels_for("fps", points) is None  # auto: the reference
        assert backends.kernels_for("knn", points, points) is not None  # the others pass
    (warning,) = caplog.records
    assert "fps on cpu uses the pure-PyTorch reference" in warning.getMessage()
    assert "picks differ" in warning.getMessage()
    assert "self-check failed" in backends.report()["fps"]["note"]["cpu"]
    backends.configure({"fps": "compiled"})
    with pytest.raises(backends.OperatorBackendError, match="its self-check failed: picks differ"):
        furthest_point_sample(points, 8)


def test_knn_keeps_gradients_by_running_the_reference(kernels):
    support, centres = cloud(), cloud(9, seed=1).requires_grad_()
    with backends.override(**ALL):
        distance, _ = knn(centres, support, 3)
        assert distance.requires_grad
        distance.sum().backward()
        assert centres.grad is not None and bool(centres.grad.abs().sum() > 0)
        with torch.no_grad():
            assert not knn(centres, support, 3)[0].requires_grad


def test_cylinder_query_steps_aside_when_the_reference_computes_in_reduced_precision(
    kernels, monkeypatch
):
    calls = []
    compiled = query._cylinder_query_compiled
    monkeypatch.setattr(
        query, "_cylinder_query_compiled", lambda *a: calls.append(1) or compiled(*a)
    )
    points = cloud()
    rotation = torch.eye(3).expand(2, 5, 3, 3)
    arguments = (points[:, :5], points, rotation, 0.3, -0.1, 0.2, 4)
    with backends.override(**ALL):
        cylinder_query(*arguments)
        assert len(calls) == 1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cylinder_query(*arguments)  # the matrix product of the reference is autocast
        assert len(calls) == 1


# ------------------------------------------------------------------------ the build


def toolchain(**changes) -> build.Toolchain:
    facts = {
        "ninja": True,
        "compiler": "/usr/bin/c++",
        "compiler_version": "c++ 11.4",
        "cuda_home": None,
        "cuda_version": None,
        "cuda_note": "no CUDA toolkit found",
    }
    return build.Toolchain(**{**facts, **changes})


@pytest.fixture
def empty_cache(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv(build.ENV_CACHE, str(tmp_path))
    monkeypatch.delenv(build.ENV_BUILD, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ({"compiler": None, "compiler_version": None}, "no C++ compiler found"),
        ({"ninja": False}, "ninja not found (pip install ninja)"),
    ],
)
def test_a_missing_toolchain_is_reported_not_raised(empty_cache, monkeypatch, missing, message):
    monkeypatch.setattr(build, "toolchain", lambda: toolchain(**missing))
    monkeypatch.setattr(build, "_compile", lambda *a: pytest.fail("nothing can be compiled"))
    result = build.ensure()
    assert result.path is None and not result.cuda and message in result.detail
    assert not list(empty_cache.iterdir())  # nothing was written


def test_a_failed_build_is_recorded_and_not_repeated(empty_cache, monkeypatch):
    attempts = []

    def failing(variant, *_):
        attempts.append(variant)
        raise RuntimeError("ninja: build stopped.\nkernels_cpu.cpp:7:10: fatal error: boom\n")

    monkeypatch.setattr(build, "toolchain", toolchain)
    monkeypatch.setattr(build, "_compile", failing)
    first = build.ensure(cpu_only=True)
    assert (
        first.path is None
        and "the cpu build failed (kernels_cpu.cpp:7:10: fatal error: boom" in first.detail
    )
    (log,) = empty_cache.glob("*/build.log")
    assert "ninja: build stopped." in log.read_text()
    second = build.ensure(cpu_only=True)
    assert attempts == ["cpu"]  # the second call did not compile again
    assert "failed before" in second.detail and "--force" in second.detail
    build.ensure(cpu_only=True, force=True)
    assert attempts == ["cpu", "cpu"]
    # Another compiler is another chance.
    monkeypatch.setattr(build, "toolchain", lambda: toolchain(compiler_version="c++ 13.2"))
    build.ensure(cpu_only=True)
    assert attempts == ["cpu", "cpu", "cpu"]


def test_building_on_first_use_can_be_switched_off(empty_cache, monkeypatch):
    monkeypatch.setenv(build.ENV_BUILD, "0")
    monkeypatch.setattr(build, "_compile", lambda *a: pytest.fail("must not compile"))
    result = build.ensure(cpu_only=True)
    assert result.path is None and result.detail == "cpu kernels are not built"


def test_cpu_kernels_are_built_when_the_cuda_build_fails(empty_cache, monkeypatch):
    if not torch.version.cuda:
        pytest.skip("this PyTorch build has no CUDA support")

    def compiling(variant, tools, directory, name, verbose):
        if variant == "cuda":
            raise RuntimeError(
                "nvcc fatal   : Value 'sm_90' is not defined for option 'gpu-architecture'"
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.so").write_bytes(b"")
        return directory / f"{name}.so"

    monkeypatch.setattr(
        build, "toolchain", lambda: toolchain(cuda_home="/opt/cuda", cuda_version="11.7")
    )
    monkeypatch.setattr(build, "_compile", compiling)
    result = build.ensure()
    assert result.path is not None and result.path.is_file() and result.built and not result.cuda
    assert result.detail.startswith("CPU kernels only (the cuda build failed (nvcc fatal")
    again = build.ensure()  # found in the cache, and the reason is still given
    assert again.path == result.path and not again.built and "failed before" in again.detail


def test_the_cuda_toolkit_must_match_the_major_version_of_pytorch(monkeypatch):
    versions = {"/opt/env": (11, 7), "/usr/local/cuda-11.8": (11, 8), "/usr/local/cuda": (12, 1)}
    monkeypatch.setattr(build, "_nvcc_version", versions.get)
    monkeypatch.setattr(build.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(torch.version, "cuda", "11.8")
    monkeypatch.setenv("CUDA_HOME", "/opt/env")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    home, version, note = build._cuda_toolkit()
    assert (home, version) == ("/usr/local/cuda-11.8", "11.8")  # exact beats $CUDA_HOME
    del versions["/usr/local/cuda-11.8"]
    assert build._cuda_toolkit()[:2] == ("/opt/env", "11.7")  # same major version: usable
    del versions["/opt/env"]
    home, version, note = build._cuda_toolkit()
    assert home is None and "no CUDA 11.x toolkit" in note and "12.1 at /usr/local/cuda" in note
    monkeypatch.setattr(torch.version, "cuda", None)
    assert build._cuda_toolkit() == (None, None, "this PyTorch build has no CUDA support")


def test_a_process_without_a_compiler_runs_on_the_reference(tmp_path):
    """End to end, in a fresh interpreter: no compiler, an empty cache, default settings."""
    script = (
        "import json, torch\n"
        "from grasppanda.ops import backends, furthest_point_sample, knn\n"
        "points = torch.rand(1, 50, 3, generator=torch.Generator().manual_seed(0))\n"
        "picks = furthest_point_sample(points, 5).tolist()\n"
        "knn(points, points, 2)\n"
        "print(json.dumps({'picks': picks, 'report': backends.report()}))\n"
    )
    environment = {
        **os.environ,
        "CXX": str(tmp_path / "no-such-compiler"),
        build.ENV_CACHE: str(tmp_path / "cache"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    environment.pop(backends.ENV, None)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["picks"][0] == [0, *output["picks"][0][1:]] and len(output["picks"][0]) == 5
    assert {output["report"][name]["cpu"] for name in backends.OPERATORS} == {"reference"}
    assert output["report"]["library"]["path"] is None
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(lines) == 1 and "no C++ compiler found" in lines[0], result.stderr
    assert not (tmp_path / "cache").exists()
