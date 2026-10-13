"""Build and locate the compiled operator kernels.

The kernels are optional. They are compiled on first use with PyTorch's own extension
tooling (:func:`torch.utils.cpp_extension.load`: a C++ compiler and ``ninja``, plus a
CUDA toolkit of the major version PyTorch was built with for the CUDA kernels) and
cached below ``$GRASPPANDA_OPS_CACHE`` (default ``<cache root>/ops``), one directory per
PyTorch version, CUDA version, platform and source revision. Nothing here raises when a
toolchain is missing or a build fails: the caller gets a :class:`Library` without a
path and a sentence saying why, and falls back to the pure-PyTorch operators.

Build ahead of time, or see what a machine can build::

    python -m grasppanda._ops.build            # build if needed and report
    python -m grasppanda._ops.build --force    # rebuild, also after a recorded failure
    python -m grasppanda._ops.build --cpu-only

A library holds CPU kernels (C++, OpenMP) and, when a toolkit was found, CUDA kernels.
The CUDA kernels are compiled to PTX for the oldest architecture PyTorch supports and
specialised by the driver when first used, so one build serves every GPU of a machine
and a build made on a login node serves its compute nodes
(``TORCH_CUDA_ARCH_LIST`` overrides the architecture list, as for any PyTorch extension).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from grasppanda.utils.paths import cache_root

__all__ = ["ENV_BUILD", "ENV_CACHE", "Library", "Toolchain", "cache_dir", "ensure", "toolchain"]

#: Directory of the compiled libraries; overrides ``<cache root>/ops``.
ENV_CACHE = "GRASPPANDA_OPS_CACHE"
#: Set to ``0`` to use only libraries that are already built (never compile on first use).
ENV_BUILD = "GRASPPANDA_OPS_BUILD"

_SOURCES = Path(__file__).parent / "src"
_CPU_SOURCES = ("kernels_cpu.cpp",)
_CUDA_SOURCES = ("kernels_cuda.cu",)
_HEADERS = ("common.h",)
#: -ffp-contract=off: a fused multiply-add rounds once where PyTorch rounds twice.
#: -fno-trapping-math, -fno-math-errno: let the compiler vectorise branch-free loops; they
#: declare floating-point exception flags unobserved and do not change any value.
_CFLAGS = (
    "-O3",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fno-trapping-math",
    "-fno-math-errno",
    "-fopenmp",
    "-fvisibility=hidden",
)
_CUDA_CFLAGS = ("-O3", "-fmad=false")
#: --as-needed: the kernels use no PyTorch symbol, so the library does not depend on libtorch.
_LDFLAGS = ("-Wl,--as-needed", "-fopenmp")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Toolchain:
    """What this machine can build with.

    Attributes:
        ninja: Whether ``ninja`` is available to PyTorch's extension tooling.
        compiler: The C++ compiler (``$CXX`` or ``c++``) and its version line.
        cuda_home: CUDA toolkit chosen for the CUDA kernels.
        cuda_note: Why that toolkit was chosen, or why there is none.
    """

    ninja: bool
    compiler: str | None
    compiler_version: str | None
    cuda_home: str | None
    cuda_version: str | None
    cuda_note: str

    @property
    def missing(self) -> str | None:
        """What a CPU build lacks, as a sentence; ``None`` when nothing."""
        if self.compiler is None:
            return "no C++ compiler found (set CXX or install g++)"
        if not self.ninja:
            return "ninja not found (pip install ninja)"
        return None

    @property
    def identity(self) -> str:
        return f"{self.compiler_version} | ninja={self.ninja} | cuda={self.cuda_version}"


@dataclass(frozen=True)
class Library:
    """Result of :func:`ensure`.

    Attributes:
        path: The shared library, or ``None`` when there is none.
        cuda: Whether it contains the CUDA kernels.
        detail: One sentence: what was found or built, or why there is nothing.
        built: Compiled by this call (``False``: found in the cache).
        seconds: Time spent compiling.
    """

    path: Path | None
    cuda: bool
    detail: str
    built: bool = False
    seconds: float = 0.0


def cache_dir() -> Path:
    """Where compiled libraries are kept."""
    configured = os.environ.get(ENV_CACHE)
    return Path(configured).expanduser() if configured else cache_root() / "ops"


# ----------------------------------------------------------------------- toolchain


def _first_line(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0].strip() if result.returncode == 0 and lines else None


def _nvcc_version(home: str) -> tuple[int, int] | None:
    nvcc = Path(home) / "bin" / "nvcc"
    if not nvcc.is_file():
        return None
    try:
        result = subprocess.run(
            [str(nvcc), "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", result.stdout)
    return (int(match[1]), int(match[2])) if match else None


def _cuda_toolkit() -> tuple[str | None, str | None, str]:
    """Pick the CUDA toolkit to compile with: ``(home, version, note)``.

    PyTorch's tooling takes ``$CUDA_HOME`` or the first ``nvcc`` on the path, whatever
    its version. Kernels have to be compiled for the CUDA *major* version PyTorch was
    built with (they link against its runtime), so the candidates are checked: an
    exact version match wins wherever it is installed, then the first toolkit of the
    same major version in the order environment, path, ``/usr/local/cuda*``.
    """
    built_with = torch.version.cuda
    if not built_with:
        return None, None, "this PyTorch build has no CUDA support"
    wanted = tuple(int(part) for part in built_with.split(".")[:2])
    candidates = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parent.parent))
    candidates += [f"/usr/local/cuda-{built_with}", "/usr/local/cuda"]
    candidates += sorted(glob.glob(f"/usr/local/cuda-{wanted[0]}.*"), reverse=True)
    found: dict[str, tuple[int, int]] = {}
    for home in candidates:
        if home and home not in found:
            version = _nvcc_version(home)
            if version is not None:
                found[home] = version
    for match in (lambda v: v == wanted, lambda v: v[0] == wanted[0]):
        for home, version in found.items():
            if match(version):
                text = f"{version[0]}.{version[1]}"
                return home, text, f"CUDA toolkit {text} at {home} (PyTorch: CUDA {built_with})"
    if found:
        seen = ", ".join(f"{v[0]}.{v[1]} at {home}" for home, v in found.items())
        return None, None, f"no CUDA {wanted[0]}.x toolkit for this PyTorch build (found {seen})"
    return None, None, f"no CUDA toolkit found (PyTorch: CUDA {built_with}; set CUDA_HOME)"


def _cpp_extension() -> Any:
    """PyTorch's extension tooling, imported quietly.

    On a machine without a GPU the module prints which ``CUDA_HOME`` it guessed while
    it is imported; the toolkit is chosen here (:func:`_cuda_toolkit`), so that line
    would only be noise next to the one line a missing backend is allowed.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        from torch.utils import cpp_extension
    return cpp_extension


def toolchain() -> Toolchain:
    """Inspect the machine; never raises."""
    try:
        ninja = bool(_cpp_extension().is_ninja_available())
    except Exception:  # an unusable extension module is a missing toolchain
        ninja = False
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    home, version, note = _cuda_toolkit()
    return Toolchain(
        ninja=ninja,
        compiler=compiler,
        compiler_version=_first_line([compiler, "--version"]) if compiler else None,
        cuda_home=home,
        cuda_version=version,
        cuda_note=note,
    )


def _arch_list() -> str:
    """``TORCH_CUDA_ARCH_LIST`` for the build: the user's, or PTX for the oldest architecture."""
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured:
        return configured
    oldest = 60  # Pascal; older architectures are deprecated by every supported toolkit
    try:
        flags = torch._C._cuda_getArchFlags()  # architectures PyTorch itself was built for
        supported = sorted(int(v) for v in re.findall(r"sm_(\d+)", flags or ""))
        oldest = next((v for v in supported if v >= oldest), oldest)
    except Exception:
        pass
    return f"{oldest // 10}.{oldest % 10}+PTX"


# ----------------------------------------------------------------------------- keys


def _digest(variant: str) -> str:
    """Hash of everything that decides whether a cached library is still the right one."""
    names = (*_CPU_SOURCES, *_HEADERS, *(_CUDA_SOURCES if variant == "cuda" else ()))
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update((_SOURCES / name).read_bytes())
    facts = [variant, torch.__version__, platform.system(), platform.machine(), *_CFLAGS, *_LDFLAGS]
    if variant == "cuda":
        facts += [str(torch.version.cuda), _arch_list(), *_CUDA_CFLAGS]
    digest.update("\0".join(facts).encode())
    return digest.hexdigest()


def _slot(variant: str) -> tuple[Path, str]:
    """Cache directory and library name of a variant (``cpu`` or ``cuda``)."""
    digest = _digest(variant)
    version = re.sub(r"[^0-9A-Za-z.]+", "_", torch.__version__)
    directory = cache_dir() / f"torch{version}-{variant}-{platform.machine()}-{digest[:12]}"
    # The name is unique per build input, so PyTorch's tooling never renames the module.
    return directory, f"grasppanda_ops_{digest[:16]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock so that concurrent processes build once and the rest wait.

    The lock belongs to the process, so it cannot be left behind by a crash. Where the
    file system has no locks the build simply is not serialised: it happens in a private
    directory and is published atomically either way.
    """
    try:
        import fcntl
    except ImportError:  # not POSIX
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        with contextlib.suppress(OSError):
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


# ---------------------------------------------------------------------------- build


@contextlib.contextmanager
def _extension_tooling(cuda_home: str | None) -> Iterator[Any]:
    """PyTorch's extension module, pointed at the chosen toolkit and architecture list."""
    cpp_extension = _cpp_extension()
    saved_home = cpp_extension.CUDA_HOME
    saved_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_home is not None:
        cpp_extension.CUDA_HOME = cuda_home
        os.environ["TORCH_CUDA_ARCH_LIST"] = _arch_list()
    try:
        yield cpp_extension
    finally:
        cpp_extension.CUDA_HOME = saved_home
        if cuda_home is not None:
            if saved_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = saved_arch


def _compile(variant: str, tools: Toolchain, directory: Path, name: str, verbose: bool) -> Path:
    """Compile one variant into ``directory`` and return the library; raises on failure."""
    cuda = variant == "cuda"
    sources = [str(_SOURCES / n) for n in (*_CPU_SOURCES, *(_CUDA_SOURCES if cuda else ()))]
    directory.mkdir(parents=True, exist_ok=True)
    # A private directory per attempt: nothing half-built is ever visible to another
    # process, and no lock file of an interrupted build can block the next one.
    workspace = directory / f"build-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    try:
        tooling = _extension_tooling(tools.cuda_home if cuda else None)
        with tooling as cpp_extension, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cpp_extension.load(
                name=name,
                sources=sources,
                extra_cflags=[*_CFLAGS, *(["-DGP_WITH_CUDA"] if cuda else [])],
                extra_cuda_cflags=list(_CUDA_CFLAGS) if cuda else None,
                extra_ldflags=list(_LDFLAGS),
                extra_include_paths=[str(_SOURCES)],
                build_directory=str(workspace),
                verbose=verbose,
                with_cuda=cuda,
                is_python_module=False,
            )
        built = workspace / f"{name}.so"
        if not built.is_file():
            raise RuntimeError(f"the build finished without producing {built.name}")
        staged = directory / f".{name}.{uuid.uuid4().hex}.so"
        shutil.copyfile(built, staged)
        os.replace(staged, directory / f"{name}.so")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return directory / f"{name}.so"


def _summary(error: BaseException) -> str:
    """The line of a build error that says what went wrong."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    telling = [line for line in lines if re.search(r"\berror\b|not found|No such file", line, re.I)]
    text = (telling or lines or [type(error).__name__])[0]
    return text if len(text) <= 200 else text[:197] + "..."


def ensure(
    *,
    build: bool | None = None,
    force: bool = False,
    cpu_only: bool = False,
    verbose: bool = False,
) -> Library:
    """Return the compiled library, building it when necessary and possible.

    The CUDA variant is preferred when PyTorch has CUDA support; without a matching
    toolkit (or after a failed CUDA build) the CPU variant is used, and CUDA tensors
    then go through the pure-PyTorch operators.

    Args:
        build: Compile when no cached library fits. ``None`` reads ``$GRASPPANDA_OPS_BUILD``
            (default: yes).
        force: Rebuild even if a library is cached or an earlier build failed.
        cpu_only: Do not look for or build the CUDA variant.
        verbose: Show the compiler output.

    Never raises: a failure is described in :attr:`Library.detail`.
    """
    try:
        return _ensure(build, force, cpu_only, verbose)
    except Exception as error:  # the compiled backend is optional under all circumstances
        return Library(None, False, f"unexpected error: {type(error).__name__}: {error}")


def _ensure(build: bool | None, force: bool, cpu_only: bool, verbose: bool) -> Library:
    if build is None:
        build = os.environ.get(ENV_BUILD, "1").strip().lower() not in ("0", "false", "no", "off")
    tools: Toolchain | None = None
    notes: list[str] = []
    variants = ["cpu"] if cpu_only or not torch.version.cuda else ["cuda", "cpu"]
    for variant in variants:
        cuda = variant == "cuda"
        directory, name = _slot(variant)
        library = directory / f"{name}.so"
        if library.is_file() and (directory / "build.json").is_file() and not force:
            return Library(library, cuda, _describe(cuda, notes))
        if not build and not force:
            notes.append(f"{variant} kernels are not built")
            continue
        tools = tools or toolchain()
        if tools.missing:  # a library cached for the other variant can still be used
            if tools.missing not in notes:
                notes.append(tools.missing)
            continue
        if cuda and tools.cuda_home is None:
            notes.append(tools.cuda_note)
            continue
        failure = directory / "failure.json"
        record = _read_json(failure) if not force else None
        if record is not None and record.get("toolchain") == tools.identity:
            notes.append(
                f"the {variant} build failed before ({record.get('error')}; log: "
                f"{directory / 'build.log'}; retry with `python -m grasppanda._ops.build --force`)"
            )
            continue
        with _locked(directory.with_name(directory.name + ".lock")):
            if library.is_file() and (directory / "build.json").is_file() and not force:
                return Library(library, cuda, _describe(cuda, notes))  # built while we waited
            _log.info(
                "Building the %s operator kernels for torch %s (once; cached in %s) ...",
                variant,
                torch.__version__,
                directory,
            )
            start = time.perf_counter()
            try:
                _compile(variant, tools, directory, name, verbose)
            except Exception as error:
                summary = _summary(error)
                with contextlib.suppress(OSError):  # a cache that cannot be written
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "build.log").write_text(str(error), encoding="utf-8")
                    _write_json(failure, {"error": summary, "toolchain": tools.identity})
                notes.append(
                    f"the {variant} build failed ({summary}; log: {directory / 'build.log'})"
                )
                continue
            seconds = time.perf_counter() - start
            with contextlib.suppress(OSError):
                failure.unlink()
            _write_json(
                directory / "build.json",
                {
                    "library": library.name,
                    "variant": variant,
                    "torch": torch.__version__,
                    "cuda_toolkit": tools.cuda_home if cuda else None,
                    "cuda_architectures": _arch_list() if cuda else None,
                    "compiler": tools.compiler_version,
                    "seconds": round(seconds, 1),
                },
            )
            _log.info("Built %s in %.0f s.", library, seconds)
            return Library(library, cuda, _describe(cuda, notes), built=True, seconds=seconds)
    return Library(None, False, "; ".join(notes) or "nothing to build")


def _describe(cuda: bool, notes: Sequence[str]) -> str:
    text = "CPU and CUDA kernels" if cuda else "CPU kernels only"
    return f"{text} ({'; '.join(notes)})" if notes else text


# ------------------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grasppanda._ops.build",
        description="Build the optional compiled operator kernels and report what is in use.",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if cached or failed")
    parser.add_argument("--cpu-only", action="store_true", help="do not build the CUDA kernels")
    parser.add_argument("--verbose", action="store_true", help="show the compiler output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Imported here: this file also runs as __main__, and the process-wide state (the
    # loaded library, the resolved backends) must live in modules that exist once.
    from grasppanda._ops import library
    from grasppanda.ops import backends
    from grasppanda.utils.console import table

    tools = toolchain()
    built = ensure(build=True, force=args.force, cpu_only=args.cpu_only, verbose=args.verbose)
    rows = [
        ["torch", f"{torch.__version__} (CUDA build {torch.version.cuda or 'none'})"],
        ["compiler", tools.compiler_version or "not found"],
        ["ninja", "yes" if tools.ninja else "not found"],
        ["CUDA toolkit", tools.cuda_note],
        ["cache", str(cache_dir())],
        ["library", str(built.path) if built.path else "none"],
        ["kernels", built.detail],
        ["built now", f"yes, in {built.seconds:.0f} s" if built.built else "no"],
    ]
    print(table(["item", "value"], rows))
    if built.path is None:
        print("\nNo compiled kernels: the pure-PyTorch operators are used (same results, slower).")
        return 1
    library.reset()  # pick up what was just built
    report = backends.report()
    print()
    devices = [key for key in ("cpu", "cuda") if any(key in entry for entry in report.values())]
    print(
        table(
            ["operator", "requested", *devices],
            [
                [name, entry["requested"], *(str(entry.get(d, "-")) for d in devices)]
                for name, entry in report.items()
                if name in backends.OPERATORS
            ],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
