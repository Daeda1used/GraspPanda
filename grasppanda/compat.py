"""Small scoped runtime shims, never edits vendored source."""
import collections.abc
import sys
import types


def legacy_torch():
    if "torch._six" not in sys.modules:
        module = types.ModuleType("torch._six")
        module.container_abcs = collections.abc
        module.string_classes = (str, bytes)
        module.int_classes = (int,)
        sys.modules["torch._six"] = module


def legacy_dgl():
    """Retain the old graph type name used by the pinned GraNet collator."""
    import importlib
    heterograph=importlib.import_module('dgl.heterograph')
    if not hasattr(heterograph,'DGLHeteroGraph'):
        heterograph.DGLHeteroGraph=heterograph.DGLGraph


def triton_driver():
    """Supply a local linker alias when containers mount only libcuda.so.1."""
    import os
    from pathlib import Path
    from triton.backends.nvidia import driver
    from .config import ROOT
    if os.environ.get('TRITON_LIBCUDA_PATH'):
        return
    directories = [Path(p) for p in driver.libcuda_dirs()]
    if any((p/'libcuda.so').is_file() for p in directories):
        return
    library = next((p/'libcuda.so.1' for p in directories if (p/'libcuda.so.1').is_file()), None)
    if library is None:
        raise ValueError('Triton requires an installed NVIDIA GPU driver')
    cache = ROOT/'environments/triton-driver'
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache/f'libcuda.so.{os.getpid()}.tmp'
    temporary.symlink_to(library.resolve())
    temporary.replace(cache/'libcuda.so')
    os.environ['TRITON_LIBCUDA_PATH'] = str(cache)
    driver.libcuda_dirs.cache_clear()
    driver.library_dirs.cache_clear()
