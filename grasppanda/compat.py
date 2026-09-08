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
