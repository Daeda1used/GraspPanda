"""Turn a released encoder checkpoint into weights for the ``backbone`` of a GraspPanda model.

Some published checkpoints of other tasks fit a registered scene encoder key for key.
This tool cuts the encoder out of such a file, checks it **strictly** against the class
it is meant for (every tensor of the encoder present with the right shape; what is left
over must be on the list of parts the format is known to carry, such as a segmentation
head), and writes a state dict whose keys start with ``backbone.``:

    python -m tools.convert_backbone_weights pointnext-s_ckpt_best.pth pointnext_s_s3dis.pth \
        --format pointnext-s
    python -m tools.convert_backbone_weights pretrained_votenet_on_sunrgbd.tar votenet_sunrgbd.pth \
        --format votenet
    python -m tools.convert_backbone_weights partition8_4096_15k_pointnet.pth csc_pointnet2.pth \
        --format csc-pointnet2

Formats (``--list`` prints them with the backbone config each one needs):

``pointnext-s``, ``pointnext-b``, ``pointnext-l``, ``pointnext-xl``
    The S3DIS segmentation checkpoints of the official PointNeXt model zoo
    (``guochengqian/PointNeXt``, ``docs/modelzoo.md``). 4 input channels (colour and
    height), room radii. The ``head.*`` entries are left behind.
``votenet``
    The official VoteNet checkpoints (``facebookresearch/votenet``, SUN RGB-D or
    ScanNet): the ``backbone_net.*`` entries, for ``PointNet2`` with one input channel
    (height) and the VoteNet radii.
``csc-pointnet2``
    The PointNet++ pre-training file of Contrastive Scene Contexts
    (``facebookresearch/ContrastiveSceneContexts``), for ``PointNet2`` at its defaults;
    the two extra propagation levels of the file are left behind.
``csc-pointnet2-dense``
    The same file, all four propagation levels, for ``PointNet2Dense``.

Use the result as an initialisation: put the printed backbone options into the model
config and load with ``runtime.load_from`` and ``runtime.allow_weight_mismatch: true``
(the other stages start from random weights, and the key report says so). No value is
changed and nothing is renamed beyond the prefix. The weights come with the terms of
their source and of the data they were trained on; this tool downloads nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

__all__ = ["FORMATS", "BackboneWeightsError", "convert", "convert_file", "main"]


class BackboneWeightsError(RuntimeError):
    """The file does not hold what the chosen format promises."""


_BASELINE_SA = [[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]]


def _pointnext(**options: Any) -> dict[str, Any]:
    return {
        "container": "model",
        "prefix": "",
        "drop": ("head.",),
        "backbone": {"type": "PointNeXt", "in_channels": 4, **options},
    }


#: format -> where the encoder sits in the file, what else the file may hold, and the
#: backbone config the weights are for.
FORMATS: dict[str, dict[str, Any]] = {
    "pointnext-s": _pointnext(),
    "pointnext-b": _pointnext(blocks=[1, 2, 3, 2, 2], sa_layers=1, sa_use_res=False),
    "pointnext-l": _pointnext(blocks=[1, 3, 5, 3, 3], sa_layers=1, sa_use_res=False),
    "pointnext-xl": _pointnext(blocks=[1, 4, 7, 4, 4], width=64, sa_layers=1, sa_use_res=False),
    "votenet": {
        "container": "model_state_dict",
        "prefix": "backbone_net.",
        "drop": ("vgen.", "pnet."),
        "backbone": {"type": "PointNet2", "in_channels": 1, "radii": [0.2, 0.4, 0.8, 1.2]},
    },
    "csc-pointnet2": {
        "container": "state_dict",
        "prefix": "",
        "drop": ("fp3.", "fp4."),
        "backbone": {"type": "PointNet2", "radii": [0.2, 0.4, 0.8, 1.2]},
    },
    "csc-pointnet2-dense": {
        "container": "state_dict",
        "prefix": "",
        "drop": (),
        "backbone": {
            "type": "PointNet2Dense",
            "npoints": [2048, 1024, 512, 256],
            "radii": [0.2, 0.4, 0.8, 1.2],
            "nsamples": [64, 32, 16, 16],
            "sa_mlps": _BASELINE_SA,
            "fp_mlps": [[256, 256], [256, 256], [256, 128], [128, 32]],
            "normalize_xyz": True,
            "interpolation_power": 1.0,
            "fps_skip_near_origin": True,
        },
    },
}


def convert(
    checkpoint: Mapping[str, Any], source_format: str, *, prefix: str = "backbone."
) -> dict[str, Tensor]:
    """Cut the encoder out of a loaded checkpoint and check it against its class."""
    from grasppanda.registry import BACKBONES

    if source_format not in FORMATS:
        raise BackboneWeightsError(
            f"unknown format '{source_format}'; supported: {', '.join(sorted(FORMATS))}"
        )
    spec = FORMATS[source_format]
    state = checkpoint.get(spec["container"]) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        keys = ", ".join(map(str, list(checkpoint)[:8])) if isinstance(checkpoint, Mapping) else ""
        raise BackboneWeightsError(
            f"format '{source_format}' expects the weights under '{spec['container']}', but the "
            f"top-level keys of the file are: {keys}"
        )
    state = dict(state)
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    # Entries outside the encoder's prefix belong to other modules of the source model;
    # inside it, only the parts the format names may be left behind.
    taken = {}
    for key, value in state.items():
        if not key.startswith(spec["prefix"]):
            continue
        name = key[len(spec["prefix"]) :]
        if spec["drop"] and name.startswith(spec["drop"]):
            continue
        taken[name] = value
    own = BACKBONES.build(dict(spec["backbone"])).state_dict()
    missing = [key for key in own if key not in taken]
    unexpected = [key for key in taken if key not in own]
    wrong = [
        f"{key}: file {tuple(taken[key].shape)}, model {tuple(own[key].shape)}"
        for key in own
        if key in taken and tuple(taken[key].shape) != tuple(own[key].shape)
    ]
    if missing or unexpected or wrong:
        raise BackboneWeightsError(
            f"the file does not fit format '{source_format}' "
            f"({spec['backbone']['type']}): {len(missing)} tensors missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]}), "
            f"{len(wrong)} of another shape (e.g. {wrong[:2]}). Nothing was written."
        )
    return {f"{prefix}{key}": value for key, value in taken.items()}


class _Inert:
    """Stand-in for a class a checkpoint names besides its tensors: it accepts whatever
    the pickle hands it and keeps nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def append(self, value: Any) -> None:
        pass

    def extend(self, values: Any) -> None:
        pass


def load_tensors_only(source: Path) -> Any:
    """``torch.load(weights_only=True)`` that also reads files with config objects in them.

    Research checkpoints often pickle their experiment config (an ``omegaconf`` object,
    an ``argparse.Namespace``) next to the weights, and the safe loader refuses the
    class. Nothing of such a class is ever imported or run here: each refused name gets
    an inert stand-in that ignores its content, and loading is tried again. The tensors
    are read by PyTorch's restricted unpickler throughout.
    """
    if not hasattr(torch.serialization, "safe_globals"):  # PyTorch < 2.5 has no allow-list
        try:
            return torch.load(source, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as error:
            raise BackboneWeightsError(
                f"{source} holds objects besides its tensors, and reading such a file without "
                "running its code needs PyTorch 2.5 or newer (this is "
                f"{torch.__version__}). Upgrade PyTorch for the conversion, or save the "
                "state dict alone in the environment the file came from and convert that."
            ) from error
    allowed: list[Any] = []
    for _ in range(64):
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(source, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as error:
            found = re.search(r"Unsupported global: GLOBAL (\S+) was not", str(error))
            if found is None:
                raise
            module, _, name = found.group(1).rpartition(".")
            if module in ("", "builtins", "__builtin__") and name == "dict":
                allowed.append(dict)
            else:
                allowed.append(type(name, (_Inert,), {"__module__": module or "builtins"}))
    raise BackboneWeightsError(f"{source} names more than 64 classes besides its tensors")


def convert_file(
    source: Path, source_format: str, *, prefix: str = "backbone."
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Convert a checkpoint file; returns the state dict and its provenance record."""
    checkpoint = load_tensors_only(source)
    converted = convert(checkpoint, source_format, prefix=prefix)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    meta = {
        "source_format": source_format,
        "source_file": source.name,
        "source_sha256": digest,
        "backbone": FORMATS[source_format]["backbone"],
        "tensors": len(converted),
    }
    return converted, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, nargs="?", help="released checkpoint")
    parser.add_argument("destination", type=Path, nargs="?", help="file to write")
    parser.add_argument("--format", choices=sorted(FORMATS), dest="source_format")
    parser.add_argument("--prefix", default="backbone.", help="prefix of the written keys")
    parser.add_argument("--list", action="store_true", help="print the formats and exit")
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in FORMATS.items():
            print(f"{name}: model.backbone = {spec['backbone']}")
        return 0
    if args.source is None or args.destination is None or args.source_format is None:
        parser.error("source, destination and --format are required")
    try:
        converted, meta = convert_file(args.source, args.source_format, prefix=args.prefix)
    except BackboneWeightsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    torch.save({"state_dict": converted, "meta": meta}, args.destination)
    print(f"wrote {len(converted)} tensors to {args.destination}")
    print(f"source sha256: {meta['source_sha256']}")
    print(f"model.backbone: {meta['backbone']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
