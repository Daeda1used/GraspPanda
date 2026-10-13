"""Fit a released encoder checkpoint to the scene encoder of a GraspPanda config.

Supported encoders and the files that were tried with them:

``PointViT``
    Point-MAE ``pretrain.pth`` (ShapeNet); MaskPoint ``pretrain_shapenet.pth``, which has
    the Point-BERT layout (``token_dim: 256``).
``PointMamba``
    PointMamba ``pretrain.pth`` (ShapeNet).
``Uni3D``
    ``modelzoo/uni3d-ti/model.pt`` and ``modelzoo/uni3d-s/model.pt`` of
    ``huggingface.co/BAAI/Uni3D`` (``timm_model: eva02_tiny_patch14_224`` and
    ``eva02_small_patch14_224``; three colour channels).
``Volt``
    ``model_last.pth`` of the experiments on ``huggingface.co/KadirYilmaz/Volt``
    (ScanNet, ScanNet200, ScanNet++; six input channels, 2 cm grid).

Such a file stores a pre-training or segmentation model: the encoder under a wrapper
prefix, next to decoders, tokens and heads. This tool keeps the tensors that fit the
encoder of the config and writes them under the names the encoder has inside the
detector, so that the file can initialise a training run::

    grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \\
        --set backbone=point-vit -o baseline_point-vit.yaml
    python -m tools.convert_encoder_checkpoint pretrain.pth point-mae_for_baseline.pth \\
        --config baseline_point-vit.yaml
    grasppanda train baseline_point-vit.yaml --opts \\
        runtime.load_from=point-mae_for_baseline.pth runtime.allow_weight_mismatch=true

``allow_weight_mismatch`` is needed because the file holds the encoder only; the trainer
lists every key it did not find. The sizes of the encoder in the config must be those of
the checkpoint. Every tensor of the encoder must be filled, otherwise nothing is
written; a projection to another output width (``proj.*``, ``head.*``, ``cls.*``) is
new by design, and ``--fresh PREFIX`` names further tensors that may stay at their
initial values (``--fresh tokenizer.proj`` for a Volt that reads other input channels
than the checkpoint).

The released weights were trained on other data (single ShapeNet objects, indoor
rooms) with other inputs. They are an initialisation; no accuracy on grasp data is
claimed for them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from grasppanda.config import Config
from grasppanda.models.backbones.point_tokens import convert_point_token_checkpoint
from grasppanda.models.backbones.volt import convert_volt_checkpoint
from grasppanda.registry import BACKBONES

CONVERTERS: dict[str, Callable[..., Any]] = {
    "PointViT": convert_point_token_checkpoint,
    "PointMamba": convert_point_token_checkpoint,
    "Uni3D": convert_point_token_checkpoint,
    "Volt": convert_volt_checkpoint,
}
#: Output projections the encoders add for a chosen width: never in a released file.
NEW_BY_DESIGN = ("proj.", "head.", "cls.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("source", type=Path, help="released checkpoint")
    parser.add_argument("output", type=Path, help="file to write")
    parser.add_argument("--config", required=True, help="config whose model.backbone is filled")
    parser.add_argument(
        "--fresh",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="encoder tensors that may stay at their initial values",
    )
    args = parser.parse_args(argv)

    node = Config.load(args.config)["model"]["backbone"]
    prefix = "backbone."
    while node["type"] not in CONVERTERS and isinstance(node.get("backbone"), dict):
        node, prefix = node["backbone"], prefix + "backbone."  # inside an adapter
    if node["type"] not in CONVERTERS:
        print(f"model.backbone of {args.config} is {node['type']}, not one of {list(CONVERTERS)}")
        return 2
    model = BACKBONES.build(dict(node))
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state, report = CONVERTERS[node["type"]](checkpoint, model, prefix=prefix)
    print(
        f"{args.source}: {len(report['loaded'])} tensors fit {node['type']} (as `{prefix}*`), "
        f"{len(report['dropped'])} dropped (decoders, tokens, heads)"
    )
    allowed = (*NEW_BY_DESIGN, *args.fresh)
    missing = [key for key in report["missing"] if not key.startswith(allowed)]
    kept_fresh = [key for key in report["missing"] if key.startswith(allowed)]
    mismatched = [
        line for line in report["mismatched"] if not any(f".{name}" in line for name in args.fresh)
    ]
    if kept_fresh:
        print(f"  left at their initial values: {', '.join(kept_fresh)}")
    for line in mismatched:
        print(f"  shape differs: {line}")
    for key in missing:
        print(f"  not in the file: {key}")
    if missing or mismatched:
        print(
            "Nothing was written: set the encoder sizes of the config to those of the file "
            "(for example `token_dim: 256` for Point-BERT and MaskPoint files), or name what "
            "may stay new with --fresh."
        )
        return 1
    torch.save({"model": state}, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
