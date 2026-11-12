"""Which per-view labels do HGGD weights fit?

HGGD trains on labels that its converter thins by a suppression: a grasp removes every
later grasp within a distance *and* a rotation angle. How dense the labels are decides how
many angle classes of the image network are positive (a cell needs two labels of one class),
and the released converter and the released checkpoint do not agree on it. This tool
measures it: for each suppression setting it reads training frames with those labels and
reports, per frame, the labels and the supervised and positive angle classes, and for each
set of weights the training loss on them.

    grasppanda prepare graspnet1b --view-grasps --camera realsense --split val_dev \\
        --config hggd/hggd_r34w8_g1b-realsense_15e      # the shipped labels; for another
                                                        # density, a config that sets
                                                        # data.train.dataset.view_grasps
    python -m tools.hggd_label_fit --density 0.02,60 --density 0.01,30 \\
        --weights authors=hggd_realsense.pth --weights trained=epoch_0015.pth \\
        --json label_fit.json

The losses are those of ``HGGD.loss`` with the weights frozen: no gradient, batch-norm on
its running statistics, the rotation anchors of the weights left where they are. The
frames are every ``--frame-step``-th frame of the scenes of ``--split`` (default: the ten
``val_dev`` scenes, training scenes), in order, in batches of four.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

CONFIG = "hggd/hggd_r34w8_g1b-realsense_15e"
LABELS = "view_grasps"
#: What the record says about itself.
ABOUT = (
    "Training loss of HGGD weights on training frames whose per-view labels were suppressed "
    "at the given distance (metres) and angle (degrees), by tools/hggd_label_fit.py: no "
    "gradient, batch-norm on its running statistics, anchors fixed. Counts are per frame: "
    "labels, (cell, angle class) entries with a label, and those that are positives of the "
    "angle loss."
)


def freeze_norms(model: nn.Module) -> None:
    """Training mode for the stages (their proposal budget), running statistics for the
    batch-norm layers: the weights as they are used, on training frames."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def measure(model: nn.Module, batches: Iterable[Any]) -> dict[str, float]:
    """Mean loss terms over ``batches`` and the label counts per frame.

    ``batches`` yields objects with ``observation`` and ``supervision``; the model is an
    ``HGGD`` whose anchors do not move (``anchors.shift_epochs`` 0).
    """
    freeze_norms(model)
    sums: dict[str, float] = defaultdict(float)
    batch_count = frames = 0
    size = tuple(model.image_options.size)
    threshold = model.proposal.positive_count_threshold
    with torch.no_grad():
        for batch in batches:
            observation, supervision = batch.observation, batch.supervision
            for name, value in model.loss(observation, supervision).items():
                if name.startswith("loss/"):
                    sums[name] += float(value)
            labels = supervision.extras[LABELS]
            target = model.proposal.encode(labels, observation.depth, size)
            sums["labels"] += sum(len(next(iter(rows.values()))) for rows in labels)
            sums["supervised"] += float((target.angle > 0).sum())
            sums["positive"] += float((target.angle > threshold).sum())
            batch_count += 1
            frames += len(observation.keys)
    if not frames:
        raise ValueError("no frame was read")
    result = {
        name: value / (batch_count if name.startswith("loss/") else frames)
        for name, value in sums.items()
    }
    result["loss"] = sum(value for name, value in result.items() if name.startswith("loss/"))
    result["frames"] = frames
    return result


def overrides(distance: float, angle: float, split: str, frame_step: int) -> list[str]:
    """The shipped config with other labels, read in order and with fixed anchors."""
    return [
        f"data.train.dataset.view_grasps.nms_distance={distance}",
        f"data.train.dataset.view_grasps.nms_angle={angle}",
        f"data.train.dataset.split={split}",
        f"data.train.dataset.frame_step={frame_step}",
        "data.train.batch_size=4",
        "data.train.shuffle=false",
        "model.anchors.shift_epochs=0",
    ]


def run(
    densities: list[tuple[float, float]],
    weights: Mapping[str, str],
    *,
    split: str,
    frame_step: int,
    device: str,
) -> dict[str, Any]:
    from grasppanda.config import Config, resolve_config_path
    from grasppanda.engine.builders import build_dataset, build_model, build_train_loader
    from grasppanda.engine.checkpoint import load_weights

    rows = []
    for distance, angle in densities:
        config = Config.load(
            resolve_config_path(CONFIG), overrides(distance, angle, split, frame_step)
        )
        cfg = config.resolved()
        loader = build_train_loader(cfg, build_dataset(cfg, "train"), rank=0, world_size=1)
        row: dict[str, Any] = {"nms_distance": distance, "nms_angle": angle, "weights": {}}
        for name, path in weights.items():
            torch.manual_seed(0)
            model = build_model(cfg).to(device)
            load_weights(model, path)
            torch.manual_seed(0)  # the draws of the region stage: the same for every weights
            result = measure(model, (batch.to(device) for batch in loader))
            for key in ("frames", "labels", "supervised", "positive"):
                row[key] = round(result[key], 2) if key != "frames" else result[key]
            row["weights"][name] = {
                key: round(value, 4) for key, value in result.items() if key.startswith("loss")
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {
        "about": ABOUT,
        "config": CONFIG,
        "split": split,
        "frame_step": frame_step,
        "rows": rows,
    }


def pair(text: str) -> tuple[float, float]:
    try:
        distance, angle = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{text}' is not DISTANCE,ANGLE (metres, degrees), e.g. 0.02,60"
        ) from None
    return distance, angle


def named(text: str) -> tuple[str, str]:
    name, separator, path = text.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"'{text}' is not NAME=PATH, e.g. authors=hggd.pth")
    return name, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--density",
        type=pair,
        action="append",
        required=True,
        metavar="DISTANCE,ANGLE",
        help="suppression of the labels, metres and degrees (repeat; the shipped one is 0.02,60)",
    )
    parser.add_argument(
        "--weights",
        type=named,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="weights in GraspPanda format under a name of your choice (repeat)",
    )
    parser.add_argument("--split", default="val_dev", help="scenes to read (default: val_dev)")
    parser.add_argument("--frame-step", type=int, default=16, help="every n-th frame (default: 16)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", type=Path, help="write the record here")
    args = parser.parse_args(argv)
    record = run(
        args.density,
        dict(args.weights),
        split=args.split,
        frame_step=args.frame_step,
        device=args.device,
    )
    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
