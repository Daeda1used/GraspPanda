"""How closely do weights fit the training frames, and do they need their training mode?

The training loss of each set of weights on training frames, twice:

* ``training``: every layer as in training, dropout drawing and batch-norm normalising by
  the batch;
* ``evaluation``: no dropout, batch-norm on its running statistics, the layers as
  ``predict`` uses them.

Two sets of weights of one config compare by either row: the one with the lower loss fits
the training scenes closer. And one set compares with itself: a loss that is higher in
``evaluation`` than in ``training`` says the weights lean on their training mode and
predict worse than they trained.

    python -m tools.train_fit finegrasp/finegrasp_sunet14_g1b-realsense_10e \\
        --weights authors=finegrasp_realsense.pth --weights trained=epoch_0010.pth \\
        --json train_fit.json

``--between`` adds the two rows between the modes, one kind of layer switched alone
(``dropout off``, ``running statistics``), and ``--off NAME=REGEX`` a row with the dropout
layers of that name off. No run is ever in these states: they locate a difference that
``evaluation`` shows, and mean nothing without one.

The losses are those of the model's ``loss`` without a gradient. The stages stay in
training mode in every row (a detector samples and assigns labels there), so the rows
differ in the named layers only. The frames are every ``--frame-step``-th frame of the
scenes of ``--split`` (default: the ten ``val_dev`` scenes, training scenes), in order, in
batches of four, with the augmentation of the config and the same random draws in each row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

#: What the record says about itself.
ABOUT = (
    "Training loss of weights on training frames by tools/train_fit.py, no gradient: "
    "`training` with every layer as in training, `evaluation` with no dropout and "
    "batch-norm on its running statistics, as the model predicts. The lower loss fits the "
    "training frames closer; a loss that is higher in `evaluation` than in `training` "
    "says the weights lean on their training mode. Any other row switches one kind of "
    "layer alone, a state no run is in."
)
NORMS = nn.modules.batchnorm._BatchNorm
DROPOUTS = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def set_mode(
    model: nn.Module, *, dropout_off: str | None = None, running_norms: bool = False
) -> list[str]:
    """Training mode, with the dropout layers whose name matches ``dropout_off`` and, with
    ``running_norms``, every batch-norm layer in their evaluation behaviour.

    Returns the names of the dropout layers that were switched off.
    """
    model.train()
    switched = []
    for name, module in model.named_modules():
        if running_norms and isinstance(module, NORMS):
            module.eval()
        named = dropout_off is not None and re.search(dropout_off, name)
        if named and isinstance(module, DROPOUTS):
            module.eval()
            switched.append(name)
    return switched


def measure(model: nn.Module, batches: Iterable[Any], seed: int = 0) -> dict[str, float]:
    """Mean of every loss term over ``batches`` in the mode the model is in, their sum as
    ``loss``, and the number of ``frames``.

    ``batches`` yields objects with ``observation`` and ``supervision``. The modes of the
    layers are left as they are (:func:`set_mode`); nothing is written to the model but
    the running statistics of a batch-norm layer in training mode, which are put back.
    """
    state = {name: value.clone() for name, value in model.state_dict().items()}
    sums: dict[str, float] = defaultdict(float)
    count = frames = 0
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch in batches:
            for name, value in model.loss(batch.observation, batch.supervision).items():
                if name.startswith("loss/"):
                    sums[name] += float(value)
            count += 1
            frames += len(batch.observation.keys)
    model.load_state_dict(state)
    if not frames:
        raise ValueError("no frame was read")
    result = {name: value / count for name, value in sums.items()}
    result["loss"] = sum(result.values())
    result["frames"] = frames
    return result


def rows(
    model: nn.Module,
    batches: Callable[[], Iterable[Any]],
    *,
    between: bool = False,
    off: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Every row of one set of weights. ``batches()`` gives the same frames each time.

    ``between`` adds the rows with one kind of layer switched alone, ``off`` one row per
    name with the dropout layers whose module name matches its pattern switched off.

    Raises:
        ValueError: A pattern of ``off`` matches no dropout layer of the model.
    """
    modes: dict[str, dict[str, Any]] = {
        "training": {},
        "evaluation": {"dropout_off": "", "running_norms": True},
    }
    if between:
        modes["dropout off"] = {"dropout_off": ""}
        modes["running statistics"] = {"running_norms": True}
    for name, pattern in (off or {}).items():
        modes[f"{name} off"] = {"dropout_off": pattern}
    result = {}
    for label, mode in modes.items():
        switched = set_mode(model, **mode)
        if mode.get("dropout_off") and not switched:
            names = [n for n, m in model.named_modules() if isinstance(m, DROPOUTS)]
            raise ValueError(
                f"'{mode['dropout_off']}' matches no dropout layer; the model has "
                f"{', '.join(names) or 'none'}"
            )
        result[label] = measure(model, batches())
    model.train()
    return result


def overrides(split: str, frame_step: int) -> list[str]:
    """The frames of the measurement: which scenes, how many, in which order."""
    return [
        f"data.train.dataset.split={split}",
        f"data.train.dataset.frame_step={frame_step}",
        "data.train.batch_size=4",
        "data.train.shuffle=false",
    ]


def run(
    config: str,
    weights: Mapping[str, str],
    *,
    opts: list[str],
    between: bool,
    off: Mapping[str, str],
    split: str,
    frame_step: int,
    device: str,
) -> dict[str, Any]:
    from grasppanda.config import Config, resolve_config_path
    from grasppanda.engine.builders import build_dataset, build_model, build_train_loader
    from grasppanda.engine.checkpoint import load_weights

    cfg = Config.load(
        resolve_config_path(config), [*opts, *overrides(split, frame_step)]
    ).resolved()

    def batches() -> Iterable[Any]:
        torch.manual_seed(0)  # the augmentation: the same in every row
        loader = build_train_loader(cfg, build_dataset(cfg, "train"), rank=0, world_size=1)
        return (batch.to(device) for batch in loader)

    record: dict[str, Any] = {
        "about": ABOUT,
        "config": config,
        "opts": opts,
        "split": split,
        "frame_step": frame_step,
        "weights": {},
    }
    for name, path in weights.items():
        model = build_model(cfg).to(device)
        load_weights(model, path)
        measured = rows(model, batches, between=between, off=off)
        record["frames"] = next(iter(measured.values()))["frames"]
        record["weights"][name] = {
            label: {key: round(value, 4) for key, value in row.items() if key != "frames"}
            for label, row in measured.items()
        }
        for label, row in record["weights"][name].items():
            print(f"{name:<16} {label:<28} loss {row['loss']:.3f}", flush=True)
    return record


def named(text: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not name or not value:
        raise argparse.ArgumentTypeError(f"'{text}' is not NAME=VALUE, e.g. authors=file.pth")
    return name, value


def pattern(text: str) -> tuple[str, str]:
    name, value = named(text)
    try:
        re.compile(value)
    except re.error as error:
        raise argparse.ArgumentTypeError(f"'{value}' is not a pattern: {error}") from None
    return name, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="the config the weights were trained with")
    parser.add_argument(
        "--weights",
        type=named,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="weights in GraspPanda format under a name of your choice (repeat)",
    )
    parser.add_argument(
        "--between",
        action="store_true",
        help="also the rows with dropout alone off and with batch-norm alone on its statistics",
    )
    parser.add_argument(
        "--off",
        type=pattern,
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="a row with the dropout layers whose module name matches switched off (repeat)",
    )
    parser.add_argument("--opts", nargs="*", default=[], help="overrides of the config")
    parser.add_argument("--split", default="val_dev", help="scenes to read (default: val_dev)")
    parser.add_argument("--frame-step", type=int, default=16, help="every n-th frame (default: 16)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", type=Path, help="write the record here")
    args = parser.parse_args(argv)
    try:
        record = run(
            args.config,
            dict(args.weights),
            opts=args.opts,
            between=args.between,
            off=dict(args.off),
            split=args.split,
            frame_step=args.frame_step,
            device=args.device,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
