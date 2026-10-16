"""``grasppanda probe``: check in about a minute that a config really trains and predicts.

    grasppanda probe experimental/gsnet_ptv3_g1b-realsense_10e
    grasppanda probe my_config.yaml --iters 20 --frames 8 --json

The probe builds the detector and the data pipeline of the config, trains a few
iterations on real frames (the split ``sample`` of the dataset) with the config's own
recipe, predicts on a few frames and reports what it saw: parameters, time and memory,
every loss term first and last with its fall and flags, the fraction of real seeds, the
units of the model without a gradient and the parts that did not change, the grasps per
frame and their scores, the decay of the learning rate, any non-finite value. It ends
with one verdict:

    runs            trained and predicted on real frames, all finite, seeds every iteration
    runs-empty      as runs, but no probed frame has a grasp
    runs-unstable   trained and predicted, but with a non-finite value or iterations without seeds
    fails           an exception after the config check passed; phase and component are named
    refused         the config check refused the config; its messages are printed
    wired           no real frame was read: a wiring check on a synthetic scene passed

Without the dataset, or without the label assets of its frames, the probe says what is
missing and how to build it, runs the synthetic wiring check and reports ``wired`` (exit
status 3), never ``runs``. Exit status: 0 runs or runs-empty (or ``wired`` with
``--synthetic``), 1 runs-unstable or fails, 2 refused or a request that cannot be run,
3 wired as a fallback.

A probe is one run, and training on a GPU is not repeatable: see
:mod:`grasppanda.combine.probe`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from grasppanda.utils.console import style

__all__ = ["register"]


EPILOG = """\
verdicts (exactly one is reported) and exit status:
  runs           0  built, trained every iteration on real frames with finite losses and
                    real seeds, every part got a gradient, predicted one grasp group per
                    frame and at least one grasp
  runs-empty     0  as runs, but every predicted group is empty: no grasp on any frame
  runs-unstable  1  built, trained and predicted, but a loss or weight became non-finite,
                    or an iteration had no real seed, or no seed was ever supervised, or a
                    part received no gradient
  fails          1  an exception after the config check passed; the phase (build, data,
                    loss, backward, step, predict), the component and the message are given
  refused        2  the config check refused the config (as `print-config --check --for
                    train`); nothing was built
  wired          3  no real frame was read (0 with --synthetic): the stages connect on a
                    synthetic scene with random labels; this says nothing about real data
A request that cannot be run (a count below 1, an unknown --split) also exits with 2.

what it needs: the dataset of the config and the label assets of the probed frames, which
for the split `sample` are small (`grasppanda prepare <dataset> --check --split sample
--config <config>` lists them). Without them the probe says what is missing and how to
build it, runs the synthetic check and reports `wired`, never `runs`.

reported, not judged (`--json` has every iteration): per loss term the fall from its
first active value to the mean of its last quarter and the flags flat, rising,
zero_at_end, constant; the units of the model (sub-modules at depth 2) that never got a
gradient and the parts whose weights did not change; the score of the predicted grasps;
the learning rate at the last step relative to the first.

the learning rate (--schedule): `recipe`, the default, runs the config's scheduler over
the few epochs of the probe (train.epochs = ceil(iterations / iterations per epoch)), so a
per-epoch decay is compressed into the probe and a long warm-up is never over. `constant`
removes the scheduler (scheduler=null) and trains every step at the recipe's peak rate: the
optimizer's rate, or `max_lr` of OneCycleLR and CyclicLR, scaled by the ratio of the batch
sizes when the probe's batches are smaller than the recipe's (the linear scaling rule).
A batch-norm momentum the recipe decays per epoch (BNMomentumHook) is held at its first
epoch's value (hooks.<i>.decay=1). Optimizer, weight decay, clipping and the other hooks
stay the recipe's. Use it to compare the trends of methods whose schedules differ; the
config hash is the same either way.

a probe is one run. The config hash, the parameter count and the first losses repeat;
later losses, seed fractions and grasp counts differ from run to run, and a combination
near the edge can be `runs` in one probe and `runs-unstable` (or `runs-empty`) in the next.

examples:
  grasppanda probe gsnet/gsnet_sunet14_g1b-realsense_10e
  grasppanda probe my_config.yaml --iters 20 --frames 8 --json
  grasppanda probe finegrasp/finegrasp_sunet14_g1b-realsense_10e --schedule constant
"""


def _at_least(minimum: int) -> Any:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got '{text}'") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    return parse


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "probe",
        help="train a few iterations on real frames and predict: does this config run?",
        description=(
            "Build the detector, train a few iterations on real frames, predict, and report "
            "one verdict: runs, runs-empty, runs-unstable, fails, refused or wired."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="config file or name of a shipped config")
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, as for `grasppanda train`",
    )
    parser.add_argument(
        "--iters", type=_at_least(1), default=20, help="training iterations (default 20)"
    )
    parser.add_argument(
        "--frames",
        type=_at_least(1),
        default=8,
        help="train on at most this many frames (default 8)",
    )
    parser.add_argument(
        "--predict-frames", type=_at_least(1), default=2, help="frames to predict on (default 2)"
    )
    parser.add_argument(
        "--split", default="sample", help="split the frames are taken from (default: sample)"
    )
    parser.add_argument("--device", help="overrides runtime.device, e.g. cpu or cuda:1")
    parser.add_argument(
        "--workers", type=_at_least(0), default=0, help="data loader processes (default 0: none)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="skip real frames: only the wiring check on a synthetic scene (verdict `wired`)",
    )
    parser.add_argument(
        "--schedule",
        choices=("recipe", "constant"),
        default="recipe",
        help="recipe: the config's scheduler over the probe's epochs (default); constant: "
        "the recipe's peak learning rate at every step, scaled down for batches smaller than "
        "the recipe's",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.set_defaults(handler=run)


def _pair(values: list[float]) -> str:
    return f"{values[0]:.4g} → {values[1]:.4g}"


def _trend(trend: dict[str, Any] | None, iterations: int) -> str:
    """The fall of a loss term and its flags, as one short phrase."""
    if not trend:
        return ""
    fall = trend.get("fall")
    if fall is None:
        text = "never active"
    else:
        text = f"{'fell' if fall >= 0 else 'rose'} {abs(fall) * 100:.0f} %"
        if trend["active"] < iterations:
            text += f" ({trend['active']} of {iterations} active)"
    flags = [flag.replace("_", " ") for flag in trend.get("flags", [])]
    return " · ".join([text, *flags])


def _print(report: Any, color: bool) -> None:
    def row(name: str, value: str) -> None:
        print(f"  {style(name.ljust(10), 'gray', enabled=color)}  {value}")

    row(
        "config",
        f"{report.config}" + (f"  (hash {report.config_hash[:12]})" if report.config_hash else ""),
    )
    if report.overrides:
        row("overrides", " ".join(report.overrides))
    if report.detector:
        size = f" · {report.parameters:,} parameters" if report.parameters else ""
        row("model", f"{report.detector}{size}")
    if report.fallback:
        row("dataset", f"NOT READ: {report.fallback}")
    if report.data == "real":
        if report.frames:
            frames, batch = len(report.frames), report.batch_size
            row("data", f"{frames} frames of the config's dataset · {batch} per iteration")
        row("probe", "as `grasppanda train` with " + " ".join(report.probe_overrides))
    elif report.data == "synthetic":
        row("data", "SYNTHETIC scene with random labels: a wiring check, not a result")
    if report.device:
        row("device", report.device)
    for note in report.notes:
        row("note", note)
    if report.iterations:
        timing = ""
        if report.seconds_per_iteration is not None:
            timing = (
                f" · {report.seconds_per_iteration:.2f} s per iteration (median, without "
                f"data; data {report.seconds_data:.2f} s), measured on this machine now"
            )
        row("trained", f"{report.iterations} of {report.iterations_requested} iterations{timing}")
        if report.peak_gpu_mib is not None:
            row("memory", f"peak {report.peak_gpu_mib / 1024:.1f} GiB on the GPU, this process")
        trends = getattr(report, "loss_trends", {}) or {}
        print(
            f"  {style('losses', 'gray', enabled=color)}      first → last"
            + (
                " · fall from the first active value to the mean of the last quarter"
                if trends
                else ""
            )
        )
        for name, values in report.losses.items():
            trend = _trend(trends.get(name), report.iterations)
            print(f"    {name:<28}{_pair(values):<22}{trend}".rstrip())
        if report.real_seeds:
            seeds = report.real_seeds
            row(
                "seeds",
                f"real: min {min(seeds):.2f}, mean {sum(seeds) / len(seeds):.2f} of the seed "
                "slots per iteration",
            )
        if report.supervised_seeds:
            seeds = report.supervised_seeds
            row(
                "" if report.real_seeds else "seeds",
                f"supervised: {seeds[0]:.2f} → {seeds[-1]:.2f}, min {min(seeds):.2f}",
            )
    if report.grasps_per_frame:
        counts = ", ".join(str(count) for count in report.grasps_per_frame)
        row("predicted", f"{len(report.grasps_per_frame)} frame(s) · {counts} grasps")
        quantiles = getattr(report, "score_quantiles", []) or []
        if any(quantiles):
            listed = "; ".join(
                "—" if q is None else " / ".join(f"{value:.3g}" for value in q) for q in quantiles
            )
            row("scores", f"min / median / max per frame: {listed}")
    chain = report.postprocessing or {}
    for sizes in chain.get("grasps", []):
        steps = " → ".join(f"{name} {left}" for name, left in zip(chain["steps"], sizes[1:]))
        row("post", f"{sizes[0]} → {steps}")
    units = getattr(report, "units_checked", 0)
    if report.no_gradient:
        of = f" (of {units} units)" if units else ""
        row("gradient", "none reached " + ", ".join(report.no_gradient) + of)
    elif units:
        parts = report.parts_checked
        row("gradient", f"reached every one of {units} units of {parts} part(s)")
    if units:
        unchanged = report.unchanged
        row("changed", "not " + ", ".join(unchanged) if unchanged else "every part")
    rates = getattr(report, "learning_rate", []) or []
    constant = getattr(report, "schedule", "recipe") == "constant"
    if constant and report.rate is not None:
        scaled = any("linear scaling rule" in note for note in report.notes)
        what = "the recipe's peak rate scaled to the batch" if scaled else "the recipe's peak rate"
        row("schedule", f"constant: {report.rate:.3g}, {what}, at every step")
    elif rates:
        factor = f", ×{report.lr_factor:.3g} at the last step" if report.lr_factor else ""
        row("lr", f"{rates[0]:.3g} at the first step → {rates[1]:.3g} at the last{factor}")
    for item in report.nonfinite:
        row("non-finite", item)
    for message in report.refusal:
        print(f"  {style('✗', 'red', enabled=color)} {message}")
    print()
    good = report.verdict in ("runs", "wired")
    mark = style("✓" if good else "✗", "bold", "green" if good else "red", enabled=color)
    if report.verdict == "runs-empty":
        mark = style("∅", "bold", "yellow", enabled=color)
    print(f"{mark} {style(report.verdict, 'bold', enabled=color)}: {report.detail}")
    if report.verdict == "wired" and report.fallback:
        print("  Real frames were not probed; `wired` says only that the stages connect.")
    print(f"  ({report.seconds:.0f} s)")


def run(args: argparse.Namespace) -> int:
    from grasppanda.combine.probe import ProbeRequestError, probe

    try:
        report = probe(
            args.config,
            args.opts,
            iters=args.iters,
            frames=args.frames,
            predict_frames=args.predict_frames,
            split=args.split,
            device=args.device,
            workers=args.workers,
            synthetic=args.synthetic,
            schedule=args.schedule,
        )
    except ProbeRequestError as error:
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        print(f"{mark} {error}", file=sys.stderr)
        return 2
    from grasppanda.ops.sparse import cpu_thread_note

    if note := cpu_thread_note():
        report.notes.append(note)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print(report, sys.stdout.isatty())
    return report.exit_status(synthetic_requested=args.synthetic)
