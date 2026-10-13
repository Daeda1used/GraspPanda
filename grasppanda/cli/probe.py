"""``grasppanda probe``: check in about a minute that a config really trains and predicts.

    grasppanda probe experimental/gsnet_ptv3_g1b-realsense_10e
    grasppanda probe my_config.yaml --iters 20 --frames 8 --json

The probe builds the detector and the data pipeline of the config, trains a few
iterations on real frames (the split ``sample`` of the dataset) with the config's own
recipe, predicts on a few frames and reports what it saw: parameters, time and memory,
every loss term first and last, the fraction of real seeds, the grasps per frame, any
non-finite value. It ends with one verdict:

    runs            trained and predicted on real frames, all finite, seeds every iteration
    runs-unstable   trained and predicted, but with a non-finite value or iterations without seeds
    fails           an exception after the config check passed; phase and component are named
    refused         the config check refused the config; its messages are printed
    wired           no real frame was read: a wiring check on a synthetic scene passed

Without the dataset, or without the label assets of its frames, the probe says what is
missing and how to build it, runs the synthetic wiring check and reports ``wired`` (exit
status 3), never ``runs``. Exit status: 0 runs (or ``wired`` with ``--synthetic``),
1 runs-unstable or fails, 2 refused or a request that cannot be run, 3 wired as a fallback.

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
                    real seeds, predicted one grasp group per frame (a group may be empty)
  runs-unstable  1  built, trained and predicted, but a loss or weight became non-finite,
                    or an iteration had no real seed, or no seed was ever supervised
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

a probe is one run. The config hash, the parameter count and the first losses repeat;
later losses, seed fractions and grasp counts differ from run to run, and a combination
near the edge can be `runs` in one probe and `runs-unstable` in the next.

examples:
  grasppanda probe gsnet/gsnet_sunet14_g1b-realsense_10e
  grasppanda probe my_config.yaml --iters 20 --frames 8 --json
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
            "one verdict: runs, runs-unstable, fails, refused or wired."
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
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.set_defaults(handler=run)


def _pair(values: list[float]) -> str:
    return f"{values[0]:.4g} → {values[1]:.4g}"


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
        print(f"  {style('losses', 'gray', enabled=color)}      first → last")
        for name, values in report.losses.items():
            print(f"    {name:<28}{_pair(values)}")
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
    chain = report.postprocessing or {}
    for sizes in chain.get("grasps", []):
        steps = " → ".join(f"{name} {left}" for name, left in zip(chain["steps"], sizes[1:]))
        row("post", f"{sizes[0]} → {steps}")
    if report.no_gradient:
        row("gradient", "none reached " + ", ".join(report.no_gradient))
    for item in report.nonfinite:
        row("non-finite", item)
    for message in report.refusal:
        print(f"  {style('✗', 'red', enabled=color)} {message}")
    print()
    good = report.verdict in ("runs", "wired")
    mark = style("✓" if good else "✗", "bold", "green" if good else "red", enabled=color)
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
        )
    except ProbeRequestError as error:
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        print(f"{mark} {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print(report, sys.stdout.isatty())
    return report.exit_status(synthetic_requested=args.synthetic)
