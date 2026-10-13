"""Time the accelerated operators on the reference and on the compiled kernels.

Every case is run on both backends back to back, ``--repeats`` times, with the device
synchronised around each call. Reported per case: minimum and median time of each
backend, and the median of the per-pair ratios, which is the number to trust on a
machine that is busy with other work. The results of the two backends are compared in
every case; a difference is printed and makes the script exit with an error.

    python tools/bench_ops.py                       # the operators, on CUDA if there is one
    python tools/bench_ops.py --device cpu --repeats 3
    python tools/bench_ops.py --models              # also PointNet++ and GSNet end to end
    python tools/bench_ops.py --markdown            # the table of docs/advanced_guides/operators.md

Inputs are synthetic table-top clouds in metres (a plane and a few objects, about the
density of a GraspNet-1Billion frame), sized like the models use them: 1024 samples
from 3 000 to 15 000 candidates; 1024 queries on 15 000 to 20 000 points with 16 to 64
neighbours; batches of 1 and 4. ``--models`` times the GraspNet-baseline PointNet++
backbone on 20 000 points (random weights) and, when the recorded GSNet fixtures are
available (``tools/make_gsnet_fixtures.py``), ``GSNet.predict`` on real frames with the
authors' weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from grasppanda.ops import (
    approach_to_rotation,
    backends,
    ball_query,
    cylinder_query,
    furthest_point_sample,
    knn,
)
from grasppanda.utils.console import table
from grasppanda.utils.paths import cache_root

REPO = Path(__file__).resolve().parents[1]
REFERENCE = dict.fromkeys(backends.OPERATORS, "reference")
COMPILED = dict.fromkeys(backends.OPERATORS, "auto")


@dataclass
class Row:
    """One benchmark case; times in milliseconds."""

    case: str
    batch: int
    reference_min: float
    reference_median: float
    compiled_min: float | None = None
    compiled_median: float | None = None
    ratio: float | None = None
    identical: bool | None = None


def tabletop(batch: int, count: int, seed: int) -> Tensor:
    """``(batch, count, 3)`` points of a table top with four objects, half a metre away."""
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(batch, count, 3, generator=generator) * torch.tensor([0.6, 0.6, 0.002])
    points[..., 2] += 0.5
    centre = torch.rand(batch, 4, 3, generator=generator) * torch.tensor([0.6, 0.6, 0.0])
    blob = torch.randint(0, 4, (batch, count), generator=generator)
    lifted = torch.gather(centre, 1, blob[..., None].expand(-1, -1, 3))
    lifted = lifted + 0.04 * torch.randn(batch, count, 3, generator=generator)
    lifted[..., 2] = 0.45 + lifted[..., 2].abs()
    on_object = torch.rand(batch, count, generator=generator) < 0.5
    return torch.where(on_object[..., None], lifted, points)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(call: Callable[[], Any], device: torch.device) -> tuple[float, Any]:
    _synchronise(device)
    start = time.perf_counter()
    result = call()
    _synchronise(device)
    return (time.perf_counter() - start) * 1e3, result


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor):
        return bool(first.shape == second.shape and torch.equal(first, second))
    if isinstance(first, np.ndarray):
        return bool(np.array_equal(first, second))
    return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))


def measure(
    case: str,
    batch: int,
    call: Callable[[], Any],
    device: torch.device,
    repeats: int,
    compiled: bool,
    exact: bool = True,
) -> Row:
    """Time ``call`` on both backends in turn; ``exact`` also compares their results."""
    with backends.override(**REFERENCE):
        _timed(call, device)  # warm-up
    reference, fast, identical = [], [], True
    if compiled:
        with backends.override(**COMPILED):
            _timed(call, device)
    for _ in range(repeats):
        with backends.override(**REFERENCE):
            elapsed, expected = _timed(call, device)
        reference.append(elapsed)
        if compiled:
            with backends.override(**COMPILED):
                elapsed, got = _timed(call, device)
            fast.append(elapsed)
            identical = identical and (not exact or _same(got, expected))
    row = Row(case, batch, min(reference), statistics.median(reference))
    if compiled:
        row.compiled_min, row.compiled_median = min(fast), statistics.median(fast)
        row.ratio = statistics.median(r / f for r, f in zip(reference, fast))
        row.identical = identical if exact else None
    return row


# ---------------------------------------------------------------------------- cases


def operator_cases(device: torch.device, batches: list[int]) -> Iterator[tuple[str, int, Callable]]:
    for batch in batches:
        for count in (3000, 8000, 15000):
            points = tabletop(batch, count, seed=count).to(device)
            for label, options in (
                ("first", {}),
                ("PointNet++ kernel", {"skip_near_origin": True, "tie_break": "pointnet2_cuda"}),
            ):
                yield (
                    f"fps: 1024 of {count}, ties: {label}",
                    batch,
                    lambda p=points, o=options: furthest_point_sample(p, 1024, **o),
                )
        seeds = tabletop(batch, 1024, seed=1).to(device)
        frames = approach_to_rotation(
            torch.randn(batch, 1024, 3, generator=torch.Generator().manual_seed(2))
        ).to(device)
        yield (
            "cylinder_query: 1024 x 1024 seeds, 16 neighbours",
            batch,
            lambda s=seeds, r=frames: cylinder_query(s, s, r, 0.05, -0.02, 0.04, 16),
        )
        for count in (15000, 20000):
            points = tabletop(batch, count, seed=count + 1).to(device)
            centres = points[:, :1024].contiguous()
            for nsample in (16, 64):
                yield (
                    f"ball_query: 1024 x {count}, {nsample} neighbours",
                    batch,
                    lambda c=centres, p=points, k=nsample: ball_query(c, p, 0.05, k),
                )
                yield (
                    f"cylinder_query: 1024 x {count}, {nsample} neighbours",
                    batch,
                    lambda c=centres, p=points, r=frames, k=nsample: cylinder_query(
                        c, p, r, 0.05, -0.02, 0.04, k
                    ),
                )
            yield (
                f"knn: {count} x 1024, k=3",
                batch,
                lambda c=centres, p=points: knn(p, c, 3),
            )
            yield (
                f"knn: 1024 x {count}, k=1",
                batch,
                lambda c=centres, p=points: knn(c, p, 1, chunk=256),
            )


def pointnet2_cases(
    device: torch.device, batches: list[int]
) -> Iterator[tuple[str, int, Callable]]:
    from grasppanda.registry import BACKBONES
    from grasppanda.structures import PointSet

    torch.manual_seed(0)
    model = BACKBONES.build({"type": "PointNet2"}).to(device).eval()
    for batch in batches:
        points = PointSet.from_dense(tabletop(batch, 20000, seed=3)).to(device)

        def forward(points=points) -> Tensor:
            with torch.no_grad():
                return model(points).feats

        yield "PointNet2 backbone forward, 20000 points", batch, forward


def gsnet_cases(
    device: torch.device, batches: list[int], fixtures: Path
) -> Iterator[tuple[str, int, Callable]]:
    """``GSNet.predict`` on recorded test frames with the authors' weights, per call."""
    from grasppanda.config import Config
    from grasppanda.registry import DETECTORS
    from grasppanda.structures import Observation, PointSet, SampleKey

    manifest = json.loads((fixtures / "manifest.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    state, _ = converter.convert_file(
        Path(manifest["checkpoint"]), "_base_/models/gsnet.yaml", camera=manifest["camera"]
    )
    model = DETECTORS.build(Config.load("_base_/models/gsnet.yaml")["model"])
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    clouds = [
        torch.from_numpy(np.load(fixtures / entry["file"])["point_clouds"][0])
        for entry in manifest["inference"]
    ]
    for batch in batches:
        if batch > len(clouds):
            continue
        observation = Observation(
            keys=[SampleKey("graspnet1b", 100 + i, manifest["camera"], i) for i in range(batch)],
            points=PointSet.from_dense(torch.stack(clouds[:batch])),
        ).to(device)

        def predict(observation=observation) -> list[Tensor]:
            return [group.translation for group in model.predict(observation)]

        yield f"GSNet.predict, {len(clouds[0])} points per frame", batch, predict


# --------------------------------------------------------------------------- report


def context(device: torch.device) -> dict[str, Any]:
    """What the numbers were measured on, including how busy the machine was."""
    facts: dict[str, Any] = {
        "torch": torch.__version__,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "load_average": [round(value, 1) for value in os.getloadavg()],
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        facts["gpu"] = torch.cuda.get_device_name(device)
        try:
            query = "--query-gpu=utilization.gpu,memory.used"
            listing = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            facts["gpu_utilisation_and_memory_before"] = listing.stdout.strip().splitlines()
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            others = [pid for pid in apps.stdout.split() if pid != str(os.getpid())]
            facts["other_gpu_processes"] = len(others)
        except (OSError, subprocess.SubprocessError):
            pass
    return facts


def _cell(low: float | None, median: float | None) -> str:
    return "-" if low is None else f"{low:.2f} / {median:.2f}"


def render(rows: list[Row], markdown: bool) -> str:
    headers = ["case", "batch", "reference (ms)", "compiled (ms)", "ratio", "identical"]
    body = [
        [
            row.case,
            str(row.batch),
            _cell(row.reference_min, row.reference_median),
            _cell(row.compiled_min, row.compiled_median),
            "-" if row.ratio is None else f"{row.ratio:.1f}x",
            {None: "-", True: "yes", False: "NO"}[row.identical],
        ]
        for row in rows
    ]
    if not markdown:
        return table(headers, body, enabled=False)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * 4 + ["---"]) + "|",
    ]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=7, help="back-to-back pairs per case")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--models", action="store_true", help="also time PointNet++ and GSNet")
    parser.add_argument("--no-operators", action="store_true", help="skip the operator cases")
    parser.add_argument(
        "--gsnet-fixtures",
        type=Path,
        default=cache_root() / "fixtures" / "gsnet",
        help="recorded GSNet frames and checkpoint (tools/make_gsnet_fixtures.py)",
    )
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--json", type=Path, help="also write the numbers and the context here")
    args = parser.parse_args()

    device = torch.device(args.device)
    facts = context(device)
    report = backends.report()
    kind = device.type
    compiled = all(report[name].get(kind) == "compiled" for name in backends.OPERATORS)
    facts["backends"] = {name: report[name].get(kind) for name in backends.OPERATORS}
    facts["library"] = report["library"]
    print(json.dumps(facts, indent=2))
    if not compiled:
        print(
            f"The compiled kernels are not in use on {kind} ({report['library']['detail']}): "
            "timing the reference only.",
            file=sys.stderr,
        )

    cases: list[tuple[str, int, Callable, bool]] = []
    if not args.no_operators:
        cases += [(*case, True) for case in operator_cases(device, args.batch)]
    if args.models:
        # Float32 activations of a network differ in the last bits between two runs on a
        # GPU, so model outputs are not compared here; the parity tests do that.
        cases += [(*case, False) for case in pointnet2_cases(device, args.batch)]
        if (args.gsnet_fixtures / "manifest.json").is_file() and kind == "cuda":
            cases += [
                (*case, False) for case in gsnet_cases(device, args.batch, args.gsnet_fixtures)
            ]
        else:
            print(
                f"No GSNet fixtures at {args.gsnet_fixtures} (or no GPU): skipped.", file=sys.stderr
            )
    rows = []
    for case, batch, call, exact in cases:
        rows.append(measure(case, batch, call, device, args.repeats, compiled, exact))
        print(f"  {case} (batch {batch}) done", file=sys.stderr, flush=True)
    print()
    print(render(rows, args.markdown))
    print(
        "\nTimes are min / median over", args.repeats, "runs; ratio is the median of paired ratios."
    )
    if args.json:
        payload = {"context": facts, "rows": [asdict(row) for row in rows]}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if any(row.identical is False for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
