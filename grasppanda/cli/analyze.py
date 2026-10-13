"""``grasppanda analyze``: where the AP of a result comes from, and how two results differ.

    grasppanda analyze <run directory | results directory | record.json> ... [--variant NAME]
    grasppanda analyze --compare A B [--variant NAME] [--plot diff.png]

It reads result records (``<run>/results/<variant>.json``, written by ``grasppanda
test``) and nothing else: no dataset, no weights, no predictions. For every record it
reports the conditions of the measurement, the precision per friction threshold, per
split and per scene, how many grasps each post-processing step left, coverage, and,
for records that carry the evaluator's breakdown, which share of the AP was earned and
lost on which object and for which reason.

``--compare A B`` pairs the scenes two records share and reports ``B - A`` per scene,
largest first, with a summary in which the scene is the unit: grasps and frames of one
scene are not independent, so intervals are computed over scenes.

``A:cd`` picks the variant ``cd`` of run ``A``; ``--variant`` picks one for every path.
Precisions are fractions in ``[0, 1]``; papers print them times 100.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grasppanda.evaluation.records import ResultRecord
from grasppanda.utils.console import style, table

__all__ = ["compare", "load_records", "overview", "register", "report"]

_SCENE = re.compile(r"scene_(\d+)$")


class AnalyzeError(ValueError):
    """The paths given to ``grasppanda analyze`` do not lead to result records."""


# --------------------------------------------------------------------------- loading


@dataclass
class Loaded:
    """A record and where it came from."""

    record: ResultRecord
    path: Path
    siblings: dict[str, ResultRecord]

    @property
    def run(self) -> Path:
        """The run directory of the record, or the directory of a loose record file."""
        parent = self.path.parent
        if parent.name == "results":
            return parent.parent
        if parent.parent.name == "results":  # results/epoch_NNNN/ of a training run
            return parent.parent.parent
        return parent

    @property
    def label(self) -> str:
        return f"{self.run} · {self.record.variant}"

    @property
    def short(self) -> str:
        """``<work dir>/<run id> · <variant>``, for a chart title."""
        return f"{'/'.join(self.run.parts[-2:])} · {self.record.variant}"


def _results_directory(path: Path) -> tuple[Path, str]:
    """The directory holding the records below ``path`` and a note on how it was chosen."""
    results = path / "results"
    if results.is_dir():
        if any(results.glob("*.json")):
            return results, ""
        epochs = sorted(
            child for child in results.iterdir() if child.is_dir() and any(child.glob("*.json"))
        )
        if epochs:
            note = (
                f"a training run: the records of its last evaluation ({epochs[-1].name}) are read"
            )
            return epochs[-1], note
        raise AnalyzeError(
            f"{results} holds no result record; was the run scored (grasppanda test)?"
        )
    if any(_is_record(file) for file in path.glob("*.json")):
        return path, ""
    runs = sorted(child for child in path.iterdir() if (child / "results").is_dir())
    if len(runs) == 1:
        return _results_directory(runs[0])
    if runs:
        listing = "\n  ".join(str(run) for run in runs)
        raise AnalyzeError(f"{path} holds {len(runs)} runs; name one of them:\n  {listing}")
    raise AnalyzeError(
        f"{path} is not a run directory: it has no results/ directory and no record. A run "
        "directory is what `grasppanda test` prints as `run`."
    )


def _is_record(file: Path) -> bool:
    try:
        ResultRecord.read(file)
    except (ValueError, TypeError, KeyError, OSError):
        return False
    return True


def load_records(spec: str, variant: str | None = None) -> tuple[list[Loaded], str]:
    """Read the records a path names.

    Args:
        spec: A record file, a results directory, a run directory, or a work directory
            holding exactly one run; optionally followed by ``:<variant>``.
        variant: Keep only this variant (the same as the ``:<variant>`` suffix).

    Returns:
        The records, in file-name order, and a note for the reader (empty when there is
        nothing to say).
    """
    path = Path(spec).expanduser()
    if not path.exists() and ":" in spec:
        head, _, tail = spec.rpartition(":")
        path, variant = Path(head).expanduser(), tail
    if not path.exists():
        raise AnalyzeError(f"{path} does not exist")
    note = ""
    if path.is_file():
        files = [path]
        directory = path.parent
    else:
        directory, note = _results_directory(path)
        files = sorted(directory.glob("*.json"))
    records: dict[str, tuple[ResultRecord, Path]] = {}
    for file in sorted(directory.glob("*.json")):
        try:
            record = ResultRecord.read(file)
        except (ValueError, TypeError, KeyError) as error:
            if file in files and path.is_file():
                raise AnalyzeError(f"{file} is not a result record: {error}") from error
            continue
        records[record.variant] = (record, file)
    siblings = {name: record for name, (record, _) in records.items()}
    chosen = [
        Loaded(record, file, siblings)
        for record, file in records.values()
        if file in files and (variant is None or record.variant == variant)
    ]
    if not chosen:
        have = ", ".join(sorted(records)) or "none"
        raise AnalyzeError(
            f"{directory} has no record of variant '{variant}'; it holds {have}"
            if variant
            else f"{directory} holds no result record"
        )
    return chosen, note


# -------------------------------------------------------------------------- analysis


def _scene_number(name: str) -> int | None:
    match = _SCENE.search(name)
    return int(match.group(1)) if match else None


def _frictions(record: ResultRecord) -> list[float] | None:
    columns = len(next(iter(record.per_scene.values()))["per_column"]) if record.per_scene else 0
    constants = (record.evaluator.get("info") or {}).get("constants") or {}
    for candidate in (constants.get("frictions"), record.breakdown.get("frictions")):
        if candidate and len(candidate) == columns:
            return [float(value) for value in candidate]
    return None


def _weighted(record: ResultRecord, names: list[str]) -> tuple[int, float, np.ndarray]:
    """Frames, mean precision and per-column mean precision over the given scenes."""
    frames = np.array([record.per_scene[name]["frames"] for name in names], dtype=np.float64)
    if frames.sum() == 0:
        return 0, float("nan"), np.zeros(0)
    mean = np.array([record.per_scene[name]["mean_precision"] for name in names])
    columns = np.array([record.per_scene[name]["per_column"] for name in names], dtype=np.float64)
    total = frames.sum()
    return int(total), float(frames @ mean / total), frames @ columns / total


def _manifest(record: ResultRecord) -> Any | None:
    from grasppanda.datasets.manifest import SplitManifest

    try:
        return SplitManifest.load(record.dataset)
    except (FileNotFoundError, ValueError, KeyError, ModuleNotFoundError):
        return None  # no manifest ships for this dataset: nothing is said about its splits


def _extent(record: ResultRecord) -> str:
    """Whether the record covers its split: the sentence that keeps a subset from being quoted."""
    scenes = len(record.per_scene)
    per_scene = sorted({entry["frames"] for entry in record.per_scene.values()})
    manifest = _manifest(record)
    if manifest is not None and record.split in manifest.splits:
        split = manifest.splits[record.split]
        whole = len(split.scenes) * len(split.frames) * max(len(record.cameras), 1)
        if record.requested >= whole:
            return f"the complete split '{record.split}' ({whole:,} frames)"
        steps = "/".join(str(count) for count in per_scene) or "0"
        return (
            f"a SUBSET of split '{record.split}': {record.requested:,} of its {whole:,} frames "
            f"({steps} of {len(split.frames)} frames per scene, {scenes} of {len(split.scenes)} "
            "scenes). Not a benchmark number."
        )
    constants = (record.evaluator.get("info") or {}).get("constants") or {}
    full = constants.get("frames_per_scene")
    if full and per_scene and per_scene[-1] < full:
        return (
            f"a SUBSET: {'/'.join(map(str, per_scene))} of {full} frames per scene. Not a "
            "benchmark number."
        )
    return f"{scenes} scenes; whether that is the whole split '{record.split}' is not recorded"


def _split_rows(record: ResultRecord) -> list[list[str]]:
    manifest = _manifest(record)
    if manifest is None:
        return []
    numbers = {name: _scene_number(name) for name in record.per_scene}
    own = set(manifest.splits[record.split].scenes) if record.split in manifest.splits else None
    rows = []
    for name, split in manifest.splits.items():
        wanted = set(split.scenes)
        if own is not None and not wanted <= own:
            continue
        names = [scene for scene, number in numbers.items() if number in wanted]
        if not names or (own is None and len({numbers[n] for n in names}) != len(wanted)):
            continue
        frames, mean, columns = _weighted(record, names)
        covered = len({numbers[n] for n in names})
        rows.append(
            [
                name,
                f"{covered}/{len(wanted)}",
                str(frames),
                f"{mean:.4f}",
                *(f"{c:.4f}" for c in columns),
            ]
        )
    return rows


def _count_lines(loaded: Loaded) -> list[str]:
    record = loaded.record
    info = record.evaluator.get("info") or {}
    frames = max(record.scored, 1)
    lines = []
    counts = record.grasp_counts
    if counts.get("per_scene"):
        totals = np.sum(list(counts["per_scene"].values()), axis=0)
        steps = counts["steps"]
        rows = [[steps[0], f"{int(totals[0]):,}", f"{totals[0] / frames:.1f}", "", ""]]
        for index in range(1, len(steps)):
            before, after = totals[index - 1], totals[index]
            removed = 100.0 * (before - after) / before if before else 0.0
            kept = 100.0 * after / totals[0] if totals[0] else 0.0
            rows.append(
                [
                    f"after {steps[index]}",
                    f"{int(after):,}",
                    f"{after / frames:.1f}",
                    f"{removed:.1f} %",
                    f"{kept:.1f} %",
                ]
            )
        headers = ["stage", "grasps", "per frame", "removed by this step", "left of the predicted"]
        lines += table(headers, rows, enabled=False).splitlines()
    elif "grasps_read" in info:
        read = info["grasps_read"]
        lines.append(f"scored: {read:,} grasps, {read / frames:.1f} per frame")
        raw = next(
            (
                other
                for other in loaded.siblings.values()
                if not other.postprocessing and other is not record
            ),
            None,
        )
        raw_read = (raw.evaluator.get("info") or {}).get("grasps_read") if raw is not None else None
        if record.postprocessing and raw_read:
            chain = " + ".join(step.get("type", "?") for step in record.postprocessing)
            lines.append(
                f"the '{raw.variant}' record of the same run read {raw_read:,}: the chain ({chain}) "
                f"removed {100.0 * (raw_read - read) / raw_read:.1f} % in total"
            )
        if record.postprocessing:
            lines.append(
                "per-step counts are not in this record (it was written before they were recorded)"
            )
    else:
        lines.append("grasp counts are not recorded by this evaluator")
    if "empty_predictions" in info:
        lines.append(f"frames with an empty prediction: {info['empty_predictions']}")
    return lines


def _shares(record: ResultRecord) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Per friction column, the share of AP earned and the shares lost by cause."""
    scenes = record.breakdown.get("per_scene") or {}
    frames = sum(entry.get("frames", 0) for entry in scenes.values())
    if not frames:
        return None
    columns = len(record.breakdown.get("frictions") or [])

    def total(name: str) -> np.ndarray:
        values = [
            np.broadcast_to(np.asarray(entry[name], dtype=np.float64), (columns,))
            for entry in scenes.values()
        ]
        return np.sum(values, axis=0) / frames

    gain = total("gain")
    return gain, {
        "no force closure": total("no_force_closure"),
        "collision": total("collision"),
        "empty": total("empty"),
        "unfilled ranks": total("unfilled"),
    }


def _object_rows(record: ResultRecord, limit: int) -> tuple[list[list[str]], int]:
    objects = record.breakdown.get("per_object") or {}
    frames = sum(
        entry.get("frames", 0) for entry in (record.breakdown.get("per_scene") or {}).values()
    )
    if not objects or not frames:
        return [], 0
    rows = []
    for object_id, entry in objects.items():
        earned = float(np.mean(entry["gain"])) if entry["gain"] else 0.0
        closure = float(np.mean(entry["no_force_closure"])) if entry["no_force_closure"] else 0.0
        lost = closure + entry["collision"] + entry["empty"]
        held = earned + lost
        rows.append((lost, object_id, entry, earned, closure, held))
    rows.sort(key=lambda row: (-row[0], row[1]))
    shown = rows if limit <= 0 else rows[:limit]
    body = [
        [
            object_id,
            str(entry["frames"]),
            f"{entry['ranked'] / max(entry['frames'], 1):.1f}",
            f"{100 * earned / frames:.2f}",
            f"{100 * closure / frames:.2f}",
            f"{100 * entry['collision'] / frames:.2f}",
            f"{100 * entry['empty'] / frames:.2f}",
            f"{earned / held:.3f}" if held else "-",
        ]
        for _, object_id, entry, earned, closure, held in shown
    ]
    return body, len(rows)


def _scene_rows(
    record: ResultRecord, limit: int, frictions: list[float] | None
) -> tuple[list[str], list[list[str]], str]:
    names = sorted(
        record.per_scene, key=lambda name: (record.per_scene[name]["mean_precision"], name)
    )
    reported = []
    constants = (record.evaluator.get("info") or {}).get("constants") or {}
    for metric, friction in constants.get("reported") or []:
        if frictions and friction in frictions:
            reported.append((metric, frictions.index(friction)))
    counts = record.grasp_counts.get("per_scene") or {}
    headers = ["scene", "frames", "mean precision", *(metric for metric, _ in reported)]
    if counts:
        headers.append("grasps/frame")

    def row(name: str) -> list[str]:
        entry = record.per_scene[name]
        cells = [name, str(entry["frames"]), f"{entry['mean_precision']:.4f}"]
        cells += [f"{entry['per_column'][column]:.4f}" for _, column in reported]
        if counts:
            cells.append(
                f"{counts[name][-1] / max(entry['frames'], 1):.1f}" if name in counts else "-"
            )
        return cells

    if limit <= 0 or len(names) <= 2 * limit:
        return headers, [row(name) for name in names], f"all {len(names)} scenes, lowest first"
    gap = ["…"] + [""] * (len(headers) - 1)
    body = [*(row(name) for name in names[:limit]), gap, *(row(name) for name in names[-limit:])]
    return (
        headers,
        body,
        f"the {limit} lowest and the {limit} highest of {len(names)}; --scenes 0 lists all",
    )


def report(loaded: Loaded, *, scenes: int = 5, objects: int = 10) -> str:
    """The text report of one record."""
    record = loaded.record
    info = record.evaluator.get("info") or {}
    frictions = _frictions(record)
    names = list(record.per_scene)
    _, mean, columns = _weighted(record, names)
    out = [style(loaded.label, "bold", enabled=False)]
    assisted = ", ".join(record.gt_assisted) or "none"
    commit = str(record.code.get("commit") or "unknown")[:10]
    dirty = {True: " (uncommitted changes)", False: ""}.get(record.code.get("dirty"), "")
    sha = str(record.weights.get("sha256") or "unknown")[:12]
    chain = (
        " + ".join(step.get("type", "?") for step in record.postprocessing)
        or "none (raw predictions)"
    )
    out += [
        f"  protocol        {record.protocol} · dataset {record.dataset} · split {record.split} · "
        f"camera {', '.join(record.cameras) or '-'}",
        f"  coverage        {record.scored} of {record.requested} frames scored "
        f"({100 * record.coverage:.1f} %), {len(record.missing)} missing",
        f"  extent          {_extent(record)}",
        f"  inputs          ground-truth-assisted: {assisted}",
        f"  post-processing {chain}",
        f"  provenance      weights sha256 {sha} · code {commit}{dirty} · seed {record.seed} · {record.created}",
    ]
    if record.partial:
        out.append(
            "  PARTIAL: not every requested frame was scored, so the record holds no headline "
            "metric. The numbers below describe the scored frames only."
        )
        if record.missing:
            listed = ", ".join(record.missing[:5]) + (" …" if len(record.missing) > 5 else "")
            out.append(f"  missing         {listed}")
    else:
        metrics = " · ".join(f"{name} {value:.4f}" for name, value in record.metrics.items())
        unit = info.get("metric_unit")
        out.append(f"  metrics         {metrics}" + (f"   ({unit})" if unit else ""))

    out += ["", "Where the precision comes from"]
    heads = (
        [f"μ≤{value:g}" for value in frictions]
        if frictions
        else [f"column {i}" for i in range(len(columns))]
    )
    out.append(
        table(
            ["friction threshold", *heads, "mean"],
            [["mean precision", *(f"{value:.4f}" for value in columns), f"{mean:.4f}"]],
            enabled=False,
        )
    )
    shares = _shares(record)
    if shares is not None:
        gain, lost = shares
        body = [["earned", *(f"{v:.4f}" for v in gain), f"{gain.mean():.4f}"]]
        body += [
            [f"lost: {name}", *(f"{v:.4f}" for v in values), f"{values.mean():.4f}"]
            for name, values in lost.items()
        ]
        total = gain + sum(lost.values())
        body.append(["sum", *(f"{v:.4f}" for v in total), f"{total.mean():.4f}"])
        out += [
            "",
            "Share of every frame's AP by outcome of the rank that owns it (earned = the precision above):",
            table(["outcome", *heads, "mean"], body, enabled=False),
        ]
    else:
        out += [
            "",
            "This record has no breakdown by cause and object (it predates it, or its evaluator has "
            "none): score the run again with `grasppanda test` to get one.",
        ]

    splits = _split_rows(record)
    if splits:
        out += [
            "",
            "By split",
            table(["split", "scenes", "frames", "mean", *heads], splits, enabled=False),
        ]

    headers, body, which = _scene_rows(record, scenes, frictions)
    out += ["", f"By scene ({which})", table(headers, body, enabled=False)]

    object_rows, count = _object_rows(record, objects)
    if object_rows:
        shown = "all" if len(object_rows) == count else f"the {len(object_rows)} that lose most"
        out += [
            "",
            f"By object ({shown} of {count}; percentage points of AP, mean over the friction thresholds)",
            table(
                [
                    "object",
                    "frames",
                    "ranked/frame",
                    "earned",
                    "no closure",
                    "collision",
                    "empty",
                    "success",
                ],
                object_rows,
                enabled=False,
            ),
            "  earned sums to the AP over all objects; success = earned / (earned + lost), rank-weighted",
        ]

    out += ["", "Grasps", *(f"  {line}" for line in _count_lines(loaded))]
    return "\n".join(out)


def overview(records: list[Loaded]) -> str:
    """One row per record: the view across splits, runs and variants."""
    metrics = list(dict.fromkeys(name for item in records for name in item.record.metrics))
    rows = []
    for item in records:
        record = item.record
        _, mean, _ = _weighted(record, list(record.per_scene))
        cells = [
            f"{record.metrics[name]:.4f}" if name in record.metrics else "-" for name in metrics
        ]
        if record.partial:
            cells = ["partial"] * len(metrics)
        rows.append(
            [
                item.short,
                record.split,
                f"{record.scored}/{record.requested}",
                str(len(record.per_scene)),
                f"{mean:.4f}",
                *cells,
            ]
        )
    headers = ["run · variant", "split", "frames", "scenes", "mean precision", *metrics]
    return table(headers, rows, enabled=False)


# ------------------------------------------------------------------------ comparison


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test: the chance of a split at least this uneven by coin flips."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(math.comb(trials, k) for k in range(extreme + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def compare(first: Loaded, second: Loaded, *, scenes: int = 10) -> tuple[str, dict[str, Any]]:
    """Scene-by-scene difference ``second - first`` with a paired summary.

    Returns:
        The text report and the numbers behind it (``names`` and ``differences`` sorted
        by size, ``mean``, ``interval``, ``wins``, ``losses``, ``ties``, ``p_sign``).
    """
    a, b = first.record, second.record
    if a.protocol != b.protocol or a.dataset != b.dataset:
        raise AnalyzeError(
            f"the two results are not comparable: A is {a.protocol} on {a.dataset}, B is "
            f"{b.protocol} on {b.dataset}"
        )
    common = sorted(set(a.per_scene) & set(b.per_scene))
    if not common:
        raise AnalyzeError("the two results share no scene")
    value_a = np.array([a.per_scene[name]["mean_precision"] for name in common])
    value_b = np.array([b.per_scene[name]["mean_precision"] for name in common])
    difference = value_b - value_a
    order = np.lexsort((np.array(common), -np.abs(difference)))
    names = [common[i] for i in order]
    ordered = difference[order]

    generator = np.random.default_rng(0)
    draws = generator.integers(0, len(common), size=(10_000, len(common)))
    low, high = np.percentile(difference[draws].mean(axis=1), [2.5, 97.5])
    wins, losses = int((difference > 0).sum()), int((difference < 0).sum())
    ties = len(common) - wins - losses
    numbers = {
        "names": names,
        "differences": ordered.tolist(),
        "mean_a": float(value_a.mean()),
        "mean_b": float(value_b.mean()),
        "mean": float(difference.mean()),
        "median": float(np.median(difference)),
        "std": float(difference.std(ddof=1)) if len(common) > 1 else 0.0,
        "interval": (float(low), float(high)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_sign": _sign_test(wins, losses),
        "scenes": len(common),
    }

    notes = []
    for what, left, right in (
        ("split", a.split, b.split),
        ("cameras", a.cameras, b.cameras),
        ("ground-truth-assisted inputs", a.gt_assisted, b.gt_assisted),
        (
            "post-processing",
            [s.get("type") for s in a.postprocessing],
            [s.get("type") for s in b.postprocessing],
        ),
        ("weights", str(a.weights.get("sha256"))[:12], str(b.weights.get("sha256"))[:12]),
        ("code", str(a.code.get("commit"))[:10], str(b.code.get("commit"))[:10]),
    ):
        if left != right:
            notes.append(f"{what}: A {left or 'none'} · B {right or 'none'}")
    frames_differ = [n for n in common if a.per_scene[n]["frames"] != b.per_scene[n]["frames"]]
    if frames_differ:
        notes.append(
            f"{len(frames_differ)} scene(s) were scored on a different number of frames in A and B"
        )
    only_a, only_b = (
        len(set(a.per_scene) - set(b.per_scene)),
        len(set(b.per_scene) - set(a.per_scene)),
    )
    if only_a or only_b:
        notes.append(f"unpaired scenes left out: {only_a} only in A, {only_b} only in B")
    if a.partial or b.partial:
        notes.append("at least one of the two records is partial (not every frame was scored)")

    out = [
        f"A  {first.label}",
        f"B  {second.label}",
        f"   protocol {a.protocol} · dataset {a.dataset} · {len(common)} paired scenes",
    ]
    out += [f"   differs: {note}" for note in notes]
    out += [
        "",
        "Paired summary over scenes (mean precision of a scene; B − A)",
        f"  mean A {numbers['mean_a']:.4f} · mean B {numbers['mean_b']:.4f} · mean difference "
        f"{numbers['mean']:+.4f} · median {numbers['median']:+.4f} · standard deviation {numbers['std']:.4f}",
        f"  95 % interval of the mean difference {low:+.4f} to {high:+.4f} (bootstrap over the "
        f"{len(common)} scenes, 10 000 resamples, seed 0)",
        f"  B higher in {wins} scenes, A higher in {losses}, equal in {ties} · two-sided sign test "
        f"p = {numbers['p_sign']:.3g}",
    ]
    if len(common) < 10:
        out.append(f"  only {len(common)} scenes: read the interval as a rough guide")
    shown = names if scenes <= 0 else names[:scenes]
    body = [
        [
            name,
            f"{a.per_scene[name]['mean_precision']:.4f}",
            f"{b.per_scene[name]['mean_precision']:.4f}",
            f"{value:+.4f}",
        ]
        for name, value in zip(shown, ordered)
    ]
    which = (
        f"all {len(names)}"
        if len(shown) == len(names)
        else f"the {len(shown)} largest of {len(names)}; --scenes 0 lists all"
    )
    out += [
        "",
        f"Per scene, largest difference first ({which})",
        table(["scene", "A", "B", "B − A"], body, enabled=False),
    ]
    return "\n".join(out), numbers


# ------------------------------------------------------------------------------ plot


def _plot_one(loaded: Loaded, path: Path) -> None:
    from grasppanda.visualization.charts import draw_analysis

    record = loaded.record
    frictions = _frictions(record)
    names = sorted(
        record.per_scene, key=lambda name: (record.per_scene[name]["mean_precision"], name)
    )
    _, mean, columns = _weighted(record, names)
    shares = _shares(record)
    parts = None
    if shares is not None:
        gain, lost = shares
        parts = {"earned (= mean precision)": float(gain.mean())}
        parts.update({f"lost: {name}": float(values.mean()) for name, values in lost.items()})
    labels = (
        [f"{value:g}" for value in frictions]
        if frictions
        else [str(i) for i in range(len(columns))]
    )
    subtitle = (
        f"{record.protocol} · {record.dataset} {record.split} · {record.scored}/{record.requested} "
        f"frames in {len(names)} scenes · ground-truth-assisted inputs: "
        f"{', '.join(record.gt_assisted) or 'none'} · mean precision {mean:.4f}"
    )
    image = draw_analysis(
        loaded.short,
        subtitle,
        labels,
        columns.tolist(),
        names,
        [record.per_scene[name]["mean_precision"] for name in names],
        parts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _plot_comparison(first: Loaded, second: Loaded, numbers: dict[str, Any], path: Path) -> None:
    from grasppanda.visualization.charts import draw_comparison

    low, high = numbers["interval"]
    subtitle = (
        f"mean difference {numbers['mean']:+.4f} (95 % interval {low:+.4f} to {high:+.4f}, bootstrap "
        f"over {numbers['scenes']} scenes) · B higher in {numbers['wins']}, A higher in {numbers['losses']}"
    )
    image = draw_comparison(
        f"A: {first.short}    B: {second.short}",
        subtitle,
        numbers["names"],
        numbers["differences"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# ------------------------------------------------------------------------------- cli


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "analyze",
        help="say where the AP of a result comes from; compare two results scene by scene",
        description=(
            "Read result records and report precision per friction threshold, split, scene and "
            "object, grasp counts through the post-processing chain, and coverage. Needs no "
            "dataset and no weights."
        ),
        epilog="A path may end in :<variant> (run_dir:cd). Precisions are fractions; papers print them times 100.",
    )
    parser.add_argument(
        "paths", nargs="*", metavar="RUN", help="run directory, results directory or record file"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="report B − A per scene with a paired summary",
    )
    parser.add_argument("--variant", metavar="NAME", help="read only this post-processing variant")
    parser.add_argument(
        "--scenes", type=int, default=None, metavar="N", help="scenes to list at each end (0: all)"
    )
    parser.add_argument(
        "--objects", type=int, default=10, metavar="N", help="objects to list (0: all; default 10)"
    )
    parser.add_argument(
        "--plot",
        metavar="FILE",
        help="write a chart (.png) of the one record, or of the comparison",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        if args.compare:
            if args.paths:
                raise AnalyzeError("--compare takes exactly two results; remove the other paths")
            pair = []
            for spec in args.compare:
                loaded, _ = load_records(spec, args.variant)
                if len(loaded) != 1:
                    have = ", ".join(item.record.variant for item in loaded)
                    raise AnalyzeError(
                        f"{spec} holds the variants {have}; pick one with {spec}:<variant> or --variant"
                    )
                pair.append(loaded[0])
            text, numbers = compare(
                pair[0], pair[1], scenes=10 if args.scenes is None else args.scenes
            )
            print(text)
            if args.plot:
                _plot_comparison(pair[0], pair[1], numbers, Path(args.plot))
                print(f"\nwrote {args.plot}")
            return 0
        if not args.paths:
            raise AnalyzeError("name at least one run directory or record, or use --compare A B")
        everything: list[Loaded] = []
        for spec in args.paths:
            loaded, note = load_records(spec, args.variant)
            if note:
                print(f"note: {spec}: {note}")
            everything += loaded
        for index, loaded in enumerate(everything):
            if index:
                print("\n" + "─" * 78 + "\n")
            print(
                report(
                    loaded, scenes=5 if args.scenes is None else args.scenes, objects=args.objects
                )
            )
        if len(everything) > 1:
            print("\n" + "─" * 78 + "\n\nAll records\n" + overview(everything))
        if args.plot:
            if len(everything) != 1:
                have = ", ".join(item.record.variant for item in everything)
                raise AnalyzeError(
                    f"--plot draws one record, but {len(everything)} were read ({have}); pick one "
                    "with --variant"
                )
            _plot_one(everything[0], Path(args.plot))
            print(f"\nwrote {args.plot}")
    except AnalyzeError as error:
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        print(f"{mark} {error}", file=sys.stderr)
        return 1
    return 0
