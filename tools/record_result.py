"""Put the result records of a whole-split evaluation into the model zoo.

    python -m tools.record_result <run directory> --method <configs directory name>

``<run directory>`` is what ``grasppanda test`` wrote (``run.json``, ``git.json``,
``results/<variant>.json``), or the work directory above it when that holds one run with
result records. The tool

1. **checks the run**: every variant scored every frame of the whole test split of its
   protocol; the code was committed and clean; the weights were hashed and loaded
   without a mismatch; the variants agree on everything but their post-processing; and
   the weights are those the metafile lists (for a converted checkpoint: the file it
   was converted from, read from the note ``tools/convert_checkpoint.py`` leaves in it);
2. **copies each record** to
   ``configs/<method>/records/<model>_<provenance>_<split>_<variant>.json``, unchanged
   but for the paths of this machine: a path below a directory named by a
   ``GRASPPANDA_*`` environment variable is written with the name of the variable
   (``$GRASPPANDA_CACHE/...``), and a record in which another absolute path is left is
   refused;
3. **prints the rows** for ``configs/<method>/metafile.yaml``, with the numbers computed
   from the per-scene values of the records, and says whether the metafile states them.

The metafile itself is edited by hand: its rows stand next to reported numbers and
comments that a tool should not rewrite, and a weights entry needs a link and a licence
that no run knows. Running the tool again changes nothing; once the rows are pasted it
ends with ``the metafile states these rows``.

A row is ``author-ckpt`` when the model's weights entry in the metafile has ``source:
author`` and ``reproduced`` when it has ``source: grasppanda``. Without a weights entry,
say which it is with ``--provenance``; the entry to add is printed with the rows.

The weights that were evaluated are tied to that entry. Weights released by this project
are the evaluated file itself: the two hashes must be equal. The authors' file was
converted first, so the hashes differ, and the tie is the note the converter leaves in
the converted file (``source_sha256``); the tool therefore reads the weights file at the
path the record names. If that file is gone, the run is refused unless
``--allow-unverified-weights`` is given, and the output then says ``NOT VERIFIED``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grasppanda.config import ConfigError
from grasppanda.datasets.manifest import SplitManifest
from grasppanda.evaluation.records import RECORD_SCHEMA
from grasppanda.hub import PROTOCOL_SPLITS, ModelEntry, Weights, load_metafile, sha256_of

REPO = Path(__file__).resolve().parents[1]

#: What ``inputs`` says for each name a record lists under ``gt_assisted``.
GT_ASSISTED = {"workspace": "workspace crop from the ground-truth segmentation"}
#: The sources a weights entry may have, by the provenance of the rows they back.
WEIGHT_SOURCES = {"author-ckpt": "author", "reproduced": "grasppanda"}
_DATASETS = {"graspnet1b": "GraspNet-1Billion", "graspclutter6d": "GraspClutter6D"}
#: Fields of a record this tool reads beside the ones it checks one by one.
_FIELDS = ("variant", "dataset", "per_scene", "postprocessing", "gt_assisted", "evaluator")
#: An absolute path inside a string: ``/a/b``, ``~/a``, ``C:/a`` or a ``file://`` URL.
#: Not the rest of ``$NAME/a/b``, of a relative path or of a web address.
_ABSOLUTE = re.compile(r"(?<![\w$}./~\\-])(?:~|[A-Za-z]:)?[/\\][\w.+~-]+[/\\]|file://")


class RecordError(ValueError):
    """A run cannot be recorded; the message lists every reason."""


@dataclass(frozen=True)
class WholeSplit:
    """The test split a headline number of one protocol covers.

    Attributes:
        split: Name of the split in a record.
        frames_per_scene: Frames every scene of the split has, per camera.
        scene_count: Scenes of the split.
        scenes: Scenes of every split a zoo row names, or ``None`` when the row has one
            split and the dataset tree alone lists its scenes.
        metrics: The numbers a zoo row of the protocol states.
    """

    split: str
    frames_per_scene: int
    scene_count: int
    scenes: dict[str, tuple[int, ...]] | None
    metrics: tuple[str, ...]


def whole_split(protocol: str) -> WholeSplit:
    """What "the whole test split" means under ``protocol``."""
    if protocol == "graspnet1b-sv-v1":
        manifest = SplitManifest.load("graspnet1b")
        test = manifest.split("test")
        scenes = {name: manifest.split(f"test_{name}").scenes for name in PROTOCOL_SPLITS[protocol]}
        return WholeSplit("test", len(test.frames), len(test.scenes), scenes, ("AP",))
    if protocol == "graspclutter6d-sv-v1":
        # 235 scenes x 13 views; the scene ids are listed by the dataset's own split file.
        return WholeSplit("test", 13, 235, None, ("AP", "AP0.8", "AP0.4"))
    raise RecordError(f"no zoo row is defined for protocol '{protocol}'")


def _by_scene(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """The per-scene table of a record by scene id (its keys are ``<camera>/scene_NNNN``)."""
    table = record.get("per_scene") or {}
    return {int(key.rsplit("scene_", 1)[-1]): entry for key, entry in table.items()}


def headline(record: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """The numbers of a record's zoo row, in percent: ``{"AP": {split: value}, ...}``.

    ``AP`` of a split is the mean over its scenes of the per-scene mean precision.
    Every scene of a whole split has the same number of frames, so this is the mean
    over the frames. ``AP0.8`` and ``AP0.4`` are the same mean of the friction column
    the evaluator reports under that name.
    """
    whole = whole_split(record["protocol"])
    constants = record["evaluator"]["info"]["constants"]
    columns = {name: constants["frictions"].index(level) for name, level in constants["reported"]}
    scenes = _by_scene(record)
    groups = whole.scenes or {PROTOCOL_SPLITS[record["protocol"]][0]: tuple(sorted(scenes))}
    numbers: dict[str, dict[str, float]] = {}
    for metric in whole.metrics:
        numbers[metric] = {}
        for split, ids in groups.items():
            entries = [scenes[scene] for scene in ids]
            values = [
                entry["mean_precision"] if metric == "AP" else entry["per_column"][columns[metric]]
                for entry in entries
            ]
            numbers[metric][split] = round(100 * sum(values) / len(values), 2)
    return numbers


def collision_filter(record: Mapping[str, Any]) -> bool:
    """Whether the post-processing chain of the record holds a collision filter."""
    return any(
        "collision" in str(step.get("type", "")).lower() for step in record["postprocessing"]
    )


# ---------------------------------------------------------------------- the run


@dataclass
class Run:
    """A run of ``grasppanda test``: its snapshot files and the records it wrote."""

    directory: Path
    info: dict[str, Any]
    git: dict[str, Any]
    records: dict[str, dict[str, Any]]

    @property
    def first(self) -> dict[str, Any]:
        return next(iter(self.records.values()))

    @property
    def config(self) -> str:
        """``<directory>/<file>`` of the run's config, as a metafile names it."""
        return "/".join(Path(str(self.info.get("config", ""))).parts[-2:])


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordError(f"cannot read {path}: {error}") from None


def load_run(path: str | Path, variants: Sequence[str] | None = None) -> Run:
    """Read a run directory, or the one run with result records below a work directory."""
    directory = Path(path)
    if not (directory / "results").is_dir():
        found = sorted(p.parent for p in directory.glob("*/results") if any(p.glob("*.json")))
        if not found:
            raise RecordError(
                f"{directory} holds no run with result records: the evaluation has not "
                "finished, or this is not a directory of `grasppanda test`"
            )
        if len(found) > 1:
            raise RecordError(
                f"{directory} holds {len(found)} runs with result records "
                f"({', '.join(p.name for p in found)}); name one of them"
            )
        directory = found[0]
    files = {p.stem: p for p in sorted((directory / "results").glob("*.json"))}
    if variants:
        missing = [name for name in variants if name not in files]
        if missing:
            raise RecordError(
                f"{directory}/results has no record of {', '.join(missing)}; "
                f"it has {', '.join(files) or 'none'}"
            )
        files = {name: files[name] for name in variants}
    if not files:
        raise RecordError(f"{directory}/results holds no result record: the run did not finish")
    order = sorted(files, key=lambda name: (name != "raw", name))  # raw first
    return Run(
        directory,
        _read(directory / "run.json"),
        _read(directory / "git.json"),
        {name: _read(files[name]) for name in order},
    )


def _record_problems(variant: str, record: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def problem(text: str) -> None:
        found.append(f"{variant}: {text}")

    if record.get("schema") != RECORD_SCHEMA:
        problem(f"record schema {record.get('schema')!r}, this tool reads {RECORD_SCHEMA}")
        return found
    absent = [key for key in _FIELDS if key not in record]
    if absent:
        problem(f"not a result record of `grasppanda test`: no {', '.join(absent)}")
        return found
    scored, requested = record.get("scored"), record.get("requested")
    if (
        record.get("partial")
        or scored != requested
        or record.get("missing")
        or not record.get("metrics")
    ):
        problem(
            f"incomplete: {scored} of {requested} frames scored, "
            f"{len(record.get('missing') or [])} missing; a partial result has no headline number"
        )
    try:
        whole = whole_split(str(record.get("protocol")))
    except RecordError as error:
        problem(str(error))
        return found
    if record.get("split") != whole.split:
        problem(f"split '{record.get('split')}' is not the whole test split ('{whole.split}')")
    if len(record.get("cameras") or []) != 1:
        problem(f"one camera per row; the record has {record.get('cameras')}")
    scenes = _by_scene(record)
    expected = {scene for ids in whole.scenes.values() for scene in ids} if whole.scenes else None
    if expected is not None and set(scenes) != expected:
        absent = sorted(expected - set(scenes))
        extra = sorted(set(scenes) - expected)
        problem(
            f"not the scenes of the split: {len(absent)} absent {absent[:5]}, {len(extra)} extra"
        )
    if len(scenes) != whole.scene_count:
        problem(f"{len(scenes)} scenes, the split has {whole.scene_count}")
    short = sorted(
        s for s, entry in scenes.items() if entry.get("frames") != whole.frames_per_scene
    )
    if short:
        problem(
            f"{len(short)} scenes without all {whole.frames_per_scene} frames (first: {short[:5]})"
        )
    if requested != whole.scene_count * whole.frames_per_scene:
        problem(
            f"{requested} frames requested, the split has "
            f"{whole.scene_count * whole.frames_per_scene}"
        )
    code = record.get("code") or {}
    if not code.get("commit"):
        problem("the record names no code commit")
    if code.get("dirty") is not False:
        problem("the code had uncommitted changes (code.dirty); commit, then run again")
    weights = record.get("weights") or {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(weights.get("sha256", ""))):
        problem("the record holds no sha256 of the weights")
    if weights.get("clean_load") is False:
        problem("the weights did not load cleanly (weights.clean_load is false)")
    return found


def problems(run: Run) -> list[str]:
    """Every reason why ``run`` cannot back a zoo row; empty when it can."""
    found = [
        text for name, record in run.records.items() for text in _record_problems(name, record)
    ]
    for key in ("protocol", "dataset", "split", "cameras", "config_hash", "seed", "gt_assisted"):
        values = {name: record.get(key) for name, record in run.records.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
            found.append(f"the variants disagree on {key}: {values}")
    for section, key in (("code", "commit"), ("weights", "sha256")):
        values = {
            name: (record.get(section) or {}).get(key) for name, record in run.records.items()
        }
        if len(set(values.values())) > 1:
            found.append(f"the variants disagree on {section}.{key}: {values}")
    filters: dict[bool, str] = {}
    for name, record in run.records.items():
        if record.get("schema") != RECORD_SCHEMA or "postprocessing" not in record:
            continue
        flag = collision_filter(record)
        if flag in filters:
            found.append(
                f"variants '{filters[flag]}' and '{name}' are both "
                f"{'with' if flag else 'without'} a collision filter and would make one "
                "row; choose with --variants"
            )
        filters[flag] = name
    code = run.first.get("code") or {}
    if run.git.get("dirty") is not False or not run.git.get("available", True):
        found.append("git.json: the run was not started from a clean commit")
    elif run.git.get("commit") != code.get("commit"):
        found.append(
            f"git.json names commit {run.git.get('commit')}, the records {code.get('commit')}"
        )
    if run.info.get("command_name", "test") != "test":
        found.append(f"run.json: the command was '{run.info.get('command_name')}', not 'test'")
    if run.info.get("config_hash") != run.first.get("config_hash"):
        found.append("run.json and the records name different config hashes")
    return found


# ---------------------------------------------------------- paths of this machine


def machine_roots(environ: Mapping[str, str] | None = None) -> list[tuple[str, str]]:
    """``(directory, "$NAME")`` of every ``GRASPPANDA_*`` variable that holds an
    absolute path, the longest directory first."""
    environ = os.environ if environ is None else environ
    roots: dict[str, str] = {}
    # GRASPPANDA_CACHE first: where two variables name one directory, it is the cache.
    for name, value in sorted(
        environ.items(), key=lambda item: (item[0] != "GRASPPANDA_CACHE", item)
    ):
        if not name.startswith("GRASPPANDA_") or not os.path.isabs(os.path.expanduser(value)):
            continue
        expanded = os.path.expanduser(value).rstrip("/\\")
        for directory in {expanded, os.path.realpath(expanded)}:
            if directory:
                roots.setdefault(directory, f"${name}")
    return sorted(roots.items(), key=lambda item: -len(item[0]))


def without_machine_paths(value: Any, roots: Sequence[tuple[str, str]]) -> Any:
    """A copy of ``value`` in which every path below a root is written with its name."""
    if isinstance(value, str):
        for directory, name in roots:
            value = re.sub(re.escape(directory) + r"(?=$|[/\\\s\"'])", lambda _, n=name: n, value)
        return value
    if isinstance(value, Mapping):
        return {key: without_machine_paths(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [without_machine_paths(item, roots) for item in value]
    return value


def absolute_paths(value: Any, where: str = "") -> list[str]:
    """``<key path>: <string>`` of every string in ``value`` that holds an absolute path."""
    if isinstance(value, str):
        return [f"{where}: {value}"] if _ABSOLUTE.search(value) else []
    if isinstance(value, Mapping):
        return [
            text
            for key, item in value.items()
            for text in absolute_paths(item, f"{where}.{key}" if where else str(key))
        ]
    if isinstance(value, list):
        return [
            text
            for index, item in enumerate(value)
            for text in absolute_paths(item, f"{where}.{index}")
        ]
    return []


# -------------------------------------------------------------------- the weights


def conversion_note(path: Path) -> dict[str, Any] | None:
    """The note of ``tools/convert_checkpoint.py`` inside a weights file, or ``None``."""
    import torch

    try:
        content = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # not a file this toolbox wrote; nothing can be read from it
        return None
    note = content.get("meta") if isinstance(content, dict) else None
    return dict(note) if isinstance(note, dict) and note.get("source_sha256") else None


@dataclass
class WeightsCheck:
    """What is known about the weights of a run, and what follows for its rows."""

    provenance: str | None
    problems: list[str]
    notes: list[str]
    #: The released file a converted checkpoint came from (``source_file``, ``source_sha256``, ...).
    note: dict[str, Any] | None = None
    #: The weights entry of the metafile the evaluated weights belong to, if any.
    entry: Weights | None = None


def check_weights(
    run: Run,
    models: Sequence[ModelEntry],
    model: ModelEntry | None,
    provenance: str | None,
    allow_unverified: bool = False,
) -> WeightsCheck:
    """Decide the provenance of the rows and tie the evaluated weights to the metafile.

    The authors' file and the converted file that was evaluated have different hashes;
    what ties them is the note the converter leaves in the converted file. When that
    file is gone the tie cannot be checked, which is a refusal unless ``allow_unverified``.

    Args:
        run: The run.
        models: Every model of the method's metafile.
        model: The model of the run's config, or ``None`` when the metafile does not
            list it. Without a weights entry of its own (a transfer config that is not
            listed yet), the run takes the entry of another model of the method whose
            file is the one that was evaluated.
        provenance: What the user said the weights are, or ``None``.
        allow_unverified: Accept an author checkpoint whose origin cannot be read.
    """
    weights = run.first.get("weights") or {}
    evaluated = str(weights.get("sha256", ""))
    check = WeightsCheck(provenance, [], [])
    path = Path(str(weights.get("path", "")))
    readable = path.is_file() and sha256_of(path) == evaluated
    if readable:
        check.note = conversion_note(path)
    entry = model.weights if model else None
    if entry is None:
        origin = {evaluated, check.note["source_sha256"] if check.note else evaluated}
        owner = next((m for m in models if m.weights and m.weights.sha256 in origin), None)
        if owner is not None:
            entry = owner.weights
            check.notes.append(f"the weights are those of the metafile's model {owner.name}")
    check.entry = entry
    if not readable and (entry is None or entry.source == "author"):
        text = (
            f"the evaluated weights file is not at {path} any more (or was replaced), so "
            "the file it was converted from cannot be read"
        )
        if entry is None or allow_unverified:
            check.notes.append(f"NOT VERIFIED: {text}")
        else:
            check.problems.append(
                f"{text} and compared with the metafile; pass --allow-unverified-weights "
                "to record the run without that check"
            )
    if entry is None:
        if provenance is None:
            check.problems.append(
                "the metafile has no weights entry for these weights: say what they are "
                "with --provenance author-ckpt (the authors' released file, converted) or "
                "--provenance reproduced (trained here); the entry to add is printed with "
                "the rows"
            )
        return check
    implied = {source: name for name, source in WEIGHT_SOURCES.items()}.get(entry.source)
    if implied is None:
        check.problems.append(f"the weights entry of the metafile has source '{entry.source}'")
        return check
    if provenance not in (None, implied):
        check.problems.append(
            f"the weights entry of the metafile has source '{entry.source}', which makes "
            f"these rows '{implied}'; --provenance {provenance} contradicts it"
        )
    check.provenance = implied
    if implied == "reproduced":
        if evaluated != entry.sha256:
            check.problems.append(
                f"the evaluated weights (sha256 {evaluated}) are not the released file of "
                f"the metafile ({entry.sha256})"
            )
        else:
            check.notes.append("the evaluated weights are the released file of the metafile")
    elif check.note is not None and check.note["source_sha256"] != entry.sha256:
        check.problems.append(
            f"the evaluated weights were converted from {check.note.get('source_file')} "
            f"(sha256 {check.note['source_sha256']}), the metafile lists {entry.sha256}"
        )
    elif check.note is not None:
        check.notes.append(
            f"the evaluated weights were converted from {check.note.get('source_file')} "
            f"(sha256 {entry.sha256[:12]}…), the file the metafile lists"
        )
    elif readable:
        check.problems.append(
            "the evaluated weights carry no note of tools/convert_checkpoint.py, so they "
            "are not a converted copy of the authors' file the metafile lists"
        )
    return check


# ------------------------------------------------------------------------ output


def record_stem(model: str, provenance: str, split: str) -> str:
    """``<model>_<provenance>_<split>``; a record file is this plus ``_<variant>.json``."""
    return f"{model}_{provenance}_{split}"


def inputs_text(gt_assisted: Sequence[str]) -> str:
    """The ``inputs`` of a row: what was derived from ground truth, ending with the
    names as the record lists them."""
    if not gt_assisted:
        return "nothing derived from ground truth (gt_assisted: none)"
    described = "; ".join(
        GT_ASSISTED.get(name, f"{name} derived from ground truth") for name in gt_assisted
    )
    return f"{described} (gt_assisted: {', '.join(gt_assisted)})"


def _mapping(values: Mapping[str, float]) -> str:
    return "{" + ", ".join(f"{split}: {value:.2f}" for split, value in values.items()) + "}"


def rows_yaml(run: Run, provenance: str, stem: str) -> str:
    """The rows of the run for a metafile, as YAML at the indentation of ``results:``."""
    first = run.first
    whole = whole_split(first["protocol"])
    lines = [
        f"      # {_DATASETS.get(first['dataset'], first['dataset'])}, the whole test split "
        f"({whole.scene_count} scenes, {first['requested']:,} frames)",
        f"      # through `grasppanda test` at commit {first['code']['commit'][:7]}, scored by "
        f"{first['evaluator']['type']}.",
    ]
    for variant, record in run.records.items():
        steps = [str(step.get("type")) for step in record["postprocessing"]]
        if len(steps) > 1:
            lines.append(f"      # Post-processing of '{variant}': {', then '.join(steps)}.")
        lines += [
            f"      - provenance: {provenance}",
            f"        collision_filter: {str(collision_filter(record)).lower()}",
            f"        evaluator: {record['evaluator']['type']}",
            f"        record: records/{stem}_{variant}.json",
            f'        inputs: "{inputs_text(record["gt_assisted"])}"',
        ]
        lines += [
            f"        {metric}: {_mapping(values)}" for metric, values in headline(record).items()
        ]
    return "\n".join(lines) + "\n"


def weights_yaml(
    run: Run, provenance: str, note: Mapping[str, Any] | None, entry: Weights | None = None
) -> str:
    """The weights entry of a model whose metafile entry has none: the entry of the
    model the weights belong to, or one to complete by hand."""
    lines = ["    weights:"]
    if entry is not None:
        return (
            "\n".join(
                [
                    *lines,
                    f"      url: {entry.url}",
                    f"      sha256: {entry.sha256}",
                    f"      license: {json.dumps(entry.license, ensure_ascii=False)}",
                    f"      source: {entry.source}",
                ]
            )
            + "\n"
        )
    if provenance == "author-ckpt":
        name = note.get("source_file") if note else "<the released file>"
        digest = note["source_sha256"] if note else "<sha256 of the released file>"
        if note:
            lines.append(
                f"      # {name}, converted with tools/convert_checkpoint.py "
                f"(--source-format {note.get('source_format')})"
            )
        lines += [f"      url: <where the authors publish {name}>", f"      sha256: {digest}"]
    else:
        lines += [
            "      url: <where this project publishes the checkpoint>",
            f"      sha256: {run.first['weights']['sha256']}",
        ]
    lines += [
        "      license: <the licence of the weights>",
        f"      source: {WEIGHT_SOURCES[provenance]}",
    ]
    return "\n".join(lines) + "\n"


def metafile_state(
    run: Run, model: ModelEntry | None, provenance: str, stem: str
) -> tuple[bool, str]:
    """Whether the metafile states the rows of the run: ``(ok, sentence)``; ``ok`` is false
    only when the metafile states other numbers for one of the records."""
    stated = {result.record: result for result in model.results} if model else {}
    present, different = 0, []
    for variant, record in run.records.items():
        result = stated.get(f"records/{stem}_{variant}.json")
        if result is None:
            continue
        present += 1
        mine = (provenance, collision_filter(record), record["evaluator"]["type"], headline(record))
        if mine != (result.provenance, result.collision_filter, result.evaluator, result.numbers()):
            different.append(variant)
    if different:
        return False, f"the metafile states OTHER rows for: {', '.join(different)}; replace them"
    if present == len(run.records):
        return True, "the metafile states these rows"
    if model is None:
        return True, "the metafile does not list this model: paste the entry above under `models:`"
    return True, (
        f"the metafile states {present} of these {len(run.records)} rows: paste the rows above "
        "under `results:` of the model"
    )


def write_records(
    run: Run, directory: Path, stem: str, roots: Sequence[tuple[str, str]], force: bool
) -> list[str]:
    """Write the records without machine paths; returns one line per file.

    Raises:
        RecordError: An absolute path is left in a record, or a file with other content
            exists and ``force`` is not set. Nothing is written then.
    """
    texts: dict[Path, str] = {}
    paths, conflicts = [], []
    for variant, record in run.records.items():
        clean = without_machine_paths(copy.deepcopy(record), roots)
        paths += [f"{variant}: {text}" for text in absolute_paths(clean)]
        path = directory / f"{stem}_{variant}.json"
        texts[path] = json.dumps(clean, indent=2, sort_keys=True) + "\n"
        if path.is_file() and path.read_text(encoding="utf-8") != texts[path] and not force:
            conflicts.append(f"{path} exists with other content; pass --force to replace it")
    if paths:
        raise RecordError(
            "a committed record must not hold a path of this machine; set the GRASPPANDA_* "
            "variable that names the directory (GRASPPANDA_CACHE, ...) and run again:\n  "
            + "\n  ".join(paths)
        )
    if conflicts:
        raise RecordError("\n  ".join(conflicts))
    lines = []
    for path, text in texts.items():
        unchanged = path.is_file() and path.read_text(encoding="utf-8") == text
        if not unchanged:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        lines.append(f"{'unchanged' if unchanged else 'wrote    '} {path}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="run directory of `grasppanda test`")
    parser.add_argument("--method", required=True, help="directory below configs/, e.g. gsnet")
    parser.add_argument(
        "--name",
        help="stem of the record files in place of the systematic "
        "<model>_<provenance>_<split> (to which _<variant>.json is appended); "
        "tests/unit/test_zoo_records.py lists every record that is named otherwise",
    )
    parser.add_argument(
        "--provenance",
        choices=sorted(WEIGHT_SOURCES),
        help="needed only while the metafile has no weights entry for the model",
    )
    parser.add_argument("--variants", nargs="+", help="variants to record (default: all)")
    parser.add_argument(
        "--allow-unverified-weights",
        action="store_true",
        help="record although the evaluated weights file is gone and cannot be tied to the "
        "authors' file of the metafile",
    )
    parser.add_argument("--force", action="store_true", help="replace record files that differ")
    parser.add_argument("--dry-run", action="store_true", help="check and print, write nothing")
    parser.add_argument(
        "--configs", type=Path, default=REPO / "configs", help="configs directory (for tests)"
    )
    args = parser.parse_args(argv)
    try:
        metafile = load_metafile(args.configs / args.method / "metafile.yaml")
        run = load_run(args.run, args.variants)
        model = next((m for m in metafile.models if m.config == run.config), None)
        found = problems(run)
        weights = check_weights(
            run, metafile.models, model, args.provenance, args.allow_unverified_weights
        )
        if found or weights.problems:
            raise RecordError(
                f"{run.directory} cannot back a zoo row:\n  "
                + "\n  ".join(found + weights.problems)
            )
        provenance = str(weights.provenance)
        name = model.name if model else Path(run.config).stem
        stem = args.name or record_stem(name, provenance, run.first["split"])
        directory = args.configs / args.method / "records"
        written = (
            [f"would write {directory / f'{stem}_{variant}.json'}" for variant in run.records]
            if args.dry_run
            else write_records(run, directory, stem, machine_roots(), args.force)
        )
    except (RecordError, ConfigError, OSError) as error:
        print(f"not recorded: {error}", file=sys.stderr)
        return 1
    first = run.first
    print(
        f"run        {run.directory}\n"
        f"           {first['protocol']} · split {first['split']} · camera {first['cameras'][0]} · "
        f"{first['scored']:,} of {first['requested']:,} frames · commit {first['code']['commit'][:7]} "
        f"(clean) · weights sha256 {first['weights']['sha256'][:12]}…"
    )
    for text in weights.notes:
        print(f"weights    {text}")
    print("\n".join(f"record     {line}" for line in written))
    print(f"\nrows for {metafile.path}, model {name}:\n")
    if model is None:
        print(
            f"  - name: {name}\n    config: {run.config}\n    protocol: {first['protocol']}\n"
            f"    camera: {first['cameras'][0]}"
        )
    if model is None or model.weights is None:
        print(weights_yaml(run, provenance, weights.note, weights.entry), end="")
    if model is None:
        print("    results:")
    print(rows_yaml(run, provenance, stem))
    ok, sentence = metafile_state(run, model, provenance, stem)
    print(sentence, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
