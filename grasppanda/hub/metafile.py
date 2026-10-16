"""Method metafiles: the machine-readable source of the model zoo.

Every ``configs/<method>/metafile.yaml`` is validated here. The rules encode the
project's reporting policy: a number never appears without its provenance and without
saying whether a collision filter was applied, and a number produced by this toolbox
must point at the evidence behind it: a result record that is committed next to the
metafile (``tools/record_result.py`` puts it there and prints the row).

A result states ``AP`` for the splits of its protocol (:data:`PROTOCOL_SPLITS`), in
percent, and may state ``AP0.8`` and ``AP0.4`` for the same splits::

    AP: {seen: 65.70, similar: 53.75, novel: 23.98}      # graspnet1b-sv-v1
    AP: {test: 21.67}                                    # graspclutter6d-sv-v1
    AP0.8: {test: 26.62}
    AP0.4: {test: 15.90}

A model's ``weights`` entry names the file its measured rows were produced with: its
``sha256`` always, and where it is published (``url``) under which ``license``. Weights
trained here (``source: grasppanda``) may be listed before they are published: ``url``
is then ``null``, ``license`` may be ``null``, and a short ``note`` says that they are
not published yet::

    weights:
      url: null
      sha256: <the 64 hex digits of the file>
      license: null
      source: grasppanda
      note: "Not published yet: ..."

The authors' checkpoint (``source: author``) is in the authors' format. ``convert`` is the
command that turns the downloaded file into weights for the config, exactly as the
method's README gives it; its two file arguments name the file the authors publish and
the converted file::

    weights:
      url: https://github.com/<owner>/<repo>/releases/download/v1/authors.tar
      sha256: <the 64 hex digits of the authors' file>
      license: "MIT; trained on GraspNet-1Billion (CC BY-NC-SA 4.0)"
      source: author
      convert: "python -m tools.convert_checkpoint authors.tar method_realsense.pth --source-format <format> --camera realsense"

Where ``weights`` names another file (weights trained here), ``author_weights`` lists the
checkpoint the authors released for the same config, in the same form, with a ``note``
on what it gives.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grasppanda.config import ConfigError, Issue, configs_root, parse_yaml

__all__ = [
    "PROTOCOL_SPLITS",
    "PROVENANCES",
    "Metafile",
    "ModelEntry",
    "Result",
    "Weights",
    "load_metafile",
    "load_metafiles",
]

#: ``reported``: taken from the paper or the authors' repository.
#: ``author-ckpt``: the authors' weights, evaluated by this toolbox.
#: ``reproduced``: trained and evaluated by this toolbox.
PROVENANCES = ("reported", "author-ckpt", "reproduced")
#: The splits a result of each pinned protocol is stated for. GraspNet-1Billion papers
#: print one AP per object-novelty split of the test scenes; GraspClutter6D has one test
#: split, for which AP, AP0.8 and AP0.4 are printed.
PROTOCOL_SPLITS = {
    "graspnet1b-sv-v1": ("seen", "similar", "novel"),
    "graspclutter6d-sv-v1": ("test",),
}
#: Average precision at one friction level, which a result may state next to ``AP``.
_AT_FRICTION = ("AP0.8", "AP0.4")
#: How a conversion command starts: the converter of the repository, run as a module.
CONVERTER = "python -m tools.convert_checkpoint"
#: Options of the converter; each takes a value.
_CONVERTER_OPTIONS = ("--source-format", "--config", "--camera")


def conversion_files(command: str) -> tuple[str, ...]:
    """The file arguments of a conversion command: ``(downloaded, converted)`` when well formed."""
    words = shlex.split(command)[len(CONVERTER.split()) :]
    files: list[str] = []
    while words:
        word = words.pop(0)
        if word in _CONVERTER_OPTIONS:
            if words:
                words.pop(0)
        elif word.startswith("--"):
            return ()  # an option the converter does not have
        else:
            files.append(word)
    return tuple(files)


@dataclass(frozen=True)
class Result:
    provenance: str
    #: ``True`` / ``False``, or ``None`` when a publication does not say.
    collision_filter: bool | None
    ap: dict[str, float]
    source: str = ""
    evaluator: str = ""
    record: str = ""
    inputs: str = ""
    seeds: int = 1
    #: ``{"AP0.8": {split: value}, "AP0.4": {...}}`` for the levels the row states.
    ap_at: dict[str, dict[str, float]] = field(default_factory=dict)

    def numbers(self) -> dict[str, dict[str, float]]:
        """Every number of the row: ``{"AP": {split: value}, "AP0.8": {...}, ...}``."""
        return {"AP": dict(self.ap), **{name: dict(values) for name, values in self.ap_at.items()}}


@dataclass(frozen=True)
class Weights:
    #: ``None`` only for weights trained here that are not published yet (see ``note``).
    url: str | None
    sha256: str
    #: ``None`` only while ``url`` is.
    license: str | None
    source: str  # "author" or "grasppanda"
    #: Why ``url`` is ``None``: a short sentence saying the weights are not published yet;
    #: for ``author_weights``, what the authors' file gives.
    note: str = ""
    #: The command that converts the authors' file (``source: author`` only), as the
    #: method's README gives it.
    convert: str = ""

    @property
    def published(self) -> bool:
        """Whether the file can be fetched from ``url``."""
        return self.url is not None

    @property
    def downloaded_name(self) -> str:
        """The authors' file as the conversion command names it; ``""`` without one."""
        return conversion_files(self.convert)[0] if self.convert else ""

    @property
    def converted_name(self) -> str:
        """The file the conversion command writes; ``""`` without one."""
        return conversion_files(self.convert)[1] if self.convert else ""


@dataclass(frozen=True)
class ModelEntry:
    name: str
    config: str
    protocol: str
    camera: str
    weights: Weights | None = None
    results: tuple[Result, ...] = ()
    #: The checkpoint the authors released for this config, where ``weights`` names
    #: another file (weights trained here).
    author_weights: Weights | None = None


@dataclass(frozen=True)
class Metafile:
    method: str
    title: str
    venue: str
    upstream: dict[str, str]
    implementation: str
    path: Path
    paper: str = ""
    models: tuple[ModelEntry, ...] = field(default_factory=tuple)


def _numbers(
    raw: Any, where: str, splits: tuple[str, ...], issues: list[Issue]
) -> dict[str, float]:
    """A ``{split: number}`` mapping of a result, or ``{}`` with an issue."""
    if not isinstance(raw, dict) or not raw or not set(raw) <= set(splits):
        issues.append(Issue(where, f"must map splits ({', '.join(splits)}) to numbers"))
        return {}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw.values()):
        issues.append(Issue(where, "values must be numbers"))
        return {}
    return {split: float(value) for split, value in raw.items()}


def _weights(given: Any, where: str, issues: list[Issue]) -> Weights | None:
    """The weights entry of a model, or ``None`` with an issue."""
    if not isinstance(given, dict):
        issues.append(Issue(where, "must be a mapping"))
        return None
    unpublished = given.get("source") == "grasppanda" and given.get("url") is None
    required = ("sha256", "source") if unpublished else ("url", "sha256", "license", "source")
    for key in required:
        if not given.get(key):
            issues.append(Issue(f"{where}.{key}", "is required"))
    if len(str(given.get("sha256", ""))) != 64:
        issues.append(Issue(f"{where}.sha256", "must be a 64-character hex digest"))
    note = str(given.get("note") or "")
    if unpublished and "not published" not in note.lower():
        issues.append(
            Issue(
                f"{where}.note",
                "weights trained here without a url need a short note saying that they are "
                "not published yet",
            )
        )
    convert = str(given.get("convert") or "")
    if convert and given.get("source") != "author":
        issues.append(
            Issue(
                f"{where}.convert",
                "only the authors' checkpoint is converted; weights trained here load as they are",
            )
        )
    elif convert and (
        not convert.startswith(CONVERTER + " ") or len(conversion_files(convert)) != 2
    ):
        issues.append(
            Issue(
                f"{where}.convert",
                f"must read '{CONVERTER} <downloaded file> <converted file> [--source-format "
                "F] [--config C] [--camera C]'",
            )
        )
    return Weights(
        url=None if unpublished else str(given.get("url", "")),
        sha256=str(given.get("sha256", "")),
        license=None if given.get("license") is None else str(given["license"]),
        source=str(given.get("source", "")),
        note=note,
        convert=convert,
    )


def _result(
    raw: Any,
    where: str,
    issues: list[Issue],
    has_weights: bool,
    splits: tuple[str, ...],
    directory: Path,
) -> Result | None:
    if not isinstance(raw, dict):
        issues.append(Issue(where, "a result must be a mapping"))
        return None
    provenance = raw.get("provenance")
    if provenance not in PROVENANCES:
        issues.append(
            Issue(
                f"{where}.provenance",
                f"must be one of {', '.join(PROVENANCES)}; got {provenance!r}",
            )
        )
    if "collision_filter" not in raw:
        issues.append(
            Issue(
                f"{where}.collision_filter",
                "is required: true, false, or null when the source does not say",
            )
        )
    collision = raw.get("collision_filter")
    if collision is not None and not isinstance(collision, bool):
        issues.append(Issue(f"{where}.collision_filter", "must be true, false or null"))
    ap = _numbers(raw.get("AP"), f"{where}.AP", splits, issues)
    ap_at = {}
    for name in _AT_FRICTION:
        if name not in raw:
            continue
        ap_at[name] = _numbers(raw[name], f"{where}.{name}", splits, issues)
        if ap and ap_at[name] and set(ap_at[name]) != set(ap):
            issues.append(Issue(f"{where}.{name}", "must be given for the same splits as AP"))
    if provenance == "reported" and not raw.get("source"):
        issues.append(Issue(f"{where}.source", "a reported number must say where it is reported"))
    if provenance in ("author-ckpt", "reproduced"):
        if collision is None:
            issues.append(
                Issue(
                    f"{where}.collision_filter",
                    "a number measured here cannot have an unknown collision filter",
                )
            )
        for key in ("evaluator", "record"):
            if not raw.get(key):
                issues.append(
                    Issue(
                        f"{where}.{key}",
                        f"is required for a '{provenance}' result: it is the evidence",
                    )
                )
        if raw.get("record") and not (directory / str(raw["record"])).is_file():
            issues.append(
                Issue(
                    f"{where}.record",
                    f"no such result record next to the metafile: {raw['record']}",
                )
            )
        if not has_weights:
            issues.append(
                Issue(
                    where,
                    f"a '{provenance}' result needs the model's weights entry: the sha256 "
                    "of the file it was measured with",
                )
            )
    return Result(
        provenance=str(provenance),
        collision_filter=collision,
        ap=ap,
        source=str(raw.get("source", "")),
        evaluator=str(raw.get("evaluator", "")),
        record=str(raw.get("record", "")),
        inputs=str(raw.get("inputs", "")),
        seeds=int(raw.get("seeds", 1)),
        ap_at=ap_at,
    )


def load_metafile(path: str | Path) -> Metafile:
    """Load and validate one metafile; raise :class:`ConfigError` listing every problem."""
    path = Path(path)
    raw = parse_yaml(path.read_text(encoding="utf-8")) or {}
    issues: list[Issue] = []
    for key in ("method", "title", "venue", "upstream", "implementation"):
        if not raw.get(key):
            issues.append(Issue(key, "is required"))
    upstream = raw.get("upstream") or {}
    if isinstance(upstream, dict):
        for key in ("repository", "license"):
            if not upstream.get(key):
                issues.append(
                    Issue(
                        f"upstream.{key}", "is required (write 'none' for a missing licence file)"
                    )
                )
    models = []
    for index, entry in enumerate(raw.get("models") or []):
        where = f"models.{index}"
        for key in ("name", "config", "protocol", "camera"):
            if not entry.get(key):
                issues.append(Issue(f"{where}.{key}", "is required"))
        splits = PROTOCOL_SPLITS.get(str(entry.get("protocol", "")))
        if entry.get("protocol") and splits is None:
            issues.append(
                Issue(
                    f"{where}.protocol",
                    f"unknown protocol {entry['protocol']!r}; known: {', '.join(PROTOCOL_SPLITS)}",
                )
            )
        splits = splits or tuple(
            dict.fromkeys(s for known in PROTOCOL_SPLITS.values() for s in known)
        )
        config = configs_root() / str(entry.get("config", ""))
        if entry.get("config") and not config.is_file():
            issues.append(Issue(f"{where}.config", f"no such shipped config: {entry['config']}"))
        weights = None
        if entry.get("weights"):
            weights = _weights(entry["weights"], f"{where}.weights", issues)
        author_weights = None
        if entry.get("author_weights"):
            author_weights = _weights(entry["author_weights"], f"{where}.author_weights", issues)
            if author_weights is not None and author_weights.source != "author":
                issues.append(
                    Issue(f"{where}.author_weights.source", "must be 'author': the authors' file")
                )
            if weights is None or weights.source == "author":
                issues.append(
                    Issue(
                        f"{where}.author_weights",
                        "is for a model whose weights are another file; the authors' "
                        "checkpoint itself is listed as its weights",
                    )
                )
        results = [
            result
            for r_index, r in enumerate(entry.get("results") or [])
            if (
                result := _result(
                    r,
                    f"{where}.results.{r_index}",
                    issues,
                    weights is not None,
                    splits,
                    path.parent,
                )
            )
        ]
        models.append(
            ModelEntry(
                name=str(entry.get("name", "")),
                config=str(entry.get("config", "")),
                protocol=str(entry.get("protocol", "")),
                camera=str(entry.get("camera", "")),
                weights=weights,
                results=tuple(results),
                author_weights=author_weights,
            )
        )
    if issues:
        raise ConfigError(issues, str(path))
    return Metafile(
        method=raw["method"],
        title=raw["title"],
        venue=raw["venue"],
        upstream={k: str(v) for k, v in upstream.items()},
        implementation=raw["implementation"],
        path=path,
        paper=str(raw.get("paper", "")),
        models=tuple(models),
    )


def load_metafiles(root: str | Path | None = None) -> list[Metafile]:
    """All metafiles below the shipped configs (or ``root``), sorted by method."""
    base = Path(root) if root else configs_root()
    return sorted(
        (load_metafile(p) for p in base.glob("*/metafile.yaml")), key=lambda m: m.method.lower()
    )
