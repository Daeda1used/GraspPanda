"""Method metafiles: the machine-readable source of the model zoo.

Every ``configs/<method>/metafile.yaml`` is validated here. The rules encode the
project's reporting policy: a number never appears without its provenance and without
saying whether a collision filter was applied, and a number produced by this toolbox
must point at the evidence behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grasppanda.config import ConfigError, Issue, configs_root, parse_yaml

__all__ = [
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
_SPLITS = ("seen", "similar", "novel")


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


@dataclass(frozen=True)
class Weights:
    url: str
    sha256: str
    license: str
    source: str  # "author" or "grasppanda"


@dataclass(frozen=True)
class ModelEntry:
    name: str
    config: str
    protocol: str
    camera: str
    weights: Weights | None = None
    results: tuple[Result, ...] = ()


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


def _result(raw: Any, where: str, issues: list[Issue], has_weights: bool) -> Result | None:
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
    ap = raw.get("AP")
    if not isinstance(ap, dict) or not ap or not set(ap) <= set(_SPLITS):
        issues.append(Issue(f"{where}.AP", f"must map splits ({', '.join(_SPLITS)}) to numbers"))
        ap = {}
    elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ap.values()):
        issues.append(Issue(f"{where}.AP", "values must be numbers"))
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
        if not has_weights:
            issues.append(
                Issue(
                    where,
                    f"a '{provenance}' result needs the model's weights entry (url and sha256)",
                )
            )
    return Result(
        provenance=str(provenance),
        collision_filter=collision,
        ap={k: float(v) for k, v in ap.items() if isinstance(v, (int, float))},
        source=str(raw.get("source", "")),
        evaluator=str(raw.get("evaluator", "")),
        record=str(raw.get("record", "")),
        inputs=str(raw.get("inputs", "")),
        seeds=int(raw.get("seeds", 1)),
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
        config = configs_root() / str(entry.get("config", ""))
        if entry.get("config") and not config.is_file():
            issues.append(Issue(f"{where}.config", f"no such shipped config: {entry['config']}"))
        weights = None
        if entry.get("weights"):
            given = entry["weights"]
            for key in ("url", "sha256", "license", "source"):
                if not given.get(key):
                    issues.append(Issue(f"{where}.weights.{key}", "is required"))
            if len(str(given.get("sha256", ""))) != 64:
                issues.append(Issue(f"{where}.weights.sha256", "must be a 64-character hex digest"))
            weights = Weights(
                *(str(given.get(k, "")) for k in ("url", "sha256", "license", "source"))
            )
        results = [
            result
            for r_index, r in enumerate(entry.get("results") or [])
            if (result := _result(r, f"{where}.results.{r_index}", issues, weights is not None))
        ]
        models.append(
            ModelEntry(
                name=str(entry.get("name", "")),
                config=str(entry.get("config", "")),
                protocol=str(entry.get("protocol", "")),
                camera=str(entry.get("camera", "")),
                weights=weights,
                results=tuple(results),
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
