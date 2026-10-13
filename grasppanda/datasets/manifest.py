"""Split manifests: which ``(scene, camera, frame)`` units a split consists of.

A benchmark split is a product of scenes and frames, so a manifest states ranges rather
than listing tens of thousands of entries. Manifests of the supported datasets ship as
package data under ``grasppanda/datasets/splits/``; a dataset whose authoritative split
files live in its own tree builds a manifest from those with :meth:`SplitManifest.build`.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

from grasppanda.structures import SampleKey

__all__ = ["Split", "SplitManifest", "expand_ranges"]


def expand_ranges(spec: Any) -> tuple[int, ...]:
    """Expand ``7``, ``"0-99"``, ``"0-255:32"`` or a list of those into sorted ids.

    Ranges are inclusive at both ends.
    """
    if isinstance(spec, bool):
        raise ValueError(f"cannot read {spec!r} as an id or a range")
    if isinstance(spec, int):
        return (spec,)
    if isinstance(spec, str):
        body, _, step_text = spec.partition(":")
        first, separator, last = body.partition("-")
        try:
            start = int(first)
            stop = int(last) if separator else start
            step = int(step_text) if step_text else 1
        except ValueError:
            raise ValueError(
                f"cannot read {spec!r} as a range; write 'first-last' or 'first-last:step'"
            ) from None
        if stop < start or step < 1:
            raise ValueError(f"empty or reversed range {spec!r}")
        return tuple(range(start, stop + 1, step))
    if isinstance(spec, Sequence):
        ids = sorted(value for item in spec for value in expand_ranges(item))
        if len(set(ids)) != len(ids):
            raise ValueError(f"ids are listed more than once in {spec!r}")
        return tuple(ids)
    raise ValueError(f"cannot read {spec!r} as an id or a range")


@dataclass(frozen=True)
class Split:
    """Scenes and frame indices of one split."""

    name: str
    scenes: tuple[int, ...]
    frames: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.scenes) * len(self.frames)


class SplitManifest:
    """The splits of one dataset.

    Args:
        dataset: Dataset name used in :class:`~grasppanda.structures.SampleKey`.
        cameras: Camera names the dataset provides.
        frames: Frame ids every scene has (per camera).
        splits: The splits by name.
    """

    def __init__(
        self,
        dataset: str,
        cameras: Sequence[str],
        frames: Iterable[int],
        splits: Mapping[str, Split],
    ) -> None:
        self.dataset = dataset
        self.cameras = tuple(cameras)
        self.frames = tuple(frames)
        self.splits = dict(splits)

    @classmethod
    def build(
        cls,
        dataset: str,
        cameras: Sequence[str],
        frames: Iterable[int],
        scenes: Mapping[str, Iterable[int]],
    ) -> SplitManifest:
        """Create a manifest in which every split has the same frames per scene."""
        frames = tuple(frames)
        splits = {name: Split(name, tuple(sorted(ids)), frames) for name, ids in scenes.items()}
        return cls(dataset, cameras, frames, splits)

    @classmethod
    def load(cls, dataset: str) -> SplitManifest:
        """Load the manifest shipped with the package for ``dataset``."""
        file = resources.files("grasppanda.datasets") / "splits" / f"{dataset}.yaml"
        if not file.is_file():
            raise FileNotFoundError(f"no split manifest is shipped for dataset '{dataset}'")
        return cls.from_mapping(yaml.safe_load(file.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SplitManifest:
        every_frame = tuple(range(int(data["frames_per_scene"])))
        splits = {}
        for name, entry in data["splits"].items():
            unknown = set(entry) - {"scenes", "exclude", "frames"}
            if unknown:
                raise ValueError(f"split '{name}': unknown keys {sorted(unknown)}")
            excluded = set(expand_ranges(entry.get("exclude", [])))
            scenes = tuple(s for s in expand_ranges(entry["scenes"]) if s not in excluded)
            frames = expand_ranges(entry["frames"]) if "frames" in entry else every_frame
            if not set(frames) <= set(every_frame):
                raise ValueError(f"split '{name}': frames outside 0..{len(every_frame) - 1}")
            splits[name] = Split(name, scenes, frames)
        return cls(str(data["dataset"]), list(data["cameras"]), every_frame, splits)

    def all_scenes(self) -> tuple[int, ...]:
        """Every scene that belongs to at least one split."""
        return tuple(sorted({scene for split in self.splits.values() for scene in split.scenes}))

    def split(self, name: str) -> Split:
        """Return a split or explain which names exist."""
        try:
            return self.splits[name]
        except KeyError:
            close = difflib.get_close_matches(name, list(self.splits), n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(
                f"'{name}' is not a split of {self.dataset}.{hint} "
                f"Splits: {', '.join(self.splits)}."
            ) from None

    def check_camera(self, camera: str) -> None:
        if camera not in self.cameras:
            raise ValueError(
                f"'{camera}' is not a camera of {self.dataset}. Cameras: {', '.join(self.cameras)}."
            )

    def keys(self, split: str, camera: str, frame_step: int = 1) -> list[SampleKey]:
        """Every unit of a split for one camera, scene by scene.

        Args:
            split: Split name.
            camera: Camera name.
            frame_step: Keep every ``frame_step``-th frame of the split, for fast
                validation.
        """
        self.check_camera(camera)
        if frame_step < 1:
            raise ValueError(f"frame_step must be at least 1, got {frame_step}")
        chosen = self.split(split)
        return [
            SampleKey(self.dataset, scene, camera, frame)
            for scene in chosen.scenes
            for frame in chosen.frames[::frame_step]
        ]

    def digest(self, split: str) -> str:
        """Hash of a split's content, for run records."""
        chosen = self.split(split)
        text = json.dumps([self.dataset, chosen.scenes, chosen.frames], separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:16]
