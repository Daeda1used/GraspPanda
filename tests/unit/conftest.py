"""Shared fixtures: a tiny set of registered components and a config writer."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
from torch import nn

from grasppanda.models import Backbone, BaseGraspDetector, HeadBundle
from grasppanda.registry import BACKBONES, DETECTORS, HEADS, INJECTED, Cfg

REPO = Path(__file__).resolve().parents[2]


@BACKBONES.register()
class TBackbone(Backbone):
    output = "dense"

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        norm: Literal["bn", "ln"] = "bn",
        dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.layer = nn.Linear(in_channels, out_channels)


@HEADS.register()
class THead(HeadBundle):
    in_axes = ("seed", "channel")

    def __init__(
        self, num_angle: int, in_channels: int = INJECTED, loss_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.num_angle, self.in_channels = num_angle, in_channels


@DETECTORS.register()
class TDetector(BaseGraspDetector):
    def __init__(self, backbone: Cfg[Backbone], head: Cfg[HeadBundle], seeds: int = 8) -> None:
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.head = HEADS.build(head, in_channels=self.backbone.out_channels)

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        if node.get("seeds", 8) > 4096:
            yield "seeds must be at most 4096."


MODEL = {
    "type": "TDetector",
    "backbone": {"type": "TBackbone", "out_channels": 16},
    "head": {"type": "THead", "num_angle": 12},
}


@pytest.fixture
def write(tmp_path: Path):
    """Write dedented YAML files below a temporary directory and return their paths."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path

    return _write
