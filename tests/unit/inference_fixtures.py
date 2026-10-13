"""Configs, weights and a second detector for the inference tests; nothing here needs data.

Two models are used. ``ToyDetector`` of the example project reads points, like the
point-cloud methods. ``ImageToyDetector``, registered here, declares that it also reads
the colour and depth images and the camera matrix, like HGGD, and draws a random number
in ``predict`` the way HGGD jitters its regions.

The pipelines are real ones: the config nodes below are those of a GraspNet-1Billion
test section (with fewer points), and only their transforms are built, never the
dataset, so no file of the dataset is touched.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from grasppanda.config import Config
from grasppanda.models import BaseGraspDetector
from grasppanda.registry import DETECTORS
from grasppanda.structures import GraspGroup, Observation

from .conftest import REPO

PROJECT = REPO / "projects/example_project"
TOY_CONFIG = str(PROJECT / "configs/toy_train.yaml")

_HEAD = f"""
custom_imports:
  - {PROJECT / "toy_components.py"}
  - {PROJECT / "toy_data.py"}
runtime: {{device: cpu}}
"""
_TOY_MODEL = """
model:
  type: ToyDetector
  num_seeds: 32
  backbone: {type: ToyBackbone, out_channels: 16}
  grouper: {type: ToyGrouper, radius: 0.05, nsample: 8}
  head: {type: ToyHead, num_angle: 12, num_depth: 4}
"""

#: A point-cloud method on the GraspNet-1Billion recipe: ground-truth workspace, sampling,
#: constant features, and a collision-filter variant.
POINT_CONFIG = (
    _HEAD
    + _TOY_MODEL
    + """
data:
  test:
    dataset:
      type: GraspNet1B
      camera: realsense
      split: test
      load_labels: false
      preprocess:
        - {type: DepthToCloud}
        - {type: SegmentationBoxWorkspace, margin: 0.02, invalid_depth_foreground: include}
      transforms:
        - {type: RandomSample, num_points: 1024}
        - {type: ConstantFeatures, channels: 3}
      seed: 0
    batch_size: 1
    num_workers: 0
evaluation:
  variants:
    raw: []
    cd: [{type: ModelFreeCollisionFilter}]
    centre: [{type: ToyCentreFilter, radius: 0.15}]
"""
)

#: An image method on the HGGD recipe: no workspace, images served beside the cloud.
IMAGE_CONFIG = (
    _HEAD
    + """
model: {type: ImageToyDetector, num_grasps: 6}
data:
  test:
    dataset:
      type: GraspNet1B
      camera: realsense
      split: test
      load_labels: false
      images: [rgb, depth]
      preprocess:
        - {type: DepthToCloud}
        - {type: NoWorkspace}
      transforms:
        - {type: RandomSample, num_points: 512}
      keep_full_points: false
      seed: 0
    batch_size: 1
    num_workers: 0
"""
)

#: A config without any data section: it declares no pipeline at all.
MODEL_ONLY_CONFIG = _HEAD + _TOY_MODEL


@DETECTORS.register()
class ImageToyDetector(BaseGraspDetector):
    """Grasps at the first points of the cloud, scored by image content plus a random draw."""

    required_inputs = frozenset({"rgb", "depth", "intrinsics", "points"})

    def __init__(self, num_grasps: int = 6) -> None:
        super().__init__()
        self.num_grasps = num_grasps
        self.gain = nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def predict(self, observation: Observation) -> list[GraspGroup]:
        points = observation.points.coords[: self.num_grasps]
        brightness = observation.rgb.mean() + observation.depth.mean()
        noise = torch.rand(len(points), device=points.device)  # as HGGD jitters its regions
        return [
            GraspGroup(
                translation=points,
                rotation=torch.eye(3, device=points.device).expand(len(points), 3, 3).clone(),
                width=torch.full((len(points),), 0.05, device=points.device),
                depth=torch.full((len(points),), 0.02, device=points.device),
                score=self.gain * brightness + noise,
                key=observation.keys[0],
            )
        ]


def write_config(directory: Path, text: str, name: str = "config.yaml") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def save_weights(config: str | Path, path: Path) -> Path:
    """Weights of the config's freshly initialised model, stored the way a converter stores them."""
    from grasppanda.engine.builders import build_model

    torch.manual_seed(1)
    model = build_model(Config.load(config).resolved())
    torch.save({"state_dict": model.state_dict()}, path)
    return path
