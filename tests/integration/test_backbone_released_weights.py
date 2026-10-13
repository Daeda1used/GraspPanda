"""Released checkpoints load strictly into the encoder classes they are listed for.

Every entry of ``grasppanda.models.backbones._pretrained.RELEASED`` is built with the
options the table gives and its file; the load must be strict: every tensor of the class
comes from the file with the same shape, and the encoder part of the file holds no
tensor the class lacks. Then the presets that start a grasp model from such a file are
built as written, where exactly the tensors their headers name keep their
initialisation.

The files are not part of the repository and are never downloaded by a test: each case
skips unless its file lies under ``$GRASPPANDA_CACHE/weights_components`` (the error of
a missing file prints the command that fetches it). This proves key names and shapes,
and that the model runs; it does not compare activations with the original code.

Converted checkpoints of other tasks (point encoders)
-----------------------------------------------------
Released checkpoints of other tasks that fit a registered point encoder key for key.

The files are never downloaded by a test. Put them into
``$GRASPPANDA_WEIGHTS/components/released_backbones/`` under the names below (how to get
them: ``docs/slots/scene_encoders_point.md``); a missing file skips its case.

Each case converts the file with ``tools.convert_backbone_weights`` (which fails unless
every tensor of the encoder is present with the right shape), loads the result into the
backbone with ``strict=True`` and runs it on a synthetic cloud. That shows the weights
*load*; it does not compare activations with the upstream implementation, so the
classes keep their status.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from grasppanda.combine.presets import find
from grasppanda.models.backbones._pretrained import RELEASED, weights_directory
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointSet
from tools.convert_backbone_weights import FORMATS, convert_file

#: The cases of the sparse classes need spconv; the converted point-encoder cases below do not.
needs_spconv = pytest.mark.skipif(
    importlib.util.find_spec("spconv") is None, reason="needs the spconv extra"
)

#: Preset → the tensors that may keep their initialisation (prefixes of state keys).
PRESETS = {
    "spunet-gaussiancross": ("conv_input.0.weight", "final."),
    "ptv3-scannet200": ("embedding.stem.conv.weight",),
    "ptv3-ppt": ("embedding.stem.conv.weight",),
    "ptv3-sonata-small": ("embedding.stem.linear.weight",),
    "litept-structured3d": ("embedding.stem.conv.weight",),
    "minkunet34c-pointcontrast": ("final.",),
    "minkunet18a-openscene": ("final.",),
}


def available(name: str) -> None:
    path = weights_directory() / RELEASED[name].file
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine ({RELEASED[name].url})")


@needs_spconv
@pytest.mark.parametrize("name", sorted(RELEASED))
def test_a_release_loads_strictly_into_its_class(name):
    available(name)
    entry = RELEASED[name]
    model = BACKBONES.build({"type": entry.component, **entry.options, "pretrained": name})
    report = model.pretrained_report
    assert report.strict, report.summary()
    assert len(report.loaded) == len(model.state_dict())
    megabytes = (weights_directory() / entry.file).stat().st_size / 1e6
    assert abs(megabytes - entry.megabytes) < 1.0, megabytes


@needs_spconv
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_a_weight_preset_builds_and_runs_with_the_stated_tensors_left_out(preset):
    node = dict(find("backbone", preset).node)
    available(node["pretrained"])
    model = BACKBONES.build(node).eval()
    report = model.pretrained_report
    left = [entry.split()[0] for entry in (*report.missing, *report.mismatched)]
    assert left and all(key.startswith(PRESETS[preset]) for key in left), left
    assert len(report.loaded) == len(model.state_dict()) - len(left)
    points = PointSet.from_dense(
        torch.rand(1, 2000, 3, generator=torch.Generator().manual_seed(0)) * 0.4
    )
    with torch.no_grad():
        out = model(points, torch.ones(len(points), model.in_channels))
    assert out.feats.shape == (2000, model.out_channels) and torch.isfinite(out.feats).all()


#: (file name, format, tensors of the encoder, SHA-256 of the file as downloaded on 2026-10-13)
CASES = [
    (
        "pointnext-s_s3dis-area5_ckpt_best.pth",
        "pointnext-s",
        106,
        "95802f5c110e9fdd8baa65290654779234188358d3a9ab654d584f85d2bc8fb5",
    ),
    (
        "pretrained_votenet_on_sunrgbd.tar",
        "votenet",
        96,
        "ff645ce392676cc400c81cd872eaf40ae77192f61ecd7747c73c489f2c7ce281",
    ),
    (
        "pretrained_votenet_on_scannet.tar",
        "votenet",
        96,
        "22e57a2b8df1315ab4c770fac95990cc30a70680651ad2f4ccd97ed92b3fcd95",
    ),
    (
        "partition8_4096_15k_pointnet.pth",
        "csc-pointnet2",
        96,
        "a6812612720295ff2625537aa1c5c88c0ba9748de2e4b4f501c84e86730c482e",
    ),
    (
        "partition8_4096_15k_pointnet.pth",
        "csc-pointnet2-dense",
        120,
        "a6812612720295ff2625537aa1c5c88c0ba9748de2e4b4f501c84e86730c482e",
    ),
]


def _directory() -> Path:
    root = os.environ.get("GRASPPANDA_WEIGHTS", "")
    return Path(root) / "components" / "released_backbones"


@pytest.mark.parametrize(("file", "source_format", "tensors", "sha256"), CASES)
def test_released_checkpoint_loads_strictly(file, source_format, tensors, sha256):
    path = _directory() / file
    if not path.is_file():
        pytest.skip(f"{path} not found")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    converted, meta = convert_file(path, source_format)
    assert len(converted) == tensors == meta["tensors"]
    node = dict(FORMATS[source_format]["backbone"])
    if "npoints" not in node and node["type"].startswith("PointNet2"):
        node["npoints"] = [2048, 1024, 512, 256]
    model = BACKBONES.build(node).eval()
    model.load_state_dict(
        {key[len("backbone.") :]: value for key, value in converted.items()}, strict=True
    )
    generator = torch.Generator().manual_seed(0)
    coords = torch.rand(4096, 3, generator=generator) * torch.tensor([2.0, 2.0, 1.0])
    feats = torch.rand(4096, model.in_channels, generator=generator) if model.in_channels else None
    with torch.no_grad():
        out = model(PointSet(coords), feats)
    assert out.feats.shape[1] == model.out_channels and torch.isfinite(out.feats).all()
    assert float(out.feats.std()) > 0
