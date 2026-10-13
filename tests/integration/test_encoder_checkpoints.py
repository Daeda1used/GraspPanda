"""Released checkpoints fill ``PointViT``, ``PointMamba`` and ``Volt`` completely.

Skipped when a file is absent. The files are looked up under
``$GRASPPANDA_WEIGHTS/components/``:

``point_tokens/point-mae_pretrain.pth``
    https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth
``point_tokens/maskpoint_pretrain_shapenet.pth``
    https://static.hliu.cc/files/projects/maskpoint/checkpoints/pretrain_shapenet.pth
``point_tokens/pointmamba_pretrain.pth``
    https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/pretrain.pth
``point_tokens/uni3d-ti.pt``, ``point_tokens/uni3d-s.pt``
    https://huggingface.co/BAAI/Uni3D/resolve/main/modelzoo/uni3d-ti/model.pt (and ``uni3d-s``)
``volt/volt-small-scannet.pth``
    https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_small/scannet/model/model_last.pth

The first three were pre-trained on ShapeNet objects, Uni3D on object collections with
colour, Volt on indoor rooms. What is
checked is that every tensor of the encoder is found in the file with its shape and
that the encoder then runs; no activation is compared with the upstream code.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from grasppanda.models.backbones.point_tokens import convert_point_token_checkpoint
from grasppanda.models.backbones.volt import convert_volt_checkpoint
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointSet

TOKENS = {
    "point-mae_pretrain.pth": {"type": "PointViT"},
    "maskpoint_pretrain_shapenet.pth": {"type": "PointViT", "token_dim": 256},
    "pointmamba_pretrain.pth": {"type": "PointMamba"},
    "uni3d-ti.pt": {"type": "Uni3D", "timm_model": "eva02_tiny_patch14_224"},
    "uni3d-s.pt": {"type": "Uni3D", "timm_model": "eva02_small_patch14_224"},
}


def released(relative: str) -> Path:
    root = os.environ.get("GRASPPANDA_WEIGHTS", "")
    path = Path(root) / "components" / relative
    if not root or not path.is_file():
        pytest.skip(f"{relative} not found under $GRASPPANDA_WEIGHTS/components")
    return path


def fill(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load strictly and check that nearly every tensor changed."""
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    after = model.state_dict()
    changed = [key for key in after if not torch.equal(after[key], before[key])]
    assert len(changed) > 0.9 * len(before)  # all but counters and freshly initialised ones


@pytest.mark.parametrize("name", list(TOKENS))
def test_point_token_checkpoint_fills_the_encoder(name):
    path = released(f"point_tokens/{name}")
    node = {**TOKENS[name], "num_group": 64}
    if node["type"] == "PointMamba":
        pytest.importorskip("mamba_ssm", reason="needs the mamba-ssm extra")
        if not torch.cuda.is_available():
            pytest.skip("the selective scan of mamba-ssm is a CUDA kernel")
    if node["type"] == "Uni3D":
        pytest.importorskip("timm", reason="needs the timm extra")
    model = BACKBONES.build(node)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state, report = convert_point_token_checkpoint(checkpoint, model, prefix="")
    assert report["missing"] == [] and report["mismatched"] == []
    kept = ("encoder.", "blocks.", "pos_embed.", "point_encoder.visual.blocks.")
    assert all(not key.startswith(kept) for key in report["dropped"])
    fill(model, state)

    device = "cuda" if node["type"] == "PointMamba" else "cpu"
    model = model.to(device).eval()
    sphere = torch.nn.functional.normalize(torch.randn(1024, 3), dim=1) * torch.rand(1024, 1)
    colour = torch.rand(1024, 3) if model.in_channels else None
    with torch.no_grad():
        out = model(PointSet(sphere.to(device)), None if colour is None else colour.to(device))
    assert out.feats.shape == (64, model.out_channels) and torch.isfinite(out.feats).all()


def test_volt_small_checkpoint_fills_the_encoder():
    path = released("volt/volt-small-scannet.pth")
    model = BACKBONES.build(
        {"type": "Volt", "embed_dim": 384, "num_heads": 6, "qk_norm": True, "up_mlp_dim": 128}
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state, report = convert_volt_checkpoint(checkpoint, model, prefix="")
    assert report["missing"] == [] and report["mismatched"] == []
    assert all(".backbone." not in key for key in report["dropped"])  # heads only
    fill(model, state)

    model.eval()
    room = torch.rand(4000, 3) * torch.tensor([3.0, 3.0, 0.3])  # a floor slab, 2 cm voxels
    with torch.no_grad():
        out = model(PointSet(room), torch.rand(4000, 6))
    assert out.feats.shape == (4000, 128) and torch.isfinite(out.feats).all()
