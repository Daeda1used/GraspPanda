"""Volt: layout of the released models, the tokeniser and its inverse, misuse.

The shared contract is in ``test_backbone_contract``; loading the released file is in
``tests/integration/test_encoder_checkpoints.py``."""

from __future__ import annotations

import pytest
import torch

from grasppanda.models.backbones.volt import Volt, convert_volt_checkpoint
from grasppanda.structures import PointSet

TINY = {
    "in_channels": 2,
    "up_mlp_dim": 6,
    "grid_size": 0.1,
    "kernel_size": 2,
    "embed_dim": 16,
    "depth": 1,
    "num_heads": 2,
    "drop_path": 0.0,
    "rope_split": [1, 2, 1],
}


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_defaults_are_volt_b_and_volt_s_has_the_released_size():
    model = Volt()
    assert (model.output, model.in_channels, model.out_channels) == ("dense", 6, 256)
    state = model.state_dict()
    assert state["tokenizer.proj.weight"].shape == (768, 125 * 6)
    assert state["decoder.unembed.proj.weight"].shape == (125 * 256, 256)
    assert state["decoder.unembed.bias"].shape == (256,)
    assert "blocks.0.attn.q_norm.weight" not in state  # qk_norm: false
    small = Volt(embed_dim=384, num_heads=6, qk_norm=True, up_mlp_dim=128)
    assert small.state_dict()["blocks.11.attn.q_norm.weight"].shape == (64,)
    # Every tensor of the released Volt-S ScanNet checkpoint has a place in this model
    # (212 tensors, checked on the file on 2026-10-13); this is the size they add up to.
    assert sum(p.numel() for p in small.parameters()) == 23_683_584
    assert Volt(out_channels=512).out_channels == 512


def test_one_token_per_occupied_block_and_every_voxel_reads_its_own_slot():
    model = Volt(**TINY).eval()
    # Two blocks of 2x2x2 voxels of 0.1: three voxels in the first, one in the second.
    coords = torch.tensor(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.03, 0.01],
            [0.11, 0.01, 0.01],
            [0.01, 0.11, 0.11],
            [0.31, 0.01, 0.01],
        ]
    )
    points = PointSet(coords)
    feats = torch.rand(5, 2)
    seen = {}
    model.tokenizer.register_forward_hook(lambda m, a, out: seen.update(tokens=out))
    with torch.no_grad():
        out = model(points, feats)
    tokens, blocks, inverse, slot = seen["tokens"]
    assert tokens.shape == (2, 16)
    assert blocks.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert inverse.tolist() == [0, 0, 0, 1] and slot.tolist() == [0, 3, 4, 4]
    assert out.feats.shape == (5, 6)
    assert torch.equal(out.feats[0], out.feats[1])  # two points of one voxel
    # The slot-by-slot read-out equals the reference formulation: every slot of every
    # token, then an index.
    unembed = model.decoder.unembed
    hidden = torch.rand(2, 6)
    reference = unembed.proj(hidden).view(-1, 8, 6)[inverse, slot] + unembed.bias
    assert torch.allclose(unembed(hidden, inverse, slot), reference, atol=1e-6)


def test_rotary_embedding_depends_on_block_offsets_only():
    model = Volt(**TINY).eval()
    blocks = torch.tensor([[0, 0, 0], [3, 1, 2], [5, 5, 5]])
    rotation = model._rotation(blocks)
    shifted = model._rotation(blocks + torch.tensor([7, 2, 4]))
    assert rotation.shape == (1, 3, 4)
    relative = rotation[0, 1] * rotation[0, 0].conj()
    assert torch.allclose(relative, shifted[0, 1] * shifted[0, 0].conj(), atol=1e-5)


def test_misuse_is_refused_with_the_way_out():
    with pytest.raises(ValueError, match=r"rope_split \[12, 12, 8\] rotates 64 channels"):
        Volt(embed_dim=384, num_heads=12)
    with pytest.raises(ValueError, match="not a multiple of 5 heads"):
        Volt(embed_dim=384, num_heads=5)


def test_converter_takes_the_backbone_of_a_pointcept_checkpoint():
    source, model = Volt(**TINY), Volt(**TINY)
    released = {f"module.backbone.{key}": value for key, value in source.state_dict().items()}
    released["module.seg_heads.0.weight"] = torch.zeros(20, 6)
    released["module.backbone.tokenizer.proj.weight"] = torch.zeros(16, 8 * 6)  # six channels
    state, report = convert_volt_checkpoint({"state_dict": released}, model)
    assert report["dropped"] == ["module.seg_heads.0.weight"]
    assert report["missing"] == ["tokenizer.proj.weight"]
    assert report["mismatched"] == [
        "module.backbone.tokenizer.proj.weight: file (16, 48), model (16, 16)"
    ]
    assert all(key.startswith("backbone.") for key in state)
    assert len(report["loaded"]) == len(source.state_dict()) - 1
