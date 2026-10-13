"""Point Transformer V3 family: published configurations and the properties its design
promises. The shared contract is in ``test_backbone_contract``; agreement with the
official implementation is in ``tests/integration/test_backbone_oracle.py``."""

from __future__ import annotations

import copy
import pickle

import pytest
import torch

from grasppanda.models.backbones import ptv3
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointSet
from grasppanda.utils.imports import MissingDependencyError


def test_missing_spconv_names_the_extra_to_install(monkeypatch):
    def unavailable(module, *, extra, purpose):
        raise MissingDependencyError(
            f"{purpose} needs '{module}', which is not installed. "
            f'Install it with: pip install "grasppanda[{extra}]"'
        )

    monkeypatch.setattr(ptv3, "require", unavailable)
    with pytest.raises(MissingDependencyError, match=r"PTv3 needs .*grasppanda\[spconv-cu118\]"):
        ptv3.PTv3()


def test_patch_layout_completes_the_last_patch_with_the_points_before_it():
    # Three clouds of 5, 12 and 3 voxels, patches of 4.
    pad, unpad, runs = ptv3._patch_layout([5, 12, 3], 4, torch.device("cpu"))
    # Cloud 0: one full patch [0..3], then its last voxel 4 completed with 1, 2, 3.
    assert pad[:8].tolist() == [0, 1, 2, 3, 4, 1, 2, 3]
    # Cloud 1: three exact patches. Cloud 2 is smaller than a patch: one patch of 3.
    assert pad[8:20].tolist() == list(range(5, 17)) and pad[20:].tolist() == [17, 18, 19]
    assert runs == [(20, 4), (3, 3)]
    # Every voxel reads its result from the slot that holds it for the first time.
    assert pad[unpad].tolist() == list(range(20))
    assert unpad[:5].tolist() == [0, 1, 2, 3, 4] and unpad[5] == 8


spconv = pytest.importorskip("spconv.pytorch", reason="needs the spconv extra")

TINY = {
    "in_channels": 3,
    "grid_size": 0.02,
    "stride": [2, 2],
    "enc_depths": [1, 1, 2],
    "enc_channels": [12, 24, 48],
    "enc_num_head": [2, 4, 8],
    "enc_patch_size": [16, 16, 16],
    "drop_path": 0.0,
}
DECODER = {
    "dec_depths": [1, 1],
    "dec_channels": [12, 24],
    "dec_num_head": [2, 4],
    "dec_patch_size": [16, 16],
}


def build(kind: str, **config):
    torch.manual_seed(0)
    extra = DECODER if kind == "PTv3" else {}
    return BACKBONES.build({"type": kind, **TINY, **extra, **config}).eval()


def scene(count: int = 180, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Points spread so that every one has its own 2 cm voxel, with position-free features."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randperm(20**3, generator=generator)[:count]
    grid = torch.stack([cells // 400, cells // 20 % 20, cells % 20], dim=1)
    coords = (grid + 0.25 + 0.5 * torch.rand(count, 3, generator=generator)) * 0.02
    return coords, torch.rand(count, 3, generator=generator)


# ---------------------------------------------------------------- published settings


@pytest.mark.parametrize(
    ("kind", "parameters", "out_channels", "source"),
    [
        # 46.2 M in the PTv3 paper; exact count from the official model (fixture ptv3_scannet).
        ("PTv3", 46_166_272, 64, "official model, ScanNet configuration"),
        # Tensor elements of the released concerto_base.pth, the Sonata base encoder.
        ("PTv3Sonata", 108_461_376, 1232, "released checkpoint"),
        # Official model built from utonia.pth (its 72 rotary frequencies are buffers).
        ("PTv3Utonia", 137_253_744, 1386, "official model"),
    ],
)
def test_default_configuration_has_the_published_size(kind, parameters, out_channels, source):
    model = BACKBONES.build({"type": kind})
    assert sum(p.numel() for p in model.parameters()) == parameters, source
    assert model.out_channels == out_channels and model.output == "dense"


def test_state_dict_uses_the_official_names():
    legacy = build("PTv3").state_dict()
    assert legacy["embedding.stem.conv.weight"].shape == (12, 5, 5, 5, 3)
    assert "embedding.stem.norm.running_mean" in legacy  # batch norm in PT-v3m1
    assert legacy["enc.enc1.down.proj.weight"].shape == (24, 12)
    assert "enc.enc1.down.norm.0.running_var" in legacy
    assert legacy["enc.enc2.block1.cpe.0.weight"].shape == (48, 3, 3, 3, 48)
    assert legacy["enc.enc0.block0.attn.qkv.weight"].shape == (36, 12)
    assert legacy["enc.enc0.block0.mlp.0.fc1.weight"].shape == (48, 12)
    assert legacy["dec.dec0.up.proj_skip.0.weight"].shape == (12, 12)
    assert "dec.dec1.up.proj.1.running_mean" in legacy
    assert not any("ls1" in key or "mask_token" in key for key in legacy)

    sonata = build("PTv3Sonata", layer_scale=0.1).state_dict()
    assert sonata["embedding.stem.linear.weight"].shape == (12, 3)
    assert sonata["embedding.mask_token"].shape == (1, 12)
    assert sonata["enc.enc1.down.norm.0.weight"].shape == (24,)
    assert sonata["enc.enc0.block0.ls1.0.gamma"].shape == (12,)
    assert not any("running_mean" in key or key.startswith("dec.") for key in sonata)

    utonia = build("PTv3Utonia").state_dict()
    assert utonia["enc.enc0.block0.attn.rope.inv_freq"].tolist() == [1.0]  # head width 6
    rpe = build("PTv3", enable_rpe=True).state_dict()
    bound = int((4 * 16) ** (1 / 3) * 2)  # the official bound for patches of 16 voxels
    assert rpe["enc.enc0.block0.attn.rpe.rpe_table"].shape == (3 * (2 * bound + 1), 2)


# --------------------------------------------------------------------------- properties


@pytest.mark.parametrize("kind", ["PTv3", "PTv3Sonata", "PTv3Utonia"])
def test_result_does_not_depend_on_the_order_of_the_points(kind):
    model = build(kind)
    coords, feats = scene()
    order = torch.randperm(len(coords), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        straight = model(PointSet(coords), feats).feats
        shuffled = model(PointSet(coords[order]), feats[order]).feats
    assert torch.allclose(shuffled, straight[order], atol=2e-5)


@pytest.mark.parametrize("kind", ["PTv3", "PTv3Sonata"])
def test_voxel_grid_starts_at_each_clouds_own_corner(kind):
    """Moving a cloud by whole voxels changes nothing: the grid origin is per cloud."""
    model = build(kind)
    coords, feats = scene()
    with torch.no_grad():
        here = model(PointSet(coords), feats).feats
        there = model(PointSet(coords + torch.tensor([7.0, -3.0, 11.0]) * 0.02), feats).feats
    assert torch.allclose(there, here, atol=2e-5)


def test_rotary_embedding_reads_metric_coordinates():
    model = build("PTv3Utonia")
    with torch.no_grad():  # freshly initialised attention is almost uniform: sharpen it
        for name, parameter in model.named_parameters():
            if name.endswith("attn.qkv.weight"):
                parameter.mul_(40.0)
    coords, feats = scene()
    # The same voxels at twice the scale: only the coordinates the rotation reads differ.
    doubled = BACKBONES.build({"type": "PTv3Utonia", **TINY, "grid_size": 0.04}).eval()
    doubled.load_state_dict(model.state_dict())
    with torch.no_grad():
        here = model(PointSet(coords), feats).feats
        shifted = model(PointSet(coords + 0.4), feats).feats
        scaled = doubled(PointSet(coords * 2), feats).feats
    # A uniform shift cancels between queries and keys; a change of scale does not.
    assert torch.allclose(shifted, here, atol=2e-4)
    assert not torch.allclose(scaled, here, atol=1e-3)


def test_points_in_one_voxel_share_the_feature_of_its_first_point():
    model = build("PTv3")
    coords, feats = scene(120)
    twins = coords[:30] + 0.003  # inside the same 2 cm voxels as the first 30 points
    other = torch.rand(30, 3, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        alone = model(PointSet(coords), feats).feats
        both = model(PointSet(torch.cat([coords, twins])), torch.cat([feats, other])).feats
    assert torch.allclose(both[:120], alone, atol=2e-5)  # the later points change nothing
    assert torch.equal(both[120:], both[:30])


def test_concatenating_fewer_stages_keeps_the_deepest_channels():
    full = build("PTv3Sonata", concat_stages=3)
    assert full.out_channels == 12 + 24 + 48
    coords, feats = scene()
    with torch.no_grad():
        everything = full(PointSet(coords), feats).feats
        for stages, width in ((2, 24 + 48), (1, 48)):
            partial = build("PTv3Sonata", concat_stages=stages)
            partial.load_state_dict(full.state_dict())
            assert partial.out_channels == width
            assert torch.allclose(
                partial(PointSet(coords), feats).feats, everything[:, -width:], atol=1e-6
            )


def test_orders_are_shuffled_in_training_only():
    coords, feats = scene(400)
    model = build(
        "PTv3Sonata"
    ).train()  # layer norm only and no dropout: shuffling is the only randomness

    def run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        with torch.no_grad():
            return model(PointSet(coords), feats).feats

    outputs = [run(seed) for seed in range(4)]
    assert any(not torch.allclose(outputs[0], other, atol=1e-4) for other in outputs[1:])
    fixed = build("PTv3Sonata", shuffle_orders=False).train()
    with torch.no_grad():
        assert torch.equal(
            fixed(PointSet(coords), feats).feats, fixed(PointSet(coords), feats).feats
        )


def test_the_decoder_quirk_of_the_first_release_is_a_named_option():
    official = build("PTv3")
    fused = build("PTv3", fuse_before_decoder_cpe=True)
    fused.load_state_dict(official.state_dict())
    coords, feats = scene()
    with torch.no_grad():
        assert not torch.allclose(
            official(PointSet(coords), feats).feats, fused(PointSet(coords), feats).feats, atol=1e-4
        )


def test_model_can_be_copied_pickled_and_moved():
    model = build("PTv3Utonia")
    coords, feats = scene()
    with torch.no_grad():
        expected = model(PointSet(coords), feats).feats
        for clone in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
            assert torch.equal(clone(PointSet(coords), feats).feats, expected)


@pytest.mark.parametrize(
    ("kind", "change", "message"),
    [
        ("PTv3", {"in_channels": 0}, "at least one input feature channel"),
        ("PTv3", {"grid_size": 0.0}, "grid_size must be positive"),
        ("PTv3", {"order": []}, "at least one space-filling curve"),
        ("PTv3", {"enc_num_head": [2, 4]}, "one entry per stage"),
        ("PTv3", {"dec_depths": [1]}, "one per decoder stage"),
        ("PTv3", {"enc_num_head": [5, 4, 8]}, "12 channels cannot be split into 5 heads"),
        ("PTv3", {"stride": [3, 2]}, "power-of-two stride"),
        ("PTv3Sonata", {"concat_stages": 4}, r"concat_stages must be in 1\.\.3"),
        ("PTv3Utonia", {"enc_num_head": [3, 4, 8]}, "head width divisible by 6"),
    ],
)
def test_inconsistent_configurations_are_rejected(kind, change, message):
    extra = DECODER if kind == "PTv3" else {}
    with pytest.raises(ValueError, match=message):
        BACKBONES.build({"type": kind, **TINY, **extra, **change})


def test_ptv3_without_decoder_is_the_encoder_half_of_the_full_model():
    """`enc_mode` builds no decoder; the encoder keeps its names, so an encoder-only
    checkpoint loads strictly and a full one leaves only decoder tensors over."""
    full = build("PTv3")
    encoder = build("PTv3", enc_mode=True)
    assert not hasattr(encoder, "dec") and encoder.out_channels == TINY["enc_channels"][-1]
    state = full.state_dict()
    assert set(encoder.state_dict()) == {key for key in state if not key.startswith("dec.")}
    result = encoder.load_state_dict(state, strict=False)
    assert not result.missing_keys
    assert all(key.startswith("dec.") for key in result.unexpected_keys)
    coords, feats = scene()
    out = encoder.train()(PointSet(coords), feats)
    assert out.feats.shape == (len(coords), encoder.out_channels)
    out.feats.sum().backward()
    wide = build("PTv3", enc_mode=True, concat_stages=None)
    assert wide.out_channels == sum(TINY["enc_channels"])
