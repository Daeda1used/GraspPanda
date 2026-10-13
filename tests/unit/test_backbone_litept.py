"""LitePT: published configurations, PointROPE against its definition, and the properties
the design promises. The shared contract is in ``test_backbone_contract``; agreement with
the official implementation is in ``tests/integration/test_backbone_litept_oracle.py``."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import torch

from grasppanda.models.backbones import litept
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointSet
from grasppanda.utils.imports import MissingDependencyError


def test_missing_spconv_names_the_extra_to_install(monkeypatch):
    def unavailable(module, *, extra, purpose):
        raise MissingDependencyError(
            f"{purpose} needs '{module}', which is not installed. "
            f'Install it with: pip install "grasppanda[{extra}]"'
        )

    monkeypatch.setattr(litept, "require", unavailable)
    with pytest.raises(MissingDependencyError, match=r"LitePT needs .*grasppanda\[spconv-cu118\]"):
        litept.LitePT()


def test_component_metadata():
    cls = BACKBONES.get("LitePT")
    assert cls is litept.LitePT and cls.output == "dense"
    assert cls.status == "weights-verified"  # tests/integration/test_backbone_litept_oracle.py
    assert cls.source == "LitePT (Yue et al., CVPR 2026)" and cls.extras == ("spconv",)


# ------------------------------------------------------------------------- PointROPE


def rotated_pair_by_pair(tokens: np.ndarray, position: np.ndarray, base: float) -> np.ndarray:
    """PointROPE as the paper defines it, one pair of channels at a time (float64).

    A head's channels are ``[u_x, v_x, u_y, v_y, u_z, v_z]``, each of ``pairs`` channels.
    Pair ``i`` of an axis is the point ``(u[i], v[i])`` of a plane, turned by the angle
    ``coordinate * base ** (-i / pairs)``.
    """
    pairs = tokens.shape[-1] // 6
    out = np.empty_like(tokens)
    for axis in range(3):
        for i in range(pairs):
            angle = position[:, axis, None] * base ** (-i / pairs)
            u, v = axis * 2 * pairs + i, axis * 2 * pairs + pairs + i
            out[..., u] = tokens[..., u] * np.cos(angle) - tokens[..., v] * np.sin(angle)
            out[..., v] = tokens[..., v] * np.cos(angle) + tokens[..., u] * np.sin(angle)
    return out


@pytest.mark.parametrize(("width", "base"), [(6, 100.0), (18, 100.0), (24, 10.0)])
def test_point_rope_equals_its_definition_written_out(width, base):
    generator = torch.Generator().manual_seed(width)
    tokens = torch.randn(50, 3, width, generator=generator, dtype=torch.float64)
    position = torch.randint(0, 40, (50, 3), generator=generator)
    expected = rotated_pair_by_pair(tokens.numpy(), position.numpy().astype(np.float64), base)
    # The angles are float32 by design (the official table is), hence the tolerance.
    assert np.abs(litept._point_rope(tokens, position, base).numpy() - expected).max() < 2e-5
    single = litept._point_rope(tokens.float(), position, base)
    assert single.dtype == torch.float32
    assert np.abs(single.numpy() - expected).max() < 2e-5


def test_point_rope_scores_depend_on_voxel_offsets_only():
    generator = torch.Generator().manual_seed(0)
    q, k = torch.randn(2, 30, 2, 18, generator=generator, dtype=torch.float64)
    position = torch.randint(0, 20, (30, 3), generator=generator)
    shift = torch.tensor([5, 11, 2])

    def scores(offset: torch.Tensor) -> torch.Tensor:
        rotated_q = litept._point_rope(q, position + offset, 100.0)
        rotated_k = litept._point_rope(k, position + offset, 100.0)
        return torch.einsum("ihd,jhd->hij", rotated_q, rotated_k)

    here = scores(torch.zeros(3, dtype=torch.long))
    assert torch.allclose(scores(shift), here, atol=1e-4)
    plain = torch.einsum("ihd,jhd->hij", q, k)
    assert not torch.allclose(here, plain, atol=1e-2)  # ...and the rotation does something
    # A voxel attends to itself as without any position.
    assert torch.allclose(here.diagonal(dim1=1, dim2=2), plain.diagonal(dim1=1, dim2=2), atol=1e-4)


def test_half_precision_tokens_are_rotated_in_float32():
    """Grid coordinates of a few thousand are beyond what float16 angles could carry."""
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randn(40, 2, 12, generator=generator).half()
    position = torch.randint(1000, 3000, (40, 3), generator=generator)
    rotated = litept._point_rope(tokens, position, 100.0)
    assert rotated.dtype == torch.float16
    exact = litept._point_rope(tokens.float(), position, 100.0)
    assert torch.allclose(rotated.float(), exact, atol=4e-3)


# -------------------------------------------------------------- the network (spconv)

spconv = pytest.importorskip("spconv.pytorch", reason="needs the spconv extra")

TINY = {
    "in_channels": 3,
    "grid_size": 0.02,
    "stride": [2, 2],
    "enc_depths": [1, 1, 2],
    "enc_channels": [12, 24, 48],
    "enc_num_head": [2, 4, 8],
    "enc_patch_size": [16, 16, 16],
    "enc_conv": [True, True, False],
    "enc_attn": [False, True, True],
    "enc_rope_freq": [100.0, 100.0, 100.0],
    "dec_depths": [1, 1],
    "dec_channels": [12, 24],
    "dec_num_head": [2, 4],
    "dec_patch_size": [16, 16],
    "dec_conv": [True, False],
    "dec_attn": [False, True],
    "dec_rope_freq": [100.0, 100.0],
    "drop_path": 0.0,
}

#: Published models other than the default, as changes to it (official configs
#: ``insseg-litept-small-v1m2``, ``semseg-litept-base-v1m1``, ``semseg-litept-large-v1m1``).
SMALL_WITH_DECODER = {
    "dec_depths": [2, 2, 2, 2],
    "dec_conv": [True, True, True, False],
    "dec_attn": [False, False, False, True],
}
BASE = {
    "enc_depths": [3, 3, 3, 12, 3],
    "enc_channels": [54, 108, 216, 432, 576],
    "enc_num_head": [3, 6, 12, 24, 32],
    "dec_channels": [72, 108, 216, 432],
    "dec_num_head": [4, 6, 12, 24],
}
LARGE = {
    "enc_depths": [3, 3, 3, 12, 3],
    "enc_channels": [72, 144, 288, 576, 864],
    "enc_num_head": [4, 8, 16, 32, 48],
    "dec_channels": [72, 144, 288, 576],
    "dec_num_head": [4, 8, 16, 32],
}


def build(**config):
    torch.manual_seed(0)
    return BACKBONES.build({"type": "LitePT", **TINY, **config}).eval()


def sharpen(model) -> None:
    """Freshly initialised attention is almost uniform: make it depend on its logits."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("attn.qkv.weight"):
                parameter.mul_(40.0)


def scene(count: int = 180, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Points spread so that every one has its own 2 cm voxel, with position-free features."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randperm(20**3, generator=generator)[:count]
    grid = torch.stack([cells // 400, cells // 20 % 20, cells % 20], dim=1)
    coords = (grid + 0.25 + 0.5 * torch.rand(count, 3, generator=generator)) * 0.02
    return coords, torch.rand(count, 3, generator=generator)


# ---------------------------------------------------------------- published settings


@pytest.mark.parametrize(
    ("change", "parameters", "paper"),
    [
        # Exact counts are those of the official model built from the published config,
        # recorded with the fixtures of the oracle test; the paper rounds them.
        ({}, 12_712_932, "LitePT-S, 12.7 M"),
        (SMALL_WITH_DECODER, 15_988_716, "LitePT-S*, 16.0 M"),
        (BASE, 45_112_788, "LitePT-B, 45.1 M"),
        (LARGE, 85_880_448, "LitePT-L, 85.9 M"),
    ],
)
def test_published_configurations_have_the_published_size(change, parameters, paper):
    model = BACKBONES.build({"type": "LitePT", **change})
    assert sum(p.numel() for p in model.parameters()) == parameters, paper
    assert model.out_channels == 72 and model.in_channels == 6 and model.output == "dense"


def test_state_dict_uses_the_official_names():
    state = build().state_dict()
    assert state["embedding.stem.conv.weight"].shape == (12, 5, 5, 5, 3)
    assert "embedding.stem.norm.running_mean" in state
    # A stage that only convolves: convolution, linear layer, layer norm.
    assert state["enc.enc0.block0.conv.0.weight"].shape == (12, 3, 3, 3, 12)
    assert state["enc.enc0.block0.conv.1.weight"].shape == (12, 12)
    assert state["enc.enc0.block0.conv.2.weight"].shape == (12,)
    assert not any(key.startswith("enc.enc0.block0.") and "conv" not in key for key in state)
    # A stage that convolves and attends, and one that only attends (a norm in place of
    # the convolution).
    assert "enc.enc1.block0.conv.0.bias" in state and "enc.enc1.block0.norm1.0.weight" in state
    assert state["enc.enc2.block1.norm0.0.weight"].shape == (48,)
    assert state["enc.enc2.block1.attn.qkv.weight"].shape == (144, 48)
    assert state["enc.enc2.block1.mlp.0.fc1.weight"].shape == (192, 48)
    assert not any(key.startswith("enc.enc2.block1.conv") for key in state)
    assert state["enc.enc1.down.proj.weight"].shape == (24, 12)
    assert "enc.enc1.down.norm.0.running_var" in state
    assert state["dec.dec0.up.proj_skip.0.weight"].shape == (12, 12)
    assert "dec.dec1.up.proj.1.running_mean" in state
    assert "dec.dec1.block0.attn.proj.bias" in state and "dec.dec0.block0.conv.0.weight" in state
    # PointROPE has no parameters and no buffers.
    assert not any("rope" in key for key in state)
    # The published decoder is the unpooling alone.
    default = BACKBONES.build({"type": "LitePT"}).state_dict()
    assert not any(".block" in key for key in default if key.startswith("dec."))


# --------------------------------------------------------------------------- properties


def test_result_does_not_depend_on_the_order_of_the_points():
    model = build()
    coords, feats = scene()
    order = torch.randperm(len(coords), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        straight = model(PointSet(coords), feats).feats
        shuffled = model(PointSet(coords[order]), feats[order]).feats
    assert torch.allclose(shuffled, straight[order], atol=2e-5)


def test_positions_are_voxel_offsets_whatever_the_origin_and_the_scale():
    """PointROPE reads integer grid coordinates: moving the cloud, or scaling it together
    with the grid, changes nothing, while the base of the rotation does."""
    # Heads of width 12: two frequencies per axis, 1 and base ** -0.5. (A head of width 6
    # has the single frequency 1, whatever the base.)
    wide = {"enc_num_head": [2, 2, 4], "dec_num_head": [2, 2]}
    model = build(**wide)
    sharpen(model)
    coords, feats = scene()
    doubled = build(**wide, grid_size=0.04)
    doubled.load_state_dict(model.state_dict())
    slower = build(**wide, enc_rope_freq=[100.0, 5.0, 5.0], dec_rope_freq=[100.0, 5.0])
    slower.load_state_dict(model.state_dict())
    with torch.no_grad():
        here = model(PointSet(coords), feats).feats
        moved = model(PointSet(coords + torch.tensor([7.0, -3.0, 11.0]) * 0.02), feats).feats
        scaled = doubled(PointSet(coords * 2), feats).feats
        other_base = slower(PointSet(coords), feats).feats
    assert torch.allclose(moved, here, atol=2e-4)
    assert torch.allclose(scaled, here, atol=2e-4)
    assert not torch.allclose(other_base, here, atol=1e-3)


def test_points_in_one_voxel_share_the_feature_of_its_first_point():
    model = build()
    coords, feats = scene(120)
    twins = coords[:30] + 0.003  # inside the same 2 cm voxels as the first 30 points
    other = torch.rand(30, 3, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        alone = model(PointSet(coords), feats).feats
        both = model(PointSet(torch.cat([coords, twins])), torch.cat([feats, other])).feats
    assert torch.allclose(both[:120], alone, atol=2e-5)  # the later points change nothing
    assert torch.equal(both[120:], both[:30])


def test_only_stages_that_attend_are_serialised(monkeypatch):
    """The fine stages of the published models convolve only and never pay for a curve."""
    calls = []

    def counting(grid, batch, depths, orders):
        calls.append(len(grid))
        return original(grid, batch, depths, orders)

    original = litept._codes
    monkeypatch.setattr(litept, "_codes", counting)
    coords, feats = scene()
    last_only = build(enc_attn=[False, False, True], dec_attn=[False, False])
    with torch.no_grad():
        last_only(PointSet(coords), feats)
    assert len(calls) == 1 and calls[0] < len(coords)  # the coarsest level only
    calls.clear()
    with torch.no_grad():
        build()(PointSet(coords), feats)  # the decoder reuses the order of its level
    assert len(calls) == 2


def test_orders_are_shuffled_in_training_only():
    coords, feats = scene(400)
    # No dropout and no stochastic depth: shuffling is the only randomness. (Batch norm
    # uses the statistics of the batch in training mode, which are the same every time.)
    model = build(enc_depths=[1, 1, 4]).train()

    def run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        with torch.no_grad():
            return model(PointSet(coords), feats).feats

    outputs = [run(seed) for seed in range(4)]
    assert any(not torch.allclose(outputs[0], other, atol=1e-4) for other in outputs[1:])
    fixed = build(enc_depths=[1, 1, 4], shuffle_orders=False).train()
    with torch.no_grad():
        assert torch.equal(
            fixed(PointSet(coords), feats).feats, fixed(PointSet(coords), feats).feats
        )
    model.eval()
    assert torch.equal(run(0), run(1))


def test_model_can_be_copied_pickled_and_moved():
    model = build()
    coords, feats = scene()
    with torch.no_grad():
        expected = model(PointSet(coords), feats).feats
        for clone in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
            assert torch.equal(clone(PointSet(coords), feats).feats, expected)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"in_channels": 0}, "at least one input feature channel"),
        ({"grid_size": 0.0}, "grid_size must be positive"),
        ({"order": []}, "at least one space-filling curve"),
        ({"enc_conv": [True, True]}, "one entry per stage"),
        ({"dec_attn": [False]}, "one per decoder stage"),
        ({"stride": [2, 0]}, "stride must be positive"),
        ({"enc_num_head": [2, 5, 8]}, "24 channels cannot be split into 5 heads"),
        ({"enc_num_head": [2, 4, 6]}, "head width divisible by 6"),
        ({"dec_attn": [True, True]}, "decoder stage 0 attends but encoder stage 0 does not"),
        ({"dec_patch_size": [16, 8]}, "a level has one patch layout"),
    ],
)
def test_inconsistent_configurations_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        BACKBONES.build({"type": "LitePT", **TINY, **change})


def test_heads_of_stages_that_do_not_attend_are_not_constrained():
    # 12 channels in 5 heads would be impossible, but stage 0 only convolves.
    build(enc_num_head=[5, 4, 8])
