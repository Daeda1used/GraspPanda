"""Point-token encoders: PointViT, the checkpoint converter and PointMamba.

The shared contract of ``PointViT`` is in ``test_backbone_contract``. ``PointMamba``
needs the optional ``mamba-ssm`` library and a GPU (its selective scan is a CUDA
kernel); its tests are skipped without them. Loading the released files is the subject
of ``tests/integration/test_encoder_checkpoints.py``."""

from __future__ import annotations

import pytest
import torch

from grasppanda.models.backbones.point_tokens import PointViT, convert_point_token_checkpoint
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

TINY = {"num_group": 12, "group_size": 8, "embed_dim": 16, "depth": 2, "token_dim": 16}


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_defaults_are_the_point_mae_encoder_with_1024_tokens():
    model = PointViT()
    assert (model.output, model.in_channels, model.out_channels) == ("subsampled", 0, 384)
    assert (model.num_group, model.group_size) == (1024, 32)
    state = model.state_dict()
    assert state["encoder.first_conv.0.weight"].shape == (128, 3, 1)
    assert state["encoder.second_conv.3.weight"].shape == (384, 512, 1)
    assert state["pos_embed.2.weight"].shape == (384, 128)
    assert state["blocks.blocks.11.attn.qkv.weight"].shape == (1152, 384)
    assert "blocks.blocks.0.attn.qkv.bias" not in state  # qkv_bias: false, as released
    assert not any(key.startswith(("reduce_dim", "proj")) for key in state)
    # Point-BERT: a 256-wide patch encoder joined to the transformer by a linear layer.
    assert PointViT(token_dim=256).state_dict()["reduce_dim.weight"].shape == (384, 256)
    with pytest.raises(ValueError, match="not a multiple of num_heads"):
        PointViT(embed_dim=100, num_heads=6)


def test_tokens_live_on_the_furthest_point_sample_of_every_cloud():
    model = PointViT(**TINY, num_heads=2, drop_path_rate=0.0).eval()
    points = PointSet.cat([PointSet(torch.rand(40, 3)), PointSet(torch.rand(57, 3) + 3.0)])
    with torch.no_grad():
        out = model(points, None)
    assert out.feats.shape == (24, 16)
    assert out.points.sizes.tolist() == [12, 12]
    index = out.points.parent_index
    assert (index[:12] < 40).all() and (index[12:] >= 40).all()
    # centre_clouds: moving a cloud as a whole does not change its features.
    moved = PointSet.cat([PointSet(points.coords[:40] - 7.0), PointSet(points.coords[40:])])
    with torch.no_grad():
        again = model(moved, None)
    assert torch.allclose(out.feats, again.feats, atol=1e-4)
    with pytest.raises(ContractError, match=r"at least 13\s+points; cloud 0 of the batch has 12"):
        model(PointSet(torch.rand(12, 3)), None)


def test_converter_strips_the_wrappers_and_reports_what_it_leaves():
    model = PointViT(**TINY, num_heads=2)
    source = PointViT(**TINY, num_heads=2)
    released = {f"module.MAE_encoder.{key}": value for key, value in source.state_dict().items()}
    released["module.MAE_decoder.blocks.0.norm1.weight"] = torch.zeros(16)
    released["module.mask_token"] = torch.zeros(1, 1, 16)
    released["module.MAE_encoder.pos_embed.2.bias"] = torch.zeros(7)  # a wrong shape
    state, report = convert_point_token_checkpoint({"base_model": released}, model)
    assert report["missing"] == ["pos_embed.2.bias"]
    assert report["mismatched"] == ["module.MAE_encoder.pos_embed.2.bias: file (7,), model (16,)"]
    assert report["dropped"] == ["module.MAE_decoder.blocks.0.norm1.weight", "module.mask_token"]
    assert len(report["loaded"]) == len(source.state_dict()) - 1
    assert all(key.startswith("backbone.") for key in state)
    bare, _ = convert_point_token_checkpoint({"base_model": released}, model, prefix="")
    model.load_state_dict(bare, strict=False)
    assert torch.equal(model.norm.weight, source.norm.weight)
    # Point-BERT files store the encoder under `transformer_q.`.
    bert = {f"transformer_q.{key}": value for key, value in source.state_dict().items()}
    assert not convert_point_token_checkpoint({"base_model": bert}, model)[1]["missing"]


# ------------------------------------------------------------------------ PointMamba


@pytest.fixture
def mamba():
    pytest.importorskip("mamba_ssm", reason="needs the mamba-ssm extra")
    if not torch.cuda.is_available():
        pytest.skip("the selective scan of mamba-ssm is a CUDA kernel")


def test_pointmamba_contract_on_a_gpu(mamba):
    model = BACKBONES.build({"type": "PointMamba", **TINY, "out_channels": 10}).cuda()
    assert (model.output, model.status, model.extras) == ("subsampled", "bridge", ("mamba-ssm",))
    assert model.state_dict()["blocks.layers.1.mixer.in_proj.weight"].shape == (64, 16)
    points = PointSet.cat([PointSet(torch.rand(40, 3)), PointSet(torch.rand(57, 3))]).to("cuda")
    out = model(points, None)
    assert out.feats.shape == (24, 10) and out.points.sizes.tolist() == [12, 12]
    assert torch.equal(out.points.coords, points.coords[out.points.parent_index])
    out.feats.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Each cloud is serialised on its own: alone it gives the same features, up to the
    # arithmetic of the CUDA scan (differences of 1e-4 were seen on features of size 1).
    model.eval()
    with torch.no_grad():
        together = model(points, None).feats
        alone = model(PointSet(points.coords[:40]), None).feats
    assert torch.allclose(together[:12], alone, atol=2e-3)


def test_pointmamba_says_that_it_needs_a_gpu(mamba):
    model = BACKBONES.build({"type": "PointMamba", **TINY})
    with pytest.raises(ContractError, match="PointMamba runs on CUDA only"):
        model(PointSet(torch.rand(40, 3)), None)


def test_pointmamba_names_its_extra_when_the_library_is_missing(monkeypatch):
    import builtins

    from grasppanda.utils.imports import MissingDependencyError

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("mamba_ssm"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setitem(__import__("sys").modules, "mamba_ssm", None)
    with pytest.raises(MissingDependencyError, match=r"grasppanda\[mamba-ssm\]"):
        BACKBONES.build({"type": "PointMamba", **TINY})


# ----------------------------------------------------------------------------- Uni3D


def test_uni3d_runs_the_blocks_of_a_timm_model_over_the_tokens():
    pytest.importorskip("timm", reason="needs the timm extra")
    node = {"num_group": 12, "group_size": 8, "timm_model": "eva02_tiny_patch14_224"}
    model = BACKBONES.build({"type": "Uni3D", **node, "out_channels": 10})
    assert (model.output, model.status, model.extras) == ("subsampled", "bridge", ("timm",))
    state = model.state_dict()
    assert state["encoder.first_conv.0.weight"].shape == (128, 6, 1)  # offsets and colour
    assert state["reduce_dim.weight"].shape == (192, 512)  # `encoder2trans` of the reference
    assert state["visual.blocks.11.attn.qkv.weight"].shape == (576, 192)
    assert state["cls_token"].shape == (1, 1, 192)
    assert not any(key.startswith("visual.patch_embed") for key in state)  # blocks only
    points = PointSet.cat([PointSet(torch.rand(40, 3)), PointSet(torch.rand(57, 3))])
    out = model(points, torch.rand(97, 3))
    assert out.feats.shape == (24, 10) and out.points.sizes.tolist() == [12, 12]
    out.feats.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # A Uni3D file: the encoder under `point_encoder.`, the bridge layer as `encoder2trans`.
    released = {}
    for key, value in BACKBONES.build({"type": "Uni3D", **node}).state_dict().items():
        key = key.replace("reduce_dim.", "encoder2trans.")
        released[f"point_encoder.{key}"] = value
    released["logit_scale"] = torch.zeros(())
    plain = BACKBONES.build({"type": "Uni3D", **node})
    _, report = convert_point_token_checkpoint({"module": released}, plain)
    assert report["missing"] == [] and report["dropped"] == ["logit_scale"]
    with pytest.raises(ValueError, match="has no `blocks` and `norm`"):
        BACKBONES.build({"type": "Uni3D", **{**node, "timm_model": "resnet18"}})
