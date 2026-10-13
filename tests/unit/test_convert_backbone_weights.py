"""The converter of released encoder checkpoints, on files written by the test itself.

Each format is fed a checkpoint with the layout of its source (container key, prefix,
the extra parts such a file carries) and random tensors of the right shapes. The real
files are covered by ``tests/integration/test_backbone_released_weights.py``.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from grasppanda.registry import BACKBONES
from tools.convert_backbone_weights import (
    FORMATS,
    BackboneWeightsError,
    convert,
    load_tensors_only,
    main,
)


def _state(node: dict) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {key: torch.rand_like(value.float()) for key, value in _build(node).items()}


def _build(node: dict) -> dict[str, torch.Tensor]:
    return BACKBONES.build(dict(node)).state_dict()


def _source(name: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """A checkpoint laid out like the released file of a format, and its encoder tensors."""
    spec = FORMATS[name]
    if name == "csc-pointnet2":  # the file holds four propagation levels
        encoder = _state(FORMATS["csc-pointnet2-dense"]["backbone"])
    else:
        encoder = _state(spec["backbone"])
    inside = {f"{spec['prefix']}{key}": value for key, value in encoder.items()}
    extra = {
        "pointnext-s": {"head.head.0.0.weight": torch.rand(32, 32, 1)},
        "votenet": {"vgen.conv1.weight": torch.rand(4, 4), "pnet.conv1.bias": torch.rand(4)},
    }.get(name, {})
    return {spec["container"]: {**inside, **extra}, "epoch": 3}, encoder


@pytest.mark.parametrize("name", ["pointnext-s", "votenet", "csc-pointnet2", "csc-pointnet2-dense"])
def test_the_encoder_is_cut_out_unchanged_and_fits_its_class_strictly(name):
    checkpoint, encoder = _source(name)
    converted = convert(checkpoint, name)
    assert all(key.startswith("backbone.") for key in converted)
    model = BACKBONES.build(dict(FORMATS[name]["backbone"]))
    model.load_state_dict({key[len("backbone.") :]: value for key, value in converted.items()})
    for key, value in converted.items():
        assert torch.equal(value, encoder[key[len("backbone.") :]])
    if name == "csc-pointnet2":
        assert not any(".fp3." in key or ".fp4." in key for key in converted)
        assert len(converted) < len(encoder)
    else:
        assert len(converted) == len(encoder)


def test_every_format_names_a_backbone_that_can_be_built():
    for name, spec in FORMATS.items():
        if name == "pointnext-xl":  # 41 M parameters: its options are those of the size test
            continue
        assert _build(spec["backbone"]), name
    assert FORMATS["pointnext-xl"]["backbone"]["width"] == 64


def test_a_file_of_another_size_or_layout_is_refused_with_counts():
    checkpoint, _ = _source("pointnext-s")
    with pytest.raises(BackboneWeightsError, match=r"does not fit format 'pointnext-b'.*missing"):
        convert(checkpoint, "pointnext-b")
    with pytest.raises(BackboneWeightsError, match="expects the weights under 'model_state_dict'"):
        convert(checkpoint, "votenet")
    with pytest.raises(BackboneWeightsError, match="unknown format"):
        convert(checkpoint, "pointnet")
    broken = {"model": dict(checkpoint["model"])}
    broken["model"]["encoder.stray.weight"] = torch.rand(2)
    with pytest.raises(BackboneWeightsError, match="1 unexpected"):
        convert(broken, "pointnext-s")


ALLOW_LIST = hasattr(torch.serialization, "safe_globals")  # PyTorch 2.5 and newer


@pytest.mark.skipif(ALLOW_LIST, reason="PyTorch has the allow-list of the safe loader")
def test_without_the_allow_list_a_file_with_objects_is_refused_with_the_way_out(tmp_path):
    checkpoint, _ = _source("csc-pointnet2")
    plain = tmp_path / "plain.pth"
    torch.save(checkpoint, plain)
    assert set(load_tensors_only(plain)) == set(checkpoint)  # tensors alone are read
    checkpoint["config"] = argparse.Namespace(lr=0.1)
    source = tmp_path / "pretrain.pth"
    torch.save(checkpoint, source)
    with pytest.raises(BackboneWeightsError, match=r"needs PyTorch 2\.5 or newer"):
        load_tensors_only(source)


@pytest.mark.skipif(not ALLOW_LIST, reason="needs the allow-list of PyTorch 2.5 or newer")
def test_a_config_object_in_the_file_is_read_as_an_inert_stand_in(tmp_path, capsys):
    checkpoint, encoder = _source("csc-pointnet2")
    checkpoint["config"] = argparse.Namespace(lr=0.1, nested=argparse.Namespace(a=[1, 2]))
    source, target = tmp_path / "pretrain.pth", tmp_path / "out.pth"
    torch.save(checkpoint, source)
    with pytest.raises(Exception, match="Unsupported global"):
        torch.load(source, weights_only=True)
    loaded = load_tensors_only(source)
    assert type(loaded["config"]).__name__ == "Namespace"
    assert not isinstance(loaded["config"], argparse.Namespace)  # never the real class
    assert not vars(loaded["config"])
    assert torch.equal(loaded["state_dict"]["sa1.mlp_module.layer0.conv.weight"],
                       encoder["sa1.mlp_module.layer0.conv.weight"])  # fmt: skip

    assert main([str(source), str(target), "--format", "csc-pointnet2"]) == 0
    written = torch.load(target, weights_only=True)
    assert written["meta"]["backbone"]["type"] == "PointNet2"
    assert len(written["state_dict"]) == written["meta"]["tensors"] == 96
    assert "model.backbone" in capsys.readouterr().out
    assert main([str(source), str(target), "--format", "votenet"]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["--list"]) == 0
    assert "pointnext-xl" in capsys.readouterr().out
