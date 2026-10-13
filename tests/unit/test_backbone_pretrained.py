"""The ``pretrained`` option of the encoders, on files written by the test itself.

Released checkpoints are the subject of
``tests/integration/test_backbone_released_weights.py``; nothing is downloaded here.
"""

from __future__ import annotations

import sys

import pytest
import torch

pytest.importorskip("spconv", reason="needs the spconv extra")

from grasppanda.models.backbones import _pretrained
from grasppanda.models.backbones._pretrained import (
    RELEASED,
    PretrainedError,
    load_pretrained,
    read_state,
)
from grasppanda.registry import BACKBONES

TINY = {
    "type": "SpUNet",
    "in_channels": 3,
    "out_channels": 5,
    "grid_size": 0.25,
    "base_channels": 4,
    "channels": [6, 8, 7, 5],
    "layers": [1, 2, 2, 1],
}


class _Scheduler:
    """Stands for an object of a foreign code base inside a training checkpoint."""

    def __init__(self) -> None:
        self.total_steps = 7


def pointcept_file(tmp_path, model, name="released.pth", prefix="module.backbone."):
    """A training checkpoint in the Pointcept arrangement: prefix, 1×1 kernels as
    ``(out, 1, 1, 1, in)``, a class head beside the encoder and a pickled scheduler."""
    state = {}
    for key, value in model.state_dict().items():
        if value.ndim == 2:  # undo convert_pointcept_state
            value = value.t().reshape(value.shape[0], 1, 1, 1, value.shape[1])
        state[prefix + key] = value.clone()
    state["module.seg_head.weight"] = torch.zeros(2, 5)
    path = tmp_path / name
    torch.save({"state_dict": state, "epoch": 3, "scheduler": _Scheduler()}, path)
    return path


def test_a_training_checkpoint_loads_strictly_and_nothing_of_it_is_imported(tmp_path, monkeypatch):
    source = BACKBONES.build(TINY)
    path = pointcept_file(tmp_path, source)
    # The class the file pickles is gone: a loader that imported it would fail here.
    monkeypatch.delattr(sys.modules[__name__], "_Scheduler")
    model = BACKBONES.build({**TINY, "pretrained": str(path)})
    report = model.pretrained_report
    assert report.strict and report.prefix == "module.backbone." and report.outside == 1
    assert len(report.loaded) == len(source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key
    assert "tensors loaded" in report.summary()


def test_a_tensor_that_does_not_fit_is_refused_or_skipped_by_name(tmp_path):
    path = pointcept_file(tmp_path, BACKBONES.build(TINY))
    other = {**TINY, "in_channels": 6, "out_channels": 9}
    with pytest.raises(PretrainedError, match="pretrained_mismatch: skip") as refused:
        BACKBONES.build({**other, "pretrained": str(path)})
    assert "conv_input.0.weight" in str(refused.value)
    model = BACKBONES.build({**other, "pretrained": str(path), "pretrained_mismatch": "skip"})
    report = model.pretrained_report
    assert not report.strict
    assert {entry.split()[0] for entry in report.mismatched} == {
        "conv_input.0.weight",
        "final.weight",
        "final.bias",
    }
    assert len(report.loaded) == len(model.state_dict()) - 3
    with pytest.raises(ValueError, match="`error` or `skip`"):
        BACKBONES.build({**TINY, "pretrained": str(path), "pretrained_mismatch": "ignore"})


def test_a_file_of_another_architecture_is_refused(tmp_path):
    path = tmp_path / "other.pth"
    torch.save({"layer.weight": torch.zeros(3, 3)}, path)
    with pytest.raises(PretrainedError, match="not a checkpoint of this architecture"):
        BACKBONES.build({**TINY, "pretrained": str(path)})


def test_a_release_that_is_absent_prints_how_to_fetch_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    with pytest.raises(PretrainedError) as missing:
        BACKBONES.build({**TINY, "pretrained": "gaussiancross-scannet-lin"})
    message = str(missing.value)
    entry = RELEASED["gaussiancross-scannet-lin"]
    assert f"curl -L -o {tmp_path / 'weights_components' / entry.file} {entry.url}" in message
    assert "pretrained: null" in message and "MIT" in message
    with pytest.raises(PretrainedError, match="neither a file nor a known release"):
        BACKBONES.build({**TINY, "pretrained": "no-such-weights"})


def test_a_release_is_found_under_the_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path))
    (tmp_path / "weights_components").mkdir()
    source = BACKBONES.build(TINY)
    pointcept_file(tmp_path / "weights_components", source, RELEASED["volt-teacher-scannet"].file)
    model = BACKBONES.build({**TINY, "pretrained": "volt-teacher-scannet"})
    assert model.pretrained_report.strict


def test_the_table_of_releases_names_registered_classes_and_their_options():
    from grasppanda.registry import component_parameters

    for name, entry in RELEASED.items():
        assert name == name.lower() and entry.url.startswith("https://"), name
        params, _ = component_parameters(BACKBONES.get(entry.component))
        assert set(entry.options) <= set(params), name
        assert {"pretrained", "pretrained_mismatch"} <= set(params), entry.component
        assert entry.licence and entry.megabytes > 0 and entry.trained_on and entry.paper
    files = [entry.file for entry in RELEASED.values()]
    assert len(set(files)) == len(files)


def test_plain_state_dicts_and_stored_tensors_read_the_same(tmp_path):
    state = {"a.weight": torch.arange(24.0).reshape(2, 3, 4)[:, ::2], "b": torch.tensor([3])}
    plain, foreign = tmp_path / "plain.pth", tmp_path / "foreign.pth"
    torch.save(state, plain)
    torch.save({"model": state, "extra": _Scheduler()}, foreign)
    for path in (plain, foreign):
        loaded = read_state(path)
        assert set(loaded) == set(state)
        for key, value in state.items():
            assert torch.equal(loaded[key], value) and loaded[key].dtype == value.dtype
    assert isinstance(_pretrained._read_inert(foreign)["extra"], _pretrained._Inert)


def test_load_pretrained_is_usable_on_any_module(tmp_path):
    layer = torch.nn.Linear(3, 2)
    path = tmp_path / "linear.pth"
    torch.save(
        {"net": {"encoder.weight": layer.weight.detach() + 1, "encoder.bias": layer.bias}}, path
    )
    target = torch.nn.Linear(3, 2)
    report = load_pretrained(target, str(path))
    assert report.strict and report.prefix == "encoder."
    assert torch.equal(target.weight, layer.weight + 1)


def test_one_set_of_prompt_driven_normalisation_layers_is_selected(tmp_path):
    """A PPT checkpoint stores `<layer>.norm.<i>.<tensor>`; one set becomes the layer."""
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    state = {"backbone.0.weight": source[0].weight.detach(), "backbone.0.bias": source[0].bias}
    for index in range(3):
        for name, value in source[1].state_dict().items():
            state[f"backbone.1.norm.{index}.{name}"] = value + index
    path = tmp_path / "ppt.pth"
    torch.save({"state_dict": state}, path)
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    report = load_pretrained(target, str(path), condition=2)
    assert report.strict and report.prefix == "backbone."
    assert torch.equal(target[1].running_mean, source[1].running_mean + 2)
    # Without the option the normalisation layers are simply absent from the file.
    with pytest.raises(PretrainedError, match="does not fit the encoder"):
        load_pretrained(target, str(path))
    with pytest.raises(PretrainedError, match=r"it has the sets \[0, 1, 2\]"):
        load_pretrained(target, str(path), condition=5)
    plain = tmp_path / "plain.pth"
    torch.save(source.state_dict(), plain)
    with pytest.raises(PretrainedError, match="it has none"):
        load_pretrained(target, str(plain), condition=0)
