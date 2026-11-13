"""``tools/train_fit.py``: the training loss of weights with layers as in training or as in
evaluation."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import train_fit as tool

from .conftest import REPO


class Toy(nn.Module):
    """A loss through a batch-norm layer and two dropout layers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(4, 1))

    def loss(self, observation, supervision) -> dict[str, torch.Tensor]:
        out = self.head(self.drop(self.norm(self.linear(observation.points))))
        return {"loss/a": out.square().mean(), "loss/b": out.abs().mean(), "metric/c": out.mean()}


def toy() -> Toy:
    torch.manual_seed(0)
    return Toy()


def batches(count: int = 3, frames: int = 4):
    generator = torch.Generator().manual_seed(1)
    return [
        SimpleNamespace(
            observation=SimpleNamespace(
                points=torch.randn(frames, 4, generator=generator), keys=list(range(frames))
            ),
            supervision=None,
        )
        for _ in range(count)
    ]


def modes(model: nn.Module) -> dict[str, bool]:
    return {name: module.training for name, module in model.named_modules() if name}


def test_set_mode_switches_the_named_layers_and_nothing_else():
    model = toy()
    assert tool.set_mode(model) == [] and all(modes(model).values())
    assert tool.set_mode(model, dropout_off="") == ["drop", "head.0"]
    assert modes(model) == {
        "linear": True, "norm": True, "drop": False, "head": True, "head.0": False, "head.1": True
    }  # fmt: skip
    assert tool.set_mode(model, dropout_off=r"^drop$") == ["drop"]
    assert modes(model)["head.0"] and not modes(model)["drop"]  # every call starts from training
    assert tool.set_mode(model, running_norms=True) == []
    assert not modes(model)["norm"] and modes(model)["drop"] and model.training


def test_measure_averages_the_loss_terms_and_leaves_the_weights_alone():
    model = toy()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    tool.set_mode(model, dropout_off="")
    result = tool.measure(model, batches())
    assert result["frames"] == 12
    assert set(result) == {"loss/a", "loss/b", "loss", "frames"}  # a metric is no loss term
    assert result["loss"] == pytest.approx(result["loss/a"] + result["loss/b"])
    # batch-norm in training mode moves its running statistics; they are put back
    after = model.state_dict()
    assert all(torch.equal(after[name], value) for name, value in before.items())
    with torch.no_grad():
        expected = [float(model.loss(b.observation, None)["loss/a"]) for b in batches()]
    assert result["loss/a"] == pytest.approx(sum(expected) / 3)
    with pytest.raises(ValueError, match="no frame"):
        tool.measure(model, [])


def test_rows_are_the_two_modes_and_on_request_what_lies_between():
    model = toy()
    assert list(tool.rows(model, batches)) == ["training", "evaluation"]
    result = tool.rows(model, batches, between=True, off={"first": r"^drop$"})
    assert list(result) == [
        "training", "evaluation", "dropout off", "running statistics", "first off"
    ]  # fmt: skip
    assert model.training and all(modes(model).values())
    # the same frames and the same draws in every measurement: a row repeats
    again = tool.rows(model, batches, between=True)
    assert again == {name: result[name] for name in again}
    # each switch moves the loss of the toy, and the named layer is one of the two dropouts
    losses = [row["loss"] for row in result.values()]
    assert len(set(losses)) == len(losses)


def test_a_pattern_that_matches_no_dropout_layer_names_the_layers():
    with pytest.raises(ValueError, match=r"matches no dropout layer; the model has drop, head\.0"):
        tool.rows(toy(), batches, off={"absent": r"^no_such_layer$"})


def test_names_and_patterns_are_checked_on_the_command_line():
    assert tool.named("authors=some/file.pth") == ("authors", "some/file.pth")
    assert tool.pattern(r"crop=grouper\.branches\.\d+") == ("crop", r"grouper\.branches\.\d+")
    for parse, text in ((tool.named, "file.pth"), (tool.pattern, "crop=("), (tool.pattern, "x")):
        with pytest.raises(argparse.ArgumentTypeError):
            parse(text)


def test_the_overrides_choose_the_frames_and_nothing_of_the_model():
    from grasppanda.config import Config, resolve_config_path

    path = resolve_config_path("economicgrasp/economicgrasp_sunet14_g1b-realsense_10e")
    shipped, changed = Config.load(path), Config.load(path, tool.overrides("val_dev", 16))
    assert changed.select("data.train.dataset.split") == "val_dev"
    assert changed.select("data.train.dataset.frame_step") == 16
    assert changed.select("data.train.shuffle") is False
    assert changed.select("model") == shipped.select("model")
    assert changed.select("data.train.dataset.transforms") == shipped.select(
        "data.train.dataset.transforms"
    )


# ------------------------------------------------------------------ the committed records


def record(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def flat(path: str) -> str:
    return " ".join((REPO / path).read_text(encoding="utf-8").split())


def test_the_ablation_states_the_recorded_fit_of_its_multi_scale_run():
    """The run with the four-radius crop alone: the record is that of the sweep cell, and
    its loss with the layers as in evaluation is not above the one in training."""
    from grasppanda.sweep import load_sweep

    found = record("projects/finegrasp_ablation/records/train_fit.json")
    cell = next(
        cell
        for cell in load_sweep(REPO / "projects/finegrasp_ablation/changes.yaml").plan()
        if cell.choices["change"] == "multi-scale"
    )
    assert found["about"] == tool.ABOUT and found["opts"] == cell.overrides
    assert (found["split"], found["frames"]) == ("val_dev", 160)
    rows = found["weights"]["multi-scale"]
    assert list(rows) == ["training", "evaluation"]
    assert rows["evaluation"]["loss"] < rows["training"]["loss"]
    assert (
        f"on {found['frames']} training frames its loss is {rows['training']['loss']:.2f} with "
        f"the layers as in training and {rows['evaluation']['loss']:.2f} as in evaluation"
    ) in flat("projects/finegrasp_ablation/README.md")


def test_the_readme_of_finegrasp_states_the_recorded_fit():
    """FineGrasp trained here fits the training frames closer than the authors' weights, on
    every loss term; the two sets of weights of EconomicGrasp, whose training here lands on
    its authors' checkpoint, are close on the same frames."""
    fine = record("configs/finegrasp/diagnostics/train_fit.json")
    economic = record("configs/economicgrasp/diagnostics/train_fit.json")
    for found in (fine, economic):
        assert found["about"] == tool.ABOUT and found["opts"] == []
        assert (found["split"], found["frames"]) == ("val_dev", 160)
        assert list(found["weights"]) == ["authors", "trained"]
    here, authors = (fine["weights"][name]["evaluation"] for name in ("trained", "authors"))
    assert all(here[term] < authors[term] for term in authors), "on every term"
    apart = abs(
        economic["weights"]["trained"]["evaluation"]["loss"]
        - economic["weights"]["authors"]["evaluation"]["loss"]
    )
    assert apart < 0.1 < authors["loss"] - here["loss"]
    assert (
        f"On {fine['frames']} training frames its training loss is the lower one, "
        f"{here['loss']:.2f} against the authors' {authors['loss']:.2f}, on every term "
        f"(EconomicGrasp's two: {apart:.2f} apart;"
    ) in flat("configs/finegrasp/README.md")


def test_the_graspness_labels_of_finegrasp_are_checked_through_the_released_weights():
    """The authors' graspness files are not published. Their weights tell which kind they
    were trained on: the graspness loss on the labels generated here, normalised per
    object, is far below the one on labels normalised over the scene, and every other
    term is the same, because nothing else reads these labels."""
    ours = record("configs/finegrasp/diagnostics/train_fit.json")
    scene = record("configs/finegrasp/diagnostics/train_fit_scene_graspness.json")
    assert scene["about"] == tool.ABOUT
    assert scene["opts"] == ["data.train.dataset.graspness.normalization=scene"]
    assert (scene["split"], scene["frames"]) == (ours["split"], ours["frames"])
    for name in ("authors", "trained"):
        here, there = (found["weights"][name]["evaluation"] for found in (ours, scene))
        assert there["loss/graspness"] > 2 * here["loss/graspness"], name
        assert all(
            there[term] == here[term]
            for term in here
            if term.startswith("loss/") and term != "loss/graspness"
        ), name
    here, there = (found["weights"]["authors"]["evaluation"] for found in (ours, scene))
    assert (
        f"graspness loss, {here['loss/graspness']:.2f} on these, "
        f"{there['loss/graspness']:.2f} on scene-normalised ones"
    ) in flat("configs/finegrasp/README.md")
