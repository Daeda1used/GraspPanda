"""``tools/hggd_label_fit.py``: the loss of HGGD weights on labels of a given density."""

from __future__ import annotations

import argparse
import json

import pytest
import torch
from torch import nn

from tools import hggd_label_fit as tool

from .conftest import REPO
from .test_hggd_detector import batch, tiny
from .test_zoo_pages import table_under

#: How HGGD's README names a suppression setting → (distance in metres, angle in degrees).
SHOWN = {
    "2 cm, 60°": (0.02, 60.0),
    "1 cm, 30°": (0.01, 30.0),
    "0.5 cm, 30°": (0.005, 30.0),
    "none": (1e-06, 0.001),
}


def test_measure_reports_the_loss_and_the_label_counts_per_frame():
    model = tiny()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    torch.manual_seed(2)
    result = tool.measure(model, [batch(2), batch(2)])
    assert result["frames"] == 4 and result["labels"] == 60  # per frame
    assert 0 <= result["positive"] <= result["supervised"] <= result["labels"]
    assert result["supervised"] > 0
    terms = [value for name, value in result.items() if name.startswith("loss/")]
    assert len(terms) == 5 and result["loss"] == pytest.approx(sum(terms))
    # the weights as they are used: stages in training mode, norms on running statistics,
    # and nothing written, not even a running mean
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert model.training and norms and not any(norm.training for norm in norms)
    after = model.state_dict()
    assert all(torch.equal(after[name], value) for name, value in before.items())


def test_measure_refuses_to_average_nothing():
    with pytest.raises(ValueError, match="no frame"):
        tool.measure(tiny(), [])


def test_a_density_is_two_numbers_and_weights_have_a_name():
    assert tool.pair("0.01,30") == (0.01, 30.0)
    assert tool.named("authors=some/file.pth") == ("authors", "some/file.pth")
    for parse, text in ((tool.pair, "0.01"), (tool.pair, "a,b"), (tool.named, "file.pth")):
        with pytest.raises(argparse.ArgumentTypeError):
            parse(text)


def test_the_overrides_change_the_labels_and_nothing_of_the_model():
    """A density reads the shipped config with other labels: the suppression of the label
    asset, which frames, in which order; the anchors of the weights stay where they are."""
    from grasppanda.config import Config, resolve_config_path

    shipped = Config.load(resolve_config_path(tool.CONFIG))
    changed = Config.load(
        resolve_config_path(tool.CONFIG), tool.overrides(0.01, 30.0, "val_dev", 16)
    )
    labels = changed.select("data.train.dataset.view_grasps")
    assert (labels["nms_distance"], labels["nms_angle"]) == (0.01, 30.0)
    assert changed.select("data.train.shuffle") is False
    assert changed.select("model.anchors.shift_epochs") == 0
    model, other = dict(changed.select("model")), dict(shipped.select("model"))
    model["anchors"] = {**model["anchors"], "shift_epochs": other["anchors"]["shift_epochs"]}
    assert model == other


def test_the_readme_of_hggd_states_the_recorded_label_fit():
    """The table and the sentences about why the model trained here is below the authors'
    checkpoint are those of ``configs/hggd/diagnostics/label_fit.json``, and the record says
    what the README concludes: the checkpoint fits denser labels, the other model does not."""
    record = json.loads(
        (REPO / "configs/hggd/diagnostics/label_fit.json").read_text(encoding="utf-8")
    )
    rows = {(row["nms_distance"], row["nms_angle"]): row for row in record["rows"]}
    assert record["config"] == tool.CONFIG and len(rows) == len(record["rows"]) >= len(SHOWN)
    readme = (REPO / "configs/hggd/README.md").read_text(encoding="utf-8")
    table = table_under(readme, "**What was found.**")
    assert [cells[0].split(":")[0] for cells in table] == list(SHOWN)
    for cells in table:
        row = rows[SHOWN[cells[0].split(":")[0]]]
        losses = row["weights"]
        assert cells[1:] == [
            f"{row['labels']:.0f}",
            f"{row['positive']:.0f}",
            f"{losses['authors']['loss']:.2f}",
            f"{losses['trained']['loss']:.2f}",
        ], cells[0]
    shipped = rows[SHOWN["2 cm, 60°"]]
    flat = " ".join(readme.split())
    for sentence in (
        f"these labels leave {shipped['positive']:.0f} positives a frame",
        f"Its training loss on them is {shipped['weights']['authors']['loss']:.2f}",
        f"reaches {shipped['weights']['trained']['loss']:.2f}",
        f"Training loss on {shipped['frames']} frames",
        f"leaves about {round(shipped['labels'], -1):.0f}",
    ):
        assert sentence in flat, sentence
    # the finding: by density, the checkpoint's loss falls to less than half and ends below
    # the other model's, which moves by little and starts far below the checkpoint's
    by_density = sorted(rows.values(), key=lambda row: row["positive"])
    authors = [row["weights"]["authors"]["loss"] for row in by_density]
    trained = [row["weights"]["trained"]["loss"] for row in by_density]
    assert by_density[0] is shipped
    assert authors == sorted(authors, reverse=True) and authors[-1] < authors[0] / 2
    assert trained[0] < authors[0] / 1.5 and authors[-1] < trained[-1]
    assert max(trained) - min(trained) < 0.25
