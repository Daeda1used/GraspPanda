"""The numbers a project prints are those of result records committed with it."""

from __future__ import annotations

import json

from grasppanda.sweep import load_sweep
from tools import record_result

from .conftest import REPO
from .test_zoo_pages import table_under

PROJECT = REPO / "projects/encoder_comparison"
#: Run of ``gsnet_encoders.yaml`` → stem of its two records (``_raw.json``, ``_cd.json``).
#: The shipped encoder's run is also GSNet's third training, so its records live there.
ENCODER_RECORDS = {
    "sunet14": "configs/gsnet/repeats/gsnet_sunet14_g1b-realsense_10e_run3_test",
    "spunet": "projects/encoder_comparison/records/spunet_test",
    "ptv3": "projects/encoder_comparison/records/ptv3_test",
}


def test_the_encoder_comparison_prints_what_its_records_hold():
    """Every row of its results is a run of the sweep scored on the whole test split: the
    AP of the row is the headline of the run's records, the weights of a record are those
    the sweep trained in the work directory of that run, and a run without records shows
    no AP."""
    planned = {
        cell.choices["config"]: cell.config.select("runtime.work_dir")
        for cell in load_sweep(PROJECT / "gsnet_encoders.yaml").plan()
    }
    rows = {
        row[0].strip("`"): row
        for row in table_under((PROJECT / "README.md").read_text(encoding="utf-8"), "## Results")
    }
    assert list(rows) == ["sunet14", "spunet", "ptv3", "litept"]
    assert set(rows) == set(planned)
    for name, row in rows.items():
        if name not in ENCODER_RECORDS:
            assert not any(character.isdigit() for cell in row[2:] for character in cell), name
            continue
        for cell, variant in zip(row[2:4], ("raw", "cd")):
            path = REPO / f"{ENCODER_RECORDS[name]}_{variant}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["variant"] == variant and not record["partial"], path
            assert record["scored"] == record["requested"] == 23040, path
            assert record["weights"]["path"].startswith(f"{planned[name]}/"), path
            numbers = record_result.headline(record)["AP"]
            assert cell == " / ".join(f"{value:.2f}" for value in numbers.values()), path
