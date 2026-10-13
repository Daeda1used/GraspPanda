"""Every measured number of the model zoo has its evidence, for all methods.

A row of a ``configs/<method>/metafile.yaml`` with provenance ``author-ckpt`` or
``reproduced`` names a result record. This file holds every such row to it: the record
is committed, covers the whole test split of its protocol, was produced by clean code
with hashed weights, and its per-scene values give exactly the numbers of the row. The
arithmetic is written out here, independent of ``tools/record_result.py``, and the tool
is then held to the same numbers.

``test_zoo_pages.py`` compares the pages that show these numbers with the metafiles.
"""

from __future__ import annotations

import json
import re

import pytest

from grasppanda.hub import load_metafiles
from tools import record_result

from .conftest import REPO

#: Per protocol: the dataset, and scenes x frames per scene of its whole test split.
WHOLE_SPLIT = {
    "graspnet1b-sv-v1": ("graspnet1b", 90, 256),
    "graspclutter6d-sv-v1": ("graspclutter6d", 235, 13),
}
#: The scenes behind the three numbers of a GraspNet-1Billion row.
G1B_SCENES = {"seen": range(100, 130), "similar": range(130, 160), "novel": range(160, 190)}
FRICTION = {"AP0.8": 0.8, "AP0.4": 0.4}
WEIGHT_SOURCE = {"author-ckpt": "author", "reproduced": "grasppanda"}
#: Records stored under another name than ``<model>_<provenance>_<split>_<variant>.json``
#: (``tools/record_result.py --name``), as the row names them, each with its reason.
NAMED_OTHERWISE: dict[str, str] = {}
METAFILES = load_metafiles()
MEASURED = [
    (meta, model, result)
    for meta in METAFILES
    for model in meta.models
    for result in model.results
    if result.provenance != "reported"
]


def variant_of(result) -> str:
    return "cd" if result.collision_filter else "raw"


@pytest.mark.parametrize(
    ("meta", "model", "result"),
    [
        pytest.param(*row, id=f"{row[1].name}:{row[2].provenance}:{variant_of(row[2])}")
        for row in MEASURED
    ],
)
def test_every_measured_row_has_a_whole_split_record_that_gives_its_numbers(meta, model, result):
    record = json.loads((meta.path.parent / result.record).read_text(encoding="utf-8"))
    dataset, scenes, frames = WHOLE_SPLIT[model.protocol]

    # the record is of this model's protocol and camera, and of the evaluator the row names
    assert record["schema"] == 1 and record["protocol"] == model.protocol
    assert record["dataset"] == dataset and record["cameras"] == [model.camera]
    assert record["evaluator"]["type"] == result.evaluator

    # the whole test split, every frame scored
    assert record["split"] == "test" and record["partial"] is False and record["missing"] == []
    assert record["scored"] == record["requested"] == scenes * frames
    assert all(key.startswith(f"{model.camera}/scene_") for key in record["per_scene"])
    per_scene = {int(key[-4:]): entry for key, entry in record["per_scene"].items()}
    assert len(per_scene) == scenes and {entry["frames"] for entry in per_scene.values()} == {
        frames
    }

    # clean code, hashed weights that loaded without a mismatch
    assert record["code"]["dirty"] is False and re.fullmatch(
        r"[0-9a-f]{40}", record["code"]["commit"]
    )
    assert re.fullmatch(r"[0-9a-f]{64}", record["weights"]["sha256"])
    assert record["weights"]["clean_load"] is True
    assert model.weights is not None and model.weights.source == WEIGHT_SOURCE[result.provenance]
    if result.provenance == "reproduced":  # the released file is the evaluated file
        assert model.weights.sha256 == record["weights"]["sha256"]

    # the row says what the record says about post-processing and ground-truth inputs
    filtered = any("Collision" in step["type"] for step in record["postprocessing"])
    assert result.collision_filter is filtered and (record["variant"] == "raw") is not filtered
    assert f"(gt_assisted: {', '.join(record['gt_assisted']) or 'none'})" in result.inputs

    # exactly the numbers of the row: the mean over the scenes of each split, in percent
    groups = G1B_SCENES if model.protocol == "graspnet1b-sv-v1" else {"test": sorted(per_scene)}
    frictions = record["evaluator"]["info"]["constants"]["frictions"]
    for metric, stated in result.numbers().items():
        assert set(stated) == set(groups), f"{metric} must be stated for every split"
        for split, number in stated.items():
            entries = [per_scene[scene] for scene in groups[split]]
            values = [
                entry["mean_precision"]
                if metric == "AP"
                else entry["per_column"][frictions.index(FRICTION[metric])]
                for entry in entries
            ]
            assert round(100 * sum(values) / len(values), 2) == number, (metric, split)
    overall = sum(entry["mean_precision"] for entry in per_scene.values()) / scenes
    assert record["metrics"]["AP"] == pytest.approx(overall, abs=1e-12)

    # the systematic name, and the tool that wrote the file computes the same row
    name = f"{model.name}_{result.provenance}_{record['split']}_{record['variant']}.json"
    assert result.record == f"records/{name}" or result.record in NAMED_OTHERWISE
    assert record_result.headline(record) == result.numbers()
    assert record_result.collision_filter(record) is result.collision_filter


def test_records_and_rows_correspond_one_to_one_and_hold_no_machine_path():
    named = [(meta.path.parent / result.record) for meta, _, result in MEASURED]
    assert len(set(named)) == len(named), "two rows name one record"
    committed = sorted((REPO / "configs").glob("*/records/*"))
    assert sorted(named) == committed, "a record without a row, or a row without a record"
    for path in committed:
        text = path.read_text(encoding="utf-8")
        record = json.loads(text)
        assert record_result.absolute_paths(record) == [], path.name
        assert record["weights"]["path"].startswith("$GRASPPANDA_"), path.name
        assert text == json.dumps(record, indent=2, sort_keys=True) + "\n", path.name


def test_the_rows_of_one_model_are_one_per_variant_from_one_run():
    for meta in METAFILES:
        for model in meta.models:
            rows = [result for result in model.results if result.provenance != "reported"]
            if not rows:
                continue
            assert sorted(variant_of(result) for result in rows) == ["cd", "raw"], model.name
            records = [json.loads((meta.path.parent / r.record).read_text()) for r in rows]
            for key in ("config_hash", "seed", "split", "cameras", "gt_assisted"):
                assert records[0][key] == records[1][key], (model.name, key)
            assert records[0]["weights"]["sha256"] == records[1]["weights"]["sha256"], model.name
            assert records[0]["code"]["commit"] == records[1]["code"]["commit"], model.name
