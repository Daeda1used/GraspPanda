"""The numbers a project prints are those of result records committed with it."""

from __future__ import annotations

import json
from itertools import pairwise

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
    "litept": "projects/encoder_comparison/records/litept_test",
}
#: Encoder of a run of ``gsnet_recipe.yaml`` → stem of its two records.
RECIPE_RECORDS = {
    "sunet14": "projects/encoder_comparison/records/sunet14_adamw-onecycle_test",
    "ptv3": "projects/encoder_comparison/records/ptv3_adamw-onecycle_test",
    "litept": "projects/encoder_comparison/records/litept_adamw-onecycle_test",
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


def encoder_ap(name: str, variant: str) -> dict[str, float]:
    """Seen / similar / novel AP of a recorded run of the encoder comparison."""
    path = REPO / f"{ENCODER_RECORDS[name]}_{variant}.json"
    return record_result.headline(json.loads(path.read_text(encoding="utf-8")))["AP"]


def below_shipped(name: str) -> list[float]:
    """How far a run ends below the shipped encoder, on every split of both variants."""
    return [
        encoder_ap("sunet14", variant)[split] - value
        for variant in ("raw", "cd")
        for split, value in encoder_ap(name, variant).items()
    ]


def test_the_loss_column_of_the_encoder_comparison_is_the_recorded_last_epoch():
    losses = json.loads((PROJECT / "records/training_loss.json").read_text(encoding="utf-8"))
    rows = table_under((PROJECT / "README.md").read_text(encoding="utf-8"), "## Results")
    assert set(losses["runs"]) == set(ENCODER_RECORDS)
    for row in rows:
        name = row[0].strip("`")
        if name in ENCODER_RECORDS:
            assert len(losses["runs"][name]) == losses["epochs"] == 10
            assert row[4] == f"{losses['runs'][name][-1]:.2f}", name


def test_what_the_encoder_comparison_says_is_computed_from_its_records():
    """The sentences of the project and of the evidence page: how far SpUNet, PTv3 and
    LitePT end from the shipped encoder, the parameters, the two losses that show PTv3 has
    not fitted the training scenes and the three that show LitePT's loss rising."""
    losses = json.loads((PROJECT / "records/training_loss.json").read_text(encoding="utf-8"))
    runs = losses["runs"]
    spunet, ptv3 = below_shipped("spunet"), below_shipped("ptv3")
    litept = below_shipped("litept")
    rows = {
        row[0].strip("`"): row
        for row in table_under((PROJECT / "README.md").read_text(encoding="utf-8"), "## Results")
    }
    parameters = {name: float(row[1].removesuffix(" M")) for name, row in rows.items()}
    sentences = [
        f"{parameters['spunet'] / parameters['sunet14']:.1f} times the parameters",
        f"{min(spunet):.1f} to {max(spunet):.1f} AP",
        f"{ptv3[0]:.1f} AP lower on seen scenes",
        f"{litept[0]:.1f} AP lower on seen scenes",
        # LitePT's loss: after the second epoch, at its highest after that, at the end
        f"rose from {runs['litept'][1]:.2f} after the second epoch "
        f"to {max(runs['litept'][1:]):.2f} after the fourth",
        f"ends at {runs['litept'][-1]:.2f}",
    ]
    assert max(spunet) < 1.12 < min(ptv3 + litept), "what one seed moves separates the findings"
    assert runs["ptv3"][-1] > runs["sunet14"][4], "PTv3 after ten epochs, the shipped after five"
    assert runs["litept"].index(max(runs["litept"][1:])) == 3, "the fourth epoch"
    assert runs["litept"][-1] > runs["sunet14"][1], "LitePT after ten epochs, the shipped after two"
    pages = {
        "the project": (PROJECT / "README.md").read_text(encoding="utf-8"),
        "the evidence page": (REPO / "docs/advanced_guides/trained_swaps.md").read_text(
            encoding="utf-8"
        ),
    }
    for where, page in pages.items():
        flat = " ".join(page.split())
        for sentence in [*sentences, f"{runs['ptv3'][-1]:.2f}", f"{runs['sunet14'][4]:.2f}"]:
            assert sentence in flat, f"{where} does not say: {sentence}"
    # the page of the 250-iteration check points at the swap it called on a par
    short = (REPO / "docs/advanced_guides/learns_like_reference.md").read_text(encoding="utf-8")
    assert f"on a par here and {ptv3[0]:.0f} and {litept[0]:.0f} AP lower there" in short
    project = " ".join(pages["the project"].split())
    assert f"above the shipped encoder's after two epochs ({runs['sunet14'][1]:.2f})" in project


def test_the_evidence_page_shows_the_results_of_the_project():
    """The table of the docs page (an include) is the table of the project's README."""
    include = (REPO / "docs/includes/trained_encoders.md").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    assert include.strip() in readme
    assert table_under(include, "|") == table_under(readme, "## Results")
    page = (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8")
    assert page.count('--8<-- "docs/includes/trained_encoders.md"') == 1


def test_the_evidence_page_quotes_the_cheaper_checks_from_their_records():
    """Each swap trained in full, with what the probe and the 250-iteration check had said
    about it and how far it ended from the shipped encoder on seen scenes."""
    cells = {
        (cell["method"], cell["option"]): cell
        for cell in json.loads(
            (REPO / "docs/advanced_guides/learns_like_reference.json").read_text(encoding="utf-8")
        )["cells"]
        if cell["kind"] == "encoder"
    }
    page = (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8")
    rows = table_under(page, "## What the cheaper checks had said")
    assert [row[0].strip("`") for row in rows] == ["spunet", "ptv3", "litept"]
    for row in rows:
        name = row[0].strip("`")
        cell = cells["GSNet", f"backbone={name}"]
        assert row[1:3] == [cell["level1"], cell["outcome"]], name
        if name in ENCODER_RECORDS:
            assert row[3].startswith(f"−{below_shipped(name)[0]:.2f} AP"), name
        else:
            assert row[3] == "training", name
    # the swap trained under another method had passed the two checks as well
    other = cells["EconomicGrasp", "backbone=ptv3"]
    assert (other["level1"], other["outcome"]) == ("runs", "on a par")
    assert "All three passed both, and so did PTv3 under EconomicGrasp." in " ".join(page.split())


# -------------------------------------------------------------------- encoder or recipe


def recipe_ap(name: str, variant: str) -> dict[str, float]:
    """Seen / similar / novel AP of a recorded run of ``gsnet_recipe.yaml``."""
    path = REPO / f"{RECIPE_RECORDS[name]}_{variant}.json"
    return record_result.headline(json.loads(path.read_text(encoding="utf-8")))["AP"]


def test_the_two_by_two_of_encoder_and_recipe_prints_what_its_records_hold():
    """A row under GSNet's recipe repeats the row of the results; a row under the preset is
    a run of ``gsnet_recipe.yaml`` scored on the whole test split, with the weights that
    sweep trained and the loss of its last epoch; a run without records shows no number."""
    planned = {
        cell.choices["config"]: cell.config.select("runtime.work_dir")
        for cell in load_sweep(PROJECT / "gsnet_recipe.yaml").plan()
    }
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    results = {row[0].strip("`"): row for row in table_under(readme, "## Results")}
    losses = json.loads((PROJECT / "records/training_loss.json").read_text(encoding="utf-8"))
    assert set(losses["recipe_runs"]) == set(RECIPE_RECORDS)
    rows = table_under(readme, "## Encoder or recipe")
    assert [(row[0].strip("`"), row[1]) for row in rows] == [
        (name, recipe) for name in planned for recipe in ("GSNet's", "one-cycle")
    ]
    assert "*one-cycle* below" in readme  # the text above the table says which preset that is
    for row in rows:
        name, recipe = row[0].strip("`"), row[1]
        if recipe == "GSNet's":
            assert row[2:] == results[name][2:], name
        elif name not in RECIPE_RECORDS:
            assert not any(character.isdigit() for cell in row[2:] for character in cell), name
        else:
            for cell, variant in zip(row[2:4], ("raw", "cd")):
                path = REPO / f"{RECIPE_RECORDS[name]}_{variant}.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                assert record["variant"] == variant and not record["partial"], path
                assert record["scored"] == record["requested"] == 23040, path
                assert record["weights"]["path"].startswith(f"{planned[name]}/"), path
                numbers = record_result.headline(record)["AP"]
                assert cell == " / ".join(f"{value:.2f}" for value in numbers.values()), path
            assert len(losses["recipe_runs"][name]) == losses["epochs"]
            assert row[4] == f"{losses['recipe_runs'][name][-1]:.2f}", name


def test_what_the_two_by_two_says_is_computed_from_its_records():
    """What the recipe does to the shipped encoder, in the project and on the evidence
    page."""
    losses = json.loads((PROJECT / "records/training_loss.json").read_text(encoding="utf-8"))
    shipped = [
        value - encoder_ap("sunet14", variant)[split]
        for variant in ("raw", "cd")
        for split, value in recipe_ap("sunet14", variant).items()
    ]
    assert max(abs(value) for value in shipped) < 1.12, "inside what one seed moves"
    project = " ".join((PROJECT / "README.md").read_text(encoding="utf-8").split())
    page = " ".join(
        (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8").split()
    )
    moved = f"{signed(shipped[0])} AP on seen and {signed(shipped[2])} on novel scenes"
    assert moved in project and moved in page
    fitted = (
        f"({losses['recipe_runs']['sunet14'][-1]:.2f} against {losses['runs']['sunet14'][-1]:.2f})"
    )
    assert fitted in project
    # PTv3: what the recipe gains it, and what is left to the shipped encoder under the
    # same recipe, on seen and on novel scenes without the collision filter
    gained = recipe_ap("ptv3", "raw")["seen"] - encoder_ap("ptv3", "raw")["seen"]
    left = {
        split: recipe_ap("sunet14", "raw")[split] - recipe_ap("ptv3", "raw")[split]
        for split in ("seen", "novel")
    }
    before = below_shipped("ptv3")[0]
    assert 0.5 < (before - left["seen"]) / before < 1, "more than half of the gap"
    recipe_losses = losses["recipe_runs"]
    assert recipe_losses["ptv3"][-1] > recipe_losses["sunet14"][-1]
    for sentence in (
        f"It gains {gained:.1f} AP on seen scenes and ends {left['seen']:.1f} AP below the "
        "shipped encoder under the same recipe",
        f"on novel scenes the two are {left['novel']:.2f} apart",
        f"Its loss, {recipe_losses['ptv3'][-1]:.2f}, stays above the shipped encoder's "
        f"{recipe_losses['sunet14'][-1]:.2f}",
    ):
        assert sentence in project and sentence in page, sentence
    assert f"where it was {before:.1f} below under GSNet's" in project
    checks = table_under(
        (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8"),
        "## What the cheaper checks had said",
    )
    assert f"−{left['seen']:.2f} under one-cycle" in checks[1][3]
    # LitePT: a loss that falls in every epoch, what the recipe gains it on seen scenes
    # and what is left to the shipped encoder under the same recipe: a third of the gap less
    steady = recipe_losses["litept"]
    assert all(later < earlier for earlier, later in pairwise(steady))
    gained = recipe_ap("litept", "raw")["seen"] - encoder_ap("litept", "raw")["seen"]
    left = recipe_ap("sunet14", "raw")["seen"] - recipe_ap("litept", "raw")["seen"]
    before = below_shipped("litept")[0]
    assert 0.3 < (before - left) / before < 0.4, "a third of the gap"
    for sentence in (
        f"Its loss now falls in every epoch and ends at {steady[-1]:.2f}.",
        f"It gains {gained:.1f} AP on seen scenes and ends {left:.1f} AP below the shipped "
        "encoder under the same recipe",
    ):
        assert sentence in project and sentence in page, sentence
    assert f"where it was {before:.1f} below under GSNet's" in project
    assert checks[2][3] == f"−{before:.2f} AP; −{left:.2f} under one-cycle"


def test_the_evidence_page_shows_the_two_by_two_of_the_project():
    include = (REPO / "docs/includes/trained_recipes.md").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    assert include.strip() in readme
    assert table_under(include, "|") == table_under(readme, "## Encoder or recipe")
    page = (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8")
    assert page.count('--8<-- "docs/includes/trained_recipes.md"') == 1


# ------------------------------------------------------- the same swap under EconomicGrasp

#: EconomicGrasp with the encoder it ships (the model zoo's row) and with PTv3 → the config
#: that was trained and the stem of the two records.
ECONOMICGRASP_RUNS = {
    "sunet14": (
        "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e",
        "configs/economicgrasp/records/economicgrasp_sunet14_g1b-realsense_10e_reproduced_test",
    ),
    "ptv3": (
        "experimental/economicgrasp_ptv3_g1b-realsense_10e",
        "projects/encoder_comparison/records/economicgrasp_ptv3_test",
    ),
}


def economicgrasp_record(name: str, variant: str) -> dict:
    path = REPO / f"{ECONOMICGRASP_RUNS[name][1]}_{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def below_shipped_economicgrasp() -> list[float]:
    """How far PTv3 ends below the encoder EconomicGrasp ships: seen, similar, novel without
    the collision filter, then with it."""
    return [
        value - record_result.headline(economicgrasp_record("ptv3", variant))["AP"][split]
        for variant in ("raw", "cd")
        for split, value in record_result.headline(economicgrasp_record("sunet14", variant))[
            "AP"
        ].items()
    ]


def test_the_swap_under_economicgrasp_prints_what_its_records_hold():
    """Both rows are a shipped config trained with seed 0 for its ten epochs and scored on
    the whole test split: the weights of a record were trained from the config of its row,
    and the loss is that of the last epoch."""
    from grasppanda.config import Config, resolve_config_path

    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    rows = {
        row[0].strip("`"): row
        for row in table_under(readme, "## The same swap under EconomicGrasp")
    }
    assert list(rows) == list(ECONOMICGRASP_RUNS)
    losses = json.loads((PROJECT / "records/training_loss.json").read_text(encoding="utf-8"))
    runs = losses["economicgrasp_runs"]
    assert set(runs) == set(rows)
    for name, row in rows.items():
        trained = Config.load(resolve_config_path(ECONOMICGRASP_RUNS[name][0])).digest()
        for cell, variant in zip(row[1:3], ("raw", "cd")):
            record = economicgrasp_record(name, variant)
            assert record["variant"] == variant and not record["partial"], name
            assert record["scored"] == record["requested"] == 23040, name
            assert record["seed"] == 0 and record["weights"]["epoch"] == 10, name
            assert record["weights"]["config_hash"] == trained, name
            numbers = record_result.headline(record)["AP"]
            assert cell == " / ".join(f"{value:.2f}" for value in numbers.values()), name
        assert len(runs[name]) == losses["epochs"]
        assert row[3] == f"{runs[name][-1]:.2f}", name
    # the shipped encoder's training is the ablation's run without a change: one record
    ablation = json.loads(
        (REPO / "projects/finegrasp_ablation/records/training_loss.json").read_text(
            encoding="utf-8"
        )
    )
    assert runs["sunet14"] == ablation["runs"]["none"]


def test_what_the_swap_under_economicgrasp_says_is_computed_from_its_records():
    """The gap under EconomicGrasp beside the two under GSNet, and the loss that shows PTv3
    has not fitted the training scenes there either: in the project and on the evidence
    page."""
    gap = below_shipped_economicgrasp()
    under_gsnet = below_shipped("ptv3")[0]
    one_cycle = recipe_ap("sunet14", "raw")["seen"] - recipe_ap("ptv3", "raw")["seen"]
    assert min(gap) > 1.12, "lower on every split, outside what one seed moves"
    assert abs(gap[0] - one_cycle) < 1.12 < under_gsnet - gap[0], "the smaller of the two gaps"
    losses = json.loads((PROJECT / "records/training_loss.json").read_text(encoding="utf-8"))
    runs = losses["economicgrasp_runs"]
    assert runs["sunet14"][4] < runs["ptv3"][-1] < runs["sunet14"][3], "after four, not five"
    project = " ".join((PROJECT / "README.md").read_text(encoding="utf-8").split())
    page = " ".join(
        (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8").split()
    )
    fitted = (
        f"After ten epochs its loss, {runs['ptv3'][-1]:.2f}, is where the shipped encoder was "
        f"after four ({runs['sunet14'][3]:.2f})"
    )
    assert fitted in project and fitted in page
    assert (
        f"PTv3 ends {gap[0]:.1f} AP lower on seen scenes, {gap[2]:.1f} on novel." in project
        and f"the one-cycle recipe ({one_cycle:.1f}), not the one under GSNet's own "
        f"({under_gsnet:.1f})"
        in project
    )
    assert (
        f"PTv3: {gap[0]:.1f} AP lower on seen scenes." in page
        and f"under one-cycle ({one_cycle:.1f}), not the one under its own recipe "
        f"({under_gsnet:.1f})"
        in page
    )


def test_the_evidence_page_shows_the_swap_under_economicgrasp():
    include = (REPO / "docs/includes/trained_economicgrasp.md").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    assert include.strip() in readme
    assert table_under(include, "|") == table_under(readme, "## The same swap under EconomicGrasp")
    page = (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8")
    assert page.count('--8<-- "docs/includes/trained_economicgrasp.md"') == 1


# ------------------------------------------------------------------ FineGrasp's changes

ABLATION = REPO / "projects/finegrasp_ablation"
#: Run of ``changes.yaml`` → stem of its two records. The run without a change is the
#: model trained here of EconomicGrasp, whose records are the model zoo's.
ABLATION_RECORDS = {
    "none": "configs/economicgrasp/records/economicgrasp_sunet14_g1b-realsense_10e_reproduced_test",
    "graspness": "projects/finegrasp_ablation/records/graspness_test",
    "normals": "projects/finegrasp_ablation/records/normals_test",
    "multi-scale": "projects/finegrasp_ablation/records/multi-scale_test",
    "all": "projects/finegrasp_ablation/records/all_test",
}
#: FineGrasp in the model zoo: the authors' checkpoint and the model trained here with
#: FineGrasp's own recipe → stem of the two records.
FINEGRASP_RECORDS = {
    provenance: f"configs/finegrasp/records/finegrasp_sunet14_g1b-realsense_10e_{provenance}_test"
    for provenance in ("author-ckpt", "reproduced")
}


def ablation_record(name: str, variant: str) -> dict:
    path = REPO / f"{ABLATION_RECORDS[name]}_{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ablation_rows() -> dict[str, list[str]]:
    """Rows of the project's results by the name of the change."""
    rows = table_under((ABLATION / "README.md").read_text(encoding="utf-8"), "## Results")
    return {row[0].partition(":")[0].strip("`"): row for row in rows}


def test_the_ablation_prints_what_its_records_hold():
    """Every row of its results is a training scored on the whole test split with seed 0:
    the runs of the sweep by the weights in their work directory, the run without a change
    by the records of EconomicGrasp; a run without records shows no AP."""
    planned = {
        cell.choices["change"]: cell.config.select("runtime.work_dir")
        for cell in load_sweep(ABLATION / "changes.yaml").plan()
    }
    rows = ablation_rows()
    assert list(rows) == ["none", *planned]
    for name, row in rows.items():
        if name not in ABLATION_RECORDS:
            assert not any(character.isdigit() for cell in row[1:] for character in cell), name
            continue
        for cell, variant in zip(row[1:3], ("raw", "cd")):
            record = ablation_record(name, variant)
            assert record["variant"] == variant and not record["partial"], name
            assert record["scored"] == record["requested"] == 23040, name
            assert record["seed"] == 0, name
            if name in planned:
                assert record["weights"]["path"].startswith(f"{planned[name]}/"), name
            numbers = record_result.headline(record)["AP"]
            assert cell == " / ".join(f"{value:.2f}" for value in numbers.values()), name


def from_no_change(name: str) -> list[float]:
    """What a change adds to EconomicGrasp: seen, similar, novel without the collision
    filter, then with it."""
    return [
        value - record_result.headline(ablation_record("none", variant))["AP"][split]
        for variant in ("raw", "cd")
        for split, value in record_result.headline(ablation_record(name, variant))["AP"].items()
    ]


def below_the_authors(stem: str) -> list[float]:
    """How far a FineGrasp model ends below the authors' checkpoint: seen, similar, novel
    without the collision filter, then with it. ``stem`` names its two records."""

    def ap(stem: str, variant: str) -> dict[str, float]:
        record = json.loads((REPO / f"{stem}_{variant}.json").read_text(encoding="utf-8"))
        return record_result.headline(record)["AP"]

    return [
        value - ap(stem, variant)[split]
        for variant in ("raw", "cd")
        for split, value in ap(FINEGRASP_RECORDS["author-ckpt"], variant).items()
    ]


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{abs(value):.2f}"


def test_what_the_ablation_says_is_computed_from_its_records():
    readme = " ".join((ABLATION / "README.md").read_text(encoding="utf-8").split())
    graspness = from_no_change("graspness")
    assert max(abs(value) for value in graspness) < 1.12, "inside what one seed moves"
    for sentence in (
        f"by more than {max(abs(value) for value in graspness):.1f} AP",
        f"{signed(graspness[0])} on seen and {signed(graspness[1])} on similar",
    ):
        assert sentence in readme, sentence
    losses = json.loads((ABLATION / "records/training_loss.json").read_text(encoding="utf-8"))
    runs = losses["runs"]
    assert set(runs) <= set(ABLATION_RECORDS)
    assert all(len(values) == losses["epochs"] == 10 for values in runs.values())
    # the normals: seen scenes without the collision filter just beyond what a seed moves,
    # the five other numbers inside it, and the training loss of EconomicGrasp
    normals = from_no_change("normals")
    assert normals[0] == min(normals) < -1.12 < min(normals[1:])
    assert max(normals) < 1.12, "no gain"
    assert abs(runs["normals"][-1] - runs["none"][-1]) < 0.05
    for sentence in (
        f"{signed(min(normals[1:]))} to {signed(max(normals[1:]))} AP",
        f"filter, is {signed(normals[0])}",
        f"{runs['normals'][-1]:.2f} against {runs['none'][-1]:.2f}",
    ):
        assert sentence in readme, sentence
    # the four-radius crop: lower by more than a seed moves, with the lower training loss
    scales = from_no_change("multi-scale")
    assert max(scales) < -1.12, "lower on every split, outside what one seed moves"
    assert runs["multi-scale"][-1] < runs["none"][-1]
    for sentence in (
        f"{signed(scales[0])} AP on seen, {signed(scales[1])} on similar and "
        f"{signed(scales[2])} on novel scenes",
        f"{runs['multi-scale'][-1]:.2f} against EconomicGrasp's {runs['none'][-1]:.2f}",
    ):
        assert sentence in readme, sentence
    # all three, the FineGrasp model: seen scenes without the filter at the edge of what a
    # seed moves, nothing else beyond it
    everything = from_no_change("all")
    assert everything[0] == max(everything) and 1.12 < everything[0] < 1.5
    assert max(abs(value) for value in everything[1:]) < 1.12
    # below the authors' checkpoint under EconomicGrasp's recipe as under FineGrasp's own
    below = {
        "all": below_the_authors(ABLATION_RECORDS["all"]),
        "own": below_the_authors(FINEGRASP_RECORDS["reproduced"]),
    }
    assert min(below["all"]) > 0.9 and min(below["own"]) > 1.12
    for sentence in (
        f"it ends {signed(everything[0])} AP on seen scenes",
        f"{signed(everything[1])} on similar and {signed(everything[2])} on novel",
        "with the filter " + " / ".join(signed(value) for value in everything[3:]),
        f"{min(below['all']):.1f} to {max(below['all']):.1f} AP below the authors' checkpoint",
        f"own recipe is {min(below['own']):.1f} to {max(below['own']):.1f} below",
        # the four-radius crop alone against the crop with the two other changes
        f"`multi-scale` alone loses {-max(scales[:3]):.2f} to {-min(scales[:3]):.2f} AP",
    ):
        assert sentence in readme, sentence
    assert min(everything[:2]) > -0.5 and max(scales[:2]) < -1.12, "none of that on seen, similar"
    # FineGrasp's own README draws the conclusion from the same records
    finegrasp = " ".join((REPO / "configs/finegrasp/README.md").read_text(encoding="utf-8").split())
    assert (
        f"under EconomicGrasp's the model ends {min(below['all']):.1f} to "
        f"{max(below['all']):.1f} AP below the checkpoint" in finegrasp
    )


def test_the_evidence_page_shows_the_ablation_of_the_project():
    """Its table is the project's (an include), and its three findings are the project's
    sentences in short: the same numbers, computed from the records there."""
    include = (REPO / "docs/includes/trained_changes.md").read_text(encoding="utf-8")
    readme = (ABLATION / "README.md").read_text(encoding="utf-8")
    assert include.strip() in readme
    assert table_under(include, "|") == table_under(readme, "## Results")
    page = (REPO / "docs/advanced_guides/trained_swaps.md").read_text(encoding="utf-8")
    assert page.count('--8<-- "docs/includes/trained_changes.md"') == 1
    flat = " ".join(page.split())
    normals, scales = from_no_change("normals"), from_no_change("multi-scale")
    everything = from_no_change("all")
    below = {
        "all": below_the_authors(ABLATION_RECORDS["all"]),
        "own": below_the_authors(FINEGRASP_RECORDS["reproduced"]),
    }
    for sentence in (
        f"{signed(normals[0])} AP on seen scenes for `normals`",
        f"`multi-scale` alone ends {-max(scales[:3]):.2f} to {-min(scales[:3]):.2f} AP lower",
        f"{signed(everything[0])} AP on seen and {signed(everything[2])} on novel scenes",
        f"{min(below['all']):.1f} to {max(below['all']):.1f} AP below the authors' checkpoint",
        f"own recipe is {min(below['own']):.1f} to {max(below['own']):.1f} below",
    ):
        assert sentence in flat, sentence
