"""Where an option works, as every page says it: the reach of an option over the methods it
applies to (``grasppanda.combine.reach``), held to the measured records."""

from __future__ import annotations

import pytest

from grasppanda.combine import reach as reach_module
from grasppanda.combine.grid import FAMILIES, KNOWN_FAILURES, METHODS
from grasppanda.combine.options import _class_index, measured
from grasppanda.combine.reach import Level2, Reach, level2, method_order, reach

LEVELS = ("composes", "runs", "learns", "predicts")


@pytest.fixture(scope="module")
def records():
    return measured()


def test_methods_are_in_the_order_of_their_papers_then_the_other_family():
    assert method_order() == (
        "GraspNet-baseline",
        "GSNet",
        "Scale-Balanced-Grasp",
        "EconomicGrasp",
        "FineGrasp",
        "HGGD",
    )
    assert method_order() == (*METHODS, *FAMILIES)


def test_every_known_failure_is_still_the_failure_of_its_record(records):
    for cell, (words, message) in KNOWN_FAILURES.items():
        record = records.get(cell)
        assert record is not None, cell
        assert record["verdict"] == "fails", cell
        assert message in record["failure"]["message"], cell
        assert words


def test_the_denominator_is_every_swap_and_nothing_is_dropped(records):
    """Every method an option has a cell under is counted: as reached, missed (with why),
    undecided (with why), or as the method that ships it."""
    for (slot, name), by_method in _class_index().items():
        if not any(by_method.values()):
            continue
        for level in LEVELS:
            found = reach(slot, name, level, records)
            named = {*found.own, *found.reached, *found.missed, *found.undecided}
            assert named == set(by_method), (slot, name, level)
            assert len(named) == len(found.own) + len(found.reached) + len(found.missed) + len(
                found.undecided
            ), (slot, name, level)
            if found.tone == "ok":
                assert set(found.reached) == set(found.swaps), (slot, name, level)


def test_a_refused_or_failed_swap_stays_named_at_every_later_level(records):
    """A refusal that waits for the user's consent, or a request that is no combination,
    is undecided at every level instead; every other refusal and failure is missed."""
    from grasppanda.combine.levels import refusal_kind

    for (slot, name), by_method in _class_index().items():
        for method, cells in by_method.items():
            recorded = [records.get(cell) for cell in cells if records.get(cell)]
            verdicts = {record["verdict"] for record in recorded}
            if not verdicts or not verdicts <= {"refused", "fails"}:
                continue
            found = [reach(slot, name, level, records) for level in LEVELS[1:]]
            if method in found[0].own:
                continue
            kinds = {refusal_kind(record) for record in recorded if record["verdict"] == "refused"}
            if kinds and kinds <= {"policy", "not-a-combination"}:
                assert all(method in item.undecided for item in found), (slot, name, method)
            else:
                assert all(method in item.missed for item in found), (slot, name, method)


def test_the_method_that_ships_an_option_is_not_counted_as_a_swap(records):
    compact = reach("assigner", "compact-label", "runs", records)
    assert set(compact.own) == {"EconomicGrasp", "FineGrasp"}
    assert compact.fraction == "0/3"
    economic = reach("head", "economic", "runs", records)
    assert set(economic.own) == {"EconomicGrasp", "FineGrasp"}
    assert "own in EconomicGrasp, FineGrasp" in economic.text


def test_words_for_refusals_failures_and_options_without_a_cell(records):
    amp = reach("hooks", "amp", "runs", records)
    assert amp.text == "0/6 · not supported yet" and amp.tone == "no"
    jitter = reach("augment", "jitter", "learns", records)
    assert jitter.missed == {"HGGD": "not supported yet"}
    swa = reach("hooks", "swa", "runs", records)
    assert set(swa.missed) == {"GraspNet-baseline", "Scale-Balanced-Grasp"}
    assert all(words.startswith("not verified") for words in swa.missed.values())
    painted = reach("backbone", "image-painted", "runs", records)
    assert painted.fraction == "no record" and "point painting" in painted.text
    one = reach("image_encoder", "dinov2-small-reg-depth-stem", "runs", records)
    assert one.fraction == "HGGD ✓"


def test_level_two_is_read_from_single_term_cells_and_skips_what_it_cannot_compare(records):
    assert level2("post", "collision", records).fraction == "—"
    assert level2("schedule", "onecycle", records).fraction == "—"
    ptv3 = level2("backbone", "ptv3", records)
    assert set(ptv3.outcomes) == set(METHODS)
    assert (
        ptv3.tone
        == reach_module.LEVEL2_TONE[
            min(ptv3.outcomes.values(), key=reach_module.LEVEL2_ORDER.index)
        ]
    )
    # the method that ships an option is not compared with itself
    assert "EconomicGrasp" not in level2("head", "economic", records).outcomes


def test_the_texts_are_plain():
    found = Reach("runs", ("GSNet", "HGGD"), reached=("GSNet",), missed={"HGGD": "fails"})
    assert found.text == "1/2 · not HGGD (fails)" and found.tone == "warn"
    assert Level2({"GSNet": "fails"}).text == "GSNet: did not train"
    mixed = Level2({"GSNet": "on a par", "HGGD": "diverging", "FineGrasp": "worse"})
    assert mixed.fraction == "1/3 on a par · 1 diverging · 1 worse" and mixed.tone == "no"


def test_level_two_leaves_out_what_cannot_be_compared(records):
    """A swap that did not run at level 1, and a cell that trained on a schedule of its own
    beside a constant-rate reference, are named and not counted."""
    spunet = level2("backbone", "spunet", records)
    assert "FineGrasp" not in spunet.outcomes
    assert spunet.excluded["FineGrasp"] == reach_module.LEVEL2_EXCLUDED["runs"]
    onecycle = level2("recipe", "adamw-onecycle", records)
    assert onecycle.excluded["GSNet"] == reach_module.LEVEL2_EXCLUDED["schedule"]
    # it writes a scheduler of its own: not compared under FineGrasp either, although
    # FineGrasp's reference ran on FineGrasp's own warm-up
    assert not onecycle.outcomes and onecycle.fraction == "not compared"
    for slot, name in _class_index():
        found = level2(slot, name, records)
        ran = set(reach(slot, name, "runs", records).reached)
        assert set(found.outcomes) <= ran, (slot, name)


def test_a_shipped_equivalent_is_the_method_as_shipped(records):
    """A preset that writes out the method's own component counts as own, not as a swap:
    its record has the parameters of the method's reference."""
    from grasppanda.combine.grid import SHIPPED_EQUIVALENTS

    for cell in SHIPPED_EQUIVALENTS:
        record = records.get(cell)
        reference = records.reference(record["method"])
        assert record["parameters"] == reference["parameters"], cell
    residual = reach("image_encoder", "hggd-residual", "runs", records)
    assert residual.own == ("HGGD",) and residual.fraction == "own only"
