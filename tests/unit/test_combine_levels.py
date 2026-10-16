"""Levels of a measured cell and classes of an option (``grasppanda.combine.levels``).

The records of judge 2 below are real: probes of 2026-10-15 on the frames of the split
``sample`` (``$GRASPPANDA_CACHE/probe_grid/w03-check`` and ``w1a-dry``), exported with
``tools/probe_grid.compact`` and trimmed to the fields a level is decided from. They are
not in the committed records; they are the examples the definitions were written against.
"""

from __future__ import annotations

import pytest

from grasppanda.combine.levels import (
    ACTIVE_SHARE,
    CAVEAT,
    CLASS_WORDS,
    FELL_BY,
    LEVELS,
    NOT_CHECKED,
    NOT_REPEATED,
    UNDECIDED,
    cell_levels,
    option_class,
    refusal_kind,
    refusal_kinds,
)
from grasppanda.combine.probe import FLAT_BELOW

SEED = ("GSNet", "EconomicGrasp", "FineGrasp", "GraspNet-baseline", "Scale-Balanced-Grasp")

# --------------------------------------------------------------- real judge-2 records

GSNET = {
    "id": "reference/GSNet",
    "verdict": "runs",
    "judge": 2,
    "iterations": 20,
    "grasps_per_frame": [342, 369],
    "trend": {
        "loss/objectness": [0.7893, 0.617, 0.218],
        "loss/graspness": [5.176, 0.3662, 0.929],
        "loss/view": [6.291, 0.3737, 0.941],
        "loss/score": [1.515, 0.0883, 0.942],
        "loss/width": [3.654, 0.2216, 0.939],
        "loss": [17.43, 1.667, 0.904],
    },
    "no_gradient": [],
    "unchanged": [],
    "score_quantiles": [[0.1138, 0.1746, 0.2684], [0.1128, 0.1943, 0.2921]],
}
POINTNEXT = {
    "id": "encoder/GSNet/PointNeXt@pointnext",
    "verdict": "runs-empty",
    "judge": 2,
    "iterations": 20,
    "grasps_per_frame": [0, 0],
    "trend": {
        "loss/objectness": [0.722, 0.6685, 0.074],
        "loss/graspness": [0.8836, 0.2261, 0.744],
        "loss/view": [2.824, 0.9641, 0.659],
        "loss/score": [1.359, 0.08886, 0.935],
        "loss/width": [2.072, 0.198, 0.904],
        "loss": [7.86, 2.146, 0.727],
    },
    "no_gradient": [],
    "unchanged": [],
    "score_quantiles": [None, None],
}
TWO_VECTOR = {
    "id": "stage/GraspNet-baseline/view/two-vector",
    "verdict": "runs",
    "judge": 2,
    "iterations": 20,
    "grasps_per_frame": [1024, 1024],
    "trend": {
        "loss/view": [1.35, 0.3381, 0.75],
        "loss/view_frame": [1.037, 1.235, -0.19],
        "loss/score": [0.01787, 0.03113, -0.742],
        "loss/angle": [0.469, 0.4112, 0.123],
        "loss/width": [0.6806, 0.1962, 0.712],
        "loss/tolerance": [0.8189, 0.3918, 0.522],
        "loss": [3.337, 1.901, 0.43],
    },
    "active": {"loss/view_frame": 14},
    "flags": {"loss/view_frame": ["zero_at_end"]},
    "no_gradient": [],
    "unchanged": [],
    "score_quantiles": [[-11.24, 0.3176, 10.34], [-14.35, -0.136, 10.44]],
}
BALL = {
    "id": "stage/EconomicGrasp/grouper/ball",
    "verdict": "runs",
    "judge": 2,
    "iterations": 20,
    "grasps_per_frame": [1024, 1024],
    "trend": {
        "loss/objectness": [0.7845, 0.574, 0.268],
        "loss/graspness": [5.216, 0.3399, 0.935],
        "loss/view": [6.545, 0.7124, 0.891],
        "loss/angle": [2.733, 0.801, 0.707],
        "loss/depth": [1.139, 0.4223, 0.629],
        "loss/score": [1.305, 0.6517, 0.5],
        "loss/width": [1.395, 0.6965, 0.501],
        "loss": [19.12, 4.198, 0.78],
    },
    "no_gradient": [],
    "unchanged": [],
    "score_quantiles": [[-2.609, -2.153, -1.776], [-2.451, -2.081, -1.339]],
}
FINEGRASP_COLLISION = {
    "id": "post/FineGrasp/collision",
    "verdict": "runs-unstable",
    "judge": 2,
    "unstable": "4 of 20 iterations had no real seed (iteration 2, 3, 10, 11).",
    "iterations": 20,
    "grasps_per_frame": [1024, 1024],
    "trend": {
        "loss/objectness": [0.8058, 0.676, 0.161],
        "loss/view": [5.497, 0.6185, 0.887],
        "loss/score": [2.563, 0.7692, 0.7],
        "loss": [19.76, 5.734, 0.71],
    },
    "active": {"loss/view": 16, "loss/score": 15},
    "no_gradient": [],
    "unchanged": [],
    "score_quantiles": [[-4.584, -4.584, -4.584], [-4.584, -4.584, -4.584]],
}
FINEGRASP_ADAMW = {
    "id": "recipe/FineGrasp/optimizer/adamw",
    "verdict": "runs",
    "judge": 2,
    "iterations": 20,
    "grasps_per_frame": [1024, 1024],
    "trend": {
        "loss/objectness": [0.8058, 0.6364, 0.21],
        "loss/angle": [1.429, 1.025, 0.283],
        "loss/score": [2.563, 0.8584, 0.665],
        "loss": [19.76, 4.863, 0.754],
    },
    "no_gradient": [],
    "unchanged": [],
    "score_quantiles": [[-1.691, -1.686, -1.679], [-1.693, -1.685, -1.678]],
}


def states(record: dict, numbers: dict | None = None) -> list[str]:
    found = cell_levels(record, numbers)
    return [found.states[level] for level in LEVELS]


def test_the_definitions_are_in_one_place_with_their_thresholds():
    assert list(LEVELS) == ["composes", "runs", "learns", "predicts"]
    assert FELL_BY == FLAT_BELOW  # a term that fell is a term that is not flat
    assert f"{FELL_BY * 100:g} %" in LEVELS["learns"]
    assert f"{ACTIVE_SHARE * 100:g} %" in LEVELS["learns"]
    assert "zero_at_end" in LEVELS["learns"] and "repeat" in LEVELS["runs"]
    assert CAVEAT == (
        "one run, seed 0, 20 iterations, 8 frames of one scene; a class counts verdicts, "
        "not accuracy"
    )
    assert set(UNDECIDED) >= {NOT_CHECKED, NOT_REPEATED, "needs consent", "not probed"}
    assert list(CLASS_WORDS) == ["universal", "most", "some", "specific", "none"]


def test_the_real_probes_of_judge_two_are_decided_as_the_audit_reads_them():
    # GSNet as shipped: every term fell 22-94 %, grasps on both frames, scores spread.
    assert states(GSNET) == ["yes", "yes", "yes", "yes"]
    # PointNeXt under GSNet: it learns (objectness only 7 %, above the 5 % margin), it
    # predicts nothing after 20 iterations.
    found = cell_levels(POINTNEXT)
    assert states(POINTNEXT) == ["yes", "yes", "yes", "no"]
    assert found.why["predicts"] == ["no grasp on any frame"]
    # The two-vector view under GraspNet-baseline: the frame term is 0 in 6 of 20
    # iterations, also at the last one; that alone does not decide (active 14 of 20), but
    # over its active iterations it rose 19 %, and the score term rose 74 % from a low
    # first value.
    found = cell_levels(TWO_VECTOR)
    assert states(TWO_VECTOR) == ["yes", "yes", "no", "yes"]
    assert found.why["learns"] == ["view_frame rose 19 %", "score rose 74 %"]
    assert found.notes == ["view_frame is 0 at the last iteration (active in 14 of 20)"]
    # The ball grouper under EconomicGrasp: at 20 iterations every term fell.
    assert states(BALL) == ["yes", "yes", "yes", "yes"]
    # AdamW under FineGrasp: scores nearly collapsed but not identical: it predicts.
    assert states(FINEGRASP_ADAMW) == ["yes", "yes", "yes", "yes"]


def test_an_unstable_run_counts_only_after_a_repeat():
    assert states(FINEGRASP_COLLISION) == ["yes", NOT_REPEATED, NOT_REPEATED, NOT_REPEATED]
    again = {**FINEGRASP_COLLISION, "repeat": {"verdict": "runs-unstable", "unstable": "..."}}
    found = cell_levels(again)
    assert states(again) == ["yes", "no", "no", "no"] and found.repeated == "runs-unstable"
    assert found.why["runs"] == ["unstable, and the repeat was `runs-unstable`"]
    # A repeat that ran clean: the cell runs, and the higher levels are read from it.
    clean = {**FINEGRASP_COLLISION, "repeat": {**GSNET, "verdict": "runs"}}
    assert states(clean) == ["yes", "yes", "yes", "yes"]
    assert cell_levels(clean).notes[0].startswith("unstable in the first probe")
    collapsed = {**FINEGRASP_COLLISION, "repeat": {**FINEGRASP_COLLISION, "verdict": "runs"}}
    found = cell_levels(collapsed)
    assert found.states["predicts"] == "no"
    assert found.why["predicts"] == [
        "every score of frame 1 is -4.584",
        "every score of frame 2 is -4.584",
    ]
    # A bare verdict string and a repeat without numbers of judge 2.
    assert states({**FINEGRASP_COLLISION, "repeat": "runs"})[1:] == [
        "yes",
        NOT_CHECKED,
        NOT_CHECKED,
    ]


def test_a_record_of_the_first_judge_runs_at_most():
    first = {"id": "x", "verdict": "runs", "grasps_per_frame": [0, 0], "losses": {"loss": [2, 1]}}
    assert states(first) == ["yes", "yes", NOT_CHECKED, NOT_CHECKED]
    assert states({"id": "x", "verdict": "runs-unstable"})[1] == NOT_REPEATED
    failed = {"id": "x", "verdict": "fails", "failure": {"phase": "build"}}
    assert states(failed) == ["yes", "no", "no", "no"]
    crashed = {"id": "x", "verdict": "fails", "failure": {"phase": "compose"}}
    assert states(crashed) == ["no", "no", "no", "no"]
    # A cell that shares a probe is decided on the numbers of the cell it names.
    sharer = {"id": "y", "verdict": "runs", "probed_as": GSNET["id"]}
    assert states(sharer, GSNET) == ["yes", "yes", "yes", "yes"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"no_gradient": ["head.score_net"]}, "no gradient: head.score_net"),
        ({"unchanged": ["backbone"]}, "unchanged: backbone"),
        ({"flags": {"loss/score": ["rising"]}}, "score rising"),
        ({"flags": {"loss/score": ["constant"]}}, "score constant"),
        ({"active": {"loss/score": 9}}, "score active in 9 of 20 iterations"),
        ({"trend": {"loss/score": [None, None, None]}}, "score never active"),
        ({"trend": {"loss/score": [1.0, 0.98, 0.02]}}, "score fell only 2 %"),
        ({"trend": {}}, "no loss term recorded"),
    ],
)
def test_what_keeps_a_probe_from_learning(change, reason):
    found = cell_levels({**GSNET, **change})
    assert found.states["learns"] == "no" and reason in found.why["learns"]


def test_what_keeps_a_probe_from_predicting():
    found = cell_levels(
        {**GSNET, "grasps_per_frame": [12, 0], "score_quantiles": [[0.1, 0.2, 0.3], None]}
    )
    assert found.why["predicts"] == ["no grasp on frame 2"]
    found = cell_levels({**GSNET, "score_quantiles": [[0.1, 0.2, 0.3]]})
    assert found.why["predicts"] == ["the scores were not recorded for every frame"]


def test_a_refusal_carries_its_kind_and_a_policy_is_no_failure():
    structural = {"id": "x", "verdict": "refused", "refusal": ["A emits X, but B expects Y."]}
    assert states(structural) == ["no"] * 4 and refusal_kind(structural) == ""
    kinds = {**structural, "refusal_kinds": ["structural"]}
    assert refusal_kind(kinds) == "structural"
    assert cell_levels(kinds).why["composes"] == ["refused (structural)"]
    lifted = {
        **structural,
        "refusal": ["a", "b"],
        "refusal_kinds": [
            {"kind": "not-implemented", "lift": "compose rewrites the labels"},
            "policy",
        ],
    }
    assert refusal_kind(lifted) == "not-implemented"
    assert refusal_kinds(lifted)[0] == {
        "message": "a",
        "kind": "not-implemented",
        "lift": "compose rewrites the labels",
    }
    mapping = {**structural, "refusal_kinds": {"A emits X, but B expects Y.": "structural"}}
    assert refusal_kind(mapping) == "structural"
    lifts = {**structural, "refusal_kinds": ["not-implemented"], "refusal_lifts": ["an adapter"]}
    assert refusal_kinds(lifts)[0]["lift"] == "an adapter"
    # Records exported before compose reported kinds: the remote-code policy is still told.
    remote = {
        "id": "family-slot/HGGD/image_encoder/hf-radio-v3-base",
        "verdict": "refused",
        "refusal": [
            "model.image_encoder: HFImageEncoder: `model_args.trust_remote_code: true` runs "
            "Python code ... state that you accept it with `accept_remote_code: true`."
        ],
    }
    assert refusal_kind(remote) == "policy"
    assert states(remote) == ["needs consent"] * 4
    bare = {**structural, "refusal_kinds": ["not-a-combination"]}
    assert states(bare) == ["not a combination"] * 4


def test_classes_count_methods_level_by_level():
    yes = dict.fromkeys(SEED, "yes")
    assert option_class("runs", yes, SEED, SEED).name == "universal"
    found = option_class("runs", {**yes, "GSNet": "no", "FineGrasp": "no"}, SEED, SEED)
    assert (found.name, found.reached, found.decided) == ("most", 3, 5)
    assert (
        option_class(
            "runs", {**dict.fromkeys(SEED, "no"), "GSNet": "yes", "FineGrasp": "yes"}, SEED, SEED
        ).name
        == "some"
    )
    assert (
        option_class("runs", {**dict.fromkeys(SEED, "no"), "GSNet": "yes"}, SEED, SEED).name
        == "specific"
    )
    assert option_class("runs", dict.fromkeys(SEED, "no"), SEED, SEED).name == "none"
    # Undecided cells are named, never counted as failures.
    found = option_class("runs", {**yes, "GSNet": NOT_REPEATED}, SEED, SEED)
    assert (
        found.name == "4 of 4 probed" and found.text == "4 of 4 probed, +1 unstable, not repeated"
    )
    found = option_class("learns", dict.fromkeys(SEED, NOT_CHECKED), SEED, SEED)
    assert (found.name, found.decided, found.text) == (NOT_CHECKED, 0, NOT_CHECKED)
    mixed = option_class(
        "learns", {**dict.fromkeys(SEED, NOT_CHECKED), "GSNet": NOT_REPEATED}, SEED, SEED
    )
    assert mixed.name == "undecided" and mixed.text == "4 not checked, 1 unstable, not repeated"
    # A missing record is not probed; a slot probed under fewer methods is relative.
    found = option_class("runs", {"GSNet": "yes", "HGGD": "yes"}, (*SEED, "HGGD"), SEED)
    assert found.name == "2 of 2 probed" and found.text == "2 of 2 probed, +4 not probed"
    assert option_class("runs", {"HGGD": "yes"}, ["HGGD"], SEED).name == "1 of 1 probed"
    every = {**yes, "HGGD": "yes"}
    assert option_class("runs", every, (*SEED, "HGGD"), SEED).name == "universal"
    # A policy refusal under the one method of a slot is no "none".
    assert (
        option_class("composes", {"HGGD": "needs consent"}, ["HGGD"], SEED).name == "needs consent"
    )


def test_a_level_the_method_as_shipped_does_not_reach_is_undecided_under_it():
    """FineGrasp at the constant rate: its own reference gives every grasp one score, so
    `predicts` says nothing about an option under FineGrasp. A cell held against such a
    reference is undecided at that level, not `no`; a cell that reaches it keeps it."""
    from grasppanda.combine.levels import BELOW_REFERENCE
    from grasppanda.combine.options import Measured

    flat = [[-0.29, -0.29, -0.29], [-0.29, -0.29, -0.29]]
    reference = {**GSNET, "id": "reference/FineGrasp", "method": "FineGrasp"}
    reference["score_quantiles"] = flat
    collapsed = {**GSNET, "id": "encoder/FineGrasp/x", "method": "FineGrasp"}
    collapsed["score_quantiles"] = flat
    spread = {**GSNET, "id": "encoder/FineGrasp/y", "method": "FineGrasp"}
    under_gsnet = {**GSNET, "id": "encoder/GSNet/x", "method": "GSNet", "score_quantiles": flat}
    shipped = {**GSNET, "method": "GSNet"}
    measured = Measured({"cells": [reference, collapsed, spread, under_gsnet, shipped]})
    assert measured.levels(reference).states["predicts"] == "no"  # the reference itself
    held = measured.levels(collapsed)
    assert held.states["predicts"] == BELOW_REFERENCE and "predicts" not in held.why
    assert any(
        note.startswith("predicts: not reached, as the method as shipped") for note in held.notes
    )
    assert measured.levels(spread).states["predicts"] == "yes"
    # Under a method whose reference predicts, the same numbers are a plain `no`.
    assert measured.levels(under_gsnet).states["predicts"] == "no"
    assert BELOW_REFERENCE in UNDECIDED
    # A class does not count the method where the level is undecided.
    found = option_class("predicts", {"FineGrasp": BELOW_REFERENCE, "GSNet": "yes"}, SEED, SEED)
    assert found.text.startswith("1 of 1 probed")
