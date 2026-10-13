"""The contract rules of the seed-based skeleton, one by one.

``stage_contract_problems`` compares whatever stages it is given, so each rule is
exercised on the two or three stages it concerns. Every refusal has to name both sides
of the conflict and a way out that exists in the registries. That the rules together
leave no combination that passes the check and fails later is the subject of
``test_swap_matrix.py``.
"""

from __future__ import annotations

import pytest

from grasppanda.config import Config, ConfigError
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems
from grasppanda.registry import ASSIGNERS, BACKBONES, DETECTORS, HEADS, PROPOSALS, VIEWS
from grasppanda.structures import ContractError, Gripper

OBJECT_SEEDS = {"type": "NearestLabelAssigner", "supervise": "object_seeds", "view_widths": True}


def only(problems: list[str]) -> str:
    assert len(problems) == 1, problems
    return problems[0]


def test_compatible_stages_have_no_problems_and_unknown_slots_are_an_error():
    assert stage_contract_problems() == []
    assert stage_contract_problems(grouper={"type": "CylinderGroup"}) == []
    assert (
        stage_contract_problems(
            backbone={"type": "SparseUNet14"},
            proposal={"type": "GraspnessFPS"},
            view={"type": "TemplateViewStage"},
            assigner={"type": "NearestLabelAssigner"},
            grouper={"type": "CylinderGroup"},
            head={"type": "SWADHead"},
        )
        == []
    )
    with pytest.raises(ValueError, match="unknown slot"):
        stage_contract_problems(neck={"type": "FeaturePropagation"})
    assert list(STAGE_REGISTRIES) == ["backbone", "proposal", "view", "assigner", "grouper", "head"]


# ------------------------------------------------------- where the scene features live


def test_a_subsampled_encoder_under_a_dense_proposal_names_the_adapter():
    message = only(
        stage_contract_problems(backbone={"type": "PointNet2"}, proposal={"type": "GraspnessFPS"})
    )
    assert "GraspnessFPS requires 'dense' scene features" in message
    assert "PointNet2 returns 'subsampled' ones" in message
    assert "{type: FeaturePropagation, backbone: {...}}" in message
    assert "FeaturePropagation" in BACKBONES  # the way out exists


def test_a_dense_encoder_under_a_subsampled_proposal_names_both_ways_out():
    message = only(
        stage_contract_problems(
            backbone={"type": "SparseUNet14"}, proposal={"type": "SubsampledSeeds"}
        )
    )
    assert "SubsampledSeeds requires 'subsampled' scene features" in message
    assert "SparseUNet14 returns 'dense' ones" in message
    assert (
        "GraspnessFPS" in message and "{type: FurthestPointSubsample, backbone: {...}}" in message
    )
    assert "FurthestPointSubsample" in BACKBONES and "GraspnessFPS" in PROPOSALS
    wrapped = {"type": "FurthestPointSubsample", "backbone": {"type": "SparseUNet14"}}
    assert stage_contract_problems(backbone=wrapped, proposal={"type": "SubsampledSeeds"}) == []


# ---------------------------------------------------------------------- feature axes


def test_an_axis_mismatch_lists_the_components_that_would_fit():
    message = only(
        stage_contract_problems(grouper={"type": "CylinderGroup"}, head={"type": "OperationHead"})
    )
    assert "CylinderGroup emits features with axes ('seed', 'channel')" in message
    assert "OperationHead expects ('seed', 'depth', 'channel')" in message
    assert "Groupers that emit ('seed', 'depth', 'channel'): " in message
    assert "DepthCylinderGroup" in message.split("Groupers that emit")[1]
    fitting = message.split("Head bundles that expect ('seed', 'channel'): ")[1]
    assert "SWADHead" in fitting and "EconomicHead" in fitting and "OperationHead" not in fitting


# ------------------------------------------------------------------ the label format


def test_assigner_and_head_agree_on_the_label_set_under_every_detector():
    message = only(
        stage_contract_problems(
            assigner={"type": "CompactLabelAssigner"}, head={"type": "SWADHead"}
        )
    )
    assert "CompactLabelAssigner returns 'compact' matched labels" in message
    assert "SWADHead is supervised with 'dense' ones" in message
    assert ASSIGNERS.get("NearestLabelAssigner").label_format == "dense"
    assert HEADS.get("EconomicHead").label_format == "compact"


def test_a_view_stage_states_which_label_sets_it_can_read():
    assert VIEWS.get("TemplateViewStage").label_formats == ("dense", "compact")
    assert VIEWS.get("ObjectnessViewStage").label_formats == ("dense",)
    compact = {"type": "CompactLabelAssigner"}
    assert stage_contract_problems(view={"type": "TemplateViewStage"}, assigner=compact) == []
    message = only(stage_contract_problems(view={"type": "ObjectnessViewStage"}, assigner=compact))
    assert "ObjectnessViewStage computes its targets from 'dense' matched labels" in message
    assert "CompactLabelAssigner returns 'compact' ones" in message
    assert "View stages that read 'compact' labels: " in message and "TemplateViewStage" in message


# ------------------------------------------------------ fields of the matched labels


def test_a_view_stage_names_the_assigner_options_it_depends_on():
    view = {"type": "ObjectnessViewStage"}
    message = only(stage_contract_problems(view=view, assigner={"type": "NearestLabelAssigner"}))
    assert "ObjectnessViewStage reads 'on_object' and 'view_width' of the matched labels" in message
    assert (
        "NearestLabelAssigner returns only with `supervise: object_seeds` and `view_widths: true`"
        in message
    )
    half = {"type": "NearestLabelAssigner", "supervise": "object_seeds"}
    message = only(stage_contract_problems(view=view, assigner=half))
    assert "reads 'view_width'" in message and "`view_widths: true`" in message
    assert "supervise" not in message
    assert stage_contract_problems(view=view, assigner=OBJECT_SEEDS) == []
    # ScaleBalancedAssigner always returns the widths of every view, and the weight.
    assert stage_contract_problems(view=view, assigner={"type": "ScaleBalancedAssigner"}) == []
    every_seed = {"type": "ScaleBalancedAssigner", "supervise": "all"}
    assert "`supervise: object_seeds`" in only(
        stage_contract_problems(view=view, assigner=every_seed)
    )


def test_seed_weights_are_never_dropped_silently():
    """Before this rule a GSNet config with ``ScaleBalancedAssigner`` trained without any
    scale balancing: the SWAD head and the template view stage do not read the weight."""
    balanced = {"type": "ScaleBalancedAssigner"}
    message = only(
        stage_contract_problems(
            view={"type": "TemplateViewStage"}, assigner=balanced, head={"type": "SWADHead"}
        )
    )
    assert "ScaleBalancedAssigner returns a loss weight per seed" in message
    assert "TemplateViewStage and SWADHead do not apply seed weights" in message
    assert "silently do nothing" in message
    assert (
        "ObjectnessViewStage" in message.split("view stages: ")[1].split(";")[0]
        and "head bundles: OperationHead" in message
    )
    message = only(
        stage_contract_problems(
            view={"type": "ObjectnessViewStage"}, assigner=balanced, head={"type": "SWADHead"}
        )
    )
    assert "but SWADHead does not apply seed weights" in message
    assert (
        stage_contract_problems(
            view={"type": "ObjectnessViewStage"}, assigner=balanced, head={"type": "OperationHead"}
        )
        == []
    )
    assert VIEWS.get("ObjectnessViewStage").applies_seed_weights
    assert HEADS.get("OperationHead").applies_seed_weights
    assert not HEADS.get("SWADHead").applies_seed_weights
    assert not HEADS.get("EconomicHead").applies_seed_weights


# ------------------------------------------------------------------- scores of seeds


def test_a_seed_score_has_exactly_one_owner():
    message = only(
        stage_contract_problems(
            proposal={"type": "GraspnessFPS"}, view={"type": "ObjectnessViewStage"}
        )
    )
    assert (
        "GraspnessFPS and ObjectnessViewStage both predict and supervise the 'objectness'"
        in message
    )
    assert "view stage that does not predict it (" in message and "TemplateViewStage" in message
    assert "proposal that does not (" in message and "SubsampledSeeds" in message
    assert PROPOSALS.get("GraspnessFPS").seed_scores == ("graspness", "objectness")
    assert PROPOSALS.get("SubsampledSeeds").seed_scores == ()


def test_a_score_the_decoder_reads_must_be_predicted_by_some_stage():
    """Otherwise the model trains and fails at its first evaluation."""
    head = {"type": "OperationHead", "score_decode": "score_x_objectness_x_tolerance"}
    stages = {"proposal": {"type": "SubsampledSeeds"}, "view": {"type": "TemplateViewStage"}}
    message = only(stage_contract_problems(**stages, head=head))
    assert (
        "OperationHead as configured reads the seed score 'objectness' when it decodes" in message
    )
    assert "neither SubsampledSeeds nor TemplateViewStage predicts it" in message
    assert "GraspnessFPS" in message and "ObjectnessViewStage" in message
    assert stage_contract_problems(**stages, head={"type": "OperationHead"}) == []
    assert (
        stage_contract_problems(**{**stages, "view": {"type": "ObjectnessViewStage"}}, head=head)
        == []
    )
    assert (
        stage_contract_problems(**{**stages, "proposal": {"type": "GraspnessFPS"}}, head=head) == []
    )


# ----------------------------------------------------- what two stages state separately


def test_depth_bins_of_grouper_and_head_are_compared_with_their_defaults():
    grouper = {"type": "DepthCylinderGroup", "hmax": [0.01, 0.02, 0.03]}
    message = only(stage_contract_problems(grouper=grouper, head={"type": "OperationHead"}))
    assert "lists 3 far end(s) (hmax)" in message and "num_depth = 4" in message
    three = {"type": "OperationHead", "num_depth": 3}
    assert stage_contract_problems(grouper=grouper, head=three) == []


def test_one_score_definition_is_shared_only_by_stages_that_use_it():
    view = {"type": "ObjectnessViewStage"}
    fixed = {"type": "OperationHead", "score_target": "log_friction_fixed_max"}
    message = only(stage_contract_problems(view=view, head=fixed))
    assert (
        "score_target = 'log_friction_batch_max'" in message
        and "'log_friction_fixed_max'" in message
    )
    # The SWAD head and the template view stage have targets of their own kind: an option
    # that merely shares a name (max_friction of the view graspness) is not compared.
    assert stage_contract_problems(view=view, head={"type": "SWADHead"}) == []
    assert stage_contract_problems(view={"type": "TemplateViewStage"}, head=fixed) == []


# ------------------------------------------- the same verdict at every point in time


def tiny(**slots) -> dict:
    model = {
        "type": "GSNet",
        "backbone": {
            "type": "PointNet2",
            "npoints": [24, 12],
            "radii": [0.3, 0.6],
            "nsamples": [4, 4],
            "sa_mlps": [[8], [8]],
            "fp_mlps": [[8]],
        },
        "proposal": {"type": "SubsampledSeeds", "num_seeds": 24},
        "view": {"type": "ObjectnessViewStage", "num_views": 10},
        "assigner": dict(OBJECT_SEEDS),
        "grouper": {"type": "DepthCylinderGroup", "hmax": [0.1, 0.2], "mlp_channels": [8]},
        "head": {"type": "OperationHead", "num_angle": 3, "num_depth": 2, "hidden_channels": 8},
    }
    model.update(slots)
    return model


BROKEN = [
    ({"assigner": {"type": "NearestLabelAssigner"}}, "returns only with `supervise: object_seeds`"),
    ({"assigner": {"type": "CompactLabelAssigner"}}, "must agree on the label set"),
    ({"proposal": {"type": "GraspnessFPS", "num_seeds": 24}}, "requires 'dense' scene features"),
    (
        {"head": {"type": "OperationHead", "num_angle": 3, "num_depth": 3, "hidden_channels": 8}},
        "num_depth = 3",
    ),
    (
        {
            "head": {
                "type": "OperationHead",
                "num_angle": 3,
                "num_depth": 2,
                "hidden_channels": 8,
                "score_target": "log_friction_fixed_max",
            }
        },
        "both must encode scores the same way",
    ),
]


@pytest.mark.parametrize(
    "detector", ["GSNet", "EconomicGrasp", "GraspNetBaseline", "ScaleBalancedGrasp"]
)
@pytest.mark.parametrize(("slots", "fragment"), BROKEN)
def test_config_check_and_build_agree_for_every_detector_type(detector, slots, fragment):
    """The rules belong to the skeleton: the detector type does not change the verdict,
    and a detector built from nodes without the config checker refuses the same things."""
    Config.from_dict({"model": tiny(type=detector)})
    DETECTORS.build(tiny(type=detector))
    node = tiny(type=detector, **slots)
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": node})
    messages = [issue.message for issue in error.value.issues]
    assert any(fragment in message for message in messages), messages
    assert len(messages) == len(set(messages))  # a rule a subclass repeats is reported once
    with pytest.raises(ContractError, match=detector):
        DETECTORS.build(node)


def test_built_components_are_judged_by_their_actual_options():
    plain = ASSIGNERS.build(
        {"type": "NearestLabelAssigner"}, template=_template(), gripper=Gripper()
    )
    with pytest.raises(ContractError, match="supervise: object_seeds.*view_widths: true"):
        DETECTORS.build(tiny(assigner=plain))
    ready = ASSIGNERS.build(dict(OBJECT_SEEDS), template=_template(), gripper=Gripper())
    assert DETECTORS.build(tiny(assigner=ready)).assigner is ready


def _template():
    from grasppanda.models.views.template import ViewTemplate

    return ViewTemplate.fibonacci(10)
