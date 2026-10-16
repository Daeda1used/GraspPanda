"""``grasppanda compose``: a base config plus ``slot=choice`` pairs → a checked config.

What is asserted is what a user relies on: the composed file is the config they asked
for (same hash as writing it by hand), every adaptation is announced, a refused
combination quotes the contract rules and writes nothing, and the shipped experimental
configs are what compose writes today.
"""

from __future__ import annotations

import json

import pytest

from grasppanda.cli.main import main
from grasppanda.combine import yamltext
from grasppanda.combine.compose import (
    CONTRACT_GROUPS,
    ComposeError,
    component_slots,
    compose,
    is_adapter,
    recompose,
)
from grasppanda.config import Config, ConfigError, available_configs, configs_root, parse_yaml
from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES, stage_contract_problems
from grasppanda.registry import BACKBONES, DETECTORS

GSNET = "gsnet/gsnet_sunet14_g1b-realsense_10e"
ECONOMIC = "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e"
FINEGRASP = "finegrasp/finegrasp_sunet14_g1b-realsense_10e"
BASELINE = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"
SBG = "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e"
HGGD = "hggd/hggd_r34w8_g1b-realsense_15e"
EXPERIMENTAL = [name for name in available_configs() if name.startswith("experimental/")]


def loaded(result, tmp_path, name="composed.yaml") -> Config:
    """The composed config as the file a user gets."""
    path = tmp_path / name
    path.write_text(result.text(path), encoding="utf-8")
    return Config.load(path)


# ----------------------------------------------------------------- what gets composed


def test_a_preset_becomes_the_one_line_swap_it_stands_for(tmp_path):
    result = compose(GSNET, ["backbone=ptv3"])
    assert result.accepted and result.adapter is None
    assert result.body == {
        "model": {"backbone": {"type": "PTv3", "in_channels": 3, "grid_size": 0.005}}
    }
    by_hand = Config.load(
        GSNET, ["model.backbone={type: PTv3, in_channels: 3, grid_size: 0.005}"]
    ).digest()
    assert result.config_hash == by_hand == loaded(result, tmp_path).digest()
    assert result.stages == {
        "backbone": "PTv3",
        "proposal": "GraspnessFPS",
        "view": "TemplateViewStage",
        "assigner": "NearestLabelAssigner",
        "grouper": "CylinderGroup",
        "head": "SWADHead",
    }
    assert result.origins["backbone"] == "preset `ptv3`" and result.origins["head"] == "base config"
    assert [verdict.slots for verdict in result.verdicts] == list(CONTRACT_GROUPS)
    assert not any(verdict.problems for verdict in result.verdicts)


def test_a_subsampling_encoder_under_a_dense_proposal_gets_the_adapter_and_is_told():
    result = compose(GSNET, ["backbone=pointnet2"])
    assert result.accepted and result.adapter == "FeaturePropagation"
    backbone = result.body["model"]["backbone"]
    assert backbone["type"] == "FeaturePropagation" and backbone["backbone"]["type"] == "PointNet2"
    assert result.stages["backbone"] == "PointNet2 in FeaturePropagation"
    (told,) = [action for action in result.actions if action.startswith("wrapped ")]
    # The reason is the first sentence of the contract rule, not a text of compose's own.
    (rule,) = stage_contract_problems(
        backbone={"type": "PointNet2"}, proposal={"type": "GraspnessFPS"}
    )
    assert rule.split(". ")[0] in told and "FeaturePropagation" in told


def test_a_dense_encoder_under_subsampled_seeds_keeps_the_reference_seeds():
    result = compose(BASELINE, ["backbone=sparse-unet14"])
    assert result.accepted and result.adapter == "FurthestPointSubsample"
    backbone = result.body["model"]["backbone"]
    assert backbone == {
        "type": "FurthestPointSubsample",
        "npoint": 1024,  # = model.proposal.num_seeds
        "fps_skip_near_origin": True,  # the rules of the encoder it replaces
        "fps_tie_break": "pointnet2_cuda",
        "backbone": {
            "type": "SparseUNet14",
            "in_channels": 3,
            "out_channels": 256,  # the width of PointNet2, not the 512 of the preset
            "voxel_size": 0.005,
        },
    }
    told = " ".join(result.actions)
    assert "out_channels: 256" in told and "npoint: 1024" in told and "fps_tie_break" in told
    assert result.comments[("model", "backbone", "npoint")] == "= model.proposal.num_seeds"
    # A proposal with another seed count moves the adapter with it, however it was set:
    # by an override of the base config's proposal ...
    other = compose(BASELINE, ["backbone=sparse-unet14"], ["model.proposal.num_seeds=512"])
    assert other.accepted and other.body["model"]["backbone"]["npoint"] == 512
    assert other.body["model"]["proposal"] == {"num_seeds": 512}
    assert any("npoint: 512, the number of seeds" in action for action in other.actions)
    # ... or by an override of a proposal that was --set.
    fewer = compose(GSNET, ["proposal=SubsampledSeeds"], ["model.proposal.num_seeds=512"])
    assert fewer.accepted and fewer.body["model"]["backbone"]["npoint"] == 512
    # ... and here the encoder is the base config's own, written out inside the adapter.
    assert fewer.body["model"]["backbone"]["backbone"]["type"] == "SparseUNet14"
    assert fewer.body["model"]["backbone"]["backbone"]["encoder_channels"] == [32, 64, 128, 256]


def test_without_adapters_the_refusal_is_the_contract_rule():
    result = compose(GSNET, ["backbone=pointnet2"], adapters=False)
    assert not result.accepted and result.adapter is None and result.config_hash is None
    (problem,) = result.problems
    assert "GraspnessFPS requires 'dense' scene features" in problem
    assert "{type: FeaturePropagation, backbone: {...}}" in problem  # the fix, by name
    assert result.verdicts[0].slots == ("backbone", "proposal") and result.verdicts[0].problems
    assert result.issues == []  # the checker's copy of the same message is not repeated
    assert "--no-adapter" in result.command


def test_the_encoder_reads_the_features_the_method_feeds():
    """FineGrasp feeds [x, y, z | normal]; an encoder preset written for constant ones
    is given six input channels, and that is announced."""
    result = compose(FINEGRASP, ["backbone=ptv3"])
    assert result.accepted and result.body["model"]["backbone"]["in_channels"] == 6
    assert any(action.startswith("in_channels: 6") for action in result.actions)
    wrapped = compose(FINEGRASP, ["backbone=pointnet2"])
    assert wrapped.body["model"]["backbone"]["backbone"]["in_channels"] == 6
    # A method that feeds constant ones leaves the preset's choice alone.
    assert (
        compose(ECONOMIC, ["backbone=pointnet2"]).body["model"]["backbone"]["backbone"][
            "in_channels"
        ]
        == 0
    )


def test_overrides_address_the_component_that_was_set_and_win(tmp_path):
    result = compose(
        BASELINE,
        ["backbone=sparse-unet14"],
        ["model.backbone.voxel_size=0.01", "model.backbone.out_channels=128", "train.epochs=3"],
    )
    inner = result.body["model"]["backbone"]["backbone"]
    assert inner["voxel_size"] == 0.01 and inner["out_channels"] == 128  # not the fitted 256
    assert result.body["train"] == {"epochs": 3} and result.accepted
    assert not any(action.startswith("out_channels:") for action in result.actions)
    config = loaded(result, tmp_path)
    assert config.select("train.epochs") == 3
    assert config.select("model.backbone.backbone.voxel_size") == 0.01
    with pytest.raises(ComposeError, match="use `--set backbone="):
        compose(GSNET, ["backbone=ptv3"], ["model.backbone.type=LitePT"])
    # A child config cannot reach into an inherited list; compose says what to write instead.
    with pytest.raises(ComposeError, match=r"give the whole list \(`hooks=\[...\]`\)"):
        compose(GSNET, ["backbone=ptv3"], ["hooks.0.interval=5"])
    with pytest.raises(ComposeError, match=r"\(`data.train.dataset.transforms=\[...\]`\)"):
        compose(GSNET, ["backbone=ptv3"], ["data.train.dataset.transforms.0.num_points=1000"])
    with pytest.raises(ComposeError, match=r"`model.backbone.enc_depths=\[...\]`"):
        compose(GSNET, ["backbone=ptv3"], ["model.backbone.enc_depths.0=3"])
    whole = compose(GSNET, ["backbone=ptv3"], ["model.backbone.enc_depths=[1, 1, 1, 2, 1]"])
    assert whole.accepted and whole.body["model"]["backbone"]["enc_depths"] == [1, 1, 1, 2, 1]


def test_a_stage_that_reads_a_label_asset_brings_its_dataset_section():
    result = compose(BASELINE, ["assigner=scale-balanced"])
    assert result.accepted
    statistic = result.body["data"]["train"]["dataset"]["scale_statistic"]
    assert statistic["source"] == "native" and statistic["bins"] == 32
    assert any(
        action.startswith("added `data.train.dataset.scale_statistic`") for action in result.actions
    )
    # A base config that serves the section keeps its own, untouched.
    own = compose(SBG, ["assigner=scale-balanced"])
    assert "data" not in own.body and own.config_hash == Config.load(SBG).digest()
    fine = compose(FINEGRASP, ["proposal=graspness-fps"])
    assert "data" not in fine.body  # FineGrasp's instance-normalised graspness stays
    # A compact head with a compact assigner on a recipe of dense grids: the recipe is
    # rewritten for compact labels (the compact section, without the dense label step).
    compact = compose(GSNET, ["assigner=compact-label", "head=economic"])
    assert compact.accepted
    dataset = compact.body["data"]["train"]["dataset"]
    assert dataset["compact_labels"]["view_friction_threshold"] == 0.6  # = the view's
    assert all(step["type"] != "SampleGraspLabels" for step in dataset["transforms"])


@pytest.mark.parametrize(
    ("base", "sets", "fragments", "kinds"),
    [
        (GSNET, ["grouper=depth-embed"], ["is itself an axis adapter"], ["structural"]),
        (
            BASELINE,
            ["grouper=voxel-set-abstraction"],
            ["No axis adapter turns a scene-support crop"],
            ["not-implemented"],
        ),
        (
            BASELINE,
            ["proposal=objectness-fps"],
            ["selects its seeds by objectness"],
            ["structural"],
        ),
        (
            ECONOMIC,
            ["view=objectness-view"],
            ["computes its targets from 'dense' matched labels"],
            ["structural"],
        ),
        (
            GSNET,
            ["assigner=compact-label"],
            ["must agree on the label set"],
            ["structural", "fixed-by-compose"],
        ),
    ],
)
def test_a_refused_combination_quotes_the_rules_and_writes_nothing(
    base, sets, fragments, kinds, tmp_path, capsys
):
    result = compose(base, sets)
    assert not result.accepted and result.config_hash is None
    for fragment in fragments:
        assert any(fragment in problem for problem in result.problems), result.problems
    assert result.refusal_kinds() == kinds
    # Every problem is a message of the contract rules on the composed stage set.
    model = result.merged["model"]
    stages = {slot: model[slot] for slot in STAGE_REGISTRIES}
    assert result.problems == list(dict.fromkeys(stage_contract_problems(**stages)))
    target = tmp_path / "refused.yaml"
    arguments = ["compose", base, "-o", str(target)]
    for item in sets:
        arguments += ["--set", item]
    assert main(arguments) == 1
    out = " ".join(capsys.readouterr().out.split())  # the report wraps its paragraphs
    assert not target.exists() and "refused: nothing was written" in out
    assert all(fragment in out for fragment in fragments)
    assert all(f"[{kind}]" in out for kind in kinds)  # the report says why, kind by kind


#: Refusals compose lifts with an adapter or an option (W4.2, W4.4, W4.6): the request,
#: what compose says it did, and what `--no-adapter` (or a config written without it)
#: is refused with.
LIFTED = [
    (
        GSNET,
        ["grouper=depth-ball"],
        "grouper: wrapped DepthLocalGroup in the axis adapter DepthPoolGroup (preset `depth-pool`)",
        "wrap the grouper in the axis adapter DepthPoolGroup (preset `depth-pool`)",
    ),
    (
        BASELINE,
        ["grouper=ball"],
        "grouper: wrapped LocalGroup in the axis adapter DepthEmbedGroup (preset `depth-embed`)",
        "wrap the grouper in the axis adapter DepthEmbedGroup (preset `depth-embed`)",
    ),
    (
        SBG,
        ["view=lattice-view"],
        "head: score_decode: score_x_tolerance.",
        "Set `score_decode: score_x_tolerance`, the decode option of the head",
    ),
    (
        BASELINE,
        ["assigner=NearestLabelAssigner"],
        "assigner: `supervise: object_seeds` and `view_widths: true`",
        "returns only with `supervise: object_seeds` and `view_widths: true`",
    ),
    (
        BASELINE,
        ["proposal=graspness-ffps"],
        "proposal: select_by: graspness_only.",
        "Let ObjectnessViewStage own it and ScoredSeeds leave it: `select_by: graspness_only`",
    ),
    (
        GSNET,
        ["view=objectness-view"],
        "proposal: ScoredSeeds with select_by: graspness_only",
        "`type: ScoredSeeds` with `select_by: graspness_only`",
    ),
]


@pytest.mark.parametrize(("base", "sets", "action", "fix"), LIFTED)
def test_compose_lifts_what_an_adapter_or_an_option_lifts_and_says_so(base, sets, action, fix):
    result = compose(base, sets)
    assert result.accepted, result.refusal()
    assert any(told.startswith(action) for told in result.actions), result.actions
    assert result.warnings == []
    # The composed file loads to the config that was checked; its stages agree.
    model = result.merged["model"]
    assert stage_contract_problems(**{slot: model[slot] for slot in STAGE_REGISTRIES}) == []
    # Without the lift the pair is refused, and the refusal names what compose would do.
    refused = compose(base, sets, adapters=False)
    assert not refused.accepted and refused.config_hash is None
    assert any(fix in problem for problem in refused.problems), refused.problems
    assert set(refused.refusal_kinds()) == {"fixed-by-compose"}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("base", "sets"),
    [
        (BASELINE, ["grouper=ball"]),  # W4.2, an axis adapter
        (GSNET, ["assigner=scale-balanced"]),  # W4.3, seed weights in the view and the head
        (SBG, ["view=lattice-view"]),  # W4.4, the decode option
        (BASELINE, ["proposal=graspness-fps"]),  # W4.6, graspness only
        (ECONOMIC, ["head=swad"]),  # W4.5, dense labels on a compact recipe
        (GSNET, ["head=economic"]),  # W4.5, compact labels on a dense recipe
    ],
)
def test_a_lifted_composition_is_wired_at_the_sizes_of_its_config(base, sets, tmp_path):
    """The composed file of each lift, at its own sizes, builds, back-propagates and
    predicts on a synthetic scene (`grasppanda probe --synthetic`)."""
    import importlib.util

    from grasppanda.combine.probe import probe

    if base in (GSNET, ECONOMIC) and importlib.util.find_spec("spconv") is None:
        pytest.skip("needs the spconv extra")
    result = compose(base, sets)
    assert result.accepted, result.refusal()
    path = tmp_path / "lifted.yaml"
    path.write_text(result.text(path), encoding="utf-8")
    report = probe(path, iters=2, synthetic=True, device="cpu")
    assert report.verdict == "wired", (report.detail, report.failure)


def test_the_refusal_kinds_and_the_warnings_are_in_the_report(capsys, monkeypatch):
    """W4.1 and W4.7: the JSON report carries the kind of every reason, parallel to
    `refusal`, and the warnings of an accepted stage set."""
    assert main(["compose", ECONOMIC, "--set", "assigner=NearestLabelAssigner", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["refusal"] == [
        *report["problems"],
        *(f"{issue['path']}: {issue['message']}" for issue in report["issues"]),
    ]
    assert report["refusal_kinds"] == ["structural", "fixed-by-compose"]
    assert [issue["kind"] for issue in report["issues"]] == ["fixed-by-compose"]
    assert report["contracts"][4]["kinds"] == ["structural"]  # assigner · head
    assert report["warnings"] == []
    # A view stage that cannot apply a weight per seed: accepted, with a warning.
    from grasppanda.registry import VIEWS

    monkeypatch.setattr(VIEWS.get("TemplateViewStage"), "applies_seed_weights", False)
    assert main(["compose", GSNET, "--set", "assigner=scale-balanced", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    (warning,) = report["warnings"]
    assert "TemplateViewStage does not apply seed weights: its loss is unweighted" in warning
    assert "Warnings (accepted, but read them):" in report["config"]
    assert report["refusal"] == report["refusal_kinds"] == []


def test_the_bare_component_is_composed_with_its_defaults_and_points_at_the_preset():
    result = compose(GSNET, ["backbone=PTv3"])
    assert result.accepted and result.body == {"model": {"backbone": {"type": "PTv3"}}}
    assert result.origins["backbone"] == "constructor defaults"
    assert any(
        "not a recipe for table-top clouds" in action and "`--set backbone=ptv3`" in action
        for action in result.actions
    )
    bare = compose(GSNET, ["grouper=MultiScaleGroup"])
    assert not bare.accepted and "requires 'branch'" in bare.issues[0].message
    assert compose(GSNET, ["grouper=multi-scale"]).accepted  # which the preset supplies


def test_requests_that_cannot_be_composed_say_why():
    with pytest.raises(ComposeError, match=r"no preset 'ptv33' for slot 'backbone'.*'ptv3'"):
        compose(GSNET, ["backbone=ptv33"])
    with pytest.raises(ComposeError) as far:  # nothing near: no guess is offered
        compose(GSNET, ["backbone=resnet50"])
    assert "Did you mean" not in str(far.value) and "Presets: " in str(far.value)
    with pytest.raises(ComposeError, match=r"GSNet has no slot 'neck'"):
        compose(GSNET, ["neck=ptv3"])
    with pytest.raises(ComposeError, match=r"slot 'backbone' is set twice"):
        compose(GSNET, ["backbone=ptv3", "backbone=litept"])
    with pytest.raises(ComposeError, match=r"must look like `slot=preset`"):
        compose(GSNET, ["backbone"])
    with pytest.raises(ComposeError, match=r"is an adapter: FeaturePropagation wraps"):
        compose(GSNET, ["backbone=feature-propagation"])
    with pytest.raises(ComposeError, match=r"is an adapter"):
        compose(GSNET, ["backbone=FurthestPointSubsample"])
    with pytest.raises(ConfigError):
        compose("gsnet/no_such_config", ["backbone=ptv3"])


def test_a_detector_family_of_its_own_says_which_slots_it_has(tmp_path, capsys):
    assert list(component_slots(DETECTORS.get("HGGD"))) == ["image_encoder"]
    assert list(component_slots(DETECTORS.get("GSNet"))) == [
        "backbone",
        "proposal",
        "view",
        "grouper",
        "head",
        "assigner",
    ]
    with pytest.raises(ComposeError) as error:
        compose(HGGD, ["backbone=ptv3"])
    message = str(error.value)
    assert "HGGD has no slot 'backbone'" in message and "Its slots: image_encoder." in message
    assert "detector family of its own" in message and "grasppanda components HGGD" in message
    assert "--opts model.<part>.<option>=<value>" in message
    assert main(["compose", HGGD, "--set", "backbone=ptv3"]) == 2
    assert "HGGD has no slot 'backbone'" in capsys.readouterr().err
    # Its options can still be changed, and the config check is the verdict.
    result = compose(HGGD, overrides=["model.encoder.base_width=16"])
    assert result.accepted and not result.seed_based and result.verdicts == []
    assert loaded(result, tmp_path).select("model.encoder.base_width") == 16
    assert main(["compose", HGGD, "--opts", "model.encoder.base_width=16"]) == 0
    assert "not built on the seed-based skeleton" in capsys.readouterr().out


def test_adapters_are_found_from_the_registry_not_from_a_list():
    adapters = sorted(name for name in BACKBONES.names() if is_adapter(BACKBONES.get(name)))
    assert adapters == [
        "FeaturePropagation",
        "FurthestPointSubsample",
        "ImagePainted",
        "ImagePaintedSubsampled",
    ]
    # ... and only those a contract can ask for are ever inserted.
    from grasppanda.combine.compose import _adapters_for

    dense, proposal = {"type": "SparseUNet14"}, {"type": "SubsampledSeeds"}
    assert _adapters_for(dense, proposal) == ["FurthestPointSubsample"]
    assert not is_adapter(BACKBONES.get("PointNet2"))


# ------------------------------------------------- what compose reads from the contracts


SUBSAMPLED_SET = [
    "proposal=SubsampledSeeds",
    "view=objectness-view",
    "assigner=nearest-label-object-seeds",
    "grouper=depth-cylinder-multi-scale",
    "head=operation",
]


def test_the_encoder_is_given_the_width_a_grouper_fixes():
    """The gated seed residual needs 256-channel seeds. A new encoder with an
    `out_channels` option gets that width, whatever the encoder it replaces returned; one
    without the option gets the projected variant of the grouper, as does the base config's
    own encoder."""
    fitted = compose(GSNET, ["backbone=sparse-unet14", *SUBSAMPLED_SET])
    assert fitted.accepted and fitted.adapter == "FurthestPointSubsample"
    inner = fitted.body["model"]["backbone"]["backbone"]
    assert inner["out_channels"] == 256  # not the 512 of the encoder it replaces
    assert any("the seed features DepthCylinderGroup needs" in a for a in fitted.actions)
    assert fitted.comments[("model", "backbone", "backbone", "out_channels")] == (
        "the seed width DepthCylinderGroup needs as configured"
    )
    # The same grouper set by an override counts like one set by --set ...
    by_option = compose(BASELINE, ["backbone=sparse-unet14"], ["model.grouper.seed_residual=gated"])
    assert by_option.accepted
    assert by_option.body["model"]["backbone"]["backbone"]["out_channels"] == 256
    # ... and without the residual the width of the replaced encoder is kept.
    plain = compose(SBG, ["backbone=sparse-unet14"], ["model.grouper.seed_residual=none"])
    assert any("the encoder it replaces (PointNet2Transformer)" in a for a in plain.actions)
    # An encoder without the option, and the base config's own encoder, get the projected
    # variant of the grouper; with --no-adapter the width rule refuses them, with both widths.
    for sets in (["backbone=ptv3", *SUBSAMPLED_SET], SUBSAMPLED_SET):
        projected = compose(GSNET, sets)
        assert projected.accepted
        assert projected.body["model"]["grouper"]["type"] == "DepthCylinderProjectionGroup"
        assert any("DepthCylinderProjectionGroup in place of" in a for a in projected.actions)
        refused = compose(GSNET, sets, adapters=False)
        assert not refused.accepted and refused.config_hash is None
        assert any(
            "adds the seed features to its 256-channel crop features" in problem
            and "`type: DepthCylinderProjectionGroup`" in problem
            for problem in refused.problems
        )
    assert any(
        "PTv3 returns 64-channel features" in problem
        for problem in compose(GSNET, ["backbone=ptv3", *SUBSAMPLED_SET], adapters=False).problems
    )
    # The user's own width wins over the fit, and is then what the rule judges.
    own = compose(
        GSNET,
        ["backbone=sparse-unet14", *SUBSAMPLED_SET],
        ["model.backbone.out_channels=128"],
        adapters=False,
    )
    assert not own.accepted
    assert any("returns 128-channel features" in problem for problem in own.problems)


@pytest.mark.parametrize(
    ("base", "sets", "labels", "assigner"),
    [
        (ECONOMIC, ["head=swad"], "dense", "NearestLabelAssigner"),
        (FINEGRASP, ["head=decoupled-grid"], "dense", "NearestLabelAssigner"),
        (ECONOMIC, ["head=operation"], "dense", "NearestLabelAssigner"),
        (GSNET, ["head=economic"], "compact", "CompactLabelAssigner"),
    ],
)
def test_the_data_recipe_follows_the_label_set_of_the_head(base, sets, labels, assigner):
    """W4.5: a head bundle of the other label set brings its labels: the dataset section is
    rewritten after a shipped recipe that serves them, the assigner after a shipped config
    whose assigner returns them, and the report says both."""
    result = compose(base, sets)
    assert result.accepted, result.refusal()
    dataset = result.merged["dataset"]
    told = " ".join(result.actions)
    assert f"labels: {labels}." in told and f"assigner: {assigner}, the assigner of" in told
    steps = [step["type"] for step in dataset["transforms"]]
    if labels == "dense":
        assert dataset.get("compact_labels") is None and "SampleGraspLabels" in steps
        # The dense step sits before the first step that moves the points, as in GSNet.
        assert steps.index("SampleGraspLabels") < steps.index("RandomFlip")
        assert "grasppanda prepare <dataset> --labels" in told
    else:
        assert dataset["compact_labels"] and "SampleGraspLabels" not in steps
        assert "--compact-labels" in told
    assert result.merged["model"]["assigner"]["type"] == assigner
    # `--no-adapter` writes nothing on its own: the label sets disagree.
    refused = compose(base, sets, adapters=False)
    assert not refused.accepted and "structural" in refused.refusal_kinds()
    # `--set labels=` asks for it in so many words, also without the automatic changes.
    asked = compose(base, [*sets, f"labels={labels}"], adapters=False)
    assert f"--set labels={labels}" in asked.command
    assert any(
        action.startswith(f"labels: {labels}. `--set labels={labels}`") for action in asked.actions
    )


def test_a_data_recipe_refusal_that_remains_prints_the_exact_words_that_rewrite_it():
    """A dense assigner chosen under a compact head stays refused (the label sets differ),
    and the hint prints the `--set labels=` and the `--unset`/`--opts` it stands for;
    applied as printed, the data refusal is gone and only the genuine one remains."""
    result = compose(ECONOMIC, ["assigner=NearestLabelAssigner"])
    assert not result.accepted
    assert result.refusal_kinds() == ["structural", "fixed-by-compose"]
    (data,) = [hint for hint in result.hints if hint.startswith("To make the training data")]
    assert "`--set labels=dense`" in data
    words = data.split("which writes `")[1].split("` (the recipe of")[0]
    assert words.startswith("--unset data.train.dataset.compact_labels --opts ")
    import shlex

    parts = shlex.split(words)
    unsets = [parts[1]]
    overrides = parts[3:]
    by_hand = compose(ECONOMIC, ["assigner=NearestLabelAssigner"], overrides, unsets=unsets)
    assert by_hand.refusal_kinds() == ["structural"]  # NearestLabelAssigner vs EconomicHead
    assert any("the request set the assigner" in hint for hint in result.hints)
    # The other direction, and a hint per derived asset a bare component reads.
    compact = compose(GSNET, ["assigner=compact-label"])
    assert any("`--set labels=compact`" in hint for hint in compact.hints)
    bare = compose(BASELINE, ["proposal=GraspnessFPS", "view=TemplateViewStage"])
    assert any(
        hint.startswith("To serve `point_labels.graspness`: `--opts 'data.train.dataset.graspness=")
        for hint in bare.hints
    )


def test_unset_puts_an_option_back_to_the_default_of_its_component(tmp_path):
    result = compose(
        ECONOMIC, ["head=economic-focal"], unsets=["data.train.dataset.compact_labels"]
    )
    assert result.body["data"]["train"]["dataset"]["compact_labels"] is None
    assert "--unset data.train.dataset.compact_labels" in result.command
    assert any(
        action.startswith("unset `data.train.dataset.compact_labels`") for action in result.actions
    )
    assert not result.accepted  # the compact head then reads labels nobody serves
    # The written file says how it was made and composes again to itself.
    again = compose(GSNET, ["head=swad-focal"], unsets=["model.head.width_loss"])
    path = tmp_path / "unset.yaml"
    path.write_text(again.text(path), encoding="utf-8")
    assert recompose(path).config_hash == again.config_hash
    with pytest.raises(ComposeError, match="sets nothing there"):
        compose(GSNET, [], unsets=["data.train.dataset.no_such_key"])
    with pytest.raises(ComposeError, match="the label set is one of dense, compact"):
        compose(GSNET, ["labels=sparse"])


def test_a_component_named_by_its_type_is_pointed_at_the_preset_with_its_dataset_section():
    bare = compose(BASELINE, ["proposal=GraspnessFPS", "view=TemplateViewStage"])
    assert not bare.accepted
    (issue,) = bare.issues
    assert "GraspnessFPS reads `supervision.point_labels['graspness']`" in issue.message
    assert any("`--set proposal=graspness-fps`" in hint for hint in bare.hints)
    assert not any(hint.startswith("Or compose from") for hint in bare.hints)  # none serves all
    # With the preset the section is added, and the same stages are accepted.
    preset = compose(BASELINE, ["proposal=graspness-fps", "view=template-view"])
    assert preset.accepted and "graspness" in preset.body["data"]["train"]["dataset"]
    tolerance = compose(GSNET, ["grouper=DepthCylinderGroup", "head=OperationHead"])
    assert not tolerance.accepted
    assert any("`--set head=operation`" in hint for hint in tolerance.hints)
    assert compose(GSNET, ["grouper=DepthCylinderGroup", "head=operation"]).accepted
    histogram = compose(BASELINE, ["assigner=ScaleBalancedAssigner"])
    assert any("`--set assigner=scale-balanced`" in hint for hint in histogram.hints)
    # A request refused by a stage contract alone gets no hint about data.
    assert compose(GSNET, ["grouper=depth-embed"]).hints == []


# -------------------------------------------------------------------- the base config


def test_a_composed_config_is_composed_again_from_its_base(tmp_path):
    """A chain is two levels deep, so a composed config cannot be inherited from. Its
    header says how it was made: the two requests are merged and composed from its base."""
    first = tmp_path / "gsnet_pointnet2.yaml"
    assert main(["compose", GSNET, "--set", "backbone=pointnet2", "-o", str(first)]) == 0
    again = compose(first, ["grouper=multi-scale"], ["train.epochs=3"])
    assert again.accepted and again.base == GSNET
    assert again.command == (
        f"grasppanda compose {GSNET} --set backbone=pointnet2 --set grouper=multi-scale "
        "--opts train.epochs=3"
    )
    assert (
        again.config_hash
        == compose(
            GSNET, ["backbone=pointnet2", "grouper=multi-scale"], ["train.epochs=3"]
        ).config_hash
    )
    assert "is itself a composed config" in again.actions[0] and GSNET in again.actions[0]
    # A --set of the new request replaces the one of the same slot; so does an override.
    second = tmp_path / "second.yaml"
    second.write_text(again.text(second), encoding="utf-8")
    replaced = compose(second, ["backbone=ptv3"], ["train.epochs=5"])
    assert replaced.command == (
        f"grasppanda compose {GSNET} --set grouper=multi-scale --set backbone=ptv3 "
        "--opts train.epochs=5"
    )
    # The shipped experimental configs are composed configs.
    shipped = compose(EXPERIMENTAL[0], ["grouper=CylinderInteractionGroup"])
    assert shipped.accepted and shipped.base in (GSNET, ECONOMIC, BASELINE)


def test_a_base_that_cannot_be_inherited_from_says_what_to_compose_from(tmp_path, capsys):
    # A composed file that was edited: its header no longer says what it is.
    edited = tmp_path / "edited.yaml"
    text = compose(GSNET, ["backbone=ptv3"]).text(edited)
    edited.write_text(text + "\ntrain:\n  epochs: 3\n", encoding="utf-8")
    with pytest.raises(ComposeError) as error:
        compose(edited, ["grouper=multi-scale"])
    message = str(error.value)
    assert "limited to 2 levels" in message and "edited afterwards" in message
    assert f"Compose from `{GSNET}` in one command" in message and "--set backbone=ptv3" in message
    # A shipped child config that is not composed: the cross-dataset configs.
    child = "gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e"
    with pytest.raises(ComposeError) as error:
        compose(child, ["backbone=ptv3"])
    assert f"Compose from `{GSNET}` instead" in str(error.value)
    assert main(["compose", child, "--set", "backbone=ptv3"]) == 2
    assert "cannot be the base of a composed config" in capsys.readouterr().err
    # Every shipped config either composes or says so.
    bases = 0
    for name in available_configs():
        if name.startswith("_base_/"):
            continue
        try:
            bases += compose(name).accepted
        except ComposeError as refusal:
            assert "cannot be the base of a composed config" in str(refusal), name
    assert bases >= 16


# ------------------------------------------------------------------------ the file


def test_the_written_file_says_how_it_was_made_and_composes_again(tmp_path):
    result = compose(BASELINE, ["backbone=sparse-unet14"], ["train.epochs=3"])
    outside = tmp_path / "mine.yaml"
    text = result.text(outside)
    assert text.startswith("# EXPERIMENTAL. No AP is claimed. Composed by `grasppanda compose`")
    assert f"#   base  {BASELINE} (GraspNetBaseline)\n#   set   backbone `sparse-unet14`\n" in text
    assert "No AP is claimed" in text and "What compose did, and why:" in text
    assert (
        "What must be re-tuned, and why" in text and "backbone `furthest-point-subsample`" in text
    )
    assert f"#   grasppanda compose {BASELINE} \\\n" in text
    assert "#       --set backbone=sparse-unet14 --opts train.epochs=3\n" in text
    # lines naming the output file grow with the temporary directory the test runs in
    lines = [line for line in text.splitlines() if str(outside) not in line]
    assert max(len(line) for line in lines if not line.startswith("  - ")) <= 100
    # Outside the config tree the base is an absolute path, so the file works anywhere.
    base = parse_yaml(text)["_base_"]
    assert base == [str(configs_root().resolve() / f"{BASELINE}.yaml")]
    outside.write_text(text, encoding="utf-8")
    again = recompose(outside)
    assert again.command == result.command and again.text(outside) == text
    assert Config.load(outside).digest() == result.config_hash
    # Inside it, a relative path, as shipped configs do.
    inside = configs_root() / "experimental" / "not_written.yaml"
    assert parse_yaml(result.text(inside))["_base_"] == [f"../{BASELINE}.yaml"]
    assert "grasppanda probe experimental/not_written" in result.text(inside)


@pytest.mark.parametrize("name", EXPERIMENTAL)
def test_shipped_experimental_configs_are_what_compose_writes(name):
    """The numbers of an experimental config live in a preset; the file is compose's
    output for the command in its header. Edit the preset and regenerate
    (``python -m tools.make_combination_docs``), never the numbers in the file."""
    path = configs_root() / f"{name}.yaml"
    result = recompose(path)
    assert result.accepted and all(choice.preset is not None for choice in result.choices)
    assert result.text(path) == path.read_text(encoding="utf-8"), (
        f"{path} is not what `{result.command}` writes today; "
        "run `python -m tools.make_combination_docs`"
    )
    assert Config.load(name).digest() == result.config_hash
    assert EXPERIMENTAL and len(EXPERIMENTAL) >= 8


def test_yaml_text_keeps_comments_and_parses_back():
    data = {
        "model": {
            "backbone": {"type": "PTv3", "grid_size": 0.005, "order": ["z", "hilbert"]},
            "head": {"loss": {"type": "SmoothL1", "weight": 1e-05, "beta": None}},
        },
        "train": {"amp": False, "note": "yes"},
    }
    comments = {("model", "backbone", "type"): "dense", ("model", "head"): "a bundle"}
    text = yamltext.dump(data, comments)
    assert parse_yaml(text) == data
    assert "    type: PTv3                    # dense\n" in text
    assert "  head:                           # a bundle\n" in text
    assert "order: [z, hilbert]" in text and "note: 'yes'" in text and "beta: null" in text
    assert "\n\ntrain:\n" in text  # a blank line between top-level sections
    assert yamltext.flow(3) == "3" and yamltext.flow([1, 2.5]) == "[1, 2.5]"
    assert yamltext.comment_block(["a", "", "b"]) == "# a\n#\n# b\n"


# ------------------------------------------------------------------- the command line


def test_the_command_writes_the_file_and_reports_stage_by_stage(tmp_path, capsys):
    target = tmp_path / "out" / "gsnet_pointnet2.yaml"
    assert main(["compose", GSNET, "--set", "backbone=pointnet2", "-o", str(target)]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "PointNet2 in FeaturePropagation" in out and "what compose did" in out
    assert "✓ backbone · proposal" in out and "✓ proposal · view · head" in out
    assert "✓ valid (hash c5b4b20c28d3)" in out and f"✓ written {target}" in out
    assert f"grasppanda probe {target}" in out
    assert Config.load(target).digest().startswith("c5b4b20c28d3")
    # Without -o nothing is written and the report says how to get the file.
    assert main(["compose", GSNET, "--set", "backbone=ptv3"]) == 0
    assert "Pass -o FILE" in capsys.readouterr().out


def test_the_command_as_json_and_its_listings(capsys):
    assert main(["compose", GSNET, "--set", "backbone=pointnet2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"] and report["adapter"] == "FeaturePropagation"
    assert report["config_hash"].startswith("c5b4b20c28d3") and report["output"] is None
    assert parse_yaml(report["config"])["model"]["backbone"]["type"] == "FeaturePropagation"
    assert [entry["slots"] for entry in report["contracts"]] == [list(g) for g in CONTRACT_GROUPS]
    request = ["compose", GSNET, "--set", "view=ObjectnessViewStage", "--no-adapter", "--json"]
    assert main(request) == 1
    refused = json.loads(capsys.readouterr().out)
    assert not refused["accepted"] and len(refused["problems"]) == 2
    assert refused["refusal_kinds"] == ["fixed-by-compose", "fixed-by-compose"]

    assert main(["compose", "--list"]) == 0
    listing = capsys.readouterr().out
    for word in ("backbone", "ptv3", "PTv3", "multi-scale", "components without a preset"):
        assert word in listing
    assert "CylinderGroup" in listing and "--set grouper=<preset or component>" in listing
    assert main(["compose", "--list", "head"]) == 0
    assert "ptv3" not in capsys.readouterr().out
    assert main(["compose", "--list", "neck"]) == 2
    assert "unknown slot 'neck'" in capsys.readouterr().err
    assert main(["compose", "--show", "backbone=ptv3"]) == 0
    shown = capsys.readouterr().out
    assert "What must be re-tuned, and why:" in shown and "grid_size: 0.005" in shown
    assert main(["compose", "--show", "backbone=nope"]) == 2
    assert main(["compose"]) == 2
    assert "compose needs a base config" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stop:
        main(["compose", "--help"])
    text = capsys.readouterr().out
    assert stop.value.code == 0 and "--set" in text
    # The help is enough to use the command: slots, examples, the base, exit statuses.
    for fragment in (
        "backbone  proposal  view  assigner  grouper  head",
        "examples:",
        "exit status:",
        "composed again from its own",
        "grasppanda probe",
    ):
        assert fragment in text, fragment
    # Slots are also known by the plural `grasppanda components --slot` uses.
    assert main(["compose", "--list", "backbones"]) == 0
    assert "ptv3" in capsys.readouterr().out
    assert main(["compose", "--show", "ptv3"]) == 2
    assert "--show takes `slot=preset`" in capsys.readouterr().err
    assert main(["compose", "--show", "encoder=ptv3"]) == 2
    assert "Slots: backbone, proposal" in capsys.readouterr().err


def test_the_command_does_not_overwrite_a_file_or_its_own_base(tmp_path, capsys):
    target = tmp_path / "mine.yaml"
    arguments = ["compose", GSNET, "--set", "backbone=ptv3", "-o", str(target)]
    assert main(arguments) == 0
    first = target.read_text(encoding="utf-8")
    capsys.readouterr()
    other = ["compose", GSNET, "--set", "backbone=litept", "-o", str(target)]
    assert main(other) == 2  # the file exists
    assert "exists" in capsys.readouterr().err and target.read_text(encoding="utf-8") == first
    assert main([*other, "--force"]) == 0 and target.read_text(encoding="utf-8") != first
    capsys.readouterr()
    # Writing onto the base config would make it inherit from itself.
    base = tmp_path / "base.yaml"
    shipped = (configs_root() / f"{GSNET}.yaml").read_text(encoding="utf-8")
    base.write_text(shipped.replace("../_base_/", f"{configs_root()}/_base_/"), encoding="utf-8")
    before = base.read_text(encoding="utf-8")
    assert main(["compose", str(base), "--set", "backbone=ptv3", "-o", str(base), "--force"]) == 2
    assert "inherit from itself" in " ".join(capsys.readouterr().err.split())  # wrapped
    assert base.read_text(encoding="utf-8") == before
    # A refused request never touches the output, existing or not.
    assert (
        main(["compose", GSNET, "--set", "grouper=depth-embed", "-o", str(target), "--force"]) == 1
    )
    assert "LitePT" in target.read_text(encoding="utf-8")


# ------------------------------------------------- label grids, seed counts, ranges


def _refusal(base, sets=(), opts=()):
    result = compose(base, list(sets), list(opts))
    assert not result.accepted
    return " ".join(result.refusal())


def test_a_stage_on_another_grid_than_the_labels_is_refused_at_the_config_check():
    """These composed to `valid` and stopped in the first iteration on real frames."""
    text = _refusal(GSNET, opts=["model.head.num_angle=6"])
    assert "SWADHead has num_angle = 6" in text and "set `num_angle: 12`" in text
    text = _refusal(ECONOMIC, opts=["model.head.num_depth=5"])
    assert "EconomicHead has num_depth = 5" in text and "set `num_depth: 4`" in text
    text = _refusal(GSNET, opts=["model.view.num_views=120"])
    assert "TemplateViewStage has num_views = 120" in text and "lattice of 300 views" in text
    assert "--set view=lattice-view" in text
    # A stage that keeps its own lattice apart from the labels' may score 120 views ...
    assert compose(GSNET, ["view=lattice-fibonacci-120"]).accepted
    # ... but not read labels on another lattice than the dataset's.
    text = _refusal(GSNET, ["view=lattice-fibonacci-120"], ["model.view.label_views=120"])
    assert "label_views = 120" in text and "Set `label_views: 300`" in text


def test_seeds_taken_from_the_encoder_must_be_as_many_as_it_returns():
    text = _refusal(BASELINE, opts=["model.proposal.num_seeds=512"])
    assert "SubsampledSeeds is configured for num_seeds = 512" in text
    assert "PointNet2 as configured returns 1024 points per cloud" in text
    text = _refusal(
        BASELINE, ["backbone=pointnet2-msg"], ["model.backbone.npoints=[2048,512,256,128]"]
    )
    assert "PointNet2MSG as configured returns 512 points per cloud" in text
    assert stage_contract_problems(
        backbone={"type": "PointViT", "num_group": 256}, proposal={"type": "SubsampledSeeds"}
    )
    assert not stage_contract_problems(
        backbone={"type": "PointViT", "num_group": 1024}, proposal={"type": "SubsampledSeeds"}
    )


def test_option_values_out_of_range_are_refusals_not_build_failures():
    text = _refusal(GSNET, ["view=lattice-topk"], ["model.view.top_k=400"])
    assert "LatticeViewStage has top_k = 400, but its lattice has 300 views" in text
    text = _refusal(GSNET, ["head=swad-focal"], ["model.head.friction_range=[0.5,0.2]"])
    assert "friction_range = [0.5, 0.2]" in text and "0 < min < max" in text


def test_an_adapter_that_compose_does_not_write_points_to_its_fragment():
    with pytest.raises(ComposeError, match="Compose does not write this wrapper.*section 5"):
        compose(GSNET, ["backbone=image-painted"])
    with pytest.raises(ComposeError, match="compose inserts the adapter"):
        compose(GSNET, ["backbone=feature-propagation"])


def test_a_config_that_names_a_shipped_base_counts_its_levels(tmp_path):
    """A config of one's own may inherit a shipped config by name, from any directory;
    compose walks its chain as loading does, so it says the chain is too deep instead of
    writing a config that loading would then refuse."""
    mine = tmp_path / "mine.yaml"
    mine.write_text(f"_base_: [{GSNET}.yaml]\ntrain: {{epochs: 2}}\n", encoding="utf-8")
    Config.load(mine)  # two levels: loads
    with pytest.raises(ComposeError, match="limited to 2 levels"):
        compose(mine, ["grouper=multi-scale"])
