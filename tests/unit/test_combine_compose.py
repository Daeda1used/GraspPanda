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
    # A proposal with another seed count moves the adapter with it.
    other = compose(BASELINE, ["backbone=sparse-unet14"], ["model.proposal.num_seeds=512"])
    assert other.body["model"]["backbone"]["npoint"] == 1024  # the override is not a --set
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
    with pytest.raises(ComposeError, match=r"give the whole list \(`hooks.0=\[...\]`\)"):
        compose(GSNET, ["backbone=ptv3"], ["hooks.0.interval=5"])
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
    # Compact labels on a dataset that reads dense grids: the dataset's own check refuses.
    refused = compose(GSNET, ["assigner=compact-label", "head=economic"])
    assert not refused.accepted and not refused.problems
    (issue,) = refused.issues
    assert issue.path == "data.train.dataset" and "SampleGraspLabels" in issue.message


@pytest.mark.parametrize(
    ("base", "sets", "fragments"),
    [
        (GSNET, ["view=ObjectnessViewStage"], ["both predict and supervise the 'objectness'"]),
        (GSNET, ["assigner=scale-balanced"], ["do not apply seed weights"]),
        (GSNET, ["head=economic"], ["must agree on the label set"]),
        (GSNET, ["grouper=DepthCylinderGroup"], ["SWADHead expects ('seed', 'channel')"]),
        (BASELINE, ["assigner=NearestLabelAssigner"], ["`supervise: object_seeds`"]),
        (BASELINE, ["proposal=graspness-fps"], ["both predict and supervise the 'objectness'"]),
        (SBG, ["view=TemplateViewStage"], ["reads the seed score 'objectness'"]),
    ],
)
def test_a_refused_combination_quotes_the_rules_and_writes_nothing(
    base, sets, fragments, tmp_path, capsys
):
    result = compose(base, sets)
    assert not result.accepted and result.config_hash is None
    for fragment in fragments:
        assert any(fragment in problem for problem in result.problems), result.problems
    # Every problem is a message of the contract rules on the composed stage set.
    model = Config.load(base).to_dict()["model"]
    stages = {slot: model[slot] for slot in STAGE_REGISTRIES}
    for choice in result.choices:
        if choice.slot != "backbone":
            stages[choice.slot] = choice.node
    stages["backbone"] = result.body["model"].get("backbone", stages["backbone"])
    assert result.problems == list(dict.fromkeys(stage_contract_problems(**stages)))
    target = tmp_path / "refused.yaml"
    arguments = ["compose", base, "-o", str(target)]
    for item in sets:
        arguments += ["--set", item]
    assert main(arguments) == 1
    out = " ".join(capsys.readouterr().out.split())  # the report wraps its paragraphs
    assert not target.exists() and "refused: nothing was written" in out
    assert all(fragment in out for fragment in fragments)


def test_the_bare_component_is_composed_with_its_defaults_and_points_at_the_preset():
    result = compose(GSNET, ["backbone=PTv3"])
    assert result.accepted and result.body == {"model": {"backbone": {"type": "PTv3"}}}
    assert result.origins["backbone"] == "constructor defaults"
    assert any("Presets for it: ptv3" in action for action in result.actions)
    bare = compose(GSNET, ["grouper=MultiScaleGroup"])
    assert not bare.accepted and "requires 'branch'" in bare.issues[0].message
    assert compose(GSNET, ["grouper=multi-scale"]).accepted  # which the preset supplies


def test_requests_that_cannot_be_composed_say_why():
    with pytest.raises(ComposeError, match=r"no preset 'ptv33' for slot 'backbone'.*'ptv3'"):
        compose(GSNET, ["backbone=ptv33"])
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


def test_a_detector_without_slots_says_what_cannot_be_done_yet(tmp_path, capsys):
    assert component_slots(DETECTORS.get("HGGD")) == {}
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
    assert "HGGD has no component slots" in message and "grasppanda components HGGD" in message
    assert "--opts model.<part>.<option>=<value>" in message
    assert main(["compose", HGGD, "--set", "backbone=ptv3"]) == 2
    assert "HGGD has no component slots" in capsys.readouterr().err
    # Its options can still be changed, and the config check is the verdict.
    result = compose(HGGD, overrides=["model.encoder.base_width=16"])
    assert result.accepted and not result.seed_based and result.verdicts == []
    assert loaded(result, tmp_path).select("model.encoder.base_width") == 16
    assert main(["compose", HGGD, "--opts", "model.encoder.base_width=16"]) == 0
    assert "not built on the seed-based skeleton" in capsys.readouterr().out


def test_adapters_are_found_from_the_registry_not_from_a_list():
    adapters = sorted(name for name in BACKBONES.names() if is_adapter(BACKBONES.get(name)))
    assert adapters == ["FeaturePropagation", "FurthestPointSubsample"]
    assert not is_adapter(BACKBONES.get("PointNet2"))


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
    assert max(len(line) for line in text.splitlines() if not line.startswith("  - ")) <= 100
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
    assert "✓ valid (hash 3ee8770e1e94)" in out and f"✓ written {target}" in out
    assert f"grasppanda probe {target}" in out
    assert Config.load(target).digest().startswith("3ee8770e1e94")
    # Without -o nothing is written and the report says how to get the file.
    assert main(["compose", GSNET, "--set", "backbone=ptv3"]) == 0
    assert "Pass -o FILE" in capsys.readouterr().out


def test_the_command_as_json_and_its_listings(capsys):
    assert main(["compose", GSNET, "--set", "backbone=pointnet2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"] and report["adapter"] == "FeaturePropagation"
    assert report["config_hash"].startswith("3ee8770e1e94") and report["output"] is None
    assert parse_yaml(report["config"])["model"]["backbone"]["type"] == "FeaturePropagation"
    assert [entry["slots"] for entry in report["contracts"]] == [list(g) for g in CONTRACT_GROUPS]
    assert main(["compose", GSNET, "--set", "view=ObjectnessViewStage", "--json"]) == 1
    refused = json.loads(capsys.readouterr().out)
    assert not refused["accepted"] and len(refused["problems"]) == 2

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
    assert stop.value.code == 0 and "--set" in capsys.readouterr().out
