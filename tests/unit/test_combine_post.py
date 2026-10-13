"""Post-processing chains from short text: presets, compose, the commands, the probe."""

from __future__ import annotations

import argparse
import types

import pytest
import torch

from grasppanda.cli import test as test_command
from grasppanda.combine.compose import ComposeError, compose, recompose
from grasppanda.combine.grid import FAMILIES, METHODS, POST_METHODS, POST_NOT_PROBED, cells
from grasppanda.combine.post import PostSpecError, chain_nodes, chain_override, step_node
from grasppanda.combine.presets import presets
from grasppanda.combine.probe import ProbeReport, _post_process
from grasppanda.config import Config, ConfigError
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation, PointSet, SampleKey

GSNET, HGGD = METHODS["GSNet"], FAMILIES["HGGD"]


def test_a_step_is_a_preset_or_a_component_with_a_main_option_or_any_options():
    assert step_node("top-k") == {"type": "TopK", "k": 50}
    assert step_node("top-k=20") == {"type": "TopK", "k": 20}
    assert step_node("TopK") == {"type": "TopK"}
    assert step_node("gripper-limits=0.085")["max_width"] == 0.085
    assert step_node("workspace:x=[-0.3,0.3],z=[0.2,0.9]") == {
        "type": "WorkspaceFilter",
        "x": [-0.3, 0.3],
        "z": [0.2, 0.9],
    }
    assert step_node("score-threshold:threshold=1.5,mode=absolute")["mode"] == "absolute"
    assert step_node("approach-cone=45")["max_angle"] == 45
    assert step_node("ApproachAngleFilter:direction=camera")["direction"] == "camera"
    chain = chain_nodes(["collision+nms", "top-k=1e+1"])
    assert [node["type"] for node in chain] == ["ModelFreeCollisionFilter", "GraspNMS", "TopK"]
    for node in chain_nodes("+".join(presets()["post"])):  # every preset builds
        if node["type"] != "GripperMeshCollisionFilter":
            POSTPROCESSORS.build(node)


def test_a_step_that_cannot_be_read_says_what_can():
    with pytest.raises(PostSpecError, match=r"no post-processing step 'topk'. Did you mean"):
        step_node("topk")
    with pytest.raises(PostSpecError, match=r"TopK has no option 'n'. Its options: k"):
        step_node("top-k:n=5")
    with pytest.raises(PostSpecError, match=r"names no main option.*`nms:<option>=<value>`"):
        step_node("nms=0.05")
    with pytest.raises(PostSpecError, match="empty post-processing chain"):
        chain_nodes([])


@pytest.mark.parametrize("base", [GSNET, HGGD], ids=["GSNet", "HGGD"])
def test_compose_writes_the_chain_as_the_variant_post_of_any_detector(base, tmp_path):
    composed = compose(base, ["post=nms+top-k=20+workspace:z=[0.2,0.9]"])
    assert composed.accepted, composed.refusal()
    assert composed.body["evaluation"]["variants"]["post"] == [
        {"type": "GraspNMS", "translation_threshold": 0.03, "rotation_threshold": 30.0},
        {"type": "TopK", "k": 20},
        {"type": "WorkspaceFilter", "z": [0.2, 0.9]},
    ]
    assert "--set 'post=nms+top-k=20+workspace:z=[0.2,0.9]'" in composed.command
    file = tmp_path / "composed.yaml"
    file.write_text(composed.text(file), encoding="utf-8")
    variants = Config.load(file)["evaluation"]["variants"]
    assert list(variants) == ["raw", "cd", "post"]  # beside the variants of the base config
    assert recompose(file).config_hash == composed.config_hash


def test_compose_refuses_a_wrong_chain_with_the_reason():
    with pytest.raises(ComposeError, match="no post-processing step 'top'"):
        compose(GSNET, ["post=top"])
    with pytest.raises(ComposeError, match="set twice.*joined by `\\+`"):
        compose(GSNET, ["post=nms", "post=top-k"])
    refused = compose(GSNET, ["post=top-k=many"])
    assert not refused.accepted
    assert any("evaluation.variants.post" in line for line in refused.refusal())


def test_the_post_option_of_test_and_infer_is_an_override_of_the_variant_post():
    args = argparse.Namespace(
        opts=[], weights=None, split=None, work_dir=None, post=["nms", "top-k=20"]
    )
    (override,) = test_command.overrides(args)
    assert override == chain_override(["nms", "top-k=20"])
    config = Config.load(GSNET, [override])
    assert [node["type"] for node in config["evaluation"]["variants"]["post"]] == [
        "GraspNMS",
        "TopK",
    ]
    args.post = ["topk"]
    with pytest.raises(ConfigError, match="no post-processing step 'topk'"):
        test_command.overrides(args)


def test_the_probe_records_what_every_step_of_the_variant_post_left():
    key = SampleKey("unit", 0, "none", 0)
    grasps = GraspGroup(
        translation=torch.rand(30, 3),
        rotation=torch.eye(3).expand(30, 3, 3).clone(),
        width=torch.linspace(0.02, 0.1, 30),
        depth=torch.full((30,), 0.02),
        score=torch.rand(30),
    )
    observation = Observation(keys=[key], points=PointSet(torch.rand(50, 3)))
    cfg = {"evaluation": {"variants": {"raw": [], "post": chain_nodes("gripper-limits+top-k=5")}}}
    report = ProbeReport()
    _post_process(report, cfg, [(observation, [grasps])], torch.device("cpu"))
    assert report.postprocessing["steps"] == ["GripperLimits", "TopK"]
    ((predicted, limited, kept),) = report.postprocessing["grasps"]
    assert (predicted, kept) == (30, 5) and 5 < limited < 30
    assert len(report.postprocessing["seconds"][0]) == 2
    untouched = ProbeReport()
    _post_process(untouched, {"evaluation": {"variants": {"raw": []}}}, [], torch.device("cpu"))
    assert untouched.postprocessing == {}


def test_every_post_preset_has_its_cells_or_a_stated_reason():
    names = set(presets()["post"])
    assert set(POST_NOT_PROBED) <= names
    post = list(cells(("post",)))
    assert {(cell.variant, cell.method) for cell in post} == {
        (name, method) for name in names - set(POST_NOT_PROBED) for method in POST_METHODS
    }
    assert all(cell.sets == (f"post={cell.variant}",) for cell in post)
    # every registered post-processor has a preset: nobody has to guess its options
    in_tree = {
        name
        for name in POSTPROCESSORS.names()
        if POSTPROCESSORS.get(name).__module__.startswith("grasppanda.")
    }
    assert in_tree == {preset.type for preset in presets()["post"].values()}
    assert isinstance(types.MappingProxyType(POST_NOT_PROBED)["target-mask"], str)
