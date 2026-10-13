"""The component catalogue: ``grasppanda components`` and the generated pages.

The catalogue is only worth having if it cannot lie. So: every registered component
describes itself, the command shows what the registries hold, and the two generated
documentation pages are compared with what the code produces right now.
"""

from __future__ import annotations

import importlib.util
import inspect
import json

import pytest

from grasppanda.cli.components import (
    CONTRACT_FIELDS,
    FIELD_HELP,
    SLOTS,
    collect,
    compatibility,
    parameters,
    resolve_slot,
)
from grasppanda.cli.main import main
from grasppanda.config import Config, available_configs
from grasppanda.config.validate import iter_components
from grasppanda.registry import all_registries

from .conftest import REPO

STATUSES = {"tested", "weights-verified", ""}


def docs_tool():
    path = REPO / "tools" / "make_component_docs.py"
    spec = importlib.util.spec_from_file_location("make_component_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def in_tree():
    return collect(in_tree_only=True)


# ------------------------------------------------------- what components say of themselves


def test_every_registry_is_a_documented_slot_and_every_contract_field_is_explained():
    assert set(all_registries()) == set(SLOTS)
    assert set(CONTRACT_FIELDS) <= set(SLOTS)
    used = {field for fields in CONTRACT_FIELDS.values() for field in fields}
    assert used == set(FIELD_HELP)


def test_every_component_states_its_source_summary_and_status():
    """On the class itself: a subclass does not inherit a citation or a proof."""
    entries = in_tree()
    assert len(entries) > 40
    for entry in entries:
        cls = all_registries()[entry.registry].get(entry.name)
        where = f"{entry.registry}.{entry.name}"
        assert isinstance(vars(cls).get("source"), str) and entry.source.strip(), where
        assert entry.summary.endswith((".", "!", "?")) and len(entry.summary) > 10, where
        assert "\n" not in entry.summary and ":class:" not in entry.summary, where
        assert entry.status in STATUSES, where
        assert entry.status == vars(cls).get("status", ""), where
        assert all(isinstance(extra, str) for extra in entry.extras), where
        assert inspect.getdoc(cls), where
    by_name = {entry.name: entry for entry in entries}
    assert by_name["SparseUNet14"].extras == ("spconv",)
    assert by_name["PointNet2"].extras == ()
    assert "Wang et al., ICCV 2021" in by_name["GSNet"].source
    assert by_name["FurthestPointSubsample"].summary == (
        "Turn a `dense` backbone into a `subsampled` one by furthest point sampling."
    )


def test_stages_of_the_pipeline_declare_every_field_of_their_contract():
    for entry in in_tree():
        for field in CONTRACT_FIELDS.get(entry.registry, ()):
            assert field in entry.contract, f"{entry.name} does not declare {field}"
    contracts = {entry.name: entry.contract for entry in in_tree()}
    assert contracts["PointNet2"] == {"output": "subsampled"}
    assert contracts["GraspnessFPS"] == {
        "requires": "dense",
        "seed_scores": ["graspness", "objectness"],
    }
    assert contracts["DepthCylinderGroup"] == {
        "support": "scene",
        "out_axes": ["seed", "depth", "channel"],
    }
    assert contracts["EconomicHead"]["label_format"] == "compact"
    assert contracts["ScaleBalancedAssigner"]["optional_outputs"]["weight"] is None
    assert contracts["HGGD"]["required_inputs"] == ["depth", "intrinsics", "points", "rgb"]


def test_shipped_configs_are_attributed_by_the_slot_a_component_sits_in():
    usage = {(entry.registry, entry.name): entry.used_by for entry in in_tree()}
    gsnet = "gsnet/gsnet_sunet14_g1b-realsense_10e"
    assert gsnet in usage["BACKBONES", "SparseUNet14"]
    assert gsnet in usage["HEADS", "SWADHead"] and gsnet in usage["LOSSES", "SmoothL1"]
    assert gsnet in usage["POSTPROCESSORS", "ModelFreeCollisionFilter"]
    assert (
        gsnet in usage["TRANSFORMS", "RandomRotation"] and gsnet in usage["DATASETS", "GraspNet1B"]
    )
    assert gsnet not in usage["HEADS", "EconomicHead"]
    assert all(config in available_configs() for configs in usage.values() for config in configs)
    # iter_components is the walker behind it: paths, registries and names of one config.
    found = {
        (".".join(map(str, path)), registry.name, name)
        for path, registry, name in iter_components(Config.load(gsnet).to_dict())
    }
    assert ("model", "DETECTORS", "GSNet") in found
    assert ("model.proposal.graspness_loss", "LOSSES", "SmoothL1") in found
    assert ("data.train.dataset.transforms.0", "TRANSFORMS", "RandomSample") in found
    assert ("evaluation.variants.cd.0", "POSTPROCESSORS", "ModelFreeCollisionFilter") in found
    assert ("optimizer", "OPTIMIZERS", "Adam") in found


# ------------------------------------------------------------------------ the command


def test_the_command_lists_every_slot_and_component(capsys):
    assert main(["components"]) == 0
    out = capsys.readouterr().out
    for registry, (where, _) in SLOTS.items():
        assert registry in out and where in out
    for entry in in_tree():
        assert entry.name in out
    assert "weights-verified" in out and "GSNet (Wang et al., ICCV 2021)" in out
    assert "AdamW" in out and "OneCycleLR" in out  # third-party registries, names only
    assert "grasppanda components <name>" in out


@pytest.mark.parametrize("spelling", ["backbones", "backbone", "BACKBONES", "model.backbone"])
def test_one_slot_can_be_selected_in_any_spelling(spelling, capsys):
    assert resolve_slot(spelling) == "BACKBONES"
    assert main(["components", "--slot", spelling]) == 0
    out = capsys.readouterr().out
    assert "SparseUNet14" in out and "SWADHead" not in out
    assert "output:" in out and "subsampled" in out  # the legend of the contract column


def test_an_unknown_slot_or_component_is_an_error_with_candidates(capsys):
    assert main(["components", "--slot", "necks"]) == 2
    assert "unknown slot 'necks'" in capsys.readouterr().err
    assert main(["components", "SparseUnet14"]) == 2
    assert "Did you mean 'SparseUNet14'" in capsys.readouterr().err


def test_json_output_is_the_collected_data(capsys):
    assert main(["components", "--slot", "head", "--slot", "grouper", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {item["registry"] for item in data} == {"GROUPERS", "HEADS"}
    swad = next(item for item in data if item["name"] == "SWADHead")
    assert swad["contract"]["in_axes"] == ["seed", "channel"]
    assert swad["status"] == "tested" and swad["source"] and swad["summary"]
    assert "gsnet/gsnet_sunet14_g1b-realsense_10e" in swad["used_by"]
    assert all("gsnet" in config for config in swad["used_by"])  # GSNet and its swap configs
    assert set(swad) == {
        "registry",
        "name",
        "module",
        "summary",
        "source",
        "status",
        "contract",
        "extras",
        "used_by",
        "in_tree",
    }


def test_one_component_shows_its_options_and_what_it_combines_with(capsys):
    assert main(["components", "PointNet2"]) == 0
    out = capsys.readouterr().out
    assert "PointNet2  BACKBONES  (model.backbone)" in out
    assert "weights-verified" in out and "nothing beyond the core install" in out
    assert "npoints" in out and "[2048, 1024, 512, 256]" in out and "fps_tie_break" in out
    assert "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e" in out
    assert "✓ SubsampledSeeds" in out
    assert "✗ GraspnessFPS: GraspnessFPS requires 'dense' scene features" in out
    assert "{type: FeaturePropagation, backbone: {...}}" in out  # the way out, in place

    assert main(["components", "SWADHead"]) == 0
    out = capsys.readouterr().out
    assert "injected by the parent" in out  # in_channels and gripper come from the detector
    assert "✓ CylinderGroup" in out and "✗ DepthCylinderGroup" in out
    assert "✗ ScaleBalancedAssigner" in out and "✓ NearestLabelAssigner" in out


def test_parameters_and_compatibility_as_data():
    entries = {entry.name: entry for entry in in_tree()}
    options = {name: (kind, default) for name, kind, default in parameters(entries["GraspnessFPS"])}
    assert options["num_seeds"] == ("int", "1024")
    assert options["fps_skip_near_origin"] == ("bool", "true")
    assert options["in_channels"][1] == "injected by the parent"
    assert options["objectness_loss"][1] == "null"
    verdicts = compatibility(entries["ObjectnessViewStage"])
    assert set(verdicts) == {"PROPOSALS", "ASSIGNERS", "HEADS"}
    assert verdicts["PROPOSALS"]["SubsampledSeeds"] == []
    assert "both predict and supervise" in verdicts["PROPOSALS"]["GraspnessFPS"][0]
    assert "`supervise: object_seeds`" in verdicts["ASSIGNERS"]["NearestLabelAssigner"][0]
    assert compatibility(entries["GraspNMS"]) == {}  # not a stage of the skeleton


def test_out_of_tree_components_of_a_config_are_listed_as_plugins(capsys):
    project = REPO / "projects" / "example_project" / "configs"
    config = next(iter(sorted(project.glob("*.yaml"))))
    assert main(["components", "--config", str(config), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    plugins = [item for item in data if not item["in_tree"] and item["summary"]]
    assert plugins, "the example project registers components of its own"
    assert all(item["module"].split(".")[0] != "grasppanda" for item in plugins)


# ---------------------------------------------------------------- the generated pages


def test_the_generated_pages_are_current():
    """Fails when a component, a contract, a rule or a shipped config changed without
    `python -m tools.make_component_docs`."""
    tool = docs_tool()
    outdated = [path.relative_to(REPO).as_posix() for path in tool.stale()]
    assert not outdated, f"stale: {outdated}; run `python -m tools.make_component_docs`"


def test_the_catalogue_page_holds_every_component_and_only_in_tree_ones():
    page = (REPO / "docs" / "components.md").read_text(encoding="utf-8")
    assert page.startswith("<!-- Generated by tools/make_component_docs.py")
    for entry in in_tree():
        assert f"| `{entry.name}` |" in page, entry.name
        assert entry.source in page
    for name in ("TBackbone", "THead", "TDetector"):  # registered by the test suite
        assert name not in page
    # Optimizers and schedulers of PyTorch differ between its versions: named in the
    # terminal, where the installed version is the one that counts, not on the page.
    assert "## Head bundles" in page and "`Adam`" not in page and "`OneCycleLR`" not in page
    assert "`grasppanda components --slot optimizers` lists them" in page


def test_the_swapping_page_states_what_the_matrix_test_proves():
    from . import test_swap_matrix as matrix

    tool = docs_tool()
    page = (REPO / "docs" / "advanced_guides" / "swapping.md").read_text(encoding="utf-8")
    for name in ("encoders", "pairs", "stage-sets"):
        assert (
            f"<!-- generated:{name}:begin -->" in page and f"<!-- generated:{name}:end -->" in page
        )
    # Encoders: the page says "as is" or "in <adapter>" exactly where the test ran it so.
    verdicts = tool.encoder_verdicts()
    assert set(verdicts) == set(matrix.ENCODERS)
    for name, per_method in verdicts.items():
        assert set(per_method) == set(matrix.METHODS)
        for detector, verdict in per_method.items():
            node = matrix.method(detector, matrix.encoder(name))
            wanted = all_registries()["PROPOSALS"].get(node["proposal"]["type"]).requires
            direct = matrix.kind_of(node["backbone"]) == wanted
            assert (verdict == "") == direct, (name, detector)
            if not direct:
                assert verdict == matrix.ADAPTERS[wanted]({})["type"], (name, detector)
    # Stage sets: the same sets the matrix found legal and ran, by component type.
    documented = {
        tuple(chosen[slot].split(" (")[0] for slot in tool.SLOT_ORDER)
        for chosen in tool.legal_stage_sets()
    }
    ran = {
        (
            chosen["kind"],
            *(matrix.STAGES[slot][chosen[slot]]["type"] for slot in tool.SLOT_ORDER[1:]),
        )
        for chosen in matrix.stage_sets()
        if not matrix.refusal(matrix.node_of("GSNet", chosen))
    }
    assert documented == ran
    assert f"**{len(tool.legal_stage_sets())} of " in page
