"""Composed configs on real GraspNet-1Billion frames: the probe behind the measured page.

Needs the dataset, the label assets of its split ``sample`` and a GPU; every test skips
cleanly without them. The full grid is ``python -m tools.probe_grid run`` and is not
repeated here: these tests keep its machinery honest on a handful of cells.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from grasppanda.combine.compose import compose
from grasppanda.combine.probe import probe
from grasppanda.utils.paths import cache_root

pytest.importorskip("spconv", reason="needs the spconv extra")
pytestmark = [pytest.mark.gpu, pytest.mark.dataset, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
GSNET = "gsnet/gsnet_sunet14_g1b-realsense_10e"
BASELINE = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"


def real(report):
    """The report of a probe that read real frames; skip when this machine cannot."""
    if report.fallback is not None:
        pytest.skip(f"real frames cannot be probed here: {report.fallback[:300]}")
    return report


def test_a_shipped_method_runs_on_the_sample_frames(graspnet1b_root, cuda):
    report = real(probe(GSNET, iters=4))
    assert report.verdict == "runs" and report.data == "real", report.detail
    assert report.iterations == 4 and report.batch_size == 4 and len(report.frames) == 8
    assert all(frame.startswith("graspnet1b/scene_0000/realsense/") for frame in report.frames)
    assert report.parameters > 10_000_000 and report.peak_gpu_mib > 100
    assert set(report.losses) == {
        "loss/objectness",
        "loss/graspness",
        "loss/view",
        "loss/score",
        "loss/width",
        "loss",
    }
    assert len(report.real_seeds) == len(report.supervised_seeds) == 4
    assert report.probe_overrides[0] == "data.train.dataset.split=sample"
    assert "train.epochs=2" in report.probe_overrides  # two batches of four per epoch
    assert report.predicted_frames == [
        "graspnet1b/scene_0000/realsense/0000",
        "graspnet1b/scene_0000/realsense/0128",
    ]


def test_a_composed_swap_with_an_adapter_runs(graspnet1b_root, cuda, tmp_path):
    """A dense encoder under GraspNet-baseline: compose inserts the sub-sampling adapter
    and fits the width, and the config trains on real frames with the tolerance labels."""
    composed = compose(BASELINE, ["backbone=sparse-unet14"])
    assert composed.accepted and composed.adapter == "FurthestPointSubsample"
    path = tmp_path / "baseline_sunet14.yaml"
    path.write_text(composed.text(path), encoding="utf-8")
    report = real(probe(path, iters=4))
    assert report.verdict == "runs", report.detail
    assert report.batch_size == 2 and "loss/tolerance" in report.losses
    assert min(report.real_seeds) == 1.0  # a furthest-point subsample has no padding


def test_a_stage_set_runs_under_a_recipe_that_serves_its_labels_and_is_refused_elsewhere(
    graspnet1b_root, cuda, tmp_path
):
    """GSNet's five stages composed onto GraspNet-baseline's recipe: compose adds the
    graspness section the proposal reads, and the config trains on real frames. Onto
    EconomicGrasp's recipe, which serves compact labels, the same stages are refused by
    the config check; before that rule they failed at the first loss."""
    stages = [
        "backbone=sparse-unet14",
        "proposal=graspness-fps",
        "view=template-view",
        "assigner=NearestLabelAssigner",
        "grouper=CylinderGroup",
        "head=swad",
    ]
    composed = compose(BASELINE, stages)
    assert composed.accepted and composed.adapter is None
    assert "graspness" in composed.body["data"]["train"]["dataset"]
    path = tmp_path / "gsnet_stages_baseline_recipe.yaml"
    path.write_text(composed.text(path), encoding="utf-8")
    report = real(probe(path, iters=4))
    assert report.verdict in ("runs", "runs-unstable"), report.detail
    assert {"loss/graspness", "loss/view", "loss/score", "loss/width"} <= set(report.losses)
    assert report.batch_size == 2  # the recipe is GraspNet-baseline's
    refused = compose("economicgrasp/economicgrasp_sunet14_g1b-realsense_10e", stages)
    assert not refused.accepted and not refused.problems
    assert "NearestLabelAssigner reads `supervision.objects`" in refused.issues[0].message


def test_an_encoder_of_another_width_is_refused_or_projected_before_a_gpu_is_touched():
    """Scale-Balanced-Grasp adds seed features to crop features, so the encoder's width
    must be the crop's; PTv3 returns 64 channels. It used to pass the config check and
    fail when the grouper was built. Compose puts the projected variant of the grouper
    in place; without that, the config check refuses the pair."""
    base = "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e"
    composed = compose(base, ["backbone=ptv3"])
    assert composed.accepted
    assert composed.body["model"]["grouper"]["type"] == "DepthCylinderProjectionGroup"
    refused = compose(base, ["backbone=ptv3"], adapters=False)
    assert not refused.accepted
    assert any("PTv3 returns 64-channel features" in problem for problem in refused.problems)


def test_missing_assets_are_named_and_the_verdict_is_only_wired(
    graspnet1b_root, cuda, tmp_path, monkeypatch
):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "empty_cache"))
    report = probe(GSNET, iters=4)
    assert report.verdict == "wired" and report.data == "synthetic"
    assert "a label asset of the frames is missing" in report.fallback
    assert "grasppanda prepare graspnet1b --labels" in report.fallback
    assert "--split sample" in report.fallback
    assert report.exit_status() == 3


def test_the_runner_records_a_cell_with_its_commands_and_resumes(graspnet1b_root, cuda):
    spec = importlib.util.spec_from_file_location("probe_grid", REPO / "tools" / "probe_grid.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules["probe_grid"] = runner
    spec.loader.exec_module(runner)
    name = f"pytest-{uuid.uuid4().hex[:8]}"
    arguments = ["run", "--name", name, "--iters", "2", "--jobs", "1"]
    arguments += [
        "--only",
        "reference/GSNet",
        "encoder/GSNet/SparseUNet14",
        "stage/GSNet/head/economic",
    ]
    try:
        assert runner.main(arguments) == 0
        records = runner.load(name)
        assert set(records) == {
            "reference/GSNet",
            "encoder/GSNet/SparseUNet14",
            "stage/GSNet/head/economic",
        }
        reference = records["reference/GSNet"]
        own = records["encoder/GSNet/SparseUNet14"]
        refused = records["stage/GSNet/head/economic"]
        if reference["probe"].get("fallback"):
            pytest.skip(
                f"real frames cannot be probed here: {reference['probe']['fallback'][:200]}"
            )
        assert reference["verdict"] == "runs"
        assert reference["probe_command"] == f"grasppanda probe {GSNET} --iters 2 --frames 8"
        # The method's own encoder composes to the method's config: one experiment.
        assert own["config_hash"] == reference["config_hash"]
        assert own["probed_as"] == "reference/GSNet" and own["probe"] == reference["probe"]
        assert own["compose_command"] == f"grasppanda compose {GSNET} --set backbone=sparse-unet14"
        assert refused["verdict"] == "refused" and refused["refused_by"] == "compose"
        assert any("must agree on the label set" in message for message in refused["refusal"])
        # A second run finds everything recorded and probes nothing.
        before = (runner.directory(name) / "results.jsonl").read_text(encoding="utf-8")
        assert runner.main(arguments) == 0
        assert (runner.directory(name) / "results.jsonl").read_text(encoding="utf-8") == before
        compact = runner.compact(reference)
        assert compact["verdict"] == "runs" and "shared_machine" in compact
        assert json.dumps(compact)  # plain data
    finally:
        shutil.rmtree(cache_root() / "probe_grid" / name, ignore_errors=True)
