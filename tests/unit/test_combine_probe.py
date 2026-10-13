"""``grasppanda probe``: one verdict per config, and never more than was run.

The real path is exercised on the example project (a toy detector and a toy dataset that
train on a CPU in a second); the wiring check on tiny seed-based models and on a tiny
HGGD. What a probe says about real GraspNet-1Billion frames is the subject of
``tests/integration/test_combine_grid.py``.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import yaml

from grasppanda.cli.main import main
from grasppanda.combine.probe import EXIT_STATUS, VERDICTS, ProbeReport, _component, _phase, probe
from grasppanda.combine.synthetic import HISTOGRAM_KEY, random_batch, scene_batch
from grasppanda.config import Config
from grasppanda.models.assigners import scale_balance
from grasppanda.structures import COMPACT_LABELS

from . import test_swap_matrix as matrix
from .conftest import REPO

TOY = str(REPO / "projects/example_project/configs/toy_train.yaml")
needs_spconv = pytest.mark.skipif(
    importlib.util.find_spec("spconv") is None, reason="needs the spconv extra"
)


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def write(tmp_path: Path, name: str, content: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return str(path)


def tiny(detector: str, backbone: str, **replace: str) -> dict:
    """A tiny seed-based model on CPU: the swap matrix's stages around one encoder."""
    return {
        "model": matrix.method(detector, matrix.encoder(backbone), **replace),
        "optimizer": {"type": "Adam", "lr": 1e-3},
        "runtime": {"device": "cpu"},
    }


# ----------------------------------------------------------------------- the vocabulary


def test_the_verdicts_are_a_fixed_documented_vocabulary():
    assert list(VERDICTS) == ["runs", "runs-unstable", "fails", "refused", "wired"]
    assert set(EXIT_STATUS) == set(VERDICTS)
    assert all(len(meaning) > 60 and meaning.endswith(".") for meaning in VERDICTS.values())
    assert EXIT_STATUS["runs"] == 0 and all(
        EXIT_STATUS[name] != 0 for name in ("runs-unstable", "fails", "refused")
    )
    # `wired` is success only when the wiring check is what was asked for.
    report = ProbeReport(verdict="wired")
    assert report.exit_status() == 3 and report.exit_status(synthetic_requested=True) == 0
    page = (REPO / "docs/user_guides/combinations.md").read_text(encoding="utf-8")
    # The guide defines every verdict in the words of the code, so the two cannot drift.
    for name, meaning in VERDICTS.items():
        assert f"| `{name}` | {meaning} |" in page, name


# ---------------------------------------------------------------------------- real path


def test_a_config_that_trains_is_probed_with_the_real_trainer(capsys):
    report = probe(TOY, iters=6, predict_frames=3)
    assert report.verdict == "runs" and report.data == "real" and report.fallback is None
    assert report.iterations == report.iterations_requested == 6 and report.batch_size == 4
    assert report.detector == "ToyDetector" and report.parameters == 7601
    assert report.device == "cpu" and report.peak_gpu_mib is None
    assert set(report.losses) == {"loss/score", "loss/width", "loss"}
    assert all(len(pair) == 2 for pair in report.losses.values())
    first, last = report.losses["loss"]
    assert first == pytest.approx(sum(report.losses[k][0] for k in ("loss/score", "loss/width")))
    assert last < first  # six steps of Adam on a learnable toy problem
    assert len(report.frames) == 24 and report.frames[0].startswith("toy/")
    assert len(report.grasps_per_frame) == len(report.predicted_frames) == 3
    assert report.nonfinite == [] and report.failure is None and report.refusal == []
    assert report.seconds_per_iteration is not None and report.seconds_per_iteration >= 0
    # What the probe changed is spelled as overrides of `grasppanda train` ...
    assert report.probe_overrides == [
        "data.train.num_workers=0",
        "evaluation.interval=0",
        "evaluation.save_best=null",
        "train.epochs=1",
        "data.val.num_workers=0",
    ]
    assert any("has no `split` option" in note for note in report.notes)
    # ... and it is the same experiment: the hash is that of the config as given.
    assert report.config_hash == Config.load(TOY).digest()
    assert json.loads(json.dumps(report.to_dict()))["verdict"] == "runs"


def test_the_probe_is_the_trainer_of_the_config_not_a_loop_of_its_own(monkeypatch):
    """Same weights after the same number of iterations as ``Trainer.train``: recipe,
    seed, data order, hooks."""
    from grasppanda.engine import trainer as engine

    seen: dict[str, torch.Tensor] = {}
    original = engine.Trainer.train

    def recording(self):
        try:
            return original(self)
        finally:
            seen.update({k: v.detach().clone() for k, v in self.state.model.state_dict().items()})

    monkeypatch.setattr(engine.Trainer, "train", recording)
    report = probe(TOY, iters=8)
    assert report.verdict == "runs"
    probed = dict(seen)
    seen.clear()

    from grasppanda.engine.run import Run, resolve_device

    config = Config.load(TOY, [*report.probe_overrides], command="train")
    cfg = config.resolved()
    run = Run(Path("unused"), "plain", "train", config, cfg, resolve_device("cpu"))
    plain = engine.Trainer(run)
    plain.hooks = []
    plain.train()  # one epoch of eight batches: exactly the eight iterations of the probe
    assert plain.state.iteration == 8 and probed.keys() == seen.keys()
    assert all(torch.equal(probed[name], seen[name]) for name in probed)


def test_a_refused_config_is_refused_before_anything_is_built(capsys):
    report = probe(TOY, ["model.head.num_angles=6"])
    assert report.verdict == "refused" and report.parameters is None and report.iterations == 0
    assert any("Did you mean 'num_angle'" in message for message in report.refusal)
    assert main(["probe", TOY, "--opts", "model.grouper.type=ToyDepthGrouper"]) == 2
    out = capsys.readouterr().out
    assert "✗ refused: the config check reports 1 problem(s)." in out
    assert "ToyHead expects ('seed', 'channel')" in out
    # A config `train` cannot run is refused by the same check as `print-config --for train`.
    model_only = str(REPO / "projects/example_project/configs/toy_detector.yaml")
    assert probe(model_only).refusal == [
        "data.train: `grasppanda train` needs this section, but it is not set."
    ]


def test_a_failure_names_the_phase_and_the_component(tmp_path):
    """Passes the config check, fails when built: the gated seed residual of
    Scale-Balanced-Grasp needs the encoder's width, which only the built encoder knows."""
    config = tiny("ScaleBalancedGrasp", "PointNet2")
    config["model"]["grouper"]["mlp_channels"] = [8, 8]  # the encoder returns 12 channels
    path = write(tmp_path, "narrow.yaml", config)
    report = probe(path, synthetic=True)
    assert report.verdict == "fails" and report.iterations == 0
    assert report.failure["phase"] == "build" and report.failure["error"] == "ValueError"
    assert report.failure["component"] == "grouper DepthCylinderGroup"
    assert "12-channel seed features to 8-channel crop features" in report.failure["message"]
    assert report.detail.startswith("at build, in the grouper DepthCylinderGroup: ValueError:")
    assert main(["probe", path, "--synthetic"]) == 1


def test_phase_and_component_are_read_from_the_traceback():
    from grasppanda.engine.builders import build_model
    from grasppanda.registry import BuildError

    with pytest.raises(BuildError) as caught:
        build_model({"model": {"type": "GSNet", "backbone": {"type": "PointNet2"}}})
    assert _phase(caught.value, "loss") == "build"

    def backward():
        raise RuntimeError("in a detector's loss")

    with pytest.raises(RuntimeError) as plain:
        backward()
    assert _phase(plain.value, "loss") == "loss" and _component(plain.value) == ""


# ------------------------------------------------------------------ the wiring check


def test_without_the_dataset_the_probe_says_so_and_only_claims_wired(tmp_path, capsys):
    config = tiny("GSNet", "PointNet2Dense")
    config["data"] = {
        "train": {
            "dataset": {
                "type": "GraspNet1B",
                "camera": "realsense",
                "split": "train",
                "root": str(tmp_path / "no_dataset_here"),
                "preprocess": [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}],
                "transforms": [{"type": "RandomSample", "num_points": 600}],
            },
            "batch_size": 2,
        }
    }
    path = write(tmp_path, "tiny_gsnet.yaml", config)
    report = probe(path, iters=20)
    assert report.verdict == "wired" and report.data == "synthetic"
    assert "the dataset is not on this machine" in report.fallback
    assert "no_dataset_here" in report.fallback and "GRASPPANDA_GRASPNET1B_ROOT" in report.fallback
    assert report.probe_overrides == [] and all(f.startswith("synthetic/") for f in report.frames)
    assert any("graspness_threshold was lowered" in note for note in report.notes)
    assert report.iterations == report.iterations_requested == 2  # a wiring check is short
    assert "synthetic scene with random labels" in report.detail
    assert set(report.losses) >= {"loss/objectness", "loss/graspness", "loss/view", "loss"}
    assert report.grasps_per_frame and report.real_seeds == []
    # As a fallback, `wired` is not success: exit status 3, and the output says why.
    assert main(["probe", path]) == 3
    out = capsys.readouterr().out
    assert "NOT READ: the dataset is not on this machine" in out
    assert "SYNTHETIC scene with random labels" in out and "✓ wired:" in out
    assert "Real frames were not probed" in out
    # Asked for, it is: the check that was requested passed.
    assert main(["probe", path, "--synthetic", "--json"]) == 0
    asked = json.loads(capsys.readouterr().out)
    assert asked["verdict"] == "wired" and asked["fallback"] is None


@pytest.mark.parametrize(
    ("detector", "backbone"),
    [
        ("GSNet", "PointNet2Dense"),
        ("EconomicGrasp", "PointNeXt"),
        ("GraspNetBaseline", "PointNet2"),
        ("ScaleBalancedGrasp", "PointNet2Transformer"),
    ],
)
def test_every_seed_based_method_passes_the_wiring_check_in_small(detector, backbone, tmp_path):
    report = probe(write(tmp_path, "tiny.yaml", tiny(detector, backbone)), synthetic=True)
    assert report.verdict == "wired", report.detail
    assert report.iterations == 2 and report.nonfinite == [] and len(report.grasps_per_frame) == 2
    assert all(torch.isfinite(torch.tensor(pair)).all() for pair in report.losses.values())


def test_the_wiring_check_knows_an_image_based_detector(tmp_path):
    from .test_hggd_detector import TINY

    config = {"model": TINY, "optimizer": {"type": "AdamW", "lr": 1e-3}}
    config["runtime"] = {"device": "cpu"}
    report = probe(write(tmp_path, "tiny_hggd.yaml", config), synthetic=True)
    assert report.verdict == "wired", report.detail
    assert set(report.losses) >= {"loss/heatmap", "loss/anchor", "loss"}


def test_the_scene_batch_has_every_label_at_the_sizes_asked_for():
    observation, supervision = scene_batch(
        frames=2, points=500, views=7, angles=3, depths=2, label_points=9, feature_channels=6
    )
    assert observation.points.coords.shape == (1000, 3) and observation.point_feats.shape == (
        1000,
        6,
    )
    assert torch.equal(observation.point_feats[:, :3], observation.points.coords)
    assert observation.rgb is None and len(observation.keys) == 2
    depth = observation.points.coords[:, 2]
    assert float(depth.min()) > 0.2 and float(depth.max()) < 1.5  # metres, a table top
    assert set(supervision.point_labels) == {"objectness", "graspness"}
    assert 0 < int(supervision.point_labels["objectness"].sum()) < 1000
    for frame in range(2):
        dense, compact = supervision.objects[frame], supervision.extras[COMPACT_LABELS][frame]
        assert dense and len(dense) == len(compact)
        assert dense[0].scores.shape == dense[0].tolerance.shape == (9, 7, 3, 2)
        assert compact[0].view_graspness.shape == (9, 7)
        # Label points lie on the cloud: a seed can be within any matching distance.
        cloud = observation.points.coords[frame * 500 : (frame + 1) * 500]
        placed = dense[0].points @ dense[0].pose[:3, :3].T + dense[0].pose[:3, 3]
        assert float(torch.cdist(placed, cloud).min(dim=1).values.max()) < 0.01
    assert HISTOGRAM_KEY == scale_balance.HISTOGRAM_KEY and HISTOGRAM_KEY in supervision.extras
    again, _ = scene_batch(frames=2, points=500, views=7, angles=3, depths=2, label_points=9)
    assert torch.equal(again.points.coords, observation.points.coords) and again.point_feats is None
    pictures, labels = scene_batch(frames=1, points=300, images=True)
    assert pictures.rgb.shape == (1, 3, 720, 1280) and pictures.depth.shape == (1, 720, 1280)
    assert pictures.intrinsics.shape == (1, 3, 3)
    assert labels.extras["view_grasps"][0]["centers"].shape == (60, 2)


def test_the_random_batch_is_the_swap_matrix_batch():
    observation, supervision = random_batch(seed=3)
    theirs, _ = matrix.batch(seed=3)
    assert torch.equal(observation.points.coords, theirs.points.coords)
    assert supervision.objects[0][0].scores.shape == (20, 10, 3, 2)


def test_the_command_prints_the_numbers_and_one_verdict(capsys):
    assert main(["probe", TOY, "--iters", "4", "--predict-frames", "1"]) == 0
    out = capsys.readouterr().out
    for fragment in (
        "ToyDetector · 7,601 parameters",
        "frames of the config's dataset · 4 per iteration",
        "as `grasppanda train` with data.train.num_workers=0",
        "trained     4 of 4 iterations",
        "measured on this machine now",
        "loss/score",
        "predicted   1 frame(s)",
        "✓ runs: trained 4 iterations on 16 frames of the dataset and predicted on 1 frame(s)",
    ):
        assert fragment in out, fragment
    assert main(["probe", TOY, "--iters", "4", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "runs" and report["iterations"] == 4
    with pytest.raises(SystemExit) as stop:
        main(["probe", "--help"])
    assert stop.value.code == 0 and "--synthetic" in capsys.readouterr().out
