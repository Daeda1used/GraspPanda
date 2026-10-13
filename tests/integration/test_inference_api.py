"""``GraspInferencer`` and ``grasppanda analyze`` on real frames, real weights and real runs.

Skipped without GraspNet-1Billion, a CUDA device and converted weights. The weights are
looked up below ``$GRASPPANDA_CACHE`` (they are converted author checkpoints under
non-commercial licences, or weights trained on such data, and are not committed)::

    runs/author_ckpt_smoke/{baseline_rs,sbg_rs,hggd_rs}.pth
    weights/converted/{gsnet_local_enhanced_rs_e18,economicgrasp_author_kinect}.pth

What is compared, and how tightly.

* **The engine's dump.** ``grasppanda infer`` (the prediction half of ``grasppanda
  test``) is run on the eight frames of the ``sample`` split, with the four loader
  workers of the shipped configs. For the four seed-based methods every dumped array,
  raw and collision-filtered, has to be **bit-identical** to what the inferencer returns
  for that frame.
* **HGGD** draws random numbers inside ``predict`` (region jitter, sampling starts). A
  run seeds once and draws frame after frame, so only its first frame is drawn from
  the same numbers as an inferencer call, which seeds per call. That frame has to agree
  within ``1e-5`` (two runs of the same GPU kernels differ by about ``2e-7``); the
  later frames of a run are a different, equally valid, draw.
* **Arrays versus files.** Feeding the inferencer the arrays of a frame (depth,
  intrinsics, segmentation as the mask, the camera-to-table transform) has to give the
  very points the dataset reader gives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from grasppanda.apis import GraspInferencer
from grasppanda.cli.main import main
from grasppanda.config import Config
from grasppanda.evaluation.records import ResultRecord

#: method: (config, weights below the cache root, camera, bit-identical to the engine's dump)
METHODS = {
    "graspnet_baseline": (
        "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e",
        "runs/author_ckpt_smoke/baseline_rs.pth",
        "realsense",
        True,
    ),
    "scale_balanced_grasp": (
        "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e",
        "runs/author_ckpt_smoke/sbg_rs.pth",
        "realsense",
        True,
    ),
    "gsnet": (
        "gsnet/gsnet_sunet14_g1b-realsense_10e",
        "weights/converted/gsnet_local_enhanced_rs_e18.pth",
        "realsense",
        True,
    ),
    "economicgrasp": (
        "economicgrasp/economicgrasp_sunet14_g1b-kinect_10e",
        "weights/converted/economicgrasp_author_kinect.pth",
        "kinect",
        True,
    ),
    "hggd": (
        "hggd/hggd_r34w8_g1b-realsense_15e",
        "runs/author_ckpt_smoke/hggd_rs.pth",
        "realsense",
        False,
    ),
}
SAMPLE_FRAMES = tuple(range(0, 256, 32))  # the `sample` split: eight frames of scene 0


def weights_of(asset_cache: Path, method: str) -> Path:
    path = asset_cache / METHODS[method][1]
    if not path.is_file():
        pytest.skip(f"no converted weights for {method} at {path}")
    return path


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.parametrize("method", list(METHODS))
def test_a_dataset_frame_equals_what_the_engine_dumps(
    method, graspnet1b_root, asset_cache, cuda, tmp_path, capsys
):
    config, _, camera, exact = METHODS[method]
    weights = weights_of(asset_cache, method)
    run = tmp_path / "run"
    arguments = ["infer", config, "--weights", str(weights), "--output", str(run)]
    assert main([*arguments, "--opts", "data.test.dataset.split=sample"]) == 0
    assert "✓ 8 frames" in capsys.readouterr().out
    assert Config.load(config).select("data.test.num_workers") == 4  # the case that matters

    inferencer = GraspInferencer(config, weights)
    assert "raw" in inferencer.variants and "cd" in inferencer.variants
    frames = SAMPLE_FRAMES if exact else SAMPLE_FRAMES[:1]
    for frame in frames:
        name = f"scene_0000/{camera}/{frame:04d}"
        for variant in ("raw", "cd"):
            dumped = np.load(run / "predictions" / variant / f"{name}.npy")
            mine = inferencer(frame=f"graspnet1b:{name}", variant=variant).to_graspnet_array()
            assert mine.shape == dumped.shape and len(mine) > 0, (name, variant)
            if exact:
                assert np.array_equal(mine, dumped), (name, variant)
            else:
                assert np.abs(mine - dumped).max() < 1e-5, (name, variant)
        filtered = inferencer(frame=f"graspnet1b:{name}", collision_filter=True)
        again = filtered.to_graspnet_array()  # the `cd` chain, by another name
        assert again.shape == mine.shape
        assert np.array_equal(again, mine) if exact else np.abs(again - mine).max() < 1e-5
    # The same call twice gives the same grasps, also for the method that draws at random.
    first = inferencer(frame=f"graspnet1b:scene_0000/{camera}/0064").to_graspnet_array()
    second = inferencer(frame=f"graspnet1b:scene_0000/{camera}/0064").to_graspnet_array()
    assert first.shape == second.shape and np.abs(first - second).max() < 1e-5


def frame_arrays(root: Path, scene: int, camera: str, frame: int) -> dict:
    """What a user would hold for this frame: arrays, with the segmentation as a mask."""
    from scipy.io import loadmat

    directory = root / "scenes" / f"scene_{scene:04d}" / camera
    meta = loadmat(str(directory / "meta" / f"{frame:04d}.mat"))
    poses = np.load(directory / "camera_poses.npy")
    return {
        "depth": np.array(Image.open(directory / "depth" / f"{frame:04d}.png")),
        "rgb": np.array(Image.open(directory / "rgb" / f"{frame:04d}.png")),
        "mask": np.array(Image.open(directory / "label" / f"{frame:04d}.png")),
        "intrinsics": meta["intrinsic_matrix"],
        "depth_scale": float(meta["factor_depth"].reshape(-1)[0]),
        "table_from_camera": np.load(directory / "cam0_wrt_table.npy") @ poses[frame],
    }


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.parametrize("method", ["graspnet_baseline", "hggd"])
def test_arrays_of_a_frame_go_through_the_same_pipeline_as_the_dataset(
    method, graspnet1b_root, asset_cache, cuda
):
    config, _, camera, exact = METHODS[method]
    inferencer = GraspInferencer(config, weights_of(asset_cache, method))
    scene, frame = 100, 64
    reference = inferencer.observe(frame=f"graspnet1b:scene_{scene:04d}/{camera}/{frame:04d}")
    arrays = frame_arrays(graspnet1b_root, scene, camera, frame)
    if method == "hggd":  # no workspace in this pipeline: a mask would be ignored, with a warning
        del arrays["mask"], arrays["table_from_camera"]
    else:
        del arrays["rgb"]
    # The dataset draws the sample of a frame from (seed, scene, frame, camera index).
    generator = np.random.default_rng([0, scene, frame, 0])
    observation = inferencer.observe(**arrays, seed=generator)
    assert torch.equal(observation.points.coords, reference.points.coords)
    assert torch.equal(observation.intrinsics, reference.intrinsics)
    if method == "hggd":
        assert torch.equal(observation.rgb, reference.rgb)
        assert torch.equal(observation.depth, reference.depth)
        assert observation.full_points is None and reference.full_points is None
    else:
        assert torch.equal(observation.full_points.coords, reference.full_points.coords)
    mine = inferencer.predict(observation, collision_filter=True).to_graspnet_array()
    theirs = inferencer.predict(reference, collision_filter=True).to_graspnet_array()
    assert mine.shape == theirs.shape and len(mine) > 0
    assert np.array_equal(mine, theirs) if exact else np.abs(mine - theirs).max() < 1e-5


@pytest.mark.dataset
@pytest.mark.gpu
def test_stored_dumps_of_grasppanda_test_are_reproduced(graspnet1b_root, asset_cache, cuda):
    """Against a run directory that ``grasppanda test`` wrote earlier, if this machine has one."""
    run = asset_cache / "runs/inference_api/baseline_rs/test_seen_step64"
    if not (run / "predictions/raw").is_dir():
        pytest.skip(f"no stored run at {run}")
    inferencer = GraspInferencer(
        METHODS["graspnet_baseline"][0], weights_of(asset_cache, "graspnet_baseline")
    )
    for name in (
        "scene_0100/realsense/0000",
        "scene_0100/realsense/0128",
        "scene_0117/realsense/0192",
    ):
        for variant in ("raw", "cd"):
            dumped = np.load(run / "predictions" / variant / f"{name}.npy")
            mine = inferencer(frame=f"graspnet1b:{name}", variant=variant).to_graspnet_array()
            assert np.array_equal(mine, dumped), (name, variant)


@pytest.mark.dataset
@pytest.mark.gpu
def test_graspclutter6d_frames_go_the_same_way(graspclutter6d_root, asset_cache, cuda, tmp_path):
    """Nothing in the API is specific to GraspNet-1Billion: another reader, same guarantee."""
    from grasppanda.engine.evaluation import build_job, predict

    weights = weights_of(asset_cache, "gsnet")
    config = tmp_path / "gsnet_gc6d.yaml"
    root = Config.load("gsnet/gsnet_sunet14_g1b-realsense_10e").path.parent.parent
    config.write_text(
        f"_base_:\n  - {root}/_base_/models/gsnet.yaml\n"
        f"  - {root}/_base_/datasets/graspclutter6d_d435.yaml\n"
        f"  - {root}/_base_/default_runtime.yaml\n"
        "data:\n  test:\n    dataset: {keep_full_points: true}\n"
        "evaluation:\n  variants:\n    raw: []\n    cd: [{type: ModelFreeCollisionFilter}]\n"
    )
    inferencer = GraspInferencer(config, weights)
    job = build_job(inferencer.cfg, "test", with_evaluator=False)
    job.loader.sampler.indices = range(0, 4)  # the first four frames of the test split
    dump = predict(inferencer.model, job, tmp_path / "dump", device=inferencer.device)
    assert len(dump.keys) == 4 and dump.gt_assisted == ["workspace"]
    for key in dump.keys:
        assert key.dataset == "graspclutter6d" and key.camera == "realsense-d435"
        for variant in ("raw", "cd"):
            file = (
                tmp_path
                / "dump"
                / variant
                / f"scene_{key.scene:04d}/{key.camera}/{key.frame:04d}.npy"
            )
            mine = inferencer(frame=key, variant=variant).to_graspnet_array()
            assert np.array_equal(mine, np.load(file)), (key, variant)


# -------------------------------------------------------------------------- analyze


def test_analyze_reads_a_real_run_directory(asset_cache, tmp_path, capsys):
    """Records written by ``grasppanda test`` before the breakdown and the counts existed."""
    work = asset_cache / "runs/author_ckpt_smoke/hggd_rs"
    if not work.is_dir():
        pytest.skip(f"no run directory at {work}")
    plot = tmp_path / "cd.png"
    assert main(["analyze", str(work), "--variant", "cd", "--plot", str(plot)]) == 0
    printed = capsys.readouterr().out
    (file,) = work.glob("*/results/cd.json")
    record = ResultRecord.read(file)
    stored = json.loads(file.read_text())["metrics"]
    assert not record.partial and record.requested == 120 and len(record.per_scene) == 30
    for name, value in stored.items():
        assert f"{name} {value:.4f}" in printed
    assert "120 of 120 frames scored (100.0 %), 0 missing" in printed
    assert (
        "a SUBSET of split 'test_seen': 120 of its 7,680 frames (4 of 256 frames per scene, 30 of 30 scenes)"
        in printed
    )
    assert "ground-truth-assisted: none" in printed
    assert "post-processing ModelFreeCollisionFilter + GraspNMS" in printed
    line = next(row for row in printed.splitlines() if row.startswith("mean precision"))
    assert np.isclose(float(line.split()[-1]), stored["AP"], atol=5e-5)
    assert "the chain (ModelFreeCollisionFilter + GraspNMS) removed" in printed
    assert "This record has no breakdown by cause and object" in printed
    with Image.open(plot) as image:
        assert image.size[0] == 1100

    assert main(["analyze", "--compare", f"{work}:raw", f"{work}:cd"]) == 0
    compared = capsys.readouterr().out
    assert "30 paired scenes" in compared and "differs: post-processing" in compared
    means = {
        n: np.mean(
            [
                s["mean_precision"]
                for s in ResultRecord.read(file.with_name(f"{n}.json")).per_scene.values()
            ]
        )
        for n in ("raw", "cd")
    }
    assert f"mean difference {means['cd'] - means['raw']:+.4f}" in compared


def test_analyze_splits_the_ap_of_a_run_scored_by_this_code(asset_cache, capsys):
    run = asset_cache / "runs/inference_api/baseline_rs/test_seen_step64"
    if not (run / "results/cd.json").is_file():
        pytest.skip(f"no run directory at {run}")
    record = ResultRecord.read(run / "results/cd.json")
    if not record.breakdown:
        pytest.skip("the stored run predates the breakdown")
    assert main(["analyze", str(run), "--variant", "cd", "--objects", "0"]) == 0
    printed = capsys.readouterr().out
    earned = next(row for row in printed.splitlines() if row.startswith("earned"))
    assert np.isclose(float(earned.split()[-1]), record.metrics["AP"], atol=5e-5)
    total = next(row for row in printed.splitlines() if row.startswith("sum"))
    assert total.split()[1:] == ["1.0000"] * 7
    assert "after ModelFreeCollisionFilter" in printed and "By object (all of" in printed
    objects = record.breakdown["per_object"]
    gains = np.sum([np.mean(entry["gain"]) for entry in objects.values()]) / record.scored
    assert np.isclose(gains, record.metrics["AP"], atol=1e-9)  # the objects' shares are the AP
