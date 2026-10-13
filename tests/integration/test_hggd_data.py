"""HGGD's inputs from the GraspNet-1Billion reader, against what the reference loader made.

The reference values are the inputs recorded by ``tools/make_hggd_fixtures.py`` (see
``test_hggd_parity.py`` for how to regenerate them). Here the same frames are read by
:class:`~grasppanda.datasets.graspnet1b.GraspNet1B` with HGGD's data recipe
(``configs/_base_/datasets/graspnet1b_<camera>_hggd.yaml``) and compared:

* the network image built from ``observation.rgb`` / ``observation.depth`` must be the
  reference's (colour identical, depth within the rounding of a mean);
* the 25 600 recorded scene points must be points of the reader's unmasked cloud (the
  reference draws them at random; its draw is replayed through the recorded indices);
* no input is derived from ground truth, so nothing is ``gt_assisted``;
* predictions made from these inputs with the reference's random draws replayed score
  what the reference's own dump scores under the benchmark evaluator;
* with the per-view label asset of the ``sample`` split built
  (``grasppanda prepare graspnet1b --view-grasps --camera realsense --split sample
  --config _base_/datasets/graspnet1b_realsense_hggd.yaml``), the training reader feeds
  the detector and the authors' weights have the loss of a trained model on it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.config import Config
from grasppanda.datasets.assets import AssetError
from grasppanda.evaluation import save_prediction
from grasppanda.models.detectors.hggd import RegionDraws
from grasppanda.models.detectors.hggd.image import encode_image
from grasppanda.registry import DATASETS, DETECTORS, EVALUATORS, POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation, PointSet, SampleKey, collate

pytestmark = [pytest.mark.dataset, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
CAMERAS = ("realsense", "kinect")


@pytest.fixture(scope="module", params=CAMERAS)
def recorded(request, asset_cache, graspnet1b_root):
    """The camera, its fixture manifest, the reader and the recorded frames."""
    camera = request.param
    directory = asset_cache / "fixtures" / "hggd"
    file = directory / f"manifest_{camera}.json"
    if not file.is_file():
        pytest.skip(f"no HGGD reference fixtures for {camera} at {directory}")
    manifest = json.loads(file.read_text())
    node = Config.load(f"_base_/datasets/graspnet1b_{camera}_hggd.yaml").resolved()
    node = dict(node["data"]["test"]["dataset"], keep_full_points=True)
    dataset = DATASETS.build(node)
    index = {key: position for position, key in enumerate(dataset.keys)}
    frames = []
    for entry in manifest["inference"]:
        key = SampleKey("graspnet1b", entry["scene"], camera, entry["frame"])
        frames.append((key, dict(np.load(directory / entry["file"])), dataset[index[key]]))
    return camera, manifest, dataset, frames


def test_images_are_served_and_encode_to_the_reference_input(recorded):
    _, _, _, frames = recorded
    for key, record, batch in frames:
        observation = batch.observation
        assert observation.keys == [key]
        assert tuple(observation.rgb.shape) == (1, 3, 720, 1280)
        assert tuple(observation.depth.shape) == (1, 720, 1280)
        assert observation.rgb.dtype == observation.depth.dtype == torch.float32
        assert float(observation.rgb.min()) >= 0.0 and float(observation.rgb.max()) <= 1.0
        # the sensor depth, metres here and (W, H) millimetres in the reference
        millimetres = observation.depth[0].numpy().T * 1000.0
        assert np.abs(millimetres - record["depth"]).max() < 1e-3
        image = encode_image(observation.rgb, observation.depth, (640, 360))[0].numpy()
        assert image.shape == record["image"].shape == (4, 640, 360)
        assert np.array_equal(image[1:], record["image"][1:])  # colour: bit for bit
        assert np.abs(image[0] - record["image"][0]).max() < 5e-7  # depth: a float32 mean apart
        other = encode_image(observation.rgb, observation.depth, (640, 360), resize="torch_bicubic")
        assert np.abs(other[0, 1:].numpy() - record["image"][1:]).max() > 1 / 255  # not the same


def test_the_recorded_cloud_is_a_sample_of_the_unmasked_cloud(recorded):
    _, manifest, _, frames = recorded
    for _, record, batch in frames:
        observation = batch.observation
        assert observation.gt_assisted() == [] and observation.provenance == {}
        assert len(observation.points) == 25600
        valid = record["depth"] > 0  # (W, H): the reference enumerates pixels column by column
        assert len(observation.full_points) == int(valid.sum())  # every pixel with depth
        u, v = np.nonzero(valid)
        picked = record["sample_index"]
        pixel = v[picked] * 1280 + u[picked]  # the reader enumerates them row by row
        order = np.sort(v * 1280 + u)
        rows = np.searchsorted(order, pixel)
        mine = observation.full_points.coords.numpy()[rows]
        # The reference back-projects with constants rounded to two decimals, in float32.
        assert np.abs(mine - record["points"]).max() < 2e-5
        calibration = observation.intrinsics[0].numpy()
        assert np.abs(calibration - np.asarray(manifest["intrinsics"])).max() < 0.01
        # the sampled cloud of the reader is a subset of the same cloud
        own = observation.points.coords.numpy()
        full = observation.full_points.coords.numpy()
        assert len(np.unique(own, axis=0)) == len(own) <= len(full)


@pytest.mark.gpu
@pytest.mark.slow
def test_replayed_predictions_score_what_the_reference_dump_scores(recorded, cuda, tmp_path):
    """Dataset → model → post-processors → evaluator, against the reference's own dump."""
    camera, manifest, _, frames = recorded
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the checkpoint the fixtures were recorded with is missing: {checkpoint}")
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    state, _ = converter.convert_file(checkpoint, camera=camera, source_format="hggd")
    config = Config.load(f"hggd/hggd_r34w8_g1b-{camera}_15e").resolved()
    model = DETECTORS.build(config["model"])
    model.load_state_dict(state, strict=True)
    model.to(cuda).eval()
    chains = {
        name: [POSTPROCESSORS.build(node) for node in chain]
        for name, chain in config["evaluation"]["variants"].items()
    }
    assert set(chains) == {"raw", "cd"}
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    keys = []
    for key, record, batch in frames:
        keys.append(key)
        device = cuda
        # The reader's images and calibration; the reference's sampled cloud and draws.
        observation = Observation(
            keys=[key],
            intrinsics=batch.observation.intrinsics.to(device),
            depth=batch.observation.depth.to(device),
            rgb=batch.observation.rgb.to(device),
            points=PointSet(torch.as_tensor(record["points"], device=device)),
        )
        draws = RegionDraws(
            resample=list(torch.as_tensor(record["resample_0"], device=device)),
            jitter=torch.as_tensor(record["jitter_0"], device=device),
            start=torch.as_tensor(record["fps_start_0"], device=device),
        )
        with torch.no_grad():
            (group,) = model.decode(model.run_stages(observation, draws=[draws]), observation)
        assert group.key == key and len(group) == len(record["decoded_score"])
        for name, chain in chains.items():
            processed = group
            for step in chain:
                processed = step(processed, observation)
            save_prediction(tmp_path / "ours" / name, processed)
        for name, dump in (("raw", "final_raw"), ("cd", "final_cd")):
            theirs = GraspGroup.from_graspnet_array(record[dump], key=key)
            save_prediction(tmp_path / "reference" / name, theirs)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
    evaluator = EVALUATORS.build(config["evaluation"]["evaluator"])
    scores = {}
    for who in ("ours", "reference"):
        for name in chains:
            result = evaluator.evaluate(tmp_path / who / name, keys)
            assert result.complete
            scores[who, name] = {key: result.precision[key].mean() * 100 for key in keys}
    print(f"\nHGGD {camera}: AP on {len(keys)} recorded frames (partial coverage, not a headline)")
    for name in chains:
        ours = np.mean(list(scores["ours", name].values()))
        theirs = np.mean(list(scores["reference", name].values()))
        print(f"  {name:4s} GraspPanda {ours:6.2f}   reference dump {theirs:6.2f}")
        for key in keys:
            # Same grasps up to float32 and a region sampled in another order now and then.
            assert abs(scores["ours", name][key] - scores["reference", name][key]) < 1.5, key
        assert abs(ours - theirs) < 0.5


@pytest.mark.gpu
def test_training_reader_feeds_the_detector_through_the_label_asset(asset_cache, cuda):
    """Dataset config → images, cloud and per-view labels → ``HGGD.loss``."""
    manifest_file = asset_cache / "fixtures" / "hggd" / "manifest_realsense.json"
    if not manifest_file.is_file():
        pytest.skip("no HGGD fixtures: the authors' checkpoint is located through their manifest")
    checkpoint = Path(json.loads(manifest_file.read_text())["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the authors' checkpoint is missing: {checkpoint}")
    config = Config.load("hggd/hggd_r34w8_g1b-realsense_15e").resolved()
    node = dict(config["data"]["train"]["dataset"], split="sample", seed=0)
    try:
        dataset = DATASETS.build(node)
    except AssetError as error:
        pytest.skip(f"the view-grasp asset of the sample split is not built: {error}")
    assert [record["kind"] for record in dataset.asset_records()] == ["view_grasps"]
    batch = collate([dataset[index] for index in range(4)])
    assert batch.supervision.objects == [[], [], [], []]  # no dense label grid is read
    labels = batch.supervision.extras["view_grasps"]
    assert len(labels) == 4 and all(len(entry["theta"]) > 20 for entry in labels)
    assert all(
        0.2 < float(entry["depth"].min()) < float(entry["depth"].max()) < 1.5 for entry in labels
    )
    assert batch.observation.gt_assisted() == []
    assert tuple(batch.observation.rgb.shape) == (4, 3, 720, 1280)

    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    state, _ = converter.convert_file(checkpoint, camera="realsense", source_format="hggd")
    model = DETECTORS.build(config["model"])
    model.load_state_dict(state, strict=True)
    model.to(cuda).train()
    model.on_train_epoch(10, 15)  # past the anchor-shifting epochs
    torch.manual_seed(0)
    batch = batch.to(cuda)
    losses = {
        name: float(value)
        for name, value in model.loss(batch.observation, batch.supervision).items()
    }
    print("\nHGGD, authors' RealSense weights on four `sample` frames:", losses)
    # A trained model: the reference's own training batch has 1.25 / 1.32 / 0.15 / 0.51 / 0.15.
    assert losses["loss/heatmap"] < 3 and losses["loss/angle"] < 3 and losses["loss/offsets"] < 0.5
    assert losses["loss/anchor"] < 1.5 and losses["loss/anchor_offset"] < 0.5
    assert losses["metric/regions"] > 100 and losses["metric/supervised_regions"] > 0.5
