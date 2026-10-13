"""GraspClutter6D as every ported method family reads it, on the real dataset.

Skipped without the dataset. Tests that read a derived asset are skipped, with the
command that builds it, when the asset is absent; they need it for the training scene of
the ``sample`` split only (scene 5, which holds three copies of several objects)::

    grasppanda prepare graspclutter6d --labels --scenes 5
    grasppanda prepare graspclutter6d --graspness --camera realsense-d435 --scenes 5
    grasppanda prepare graspclutter6d --compact-labels --scenes 5 \\
        --config economicgrasp/economicgrasp_sunet14_gc6d-d435_10e
    grasppanda prepare graspclutter6d --tolerance --scenes 5 \\
        --config _base_/datasets/graspclutter6d_d435_tolerance.yaml
    grasppanda prepare graspclutter6d --view-grasps --camera realsense-d435 --scenes 5 \\
        --config _base_/datasets/graspclutter6d_d435_hggd.yaml

Stated tolerances:

* a per-view grasp label of an object lies on that object: its centre pixel falls on the
  full (amodal) mask of an instance of the object for at least 90 % of the labels, and
  its depth is within -0.1 .. +0.3 m of the sensor depth at that pixel for at least 95 %.
  (Measured when the test was written, 271 labels of four views: 96.7 % on the mask, the
  others in a hole or between two parts of their object; label depth minus sensor depth
  +2 mm in the median, -41 / +102 mm at the 1st / 99th percentile; a centre lies inside
  its object, so it can be behind the surface or in front of what is seen through a
  gap.) This bounds the pose source, the pose units and the per-frame calibration of the
  label generator;
* point-wise graspness is a min-max normalised map: 0 and 1 are attained on every frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from grasppanda.config import Config
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspclutter6d import IMAGE_SIZES, GraspClutter6D
from grasppanda.datasets.scene import read_image
from grasppanda.registry import DATASETS, DETECTORS
from grasppanda.structures import COMPACT_LABELS, SampleKey

pytestmark = pytest.mark.dataset

GSNET = "gsnet/gsnet_sunet14_gc6d-d435_10e"
ECONOMIC = "economicgrasp/economicgrasp_sunet14_gc6d-d435_10e"
TOLERANCE = "_base_/datasets/graspclutter6d_d435_tolerance.yaml"
HGGD = "_base_/datasets/graspclutter6d_d435_hggd.yaml"
#: views of the sample scene that make a training batch
VIEWS = (0, 5, 9, 12)


def node_of(config: str, loader: str = "train", **overrides: Any) -> dict[str, Any]:
    return {**Config.load(config).resolved()["data"][loader]["dataset"], **overrides}


def sample_dataset(config: str, **overrides: Any) -> GraspClutter6D:
    """The training reader of a config on the one scene of the ``sample`` split."""
    try:
        return DATASETS.build(node_of(config, split="sample", seed=0, **overrides))
    except AssetError as error:
        lines = str(error).splitlines()
        pytest.skip(f"derived assets are not built: {lines[0]} {lines[-1].strip()}")


# ----------------------------------------------------------------------- tree, splits


def test_subsets_are_inside_the_official_splits(graspclutter6d_root):
    manifest = GraspClutter6D.load_manifest(graspclutter6d_root)
    train, test = set(manifest.split("train").scenes), set(manifest.split("test").scenes)
    mini, validation = manifest.split("test_mini").scenes, manifest.split("val_dev").scenes
    development = manifest.split("train_dev").scenes
    assert (len(mini), len(validation), len(development)) == (24, 41, 372)
    assert set(mini) < test and set(validation) < train
    assert set(development) | set(validation) == train and not set(development) & set(validation)
    assert len(manifest.keys("test_mini", "realsense-d435")) == 312
    assert mini == tuple(sorted(test)[0::10]) and validation == tuple(sorted(train)[5::10])


def test_test_objects_never_occur_in_training_scenes(graspclutter6d_root):
    """The benchmark is cross-object: 123 training and 68 test models, disjoint."""
    dataset = GraspClutter6D("realsense-d435", "all", [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}], [], load_labels=False)  # fmt: skip
    train = dataset.object_ids(dataset.manifest.split("train").scenes)
    test = dataset.object_ids(dataset.manifest.split("test").scenes)
    assert (len(train), len(test)) == (123, 68) and not train & test
    assert test == set(range(1, 69)) and min(train) == 69 and max(train) == 198
    covered = dataset.object_ids(dataset.manifest.split("test_mini").scenes)
    assert len(covered) == 66 and covered < test


def test_tree_holds_what_reader_and_evaluator_need(graspclutter6d_root):
    assert GraspClutter6D.check_tree(graspclutter6d_root, ["realsense-d435"]) == []


# ----------------------------------------------------------------------------- images


@pytest.mark.parametrize("camera", list(IMAGE_SIZES))
def test_colour_and_depth_images_of_every_camera(graspclutter6d_root, camera):
    dataset = GraspClutter6D(
        camera,
        "test_mini",
        [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}],
        [{"type": "RandomSample", "num_points": 25600}],
        load_labels=False,
        images=["rgb", "depth"],
        keep_full_points=False,
        seed=0,
    )
    assert dataset.image_size == IMAGE_SIZES[camera]
    observation = dataset[7].observation
    height, width = IMAGE_SIZES[camera]
    assert observation.rgb.shape == (1, 3, height, width) and observation.rgb.dtype == torch.float32
    assert float(observation.rgb.min()) >= 0.0 and float(observation.rgb.max()) <= 1.0
    assert float(observation.rgb.std()) > 0.05  # a picture, not a constant
    depth = observation.depth[0]
    assert depth.shape == (height, width)
    measured = depth[depth > 0]
    assert 0.3 < float(measured.median()) < 2.0  # metres, whatever the raw unit
    points = observation.points.coords
    assert points.shape == (25600, 3) and observation.gt_assisted() == []
    # the cloud is the back-projected depth image: every point's z is a depth value
    key = dataset.keys[7]
    frame = dataset.frame(key)
    assert np.isin(points[:, 2].numpy(), frame.cloud[..., 2].astype(np.float32)).all()


def test_hggd_test_loader_serves_images_three_times_the_network_image(graspclutter6d_root):
    dataset = DATASETS.build(node_of(HGGD, "test", split="test_mini"))
    batch = dataset.collate([dataset[0], dataset[13]])
    assert batch.observation.rgb.shape == (2, 3, 1080, 1920)
    assert batch.observation.depth.shape == (2, 1080, 1920)
    assert batch.observation.points.batch_offsets.tolist() == [0, 25600, 51200]
    assert batch.supervision is None and batch.observation.gt_assisted() == []
    model = Config.load("hggd/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e").resolved()["model"]
    width, height = model["image"]["size"]
    assert (1920 / width, 1080 / height) == (3.0, 3.0)
    focal = batch.observation.intrinsics[:, 0, 0] / 3
    assert ((focal > 460) & (focal < 475)).all()  # GraspNet-1Billion RealSense: 463.6


@pytest.mark.parametrize("camera", list(IMAGE_SIZES))
def test_shipped_config_of_every_camera_serves_test_frames(graspclutter6d_root, camera):
    node = node_of(GraspClutter6D.default_config(camera), "test")
    assert node["camera"] == camera
    dataset = DATASETS.build(node)
    assert len(dataset) == 235 * 13
    item = dataset[26]
    assert len(item.observation.points) == 15000 and item.supervision is None
    assert item.observation.gt_assisted() == ["workspace"]
    assert len(item.observation.full_points) > 100_000


# ----------------------------------------------------------------------- label assets


def test_graspness_maps_are_normalised_per_frame_and_keyed_by_pixel(graspclutter6d_root):
    dataset = sample_dataset(GSNET)
    assert [record["kind"] for record in dataset.asset_records()] == ["grasp_labels", "graspness"]
    for view in VIEWS[:2]:
        key = SampleKey("graspclutter6d", 5, "realsense-d435", view)
        sample = dataset.sample(key)
        values = sample.point_labels["graspness"]
        assert len(values) == len(sample) > 100_000
        assert values.min() == 0.0 and values.max() == 1.0 and np.isfinite(values).all()
        stored = dataset.graspness.read_map(key)
        assert stored.shape == IMAGE_SIZES["realsense-d435"]
        assert np.isfinite(stored).sum() == len(sample)  # NaN outside the workspace crop


def test_compact_labels_are_one_entry_per_instance(graspclutter6d_root):
    dataset = sample_dataset(ECONOMIC)
    item = dataset[3]
    (objects,) = item.supervision.extras[COMPACT_LABELS]
    object_ids = dataset.scene_object_ids(5)
    assert len(object_ids) == 16 and len(set(object_ids)) == 6  # three copies of most models
    assert [labels.object_id for labels in objects] == object_ids
    assert item.supervision.objects == [[]] and len(item.observation.points) == 20000
    kept = sum(len(labels.points) for labels in objects)
    assert (
        0 < kept < sum(dataset.label_store.object_arrays(i)[0].shape[0] for i in object_ids)
    )  # only points with a collision-free grasp of friction <= 0.4 survive
    # widths of the kept grasps (score > 0; other cells hold no grasp), in millimetres
    widths = torch.cat([labels.width[labels.score > 0] for labels in objects])
    assert len(widths) > 1000 and int(widths.max()) <= 140  # the gripper of this benchmark
    assert int(widths.max()) > 100  # ... which opens wider than GraspNet-1Billion's


def test_tolerance_labels_reach_the_objects(graspclutter6d_root):
    dataset = sample_dataset(TOLERANCE)
    (objects,) = dataset[0].supervision.objects
    assert 10 <= len(objects) <= 16
    largest = 0.0
    for labels in objects:
        # the shipped recipe leaves the grids in the store; `dense()` reads them all
        assert labels.deferred is not None and "tolerance" in labels.shapes()
        grids = labels.dense()
        assert grids.tolerance is not None and grids.tolerance.shape == grids.scores.shape
        assert float(grids.tolerance.max()) <= 0.05 + 1e-9
        largest = max(largest, float(grids.tolerance.max()))
    assert largest > 0


def test_view_grasps_lie_on_their_objects_in_the_image(graspclutter6d_root):
    dataset = sample_dataset(HGGD)
    scene_dir = dataset.scene_dir(5)
    object_ids = dataset.scene_object_ids(5)
    labels_seen = on_mask = behind_surface = 0
    offsets: list[float] = []
    for view in VIEWS:
        key = SampleKey("graspclutter6d", 5, "realsense-d435", view)
        item = dataset[dataset.keys.index(key)]
        (labels,) = item.supervision.extras["view_grasps"]
        count = len(labels["score"])
        assert count > 50 and item.supervision.objects == [[]]
        assert set(labels["object_id"].tolist()) <= set(object_ids)
        assert set(np.round(labels["score"].numpy(), 4)) <= {1.0, 0.9}
        columns = labels["centers"][:, 0].round().long().numpy()
        rows = labels["centers"][:, 1].round().long().numpy()
        image = dataset.image_id(key)
        amodal: dict[int, np.ndarray] = {}
        for index, object_id in enumerate(object_ids):
            mask = read_image(scene_dir / "mask" / f"{image:06d}_{index:06d}.png") > 0
            amodal[object_id] = amodal.get(object_id, False) | mask
        hit = np.array(
            [amodal[int(i)][r, c] for i, r, c in zip(labels["object_id"], rows, columns)]
        )
        depth = item.observation.depth[0].numpy()[rows, columns]
        measured = depth > 0
        ahead = labels["depth"].numpy()[measured] - depth[measured]
        labels_seen += count
        on_mask += int(hit.sum())
        behind_surface += int(((ahead > -0.1) & (ahead < 0.3)).sum()) + int((~measured).sum())
        offsets.extend(ahead.tolist())
    print(
        f"\nview grasps of scene 5: {labels_seen} labels, {on_mask / labels_seen:.4f} on the "
        f"amodal mask of their object, {behind_surface / labels_seen:.4f} at a plausible depth; "
        f"label depth minus sensor depth, 1st / 50th / 99th percentile: "
        f"{np.percentile(offsets, [1, 50, 99]).round(3).tolist()} m"
    )
    assert on_mask / labels_seen > 0.9 and behind_surface / labels_seen > 0.95
    assert abs(float(np.median(offsets))) < 0.01


# ------------------------------------------------------------------- training batches


def train_batch(config: str, cuda: torch.device):
    resolved = Config.load(config, command="train").resolved()
    dataset = sample_dataset(config)
    items = [dataset[view] for view in VIEWS]
    torch.manual_seed(0)
    model = DETECTORS.build(resolved["model"]).to(cuda).train()
    return model, dataset.collate(items).to(cuda), items


def check_losses(model: torch.nn.Module, batch, expected: set[str]) -> dict[str, float]:
    losses = model.loss(batch.observation, batch.supervision)
    values = {name: float(value) for name, value in losses.items()}
    print("\nlosses of a GraspClutter6D batch:", {k: round(v, 4) for k, v in values.items()})
    assert expected <= set(values) and all(np.isfinite(v) for v in values.values())
    total = sum(value for name, value in losses.items() if name.startswith("loss/"))
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert float(sum((g**2).sum() for g in gradients)) > 0
    return values


@pytest.mark.gpu
@pytest.mark.slow
def test_a_real_batch_trains_gsnet(graspclutter6d_root, cuda):
    model, batch, items = train_batch(GSNET, cuda)
    assert model.gripper.max_width == 0.14
    for item in items:
        assert len(item.observation.points) == 15000
        assert set(item.supervision.point_labels) == {"objectness", "instance", "graspness"}
        (objects,) = item.supervision.objects
        assert 10 <= len(objects) <= 16  # instances with at least 50 visible points
        marks = set(item.supervision.point_labels["instance"].unique().tolist()) - {0}
        assert len(marks) > len({labels.object_id for labels in objects})  # instance-keyed
    values = check_losses(
        model, batch, {"loss/objectness", "loss/graspness", "loss/view", "loss/score", "loss/width"}
    )
    assert values["metric/supervised_seeds"] > 0.5


@pytest.mark.gpu
@pytest.mark.slow
def test_a_real_batch_trains_economicgrasp(graspclutter6d_root, cuda):
    model, batch, items = train_batch(ECONOMIC, cuda)
    assert model.gripper.max_width == 0.14
    assert all(len(item.observation.points) == 20000 for item in items)
    values = check_losses(
        model,
        batch,
        {"loss/objectness", "loss/graspness", "loss/view", "loss/angle", "loss/depth",
         "loss/score", "loss/width"},
    )  # fmt: skip
    # seeds within 5 mm of a kept label point: few in a scene this cluttered, but some
    assert 0 < values["metric/supervised_seeds"] < 0.5
