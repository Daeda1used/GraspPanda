"""The GraspClutter6D reader on a tiny synthetic BOP-style tree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from grasppanda.cli.main import main
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspclutter6d import GraspClutter6D
from grasppanda.datasets.graspness import generate_graspness
from grasppanda.datasets.labels import LabelGrid
from grasppanda.registry import DATASETS
from grasppanda.structures import ContractError, Provenance, SampleKey
from grasppanda.transforms.frame import transform_points
from grasppanda.transforms.points import FilterObjects, SampleGraspLabels

from .eval_synthetic import write_ply

GRID = LabelGrid(num_view=5, num_angle=3, num_depth=2)
VIEWS = 2
SIZES = {
    "realsense-d415": (20, 28),
    "realsense-d435": (20, 28),
    "azure-kinect": (40, 56),
    "zivid": (24, 28),
}
#: instances per scene; scene 5 holds two copies of object 7
SCENES = {5: [7, 7, 3], 9: [3, 12], 2: [12, 7, 7]}
POINTS = {3: 6, 7: 8, 12: 10}
#: raw depth units per millimetre is 1 / depth_scale
DEPTH_SCALE = {"realsense-d415": 1, "realsense-d435": 1, "azure-kinect": 0.1, "zivid": 0.1}


@DATASETS.register()
class TinyClutter(GraspClutter6D):
    name = "tinyclutter"
    label_grid = GRID
    image_sizes = SIZES
    views_per_scene = VIEWS


def make_tree(root: Path, seed: int = 0) -> dict:
    """Write a synthetic dataset in the GraspClutter6D layout and return its content."""
    rng = np.random.default_rng(seed)
    truth: dict = {"labels": {}, "collision": {}, "frames": {}}
    (root / "split_info").mkdir(parents=True)
    (root / "split_info/grasp_train_scene_ids.json").write_text(json.dumps(["000005", "000009"]))
    (root / "split_info/grasp_test_scene_ids.json").write_text(json.dumps(["000002"]))
    (root / "split_info/obj_ids_per_scene.json").write_text(
        json.dumps({str(scene): sorted(set(ids)) for scene, ids in SCENES.items()})
    )
    for folder in ("grasp_label", "models_m", "dex_models"):
        (root / folder).mkdir()
    for object_id, count in POINTS.items():
        points = rng.uniform(-0.03, 0.03, (count, 3)).astype(np.float32)
        # the model: the label points and a little more of the same surface
        write_ply(
            root / "models_m" / f"obj_{object_id:06d}.ply",
            np.concatenate([points, 1.01 * points]).astype(np.float64),
        )
        (root / "dex_models" / f"obj_{object_id:06d}.pkl").write_bytes(b"")  # presence only
        scores = rng.choice(np.array([-1, 0.2, 0.4, 0.8, 1.0], np.float32), (count, *GRID.shape))
        offsets = rng.uniform(0, 0.1, (count, *GRID.shape, 3)).astype(np.float32)
        truth["labels"][object_id] = (points, scores, offsets[..., 2])
        np.savez_compressed(
            root / "grasp_label" / f"obj_{object_id:06d}_labels.npz",
            points=points,
            offsets=offsets,
            scores=scores,
            collision=np.zeros((count, *GRID.shape), dtype=bool),
        )
    (root / "collision_label").mkdir()
    cameras = list(SIZES)
    for scene, object_ids in SCENES.items():
        collisions = [rng.random((POINTS[i], *GRID.shape)) < 0.3 for i in object_ids]
        truth["collision"][scene] = collisions
        np.savez(
            root / "collision_label" / f"{scene:06d}.npz",
            **{f"arr_{k}": array for k, array in enumerate(collisions)},
        )
        directory = root / "scenes" / f"{scene:06d}"
        for sub in ("depth", "label", "mask_visib", "rgb"):
            (directory / sub).mkdir(parents=True)
        scene_camera, scene_gt = {}, {}
        for view in range(VIEWS):
            for offset, camera in enumerate(cameras, start=1):
                image = 4 * view + offset
                height, width = SIZES[camera]
                focal = 30.0 * width / 28 + view  # intrinsics change from view to view
                intrinsics = [focal, 0, width / 2, 0, focal + 1, height / 2, 0, 0, 1]
                millimetres = np.full((height, width), 800.0)
                label = np.zeros((height, width), dtype=np.uint8)
                instances, masks = [], []
                for k, object_id in enumerate(object_ids):
                    top, left = 2 + 5 * k * height // 20, 3 + 7 * k * width // 28
                    box = (slice(top, top + height // 4), slice(left, left + width // 4 + 1))
                    millimetres[box] = 700.0 - 30 * k
                    label[box] = object_id
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[box] = 255
                    masks.append(mask)
                    z = (700.0 - 30 * k) / 1000
                    u = (box[1].start + box[1].stop - 1) / 2
                    v = (box[0].start + box[0].stop - 1) / 2
                    position = [
                        (u - width / 2) * z / focal * 1000,
                        (v - height / 2) * z / (focal + 1) * 1000,
                        z * 1000,
                    ]
                    angle = 0.2 * (k + 1) + 0.05 * view
                    rotation = [
                        np.cos(angle),
                        -np.sin(angle),
                        0,
                        np.sin(angle),
                        np.cos(angle),
                        0,
                        0,
                        0,
                        1,
                    ]
                    instances.append(
                        {"cam_R_m2c": rotation, "cam_t_m2c": position, "obj_id": object_id}
                    )
                # the first two instances overlap by one column in their visibility masks
                masks[1][:, masks[0].nonzero()[1].max()] |= masks[0][:, masks[0].nonzero()[1].max()]
                millimetres[0, 0] = 0
                raw = np.round(millimetres / DEPTH_SCALE[camera]).astype(np.uint16)
                Image.fromarray(raw).save(directory / "depth" / f"{image:06d}.png")
                colour = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                Image.fromarray(colour).save(directory / "rgb" / f"{image:06d}.png")
                Image.fromarray(np.repeat(label[..., None], 3, axis=2)).save(
                    directory / "label" / f"{image:06d}.png"
                )
                for k, mask in enumerate(masks):
                    Image.fromarray(mask).save(
                        directory / "mask_visib" / f"{image:06d}_{k:06d}.png"
                    )
                # the camera pose in the world: looking down, shifted with the view
                pose_rotation = [0, 1, 0, 1, 0, 0, 0, 0, -1]
                pose_translation = [100.0 * view, -50.0, 900.0]
                scene_camera[str(image)] = {
                    "cam_K": intrinsics,
                    "depth_scale": DEPTH_SCALE[camera],
                    "cam_R_w2c": pose_rotation,
                    "cam_t_w2c": pose_translation,
                }
                scene_gt[str(image)] = instances
                truth["frames"][scene, camera, view] = {
                    "millimetres": millimetres,
                    "colour": colour,
                    "label": label,
                    "masks": masks,
                    "intrinsics": np.array(intrinsics).reshape(3, 3),
                    "instances": instances,
                    "image": image,
                    "world_from_camera": (np.array(pose_rotation).reshape(3, 3), pose_translation),
                }
        (directory / "scene_camera.json").write_text(json.dumps(scene_camera))
        (directory / "scene_gt.json").write_text(json.dumps(scene_gt))
    return truth


PREPROCESS = [{"type": "DepthToCloud"}, {"type": "SegmentationRectWorkspace", "margin": 0.1}]
PLAIN = [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}]


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    truth = make_tree(tmp_path / "data")
    truth["root"] = tmp_path / "data"
    return truth


def build(tree: dict, **overrides) -> TinyClutter:
    arguments = {
        "camera": "realsense-d435",
        "split": "train",
        "preprocess": PREPROCESS,
        "transforms": [],
        "root": str(tree["root"]),
        "seed": 0,
    }
    arguments.update(overrides)
    if arguments.get("load_labels", True):
        probe = TinyClutter(**{**arguments, "check_assets": False})
        scenes = probe.manifest.all_scenes()
        probe.label_store.build(probe.object_ids(scenes), scenes)
    return TinyClutter(**arguments)


def test_splits_come_from_the_dataset_and_cameras_from_image_ids(tree):
    dataset = build(tree, load_labels=False)
    assert dataset.manifest.split("train").scenes == (5, 9)
    assert dataset.manifest.split("test").scenes == (2,)
    assert dataset.manifest.split("all").scenes == (2, 5, 9)
    assert dataset.manifest.split("sample").scenes == (5,)
    assert len(dataset) == 2 * VIEWS and dataset.keys[1] == SampleKey(
        "tinyclutter", 5, "realsense-d435", 1
    )
    for offset, camera in enumerate(TinyClutter.cameras, start=1):
        reader = build(tree, load_labels=False, camera=camera)
        assert [reader.image_id(key) for key in reader.keys[:VIEWS]] == [offset, 4 + offset]
    real = GraspClutter6D.load_manifest.__func__  # the real class: 13 views per scene
    assert len(real(GraspClutter6D, tree["root"]).frames) == 13


@pytest.mark.parametrize("camera", list(SIZES))
def test_intrinsics_depth_scale_and_image_size_are_per_image(tree, camera):
    dataset = build(tree, load_labels=False, camera=camera, preprocess=PLAIN)
    for key in dataset.keys[:VIEWS]:
        truth = tree["frames"][key.scene, camera, key.frame]
        frame = dataset.frame(key)
        assert frame.depth.shape == SIZES[camera]
        assert np.array_equal(frame.intrinsics, truth["intrinsics"])
        z = truth["millimetres"] / 1000.0
        assert np.allclose(frame.cloud[..., 2], z, atol=1e-12)  # metres, whatever the raw unit
        columns = np.arange(SIZES[camera][1])
        expected_x = (columns - truth["intrinsics"][0, 2]) * z / truth["intrinsics"][0, 0]
        assert np.allclose(frame.cloud[..., 0], expected_x, atol=1e-12)
        assert frame.valid.sum() == z.size - 1
    first, second = (dataset.frame(key).intrinsics[0, 0] for key in dataset.keys[:2])
    assert first != second


def test_a_depth_image_of_the_wrong_size_is_rejected(tree):
    directory = tree["root"] / "scenes/000005/depth"
    (directory / "000002.png").write_bytes((directory / "000003.png").read_bytes())
    with pytest.raises(
        ContractError, match=r"is \(40, 56\), expected \(20, 28\) for realsense-d435"
    ):
        build(tree, load_labels=False).frame(SampleKey("tinyclutter", 5, "realsense-d435", 0))


def test_poses_are_in_metres_and_the_world_frame_is_the_camera_pose(tree):
    dataset = build(tree, load_labels=False, preprocess=PLAIN)
    key = dataset.keys[1]
    truth = tree["frames"][5, "realsense-d435", 1]
    poses = dataset.object_poses(key)
    assert [object_id for object_id, _ in poses] == SCENES[5]
    for (_, pose), instance in zip(poses, truth["instances"]):
        assert pose.dtype == np.float32 and pose.shape == (4, 4)
        assert np.allclose(pose[:3, 3], np.array(instance["cam_t_m2c"]) / 1000, atol=1e-7)
        assert np.allclose(pose[:3, :3], np.array(instance["cam_R_m2c"]).reshape(3, 3), atol=1e-7)
    rotation, translation = truth["world_from_camera"]
    frame = dataset.frame(key)
    point = np.array([[0.1, -0.2, 0.7]])
    expected = point @ rotation.T + np.array(translation) / 1000
    assert np.allclose(transform_points(point, frame.table_from_camera), expected)
    with pytest.raises(ValueError, match="pose source must be one of"):
        dataset.object_poses(key, "meta")


def test_labels_are_keyed_by_instance_not_by_object(tree):
    dataset = build(tree)
    key = SampleKey("tinyclutter", 5, "realsense-d435", 0)
    truth = tree["frames"][5, "realsense-d435", 0]
    sample = dataset.sample(key)
    assert [a.object_id for a in sample.objects] == [7, 7, 3]
    assert [a.instance_id for a in sample.objects] == [1, 2, 3]
    rows, columns = np.divmod(sample.pixels, SIZES["realsense-d435"][1])
    instance = sample.point_labels["instance"]
    # the semantic label cannot separate the two copies of object 7; the instance label does
    assert set(truth["label"][rows, columns][np.isin(instance, [1, 2])]) == {7}
    overlap = (truth["masks"][0] > 0) & (truth["masks"][1] > 0)
    assert overlap.any() and (instance[overlap[rows, columns]] == 1).all()  # first instance wins
    for index in (1, 2):
        own = (truth["masks"][index] > 0) & ~overlap
        assert (instance[own[rows, columns]] == index + 1).all()
    assert np.array_equal(sample.point_labels["objectness"], (truth["label"][rows, columns] > 0))

    step = SampleGraspLabels(1.0, 0, "attach")
    picked = [np.arange(annotation.source.num_points) for annotation in sample.objects]
    step.apply(sample, picked)
    first, second, _ = sample.objects
    assert np.array_equal(first.scores, second.scores)  # the same object model ...
    assert np.array_equal(first.scores, tree["labels"][7][1])
    assert not np.array_equal(first.collision, second.collision)  # ... two placements
    for index, annotation in enumerate(sample.objects):
        assert np.array_equal(annotation.collision, tree["collision"][5][index])
        assert np.array_equal(annotation.widths, tree["labels"][annotation.object_id][2])

    counts = [int((instance == mark).sum()) for mark in (1, 2, 3)]
    fresh = FilterObjects(max(counts))(dataset.sample(key))
    assert [a.instance_id for a in fresh.objects] == [
        m for m, c in zip((1, 2, 3), counts) if c == max(counts)
    ]


def test_rect_workspace_uses_ground_truth_and_says_so(tree):
    dataset = build(tree, load_labels=False)
    key = dataset.keys[0]
    truth = tree["frames"][5, "realsense-d435", 0]
    frame = dataset.frame(key)
    height, width = SIZES["realsense-d435"]
    rows, columns = np.nonzero(truth["label"])
    top, bottom = (
        max(0, rows.min() - int(height * 0.1)),
        min(height, rows.max() + 1 + int(height * 0.1)),
    )
    left, right = (
        max(0, columns.min() - int(width * 0.1)),
        min(width, columns.max() + 1 + int(width * 0.1)),
    )
    expected = np.zeros((height, width), dtype=bool)
    expected[top:bottom, left:right] = True
    expected &= truth["millimetres"] > 0
    assert np.array_equal(frame.valid, expected) and 0 < expected.sum() < expected.size - 1
    assert frame.provenance == {"workspace": Provenance.GROUND_TRUTH}
    batch = dataset[0]
    assert batch.supervision is None and batch.observation.gt_assisted() == ["workspace"]
    assert len(batch.observation.points) == expected.sum()


def test_without_labels_and_ground_truth_crop_no_annotation_file_is_opened(tree):
    for sub in ("label", "mask_visib"):
        for file in (tree["root"] / "scenes").rglob(f"{sub}/*.png"):
            file.unlink()
    for file in (tree["root"] / "scenes").rglob("scene_gt.json"):
        file.unlink()
    dataset = build(tree, load_labels=False, preprocess=PLAIN)
    batch = dataset.collate([dataset[0], dataset[1]])
    assert batch.supervision is None and batch.observation.gt_assisted() == []
    boxed = build(
        tree,
        load_labels=False,
        preprocess=[
            PLAIN[0],
            {"type": "TableBoxWorkspace", "lower": [-9, -9, -9], "upper": [9, 9, 9]},
        ],
    )
    assert boxed[0].observation.provenance == {"workspace": Provenance.CALIBRATION}
    with pytest.raises(FileNotFoundError, match="label"):
        build(tree, load_labels=False)[0]


def test_training_item_and_collation(tree):
    transforms = [
        {"type": "RandomSample", "num_points": 64},
        {"type": "FilterObjects", "min_visible_points": 1},
        {
            "type": "SampleGraspLabels",
            "point_fraction": 0.5,
            "min_points": 2,
            "collision": "zero_scores",
        },
        {"type": "RandomFlip", "axis": "x", "prob": 0.5},
        {"type": "ConstantFeatures", "channels": 3},
    ]
    dataset = build(tree, transforms=transforms)
    batch = dataset.collate([dataset[0], dataset[3]])
    assert batch.observation.points.batch_offsets.tolist() == [0, 64, 128]
    assert batch.observation.intrinsics.shape == (2, 3, 3)
    assert not torch.equal(batch.observation.intrinsics[0], batch.observation.intrinsics[1])
    assert set(batch.supervision.point_labels) == {"objectness", "instance"}
    assert [[o.object_id for o in objects] for objects in batch.supervision.objects][1] == [3, 12]
    assert dataset.asset_records()[0]["kind"] == "grasp_labels"
    with pytest.raises(ContractError, match="no upstream-format graspness layout is known"):
        build(tree, graspness={"source": "legacy"})


def test_native_graspness_works_through_the_shared_generator(tree):
    config = {
        "source": "native",
        "friction_threshold": 0.8,
        "poses": "scene_gt",
        "dtype": "float32",
    }
    dataset = build(tree, graspness=config, check_assets=False)
    maps = dataset.graspness
    maps.asset.create()
    for scene in dataset.scenes():
        maps.mark_complete(
            scene, generate_graspness(dataset, maps, scene, dataset.graspness_config)
        )
    ready = build(tree, graspness=config)
    sample = ready.sample(ready.keys[2])
    values = sample.point_labels["graspness"]
    assert values.min() == 0 and values.max() == 1 and len(values) == len(sample)
    assert ready.asset_records()[1]["uses_ground_truth"] is True
    with pytest.raises(ContractError, match="`graspness.poses` must be one of scene_gt"):
        build(tree, graspness={**config, "poses": "annotations"})


def test_inconsistent_annotation_is_refused(tree):
    file = tree["root"] / "scenes/000005/scene_gt.json"
    content = json.loads(file.read_text())
    content["6"] = content["6"][::-1]  # view 1 of the D435 lists the instances in another order
    file.write_text(json.dumps(content))
    dataset = build(tree, load_labels=False, preprocess=PLAIN)
    dataset.object_poses(dataset.keys[0])
    with pytest.raises(
        ContractError, match=r"lists objects \[3, 7, 7\] for this image, but \[7, 7, 3\]"
    ):
        dataset.object_poses(dataset.keys[1])
    del content["6"]
    file.write_text(json.dumps(content))
    with pytest.raises(ContractError, match="scene_gt.json has no entry for image 6"):
        build(tree, load_labels=False, preprocess=PLAIN).object_poses(dataset.keys[1])


def test_split_files_are_required_and_must_be_disjoint(tree):
    (tree["root"] / "split_info/grasp_test_scene_ids.json").write_text(json.dumps(["000005"]))
    with pytest.raises(ContractError, match="train and test scenes overlap"):
        build(tree, load_labels=False)
    (tree["root"] / "split_info/grasp_test_scene_ids.json").unlink()
    with pytest.raises(FileNotFoundError, match="grasp_test_scene_ids.json not found"):
        build(tree, load_labels=False)


def test_check_tree_and_prepare(tree, capsys, tmp_path):
    root = tree["root"]
    assert TinyClutter.check_tree(root) == []
    (root / "scenes/000009/depth/000007.png").unlink()
    (root / "scenes/000009/rgb/000002.png").unlink()
    (root / "collision_label/000002.npz").unlink()
    (root / "grasp_label/obj_000012_labels.npz").unlink()
    (root / "models_m/obj_000003.ply").unlink()
    (root / "dex_models/obj_000007.pkl").unlink()
    assert TinyClutter.check_tree(root) == [
        "collision_label/000002.npz is missing",
        "scenes/000009/depth/: 1 of 8 images missing, e.g. 000007.png",
        "scenes/000009/rgb/: 1 of 8 images missing, e.g. 000002.png",
        "grasp_label/: labels of 1 objects missing, e.g. [12]",
        "models_m/: models of 1 objects missing, e.g. [3]",
        "dex_models/: pickles of 1 objects missing, e.g. [7]",
    ]
    assert TinyClutter.check_tree(root, ["zivid"]) == [
        "collision_label/000002.npz is missing",
        "grasp_label/: labels of 1 objects missing, e.g. [12]",
        "models_m/: models of 1 objects missing, e.g. [3]",
        "dex_models/: pickles of 1 objects missing, e.g. [7]",
    ]
    assert GraspClutter6D.default_config() == "_base_/datasets/graspclutter6d_d435.yaml"
    assert GraspClutter6D.default_config("zivid") == "_base_/datasets/graspclutter6d_zivid.yaml"

    import yaml

    node = {
        "type": "TinyClutter",
        "camera": "realsense-d435",
        "split": "sample",
        "root": str(root),
        "preprocess": PREPROCESS,
        "transforms": [],
    }
    config = tmp_path / "clutter.yaml"
    config.write_text(yaml.safe_dump({"data": {"train": {"dataset": node}}}))
    base = ["prepare", "tinyclutter", "--config", str(config)]
    assert main([*base, "--labels"]) == 0  # scene 5: objects 3 and 7, one scene
    assert "3 shards built and verified against their sources" in capsys.readouterr().out
    TinyClutter(**{k: v for k, v in node.items() if k != "type"})
    with pytest.raises(AssetError, match="grasppanda prepare tinyclutter --labels"):
        TinyClutter(**{**{k: v for k, v in node.items() if k != "type"}, "split": "train"})


def test_shipped_config_is_valid_and_declares_the_ground_truth_crop():
    from grasppanda.config import Config

    for camera in GraspClutter6D.cameras:
        config = Config.load(GraspClutter6D.default_config(camera))
        for loader, split in (("train", "train"), ("val", "val_dev"), ("test", "test")):
            node = config.select(f"data.{loader}.dataset")
            assert node["type"] == "GraspClutter6D" and node["camera"] == camera
            assert node["split"] == split and node["root"] is None
            assert node["preprocess"][1] == {"type": "SegmentationRectWorkspace", "margin": 0.1}
        assert config.select("data.test.dataset.load_labels") is False
        assert config.select("data.train.dataset.graspness.poses") == "scene_gt"
