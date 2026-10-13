"""Image inputs and label options that an image-based detector needs from a scene dataset.

HGGD reads the colour and depth images beside a cloud and trains on per-view labels
instead of the dense per-object grids. These are options of the scene reader; the
defaults leave what the point-cloud methods load untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from grasppanda.config import Config, ConfigError
from grasppanda.structures import ContractError, Observation, SampleKey, collate
from grasppanda.transforms.frame import Frame, PointSample, align, to_batch
from grasppanda.transforms.image import DepthToCloud, NoWorkspace

from .test_data_graspnet1b import FRAMES, PREPROCESS, SCENES, SIZE, TEST, TRAIN, build, tree  # noqa: F401

KEY = SampleKey("unit", 3, "realsense", 7)
UNMASKED = [{"type": "DepthToCloud"}, {"type": "NoWorkspace"}]
SAMPLE = [{"type": "RandomSample", "num_points": 64}]


def raw_frame(with_colour: bool = True) -> Frame:
    generator = np.random.default_rng(0)
    depth = generator.integers(400, 900, size=(6, 8)).astype(np.uint16)
    depth[0, :3] = 0
    colour = generator.integers(0, 256, size=(6, 8, 3), dtype=np.uint8) if with_colour else None
    intrinsics = np.array([[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]])
    return Frame(key=KEY, depth=depth, depth_scale=1000.0, intrinsics=intrinsics, rgb=colour)


def aligned(images=("rgb", "depth")) -> tuple[Frame, PointSample]:
    frame = NoWorkspace()(DepthToCloud()(raw_frame()))
    return frame, align(frame, images=images)


# -------------------------------------------------------------- the frame containers


def test_images_cross_the_alignment_boundary_only_when_asked():
    frame, sample = aligned()
    assert sample.rgb is frame.rgb and sample.depth.dtype == np.float32
    assert np.array_equal(sample.depth, (frame.depth / np.float32(1000.0)).astype(np.float32))
    assert sample.depth.shape == (6, 8) and (sample.depth == 0).sum() == 3  # holes stay zero
    assert len(sample) == 6 * 8 - 3  # the cloud is masked, the image is not
    plain = align(frame)
    assert plain.rgb is None and plain.depth is None  # the default: a point-cloud sample
    only_depth = align(frame, images=("depth",))
    assert only_depth.rgb is None and only_depth.depth is not None
    with pytest.raises(ValueError, match=r"unknown image\(s\) \['normals'\]"):
        align(frame, images=("normals",))
    colourless = NoWorkspace()(DepthToCloud()(raw_frame(with_colour=False)))
    with pytest.raises(ContractError, match="did not provide a colour image"):
        align(colourless, images=("rgb",))


def test_batch_holds_colour_in_unit_range_and_depth_in_metres():
    frame, sample = aligned()
    observation = to_batch(sample).observation
    assert tuple(observation.rgb.shape) == (1, 3, 6, 8) and observation.rgb.dtype == torch.float32
    assert torch.equal(
        torch.round(observation.rgb[0] * 255).to(torch.uint8),
        torch.from_numpy(frame.rgb).permute(2, 0, 1),
    )
    assert tuple(observation.depth.shape) == (1, 6, 8)
    assert float(observation.depth.max()) < 1.0 and float(observation.depth[0, 0, 0]) == 0.0
    assert observation.provides() >= {"rgb", "depth", "intrinsics", "points"}
    assert to_batch(align(frame)).observation.provides() == {"intrinsics", "points", "full_points"}
    both = collate([to_batch(aligned()[1]), to_batch(aligned()[1])])
    assert tuple(both.observation.rgb.shape) == (2, 3, 6, 8)
    assert tuple(both.observation.depth.shape) == (2, 6, 8)
    mixed = [to_batch(aligned()[1]), to_batch(align(aligned()[0]))]
    with pytest.raises(ContractError, match="'depth' is present in some frames only"):
        collate(mixed)


def test_a_sample_with_images_refuses_point_space_augmentation():
    _, sample = aligned()
    with pytest.raises(ContractError, match=r"leave the image-space data.*\(rgb, depth\)"):
        sample.apply_linear(np.diag([-1.0, 1.0, 1.0]))
    _, plain = aligned(images=())
    plain.apply_linear(np.diag([-1.0, 1.0, 1.0]))  # a cloud alone may be flipped
    # A label that does not depend on the pose of the cloud follows any map; one that is
    # expressed in the camera frame does not.
    plain.extras["scale_histogram"] = {"counts": np.zeros(2)}
    plain.apply_linear(np.eye(3))
    plain.extras["view_grasps"] = {"theta": np.zeros(2)}
    plain.camera_bound.add("view_grasps")
    with pytest.raises(ContractError, match=r"\(view_grasps\)"):
        plain.apply_linear(np.eye(3))


def test_extras_reach_the_supervision_as_tensors_one_entry_per_frame():
    _, sample = aligned()
    sample.objects = []
    sample.extras["view_grasps"] = {
        "theta": np.linspace(0, 1, 4),
        "centers": np.arange(8.0).reshape(4, 2),
        "object_id": np.arange(4),
    }
    batch = to_batch(sample)
    (labels,) = batch.supervision.extras["view_grasps"]
    assert batch.supervision.objects == [[]]
    assert labels["theta"].dtype == torch.float64 and tuple(labels["centers"].shape) == (4, 2)
    assert labels["object_id"].dtype == torch.int64
    merged = collate([batch, to_batch(sample)])
    assert len(merged.supervision.extras["view_grasps"]) == 2
    moved = merged.to("cpu").supervision.extras["view_grasps"][1]["theta"]
    assert torch.equal(moved, labels["theta"])
    sample.objects = None  # labels without supervision cannot be right
    with pytest.raises(ContractError, match="carries labels .* but no supervision"):
        to_batch(sample)


# ------------------------------------------------------------------------ the reader


@pytest.fixture
def colour_tree(tree):  # noqa: F811
    """The synthetic tree with colour images, which the point-cloud tests do not need."""
    generator = np.random.default_rng(1)
    written = {}
    for scene in SCENES:
        for camera in ("realsense", "kinect"):
            directory = tree["root"] / "scenes" / f"scene_{scene:04d}" / camera / "rgb"
            directory.mkdir(parents=True)
            for frame in range(FRAMES):
                image = generator.integers(0, 256, size=(*SIZE, 3), dtype=np.uint8)
                Image.fromarray(image).save(directory / f"{frame:04d}.png")
                written[scene, camera, frame] = image
    tree["rgb"] = written
    return tree


def test_reader_serves_images_beside_an_unmasked_cloud(colour_tree):
    dataset = build(
        colour_tree,
        load_labels=False,
        images=["rgb", "depth"],
        preprocess=UNMASKED,
        transforms=SAMPLE,
        split="test",
        keep_full_points=True,
    )
    batch = dataset[1]
    observation: Observation = batch.observation
    truth = colour_tree["frames"][2, "realsense", 1]
    assert batch.supervision is None and observation.keys == [dataset.keys[1]]
    assert torch.equal(
        torch.round(observation.rgb[0] * 255).to(torch.uint8),
        torch.from_numpy(colour_tree["rgb"][2, "realsense", 1]).permute(2, 0, 1),
    )
    assert np.allclose(observation.depth[0].numpy() * 1000.0, truth["depth"], atol=1e-3)
    assert len(observation.full_points) == int((truth["depth"] > 0).sum())  # every pixel with depth
    assert observation.gt_assisted() == [] and observation.provenance == {}
    assert len(observation.points) == 64 and dataset.reads_ground_truth is False


def test_existing_configurations_read_no_image(colour_tree, monkeypatch):
    """The default loads what it loaded before: no colour file is even opened."""
    from grasppanda.datasets import graspnet1b

    opened = []
    original = graspnet1b.read_image
    monkeypatch.setattr(
        graspnet1b, "read_image", lambda path: opened.append(path) or original(path)
    )
    dataset = build(colour_tree, load_labels=False, transforms=TEST)
    observation = dataset[0].observation
    assert observation.rgb is None and observation.depth is None
    assert not [path for path in opened if "rgb" in path.parts]
    assert observation.provides() == {"intrinsics", "points", "point_feats", "full_points"}
    assert observation.gt_assisted() == ["workspace"]  # the Graspness crop, as before


def test_images_and_point_augmentation_do_not_mix(colour_tree):
    dataset = build(colour_tree, images=["rgb", "depth"], transforms=TRAIN)
    with pytest.raises(ContractError, match="leave the image-space data"):
        for index in range(len(dataset)):  # the flip is random: some frame draws it
            dataset[index]


def test_supervision_without_the_dense_object_labels(colour_tree):
    dataset = build(
        colour_tree,
        load_labels=True,
        object_labels=False,
        images=["rgb", "depth"],
        preprocess=UNMASKED,
        transforms=SAMPLE,
        check_assets=True,
    )
    batch = dataset[0]
    assert batch.supervision is not None and batch.supervision.objects == [[]]
    assert set(batch.supervision.point_labels) == {"objectness", "instance"}
    assert dataset.asset_records() == []  # no derived asset is read
    # ... and none is required: a fresh cache without the label store is fine
    type(dataset)(
        camera="realsense",
        split="train",
        preprocess=UNMASKED,
        transforms=SAMPLE,
        root=str(colour_tree["root"]),
        object_labels=False,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"images": ["rgb"], "load_labels": False, "camera": "realsense"}, None),
        (
            {"object_labels": False, "load_labels": False},
            "only makes sense with `load_labels: true`",
        ),
        (
            {"view_grasps": {"friction_threshold": 0.2}, "load_labels": False},
            "`view_grasps` is a label and needs `load_labels: true`",
        ),
        ({"view_grasps": {"nms_angle": 270.0}}, r"nms_angle must be in \(0, 180\] degrees"),
        ({"view_grasps": {"poses": "icp"}}, "`view_grasps.poses` is 'icp'"),
    ],
)
def test_label_options_are_checked_in_the_config(colour_tree, overrides, message):
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "preprocess": PREPROCESS,
        "transforms": TEST,
        "root": str(colour_tree["root"]),
        **overrides,
    }
    config = {"data": {"train": {"dataset": node}}}
    if message is None:
        Config.from_dict(config)
        return
    with pytest.raises(ConfigError, match=message):
        Config.from_dict(config)
    with pytest.raises(ContractError, match=message):
        build(colour_tree, **overrides)


def test_unknown_images_are_rejected_by_the_validator(colour_tree):
    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "load_labels": False,
        "preprocess": PREPROCESS,
        "transforms": TEST,
        "images": ["normals"],
    }
    with pytest.raises(ConfigError, match="images"):
        Config.from_dict({"data": {"train": {"dataset": node}}})
