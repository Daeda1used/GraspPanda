"""The GraspNet-1Billion reader on a tiny synthetic tree.

``make_tree`` writes a dataset in the GraspNet-1Billion layout with 24 x 32 images and a
5 x 3 x 2 label grid, and returns what it wrote. ``TinyGraspNet`` is the real reader with
those two constants changed. Other data tests import both from here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat

from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspnet1b import GraspNet1B, quaternion_to_matrix
from grasppanda.datasets.labels import LabelGrid
from grasppanda.datasets.manifest import SplitManifest
from grasppanda.registry import DATASETS
from grasppanda.structures import Batch, ContractError, Provenance, SampleKey

GRID = LabelGrid(num_view=5, num_angle=3, num_depth=2)
SIZE = (24, 32)
FRAMES = 3
SCENES = {0: [3, 7], 1: [7, 5], 2: [5, 3]}
POINTS = {3: 7, 5: 9, 7: 11}
FOCAL, CENTRE = 40.0, (16.0, 12.0)
FRICTION = np.array([-1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)


@DATASETS.register()
class TinyGraspNet(GraspNet1B):
    name = "tinygraspnet"
    label_grid = GRID
    image_size = SIZE

    @classmethod
    def load_manifest(cls, root=None) -> SplitManifest:
        return SplitManifest.build(
            cls.name,
            cls.cameras,
            range(FRAMES),
            {"train": [0, 1], "test": [2], "all": [0, 1, 2], "sample": [0]},
        )


def _quaternion(angle: float) -> tuple[float, float, float, float]:
    """Rotation by ``angle`` about the axis (1, 2, 3) / sqrt(14), as (w, x, y, z)."""
    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    return (math.cos(angle / 2), *(math.sin(angle / 2) * axis))


def make_tree(root: Path, *, simplified: bool = False, seed: int = 0) -> dict:
    """Write a synthetic dataset below ``root`` and return its content."""
    rng = np.random.default_rng(seed)
    truth: dict = {"labels": {}, "collision": {}, "frames": {}}
    label_dir = root / ("grasp_label_simplified" if simplified else "grasp_label")
    label_dir.mkdir(parents=True)
    for object_id, count in POINTS.items():
        points = rng.uniform(-0.03, 0.03, (count, 3)).astype(np.float32)
        scores = rng.choice(FRICTION, (count, *GRID.shape))
        widths = rng.uniform(0.0, 0.1, (count, *GRID.shape)).astype(np.float32)
        truth["labels"][object_id] = (points, scores, widths)
        if simplified:
            arrays = {"points": points, "scores": scores, "width": widths}
        else:  # the official files hold (angle, depth, width) triples; own stream, so
            # that both variants of one seed carry the same labels
            filler = np.random.default_rng(seed + 1000 + object_id)
            offsets = filler.uniform(0, 1, (count, *GRID.shape, 3)).astype(np.float32)
            offsets[..., 2] = widths
            arrays = {"points": points, "scores": scores, "offsets": offsets}
        np.savez(label_dir / f"{object_id:03d}_labels.npz", **arrays)

    intrinsics = np.array([[FOCAL, 0, CENTRE[0]], [0, FOCAL, CENTRE[1]], [0, 0, 1.0]])
    table_from_first = np.diag([1.0, -1.0, -1.0, 1.0])
    table_from_first[2, 3] = 0.6
    for scene, object_ids in SCENES.items():
        name = f"scene_{scene:04d}"
        (root / "scenes" / name).mkdir(parents=True)
        (root / "scenes" / name / "object_id_list.txt").write_text(
            "".join(f"{i}\n" for i in object_ids)
        )
        collisions = [rng.random((POINTS[i], *GRID.shape)) < 0.3 for i in object_ids]
        truth["collision"][scene] = collisions
        (root / "collision_label" / name).mkdir(parents=True)
        np.savez(
            root / "collision_label" / name / "collision_labels.npz",
            **{f"arr_{k}": array for k, array in enumerate(collisions)},
        )
        for camera in TinyGraspNet.cameras:
            directory = root / "scenes" / name / camera
            for sub in ("depth", "label", "meta", "annotations"):
                (directory / sub).mkdir(parents=True)
            camera_poses = np.tile(np.eye(4), (FRAMES, 1, 1))
            camera_poses[:, 0, 3] = 0.01 * np.arange(FRAMES)
            np.save(directory / "camera_poses.npy", camera_poses)
            np.save(directory / "cam0_wrt_table.npy", table_from_first)
            for frame in range(FRAMES):
                depth = np.full(SIZE, 600, dtype=np.uint16)
                label = np.zeros(SIZE, dtype=np.uint8)
                poses = np.zeros((3, 4, len(object_ids)), dtype=np.float32)
                entries = []
                # Both objects sit right of the optical axis, so the camera centre is
                # outside the horizontal extent of the foreground.
                boxes = [(4 + frame, 10 + frame, 18, 23), (14, 20, 22 + frame, 30)]
                for k, (object_id, (top, bottom, left, right)) in enumerate(zip(object_ids, boxes)):
                    millimetres = 520 - 40 * k
                    depth[top:bottom, left:right] = millimetres
                    label[top:bottom, left:right] = object_id + 1
                    z = millimetres / 1000.0
                    u, v = (left + right - 1) / 2, (top + bottom - 1) / 2
                    position = [(u - CENTRE[0]) * z / FOCAL, (v - CENTRE[1]) * z / FOCAL, z]
                    quaternion = _quaternion(0.3 * (k + 1) + 0.1 * frame)
                    poses[:, :3, k] = quaternion_to_matrix(*quaternion)
                    poses[:, 3, k] = position
                    entries.append(
                        f"<obj><obj_id>{object_id}</obj_id><obj_name>o</obj_name>"
                        f"<obj_path>p</obj_path><pos_in_world>"
                        f"{' '.join(f'{c:.4f}' for c in position)}</pos_in_world><ori_in_world>"
                        f"{' '.join(f'{c:.4f}' for c in quaternion)}</ori_in_world></obj>"
                    )
                depth[0, 0] = depth[1, 5] = 0  # holes in the table
                depth[boxes[0][0], boxes[0][2]] = 0  # a foreground pixel without depth
                stem = f"{frame:04d}"
                Image.fromarray(depth).save(directory / "depth" / f"{stem}.png")
                Image.fromarray(label).save(directory / "label" / f"{stem}.png")
                savemat(
                    str(directory / "meta" / f"{stem}.mat"),
                    {
                        "cls_indexes": np.array(object_ids, dtype=np.int32) + 1,
                        "poses": poses,
                        "intrinsic_matrix": intrinsics,
                        "factor_depth": np.array([[1000.0]]),
                    },
                )
                (directory / "annotations" / f"{stem}.xml").write_text(
                    f"<scene>{''.join(entries)}</scene>"
                )
                truth["frames"][scene, camera, frame] = {
                    "depth": depth,
                    "label": label,
                    "poses": poses,
                    "table_from_camera": table_from_first @ camera_poses[frame],
                    "intrinsics": intrinsics,
                }
    return truth


PREPROCESS = [
    {"type": "DepthToCloud"},
    {"type": "SegmentationBoxWorkspace", "margin": 0.02, "invalid_depth_foreground": "include"},
]
TRAIN = [
    {"type": "RandomSample", "num_points": 64},
    {"type": "FilterObjects", "min_visible_points": 2},
    {
        "type": "SampleGraspLabels",
        "point_fraction": 0.25,
        "min_points": 4,
        "collision": "zero_scores",
    },
    {"type": "RandomFlip", "axis": "x", "prob": 0.5},
    {"type": "RandomRotation", "axis": "x", "max_angle": 30.0},
    {"type": "ConstantFeatures", "channels": 3},
]
TEST = [{"type": "RandomSample", "num_points": 64}, {"type": "ConstantFeatures", "channels": 3}]


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A synthetic dataset, an empty asset cache, and the truth that was written."""
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    truth = make_tree(tmp_path / "data")
    truth["root"] = tmp_path / "data"
    return truth


def build(tree: dict, **overrides) -> TinyGraspNet:
    """A dataset on the synthetic tree; the label store is built when labels are loaded."""
    arguments = {
        "camera": "realsense",
        "split": "train",
        "preprocess": PREPROCESS,
        "transforms": TRAIN,
        "root": str(tree["root"]),
        "seed": 0,
    }
    arguments.update(overrides)
    if arguments.get("load_labels", True):
        probe = TinyGraspNet(**{**arguments, "check_assets": False})
        scenes = probe.manifest.all_scenes()
        probe.label_store.build({i for s in scenes for i in probe.scene_object_ids(s)}, scenes)
    return TinyGraspNet(**arguments)


# ------------------------------------------------------------------------------ tests


def test_item_is_a_batch_of_one_frame_with_supervision(tree):
    dataset = build(tree)
    assert len(dataset) == 2 * FRAMES and dataset.keys[4] == SampleKey(
        "tinygraspnet", 1, "realsense", 1
    )
    batch = dataset[4]
    assert isinstance(batch, Batch) and len(batch) == 1
    observation, supervision = batch.observation, batch.supervision
    assert observation.keys == [dataset.keys[4]]
    assert (
        observation.points.coords.shape == (64, 3)
        and observation.points.coords.dtype == torch.float32
    )
    assert observation.point_feats.shape == (64, 3) and (observation.point_feats == 1).all()
    assert observation.intrinsics.shape == (1, 3, 3)
    assert observation.provenance == {"workspace": Provenance.GROUND_TRUTH}
    assert observation.gt_assisted() == ["workspace"]
    assert set(supervision.point_labels) == {"objectness", "instance"}
    assert supervision.point_labels["objectness"].dtype == torch.int64
    assert torch.equal(
        supervision.point_labels["objectness"], (supervision.point_labels["instance"] > 0).long()
    )
    (objects,) = supervision.objects
    assert [o.object_id for o in objects] == SCENES[1]
    for labels in objects:
        kept = max(int(POINTS[labels.object_id] * 0.25), 4)
        assert labels.points.shape == (kept, 3) and labels.pose.shape == (4, 4)
        assert labels.scores.shape == labels.widths.shape == (kept, *GRID.shape)
        assert labels.collision is None  # folded into the scores by `zero_scores`


def test_without_labels_no_ground_truth_is_read(tree):
    """`load_labels=False` gives no supervision; ground truth is opened only if declared."""
    calibrated = [
        {"type": "DepthToCloud"},
        {"type": "TableBoxWorkspace", "lower": [-0.3, -0.3, -0.05], "upper": [0.3, 0.3, 0.3]},
    ]
    for file in (tree["root"] / "scenes").rglob("label/*.png"):
        file.unlink()
    for file in (tree["root"] / "collision_label").rglob("*.npz"):
        file.unlink()
    dataset = build(tree, load_labels=False, preprocess=calibrated, transforms=TEST)
    batch = dataset[0]
    assert batch.supervision is None
    assert batch.observation.provenance == {"workspace": Provenance.CALIBRATION}
    assert batch.observation.gt_assisted() == []
    # The reference crop needs the segmentation even at test time, and says so.
    leaky = build(tree, load_labels=False, transforms=TEST)
    with pytest.raises(FileNotFoundError, match="label"):
        leaky[0]


def test_full_points_are_the_masked_cloud_before_sampling(tree):
    dataset = build(tree, load_labels=False, transforms=TEST)
    key = dataset.keys[0]
    truth = tree["frames"][0, "realsense", 0]
    frame = dataset.frame(key)
    batch = dataset[0]
    full = batch.observation.full_points.coords
    assert len(full) == frame.valid.sum() < (truth["depth"] > 0).sum()
    assert torch.equal(full, torch.from_numpy(frame.cloud[frame.valid].astype(np.float32)))
    # every sampled point is one of them
    sampled = batch.observation.points.coords
    assert (sampled[:, None, :] == full[None, :, :]).all(dim=-1).any(dim=1).all()
    assert (
        build(tree, load_labels=False, transforms=TEST, keep_full_points=False)[
            0
        ].observation.full_points
        is None
    )


def test_point_labels_stay_aligned_through_sampling(tree):
    dataset = build(tree, transforms=[{"type": "RandomSample", "num_points": 40}])
    key = dataset.keys[1]
    truth = tree["frames"][key.scene, key.camera, key.frame]
    sample = dataset.sample(key)
    dataset.transforms[0](sample)
    rows, columns = np.divmod(sample.pixels, SIZE[1])
    assert np.array_equal(sample.point_labels["instance"], truth["label"][rows, columns])
    z = truth["depth"][rows, columns] / 1000.0
    assert np.allclose(sample.points[:, 2], z)
    assert np.allclose(sample.points[:, 0], (columns - CENTRE[0]) * z / FOCAL)


def test_object_labels_equal_the_source_arrays(tree):
    """With given indices the label arrays are the source rows, colliding scores zeroed."""
    dataset = build(tree, transforms=[])
    key = SampleKey("tinygraspnet", 1, "realsense", 2)
    sample = dataset.sample(key)
    assert [a.object_id for a in sample.objects] == SCENES[1]
    step = DATASETS.get("TinyGraspNet")  # noqa: F841 - registered above
    from grasppanda.transforms.points import SampleGraspLabels

    rows = [np.array([4, 0, 9]), np.array([8, 1])]
    for mode in ("zero_scores", "attach", "ignore"):
        sample = dataset.sample(key)
        SampleGraspLabels(1.0, 0, mode).apply(sample, rows)
        for index, (annotation, chosen) in enumerate(zip(sample.objects, rows)):
            points, scores, widths = tree["labels"][annotation.object_id]
            collision = tree["collision"][1][index][chosen]
            expected = scores[chosen].copy()
            if mode == "zero_scores":
                expected[collision] = 0
            assert np.array_equal(annotation.points, points[chosen])
            assert np.array_equal(annotation.widths, widths[chosen])
            assert np.array_equal(annotation.scores, expected)
            if mode == "attach":
                assert np.array_equal(annotation.collision, collision)
            else:
                assert annotation.collision is None
            assert np.array_equal(
                annotation.pose[:3], tree["frames"][1, "realsense", 2]["poses"][:, :, index]
            )


def test_seed_makes_items_reproducible_and_none_follows_torch(tree):
    dataset = build(tree)
    first, second = dataset[2], dataset[2]
    assert torch.equal(first.observation.points.coords, second.observation.points.coords)
    assert torch.equal(
        first.supervision.objects[0][0].scores, second.supervision.objects[0][0].scores
    )
    free = build(tree, seed=None)
    torch.manual_seed(5)
    one = free[2].observation.points.coords
    two = free[2].observation.points.coords
    torch.manual_seed(5)
    again = free[2].observation.points.coords
    assert torch.equal(one, again) and not torch.equal(one, two)


def test_frames_of_different_sizes_collate(tree):
    dataset = build(
        tree,
        transforms=[
            {"type": "FilterObjects", "min_visible_points": 2},
            {
                "type": "SampleGraspLabels",
                "point_fraction": 1.0,
                "min_points": 0,
                "collision": "attach",
            },
        ],
    )
    items = [dataset[0], dataset[4]]
    sizes = [len(item.observation.points) for item in items]
    assert sizes[0] != sizes[1]
    batch = dataset.collate(items)
    assert batch.observation.points.batch_offsets.tolist() == [0, sizes[0], sum(sizes)]
    assert batch.supervision.point_labels["instance"].shape == (sum(sizes),)
    assert [len(objects) for objects in batch.supervision.objects] == [2, 2]
    assert batch.supervision.objects[1][0].collision.dtype == torch.bool


def test_frame_step_and_splits(tree):
    assert len(build(tree, load_labels=False, transforms=TEST, frame_step=2)) == 2 * 2
    assert [
        k.scene for k in build(tree, load_labels=False, transforms=TEST, split="test").keys
    ] == [2] * 3
    with pytest.raises(
        ValueError, match="'traln' is not a split of tinygraspnet. Did you mean 'train'"
    ):
        build(tree, load_labels=False, transforms=TEST, split="traln")


def test_loader_with_workers_matches_in_process_reading(tree):
    from .conftest import worker_loader

    dataset = build(tree)
    loader = worker_loader(dataset, "spawn", batch_size=2, collate_fn=dataset.collate)
    batches = list(loader)
    assert sum(len(batch) for batch in batches) == len(dataset)
    expected = dataset.collate([dataset[0], dataset[1]])
    assert torch.equal(batches[0].observation.points.coords, expected.observation.points.coords)
    assert torch.equal(
        batches[0].supervision.objects[1][1].widths, expected.supervision.objects[1][1].widths
    )


def test_missing_assets_name_the_command_that_builds_them(tree):
    with pytest.raises(AssetError, match=r"grasppanda prepare tinygraspnet --labels"):
        TinyGraspNet("realsense", "train", PREPROCESS, TRAIN, root=str(tree["root"]))
    dataset = build(tree)  # builds the store
    assert dataset.asset_records()[0]["kind"] == "grasp_labels"
    native = {"source": "native", "friction_threshold": 0.8, "poses": "annotations"}
    with pytest.raises(AssetError, match=r"prepare tinygraspnet --graspness --camera realsense"):
        build(tree, graspness=native)


def test_labels_that_were_never_loaded_are_an_error_not_a_zero(tree):
    dataset = build(tree, transforms=[{"type": "RandomSample", "num_points": 16}])
    with pytest.raises(ContractError, match="were never loaded. Add a label transform"):
        dataset[0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"preprocess": [PREPROCESS[0]]}, "exactly one workspace provider (found: none)"),
        (
            {"preprocess": [*PREPROCESS, {"type": "NoWorkspace"}]},
            "found: SegmentationBoxWorkspace, NoWorkspace",
        ),
        (
            {"preprocess": [*PREPROCESS, {"type": "RandomSample", "num_points": 8}]},
            "preprocess[2] is RandomSample, which is not an image-space transform",
        ),
        ({"transforms": [{"type": "DepthToCloud"}]}, "transforms[0] is DepthToCloud"),
        ({"load_labels": False, "graspness": {"source": "legacy"}}, "needs `load_labels: true`"),
    ],
)
def test_config_check_rejects_broken_pipelines(tree, overrides, message):
    from grasppanda.config import Config, ConfigError

    node = {
        "type": "TinyGraspNet",
        "camera": "realsense",
        "split": "train",
        "preprocess": PREPROCESS,
        "transforms": TEST,
        **overrides,
    }
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"data": {"train": {"dataset": node}}})
    assert message in str(error.value)
    with pytest.raises((ContractError, AssetError)):  # the constructor enforces it as well
        TinyGraspNet(**{k: v for k, v in node.items() if k != "type"}, root=str(tree["root"]))


def test_config_validation_knows_the_dataset_signature():
    from grasppanda.config import Config, ConfigError

    node = {"type": "TinyGraspNet", "camera": "realsense", "split": "train", "transforms": TEST}
    with pytest.raises(ConfigError, match="TinyGraspNet requires 'preprocess'"):
        Config.from_dict({"data": {"train": {"dataset": node}}})
    node["preprocess"] = [{"type": "SegmentationBoxWorkspace", "margn": 0.02}]
    with pytest.raises(ConfigError, match="Did you mean 'margin'"):
        Config.from_dict({"data": {"train": {"dataset": node}}})
    node["preprocess"] = PREPROCESS
    node["graspness"] = {"source": "native", "friction_treshold": 0.8}
    with pytest.raises(ConfigError, match="Did you mean 'friction_threshold'"):
        Config.from_dict({"data": {"train": {"dataset": node}}})
    node["graspness"] = {"source": "native"}  # the generator's parameters have no defaults
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"data": {"train": {"dataset": node}}})
    assert "`graspness.friction_threshold` is required for the native source" in str(error.value)
    assert "`graspness.poses` must be one of meta, annotations" in str(error.value)
    node["graspness"] = {"source": "legacy"}
    resolved = Config.from_dict({"data": {"train": {"dataset": node}}}).resolved()
    assert resolved["data"]["train"]["dataset"]["graspness"] == {
        "source": "legacy",
        "friction_threshold": None,
        "poses": None,
        "dtype": "float16",
        "normalization": "scene",
    }


def test_check_tree_reports_what_is_missing(tree):
    root = tree["root"]
    assert TinyGraspNet.check_tree(root) == []
    (root / "scenes/scene_0001/kinect/depth/0002.png").unlink()
    (root / "collision_label/scene_0002/collision_labels.npz").unlink()
    (root / "grasp_label/005_labels.npz").unlink()
    problems = TinyGraspNet.check_tree(root)
    assert problems == [
        "scenes/scene_0001/kinect/depth/: 1 of 3 .png files missing, e.g. 0002.png",
        "collision_label/scene_0002/collision_labels.npz is missing",
        "grasp_label/: labels of 1 objects missing, e.g. [5]",
    ]
    assert TinyGraspNet.check_tree(root, ["realsense"])[0].startswith("collision_label")
    assert TinyGraspNet.check_tree(root / "nowhere") == [
        f"dataset root {root / 'nowhere'} does not exist"
    ]


def test_poses_come_from_meta_or_from_rounded_annotations(tree):
    dataset = build(tree, load_labels=False, transforms=TEST)
    key = dataset.keys[1]
    meta = dataset.object_poses(key, "meta")
    rounded = dataset.object_poses(key, "annotations")
    assert [i for i, _ in meta] == [i for i, _ in rounded] == SCENES[0]
    for (_, exact), (_, approximate) in zip(meta, rounded):
        assert exact.dtype == approximate.dtype == np.float32 and exact.shape == (4, 4)
        assert not np.array_equal(exact, approximate)
        assert np.allclose(exact, approximate, atol=2e-4)
    (tree["root"] / "scenes/scene_0000/object_id_list.txt").write_text("7\n3\n")
    with pytest.raises(ContractError, match=r"the meta list objects \[3, 7\], but the scene's"):
        build(tree, load_labels=False, transforms=TEST).object_poses(key, "meta")


def test_shipped_dataset_configs_follow_the_graspness_recipe():
    from grasppanda.config import Config

    for camera in GraspNet1B.cameras:
        config = Config.load(GraspNet1B.default_config(camera))
        train = config.select("data.train.dataset")
        assert train["camera"] == camera and train["split"] == "train" and train["root"] is None
        assert train["preprocess"][1] == {
            "type": "SegmentationBoxWorkspace",
            "margin": 0.02,
            "invalid_depth_foreground": "include",
        }
        assert [step["type"] for step in train["transforms"]] == [
            "RandomSample",
            "FilterObjects",
            "SampleGraspLabels",
            "RandomFlip",
            "RandomRotation",
            "ConstantFeatures",
        ]
        assert train["transforms"][0]["num_points"] == 15000
        assert train["transforms"][4] == {"type": "RandomRotation", "axis": "x", "max_angle": 30.0}
        for loader in ("val", "test"):
            node = config.select(f"data.{loader}.dataset")
            assert node["preprocess"] == train["preprocess"] and node["load_labels"] is False
        assert config.select("data.val.dataset.split") == "val_dev"
        assert config.select("data.test.dataset.split") == "test"
