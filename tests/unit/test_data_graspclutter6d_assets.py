"""What every method family needs from GraspClutter6D, on the synthetic BOP-style tree:
images, the label assets of ``grasppanda prepare``, the shipped subsets and the shipped
dataset, transfer and training configs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from grasppanda.cli.main import main
from grasppanda.config import Config, configs_root
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.graspclutter6d import GraspClutter6D
from grasppanda.structures import COMPACT_LABELS, ContractError

from .test_data_graspclutter6d import (  # noqa: F401
    PLAIN,
    PREPROCESS,
    SCENES,
    SIZES,
    TinyClutter,
    build,
    tree,
)

GRASPNESS = {"source": "native", "friction_threshold": 0.8, "poses": "scene_gt", "dtype": "float32"}
COMPACT = {"point_friction_threshold": 0.4, "view_friction_threshold": 0.8, "max_width": 0.14}
TOLERANCE = {"max_friction": 0.8, "min_good_ratio": 0.5, "radius_step": 0.01, "max_radius": 0.03}
VIEW = {"poses": "scene_gt", "intrinsics": "frame", "model_points": 12, "min_width": 0.0}
SAMPLE = [{"type": "RandomSample", "num_points": 64}]
DENSE = [
    *SAMPLE,
    {"type": "SampleGraspLabels", "point_fraction": 1.0, "min_points": 1, "collision": "zero_scores"},
]  # fmt: skip


# ---------------------------------------------------------------------------- images


@pytest.mark.parametrize("camera", ["realsense-d435", "azure-kinect", "zivid"])
def test_reader_serves_colour_and_depth_images_of_the_camera(tree, camera):  # noqa: F811
    dataset = build(
        tree, load_labels=False, camera=camera, preprocess=PLAIN, images=["rgb", "depth"]
    )
    assert dataset.image_size == SIZES[camera]
    key = dataset.keys[1]
    truth = tree["frames"][key.scene, camera, key.frame]
    frame = dataset.read_frame(key)
    assert frame.rgb.dtype == np.uint8 and np.array_equal(frame.rgb, truth["colour"])
    observation = dataset[1].observation
    assert observation.rgb.shape == (1, 3, *SIZES[camera])
    expected = torch.from_numpy(truth["colour"]).permute(2, 0, 1).float() / 255
    assert torch.allclose(observation.rgb[0], expected)
    metres = torch.from_numpy(truth["millimetres"] / 1000).float()
    assert torch.allclose(observation.depth[0], metres, atol=1e-6)  # whatever the raw unit
    assert observation.depth[0, 0, 0] == 0  # the one pixel without depth
    assert observation.gt_assisted() == []


def test_images_are_read_only_when_asked_for(tree):  # noqa: F811
    for file in (tree["root"] / "scenes").rglob("rgb/*.png"):
        file.unlink()
    plain = build(tree, load_labels=False)
    assert plain[0].observation.rgb is None and plain.read_frame(plain.keys[0]).rgb is None
    with pytest.raises(FileNotFoundError):
        build(tree, load_labels=False, images=["rgb"])[0]


def test_a_colour_image_of_the_wrong_size_is_rejected(tree):  # noqa: F811
    directory = tree["root"] / "scenes/000005/rgb"
    (directory / "000002.png").write_bytes((directory / "000003.png").read_bytes())
    with pytest.raises(ContractError, match=r"colour image 000002.png is \(40, 56, 3\)"):
        build(tree, load_labels=False, images=["rgb"])[0]


def test_object_models_are_the_metre_models(tree):  # noqa: F811
    dataset = build(tree, load_labels=False)
    assert dataset.object_model_file(7) == tree["root"] / "models_m" / "obj_000007.ply"
    assert dataset.object_model_file(7).is_file()


# --------------------------------------------------------------------------- prepare


@pytest.fixture
def config(tree, tmp_path):  # noqa: F811
    """Dataset configs for the synthetic tree, one per method family."""

    def write(name: str, **node) -> str:
        base = {
            "type": "TinyClutter",
            "camera": "realsense-d435",
            "split": "train",
            "root": str(tree["root"]),
            "preprocess": PREPROCESS,
            "transforms": SAMPLE,
        }
        file = tmp_path / f"{name}.yaml"
        file.write_text(yaml.safe_dump({"data": {"train": {"dataset": {**base, **node}}}}))
        return str(file)

    return {
        "graspness": write("graspness", graspness=GRASPNESS),
        "compact": write("compact", graspness=GRASPNESS, compact_labels=COMPACT),
        "tolerance": write("tolerance", tolerance=TOLERANCE),
        "views": write(
            "views", view_grasps=VIEW, object_labels=False, images=["rgb", "depth"],
            preprocess=PLAIN,
        ),
    }  # fmt: skip


def test_prepare_builds_the_assets_of_every_method_family(tree, config, capsys):  # noqa: F811
    def prepare(which: str, *flags: str) -> tuple[int, str]:
        status = main(
            ["prepare", "tinyclutter", "--config", config[which], "--workers", "1", *flags]
        )
        captured = capsys.readouterr()
        return status, captured.out + captured.err

    status, out = prepare("compact", "--check")
    assert status == 1 and "tree is complete" in out
    assert "grasp labels 0/5 shards" in out and "graspness 0/2 scenes" in out
    assert "compact grasp labels 0/2 scenes" in out

    status, out = prepare("graspness", "--graspness")  # the dense labels come first
    assert status == 1 and "grasppanda prepare tinyclutter --labels" in out
    assert prepare("graspness", "--labels")[0] == 0

    status, out = prepare("graspness", "--graspness", "--scenes", "5")
    assert status == 0 and "1 scenes written" in out
    status, out = prepare("graspness", "--graspness")  # resumes: scene 9 only
    assert status == 0 and "1 of 2 scenes already present" in out

    status, out = prepare("compact", "--compact-labels")
    assert status == 0 and "2 scenes written and read back" in out
    status, out = prepare("tolerance", "--tolerance")
    assert status == 0 and "3 objects written and verified" in out
    status, out = prepare("views", "--view-grasps")
    assert status == 0 and "2 scenes written" in out

    for which, line in (
        ("compact", "compact grasp labels 2/2 scenes"),
        ("tolerance", "tolerance 3/3 objects"),
        ("views", "view grasps 2/2 scenes"),
    ):
        status, out = prepare(which, "--check", "--camera", "realsense-d435")
        assert status == 0 and line in out, out

    # every asset is now served by the reader its config describes
    graspness = build(tree, graspness=GRASPNESS, transforms=DENSE)[0].supervision
    assert set(graspness.point_labels) == {"objectness", "instance", "graspness"}

    compact = build(tree, graspness=GRASPNESS, compact_labels=COMPACT, transforms=SAMPLE)
    (objects,) = compact[0].supervision.extras[COMPACT_LABELS]
    assert [labels.object_id for labels in objects] == SCENES[5]  # one entry per INSTANCE
    assert compact[0].supervision.objects == [[]]  # in place of the dense grids
    assert [record["kind"] for record in compact.asset_records()] == [
        "compact_grasp_labels",
        "graspness",
    ]

    tolerant = build(tree, tolerance=TOLERANCE, transforms=DENSE)
    served = tolerant[0].supervision.objects[0]
    assert [labels.object_id for labels in served] == SCENES[5]
    assert all(labels.tolerance.shape == labels.scores.shape for labels in served)

    views = build(
        tree, view_grasps=VIEW, object_labels=False, images=["rgb", "depth"], preprocess=PLAIN,
        transforms=SAMPLE,
    )  # fmt: skip
    batch = views[0]
    (labels,) = batch.supervision.extras["view_grasps"]
    height, width = SIZES["realsense-d435"]
    assert len(labels["score"]) > 0 and batch.observation.rgb.shape == (1, 3, height, width)
    u, v = labels["centers"].T
    assert ((u > 0) & (u < width) & (v > 0) & (v < height)).all()
    assert set(labels["object_id"].tolist()) <= set(SCENES[5])
    manifest = json.loads((views.view_grasps.asset.path / "manifest.json").read_text())
    assert manifest["parameters"]["camera"] == "realsense-d435"
    assert manifest["parameters"]["poses"] == "scene_gt"

    # another camera is another asset where the labels depend on the camera
    with pytest.raises(AssetError, match="--graspness --camera zivid"):
        build(tree, graspness=GRASPNESS, camera="zivid")
    with pytest.raises(AssetError, match="--view-grasps --camera zivid"):
        build(tree, view_grasps=VIEW, object_labels=False, camera="zivid")
    build(tree, graspness=GRASPNESS, compact_labels=COMPACT, transforms=SAMPLE, camera="zivid",
          check_assets=False).compact_labels.asset.require(["scene_000005"], build="-")  # fmt: skip


def test_view_grasps_need_the_poses_and_the_calibration_this_dataset_has(tree, config, capsys):  # noqa: F811
    with pytest.raises(ContractError, match=r"`view_grasps.poses` is 'annotations'"):
        build(tree, view_grasps={**VIEW, "poses": "annotations"}, check_assets=False)
    problems = list(
        TinyClutter.check_config(
            {"preprocess": PLAIN, "transforms": [], "view_grasps": {"intrinsics": "frame"}}
        )
    )  # the default pose source is GraspNet-1Billion's annotation files
    assert any("pose sources ['scene_gt']" in problem for problem in problems)

    # the reference converter's constant camera matrices exist for GraspNet-1Billion only
    file = Path(config["views"])
    document = yaml.safe_load(file.read_text())
    document["data"]["train"]["dataset"]["view_grasps"] = {**VIEW, "intrinsics": "reference"}
    file.write_text(yaml.safe_dump(document))
    base = ["prepare", "tinyclutter", "--config", config["views"], "--workers", "1"]
    assert main([*base, "--labels"]) == 0
    capsys.readouterr()
    assert main([*base, "--view-grasps"]) == 1
    assert "use `intrinsics: frame`" in capsys.readouterr().err


def test_prepare_builds_the_scale_statistic_of_scale_balanced_grasp(
    tree,  # noqa: F811
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # on the CPU, as in CI
    statistic = {
        "source": "native",
        "bins": 3,
        "frame_step": 1,
        "num_seeds": 16,
        "min_graspable_views": 0,
        "graspable_score": 0.1,
        "max_width": 0.14,  # the gripper of this benchmark
        "seed": 0,
    }
    recipe = [
        *SAMPLE,
        {"type": "FilterObjects", "min_visible_points": 1},
        DENSE[-1],
    ]
    node = {
        "type": "TinyClutter",
        "camera": "realsense-d435",
        "split": "train",
        "root": str(tree["root"]),
        "preprocess": PREPROCESS,
        "transforms": recipe,
        "scale_statistic": statistic,
    }
    file = tmp_path / "scale.yaml"
    file.write_text(yaml.safe_dump({"data": {"train": {"dataset": node}}}))
    base = ["prepare", "tinyclutter", "--config", str(file)]
    assert main([*base, "--labels"]) == 0
    with pytest.raises(AssetError, match="grasppanda prepare tinyclutter --scale-statistic"):
        build(tree, scale_statistic=statistic, transforms=recipe)
    capsys.readouterr()
    assert main([*base, "--scale-statistic"]) == 0
    assert "graspable seeds of 4 frames in 3 bins" in capsys.readouterr().out
    dataset = build(tree, scale_statistic=statistic, transforms=recipe)
    assert dataset.asset_records()[-1]["kind"] == "scale_statistic"
    edges, counts = dataset.scale_statistic.load()
    assert len(edges) == 4 and counts.sum() > 0 and edges[-1] <= 0.14
    (histogram,) = dataset[0].supervision.extras["scale_histogram"]
    assert torch.equal(histogram["counts"], torch.from_numpy(counts))


# --------------------------------------------------------------------------- subsets


def write_splits(root, train, test) -> None:
    (root / "split_info").mkdir(parents=True)
    for name, scenes in (("train", train), ("test", test)):
        (root / "split_info" / f"grasp_{name}_scene_ids.json").write_text(
            json.dumps([f"{scene:06d}" for scene in scenes])
        )


def shipped_subsets() -> dict:
    file = configs_root().parent / "grasppanda" / "datasets" / "splits" / "graspclutter6d.yaml"
    return yaml.safe_load(file.read_text())["subsets"]


def test_shipped_subsets_are_fixed_scene_lists_inside_the_official_splits(tmp_path):
    subsets = shipped_subsets()
    mini, validation = subsets["test_mini"]["scenes"], subsets["val_dev"]["scenes"]
    assert len(mini) == len(set(mini)) == 24 and len(validation) == len(set(validation)) == 41
    assert not set(mini) & set(validation)
    assert subsets["test_mini"]["of"] == "test" and subsets["val_dev"]["of"] == "train"
    assert subsets["train_dev"] == {"of": "train", "exclude": "val_dev"}

    train, test = sorted({*validation, 5, 9, 13}), sorted({*mini, 3, 6})
    write_splits(tmp_path / "full", train, test)
    manifest = GraspClutter6D.load_manifest(tmp_path / "full")
    assert manifest.split("test_mini").scenes == tuple(mini)
    assert manifest.split("val_dev").scenes == tuple(validation)
    assert manifest.split("train_dev").scenes == (5, 9, 13)  # train minus val_dev
    assert len(manifest.keys("test_mini", "realsense-d435")) == 24 * 13
    assert manifest.split("sample").scenes == (5,)
    assert manifest.all_scenes() == tuple(sorted({*train, *test}))


def test_a_tree_with_other_split_files_does_not_offer_the_subsets(tmp_path):
    subsets = shipped_subsets()
    # one scene of test_mini has moved to the training split: the subset no longer exists
    mini = subsets["test_mini"]["scenes"]
    write_splits(tmp_path / "moved", [5, mini[0]], mini[1:])
    manifest = GraspClutter6D.load_manifest(tmp_path / "moved")
    assert set(manifest.splits) == {"train", "test", "all", "sample"}
    with pytest.raises(ValueError, match="'test_mini' is not a split of graspclutter6d"):
        manifest.split("test_mini")
    # val_dev present, so train_dev is too; test_mini still absent
    write_splits(tmp_path / "partial", [*subsets["val_dev"]["scenes"], 1], [2])
    assert set(GraspClutter6D.load_manifest(tmp_path / "partial").splits) == {
        "train", "test", "all", "sample", "val_dev", "train_dev",
    }  # fmt: skip


def test_a_reader_under_another_name_has_no_shipped_subsets(tree):  # noqa: F811
    assert set(build(tree, load_labels=False).manifest.splits) == {"train", "test", "all", "sample"}


# --------------------------------------------------------------------------- configs

TRANSFER = {
    "gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e": "gsnet_sunet14_g1b-realsense_10e",
    "economicgrasp/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e": (
        "economicgrasp_sunet14_g1b-realsense_10e"
    ),
    "economicgrasp/economicgrasp_sunet14_g1b-kinect-to-gc6d-d435_10e": (
        "economicgrasp_sunet14_g1b-kinect_10e"
    ),
    "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e": (
        "graspnet-baseline_pointnet2_g1b-realsense_18e"
    ),
    "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e": (
        "sbg_pointnet2-transformer_g1b-realsense_18e"
    ),
    "hggd/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e": "hggd_r34w8_g1b-realsense_15e",
}


@pytest.mark.parametrize("name", list(TRANSFER))
def test_transfer_config_changes_the_test_frames_and_the_evaluator_only(name):
    """Whatever the weights depend on is inherited from the GraspNet-1Billion config."""
    transfer = Config.load(name, command="test")
    source = Config.load(f"{name.split('/')[0]}/{TRANSFER[name]}", command="test")
    for section in ("model", "optimizer", "scheduler", "train"):
        assert transfer.select(section) == source.select(section), section
    runtime, inherited = dict(transfer.select("runtime")), dict(source.select("runtime"))
    assert runtime.pop("work_dir") != inherited.pop("work_dir")  # named after the config
    assert runtime == inherited
    for section in ("data.train", "data.val", "evaluation.variants", "evaluation.split"):
        assert transfer.select(section) == source.select(section), section
    assert transfer.select("evaluation.evaluator")["type"] == "GraspClutter6DEvaluator"
    assert source.select("evaluation.evaluator")["type"] == "GraspNet1BEvaluator"
    # a test config: its evaluator cannot score the inherited GraspNet-1Billion val loader
    assert transfer.select("evaluation.interval") == 0 != source.select("evaluation.interval")

    ours, theirs = transfer.select("data.test.dataset"), source.select("data.test.dataset")
    assert ours["type"] == "GraspClutter6D" and theirs["type"] == "GraspNet1B"
    assert (ours["camera"], ours["split"]) == ("realsense-d435", "test")
    assert ours["load_labels"] is False and ours["seed"] == theirs["seed"] == 0
    # the cloud the network sees: the same number of points, features and images
    for key in ("transforms", "keep_full_points", "images"):
        assert ours.get(key) == theirs.get(key), key
    assert ours["preprocess"][0] == theirs["preprocess"][0] == {"type": "DepthToCloud"}
    crop = ours["preprocess"][1]
    if theirs["preprocess"][1]["type"] == "NoWorkspace":  # HGGD: no crop in training, none here
        assert crop == {"type": "NoWorkspace"}
    else:  # the official crop of each benchmark; both are derived from ground truth
        assert theirs["preprocess"][1]["type"] == "SegmentationBoxWorkspace"
        assert crop == {"type": "SegmentationRectWorkspace", "margin": 0.1}
    for key in ("batch_size", "shuffle"):
        assert transfer.select(f"data.test.{key}") == source.select(f"data.test.{key}")


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("gsnet/gsnet_sunet14_gc6d-d435_10e", "gsnet/gsnet_sunet14_g1b-realsense_10e"),
        (
            "economicgrasp/economicgrasp_sunet14_gc6d-d435_10e",
            "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e",
        ),
    ],
)
def test_training_config_differs_from_the_graspnet_recipe_where_the_dataset_does(name, source):
    ours, theirs = Config.load(name, command="train"), Config.load(source, command="train")
    for section in ("optimizer", "scheduler", "train", "evaluation.variants"):
        assert ours.select(section) == theirs.select(section), section
    model, reference = dict(ours.select("model")), dict(theirs.select("model"))
    gripper = model.pop("gripper")
    assert gripper["max_width"] == 0.14 and reference.pop("gripper")["max_width"] == 0.1
    assert gripper["name"] == "graspclutter6d-parallel-jaw"
    assert model == reference  # nothing else of the model changes
    assert ours.select("evaluation.evaluator")["type"] == "GraspClutter6DEvaluator"
    for loader, split in (("train", "train"), ("val", "val_dev"), ("test", "test")):
        node, other = ours.select(f"data.{loader}.dataset"), theirs.select(f"data.{loader}.dataset")
        assert node["type"] == "GraspClutter6D" and node["split"] == split
        assert node["transforms"] == other["transforms"]  # points, labels, augmentation
        assert node["preprocess"][1] == {"type": "SegmentationRectWorkspace", "margin": 0.1}
    train, other = ours.select("data.train.dataset"), theirs.select("data.train.dataset")
    assert train["graspness"] == {**other["graspness"], "poses": "scene_gt"}
    if train.get("compact_labels"):
        assert train["compact_labels"] == {**other["compact_labels"], "max_width": 0.14}
        assert train["compact_labels"]["max_width"] == gripper["max_width"]


def test_method_dataset_bases_mirror_the_graspnet_ones():
    for suffix in ("_tolerance", "_hggd"):
        ours = Config.load(f"_base_/datasets/graspclutter6d_d435{suffix}.yaml")
        theirs = Config.load(f"_base_/datasets/graspnet1b_realsense{suffix}.yaml")
        for loader in ("train", "val", "test"):
            node = dict(ours.select(f"data.{loader}.dataset"))
            other = dict(theirs.select(f"data.{loader}.dataset"))
            assert node.pop("type") == "GraspClutter6D" and node["camera"] == "realsense-d435"
            for key in ("load_labels", "images", "object_labels", "keep_full_points", "tolerance"):
                assert node.get(key) == other.get(key), (suffix, loader, key)
            assert ours.select(f"data.{loader}.batch_size") == theirs.select(
                f"data.{loader}.batch_size"
            )
        train = ours.select("data.train.dataset")
        reference = theirs.select("data.train.dataset")
        if suffix == "_hggd":
            assert train["transforms"] == reference["transforms"]
            assert train["preprocess"] == reference["preprocess"]  # no crop on either dataset
            assert train["view_grasps"] == {
                **reference["view_grasps"], "poses": "scene_gt", "intrinsics": "frame",
            }  # fmt: skip
        else:
            # the one recipe line that is specific to GraspNet-1Billion is not carried over
            assert reference["transforms"][1]["exclude_object_ids"] == [18]
            assert train["transforms"][1]["exclude_object_ids"] is None
            assert train["transforms"][1]["min_visible_points"] == 50
            assert train["transforms"][2:] == reference["transforms"][2:]
            assert train["transforms"][0] == reference["transforms"][0]
