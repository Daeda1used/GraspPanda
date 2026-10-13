"""Split manifests."""

from __future__ import annotations

import pytest

from grasppanda.datasets.manifest import SplitManifest, expand_ranges
from grasppanda.structures import SampleKey


def test_expand_ranges():
    assert expand_ranges(7) == (7,)
    assert expand_ranges("3-6") == (3, 4, 5, 6)
    assert expand_ranges("0-255:64") == (0, 64, 128, 192)
    assert expand_ranges(["8-9", 2, "4"]) == (2, 4, 8, 9)
    for bad in ("9-3", "a-b", "0-10:0", True, 1.5, ["1-3", 2]):
        with pytest.raises(ValueError):
            expand_ranges(bad)


def test_graspnet1b_official_splits():
    manifest = SplitManifest.load("graspnet1b")
    assert manifest.cameras == ("realsense", "kinect") and manifest.frames == tuple(range(256))
    expected = {
        "train": range(0, 100),
        "test_seen": range(100, 130),
        "test_similar": range(130, 160),
        "test_novel": range(160, 190),
        "test": range(100, 190),
    }
    for name, scenes in expected.items():
        split = manifest.split(name)
        assert split.scenes == tuple(scenes) and split.frames == tuple(range(256))
        assert len(split) == len(scenes) * 256
    assert len(manifest.keys("train", "kinect")) == 25600
    assert manifest.all_scenes() == tuple(range(190))


def test_graspnet1b_dev_partition_is_a_fixed_scene_level_split():
    manifest = SplitManifest.load("graspnet1b")
    train, dev, val = (set(manifest.split(n).scenes) for n in ("train", "train_dev", "val_dev"))
    assert len(dev) == 90 and len(val) == 10
    assert dev | val == train and not dev & val
    # frozen: one scene from each block of ten (see the comment in the manifest)
    assert sorted(val) == [8, 11, 20, 36, 43, 54, 60, 73, 86, 93]
    assert sorted(scene // 10 for scene in val) == list(range(10))


def test_sample_split_and_frame_subsampling():
    manifest = SplitManifest.load("graspnet1b")
    sample = manifest.keys("sample", "realsense")
    assert sample == [SampleKey("graspnet1b", 0, "realsense", f) for f in range(0, 256, 32)]
    every_fourth = manifest.keys("test_seen", "kinect", frame_step=4)
    assert len(every_fourth) == 30 * 64 and every_fourth[1].frame == 4
    assert every_fourth[64] == SampleKey("graspnet1b", 101, "kinect", 0)
    assert len(manifest.keys("sample", "realsense", frame_step=3)) == 3
    with pytest.raises(ValueError, match="frame_step must be at least 1"):
        manifest.keys("train", "kinect", frame_step=0)


def test_unknown_names_are_explained():
    manifest = SplitManifest.load("graspnet1b")
    with pytest.raises(ValueError, match="'test_sen' is not a split of graspnet1b. Did you mean"):
        manifest.split("test_sen")
    with pytest.raises(ValueError, match="'zivid' is not a camera of graspnet1b"):
        manifest.keys("train", "zivid")
    with pytest.raises(FileNotFoundError, match="no split manifest is shipped for dataset 'nope'"):
        SplitManifest.load("nope")


def test_digest_identifies_split_content():
    manifest = SplitManifest.load("graspnet1b")
    digests = {name: manifest.digest(name) for name in manifest.splits}
    assert len(set(digests.values())) == len(digests)
    assert SplitManifest.load("graspnet1b").digest("train") == digests["train"]
    other = SplitManifest.build("graspnet1b", ["realsense"], range(256), {"train": range(99)})
    assert other.digest("train") != digests["train"]


def test_manifest_file_errors():
    base = {"dataset": "d", "cameras": ["c"], "frames_per_scene": 4}
    with pytest.raises(ValueError, match="unknown keys"):
        SplitManifest.from_mapping({**base, "splits": {"a": {"scenes": "0-1", "frame": [0]}}})
    with pytest.raises(ValueError, match=r"frames outside 0\.\.3"):
        SplitManifest.from_mapping({**base, "splits": {"a": {"scenes": "0-1", "frames": [4]}}})
    manifest = SplitManifest.from_mapping(
        {**base, "splits": {"a": {"scenes": "0-5", "exclude": ["2-3"], "frames": "0-3:2"}}}
    )
    assert manifest.split("a").scenes == (0, 1, 4, 5) and manifest.split("a").frames == (0, 2)
