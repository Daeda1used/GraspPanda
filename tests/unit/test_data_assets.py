"""The derived-asset store: identity, manifest, atomic writes, resumable shards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from grasppanda.datasets.assets import Asset, AssetError, AssetSpec, atomic_path, save_array
from grasppanda.transforms.base import fingerprint
from grasppanda.transforms.image import DepthToCloud, SegmentationBoxWorkspace


def spec(margin: float = 0.02, **changes) -> AssetSpec:
    chain = [DepthToCloud(), SegmentationBoxWorkspace(margin, "include")]
    fields = {
        "dataset": "demo",
        "kind": "graspness",
        "generator": "tests.generator",
        "version": 1,
        "parameters": {"camera": "realsense", "friction_threshold": 0.8},
        "inputs": {"dataset": "demo", "grasp_labels": "abc"},
        "preprocessing": [step.config() for step in chain],
        "uses_ground_truth": True,
    }
    fields.update(changes)
    return AssetSpec(**fields)


def test_hash_covers_everything_that_decides_the_content():
    reference = spec().hash
    assert spec().hash == reference and len(reference) == 16
    variants = [
        spec(margin=0.03),
        spec(version=2),
        spec(generator="tests.other"),
        spec(parameters={"camera": "kinect", "friction_threshold": 0.8}),
        spec(inputs={"dataset": "demo", "grasp_labels": "abcd"}),
        spec(provenance="legacy-import"),
        spec(preprocessing=None),
    ]
    assert len({v.hash for v in variants} | {reference}) == len(variants) + 1
    assert spec(about={"note": "documentation only"}).hash == reference
    # key order and number spelling do not matter
    assert spec(parameters={"friction_threshold": 0.8, "camera": "realsense"}).hash == reference


def test_asset_lives_under_the_cache_root_by_dataset_kind_and_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("GRASPPANDA_CACHE", str(tmp_path / "cache"))
    asset = Asset(spec())
    assert asset.path == tmp_path / "cache" / "demo" / "graspness" / spec().hash
    assert Asset(spec(), tmp_path / "elsewhere").path.parts[-4] == "elsewhere"


def test_manifest_records_identity_and_shard_completion(tmp_path):
    asset = Asset(spec(), tmp_path)
    assert not asset.exists()
    asset.create()
    asset.mark_complete("scene_000000", frames=256)
    manifest = json.loads((asset.path / "manifest.json").read_text())
    chain = [DepthToCloud(), SegmentationBoxWorkspace(0.02, "include")]
    assert manifest["generator"] == {"name": "tests.generator", "version": 1}
    assert manifest["parameters"] == {"camera": "realsense", "friction_threshold": 0.8}
    assert manifest["inputs"] == {"dataset": "demo", "grasp_labels": "abc"}
    assert manifest["preprocessing_fingerprint"] == fingerprint(chain)
    assert manifest["preprocessing"][1]["margin"] == 0.02
    assert manifest["uses_ground_truth"] is True and manifest["provenance"] == "native"
    assert manifest["hash"] == spec().hash and manifest["format"] == 1
    assert manifest["shards"]["scene_000000"]["frames"] == 256
    assert "finished" in manifest["shards"]["scene_000000"] and "created" in manifest
    assert asset.record()["uses_ground_truth"] is True


def test_generation_resumes_shard_by_shard(tmp_path):
    asset = Asset(spec(), tmp_path)
    asset.create()
    wanted = [f"scene_{i:06d}" for i in range(4)]
    for shard in wanted[:2]:
        asset.mark_complete(shard, frames=3)
    resumed = Asset(spec(), tmp_path)  # a later process
    resumed.create()
    assert resumed.missing(wanted) == wanted[2:]
    assert resumed.is_complete(wanted[0]) and not resumed.is_complete(wanted[3])
    # two writers: a completion recorded by one is kept by the other
    asset.mark_complete(wanted[2], frames=3)
    resumed.mark_complete(wanted[3], frames=3)
    assert Asset(spec(), tmp_path).missing(wanted) == []


def test_missing_shard_is_an_error_that_names_the_command(tmp_path):
    asset = Asset(spec(), tmp_path)
    asset.create()
    asset.mark_complete("scene_000000", frames=3)
    asset.require(["scene_000000"], build="grasppanda prepare demo --graspness")
    with pytest.raises(AssetError) as error:
        asset.require(
            [f"scene_{i:06d}" for i in range(8)], build="grasppanda prepare demo --graspness"
        )
    message = str(error.value)
    assert "lacks 7 of the 8 shards" in message and "scene_000001" in message
    assert "grasppanda prepare demo --graspness" in message


def test_missing_asset_says_how_an_existing_one_differs(tmp_path):
    built = Asset(spec(margin=0.03), tmp_path)
    built.create()
    with pytest.raises(AssetError) as error:
        Asset(spec(), tmp_path).require(["scene_000000"], build="grasppanda prepare demo")
    message = str(error.value)
    assert "No graspness asset of demo for this configuration" in message
    assert built.spec.hash in message
    assert "preprocessing[1].margin: 0.03 vs 0.02" in message
    assert "Build it with: grasppanda prepare demo" in message


def test_a_tampered_manifest_is_refused(tmp_path):
    asset = Asset(spec(), tmp_path)
    asset.create()
    file = asset.path / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["parameters"]["friction_threshold"] = 0.6
    file.write_text(json.dumps(manifest))
    with pytest.raises(AssetError, match=r"parameters.friction_threshold: 0.6 vs 0.8"):
        Asset(spec(), tmp_path).manifest()
    manifest["format"] = 99
    file.write_text(json.dumps(manifest))
    with pytest.raises(AssetError, match="asset format 99"):
        Asset(spec(), tmp_path).manifest()


def test_writes_are_atomic(tmp_path):
    target = tmp_path / "deep" / "array.npy"
    with pytest.raises(RuntimeError), atomic_path(target) as temporary:
        temporary.write_bytes(b"half a file")
        raise RuntimeError("killed")
    assert not target.exists() and list(target.parent.iterdir()) == []
    save_array(target, np.arange(6).reshape(2, 3))
    assert [p.name for p in target.parent.iterdir()] == ["array.npy"]
    assert np.load(target, mmap_mode="r").tolist() == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(RuntimeError), atomic_path(target) as temporary:
        temporary.write_bytes(b"garbage")
        raise RuntimeError("killed")
    assert np.load(target).sum() == 15  # the previous content survives a failed rewrite
