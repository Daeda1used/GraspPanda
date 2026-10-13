"""The inert unpickler and the per-object evaluation cache, on a synthetic dataset."""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pytest

from grasppanda.evaluation.graspnet1b import cache as cache_module
from grasppanda.evaluation.graspnet1b.cache import CacheError, ObjectCache
from grasppanda.evaluation.graspnet1b.dexnet_pickle import (
    InertUnpickler,
    file_sha256,
    read_dexnet_sdf,
)
from grasppanda.evaluation.graspnet1b.models import voxel_downsample

from .eval_synthetic import (
    dexnet_pickle_bytes,
    make_grid,
    sphere_asset,
    sphere_distance,
    stand_in_modules,
    upstream_stand_ins,
    write_dataset,
)

VOXEL = 0.008


def tree(root) -> dict[str, tuple[int, int]]:
    """Every file below ``root`` with size and modification time."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "data")


@pytest.fixture
def store(dataset, tmp_path) -> ObjectCache:
    return ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "cache")


# ------------------------------------------------------------------- unpickler


def test_pickle_is_read_without_the_upstream_modules(tmp_path):
    grid = make_grid(sphere_distance)
    path = tmp_path / "000.pkl"
    path.write_bytes(dexnet_pickle_bytes(grid))
    assert not any(name.split(".")[0] in ("graspnetAPI", "autolab_core") for name in sys.modules)
    with pytest.raises(ModuleNotFoundError):  # the standard unpickler would need them
        pickle.loads(path.read_bytes())

    extracted = read_dexnet_sdf(path)
    assert not any(name.split(".")[0] in ("graspnetAPI", "autolab_core") for name in sys.modules)
    assert np.array_equal(extracted.sdf, grid.values)
    assert np.array_equal(extracted.surface_sample, grid.surface_sample)
    assert extracted.resolution == grid.grid_to_object_scale
    assert extracted.surface_threshold == grid.surface_threshold
    assert extracted.grid_to_object[0] == grid.grid_to_object_scale
    assert np.array_equal(extracted.grid_to_object[1], grid.grid_to_object_translation)
    assert extracted.object_to_grid[0] == grid.object_to_grid_scale
    assert np.array_equal(extracted.object_to_grid[1], grid.object_to_grid_translation)
    assert (extracted.sha256, extracted.size) == file_sha256(path)


def test_pickled_classes_never_execute(tmp_path):
    """State hooks of pickled classes are not called: the classes are never looked up."""
    calls = []
    with upstream_stand_ins() as classes:
        probe = classes["Mesh3D"]
        probe.__setstate__ = lambda self, state: calls.append("setstate")
        probe.__reduce__ = lambda self: (type(self), (), {"vertices_": 1})
        payload = pickle.dumps(probe(), protocol=4)
        pickle.loads(payload)
        assert calls == ["setstate"]  # the ordinary unpickler does run it
    calls.clear()
    path = tmp_path / "probe.pkl"
    path.write_bytes(payload)
    with path.open("rb") as stream:
        loaded = InertUnpickler(stream).load()
    assert calls == [] and vars(loaded) == {"vertices_": 1}
    assert type(loaded).__name__ == "Mesh3D" and type(loaded) is not probe


@pytest.mark.parametrize(
    "payload",
    [
        pickle.dumps(os.getcwd, protocol=4),  # posix.getcwd
        b"cos\nsystem\n(S'true'\ntR.",  # os.system('true')
        b"cbuiltins\neval\n(S'1+1'\ntR.",
        pickle.dumps(np.random.default_rng(0)),  # a NumPy global outside the short allow-list
    ],
)
def test_foreign_globals_are_refused(tmp_path, payload):
    path = tmp_path / "evil.pkl"
    path.write_bytes(payload)
    with path.open("rb") as stream, pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        InertUnpickler(stream).load()


def test_getattr_calls_are_neutralised(tmp_path):
    payload = b"cbuiltins\ngetattr\n(cnumpy\nndarray\nS'__name__'\ntR."
    assert pickle.loads(payload) == "ndarray"
    path = tmp_path / "getattr.pkl"
    path.write_bytes(payload)
    with path.open("rb") as stream:
        assert InertUnpickler(stream).load() is None


def test_array_subclasses_come_back_as_plain_arrays(tmp_path):
    with stand_in_modules({"trimesh.caching": ()}):
        module = sys.modules["trimesh.caching"]
        module.TrackedArray = type("TrackedArray", (np.ndarray,), {"__module__": module.__name__})
        payload = pickle.dumps(np.arange(6.0).reshape(2, 3).view(module.TrackedArray), protocol=4)
    path = tmp_path / "tracked.pkl"
    path.write_bytes(payload)
    with path.open("rb") as stream:
        loaded = InertUnpickler(stream).load()
    assert type(loaded) is np.ndarray and np.array_equal(loaded, np.arange(6.0).reshape(2, 3))


def test_extraction_validates_what_it_finds(tmp_path):
    grid = make_grid(sphere_distance)
    path = tmp_path / "bad.pkl"
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    path.write_bytes(dexnet_pickle_bytes(grid, rotation=turned))
    with pytest.raises(ValueError, match="non-identity rotation"):
        read_dexnet_sdf(path)
    with upstream_stand_ins() as classes:
        path.write_bytes(pickle.dumps(classes["Sdf3D"](), protocol=4))
    with pytest.raises(ValueError, match="root object should be a pickled .*GraspableObject3D"):
        read_dexnet_sdf(path)
    with upstream_stand_ins() as classes:
        graspable = classes["GraspableObject3D"]()
        graspable.sdf_ = classes["Sdf3D"]()
        path.write_bytes(pickle.dumps(graspable, protocol=4))
    with pytest.raises(ValueError, match="has no attribute 'data_'"):
        read_dexnet_sdf(path)


# ----------------------------------------------------------------------- cache


def test_cache_builds_lazily_and_serves_identical_data(dataset, store):
    before = tree(dataset)
    assert not store.is_complete(0) and not store.is_complete(1)
    asset = store.load(0)
    assert store.is_complete(0) and not store.is_complete(1)  # only what was asked for
    reference = sphere_asset(0)
    assert np.array_equal(asset.sdf.values, reference.sdf.values)
    assert np.array_equal(asset.sdf.surface_sample, reference.sdf.surface_sample)
    assert asset.sdf.surface_threshold == reference.sdf.surface_threshold
    assert asset.sdf.object_to_grid_scale == reference.sdf.object_to_grid_scale
    assert np.array_equal(
        asset.sdf.grid_to_object_translation, reference.sdf.grid_to_object_translation
    )
    vertices = reference.model_points.astype(np.float32).astype(np.float64)  # as stored in the PLY
    assert np.array_equal(asset.model_points, voxel_downsample(vertices, VOXEL))
    probe = np.random.default_rng(0).uniform(-5, 70, size=(100, 3))
    assert np.array_equal(asset.sdf.signed_distance(probe), reference.sdf.signed_distance(probe))
    assert tree(dataset) == before  # the dataset tree is untouched
    assert store.directory.name == store.hash and store.directory.parent.name == "eval"


def test_manifest_and_sidecar(dataset, store):
    store.ensure([0, 1])
    manifest = json.loads((store.directory / "manifest.json").read_text())
    assert manifest["generator"] == "graspnet1b-eval-cache"
    assert manifest["uses_ground_truth"] is True
    assert manifest["parameters"] == {"model_voxel_size": VOXEL}
    assert len(manifest["dataset"]["fingerprint"]) == 64
    sidecar = json.loads((store.directory / "objects" / "001.json").read_text())
    pickle_path = dataset / "dex_models" / "001.pkl"
    assert sidecar["sources"]["dex_model"] == {
        "path": "dex_models/001.pkl",
        "size": pickle_path.stat().st_size,
        "sha256": file_sha256(pickle_path)[0],
    }
    assert sidecar["sources"]["model"]["path"] == "models/001/nontextured.ply"
    assert sidecar["sdf_shape"] == [64, 64, 64] and sidecar["surface_sample_size"] == 1000
    leftovers = [p.name for p in (store.directory / "objects").iterdir() if ".tmp" in p.name]
    assert leftovers == []
    store.verify(0)


def test_cache_is_resumable_and_reused(dataset, store, tmp_path, monkeypatch):
    assert store.ensure([0]) == [0]
    stamp = tree(store.directory)
    monkeypatch.setattr(cache_module, "read_dexnet_sdf", lambda path: pytest.fail("rebuilt"))
    again = ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "cache")
    assert again.directory == store.directory
    assert again.ensure([0]) == [] and again.load(0).object_id == 0
    assert tree(store.directory) == stamp
    monkeypatch.undo()
    assert again.ensure([0, 1]) == [1]  # picks up where the first build stopped


def test_an_interrupted_object_is_rebuilt(dataset, store):
    store.ensure([0])
    (store.directory / "objects" / "000.json").unlink()  # data written, completion marker not
    assert not store.is_complete(0)
    assert store.ensure([0]) == [0] and store.is_complete(0)


def test_read_only_cache_reports_missing_objects(dataset, store, tmp_path):
    store.ensure([0])
    reader = ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "cache", build=False)
    assert reader.load(0).object_id == 0
    with pytest.raises(CacheError, match="object 001 is missing"):
        reader.load(1)
    with pytest.raises(CacheError, match="object 001 is missing"):
        reader.ensure([0, 1])
    with pytest.raises(CacheError, match="building is disabled"):
        ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "elsewhere", build=False)


def test_hash_follows_parameters_and_dataset(dataset, store, tmp_path):
    other_voxel = ObjectCache(dataset, model_voxel_size=0.004, root=tmp_path / "cache")
    assert other_voxel.hash != store.hash
    (dataset / "dex_models" / "001.pkl").write_bytes(b"different size")
    changed = ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "cache")
    assert changed.hash != store.hash  # a changed source file is never served stale data
    moved = write_dataset(tmp_path / "copy")  # same content at another path: same cache
    assert ObjectCache(moved, model_voxel_size=VOXEL, root=tmp_path / "cache").hash == store.hash


def test_verify_detects_changed_sources(dataset, store):
    store.ensure([0])
    path = dataset / "models" / "000" / "nontextured.ply"
    content = bytearray(path.read_bytes())
    content[-20] ^= 0xFF  # same size, different content
    path.write_bytes(bytes(content))
    with pytest.raises(CacheError, match="changed since object 000 was cached"):
        store.verify(0)
    with pytest.raises(CacheError, match="not in the cache"):
        store.verify(1)


def test_cache_refuses_locations_and_states_it_cannot_trust(dataset, store, tmp_path):
    with pytest.raises(CacheError, match="inside the dataset"):
        ObjectCache(dataset, model_voxel_size=VOXEL, root=dataset / "cache")
    with pytest.raises(CacheError, match="no object sources"):
        ObjectCache(tmp_path / "nothing", model_voxel_size=VOXEL, root=tmp_path / "cache")
    with pytest.raises(CacheError, match="source .* does not exist"):
        store.load(7)
    manifest = store.directory / "manifest.json"
    stored = json.loads(manifest.read_text())
    stored["version"] = -1
    manifest.write_text(json.dumps(stored))
    with pytest.raises(CacheError, match="describes a different cache"):
        ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "cache")


def test_loaded_objects_are_bounded(dataset, tmp_path):
    small = ObjectCache(dataset, model_voxel_size=VOXEL, root=tmp_path / "cache", max_loaded=1)
    first = small.load(0)
    assert small.load(0) is first
    small.load(1)
    assert list(small._loaded) == [1]
    assert small.load(0) is not first  # read again from disk, equal content
    assert np.array_equal(small.load(0).sdf.values, first.sdf.values)
