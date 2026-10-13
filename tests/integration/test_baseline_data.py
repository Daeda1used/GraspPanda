"""The data side of GraspNet-baseline on the real dataset: tolerance labels against the
reference generator, and the shipped training recipe feeding the model.

``tolerance.npz`` is recorded by ``tools/make_baseline_fixtures.py --tolerance-objects``:
the reference generator's own ``worker`` function on a few annotated points of a few
objects (see ``tests/integration/test_baseline_parity.py`` for the command).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.config import Config
from grasppanda.datasets.assets import AssetError
from grasppanda.datasets.tolerance import ToleranceConfig, object_tolerance
from grasppanda.registry import DATASETS, DETECTORS
from grasppanda.structures import SampleKey

RUNNABLE = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Path:
    directory = asset_cache / "fixtures" / "graspnet_baseline"
    if not (directory / "manifest.json").is_file():
        pytest.skip(f"no GraspNet-baseline reference fixtures at {directory}")
    return directory


@pytest.fixture(scope="module")
def train_node(graspnet1b_root) -> dict:
    return dict(Config.load(RUNNABLE).resolved()["data"]["train"]["dataset"])


@pytest.mark.dataset
@pytest.mark.oracle
def test_tolerance_equals_the_reference_generator_on_sample_points(
    fixture_dir, graspnet1b_root, train_node
):
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    if "tolerance" not in manifest:
        pytest.skip("the fixtures were recorded without --tolerance-objects; regenerate them")
    config = ToleranceConfig(**train_node["tolerance"])
    assert config.max_friction == manifest["tolerance"]["mu_thresh"]
    assert config.min_good_ratio == manifest["tolerance"]["pos_ratio_thresh"]
    record = np.load(fixture_dir / manifest["tolerance"]["file"])
    assert np.array_equal(config.radii, record["radii"].astype(np.float32))
    compared = 0
    for object_id in record["objects"].tolist():
        with np.load(graspnet1b_root / "grasp_label" / f"{object_id:03d}_labels.npz") as label:
            points, scores = label["points"], label["scores"]
        rows, reference = record[f"rows_{object_id:03d}"], record[f"tolerance_{object_id:03d}"]
        levels = object_tolerance(points, scores, config, rows=rows)
        assert np.array_equal(config.radii[levels], reference), object_id  # value for value
        assert np.count_nonzero(reference) > 0
        compared += reference.size
    print(f"\ntolerance: {compared} grasps of {len(record['objects'])} objects identical")


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.slow
def test_the_shipped_recipe_feeds_the_model(graspnet1b_root, train_node, fixture_dir, cuda):
    """One real training batch through the shipped pipeline and ``loss()``.

    Sampling is random, so nothing is compared element by element; the batch must carry
    what the recipe promises, and with the authors' weights the losses must be those of
    a trained model (the reference's own batch of other frames of these scenes gives
    0.12 / 0.04 / 0.02 / 0.29 for objectness / view / score / angle).
    """
    try:
        dataset = DATASETS.build({**train_node, "seed": 0})
    except AssetError as error:
        pytest.skip(f"derived assets are not built: {str(error).splitlines()[0]}")
    keys = [
        SampleKey("graspnet1b", 5, "realsense", 17),
        SampleKey("graspnet1b", 63, "realsense", 100),
    ]
    items = [dataset[dataset.keys.index(key)] for key in keys]
    for item, key in zip(items, keys):
        (objects,) = item.supervision.objects
        assert len(item.observation.points) == 20000 and item.observation.point_feats is None
        assert set(item.supervision.point_labels) == {"objectness", "instance"}
        assert 3 <= len(objects) <= 10 and 18 not in {obj.object_id for obj in objects}
        visible = item.supervision.point_labels["instance"] > 0
        cloud = item.observation.points.coords
        for obj in objects:
            # the shipped recipe leaves the grids in the store; `dense()` reads them all
            shapes = obj.shapes()
            assert obj.deferred is not None and set(shapes) == {"scores", "widths", "tolerance"}
            assert shapes["tolerance"] == shapes["scores"] == (len(obj.points), 300, 12, 4)
            grids = obj.dense()
            assert grids.collision is None and float(grids.tolerance.max()) <= 0.05 + 1e-9
            assert grids.tolerance.shape == grids.scores.shape == shapes["scores"]
            assert (
                0
                < len(obj.points)
                <= max(300, dataset.label_store.object_arrays(obj.object_id)[0].shape[0] // 4)
            )
            # every label point lies within 1 cm of a visible point of some object
            placed = obj.points @ obj.pose[:3, :3].T + obj.pose[:3, 3]
            distance = torch.cdist(placed.double(), cloud[visible].double()).amin(dim=1)
            assert float(distance.max()) < 0.01 + 1e-6, (key, obj.object_id)
    batch = dataset.collate(items).to(cuda)
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    state = _converted(manifest)
    model = DETECTORS.build(Config.load(RUNNABLE).resolved()["model"])
    model.load_state_dict(state, strict=True)
    model = model.to(cuda).train()
    losses = model.loss(batch.observation, batch.supervision)
    values = {key: float(value) for key, value in losses.items()}
    print(
        "\nlosses of a batch from the shipped recipe:", {k: round(v, 4) for k, v in values.items()}
    )
    assert all(np.isfinite(v) for v in values.values())
    assert values["loss/objectness"] < 0.4 and values["loss/view"] < 0.15
    assert values["loss/score"] < 0.08 and values["loss/angle"] < 0.5
    assert values["loss/width"] < 0.01 and values["loss/tolerance"] < 0.02
    assert values["metric/objectness_accuracy"] > 0.85
    sum(v for k, v in losses.items() if k.startswith("loss/")).backward()


def _converted(manifest: dict) -> dict[str, torch.Tensor]:
    """The authors' checkpoint converted for the current model. Never cached: a converted
    state dict is only valid for the module layout it was converted for."""
    import importlib.util

    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the authors' checkpoint is missing: {checkpoint}")
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", repo / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state, _ = module.convert_file(
        checkpoint, camera=manifest["camera"], source_format="graspnet-baseline"
    )
    return state
