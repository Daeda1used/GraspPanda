"""The data side of Scale-Balanced-Grasp on the real dataset: the scale statistic against
the reference's file, and the shipped training recipe feeding the model.

The reference ships ``statistic/scale_distribution.npy`` without the script that made
it. Two things can therefore be tested, and they are different claims:

* the file is *read* exactly: the restricted reader returns what the reference's own
  loss module loaded (recorded in ``train_batch.npz`` by ``tools/make_sbg_fixtures.py``);
* the native generator produces a histogram of the same *shape* on a sample of training
  frames. The counts of the file are not reproduced: which frames it was accumulated
  over is unknown.
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
from grasppanda.datasets.scale_statistic import ScaleStatisticConfig, ScaleStatisticStore
from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY, collect_seed_scales
from grasppanda.registry import DATASETS, DETECTORS
from grasppanda.structures import SampleKey

RUNNABLE = "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e"
#: Frames of the shape comparison: every 640th training frame, 40 in all.
SAMPLE_STEP = 640


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Path:
    directory = asset_cache / "fixtures" / "scale_balanced_grasp"
    if not (directory / "manifest.json").is_file():
        pytest.skip(f"no Scale-Balanced-Grasp reference fixtures at {directory}")
    return directory


@pytest.fixture(scope="module")
def manifest(fixture_dir) -> dict:
    return json.loads((fixture_dir / "manifest.json").read_text())


@pytest.fixture(scope="module")
def reference_statistic(fixture_dir, manifest, tmp_path_factory):
    """The reference's file (the recorder's copy of it) imported into a temporary asset root."""
    if "scale_statistic_file" not in manifest:
        pytest.skip("the fixtures hold no copy of the reference's scale statistic; regenerate them")
    path = fixture_dir / manifest["scale_statistic_file"]
    if not path.is_file():
        pytest.skip(f"the reference's scale statistic is not at {path}")
    resolved = Config.load(RUNNABLE).resolved()["data"]["train"]["dataset"]["scale_statistic"]
    config = ScaleStatisticConfig(
        source="legacy_import",
        bins=resolved["bins"],
        reference_sha256=manifest["scale_statistic_sha256"],
    )
    store = ScaleStatisticStore(
        "graspnet1b", config, {"dataset": "graspnet1b"}, root=tmp_path_factory.mktemp("assets")
    )
    store.import_reference(path)
    return store.load()


@pytest.fixture(scope="module")
def train_dataset(graspnet1b_root):
    node = dict(Config.load(RUNNABLE).resolved()["data"]["train"]["dataset"])
    node.update(scale_statistic=None, seed=0)  # the statistic itself is what is examined here
    try:
        return DATASETS.build(node)
    except AssetError as error:
        pytest.skip(f"derived assets are not built: {str(error).splitlines()[0]}")


@pytest.mark.oracle
def test_the_reference_file_is_read_exactly(reference_statistic, fixture_dir, manifest):
    edges, counts = reference_statistic
    record = np.load(fixture_dir / manifest["training"]["file"])
    assert np.array_equal(edges, record["scale_intervals"])
    assert np.array_equal(counts, record["scale_counts"])
    assert len(counts) == 32 and int(counts.sum()) == 683_218


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.oracle
@pytest.mark.slow
def test_the_native_statistic_has_the_shape_of_the_reference_file(
    reference_statistic, train_dataset, cuda
):
    """Best-grasp widths of graspable seeds on 40 training frames, binned with the
    reference's edges, against the reference's counts as fractions."""
    edges, counts = reference_statistic
    config = Config.load(RUNNABLE).resolved()["data"]["train"]["dataset"]["scale_statistic"]
    widths = collect_seed_scales(
        train_dataset,
        frame_step=SAMPLE_STEP,
        num_seeds=config["num_seeds"],
        min_graspable_views=config["min_graspable_views"],
        graspable_score=config["graspable_score"],
        max_width=config["max_width"],
        device=cuda,
    ).numpy()
    frames = len(range(0, len(train_dataset), SAMPLE_STEP))
    outside = int((widths < edges[0]).sum() + (widths > edges[-1]).sum())
    mine = np.histogram(widths, bins=edges)[0] / len(widths)
    reference = counts / counts.sum()
    correlation = float(np.corrcoef(mine, reference)[0, 1])
    worst = float(np.abs(mine - reference).max())
    print(
        f"\nscale statistic on {frames} frames: {len(widths)} graspable seeds "
        f"({len(widths) / frames:.0f} per frame), widths {widths.min():.4f} .. {widths.max():.4f}, "
        f"{outside} outside the reference's range; correlation with the reference's bin "
        f"fractions {correlation:.4f}, largest difference of a bin fraction {worst:.4f}"
    )
    assert frames == 40 and len(widths) > 4000
    assert outside <= 0.002 * len(widths)
    assert correlation > 0.985 and worst < 0.012


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.slow
def test_the_shipped_recipe_feeds_the_model(reference_statistic, train_dataset, manifest, cuda):
    """One real training batch through the shipped pipeline and ``loss()``, with the
    authors' weights and the reference's histogram: the losses must be those of a
    trained model (the reference's own batch of these two frames, sampled differently,
    gives 0.13 / 0.08 / 0.012 / 0.24 for graspable / view / score / angle)."""
    edges, counts = reference_statistic
    keys = [
        SampleKey("graspnet1b", 5, "realsense", 17),
        SampleKey("graspnet1b", 63, "realsense", 100),
    ]
    items = [train_dataset[train_dataset.keys.index(key)] for key in keys]
    batch = train_dataset.collate(items)
    histogram = {"edges": torch.from_numpy(edges), "counts": torch.from_numpy(counts)}
    batch.supervision.extras[HISTOGRAM_KEY] = [histogram] * len(items)
    batch = batch.to(cuda)
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the authors' checkpoint is missing: {checkpoint}")
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint",
        Path(__file__).resolve().parents[2] / "tools" / "convert_checkpoint.py",
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    state, _ = converter.convert_file(
        checkpoint, camera=manifest["camera"], source_format="scale-balanced-grasp"
    )
    model = DETECTORS.build(Config.load(RUNNABLE).resolved()["model"])  # as the engine builds it
    model.load_state_dict(state, strict=True)
    model = model.to(cuda).train()
    losses = model.loss(batch.observation, batch.supervision)
    values = {key: float(value) for key, value in losses.items()}
    print(
        "\nlosses of a batch from the shipped recipe:", {k: round(v, 4) for k, v in values.items()}
    )
    assert all(np.isfinite(v) for v in values.values())
    assert values["loss/objectness"] < 0.4 and values["loss/view"] < 0.25
    assert values["loss/score"] < 0.05 and values["loss/angle"] < 0.5
    assert values["loss/width"] < 0.01 and values["loss/tolerance"] < 0.02
    assert values["metric/objectness_accuracy"] > 0.85
    sum(v for k, v in losses.items() if k.startswith("loss/")).backward()
