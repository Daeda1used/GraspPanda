"""FineGrasp against the authors' implementation, stage by stage, with the authors' weights.

The reference values were recorded by ``tools/make_finegrasp_fixtures.py`` from release
0.4.0 of ``robo_orchard_lab`` (MinkowskiEngine backbone; its FineGrasp sources are those
of ``HorizonRobotics/robo_orchard_lab`` at ``221ec0e``) in an environment of its own.
Here the released ``model.safetensors`` is converted with ``tools/convert_checkpoint.py``,
loaded into the GraspPanda model (spconv backbone) and fed the recorded inputs.

Regenerate the fixtures (about 1 GB, kept under the cache root, never committed) as the
docstring of ``tools/make_finegrasp_fixtures.py`` says. What the recorder supplies
because the reference does not ship it (a knn wrapper, pointnet2, the normal images) is
listed in ``tools/oracle/finegrasp_reference.py``; the normal images are the one of the
three that carries content, and ``test_normal_features_are_the_recorded_normal_image``
ties GraspPanda's ``DepthNormals`` to them.

Tolerances are those of the EconomicGrasp parity tests, for the same reasons.

* Discrete results must be *identical*: the voxel of every point and the point that
  lends a voxel its input feature, the candidate set, the seed indices, the chosen
  views, the matched label points and collision decisions.
* Activations must agree within ``RTOL = 2e-5`` of the largest reference magnitude of
  the stage.
* Gradients are compared at 1e-2 of a tensor's largest entry and the total gradient
  norm at 1e-4.

Replayed from the reference record instead of being drawn here: the training view of
every seed, the entries dropped by every dropout module, and which of two template
views that share a labelled view received the label (see the EconomicGrasp parity
tests; the label matching is the same code). The attention dropout inside the two
fusion layers cannot be recorded and is off on both sides for the training batch; its
place in the layer is pinned by ``tests/unit/test_finegrasp_grouper.py`` against
``nn.TransformerEncoderLayer``.

Every comparison is also written to ``parity_report.json`` next to the fixtures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from grasppanda.config import Config
from grasppanda.datasets.compact_labels import VIEW_ARRAYS, from_reference_arrays
from grasppanda.models.assigners.compact_label import CompactLabelAssigner
from grasppanda.models.backbones.sparse_unet import VoxelMap
from grasppanda.models.groupers.multi_scale import MultiScaleGroup
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.ops import sparse, voxelize
from grasppanda.registry import DATASETS, DETECTORS, POSTPROCESSORS
from grasppanda.structures import (
    COMPACT_LABELS,
    CompactGraspLabels,
    GraspGroup,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    Supervision,
)
from grasppanda.transforms.normals import DepthNormals

pytestmark = [pytest.mark.gpu, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
MODEL_CONFIG = "_base_/models/finegrasp.yaml"
RUN_CONFIG = "finegrasp/finegrasp_sunet14_g1b-realsense_10e"
RTOL = 2e-5
#: Reference: trainable parameters and state tensors of the authors' network.
REFERENCE_PARAMETERS = 19_145_441
REFERENCE_TENSORS = 388
BACKBONE_TAPS = (
    ["bn0"]
    + [name for i in range(1, 5) for name in (f"bn{i}", f"block{i}")]
    + [name for i in range(4, 8) for name in (f"bntr{i}", f"block{i + 1}")]
    + ["final"]
)
TOKENS = ("angle", "depth", "width", "score")
SCALES = 4

_report: dict[str, dict[str, float]] = {}


def close(stage: str, mine, reference, *, rtol: float = RTOL, atol: float = 0.0) -> None:
    """Assert ``|mine - reference| <= atol + rtol * max|reference|`` and log the numbers."""
    mine = np.asarray(mine.detach().cpu() if isinstance(mine, torch.Tensor) else mine, np.float64)
    reference = np.asarray(reference, np.float64)
    assert mine.shape == reference.shape, f"{stage}: shape {mine.shape} vs {reference.shape}"
    scale = float(np.abs(reference).max())
    worst = float(np.abs(mine - reference).max())
    entry = _report.setdefault(stage, {"max_abs": 0.0, "max_rel": 0.0, "scale": 0.0, "count": 0})
    entry["max_abs"] = max(entry["max_abs"], worst)
    entry["max_rel"] = max(entry["max_rel"], worst / scale if scale else 0.0)
    entry["scale"] = max(entry["scale"], scale)
    entry["count"] += 1
    assert worst <= atol + rtol * scale, (
        f"{stage}: max abs difference {worst:.3e} exceeds {atol:.1e} + {rtol:.1e} x {scale:.3e}"
    )


def count(stage: str, value: int) -> None:
    _report[stage] = {"count": int(value)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Iterator[Path]:
    directory = asset_cache / "fixtures" / "finegrasp"
    if not (directory / "manifest.json").is_file():
        pytest.skip(
            f"no FineGrasp reference fixtures at {directory}; record them with "
            "tools/make_finegrasp_fixtures.py (see its docstring)"
        )
    yield directory
    if _report:
        (directory / "parity_report.json").write_text(json.dumps(_report, indent=2) + "\n")
        print("\nFineGrasp parity (max abs difference, relative to the stage's largest value):")
        for stage, entry in _report.items():
            if "max_abs" in entry:
                print(f"  {stage:40s} {entry['max_abs']:.3e}   {entry['max_rel']:.2e}")
            else:
                print(f"  {stage:40s} {entry['count']}")


@pytest.fixture(scope="module")
def manifest(fixture_dir) -> dict:
    return json.loads((fixture_dir / "manifest.json").read_text())


@pytest.fixture(scope="module")
def converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model(manifest, converter, cuda):
    pytest.importorskip("spconv")
    weights = Path(manifest["weights"])
    if not weights.is_file():
        pytest.skip(f"the weights the fixtures were recorded with are missing: {weights}")
    assert sha256(weights) == manifest["weights_sha256"]
    state, meta = converter.convert_file(
        weights, camera=manifest["camera"], source_format="finegrasp"
    )
    assert meta["model_config"] == MODEL_CONFIG
    # Counted in the reference process: the same parameters, in 8 tensors more because the
    # stacked query / key / value projections of the two fusion layers are cut in three.
    assert manifest["reference_parameters"] == meta["parameters"] == REFERENCE_PARAMETERS
    assert manifest["reference_tensors"] == REFERENCE_TENSORS
    assert meta["tensors"] == REFERENCE_TENSORS + 8
    detector = DETECTORS.build(Config.load(MODEL_CONFIG)["model"])
    assert sum(p.numel() for p in detector.parameters()) == REFERENCE_PARAMETERS
    detector.load_state_dict(state, strict=True)
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield detector.to(cuda)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def inference_frames() -> list[str]:
    return ["infer_0100_0000", "infer_0118_0137", "infer_0145_0064", "infer_0175_0200"]


def observe(keys: list[SampleKey], clouds: np.ndarray, normals: np.ndarray, **more) -> Observation:
    """What the data pipeline hands the model: points, and [x, y, z | normal] features."""
    points = torch.from_numpy(clouds)
    feats = torch.cat([points, torch.from_numpy(normals)], dim=-1).reshape(-1, 6)
    return Observation(keys=keys, points=PointSet.from_dense(points), point_feats=feats, **more)


@pytest.fixture(scope="module", params=inference_frames())
def frame(request, fixture_dir, manifest):
    """One recorded test frame: the reference record and the observation built from it."""
    listed = {entry["file"] for entry in manifest["inference"]}
    assert f"{request.param}.npz" in listed, "fixtures are older than this test; regenerate them"
    record = dict(np.load(fixture_dir / f"{request.param}.npz"))
    _, scene, index = request.param.split("_")
    observation = observe(
        [SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))],
        record["point_clouds"],
        record["cloud_normal"],
        # float64, as the reference hands the cloud to its collision detector
        full_points=PointSet(torch.from_numpy(record["full_cloud"])),
    )
    return record, observation


def backbone_modules(backbone) -> dict[str, nn.Module]:
    """Reference tap name → the module here whose output is the same activation."""
    modules = {"bn0": backbone.stem.norm, "final": backbone.project}
    for stage in range(4):
        modules[f"bn{stage + 1}"] = backbone.down[stage].norm
        modules[f"block{stage + 1}"] = backbone.encoder[stage]
        modules[f"bntr{stage + 4}"] = backbone.up[stage].norm
        modules[f"block{stage + 5}"] = backbone.decoder[stage]
    return modules


@pytest.fixture(scope="module")
def inference(frame, model, cuda):
    """Natural and seed-replayed runs of one frame, plus the backbone activations."""
    record, observation = frame
    observation = observation.to(cuda)
    model.eval()
    taps: dict[str, object] = {}
    hooks = [
        module.register_forward_hook(
            lambda _m, _i, output, name=name: taps.__setitem__(name, output)
        )
        for name, module in backbone_modules(model.backbone).items()
    ]
    with torch.no_grad():
        natural = model.run_stages(observation)
        for hook in hooks:
            hook.remove()
        replayed = model.run_stages(
            observation, seed_index=torch.from_numpy(record["seed_index"]).to(cuda)
        )
    return record, observation, natural, replayed, taps


# -------------------------------------------------------------------------- tests


def test_the_recorded_run_is_the_shipped_configuration(model, manifest):
    """The fixtures describe what the runnable config builds and feeds."""
    run = Config.load(RUN_CONFIG)
    assert run["model"] == Config.load(MODEL_CONFIG)["model"]
    dataset = run.select("data.test.dataset")
    assert dataset["camera"] == manifest["camera"] and list(dataset["images"]) == ["depth"]
    expected = [
        {"type": "DepthNormals", "step": 1, "towards_camera": True, "encoding": "image8"},
        {"type": "RandomSample", "num_points": manifest["num_points"]},
        {"type": "CoordinateFeatures"},
    ]
    assert len(dataset["transforms"]) == len(expected)
    for step, node in zip(dataset["transforms"], expected):
        assert node.items() <= dict(step).items()
    assert manifest["normals"] == "image"
    reference = manifest["model_config"]  # of the reference process
    grouper = model.grouper
    assert isinstance(grouper, MultiScaleGroup) and len(grouper.branches) == SCALES
    assert grouper.radius == reference["cylinder_radius"]
    assert list(grouper.radius_scales) == reference["cylinder_groups"]
    assert reference["use_normal"] is True and model.backbone.in_channels == 6
    assert model.point_features == "observation"
    assert model.backbone.out_channels == reference["seed_feat_dim"]
    assert model.backbone.voxel_size == reference["voxel_size"] == manifest["voxel_size"]
    assert model.proposal.num_seeds == reference["num_seed_points"]
    assert model.proposal.graspness_threshold == reference["graspness_threshold"]
    assert model.view.template.num_views == reference["num_view"]
    assert model.head.codec.grid == (reference["num_angle"], reference["num_depth"])
    assert model.gripper.max_width == reference["grasp_max_width"]
    (cd,) = run["evaluation"]["variants"]["cd"]
    recorded = manifest["collision_filter"]
    assert cd["voxel_size"] == recorded["voxel_size"]
    assert cd["approach_distance"] == recorded["approach"]
    assert cd["collision_threshold"] == recorded["threshold"]


def test_normal_features_are_the_recorded_normal_image(frame, graspnet1b_root):
    """``DepthNormals`` computes, for every point of the frame, what the recorder stored
    in the normal image the reference dataset read."""
    record, observation = frame
    key = observation.keys[0]
    node = dict(Config.load(RUN_CONFIG).select("data.test.dataset"))
    dataset = DATASETS.build(node)
    sample = dataset.sample(key)
    assert isinstance(dataset.transforms[0], DepthNormals)
    sample = dataset.transforms[0](sample)
    image = record["normal_image"]
    assert image.dtype == np.uint8 and image.shape == (*sample.image_size, 3)
    stored = image.reshape(-1, 3)[sample.pixels]
    assert np.array_equal(np.rint(sample.feats * 255.0).astype(np.uint8), stored)
    assert np.array_equal(
        sample.feats, (stored / 255.0).astype(np.float32)
    )  # as the loader divides
    assert sample.depth is None  # consumed: later augmentations may move the cloud
    # The sampled points of the record are points of this frame, with these normals.
    rows = {point.tobytes(): row for row, point in enumerate(sample.points.astype(np.float32))}
    index = np.array([rows[point.tobytes()] for point in record["point_clouds"][0]])
    assert np.array_equal(sample.feats[index], record["cloud_normal"][0])


def test_voxelisation_is_identical_and_device_independent(frame, cuda):
    record, observation = frame
    points = observation.points
    reference = record["coors"][record["quantize2original"]]  # voxel of every point
    for device in ("cpu", cuda):
        moved = points.to(device)
        grid = voxelize(moved.coords, 0.005, moved.batch_index)
        assert np.array_equal(grid.voxels[grid.inverse].cpu().numpy(), reference), str(device)
        assert len(grid.voxels) == len(record["coors"])


def test_a_voxel_takes_the_features_of_the_point_the_reference_takes(frame, model, cuda):
    """With real input features the representative of a voxel matters (it did not while
    the features were ones): ``voxel_features: first`` is the reference's choice."""
    record, observation = frame
    assert model.backbone.voxel_features == "first"
    reference = record["voxel_feats"][record["quantize2original"]]  # per point: its voxel's
    for device in ("cpu", cuda):
        moved = observation.to(device)
        voxel_map = VoxelMap.build(moved.points, model.backbone.voxel_size)
        mine = voxel_map.reduce(moved.point_feats, "first")[voxel_map.inverse]
        assert np.array_equal(mine.cpu().numpy(), reference), str(device)
    assert record["voxel_feats"].shape[1] == 6


def test_backbone_matches_at_every_level(inference, model):
    record, observation, _, _, taps = inference
    grid = voxelize(observation.points.coords, model.backbone.voxel_size)
    stride = model.backbone.total_stride
    origin = torch.div(grid.voxels[:, 1:].amin(0), stride, rounding_mode="floor") * stride
    for name in BACKBONE_TAPS:
        reference_coords = record[f"tap_{name}_coords"].astype(np.int64)
        index = sparse.indices(taps[name]).long()
        # MinkowskiEngine coordinates of a level are multiples of its tensor stride.
        level = _tensor_stride(name)
        absolute = torch.cat([index[:, :1], index[:, 1:] * level + origin], dim=1)
        rows = {tuple(row): n for n, row in enumerate(absolute.tolist())}
        assert len(rows) == len(reference_coords), name
        order = torch.tensor(
            [rows[tuple(c)] for c in reference_coords.tolist()], device=index.device
        )
        close(f"backbone/{name}", sparse.features(taps[name])[order], record[f"tap_{name}_feats"])


def _tensor_stride(name: str) -> int:
    """Tensor stride of the activation a reference tap records."""
    strides = {"bn0": 1, "final": 1}
    for stage in range(1, 5):
        strides[f"bn{stage}"] = strides[f"block{stage}"] = 2**stage
        strides[f"bntr{stage + 3}"] = strides[f"block{stage + 4}"] = 2 ** (4 - stage)
    return strides[name]


def test_point_features_and_scores_match(inference):
    record, _, natural, _, _ = inference
    close("point_features", natural.feats.feats, record["point_feats"][0])
    close("objectness", natural.point_scores.objectness, record["objectness"][0])
    close("graspness", natural.point_scores.graspness, record["graspness"][0])


def test_candidates_and_seed_indices_are_identical(inference, model):
    record, _, natural, _, _ = inference
    candidates = torch.nonzero(model.proposal.candidates(natural.point_scores)).flatten()
    assert np.array_equal(candidates.cpu().numpy(), record["candidates_0"])
    assert int(natural.point_scores.num_candidates[0]) == len(record["candidates_0"])
    seeds = natural.seeds
    assert np.array_equal(seeds.points.parent_index.cpu().numpy(), record["seed_index"][0])
    assert bool(seeds.valid.all())
    assert np.array_equal(seeds.points.coords.cpu().numpy(), record["seed_xyz"][0])


def test_view_stage_matches_with_reference_seeds(inference, model):
    """Independent of the selection: the reference's seeds are injected."""
    record, _, _, out, _ = inference
    assert out.seeds.source == "replayed"
    close("view/scores", out.approach.logits, record["view_score"][0])
    assert np.array_equal(out.approach.view_index.cpu().numpy(), record["view_index"][0])
    before = out.feats.feats[out.seeds.points.parent_index]
    close("view/residual", torch.relu(model.view.hidden(before)), record["view_residual"][0])
    close("view/seed_features", out.seeds.feats, before.cpu().numpy() + record["view_residual"][0])
    close("view/approach", out.approach.direction, -record["view_xyz"][0], atol=1e-7)
    close("view/rotation", out.approach.rotation, record["view_rot"][0], atol=1e-6)


def test_every_scale_crops_as_the_reference_with_reference_seeds(inference, model, manifest):
    """The four branches are EconomicGrasp's grouper; each is compared like it, the
    per-neighbour activations on the seeds the recorder kept."""
    record, _, _, out, _ = inference
    stride = manifest["detail_stride"]
    support = PointFeatures(out.seeds.feats, out.seeds.points)
    centres, rotation = out.seeds.points.coords, out.approach.rotation
    assert np.array_equal(centres.cpu().numpy(), record["crop_centre"][0])
    for scale, branch in enumerate(model.grouper.branches):
        assert branch.radius == model.grouper.radius * model.grouper.radius_scales[scale]
        with torch.no_grad():
            grouped = branch.group(support, centres, rotation)
            tokens = branch.tokens(support, centres, rotation)
            mixed = branch.interaction(tokens)
            after = branch.post_mlp(mixed.flatten(0, 1)).unflatten(0, mixed.shape[:2])
            pooled = branch(support, centres, rotation).feats
        local = record[f"grouped_xyz_{scale}"][0]
        close(f"scale{scale}/local_xyz", grouped[..., :3], local, atol=1e-6)
        kept = slice(None, None, stride)
        close(f"scale{scale}/features", grouped[kept, :, 3:], record[f"grouped_feats_{scale}"][0])
        width = branch.token_channels - 3
        close(f"scale{scale}/shared_mlp", tokens[kept, :, :width], record[f"crop_mlp_{scale}"][0])
        close(f"scale{scale}/tokens", tokens[kept], record[f"crop_tokens_{scale}_in"][0])
        close(f"scale{scale}/interaction", mixed[kept], record[f"crop_tokens_{scale}_out"][0])
        close(f"scale{scale}/post_mlp", after[kept], record[f"crop_post_mlp_{scale}"][0])
        close(f"scale{scale}/pooled", pooled, record[f"crop_{scale}"][0])


def test_scale_fusion_matches_with_reference_seeds(inference, model):
    record, _, _, out, _ = inference
    grouper = model.grouper
    support = PointFeatures(out.seeds.feats, out.seeds.points)
    with torch.no_grad():
        tokens = grouper.scale_features(support, out.seeds.points.coords, out.approach.rotation)
        encoded = tokens
        for index, layer in enumerate(grouper.fusion.layers):
            encoded = layer(encoded)
            close(f"fusion/layer{index}", encoded, record[f"fusion_layer_{index}"])
        weights = grouper.fusion.weights(encoded)
    reference_tokens = np.stack([record[f"crop_{scale}"][0] for scale in range(SCALES)], axis=1)
    close("fusion/scale_tokens", tokens, reference_tokens)
    close("fusion/weights", weights, record["fusion_weights"], atol=1e-6)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5)
    close("fusion/fused", out.groups.feats, record["fused"][0])


def test_head_matches_with_reference_seeds(inference, model):
    record, _, _, out, _ = inference
    head = model.head
    with torch.no_grad():
        tokens = torch.stack([head.project[t](out.groups.feats) for t in TOKENS], dim=1)
        mixed = head.interaction(tokens)
    close("head/tokens", tokens, record["head_tokens_in"])
    close("head/interaction", mixed, record["head_tokens_out"])
    close("head/angle", out.raw.angle, record["angle_pred"][0])
    close("head/depth", out.raw.depth, record["depth_pred"][0])
    close("head/score", out.raw.score, record["score_pred"][0])
    close("head/width", out.raw.width, record["width_pred"][0][:, 0])
    assert out.raw.angle.shape[1] == 13 and out.raw.depth.shape[1] == 5  # the extra class


def test_decoded_grasps_match_end_to_end(inference, model):
    """No injection: inputs in, ``(1024, 17)`` out."""
    record, observation, natural, _, _ = inference
    (grasps,) = model.decode(natural, observation)
    assert grasps.key == observation.keys[0] and grasps.score_kind == "weighted_quality_logits"
    mine, reference = grasps.to_graspnet_array(), record["decoded"][0]
    assert mine.shape == reference.shape == (1024, 17)
    close("decoded/score", mine[:, 0], reference[:, 0])
    close("decoded/width", mine[:, 1], reference[:, 1], atol=1e-6)
    assert np.array_equal(mine[:, 2], reference[:, 2]), "height"
    assert np.array_equal(mine[:, 3], reference[:, 3]), "depth"
    close("decoded/rotation", mine[:, 4:13], reference[:, 4:13], atol=1e-6)
    assert np.array_equal(mine[:, 13:16], reference[:, 13:16]), "translation"
    assert np.array_equal(mine[:, 16], reference[:, 16]), "object id"


def test_the_extra_class_never_wins_on_the_recorded_frames(inference, model, manifest):
    """FineGrasp decodes over all logits, as EconomicGrasp did when it was forked."""
    _, observation, natural, _, _ = inference
    wins = [entry["extra_class_wins"] for entry in manifest["inference"]]
    assert all(entry == {"angle": 0, "depth": 0} for entry in wins)  # measured in the reference
    assert model.head.decode_argmax == "all_logits"
    (pinned,) = model.decode(natural, observation)
    model.head.decode_argmax = "supervised"
    try:
        (later,) = model.decode(natural, observation)
    finally:
        model.head.decode_argmax = "all_logits"
    assert np.array_equal(pinned.to_graspnet_array(), later.to_graspnet_array())


def test_collision_filter_of_the_run_config_makes_the_same_decisions(inference, model):
    record, observation, natural, _, _ = inference
    (node,) = Config.load(RUN_CONFIG)["evaluation"]["variants"]["cd"]
    collision = POSTPROCESSORS.build(node)
    assert isinstance(collision, ModelFreeCollisionFilter) and collision.voxel_size == 0.01
    scene = voxel_down_sample(observation.full_points.coords, collision.voxel_size)
    # Same point set as Open3D's voxel_down_sample (whose output order is unspecified).
    reference_scene = torch.from_numpy(record["collision_scene"]).to(scene)
    assert scene.shape == reference_scene.shape
    # The reference's own grasps: isolates the filter from the network.
    reference = GraspGroup.from_graspnet_array(record["decoded"][0]).to(scene.device)
    mask = collision.collision_mask(reference, scene)
    assert np.array_equal(mask.cpu().numpy(), record["collision_mask"])
    # And the full path: our grasps through the registered post-processor.
    (grasps,) = model.decode(natural, observation)
    kept = collision(grasps, observation)
    expected = record["decoded"][0][~record["collision_mask"]]
    assert len(kept) == len(expected)
    assert np.array_equal(kept.to_graspnet_array()[:, 13:16], expected[:, 13:16])


def test_a_batch_predicts_what_its_frames_predict_alone(fixture_dir, manifest, model, cuda):
    records = [dict(np.load(fixture_dir / f"{name}.npz")) for name in inference_frames()[:2]]
    observation = observe(
        [SampleKey("graspnet1b", 100 + i, manifest["camera"], i) for i in range(2)],
        np.concatenate([r["point_clouds"] for r in records]),
        np.concatenate([r["cloud_normal"] for r in records]),
    ).to(cuda)
    model.eval()
    grasps = model.predict(observation)
    assert [g.key for g in grasps] == observation.keys
    for group, record in zip(grasps, records):
        mine, reference = group.to_graspnet_array(), record["decoded"][0]
        assert np.array_equal(mine[:, 13:16], reference[:, 13:16])  # same seeds
        close("batched/score", mine[:, 0], reference[:, 0])
        close("batched/rotation", mine[:, 4:13], reference[:, 4:13], atol=1e-6)


# -------------------------------------------------------------------------- sweep


@pytest.mark.slow
def test_decisions_agree_on_a_sweep_of_test_frames(fixture_dir, manifest, model, cuda):
    """Seeds sit behind a threshold, grasps behind arg-maxes: how often do both sides agree?

    The rules are those of the EconomicGrasp sweep: candidate sets may differ only at
    points whose reference score is within round-off of the decision boundary; with
    equal sets the seeds must be identical; with the reference's seeds injected, an
    arg-max may fall the other way only where the reference's two best values are
    within round-off of each other, and every other grasp must match.
    """
    if "sweep" not in manifest:
        pytest.skip("the fixtures were recorded without a sweep; regenerate them")
    record = np.load(fixture_dir / manifest["sweep"]["file"])
    frames, points = record["point_clouds"].shape[:2]
    seeds_per_frame = model.proposal.num_seeds
    threshold = model.proposal.graspness_threshold
    model.eval()
    same_candidates = same_seeds = flipped_points = 0
    ties = {"view": 0, "angle": 0, "depth": 0}
    for start in range(0, frames, 8):
        rows = slice(start, min(start + 8, frames))
        observation = observe(
            [
                SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))
                for scene, index in zip(record["scene"][rows], record["frame"][rows])
            ],
            record["point_clouds"][rows],
            record["cloud_normal"][rows],
        ).to(cuda)
        offset = torch.arange(len(observation), device=cuda)[:, None] * points
        reference_seeds = torch.from_numpy(record["seed_index"][rows]).to(cuda) + offset
        with torch.no_grad():
            out = model.run_stages(observation)
            replay = model.run_stages(observation, seed_index=reference_seeds)
            replayed = model.decode(replay, observation)
        mask = model.proposal.candidates(out.point_scores).reshape(-1, points).cpu().numpy()
        index = out.seeds.points.parent_index.reshape(-1, seeds_per_frame) - offset
        seeds = index.cpu().numpy()
        choices = {
            "view": replay.approach.view_index,
            "angle": replay.raw.angle.argmax(dim=-1),
            "depth": replay.raw.depth.argmax(dim=-1),
        }
        choices = {k: v.reshape(-1, seeds_per_frame).cpu().numpy() for k, v in choices.items()}
        for local, frame in enumerate(range(rows.start, rows.stop)):
            objectness, graspness = record["objectness"][frame], record["graspness"][frame]
            differs = mask[local] != record["candidate_mask"][frame]
            if differs.any():
                # Only points on a decision boundary (to round-off) may change sides.
                margin = np.minimum(
                    np.abs(objectness[:, 1] - objectness[:, 0]), np.abs(graspness - threshold)
                )[differs]
                scale = max(np.abs(objectness).max(), np.abs(graspness).max())
                assert margin.max() <= RTOL * scale, (frame, float(margin.max()))
                flipped_points += int(differs.sum())
            else:
                same_candidates += 1
                assert np.array_equal(seeds[local], record["seed_index"][frame]), frame
            same_seeds += int(np.array_equal(seeds[local], record["seed_index"][frame]))

            agree = np.ones(seeds_per_frame, dtype=bool)
            for what in ("view", "angle", "depth"):  # in the order the decoder depends on them
                other = (choices[what][local] != record[f"{what}_choice"][frame]) & agree
                # A seed that already took another view has another crop: not comparable.
                allowed = record[f"{what}_margin"][frame] <= RTOL * record[f"{what}_scale"][frame]
                assert allowed[other].all(), (frame, what, int(other.sum()))
                ties[what] += int(other.sum())
                agree &= ~other
            mine, reference = replayed[local].to_graspnet_array(), record["decoded"][frame]
            assert np.array_equal(mine[:, 13:16], reference[:, 13:16]), frame
            assert np.array_equal(mine[agree, 3], reference[agree, 3]), frame
            close("sweep/decoded_score", mine[agree, 0], reference[agree, 0])
            close("sweep/decoded_width", mine[agree, 1], reference[agree, 1], atol=1e-6)
            close("sweep/decoded_rotation", mine[agree, 4:13], reference[agree, 4:13], atol=1e-6)
    count("sweep/frames", frames)
    count("sweep/frames_with_identical_candidates", same_candidates)
    count("sweep/frames_with_identical_seeds", same_seeds)
    count("sweep/candidate_flips", flipped_points)
    count("sweep/grasps", frames * seeds_per_frame)
    for what, number in ties.items():
        count(f"sweep/{what}_choices_decided_by_round_off", number)
    print(
        f"\nsweep: {same_seeds} of {frames} frames with identical seeds "
        f"({same_candidates} with identical candidates, {flipped_points} boundary flips); "
        f"arg-max ties decided by round-off among {frames * seeds_per_frame} grasps: {ties}"
    )


@pytest.mark.slow
def test_grasps_agree_on_the_reference_dump_of_the_benchmark_sample(asset_cache, model, cuda):
    """The frames whose AP ``configs/finegrasp/README.md`` reports: GraspPanda fed the
    clouds and normals that ``tools/oracle/finegrasp_reference.py --save-inputs`` stored
    must select the reference's seeds and decode the reference's grasps."""
    dump = asset_cache / "oracle" / "finegrasp_reference_0.4.0_realsense_step64"
    if not (dump / "manifest.json").is_file():
        pytest.skip(f"no reference dump at {dump}; see tools/oracle/finegrasp_reference.py")
    manifest = json.loads((dump / "manifest.json").read_text())
    assert manifest["normals"] == "image" and manifest["tf32"] == "off"
    model.eval()
    frames = same_seeds = other_choice = grasps = 0
    for entry in manifest["frames"]:
        key = SampleKey("graspnet1b", entry["scene"], manifest["camera"], entry["frame"])
        name = Path(f"scene_{key.scene:04d}") / key.camera / f"{key.frame:04d}"
        inputs = np.load(dump / "inputs" / name.with_suffix(".npz"))
        observation = observe([key], inputs["points"][None], inputs["normals"][None]).to(cuda)
        mine = model.predict(observation)[0].to_graspnet_array()
        reference = np.load(dump / "raw" / name.with_suffix(".npy"))
        frames += 1
        grasps += len(reference)
        assert mine.shape == reference.shape and entry["grasps"] == len(reference)
        if not np.array_equal(mine[:, 13:16], reference[:, 13:16]):
            continue
        same_seeds += 1
        # A view, angle or depth arg-max that fell the other way gives another grasp; the
        # sweep above shows that this happens only at ties. Here it is counted.
        agree = (mine[:, 3] == reference[:, 3]) & (
            np.abs(mine[:, 4:13] - reference[:, 4:13]).max(axis=1) < 1e-3
        )
        other_choice += int((~agree).sum())
        close("sample/decoded_score", mine[agree, 0], reference[agree, 0])
        close("sample/decoded_width", mine[agree, 1], reference[agree, 1], atol=1e-6)
        close("sample/decoded_rotation", mine[agree, 4:13], reference[agree, 4:13], atol=1e-6)
    count("sample/frames", frames)
    count("sample/frames_with_identical_seeds", same_seeds)
    count("sample/grasps", grasps)
    count("sample/grasps_with_another_arg_max", other_choice)
    assert same_seeds == frames
    assert other_choice <= 1e-4 * grasps


# ----------------------------------------------------------------------- training


class _ReplayDropout(nn.Module):
    """Dropout that drops exactly the entries a recorded run dropped."""

    def __init__(self, keep: np.ndarray, probability: float) -> None:
        super().__init__()
        self.keep, self.probability = torch.from_numpy(keep), probability

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        noise = self.keep.to(values).reshape(values.shape) / (1.0 - self.probability)
        return values * noise


@contextmanager
def replayed_dropout(model, record) -> Iterator[None]:
    """Replace every dropout module by a replay of the reference's draw.

    The attention dropout of the fusion layers is switched off instead: the reference
    record was made without it (see the module docstring).
    """
    slots: list[tuple[nn.Module, str, str | None]] = [(model.head.interaction, "dropout", "head")]
    for scale, branch in enumerate(model.grouper.branches):
        slots.append((branch.interaction, "dropout", f"crop_dropout_keep_{scale}"))
    for index, layer in enumerate(model.grouper.fusion.layers):
        slots.append((layer.interaction, "dropout", None))
        slots.append((layer.interaction, "residual_dropout", f"fusion_{index}_dropout1_keep"))
        slots.append((layer, "dropout", f"fusion_{index}_dropout_keep"))
        slots.append((layer, "residual_dropout", f"fusion_{index}_dropout2_keep"))
    originals = [getattr(owner, name) for owner, name, _ in slots]
    for (owner, name, key), original in zip(slots, originals):
        assert isinstance(original, nn.Dropout) and original.p > 0, (name, key)
        if key is None:
            setattr(owner, name, nn.Identity())
            continue
        key = "head_dropout_keep" if key == "head" else key
        # An entry that was zero before the dropout is recorded as kept; it is zero either way.
        setattr(owner, name, _ReplayDropout(record[key], original.p))
    try:
        yield
    finally:
        for (owner, name, _), original in zip(slots, originals):
            setattr(owner, name, original)


def compact_labels(manifest, record) -> list[list[CompactGraspLabels]]:
    """Supervision of the recorded batch from the files the reference read."""
    scenes = {}
    for scene, file in manifest["training"]["labels"].items():
        if not Path(file).is_file():
            pytest.skip(f"the compact labels the training batch was recorded with are gone: {file}")
        with np.load(file) as archive:
            scenes[int(scene)] = from_reference_arrays(archive)
    frames = []
    for frame, (scene, _) in enumerate(manifest["training"]["frames"]):
        arrays = scenes[scene]
        frames.append([])
        for position in range(int(record["objects_per_frame"][frame])):
            rows = arrays["instance"] == position
            assert int(rows.sum()) == int(record[f"label_points_{frame}"][position])
            pose = torch.eye(4)
            pose[:3] = torch.from_numpy(record[f"poses_{frame}"][position])
            frames[-1].append(
                CompactGraspLabels(
                    object_id=position,
                    pose=pose,
                    points=torch.from_numpy(arrays["points"][rows]),
                    view_max_friction=0.6,
                    **{name: torch.from_numpy(arrays[name][rows]) for name in VIEW_ARRAYS},
                )
            )
    return frames


@pytest.fixture(scope="module")
def training(fixture_dir, manifest, model, cuda):
    """The recorded training batch pushed through the stages with the reference's draws."""
    if "training" not in manifest:
        pytest.skip("the fixtures were recorded without a training batch; regenerate them")
    record = dict(np.load(fixture_dir / manifest["training"]["file"]))
    batch, points = record["point_clouds"].shape[:2]
    observation = observe(
        [
            SampleKey("graspnet1b", scene, manifest["camera"], index)
            for scene, index in manifest["training"]["frames"]
        ],
        record["point_clouds"],
        record["cloud_normal"],
    ).to(cuda)
    supervision = Supervision(
        objects=[[] for _ in range(batch)],
        point_labels={
            "objectness": torch.from_numpy(record["objectness_label"]).reshape(-1),
            "graspness": torch.from_numpy(record["graspness_label"]).reshape(-1),
        },
        extras={COMPACT_LABELS: compact_labels(manifest, record)},
    ).to(cuda)
    # A training-mode forward moves the running statistics of every batch norm; the model
    # is shared with the inference tests, so its state is put back afterwards.
    state = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()
    model.zero_grad()
    views = torch.from_numpy(record["sampled_view_index"]).reshape(-1).to(cuda)
    with replayed_dropout(model, record):
        out = model.run_stages(observation, supervision, view_index=views)
    template = model.view.template
    variants = {
        rule: CompactLabelAssigner(shared_views=rule, template=template)(
            out.seeds, out.approach, supervision
        )
        for rule in ("all", "highest")
    }
    # The reference's outcome of its undefined assignment, replayed: a seed is labelled
    # where the reference labelled it; every label value comes from the labels themselves.
    labelled = torch.from_numpy(record["label_angle"].reshape(-1) != 12).to(cuda)
    everything = variants["all"]
    replay = dataclasses.replace(
        everything,
        view_labelled=labelled,
        valid=everything.view_valid & (everything.distance < 0.005) & labelled,
    )
    losses = {
        **model.proposal.loss(out.point_scores, supervision),
        **model.view.loss(out.approach, replay),
        **model.head.loss(out.raw, replay),
    }
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
    model.zero_grad()
    model.load_state_dict(state)
    model.eval()
    return record, out, variants, replay, losses, gradients, (batch, points)


def test_the_training_pipeline_builds_the_inputs_the_reference_loader_built(
    fixture_dir, manifest, graspnet1b_root
):
    """Features of a training sample: the coordinates of the (possibly mirrored) cloud and
    the normals of the sensor frame, which the reference's flip leaves as they were."""
    if "training" not in manifest:
        pytest.skip("the fixtures were recorded without a training batch; regenerate them")
    record = np.load(fixture_dir / manifest["training"]["file"])
    node = dict(Config.load(RUN_CONFIG).select("data.train.dataset"))
    assert [step["type"] for step in node["transforms"]] == [
        "DepthNormals",
        "RandomSample",
        "RandomFlip",
        "CoordinateFeatures",
    ]
    # The inputs do not depend on the labels: read the frames without them (and their assets).
    node.update(load_labels=False, graspness=None, compact_labels=None)
    dataset = DATASETS.build(node)
    normals, _, flip, coordinates = dataset.transforms
    mirrored = manifest["training"]["mirrored_frames"]
    for row, (scene, index) in enumerate(manifest["training"]["frames"]):
        key = SampleKey("graspnet1b", scene, manifest["camera"], index)
        sample = coordinates(flip.apply(normals(dataset.sample(key)), mirrored[row]))
        rows = {point.tobytes(): n for n, point in enumerate(sample.feats[:, :3])}
        chosen = np.array([rows[point.tobytes()] for point in record["point_clouds"][row]])
        assert np.array_equal(sample.feats[chosen, :3], record["point_clouds"][row])
        assert np.array_equal(sample.feats[chosen, 3:], record["cloud_normal"][row])
        if mirrored[row]:  # the same normals as without the flip
            plain = normals(dataset.sample(key))
            assert np.array_equal(plain.feats[chosen], record["cloud_normal"][row])
    count("train/frames_built_by_the_pipeline", len(mirrored))
    count("train/of_them_mirrored", sum(mirrored))


@pytest.mark.slow
def test_training_forward_matches(training, model, manifest):
    record, out, _, _, _, _, (batch, points) = training
    seeds = model.proposal.num_seeds
    mirrored = manifest["training"]["mirrored_frames"]
    assert any(mirrored) and not all(mirrored), "the fixture is meant to cover a flip"
    assert manifest["training"]["fusion_attention_dropout"].startswith("switched off")
    close("train/point_features", out.feats.feats.reshape(batch, points, -1), record["point_feats"])
    close(
        "train/objectness",
        out.point_scores.objectness.reshape(batch, points, 2),
        record["objectness"],
    )
    close("train/graspness", out.point_scores.graspness.reshape(batch, points), record["graspness"])
    offset = torch.arange(batch, device=out.seeds.points.coords.device)[:, None] * points
    index = out.seeds.points.parent_index.reshape(batch, seeds) - offset
    assert np.array_equal(index.cpu().numpy(), record["seed_index"])  # no injection needed
    close("train/view_scores", out.approach.logits.reshape(batch, seeds, -1), record["view_score"])
    close("train/group_fused", out.groups.feats.reshape(batch, seeds, -1), record["fused"])
    close("train/head_angle", out.raw.angle.reshape(batch, seeds, -1), record["angle_pred"])
    close("train/head_depth", out.raw.depth.reshape(batch, seeds, -1), record["depth_pred"])
    close("train/head_score", out.raw.score.reshape(batch, seeds, -1), record["score_pred"])
    close("train/head_width", out.raw.width.reshape(batch, seeds, 1), record["width_pred"])


@pytest.mark.slow
def test_label_matching_matches(training, model, manifest):
    record, out, variants, replay, _, _, (batch, _) = training
    seeds = model.proposal.num_seeds
    matched, everything, highest = out.matched, variants["all"], variants["highest"]
    close(
        "labels/point",
        matched.label_point.reshape(batch, seeds, 3),
        record["label_point"],
        atol=1e-6,
    )
    close(
        "labels/crop_rotation",
        matched.crop_rotation.reshape(batch, seeds, 3, 3),
        record["label_crop_rotation"],
        atol=1e-6,
    )
    close(
        "labels/crop_centre",
        matched.crop_centre.reshape(batch, seeds, 3),
        record["crop_centre"],
        atol=0.0,
        rtol=0.0,
    )
    # View targets are label values re-indexed to the template: exact.
    target, supervised = model.view.targets(matched)
    assert bool(supervised.all())
    assert np.array_equal(
        target.reshape(batch, seeds, -1).cpu().numpy(), record["label_view_graspness"]
    )

    flat = {
        name: torch.from_numpy(record[f"label_{name}"].reshape(-1)).to(matched.valid.device)
        for name in ("angle", "depth", "score", "width", "valid")
    }
    labelled = flat["angle"] != 12
    # Wherever the reference labelled a seed, the labels are these, value for value.
    assert bool(everything.view_labelled.all())
    for name in ("angle", "depth", "score", "width"):
        assert torch.equal(
            getattr(everything, name)[labelled],
            flat[name][labelled].to(getattr(everything, name).dtype),
        ), name
    # With the reference's choice among shared views replayed, the validity mask is its own.
    assert torch.equal(replay.valid, flat["valid"])
    count("labels/seeds", len(labelled))
    count("labels/labelled_in_reference", int(labelled.sum()))
    count("labels/valid_in_reference", int(flat["valid"].sum()))
    # A template view is unshared when both fixed rules label it. There the reference is
    # defined, and identical.
    unshared = matched.view_labelled & highest.view_labelled
    assert bool(labelled[unshared].all())
    assert torch.equal(matched.valid[unshared], flat["valid"][unshared])
    count("labels/seeds_at_shared_views", int((~unshared).sum()))
    count("labels/valid_mask_differences_of_lowest", int((matched.valid != flat["valid"]).sum()))
    count(
        "labels/reference_repeat_differences",
        manifest["training"]["repeat_matching"]["seeds_with_other_labelled_state"],
    )


@pytest.mark.slow
def test_every_loss_term_matches(training, manifest):
    record, _, _, _, losses, _, _ = training
    assert manifest["training"]["loss_weights"] == {  # the reference reports weighted terms
        "ObjectnessLoss": 1.0,
        "GraspnessLoss": 10.0,
        "ViewLoss": 100.0,
        "AngleLoss": 1.0,
        "DepthLoss": 1.0,
        "ScoreClsLoss": 1.0,
        "WidthLoss": 10.0,
    }
    terms = ("objectness", "graspness", "view", "angle", "depth", "score", "width")
    assert {k for k in losses if k.startswith("loss/")} == {f"loss/{term}" for term in terms}
    for term in terms:
        close(f"loss/{term}", losses[f"loss/{term}"], record[f"stat_{term}_loss"], rtol=1e-5)
    total = sum(float(losses[f"loss/{term}"]) for term in terms)
    close("loss/total", np.float64(total), record["stat_total"], rtol=1e-5)
    for what in ("angle", "depth", "score"):
        close(
            f"metric/{what}_accuracy",
            losses[f"metric/{what}_accuracy"],
            record[f"stat_D: {what.capitalize()} Acc"],
            rtol=1e-6,
        )
    close(
        "metric/objectness_accuracy",
        losses["metric/objectness_accuracy"],
        record["stat_stage1_objectness_acc"],
        rtol=1e-6,
    )


@pytest.mark.slow
def test_gradients_match(training, converter):
    record, _, _, _, _, gradients, _ = training
    compared = 0
    for key in sorted(k for k in record if k.startswith("grad_") and k != "grad_norm"):
        filled = converter._translate_finegrasp(key[len("grad_") :], torch.from_numpy(record[key]))
        for name, reference in filled.items():
            close(f"grad/{name}", gradients[name], reference.numpy(), rtol=1e-2)
            compared += 1
    assert compared >= 17  # among them the stem (normal channels) and all parts of the fusion
    norm = sum(float(g.double().pow(2).sum()) for g in gradients.values()) ** 0.5
    close("grad/total_norm", np.float64(norm), record["grad_norm"], rtol=1e-4)
