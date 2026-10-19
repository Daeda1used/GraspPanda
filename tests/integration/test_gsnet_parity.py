"""GSNet against the reference implementation, stage by stage, with the authors' weights.

The reference values were recorded by ``tools/make_gsnet_fixtures.py`` from
``graspnet/graspness_unofficial`` (MinkowskiEngine backbone) in its own environment.
Here the same checkpoint is converted with ``tools/convert_checkpoint.py``, loaded into
the GraspPanda model (spconv backbone) and fed the recorded inputs.

Regenerate the fixtures (about 1 GB, kept under the cache root, never committed)::

    $GRASPPANDA_ORACLE_PYTHON tools/make_gsnet_fixtures.py \
        --reference <graspness_unofficial checkout> --checkpoint <checkpoint-rs.tar> \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --out "$GRASPPANDA_CACHE/fixtures/gsnet"

Tolerances, and why they are what they are.

* Discrete results must be *identical*: the voxel of every point, the candidate set,
  the seed indices, the chosen views, label matches and collision decisions.
* Activations must agree within ``RTOL = 2e-5`` of the largest reference magnitude of
  the stage (about 170 float32 ulps of that value). Both networks compute the same
  float32 sums of products, but MinkowskiEngine and spconv accumulate the terms of a
  sparse convolution in different orders, through the 25 sparse convolutions, 8 linear
  maps and 32 batch norms of the backbone, and the heads then multiply the result by
  weight matrices. The largest relative difference observed on the recorded frames is
  4.8e-6. For scale: the reference differs from *itself* by about 1e-2 on the same
  features when torch 1.9's default TF32 mode is left on (see ``manifest.json``),
  which is why the fixtures are recorded with TF32 off.
* Gradients agree far less tightly than activations, because the backward pass sums
  signed terms over all voxels: the *same* GraspPanda model on CPU and on GPU differs
  by up to 2e-3 of a tensor's largest gradient entry, and two runs of the reference
  training step differ by 6e-6 in total gradient norm. Per-tensor gradients are
  therefore compared at 1e-2 of the largest entry and the total gradient norm at 1e-4.

Every comparison is also written to ``parity_report.json`` next to the fixtures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.config import Config
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.ops import sparse, voxelize
from grasppanda.registry import DETECTORS
from grasppanda.structures import (
    GraspGroup,
    ObjectGraspLabels,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    Supervision,
)

pytestmark = [pytest.mark.gpu, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
RTOL = 2e-5
#: Reference: trainable parameters of the upstream GraspNet (backbone 14 646 816).
REFERENCE_PARAMETERS = 15_353_775
BACKBONE_TAPS = (
    ["bn0"]
    + [name for i in range(1, 5) for name in (f"bn{i}", f"block{i}")]
    + [name for i in range(4, 8) for name in (f"bntr{i}", f"block{i + 1}")]
    + ["final"]
)

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Path:
    directory = asset_cache / "fixtures" / "gsnet"
    if not (directory / "manifest.json").is_file():
        pytest.skip(
            f"no GSNet reference fixtures at {directory}; record them with "
            "tools/make_gsnet_fixtures.py (see this module's docstring)"
        )
    yield directory
    if _report:
        (directory / "parity_report.json").write_text(json.dumps(_report, indent=2) + "\n")
        print("\nGSNet parity (max abs difference, relative to the stage's largest value):")
        for stage, entry in _report.items():
            if "max_abs" in entry:
                print(f"  {stage:34s} {entry['max_abs']:.3e}   {entry['max_rel']:.2e}")
            else:
                print(f"  {stage:34s} {entry['count']}")


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
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the checkpoint the fixtures were recorded with is missing: {checkpoint}")
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    state, meta = converter.convert_file(
        checkpoint,
        "_base_/models/gsnet.yaml",
        camera="realsense",
        source_format="graspness-unofficial",
    )
    assert meta["parameters"] == REFERENCE_PARAMETERS and meta["tensors"] == 216
    detector = DETECTORS.build(Config.load("_base_/models/gsnet.yaml")["model"])
    assert sum(p.numel() for p in detector.parameters()) == REFERENCE_PARAMETERS
    detector.load_state_dict(state, strict=True)
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield detector.to(cuda)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def inference_frames() -> list[str]:
    return ["infer_0100_0000", "infer_0118_0137", "infer_0145_0064", "infer_0175_0200"]


@pytest.fixture(scope="module", params=inference_frames())
def frame(request, fixture_dir, manifest):
    """One recorded test frame: the reference record and the observation built from it."""
    listed = {entry["file"] for entry in manifest["inference"]}
    assert f"{request.param}.npz" in listed, "fixtures are older than this test; regenerate them"
    record = dict(np.load(fixture_dir / f"{request.param}.npz"))
    _, scene, index = request.param.split("_")
    observation = Observation(
        keys=[SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))],
        points=PointSet.from_dense(torch.from_numpy(record["point_clouds"])),
        # float64, as the reference test script hands the cloud to its collision detector
        full_points=PointSet(torch.from_numpy(record["full_cloud"])),
    )
    return record, observation


def backbone_modules(backbone) -> dict[str, torch.nn.Module]:
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


def test_voxelisation_is_identical_and_device_independent(frame, cuda):
    record, observation = frame
    points = observation.points
    reference = record["coors"][record["quantize2original"]]  # voxel of every point
    for device in ("cpu", cuda):
        moved = points.to(device)
        grid = voxelize(moved.coords, 0.005, moved.batch_index)
        assert np.array_equal(grid.voxels[grid.inverse].cpu().numpy(), reference), str(device)
        assert len(grid.voxels) == len(record["coors"])


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


def test_stages_after_the_seeds_match_with_reference_seeds(inference, model):
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
    support = PointFeatures(out.seeds.feats, out.seeds.points)
    grouped = model.grouper.group(support, out.seeds.points.coords, out.approach.rotation)
    close("group/local_xyz", grouped[..., :3], record["grouped"][0][..., :3], atol=1e-6)
    close("group/features", grouped[..., 3:], record["grouped"][0][..., 3:])
    close("group/pooled", out.groups.feats, record["crop"][0])
    close("head/score", out.raw.score, record["score_pred"][0])
    close("head/width", out.raw.width, record["width_pred"][0])


def test_decoded_grasps_match_end_to_end(inference, model):
    """No injection: inputs in, ``(1024, 17)`` out."""
    record, observation, natural, _, _ = inference
    (grasps,) = model.decode(natural, observation)
    assert grasps.key == observation.keys[0] and grasps.score_kind == "predicted_quality"
    mine, reference = grasps.to_graspnet_array(), record["decoded"][0]
    assert mine.shape == reference.shape == (1024, 17)
    close("decoded/score", mine[:, 0], reference[:, 0])
    close("decoded/width", mine[:, 1], reference[:, 1], atol=1e-6)
    assert np.array_equal(mine[:, 2], reference[:, 2]), "height"
    assert np.array_equal(mine[:, 3], reference[:, 3]), "depth"
    close("decoded/rotation", mine[:, 4:13], reference[:, 4:13], atol=1e-6)
    assert np.array_equal(mine[:, 13:16], reference[:, 13:16]), "translation"
    assert np.array_equal(mine[:, 16], reference[:, 16]), "object id"


def test_collision_filter_makes_the_same_decisions(inference, model):
    record, observation, natural, _, _ = inference
    collision = ModelFreeCollisionFilter()  # voxel 0.01, approach 0.05, threshold 0.01
    scene = voxel_down_sample(observation.full_points.coords, collision.voxel_size)
    # Same point set as Open3D's voxel_down_sample (whose output order is unspecified).
    reference_scene = torch.from_numpy(record["collision_scene"]).to(scene)
    assert scene.shape == reference_scene.shape
    nearest = torch.cdist(scene, reference_scene).argmin(dim=1)
    assert len(torch.unique(nearest)) == len(scene)  # a one-to-one match
    close(
        "collision/scene_points",
        scene,
        reference_scene[nearest].cpu().numpy(),
        atol=1e-12,
        rtol=0.0,
    )
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
    clouds = torch.stack([torch.from_numpy(r["point_clouds"][0]) for r in records])
    observation = Observation(
        keys=[SampleKey("graspnet1b", 100 + i, manifest["camera"], i) for i in range(2)],
        points=PointSet.from_dense(clouds),
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
def test_seed_selection_agrees_on_a_sweep_of_test_frames(fixture_dir, manifest, model, cuda):
    """Seeds sit behind a threshold and an arg-max: how often do both sides agree?

    On every frame the two candidate sets may differ only at points whose reference
    score is within round-off of the decision boundary. Where the sets are equal the
    seeds must be identical; in every case the reference's seeds are injected as well
    and the decoded grasps must match.
    """
    if "sweep" not in manifest:
        pytest.skip("the fixtures were recorded without a sweep; regenerate them")
    record = np.load(fixture_dir / manifest["sweep"]["file"])
    frames, count = record["point_clouds"].shape[:2]
    seeds_per_frame = model.proposal.num_seeds
    threshold = model.proposal.graspness_threshold
    model.eval()
    same_candidates = same_seeds = flipped_points = 0
    for start in range(0, frames, 8):
        rows = slice(start, min(start + 8, frames))
        observation = Observation(
            keys=[
                SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))
                for scene, index in zip(record["scene"][rows], record["frame"][rows])
            ],
            points=PointSet.from_dense(torch.from_numpy(record["point_clouds"][rows])),
        ).to(cuda)
        offset = torch.arange(len(observation), device=cuda)[:, None] * count
        reference_seeds = torch.from_numpy(record["seed_index"][rows]).to(cuda) + offset
        with torch.no_grad():
            out = model.run_stages(observation)
            replay = model.run_stages(observation, seed_index=reference_seeds)
            replayed = model.decode(replay, observation)
        mask = model.proposal.candidates(out.point_scores).reshape(-1, count).cpu().numpy()
        index = out.seeds.points.parent_index.reshape(-1, seeds_per_frame) - offset
        seeds = index.cpu().numpy()
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
            mine, reference = replayed[local].to_graspnet_array(), record["decoded"][frame]
            assert np.array_equal(mine[:, 13:16], reference[:, 13:16]), frame
            assert np.array_equal(mine[:, 3], reference[:, 3]), frame
            close("sweep/decoded_score", mine[:, 0], reference[:, 0])
            close("sweep/decoded_width", mine[:, 1], reference[:, 1], atol=1e-6)
            close("sweep/decoded_rotation", mine[:, 4:13], reference[:, 4:13], atol=1e-6)
    _report["sweep/frames"] = {"count": frames}
    _report["sweep/frames_with_identical_candidates"] = {"count": same_candidates}
    _report["sweep/frames_with_identical_seeds"] = {"count": same_seeds}
    _report["sweep/candidate_flips"] = {"count": flipped_points}
    print(
        f"\nsweep: {same_seeds} of {frames} frames with identical seeds "
        f"({same_candidates} with identical candidates, {flipped_points} boundary flips)"
    )


# ----------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def training(fixture_dir, manifest, model, cuda):
    """The recorded training batch pushed through loss() with the reference's view draw."""
    record = dict(np.load(fixture_dir / manifest["training"]["file"]))
    labels = np.load(fixture_dir / manifest["training"]["labels"])
    batch, count = record["point_clouds"].shape[:2]
    objects = []
    for frame, number in enumerate(labels["objects_per_frame"].tolist()):
        objects.append([])
        for position in range(number):
            prefix = f"f{frame}_o{position}_"
            pose = torch.eye(4)
            pose[:3] = torch.from_numpy(labels[prefix + "pose"])
            objects[-1].append(
                ObjectGraspLabels(
                    object_id=position,
                    pose=pose,
                    points=torch.from_numpy(labels[prefix + "points"]),
                    scores=torch.from_numpy(labels[prefix + "scores"]),  # collisions already zeroed
                    widths=torch.from_numpy(labels[prefix + "widths"]),
                )
            )
    observation = Observation(
        keys=[
            SampleKey("graspnet1b", scene, manifest["camera"], index)
            for scene, index in manifest["training"]["frames"]
        ],
        points=PointSet.from_dense(torch.from_numpy(record["point_clouds"])),
    ).to(cuda)
    supervision = Supervision(
        objects=objects,
        point_labels={
            "objectness": torch.from_numpy(record["objectness_label"]).reshape(-1),
            "graspness": torch.from_numpy(record["graspness_label"]).reshape(-1),
        },
    ).to(cuda)
    model.train()
    model.zero_grad()
    views = torch.from_numpy(record["sampled_view_index"]).reshape(-1).to(cuda)
    out = model.run_stages(observation, supervision, view_index=views)
    losses = {
        **model.proposal.loss(out.point_scores, supervision),
        **model.view.loss(out.approach, out.matched),
        **model.head.loss(out.raw, out.matched),
    }
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
    model.zero_grad()
    model.eval()
    return record, out, losses, gradients, (batch, count)


@pytest.mark.slow
def test_training_forward_matches(training, model):
    record, out, _, _, (batch, count) = training
    seeds = model.proposal.num_seeds
    close("train/point_features", out.feats.feats.reshape(batch, count, -1), record["point_feats"])
    close(
        "train/objectness",
        out.point_scores.objectness.reshape(batch, count, 2),
        record["objectness"],
    )
    close("train/graspness", out.point_scores.graspness.reshape(batch, count), record["graspness"])
    offset = torch.arange(batch, device=out.seeds.points.coords.device)[:, None] * count
    index = out.seeds.points.parent_index.reshape(batch, seeds) - offset
    assert np.array_equal(index.cpu().numpy(), record["seed_index"])  # no injection needed
    close("train/view_scores", out.approach.logits.reshape(batch, seeds, -1), record["view_score"])
    close("train/group_pooled", out.groups.feats.reshape(batch, seeds, -1), record["crop"])
    close("train/head_score", out.raw.score.reshape(batch, seeds, 12, 4), record["score_pred"])
    close("train/head_width", out.raw.width.reshape(batch, seeds, 12, 4), record["width_pred"])


def manifest_mirrored(record) -> bool:
    """True when at least one recorded frame was mirrored by the flip augmentation."""
    rotations = record["label_crop_rotation"]
    return bool((np.linalg.det(rotations.reshape(-1, 3, 3).astype(np.float64)) < 0).any())


@pytest.mark.slow
def test_label_matching_matches(training, model):
    record, out, _, _, (batch, _) = training
    seeds = model.proposal.num_seeds
    matched = out.matched
    assert bool(matched.valid.all())  # no distance threshold in the reference
    assert manifest_mirrored(record), "the fixture is meant to cover a mirrored (flipped) frame"
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
    target = model.view.encode_targets(matched.view_friction)
    close(
        "labels/view_graspness",
        target.reshape(batch, seeds, -1),
        record["label_view_graspness"],
        atol=1e-6,
    )
    targets = model.head.encode_targets(matched)
    close(
        "labels/score_target",
        targets.score.reshape(batch, seeds, 12, 4),
        record["label_score_target"],
        atol=1e-6,
    )
    close(
        "labels/width",
        matched.width.reshape(batch, seeds, 12, 4),
        record["label_width"],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.slow
def test_every_loss_term_matches(training):
    record, _, losses, _, _ = training
    reference = {  # the reference reports unweighted terms; ours are weighted
        "loss/objectness": 1.0 * record["loss_stage1_objectness_loss"],
        "loss/graspness": 10.0 * record["loss_stage1_graspness_loss"],
        "loss/view": 100.0 * record["loss_stage2_view_loss"],
        "loss/score": 15.0 * record["loss_stage3_score_loss"],
        "loss/width": 10.0 * record["loss_stage3_width_loss"],
    }
    assert {k for k in losses if k.startswith("loss/")} == set(reference)
    for key, value in reference.items():
        close(key, losses[key], value, rtol=1e-5)
    total = sum(float(losses[key]) for key in reference)
    close("loss/total", np.float64(total), record["loss_overall_loss"], rtol=1e-5)


@pytest.mark.slow
def test_gradients_match(training, converter):
    record, _, _, gradients, _ = training
    for key in sorted(k for k in record if k.startswith("grad_") and k != "grad_norm"):
        name, reference = converter._translate(key[len("grad_") :], torch.from_numpy(record[key]))
        close(f"grad/{name}", gradients[name], reference.numpy(), rtol=1e-2)
    norm = sum(float(g.double().pow(2).sum()) for g in gradients.values()) ** 0.5
    close("grad/total_norm", np.float64(norm), record["grad_norm"], rtol=1e-4)
