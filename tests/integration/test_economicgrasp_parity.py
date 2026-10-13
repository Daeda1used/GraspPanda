"""EconomicGrasp against the reference implementation, stage by stage, with the authors'
weights.

The reference values were recorded by ``tools/make_economicgrasp_fixtures.py`` from
``iSEE-Laboratory/EconomicGrasp`` at commit ``7361bb6`` (MinkowskiEngine backbone) in its
own environment. Here the same checkpoint is converted with
``tools/convert_checkpoint.py``, loaded into the GraspPanda model (spconv backbone) and
fed the recorded inputs.

Regenerate the fixtures (about 1 GB, kept under the cache root, never committed)::

    $GRASPPANDA_ORACLE_PYTHON -B tools/make_economicgrasp_fixtures.py \
        --reference <pristine EconomicGrasp tree> --checkpoint <checkpoint-kinect.tar> \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" \
        --out "$GRASPPANDA_CACHE/fixtures/economicgrasp"

Tolerances are those of the GSNet parity tests, for the same reasons.

* Discrete results must be *identical*: the voxel of every point, the candidate set,
  the seed indices, the chosen views, the matched label points and collision decisions.
* Activations must agree within ``RTOL = 2e-5`` of the largest reference magnitude of
  the stage: both networks compute the same float32 sums of products in different
  orders. The reference differs from *itself* by about 1e-2 on the same features when
  torch 1.9's default TF32 mode is left on (see ``manifest.json``), which is why the
  fixtures are recorded with TF32 off.
* Gradients are compared at 1e-2 of a tensor's largest entry and the total gradient
  norm at 1e-4.

Three things are *replayed* from the reference record instead of being drawn here,
because the reference draws them at random or leaves them undefined:

* the training view of every seed (``torch.multinomial``);
* the attention weights dropped by the two dropout layers;
* which of two template views that share a labelled view received the label. The
  reference decides this with an indexed assignment that has repeated indices; on a GPU
  the outcome differs between two runs on the same inputs (the manifest counts the
  seeds that changed). The label-matching test therefore requires identity wherever
  the reference is defined and only *counts* the agreement of GraspPanda's fixed rule
  on the remaining seeds; the loss tests replay the reference's outcome for them.

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
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.ops import sparse, voxelize
from grasppanda.registry import DETECTORS, POSTPROCESSORS
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

pytestmark = [pytest.mark.gpu, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
MODEL_CONFIG = "_base_/models/economicgrasp.yaml"
RUN_CONFIG = "economicgrasp/economicgrasp_sunet14_g1b-kinect_10e"
RTOL = 2e-5
#: Reference: trainable parameters of the upstream network (backbone 14 646 816).
REFERENCE_PARAMETERS = 15_750_238
BACKBONE_TAPS = (
    ["bn0"]
    + [name for i in range(1, 5) for name in (f"bn{i}", f"block{i}")]
    + [name for i in range(4, 8) for name in (f"bntr{i}", f"block{i + 1}")]
    + ["final"]
)
TOKENS = ("angle", "depth", "width", "score")

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
    directory = asset_cache / "fixtures" / "economicgrasp"
    if not (directory / "manifest.json").is_file():
        pytest.skip(
            f"no EconomicGrasp reference fixtures at {directory}; record them with "
            "tools/make_economicgrasp_fixtures.py (see this module's docstring)"
        )
    yield directory
    if _report:
        (directory / "parity_report.json").write_text(json.dumps(_report, indent=2) + "\n")
        print("\nEconomicGrasp parity (max abs difference, relative to the stage's largest value):")
        for stage, entry in _report.items():
            if "max_abs" in entry:
                print(f"  {stage:36s} {entry['max_abs']:.3e}   {entry['max_rel']:.2e}")
            else:
                print(f"  {stage:36s} {entry['count']}")


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
        checkpoint, camera=manifest["camera"], source_format="economicgrasp"
    )
    assert meta["model_config"] == MODEL_CONFIG
    assert meta["parameters"] == REFERENCE_PARAMETERS and meta["tensors"] == 260
    assert manifest["reference_parameters"] == REFERENCE_PARAMETERS  # counted upstream
    detector = DETECTORS.build(Config.load(MODEL_CONFIG)["model"])
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
    """The fixtures describe what the runnable Kinect config builds and feeds."""
    run = Config.load(RUN_CONFIG)
    assert run["model"] == Config.load(MODEL_CONFIG)["model"]
    sample = run.select("data.test.dataset")["transforms"][0]
    assert sample == {"type": "RandomSample", "num_points": manifest["num_points"]}
    assert run.select("data.test.dataset")["camera"] == manifest["camera"]
    assert model.backbone.voxel_size == manifest["voxel_size"]


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


def test_crop_matches_at_every_step_with_reference_seeds(inference, model):
    record, _, _, out, _ = inference
    grouper = model.grouper
    support = PointFeatures(out.seeds.feats, out.seeds.points)
    centres, rotation = out.seeds.points.coords, out.approach.rotation
    assert np.array_equal(centres.cpu().numpy(), record["crop_centre"][0])
    with torch.no_grad():
        grouped = grouper.group(support, centres, rotation)
        tokens = grouper.tokens(support, centres, rotation)
        mixed = grouper.interaction(tokens)
        after = grouper.post_mlp(mixed.flatten(0, 1)).unflatten(0, mixed.shape[:2])
    close("group/local_xyz", grouped[..., :3], record["grouped"][0][..., :3], atol=1e-6)
    close("group/features", grouped[..., 3:], record["grouped"][0][..., 3:])
    width = grouper.token_channels - 3
    close("group/shared_mlp", tokens[..., :width], record["crop_mlp"][0])
    close("group/tokens", tokens, record["crop_tokens_in"])  # [shared MLP | centre xyz]
    assert torch.equal(tokens[..., width:], centres[:, None, :].expand(-1, grouper.nsample, -1))
    close("group/interaction", mixed, record["crop_tokens_out"])
    close("group/post_mlp", after, record["crop_post_mlp"][0])
    close("group/pooled", out.groups.feats, record["crop"][0])


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


def test_the_extra_class_never_wins_so_both_decoders_agree(inference, model, manifest):
    """With this checkpoint the arg-max option of upstream 4119bdc changes nothing."""
    _, observation, natural, _, _ = inference
    wins = [entry["extra_class_wins"] for entry in manifest["inference"]]
    assert all(entry == {"angle": 0, "depth": 0} for entry in wins)  # measured upstream
    assert model.head.decode_argmax == "all_logits"
    (pinned,) = model.decode(natural, observation)
    model.head.decode_argmax = "supervised"
    try:
        (later,) = model.decode(natural, observation)
    finally:
        model.head.decode_argmax = "all_logits"
    assert np.array_equal(pinned.to_graspnet_array(), later.to_graspnet_array())
    assert int((natural.raw.angle.argmax(-1) == 12).sum()) == 0
    assert int((natural.raw.depth.argmax(-1) == 4).sum()) == 0


def test_collision_filter_of_the_run_config_makes_the_same_decisions(inference, model):
    record, observation, natural, _, _ = inference
    (node,) = Config.load(RUN_CONFIG)["evaluation"]["variants"]["cd"]
    collision = POSTPROCESSORS.build(node)
    assert isinstance(collision, ModelFreeCollisionFilter) and collision.voxel_size == 0.005
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
def test_decisions_agree_on_a_sweep_of_test_frames(fixture_dir, manifest, model, cuda):
    """Seeds sit behind a threshold, grasps behind arg-maxes: how often do both sides agree?

    On every frame the two candidate sets may differ only at points whose reference
    score is within round-off of the decision boundary; where the sets are equal the
    seeds must be identical. The reference's seeds are then injected. Each seed is
    decoded through three arg-maxes (view, angle class, depth class); a choice may
    differ from the reference's only where the reference's best and second-best values
    are within round-off of each other, and every other grasp must match.
    """
    if "sweep" not in manifest:
        pytest.skip("the fixtures were recorded without a sweep; regenerate them")
    record = np.load(fixture_dir / manifest["sweep"]["file"])
    assert "view_margin" in record, "fixtures are older than this test; regenerate them"
    frames, points = record["point_clouds"].shape[:2]
    seeds_per_frame = model.proposal.num_seeds
    threshold = model.proposal.graspness_threshold
    model.eval()
    same_candidates = same_seeds = flipped_points = 0
    ties = {"view": 0, "angle": 0, "depth": 0}
    for start in range(0, frames, 8):
        rows = slice(start, min(start + 8, frames))
        observation = Observation(
            keys=[
                SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))
                for scene, index in zip(record["scene"][rows], record["frame"][rows])
            ],
            points=PointSet.from_dense(torch.from_numpy(record["point_clouds"][rows])),
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


# ----------------------------------------------------------------------- training


class _ReplayDropout(nn.Module):
    """Dropout that drops exactly the entries a recorded run dropped."""

    def __init__(self, keep: np.ndarray, probability: float) -> None:
        super().__init__()
        self.keep, self.probability = torch.from_numpy(keep), probability

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        noise = self.keep.to(weights).reshape(weights.shape) / (1.0 - self.probability)
        return weights * noise


@contextmanager
def replayed_dropout(model, record) -> Iterator[None]:
    """Replace the two dropout layers by replays of the reference's draws."""
    blocks = {"crop": model.grouper.interaction, "head": model.head.interaction}
    originals = {name: block.dropout for name, block in blocks.items()}
    for name, block in blocks.items():
        # A weight that was exactly zero before dropout is recorded as dropped; it is zero
        # either way.
        block.dropout = _ReplayDropout(record[f"{name}_dropout_keep"], originals[name].p)
    try:
        yield
    finally:
        for name, block in blocks.items():
            block.dropout = originals[name]


def compact_labels(fixture_dir, manifest, record) -> list[list[CompactGraspLabels]]:
    """Supervision of the recorded batch from the files the reference generator wrote."""
    scenes = {}
    for scene, entry in manifest["labels"]["scenes"].items():
        with np.load(fixture_dir / entry["file"]) as archive:
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
    observation = Observation(
        keys=[
            SampleKey("graspnet1b", scene, manifest["camera"], index)
            for scene, index in manifest["training"]["frames"]
        ],
        points=PointSet.from_dense(torch.from_numpy(record["point_clouds"])),
    ).to(cuda)
    supervision = Supervision(
        objects=[[] for _ in range(batch)],
        point_labels={
            "objectness": torch.from_numpy(record["objectness_label"]).reshape(-1),
            "graspness": torch.from_numpy(record["graspness_label"]).reshape(-1),
        },
        extras={COMPACT_LABELS: compact_labels(fixture_dir, manifest, record)},
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
    fixed = model.head.loss(out.raw, out.matched)  # GraspPanda's own rule, for the report
    model.load_state_dict(state)
    model.eval()
    return record, out, variants, replay, losses, fixed, gradients, (batch, points)


@pytest.mark.slow
def test_training_forward_matches(training, model):
    record, out, _, _, _, _, _, (batch, points) = training
    seeds = model.proposal.num_seeds
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
    close("train/group_pooled", out.groups.feats.reshape(batch, seeds, -1), record["crop"])
    close("train/head_angle", out.raw.angle.reshape(batch, seeds, -1), record["angle_pred"])
    close("train/head_depth", out.raw.depth.reshape(batch, seeds, -1), record["depth_pred"])
    close("train/head_score", out.raw.score.reshape(batch, seeds, -1), record["score_pred"])
    close("train/head_width", out.raw.width.reshape(batch, seeds, 1), record["width_pred"])


@pytest.mark.slow
def test_label_matching_matches(training, model, manifest):
    record, out, variants, replay, _, _, _, (batch, _) = training
    seeds = model.proposal.num_seeds
    matched, everything, highest = out.matched, variants["all"], variants["highest"]
    assert any(manifest["training"]["mirrored_frames"]), "the fixture is meant to cover a flip"
    assert not all(manifest["training"]["mirrored_frames"])
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
    # Seeds without a label carry the reference's defaults, which the masks exclude.
    assert bool((flat["depth"][~labelled] == 4).all() and (flat["score"][~labelled] == 0).all())
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
    shared = ~unshared
    agree = matched.view_labelled[shared] == labelled[shared]
    count("labels/seeds_at_shared_views", int(shared.sum()))
    count("labels/shared_seeds_where_lowest_agrees", int(agree.sum()))
    count(
        "labels/shared_seeds_where_highest_agrees",
        int((highest.view_labelled[shared] == labelled[shared]).sum()),
    )
    count("labels/valid_mask_differences_of_lowest", int((matched.valid != flat["valid"]).sum()))
    count(
        "labels/reference_repeat_differences",
        manifest["training"]["repeat_matching"]["seeds_with_other_labelled_state"],
    )
    # The fixed rule is the reference's usual outcome, not a coin flip.
    assert 0 < int(shared.sum()) < len(shared) // 2
    assert int(agree.sum()) >= 0.85 * int(shared.sum())


@pytest.mark.slow
def test_every_loss_term_matches(training):
    record, _, _, _, losses, fixed, _, _ = training
    reference = {  # the reference reports unweighted terms; ours are weighted
        "loss/objectness": 1.0 * record["stat_B: Objectness Loss"],
        "loss/graspness": 10.0 * record["stat_B: Graspness Loss"],
        "loss/view": 100.0 * record["stat_B: View Loss"],
        "loss/angle": 1.0 * record["stat_B: Angle Loss"],
        "loss/depth": 1.0 * record["stat_B: Depth Loss"],
        "loss/score": 1.0 * record["stat_B: Score Loss"],
        "loss/width": 10.0 * record["stat_B: Width Loss"],
    }
    assert {k for k in losses if k.startswith("loss/")} == set(reference)
    for key, value in reference.items():
        close(key, losses[key], value, rtol=1e-5)
    total = sum(float(losses[key]) for key in reference)
    close("loss/total", np.float64(total), record["stat_A: Overall Loss"], rtol=1e-5)
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
        record["stat_D: Objectness Acc"],
        rtol=1e-6,
    )
    # What GraspPanda's fixed rule for shared views does to the head losses, relative to
    # one outcome of the reference's undefined choice. Reported, bounded loosely.
    for key in ("loss/angle", "loss/depth", "loss/score", "loss/width"):
        close(f"fixed_rule/{key}", fixed[key], reference[key], rtol=5e-2)


@pytest.mark.slow
def test_gradients_match(training, converter):
    record, _, _, _, _, _, gradients, _ = training
    for key in sorted(k for k in record if k.startswith("grad_") and k != "grad_norm"):
        name, reference = converter._translate_economicgrasp(
            key[len("grad_") :], torch.from_numpy(record[key])
        )
        close(f"grad/{name}", gradients[name], reference.numpy(), rtol=1e-2)
    norm = sum(float(g.double().pow(2).sum()) for g in gradients.values()) ** 0.5
    close("grad/total_norm", np.float64(norm), record["grad_norm"], rtol=1e-4)
