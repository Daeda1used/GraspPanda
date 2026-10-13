"""Scale-Balanced-Grasp against the reference implementation, stage by stage, with the
authors' weights.

The reference values were recorded by ``tools/make_sbg_fixtures.py`` from the pristine
``mahaoxiang822/Scale-Balanced-Grasp`` (commit ``995d6f8``, network ``GraspNet_MSCQ``
with the scale-balanced loss) in its own environment. Here the same checkpoint is
converted with ``tools/convert_checkpoint.py``, loaded into the GraspPanda model and fed
the recorded inputs.

Regenerate the fixtures (about 0.9 GB, kept under the cache root, never committed)::

    $GRASPPANDA_SBG_ORACLE_PYTHON -B tools/make_sbg_fixtures.py \
        --reference <pristine Scale-Balanced-Grasp> --checkpoint <checkpoint-rs.tar> \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --tolerance-dir <upstream tolerance> \
        --health --out "$GRASPPANDA_CACHE/fixtures/scale_balanced_grasp"

Tolerances, and why they are what they are.

* Discrete results must be *identical*: the seed indices, the neighbour sets of both
  attention layers, the chosen views, which seeds are classified as object, the depth
  bin of every decoded grasp, label matches, the graspable target, and the
  scale-balanced weight of every seed (it is one of 32 table values).
* Every stage must agree within ``RTOL = 2e-5`` of the largest reference magnitude of
  the stage *when it is fed what the reference fed it* (the tolerance of the GSNet and
  GraspNet-baseline parity tests). The largest value observed is 9.3e-6.
* End to end from the cloud, the features the heads receive and the head scores agree
  within ``END_TO_END = 1e-4``; observed up to 4.1e-5 and 3.4e-5. The excess over the
  stage-wise figure is not a difference in any stage: the seed features agree to 6e-7,
  and the gated residual ``sigmoid(W f) * f`` amplifies that rounding 30 to 70 times
  because seed features reach magnitudes of 130 to 150. With the reference's seed
  features injected the same crop agrees to 9.3e-6, and with the reference's crop
  injected the heads agree to 4e-7.
* Cylinder membership is a threshold on a float32 product; a crop whose neighbours
  differ must be explained by boundary points in double precision (none occurs on the
  recorded frames).
* The total gradient norm is compared at 1e-4 and single gradient tensors at 1e-2 of
  their largest entry, as for GSNet.

For scale: with PyTorch 1.10's default TF32 the reference differs from *itself* by up to
0.1 on the seed features and chooses other views for 2 to 8 seeds per frame (see
``manifest.json``), which is why the fixtures are recorded with TF32 off.

Every comparison is also written to ``parity_report.json`` next to the fixtures.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.config import Config
from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY, bin_weights
from grasppanda.models.heads.friction import log_friction_score, worst_friction
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.registry import DETECTORS, EVALUATORS
from grasppanda.structures import (
    GraspGroup,
    GroupFeatures,
    ObjectGraspLabels,
    Observation,
    PointSet,
    SampleKey,
    Supervision,
)

from .test_baseline_parity import explained_by_boundary, scene_support, sha256

pytestmark = [pytest.mark.gpu, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
MODEL_CONFIG = "_base_/models/scale_balanced_grasp.yaml"
RTOL = 2e-5
END_TO_END = 1e-4
#: Reference: tensors and trainable parameters of the upstream GraspNet_MSCQ, counted there.
REFERENCE_TENSORS = 190
REFERENCE_PARAMETERS = 1_577_196

_report: dict[str, dict[str, float]] = {}


def close(stage: str, mine, reference, *, rtol: float = RTOL, atol: float = 0.0) -> None:
    """Assert ``|mine - reference| <= atol + rtol * max|reference|`` and log the numbers."""
    mine = np.asarray(mine.detach().cpu() if isinstance(mine, torch.Tensor) else mine, np.float64)
    reference = np.asarray(reference, np.float64)
    assert mine.shape == reference.shape, f"{stage}: shape {mine.shape} vs {reference.shape}"
    scale = float(np.abs(reference).max()) if reference.size else 0.0
    worst = float(np.abs(mine - reference).max()) if reference.size else 0.0
    entry = _report.setdefault(stage, {"max_abs": 0.0, "max_rel": 0.0, "scale": 0.0, "count": 0})
    entry["max_abs"] = max(entry["max_abs"], worst)
    entry["max_rel"] = max(entry["max_rel"], worst / scale if scale else 0.0)
    entry["scale"] = max(entry["scale"], scale)
    entry["count"] += 1
    assert worst <= atol + rtol * scale, (
        f"{stage}: max abs difference {worst:.3e} exceeds {atol:.1e} + {rtol:.1e} x {scale:.3e}"
    )


def count(stage: str, value: int) -> None:
    _report.setdefault(stage, {"count": 0})["count"] += int(value)


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Path:
    directory = asset_cache / "fixtures" / "scale_balanced_grasp"
    if not (directory / "manifest.json").is_file():
        pytest.skip(
            f"no Scale-Balanced-Grasp reference fixtures at {directory}; record them with "
            "tools/make_sbg_fixtures.py (see this module's docstring)"
        )
    yield directory
    if _report:
        (directory / "parity_report.json").write_text(json.dumps(_report, indent=2) + "\n")
        print("\nScale-Balanced-Grasp parity (max abs difference, relative to the largest value):")
        for stage, entry in _report.items():
            if "max_abs" in entry:
                print(f"  {stage:38s} {entry['max_abs']:.3e}   {entry['max_rel']:.2e}")
            else:
                print(f"  {stage:38s} {entry['count']}")


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
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the checkpoint the fixtures were recorded with is missing: {checkpoint}")
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    state, meta = converter.convert_file(
        checkpoint, camera=manifest["camera"], source_format="scale-balanced-grasp"
    )
    assert meta["model_config"] == MODEL_CONFIG
    assert meta["parameters"] == manifest["parameters"] == REFERENCE_PARAMETERS
    assert meta["tensors"] == manifest["tensors"] == REFERENCE_TENSORS
    detector = DETECTORS.build(Config.load(MODEL_CONFIG)["model"])
    assert sum(p.numel() for p in detector.parameters()) == REFERENCE_PARAMETERS
    detector.load_state_dict(state, strict=True)
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield detector.to(cuda).eval()
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def inference_frames() -> list[str]:
    return ["infer_0100_0000", "infer_0118_0137", "infer_0145_0064", "infer_0175_0200"]


@pytest.fixture(scope="module", params=inference_frames())
def frame(request, fixture_dir, manifest):
    """One recorded test frame: the reference record and the observation built from it."""
    listed = {entry["file"] for entry in manifest.get("inference", [])}
    assert f"{request.param}.npz" in listed, "fixtures are older than this test; regenerate them"
    record = dict(np.load(fixture_dir / f"{request.param}.npz"))
    _, scene, index = request.param.split("_")
    observation = Observation(
        keys=[SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))],
        points=PointSet.from_dense(torch.from_numpy(record["point_clouds"])),
        full_points=PointSet(torch.from_numpy(record["full_cloud"])),
    )
    return record, observation


@pytest.fixture(scope="module")
def inference(frame, model, cuda):
    """The natural run of one frame with what its backbone and view stage did inside."""
    record, observation = frame
    observation = observation.to(cuda)
    model.eval()
    taps: dict[str, torch.Tensor] = {}
    blocks = {name: getattr(model.backbone, name) for name in ("transformer1", "transformer2")}
    hooks = [
        model.view.output.register_forward_pre_hook(
            lambda _module, args: taps.__setitem__("hidden", args[0])
        )
    ]
    for name, block in blocks.items():
        hooks.append(
            block.register_forward_hook(
                lambda _m, _i, output, name=name: taps.__setitem__(name, output)
            )
        )

        def spy(coords, sizes, block=block, name=name):
            taps[name + "_neighbours"] = type(block).neighbours(block, coords, sizes)
            return taps[name + "_neighbours"]

        block.neighbours = spy
    try:
        with torch.no_grad():
            natural = model.run_stages(observation)
    finally:
        for hook in hooks:
            hook.remove()
        for block in blocks.values():
            del block.neighbours
    with torch.no_grad():
        support = scene_support(observation)
        centres, rotation = natural.seeds.points.coords, natural.approach.rotation
        neighbours = model.grouper.neighbours(support, centres, rotation)
    reference = torch.from_numpy(record["cylinder_index"]).to(cuda).long()
    return record, observation, natural, taps, neighbours, reference


# -------------------------------------------------------------------------- tests


def test_backbone_matches_level_by_level(inference, model):
    record, _, natural, taps, _, _ = inference
    seeds = natural.seeds
    assert np.array_equal(seeds.points.parent_index.cpu().numpy(), record["seed_index"][0])
    assert np.array_equal(seeds.points.coords.cpu().numpy(), record["seed_xyz"][0])
    for name in ("transformer1", "transformer2"):
        mine = torch.sort(taps[name + "_neighbours"], dim=1).values.cpu().numpy()
        expected = np.sort(record[name + "_neighbours"][0], axis=1)
        assert mine.shape == expected.shape and mine.shape[1] == 16
        different = int((mine != expected).any(axis=1).sum())
        count(f"backbone/{name}_points_with_other_neighbours", different)
        assert different == 0, name  # the neighbours are indexed by sampling order
    close("backbone/transformer1", taps["transformer1"], record["transformer1_out"][0])
    close("backbone/seed_features", natural.feats.feats, record["seed_features"][0])
    assert taps["transformer2"] is natural.feats.feats and seeds.feats is natural.feats.feats


def test_sampling_order_needs_the_kernel_tie_rule(frame, model, cuda):
    """Two equally far candidates are sampled in the order of the reference kernel; with
    the lowest-index rule they would swap places and with them their neighbour lists."""
    record, observation = frame
    assert model.backbone.sa1.tie_break == "pointnet2_cuda"
    points = observation.points.to(cuda)
    block = model.backbone.transformer1
    from grasppanda.models.backbones._packed import fps_packed

    expected = np.sort(record["transformer1_neighbours"][0], axis=1)
    swapped = {}
    for rule in ("pointnet2_cuda", "lowest_index"):
        index = fps_packed(
            points.coords, [len(points)], 2048, skip_near_origin=True, tie_break=rule
        )
        mine = torch.sort(block.neighbours(points.coords[index], [2048]), dim=1).values
        swapped[rule] = int((mine.cpu().numpy() != expected).any(axis=1).sum())
    count("backbone/points_with_other_neighbours_under_lowest_index", swapped["lowest_index"])
    assert swapped["pointnet2_cuda"] == 0


def test_objectness_and_views_match(inference, model):
    record, _, natural, taps, _, _ = inference
    approach = natural.approach
    close("view/hidden", taps["hidden"], record["approach_hidden"][0])
    close("view/objectness", approach.objectness, record["objectness"][0])
    close("view/scores", approach.logits, record["view_score"][0])
    assert np.array_equal(approach.view_index.cpu().numpy(), record["view_index"][0])
    close("view/rotation", approach.rotation, record["view_rot"][0], atol=1e-6)
    assert np.array_equal(natural.seeds.valid.cpu().numpy(), record["objectness_mask"])


def test_cylinder_neighbours_agree_up_to_boundary_points(inference, model):
    _, observation, natural, _, neighbours, reference = inference
    assert tuple(neighbours.shape) == (1, 1024, 4, 4, 64) == tuple(reference.shape)
    assert model.grouper.radii == pytest.approx((0.02, 0.04, 0.06, 0.08))
    differs = (neighbours != reference).any(dim=-1)[0]  # (seed, radius, depth)
    count("group/crops", differs.numel())
    count("group/crops_with_other_neighbours", int(differs.sum()))
    assert int(differs.sum()) <= 8, "more boundary cases than rounding explains"
    cloud = observation.points.coords
    for seed, scale, depth in torch.nonzero(differs).tolist():
        crop = type("Crop", (), {})()  # the cylinder of this radius, for the boundary check
        crop.hmin, crop.nsample = model.grouper.hmin, model.grouper.nsample
        crop.radius = model.grouper.radii[scale]
        arguments = (
            crop,
            cloud,
            natural.seeds.points.coords[seed],
            natural.approach.rotation[seed],
            model.grouper.hmax[depth],
        )
        assert explained_by_boundary(*arguments, neighbours[0, seed, scale, depth])
        assert explained_by_boundary(*arguments, reference[0, seed, scale, depth])


def test_crop_and_head_match_when_fed_what_the_reference_fed_them(inference, model, cuda):
    """Each stage on the reference's own inputs: its seed features and neighbour lists
    for the crop, its crop features for the heads."""
    record, observation, natural, _, _, reference = inference
    support = scene_support(observation)
    centres, rotation = natural.seeds.points.coords, natural.approach.rotation
    seed_features = torch.from_numpy(record["seed_features"][0]).to(cuda)
    crop_features = torch.from_numpy(record["crop"][0]).to(cuda)
    with torch.no_grad():
        grouped = model.grouper.group(support, centres, rotation, reference)
        scales = [
            mlp(grouped[:, index].reshape(-1, 3)).reshape(1024, 4, 64, 256).amax(dim=2)
            for index, mlp in enumerate(model.grouper.scales)
        ]
        fused = model.grouper.fuse(torch.cat(scales, dim=-1))
        gate = torch.sigmoid(model.grouper.gate(seed_features))
        layer = model.grouper.gate
        exact_gate = torch.sigmoid(
            seed_features.double() @ layer.weight.double().T + layer.bias.double()
        )
        groups = model.grouper(support, centres, rotation, seed_features, neighbours=reference)
        head = model.head(GroupFeatures(crop_features, groups.axes))
    close("group/local_xyz", grouped, record["grouped"][0], atol=1e-7)
    for index, pooled in enumerate(scales):
        close(f"group/radius_{index}", pooled, record[f"crop_scale{index}"][0])
    close("group/fused", fused, record["fused"][0])
    # The gate is a sigmoid of a 256-term sum whose terms add up to about 1400 in
    # magnitude, so single precision leaves a few 1e-5 in it on either side. It is
    # therefore compared with the double-precision value of the same inputs: both
    # implementations must lie in the same narrow band around it.
    close("group/gate_reference_vs_double", record["gate"][0], exact_gate.cpu(), rtol=1e-4)
    close("group/gate_here_vs_double", gate, exact_gate.cpu(), rtol=1e-4)
    close("group/head_input", groups.feats, record["crop"][0])
    close("head/score", head.score, record["score_pred"][0])
    close("head/angle", head.angle, record["angle_pred"][0])
    close("head/width", head.width, record["width_pred"][0])
    close("head/tolerance", head.tolerance, record["tolerance_pred"][0])


def test_crop_and_head_match_end_to_end(inference, model):
    """No injection. The gated seed residual amplifies the rounding of the seed
    features, so the bound is ``END_TO_END``; see the module docstring."""
    record, _, natural, _, neighbours, reference = inference
    if bool((neighbours != reference).any()):
        pytest.skip("a boundary point changed a crop on this frame")
    close("end_to_end/head_input", natural.groups.feats, record["crop"][0], rtol=END_TO_END)
    close("end_to_end/score", natural.raw.score, record["score_pred"][0], rtol=END_TO_END)
    close("end_to_end/angle", natural.raw.angle, record["angle_pred"][0], rtol=END_TO_END)
    close("end_to_end/width", natural.raw.width, record["width_pred"][0], rtol=END_TO_END)
    close(
        "end_to_end/tolerance", natural.raw.tolerance, record["tolerance_pred"][0], rtol=END_TO_END
    )


def test_decoded_grasps_match_end_to_end(inference, model):
    """Inputs in, grasps out; the emitted score includes the probability of 'object'."""
    record, observation, natural, _, neighbours, reference = inference
    (grasps,) = model.decode(natural, observation)
    assert grasps.key == observation.keys[0]
    assert grasps.score_kind == "predicted_quality_x_objectness_x_tolerance"
    mine, expected = grasps.to_graspnet_array(), record["decoded"]
    assert mine.shape == expected.shape and mine.shape[0] == int(record["objectness_mask"].sum())
    assert np.array_equal(mine[:, 13:16], expected[:, 13:16]), "translation"
    exact = ~(neighbours != reference).flatten(2).any(dim=-1)[0][natural.seeds.valid]
    exact = exact.cpu().numpy()
    count("decoded/grasps", len(mine))
    count("decoded/grasps_with_a_boundary_crop", int((~exact).sum()))
    close("decoded/score", mine[exact, 0], expected[exact, 0], rtol=END_TO_END)
    close("decoded/width", mine[exact, 1], expected[exact, 1], rtol=END_TO_END, atol=1e-6)
    assert np.array_equal(mine[exact, 3], expected[exact, 3]), "depth"
    close("decoded/rotation", mine[exact, 4:13], expected[exact, 4:13], atol=1e-6)


def test_collision_filter_makes_the_same_decisions(inference, model):
    record, observation, _, _, _, _ = inference
    collision = ModelFreeCollisionFilter()  # voxel 0.01, approach 0.05, threshold 0.01
    scene = voxel_down_sample(observation.full_points.coords, collision.voxel_size)
    assert scene.shape == record["collision_scene"].shape
    reference = GraspGroup.from_graspnet_array(record["decoded"]).to(scene.device)
    mask = collision.collision_mask(reference, scene)
    assert np.array_equal(mask.cpu().numpy(), record["collision_mask"])


def test_a_batch_predicts_what_its_frames_predict_alone(fixture_dir, manifest, model, cuda):
    records = [dict(np.load(fixture_dir / f"{name}.npz")) for name in inference_frames()[:2]]
    clouds = torch.stack([torch.from_numpy(r["point_clouds"][0]) for r in records])
    observation = Observation(
        keys=[SampleKey("graspnet1b", 100 + i, manifest["camera"], i) for i in range(2)],
        points=PointSet.from_dense(clouds),
    ).to(cuda)
    model.eval()
    grasps = model.predict(observation)
    for group, record in zip(grasps, records):
        mine, reference = group.to_graspnet_array(), record["decoded"]
        assert np.array_equal(mine[:, 13:16], reference[:, 13:16])  # same seeds kept
        close("batched/score", mine[:, 0], reference[:, 0], rtol=END_TO_END)
        close("batched/rotation", mine[:, 4:13], reference[:, 4:13], atol=1e-6)


# ----------------------------------------------------------------- step 0: health


@pytest.mark.slow
@pytest.mark.dataset
def test_author_checkpoint_scores_the_same_here_as_in_the_reference(
    fixture_dir, manifest, model, cuda, graspnet1b_root, tmp_path
):
    """Step 0 and the end-to-end check in one: the reference's dumps of a small fixed
    frame set and this model's predictions on the same points, scored by one evaluator."""
    if "health" not in manifest:
        pytest.skip("the fixtures were recorded without --health; regenerate them")
    inputs = np.load(fixture_dir / manifest["health"]["inputs"])
    keys = [
        SampleKey("graspnet1b", int(scene), manifest["camera"], int(index))
        for scene, index, *_ in inputs["frames"]
    ]
    model.eval()
    evaluator = EVALUATORS.build({"type": "GraspNet1BEvaluator"})
    for key, cloud in zip(keys, inputs["point_clouds"]):
        observation = Observation(
            keys=[key], points=PointSet.from_dense(torch.from_numpy(cloud)[None])
        ).to(cuda)
        (grasps,) = model.predict(observation)
        target = tmp_path / f"scene_{key.scene:04d}" / key.camera
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / f"{key.frame:04d}.npy", grasps.to_graspnet_array())
    scores = {}
    for name, directory in (
        ("reference", fixture_dir / manifest["health"]["variants"]["raw"]),
        ("grasppanda", tmp_path),
    ):
        result = evaluator.evaluate(str(directory), keys, num_workers=2)
        assert result.complete, result.missing
        scores[name] = float(np.stack([result.precision[key] for key in keys]).mean() * 100)
    _report["health/AP_raw_reference"] = {"count": round(scores["reference"], 3)}
    _report["health/AP_raw_grasppanda"] = {"count": round(scores["grasppanda"], 3)}
    print(f"\nstep 0, {len(keys)} frames, no collision filter: {scores}")
    assert scores["reference"] > 45.0, "the author checkpoint is not healthy in the reference"
    assert abs(scores["grasppanda"] - scores["reference"]) < 0.3


# ----------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def training(fixture_dir, manifest, model, cuda):
    """The recorded training batch pushed through the stages and every loss, with the
    scale histogram the reference read from its repository."""
    if "training" not in manifest:
        pytest.skip("the fixtures were recorded without a training batch; regenerate them")
    record = dict(np.load(fixture_dir / manifest["training"]["file"]))
    labels = np.load(fixture_dir / manifest["training"]["labels"])
    batch, points = record["point_clouds"].shape[:2]
    objects = []
    for index, number in enumerate(labels["objects_per_frame"].tolist()):
        objects.append([])
        for position in range(number):
            prefix = f"f{index}_o{position}_"
            pose = torch.eye(4)
            pose[:3] = torch.from_numpy(labels[prefix + "pose"])
            objects[-1].append(
                ObjectGraspLabels(
                    object_id=position,
                    pose=pose,
                    points=torch.from_numpy(labels[prefix + "points"]),
                    scores=torch.from_numpy(labels[prefix + "scores"]),  # collisions already zeroed
                    widths=torch.from_numpy(labels[prefix + "widths"]),
                    tolerance=torch.from_numpy(labels[prefix + "tolerance"]),
                )
            )
    histogram = {
        "edges": torch.from_numpy(record["scale_intervals"]),
        "counts": torch.from_numpy(record["scale_counts"].astype(np.int64)),
    }
    observation = Observation(
        keys=[
            SampleKey("graspnet1b", scene, manifest["camera"], index)
            for scene, index in manifest["training"]["frames"]
        ],
        points=PointSet.from_dense(torch.from_numpy(record["point_clouds"])),
    ).to(cuda)
    supervision = Supervision(
        objects=objects,
        point_labels={"objectness": torch.from_numpy(record["objectness_label"]).reshape(-1)},
        extras={HISTOGRAM_KEY: [histogram] * batch},
    ).to(cuda)
    # Training mode updates the running statistics of every batch norm: restore them.
    state = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()
    model.zero_grad()
    out = model.run_stages(observation, supervision)
    losses = {
        **model.proposal.loss(out.point_scores, supervision),
        **model.view.loss(out.approach, out.matched),
        **model.head.loss(out.raw, out.matched),
    }
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
    model.zero_grad()
    model.load_state_dict(state)
    model.eval()
    return record, out, losses, gradients, (batch, points)


@pytest.mark.slow
def test_training_forward_matches(training, model):
    record, out, _, _, (batch, points) = training
    seeds = model.proposal.num_seeds
    offset = torch.arange(batch, device=out.seeds.points.coords.device)[:, None] * points
    index = out.seeds.points.parent_index.reshape(batch, seeds) - offset
    assert np.array_equal(index.cpu().numpy(), record["seed_index"])
    assert bool(out.seeds.valid.all())  # training rejects no seed
    close("train/seed_features", out.seeds.feats.reshape(batch, seeds, -1), record["seed_features"])
    close(
        "train/objectness", out.approach.objectness.reshape(batch, seeds, 2), record["objectness"]
    )
    close("train/view_scores", out.approach.logits.reshape(batch, seeds, -1), record["view_score"])
    views = out.approach.view_index.reshape(batch, seeds).cpu().numpy()
    assert np.array_equal(views, record["view_index"])
    close(
        "train/head_input",
        out.groups.feats.reshape(batch, seeds, 4, -1),
        record["crop"],
        rtol=END_TO_END,
    )
    for name, mine in (
        ("score", out.raw.score),
        ("angle", out.raw.angle),
        ("width", out.raw.width),
        ("tolerance", out.raw.tolerance),
    ):
        close(
            f"train/head_{name}",
            mine.reshape(batch, seeds, 12, 4),
            record[f"{name}_pred"],
            rtol=END_TO_END,
        )


@pytest.mark.slow
def test_label_matching_weights_and_targets_match(training, model, manifest):
    record, out, _, _, (batch, _) = training
    seeds = model.proposal.num_seeds
    matched = out.matched
    assert any(manifest["training"]["mirrored_frames"]), "the fixture is meant to cover a flip"
    close(
        "labels/point",
        matched.label_point.reshape(batch, seeds, 3),
        record["label_point"],
        atol=1e-6,
    )
    close(
        "labels/crop_rotation",
        matched.crop_rotation.reshape(batch, seeds, 3, 3),
        record["crop_rotation"],
        atol=1e-6,
    )
    worst = worst_friction(matched, model.head.score_target, model.head.max_friction)
    score = log_friction_score(matched.friction, matched.width, worst, model.gripper.max_width)
    grid = (batch, seeds, 12, 4)
    close("labels/score_target", score.reshape(grid), record["label_score_target"], atol=1e-6)
    close("labels/width", matched.width.reshape(grid), record["label_width"], rtol=0.0)
    close("labels/tolerance", matched.tolerance.reshape(grid), record["label_tolerance"], rtol=0.0)
    targets = model.view.encode_targets(matched)
    close(
        "labels/view_score",
        targets.view.reshape(batch, seeds, -1),
        record["label_view_score"],
        atol=1e-6,
    )
    # the graspable target: on an object and more than ten graspable views
    graspable = targets.objectness.reshape(batch, seeds).cpu().numpy()
    assert np.array_equal(graspable, record["graspable_label"])
    on_object = int(matched.on_object.sum())
    assert 0 < graspable.sum() < on_object, "the rule must exclude some object seeds"
    count("labels/object_seeds", on_object)
    count("labels/graspable_seeds", int(graspable.sum()))
    # the scale-balanced weights: the table and the entry of every seed, exactly
    counts = torch.from_numpy(record["scale_counts"].astype(np.int64))
    assert np.array_equal(bin_weights(counts).numpy(), record["scale_weights"])
    weight = matched.weight.reshape(batch, seeds).cpu().numpy()
    assert np.array_equal(weight, record["seed_weight"])
    count("labels/distinct_seed_weights", len(np.unique(weight)))
    assert len(np.unique(weight)) > 16, "the batch is meant to cover most width bins"


@pytest.mark.slow
def test_every_loss_term_and_metric_matches(training):
    record, _, losses, _, _ = training

    def reference(name: str) -> float:
        return float(record["scalar_" + name.replace("/", "__")])

    expected = {  # the reference sums graspable + view + 0.2 * (the four grasp terms)
        "loss/objectness": reference("loss/stage1_graspable_loss"),
        "loss/view": reference("loss/stage1_view_loss"),
        "loss/score": 0.2 * reference("loss/stage2_grasp_score_loss"),
        "loss/angle": 0.2 * reference("loss/stage2_grasp_angle_class_loss"),
        "loss/width": 0.2 * reference("loss/stage2_grasp_width_loss"),
        "loss/tolerance": 0.2 * reference("loss/stage2_grasp_tolerance_loss"),
    }
    assert {key for key in losses if key.startswith("loss/")} == set(expected)
    for key, value in expected.items():
        close(key, losses[key], np.float64(value), rtol=1e-5)
    total = sum(float(losses[key]) for key in expected)
    close("loss/total", np.float64(total), np.float64(reference("loss/overall_loss")), rtol=1e-5)
    metrics = {
        "metric/objectness_accuracy": "stage1_graspable_acc",
        "metric/objectness_precision": "stage1_graspable_prec",
        "metric/objectness_recall": "stage1_graspable_recall",
        "metric/angle_accuracy_within_0": "stage2_grasp_angle_class_acc/0_degree",
        "metric/angle_accuracy_within_1": "stage2_grasp_angle_class_acc/15_degree",
        "metric/angle_accuracy_within_2": "stage2_grasp_angle_class_acc/30_degree",
    }
    for key, name in metrics.items():
        close(key, losses[key], np.float64(reference(name)), rtol=1e-3)


@pytest.mark.slow
def test_gradients_match(training, converter):
    record, _, _, gradients, _ = training
    for key in sorted(k for k in record if k.startswith("grad_") and k != "grad_norm"):
        name, expected = converter._translate_baseline(
            key[len("grad_") :], torch.from_numpy(record[key]), "scale-balanced-grasp"
        )
        close(f"grad/{name}", gradients[name], expected.numpy(), rtol=1e-2)
    norm = sum(float(g.double().pow(2).sum()) for g in gradients.values()) ** 0.5
    close("grad/total_norm", np.float64(norm), record["grad_norm"], rtol=1e-4)
