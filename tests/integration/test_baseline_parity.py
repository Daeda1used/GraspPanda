"""GraspNet-baseline against the reference implementation, stage by stage, with the
authors' weights.

The reference values were recorded by ``tools/make_baseline_fixtures.py`` from the
pristine ``graspnet/graspnet-baseline`` (commit ``fb2f089``) in its own environment.
Here the same checkpoint is converted with ``tools/convert_checkpoint.py``, loaded into
the GraspPanda model and fed the recorded inputs.

Regenerate the fixtures (about 0.7 GB, kept under the cache root, never committed)::

    $GRASPPANDA_ORACLE_PYTHON -B tools/make_baseline_fixtures.py \
        --reference <pristine graspnet-baseline> --checkpoint <checkpoint-rs.tar> \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --tolerance-dir <upstream tolerance> \
        --tolerance-objects 0,43,62 --health --out "$GRASPPANDA_CACHE/fixtures/graspnet_baseline"

Tolerances, and why they are what they are.

* Discrete results must be *identical*: the seed indices, the chosen views, which seeds
  are classified as object, the depth bin of every decoded grasp, label matches.
* Activations must agree within ``RTOL = 2e-5`` of the largest reference magnitude of
  the stage, the tolerance of the GSNet parity test. Both networks compute the same
  float32 sums of products in different orders; the largest relative difference
  observed on the recorded frames is 3e-6 in evaluation mode and 8e-6 in training mode.
  For scale: with PyTorch's default TF32 convolutions the reference differs from
  *itself* by 7e-3 on the seed features and picks other views for about ten seeds per
  frame (see ``manifest.json``), which is why the fixtures are recorded with TF32 off.
* Cylinder membership is a threshold on a float32 product. With 20 000 scene points and
  4096 cylinders per frame, a point lands within rounding of a cylinder surface about
  once in a few frames, and the two implementations may then disagree on that point
  (1 of 16 384 crops on the four recorded frames). Such a crop must be explained by
  boundary points in double precision; everything after the crop is compared with the
  reference's neighbour lists injected, and end to end on the seeds whose crops agree.
* Counts of predictions above a threshold may differ by a few units for the same reason.
* The total gradient norm is compared at 1e-4 and single gradient tensors at 1e-2 of
  their largest entry, as for GSNet: the backward pass sums signed terms over all points.

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
from grasppanda.models.heads.friction import log_friction_score, worst_friction
from grasppanda.models.postprocessors.collision import ModelFreeCollisionFilter, voxel_down_sample
from grasppanda.registry import DETECTORS, EVALUATORS
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
MODEL_CONFIG = "_base_/models/graspnet_baseline.yaml"
RTOL = 2e-5
#: Reference: tensors and trainable parameters of the upstream GraspNet, counted there.
REFERENCE_TENSORS = 162
REFERENCE_PARAMETERS = 1_025_964
#: Half-width, in metres, of the band around a cylinder surface inside which float32
#: implementations may disagree about a point (coordinates are below 1.5 m).
BOUNDARY = 1e-6

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_support(observation: Observation) -> PointFeatures:
    """What the detector hands a scene-support grouper under a sub-sampling backbone."""
    points = observation.points
    return PointFeatures(points.coords.new_zeros(len(points), 0), points)


def explained_by_boundary(grouper, cloud, centre, rotation, hmax, listed) -> bool:
    """Whether a neighbour list is a valid answer once boundary points may fall either way.

    ``listed`` is the answer of one implementation for one cylinder: the first
    ``nsample`` members in index order, padded with the first. Membership is recomputed
    in double precision; points within ``BOUNDARY`` of the cylinder surface are free.
    """
    local = (cloud.double() - centre.double()) @ rotation.double()
    axial, radial = local[:, 0], local[:, 1:].norm(dim=1)
    margin = torch.stack([axial - grouper.hmin, hmax - axial, grouper.radius - radial]).amin(0)
    strict, free = margin > BOUNDARY, margin.abs() <= BOUNDARY
    members = torch.unique(listed)
    if not bool((strict | free)[members].all()):
        # the only legal non-member is point 0, standing for "no member at all"
        return members.tolist() == [0] and not bool(strict.any())
    full = len(members) == grouper.nsample
    upto = int(members.max()) if full else len(cloud) - 1
    required = torch.nonzero(strict[: upto + 1]).flatten()
    return bool(torch.isin(required, members).all())


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Path:
    directory = asset_cache / "fixtures" / "graspnet_baseline"
    if not (directory / "manifest.json").is_file():
        pytest.skip(
            f"no GraspNet-baseline reference fixtures at {directory}; record them with "
            "tools/make_baseline_fixtures.py (see this module's docstring)"
        )
    yield directory
    if _report:
        (directory / "parity_report.json").write_text(json.dumps(_report, indent=2) + "\n")
        print("\nGraspNet-baseline parity (max abs difference, relative to the largest value):")
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
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the checkpoint the fixtures were recorded with is missing: {checkpoint}")
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    state, meta = converter.convert_file(
        checkpoint, camera=manifest["camera"], source_format="graspnet-baseline"
    )
    assert meta["model_config"] == MODEL_CONFIG
    assert meta["parameters"] == manifest["parameters"] == REFERENCE_PARAMETERS
    assert meta["tensors"] == REFERENCE_TENSORS
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
        # float64, as the reference test script hands the cloud to its collision detector
        full_points=PointSet(torch.from_numpy(record["full_cloud"])),
    )
    return record, observation


@pytest.fixture(scope="module")
def inference(frame, model, cuda):
    """The natural run of one frame, its crops redone with the reference's neighbours,
    and the activation between the hidden layers and the output layer of the view stage."""
    record, observation = frame
    observation = observation.to(cuda)
    model.eval()
    taps: dict[str, torch.Tensor] = {}
    hook = model.view.output.register_forward_pre_hook(
        lambda _module, args: taps.__setitem__("hidden", args[0])
    )
    with torch.no_grad():
        natural = model.run_stages(observation)
        hook.remove()
        support = scene_support(observation)
        centres, rotation = natural.seeds.points.coords, natural.approach.rotation
        neighbours = model.grouper.neighbours(support, centres, rotation)
        # recorded per (seed, depth); the grouper has an axis for its radii in between
        reference = torch.from_numpy(record["cylinder_index"]).to(cuda).long()[:, :, None]
        groups = model.grouper(support, centres, rotation, neighbours=reference)
        replayed = model.head(groups)
    return record, observation, natural, taps, neighbours, reference, groups, replayed


# -------------------------------------------------------------------------- tests


def test_seeds_are_identical_and_seed_features_match(inference, model):
    record, _, natural, _, _, _, _, _ = inference
    seeds = natural.seeds
    assert seeds.source == "backbone_subsample" and len(seeds) == model.proposal.num_seeds
    assert np.array_equal(seeds.points.parent_index.cpu().numpy(), record["seed_index"][0])
    assert np.array_equal(seeds.points.coords.cpu().numpy(), record["seed_xyz"][0])
    close("backbone/seed_features", natural.feats.feats, record["seed_features"][0])
    assert seeds.feats is natural.feats.feats  # the view stage adds no residual


def test_objectness_and_views_match(inference, model):
    record, _, natural, taps, _, _, _, _ = inference
    approach = natural.approach
    close("view/hidden", taps["hidden"], record["approach_hidden"][0])
    close("view/objectness", approach.objectness, record["objectness"][0])
    close("view/scores", approach.logits, record["view_score"][0])
    assert np.array_equal(approach.view_index.cpu().numpy(), record["view_index"][0])
    close("view/approach", approach.direction, -record["view_xyz"][0], atol=1e-7)
    close("view/rotation", approach.rotation, record["view_rot"][0], atol=1e-6)
    is_object = natural.seeds.valid.cpu().numpy()
    assert np.array_equal(is_object, record["objectness_mask"])
    probability = torch.from_numpy(record["objectness"][0]).softmax(dim=-1)[:, 1]
    close("view/objectness_probability", natural.seeds.scores["objectness"], probability)


def test_cylinder_neighbours_agree_up_to_boundary_points(inference, model):
    _, observation, natural, _, neighbours, reference, _, _ = inference
    assert tuple(neighbours.shape) == (1, 1024, 1, 4, 64) == tuple(reference.shape)
    differs = (neighbours != reference).any(dim=-1)[0, :, 0]
    count("group/crops", differs.numel())
    count("group/crops_with_other_neighbours", int(differs.sum()))
    assert int(differs.sum()) <= 4, "more boundary cases than rounding explains"
    cloud = observation.points.coords
    for seed, depth in torch.nonzero(differs).tolist():
        arguments = (
            model.grouper,
            cloud,
            natural.seeds.points.coords[seed],
            natural.approach.rotation[seed],
            model.grouper.hmax[depth],
        )
        assert explained_by_boundary(*arguments, neighbours[0, seed, 0, depth]), (seed, depth)
        assert explained_by_boundary(*arguments, reference[0, seed, 0, depth]), (seed, depth)
    # a crop of the wrong cylinder is not explained (the check is not vacuous)
    wrong = (
        model.grouper,
        cloud,
        natural.seeds.points.coords[0],
        natural.approach.rotation[0],
        model.grouper.hmax[0],
    )
    assert explained_by_boundary(*wrong, neighbours[0, 0, 0, 0])
    assert not explained_by_boundary(*wrong, neighbours[0, 500, 0, 3])


def test_crops_and_head_match_with_reference_neighbours(inference, model):
    """Independent of boundary points: the reference's neighbour lists are injected."""
    record, observation, natural, _, _, reference, groups, replayed = inference
    assert record["cylinder_index"].shape == tuple(reference[:, :, 0].shape)
    support = scene_support(observation)
    centres, rotation = natural.seeds.points.coords, natural.approach.rotation
    grouped = model.grouper.group(support, centres, rotation, reference)
    close("group/local_xyz", grouped[:, 0], record["grouped"][0], atol=1e-7)
    assert groups.axes == ("seed", "depth", "channel")
    close("group/pooled", groups.feats, record["crop"][0])
    close("head/score", replayed.score, record["score_pred"][0])
    close("head/angle", replayed.angle, record["angle_pred"][0])
    close("head/width", replayed.width, record["width_pred"][0])
    close("head/tolerance", replayed.tolerance, record["tolerance_pred"][0])


def test_decoded_grasps_match_end_to_end(inference, model):
    """No injection: inputs in, grasps out. Seeds whose crops hold a boundary point are
    compared for position and approach only."""
    record, observation, natural, _, neighbours, reference, _, _ = inference
    (grasps,) = model.decode(natural, observation)
    assert grasps.key == observation.keys[0]
    assert grasps.score_kind == "predicted_quality_x_tolerance"
    mine, expected = grasps.to_graspnet_array(), record["decoded"]
    assert mine.shape == expected.shape and mine.shape[0] == int(record["objectness_mask"].sum())
    assert np.array_equal(mine[:, 13:16], expected[:, 13:16]), "translation"
    assert np.array_equal(mine[:, 2], expected[:, 2]), "height"
    assert np.array_equal(mine[:, 16], expected[:, 16]), "object id"
    exact = ~(neighbours != reference).flatten(2).any(dim=-1)[0][natural.seeds.valid]
    exact = exact.cpu().numpy()
    count("decoded/grasps", len(mine))
    count("decoded/grasps_with_a_boundary_crop", int((~exact).sum()))
    close("decoded/score", mine[exact, 0], expected[exact, 0])
    close("decoded/width", mine[exact, 1], expected[exact, 1], atol=1e-6)
    assert np.array_equal(mine[exact, 3], expected[exact, 3]), "depth"
    close("decoded/rotation", mine[exact, 4:13], expected[exact, 4:13], atol=1e-6)
    # the approach axis does not depend on the crop
    close("decoded/approach", mine[:, [4, 7, 10]], expected[:, [4, 7, 10]], atol=1e-6)


def test_collision_filter_makes_the_same_decisions(inference, model):
    record, observation, _, _, _, _, _, _ = inference
    collision = ModelFreeCollisionFilter()  # voxel 0.01, approach 0.05, threshold 0.01
    scene = voxel_down_sample(observation.full_points.coords, collision.voxel_size)
    reference_scene = torch.from_numpy(record["collision_scene"]).to(scene)
    assert scene.shape == reference_scene.shape
    # The reference's own grasps: isolates the filter from the network.
    reference = GraspGroup.from_graspnet_array(record["decoded"]).to(scene.device)
    mask = collision.collision_mask(reference, scene)
    assert np.array_equal(mask.cpu().numpy(), record["collision_mask"])
    count("collision/grasps", len(mask))
    count("collision/in_collision", int(mask.sum()))


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
        mine, reference = group.to_graspnet_array(), record["decoded"]
        assert np.array_equal(mine[:, 13:16], reference[:, 13:16])  # same seeds kept
        close("batched/score", mine[:, 0], reference[:, 0])
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
    assert scores["reference"] > 25.0, "the author checkpoint is not healthy in the reference"
    assert abs(scores["grasppanda"] - scores["reference"]) < 0.3


# ----------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def training(fixture_dir, manifest, model, cuda):
    """The recorded training batch pushed through the stages and every loss."""
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
    support = scene_support(observation)
    with torch.no_grad():
        neighbours = model.grouper.neighbours(
            support, out.matched.crop_centre, out.matched.crop_rotation
        )
    model.zero_grad()
    model.load_state_dict(state)
    model.eval()
    return record, out, losses, gradients, neighbours, (batch, points)


@pytest.mark.slow
def test_training_forward_matches(training, model):
    record, out, _, _, neighbours, (batch, points) = training
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
    assert np.array_equal(views, record["view_index"])  # arg-max in training too
    reference = torch.from_numpy(record["cylinder_index"]).to(neighbours.device)[:, :, None]
    differs = int((neighbours != reference).any(dim=-1).sum())
    count("train/crops_with_other_neighbours", differs)
    if differs:
        pytest.skip("a boundary point changed a training crop; batch statistics then differ")
    close("train/group_pooled", out.groups.feats.reshape(batch, seeds, 4, -1), record["crop"])
    for name, mine in (
        ("score", out.raw.score),
        ("angle", out.raw.angle),
        ("width", out.raw.width),
        ("tolerance", out.raw.tolerance),
    ):
        close(f"train/head_{name}", mine.reshape(batch, seeds, 12, 4), record[f"{name}_pred"])


@pytest.mark.slow
def test_label_matching_matches(training, model, manifest):
    record, out, _, _, _, (batch, _) = training
    seeds = model.proposal.num_seeds
    matched = out.matched
    assert any(manifest["training"]["mirrored_frames"]), "the fixture is meant to cover a flip"
    on_object = record["objectness_label"][np.arange(batch)[:, None], record["seed_index"]] > 0
    assert np.array_equal(matched.on_object.reshape(batch, seeds).cpu().numpy(), on_object)
    assert np.array_equal(matched.valid.reshape(batch, seeds).cpu().numpy(), on_object)
    close(
        "labels/point",
        matched.label_point.reshape(batch, seeds, 3),
        record["label_point"],
        atol=1e-6,
    )
    # the training crop sits at the label point and uses the label frame
    close(
        "labels/crop_centre",
        matched.crop_centre.reshape(batch, seeds, 3),
        record["crop_centre"],
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


@pytest.mark.slow
def test_every_loss_term_and_metric_matches(training):
    record, _, losses, _, _, _ = training

    def reference(name: str) -> float:
        return float(record["scalar_" + name.replace("/", "__")])

    expected = {  # the reference sums objectness + view + 0.2 * (the four grasp terms)
        "loss/objectness": reference("loss/stage1_objectness_loss"),
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
        "metric/objectness_accuracy": "stage1_objectness_acc",
        "metric/objectness_precision": "stage1_objectness_prec",
        "metric/objectness_recall": "stage1_objectness_recall",
        "metric/angle_accuracy_within_0": "stage2_grasp_angle_class_acc/0_degree",
        "metric/angle_accuracy_within_1": "stage2_grasp_angle_class_acc/15_degree",
        "metric/angle_accuracy_within_2": "stage2_grasp_angle_class_acc/30_degree",
    }
    for key, name in metrics.items():
        close(key, losses[key], np.float64(reference(name)), rtol=1e-3)
    # a count of predictions above a threshold: boundary cases may fall either way
    good = float(losses["metric/good_view_predictions"])
    assert abs(good - reference("stage1_pos_view_pred_count")) <= 5


@pytest.mark.slow
def test_gradients_match(training, converter):
    record, _, _, gradients, _, _ = training
    for key in sorted(k for k in record if k.startswith("grad_") and k != "grad_norm"):
        name, expected = converter._translate_baseline(
            key[len("grad_") :], torch.from_numpy(record[key])
        )
        close(f"grad/{name}", gradients[name], expected.numpy(), rtol=1e-2)
    norm = sum(float(g.double().pow(2).sum()) for g in gradients.values()) ** 0.5
    close("grad/total_norm", np.float64(norm), record["grad_norm"], rtol=1e-4)
