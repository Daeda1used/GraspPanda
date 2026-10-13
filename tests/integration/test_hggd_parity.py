"""HGGD against the reference implementation, stage by stage, with the authors' weights.

The reference values were recorded by ``tools/make_hggd_fixtures.py`` from
``THU-VCLab/HGGD`` (commit ``69580ff``) in its own environment (torch 1.11, pytorch3d,
torchvision, numba, cupoch). Here the same checkpoints are converted with
``tools/convert_checkpoint.py``, loaded into the GraspPanda model and fed the recorded
inputs.

Regenerate the fixtures (about 240 MB, kept under the cache root, never committed)::

    $GRASPPANDA_ORACLE_PYTHON -B tools/make_hggd_fixtures.py --training \
        --reference <HGGD checkout> --checkpoint <checkpoint-realsense.tar> \
        --dataset-root "$GRASPPANDA_GRASPNET1B_ROOT" --camera realsense \
        --out "$GRASPPANDA_CACHE/fixtures/hggd"          # and the same with kinect

How each stage is compared, and why.

* **Networks.** Same float32 sums of products through cuDNN in two torch versions;
  activations must agree within ``RTOL = 2e-5`` of the stage's largest value (observed:
  below 2e-6).
* **Discrete results must be identical**: candidate pixels, the selected partial grasps'
  pixels, the number of scene points in every region, which regions are kept, the
  voxels of the collision cloud, every collision decision and the grasps NMS keeps.
* **Double-precision chains** (lifting, decoding) are compared at ``1e-6`` and ``1e-9``:
  the only inputs that differ are the camera constants held as float32 here.
* **Library semantics reproduced, not imported.** pytorch3d's nearest-neighbour pooling
  and its furthest point sampling, torchvision's blur, cupoch's voxel grid and the numba
  loops are re-implemented on ``grasppanda.ops`` and plain PyTorch. Two of them choose
  between candidates by a float32 comparison and therefore cannot be bit-identical to
  another kernel: the 8th-nearest feature cell of a point, and the next sample of the
  furthest point sampling. Their agreement is measured and asserted as a rate.
* **Replayed, because they are random in the reference**: the 25 600 sampled scene
  points (the recorded cloud is the input), and per region the re-sampling indices, the
  0.5 mm jitter and the start point of the sampling. With them replayed, the regions
  must contain the same points.
* **Training.** Dropout is replayed through the recorded mask. Region features in
  training mode pass batch norm over all regions of the batch, so a single region with
  another sample order moves every activation a little; the region network is
  therefore also compared on the recorded regions, where it has to match tightly.

Every comparison is also written to ``parity_report_<camera>.json`` next to the fixtures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from grasppanda.config import Config
from grasppanda.models.detectors.hggd import PartialGraspSet, RegionDraws, RegionSet
from grasppanda.models.detectors.hggd.region import attach_features, cell_points
from grasppanda.registry import DETECTORS, POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation, PointSet, SampleKey

pytestmark = [pytest.mark.gpu, pytest.mark.oracle]

REPO = Path(__file__).resolve().parents[2]
RTOL = 2e-5
#: Trainable parameters of the reference networks (``AnchorGraspNet``, ``PointMultiGraspNet``).
REFERENCE_PARAMETERS = {"image_net": 1_618_921, "local_net": 1_806_669}
REFERENCE_TENSORS = 297 + 76 + 2  # two state dicts and the two anchor vectors
SIZE = (640, 360)
CAMERAS = ("realsense", "kinect")
#: The reference test script's collision filter and NMS, as configured in configs/hggd/.
COLLISION = {
    "type": "ModelFreeCollisionFilter",
    "cloud": "points",
    "voxel_size": 0.01,
    "approach_distance": 0.05,
    "finger_length": 0.04,
    "height": 0.01,
    "depth": 0.02,
    "collision_threshold": 0.0,
    "min_points_between": 2,
    "precision": "float16",
}

_reports: dict[str, dict[str, dict[str, float]]] = {}


def note(camera: str, stage: str, **values: float) -> None:
    """Record a measured rate or count in the report."""
    entry = _reports.setdefault(camera, {}).setdefault(stage, {})
    for name, value in values.items():
        entry[name] = entry.get(name, 0) + value


def close(camera: str, stage: str, mine, reference, *, rtol: float = RTOL, atol: float = 0.0):
    """Assert ``|mine - reference| <= atol + rtol * max|reference|`` and log the numbers."""
    mine = np.asarray(mine.detach().cpu() if isinstance(mine, torch.Tensor) else mine, np.float64)
    reference = np.asarray(reference, np.float64)
    assert mine.shape == reference.shape, f"{stage}: shape {mine.shape} vs {reference.shape}"
    scale = float(np.abs(reference).max()) if reference.size else 0.0
    worst = float(np.abs(mine - reference).max()) if reference.size else 0.0
    entry = _reports.setdefault(camera, {}).setdefault(
        stage, {"max_abs": 0.0, "max_rel": 0.0, "scale": 0.0, "count": 0}
    )
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


def rows16(points: np.ndarray) -> np.ndarray:
    """Rows of an ``(N, 3)`` array rounded to half floats, as a sorted structured array."""
    half = np.ascontiguousarray(points.astype(np.float16))
    return np.sort(half.view([("x", "f2"), ("y", "f2"), ("z", "f2")]).ravel())


# ----------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def fixture_dir(asset_cache) -> Path:
    directory = asset_cache / "fixtures" / "hggd"
    if not any((directory / f"manifest_{camera}.json").is_file() for camera in CAMERAS):
        pytest.skip(
            f"no HGGD reference fixtures at {directory}; record them with "
            "tools/make_hggd_fixtures.py (see this module's docstring)"
        )
    yield directory
    for camera, report in _reports.items():
        (directory / f"parity_report_{camera}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"\nHGGD parity, {camera} (max abs difference, relative to the stage's largest value):"
        )
        for stage, entry in report.items():
            if "max_abs" in entry:
                print(f"  {stage:38s} {entry['max_abs']:.3e}   {entry['max_rel']:.2e}")
            else:
                print(f"  {stage:38s} {entry}")


@pytest.fixture(scope="module")
def converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=CAMERAS)
def reference(request, fixture_dir, converter, cuda):
    """Manifest of one camera's fixtures and the model with that camera's checkpoint."""
    camera = request.param
    file = fixture_dir / f"manifest_{camera}.json"
    if not file.is_file():
        pytest.skip(f"no fixtures for the {camera} checkpoint at {fixture_dir}")
    manifest = json.loads(file.read_text())
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip(f"the checkpoint the fixtures were recorded with is missing: {checkpoint}")
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    state, meta = converter.convert_file(checkpoint, camera=camera, source_format="hggd")
    model = DETECTORS.build(Config.load("_base_/models/hggd.yaml")["model"])
    model.load_state_dict(state, strict=True)
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield camera, manifest, model.to(cuda), state, meta
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def observation_of(record: dict, keys: list[SampleKey], device) -> Observation:
    """The observation the recorded frame(s) correspond to (colour enters as ``image``)."""
    depth = torch.as_tensor(record["depth"], device=device)
    depth = depth[None] if depth.ndim == 2 else depth
    points = torch.as_tensor(record["points"], device=device)
    points = points[None] if points.ndim == 2 else points
    batch = len(keys)
    return Observation(
        keys=keys,
        intrinsics=torch.as_tensor(record["intrinsics"], dtype=torch.float32, device=device)[
            None
        ].repeat(batch, 1, 1),
        depth=depth.transpose(1, 2) / 1000.0,  # recorded (W, H) millimetres
        rgb=torch.zeros(batch, 3, depth.shape[2], depth.shape[1], device=device),
        points=PointSet.from_dense(points),
    )


def draws_of(record: dict, call: int, device) -> RegionDraws:
    return RegionDraws(
        resample=list(torch.as_tensor(record[f"resample_{call}"], device=device)),
        jitter=torch.as_tensor(record[f"jitter_{call}"], device=device),
        start=torch.as_tensor(record[f"fps_start_{call}"], device=device),
    )


def partial_grasps(record: dict, prefix: str, device, centre=None) -> PartialGraspSet:
    count = len(record[f"{prefix}theta"])
    tensor = lambda name: torch.as_tensor(record[f"{prefix}{name}"], device=device)  # noqa: E731
    return PartialGraspSet(
        frame=torch.zeros(count, dtype=torch.long, device=device),
        pixel=tensor("pixel").long(),
        theta=tensor("theta"),
        width=tensor("width"),
        depth_offset=tensor("depth_offset") / 1000.0,  # recorded in millimetres
        score=tensor("score"),
        centre=None if centre is None else torch.as_tensor(centre, device=device),
    )


@pytest.fixture(scope="module")
def frames(reference, fixture_dir, cuda):
    """Every recorded test frame of the camera: record, observation and a natural run."""
    camera, manifest, model, _, _ = reference
    model.eval()
    out = []
    for entry in manifest["inference"]:
        record = dict(np.load(fixture_dir / entry["file"]))
        key = SampleKey("graspnet1b", entry["scene"], camera, entry["frame"])
        observation = observation_of(record, [key], cuda)
        image = torch.as_tensor(record["image"], device=cuda)[None]
        with torch.no_grad():
            stages = model.run_stages(observation, image=image, draws=[draws_of(record, 0, cuda)])
        out.append((record, observation, stages))
    assert out, "the manifest lists no inference frame; regenerate the fixtures"
    return out


# ------------------------------------------------------------------- the checkpoint


def test_checkpoint_converts_strictly_and_completely(reference):
    camera, manifest, model, _, meta = reference
    assert meta["tensors"] == REFERENCE_TENSORS == len(model.state_dict())
    for part, expected in REFERENCE_PARAMETERS.items():
        assert sum(p.numel() for p in getattr(model, part).parameters()) == expected
    assert sum(p.numel() for p in model.parameters()) == sum(REFERENCE_PARAMETERS.values())
    original = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=True)
    for name in ("gamma", "beta"):  # the learned anchors travel with the weights
        anchors = getattr(model.anchors, name)
        assert anchors.dtype == torch.float64
        assert torch.equal(anchors.cpu(), original[name].double().cpu())
    assert not torch.equal(model.anchors.gamma, torch.linspace(-6 / 7, 6 / 7, 7).to(anchors))
    note(camera, "checkpoint", tensors=meta["tensors"], parameters=meta["parameters"])


# ----------------------------------------------------------------------- inference


def test_image_network_matches(reference, frames):
    camera, _, model, _, _ = reference
    for record, _, stages in frames:
        taps = model.image_net.encode(stages.image)
        for level, tap in enumerate(taps):
            close(camera, f"encoder stride {2 ** (level + 1)}", tap, record[f"backbone_{level}"])
        close(camera, "heat-map logits", stages.raw.heatmap[0], record["heatmap_logits"])
        close(camera, "angle logits", stages.raw.angle[0], record["angle_logits"])
        close(camera, "angle offsets", stages.raw.angle_offset[0], record["angle_offset_raw"])
        close(camera, "depth offsets", stages.raw.depth_offset[0], record["depth_offset_raw"])
        close(camera, "width offsets", stages.raw.width_offset[0], record["width_offset_raw"])
        close(camera, "feature map", stages.raw.features[0], record["features"])
        assert not stages.raw.features.requires_grad


def test_activated_maps_match_and_candidates_are_identical(reference, frames):
    camera, _, model, _, _ = reference
    for record, _, stages in frames:
        close(camera, "blurred heat-map", stages.maps.heatmap[0], record["heatmap"])
        close(camera, "angle scores", stages.maps.angle[0], record["angle_score"])
        close(camera, "clamped angle offsets", stages.maps.angle_offset[0], record["angle_offset"])
        close(camera, "clamped depth offsets", stages.maps.depth_offset[0], record["depth_offset"])
        close(camera, "clamped width offsets", stages.maps.width_offset[0], record["width_offset"])
        mine = model.proposal.candidates(stages.maps.heatmap[0], len(record["candidates"]))
        assert np.array_equal(mine.cpu().numpy(), record["candidates"])
        # ... and from the reference's own heat-map, to separate the stage from the network
        heat = torch.as_tensor(record["heatmap"], device=mine.device)
        again = model.proposal.candidates(heat, len(record["candidates"]))
        assert np.array_equal(again.cpu().numpy(), record["candidates"])
        note(camera, "candidate pixels identical", frames=1, pixels=len(mine))


def test_partial_grasps_are_the_reference_ones(reference, frames):
    camera, _, model, _, _ = reference
    for record, _, stages in frames:
        proposals, _ = model.proposal(stages.raw, training=False)
        assert np.array_equal(proposals.pixel.cpu().numpy(), record["proposal_pixel"])
        close(camera, "proposal theta", proposals.theta, record["proposal_theta"], rtol=1e-6)
        close(camera, "proposal width (px)", proposals.width, record["proposal_width"], rtol=2e-6)
        close(
            camera,
            "proposal depth offset (mm)",
            proposals.depth_offset * 1000,
            record["proposal_depth_offset"],
            rtol=5e-6,
        )
        close(camera, "proposal score", proposals.score, record["proposal_score"], rtol=5e-6)
        assert proposals.theta.dtype == torch.float64 and proposals.centre is None
        note(camera, "partial grasps identical pixels", frames=1, grasps=len(proposals))


def test_image_features_reach_the_points(reference, frames):
    """pytorch3d's nearest-cell pooling, rebuilt on ``grasppanda.ops.knn``."""
    camera, _, model, _, _ = reference
    for record, observation, _ in frames:
        cells = cell_points(observation.depth[0], observation.intrinsics[0], (80, 45))
        close(camera, "feature-cell points", cells.permute(2, 0, 1), record["xyz_map"], rtol=1e-6)
        features = torch.as_tensor(record["features"], device=cells.device)
        points = observation.points.coords
        attached = attach_features(points, cells, features, model.region.feature_neighbours)
        different = (attached.cpu().numpy() != record["fused"][:, 3:]).any(axis=1)
        # A point whose 8th and 9th nearest cells are equally far in float32 may pool
        # another cell than pytorch3d's kernel did.
        assert different.mean() < 1e-3, f"{different.sum()} of {len(different)} points differ"
        note(
            camera,
            "points with other pooled features",
            points=len(different),
            different=int(different.sum()),
        )
        # With the reference's own neighbours the pooled features are identical.
        index = torch.as_tensor(record["knn_index"], device=cells.device).long()
        pooled = features.reshape(features.shape[0], -1).T[index].amax(dim=1)
        assert np.array_equal(pooled.cpu().numpy(), record["fused"][:, 3:])


def test_centres_are_lifted_like_the_reference(reference, frames):
    camera, _, model, _, _ = reference
    for record, observation, _ in frames:
        grasps = partial_grasps(record, "proposal_", observation.depth.device)
        centre = model.region.lift(grasps, observation.depth[0], observation.intrinsics[0], SIZE)
        close(camera, "lifted centres", centre, record["lifted_centre"], rtol=1e-6)
        u, v = grasps.pixel[:, 0] * 2, grasps.pixel[:, 1] * 2
        without_depth = int((observation.depth[0][v, u] == 0).sum())
        note(camera, "centres on pixels without depth", frames=1, centres=without_depth)
        in_place = type(model.region)({"zero_depth_centres": "in_place"})
        moved = in_place.lift(grasps, observation.depth[0], observation.intrinsics[0], SIZE)
        if without_depth and (observation.depth[0][v, u] == 0)[:-without_depth].any():
            # the reference's order really differs from keeping every centre in place
            assert not torch.allclose(moved, centre)


def test_regions_hold_the_reference_points_with_replayed_draws(reference, frames):
    """pytorch3d's furthest point sampling, rebuilt on ``grasppanda.ops``."""
    camera, _, model, _, _ = reference
    for record, observation, stages in frames:
        device = observation.depth.device
        grasps = partial_grasps(record, "proposal_", device)
        centre = torch.as_tensor(record["lifted_centre"], device=device)
        radius = model.region.radius(grasps, centre, observation.intrinsics[0], 2.0)
        scene = torch.as_tensor(record["fused"], device=device)
        regions, kept, support = model.region.group(
            scene, centre, radius, draws_of(record, 0, device)
        )
        lengths = torch.where(support < 512, torch.full_like(support, 512), support)
        assert np.array_equal(lengths.cpu().numpy(), record["fps_lengths_0"])  # ball membership
        assert int(kept.sum()) == len(record["region"])
        close(camera, "region centres", centre[kept], record["region_centre"], rtol=0, atol=0)
        mine, theirs = regions.cpu().numpy(), record["region"]
        same_order = (mine == theirs).reshape(len(mine), -1).all(axis=1)
        same_set = (np.sort(mine[..., 0], axis=1) == np.sort(theirs[..., 0], axis=1)).all(axis=1)
        # A near-tie between two candidates may be resolved the other way by another kernel.
        assert same_set.mean() >= 0.95, f"{(~same_set).sum()} of {len(same_set)} regions differ"
        note(
            camera,
            "regions",
            regions=len(mine),
            same_points=int(same_set.sum()),
            same_order=int(same_order.sum()),
        )
        # The natural run (own features, own lifting) ends in the same regions.
        assert len(stages.regions) == len(theirs)
        close(
            camera,
            "natural region centres",
            stages.regions.grasps.centre,
            record["region_centre"],
            rtol=1e-6,
        )


def test_region_network_matches_on_the_reference_regions(reference, frames):
    camera, _, model, _, _ = reference
    for record, observation, _ in frames:
        device = observation.depth.device
        regions = torch.as_tensor(record["region"], device=device)
        info = torch.as_tensor(record["grasp_info"], device=device)
        with torch.no_grad():
            aligned = model.local_net.align(regions[..., :3])
            vector = model.local_net.region_vector(regions)
            logits, offsets = model.local_net(regions, info)
        close(camera, "alignment matrices", aligned, record["stn"])
        close(camera, "region vectors", vector, record["local_feature"])
        close(camera, "anchor logits", logits, record["anchor_logits"])
        close(camera, "raw centre offsets", offsets, record["offset_raw"])


def reference_regions(record: dict, device) -> RegionSet:
    grasps = partial_grasps(record, "region_", device, centre=record["region_centre"])
    return RegionSet(
        points=torch.as_tensor(record["region"], device=device),
        grasps=grasps,
        kept=torch.ones(len(grasps), dtype=torch.bool, device=device),
        support=torch.zeros(len(grasps), dtype=torch.long, device=device),
    )


def test_decoding_matches_in_double_precision(reference, frames):
    camera, _, model, _, _ = reference
    for record, observation, _ in frames:
        device = observation.depth.device
        intrinsics = torch.as_tensor(record["intrinsics"], device=device)[None]  # float64
        groups = model.anchors.decode(
            torch.as_tensor(record["anchor_logits"], device=device),
            torch.as_tensor(record["offset_raw"], device=device),
            reference_regions(record, device),
            intrinsics,
            2.0,
            observation.keys,
        )
        (group,) = groups
        assert group.key == observation.keys[0] and group.score_kind == "anchor_probability"
        close(
            camera,
            "decoded translation",
            group.translation,
            record["decoded_translation"],
            rtol=1e-9,
        )
        close(camera, "decoded rotation", group.rotation, record["decoded_rotation"], rtol=1e-9)
        close(camera, "decoded width", group.width, record["decoded_width"], rtol=1e-9)
        close(camera, "decoded height", group.height, record["decoded_height"], rtol=1e-9)
        close(camera, "decoded depth", group.depth, record["decoded_depth"], rtol=1e-9)
        assert np.array_equal(group.score.cpu().numpy(), record["decoded_score"])
        array = group.to_graspnet_array(np.float64)
        close(
            camera,
            "decoded (G, 17) array",
            array[:, :16],
            record["decoded_array"][:, :16],
            rtol=1e-9,
        )
        # The reference frame is a rotation only up to |fx - fy| / f: the default
        # tolerance of GraspGroup.validate() rejects it, the head's own accepts it.
        frame = group.rotation
        deviation = (frame.transpose(1, 2) @ frame - torch.eye(3).to(frame)).abs().amax()
        tolerance = model.anchors.rotation_tolerance(intrinsics[0])
        assert 1e-4 < float(deviation) < tolerance < 1e-3
        group.validate(atol=tolerance)
        note(camera, "rotation deviation from orthonormal", frames=1, summed=float(deviation))


def test_predictions_end_to_end_match_where_the_regions_do(reference, frames):
    """The whole chain from the recorded input, with the reference's random draws."""
    camera, _, model, _, _ = reference
    for record, observation, stages in frames:
        (group,) = model.decode(stages, observation)
        assert group.translation.dtype == torch.float32
        same = stages.regions.points.cpu().numpy() == record["region"]
        same = same.reshape(len(same), -1).all(axis=1)
        note(
            camera,
            "end to end",
            frames=1,
            grasps=len(group),
            reference=len(record["decoded_score"]),
        )
        if not same.all():
            continue  # a region sampled in another order; covered by the tests above
        close(camera, "end-to-end anchor logits", stages.logits, record["anchor_logits"])
        mine = group.to_graspnet_array(np.float64)
        theirs = record["decoded_array"]
        if len(mine) == len(theirs):
            close(camera, "end-to-end grasps", mine[:, :16], theirs[:, :16], rtol=0, atol=2e-5)
            note(camera, "end to end", compared=1)


def reference_group(record: dict, key: SampleKey, device) -> GraspGroup:
    tensor = lambda name: torch.as_tensor(record[name], device=device)  # noqa: E731
    return GraspGroup(
        translation=tensor("decoded_translation"),
        rotation=tensor("decoded_rotation"),
        width=tensor("decoded_width"),
        depth=tensor("decoded_depth"),
        score=tensor("decoded_score"),
        height=tensor("decoded_height"),
        key=key,
    )


def test_collision_filter_makes_the_same_decisions(reference, frames):
    """cupoch's voxel grid and the half-float box test, rebuilt in PyTorch."""
    camera, _, _, _, _ = reference
    collision = POSTPROCESSORS.build(COLLISION)
    for record, observation, _ in frames:
        device = observation.depth.device
        grasps = reference_group(record, observation.keys[0], device)
        voxels = collision.down_sample(observation.points.coords)
        # The same voxels. Their centroids are float32 sums, which cupoch reduces in
        # parallel: once in several thousand voxels the last bit, and with it the
        # half-float rounding of one coordinate, comes out differently.
        mine, theirs = rows16(voxels.cpu().numpy()), rows16(record["collision_scene"])
        assert len(mine) == len(theirs)
        assert (mine != theirs).mean() < 1e-3
        rejected = collision.collision_mask(grasps, voxels).cpu().numpy()
        assert np.array_equal(~rejected, record["collision_free"])
        kept = collision(grasps, observation)  # through the post-processor interface
        assert len(kept) == int(record["collision_free"].sum())
        note(
            camera,
            "collision filter",
            grasps=len(rejected),
            same_decisions=int(((~rejected) == record["collision_free"]).sum()),
            voxels=len(mine),
            same_voxels=int((mine == theirs).sum()),
        )
        # The gripper-frame coordinates: exact products rounded to half floats agree with
        # the reference's half-precision product except for a rounding now and then.
        local = torch.as_tensor(record["collision_local"])
        half = torch.float16
        scene = torch.as_tensor(record["collision_scene"])  # the reference's voxel order
        offset = scene[None] - grasps.translation[: len(local), None].cpu().to(half)
        mine = (offset.double() @ grasps.rotation[: len(local)].cpu().to(half).double()).to(half)
        equal = float((mine == local).float().mean())
        assert equal > 0.9999, f"only {equal:.6f} of the half-float coordinates agree"
        note(
            camera,
            "half-float coordinates",
            values=local.numel(),
            different=int((mine != local).sum()),
        )


def test_nms_keeps_the_reference_grasps_in_the_reference_order(reference, frames):
    camera, _, _, _, _ = reference
    nms = POSTPROCESSORS.build(
        {"type": "GraspNMS", "translation_threshold": 0.03, "rotation_threshold": 30.0}
    )
    for record, observation, _ in frames:
        grasps = reference_group(record, observation.keys[0], observation.depth.device)
        free = torch.as_tensor(record["collision_free"], device=grasps.score.device)
        for name, group in (("final_cd", grasps[free]), ("final_raw", grasps)):
            kept = nms(group, observation).to_graspnet_array(np.float64)
            close(camera, f"NMS output ({name})", kept[:, :16], record[name][:, :16], rtol=1e-12)
            note(camera, f"NMS survivors ({name})", grasps=len(kept))


def test_post_processed_dump_is_the_reference_dump(reference, frames):
    """Collision filter then NMS on the reference's decoded grasps: the script's dump."""
    camera, _, _, _, _ = reference
    chain = [POSTPROCESSORS.build(COLLISION), POSTPROCESSORS.build({"type": "GraspNMS"})]
    for record, observation, _ in frames:
        group = reference_group(record, observation.keys[0], observation.depth.device)
        for step in chain:
            group = step(group, observation)
        mine = group.to_graspnet_array(np.float64)
        close(
            camera,
            "dump after collision filter and NMS",
            mine[:, :16],
            record["final_cd"][:, :16],
            rtol=1e-12,
        )


# ------------------------------------------------------------------------ training


class _ReplayedDropout(nn.Module):
    """Multiplies by the mask the reference drew."""

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


def labels_of(record: dict, frame: int, device) -> dict[str, torch.Tensor]:
    """A frame's upstream label file in the units of the view-grasp asset."""
    tensor = lambda name: torch.as_tensor(record[f"{name}_{frame}"], device=device)  # noqa: E731
    return {
        "centers": tensor("centers_2d"),
        "depth": tensor("center_z_depths") / 1000.0,
        "theta": tensor("thetas_rad"),
        "gamma": tensor("gammas_rad"),
        "beta": tensor("betas_rad"),
        "width": tensor("widths_2d"),
        "height": tensor("heights_2d"),
        "score": tensor("scores_from_6d"),
        "object_id": tensor("object_ids").long(),
    }


@pytest.fixture(scope="module")
def training(reference, fixture_dir, cuda):
    """The recorded training batch run through the model, with replayed randomness."""
    camera, manifest, model, state, _ = reference
    if not manifest.get("training"):
        pytest.skip("the fixtures hold no training batch; record with --training")
    record = dict(np.load(fixture_dir / manifest["training"]["file"]))
    files = dict(np.load(fixture_dir / manifest["training"]["labels"]))
    keys = [
        SampleKey("graspnet1b", scene, camera, frame)
        for scene, frame in manifest["training"]["frames"]
    ]
    observation = observation_of(record, keys, cuda)
    labels = [labels_of(files, frame, cuda) for frame in range(len(keys))]
    before, after = record["dropout_before"], record["dropout_after"]
    mask = np.where(before != 0, after / np.where(before != 0, before, 1.0), 1.0).astype(np.float32)
    model.load_state_dict(state, strict=True)  # batch-norm statistics as in the checkpoint
    model.train()
    original = model.local_net.dropout
    model.local_net.dropout = _ReplayedDropout(torch.as_tensor(mask, device=cuda))
    model.zero_grad()
    draws = [draws_of(record, call, cuda) for call in range(int(record["region_calls"]))]
    stages = model.run_stages(
        observation, image=torch.as_tensor(record["image"], device=cuda), draws=draws
    )
    yield camera, model, record, observation, labels, stages
    model.local_net.dropout = original
    model.load_state_dict(state, strict=True)
    model.eval()


@pytest.mark.slow
def test_dense_targets_match(training):
    camera, model, record, observation, labels, _ = training
    targets = model.proposal.encode(labels, observation.depth, SIZE)
    assert np.array_equal(targets.heatmap.cpu().numpy(), record["target_heatmap"])
    assert np.array_equal(targets.angle.cpu().numpy(), record["target_angle"])
    assert np.array_equal(targets.width_offset.cpu().numpy(), record["target_width_offset"])
    close(
        camera,
        "target angle offsets",
        targets.angle_offset,
        record["target_angle_offset"],
        rtol=5e-6,
    )
    # The depth image is metres here and millimetres there: one float32 rounding apart.
    close(
        camera,
        "target depth offsets",
        targets.depth_offset,
        record["target_depth_offset"],
        rtol=2e-5,
    )
    note(
        camera,
        "dense targets",
        labels=sum(len(entry["theta"]) for entry in labels),
        peaks=int((targets.heatmap == 1).sum()),
        supervised_cells=int((targets.angle > 0).sum()),
    )


@pytest.mark.slow
def test_training_forward_and_image_losses_match(training):
    camera, model, record, observation, labels, stages = training
    close(camera, "train heat-map logits", stages.raw.heatmap, record["heatmap_logits"])
    close(camera, "train angle logits", stages.raw.angle, record["angle_logits"])
    close(camera, "train feature map", stages.raw.features, record["features"])
    close(camera, "train blurred heat-map", stages.maps.heatmap, record["heatmap"])
    targets = model.proposal.encode(labels, observation.depth, SIZE)
    losses = model.proposal.loss(stages.raw, targets)
    close(camera, "loss/heatmap", losses["loss/heatmap"], record["loss_heatmap"], rtol=1e-5)
    close(camera, "loss/offsets", losses["loss/offsets"], record["loss_regression"], rtol=1e-5)
    close(camera, "loss/angle", losses["loss/angle"], record["loss_angle"], rtol=1e-5)


@pytest.mark.slow
def test_training_proposals_regions_and_matches_are_the_reference_ones(training):
    camera, model, record, observation, labels, stages = training
    frames = len(observation.keys)
    assert stages.proposals.tolist() == [len(record[f"proposal_{f}_theta"]) for f in range(frames)]
    grasps = stages.regions.grasps
    centres = np.concatenate([record[f"region_centre_{f}"] for f in range(frames)])
    thetas = np.concatenate([record[f"region_{f}_theta"] for f in range(frames)])
    assert len(stages.regions) == len(record["region"]) == len(centres)
    close(camera, "train region centres", grasps.centre, centres, rtol=1e-6)
    close(camera, "train region theta", grasps.theta, thetas, rtol=1e-6)
    mine, theirs = stages.regions.points.detach().cpu().numpy(), record["region"]
    # Centres are 1e-7 apart here (own lifting), so compare the relative coordinates loosely.
    same_order = np.abs(mine[..., :3] - theirs[..., :3]).reshape(len(mine), -1).max(axis=1) < 1e-6
    same_set = (
        np.abs(np.sort(mine[..., 0], axis=1) - np.sort(theirs[..., 0], axis=1)).max(axis=1) < 1e-6
    )
    assert same_set.mean() >= 0.95
    note(
        camera,
        "train regions",
        regions=len(mine),
        same_points=int(same_set.sum()),
        same_order=int(same_order.sum()),
    )
    matched = model.anchors.match(stages.regions, labels, observation.intrinsics)
    sizes = np.array([len(rows) for rows in matched])
    assert np.array_equal(sizes, np.diff(record["matched_offsets"]))  # label matches identical
    close(camera, "matched labels", torch.cat(matched), record["matched_labels"], rtol=1e-6)
    similarity, offset = model.anchors.encode(grasps.theta, matched)
    close(camera, "anchor targets", similarity, record["target_anchor"], rtol=1e-6)
    close(camera, "offset targets", offset, record["target_offset"], rtol=2e-5)
    positive = similarity > model.anchors.positive_similarity
    assert np.array_equal(positive.cpu().numpy(), record["target_anchor"] > 0.99)
    note(
        camera,
        "label matching",
        regions_with_labels=int((sizes > 0).sum()),
        matches=int(sizes.sum()),
        positive_anchors=int(positive.sum()),
    )


@pytest.mark.slow
def test_region_network_and_its_losses_match_on_the_reference_regions(training):
    """Training mode: batch statistics, replayed dropout."""
    camera, model, record, _, _, _ = training
    device = model.anchors.gamma.device
    regions = torch.as_tensor(record["region"], device=device)
    info = torch.as_tensor(record["grasp_info"], device=device)
    logits, offsets = model.local_net(regions, info)
    close(camera, "train anchor logits", logits, record["anchor_logits"])
    close(camera, "train centre offsets", offsets, record["offset_raw"])
    losses = model.anchors.loss(
        logits,
        offsets,
        torch.as_tensor(record["target_anchor"], device=device),
        torch.as_tensor(record["target_offset"], device=device),
    )
    close(camera, "loss/anchor", losses["loss/anchor"], record["loss_anchor"], rtol=1e-5)
    close(
        camera, "loss/anchor_offset", losses["loss/anchor_offset"], record["loss_offset"], rtol=1e-5
    )


@pytest.mark.slow
def test_total_loss_and_gradients_match(training):
    camera, model, record, observation, labels, stages = training
    targets = model.proposal.encode(labels, observation.depth, SIZE)
    losses = model.proposal.loss(stages.raw, targets)
    matched = model.anchors.match(stages.regions, labels, observation.intrinsics)
    similarity, offset = model.anchors.encode(stages.regions.grasps.theta, matched)
    losses.update(model.anchors.loss(stages.logits, stages.offsets, similarity, offset))
    assert set(losses) == {
        "loss/heatmap",
        "loss/angle",
        "loss/offsets",
        "loss/anchor",
        "loss/anchor_offset",
    }
    total = sum(losses.values())
    # A region sampled in another order moves the batch statistics of all of them.
    close(camera, "total loss", total, record["loss"], rtol=1e-4)
    model.zero_grad()
    total.backward(retain_graph=True)
    parameters = dict(model.named_parameters())
    ours = {
        "grad_anchor.backbone.net.conv1.0.weight": "image_net.stem.0.weight",
        "grad_anchor.hmap.1.weight": "image_net.heatmap.out.weight",
        "grad_anchor.cls_mask_conv.1.weight": "image_net.angle.out.weight",
        "grad_local.pointnet.conv1.weight": "local_net.embed.linear.weight",
        "grad_local.anchor_mlp.3.weight": "local_net.anchor.out.weight",
        "grad_local.offset_mlp.3.weight": "local_net.offset.out.weight",
    }
    for theirs, name in ours.items():
        gradient = parameters[name].grad.reshape(record[theirs].shape)
        close(camera, f"gradient of {name}", gradient, record[theirs], rtol=2e-2)
    # The region losses do not reach the image network: its feature map is detached.
    model.zero_grad()
    region_only = model.anchors.loss(stages.logits, stages.offsets, similarity, offset)
    sum(region_only.values()).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.image_net.parameters())
    assert any(p.grad is not None and p.grad.any() for p in model.local_net.parameters())


@pytest.mark.slow
def test_anchor_shift_matches(training):
    camera, model, record, _, _, _ = training
    labels = torch.as_tensor(record["total_labels"])
    for name, column in (("gamma", 4), ("beta", 5)):
        start = torch.as_tensor(record[f"anchor_{name}"])
        shifted = model.anchors._shifted(start, labels[:, column])
        close(camera, f"shifted {name} anchors", shifted, record[f"shifted_{name}"], rtol=1e-12)
        assert not torch.equal(shifted, start)
