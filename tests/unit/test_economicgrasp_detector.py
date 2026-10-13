"""EconomicGrasp end to end on tiny inputs: the shipped configs and metafile, the label
contract between assigner and head, loss and predict, the view stage on compact labels,
the recipe, and the checkpoint converter."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from grasppanda.cli.main import main
from grasppanda.config import Config, ConfigError
from grasppanda.datasets.compact_labels import CompactLabelConfig, CompactLabelStore
from grasppanda.datasets.labels import GRASPNET_LABEL_GRID
from grasppanda.engine.optim import build_optimizers, build_schedulers
from grasppanda.hub import load_metafile
from grasppanda.models.assigners.compact_label import CompactLabelAssigner
from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.detectors.economicgrasp import EconomicGrasp
from grasppanda.models.detectors.seed_based import SeedGraspDetector
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import DETECTORS, GROUPERS, HEADS, PROPOSALS, VIEWS
from grasppanda.structures import (
    COMPACT_LABELS,
    CompactGraspLabels,
    ContractError,
    Gripper,
    Observation,
    PointSet,
    SampleKey,
    Supervision,
)

from .conftest import REPO
from .test_gsnet_detector import (  # noqa: F401 - registers the unit-test backbone
    GSNetUnitDenseBackbone,
    load_converter,
    reference_state,
    requires_spconv,
)

MODEL_YAML = REPO / "configs" / "_base_" / "models" / "economicgrasp.yaml"
RUN_CONFIGS = {
    camera: f"economicgrasp/economicgrasp_sunet14_g1b-{camera}_10e"
    for camera in ("kinect", "realsense")
}
#: Trainable parameters of the reference network, counted there.
REFERENCE_PARAMETERS = 15_750_238
VIEW_COUNT, ANGLES, DEPTHS, SEEDS, POINTS = 10, 3, 2, 12, 60
SLOTS = ("backbone", "proposal", "view", "assigner", "grouper", "head")


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------------------------------ helpers


def tiny_config(**slots) -> dict:
    model = {
        "type": "EconomicGrasp",
        "backbone": {"type": "GSNetUnitDenseBackbone", "out_channels": 16},
        "proposal": {"type": "GraspnessFPS", "num_seeds": SEEDS},
        "view": {"type": "TemplateViewStage", "num_views": VIEW_COUNT},
        "assigner": {"type": "CompactLabelAssigner", "max_match_distance": 0.08},
        "grouper": {
            "type": "CylinderInteractionGroup",
            "radius": 0.3,
            "hmin": -0.1,
            "hmax": 0.2,
            "nsample": 4,
            "mlp_channels": [8, 8],
            "post_mlp_channels": [8, 8],
        },
        "head": {
            "type": "EconomicHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "token_channels": 8,
        },
    }
    model.update(slots)
    return model


def tiny_model(**slots) -> EconomicGrasp:
    torch.manual_seed(0)
    model = DETECTORS.build(tiny_config(**slots))
    with torch.no_grad():  # make most points candidates so that every frame has real seeds
        model.proposal.score.weight.mul_(0.01)
        model.proposal.score.bias.copy_(torch.tensor([0.0, 1.0, 1.0]))
    return model


def batch(frames: int = 2, seed: int = 0) -> tuple[Observation, Supervision]:
    generator = torch.Generator().manual_seed(seed)
    cloud = torch.rand(frames, POINTS, 3, generator=generator)
    observation = Observation(
        keys=[SampleKey("unit", 7, "none", frame) for frame in range(frames)],
        points=PointSet.from_dense(cloud),
    )
    shape = (25, VIEW_COUNT)

    def integers(high: int) -> Tensor:
        return torch.randint(0, high, shape, generator=generator).to(torch.uint8)

    labels = []
    for frame in range(frames):
        labels.append([])
        for _ in range(2):
            pose = torch.eye(4)
            pose[:3, :3] = approach_to_rotation(
                torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
            )
            # Label points on the cloud, so that many seeds lie within the matching distance.
            pick = torch.randint(0, POINTS, (25,), generator=generator)
            placed = cloud[frame, pick] + 0.01 * torch.randn(25, 3, generator=generator)
            pose[:3, 3] = torch.rand(3, generator=generator)
            labels[-1].append(
                CompactGraspLabels(
                    object_id=len(labels),
                    pose=pose,
                    points=(placed - pose[:3, 3]) @ pose[:3, :3],
                    view_graspness=torch.rand(shape, generator=generator),
                    angle=integers(ANGLES),
                    depth=integers(DEPTHS),
                    score=integers(11),
                    width=integers(120),
                    view_max_friction=0.6,
                )
            )
    supervision = Supervision(
        objects=[[] for _ in range(frames)],
        point_labels={
            "objectness": torch.randint(0, 2, (frames * POINTS,), generator=generator),
            "graspness": torch.rand(frames * POINTS, generator=generator),
        },
        extras={COMPACT_LABELS: labels},
    )
    return observation, supervision


# ------------------------------------------------------------------------- config


def test_the_shipped_model_config_is_valid(capsys):
    config = Config.load(MODEL_YAML)
    assert config["model"]["type"] == "EconomicGrasp"
    assert main(["print-config", str(MODEL_YAML), "--check"]) == 0
    out = capsys.readouterr().out
    for component in (
        "model: EconomicGrasp",
        "model.backbone: SparseUNet14",
        "model.proposal: GraspnessFPS",
        "model.view: TemplateViewStage",
        "model.assigner: CompactLabelAssigner",
        "model.grouper: CylinderInteractionGroup",
        "model.head: EconomicHead",
    ):
        assert component in out


def test_the_shipped_config_states_exactly_the_component_defaults():
    """The YAML spells the reference model out; it must not drift from the code defaults."""
    explicit = Config.load(MODEL_YAML).resolved()["model"]
    bare = {"type": "EconomicGrasp", **{slot: {"type": explicit[slot]["type"]} for slot in SLOTS}}
    minimal = Config.from_dict({"model": bare}).resolved()["model"]
    assert explicit["point_features"] == minimal["point_features"] == "ones"
    assert Gripper(**explicit["gripper"]) == Gripper() and minimal["gripper"] is None
    loss_slots = []
    for slot in SLOTS:
        assert set(explicit[slot]) == set(minimal[slot]), slot  # every option is written down
        for key, value in explicit[slot].items():
            if isinstance(value, dict) and "type" in value:
                loss_slots.append((slot, key))  # default `None` = the reference loss, see below
                assert minimal[slot][key] is None
            else:
                assert value == minimal[slot][key], (slot, key)
    built = {
        "proposal": PROPOSALS.build({"type": "GraspnessFPS"}, in_channels=4),
        "view": VIEWS.build({"type": "TemplateViewStage"}, in_channels=4),
        "head": HEADS.build({"type": "EconomicHead"}, in_channels=4, gripper=Gripper()),
    }
    assert sorted(loss_slots) == [
        ("head", "angle_loss"),
        ("head", "depth_loss"),
        ("head", "score_loss"),
        ("head", "width_loss"),
        ("proposal", "graspness_loss"),
        ("proposal", "objectness_loss"),
        ("view", "loss"),
    ]
    for slot, key in loss_slots:
        default = getattr(built[slot], {"loss": "loss_fn"}.get(key, key))
        node = explicit[slot][key]
        assert type(default).__name__ == node["type"] and default.weight == node["weight"]
        if "beta" in node:
            assert default.beta == node["beta"]


def test_the_first_three_stages_are_configured_as_in_gsnet():
    """Backbone, proposal and view stage are shared with GSNet, values included."""
    economic = Config.load(MODEL_YAML).resolved()["model"]
    gsnet = Config.load(REPO / "configs" / "_base_" / "models" / "gsnet.yaml").resolved()["model"]
    for slot in ("backbone", "proposal", "view", "gripper", "point_features"):
        assert economic[slot] == gsnet[slot], slot
    plain = {k: v for k, v in gsnet["grouper"].items() if k != "type"}
    assert plain.items() <= economic["grouper"].items()  # the same cylinder and first MLP


@pytest.mark.parametrize("camera", ["kinect", "realsense"])
def test_runnable_configs_follow_the_reference_recipe(camera):
    config = Config.load(RUN_CONFIGS[camera]).resolved()
    assert config["model"] == Config.load(MODEL_YAML).resolved()["model"]
    train = config["data"]["train"]
    assert train["batch_size"] == 4 and train["dataset"]["camera"] == camera
    assert [step["type"] for step in train["dataset"]["transforms"]] == [
        "RandomSample",
        "RandomFlip",
        "ConstantFeatures",
    ]
    assert train["dataset"]["transforms"][1] == {"type": "RandomFlip", "axis": "x", "prob": 0.5}
    compact = CompactLabelConfig(**train["dataset"]["compact_labels"])
    assert compact == CompactLabelConfig(0.4, 0.6, 0.1)
    # One owner per constant: the asset's values are the model's.
    assert compact.view_friction_threshold == config["model"]["view"]["max_friction"]
    assert compact.max_width == config["model"]["gripper"]["max_width"]
    for loader in ("train", "val", "test"):
        sample = config["data"][loader]["dataset"]["transforms"][0]
        assert sample == {"type": "RandomSample", "num_points": 20000}
    assert config["data"]["test"]["dataset"]["keep_full_points"] is True
    assert config["data"]["test"]["dataset"].get("compact_labels") is None  # no labels at test time
    adam = {"type": "Adam", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]}
    assert adam.items() <= config["optimizer"].items()
    assert config["scheduler"]["type"] == "CosineAnnealingLR" and config["train"]["epochs"] == 10
    assert config["train"]["scheduler_unit"] == "epoch"
    variants = config["evaluation"]["variants"]
    assert variants["raw"] == [] and len(variants["cd"]) == 1
    assert variants["cd"][0]["type"] == "ModelFreeCollisionFilter"
    assert variants["cd"][0]["voxel_size"] == config["model"]["backbone"]["voxel_size"] == 0.005
    assert variants["cd"][0]["collision_threshold"] == 0.01


def test_both_cameras_read_the_same_compact_label_asset():
    specs = []
    for name in RUN_CONFIGS.values():
        node = Config.load(name).resolved()["data"]["train"]["dataset"]
        compact = CompactLabelConfig(**node["compact_labels"])
        specs.append(CompactLabelStore.spec("graspnet1b", compact, GRASPNET_LABEL_GRID, "labels"))
    assert specs[0].hash == specs[1].hash and specs[0].preprocessing is None


@pytest.mark.parametrize(("name", "epochs"), [("economicgrasp_10e", 10), ("economicgrasp_20e", 20)])
def test_the_schedule_is_the_per_epoch_cosine_of_the_reference(name, epochs):
    config = Config.load(REPO / "configs" / "_base_" / "schedules" / f"{name}.yaml").resolved()
    assert config["train"]["epochs"] == epochs
    model = torch.nn.Linear(2, 2)
    optimizers = build_optimizers(model, config["optimizer"])
    (scheduler,) = build_schedulers(config["scheduler"], optimizers, total_steps=epochs).values()
    (optimizer,) = optimizers.values()
    assert type(optimizer).__name__ == "Adam" and optimizer.defaults["weight_decay"] == 0.0
    for epoch in range(epochs):
        reference = 1e-3 * (math.cos(epoch / epochs * math.pi) + 1) * 0.5  # upstream train.py
        assert optimizer.param_groups[0]["lr"] == pytest.approx(reference, rel=1e-9)
        optimizer.step()
        scheduler.step()
    assert reference > 0  # the last epoch still trains


def test_the_metafile_is_valid_and_states_the_reported_numbers():
    meta = load_metafile(REPO / "configs" / "economicgrasp" / "metafile.yaml")
    assert meta.method == "EconomicGrasp" and meta.implementation == "clean-room"
    assert meta.upstream["commit"] == "7361bb6"
    by_camera = {model.camera: model for model in meta.models}
    assert set(by_camera) == {"kinect", "realsense", "realsense-d435"}
    for camera, name in RUN_CONFIGS.items():
        model = by_camera[camera]
        assert model.config == f"{name}.yaml" and model.protocol == "graspnet1b-sv-v1"
        assert model.results[0].provenance == "reported"
        assert model.results[0].collision_filter is False
        # the authors' file; what was measured with it is held to its records, for all
        # methods, in test_zoo_records.py
        assert model.weights.source == "author" and len(model.weights.sha256) == 64
        assert "reproduced" not in {result.provenance for result in model.results}  # not trained
    assert by_camera["kinect"].results[0].ap == {"seen": 62.59, "similar": 51.73, "novel": 19.54}
    assert by_camera["realsense"].results[0].ap == {"seen": 68.21, "similar": 61.19, "novel": 25.48}
    # the transfer config takes the RealSense weights and is scored under the other protocol
    transfer = by_camera["realsense-d435"]
    assert transfer.name == "economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e"
    assert transfer.config == f"economicgrasp/{transfer.name}.yaml"
    assert transfer.protocol == "graspclutter6d-sv-v1"
    assert transfer.weights == by_camera["realsense"].weights
    assert transfer.results[0].numbers() == {
        "AP": {"test": 21.67},
        "AP0.8": {"test": 26.62},
        "AP0.4": {"test": 15.90},
    }
    assert transfer.results[0].collision_filter is None  # the paper does not say


# ------------------------------------------------------------- the label contract


@pytest.mark.parametrize(
    ("slot", "replacement", "fragments"),
    [
        (
            "head",
            {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS},
            ("CompactLabelAssigner returns 'compact'", "SWADHead is supervised with 'dense'"),
        ),
        (
            "assigner",
            {"type": "NearestLabelAssigner"},
            ("NearestLabelAssigner returns 'dense'", "EconomicHead is supervised with 'compact'"),
        ),
    ],
)
def test_assigner_and_head_must_agree_on_the_label_set(slot, replacement, fragments):
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_config(**{slot: replacement})})
    (issue,) = error.value.issues
    assert issue.path == "model"
    for fragment in fragments:
        assert fragment in issue.message
    built = tiny_model()
    with pytest.raises(ContractError, match="must agree on the label set"):
        DETECTORS.build(tiny_config(**{slot: replacement, "backbone": built.backbone}))


def test_the_stage_contracts_of_the_skeleton_still_apply():
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_config(grouper={"type": "GSNetUnitDepthGrouper"})})
    assert "EconomicHead expects ('seed', 'channel')" in error.value.issues[0].message


def test_every_seed_based_detector_compares_the_label_sets_up_front():
    """The label contract belongs to the skeleton, not to EconomicGrasp alone. GSNet used
    to accept both mixes and fail at the first loss, one of them with an AttributeError."""
    swad = {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS}
    for replacement, fragment in (
        ({"assigner": {"type": "NearestLabelAssigner"}}, "NearestLabelAssigner returns 'dense'"),
        ({"head": swad}, "SWADHead is supervised with 'dense'"),
    ):
        node = {**tiny_config(), "type": "GSNet", **replacement}
        with pytest.raises(ConfigError) as error:
            Config.from_dict({"model": node})
        (issue,) = error.value.issues
        assert fragment in issue.message
        with pytest.raises(ContractError, match="must agree on the label set"):
            DETECTORS.build(node)


def test_children_receive_what_the_parent_knows():
    model = tiny_model()
    assert isinstance(model, SeedGraspDetector) and model.required_inputs == frozenset({"points"})
    assert model.grouper.in_channels == 16 and model.head.in_channels == model.grouper.out_channels
    assert model.assigner.template is model.view.template  # one template, shared
    assert model.head.gripper is model.gripper and model.gripper == Gripper()
    assert isinstance(model.assigner, CompactLabelAssigner)


# --------------------------------------------------------------------- loss, predict


def test_loss_reports_weighted_stage_losses_and_backpropagates_everywhere():
    model = tiny_model().train()
    observation, supervision = batch()
    losses = model.loss(observation, supervision)
    expected = {"objectness", "graspness", "view", "angle", "depth", "score", "width"}
    assert {key.split("/", 1)[1] for key in losses if key.startswith("loss/")} == expected
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert 0.0 < losses["metric/supervised_seeds"].item() < 1.0  # distance and shared views
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert set(model(observation, supervision)) == set(losses)


def test_the_view_loss_runs_over_every_seed_and_the_grasp_losses_over_valid_ones():
    model = tiny_model().train()
    observation, supervision = batch()
    views = torch.arange(2 * SEEDS) % VIEW_COUNT
    out = model.run_stages(observation, supervision, view_index=views)
    matched = out.matched
    assert bool(matched.view_valid.all()) and 0 < matched.valid.sum() < 2 * SEEDS
    view = model.view.loss(out.approach, matched)["loss/view"]
    expected = 100.0 * F.smooth_l1_loss(out.approach.logits, matched.view_graspness)
    assert view.item() == pytest.approx(expected.item(), rel=1e-6)
    head = model.head.loss(out.raw, matched)
    valid = matched.valid
    angle = F.cross_entropy(out.raw.angle[valid], matched.angle[valid])
    assert head["loss/angle"].item() == pytest.approx(angle.item(), rel=1e-6)
    # The crop frame comes from the labels (object pose × template), centred at the seed.
    assert torch.equal(matched.crop_centre, out.seeds.points.coords)
    assert not torch.allclose(matched.crop_rotation, out.approach.rotation)


def test_predict_returns_one_valid_group_per_frame_and_never_sees_labels():
    model = tiny_model().eval()
    observation, _ = batch()
    grasps = model.predict(observation)
    assert len(grasps) == 2 and [g.key for g in grasps] == observation.keys
    cloud = observation.points.dense()
    for frame, group in enumerate(grasps):
        assert len(group) == SEEDS and group.score_kind == "weighted_quality_logits"
        assert group.validate().to_graspnet_array().shape == (SEEDS, 17)
        assert (group.width >= 0).all() and (group.width <= 0.1).all()
        # Depth bins 1-2, or one step beyond when the never-supervised class wins.
        assert set((group.depth * 100).round().int().tolist()) <= {1, 2, 3}
        offset = (group.translation[:, None, :] - cloud[frame][None, :, :]).abs().amax(dim=-1)
        assert offset.min(dim=1).values.max() == 0  # every grasp sits on a point of its own frame
    again = model.predict(observation)
    assert torch.equal(grasps[0].score, again[0].score)  # eval mode: no sampling, no dropout
    assert not grasps[0].score.requires_grad and isinstance(model(observation), list)


def test_training_mode_is_stochastic_through_view_sampling_and_dropout():
    model = tiny_model().train()
    observation, supervision = batch()
    views = torch.arange(2 * SEEDS) % VIEW_COUNT
    first = model.run_stages(observation, supervision, view_index=views)
    second = model.run_stages(observation, supervision, view_index=views)
    assert not torch.equal(first.raw.score, second.raw.score)  # attention dropout
    model.grouper.interaction.dropout.p = model.head.interaction.dropout.p = 0.0
    first = model.run_stages(observation, supervision, view_index=views)
    second = model.run_stages(observation, supervision, view_index=views)
    assert torch.equal(first.raw.score, second.raw.score)  # the view draw was replayed


def test_padded_and_missing_seeds_are_never_emitted_or_supervised():
    model = tiny_model().train()
    observation, supervision = batch()
    chosen = torch.zeros(2 * POINTS, dtype=torch.bool)
    chosen[[3, 11, 27, 40, 59]] = True  # five candidates in frame 0, none in frame 1
    model.proposal.candidates = lambda scores: chosen
    out = model.run_stages(observation, supervision)
    assert len(out.seeds) == 2 * SEEDS and out.seeds.valid.sum() == 5
    assert not out.matched.view_valid[SEEDS:].any() and not out.matched.valid[SEEDS:].any()
    assert out.matched.view_valid[:SEEDS].all()  # padding counts, as in the reference
    first, second = model.eval().decode(model.run_stages(observation), observation)
    assert len(first) == 5 and len(second) == 0 and second.to_graspnet_array().shape == (0, 17)


# ---------------------------------------------------------------- shared view stage


def test_the_view_stage_takes_encoded_targets_from_compact_labels():
    model = tiny_model().train()
    observation, supervision = batch()
    out = model.run_stages(observation, supervision)
    target, supervised = model.view.targets(out.matched)
    assert target is out.matched.view_graspness and supervised is out.matched.view_valid
    out.matched.view_valid[:5] = False
    masked = model.view.loss(out.approach, out.matched)["loss/view"]
    expected = 100.0 * F.smooth_l1_loss(out.approach.logits[5:], out.matched.view_graspness[5:])
    assert masked.item() == pytest.approx(expected.item(), rel=1e-6)
    out.matched.view_max_friction = 0.8
    with pytest.raises(ContractError, match="max_friction = 0.8, but TemplateViewStage is"):
        model.view.loss(out.approach, out.matched)
    out.matched.view_max_friction = 0.6
    out.matched.view_graspness = out.matched.view_graspness[:, :4]
    with pytest.raises(ContractError, match="describe 4 views, but the view template has 10"):
        model.view.loss(out.approach, out.matched)


def test_dense_labels_are_still_encoded_by_the_view_stage_itself():
    """GSNet's path through the generalised stage: unchanged."""
    torch.manual_seed(0)
    stage = VIEWS.build({"type": "TemplateViewStage", "num_views": VIEW_COUNT}, in_channels=8)
    generator = torch.Generator().manual_seed(1)
    levels = torch.tensor([-1.0, 0.0, 0.2, 0.5, 0.6, 0.9])
    friction = levels[torch.randint(0, 6, (6, VIEW_COUNT, ANGLES, DEPTHS), generator=generator)]
    valid = torch.tensor([True, True, False, True, True, True])
    zeros = torch.zeros(6, 3)
    matched = MatchedLabels(
        valid=valid,
        label_point=zeros,
        distance=torch.zeros(6),
        object_index=torch.zeros(6, dtype=torch.long),
        crop_centre=zeros,
        crop_rotation=torch.eye(3).expand(6, 3, 3),
        view_friction=friction,
        friction=friction[:, 0],
        width=torch.zeros(6, ANGLES, DEPTHS),
    )
    target, supervised = stage.targets(matched)
    assert torch.equal(target, stage.encode_targets(friction)) and supervised is valid
    good = ((friction > 0) & (friction <= 0.6)).flatten(-2).float().mean(-1)
    low, high = good.amin(-1, keepdim=True), good.amax(-1, keepdim=True)
    assert torch.allclose(target, (good - low) / (high - low + 1e-5), atol=1e-6)


# ------------------------------------------------------------- the reference model


@requires_spconv
def test_the_reference_model_has_the_reference_size():
    model = DETECTORS.build(Config.load(MODEL_YAML)["model"])
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_PARAMETERS
    assert len(model.state_dict()) == 260  # as many tensors as the reference checkpoint
    sizes = {
        name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()
    }
    assert sizes == {  # per stage, as counted in the reference checkpoint
        "backbone": 14_646_816,
        "proposal": 1_539,
        "view": 416_556,
        "grouper": 601_142,
        "head": 84_185,
    }
    assert model.backbone.voxel_size == 0.005 and model.proposal.num_seeds == 1024
    assert model.view.template.num_views == 300 and model.head.codec.grid == (12, 4)
    assert model.assigner.max_match_distance == 0.005 and model.grouper.nsample == 16


# ----------------------------------------------------------------------- converter


def economic_state(seed: int = 0) -> dict[str, Tensor]:
    """A random state dict with the layout of the reference network (MinkowskiEngine)."""
    generator = torch.Generator().manual_seed(seed + 1)
    shared = reference_state(seed)  # the backbone and the graspable layer are GSNet's
    state = {k: v for k, v in shared.items() if k.startswith(("backbone.", "graspable."))}

    def affine(name: str, cin: int, cout: int, *, conv: bool) -> None:
        shape = (cout, cin, 1) if conv else (cout, cin)
        state[f"{name}.weight"] = torch.randn(shape, generator=generator)
        state[f"{name}.bias"] = torch.randn(cout, generator=generator)

    def norm(prefix: str, channels: int) -> None:
        for name in ("weight", "bias", "running_mean"):
            state[f"{prefix}.{name}"] = torch.randn(channels, generator=generator)
        state[f"{prefix}.running_var"] = torch.rand(channels, generator=generator) + 0.5
        state[f"{prefix}.num_batches_tracked"] = torch.tensor(7)

    def interaction(prefix: str, channels: int) -> None:
        for slot in range(4):
            affine(f"{prefix}.msa.linears.{slot}", channels, channels, conv=False)
        state[f"{prefix}.norm1.weight"] = torch.randn(channels, generator=generator)
        state[f"{prefix}.norm1.bias"] = torch.randn(channels, generator=generator)

    affine("view.conv1", 512, 512, conv=True)
    affine("view.conv2", 512, 300, conv=True)
    for mlp, first in (("mlps", 515), ("mlps2", 259)):
        for layer, cin in ((0, first), (1, 256)):
            state[f"cy_group.{mlp}.layer{layer}.conv.weight"] = torch.randn(
                256, cin, 1, 1, generator=generator
            )
            norm(f"cy_group.{mlp}.layer{layer}.bn.bn", 256)
    interaction("cy_group.local_interaction_module", 259)
    for token, outputs in (("angle", 13), ("depth", 5), ("width", 1), ("score", 6)):
        affine(f"grasp_head.conv_{token}_feature", 256, 64, conv=True)
        affine(f"grasp_head.conv_{token}", 64, outputs, conv=True)
    interaction("grasp_head.global_interaction_module", 64)
    return state


@requires_spconv
def test_converter_maps_every_reference_tensor_to_its_place():
    converter = load_converter()
    source = economic_state()
    assert len(source) == 260
    converted = converter.convert_economicgrasp_reference(source)
    model = DETECTORS.build(Config.load(MODEL_YAML)["model"])
    converter.verify_against(model, converted)
    model.load_state_dict(converted, strict=True)
    assert sum(v.numel() for v in converted.values()) == sum(v.numel() for v in source.values())

    assert torch.equal(model.backbone.project.weight, source["backbone.final.kernel"].T)
    assert torch.equal(
        model.proposal.score.weight, source["graspable.conv_graspable.weight"][..., 0]
    )
    assert torch.equal(model.view.hidden.weight, source["view.conv1.weight"][..., 0])
    assert torch.equal(model.view.score.bias, source["view.conv2.bias"])
    assert torch.equal(
        model.grouper.mlp[0].linear.weight, source["cy_group.mlps.layer0.conv.weight"][:, :, 0, 0]
    )
    assert torch.equal(
        model.grouper.post_mlp[1].norm.running_var,
        source["cy_group.mlps2.layer1.bn.bn.running_var"],
    )
    assert torch.equal(
        model.grouper.post_mlp[0].linear.weight,
        source["cy_group.mlps2.layer0.conv.weight"][:, :, 0, 0],
    )
    for slot, role in enumerate(("query", "key", "value", "output")):
        crop = source[f"cy_group.local_interaction_module.msa.linears.{slot}.weight"]
        head = source[f"grasp_head.global_interaction_module.msa.linears.{slot}.bias"]
        assert torch.equal(getattr(model.grouper.interaction, role).weight, crop)
        assert torch.equal(getattr(model.head.interaction, role).bias, head)
    assert torch.equal(
        model.grouper.interaction.norm.weight,
        source["cy_group.local_interaction_module.norm1.weight"],
    )
    for token in ("angle", "depth", "width", "score"):
        assert torch.equal(
            model.head.project[token].weight,
            source[f"grasp_head.conv_{token}_feature.weight"][..., 0],
        )
        assert torch.equal(model.head.predict[token].bias, source[f"grasp_head.conv_{token}.bias"])
    assert int(model.grouper.mlp[0].norm.num_batches_tracked) == 7


@requires_spconv
def test_converter_refuses_checkpoints_that_do_not_fit():
    converter = load_converter()
    source = economic_state()
    model = DETECTORS.build(Config.load(MODEL_YAML)["model"])
    with pytest.raises(
        converter.ConversionError, match="unexpected tensor 'extra.weight' in an EconomicGrasp"
    ):
        converter.convert_economicgrasp_reference({**source, "extra.weight": torch.zeros(1)})
    # A Graspness checkpoint is not an EconomicGrasp checkpoint, and the other way round.
    with pytest.raises(
        converter.ConversionError, match="unexpected tensor 'rotation.conv1.weight'"
    ):
        converter.convert_economicgrasp_reference(reference_state())
    with pytest.raises(converter.ConversionError, match="in a graspness_unofficial checkpoint"):
        converter.convert_graspness_reference(source)
    partial = dict(source)
    del partial["grasp_head.conv_width.bias"]
    with pytest.raises(
        converter.ConversionError, match=r"not filled \(1\): head.predict.width.bias"
    ):
        converter.verify_against(model, converter.convert_economicgrasp_reference(partial))
    wrong = dict(source)
    wrong["grasp_head.conv_angle.weight"] = torch.zeros(12, 64, 1)  # a head without the extra class
    with pytest.raises(converter.ConversionError, match="shape mismatch"):
        converter.verify_against(model, converter.convert_economicgrasp_reference(wrong))


@requires_spconv
def test_converter_command_writes_weights_with_provenance(tmp_path, capsys):
    converter = load_converter()
    source = tmp_path / "reference.tar"
    torch.save({"epoch": 10, "model_state_dict": economic_state()}, source)
    destination = tmp_path / "out" / "economicgrasp.pth"
    arguments = [str(source), str(destination), "--source-format", "economicgrasp"]
    assert converter.main([*arguments, "--camera", "kinect"]) == 0
    assert "260 tensors, 15,750,238 parameters" in capsys.readouterr().out
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    meta = saved["meta"]
    assert meta["camera"] == "kinect" and meta["source_epoch"] == 10
    assert meta["parameters"] == REFERENCE_PARAMETERS and len(meta["source_sha256"]) == 64
    assert meta["model"]["type"] == "EconomicGrasp" and meta["source_format"] == "economicgrasp"
    assert meta["model_config"] == "_base_/models/economicgrasp.yaml"
    DETECTORS.build(Config.load(MODEL_YAML)["model"]).load_state_dict(
        saved["state_dict"], strict=True
    )
    # Without the format the file is read as a Graspness checkpoint, which it is not.
    assert converter.main([str(source), str(tmp_path / "never.pth")]) == 1
    assert "conversion failed" in capsys.readouterr().err and not (tmp_path / "never.pth").exists()
    with pytest.raises(converter.ConversionError, match="unknown source format 'no-such'"):
        converter.convert_file(source, source_format="no-such")
    assert set(converter.SOURCE_FORMATS) >= {"graspness-unofficial", "economicgrasp"}
    assert GROUPERS.get("CylinderInteractionGroup").status == "tested"
