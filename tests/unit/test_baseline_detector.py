"""GraspNet-baseline end to end on tiny inputs: config validation, the contracts that
keep its parts apart from GSNet's, loss and predict, and the checkpoint converter."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Iterator
from itertools import pairwise

import pytest
import torch
from torch import Tensor, nn

from grasppanda.cli.main import main
from grasppanda.config import Config, ConfigError
from grasppanda.models import Backbone
from grasppanda.models.detectors.graspnet_baseline import GraspNetBaseline
from grasppanda.models.detectors.seed_based import SeedGraspDetector
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import ASSIGNERS, BACKBONES, DETECTORS, HEADS, VIEWS
from grasppanda.structures import (
    ContractError,
    Gripper,
    ObjectGraspLabels,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    Supervision,
)

from .conftest import REPO

BASELINE_YAML = REPO / "configs" / "_base_" / "models" / "graspnet_baseline.yaml"
RUNNABLE = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"
#: Tensors and trainable parameters of the reference GraspNet (graspnet-baseline), counted there.
REFERENCE_TENSORS, REFERENCE_PARAMETERS = 162, 1_025_964
VIEW_COUNT, ANGLES, DEPTHS, SEEDS, POINTS, CHANNELS = 10, 3, 2, 12, 48, 16


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------- components used only by these tests


@BACKBONES.register()
class BaselineUnitSubsampledBackbone(Backbone):
    """Per-point MLP on xyz, returned on every fourth point: a tiny sub-sampling encoder."""

    output = "subsampled"

    def __init__(self, in_channels: int = 0, out_channels: int = CHANNELS, stride: int = 4) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        self.mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, out_channels))

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        keep = torch.arange(0, len(points), self.stride)
        return PointFeatures(
            self.mlp(points.coords[keep]), points.subset(keep, points.batch_offsets // self.stride)
        )


@BACKBONES.register()
class BaselineUnitDenseBackbone(BaselineUnitSubsampledBackbone):
    """The same MLP on every point: what a sparse U-Net looks like to the other stages."""

    output = "dense"

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        return PointFeatures(self.mlp(points.coords), points)


# ------------------------------------------------------------------------ helpers


def tiny_config(**slots) -> dict:
    model = {
        "type": "GraspNetBaseline",
        "backbone": {"type": "BaselineUnitSubsampledBackbone"},
        "proposal": {"type": "SubsampledSeeds", "num_seeds": SEEDS},
        "view": {"type": "ObjectnessViewStage", "num_views": VIEW_COUNT},
        "assigner": {
            "type": "NearestLabelAssigner",
            "crop_centre": "label_point",
            "supervise": "object_seeds",
            "view_widths": True,
        },
        "grouper": {
            "type": "DepthCylinderGroup",
            "radius": 0.3,
            "hmin": -0.1,
            "hmax": [0.1, 0.25],
            "nsample": 5,
            "mlp_channels": [8, 8],
        },
        "head": {
            "type": "OperationHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "hidden_channels": 8,
        },
    }
    model.update(slots)
    return model


def tiny_model(**slots) -> GraspNetBaseline:
    torch.manual_seed(0)
    return DETECTORS.build(tiny_config(**slots))


def batch(frames: int = 2, seed: int = 0) -> tuple[Observation, Supervision]:
    generator = torch.Generator().manual_seed(seed)
    cloud = torch.rand(frames, POINTS, 3, generator=generator)
    observation = Observation(
        keys=[SampleKey("unit", 7, "none", frame) for frame in range(frames)],
        points=PointSet.from_dense(cloud),
    )
    shape = (20, VIEW_COUNT, ANGLES, DEPTHS)
    levels = torch.tensor([-1.0, 0.1, 0.3, 0.6, 1.0])
    objects = []
    for _ in range(frames):
        objects.append([])
        for _ in range(2):
            pose = torch.eye(4)
            pose[:3, :3] = approach_to_rotation(
                torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
            )
            pose[:3, 3] = torch.rand(3, generator=generator)
            objects[-1].append(
                ObjectGraspLabels(
                    object_id=len(objects),
                    pose=pose,
                    points=torch.rand(20, 3, generator=generator) * 0.4 - 0.2,
                    scores=levels[torch.randint(0, 5, shape, generator=generator)],
                    widths=torch.rand(shape, generator=generator) * 0.12,
                    tolerance=torch.randint(0, 51, shape, generator=generator) * 0.001,
                )
            )
    objectness = (torch.rand(frames * POINTS, generator=generator) < 0.7).long()
    return observation, Supervision(objects=objects, point_labels={"objectness": objectness})


# ------------------------------------------------------------------------- config


def test_the_shipped_configs_are_valid(capsys):
    assert Config.load(BASELINE_YAML)["model"]["type"] == "GraspNetBaseline"
    assert main(["print-config", str(BASELINE_YAML), "--check"]) == 0
    out = capsys.readouterr().out
    for component in (
        "model: GraspNetBaseline",
        "model.backbone: PointNet2",
        "model.proposal: SubsampledSeeds",
        "model.view: ObjectnessViewStage",
        "model.assigner: NearestLabelAssigner",
        "model.grouper: DepthCylinderGroup",
        "model.head: OperationHead",
    ):
        assert component in out
    runnable = Config.load(RUNNABLE).resolved()
    assert runnable["train"]["epochs"] == 18 and runnable["data"]["train"]["batch_size"] == 2
    scheduler = runnable["scheduler"]
    assert (scheduler["type"], scheduler["milestones"], scheduler["gamma"]) == (
        "MultiStepLR",
        [8, 12, 16],
        0.1,
    )
    assert runnable["optimizer"]["type"] == "Adam" and runnable["optimizer"]["lr"] == 1e-3
    assert [hook["type"] for hook in runnable["hooks"]] == ["BNMomentumHook"]
    assert set(runnable["evaluation"]["variants"]) == {"raw", "cd"}
    train = runnable["data"]["train"]["dataset"]
    assert train["tolerance"]["max_friction"] == 0.55 and train.get("graspness") is None
    assert runnable["data"]["val"]["dataset"]["split"] == "val_dev"  # not a test split


def test_the_shipped_config_states_exactly_the_component_defaults():
    """The YAML spells the reference model out; it must not drift from the code defaults.
    The exceptions are the three assigner options that differ from the Graspness defaults
    and the sampling tie rule of the backbone, whose default is not the reference's."""
    explicit = Config.load(BASELINE_YAML).resolved()["model"]
    slots = ("backbone", "proposal", "view", "assigner", "grouper", "head")
    bare = {
        "type": "GraspNetBaseline",
        **{slot: {"type": explicit[slot]["type"]} for slot in slots},
    }
    # Not validated: with the assigner's own defaults the bare model is refused, because
    # the view stage needs the two assigner options listed below.
    minimal = Config.from_dict({"model": bare}, validate=False).resolved()["model"]
    assert explicit["point_features"] == minimal["point_features"] == "ones"
    assert Gripper(**explicit["gripper"]) == Gripper() and minimal["gripper"] is None
    baseline_behaviour = {
        ("assigner", "crop_centre"): "label_point",
        ("assigner", "supervise"): "object_seeds",
        ("assigner", "view_widths"): True,
        ("backbone", "fps_tie_break"): "pointnet2_cuda",
    }
    loss_slots = []
    for slot in slots:
        assert set(explicit[slot]) == set(minimal[slot]), slot  # every option is written down
        for key, value in explicit[slot].items():
            if isinstance(value, dict) and "type" in value:
                loss_slots.append((slot, key))  # default `None` = the reference loss, see below
                assert minimal[slot][key] is None
            elif (slot, key) in baseline_behaviour:
                assert value == baseline_behaviour[slot, key] != minimal[slot][key]
            else:
                assert value == minimal[slot][key] or list(value) == list(minimal[slot][key]), (
                    slot,
                    key,
                )
    built = {
        "view": VIEWS.build({"type": "ObjectnessViewStage"}, in_channels=4, gripper=Gripper()),
        "head": HEADS.build({"type": "OperationHead"}, in_channels=4, gripper=Gripper()),
    }
    assert sorted(loss_slots) == [
        ("head", "angle_loss"),
        ("head", "score_loss"),
        ("head", "tolerance_loss"),
        ("head", "width_loss"),
        ("view", "objectness_loss"),
        ("view", "view_loss"),
    ]
    for slot, key in loss_slots:
        default, node = getattr(built[slot], key), explicit[slot][key]
        assert type(default).__name__ == node["type"] and default.weight == node["weight"], (
            slot,
            key,
        )
        if "beta" in node:
            assert default.beta == node["beta"]


MIXES = [
    (  # GSNet's proposal scores every scene point: it needs dense features
        {"proposal": {"type": "GraspnessFPS", "num_seeds": SEEDS}},
        ("GraspnessFPS requires 'dense' scene features", "sub-sampled"),
    ),
    (  # a dense encoder under the baseline's proposal
        {"backbone": {"type": "BaselineUnitDenseBackbone"}},
        ("SubsampledSeeds requires 'subsampled' scene features", "BaselineUnitDenseBackbone"),
    ),
    (  # GSNet's head has no depth axis
        {"head": {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS}},
        ("axes ('seed', 'depth', 'channel')", "SWADHead expects ('seed', 'channel')"),
    ),
    (  # GSNet's grouper has no depth axis
        {"grouper": {"type": "CylinderGroup"}},
        ("axes ('seed', 'channel')", "OperationHead expects ('seed', 'depth', 'channel')"),
    ),
    (  # the two stages would encode scores differently
        {
            "head": {
                "type": "OperationHead",
                "num_angle": ANGLES,
                "num_depth": DEPTHS,
                "score_target": "log_friction_fixed_max",
            },
            "view": {
                "type": "ObjectnessViewStage",
                "num_views": VIEW_COUNT,
                "score_target": "log_friction_batch_max",
            },
        },
        ("score_target = 'log_friction_batch_max'", "'log_friction_fixed_max'"),
    ),
    (  # three crops per seed for a head with two depth bins
        {"grouper": {"type": "DepthCylinderGroup", "hmax": [0.01, 0.02, 0.03]}},
        ("lists 3 far end(s)", "num_depth = 2"),
    ),
]


@pytest.mark.parametrize(("slots", "fragments"), MIXES)
def test_incompatible_slots_are_config_errors_before_anything_is_built(slots, fragments):
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_config(**slots)})
    # Every broken contract is its own issue: GSNet's proposal under the baseline's view
    # stage also means two owners for the objectness of a seed.
    assert all(issue.path == "model" for issue in error.value.issues)
    issue = error.value.issues[0]
    for fragment in fragments:
        assert fragment in issue.message


def test_gsnet_refuses_the_baseline_parts_too():
    gsnet = {
        "type": "GSNet",
        "backbone": {"type": "BaselineUnitDenseBackbone"},
        "proposal": {"type": "GraspnessFPS", "num_seeds": SEEDS},
        "view": {"type": "TemplateViewStage", "num_views": VIEW_COUNT},
        "assigner": {"type": "NearestLabelAssigner"},
        "grouper": {"type": "CylinderGroup", "mlp_channels": [8]},
        "head": {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS},
    }
    Config.from_dict({"model": gsnet})  # the plain model is fine
    for slot, replacement, fragment in (
        ("proposal", {"type": "SubsampledSeeds"}, "requires 'subsampled' scene features"),
        ("grouper", {"type": "DepthCylinderGroup"}, "SWADHead expects ('seed', 'channel')"),
        ("head", {"type": "OperationHead"}, "OperationHead expects ('seed', 'depth', 'channel')"),
    ):
        with pytest.raises(ConfigError) as error:
            Config.from_dict({"model": {**gsnet, slot: replacement}})
        assert fragment in error.value.issues[0].message


def test_incompatible_prebuilt_components_fail_at_build_time():
    with pytest.raises(ContractError, match="requires 'subsampled' scene features"):
        DETECTORS.build(tiny_config(backbone=BaselineUnitDenseBackbone()))
    fixed = HEADS.build(
        {
            "type": "OperationHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "score_target": "log_friction_fixed_max",
        },
        in_channels=8,
        gripper=Gripper(),
    )
    with pytest.raises(ContractError, match="both must\\s+encode scores the same way"):
        DETECTORS.build(tiny_config(head=fixed))
    three = HEADS.build(
        {"type": "OperationHead", "num_angle": ANGLES, "num_depth": 3},
        in_channels=8,
        gripper=Gripper(),
    )
    with pytest.raises(ContractError, match=r"makes 2 crop\(s\) per seed.*num_depth = 3"):
        DETECTORS.build(tiny_config(head=three))


def test_children_receive_what_the_parent_knows():
    model = tiny_model()
    assert isinstance(model, SeedGraspDetector) and model.required_inputs == frozenset({"points"})
    assert model.view.in_channels == CHANNELS and model.head.in_channels == 8
    assert model.assigner.template is model.view.template
    assert model.view.gripper is model.head.gripper is model.gripper
    narrow = DETECTORS.build(tiny_config(gripper={"max_width": 0.08}))
    assert narrow.view.gripper.max_width == narrow.head.gripper.max_width == 0.08


# --------------------------------------------------------------------- loss, predict


def test_loss_reports_weighted_stage_losses_and_backpropagates_everywhere():
    model = tiny_model().train()
    observation, supervision = batch()
    losses = model.loss(observation, supervision)
    expected = {
        "loss/objectness",
        "loss/view",
        "loss/score",
        "loss/angle",
        "loss/width",
        "loss/tolerance",
    }
    assert {key for key in losses if key.startswith("loss/")} == expected
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert losses["metric/real_seeds"].item() == 1.0  # training rejects no seed
    on_object = supervision.point_labels["objectness"][::4].float().mean()
    assert losses["metric/supervised_seeds"].item() == pytest.approx(on_object.item())
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert set(model(observation, supervision)) == set(losses)


def test_training_crops_use_label_points_and_frames_and_inference_the_seeds():
    model = tiny_model().train()
    observation, supervision = batch()
    with_labels = model.run_stages(observation, supervision)
    without = model.run_stages(observation)
    matched = with_labels.matched
    assert without.matched is None
    assert torch.equal(matched.crop_centre, matched.label_point)
    assert not torch.allclose(matched.crop_centre, with_labels.seeds.points.coords)
    assert not torch.allclose(matched.crop_rotation, with_labels.approach.rotation)
    assert not torch.allclose(with_labels.groups.feats, without.groups.feats)
    assert with_labels.groups.axes == ("seed", "depth", "channel")
    assert with_labels.groups.feats.shape == (2 * SEEDS, DEPTHS, 8)


def test_a_scene_grouper_under_a_subsampling_backbone_gets_geometry_only():
    model = tiny_model().eval()
    observation, _ = batch()
    received = []
    model.grouper.register_forward_pre_hook(lambda _module, args: received.append(args[0]))
    out = model.run_stages(observation)
    (support,) = received
    assert support.points is observation.points and support.feats.shape == (2 * POINTS, 0)
    assert out.feats.points.parent_id == observation.points.id  # the backbone's own subsample
    assert out.seeds.points is out.feats.points and len(out.seeds) == 2 * SEEDS


def test_predict_drops_background_seeds_and_never_sees_labels():
    model = tiny_model().eval()
    observation, _ = batch()
    out = model.run_stages(observation)
    is_object = out.approach.objectness.argmax(dim=-1) == 1
    assert torch.equal(out.seeds.valid, is_object)
    grasps = model.predict(observation)
    assert len(grasps) == 2 and [g.key for g in grasps] == observation.keys
    assert [len(g) for g in grasps] == is_object.reshape(2, SEEDS).sum(dim=1).tolist()
    cloud = observation.points.dense()
    for index, group in enumerate(grasps):
        assert group.score_kind == "predicted_quality_x_tolerance" and group.gripper == Gripper()
        assert group.validate().to_graspnet_array().shape == (len(group), 17)
        assert (group.width >= 0).all() and (group.width <= 0.1).all()
        assert set((group.depth * 100).round().int().tolist()) <= {1, 2}
        offset = (group.translation[:, None, :] - cloud[index][None, :, :]).abs().amax(dim=-1)
        assert len(group) == 0 or offset.min(dim=1).values.max() == 0  # grasps sit on scene points
    with torch.no_grad():  # a network that calls everything background predicts nothing
        model.view.output.weight[:2].zero_()
        model.view.output.bias[:2] = torch.tensor([1.0, 0.0])
    assert [len(g) for g in model.predict(observation)] == [0, 0]
    everything = tiny_model(
        view={"type": "ObjectnessViewStage", "num_views": VIEW_COUNT, "reject_background": False}
    ).eval()
    assert [len(g) for g in everything.predict(observation)] == [SEEDS, SEEDS]


def test_the_seed_count_must_match_what_the_backbone_returns():
    model = tiny_model(proposal={"type": "SubsampledSeeds", "num_seeds": 10}).eval()
    with pytest.raises(ContractError, match="configured for 10 seeds per frame"):
        model.predict(batch()[0])


def test_the_assigner_must_be_configured_for_this_family():
    plain = {"type": "NearestLabelAssigner"}
    with pytest.raises(ConfigError) as error:  # refused when the config is checked ...
        Config.from_dict({"model": tiny_config(assigner=plain)})
    assert "`supervise: object_seeds` and `view_widths: true`" in error.value.issues[0].message
    with pytest.raises(ContractError, match="supervise: object_seeds.*view_widths: true"):
        tiny_model(assigner=plain)  # ... and again when the stages are built
    half = {**plain, "supervise": "object_seeds"}
    with pytest.raises(ConfigError, match="'view_width'.*only with `view_widths: true`"):
        Config.from_dict({"model": tiny_config(assigner=half)})
    observation, supervision = batch()
    for objects in supervision.objects:
        for obj in objects:
            obj.tolerance = None
    with pytest.raises(ContractError, match="supervises a grasp tolerance"):
        tiny_model().train().loss(observation, supervision)
    assert ASSIGNERS.get("NearestLabelAssigner") is type(tiny_model().assigner)


# ------------------------------------------------------------- the reference model


def test_the_reference_model_has_the_reference_size():
    model = DETECTORS.build(Config.load(BASELINE_YAML)["model"])
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_PARAMETERS
    assert len(model.state_dict()) == REFERENCE_TENSORS
    assert model.backbone.output == "subsampled" and model.backbone.out_channels == 256
    assert model.proposal.num_seeds == 1024 and model.view.template.num_views == 300
    assert model.grouper.hmax == (0.01, 0.02, 0.03, 0.04) and model.grouper.nsample == 64
    assert model.head.codec.grid == (12, 4) and model.grouper.out_channels == 256
    assert model.assigner.crop_centre == "label_point"


# ----------------------------------------------------------------------- converter


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_state(seed: int = 0) -> dict[str, Tensor]:
    """A random state dict with the layout of the reference GraspNet (graspnet-baseline)."""
    generator = torch.Generator().manual_seed(seed)
    state: dict[str, Tensor] = {}

    def norm(prefix: str, channels: int) -> None:
        for name in ("weight", "bias", "running_mean"):
            state[f"{prefix}.{name}"] = torch.randn(channels, generator=generator)
        state[f"{prefix}.running_var"] = torch.rand(channels, generator=generator) + 0.5
        state[f"{prefix}.num_batches_tracked"] = torch.tensor(7)

    def shared_mlp(prefix: str, widths: list[int]) -> None:
        for layer, (cin, cout) in enumerate(pairwise(widths)):
            state[f"{prefix}.layer{layer}.conv.weight"] = torch.randn(
                cout, cin, 1, 1, generator=generator
            )
            norm(f"{prefix}.layer{layer}.bn.bn", cout)

    def small_head(prefix: str, widths: list[int]) -> None:
        for layer, (cin, cout) in enumerate(pairwise(widths), start=1):
            state[f"{prefix}.conv{layer}.weight"] = torch.randn(cout, cin, 1, generator=generator)
            state[f"{prefix}.conv{layer}.bias"] = torch.randn(cout, generator=generator)
        for layer, channels in enumerate(widths[1:-1], start=1):
            norm(f"{prefix}.bn{layer}", channels)

    backbone = "view_estimator.backbone"
    shared_mlp(f"{backbone}.sa1.mlp_module", [3, 64, 64, 128])
    for level in (2, 3, 4):
        shared_mlp(f"{backbone}.sa{level}.mlp_module", [259 if level > 2 else 131, 128, 128, 256])
    for level in (1, 2):
        shared_mlp(f"{backbone}.fp{level}.mlp", [512, 256, 256])
    small_head("view_estimator.vpmodule", [256, 256, 302, 302])
    shared_mlp("grasp_generator.crop.mlps", [3, 64, 128, 256])
    small_head("grasp_generator.operation", [256, 128, 128, 36])
    small_head("grasp_generator.tolerance", [256, 128, 128, 12])
    return state


def test_converter_maps_every_reference_tensor_to_its_place():
    converter = load_converter()
    source = reference_state()
    assert len(source) == REFERENCE_TENSORS
    converted = converter.convert_graspnet_baseline(source)
    model = DETECTORS.build(Config.load(BASELINE_YAML)["model"])
    converter.verify_against(model, converted)
    model.load_state_dict(converted, strict=True)
    view, head = "view_estimator.vpmodule", "grasp_generator"
    assert torch.equal(
        model.backbone.sa3.mlp_module.layer1.conv.weight,
        source["view_estimator.backbone.sa3.mlp_module.layer1.conv.weight"],
    )
    assert torch.equal(model.view.hidden1.weight, source[f"{view}.conv1.weight"][..., 0])
    assert torch.equal(model.view.hidden2.bias, source[f"{view}.conv2.bias"])
    assert torch.equal(model.view.output.weight, source[f"{view}.conv3.weight"][..., 0])
    assert torch.equal(model.view.norm2.running_var, source[f"{view}.bn2.running_var"])
    assert torch.equal(
        model.grouper.scales[0][2].linear.weight,
        source[f"{head}.crop.mlps.layer2.conv.weight"][:, :, 0, 0],
    )
    assert torch.equal(
        model.grouper.scales[0][0].norm.running_mean,
        source[f"{head}.crop.mlps.layer0.bn.bn.running_mean"],
    )
    assert torch.equal(
        model.head.operation.output.weight, source[f"{head}.operation.conv3.weight"][..., 0]
    )
    assert torch.equal(model.head.tolerance.hidden2.bias, source[f"{head}.tolerance.conv2.bias"])
    assert torch.equal(model.head.tolerance.norm1.bias, source[f"{head}.tolerance.bn1.bias"])
    assert int(model.head.operation.norm2.num_batches_tracked) == 7


def test_converter_refuses_checkpoints_that_do_not_fit():
    converter = load_converter()
    source = reference_state()
    model = DETECTORS.build(Config.load(BASELINE_YAML)["model"])
    for key in ("extra.weight", "view_estimator.vpmodule.conv4.weight", "grasp_generator.crop.x"):
        with pytest.raises(converter.ConversionError, match=f"unexpected tensor '{key}'"):
            converter.convert_graspnet_baseline({**source, key: torch.zeros(1)})
    with pytest.raises(converter.ConversionError, match="unexpected backbone tensor"):
        converter.convert_graspnet_baseline({"view_estimator.backbone.sa9.weight": torch.zeros(1)})
    partial = dict(source)
    del partial["grasp_generator.tolerance.conv1.bias"]
    with pytest.raises(
        converter.ConversionError, match=r"not filled \(1\): head.tolerance.hidden1.bias"
    ):
        converter.verify_against(model, converter.convert_graspnet_baseline(partial))
    wrong = dict(source)
    wrong["view_estimator.vpmodule.conv3.weight"] = torch.zeros(122, 122, 1)  # 120 views
    with pytest.raises(converter.ConversionError, match="shape mismatch"):
        converter.verify_against(model, converter.convert_graspnet_baseline(wrong))
    # a GSNet checkpoint is not a baseline checkpoint, and the other way round
    with pytest.raises(converter.ConversionError, match="graspness_unofficial checkpoint"):
        converter.convert_graspness_reference(source)


def test_converter_command_writes_weights_with_provenance(tmp_path, capsys):
    converter = load_converter()
    source = tmp_path / "reference.tar"
    torch.save({"epoch": 18, "model_state_dict": reference_state()}, source)
    destination = tmp_path / "out" / "baseline.pth"
    arguments = [str(source), str(destination), "--source-format", "graspnet-baseline"]
    assert converter.main([*arguments, "--camera", "realsense"]) == 0
    assert "162 tensors, 1,025,964 parameters" in capsys.readouterr().out
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    meta = saved["meta"]
    assert meta["camera"] == "realsense" and meta["source_epoch"] == 18
    assert meta["parameters"] == REFERENCE_PARAMETERS and len(meta["source_sha256"]) == 64
    assert meta["model"]["type"] == "GraspNetBaseline"
    assert meta["source_format"] == "graspnet-baseline"
    assert meta["model_config"] == "_base_/models/graspnet_baseline.yaml"
    DETECTORS.build(Config.load(BASELINE_YAML)["model"]).load_state_dict(
        saved["state_dict"], strict=True
    )
    # the format is required; with another method's format the file is refused
    wrong = [str(source), str(tmp_path / "never.pth"), "--source-format", "graspness-unofficial"]
    assert converter.main(wrong) == 1
    assert "conversion failed" in capsys.readouterr().err and not (tmp_path / "never.pth").exists()
