"""GSNet end to end on tiny inputs: config validation, contracts between slots,
loss and predict, padding, replay hooks, and the checkpoint converter."""

from __future__ import annotations

import copy
import importlib.util
import math
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor, nn

from grasppanda.cli.main import main
from grasppanda.config import Config, ConfigError
from grasppanda.models import Backbone, Grouper
from grasppanda.models.detectors.gsnet import GSNet
from grasppanda.models.detectors.seed_based import SeedGraspDetector
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import (
    BACKBONES,
    DETECTORS,
    GROUPERS,
    HEADS,
    INJECTED,
    PROPOSALS,
    VIEWS,
    BuildError,
)
from grasppanda.structures import (
    ContractError,
    Gripper,
    GroupFeatures,
    ObjectGraspLabels,
    Observation,
    PointFeatures,
    PointSet,
    SampleKey,
    Supervision,
)

from .conftest import REPO

GSNET_YAML = REPO / "configs" / "_base_" / "models" / "gsnet.yaml"
#: Trainable parameters of the reference GraspNet (graspness_unofficial), counted there.
REFERENCE_PARAMETERS = 15_353_775
has_spconv = importlib.util.find_spec("spconv") is not None
requires_spconv = pytest.mark.skipif(not has_spconv, reason="needs the spconv extra")
VIEW_COUNT, ANGLES, DEPTHS, SEEDS, POINTS = 10, 3, 2, 12, 60


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------- components used only by these tests


@BACKBONES.register()
class GSNetUnitDenseBackbone(Backbone):
    """Per-point MLP on [xyz | input features]: a dense backbone that needs no extra."""

    output = "dense"

    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mlp = nn.Sequential(
            nn.Linear(3 + in_channels, 16), nn.ReLU(), nn.Linear(16, out_channels)
        )

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        return PointFeatures(self.mlp(torch.cat([points.coords, feats], dim=1)), points)


@BACKBONES.register()
class GSNetUnitSubsampledBackbone(GSNetUnitDenseBackbone):
    """Returns features on every second point, like a set-abstraction backbone."""

    output = "subsampled"

    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures:
        keep = torch.arange(0, len(points), 2)
        dense = super().forward(points, feats).feats
        return PointFeatures(dense[keep], points.subset(keep, points.batch_offsets // 2))


@GROUPERS.register()
class GSNetUnitDepthGrouper(Grouper):
    """Declares a depth axis that the SWAD head does not accept."""

    support = "seeds"
    out_axes = ("seed", "depth", "channel")

    def __init__(self, num_depth: int = 4, in_channels: int = INJECTED) -> None:
        super().__init__()
        self.num_depth, self.out_channels = num_depth, in_channels

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        return GroupFeatures(
            support.feats[:, None, :].expand(-1, self.num_depth, -1), self.out_axes
        )


@GROUPERS.register()
class GSNetUnitSceneGrouper(Grouper):
    """Groups over the scene cloud: remembers what it was handed."""

    support = "scene"
    out_axes = ("seed", "channel")

    def __init__(self, in_channels: int = INJECTED) -> None:
        super().__init__()
        self.out_channels = in_channels
        self.received: PointFeatures | None = None

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        self.received = support
        pooled = support.feats.mean(dim=0, keepdim=True).expand(len(centres), -1)
        return GroupFeatures(pooled, self.out_axes)


# ------------------------------------------------------------------------ helpers


def tiny_config(**slots) -> dict:
    model = {
        "type": "GSNet",
        "backbone": {"type": "GSNetUnitDenseBackbone", "out_channels": 16},
        "proposal": {"type": "GraspnessFPS", "num_seeds": SEEDS},
        "view": {"type": "TemplateViewStage", "num_views": VIEW_COUNT},
        "assigner": {"type": "NearestLabelAssigner"},
        "grouper": {
            "type": "CylinderGroup",
            "radius": 0.3,
            "hmin": -0.1,
            "hmax": 0.2,
            "nsample": 4,
            "mlp_channels": [8, 8],
        },
        "head": {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS},
    }
    model.update(slots)
    return model


def tiny_model(**slots) -> GSNet:
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
                )
            )
    supervision = Supervision(
        objects=objects,
        point_labels={
            "objectness": torch.randint(0, 2, (frames * POINTS,), generator=generator),
            "graspness": torch.rand(frames * POINTS, generator=generator),
        },
    )
    return observation, supervision


# ------------------------------------------------------------------------- config


def test_the_shipped_config_is_valid(capsys):
    config = Config.load(GSNET_YAML)
    assert config["model"]["type"] == "GSNet"
    assert main(["print-config", str(GSNET_YAML), "--check"]) == 0
    out = capsys.readouterr().out
    for component in (
        "model: GSNet",
        "model.backbone: SparseUNet14",
        "model.proposal: GraspnessFPS",
        "model.view: TemplateViewStage",
        "model.assigner: NearestLabelAssigner",
        "model.grouper: CylinderGroup",
        "model.head: SWADHead",
    ):
        assert component in out


def test_the_shipped_config_states_exactly_the_component_defaults():
    """The YAML spells the reference model out; it must not drift from the code defaults."""
    explicit = Config.load(GSNET_YAML).resolved()["model"]
    slots = ("backbone", "proposal", "view", "assigner", "grouper", "head")
    bare = {"type": "GSNet", **{slot: {"type": explicit[slot]["type"]} for slot in slots}}
    minimal = Config.from_dict({"model": bare}).resolved()["model"]
    assert explicit["point_features"] == minimal["point_features"] == "ones"
    assert Gripper(**explicit["gripper"]) == Gripper() and minimal["gripper"] is None
    loss_slots = []
    for slot in slots:
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
        "head": HEADS.build({"type": "SWADHead"}, in_channels=4, gripper=Gripper()),
    }
    attribute = {"loss": "loss_fn"}
    assert sorted(loss_slots) == [
        ("head", "score_loss"),
        ("head", "width_loss"),
        ("proposal", "graspness_loss"),
        ("proposal", "objectness_loss"),
        ("view", "loss"),
    ]
    for slot, key in loss_slots:
        default = getattr(built[slot], attribute.get(key, key))
        node = explicit[slot][key]
        assert type(default).__name__ == node["type"] and default.weight == node["weight"], (
            slot,
            key,
        )
        if "beta" in node:
            assert default.beta == node["beta"]


@pytest.mark.parametrize(
    ("slot", "replacement", "fragments"),
    [
        (
            "backbone",
            {"type": "GSNetUnitSubsampledBackbone"},
            (
                "GraspnessFPS requires 'dense' scene features",
                "GSNetUnitSubsampledBackbone",
                "sub-sampled",
            ),
        ),
        (
            "grouper",
            {"type": "GSNetUnitDepthGrouper"},
            (
                "emits features with axes ('seed', 'depth', 'channel')",
                "SWADHead expects ('seed', 'channel')",
            ),
        ),
    ],
)
def test_incompatible_slots_are_config_errors_before_anything_is_built(
    slot, replacement, fragments
):
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_config(**{slot: replacement})})
    (issue,) = error.value.issues
    assert issue.path == "model"
    for fragment in fragments:
        assert fragment in issue.message


def test_incompatible_prebuilt_components_fail_at_build_time():
    with pytest.raises(ContractError, match="requires 'dense' scene features"):
        DETECTORS.build(tiny_config(backbone=GSNetUnitSubsampledBackbone(out_channels=16)))
    with pytest.raises(ContractError, match=r"SWADHead expects \('seed', 'channel'\)"):
        DETECTORS.build(tiny_config(grouper=GSNetUnitDepthGrouper(in_channels=16)))


def test_children_receive_what_the_parent_knows_and_conflicts_are_errors():
    model = tiny_model()
    assert isinstance(model, SeedGraspDetector) and model.required_inputs == frozenset({"points"})
    assert model.proposal.in_channels == model.view.in_channels == 16
    assert (
        model.grouper.in_channels == 16
        and model.head.in_channels == model.grouper.out_channels == 8
    )
    assert model.assigner.template is model.view.template  # one template, shared
    assert model.head.gripper is model.gripper and model.gripper == Gripper()
    with pytest.raises(BuildError, match="in the config, but the parent component provides 8"):
        DETECTORS.build(tiny_config(head={"type": "SWADHead", "in_channels": 32}))
    custom = DETECTORS.build(tiny_config(gripper={"max_width": 0.08}))  # config nodes are mappings
    assert custom.gripper.max_width == 0.08 and custom.head.gripper is custom.gripper


# --------------------------------------------------------------------- loss, predict


def test_loss_reports_weighted_stage_losses_and_backpropagates_everywhere():
    model = tiny_model().train()
    observation, supervision = batch()
    losses = model.loss(observation, supervision)
    expected = {"loss/objectness", "loss/graspness", "loss/view", "loss/score", "loss/width"}
    assert {key for key in losses if key.startswith("loss/")} == expected
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert losses["metric/supervised_seeds"].item() == 1.0
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # forward(observation, supervision) is loss(); forward(observation) is predict()
    assert set(model(observation, supervision)) == set(losses)


def test_predict_returns_one_valid_group_per_frame_and_never_sees_labels():
    model = tiny_model().eval()
    observation, _ = batch()
    grasps = model.predict(observation)
    assert len(grasps) == 2 and [g.key for g in grasps] == observation.keys
    cloud = observation.points.dense()
    for frame, group in enumerate(grasps):
        assert len(group) == SEEDS and group.score_kind == "predicted_quality"
        assert group.frame == "camera" and group.gripper == Gripper()
        array = group.validate().to_graspnet_array()
        assert array.shape == (SEEDS, 17)
        assert (group.width >= 0).all() and (group.width <= 0.1).all()
        assert set((group.depth * 100).round().int().tolist()) <= {1, 2}  # two depth bins
        offset = (group.translation[:, None, :] - cloud[frame][None, :, :]).abs().amax(dim=-1)
        assert offset.min(dim=1).values.max() == 0  # every grasp sits on a point of its own frame
    again = model.predict(observation)
    assert torch.equal(grasps[0].score, again[0].score)  # eval mode: no sampling
    assert not grasps[0].score.requires_grad and isinstance(model(observation), list)


def test_padded_and_missing_seeds_are_never_emitted():
    model = tiny_model().eval()
    observation, _ = batch()
    chosen = torch.zeros(2 * POINTS, dtype=torch.bool)
    chosen[[3, 11, 27, 40, 59]] = True  # five candidates in frame 0, none in frame 1
    model.proposal.candidates = lambda scores: chosen
    out = model.run_stages(observation)
    assert len(out.seeds) == 2 * SEEDS and out.seeds.valid.sum() == 5  # tensors stay dense
    first, second = model.decode(out, observation)
    assert len(first) == 5 and len(second) == 0 and second.key == observation.keys[1]
    assert sorted(first.translation.tolist()) == sorted(observation.points.coords[chosen].tolist())
    assert second.to_graspnet_array().shape == (0, 17)  # an empty prediction is a valid result


def test_training_crops_use_label_frames_and_inference_the_template():
    model = tiny_model().train()
    observation, supervision = batch()
    seed_index = torch.stack([torch.arange(SEEDS), POINTS + torch.arange(SEEDS)])
    views = torch.arange(2 * SEEDS) % VIEW_COUNT
    with_labels = model.run_stages(
        observation, supervision, seed_index=seed_index, view_index=views
    )
    without = model.run_stages(observation, seed_index=seed_index, view_index=views)
    assert torch.equal(with_labels.seeds.points.parent_index, seed_index.reshape(-1))
    assert torch.equal(with_labels.approach.view_index, views) and without.matched is None
    assert with_labels.seeds.source == "replayed"
    matched = with_labels.matched
    assert not torch.allclose(
        matched.crop_rotation, with_labels.approach.rotation
    )  # object pose applied
    assert torch.equal(matched.crop_centre, with_labels.seeds.points.coords)
    support = PointFeatures(with_labels.seeds.feats, with_labels.seeds.points)
    expected = model.grouper(support, matched.crop_centre, matched.crop_rotation).feats
    assert torch.allclose(with_labels.groups.feats, expected, atol=1e-6)
    assert not torch.allclose(with_labels.groups.feats, without.groups.feats)


def test_input_feature_policy_is_explicit():
    observation, _ = batch()
    ones = tiny_model().eval()
    seen = []
    ones.backbone.register_forward_pre_hook(lambda _module, args: seen.append(args[1]))
    ones.predict(observation)
    assert torch.equal(seen[0], torch.ones(2 * POINTS, 3))  # geometry only

    given = tiny_model(point_features="observation").eval()
    assert given.required_inputs == frozenset({"points", "point_feats"})
    with pytest.raises(ContractError, match="observation.point_feats is missing"):
        given.predict(observation)
    observation.point_feats = torch.rand(2 * POINTS, 3)
    assert len(given.predict(observation)) == 2
    with pytest.raises(ValueError, match="unknown point_features"):
        DETECTORS.build(tiny_config(point_features="rgb"))


def test_a_scene_support_grouper_is_handed_the_scene_features():
    model = tiny_model(grouper={"type": "GSNetUnitSceneGrouper"}).eval()
    observation, _ = batch()
    out = model.run_stages(observation)
    received = model.grouper.received
    assert received.points is observation.points and received.feats.shape == (2 * POINTS, 16)
    assert torch.equal(received.feats, out.feats.feats)


def test_malformed_observations_and_duplicate_loss_keys_are_refused():
    model = tiny_model().eval()
    observation, supervision = batch()
    with pytest.raises(ContractError, match="needs observation.points"):
        model.predict(Observation(keys=observation.keys))
    with pytest.raises(ContractError, match=r"1 sample key\(s\) for a batch of 2"):
        model.predict(Observation(keys=observation.keys[:1], points=observation.points))
    clashing = copy.deepcopy(model)
    clashing.view.loss = lambda approach, matched: {"loss/objectness": torch.zeros(())}
    with pytest.raises(ContractError, match=r"reports \['loss/objectness'\], which another stage"):
        clashing.loss(observation, supervision)


# ------------------------------------------------------------- the reference model


@requires_spconv
def test_the_reference_model_has_the_reference_size():
    model = DETECTORS.build(Config.load(GSNET_YAML)["model"])
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_PARAMETERS
    assert len(model.state_dict()) == 216  # as many tensors as the reference checkpoint
    assert model.backbone.voxel_size == 0.005 and model.proposal.num_seeds == 1024
    assert model.view.template.num_views == 300 and model.head.codec.grid == (12, 4)
    assert (model.grouper.radius, model.grouper.nsample, model.grouper.out_channels) == (
        0.05,
        16,
        256,
    )


@requires_spconv
def test_a_small_sparse_model_trains_and_predicts():
    backbone = {
        "type": "SparseUNet14",
        "out_channels": 16,
        "stem_channels": 4,
        "encoder_channels": [4, 6, 8, 8],
        "decoder_channels": [8, 6, 6, 6],
        "voxel_size": 0.05,
    }
    model = tiny_model(backbone=backbone).train()
    observation, supervision = batch()
    losses = model.loss(observation, supervision)
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.backbone.stem.conv.weight.grad.abs().sum() > 0
    grasps = model.eval().predict(observation)
    assert [len(group) for group in grasps] == [SEEDS, SEEDS]


# ----------------------------------------------------------------------- converter


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_state(seed: int = 0) -> dict[str, Tensor]:
    """A random state dict with the layout of the reference GraspNet (MinkowskiEngine)."""
    generator = torch.Generator().manual_seed(seed)
    state: dict[str, Tensor] = {}

    def kernel(key: str, volume: int, cin: int, cout: int) -> None:
        shape = (cin, cout) if volume == 1 else (volume, cin, cout)  # a 1x1 kernel is stored 2-D
        state[key] = torch.randn(shape, generator=generator)

    def norm(prefix: str, channels: int) -> None:
        for name in ("weight", "bias", "running_mean"):
            state[f"{prefix}.{name}"] = torch.randn(channels, generator=generator)
        state[f"{prefix}.running_var"] = torch.rand(channels, generator=generator) + 0.5
        state[f"{prefix}.num_batches_tracked"] = torch.tensor(7)

    def block(prefix: str, cin: int, cout: int) -> None:
        kernel(f"{prefix}.conv1.kernel", 27, cin, cout)
        norm(f"{prefix}.norm1.bn", cout)
        kernel(f"{prefix}.conv2.kernel", 27, cout, cout)
        norm(f"{prefix}.norm2.bn", cout)
        if cin != cout:
            kernel(f"{prefix}.downsample.0.kernel", 1, cin, cout)
            norm(f"{prefix}.downsample.1.bn", cout)

    kernel("backbone.conv0p1s1.kernel", 125, 3, 32)
    norm("backbone.bn0.bn", 32)
    widths = [32, 32, 64, 128, 256]
    for stage, stride in enumerate((1, 2, 4, 8), start=1):
        kernel(f"backbone.conv{stage}p{stride}s2.kernel", 8, widths[stage - 1], widths[stage - 1])
        norm(f"backbone.bn{stage}.bn", widths[stage - 1])
        block(f"backbone.block{stage}.0", widths[stage - 1], widths[stage])
    width = 256
    for stage, stride, skip in zip(range(4, 8), (16, 8, 4, 2), (128, 64, 32, 32)):
        kernel(f"backbone.convtr{stage}p{stride}s2.kernel", 8, width, 192)
        norm(f"backbone.bntr{stage}.bn", 192)
        block(f"backbone.block{stage + 1}.0", 192 + skip, 192)
        width = 192
    kernel("backbone.final.kernel", 1, 192, 512)
    state["backbone.final.bias"] = torch.randn(1, 512, generator=generator)
    for name, cin, cout in (
        ("graspable.conv_graspable", 512, 3),
        ("rotation.conv1", 512, 512),
        ("rotation.conv2", 512, 300),
        ("swad.conv1", 256, 256),
        ("swad.conv_swad", 256, 96),
    ):
        state[f"{name}.weight"] = torch.randn(cout, cin, 1, generator=generator)
        state[f"{name}.bias"] = torch.randn(cout, generator=generator)
    for layer, cin in ((0, 515), (1, 256)):
        state[f"crop.mlps.layer{layer}.conv.weight"] = torch.randn(
            256, cin, 1, 1, generator=generator
        )
        norm(f"crop.mlps.layer{layer}.bn.bn", 256)
    return state


@requires_spconv
def test_converter_maps_every_reference_tensor_to_its_place():
    converter = load_converter()
    source = reference_state()
    assert len(source) == 216
    converted = converter.convert_graspness_reference(source)
    model = DETECTORS.build(Config.load(GSNET_YAML)["model"])
    converter.verify_against(model, converted)
    model.load_state_dict(converted, strict=True)

    stem = source["backbone.conv0p1s1.kernel"]
    for i, j, k in [(0, 0, 0), (4, 1, 2), (2, 3, 4)]:  # cell n = i + 5 j + 25 k
        assert torch.equal(
            model.backbone.stem.conv.weight[:, i, j, k, :], stem[i + 5 * j + 25 * k].T
        )
    assert torch.equal(model.backbone.project.weight, source["backbone.final.kernel"].T)
    assert torch.equal(model.backbone.project.bias, source["backbone.final.bias"][0])
    assert torch.equal(
        model.backbone.encoder[1].shortcut.conv.weight,
        source["backbone.block2.0.downsample.0.kernel"].T,
    )
    assert torch.equal(
        model.backbone.decoder[0].norm2.running_var,
        source["backbone.block5.0.norm2.bn.running_var"],
    )
    up = source["backbone.convtr7p2s2.kernel"]
    assert torch.equal(model.backbone.up[3].conv.weight[:, 1, 0, 1, :], up[1 + 0 + 4].T)
    assert torch.equal(
        model.proposal.score.weight, source["graspable.conv_graspable.weight"][..., 0]
    )
    assert torch.equal(model.view.score.bias, source["rotation.conv2.bias"])
    assert torch.equal(
        model.grouper.mlp[0].linear.weight, source["crop.mlps.layer0.conv.weight"][:, :, 0, 0]
    )
    assert torch.equal(
        model.grouper.mlp[1].norm.running_mean, source["crop.mlps.layer1.bn.bn.running_mean"]
    )
    assert torch.equal(model.head.output.weight, source["swad.conv_swad.weight"][..., 0])
    assert int(model.backbone.stem.norm.num_batches_tracked) == 7


@requires_spconv
def test_converter_refuses_checkpoints_that_do_not_fit():
    converter = load_converter()
    source = reference_state()
    model = DETECTORS.build(Config.load(GSNET_YAML)["model"])
    with pytest.raises(converter.ConversionError, match="unexpected tensor 'extra.weight'"):
        converter.convert_graspness_reference({**source, "extra.weight": torch.zeros(1)})
    with pytest.raises(converter.ConversionError, match="unexpected backbone tensor"):
        converter.convert_graspness_reference(
            {"backbone.block9.0.conv1.kernel": torch.zeros(27, 1, 1)}
        )
    partial = dict(source)
    del partial["swad.conv1.bias"]
    with pytest.raises(converter.ConversionError, match=r"not filled \(1\): head.hidden.bias"):
        converter.verify_against(model, converter.convert_graspness_reference(partial))
    wrong = dict(source)
    wrong["rotation.conv2.weight"] = torch.zeros(120, 512, 1)  # a 120-view checkpoint
    with pytest.raises(converter.ConversionError, match="shape mismatch"):
        converter.verify_against(model, converter.convert_graspness_reference(wrong))


@requires_spconv
def test_converter_command_writes_weights_with_provenance(tmp_path, capsys):
    converter = load_converter()
    source = tmp_path / "reference.tar"
    torch.save({"epoch": 10, "model_state_dict": reference_state()}, source)
    destination = tmp_path / "out" / "gsnet.pth"
    assert converter.main([str(source), str(destination), "--camera", "realsense"]) == 0
    assert "216 tensors, 15,353,775 parameters" in capsys.readouterr().out
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(saved) == {"state_dict", "meta"}
    meta = saved["meta"]
    assert meta["camera"] == "realsense" and meta["source_epoch"] == 10
    assert meta["parameters"] == REFERENCE_PARAMETERS and len(meta["source_sha256"]) == 64
    assert meta["model"]["type"] == "GSNet" and meta["source_format"] == "graspness-unofficial"
    DETECTORS.build(Config.load(GSNET_YAML)["model"]).load_state_dict(
        saved["state_dict"], strict=True
    )

    broken = tmp_path / "broken.tar"
    torch.save({"model_state_dict": {"nonsense": torch.zeros(1)}}, broken)
    assert converter.main([str(broken), str(tmp_path / "never.pth")]) == 1
    assert "conversion failed" in capsys.readouterr().err and not (tmp_path / "never.pth").exists()
