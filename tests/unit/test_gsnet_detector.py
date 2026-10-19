"""GSNet on tiny inputs: its shipped config, its own loss terms and decoding, the
reference size and the checkpoint converter.

The properties of the seed-based skeleton GSNet is built on are tested under every
seed-based method in tests/unit/test_all_methods_skeleton.py."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor, nn

from grasppanda.cli.main import main
from grasppanda.config import Config
from grasppanda.models import Backbone, Grouper
from grasppanda.models.detectors.gsnet import GSNet
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import (
    BACKBONES,
    DETECTORS,
    GROUPERS,
    HEADS,
    INJECTED,
    PROPOSALS,
    VIEWS,
)
from grasppanda.structures import (
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


# ---------------------------------------------------------------- what is GSNet's
# The properties of the skeleton (refusals at the config check and at build time, what the
# children receive, loss and predict, padding, crops at the labelled frames, the input
# features, scene support, malformed input) are tested under every seed-based method in
# tests/unit/test_all_methods_skeleton.py. What follows is GSNet's own.


def test_gsnet_reports_its_five_terms_and_decodes_every_seed_in_two_depth_bins():
    model = tiny_model().train()
    observation, supervision = batch()
    losses = model.loss(observation, supervision)
    expected = {"loss/objectness", "loss/graspness", "loss/view", "loss/score", "loss/width"}
    assert {key for key in losses if key.startswith("loss/")} == expected
    assert losses["metric/supervised_seeds"].item() == 1.0
    grasps = model.eval().predict(observation)
    for group in grasps:
        assert len(group) == SEEDS and group.score_kind == "predicted_quality"
        assert set((group.depth * 100).round().int().tolist()) <= {1, 2}  # two depth bins


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
    gsnet = ["--source-format", "graspness-unofficial"]
    assert converter.main([str(source), str(destination), *gsnet, "--camera", "realsense"]) == 0
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
    assert converter.main([str(broken), str(tmp_path / "never.pth"), *gsnet]) == 1
    assert "conversion failed" in capsys.readouterr().err and not (tmp_path / "never.pth").exists()
