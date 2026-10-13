"""FineGrasp as a configuration of shared components: what its configs share with
EconomicGrasp and what they change, the recipe, the metafile, the model end to end on
tiny inputs, and the checkpoint converter."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

import grasppanda.engine.schedulers
from grasppanda.cli.main import main
from grasppanda.config import Config
from grasppanda.datasets.compact_labels import CompactLabelConfig, CompactLabelStore
from grasppanda.datasets.graspness import GraspnessConfig, GraspnessMaps
from grasppanda.datasets.labels import GRASPNET_LABEL_GRID
from grasppanda.engine.optim import build_optimizers, build_schedulers
from grasppanda.hub import load_metafile
from grasppanda.models.detectors.economicgrasp import EconomicGrasp
from grasppanda.models.groupers.multi_scale import MultiScaleGroup
from grasppanda.registry import DETECTORS
from grasppanda.structures import ContractError, Observation

from .conftest import REPO
from .test_economicgrasp_detector import SEEDS, batch, economic_state, tiny_config
from .test_gsnet_detector import (  # noqa: F401 - registers the unit-test backbone
    GSNetUnitDenseBackbone,
    load_converter,
    requires_spconv,
)
from .test_gsnet_detector import tiny_config as gsnet_tiny_config

BASE = REPO / "configs" / "_base_"
MODEL_YAML = BASE / "models" / "finegrasp.yaml"
ECONOMIC_YAML = BASE / "models" / "economicgrasp.yaml"
RUN_CONFIG = "finegrasp/finegrasp_sunet14_g1b-realsense_10e"
ECONOMIC_RUN = "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e"
GSNET_RUN = "gsnet/gsnet_sunet14_g1b-realsense_10e"
#: Trainable parameters and state tensors of the reference network, counted there.
REFERENCE_PARAMETERS = 19_145_441
REFERENCE_TENSORS = 388


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------------------------------- config


def test_the_shipped_model_config_is_valid(capsys):
    assert main(["print-config", str(MODEL_YAML), "--check"]) == 0
    out = capsys.readouterr().out
    for component in (
        "model: EconomicGrasp",
        "model.backbone: SparseUNet14",
        "model.grouper: MultiScaleGroup",
        "model.grouper.branch: CylinderInteractionGroup",
        "model.head: EconomicHead",
    ):
        assert component in out


def test_the_model_is_economicgrasp_with_two_stages_reconfigured():
    """The written diff of configs/finegrasp/README.md, as a test."""
    fine = Config.load(MODEL_YAML).resolved()["model"]
    economic = Config.load(ECONOMIC_YAML).resolved()["model"]
    assert fine["type"] == economic["type"] == "EconomicGrasp"  # no detector of its own
    for slot in ("proposal", "view", "assigner", "head", "gripper"):  # same component, same values
        assert fine[slot] == economic[slot], slot
    # An option of an existing component: what the encoder is fed.
    assert (fine["point_features"], economic["point_features"]) == ("observation", "ones")
    changed = {k for k in fine["backbone"] if fine["backbone"][k] != economic["backbone"][k]}
    assert changed == {"in_channels"}
    assert (fine["backbone"]["in_channels"], economic["backbone"]["in_channels"]) == (6, 3)
    # A new component around the existing one: every scale is EconomicGrasp's grouper.
    grouper = fine["grouper"]
    assert grouper["type"] == "MultiScaleGroup"
    assert economic["grouper"]["type"] == grouper["branch"]["type"] == "CylinderInteractionGroup"
    shared = {k: v for k, v in economic["grouper"].items() if k != "radius"}
    assert shared.items() <= grouper["branch"].items()
    assert (grouper["radius"], economic["grouper"]["radius"]) == (0.07, 0.05)
    assert grouper["radius_scales"] == [0.25, 0.5, 0.75, 1.0]


def test_the_shipped_config_states_exactly_the_grouper_defaults():
    explicit = Config.load(MODEL_YAML).resolved()["model"]["grouper"]
    bare = {
        "type": "EconomicGrasp",
        **{
            slot: {"type": node["type"]}
            for slot, node in Config.load(MODEL_YAML).resolved()["model"].items()
            if isinstance(node, dict) and "type" in node
        },
    }
    bare["grouper"]["branch"] = {"type": "CylinderInteractionGroup"}
    minimal = Config.from_dict({"model": bare}).resolved()["model"]["grouper"]
    assert set(explicit) == set(minimal)  # every option is written down
    for key in explicit:
        if key != "branch":
            assert explicit[key] == minimal[key], key
    written = {k: v for k, v in explicit["branch"].items() if k != "radius"}
    assert written == {k: v for k, v in minimal["branch"].items() if k != "radius"}


def test_the_runnable_config_follows_the_reference_recipe():
    config = Config.load(RUN_CONFIG).resolved()
    economic = Config.load(ECONOMIC_RUN).resolved()
    assert config["model"] == Config.load(MODEL_YAML).resolved()["model"]
    train = config["data"]["train"]["dataset"]
    assert [step["type"] for step in train["transforms"]] == [
        "DepthNormals",
        "RandomSample",
        "RandomFlip",
        "CoordinateFeatures",
    ]
    normals = {"step": 1, "towards_camera": True, "encoding": "image8", "keep_depth": False}
    assert normals.items() <= train["transforms"][0].items()
    assert train["transforms"][2] == {"type": "RandomFlip", "axis": "x", "prob": 0.5}
    assert list(train["images"]) == ["depth"]  # the normals are read off the depth image
    # EconomicGrasp's data recipe apart from the features and the graspness labels.
    theirs = economic["data"]["train"]["dataset"]
    assert train["compact_labels"] == theirs["compact_labels"]
    assert train["preprocess"] == theirs["preprocess"]
    assert train["transforms"][1:3] == theirs["transforms"][:2]
    assert (train["graspness"]["normalization"], theirs["graspness"]["normalization"]) == (
        "instance",
        "scene",
    )
    same = {k: v for k, v in train["graspness"].items() if k != "normalization"}
    assert same == {k: v for k, v in theirs["graspness"].items() if k != "normalization"}
    for loader in ("val", "test"):
        dataset = config["data"][loader]["dataset"]
        assert [step["type"] for step in dataset["transforms"]] == [
            "DepthNormals",
            "RandomSample",
            "CoordinateFeatures",
        ]
        assert dataset["transforms"][0] == train["transforms"][0]
        assert dataset["transforms"][1] == {"type": "RandomSample", "num_points": 20000}
        assert dataset["keep_full_points"] is True and list(dataset["images"]) == ["depth"]
        assert dataset.get("graspness") is None and dataset.get("compact_labels") is None
    variants = config["evaluation"]["variants"]
    assert variants["raw"] == [] and len(variants["cd"]) == 1
    cd = {"voxel_size": 0.01, "approach_distance": 0.05, "collision_threshold": 0.01}
    assert variants["cd"][0]["type"] == "ModelFreeCollisionFilter"
    assert cd.items() <= variants["cd"][0].items()


def test_the_compact_labels_are_economicgrasps_asset_and_the_graspness_is_a_variant():
    nodes = {
        name: Config.load(name).resolved()["data"]["train"]["dataset"]
        for name in (RUN_CONFIG, ECONOMIC_RUN, GSNET_RUN)
    }
    compact = [
        CompactLabelStore.spec(
            "graspnet1b",
            CompactLabelConfig(**nodes[name]["compact_labels"]),
            GRASPNET_LABEL_GRID,
            "x",
        ).hash
        for name in (RUN_CONFIG, ECONOMIC_RUN)
    ]
    assert compact[0] == compact[1]

    def graspness(name: str) -> str:
        node = nodes[name]
        config = GraspnessConfig(**node["graspness"])
        return GraspnessMaps.spec(
            "graspnet1b", "realsense", config, node["preprocess"], True, "x"
        ).hash

    assert graspness(ECONOMIC_RUN) == graspness(GSNET_RUN) != graspness(RUN_CONFIG)


@pytest.mark.parametrize("name", [GSNET_RUN, ECONOMIC_RUN])
def test_instance_normalised_graspness_is_one_override_away_for_the_other_methods(name):
    """The controlled variant: the same method on the other label asset."""
    override = "data.train.dataset.graspness.normalization=instance"
    variant = Config.load(name, [override]).resolved()
    plain = Config.load(name).resolved()
    assert variant["data"]["train"]["dataset"]["graspness"]["normalization"] == "instance"
    assert variant["model"] == plain["model"]
    fine = Config.load(RUN_CONFIG).resolved()["data"]["train"]["dataset"]

    def asset(node: dict) -> str:
        config = GraspnessConfig(**node["graspness"])
        return GraspnessMaps.spec(
            "graspnet1b", "realsense", config, node["preprocess"], True, "x"
        ).hash

    assert asset(variant["data"]["train"]["dataset"]) == asset(fine)  # FineGrasp's asset, shared
    assert asset(plain["data"]["train"]["dataset"]) != asset(fine)


# ----------------------------------------------------------------------- schedule


def reference_factor(step: int, warmup: float, total: float, minimum: float = 1e-5) -> float:
    """The learning-rate factor of the reference configuration, as it states it."""
    if step < warmup:
        return (1 - minimum) * step / warmup + minimum
    ratio = (step - warmup) / total
    return (1 - minimum) * 0.5 * (1 + math.cos(math.pi * ratio)) + minimum


def test_the_schedule_is_the_warm_up_and_cut_off_cosine_of_the_reference():
    config = Config.load(BASE / "schedules" / "finegrasp_10e.yaml").resolved()
    run = Config.load(RUN_CONFIG).resolved()
    assert run["optimizer"] == config["optimizer"] and run["scheduler"] == config["scheduler"]
    train = run["train"]
    assert train["epochs"] == 10 and train["scheduler_unit"] == "iteration"
    assert (train["grad_clip"], train["grad_clip_mode"]) == (10.0, "norm")
    # Reference: max_step = epochs * (frames / total batch); one epoch of warm-up.
    batch_size = run["data"]["train"]["batch_size"]
    assert batch_size == 32  # the paper's 8 processes x 4
    steps_per_epoch = 25600 / batch_size
    total = int(train["epochs"] * steps_per_epoch)
    assert config["scheduler"]["warmup_steps"] == steps_per_epoch == 800
    assert config["scheduler"]["cosine_steps"] == total == 8000

    model = torch.nn.Linear(2, 2)
    optimizers = build_optimizers(model, config["optimizer"])
    (scheduler,) = build_schedulers(config["scheduler"], optimizers, total_steps=total).values()
    (optimizer,) = optimizers.values()
    assert type(optimizer).__name__ == "Adam" and optimizer.defaults["weight_decay"] == 1e-5
    rates = []
    for step in range(total):
        expected = 8e-3 * reference_factor(step, steps_per_epoch, total)
        rates.append(optimizer.param_groups[0]["lr"])
        assert rates[-1] == pytest.approx(expected, rel=1e-12), step
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(8e-8) and max(rates) == rates[800] == pytest.approx(8e-3)
    # The cosine's period is the whole run, so it stops one epoch short of its minimum.
    assert rates[-1] == pytest.approx(8e-3 * 0.0245, rel=2e-2)
    state = scheduler.state_dict()  # plain numbers: a checkpoint can hold it
    assert state["warmup_steps"] == 800 and json.dumps(state)


def test_the_scheduler_refuses_meaningless_lengths():
    optimizer = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=1.0)
    scheduler = grasppanda.engine.schedulers.WarmupCosineLR
    with pytest.raises(ValueError, match="warmup_steps >= 0 and cosine_steps >= 1"):
        scheduler(optimizer, warmup_steps=-1, cosine_steps=10)
    with pytest.raises(ValueError, match="cosine_steps >= 1"):
        scheduler(optimizer, warmup_steps=0, cosine_steps=0)
    with pytest.raises(ValueError, match="min_factor must be in"):
        scheduler(optimizer, warmup_steps=1, cosine_steps=1, min_factor=2.0)
    plain = scheduler(optimizer, warmup_steps=0, cosine_steps=4, min_factor=0.0)
    assert plain.factor(0) == 1.0 and plain.factor(2) == pytest.approx(0.5)
    assert plain.factor(4) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------- metafile


def test_the_metafile_is_valid_and_carries_reported_numbers_only():
    meta = load_metafile(REPO / "configs" / "finegrasp" / "metafile.yaml")
    assert meta.method == "FineGrasp" and meta.implementation == "clean-room"
    assert meta.upstream["repository"] == "HorizonRobotics/robo_orchard_lab"
    assert meta.upstream["commit"].startswith("221ec0e")
    (model,) = meta.models
    assert model.config == f"{RUN_CONFIG}.yaml" and model.camera == "realsense"
    assert model.weights is not None and model.weights.source == "author"
    assert model.weights.sha256.startswith("30c1f335")
    assert [(r.provenance, r.collision_filter) for r in model.results] == [
        ("reported", False),
        ("reported", True),
    ]
    assert model.results[0].ap == {"seen": 71.67, "similar": 62.83, "novel": 27.40}
    assert model.results[1].ap == {"seen": 73.71, "similar": 64.56, "novel": 28.14}


# --------------------------------------------------------------- tiny end to end


def fine_tiny_config(**slots) -> dict:
    grouper = tiny_config()["grouper"]
    branch = {k: v for k, v in grouper.items() if k != "radius"}
    model = tiny_config(
        point_features="observation",
        backbone={"type": "GSNetUnitDenseBackbone", "in_channels": 6, "out_channels": 16},
        grouper={
            "type": "MultiScaleGroup",
            "branch": branch,
            "radius": 0.4,
            "radius_scales": [0.5, 1.0],
            "fusion_heads": 2,
            "fusion_feedforward_channels": 16,
        },
    )
    model.update(slots)
    return model


def fine_tiny_model(**slots) -> EconomicGrasp:
    torch.manual_seed(0)
    model = DETECTORS.build(fine_tiny_config(**slots))
    with torch.no_grad():  # make most points candidates so that every frame has real seeds
        model.proposal.score.weight.mul_(0.01)
        model.proposal.score.bias.copy_(torch.tensor([0.0, 1.0, 1.0]))
    return model


def with_normals(observation: Observation, seed: int = 0) -> Observation:
    """Add [x, y, z | normal] features, as the FineGrasp data pipeline does."""
    generator = torch.Generator().manual_seed(seed)
    normals = torch.rand(len(observation.points), 3, generator=generator)
    observation.point_feats = torch.cat([observation.points.coords, normals], dim=1)
    return observation


def test_loss_and_predict_run_through_the_multi_scale_crop():
    model = fine_tiny_model().train()
    assert isinstance(model, EconomicGrasp) and isinstance(model.grouper, MultiScaleGroup)
    assert model.required_inputs == frozenset({"points", "point_feats"})
    assert model.head.in_channels == model.grouper.out_channels == 8
    observation, supervision = batch()
    observation = with_normals(observation)
    losses = model.loss(observation, supervision)
    expected = {"objectness", "graspness", "view", "angle", "depth", "score", "width"}
    assert {key.split("/", 1)[1] for key in losses if key.startswith("loss/")} == expected
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    grasps = model.eval().predict(observation)
    assert [len(group) for group in grasps] == [SEEDS, SEEDS]
    assert torch.equal(grasps[0].score, model.predict(observation)[0].score)


def test_the_normals_reach_the_encoder_and_change_the_result():
    model = fine_tiny_model().eval()
    observation, _ = batch()
    first = model.predict(with_normals(observation, seed=0))[0].score
    second = model.predict(with_normals(observation, seed=1))[0].score
    assert not torch.equal(first, second)
    observation.point_feats = None
    with pytest.raises(ContractError, match="observation.point_feats is missing"):
        model.predict(observation)
    observation.point_feats = observation.points.coords  # three channels for a six-channel stem
    with pytest.raises((ContractError, RuntimeError)):
        model.predict(observation)


def test_the_multi_scale_crop_also_serves_gsnet():
    """Same support and axes as the single cylinder: the swap is legal under SWAD too."""
    branch = {k: v for k, v in gsnet_tiny_config()["grouper"].items() if k != "radius"}
    node = gsnet_tiny_config(
        grouper={
            "type": "MultiScaleGroup",
            "branch": branch,
            "radius": 0.4,
            "radius_scales": [0.5, 1.0],
            "fusion_layers": 1,
            "fusion_heads": 2,
            "fusion_feedforward_channels": 16,
        }
    )
    torch.manual_seed(0)
    model = DETECTORS.build(node).eval()
    with torch.no_grad():
        model.proposal.score.weight.mul_(0.01)
        model.proposal.score.bias.copy_(torch.tensor([0.0, 1.0, 1.0]))
    observation, _ = batch()
    assert [len(group) for group in model.predict(observation)] == [SEEDS, SEEDS]


# ------------------------------------------------------------- the reference model


@requires_spconv
def test_the_reference_model_has_the_reference_size():
    model = DETECTORS.build(Config.load(MODEL_YAML)["model"])
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_PARAMETERS
    # The stacked query / key / value projections of the two fusion layers are three
    # tensors each here: weight and bias of two layers, two tensors more apiece.
    assert len(model.state_dict()) == REFERENCE_TENSORS + 8
    sizes = {
        name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()
    }
    economic = {"proposal": 1_539, "view": 416_556, "head": 84_185}
    assert {name: sizes[name] for name in economic} == economic  # EconomicGrasp's, unchanged
    assert sizes["backbone"] == 14_646_816 + 125 * 3 * 32  # three more input channels in the stem
    branches = [sum(p.numel() for p in b.parameters()) for b in model.grouper.branches]
    assert branches == [601_142] * 4  # four times EconomicGrasp's grouper
    fusion = sum(p.numel() for p in model.grouper.fusion.parameters())
    assert fusion == 2 * 789_760 + 257 and sizes["grouper"] == 4 * 601_142 + fusion
    assert model.grouper.radii == (0.07 * 0.25, 0.07 * 0.5, 0.07 * 0.75, 0.07)
    assert model.backbone.in_channels == 6 and model.point_features == "observation"


# ----------------------------------------------------------------------- converter


def finegrasp_state(seed: int = 0) -> dict[str, Tensor]:
    """A random state dict with the layout of the reference FineGrasp."""
    generator = torch.Generator().manual_seed(seed + 2)
    economic = economic_state(seed)
    state = {k: v for k, v in economic.items() if not k.startswith("cy_group.")}
    state["backbone.conv0p1s1.kernel"] = torch.randn(125, 6, 32, generator=generator)
    for scale in range(4):
        for key, value in economic.items():
            if key.startswith("cy_group."):
                name = key.replace("cy_group.", f"cy_groups.{scale}.", 1)
                state[name] = value + scale if value.is_floating_point() else value.clone()

    def tensor(name: str, *shape: int) -> None:
        state[f"fuse_multi_scale.{name}"] = torch.randn(shape, generator=generator)

    for layer in range(2):
        prefix = f"transformer.layers.{layer}"
        tensor(f"{prefix}.self_attn.in_proj_weight", 768, 256)
        tensor(f"{prefix}.self_attn.in_proj_bias", 768)
        tensor(f"{prefix}.self_attn.out_proj.weight", 256, 256)
        tensor(f"{prefix}.self_attn.out_proj.bias", 256)
        tensor(f"{prefix}.linear1.weight", 1024, 256)
        tensor(f"{prefix}.linear1.bias", 1024)
        tensor(f"{prefix}.linear2.weight", 256, 1024)
        tensor(f"{prefix}.linear2.bias", 256)
        for norm in ("norm1", "norm2"):
            tensor(f"{prefix}.{norm}.weight", 256)
            tensor(f"{prefix}.{norm}.bias", 256)
    tensor("attn.weight", 1, 256)
    tensor("attn.bias", 1)
    return state


@requires_spconv
def test_converter_maps_every_reference_tensor_to_its_place():
    converter = load_converter()
    source = finegrasp_state()
    assert len(source) == REFERENCE_TENSORS
    converted = converter.convert_finegrasp_reference(source)
    assert len(converted) == REFERENCE_TENSORS + 8
    model = DETECTORS.build(Config.load(MODEL_YAML)["model"])
    converter.verify_against(model, converted)
    model.load_state_dict(converted, strict=True)
    assert sum(v.numel() for v in converted.values()) == sum(v.numel() for v in source.values())

    assert tuple(model.backbone.stem.conv.weight.shape)[-1] == 6
    for scale, branch in enumerate(model.grouper.branches):  # the smallest radius first
        prefix = f"cy_groups.{scale}"
        assert torch.equal(
            branch.mlp[0].linear.weight, source[f"{prefix}.mlps.layer0.conv.weight"][:, :, 0, 0]
        )
        assert torch.equal(
            branch.post_mlp[1].norm.running_var, source[f"{prefix}.mlps2.layer1.bn.bn.running_var"]
        )
        for slot, role in enumerate(("query", "key", "value", "output")):
            reference = source[f"{prefix}.local_interaction_module.msa.linears.{slot}.weight"]
            assert torch.equal(getattr(branch.interaction, role).weight, reference)
    for index, layer in enumerate(model.grouper.fusion.layers):
        prefix = f"fuse_multi_scale.transformer.layers.{index}"
        stacked = source[f"{prefix}.self_attn.in_proj_weight"]
        bias = source[f"{prefix}.self_attn.in_proj_bias"]
        for part, role in enumerate(("query", "key", "value")):
            rows = slice(part * 256, (part + 1) * 256)
            assert torch.equal(getattr(layer.interaction, role).weight, stacked[rows])
            assert torch.equal(getattr(layer.interaction, role).bias, bias[rows])
        assert torch.equal(
            layer.interaction.output.weight, source[f"{prefix}.self_attn.out_proj.weight"]
        )
        assert torch.equal(layer.interaction.norm.bias, source[f"{prefix}.norm1.bias"])
        assert torch.equal(layer.expand.weight, source[f"{prefix}.linear1.weight"])
        assert torch.equal(layer.contract.bias, source[f"{prefix}.linear2.bias"])
        assert torch.equal(layer.norm.weight, source[f"{prefix}.norm2.weight"])
    assert torch.equal(model.grouper.fusion.score.weight, source["fuse_multi_scale.attn.weight"])
    assert torch.equal(model.view.hidden.weight, source["view.conv1.weight"][..., 0])
    assert torch.equal(
        model.head.project["angle"].weight, source["grasp_head.conv_angle_feature.weight"][..., 0]
    )


@requires_spconv
def test_converter_refuses_checkpoints_that_do_not_fit():
    converter = load_converter()
    source = finegrasp_state()
    model = DETECTORS.build(Config.load(MODEL_YAML)["model"])
    with pytest.raises(
        converter.ConversionError, match="unexpected tensor 'extra.weight' in a FineGrasp"
    ):
        converter.convert_finegrasp_reference({**source, "extra.weight": torch.zeros(1)})
    # An EconomicGrasp checkpoint is not a FineGrasp checkpoint, and the other way round.
    with pytest.raises(converter.ConversionError, match="unexpected tensor 'cy_group.mlps"):
        converter.convert_finegrasp_reference(economic_state())
    with pytest.raises(converter.ConversionError, match="in an EconomicGrasp checkpoint"):
        converter.convert_economicgrasp_reference(source)
    partial = dict(source)
    del partial["fuse_multi_scale.transformer.layers.1.self_attn.in_proj_bias"]
    with pytest.raises(converter.ConversionError, match=r"not filled \(3\)"):
        converter.verify_against(model, converter.convert_finegrasp_reference(partial))
    odd = dict(source)
    odd["fuse_multi_scale.transformer.layers.0.self_attn.in_proj_bias"] = torch.zeros(767)
    with pytest.raises(converter.ConversionError, match="not three stacked projections"):
        converter.convert_finegrasp_reference(odd)
    economic_stem = dict(source)
    economic_stem["backbone.conv0p1s1.kernel"] = torch.zeros(125, 3, 32)  # no normal channels
    with pytest.raises(converter.ConversionError, match="shape mismatch"):
        converter.verify_against(model, converter.convert_finegrasp_reference(economic_stem))
    fifth = {**source, "cy_groups.4.mlps.layer0.conv.weight": torch.zeros(256, 515, 1, 1)}
    with pytest.raises(converter.ConversionError, match="no such tensor in the model"):
        converter.verify_against(model, converter.convert_finegrasp_reference(fifth))


def write_safetensors(path, tensors: dict[str, Tensor]) -> None:
    """The file format by hand: header length, JSON header, raw little-endian data."""
    tags = {torch.float32: "F32", torch.int64: "I64", torch.float64: "F64", torch.bool: "BOOL"}
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, tensor in tensors.items():
        blob = tensor.contiguous().numpy().tobytes()
        header[name] = {
            "dtype": tags[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def test_safetensors_files_are_read_without_the_library(tmp_path):
    converter = load_converter()
    tensors = {
        "weight": torch.randn(3, 4),
        "steps": torch.tensor(7),
        "double": torch.arange(5, dtype=torch.float64),
        "mask": torch.tensor([True, False]),
    }
    write_safetensors(tmp_path / "model.safetensors", tensors)
    loaded = converter.load_safetensors(tmp_path / "model.safetensors")
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype and torch.equal(loaded[name], tensor)
    header = {"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    encoded = json.dumps(header).encode()
    (tmp_path / "odd.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0")
    with pytest.raises(converter.ConversionError, match="unsupported dtype BF16"):
        converter.load_safetensors(tmp_path / "odd.safetensors")


@requires_spconv
def test_converter_command_reads_the_released_file_format(tmp_path, capsys):
    converter = load_converter()
    source = tmp_path / "model.safetensors"
    write_safetensors(source, finegrasp_state())
    destination = tmp_path / "out" / "finegrasp.pth"
    arguments = [str(source), str(destination), "--source-format", "finegrasp"]
    assert converter.main([*arguments, "--camera", "realsense"]) == 0
    assert "396 tensors, 19,145,441 parameters" in capsys.readouterr().out
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    meta = saved["meta"]
    assert meta["camera"] == "realsense" and meta["source_epoch"] is None
    assert meta["parameters"] == REFERENCE_PARAMETERS and len(meta["source_sha256"]) == 64
    assert meta["model"]["type"] == "EconomicGrasp" and meta["source_format"] == "finegrasp"
    assert meta["model"]["grouper"]["type"] == "MultiScaleGroup"
    assert meta["model_config"] == "_base_/models/finegrasp.yaml"
    DETECTORS.build(Config.load(MODEL_YAML)["model"]).load_state_dict(
        saved["state_dict"], strict=True
    )
    # Read as an EconomicGrasp checkpoint, the file does not fit.
    wrong = [str(source), str(tmp_path / "never.pth"), "--source-format", "economicgrasp"]
    assert converter.main(wrong) == 1
    assert "conversion failed" in capsys.readouterr().err and not (tmp_path / "never.pth").exists()
    assert "finegrasp" in converter.SOURCE_FORMATS
