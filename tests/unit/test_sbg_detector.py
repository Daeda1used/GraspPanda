"""Scale-Balanced-Grasp end to end on tiny inputs: the shipped configs, loss and predict
with the scale-balanced parts switched on, and the checkpoint converter."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from itertools import pairwise

import pytest
import torch
from torch import Tensor

from grasppanda.cli.main import main
from grasppanda.config import Config, ConfigError
from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY
from grasppanda.models.detectors.graspnet_baseline import GraspNetBaseline
from grasppanda.models.detectors.scale_balanced_grasp import ScaleBalancedGrasp
from grasppanda.registry import DETECTORS
from grasppanda.structures import ContractError, Supervision

from .conftest import REPO
from .test_baseline_detector import ANGLES, DEPTHS, POINTS, SEEDS, VIEW_COUNT, batch

SBG_YAML = REPO / "configs" / "_base_" / "models" / "scale_balanced_grasp.yaml"
RUNNABLE = "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e"
#: Tensors and trainable parameters of the reference GraspNet_MSCQ, counted there.
REFERENCE_TENSORS, REFERENCE_PARAMETERS = 190, 1_577_196
CHANNELS = 8


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny_config(**slots) -> dict:
    model = {
        "type": "ScaleBalancedGrasp",
        "backbone": {
            "type": "PointNet2Transformer",
            "npoints": [24, SEEDS],
            "radii": [0.4, 0.8],
            "nsamples": [6, 5],
            "sa_mlps": [[8, 8], [8, CHANNELS]],
            "attention_neighbours": 4,
        },
        "proposal": {"type": "SubsampledSeeds", "num_seeds": SEEDS},
        "view": {
            "type": "ObjectnessViewStage",
            "num_views": VIEW_COUNT,
            "min_graspable_views": 1,
        },
        "assigner": {"type": "ScaleBalancedAssigner"},
        "grouper": {
            "type": "DepthCylinderGroup",
            "radius": 0.4,
            "hmin": -0.1,
            "hmax": [0.1, 0.25],
            "nsample": 5,
            "mlp_channels": [8, CHANNELS],
            "radius_scales": [0.5, 1.0],
            "seed_residual": "gated",
        },
        "head": {
            "type": "OperationHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "hidden_channels": 8,
            "score_supervision": "seeds_with_a_supervised_depth",
            "score_decode": "score_x_objectness_x_tolerance",
        },
    }
    model.update(slots)
    return model


def tiny_model(**slots) -> ScaleBalancedGrasp:
    torch.manual_seed(0)
    return DETECTORS.build(tiny_config(**slots))


def weighted_batch(frames: int = 2):
    observation, supervision = batch(frames)
    histogram = {
        "edges": torch.linspace(0.0, 0.12, 5, dtype=torch.float64),
        "counts": torch.tensor([3, 20, 50, 9]),
    }
    return observation, Supervision(
        supervision.objects, supervision.point_labels, {HISTOGRAM_KEY: [histogram] * frames}
    )


# ------------------------------------------------------------------------- config


def test_the_shipped_configs_are_valid(capsys):
    assert Config.load(SBG_YAML)["model"]["type"] == "ScaleBalancedGrasp"
    assert main(["print-config", str(SBG_YAML), "--check"]) == 0
    out = capsys.readouterr().out
    for component in (
        "model: ScaleBalancedGrasp",
        "model.backbone: PointNet2Transformer",
        "model.proposal: SubsampledSeeds",
        "model.view: ObjectnessViewStage",
        "model.assigner: ScaleBalancedAssigner",
        "model.grouper: DepthCylinderGroup",
        "model.head: OperationHead",
    ):
        assert component in out
    runnable = Config.load(RUNNABLE).resolved()
    assert runnable["train"]["epochs"] == 18 and runnable["train"]["scheduler_unit"] == "iteration"
    assert runnable["scheduler"]["type"] == "OneCycleLR" and runnable["scheduler"]["max_lr"] == 1e-3
    assert runnable["data"]["train"]["batch_size"] == 2
    assert [hook["type"] for hook in runnable["hooks"]] == ["BNMomentumHook"]
    assert set(runnable["evaluation"]["variants"]) == {"raw", "cd"}
    train = runnable["data"]["train"]["dataset"]
    assert train["scale_statistic"]["source"] == "native" and train["scale_statistic"]["bins"] == 32
    assert train["tolerance"]["max_friction"] == 0.55
    assert runnable["data"]["val"]["dataset"].get("scale_statistic") is None
    assert runnable["data"]["val"]["dataset"]["split"] == "val_dev"  # not a test split


def test_the_reference_model_differs_from_the_baseline_exactly_where_the_method_does():
    sbg = Config.load(SBG_YAML).resolved()["model"]
    baseline = Config.load(REPO / "configs/_base_/models/graspnet_baseline.yaml").resolved()[
        "model"
    ]
    assert sbg["backbone"]["type"] == "PointNet2Transformer" != baseline["backbone"]["type"]
    assert sbg["assigner"]["type"] == "ScaleBalancedAssigner"
    assert sbg["proposal"] == baseline["proposal"]
    different = {
        slot: sorted(k for k in sbg[slot] if sbg[slot][k] != baseline[slot][k])
        for slot in ("view", "grouper", "head")
    }
    assert different == {
        "view": ["min_graspable_views"],
        "grouper": ["radius", "radius_scales", "seed_residual"],
        "head": ["score_decode", "score_supervision"],
    }
    assert sbg["view"]["min_graspable_views"] == 10 and sbg["grouper"]["radius"] == 0.08
    assert sbg["grouper"]["radius_scales"] == [0.25, 0.5, 0.75, 1.0]
    for key in ("crop_centre", "supervise", "max_match_distance", "padded_seeds"):
        assert sbg["assigner"][key] == baseline["assigner"][key], key


def test_the_reference_model_has_the_reference_size():
    model = DETECTORS.build(Config.load(SBG_YAML)["model"])
    assert isinstance(model, GraspNetBaseline)  # the same flow and the same checks
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_PARAMETERS
    assert len(model.state_dict()) == REFERENCE_TENSORS
    assert model.backbone.output == "subsampled" and model.backbone.out_channels == 256
    assert model.grouper.radii == pytest.approx((0.02, 0.04, 0.06, 0.08))
    assert model.grouper.fuse.in_features == 1024 and model.grouper.uses_seed_features
    assert model.assigner.gripper is model.gripper


def test_mixing_is_still_a_config_error():
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_config(head={"type": "SWADHead"})})
    assert "SWADHead expects ('seed', 'channel')" in error.value.issues[0].message
    with pytest.raises(ConfigError) as error:
        Config.from_dict(
            {"model": tiny_config(proposal={"type": "GraspnessFPS", "num_seeds": SEEDS})}
        )
    assert "GraspnessFPS requires 'dense' scene features" in error.value.issues[0].message


# --------------------------------------------------------------------- loss, predict


def test_loss_uses_the_seed_weights_and_backpropagates_everywhere():
    model = tiny_model().train()
    observation, supervision = weighted_batch()
    out = model.run_stages(observation, supervision)
    assert out.matched.weight is not None and len(torch.unique(out.matched.weight)) > 1
    assert out.groups.feats.shape == (2 * SEEDS, DEPTHS, CHANNELS)
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
    assert all(torch.isfinite(value) for value in losses.values())
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.grouper.gate.weight.grad.abs().sum() > 0  # the seed residual is trained
    assert model.backbone.transformer1.query.weight.grad.abs().sum() > 0


def test_the_weights_change_the_losses_and_need_the_histogram():
    model = tiny_model().train()
    observation, supervision = weighted_batch()
    torch.manual_seed(1)
    weighted = model.loss(observation, supervision)
    flat = Supervision(
        supervision.objects,
        supervision.point_labels,
        {
            HISTOGRAM_KEY: [
                {"edges": torch.tensor([0.0, 0.06, 0.12]).double(), "counts": torch.tensor([7, 7])}
            ]
            * 2
        },
    )
    uniform = model.loss(observation, flat)  # equal counts: every seed has weight 1
    assert float(uniform["loss/objectness"]) == pytest.approx(float(weighted["loss/objectness"]))
    assert float(uniform["loss/view"]) != pytest.approx(float(weighted["loss/view"]), rel=1e-4)
    bare = Supervision(supervision.objects, supervision.point_labels)
    with pytest.raises(ContractError, match="scale_statistic"):
        model.loss(observation, bare)


def test_predict_needs_no_histogram_and_scores_include_objectness():
    model = tiny_model().eval()
    observation, _ = weighted_batch()
    out = model.run_stages(observation)
    grasps = model.predict(observation)
    assert [len(g) for g in grasps] == out.seeds.valid.reshape(2, SEEDS).sum(dim=1).tolist()
    assert all(g.score_kind == "predicted_quality_x_objectness_x_tolerance" for g in grasps)
    for group in grasps:
        group.validate()
    plain = tiny_model(
        head={
            "type": "OperationHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "hidden_channels": 8,
            "score_decode": "score_x_tolerance",
        }
    ).eval()
    plain.load_state_dict(model.state_dict())
    probability = out.seeds.scores["objectness"][out.seeds.valid]
    with_objectness = torch.cat([g.score for g in grasps])
    without = torch.cat([g.score for g in plain.predict(observation)])
    assert torch.allclose(with_objectness, without * probability, atol=1e-6)
    assert len(observation.points) // 2 == POINTS


# ----------------------------------------------------------------------- converter


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_state(seed: int = 0) -> dict[str, Tensor]:
    """A random state dict with the layout of the reference GraspNet_MSCQ."""
    generator = torch.Generator().manual_seed(seed)
    state: dict[str, Tensor] = {}

    def norm(prefix: str, channels: int) -> None:
        for name in ("weight", "bias", "running_mean"):
            state[f"{prefix}.{name}"] = torch.randn(channels, generator=generator)
        state[f"{prefix}.running_var"] = torch.rand(channels, generator=generator) + 0.5
        state[f"{prefix}.num_batches_tracked"] = torch.tensor(5)

    def shared_mlp(prefix: str, widths: list[int]) -> None:
        for layer, (cin, cout) in enumerate(pairwise(widths)):
            state[f"{prefix}.layer{layer}.conv.weight"] = torch.randn(
                cout, cin, 1, 1, generator=generator
            )
            norm(f"{prefix}.layer{layer}.bn.bn", cout)

    def linear(key: str, cout: int, cin: int, bias: bool = True) -> None:
        state[f"{key}.weight"] = torch.randn(cout, cin, generator=generator)
        if bias:
            state[f"{key}.bias"] = torch.randn(cout, generator=generator)

    def small_head(prefix: str, widths: list[int]) -> None:
        for layer, (cin, cout) in enumerate(pairwise(widths), start=1):
            state[f"{prefix}.conv{layer}.weight"] = torch.randn(cout, cin, 1, generator=generator)
            state[f"{prefix}.conv{layer}.bias"] = torch.randn(cout, generator=generator)
        for layer, channels in enumerate(widths[1:-1], start=1):
            norm(f"{prefix}.bn{layer}", channels)

    backbone = "view_estimator.backbone"
    shared_mlp(f"{backbone}.sa1.mlp_module", [3, 64, 64, 128])
    shared_mlp(f"{backbone}.sa2.mlp_module", [131, 128, 128, 256])
    for block, channels in (("transformer1", 128), ("transformer2", 256)):
        prefix = f"{backbone}.{block}"
        linear(f"{prefix}.fc1", channels, channels)
        linear(f"{prefix}.fc2", channels, channels)
        linear(f"{prefix}.fc_delta.0", channels, 3)
        linear(f"{prefix}.fc_delta.2", channels, channels)
        linear(f"{prefix}.fc_gamma.0", channels, channels)
        linear(f"{prefix}.fc_gamma.2", channels, channels)
        for name in ("w_qs", "w_ks", "w_vs"):
            linear(f"{prefix}.{name}", channels, channels, bias=False)
    small_head("view_estimator.vpmodule", [256, 256, 302, 302])
    for crop in range(1, 5):
        shared_mlp(f"grasp_generator.crop{crop}.mlps", [3, 64, 128, 256])
    small_head("grasp_generator.operation", [256, 128, 128, 36])
    small_head("grasp_generator.tolerance", [256, 128, 128, 12])
    state["grasp_generator.fuse_multi_scale.weight"] = torch.randn(
        256, 1024, 1, generator=generator
    )
    state["grasp_generator.fuse_multi_scale.bias"] = torch.randn(256, generator=generator)
    state["grasp_generator.gate_fusion.0.weight"] = torch.randn(256, 256, 1, generator=generator)
    state["grasp_generator.gate_fusion.0.bias"] = torch.randn(256, generator=generator)
    return state


def test_converter_maps_every_reference_tensor_to_its_place():
    converter = load_converter()
    source = reference_state()
    assert len(source) == REFERENCE_TENSORS
    converted = converter.convert_scale_balanced_grasp(source)
    model = DETECTORS.build(Config.load(SBG_YAML)["model"])
    converter.verify_against(model, converted)
    model.load_state_dict(converted, strict=True)
    backbone, generator = "view_estimator.backbone", "grasp_generator"
    block = model.backbone.transformer2
    assert torch.equal(block.embed.weight, source[f"{backbone}.transformer2.fc1.weight"])
    assert torch.equal(block.project.bias, source[f"{backbone}.transformer2.fc2.bias"])
    assert torch.equal(
        block.position[0].weight, source[f"{backbone}.transformer2.fc_delta.0.weight"]
    )
    assert torch.equal(block.attention[2].bias, source[f"{backbone}.transformer2.fc_gamma.2.bias"])
    assert torch.equal(block.query.weight, source[f"{backbone}.transformer2.w_qs.weight"])
    assert torch.equal(block.key.weight, source[f"{backbone}.transformer2.w_ks.weight"])
    assert torch.equal(block.value.weight, source[f"{backbone}.transformer2.w_vs.weight"])
    assert torch.equal(
        model.backbone.sa2.mlp_module.layer0.conv.weight,
        source[f"{backbone}.sa2.mlp_module.layer0.conv.weight"],
    )
    for crop in range(1, 5):  # crop1 is the smallest radius, scale 0
        assert torch.equal(
            model.grouper.scales[crop - 1][1].linear.weight,
            source[f"{generator}.crop{crop}.mlps.layer1.conv.weight"][:, :, 0, 0],
        )
    assert torch.equal(
        model.grouper.scales[3][2].norm.running_var,
        source[f"{generator}.crop4.mlps.layer2.bn.bn.running_var"],
    )
    assert torch.equal(
        model.grouper.fuse.weight, source[f"{generator}.fuse_multi_scale.weight"][..., 0]
    )
    assert torch.equal(model.grouper.gate.bias, source[f"{generator}.gate_fusion.0.bias"])
    assert torch.equal(
        model.view.output.weight, source["view_estimator.vpmodule.conv3.weight"][..., 0]
    )
    assert torch.equal(
        model.head.tolerance.output.bias, source[f"{generator}.tolerance.conv3.bias"]
    )


def test_converter_keeps_the_two_formats_apart():
    converter = load_converter()
    source = reference_state()
    with pytest.raises(converter.ConversionError, match="unexpected backbone tensor"):
        converter.convert_graspnet_baseline(source)  # attention blocks are not baseline
    head_only = {k: v for k, v in source.items() if "backbone" not in k}
    with pytest.raises(converter.ConversionError, match="crop1.*graspnet-baseline checkpoint"):
        converter.convert_graspnet_baseline(head_only)
    extra = {**source, "grasp_generator.crop5.mlps.layer0.conv.weight": torch.zeros(64, 3, 1, 1)}
    model = DETECTORS.build(Config.load(SBG_YAML)["model"])
    with pytest.raises(converter.ConversionError, match="no such tensor in the model"):
        converter.verify_against(model, converter.convert_scale_balanced_grasp(extra))
    partial = dict(source)
    del partial["grasp_generator.gate_fusion.0.bias"]
    with pytest.raises(converter.ConversionError, match=r"not filled \(1\): grouper.gate.bias"):
        converter.verify_against(model, converter.convert_scale_balanced_grasp(partial))
    with pytest.raises(converter.ConversionError, match="scale-balanced-grasp checkpoint"):
        converter.convert_scale_balanced_grasp({"grasp_generator.shift.weight": torch.zeros(1)})


def test_converter_command_writes_weights_with_provenance(tmp_path, capsys):
    converter = load_converter()
    source = tmp_path / "reference.tar"
    torch.save({"epoch": 18, "model_state_dict": reference_state()}, source)
    destination = tmp_path / "sbg.pth"
    arguments = [str(source), str(destination), "--source-format", "scale-balanced-grasp"]
    assert converter.main([*arguments, "--camera", "realsense"]) == 0
    assert "190 tensors, 1,577,196 parameters" in capsys.readouterr().out
    meta = torch.load(destination, map_location="cpu", weights_only=True)["meta"]
    assert meta["source_format"] == "scale-balanced-grasp"
    assert meta["model"]["type"] == "ScaleBalancedGrasp"
    assert meta["model_config"] == "_base_/models/scale_balanced_grasp.yaml"
    assert meta["parameters"] == REFERENCE_PARAMETERS
