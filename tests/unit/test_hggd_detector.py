"""HGGD as a detector: the outer contract, its configs, its networks and its hooks.

HGGD shares no stage with the seed-based family, so these tests are also the check
that the outer contracts (``loss`` / ``predict``, ``Observation`` / ``Supervision``,
``GraspGroup``, the config validator, the hooks) carry a method that has no seeds, no
view and no crop frame.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from grasppanda.config import Config, ConfigError
from grasppanda.datasets import BaseDataset
from grasppanda.engine.commands import infer, train
from grasppanda.engine.errors import EngineError
from grasppanda.engine.hooks.bn_momentum import BNMomentumHook
from grasppanda.engine.hooks.epoch_signal import EpochSignalHook
from grasppanda.evaluation import load_prediction
from grasppanda.models import BaseGraspDetector
from grasppanda.models.detectors.hggd import HGGD, ImageNet, LocalNet, RegionDraws
from grasppanda.registry import DATASETS, DETECTORS, GROUPERS, HEADS, HOOKS, PROPOSALS, VIEWS
from grasppanda.structures import (
    Batch,
    ContractError,
    GraspGroup,
    Observation,
    PointSet,
    SampleKey,
    Supervision,
    collate,
)

REPO = Path(__file__).resolve().parents[2]
SENSOR = (80, 128)  # (H, W); the tiny network image is 64 x 40
TINY = {
    "type": "HGGD",
    "image": {"size": [64, 40]},
    "proposal": {"max_proposals": 6, "train_max_proposals": 8},
    "region": {"num_points": 32, "min_points": 8},
    "anchors": {"top_k": 4, "shift_epochs": 0},
}
CONFIGS = ("hggd/hggd_r34w8_g1b-realsense_15e", "hggd/hggd_r34w8_g1b-kinect_15e")


def tiny(**overrides) -> HGGD:
    torch.manual_seed(0)
    node = {**TINY, **{k: {**TINY.get(k, {}), **v} for k, v in overrides.items()}}
    return DETECTORS.build(node)


def frame(index: int, seed: int = 0) -> Batch:
    """One synthetic frame: a tilted plane seen by a 128 x 80 sensor, with images."""
    generator = torch.Generator().manual_seed(seed + index)
    height, width = SENSOR
    v, u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    depth = 0.5 + 0.001 * u + 0.02 * torch.rand(height, width, generator=generator)
    depth[:3, :5] = 0.0  # a corner without depth
    intrinsics = torch.tensor([[100.0, 0.0, 63.5], [0.0, 101.0, 39.5], [0.0, 0.0, 1.0]])
    valid = depth > 0
    cloud = torch.stack([(u - 63.5) * depth / 100.0, (v - 39.5) * depth / 101.0, depth], dim=-1)
    points = cloud[valid]
    picked = torch.randperm(len(points), generator=generator)[:1500]
    key = SampleKey("hggdunit", 5, "realsense", index)
    observation = Observation(
        keys=[key],
        intrinsics=intrinsics[None],
        depth=depth[None],
        rgb=torch.rand(1, 3, height, width, generator=generator),
        points=PointSet(points[picked]),
    )
    count = 60
    centre_u = torch.rand(count, generator=generator, dtype=torch.float64) * 100 + 14
    centre_v = torch.rand(count, generator=generator, dtype=torch.float64) * 60 + 10
    labels = {
        "centers": torch.stack([centre_u, centre_v], dim=1),
        "depth": depth[centre_v.long(), centre_u.long()].double() + 0.005,
        "theta": (torch.rand(count, generator=generator, dtype=torch.float64) - 0.5) * 2 * math.pi,
        "gamma": (torch.rand(count, generator=generator, dtype=torch.float64) - 0.5),
        "beta": (torch.rand(count, generator=generator, dtype=torch.float64) - 0.5),
        "width": torch.full((count,), 30.0, dtype=torch.float64),
        "height": torch.full((count,), 12.0, dtype=torch.float64),
        "score": torch.ones(count, dtype=torch.float64),
        "object_id": torch.zeros(count, dtype=torch.long),
    }
    return Batch(observation, Supervision(objects=[[]], extras={"view_grasps": [labels]}))


def batch(frames: int = 2) -> Batch:
    return collate([frame(index) for index in range(frames)])


# ------------------------------------------------------------- registration, configs


def test_hggd_is_a_detector_family_of_its_own():
    assert DETECTORS.get("HGGD") is HGGD and issubclass(HGGD, BaseGraspDetector)
    assert HGGD.required_inputs == frozenset({"rgb", "depth", "intrinsics", "points"})
    # Its stages are private: none of them is offered to the slots of the seed-based family.
    private = {"HeatmapStage", "RegionStage", "AnchorHead", "ImageNet", "LocalNet"}
    for registry in (PROPOSALS, VIEWS, GROUPERS, HEADS, DETECTORS):
        assert not private & set(registry.names())
    assert "EpochSignalHook" in HOOKS.names()


@pytest.mark.parametrize("name", CONFIGS)
def test_shipped_configs_validate_and_state_the_reference_recipe(name):
    config = Config.load(name)
    cfg = config.resolved()
    camera = "kinect" if "kinect" in name else "realsense"
    assert cfg["model"]["type"] == "HGGD"
    assert cfg["model"]["image"] == {"size": [640, 360], "resize": "pil_bicubic", "depth_clip": 1.0}
    assert cfg["model"]["proposal"]["max_proposals"] == 48
    assert cfg["model"]["proposal"]["train_max_proposals"] == 128
    assert cfg["model"]["anchors"]["top_k"] == 10 and cfg["model"]["anchors"]["shift_epochs"] == 5
    assert cfg["optimizer"] == {"type": "AdamW", "lr": 0.01, "weight_decay": 0.01} | {
        k: v for k, v in cfg["optimizer"].items() if k not in ("type", "lr", "weight_decay")
    }
    assert cfg["scheduler"]["type"] == "StepLR" and cfg["scheduler"]["step_size"] == 5
    train = cfg["train"]
    assert (train["epochs"], train["accumulate"], train["grad_clip_mode"]) == (15, 2, "value")
    assert train["grad_clip"] == 0.5  # the reference clips the SUM of two batches at 1
    assert [hook["type"] for hook in cfg["hooks"]] == ["EpochSignalHook", "BNMomentumHook"]
    assert cfg["hooks"][1]["module"] == "image_net" and cfg["hooks"][1]["delay"] == 1
    for section in ("train", "val", "test"):
        dataset = cfg["data"][section]["dataset"]
        assert dataset["camera"] == camera and dataset["images"] == ["rgb", "depth"]
        assert [step["type"] for step in dataset["preprocess"]] == ["DepthToCloud", "NoWorkspace"]
        assert dataset["transforms"] == [{"type": "RandomSample", "num_points": 25600}]
    training = cfg["data"]["train"]
    assert (training["batch_size"], training["drop_last"]) == (4, True)
    assert training["dataset"]["object_labels"] is False
    assert training["dataset"]["view_grasps"]["friction_threshold"] == 0.2
    variants = cfg["evaluation"]["variants"]
    assert list(variants) == ["raw", "cd"] and variants["raw"] == []
    assert [step["type"] for step in variants["cd"]] == ["ModelFreeCollisionFilter", "GraspNMS"]
    collision = variants["cd"][0]
    assert collision["cloud"] == "points" and collision["precision"] == "float16"
    assert (collision["height"], collision["depth"], collision["min_points_between"]) == (
        0.01,
        0.02,
        2,
    )
    assert cfg["evaluation"]["evaluator"]["type"] == "GraspNet1BEvaluator"


def test_reference_model_has_the_reference_parameter_counts():
    model = DETECTORS.build(Config.load("_base_/models/hggd.yaml")["model"])
    assert sum(p.numel() for p in model.image_net.parameters()) == 1_618_921
    assert sum(p.numel() for p in model.local_net.parameters()) == 1_806_669
    assert not list(model.proposal.parameters()) and not list(model.region.parameters())
    assert not list(model.anchors.parameters()) and len(list(model.anchors.buffers())) == 2
    assert len(model.state_dict()) == 297 + 76 + 2


def test_config_mistakes_are_reported_before_anything_is_built():
    with pytest.raises(ConfigError, match="max_proposal"):
        Config.from_dict({"model": {"type": "HGGD", "proposal": {"max_proposal": 48}}})
    with pytest.raises(ConfigError, match="pil_bicubic"):
        Config.from_dict({"model": {"type": "HGGD", "image": {"resize": "bilinear"}}})
    with pytest.raises(ConfigError, match="not a multiple of proposal.stride"):
        Config.from_dict({"model": {"type": "HGGD", "image": {"size": [640, 364]}}})
    with pytest.raises(ConfigError, match="only 49 anchors"):
        Config.from_dict({"model": {"type": "HGGD", "anchors": {"top_k": 50}}})
    with pytest.raises(ConfigError, match="backbone"):  # no seed-based slot exists here
        Config.from_dict({"model": {"type": "HGGD", "backbone": {"type": "SparseUNet14"}}})
    with pytest.raises(ContractError, match="not a multiple of proposal.stride"):
        HGGD(image={"size": [60, 40]})


# -------------------------------------------------------------------------- networks


def test_image_network_shapes_and_the_sizes_it_is_laid_out_for():
    network = ImageNet(in_channels=4, num_angles=6, stride=8).eval()
    out = network(torch.randn(2, 4, 64, 40))
    assert tuple(out.heatmap.shape) == (2, 64, 40)
    for maps in (out.angle, out.angle_offset, out.depth_offset, out.width_offset):
        assert tuple(maps.shape) == (2, 6, 8, 5)
    assert tuple(out.features.shape) == (2, 32, 8, 5) and network.feature_channels == 32
    assert not out.features.requires_grad and out.heatmap.requires_grad
    attached = ImageNet(detach_features=False)(torch.randn(1, 4, 64, 40))
    assert attached.features.requires_grad
    encoder_channels = [m.shape[1] for m in network.encode(torch.randn(1, 4, 64, 40))]
    assert encoder_channels == [8, 16, 32, 64, 128]  # ResNet-34 layout at width 8
    with pytest.raises(ContractError, match="laid out for 640 x 360"):
        network(torch.randn(1, 4, 64, 48))
    with pytest.raises(ValueError, match="stride must be 2, 4, 8 or 16"):
        ImageNet(stride=3)


def test_region_network_reads_regions_and_the_partial_grasp():
    network = LocalNet(feature_channels=32, num_anchors=49).eval()
    regions = torch.randn(5, 64, 35)
    info = torch.randn(5, 3)
    logits, offsets = network(regions, info)
    assert tuple(logits.shape) == (5, 49) and tuple(offsets.shape) == (5, 49, 3)
    shuffled = regions[:, torch.randperm(64)]
    assert torch.allclose(network(shuffled, info)[0], logits, atol=1e-5)  # a set, not a sequence
    assert not torch.allclose(network(regions, info + 1.0)[0], logits)
    with pytest.raises(ContractError, match=r"expects regions \(R, S, 35\)"):
        network(torch.randn(5, 64, 6), info)
    with pytest.raises(ContractError, match="side inputs for 5 regions"):
        network(regions, info[:3])


# --------------------------------------------------------------------------- predict


def test_predict_returns_keyed_groups_with_a_declared_score():
    model = tiny().eval()
    data = batch(2)
    torch.manual_seed(1)
    groups = model.predict(data.observation)
    assert len(groups) == 2 and all(isinstance(group, GraspGroup) for group in groups)
    assert [group.key for group in groups] == data.observation.keys
    for group in groups:
        assert 0 < len(group) <= 6 * 4  # at most top_k anchors of max_proposals regions
        assert group.score_kind == "anchor_probability" and group.frame == "camera"
        assert group.translation.dtype == torch.float32
        assert float(group.score.min()) > 0 and float(group.score.max()) < 1
        assert torch.all(group.depth == 0.02) and torch.all(group.width > 0)
        assert torch.all(group.translation[:, 2] > 0.3)  # in front of the camera, near the plane
    assert model(data.observation)[0].key == groups[0].key  # forward() without labels predicts


def test_a_frame_without_proposals_gets_an_explicit_empty_group():
    model = tiny(proposal={"score_threshold": 1.0}).eval()  # no heat value exceeds 1
    data = batch(2)
    groups = model.predict(data.observation)
    assert [len(group) for group in groups] == [0, 0]
    assert [group.key for group in groups] == data.observation.keys
    assert all(group.score_kind == "anchor_probability" for group in groups)
    out = model.run_stages(data.observation)
    assert out.proposals.tolist() == [0, 0] and len(out.regions) == 0
    assert tuple(out.logits.shape) == (0, 49)
    starved = tiny(proposal={"anchor_width": 1e-3}).eval()  # proposals whose balls hold no point
    out = starved.run_stages(data.observation)
    assert out.proposals.tolist() == [6, 6] and len(out.regions) == 0 and not out.regions.kept.any()
    assert [len(group) for group in starved.predict(data.observation)] == [0, 0]


def test_recorded_draws_make_a_run_repeatable():
    model = tiny().eval()
    data = frame(0)
    with torch.no_grad():
        first = model.run_stages(data.observation)
    support = first.regions.support
    longest = max(32, int(support.max()))
    draws = RegionDraws(
        resample=[],
        jitter=torch.zeros(len(support), longest, 3),
        start=torch.zeros(len(support)).long(),
    )
    with torch.no_grad():
        one = model.run_stages(data.observation, draws=[draws])
        two = model.run_stages(data.observation, draws=[draws])
    assert torch.equal(one.regions.points, two.regions.points) and torch.equal(
        one.logits, two.logits
    )
    forced = [torch.tensor([[20, 10], [40, 30]])]
    replayed = model.run_stages(data.observation, candidates=forced, draws=None)
    assert replayed.regions.grasps.pixel.tolist() == [[20, 10], [40, 30]]


def test_missing_or_inconsistent_inputs_are_named():
    model = tiny().eval()
    observation = frame(0).observation
    without = Observation(
        keys=observation.keys, intrinsics=observation.intrinsics, points=observation.points
    )
    with pytest.raises(
        ContractError, match=r"\['depth', 'rgb'\] is missing.*images: \[rgb, depth\]"
    ):
        model.predict(without)
    odd = Observation(
        keys=observation.keys,
        intrinsics=observation.intrinsics,
        depth=torch.rand(1, 90, 128),
        rgb=torch.rand(1, 3, 90, 128),
        points=observation.points,
    )
    with pytest.raises(ContractError, match="not an integer multiple of the network image"):
        model.predict(odd)
    two_keys = Observation(
        keys=observation.keys * 2,
        intrinsics=observation.intrinsics,
        depth=observation.depth,
        rgb=observation.rgb,
        points=observation.points,
    )
    with pytest.raises(ContractError, match=r"lists 2 frame\(s\) but holds 1 cloud"):
        model.predict(two_keys)


# ------------------------------------------------------------------------------ loss


def test_loss_returns_every_term_and_trains_both_networks():
    model = tiny().train()
    data = batch(2)
    torch.manual_seed(2)
    losses = model.loss(data.observation, data.supervision)
    terms = {name for name in losses if name.startswith("loss/")}
    assert terms == {
        "loss/heatmap",
        "loss/angle",
        "loss/offsets",
        "loss/anchor",
        "loss/anchor_offset",
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert {"metric/proposals", "metric/regions", "metric/supervised_regions"} <= set(losses)
    assert float(losses["metric/proposals"]) == 8.0  # the training budget, no heat threshold
    sum(losses[name] for name in terms).backward()
    for part in (model.image_net, model.local_net):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert isinstance(model(data.observation, data.supervision), dict)  # forward() with labels


def test_region_losses_do_not_reach_the_image_network_unless_asked():
    data = batch(2)
    for detach, reaches in ((True, False), (False, True)):
        model = tiny(encoder={"detach_features": detach}).train()
        torch.manual_seed(3)
        losses = model.loss(data.observation, data.supervision)
        (losses["loss/anchor"] + losses["loss/anchor_offset"]).backward()
        touched = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.image_net.parameters()
        )
        assert touched is reaches


def test_loss_explains_missing_labels_and_a_missing_epoch():
    data = batch(1)
    model = tiny().train()
    with pytest.raises(ContractError, match="per-view grasp labels.*--view-grasps"):
        model.loss(data.observation, Supervision(objects=[[]]))
    scheduled = tiny(anchors={"shift_epochs": 5}).train()
    with pytest.raises(ContractError, match="EpochSignalHook"):
        scheduled.loss(data.observation, data.supervision)
    scheduled.eval()  # evaluation-mode losses need no epoch: nothing is shifted
    assert "loss/heatmap" in scheduled.loss(data.observation, data.supervision)
    wrong = Supervision(
        objects=[[], []], extras={"view_grasps": data.supervision.extras["view_grasps"] * 2}
    )
    with pytest.raises(ContractError, match=r"labels of 2 frame\(s\) for a batch of 1"):
        model.loss(data.observation, wrong)


def test_anchors_follow_the_labels_during_the_first_epochs_only():
    model = tiny(anchors={"shift_epochs": 2, "shift_pool": 20, "match_radius": 0.5}).train()
    data = batch(2)
    start = model.anchors.gamma.clone()
    model.on_train_epoch(0, 15)
    torch.manual_seed(4)
    losses = model.loss(data.observation, data.supervision)
    assert float(losses["metric/anchors_moved"]) == 1.0
    assert not torch.equal(model.anchors.gamma, start)
    moved = model.anchors.gamma.clone()
    model.on_train_epoch(2, 15)  # past shift_epochs
    losses = model.loss(data.observation, data.supervision)
    assert float(losses["metric/anchors_moved"]) == 0.0 and torch.equal(model.anchors.gamma, moved)
    assert torch.equal(model.state_dict()["anchors.gamma"], moved)  # saved with the weights


# ----------------------------------------------------------------------------- hooks


def state_for(model: nn.Module, epoch: int) -> SimpleNamespace:
    log = SimpleNamespace(info=lambda *args, **kwargs: None)
    return SimpleNamespace(
        model=model, epoch=epoch, max_epochs=15, run=SimpleNamespace(is_main=True, log=log)
    )


def test_epoch_signal_reaches_the_model_and_fails_when_nobody_listens():
    model = tiny(anchors={"shift_epochs": 5})
    hook = HOOKS.build({"type": "EpochSignalHook"})
    assert isinstance(hook, EpochSignalHook)
    model.anchors._pooled = 7
    hook.before_epoch(state_for(model, 3))
    assert model._epoch == 3 and model.anchors._pooled == 0  # the pool starts empty every epoch
    with pytest.raises(EngineError, match=r"Linear defines no on_train_epoch"):
        hook.before_epoch(state_for(nn.Linear(2, 2), 0))


def test_batch_norm_schedule_can_lag_and_be_limited_to_a_sub_module():
    model = tiny()
    hook = BNMomentumHook(
        initial=0.5, decay=0.5, step=2, minimum=0.001, delay=1, module="image_net"
    )
    assert [hook.momentum(epoch) for epoch in range(6)] == [0.5, 0.5, 0.5, 0.25, 0.25, 0.125]
    assert BNMomentumHook().momentum(2) == 0.25  # without the lag
    hook.before_epoch(state_for(model, 3))
    image = [m for m in model.image_net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    local = [m for m in model.local_net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert image and all(layer.momentum == 0.25 for layer in image)
    assert local and all(layer.momentum == 0.1 for layer in local)  # PyTorch's default, untouched
    with pytest.raises(EngineError, match="no such sub-module"):
        BNMomentumHook(module="backbone").before_epoch(state_for(model, 0))
    with pytest.raises(ValueError, match="delay must not be negative"):
        BNMomentumHook(delay=-1)


# ------------------------------------------------------------------------ conversion


@pytest.fixture(scope="module")
def converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", REPO / "tools" / "convert_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversion_is_strict_about_parts_and_tensors(converter):
    assert converter.SOURCE_FORMATS["hggd"][1] == "_base_/models/hggd.yaml"
    with pytest.raises(
        converter.ConversionError, match=r"not an HGGD checkpoint: \['local', 'beta'\]"
    ):
        converter.convert_hggd_reference({"anchor": {}, "gamma": torch.zeros(7)})
    parts = {"anchor": {}, "local": {}, "gamma": torch.zeros(7), "beta": torch.zeros(7)}
    stray = dict(parts, anchor={"backbone.net.fc.weight": torch.zeros(2)})
    with pytest.raises(
        converter.ConversionError, match="unexpected tensor 'anchor.backbone.net.fc"
    ):
        converter.convert_hggd_reference(stray)
    stray = dict(parts, local={"pointnet.fstn.conv1.weight": torch.zeros(2)})
    with pytest.raises(converter.ConversionError, match="unexpected tensor 'local.pointnet.fstn"):
        converter.convert_hggd_reference(stray)
    few = {
        "anchor": {
            "backbone.net.conv3_x.1.residual_function.3.weight": torch.zeros(32, 32, 3, 3),
            "trconv.2.deconv.1.running_mean": torch.zeros(16),
            "cls_mask_conv.0.conv.0.bias": torch.zeros(32),
            "hmap.1.weight": torch.zeros(1, 32, 1, 1),
        },
        "local": {
            "pointnet.conv2.weight": torch.ones(128, 96, 1),
            "pointnet.stn.fc3.bias": torch.zeros(9),
            "offset_mlp.3.weight": torch.zeros(147, 256),
        },
        "gamma": torch.linspace(-0.7, 0.7, 7),
        "beta": torch.linspace(-0.8, 0.8, 7),
    }
    converted = converter.convert_hggd_reference(few)
    assert set(converted) == {
        "image_net.stages.1.1.conv2.weight",
        "image_net.ups.2.norm.running_mean",
        "image_net.angle.conv.bias",
        "image_net.heatmap.out.weight",
        "local_net.fuse.linear.weight",
        "local_net.align.out.bias",
        "local_net.offset.out.weight",
        "anchors.gamma",
        "anchors.beta",
    }
    assert tuple(converted["local_net.fuse.linear.weight"].shape) == (128, 96)  # 1x1 conv -> linear
    assert converted["anchors.gamma"].dtype == torch.float64
    model = DETECTORS.build(Config.load("_base_/models/hggd.yaml")["model"])
    names = set(model.state_dict())
    assert set(converted) <= names  # every converted name exists in the model
    with pytest.raises(converter.ConversionError, match="not filled"):
        converter.verify_against(model, converted)  # an incomplete conversion never loads


# ------------------------------------------------------------- through the engine


@DATASETS.register()
class HGGDUnitFrames(BaseDataset):
    """The synthetic RGB-D frames above as a dataset with per-view labels."""

    name = "hggdunit"

    def __init__(self, frames: int = 4, labels: bool = True) -> None:
        self.labels = labels
        self._keys = [SampleKey("hggdunit", 5, "realsense", index) for index in range(frames)]

    @property
    def keys(self) -> list[SampleKey]:
        return self._keys

    def __getitem__(self, index: int) -> Batch:
        item = frame(index)
        return item if self.labels else Batch(item.observation)


def engine_config(tmp_path: Path, **runtime) -> Config:
    loader = {"dataset": {"type": "HGGDUnitFrames"}, "batch_size": 1, "num_workers": 0}
    quiet = {
        "dataset": {"type": "HGGDUnitFrames", "labels": False},
        "batch_size": 2,
        "num_workers": 0,
    }
    anchors = {"top_k": 4, "shift_epochs": 1, "shift_pool": 10, "match_radius": 0.5}
    collision = {
        "type": "ModelFreeCollisionFilter",
        "cloud": "points",
        "finger_length": 0.04,
        "height": 0.01,
        "depth": 0.02,
        "collision_threshold": 0.0,
        "min_points_between": 2,
        "precision": "float16",
    }
    return Config.from_dict(
        {
            "model": {**TINY, "anchors": anchors},
            "data": {"train": {**loader, "drop_last": True}, "test": quiet},
            # the reference recipe in small: one AdamW, step schedule, two batches per step,
            # clipping by value, and the two hooks the method needs
            "optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-2},
            "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.1},
            "train": {"epochs": 2, "accumulate": 2, "grad_clip": 0.5, "grad_clip_mode": "value"},
            "hooks": [
                {"type": "EpochSignalHook"},
                {"type": "BNMomentumHook", "delay": 1, "module": "image_net"},
            ],
            "evaluation": {
                "interval": 0,
                "variants": {"raw": [], "cd": [collision, {"type": "GraspNMS"}]},
            },
            "runtime": {"work_dir": str(tmp_path / "work"), "device": "cpu", **runtime},
        }
    )


def test_the_trainer_and_the_prediction_loop_run_hggd_unchanged(tmp_path):
    """Trainer, hooks, checkpoint, prediction and post-processing: no HGGD special case."""
    initial = tiny().anchors.gamma.clone()
    state = train(engine_config(tmp_path))
    model = state.model
    assert isinstance(model, HGGD) and state.epoch == 2 and state.iteration == 8
    assert sorted(state.optimizers) == ["main"]  # one optimizer over both networks
    assert state.optimizers["main"].param_groups[0]["lr"] == pytest.approx(1e-5)
    assert model._epoch == 1  # the hook told the model the epoch
    assert not torch.equal(model.anchors.gamma.cpu(), initial)  # the anchors moved in epoch 0
    checkpoint = sorted((state.run.dir / "checkpoints").glob("*.pth"))[-1]
    dump = infer(engine_config(tmp_path, load_from=str(checkpoint)))
    assert len(dump.keys) == 4 and dump.gt_assisted == []  # nothing derived from ground truth
    raw = [load_prediction(dump.root / "raw", key) for key in dump.keys]
    filtered = [load_prediction(dump.root / "cd", key) for key in dump.keys]
    assert all(len(after) <= len(before) for before, after in zip(raw, filtered))
    assert any(len(group) > 0 for group in raw)
    for group in filtered:  # the NMS output is sorted by score
        assert torch.all(group.score[:-1] >= group.score[1:])
    # the anchors the run learned came back with the weights
    reloaded = DETECTORS.build(engine_config(tmp_path).resolved()["model"])
    assert torch.equal(reloaded.anchors.gamma, initial)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = [
        value
        for holder in (saved, *[v for v in saved.values() if isinstance(v, dict)])
        for name, value in holder.items()
        if name == "anchors.gamma"
    ]
    assert tensors and torch.equal(tensors[0], model.anchors.gamma.cpu())
