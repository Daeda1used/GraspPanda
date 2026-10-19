"""The seed-based skeleton under every seed-based method, on tiny models and the CPU.

GraspNet-baseline, GSNet, Scale-Balanced-Grasp, EconomicGrasp and FineGrasp are built on
one skeleton (:class:`~grasppanda.models.detectors.seed_based.SeedGraspDetector`): four
detector types, FineGrasp being the EconomicGrasp type with observed point features and a
multi-scale grouper. What the skeleton promises (refusing stages that do not fit, handing
its children what it knows, reporting the losses of its stages, decoding only real seeds,
cropping at the labelled frames, its input-feature policy, scene features for a grouper
that asks for them, refusing malformed input) is checked here under a tiny model of every
method: the stage types of the method's shipped config in the tiny sizes of the swap
matrix (``tests/unit/test_swap_matrix.py``). The terms a model reports are held to the
method's measured reference record. Point painting is checked under the four types.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Iterator
from functools import cache
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from grasppanda.combine.grid import METHODS
from grasppanda.config import Config, ConfigError
from grasppanda.models import Grouper
from grasppanda.models.backbones.image_painting import ImagePainted, ImagePaintedSubsampled
from grasppanda.models.detectors.seed_based import SeedGraspDetector
from grasppanda.models.image_encoders import ImageEncoder
from grasppanda.registry import (
    BACKBONES,
    DETECTORS,
    GROUPERS,
    HEADS,
    INJECTED,
    PROPOSALS,
    BuildError,
)
from grasppanda.structures import (
    ContractError,
    Gripper,
    GroupFeatures,
    Observation,
    PointFeatures,
)

from .conftest import REPO
from .test_swap_matrix import ENCODERS, FRAMES, POINTS, SEEDS, STAGES, VIEW_COUNT, WIDTH, built
from .test_swap_matrix import METHODS as TINY_STAGES
from .test_swap_matrix import batch as random_frames

MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"
#: A tiny scene encoder of each kind that needs no optional library.
TINY_ENCODERS = {"dense": "PointNeXt", "subsampled": "PointNet2"}


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------- components used only by these tests


class _SceneGrouper(Grouper):
    """Groups over the scene: remembers the support it was handed, returns zeros."""

    support = "scene"

    def __init__(self, in_channels: int = INJECTED) -> None:
        super().__init__()
        self.out_channels = in_channels
        self.received: PointFeatures | None = None

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        self.received = support
        return GroupFeatures(centres.new_zeros(len(centres), self.out_channels), self.out_axes)


@GROUPERS.register()
class AllMethodsSceneGrouper(_SceneGrouper):
    out_axes = ("seed", "channel")


@GROUPERS.register()
class AllMethodsSceneDepthGrouper(_SceneGrouper):
    out_axes = ("seed", "depth", "channel")

    def __init__(self, num_depth: int = 2, in_channels: int = INJECTED) -> None:
        super().__init__(in_channels)
        self.num_depth = num_depth

    def forward(self, support: PointFeatures, centres: Tensor, rotation: Tensor) -> GroupFeatures:
        self.received = support
        feats = centres.new_zeros(len(centres), self.num_depth, self.out_channels)
        return GroupFeatures(feats, self.out_axes)


class Identity(ImageEncoder):
    """Returns the image and a half-resolution copy."""

    input_layout = "rgb"
    in_channels = 3
    out_channels = (3, 3)
    out_strides = (1, 2)

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, image):
        return [image * self.scale, nn.functional.avg_pool2d(image, 2) * self.scale]


# ------------------------------------------------------------------------ helpers


def shipped(method: str) -> dict[str, Any]:
    return Config.load(METHODS[method]).resolved()["model"]


def wanted_kind(node: dict[str, Any]) -> str:
    """The kind of scene features the proposal of a model node requires."""
    return PROPOSALS.get(node["proposal"]["type"]).requires


def tiny_encoder(kind: str, **options: Any) -> dict[str, Any]:
    name = TINY_ENCODERS[kind]
    return {"type": name, **ENCODERS[name][0], **options}


def tiny_node(method: str, **replace: Any) -> dict[str, Any]:
    """The tiny model of a method: the stage types and the input-feature policy of its
    shipped config, each stage in its tiny configuration; ``replace`` overrides entries."""
    model = shipped(method)
    detector = model["type"]
    chosen = dict(TINY_STAGES[detector])
    for slot, variant in chosen.items():
        if STAGES[slot][variant]["type"] != model[slot]["type"]:  # a variant of the type
            chosen[slot] = model[slot]["type"]
    node: dict[str, Any] = {
        "type": detector,
        **{slot: copy.deepcopy(STAGES[slot][variant]) for slot, variant in chosen.items()},
        "point_features": model.get("point_features", "ones"),
    }
    channels = (
        model["backbone"].get("in_channels") if node["point_features"] == "observation" else None
    )
    node["backbone"] = tiny_encoder(
        wanted_kind(node), **({"in_channels": channels} if channels else {})
    )
    node.update(replace)
    return node


def frames(node: dict[str, Any], seed: int = 0):
    """Two random frames with everything a stage reads, and the point features a model
    with ``point_features: observation`` reads."""
    observation, supervision = random_frames(seed)
    if node.get("point_features") == "observation":
        generator = torch.Generator().manual_seed(seed)
        channels = node["backbone"]["in_channels"]
        observation.point_feats = torch.rand(len(observation.points), channels, generator=generator)
    return observation, supervision


def other_grouper(node: dict[str, Any]) -> dict[str, Any]:
    """A plain grouper whose axes the head of ``node`` does not accept."""
    expected = HEADS.get(node["head"]["type"]).in_axes
    name = "CylinderGroup" if "depth" in expected else "DepthCylinderGroup"
    assert GROUPERS.get(name).out_axes != expected
    return copy.deepcopy(STAGES["grouper"][name])


@cache
def recorded_terms(method: str) -> frozenset[str]:
    """The loss terms of the method's measured reference run."""
    cells = json.loads(MEASURED.read_text(encoding="utf-8"))["cells"]
    record = next(record for record in cells if record["id"] == f"reference/{method}")
    return frozenset(record["losses"]) - {"loss"}


# ------------------------------------------------------------------ the tiny models


@pytest.mark.parametrize("method", METHODS)
def test_the_tiny_model_has_the_stage_types_of_the_shipped_config(method):
    model, node = shipped(method), tiny_node(method)
    assert node["type"] == model["type"]
    for slot in TINY_STAGES[model["type"]]:
        assert node[slot]["type"] == model[slot]["type"], slot
    assert node["point_features"] == model.get("point_features", "ones")
    assert BACKBONES.get(model["backbone"]["type"]).output == wanted_kind(node)


# ------------------------------------------------------------------------- config


@pytest.mark.parametrize("method", METHODS)
def test_a_stage_that_does_not_fit_is_refused_before_anything_is_built(method):
    node = tiny_node(method)
    kind = wanted_kind(node)
    wrong = tiny_encoder("subsampled" if kind == "dense" else "dense")
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_node(method, backbone=wrong)})
    (issue,) = error.value.issues
    assert issue.path == "model"
    assert f"{node['proposal']['type']} requires '{kind}' scene features" in issue.message
    assert f"but {wrong['type']} returns" in issue.message
    grouper = other_grouper(node)
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"model": tiny_node(method, grouper=grouper)})
    (issue,) = error.value.issues
    emitted = GROUPERS.get(grouper["type"]).out_axes
    expected = HEADS.get(node["head"]["type"]).in_axes
    assert f"{grouper['type']} emits features with axes {emitted}" in issue.message
    assert f"but {node['head']['type']} expects {expected}" in issue.message


@pytest.mark.parametrize("method", METHODS)
def test_a_built_stage_that_does_not_fit_fails_at_build_time(method):
    node = tiny_node(method)
    kind = wanted_kind(node)
    wrong = BACKBONES.build(tiny_encoder("subsampled" if kind == "dense" else "dense"))
    with pytest.raises(ContractError, match=f"requires '{kind}' scene features"):
        DETECTORS.build(tiny_node(method, backbone=wrong))
    grouper = GROUPERS.build(other_grouper(node), in_channels=WIDTH)
    with pytest.raises(ContractError, match=rf"{node['head']['type']} expects \("):
        DETECTORS.build(tiny_node(method, grouper=grouper))


@pytest.mark.parametrize("method", METHODS)
def test_children_receive_what_the_parent_knows_and_conflicts_are_errors(method):
    node = tiny_node(method)
    model = built(node)
    assert isinstance(model, SeedGraspDetector)
    observed = {"point_feats"} if node["point_features"] == "observation" else set()
    assert model.required_inputs == frozenset({"points", *observed})
    for stage in (model.proposal, model.view, model.grouper):
        if hasattr(stage, "in_channels"):  # a stage that reads the encoder's features
            assert stage.in_channels == model.backbone.out_channels == WIDTH, type(stage)
    assert model.head.in_channels == model.grouper.out_channels
    assert model.assigner.template is model.view.template  # one template, shared
    assert model.head.gripper is model.gripper and model.gripper == Gripper()
    provided = model.grouper.out_channels
    head = {**node["head"], "in_channels": provided + 20}
    with pytest.raises(
        BuildError, match=f"in the config, but the parent component provides {provided}"
    ):
        DETECTORS.build(tiny_node(method, head=head))
    custom = DETECTORS.build(tiny_node(method, gripper={"max_width": 0.08}))
    assert custom.gripper.max_width == 0.08 and custom.head.gripper is custom.gripper


# --------------------------------------------------------------------- loss, predict


@pytest.mark.parametrize("method", METHODS)
def test_the_loss_reports_the_terms_of_the_method_and_reaches_every_parameter(method):
    node = tiny_node(method)
    model = built(node).train()
    observation, supervision = frames(node)
    losses = model.loss(observation, supervision)
    terms = {key for key in losses if key.startswith("loss/")}
    assert terms == recorded_terms(method), "the terms of the measured reference run"
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert 0.0 < losses["metric/supervised_seeds"].item() <= 1.0
    sum(losses[key] for key in terms).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # forward(observation, supervision) is loss(); forward(observation) is predict()
    assert set(model(observation, supervision)) == set(losses)


@pytest.mark.parametrize("method", METHODS)
def test_predict_returns_one_valid_group_per_frame_and_never_sees_labels(method):
    node = tiny_node(method)
    model = built(node).eval()
    observation, _ = frames(node)
    grasps = model.predict(observation)
    assert [group.key for group in grasps] == observation.keys
    cloud = observation.points.dense()
    for frame, group in enumerate(grasps):
        assert len(group) <= SEEDS and group.frame == "camera" and group.gripper == model.gripper
        assert group.validate().to_graspnet_array().shape == (len(group), 17)
        assert (group.width >= 0).all() and (group.width <= model.gripper.max_width).all()
        if len(group):  # every grasp sits on a point of its own frame
            offset = (group.translation[:, None, :] - cloud[frame][None, :, :]).abs().amax(-1)
            assert offset.min(dim=1).values.max() == 0
    again = model.predict(observation)  # eval mode: no sampling
    assert all(torch.equal(a.score, b.score) for a, b in zip(grasps, again))
    assert not any(group.score.requires_grad for group in grasps)
    assert isinstance(model(observation), list)


@pytest.mark.parametrize("method", METHODS)
def test_padded_and_invalid_seeds_are_never_emitted(method):
    """Tensors stay dense, padding is marked invalid, and decoding drops it: five real
    seeds in the first frame and none in the second give five grasps and an empty group."""
    node = tiny_node(method)
    model = built(node).eval()
    observation, _ = frames(node)
    chosen = torch.zeros(FRAMES * SEEDS, dtype=torch.bool)
    chosen[[0, 3, 5, 7, 11]] = True
    if hasattr(model.proposal, "candidates"):  # the proposal pads a frame without candidates
        candidates = torch.zeros(FRAMES * POINTS, dtype=torch.bool)
        candidates[[3, 11, 27, 40, 59]] = True  # all in the first frame
        model.proposal.candidates = lambda scores: candidates
        out = model.run_stages(observation)
        assert len(out.seeds) == FRAMES * SEEDS and int(out.seeds.valid.sum()) == 5
        real = observation.points.coords[candidates]
    else:  # the seeds are the encoder's subsample: mark them as a proposal marks padding
        out = model.run_stages(observation)
        assert len(out.seeds) == FRAMES * SEEDS
        out.seeds.valid = chosen
        real = out.seeds.points.coords[chosen]
    first, second = model.decode(out, observation)
    assert len(first) == 5 and len(second) == 0 and second.key == observation.keys[1]
    assert sorted(first.translation.tolist()) == sorted(real.tolist())
    assert second.to_graspnet_array().shape == (0, 17)  # an empty prediction is a valid result


def crop(model: SeedGraspDetector, out, observation: Observation, centres, rotation) -> Tensor:
    """What the grouper gathers at ``centres`` and ``rotation``, from the support its
    contract asks for (the skeleton's docstring)."""
    if model.grouper.support == "seeds":
        support = PointFeatures(out.seeds.feats, out.seeds.points)
    elif out.feats.points.id == observation.points.id:
        support = out.feats
    else:
        scene = observation.points
        support = PointFeatures(scene.coords.new_zeros(len(scene), 0), scene)
    extra = (out.seeds.feats,) if getattr(model.grouper, "uses_seed_features", False) else ()
    return model.grouper(support, centres, rotation, *extra).feats


@pytest.mark.parametrize("method", METHODS)
def test_training_crops_use_label_frames_and_inference_the_template(method):
    node = tiny_node(method)
    model = built(node).train()
    model.grouper.eval()  # no dropout inside the crop, so that it can be recomputed
    observation, supervision = frames(node)
    views = torch.arange(FRAMES * SEEDS) % VIEW_COUNT
    replay: dict[str, Tensor] = {"view_index": views}
    seed_replay = "seed_index" in inspect.signature(model.proposal.forward).parameters
    if seed_replay:  # a proposal that draws its seeds can replay a recorded draw
        replay["seed_index"] = torch.stack([torch.arange(SEEDS), POINTS + torch.arange(SEEDS)])
    with_labels = model.run_stages(observation, supervision, **replay)
    without = model.run_stages(observation, **replay)
    if seed_replay:
        assert torch.equal(with_labels.seeds.points.parent_index, replay["seed_index"].reshape(-1))
        assert with_labels.seeds.source == "replayed"
    assert torch.equal(with_labels.approach.view_index, views) and without.matched is None
    matched = with_labels.matched
    assert not torch.allclose(matched.crop_rotation, with_labels.approach.rotation)  # object pose
    expected = crop(model, with_labels, observation, matched.crop_centre, matched.crop_rotation)
    assert torch.allclose(with_labels.groups.feats, expected, atol=1e-6)
    template = crop(
        model, without, observation, without.seeds.points.coords, without.approach.rotation
    )
    assert torch.allclose(without.groups.feats, template, atol=1e-6)
    assert not torch.allclose(with_labels.groups.feats, without.groups.feats)


@pytest.mark.parametrize("method", METHODS)
def test_the_input_feature_policy_is_explicit(method):
    node = tiny_node(method)
    channels = node["backbone"].get("in_channels") or 3
    ones = built(tiny_node(method, point_features="ones")).eval()
    observation, _ = frames(tiny_node(method, point_features="ones"))
    seen = []
    ones.backbone.register_forward_pre_hook(lambda _module, args: seen.append(args[1]))
    ones.predict(observation)
    width = ones.backbone.in_channels
    assert torch.equal(seen[0], torch.ones(FRAMES * POINTS, width))  # geometry only
    backbone = {**node["backbone"], "in_channels": channels}
    given = built(tiny_node(method, point_features="observation", backbone=backbone)).eval()
    assert given.required_inputs == frozenset({"points", "point_feats"})
    with pytest.raises(ContractError, match="observation.point_feats is missing"):
        given.predict(observation)
    observation.point_feats = torch.rand(FRAMES * POINTS, channels)
    assert len(given.predict(observation)) == FRAMES
    with pytest.raises(ValueError, match="unknown point_features"):
        DETECTORS.build(tiny_node(method, point_features="rgb"))


@pytest.mark.parametrize("method", METHODS)
def test_a_scene_support_grouper_is_handed_the_scene_features(method):
    """Dense scene features where the encoder keeps every point; the scene points with
    no feature channel where it returns a subsample."""
    node = tiny_node(method)
    depth = "depth" in HEADS.get(node["head"]["type"]).in_axes
    grouper = (
        {"type": "AllMethodsSceneDepthGrouper", "num_depth": node["head"]["num_depth"]}
        if depth
        else {"type": "AllMethodsSceneGrouper"}
    )
    model = built(tiny_node(method, grouper=grouper)).eval()
    observation, _ = frames(node)
    out = model.run_stages(observation)
    received = model.grouper.received
    assert received.points is observation.points
    if wanted_kind(node) == "dense":
        assert received.feats.shape == (FRAMES * POINTS, WIDTH)
        assert torch.equal(received.feats, out.feats.feats)
    else:
        assert received.feats.shape == (FRAMES * POINTS, 0)


@pytest.mark.parametrize("method", METHODS)
def test_malformed_observations_and_duplicate_loss_keys_are_refused(method):
    node = tiny_node(method)
    model = built(node)
    observation, supervision = frames(node)
    with pytest.raises(ContractError, match="needs observation.points"):
        model.eval().predict(Observation(keys=observation.keys))
    short = Observation(
        keys=observation.keys[:1], points=observation.points, point_feats=observation.point_feats
    )
    with pytest.raises(ContractError, match=r"1 sample key\(s\) for a batch of 2"):
        model.predict(short)
    out = model.train().run_stages(observation, supervision)
    key = sorted(model.view.loss(out.approach, out.matched))[0]
    clashing = copy.deepcopy(model)
    clashing.head.loss = lambda raw, matched: {key: torch.zeros(())}
    with pytest.raises(ContractError, match=rf"reports \['{key}'\], which another stage"):
        clashing.loss(observation, supervision)


# ------------------------------------------------------------------- point painting


def detector_types() -> list[str]:
    return list(dict.fromkeys(shipped(method)["type"] for method in METHODS))


@pytest.mark.parametrize("detector", detector_types())
def test_the_skeleton_hands_the_observation_to_a_painted_encoder_of_every_type(detector):
    """Point painting wraps the scene encoder of any seed-based type: the wrapper of the
    kind the type's proposal requires builds there, asks for the colour image and the
    intrinsics, and a frame without the image is refused by the wrapper itself."""
    method = next(method for method in METHODS if shipped(method)["type"] == detector)
    node = tiny_node(method, point_features="ones")
    kind = wanted_kind(node)
    painted = next(cls for cls in (ImagePainted, ImagePaintedSubsampled) if cls.output == kind)
    paint = 4
    wrapper = {
        "type": painted.__name__,
        "backbone": tiny_encoder(kind, in_channels=3 + paint),
        "image_encoder": Identity(),
        "paint_channels": paint,
    }
    model = DETECTORS.build(tiny_node(method, point_features="ones", backbone=wrapper))
    assert {"points", "rgb", "intrinsics"} <= model.required_inputs
    assert model.backbone.backbone.in_channels == 3 + paint and model.backbone.in_channels == 3
    observation, _ = frames(node)
    with pytest.raises(ContractError, match=f"{painted.__name__} reads the colour image"):
        model.eval().predict(observation)
