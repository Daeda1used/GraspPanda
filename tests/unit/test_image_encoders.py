"""The image-encoder slot: its contract, HGGD as its first consumer, the library bridges.

Everything here runs on CPU with random weights and downloads nothing. The bridge tests
skip when their library is absent (the minimal install); what a pretrained model does on
real frames is recorded by ``tools/probe_grid.py`` (``family-slot`` cells).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from grasppanda.combine.compose import ComposeError, component_slots, compose
from grasppanda.combine.grid import FAMILIES, cells, family_slots
from grasppanda.combine.presets import presets
from grasppanda.models.detectors.hggd import HGGD, ImageNet
from grasppanda.models.image_encoders import HGGDResidualEncoder, ImageEncoder
from grasppanda.registry import DETECTORS, IMAGE_ENCODERS, component_parameters
from grasppanda.structures import ContractError
from grasppanda.utils.imports import MissingDependencyError

from .test_hggd_detector import TINY, batch

HGGD_CONFIG = FAMILIES["HGGD"]


class Patches(ImageEncoder):
    """A stand-in for a ViT: one map at stride 14, sizes rounded down."""

    def __init__(self, layout: str = "rgb", channels: int = 12) -> None:
        super().__init__()
        self.input_layout = layout
        self.in_channels = 3 if layout == "rgb" else 4
        self.out_channels, self.out_strides = (channels,), (14,)
        self.embed = nn.Conv2d(self.in_channels, channels, 14, 14)
        self.seen: tuple[int, ...] = ()

    def forward(self, image):
        self.seen = tuple(image.shape)
        return [self.embed(image)]


class Pyramid(ImageEncoder):
    """A stand-in for a convolutional pyramid: strides 4, 8 and 16."""

    input_layout = "rgb"
    in_channels = 3
    out_channels = (8, 16, 24)
    out_strides = (4, 8, 16)

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [nn.Conv2d(3, 8, 3, 4, 1), nn.Conv2d(8, 16, 3, 2, 1), nn.Conv2d(16, 24, 3, 2, 1)]
        )

    def forward(self, image):
        maps = []
        for stage in self.stages:
            image = stage(image)
            maps.append(image)
        return maps


# ------------------------------------------------------------------- the slot itself


def test_the_registry_holds_the_reference_encoder_and_the_three_bridges():
    assert IMAGE_ENCODERS.base_class is ImageEncoder
    assert set(IMAGE_ENCODERS.names()) >= {
        "HGGDResidualEncoder",
        "TimmImageEncoder",
        "HFImageEncoder",
        "TorchvisionImageEncoder",
    }
    for name in ("TimmImageEncoder", "HFImageEncoder", "TorchvisionImageEncoder"):
        cls = IMAGE_ENCODERS.get(name)
        assert cls.status == "bridge" and len(cls.extras) == 1
        params, var_kwargs = component_parameters(cls)
        assert not var_kwargs and {"depth", "frozen", "model_args", "input_size"} <= set(params)


def test_the_reference_encoder_states_its_contract_and_trains():
    encoder = HGGDResidualEncoder()
    assert (encoder.in_channels, encoder.input_layout) == (4, "drgb")
    assert encoder.out_strides == (2, 4, 8, 16, 32)
    assert encoder.out_channels == (8, 16, 32, 64, 128)
    maps = encoder(torch.randn(2, 4, 64, 32))
    assert [tuple(m.shape) for m in maps] == [
        (2, width, 64 // stride, 32 // stride)
        for width, stride in zip(encoder.out_channels, encoder.out_strides)
    ]
    maps[-1].sum().backward()
    assert encoder.stem[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="four stages"):
        HGGDResidualEncoder(blocks=(2, 2))


# -------------------------------------------------------------- HGGD's image network


def test_the_reference_encoder_in_the_slot_is_the_reference_model():
    """Same parameter names, same initial values: checkpoints and seeds keep their meaning."""
    torch.manual_seed(0)
    default = DETECTORS.build(dict(TINY))
    torch.manual_seed(0)
    slotted = DETECTORS.build({**TINY, "image_encoder": {"type": "HGGDResidualEncoder"}})
    ours, theirs = default.state_dict(), slotted.state_dict()
    assert list(ours) == list(theirs)
    assert all(torch.equal(ours[name], theirs[name]) for name in ours)
    assert "image_net.stem.0.weight" in ours and default.image_net.reference
    wide = DETECTORS.build(
        {**TINY, "image_encoder": {"type": "HGGDResidualEncoder", "base_width": 16}}
    )
    assert wide.image_net.feature_channels == 64


@pytest.mark.parametrize("layout", ["rgb", "rgbd"])
def test_a_foreign_encoder_sees_an_upright_image_in_its_own_layout(layout):
    encoder = Patches(layout)
    network = ImageNet(encoder=encoder, decoder_channels=16, num_angles=6, stride=8)
    assert not network.reference and network.feature_channels == 16
    image = torch.rand(2, 4, 64, 40)  # HGGD: (B, [depth, R, G, B], U, V), u along the width
    out = network(image)
    assert encoder.seen == (2, 3 if layout == "rgb" else 4, 40, 64)  # rows are image rows
    assert out.heatmap.shape == (2, 64, 40)
    assert out.angle.shape == (2, 6, 8, 5) and out.features.shape == (2, 16, 8, 5)
    assert not out.features.requires_grad  # detach_features, as with the reference encoder
    out.heatmap.sum().backward()
    assert encoder.embed.weight.grad.abs().sum() > 0
    # Colour first, depth last: the encoder's rgbd is HGGD's drgb re-ordered.
    probe = torch.zeros(1, 4, 64, 40)
    probe[:, 0] = 1.0  # depth only
    seen = []
    encoder.embed.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    network(probe)
    assert seen[0][:, :3].abs().sum() == 0
    assert (seen[0][:, 3:].sum() > 0) == (layout == "rgbd")


def test_a_pyramid_of_any_strides_gives_the_maps_hggd_needs():
    network = ImageNet(encoder=Pyramid(), stride=4, decoder_channels=16)
    out = network(torch.rand(1, 4, 64, 40))
    assert out.heatmap.shape == (1, 64, 40) and out.features.shape == (1, 16, 16, 10)
    with pytest.raises(ContractError, match="not a multiple of the proposal stride"):
        network(torch.rand(1, 4, 66, 40))
    drgb = Patches()
    drgb.input_layout = "drgb"
    with pytest.raises(ContractError, match="HGGD can serve 'rgb' and 'rgbd'"):
        ImageNet(encoder=drgb)


def test_hggd_trains_and_predicts_with_a_foreign_encoder():
    torch.manual_seed(0)
    model = DETECTORS.build(
        {**TINY, "image_encoder": Patches("rgbd"), "encoder": {"decoder_channels": 16}}
    )
    assert isinstance(model, HGGD) and model.local_net is not None
    data = batch(2)
    losses = model.train().loss(data.observation, data.supervision)
    total = sum(value for name, value in losses.items() if name.startswith("loss/"))
    assert torch.isfinite(total)
    total.backward()
    assert model.image_net.encoder.embed.weight.grad.abs().sum() > 0
    groups = model.eval().predict(data.observation)
    assert len(groups) == 2


def test_options_of_the_reference_encoder_are_refused_next_to_another_encoder():
    with pytest.raises(ContractError, match="encoder.base_width and encoder.blocks describe"):
        DETECTORS.build({**TINY, "image_encoder": Patches(), "encoder": {"base_width": 16}})


# ------------------------------------------------------------------- compose and grid


def test_compose_puts_an_image_encoder_into_hggd():
    assert component_slots(DETECTORS.get("HGGD")) == {"image_encoder": IMAGE_ENCODERS}
    assert family_slots("HGGD") == {"image_encoder": IMAGE_ENCODERS}
    result = compose(HGGD_CONFIG, ["image_encoder=resnet34"])
    assert result.accepted and not result.seed_based
    assert result.body["model"]["image_encoder"]["type"] == "TimmImageEncoder"
    assert "image_encoder: preset `resnet34`" in result.text()
    bare = compose(HGGD_CONFIG, ["image_encoder=HGGDResidualEncoder"])
    assert bare.accepted
    with pytest.raises(ComposeError, match=r"no preset 'resnet43'.*Did you mean 'resnet34'"):
        compose(HGGD_CONFIG, ["image_encoder=resnet43"])
    wrong = compose(HGGD_CONFIG, ["image_encoder=resnet34"], ["model.image_encoder.depth=rgbd"])
    assert not wrong.accepted and "depth" in wrong.issues[0].path


def test_every_image_encoder_preset_is_a_cell_and_passes_the_config_check():
    members = presets()["image_encoder"]
    assert len(members) >= 15
    ids = {cell.id for cell in cells(("family-slot",))}
    assert ids == {f"family-slot/HGGD/image_encoder/{name}" for name in members}
    for name, preset in members.items():
        params, _ = component_parameters(IMAGE_ENCODERS.get(preset.type))
        assert set(preset.node) - {"type"} <= set(params), name
        result = compose(HGGD_CONFIG, [f"image_encoder={name}"])
        remote = (preset.node.get("model_args") or {}).get("trust_remote_code")
        if remote:  # code of the hub repository: refused until the config accepts it
            (message,) = result.refusal()
            assert "accept_remote_code: true" in message and preset.node["name"] in message
            accept = ["model.image_encoder.accept_remote_code=true"]
            result = compose(HGGD_CONFIG, [f"image_encoder={name}"], accept)
        assert result.accepted, (name, result.issues)
        # A preset never fetches weights under non-commercial terms unasked.
        if "NON-COMMERCIAL" in preset.path.read_text(encoding="utf-8"):
            assert preset.node.get("pretrained") is False, name


def test_a_bridge_says_once_which_weights_its_library_fetches(capsys):
    from grasppanda.models.image_encoders import bridges

    bridges._ANNOUNCED.clear()
    bridges._announce_weights("TimmImageEncoder", "timm", "x.tag", "cc-by-nc-4.0")
    bridges._announce_weights("TimmImageEncoder", "timm", "x.tag", "cc-by-nc-4.0")
    bridges._announce_weights("HFImageEncoder", "transformers", "org/model", None)
    told = capsys.readouterr().err.splitlines()
    assert len(told) == 2
    assert "'x.tag' come through timm" in told[0] and "(NON-COMMERCIAL)" in told[0]
    assert "not stated by the library" in told[1] and "NON-COMMERCIAL" not in told[1]


# -------------------------------------------------------------------------- bridges


def test_a_bridge_without_its_library_says_what_to_install():
    for name, module, node in (
        ("TimmImageEncoder", "timm", {"name": "resnet18"}),
        ("HFImageEncoder", "transformers", {"name": "facebook/dinov2-small"}),
        ("TorchvisionImageEncoder", "torchvision", {"name": "resnet18"}),
    ):
        if importlib.util.find_spec(module) is None:
            with pytest.raises(MissingDependencyError, match=rf'pip install -e "\.\[{module}\]"'):
                IMAGE_ENCODERS.build({"type": name, **node})


def _check_depth_modes(build, strides):
    """What every bridge promises about colour, depth and freezing."""
    torch.manual_seed(0)
    colour = build(depth="rgb_only").eval()
    assert (colour.in_channels, colour.input_layout) == (3, "rgb")
    assert colour.out_strides == strides and len(colour.out_channels) == len(strides)
    image = torch.rand(2, 4, 64, 96)
    maps = colour(image[:, :3])
    assert [tuple(m.shape[:2]) for m in maps] == [(2, width) for width in colour.out_channels]
    assert [round(64 / m.shape[2]) for m in maps] == list(strides)
    with pytest.raises(ContractError, match=r"takes a \(B, 3, H, W\) image \(rgb\)"):
        colour(image)
    for mode in ("fourth_channel", "depth_stem"):
        encoder = build(depth=mode, frozen=True)
        assert (encoder.in_channels, encoder.input_layout) == (4, "rgbd")
        own = encoder.state_dict()
        same = {k: v for k, v in colour.state_dict().items() if v.shape == own[k].shape}
        encoder.load_state_dict(same, strict=False)
        with torch.no_grad():
            if mode == "fourth_channel":  # the colour filters of the widened convolution
                convs = [m for m in encoder.modules() if isinstance(m, nn.Conv2d)]
                wide = next(m for m in convs if m.in_channels == 4)
                convs = [m for m in colour.modules() if isinstance(m, nn.Conv2d)]
                narrow = next(m for m in convs if m.in_channels == 3)
                wide.weight[:, :3] = narrow.weight
                assert wide.weight[:, 3:].abs().sum() == 0
        encoder.eval()
        # Depth starts without effect: the network is the colour model until it is trained.
        for ours, theirs in zip(encoder(image), maps):
            assert torch.allclose(ours, theirs, atol=1e-5)
        encoder.train()
        assert not encoder.model.training  # frozen statistics
        sum(m.sum() for m in encoder(image)).backward()
        trained = [name for name, p in encoder.named_parameters() if p.grad is not None]
        assert trained and all(p.grad is None for p in encoder.parameters() if not p.requires_grad)
        assert any(p.grad.abs().sum() > 0 for p in encoder.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="depth must be rgb_only, fourth_channel or depth_stem"):
        build(depth="hha")


def test_timm_bridge():
    pytest.importorskip("timm")
    from grasppanda.models.image_encoders import TimmImageEncoder

    def build(**options):
        return TimmImageEncoder("resnet10t", **options)

    _check_depth_modes(build, (2, 4, 8, 16, 32))
    coarse = TimmImageEncoder("resnet10t", out_indices=[2, 4])
    assert coarse.out_strides == (8, 32) and coarse.out_channels == (128, 512)
    with pytest.raises(ContractError, match="timm cannot build 'no_such_model'"):
        TimmImageEncoder("no_such_model")
    # A ViT: one size only, so the image is resampled to it; maps at the patch stride.
    vit = TimmImageEncoder(
        "vit_tiny_patch16_224",
        model_args={"img_size": [64, 96], "depth": 2},
        input_size=[64, 96],
        out_indices=[1],
    ).eval()
    assert vit.out_strides == (16,)
    assert vit(torch.rand(1, 3, 60, 100))[0].shape == (1, 192, 4, 6)
    network = ImageNet(encoder=build(depth="fourth_channel"), decoder_channels=16)
    assert network(torch.rand(1, 4, 64, 40)).heatmap.shape == (1, 64, 40)


def test_torchvision_bridge():
    pytest.importorskip("torchvision")
    from grasppanda.models.image_encoders import TorchvisionImageEncoder

    def build(**options):
        return TorchvisionImageEncoder("resnet18", **options)

    _check_depth_modes(build, (4, 8, 16, 32))
    mobile = TorchvisionImageEncoder("mobilenet_v3_small", min_stride=8)
    assert mobile.out_strides == (8, 16, 32)
    assert [m.shape[1] for m in mobile(torch.rand(1, 3, 64, 64))] == list(mobile.out_channels)
    with pytest.raises(ContractError, match="cannot cut 'vit_b_16' into stages"):
        TorchvisionImageEncoder("vit_b_16")


def test_hugging_face_bridge():
    pytest.importorskip("transformers")
    from grasppanda.models.image_encoders import HFImageEncoder

    tiny = {
        "model_type": "dinov2",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "image_size": 28,
        "patch_size": 14,
    }

    def build(**options):
        return HFImageEncoder(config=tiny, size_multiple=14, probe_size=(56, 56), **options)

    torch.manual_seed(0)
    colour = build().eval()
    assert colour.out_strides == (14,) and colour.out_channels == (32,)
    # 64 x 96 is resampled up to 70 x 98, five by seven patches.
    assert colour(torch.rand(1, 3, 64, 96))[0].shape == (1, 32, 5, 7)
    for mode in ("fourth_channel", "depth_stem"):
        encoder = build(depth=mode, frozen=True)
        assert encoder.in_channels == 4
        sum(m.sum() for m in encoder(torch.rand(1, 4, 56, 56))).backward()
        assert any(p.grad is not None for p in encoder.parameters())
    # Hidden states of a plain model, prefix tokens dropped.
    states = HFImageEncoder(
        config={**tiny, "model_type": "vit", "patch_size": 14, "image_size": 56},
        api="model",
        out_indices=[1, 2],
        probe_size=(56, 56),
    )
    assert states.out_strides == (14, 14) and states.out_channels == (32, 32)
    with pytest.raises(ValueError, match="exactly one of `name` and `config`"):
        HFImageEncoder()
