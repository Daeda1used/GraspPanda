"""PointNeXt: published sizes, state-dict layout and the ratio-based sampling.

The shared contract is in ``test_backbone_contract``; agreement with the official
implementation (which needs no weights file) is in
``tests/integration/test_backbone_oracle.py``."""

from __future__ import annotations

import pytest
import torch

from grasppanda.models.backbones.pointnext import PointNeXt
from grasppanda.structures import ContractError, PointSet


def seg_head(width: int, classes: int = 13) -> int:
    """Parameters of the official segmentation head, which the published counts include:
    a point-wise convolution with batch norm, then the 13-way S3DIS classifier."""
    return width * width + 2 * width + width * classes + classes


@pytest.mark.parametrize(
    ("name", "config", "published_millions"),
    [
        # "Params.(M)" in the header of cfgs/s3dis/pointnext-{s,b,l,xl}.yaml of the
        # official repository.
        ("S", {}, 0.792),
        ("B", {"blocks": [1, 2, 3, 2, 2], "sa_layers": 1, "sa_use_res": False}, 3.828),
        ("L", {"blocks": [1, 3, 5, 3, 3], "sa_layers": 1, "sa_use_res": False}, 7.125),
        (
            "XL",
            {"blocks": [1, 4, 7, 4, 4], "width": 64, "sa_layers": 1, "sa_use_res": False},
            41.576,
        ),
    ],
)
def test_published_configurations_have_the_published_size(name, config, published_millions):
    model = PointNeXt(**config)
    backbone = sum(p.numel() for p in model.parameters())
    assert round((backbone + seg_head(model.out_channels)) / 1e6, 3) == published_millions, name
    assert model.output == "dense" and model.in_channels == 4


def test_defaults_are_pointnext_s_with_the_official_state_dict_layout():
    model = PointNeXt()
    assert sum(p.numel() for p in model.parameters()) == 790_720  # official BaseSeg without head
    assert model.out_channels == 32
    state = model.state_dict()
    assert state["encoder.encoder.0.0.convs.0.0.weight"].shape == (32, 4, 1)  # point-wise stem
    assert state["encoder.encoder.0.0.convs.0.0.bias"].shape == (32,)
    assert state["encoder.encoder.1.0.convs.0.0.weight"].shape == (32, 35, 1, 1)  # 32 + offsets
    assert state["encoder.encoder.1.0.convs.1.0.weight"].shape == (64, 32, 1, 1)
    assert state["encoder.encoder.1.0.skipconv.0.weight"].shape == (64, 32, 1)
    assert state["encoder.encoder.4.0.convs.1.1.running_var"].shape == (512,)
    assert state["decoder.decoder.3.0.convs.0.0.weight"].shape == (256, 256 + 512, 1)
    assert state["decoder.decoder.0.0.convs.1.0.weight"].shape == (32, 32, 1)
    stages = model.encoder.encoder
    assert [stages[index][0].radius for index in range(1, 5)] == [0.1, 0.2, 0.4, 0.8]

    larger = PointNeXt(blocks=[1, 2, 3, 2, 2], sa_layers=1, sa_use_res=False)
    state = larger.state_dict()
    assert state["encoder.encoder.2.1.convs.convs.0.0.weight"].shape == (128, 131, 1, 1)
    assert state["encoder.encoder.2.2.pwconv.0.0.weight"].shape == (512, 128, 1)  # expansion 4
    assert state["encoder.encoder.2.2.pwconv.1.1.weight"].shape == (128,)
    assert not any("skipconv" in key for key in state)
    # Inverted residual blocks look twice as far as the abstraction of their stage.
    assert larger.encoder.encoder[1][0].radius == 0.1 and larger.encoder.encoder[1][1].radius == 0.2


def test_every_stage_keeps_a_quarter_of_each_cloud():
    model = PointNeXt(
        in_channels=2, width=8, strides=[1, 4, 4], blocks=[1, 1, 1], radius=0.3, nsample=8
    )
    model.eval()
    sizes = (70, 101)
    points = PointSet.cat([PointSet(torch.rand(size, 3)) for size in sizes])
    seen = []
    hooks = [
        stage[0].register_forward_hook(lambda module, args, output: seen.append(output[2]))
        for stage in model.encoder.encoder[1:]
    ]
    with torch.no_grad():
        out = model(points, torch.rand(len(points), 2))
    for hook in hooks:
        hook.remove()
    assert seen == [[17, 25], [4, 6]]  # floor(n / 4), cloud by cloud
    assert out.feats.shape == (171, 8)
    with pytest.raises(ContractError, match="every cloud needs at least 48 points"):
        model(PointSet(torch.rand(47, 3)), torch.rand(47, 2))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"in_channels": 0}, "at least one input feature channel"),
        ({"strides": [1, 4, 4, 4]}, "one entry per stage"),
        ({"strides": [4, 4, 4, 4, 4]}, "the first stride must be 1"),
        ({"strides": [1, 4, 1, 4, 4]}, "the others at least 2"),
        ({"blocks": [1, 0, 1, 1, 1]}, "must be positive"),
    ],
)
def test_inconsistent_configurations_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        PointNeXt(**change)
