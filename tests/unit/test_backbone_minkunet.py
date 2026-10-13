"""MinkUNet on tiny inputs: the published sizes, the contract, the MinkowskiEngine layout."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("spconv", reason="needs the spconv extra")

from grasppanda.models.backbones.minkunet import MinkUNet, convert_minkowski_state
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet

#: Values in the 374 tensors of the released Res16UNet34C of PointContrast (3 input
#: channels, 3³ stem, 32 outputs) without the batch-norm counters: its parameters.
RES16UNET34C_PARAMETERS = 37_847_808


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny(**overrides) -> MinkUNet:
    config = {
        "type": "MinkUNet",
        "in_channels": 3,
        "out_channels": 6,
        "voxel_size": 0.05,
        "layers": [1, 2, 1, 1, 1, 1, 2, 1],
        "planes": [4, 6, 8, 10, 8, 6, 5, 4],
        "init_dim": 4,
    }
    return BACKBONES.build({**config, **overrides})


def cloud(points: int = 400, clouds: int = 2, seed: int = 0) -> PointSet:
    generator = torch.Generator().manual_seed(seed)
    return PointSet.from_dense(torch.rand(clouds, points, 3, generator=generator))


def parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_the_defaults_are_res16unet34c_as_released():
    model = BACKBONES.build({"type": "MinkUNet", "out_channels": 32})
    assert parameters(model) == RES16UNET34C_PARAMETERS
    assert len(model.state_dict()) == 374
    assert [len(getattr(model, f"block{i}")) for i in range(1, 9)] == [2, 3, 4, 6, 2, 2, 2, 2]
    assert model.conv0p1s1.weight.shape == (32, 3, 3, 3, 3)
    # Without the final layer the features of the last decoder stage are returned.
    assert BACKBONES.build({"type": "MinkUNet"}).out_channels == 96
    assert MinkUNet.output == "dense" and MinkUNet.status == "tested"


@pytest.mark.parametrize("block", ["basic", "bottleneck"])
def test_one_feature_per_input_point_and_gradients_everywhere(block):
    model = tiny(block=block).train()
    points = cloud()
    feats = torch.ones(len(points), 3)
    out = model(points, feats)
    assert out.points is points and out.feats.shape == (len(points), 6)
    assert torch.isfinite(out.feats).all()
    out.feats.sum().backward()
    silent = [name for name, p in model.named_parameters() if p.grad is None]
    assert not silent, silent


def test_a_five_cell_stem_runs_beside_the_three_cell_blocks_of_its_level():
    model = tiny(stem_kernel=5, final_bias=False).eval()
    assert model.conv0p1s1.weight.shape[1:4] == (5, 5, 5) and model.final.bias is None
    points = cloud(200, 1)
    assert model(points, torch.ones(len(points), 3)).feats.shape == (200, 6)


def test_without_a_final_layer_the_width_follows_the_block():
    assert tiny(out_channels=None).out_channels == 4
    wide = tiny(out_channels=None, block="bottleneck")
    assert wide.out_channels == 16
    points = cloud(200, 1)
    assert wide.eval()(points, torch.ones(len(points), 3)).feats.shape == (200, 16)


def test_points_of_one_voxel_share_a_feature_and_clouds_do_not_mix():
    model = tiny().eval()
    coords = torch.rand(1, 300, 3, generator=torch.Generator().manual_seed(1))
    coords[0, 1] = coords[0, 0] + 1e-4  # the same 5 cm voxel
    points = PointSet.from_dense(coords)
    alone = model(points, torch.ones(300, 3)).feats
    assert torch.equal(alone[0], alone[1])
    both = PointSet.from_dense(torch.cat([coords, coords.flip(1) * 0.5]))
    assert torch.allclose(model(both, torch.ones(600, 3)).feats[:300], alone, atol=1e-5)


def test_a_minkowski_state_dict_is_relaid_key_for_key():
    model = tiny()
    upstream = {}
    for key, value in model.state_dict().items():
        module = key.rsplit(".", 1)[0]
        if key.endswith("weight") and value.ndim == 5:  # (out, i, j, k, in) → (volume, in, out)
            edge = value.shape[1]
            upstream[f"{module}.kernel"] = value.permute(3, 2, 1, 4, 0).reshape(
                edge**3, value.shape[4], value.shape[0]
            )
        elif key.endswith("weight") and value.ndim == 2:  # a 1×1 kernel is (in, out)
            upstream[f"{module}.kernel"] = value.t()
        elif key == "final.bias":
            upstream[key] = value[None]
        else:  # batch norm, wrapped in a module called `bn`
            upstream[f"{module}.bn.{key.rsplit('.', 1)[1]}"] = value
    converted = convert_minkowski_state(upstream)
    assert set(converted) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(converted[key], value), key
    with pytest.raises(ValueError, match="unsupported kernel volume"):
        convert_minkowski_state({"conv.kernel": torch.zeros(9, 2, 2)})


def test_misuse_is_refused_with_the_reason():
    model = tiny()
    points = cloud(50, 1)
    with pytest.raises(ContractError, match=r"expects input features of shape \(50, 3\)"):
        model(points, torch.ones(50, 6))
    with pytest.raises(ValueError, match="eight entries each"):
        tiny(layers=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="`basic` or `bottleneck`"):
        tiny(block="preact")
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        tiny(voxel_size=0.0)
