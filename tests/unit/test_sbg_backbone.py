"""PointNet2Transformer: the attention layer against the formula written out in loops,
the neighbours against brute force, the contract of the backbone and its misuse."""

from __future__ import annotations

import math

import pytest
import torch

from grasppanda.models.backbones.pointnet2_transformer import PointNet2Transformer, _VectorAttention
from grasppanda.registry import BACKBONES
from grasppanda.structures import ContractError, PointSet


def small(**overrides) -> PointNet2Transformer:
    torch.manual_seed(0)
    config = {
        "type": "PointNet2Transformer",
        "npoints": [24, 10],
        "radii": [0.4, 0.8],
        "nsamples": [6, 5],
        "sa_mlps": [[8, 12], [12, 16]],
        "attention_neighbours": 4,
    }
    return BACKBONES.build({**config, **overrides})


def clouds(sizes=(60, 60), seed: int = 0) -> PointSet:
    generator = torch.Generator().manual_seed(seed)
    return PointSet.cat([PointSet(torch.rand(size, 3, generator=generator)) for size in sizes])


def test_attention_layer_follows_the_formula():
    torch.manual_seed(1)
    layer = _VectorAttention(channels=6, neighbours=3)
    coords, feats = torch.rand(9, 3), torch.randn(9, 6)
    out = layer(coords, feats, [9])
    embedded = layer.embed(feats)
    query, key, value = layer.query(embedded), layer.key(embedded), layer.value(embedded)
    for point in range(9):
        distance = (coords - coords[point]).pow(2).sum(-1)
        nearest = sorted(range(9), key=lambda other: (float(distance[other]), other))[:3]
        assert nearest[0] == point  # a point is its own nearest neighbour
        position = [layer.position(coords[point] - coords[other]) for other in nearest]
        logits = torch.stack(
            [layer.attention(query[point] - key[o] + p) for o, p in zip(nearest, position)]
        )
        weight = torch.softmax(logits / math.sqrt(6), dim=0)  # over the neighbours, per channel
        mixed = sum(weight[n] * (value[o] + position[n]) for n, o in enumerate(nearest))
        assert torch.allclose(out[point], layer.project(mixed) + feats[point], atol=1e-5), point
    assert all(module.bias is None for module in (layer.query, layer.key, layer.value))
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in layer.modules())


def test_neighbours_stay_inside_a_cloud_and_ties_go_to_the_lower_index():
    layer = _VectorAttention(channels=4, neighbours=3)
    coords = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [0.0, 1, 0], [8.0, 9, 9], [8.5, 9, 9], [9.0, 9, 9]]
    )
    index = layer.neighbours(coords, [4, 3])
    assert index[0].tolist() == [0, 1, 2]  # points 1, 2 and 3 are equally far: lowest indices
    assert index[4:].min() >= 4 and index[:4].max() <= 3  # never across clouds
    assert index[5].tolist() == [5, 4, 6]


def test_forward_returns_features_on_the_sampled_subset():
    model = small().eval()
    points = clouds()
    out = model(points, None)
    assert model.output == "subsampled" and model.out_channels == 16
    assert out.feats.shape == (20, 16) and out.points.parent_id == points.id
    assert out.points.batch_offsets.tolist() == [0, 10, 20]
    index = out.points.parent_index
    assert index[:10].max() < 60 <= index[10:].min()  # every cloud keeps its own points
    assert torch.equal(out.points.coords, points.coords[index])
    alone = model(PointSet(points.coords[:60]), None)
    assert torch.allclose(alone.feats, out.feats[:10], atol=1e-5)  # clouds do not interact
    assert [name for name, _ in model.named_children()] == [
        "sa1",
        "transformer1",
        "sa2",
        "transformer2",
    ]


def test_every_parameter_receives_a_gradient_and_sizes_may_differ():
    model = small().train()
    out = model(clouds(sizes=(40, 70)), None)
    assert out.feats.shape == (20, 16)
    out.feats.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_the_sampling_tie_rule_is_an_option_and_decides_the_point_order():
    # a lattice: equal distances abound, so the two rules pick in another order
    axis = torch.arange(6.0) / 5
    lattice = PointSet(torch.cartesian_prod(axis, axis, axis) + 1.0)
    kernel, first = small(), small(fps_tie_break="lowest_index")
    assert kernel.sa1.tie_break == "pointnet2_cuda" and first.sa1.tie_break == "lowest_index"
    first.load_state_dict(kernel.state_dict())
    one, other = kernel.eval()(lattice, None), first.eval()(lattice, None)
    assert not torch.equal(one.points.parent_index, other.points.parent_index)
    with pytest.raises(ValueError, match="unknown fps_tie_break 'random'"):
        small(fps_tie_break="random")


def test_defaults_are_the_reference_backbone():
    model = BACKBONES.build({"type": "PointNet2Transformer"})
    assert model.npoints == (2048, 1024) and model.out_channels == 256 and model.in_channels == 0
    assert (model.sa1.radius, model.sa1.nsample, model.sa2.radius, model.sa2.nsample) == (
        0.04,
        64,
        0.1,
        32,
    )
    assert model.transformer1.channels == 128 and model.transformer2.num_neighbours == 16
    attention = sum(p.numel() for p in model.transformer2.parameters())
    assert attention == 2 * (256 * 257) + (3 * 256 + 256) + 3 * (256 * 257) + 3 * 256 * 256


def test_misuse_is_explained():
    with pytest.raises(ValueError, match="one entry per level"):
        small(radii=[0.4])
    with pytest.raises(ValueError, match="must not increase with depth"):
        small(npoints=[10, 24])
    with pytest.raises(ValueError, match=r"attention_neighbours must be in 1..10"):
        small(attention_neighbours=11)
    with pytest.raises(ContractError, match="samples 24 points per cloud .* has only 12"):
        small()(clouds(sizes=(60, 12)), None)
    with pytest.raises(ContractError, match="in_channels=0 .* received 3 feature channels"):
        small()(clouds(), torch.zeros(120, 3))
