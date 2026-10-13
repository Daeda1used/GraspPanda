"""MultiScaleGroup and its fusion: against ``nn.TransformerEncoder`` weight for weight,
against its own branches, and the configurations both must refuse."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

from grasppanda.models.groupers.cylinder_interaction import CylinderInteractionGroup
from grasppanda.models.groupers.multi_scale import MultiScaleGroup, ScaleAttentionFusion
from grasppanda.models.interaction import TokenInteraction
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import GROUPERS, BuildError
from grasppanda.structures import ContractError, PointFeatures, PointSet

from .test_gsnet_detector import load_converter

BRANCH = {
    "type": "CylinderInteractionGroup",
    "hmin": -0.1,
    "hmax": 0.25,
    "nsample": 5,
    "mlp_channels": [8, 16],
    "post_mlp_channels": [16, 16],
}


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def crop_inputs(frames: int = 2, seeds: int = 9, channels: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    cloud = torch.rand(frames, seeds, 3, generator=generator)
    points = PointSet.from_dense(cloud)
    support = PointFeatures(torch.randn(frames * seeds, channels, generator=generator), points)
    rotation = approach_to_rotation(
        torch.randn(frames * seeds, 3, generator=generator), torch.zeros(frames * seeds)
    )
    return support, points.coords, rotation


def build(**options) -> MultiScaleGroup:
    torch.manual_seed(0)
    node = {
        "type": "MultiScaleGroup",
        "branch": BRANCH,
        "radius": 0.4,
        "radius_scales": [0.5, 1.0],
        "fusion_heads": 4,
        "fusion_feedforward_channels": 32,
    }
    node.update(options)
    return GROUPERS.build(node, in_channels=6)


# ------------------------------------------------------------------------- fusion


def reference_fusion(channels: int, heads: int, hidden: int, layers: int, dropout: float):
    """What the FineGrasp reference builds: ``nn.TransformerEncoder`` over the scales
    (sequence first) and a linear score per token."""
    layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=hidden, dropout=dropout
    )
    encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
    return encoder, nn.Linear(channels, 1)


def load_reference(fusion: ScaleAttentionFusion, encoder: nn.Module, score: nn.Module) -> None:
    """Copy the reference weights through the checkpoint converter's own mapping."""
    converter = load_converter()
    source = {f"fuse_multi_scale.transformer.{k}": v for k, v in encoder.state_dict().items()}
    source.update({f"fuse_multi_scale.attn.{k}": v for k, v in score.state_dict().items()})
    converted = {}
    for key, value in source.items():
        converted.update(converter._translate_finegrasp(key, value))
    prefix = "grouper.fusion."
    assert all(name.startswith(prefix) for name in converted)
    fusion.load_state_dict({k[len(prefix) :]: v for k, v in converted.items()}, strict=True)


@pytest.mark.parametrize(("heads", "layers"), [(1, 1), (4, 2)])
def test_fusion_is_a_post_norm_transformer_encoder_with_attention_pooling(heads, layers):
    torch.manual_seed(0)
    encoder, score = reference_fusion(16, heads, 40, layers, dropout=0.1)
    fusion = ScaleAttentionFusion(16, layers, heads, 40, dropout=0.1)
    load_reference(fusion, encoder, score)
    tokens = torch.randn(7, 4, 16)  # (seeds, scales, channels)
    encoder.eval(), fusion.eval()
    with torch.no_grad():
        theirs = encoder(tokens.transpose(0, 1))  # (scales, seeds, channels)
        weights = torch.softmax(score(theirs), dim=0)
        expected = (theirs * weights).sum(dim=0)
        assert torch.allclose(fusion.encode(tokens), theirs.transpose(0, 1), atol=2e-6)
        assert torch.allclose(fusion.weights(fusion.encode(tokens)), weights[..., 0].T, atol=2e-6)
        assert torch.allclose(fusion(tokens), expected, atol=2e-6)


def test_fusion_matches_the_reference_layer_in_training_mode_without_dropout():
    """Train and eval mode differ by dropout only; with rate 0 the training paths agree."""
    torch.manual_seed(1)
    encoder, score = reference_fusion(8, 2, 12, 2, dropout=0.0)
    fusion = ScaleAttentionFusion(8, 2, 2, 12, dropout=0.0)
    load_reference(fusion, encoder, score)
    tokens = torch.randn(5, 3, 8, requires_grad=True)
    encoder.train(), fusion.train()
    theirs = encoder(tokens.transpose(0, 1))
    expected = (theirs * torch.softmax(score(theirs), dim=0)).sum(dim=0)
    mine = fusion(tokens)
    assert torch.allclose(mine, expected, atol=2e-6)
    (gradient,) = torch.autograd.grad(mine.sum(), tokens, retain_graph=True)
    (reference,) = torch.autograd.grad(expected.sum(), tokens)
    assert torch.allclose(gradient, reference, atol=1e-5)


def test_fusion_dropout_sits_where_the_reference_layer_has_it():
    """Four places per layer: attention weights, both residual branches, after the ReLU."""
    fusion = ScaleAttentionFusion(8, 1, 2, 12, dropout=0.25)
    (layer,) = fusion.layers
    reference = nn.TransformerEncoderLayer(8, 2, 12, dropout=0.25)
    assert reference.self_attn.dropout == layer.interaction.dropout.p == 0.25
    assert reference.dropout1.p == layer.interaction.residual_dropout.p == 0.25
    assert reference.dropout.p == layer.dropout.p == 0.25
    assert reference.dropout2.p == layer.residual_dropout.p == 0.25
    seen = {}
    hooks = {
        "attention": layer.interaction.dropout,
        "residual1": layer.interaction.residual_dropout,
        "hidden": layer.dropout,
        "residual2": layer.residual_dropout,
    }
    for name, module in hooks.items():
        module.register_forward_hook(lambda _m, i, _o, name=name: seen.__setitem__(name, i[0]))
    torch.manual_seed(0)
    tokens = torch.randn(6, 4, 8)
    fusion.train()
    assert not torch.equal(fusion(tokens), fusion(tokens))
    assert seen["attention"].shape == (6, 2, 4, 4)  # weights per head, rows sum to one
    assert torch.allclose(seen["attention"].sum(-1), torch.ones(6, 2, 4), atol=1e-6)
    assert seen["residual1"].shape == seen["residual2"].shape == (6, 4, 8)
    assert seen["hidden"].shape == (6, 4, 12) and bool((seen["hidden"] >= 0).all())  # after ReLU
    fusion.eval()
    assert torch.equal(fusion(tokens), fusion(tokens))


def test_fusion_initialises_its_attention_like_the_reference_layer():
    torch.manual_seed(0)
    fusion = ScaleAttentionFusion(64, 2, 8, 128)
    bound = (6.0 / (64 + 3 * 64)) ** 0.5  # xavier-uniform over the stacked (3C, C) projection
    for layer in fusion.layers:
        attention = layer.interaction
        for projection in (attention.query, attention.key, attention.value):
            assert float(projection.weight.abs().max()) <= bound
            assert float(projection.weight.abs().max()) > 0.9 * bound
            assert not projection.bias.any()
        assert not attention.output.bias.any()
        assert bool(layer.expand.bias.any())  # the feed-forward layers keep nn.Linear's default


def test_fusion_treats_seeds_independently_and_weights_sum_to_one():
    torch.manual_seed(2)
    fusion = ScaleAttentionFusion(8, 2, 2, 16).eval()
    tokens = torch.randn(6, 4, 8)
    with torch.no_grad():
        reference = fusion(tokens)
        changed = tokens.clone()
        changed[3] += 1.0
        assert torch.equal(fusion(changed)[[0, 1, 2, 4, 5]], reference[[0, 1, 2, 4, 5]])
        weights = fusion.weights(fusion.encode(tokens))
    assert weights.shape == (6, 4) and torch.allclose(weights.sum(1), torch.ones(6), atol=1e-6)
    with pytest.raises(ValueError, match=r"expected tokens of shape \(N, S, 8\)"):
        fusion(torch.zeros(6, 4, 9))
    with pytest.raises(ValueError, match="num_layers >= 0"):
        ScaleAttentionFusion(8, -1)


# ------------------------------------------------------------- residual dropout


def test_residual_dropout_is_off_by_default_and_draws_nothing():
    """EconomicGrasp's blocks are unchanged: no module that could consume a random number."""
    block = TokenInteraction(8, dropout=0.5)
    assert isinstance(block.residual_dropout, nn.Identity)
    assert "residual_dropout" not in "".join(block.state_dict())
    with pytest.raises(ValueError, match="residual_dropout must be in"):
        TokenInteraction(8, residual_dropout=1.0)
    block = TokenInteraction(8, residual_dropout=0.5)
    assert isinstance(block.residual_dropout, nn.Dropout) and block.dropout.p == 0.0
    tokens = torch.randn(32, 4, 8)
    block.train()
    assert not torch.equal(block(tokens), block(tokens))
    seen = []
    block.residual_dropout.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0]))
    block(tokens)
    assert seen[0].shape == tokens.shape  # the attention output, before the residual sum
    block.eval()
    assert torch.equal(block(tokens), block(tokens))


# ---------------------------------------------------------------------- the group


def test_every_scale_is_a_branch_of_its_own_radius_with_its_own_weights():
    group = build()
    assert group.support == "seeds" and group.out_axes == ("seed", "channel")
    assert group.radii == (0.4 * 0.5, 0.4 * 1.0) and group.out_channels == 16
    assert [type(branch) for branch in group.branches] == [CylinderInteractionGroup] * 2
    assert [branch.radius for branch in group.branches] == list(group.radii)
    assert all(branch.in_channels == 6 and branch.nsample == 5 for branch in group.branches)
    first, second = (branch.mlp[0].linear.weight for branch in group.branches)
    assert first is not second and not torch.equal(first, second)
    reference = build(radius=0.07, radius_scales=[0.25, 0.5, 0.75, 1.0])
    # The radii are the reference's products, bit for bit (0.07 * 0.75 is not 0.0525).
    assert reference.radii == (0.07 * 0.25, 0.07 * 0.5, 0.07 * 0.75, 0.07 * 1.0)
    assert "radius_scales=[0.25, 0.5, 0.75, 1.0]" in repr(reference)


def test_forward_is_the_fusion_of_the_branch_crops():
    group = build().eval()
    support, centres, rotation = crop_inputs()
    with torch.no_grad():
        crops = [branch(support, centres, rotation).feats for branch in group.branches]
        tokens = group.scale_features(support, centres, rotation)
        out = group(support, centres, rotation)
    assert tokens.shape == (18, 2, 16) and torch.equal(tokens, torch.stack(crops, dim=1))
    assert out.axes == ("seed", "channel") and out.feats.shape == (18, 16)
    assert torch.equal(out.feats, group.fusion(tokens))
    # The scales see different neighbourhoods.
    assert not torch.allclose(crops[0], crops[1])


def test_one_scale_without_layers_is_its_branch():
    """A soft-max over one token is 1: the wrapper adds nothing but the fusion."""
    group = build(radius_scales=[1.0], fusion_layers=0).eval()
    support, centres, rotation = crop_inputs()
    with torch.no_grad():
        alone = group.branches[0](support, centres, rotation).feats
        assert torch.equal(group(support, centres, rotation).feats, alone)
    plain = GROUPERS.build({**BRANCH, "radius": 0.4}, in_channels=6).eval()
    plain.load_state_dict(group.branches[0].state_dict())
    with torch.no_grad():
        assert torch.equal(plain(support, centres, rotation).feats, alone)


def test_any_seed_grouper_can_be_the_branch():
    group = build(branch={"type": "CylinderGroup", "nsample": 4, "mlp_channels": [8, 16]}).eval()
    support, centres, rotation = crop_inputs()
    with torch.no_grad():
        assert group(support, centres, rotation).feats.shape == (18, 16)
    assert type(group.branches[0]).__name__ == "CylinderGroup"


def test_gradients_reach_every_branch_and_the_fusion():
    group = build().train()
    support, centres, rotation = crop_inputs()
    group(support, centres, rotation).feats.sum().backward()
    for name, parameter in group.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert any(name.startswith("branches.1.") for name, _ in group.named_parameters())
    assert any(name.startswith("fusion.layers.1.") for name, _ in group.named_parameters())


# --------------------------------------------------------------------- rejections


def test_the_radius_has_one_owner():
    problems = list(MultiScaleGroup.check_config({"branch": {**BRANCH, "radius": 0.05}}))
    assert len(problems) == 1 and "remove `radius` from the branch" in problems[0]
    assert list(MultiScaleGroup.check_config({"branch": BRANCH})) == []
    with pytest.raises(ContractError, match="remove `radius: 0.2` from the branch"):
        build(branch={**BRANCH, "radius": 0.2})
    # A resolved config carries the default radius of the branch: that is not a second owner.
    assert build(branch={**BRANCH, "radius": 0.05}).radii == (0.2, 0.4)


def test_branches_of_another_kind_are_refused_before_and_at_build():
    scene = {"type": "DepthCylinderGroup"}
    problems = list(MultiScaleGroup.check_config({"branch": scene}))
    assert any("declares support = 'scene'" in problem for problem in problems)
    assert any("emits axes ('seed', 'depth', 'channel')" in problem for problem in problems)
    with pytest.raises(ContractError, match="crops among the seeds, but its branch DepthCylinder"):
        build(branch=scene)
    built = GROUPERS.build({**BRANCH, "radius": 0.4}, in_channels=6)
    with pytest.raises(BuildError, match="needs the branch as a config node"):
        build(branch=built)
    for options in ({"radius": 0.0}, {"radius_scales": []}, {"radius_scales": [0.5, -1.0]}):
        with pytest.raises(ValueError, match="radius > 0 and positive radius_scales"):
            build(**options)
