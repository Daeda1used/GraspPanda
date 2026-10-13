"""TokenInteraction against attention written out by hand, CylinderInteractionGroup
step by step, and the inputs both must refuse."""

from __future__ import annotations

import math

import pytest
import torch

from grasppanda.models.groupers.cylinder import CylinderGroup
from grasppanda.models.groupers.cylinder_interaction import CylinderInteractionGroup
from grasppanda.models.interaction import TokenInteraction
from grasppanda.ops import approach_to_rotation
from grasppanda.registry import GROUPERS, BuildError
from grasppanda.structures import ContractError, PointFeatures, PointSet

RADIUS, HMIN, HMAX, NSAMPLE = 0.3, -0.1, 0.25, 5


# -------------------------------------------------------------------- interaction


def attention_oracle(block: TokenInteraction, tokens: torch.Tensor) -> torch.Tensor:
    """LayerNorm(x + W_o Σ_j softmax_j(q_i · k_j / sqrt(d)) v_j), one token at a time."""
    out = torch.empty_like(tokens)
    width = block.channels // block.num_heads
    for row, sequence in enumerate(tokens):
        query, key, value = block.query(sequence), block.key(sequence), block.value(sequence)
        mixed = torch.zeros_like(sequence)
        for head in range(block.num_heads):
            part = slice(head * width, (head + 1) * width)
            for i in range(len(sequence)):
                weights = torch.stack(
                    [query[i, part] @ key[j, part] / math.sqrt(width) for j in range(len(sequence))]
                ).softmax(dim=0)
                mixed[i, part] = (weights[:, None] * value[:, part]).sum(dim=0)
        out[row] = block.norm(sequence + block.output(mixed))
    return out


@pytest.mark.parametrize("heads", [1, 2])
def test_interaction_is_scaled_dot_product_attention_with_residual_and_norm(heads):
    torch.manual_seed(0)
    block = TokenInteraction(6, num_heads=heads).eval()
    tokens = torch.randn(3, 4, 6)
    with torch.no_grad():
        assert torch.allclose(block(tokens), attention_oracle(block, tokens), atol=1e-6)
        weights = block.attention(tokens)
    assert weights.shape == (3, heads, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, heads, 4), atol=1e-6)


def test_tokens_interact_within_a_row_only_and_order_does_not_matter():
    torch.manual_seed(1)
    block = TokenInteraction(8).eval()
    tokens = torch.randn(5, 4, 8)
    with torch.no_grad():
        reference = block(tokens)
        changed = tokens.clone()
        changed[2] += 1.0
        assert torch.equal(block(changed)[[0, 1, 3, 4]], reference[[0, 1, 3, 4]])
        order = torch.tensor([2, 0, 3, 1])  # no positional encoding: permutation-equivariant
        assert torch.allclose(block(tokens[:, order]), reference[:, order], atol=1e-6)


def test_dropout_acts_on_attention_weights_in_training_only():
    torch.manual_seed(2)
    block = TokenInteraction(8, dropout=0.5)
    tokens = torch.randn(64, 4, 8)
    assert isinstance(block.dropout, torch.nn.Dropout) and block.dropout.p == 0.5
    block.train()
    assert not torch.equal(block(tokens), block(tokens))
    block.eval()
    assert torch.equal(block(tokens), block(tokens))
    seen = []
    block.dropout.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0]))
    block(tokens)
    assert seen[0].shape == (64, 1, 4, 4) and torch.allclose(seen[0].sum(-1), torch.ones(64, 1, 4))


def test_interaction_rejects_bad_sizes():
    with pytest.raises(ValueError, match="num_heads must divide channels"):
        TokenInteraction(6, num_heads=4)
    with pytest.raises(ValueError, match="dropout must be in"):
        TokenInteraction(6, dropout=1.0)
    with pytest.raises(ValueError, match=r"expected tokens of shape \(N, T, 6\)"):
        TokenInteraction(6)(torch.zeros(3, 6))


# ------------------------------------------------------------------------ grouper


def grouper(channels: int = 4, **overrides) -> CylinderInteractionGroup:
    torch.manual_seed(0)
    config = {
        "type": "CylinderInteractionGroup",
        "radius": RADIUS,
        "hmin": HMIN,
        "hmax": HMAX,
        "nsample": NSAMPLE,
        "mlp_channels": [6, 7],
        "post_mlp_channels": [9, 8],
    }
    return GROUPERS.build({**config, **overrides}, in_channels=channels)


def inputs(batch: int = 2, per_frame: int = 40, channels: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    points = PointSet.from_dense(torch.rand(batch, per_frame, 3, generator=generator))
    feats = torch.randn(batch * per_frame, channels, generator=generator)
    rotation = approach_to_rotation(
        torch.randn(batch * per_frame, 3, generator=generator),
        torch.rand(batch * per_frame, generator=generator) * math.pi,
    )
    return PointFeatures(feats, points), points.coords.clone(), rotation


def shared_mlp(layers, values: torch.Tensor) -> torch.Tensor:
    """Linear (no bias) → batch norm → ReLU, the same for every neighbour."""
    for layer in layers:
        values = torch.relu(layer.norm(layer.linear(values).transpose(1, 2)).transpose(1, 2))
    return values


def test_the_crop_starts_as_the_plain_cylinder_group_does():
    stage = grouper()
    plain = GROUPERS.build(
        {
            "type": "CylinderGroup",
            "radius": RADIUS,
            "hmin": HMIN,
            "hmax": HMAX,
            "nsample": NSAMPLE,
            "mlp_channels": [6, 7],
        },
        in_channels=4,
    )
    assert isinstance(stage, CylinderGroup)
    support, centres, rotation = inputs()
    assert torch.equal(
        stage.group(support, centres, rotation), plain.group(support, centres, rotation)
    )
    assert stage.support == "seeds" and stage.out_axes == ("seed", "channel")


def test_forward_is_mlp_centre_coordinates_interaction_mlp_and_max():
    stage = grouper().eval()
    support, centres, rotation = inputs()
    with torch.no_grad():
        out = stage(support, centres, rotation)
        grouped = stage.group(support, centres, rotation)
        hidden = shared_mlp(stage.mlp, grouped)
        tokens = torch.cat([hidden, centres[:, None, :].expand(-1, NSAMPLE, -1)], dim=-1)
        assert torch.allclose(stage.tokens(support, centres, rotation), tokens, atol=1e-6)
        mixed = attention_oracle(stage.interaction, tokens)
        expected = shared_mlp(stage.post_mlp, mixed).amax(dim=1)
    assert out.axes == ("seed", "channel") and out.feats.shape == (80, 8)
    assert stage.out_channels == 8 and stage.token_channels == 7 + 3
    assert torch.allclose(out.feats, expected, atol=1e-5)
    assert stage.interaction.channels == 10 and stage.post_mlp[0].linear.in_features == 10
    assert stage.post_mlp[0].linear.bias is None


def test_the_centre_coordinates_are_those_of_the_crop_and_optional():
    stage = grouper().eval()
    support, centres, rotation = inputs()
    with torch.no_grad():
        tokens = stage.tokens(support, centres, rotation)
        assert torch.equal(tokens[..., 7:], centres[:, None, :].expand(-1, NSAMPLE, -1))
        # The crop feature therefore depends on where the scene sits in the camera frame ...
        shifted = PointFeatures(support.feats, PointSet.from_dense(support.points.dense() + 0.5))
        assert not torch.allclose(
            stage(shifted, centres + 0.5, rotation).feats, stage(support, centres, rotation).feats
        )
        # ... unless the option is switched off.
        free = grouper(centre_coordinates=False).eval()
        assert free.token_channels == 7 and free.interaction.channels == 7
        assert torch.allclose(
            free(shifted, centres + 0.5, rotation).feats,
            free(support, centres, rotation).feats,
            atol=1e-5,
        )


def test_gradients_reach_the_support_features_and_every_parameter():
    stage = grouper().train()
    support, centres, rotation = inputs()
    support.feats.requires_grad_(True)
    stage(support, centres, rotation).feats.square().sum().backward()
    assert support.feats.grad.abs().sum() > 0
    assert all(p.grad is not None for p in stage.parameters())


def test_contract_violations_are_explained():
    stage = grouper()
    support, centres, rotation = inputs()
    with pytest.raises(ContractError, match="built for 4-channel support features, got 9"):
        stage(PointFeatures(torch.zeros(80, 9), support.points), centres, rotation)
    with pytest.raises(ContractError, match=r"groups among the seeds .* M = 80"):
        stage(support, centres[:30], rotation[:30])


def test_defaults_are_the_reference_crop_and_bad_values_are_rejected():
    stage = GROUPERS.build({"type": "CylinderInteractionGroup"}, in_channels=512)
    assert (stage.radius, stage.hmin, stage.hmax, stage.nsample) == (0.05, -0.02, 0.04, 16)
    assert stage.mlp[0].linear.in_features == 515 and stage.token_channels == 259
    assert stage.interaction.num_heads == 1 and stage.interaction.dropout.p == 0.05
    assert stage.post_mlp[0].linear.in_features == 259 and stage.out_channels == 256
    assert sum(p.numel() for p in stage.parameters()) == 601_142  # counted in the reference
    for layer, fan_in in zip(stage.post_mlp, (259, 256)):  # the reference's shared-MLP init
        assert layer.linear.weight.std().item() == pytest.approx(math.sqrt(2 / fan_in), rel=0.05)
    with pytest.raises(BuildError, match="needs in_channels"):
        GROUPERS.build({"type": "CylinderInteractionGroup"})
    with pytest.raises(ValueError, match="at least one post-interaction width"):
        GROUPERS.build({"type": "CylinderInteractionGroup", "post_mlp_channels": []}, in_channels=8)
    with pytest.raises(ValueError, match="CylinderGroup needs radius > 0"):
        GROUPERS.build({"type": "CylinderInteractionGroup", "nsample": 0}, in_channels=8)
    with pytest.raises(ValueError, match="num_heads must divide channels"):
        GROUPERS.build({"type": "CylinderInteractionGroup", "attention_heads": 2}, in_channels=8)
