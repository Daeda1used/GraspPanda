"""TemplateViewStage and its ViewTemplate: selection policies, targets, loss, misuse."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from grasppanda.models.assigners.matched import MatchedLabels
from grasppanda.models.views.template import ViewTemplate
from grasppanda.models.views.template_view import TemplateViewStage
from grasppanda.ops import approach_to_rotation, fibonacci_views
from grasppanda.registry import VIEWS
from grasppanda.structures import ContractError, PointSet, SeedSet


def stage(num_views: int = 30, channels: int = 8, **overrides) -> TemplateViewStage:
    torch.manual_seed(0)
    return VIEWS.build(
        {"type": "TemplateViewStage", "num_views": num_views, **overrides}, in_channels=channels
    )


def seeds(count: int = 12, channels: int = 8, seed: int = 0) -> SeedSet:
    generator = torch.Generator().manual_seed(seed)
    points = PointSet(torch.rand(count, 3, generator=generator))
    return SeedSet(points=points, feats=torch.randn(count, channels, generator=generator))


def matched(view_friction: torch.Tensor, valid: torch.Tensor | None = None) -> MatchedLabels:
    count, _, angles, depths = view_friction.shape
    zeros = torch.zeros(count, 3)
    return MatchedLabels(
        valid=torch.ones(count, dtype=torch.bool) if valid is None else valid,
        label_point=zeros,
        distance=torch.zeros(count),
        object_index=torch.zeros(count, dtype=torch.long),
        crop_centre=zeros,
        crop_rotation=torch.eye(3).expand(count, 3, 3),
        view_friction=view_friction,
        friction=torch.zeros(count, angles, depths),
        width=torch.zeros(count, angles, depths),
    )


# ----------------------------------------------------------------------- template


def test_template_is_the_fibonacci_lattice_with_its_zero_angle_frames():
    template = ViewTemplate.fibonacci(300)
    assert template.num_views == len(template) == 300
    assert torch.equal(template.views, fibonacci_views(300))
    assert torch.equal(template.rotations, approach_to_rotation(-fibonacci_views(300)))
    index = torch.tensor([[0, 7], [299, 150]])
    assert torch.equal(template.approach(index), -template.views[index])
    assert template.rotation(index).shape == (2, 2, 3, 3)
    assert torch.allclose(template.rotation(index)[..., 0], template.approach(index), atol=1e-6)
    assert template.state_dict() == {}  # derived from num_views: not part of a checkpoint
    assert template.double().views.dtype == torch.float64  # buffers follow the module
    with pytest.raises(ValueError, match=r"views must be \(V, 3\)"):
        ViewTemplate(torch.zeros(5, 2))


def test_the_stage_owns_its_template():
    view = stage(num_views=30)
    assert isinstance(view.template, ViewTemplate) and view.template.num_views == 30
    assert view.score.out_features == 30 and view.hidden.out_features == 8
    assert not any("template" in key for key in view.state_dict())
    assert view.loss_fn.weight == 100.0 and view.max_friction == 0.6


# ---------------------------------------------------------------------- selection


def test_inference_takes_the_best_view_and_returns_refined_seeds():
    view = stage().eval()
    inputs = seeds()
    approach, refined = view(inputs)
    residual = F.relu(view.hidden(inputs.feats))
    assert torch.allclose(approach.logits, view.score(residual), atol=1e-6)
    assert torch.equal(approach.view_index, approach.logits.argmax(dim=-1))
    assert torch.equal(approach.direction, view.template.approach(approach.view_index))
    assert torch.equal(approach.rotation, view.template.rotation(approach.view_index))
    assert torch.allclose(approach.direction.norm(dim=-1), torch.ones(12), atol=1e-6)
    assert torch.allclose(refined.feats, inputs.feats + residual, atol=1e-6)
    assert refined.points is inputs.points and refined is not inputs
    assert approach.logits.requires_grad and not approach.view_index.is_floating_point()


def test_training_samples_views_in_proportion_to_their_normalised_scores():
    view = stage(num_views=5).train()
    # Views 0-2 hold the row minimum (weight 0 after min-max scaling); views 3 and 4 get
    # weights 1/3 and 1, so view 4 must be drawn about three times as often as view 3.
    logits = torch.tensor([[-2.0, -2.0, -2.0, -1.0, 1.0]]).repeat(4000, 1)
    torch.manual_seed(1)
    drawn = view.select(logits)
    counts = torch.bincount(drawn, minlength=5)
    assert counts[:3].sum() == 0
    assert counts[4] / counts[3] == pytest.approx(3.0, rel=0.15)
    torch.manual_seed(1)
    assert torch.equal(view.select(logits), drawn)  # reproducible from the global seed
    assert torch.equal(view.eval().select(logits), torch.full((4000,), 4))


def test_training_policy_is_a_named_option():
    logits = torch.randn(64, 30, generator=torch.Generator().manual_seed(2))
    greedy = stage(train_selection="argmax").train()
    assert torch.equal(greedy.select(logits), logits.argmax(dim=-1))
    sampling = stage().train()
    assert sampling.train_selection == "multinomial"
    assert not torch.equal(sampling.select(logits), logits.argmax(dim=-1))
    with pytest.raises(ValueError, match="unknown train_selection 'best'"):
        TemplateViewStage(train_selection="best", in_channels=8)


def test_a_seed_without_preference_is_sampled_uniformly_instead_of_failing():
    view = stage(num_views=4).train()
    logits = torch.zeros(2000, 4)
    torch.manual_seed(0)
    counts = torch.bincount(view.select(logits), minlength=4)
    assert (counts > 400).all()  # every view is drawn; the reference would raise here


def test_a_recorded_draw_can_be_replayed():
    view = stage().train()
    inputs = seeds()
    recorded = torch.arange(12) % 30
    approach, _ = view(inputs, view_index=recorded)
    assert torch.equal(approach.view_index, recorded)
    assert torch.equal(approach.rotation, view.template.rotation(recorded))
    with pytest.raises(ContractError, match=r"one view per seed \(12,\)"):
        view(inputs, view_index=recorded[:5])
    with pytest.raises(ContractError, match=r"needs \(M, 8\) seed features"):
        view(seeds(channels=6))


# ------------------------------------------------------------------ targets, loss


def test_view_targets_are_the_normalised_share_of_good_grasps():
    view = stage(num_views=6)
    generator = torch.Generator().manual_seed(3)
    levels = torch.tensor([-1.0, 0.0, 0.2, 0.5, 0.7, 1.0])
    friction = levels[torch.randint(0, 6, (5, 6, 3, 2), generator=generator)]
    friction[4] = -1.0  # a seed without any grasp: every view scores 0
    target = view.encode_targets(friction)
    for seed in range(5):
        share = [
            sum(0 < float(v) <= view.max_friction for v in friction[seed, v_].flatten()) / 6
            for v_ in range(6)
        ]
        low, high = min(share), max(share)
        for v_ in range(6):
            expected = (share[v_] - low) / (high - low + 1e-5)
            assert target[seed, v_].item() == pytest.approx(expected, abs=1e-6)
    assert target[4].abs().max() == 0 and target[:4].max() <= 1.0
    strict = stage(num_views=6, max_friction=0.2).encode_targets(friction)
    assert not torch.equal(strict, target)  # the friction limit is this stage's own option
    # The limit itself counts: labels are float32, and 0.6 must pass a limit of 0.6.
    edge = torch.stack([torch.full((3, 2), 0.6), torch.full((3, 2), 0.7)])[None]
    assert view.encode_targets(edge.expand(1, 2, 3, 2)).tolist() == [
        [pytest.approx(1.0, abs=1e-4), 0.0]
    ]


def test_loss_regresses_all_views_of_the_supervised_seeds():
    view = stage(num_views=6).eval()
    inputs = seeds(count=5)
    approach, _ = view(inputs)
    friction = torch.rand(5, 6, 3, 2, generator=torch.Generator().manual_seed(4))
    target = view.encode_targets(friction)
    full = view.loss(approach, matched(friction))
    assert set(full) == {"loss/view"}
    assert full["loss/view"].item() == pytest.approx(
        100.0 * F.smooth_l1_loss(approach.logits, target).item(), rel=1e-6
    )
    valid = torch.tensor([True, False, True, True, False])
    partial = view.loss(approach, matched(friction, valid))
    assert partial["loss/view"].item() == pytest.approx(
        100.0 * F.smooth_l1_loss(approach.logits[valid], target[valid]).item(), rel=1e-6
    )
    partial["loss/view"].backward()
    assert view.score.weight.grad.abs().sum() > 0


def test_labels_on_another_view_lattice_are_refused():
    view = stage(num_views=6)
    approach, _ = view(seeds(count=5))
    with pytest.raises(ContractError, match="describe 8 views, but the view template has 6"):
        view.loss(approach, matched(torch.zeros(5, 8, 3, 2)))
