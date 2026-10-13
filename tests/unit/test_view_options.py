"""The view stages beyond the references: other lattices, classification, soft-arg-max,
top-k sampling, and regression of the approach or of a full rotation.

Contract (what the assigner, the grouper and the head receive), a backward pass, the
likeliest misuse, and every stage inside each of the five seed-based methods.
"""

from __future__ import annotations

import math

import pytest
import torch

from grasppanda.models.assigners.matched import CompactMatchedLabels, MatchedLabels
from grasppanda.models.detectors.seed_based import stage_contract_problems
from grasppanda.ops import (
    approach_to_rotation,
    fibonacci_views,
    nearest_view,
    two_vector_rotation,
)
from grasppanda.registry import VIEWS
from grasppanda.structures import ContractError, PointSet, SeedSet
from grasppanda.utils.imports import MissingDependencyError

from .test_swap_matrix import BRIDGE_STAGES, METHODS, STAGES, encoder, method, refusal, run

LABEL_VIEWS, ANGLES, DEPTHS, CHANNELS, COUNT = 20, 4, 2, 8, 16
REGRESSION = ("DirectionViewStage", "TwoVectorViewStage", "RomaRotationViewStage")
NEEDS = {
    "IcosphereViewStage": "trimesh",
    "HealpixViewStage": "healpix",
    "RomaRotationViewStage": "roma",
}


def build(kind: str, **options):
    if kind in NEEDS:
        pytest.importorskip(NEEDS[kind])
    torch.manual_seed(0)
    node = {"type": kind, "label_views": LABEL_VIEWS, **options}
    if kind in ("TwoVectorViewStage", "RomaRotationViewStage"):
        node.setdefault("num_angle", ANGLES)
    return VIEWS.build(node, in_channels=CHANNELS)


def seeds(seed: int = 0) -> SeedSet:
    generator = torch.Generator().manual_seed(seed)
    points = PointSet(torch.rand(COUNT, 3, generator=generator))
    return SeedSet(points=points, feats=torch.randn(COUNT, CHANNELS, generator=generator))


def dense(approach, seed: int = 0, weight: torch.Tensor | None = None) -> MatchedLabels:
    """Dense labels whose crop frame is the label frame of the snapped view."""
    generator = torch.Generator().manual_seed(seed)
    view_friction = torch.rand(COUNT, LABEL_VIEWS, ANGLES, DEPTHS, generator=generator)
    view_friction[0] = 0.0  # a seed without any grasp
    valid = torch.ones(COUNT, dtype=torch.bool)
    valid[1] = False
    rows = torch.arange(COUNT)
    return MatchedLabels(
        valid=valid,
        label_point=torch.zeros(COUNT, 3),
        distance=torch.zeros(COUNT),
        object_index=torch.zeros(COUNT, dtype=torch.long),
        crop_centre=torch.zeros(COUNT, 3),
        crop_rotation=approach_to_rotation(-fibonacci_views(LABEL_VIEWS))[approach.view_index],
        view_friction=view_friction,
        friction=view_friction[rows, approach.view_index],
        width=torch.zeros(COUNT, ANGLES, DEPTHS),
        weight=weight,
    )


def compact(approach, seed: int = 0, max_friction: float = 0.6) -> CompactMatchedLabels:
    generator = torch.Generator().manual_seed(seed)
    valid = torch.ones(COUNT, dtype=torch.bool)
    valid[1] = False
    return CompactMatchedLabels(
        valid=valid,
        view_valid=valid,
        label_point=torch.zeros(COUNT, 3),
        distance=torch.zeros(COUNT),
        object_index=torch.zeros(COUNT, dtype=torch.long),
        crop_centre=torch.zeros(COUNT, 3),
        crop_rotation=approach_to_rotation(-fibonacci_views(LABEL_VIEWS))[approach.view_index],
        view_graspness=torch.rand(COUNT, LABEL_VIEWS, generator=generator),
        view_max_friction=max_friction,
        view_labelled=torch.ones(COUNT, dtype=torch.bool),
        angle=torch.randint(0, ANGLES, (COUNT,), generator=generator),
        depth=torch.zeros(COUNT, dtype=torch.long),
        score=torch.ones(COUNT),
        width=torch.zeros(COUNT),
    )


def check_approach(stage, approach, refined, inputs) -> None:
    """What every neighbour relies on, whatever the parameterisation."""
    assert approach.direction.shape == (COUNT, 3)
    assert torch.allclose(approach.direction.norm(dim=-1), torch.ones(COUNT), atol=1e-5)
    assert approach.rotation.shape == (COUNT, 3, 3)
    assert torch.allclose(approach.rotation[..., 0], approach.direction, atol=1e-5)
    assert torch.allclose(approach.rotation, approach_to_rotation(approach.direction), atol=1e-5)
    # The view index addresses the label lattice (the assigner's), and is its nearest view.
    assert stage.template.num_views == LABEL_VIEWS
    assert approach.view_index.dtype == torch.long and approach.view_index.shape == (COUNT,)
    nearest = nearest_view(approach.direction, -stage.template.views)
    assert torch.equal(approach.view_index, nearest)
    assert len(refined) == COUNT and refined.feats.shape == inputs.feats.shape
    assert not torch.equal(refined.feats, inputs.feats)  # the residual path


# ------------------------------------------------------------------------ lattices


def test_lattices_are_unit_vectors_of_the_documented_sizes():
    from grasppanda.ops import healpix_views, icosphere_views

    trimesh = pytest.importorskip("trimesh")
    assert trimesh is not None
    for level, count in ((0, 12), (1, 42), (2, 162)):
        views = icosphere_views(level)
        assert views.shape == (count, 3)
        assert torch.allclose(views.norm(dim=-1), torch.ones(count), atol=1e-6)
    with pytest.raises(ValueError, match="subdivisions must be between 0 and 5"):
        icosphere_views(9)
    pytest.importorskip("healpix")
    for nside, count in ((1, 12), (2, 48), (4, 192), (5, 300)):
        views = healpix_views(nside)
        assert views.shape == (count, 3)
        assert torch.allclose(views.norm(dim=-1), torch.ones(count), atol=1e-6)
    # Equal-area pixels: the views average to the centre of the sphere.
    assert healpix_views(4).mean(dim=0).abs().max() < 1e-6


def test_a_bridge_names_the_extra_when_its_library_is_missing(monkeypatch):
    import grasppanda.utils.imports as imports

    real = imports.importlib.import_module
    VIEWS.names()  # the registry imports its package before the libraries go missing

    def missing(name):
        if name.split(".")[0] in NEEDS.values():
            raise ImportError(name)
        return real(name)

    monkeypatch.setattr(imports.importlib, "import_module", missing)
    for kind, extra in NEEDS.items():
        with pytest.raises(MissingDependencyError, match=rf"grasppanda\[{extra}\]"):
            VIEWS.build({"type": kind}, in_channels=CHANNELS)


def test_two_vector_rotation_is_the_gram_schmidt_frame():
    generator = torch.Generator().manual_seed(0)
    first, second = torch.randn(7, 3, generator=generator), torch.randn(7, 3, generator=generator)
    frame = two_vector_rotation(first, second)
    assert torch.allclose(frame @ frame.transpose(-1, -2), torch.eye(3).expand(7, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(frame), torch.ones(7), atol=1e-5)
    assert torch.allclose(frame[..., 0], first / first.norm(dim=-1, keepdim=True), atol=1e-6)
    assert bool(((frame[..., 1] * second).sum(dim=-1) > 0).all())
    roma = pytest.importorskip("roma")
    reference = roma.special_gramschmidt(torch.stack([first, second], dim=-1))
    assert torch.allclose(frame, reference, atol=1e-5)


# ------------------------------------------------------------------- lattice stages


def test_the_default_lattice_stage_computes_what_the_template_stage_computes():
    mine = build("LatticeViewStage", num_views=LABEL_VIEWS, train_selection="argmax")
    reference = VIEWS.build(
        {"type": "TemplateViewStage", "num_views": LABEL_VIEWS, "train_selection": "argmax"},
        in_channels=CHANNELS,
    )
    reference.load_state_dict(mine.state_dict())
    assert mine.identity
    inputs = seeds()
    approach, refined = mine(inputs)
    expected, expected_seeds = reference(inputs)
    assert torch.equal(approach.view_index, expected.view_index)
    assert torch.allclose(approach.logits, expected.logits)
    assert torch.allclose(approach.rotation, expected.rotation)
    assert torch.allclose(refined.feats, expected_seeds.feats)
    labels = dense(approach)
    assert torch.allclose(
        mine.loss(approach, labels)["loss/view"], reference.loss(expected, labels)["loss/view"]
    )


@pytest.mark.parametrize(
    ("kind", "options", "views"),
    [
        ("LatticeViewStage", {"num_views": 9}, 9),
        ("LatticeViewStage", {"num_views": 64}, 64),
        ("LatticeViewStage", {"num_views": 40, "hemisphere": True}, 20),
        ("IcosphereViewStage", {"subdivisions": 1}, 42),
        ("HealpixViewStage", {"nside": 2}, 48),
    ],
)
@pytest.mark.parametrize("target", ["score", "best_view"])
@pytest.mark.parametrize("labels", [dense, compact])
def test_a_lattice_stage_keeps_its_lattice_apart_from_the_label_lattice(
    kind, options, views, target, labels
):
    stage = build(kind, target=target, **options)
    assert stage.lattice.num_views == views and stage.template.num_views == LABEL_VIEWS
    assert stage.score.out_features == views and not stage.identity
    if options.get("hemisphere"):
        assert bool((stage.lattice.approach(torch.arange(views))[:, 2] > 0).all())
    inputs = seeds()
    for mode in (stage.train(), stage.eval()):
        approach, refined = mode(inputs)
        assert approach.logits.shape == (COUNT, views)
        # The approach is a view of the stage's own lattice; the index is its label view.
        chosen = (approach.direction @ -stage.lattice.views.T).argmax(dim=-1)
        assert torch.allclose(approach.direction, stage.lattice.approach(chosen), atol=1e-6)
        assert torch.equal(approach.view_index, stage.to_label[chosen])
        if views <= LABEL_VIEWS or not stage.training:
            check_approach(stage, approach, refined, inputs)
    stage.train()
    approach, _ = stage(inputs)
    matched = labels(approach)
    losses = stage.loss(approach, matched)
    assert torch.isfinite(losses["loss/view"]) and losses["loss/view"] > 0
    losses["loss/view"].backward()
    assert stage.score.weight.grad.abs().sum() > 0 and stage.hidden.weight.grad.abs().sum() > 0
    target_value, supervised, _ = stage.targets(matched)
    assert not supervised[1]
    if target == "best_view":
        assert "metric/view_accuracy" in losses and target_value.shape == (COUNT,)
        assert int(target_value.max()) < views
        if labels is dense:
            assert not supervised[0]  # no graspable view: nothing to classify
    else:
        assert target_value.shape == (COUNT, views)
        assert float(target_value.min()) >= 0 and float(target_value.max()) <= 1


def test_the_target_of_a_prediction_view_is_the_graspness_of_its_nearest_label_view():
    stage = build("LatticeViewStage", num_views=7, target="best_view")
    approach, _ = stage(seeds())
    matched = dense(approach)
    good = (matched.view_friction > 0) & (matched.view_friction <= 0.6)
    graspness = good.flatten(-2).float().mean(dim=-1)
    expected = graspness[:, stage.to_label].argmax(dim=-1)
    assert torch.equal(stage.targets(matched)[0], expected)


def test_soft_argmax_returns_a_direction_between_views_and_argmax_at_low_temperature():
    stage = build(
        "LatticeViewStage", num_views=50, selection="soft_argmax", top_k=4, temperature=0.05
    ).eval()
    inputs = seeds()
    approach, refined = stage(inputs)
    check_approach(stage, approach, refined, inputs)
    on_lattice = (approach.direction @ -stage.lattice.views.T).amax(dim=-1)
    assert bool((on_lattice < 1 - 1e-4).any())  # not a lattice view
    best = stage.lattice.approach(approach.logits.argmax(dim=-1))
    spacing = math.radians(40)  # the 4 best of 50 views are not all neighbours
    assert float((approach.direction * best).sum(dim=-1).mean()) > math.cos(spacing)
    stage.temperature = 1e-4
    sharp, _ = stage(inputs)
    assert torch.allclose(sharp.direction, best, atol=1e-4)
    # Training does not use the read-out: it selects a lattice view.
    trained, _ = stage.train()(inputs)
    assert bool(((trained.direction @ -stage.lattice.views.T).amax(dim=-1) > 1 - 1e-5).all())


def test_topk_sampling_draws_among_the_best_views_only():
    stage = build("LatticeViewStage", num_views=30, train_selection="topk", top_k=3).train()
    logits = torch.randn(COUNT, 30, generator=torch.Generator().manual_seed(1))
    allowed = logits.topk(3, dim=-1).indices
    seen = set()
    for _ in range(40):
        choice = stage.select(logits)
        assert bool((choice[:, None] == allowed).any(dim=-1).all())
        seen.add(int(choice[0]))
    assert len(seen) > 1  # it samples
    assert torch.equal(stage.eval().select(logits), logits.argmax(dim=-1))
    assert stage.train().select(torch.zeros(COUNT, 30)).shape == (COUNT,)


def test_seed_weights_change_the_lattice_loss():
    stage = build("LatticeViewStage", num_views=12)
    approach, _ = stage(seeds())
    plain = stage.loss(approach, dense(approach))["loss/view"]
    weight = torch.zeros(COUNT)
    weight[2] = 1.0
    weighted = stage.loss(approach, dense(approach, weight=weight))["loss/view"]
    assert not torch.allclose(plain, weighted)
    assert VIEWS.get("LatticeViewStage").applies_seed_weights
    problems = stage_contract_problems(
        view={"type": "LatticeViewStage"}, assigner={"type": "ScaleBalancedAssigner"}
    )
    assert not any("LatticeViewStage do" in message for message in problems), problems


def test_misuse_of_a_lattice_stage_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="unknown target 'softmax'"):
        build("LatticeViewStage", target="softmax")
    with pytest.raises(ValueError, match="train_selection 'topk' needs top_k"):
        build("LatticeViewStage", train_selection="topk")
    with pytest.raises(ValueError, match="top_k must be null or between 1 and the 10 views"):
        build("LatticeViewStage", num_views=10, top_k=11)
    with pytest.raises(ValueError, match="temperature must be positive"):
        build("LatticeViewStage", temperature=0.0)
    stage = build("LatticeViewStage", num_views=12)
    approach, _ = stage(seeds())
    matched = dense(approach)
    matched.view_friction = matched.view_friction[:, :5]
    with pytest.raises(ContractError, match=r"describe 5 views, but LatticeViewStage was built"):
        stage.loss(approach, matched)
    with pytest.raises(ContractError, match="max_friction = 0.4, but LatticeViewStage"):
        stage.loss(approach, compact(approach, max_friction=0.4))
    with pytest.raises(ContractError, match=r"view_index must hold one view per seed"):
        stage(seeds(), view_index=torch.zeros(3, dtype=torch.long))
    bad = seeds()
    bad.feats = bad.feats[:, :3]
    with pytest.raises(ContractError, match=rf"needs \(M, {CHANNELS}\) seed features"):
        stage(bad)


# ---------------------------------------------------------------- regression stages


def regression_cases():
    yield "DirectionViewStage", {}
    yield "TwoVectorViewStage", {"keep": "approach"}
    yield "TwoVectorViewStage", {"keep": "baseline"}
    for representation in ("6d", "9d", "quaternion", "rotvec"):
        yield "RomaRotationViewStage", {"representation": representation}


@pytest.mark.parametrize(("kind", "options"), list(regression_cases()))
@pytest.mark.parametrize("labels", [dense, compact])
def test_a_regression_stage_hands_over_a_direction_its_frame_and_the_nearest_label_view(
    kind, options, labels
):
    stage = build(kind, **options)
    inputs = seeds()
    approach, refined = stage(inputs)
    check_approach(stage, approach, refined, inputs)
    assert approach.logits is None and not approach.direction.requires_grad
    with_frame = kind != "DirectionViewStage"
    if with_frame:
        frame = approach.frame
        assert frame.shape == (COUNT, 3, 3)
        eye = torch.eye(3).expand(COUNT, 3, 3)
        assert torch.allclose(frame @ frame.transpose(-1, -2), eye, atol=1e-4)
        assert torch.allclose(torch.linalg.det(frame), torch.ones(COUNT), atol=1e-4)
        assert torch.allclose(frame[..., 0], approach.direction, atol=1e-5)
    else:
        assert approach.frame is None
    matched = labels(approach)
    losses = stage.loss(approach, matched)
    expected = {"loss/view", "metric/view_error_deg"}
    if with_frame:
        expected |= {"loss/view_frame", "metric/view_frame_error_deg"}
    assert set(losses) == expected
    assert all(torch.isfinite(value) for value in losses.values())
    sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
    assert stage.output.weight.grad.abs().sum() > 0 and stage.hidden.weight.grad.abs().sum() > 0
    # A replayed view replaces the regressed direction for the neighbours.
    replay = torch.arange(COUNT) % LABEL_VIEWS
    replayed, _ = stage(inputs, view_index=replay)
    assert torch.equal(replayed.view_index, replay)
    assert torch.allclose(replayed.direction, stage.template.approach(replay))


def test_the_direction_target_is_the_best_label_view_and_the_loss_vanishes_there():
    stage = build("DirectionViewStage")
    approach, _ = stage(seeds())
    matched = dense(approach)
    target, supervised, weight = stage.direction_target(matched)
    assert weight is None and not supervised[0] and not supervised[1] and supervised[2:].all()
    good = (matched.view_friction > 0) & (matched.view_friction <= 0.6)
    best = good.flatten(-2).float().mean(dim=-1).argmax(dim=-1)
    assert torch.allclose(target, -fibonacci_views(LABEL_VIEWS)[best])
    approach.raw = target * 3.0  # any positive length
    losses = stage.loss(approach, matched)
    assert float(losses["loss/view"]) < 1e-6 and float(losses["metric/view_error_deg"]) < 0.1


def test_the_frame_target_is_the_label_frame_at_its_best_angle_up_to_half_a_turn():
    stage = build("TwoVectorViewStage")
    approach, _ = stage(seeds())
    matched = dense(approach)
    goal, usable = stage.frame_target(matched)
    assert not usable[0] and not usable[1] and usable[2:].all()
    cost = torch.where(
        (matched.friction > 0) & (matched.friction <= 0.6), matched.friction, torch.inf
    )
    angle = cost.amin(dim=-1).argmin(dim=-1) * (math.pi / ANGLES)
    lattice = -fibonacci_views(LABEL_VIEWS)[approach.view_index]
    assert torch.allclose(goal[2:], approach_to_rotation(lattice, angle)[2:], atol=1e-5)
    # The half-turn copy of the target costs nothing.
    flipped = approach_to_rotation(lattice, angle + math.pi)
    approach.raw = torch.cat([flipped[..., 0], flipped[..., 1]], dim=-1)
    losses = stage.loss(approach, matched)
    assert float(losses["loss/view_frame"]) < 1e-6
    assert float(losses["metric/view_frame_error_deg"]) < 0.5
    # A mirrored crop frame (flip augmentation) is no rotation target.
    matched.crop_rotation = matched.crop_rotation * torch.tensor([1.0, 1.0, -1.0])
    assert not stage.frame_target(matched)[1].any()
    assert float(stage.loss(approach, matched)["loss/view_frame"]) == 0.0


def test_the_two_orders_of_the_gram_schmidt_step_keep_different_vectors():
    raw = torch.randn(COUNT, 6, generator=torch.Generator().manual_seed(3))
    a, b = raw[:, :3], raw[:, 3:]
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)  # noqa: E731
    _, kept_approach = build("TwoVectorViewStage", keep="approach").to_frame(raw)
    _, kept_baseline = build("TwoVectorViewStage", keep="baseline").to_frame(raw)
    assert torch.allclose(kept_approach[..., 0], unit(a), atol=1e-5)
    assert torch.allclose(kept_baseline[..., 1], unit(b), atol=1e-5)
    assert not torch.allclose(kept_baseline[..., 0], unit(a), atol=1e-3)
    assert torch.allclose(torch.linalg.det(kept_baseline), torch.ones(COUNT), atol=1e-4)


def test_misuse_of_a_regression_stage_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="unknown keep 'both'"):
        build("TwoVectorViewStage", keep="both")
    stage = build("TwoVectorViewStage", num_angle=ANGLES + 1)
    approach, _ = stage(seeds())
    with pytest.raises(ContractError, match=rf"have {ANGLES} in-plane angle bins"):
        stage.loss(approach, dense(approach))
    other = build("LatticeViewStage", num_views=5)
    with pytest.raises(ContractError, match="needs the approach of its own forward"):
        stage.loss(other(seeds())[0], dense(approach))
    # The config check compares the angle grids before anything is built.
    (message,) = stage_contract_problems(
        view={"type": "TwoVectorViewStage", "num_angle": 6},
        head={"type": "SWADHead", "num_angle": 12},
    )
    assert message.startswith(
        "TwoVectorViewStage has num_angle = 6 and SWADHead has num_angle = 12"
    )
    assert not stage_contract_problems(
        view={"type": "TwoVectorViewStage"}, head={"type": "SWADHead"}
    )
    assert not stage_contract_problems(
        view={"type": "DirectionViewStage"}, head={"type": "SWADHead", "num_angle": 7}
    )
    pytest.importorskip("roma")
    with pytest.raises(ValueError, match="unknown representation 'euler'"):
        build("RomaRotationViewStage", representation="euler")


def test_the_angular_cost_charges_a_wrong_view_by_its_angle_to_the_right_one():
    from grasppanda.models.views.angular_cost import AngularCost

    directions = -fibonacci_views(30)
    cost = AngularCost(directions)
    assert cost.state_dict() == {} and torch.allclose(
        cost.cost.diagonal(), torch.zeros(30), atol=1e-6
    )
    assert float(cost.cost.min()) >= -1e-6 and float(cost.cost.max()) <= 1 + 1e-6
    target = torch.tensor([3, 7, 11])
    sure = torch.full((3, 30), -50.0)
    sure[torch.arange(3), target] = 50.0
    assert float(cost(sure, target)) < 1e-6  # all mass on the right view
    near = (directions @ directions[3]).topk(2).indices[1]  # the nearest other view
    far = (directions @ directions[3]).argmin()
    wrong = torch.full((2, 30), -50.0)
    wrong[0, near], wrong[1, far] = 50.0, 50.0
    both = torch.tensor([3, 3])
    near_cost = cost(wrong[:1], both[:1])
    far_cost = cost(wrong[1:], both[1:])
    assert 0 < float(near_cost) < 0.1 < 0.9 < float(far_cost) <= 1.0
    expected = (1 - directions[3] @ directions[far]) / 2
    assert torch.allclose(far_cost, expected, atol=1e-5)
    # The same number as the upstream formula: expected cost under the softmax.
    logits = torch.randn(5, 30, generator=torch.Generator().manual_seed(0))
    labels = torch.tensor([0, 5, 9, 20, 29])
    matrix = (1 - directions @ directions.T) / 2
    upstream = (matrix[labels] * logits.softmax(dim=1)).sum(dim=1).mean()
    assert torch.allclose(cost(logits, labels), upstream, atol=1e-6)
    mask = torch.tensor([True, False, True, False, False])
    assert torch.allclose(cost(logits, labels, mask), cost(logits[mask], labels[mask]), atol=1e-6)
    assert float(cost(logits, labels, torch.zeros(5, dtype=torch.bool))) == 0.0
    assert float(cost.mean_error(sure, target)) < 1e-3
    with pytest.raises(ValueError, match=r"directions must be \(V, 3\)"):
        AngularCost(torch.zeros(4, 2))


def test_the_angular_weight_adds_its_term_to_the_classification_loss_only():
    stage = build("LatticeViewStage", num_views=15, target="best_view", angular_weight=2.0)
    approach, _ = stage(seeds())
    losses = stage.loss(approach, dense(approach))
    assert {
        "loss/view",
        "loss/view_angular",
        "metric/view_accuracy",
        "metric/view_error_deg",
    } == set(losses)
    target, supervised, _ = stage.targets(dense(approach))
    expected = 2.0 * stage.angular_cost(approach.logits, target, supervised)
    assert torch.allclose(losses["loss/view_angular"], expected)
    losses["loss/view_angular"].backward()
    assert stage.score.weight.grad.abs().sum() > 0
    plain = build("LatticeViewStage", num_views=15, target="best_view")
    assert "loss/view_angular" not in plain.loss(plain(seeds())[0], dense(approach))
    with pytest.raises(ValueError, match="angular_weight needs target 'best_view'"):
        build("LatticeViewStage", angular_weight=1.0)
    with pytest.raises(ValueError, match="angular_weight must not be negative"):
        build("LatticeViewStage", target="best_view", angular_weight=-1.0)


def test_the_von_mises_fisher_likelihood_rewards_confidence_only_where_it_is_right():
    stage = build("DirectionViewStage", concentration=True)
    assert stage.output.out_features == 4
    inputs = seeds()
    approach, refined = stage(inputs)
    check_approach(stage, approach, refined, inputs)
    matched = dense(approach)
    losses = stage.loss(approach, matched)
    assert set(losses) == {"loss/view", "metric/view_error_deg", "metric/view_concentration"}
    losses["loss/view"].backward()
    assert stage.output.weight.grad[3].abs().sum() > 0  # the concentration is trained
    target, _, _ = stage.direction_target(matched)

    def value(direction, raw_kappa):
        approach.raw = torch.cat([direction, torch.full((COUNT, 1), raw_kappa)], dim=-1)
        return float(stage.loss(approach, matched)["loss/view"])

    # On target, more concentration is better; off target, it is worse.
    assert value(target, 20.0) < value(target, 2.0) < value(target, -5.0)
    assert value(-target, 20.0) > value(-target, 2.0)
    assert value(target, 5.0) < value(-target, 5.0)
    # The density integrates to one: check the normaliser at one concentration.
    kappa = torch.tensor(3.0)
    cos = torch.linspace(-1, 1, 20001)
    log_c = torch.log(kappa) - math.log(2 * math.pi) - kappa - torch.log1p(-torch.exp(-2 * kappa))
    integral = torch.trapezoid(torch.exp(kappa * cos + log_c), cos) * 2 * math.pi
    assert abs(float(integral) - 1.0) < 1e-3
    other = build("DirectionViewStage")
    with pytest.raises(ContractError, match="needs the approach of its own forward"):
        stage.loss(other(inputs)[0], matched)


# ------------------------------------------------------- inside the five methods


def options_in_the_matrix():
    for name in ("LatticeViewStage", "DirectionViewStage", "TwoVectorViewStage"):
        yield name, STAGES["view"][name], None
    for name, (node, module) in BRIDGE_STAGES["view"].items():
        yield name, node, module
    lattice = STAGES["view"]["LatticeViewStage"]
    yield "best-view", {**lattice, "target": "best_view", "train_selection": "argmax"}, None
    yield "soft-argmax", {**lattice, "selection": "soft_argmax", "top_k": 3}, None
    yield "top-k", {**lattice, "train_selection": "topk", "top_k": 3}, None
    yield "hemisphere", {**lattice, "num_views": 12, "hemisphere": True}, None
    best = {**lattice, "target": "best_view", "train_selection": "argmax"}
    yield "angular-cost", {**best, "angular_weight": 1.0}, None
    yield "vmf", {**STAGES["view"]["DirectionViewStage"], "concentration": True}, None


@pytest.mark.parametrize("detector", list(METHODS))
@pytest.mark.parametrize(
    ("name", "node", "module"), list(options_in_the_matrix()), ids=lambda value: str(value)[:24]
)
def test_every_view_option_trains_and_predicts_inside_every_method(detector, name, node, module):
    """One training step and one prediction with the stages of the method's reference
    model around the option. Scale-Balanced-Grasp decodes with the objectness of a seed,
    which only its own view stage predicts: there the config is refused, and it runs with
    the decode option that does not read it."""
    if module is not None:
        pytest.importorskip(module)
    backbone = encoder("PointNet2" if detector != "GSNet" else "PointNet2Dense")
    if METHODS[detector]["proposal"] == "GraspnessFPS":
        backbone = encoder("PointNet2Dense")
    model = method(detector, backbone)
    model["view"] = dict(node)
    problems = refusal(model)
    if detector == "ScaleBalancedGrasp":
        (message,) = problems
        assert "reads the seed score 'objectness'" in message
        model["head"] = dict(STAGES["head"]["OperationHead"])
        problems = refusal(model)
    assert problems == []
    built = run(model)
    assert type(built.view).__name__ == node["type"]
