"""Samplers: the contract every one keeps, each against its definition, and misuse."""

from __future__ import annotations

import pytest
import torch

from grasppanda.cli.components import collect
from grasppanda.models.samplers.base import Sampler
from grasppanda.ops import furthest_point_sample, generalized_furthest_point_sample
from grasppanda.registry import SAMPLERS
from grasppanda.utils.imports import MissingDependencyError

#: Sampler -> the library it calls; absent from the minimal environment.
BRIDGES = {"FpsampleFPS": "fpsample", "TorchClusterFPS": "torch_cluster"}


def cloud(count: int = 400, channels: int = 16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    coords = torch.rand(count, 3, generator=generator) + 0.5
    scores = torch.rand(count, generator=generator)
    feats = torch.randn(count, channels, generator=generator)
    return coords, scores, feats


def in_tree() -> list[str]:
    return [entry.name for entry in collect(["SAMPLERS"], in_tree_only=True)]


def build(name: str, **options) -> Sampler:
    if name in BRIDGES:
        pytest.importorskip(BRIDGES[name])
    return SAMPLERS.build({"type": name, **options})


# ------------------------------------------------------------------- the contract


@pytest.mark.parametrize("name", in_tree())
def test_a_sampler_returns_distinct_indices_of_the_candidates(name):
    sampler = build(name)
    assert isinstance(sampler, Sampler) and set(sampler.needs) <= {"scores", "feats"}
    coords, scores, feats = cloud()
    picks = sampler.sample(coords, 64, scores=scores, feats=feats)
    assert picks.dtype == torch.long and picks.ndim == 1 and picks.device == coords.device
    assert 0 < len(picks) <= 64 and len(picks.unique()) == len(picks)
    assert int(picks.min()) >= 0 and int(picks.max()) < len(coords)
    # Asked for more than there is: every candidate at most once, never an error.
    few = sampler.sample(coords[:10], 64, scores=scores[:10], feats=feats[:10])
    assert len(few) <= 10 and len(few.unique()) == len(few)
    one = sampler.sample(coords[:1], 4, scores=scores[:1], feats=feats[:1])
    assert one.tolist() == [0]
    assert type(sampler).__name__ in repr(sampler)


@pytest.mark.parametrize("name", in_tree())
def test_a_sampler_says_what_it_needs_and_refuses_a_call_without_it(name):
    sampler = build(name)
    coords, scores, feats = cloud(50)
    given = {"scores": scores, "feats": feats}
    for need in sampler.needs:
        with pytest.raises(ValueError, match=f"needs {need}"):
            sampler.sample(coords, 8, **{k: v for k, v in given.items() if k != need})
    # What it does not need may be left out.
    kept = {key: value for key, value in given.items() if key in sampler.needs}
    assert len(sampler.sample(coords, 8, **kept)) == 8


@pytest.mark.parametrize("name", sorted(BRIDGES))
def test_a_bridge_without_its_library_names_the_install_command(name, monkeypatch):
    import grasppanda.models.samplers.bridges as module

    def missing(library, *, extra, purpose):
        raise MissingDependencyError(
            f'{purpose} needs {library}: pip install "grasppanda[{extra}]"'
        )

    monkeypatch.setattr(module, "require", missing)
    with pytest.raises(Exception, match=r"pip install \"grasppanda\["):
        SAMPLERS.build({"type": name})
    assert SAMPLERS.get(name).status == "bridge" and SAMPLERS.get(name).extras


# ------------------------------------------------------------- each by its definition


def test_the_general_kernel_is_plain_fps_without_features_and_weights():
    coords, scores, _ = cloud(300)
    plain = furthest_point_sample(coords, 40)
    assert torch.equal(generalized_furthest_point_sample(coords, 40), plain)
    assert torch.equal(build("FPS").sample(coords, 40), plain)
    # gamma = 0: furthest point sampling started at the best-scored point.
    best = int(scores.argmax())
    order = torch.arange(300)
    order[0], order[best] = best, 0
    expected = order[furthest_point_sample(coords[order], 40)]
    assert torch.equal(build("ScoreWeightedFPS", gamma=0.0).sample(coords, 40, scores), expected)
    # Duplicated coordinates never yield a repeated index.
    doubled = torch.cat([coords[:5], coords[:5]])
    assert sorted(generalized_furthest_point_sample(doubled, 10).tolist()) == list(range(10))
    assert len(build("FPS").sample(doubled, 10)) == 5  # the reference kernel repeats: dropped
    with pytest.raises(ValueError, match="cannot sample 11 points from 10"):
        generalized_furthest_point_sample(doubled, 11)
    with pytest.raises(ValueError, match=r"expected \(10,\) weights"):
        generalized_furthest_point_sample(doubled, 2, weights=torch.ones(3))


def brute_force(distance: torch.Tensor, count: int, weights=None, start: int = 0) -> list[int]:
    """Furthest point sampling on a full distance matrix, as the reference kernels do."""
    picks, running = [start], distance[start].clone()
    while len(picks) < count:
        gain = running if weights is None else running * weights
        gain[picks] = -1
        picks.append(int(gain.argmax()))
        running = torch.minimum(running, distance[picks[-1]])
    return picks


def test_feature_fps_uses_the_distance_of_the_released_code_or_of_the_paper():
    coords, _, feats = cloud(200, channels=8)
    joint = torch.cat([coords, feats], dim=1)
    concat = torch.cdist(joint, joint).square()  # 3DSSD: squared L2 of [xyz, features]
    picks = build("FeatureFPS").sample(coords, 30, feats=feats)
    assert picks.tolist() == brute_force(concat, 30)
    summed = 4.0 * torch.cdist(coords, coords) + torch.cdist(feats, feats)  # Eq. 1, lambda = 4
    picks = build("FeatureFPS", metric="sum", coord_weight=4.0).sample(coords, 30, feats=feats)
    assert picks.tolist() == brute_force(summed, 30)
    # Features that are all equal leave the coordinates: D-FPS.
    flat = build("FeatureFPS").sample(coords, 30, feats=torch.ones(200, 8))
    assert torch.equal(flat, furthest_point_sample(coords, 30))
    with pytest.raises(ValueError, match="unknown metric 'cosine'"):
        build("FeatureFPS", metric="cosine")


def test_fusion_fps_is_feature_picks_then_coordinate_picks_among_the_rest():
    coords, _, feats = cloud(200, channels=8)
    picks = build("FusionFPS", feature_fraction=0.25).sample(coords, 40, feats=feats)
    by_feature = build("FeatureFPS").sample(coords, 10, feats=feats)
    assert torch.equal(picks[:10], by_feature) and len(picks.unique()) == 40
    rest = torch.tensor([i for i in range(200) if i not in set(by_feature.tolist())])
    assert torch.equal(picks[10:], rest[furthest_point_sample(coords[rest], 30)])
    assert torch.equal(
        build("FusionFPS", feature_fraction=0.0).sample(coords, 40, feats=feats),
        furthest_point_sample(coords, 40),
    )


def test_score_weighted_fps_is_the_sasa_kernel_and_prefers_high_scores():
    coords, scores, _ = cloud(300)
    squared = torch.cdist(coords, coords).square()
    start = int(scores.argmax())
    picks = build("ScoreWeightedFPS", gamma=2.0).sample(coords, 30, scores)
    assert picks.tolist() == brute_force(squared, 30, weights=scores**2.0, start=start)
    # The paper's form with gamma equals the kernel's form with 2 gamma.
    paper = build("ScoreWeightedFPS", gamma=1.0, distance="euclidean").sample(coords, 30, scores)
    assert torch.equal(paper, picks)
    plain = furthest_point_sample(coords, 30)
    assert scores[picks].mean() > scores[plain].mean() + 0.1


def test_top_k_random_and_the_two_score_orders():
    coords, scores, _ = cloud(300)
    top = build("TopK").sample(coords, 20, scores)
    assert torch.equal(top, scores.topk(20).indices)
    tied = build("TopK").sample(coords, 5, torch.ones(300))
    assert tied.tolist() == [0, 1, 2, 3, 4]  # equal scores: by index
    first, again = (build("RandomSampler", seed=7).sample(coords, 20) for _ in range(2))
    assert torch.equal(first, again) and not torch.equal(first, torch.arange(20))
    # FPS for coverage, then the best of the rest.
    mixed = build("FPSThenTopK", fps_fraction=0.5).sample(coords, 20, scores)
    assert torch.equal(mixed[:10], furthest_point_sample(coords, 10))
    left = scores.clone()
    left[mixed[:10]] = -1
    assert torch.equal(mixed[10:], left.topk(10).indices)


def test_score_nms_and_the_voxel_grid_return_what_the_scene_has():
    coords, scores, _ = cloud(400)
    picks = build("ScoreNMS", min_distance=0.15).sample(coords, 400, scores)
    assert 1 < len(picks) < 400 and picks[0] == scores.argmax()
    spacing = torch.cdist(coords[picks], coords[picks]) + torch.eye(len(picks))
    assert float(spacing.min()) > 0.15
    assert (scores[picks][:-1] >= scores[picks][1:]).all()  # in order of score
    # Every point is accepted or lies within the radius of an accepted one.
    assert float(torch.cdist(coords, coords[picks]).min(dim=1).values.max()) <= 0.15

    sampler = build("VoxelGridSampler", voxel_size=0.25)
    cells = torch.floor(coords / 0.25).long()
    occupied = len(torch.unique(cells, dim=0))
    picks = sampler.sample(coords, 400, scores)
    assert len(picks) == occupied and len(torch.unique(cells[picks], dim=0)) == occupied
    for pick in picks.tolist():  # the best-scored point of its cell
        same = (cells == cells[pick]).all(dim=1)
        assert scores[pick] == scores[same].max()
    assert len(sampler.sample(coords, 400)) == occupied  # without scores: the first of a cell
    assert len(sampler.sample(coords, 7, scores)) == 7  # more cells than seeds: thinned
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        build("VoxelGridSampler", voxel_size=0.0)


def spacing(coords: torch.Tensor, picks: torch.Tensor) -> float:
    chosen = coords[picks]
    return float((torch.cdist(chosen, chosen) + 9.0 * torch.eye(len(picks))).min())


def test_library_fps_agrees_with_the_reference_or_spreads_as_well():
    coords, _, _ = cloud(500)
    plain = furthest_point_sample(coords, 50)
    for name, options, same in (
        ("FpsampleFPS", {"method": "vanilla"}, True),
        ("TorchClusterFPS", {}, True),
        # The bucket methods of fpsample 1.0.2 do not begin at the start index.
        ("FpsampleFPS", {"method": "bucket", "height": 3}, False),
        ("FpsampleFPS", {"method": "bucket_kdtree"}, False),
    ):
        pytest.importorskip(BRIDGES[name])
        sampler = SAMPLERS.build({"type": name, **options})
        picks = sampler.sample(coords, 50)
        assert torch.equal(picks, sampler.sample(coords, 50)), (name, options)  # repeatable
        if same:
            assert torch.equal(picks, plain), (name, options)
        assert spacing(coords, picks) > 0.9 * spacing(coords, plain), (name, options)
    # A tree higher than the frame allows is lowered, not an error.
    tall = SAMPLERS.build({"type": "FpsampleFPS", "method": "bucket", "height": 9})
    assert len(tall.sample(coords[:20], 8)) == 8
    with pytest.raises(ValueError, match="unknown method 'fast'"):
        SAMPLERS.build({"type": "FpsampleFPS", "method": "fast"})
