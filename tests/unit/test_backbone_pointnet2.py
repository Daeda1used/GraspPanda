"""PointNet++: published configurations, a written-out oracle and the options that differ
between its two lineages. The shared contract is checked in ``test_backbone_contract``."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch

from grasppanda.models.backbones._packed import fps_packed
from grasppanda.models.backbones.pointnet2 import PointNet2, PointNet2Dense
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointSet

SMALL = {
    "npoints": [40, 20, 10],
    "radii": [0.25, 0.45, 0.8],
    "nsamples": [6, 5, 4],
    "sa_mlps": [[8, 12], [12, 16], [16, 16]],
}


def cloud(count: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(count, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def randomise(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Random weights and batch-norm statistics, in double precision for a tight comparison."""
    generator = torch.Generator().manual_seed(seed)
    state = model.state_dict()
    for name, tensor in state.items():
        if tensor.is_floating_point():
            values = torch.rand(tensor.shape, generator=generator)
            state[name] = values + 0.5 if name.endswith("running_var") else values - 0.5
    model.load_state_dict(state)
    return model.double().eval()


# ------------------------------------------------------------------- written-out oracle


def shared_mlp(state: dict, prefix: str, x: np.ndarray) -> np.ndarray:
    """Conv 1×1 without bias, batch norm with running statistics, ReLU; ``x`` is (..., C)."""
    layer = 0
    while f"{prefix}.layer{layer}.conv.weight" in state:
        key = f"{prefix}.layer{layer}"
        x = x @ state[f"{key}.conv.weight"][:, :, 0, 0].T
        scale = state[f"{key}.bn.bn.weight"] / np.sqrt(state[f"{key}.bn.bn.running_var"] + 1e-5)
        x = (x - state[f"{key}.bn.bn.running_mean"]) * scale + state[f"{key}.bn.bn.bias"]
        x = np.maximum(x, 0.0)
        layer += 1
    return x


def pointnet2_oracle(model, coords: np.ndarray, feats: np.ndarray | None, config: dict):
    """PointNet++ as the paper describes it, one point at a time."""
    state = {key: value.numpy() for key, value in model.state_dict().items()}
    normalize, power = config["normalize_xyz"], config["interpolation_power"]
    levels = [(np.arange(len(coords)), coords, feats)]
    for index, (npoint, radius, nsample) in enumerate(
        zip(config["npoints"], config["radii"], config["nsamples"])
    ):
        _, xyz, f = levels[-1]
        chosen = [0]
        distance = np.full(len(xyz), np.inf)
        for _ in range(npoint - 1):
            distance = np.minimum(distance, ((xyz - xyz[chosen[-1]]) ** 2).sum(-1))
            chosen.append(int(distance.argmax()))
        pooled = []
        for centre in xyz[chosen]:
            hits = np.flatnonzero(((xyz - centre) ** 2).sum(-1) < radius**2)[:nsample]
            hits = np.concatenate([hits, np.full(nsample - len(hits), hits[0])])
            group = (xyz[hits] - centre) / (radius if normalize else 1.0)
            if f is not None:
                group = np.concatenate([group, f[hits]], axis=1)
            pooled.append(shared_mlp(state, f"sa{index + 1}.mlp_module", group).max(0))
        levels.append((levels[-1][0][chosen], xyz[chosen], np.stack(pooled)))
    carried = levels[-1][2]
    for index in range(len(config["fp_mlps"])):
        target = len(config["npoints"]) - 1 - index
        _, xyz, skip = levels[target]
        known = levels[target + 1][1]
        rows = []
        for point in xyz:
            distance = np.sqrt(((known - point) ** 2).sum(-1))
            nearest = np.argsort(distance, kind="stable")[:3]
            weight = 1.0 / (distance[nearest] ** power + 1e-8)
            rows.append((weight[:, None] * carried[nearest]).sum(0) / weight.sum())
        merged = np.stack(rows) if skip is None else np.concatenate([np.stack(rows), skip], axis=1)
        carried = shared_mlp(state, f"fp{index + 1}.mlp", merged)
    return levels[len(config["npoints"]) - len(config["fp_mlps"])][0], carried


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("PointNet2", {"fp_mlps": [[16, 12]], "normalize_xyz": True, "interpolation_power": 1.0}),
        (
            "PointNet2",
            {"fp_mlps": [[16], [14]], "normalize_xyz": False, "interpolation_power": 2.0},
        ),
        (
            "PointNet2",
            {"fp_mlps": [], "in_channels": 2, "normalize_xyz": True, "interpolation_power": 1.0},
        ),
        (
            "PointNet2Dense",
            {
                "fp_mlps": [[16], [12], [9]],
                "in_channels": 2,
                "normalize_xyz": False,
                "interpolation_power": 2.0,
            },
        ),
        (
            "PointNet2Dense",
            {"fp_mlps": [[16], [12], [9]], "normalize_xyz": True, "interpolation_power": 1.0},
        ),
    ],
)
def test_forward_equals_the_definition_written_out(kind, extra):
    config = {**SMALL, **extra, "fps_skip_near_origin": False}
    model = randomise(BACKBONES.build({"type": kind, **config}))
    coords = cloud(90, seed=1)
    feats = cloud(90, seed=2)[:, : model.in_channels] if model.in_channels else None
    with torch.no_grad():
        out = model(PointSet(coords), feats)
    rows, expected = pointnet2_oracle(
        model, coords.numpy(), None if feats is None else feats.numpy(), config
    )
    if model.output == "subsampled":
        assert out.points.parent_index.tolist() == rows.tolist()
    assert out.feats.shape == expected.shape
    assert np.allclose(out.feats.numpy(), expected, atol=1e-10)


# ---------------------------------------------------------------- published settings


def test_defaults_are_the_graspnet_baseline_backbone():
    model = PointNet2()
    assert (model.output, model.in_channels, model.out_channels) == ("subsampled", 0, 256)
    # Counted from the official implementation (fixture pointnet2_graspnet): 641 856.
    assert sum(p.numel() for p in model.parameters()) == 641_856
    state = model.state_dict()
    assert len(state) == 96  # 4 x 3 + 2 x 2 layers, six entries each
    # Names and shapes of the VoteNet / GraspNet-baseline state dict.
    assert state["sa1.mlp_module.layer0.conv.weight"].shape == (64, 3, 1, 1)
    assert state["sa2.mlp_module.layer0.conv.weight"].shape == (128, 131, 1, 1)
    assert state["sa4.mlp_module.layer2.bn.bn.running_var"].shape == (256,)
    assert state["fp1.mlp.layer0.conv.weight"].shape == (256, 512, 1, 1)
    assert state["fp2.mlp.layer1.bn.bn.num_batches_tracked"].shape == ()
    assert [model.sa1.radius, model.sa2.radius, model.sa3.radius, model.sa4.radius] == [
        0.04,
        0.1,
        0.2,
        0.3,
    ]
    assert [layer.nsample for layer in (model.sa1, model.sa2, model.sa3, model.sa4)] == [
        64,
        32,
        16,
        16,
    ]


def test_default_seeds_are_the_first_1024_furthest_points_of_the_scene():
    model = PointNet2().eval()
    coords = torch.rand(2600, 3, generator=torch.Generator().manual_seed(3)) + 0.3
    with torch.no_grad():
        out = model(PointSet(coords))
    assert out.feats.shape == (1024, 256)
    # Furthest point samples are nested, so sampling 2048 and then 1024 of those is the
    # same as sampling 1024 directly: upstream reads its seed indices off the first level.
    assert torch.equal(out.points.parent_index, furthest_point_sample(coords, 1024))


@pytest.mark.parametrize("skip_near_origin", [False, True])
def test_sampling_a_sample_again_is_taking_its_first_rows(skip_near_origin):
    """The shortcut every level below the first relies on, against the search it replaces."""
    generator = torch.Generator().manual_seed(8)
    first, second = (torch.rand(n, 3, generator=generator) + 0.2 for n in (150, 97))
    second[0] = 0.0  # a cloud may start at the frame origin ...
    second[5] = 0.01  # ... and hold more points the upstream kernel never selects
    coords, sizes = torch.cat([first, second]), [150, 97]
    sampled = coords[fps_packed(coords, sizes, [60, 40], skip_near_origin=skip_near_origin)]
    searched = fps_packed(sampled, [60, 40], [20, 13], skip_near_origin=skip_near_origin)
    taken = fps_packed(
        sampled, [60, 40], [20, 13], skip_near_origin=skip_near_origin, presampled=True
    )
    assert torch.equal(taken, searched)
    assert taken.tolist() == [*range(20), *range(60, 73)]
    with pytest.raises(ValueError, match="cannot sample"):
        fps_packed(sampled, [60, 40], [20, 41], presampled=True)


def test_dense_defaults_are_the_segmentation_network_of_the_paper():
    model = PointNet2Dense()
    assert (model.output, model.in_channels, model.out_channels) == ("dense", 0, 128)
    # Derived from the published layer list (pointnet2_sem_seg.py): bias-free 1x1
    # convolutions plus batch-norm scale and shift. The TensorFlow original also keeps a
    # bias per convolution (3 712 more), which batch norm makes redundant.
    widths = [
        [3, 32, 32, 64],
        [67, 64, 64, 128],
        [131, 128, 128, 256],
        [259, 256, 256, 512],
        [768, 256, 256],
        [384, 256, 256],
        [320, 256, 128],
        [128, 128, 128, 128],
    ]
    expected = sum(a * b + 2 * b for mlp in widths for a, b in pairwise(mlp))
    assert sum(p.numel() for p in model.parameters()) == expected == 945_824
    assert [model.sa1.npoint, model.sa2.npoint, model.sa3.npoint, model.sa4.npoint] == [
        1024,
        256,
        64,
        16,
    ]


# ---------------------------------------------------------------------------- options


def test_the_sampling_kernel_quirk_is_a_named_option():
    coords = cloud(120, seed=4).float() + 0.5
    coords[7] = 0.0  # the furthest point from everything else, at the frame origin
    config = {**SMALL, "fp_mlps": []}
    exact = PointNet2(**config, fps_skip_near_origin=False).eval()
    upstream = PointNet2(**config, fps_skip_near_origin=True).eval()
    with torch.no_grad():
        assert 7 in exact(PointSet(coords)).points.parent_index.tolist()
        assert 7 not in upstream(PointSet(coords)).points.parent_index.tolist()


def test_lineage_options_change_the_features():
    coords = cloud(100, seed=5).float()
    config = {**SMALL, "fp_mlps": [[16, 16]]}
    reference = PointNet2(**config).eval()
    with torch.no_grad():
        base = reference(PointSet(coords)).feats
        for option in ({"normalize_xyz": False}, {"interpolation_power": 2.0}):
            other = PointNet2(**config, **option).eval()
            other.load_state_dict(reference.state_dict())
            assert not torch.allclose(other(PointSet(coords)).feats, base, atol=1e-4), option


def test_the_number_of_propagation_levels_decides_the_output_set():
    coords = cloud(100, seed=6).float()
    for levels, rows in ((0, 10), (1, 20), (2, 40)):
        model = PointNet2(**SMALL, fp_mlps=[[16]] * levels).eval()
        with torch.no_grad():
            assert len(model(PointSet(coords)).points) == rows


@pytest.mark.parametrize(
    ("kind", "change", "message"),
    [
        (PointNet2, {"radii": [0.1, 0.2]}, "one entry per set-abstraction level"),
        (PointNet2, {"npoints": [20, 40, 10]}, "npoints must not increase with depth"),
        (PointNet2, {"fp_mlps": [[8], [8], [8]]}, "Use PointNet2Dense"),
        (PointNet2, {"fp_mlps": [[]]}, "every MLP needs at least one layer"),
        (PointNet2, {"npoints": [40, 20, 2], "fp_mlps": [[8]]}, "interpolates over 3 points"),
        (PointNet2Dense, {"fp_mlps": [[8], [8]]}, "Use PointNet2 for features on a subsample"),
    ],
)
def test_inconsistent_configurations_are_rejected(kind, change, message):
    with pytest.raises(ValueError, match=message):
        kind(**{**SMALL, "fp_mlps": [[8]], **change})
