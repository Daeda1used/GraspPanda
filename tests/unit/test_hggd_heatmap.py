"""HGGD's heat-map stage: decoding rules, dense targets and losses, against plain loops."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.detectors.hggd import HeatmapStage, ImagePredictions, ProposalOptions
from grasppanda.models.detectors.hggd.heatmap import focal_loss, gaussian_blur
from grasppanda.structures import ContractError

U, V = 64, 40  # network image (width, height); angle maps are 8 x 5


def stage(**options) -> HeatmapStage:
    return HeatmapStage(ProposalOptions(**options))


def maps(heat: torch.Tensor, seed: int = 0) -> ImagePredictions:
    """Activated maps with the given ``(U, V)`` heat and random angle maps."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 6, U // 8, V // 8)
    return ImagePredictions(
        heatmap=heat[None],
        angle=torch.rand(shape, generator=generator),
        angle_offset=torch.rand(shape, generator=generator) - 0.5,
        depth_offset=torch.rand(shape, generator=generator) - 0.5,
        width_offset=2 * torch.rand(shape, generator=generator) - 1,
        features=torch.zeros(1, 4, U // 8, V // 8),
    )


# ------------------------------------------------------------------------- decoding


def test_blur_is_a_normalised_gaussian_with_reflected_borders():
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(2, 12, 9, generator=generator)
    blurred = gaussian_blur(image, 5, 1.5)
    offsets = np.arange(-2, 3)
    line = np.exp(-0.5 * (offsets / 1.5) ** 2)
    line /= line.sum()
    padded = np.pad(image.numpy(), ((0, 0), (2, 2), (2, 2)), mode="reflect")
    expected = np.zeros_like(image.numpy())
    for i in range(12):
        for j in range(9):
            window = padded[:, i : i + 5, j : j + 5]
            expected[:, i, j] = (window * np.outer(line, line)).sum(axis=(1, 2))
    assert np.abs(blurred.numpy() - expected).max() < 1e-6
    assert torch.allclose(gaussian_blur(torch.ones(1, 12, 9), 9, 10.0), torch.ones(1, 12, 9))
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        gaussian_blur(image, 4, 1.0)


def brute_candidates(heat: np.ndarray, count: int, grid: int) -> np.ndarray:
    """The reference rule, cell by cell."""
    size_u, size_v = heat.shape
    cells = heat[: size_u // grid * grid, : size_v // grid * grid]
    cells = cells.reshape(size_u // grid, grid, size_v // grid, grid).max(axis=(1, 3))
    order = np.argsort(-cells.reshape(-1), kind="stable")[:count]
    pixels = []
    for index in order:
        cu, cv = divmod(int(index), cells.shape[1])
        top, bottom = max(0, cu * grid - 1), min((cu + 1) * grid + 1, size_u - 1)
        left, right = max(0, cv * grid - 1), min((cv + 1) * grid + 1, size_v - 1)
        window = heat[top:bottom, left:right]
        best = int(np.argmax(window))
        pixels.append((top + best // window.shape[1], left + best % window.shape[1]))
    return np.array(pixels)


def test_candidates_are_the_brightest_pixels_of_the_best_grown_cells():
    generator = torch.Generator().manual_seed(1)
    heat = torch.rand(U, V, generator=generator)
    mine = stage().candidates(heat, 25)
    assert np.array_equal(mine.numpy(), brute_candidates(heat.numpy(), 25, 8))
    everything = stage().candidates(heat, 10_000)  # more than there are cells
    assert len(everything) == (U // 8) * (V // 8)


def test_the_last_row_and_column_of_the_image_are_never_candidates():
    """Reference quirk: the grown window of a border cell ends one pixel early."""
    heat = torch.zeros(U, V) + 1e-3
    heat[U - 1, 10] = 1.0  # the brightest pixel of the image, in the last column of pixels
    heat[20, V - 1] = 0.9
    pixels = stage().candidates(heat, 40)
    assert not (pixels[:, 0] == U - 1).any() and not (pixels[:, 1] == V - 1).any()
    heat[U - 2, 10] = 0.8  # one pixel further in it is found
    assert (stage().candidates(heat, 40) == torch.tensor([U - 2, 10])).all(dim=1).any()


def test_selection_decodes_the_best_angle_class_of_the_pixels_cell():
    heat = torch.full((U, V), 1e-3)
    heat[19, 12] = 0.7
    activated = maps(heat)
    grasps = stage().select(activated, 0, training=False)
    assert len(grasps) == 1 and grasps.pixel.tolist() == [[19, 12]]
    cell = (19 // 8, 12 // 8)
    best = int(activated.angle[0, :, cell[0], cell[1]].argmax())
    at = (0, best, cell[0], cell[1])
    theta = math.pi / 6 * (best + float(activated.angle_offset[at]) + 0.5) - math.pi / 2
    assert grasps.theta.dtype == torch.float64 and abs(float(grasps.theta) - theta) < 1e-12
    assert -math.pi / 2 <= float(grasps.theta) <= math.pi / 2
    width = 50.0 * float(torch.exp(activated.width_offset[at]))
    assert abs(float(grasps.width) - width) < 1e-12
    assert abs(float(grasps.depth_offset) - float(activated.depth_offset[at]) * 0.04) < 1e-12
    assert abs(float(grasps.depth_offset)) <= 0.02 + 1e-12  # metres
    assert float(grasps.score) == float(heat[19, 12]) and grasps.centre is None


def test_score_threshold_applies_at_inference_and_not_in_training():
    heat = torch.full((U, V), 0.005)
    heat[19, 12] = 0.7
    decoder = stage()
    assert len(decoder.select(maps(heat), 0, training=False)) == 1
    trained = decoder.select(maps(heat), 0, training=True)
    assert len(trained) > 1  # reference: no heat threshold while training
    assert len(trained) <= decoder.train_max_proposals
    assert (
        len(stage(max_proposals=3, score_threshold=0.0).select(maps(heat), 0, training=False)) == 3
    )


def test_greedy_thinning_needs_distance_and_angle_to_be_close():
    decoder = stage(min_distance=8.0, min_angle=30.0)
    pixel = np.array([[10, 10], [13, 10], [13, 10], [30, 10], [10, 15]])
    theta = np.radians([0.0, 10.0, 45.0, 0.0, 85.0])
    # 1: near and similar angle -> dropped. 2: near, other angle -> kept. 3: far -> kept.
    # 4: near, and 85 degrees is 5 degrees from -90 = +90 ... but 85 from 0: kept.
    assert decoder._thin(pixel, theta, 10) == [0, 2, 3, 4]
    wrapped = decoder._thin(np.array([[10, 10], [11, 10]]), np.radians([-80.0, 80.0]), 10)
    assert wrapped == [0]  # -80 and 80 degrees are 20 degrees apart for a symmetric gripper
    assert decoder._thin(pixel, theta, 2) == [0, 2]  # the budget stops the scan
    assert stage(min_distance=0.0)._thin(pixel, theta, 10) == [0, 1, 2, 3, 4]


def test_recorded_candidates_replace_the_search_and_must_cover_the_batch():
    heat = torch.rand(U, V, generator=torch.Generator().manual_seed(2))
    raw = maps(torch.logit(heat.clamp(0.01, 0.99)))
    decoder = stage(blur_kernel=1, score_threshold=0.0, min_distance=0.0)
    forced = torch.tensor([[5, 5], [40, 20]])
    grasps, activated = decoder(raw, training=False, candidates=[forced])
    assert grasps.pixel.tolist() == forced.tolist() and grasps.frame.tolist() == [0, 0]
    assert float(activated.angle.min()) > 0 and float(activated.angle.max()) < 1
    assert float(activated.width_offset.abs().max()) <= 1.0
    with pytest.raises(ContractError, match="recorded candidates for 2 frame"):
        decoder(raw, candidates=[forced, forced])


def test_misconfiguration_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        stage(stride=0)
    with pytest.raises(ValueError, match="blur_kernel must be odd"):
        stage(blur_kernel=8)
    with pytest.raises(TypeError, match="has no option"):
        HeatmapStage({"strides": 8})


# -------------------------------------------------------------------------- targets


def random_labels(count: int, seed: int, width: int = 2 * U, height: int = 2 * V) -> dict:
    generator = np.random.default_rng(seed)
    return {
        "centers": np.stack(
            [generator.uniform(4, width - 4, count), generator.uniform(4, height - 4, count)],
            axis=1,
        ),
        "theta": generator.uniform(-math.pi, math.pi, count),
        "width": generator.uniform(20, 120, count),
        "height": generator.uniform(10, 30, count),
        "depth": generator.uniform(0.4, 0.6, count),
        "score": generator.choice([0.9, 1.0, 0.5], count),
        "gamma": np.zeros(count),
        "beta": np.zeros(count),
    }


def loop_targets(labels: dict, depth: np.ndarray, options: ProposalOptions) -> dict:
    """The dense targets written as the reference's sequence of loops."""
    scale = depth.shape[1] // U
    keep = labels["score"] >= options.label_min_score
    theta = labels["theta"][keep].copy()
    outside = (theta > math.pi / 2) | (theta < -math.pi / 2)
    theta[outside] = (theta[outside] + math.pi / 2) % math.pi - math.pi / 2
    rectangles = []
    for centre, angle, jaw, finger in zip(
        labels["centers"][keep], theta, labels["width"][keep], labels["height"][keep]
    ):
        axis = jaw / 2 * np.array([np.cos(angle), -np.sin(angle)])
        normal = np.array([-axis[1], axis[0]])
        normal = normal / np.linalg.norm(normal) * finger / 2
        corners = [
            centre + normal + axis,
            centre + normal - axis,
            centre - normal - axis,
            centre - normal + axis,
        ]
        rectangles.append(np.array(corners, dtype=np.int32) / scale)
    heatmap = np.zeros((U, V))
    radius = options.peak_radius
    sigma = (2 * radius + 1) / 6
    drawn = []
    for corners in rectangles:
        u, v = corners.mean(0).astype(np.int32)
        if any(math.hypot(u - a, v - b) < options.peak_min_distance for a, b in drawn):
            continue
        drawn.append((u, v))
        for i in range(max(u - radius, 0), min(u + radius + 1, U)):
            for j in range(max(v - radius, 0), min(v + radius + 1, V)):
                value = math.exp(-((i - u) ** 2 + (j - v) ** 2) / (2 * sigma * sigma))
                heatmap[i, j] = max(heatmap[i, j], value)
    shape = (options.num_angles, U // 8, V // 8)
    count, angle_sum, depth_sum, width_sum = (np.zeros(shape) for _ in range(4))
    millimetres = depth.astype(np.float32) * np.float32(1000.0)
    step = math.pi / options.num_angles
    for corners, angle, surface in zip(rectangles, theta, labels["depth"][keep] * 1000.0):
        centre = corners.mean(0).astype(np.int32)
        u, v = centre // 8
        if not (0 <= u < shape[1] and 0 <= v < shape[2]):
            continue
        x, y = centre * scale
        window = millimetres[
            max(0, y - 2) : min(depth.shape[0] - 1, y + 2),
            max(0, x - 2) : min(depth.shape[1] - 1, x + 2),
        ]
        observed = np.median(window) if window.size else 0.0
        offset = None
        if observed > 0:
            offset = np.clip((surface - observed) / (options.anchor_depth * 2000.0), -0.5, 0.5)
        jaw = np.sqrt(np.sum(np.square(corners[0] - corners[1])))
        clipped = np.clip(angle, -math.pi / 2 + 1e-8, math.pi / 2 - 1e-8)
        bin_index, remainder = divmod(clipped + math.pi / 2, step)
        at = (int(bin_index), u, v)
        count[at] += 1
        angle_sum[at] += remainder / step - 0.5
        width_sum[at] += np.log(jaw / options.anchor_width)
        if offset is not None:
            depth_sum[at] += offset
        elif count[at] > 1 and depth_sum[at] > 0:
            depth_sum[at] *= count[at] / (count[at] - 1)
    divisor = count + (count == 0)
    return {
        "heatmap": heatmap,
        "angle": 2 / (1 + np.exp(-count)) - 1,
        "angle_offset": angle_sum / divisor,
        "depth_offset": depth_sum / divisor,
        "width_offset": width_sum / divisor,
    }


@pytest.mark.parametrize("holes", [False, True])
def test_dense_targets_equal_the_reference_loops(holes):
    encoder = stage()
    generator = np.random.default_rng(5)
    depth = generator.uniform(0.45, 0.55, size=(2 * V, 2 * U)).astype(np.float32)
    if holes:  # labels over pixels without depth take the order-dependent branch
        depth[generator.random(depth.shape) < 0.55] = 0.0
    labels = random_labels(400, seed=6)
    mine = encoder.encode_frame(labels, depth, (U, V))
    reference = loop_targets(labels, depth, encoder.options)
    for name, expected in reference.items():
        assert mine[name].shape == expected.shape, name
        assert np.abs(mine[name] - expected).max() < 1e-12, name
    assert mine["heatmap"].max() == 1.0 and (mine["angle"] > 0).sum() > 50
    assert (mine["angle"] >= 0.5).sum() > 0  # some cells hold two labels or more
    if holes:
        assert (mine["depth_offset"] == 0).sum() > (reference["angle"] == 0).sum()


def test_targets_ignore_weak_labels_and_subsample_many():
    encoder = stage(max_labels=50)
    depth = np.full((2 * V, 2 * U), 0.5, dtype=np.float32)
    labels = random_labels(300, seed=7)
    weak = dict(labels, score=np.full(300, 0.5))
    assert encoder.encode_frame(weak, depth, (U, V))["heatmap"].max() == 0.0
    labels["score"][:] = 1.0
    subset = np.arange(50)
    some = encoder.encode_frame(labels, depth, (U, V), subset=subset)
    first = {name: value[:50] for name, value in labels.items()}
    alone = encoder.encode_frame(first, depth, (U, V))
    assert all(np.array_equal(some[name], alone[name]) for name in some)
    drawn = encoder.encode_frame(labels, depth, (U, V))  # a random subset of 50
    assert 0 < (drawn["angle"] > 0).sum() <= 50


def test_batch_targets_are_float32_tensors_on_the_depth_device():
    encoder = stage()
    depth = torch.full((2, 2 * V, 2 * U), 0.5)
    labels = [
        {k: torch.from_numpy(v) for k, v in random_labels(30, seed=s).items()} for s in (1, 2)
    ]
    targets = encoder.encode(labels, depth, (U, V))
    assert tuple(targets.heatmap.shape) == (2, U, V) and targets.heatmap.dtype == torch.float32
    assert tuple(targets.angle.shape) == (2, 6, U // 8, V // 8)
    assert not torch.equal(targets.heatmap[0], targets.heatmap[1])
    with pytest.raises(ContractError, match=r"labels of 1 frame\(s\) for a batch of 2"):
        encoder.encode(labels[:1], depth, (U, V))
    with pytest.raises(ContractError, match="not an integer reduction"):
        encoder.encode_frame(random_labels(3, 0), np.zeros((70, 100), np.float32), (U, V))


# --------------------------------------------------------------------------- losses


def test_focal_loss_matches_its_definition():
    logits = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
    target = torch.tensor([[1.0, 0.0, 0.6, 0.995]])
    p = torch.sigmoid(logits)
    positive = 0.5 * torch.log(p) * (1 - p) ** 2
    negative = 0.5 * torch.log(1 - p) * p**2
    plain = -(positive[0, [0, 3]].sum() + negative[0, [1, 2]].sum()) / 2
    assert torch.isclose(focal_loss(logits, target, positive_threshold=0.99), plain)
    spared = -(positive[0, [0, 3]].sum() + negative[0, 1] + negative[0, 2] * 0.4**4) / 2
    assert torch.isclose(
        focal_loss(logits, target, positive_threshold=0.99, suppress_near_positives=True), spared
    )
    none = focal_loss(logits, torch.zeros_like(target), positive_threshold=0.99)
    assert torch.isclose(none, -negative.sum())  # no positive: the sum, undivided
    weighted = focal_loss(logits, target, positive_threshold=0.5, alpha=0.25)
    assert torch.isfinite(weighted) and not torch.isclose(weighted, plain)
    extreme = focal_loss(
        torch.tensor([[80.0, -80.0]]), torch.tensor([[0.0, 1.0]]), positive_threshold=0.99
    )
    assert torch.isfinite(extreme)  # probabilities are kept away from 0 and 1


def test_loss_terms_are_weighted_and_supervise_only_labelled_cells():
    encoder = stage(heatmap_weight=1.0, offset_weight=5.0, angle_weight=1.0)
    depth = torch.full((1, 2 * V, 2 * U), 0.5)
    labels = [{k: torch.from_numpy(v) for k, v in random_labels(200, seed=3).items()}]
    targets = encoder.encode(labels, depth, (U, V))
    generator = torch.Generator().manual_seed(0)
    raw = ImagePredictions(
        heatmap=torch.randn(1, U, V, generator=generator, requires_grad=True),
        angle=torch.randn(1, 6, U // 8, V // 8, generator=generator, requires_grad=True),
        angle_offset=torch.randn(1, 6, U // 8, V // 8, generator=generator, requires_grad=True),
        depth_offset=torch.randn(1, 6, U // 8, V // 8, generator=generator, requires_grad=True),
        width_offset=torch.randn(1, 6, U // 8, V // 8, generator=generator, requires_grad=True),
        features=torch.zeros(1, 4, U // 8, V // 8),
    )
    losses = encoder.loss(raw, targets)
    assert set(losses) == {"loss/heatmap", "loss/angle", "loss/offsets"}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    losses["loss/offsets"].backward()
    unlabelled = targets.angle[0] == 0
    assert not raw.width_offset.grad[0][unlabelled].any()  # no label, no gradient
    assert raw.width_offset.grad[0][~unlabelled].any()
    saturated = raw.width_offset.detach().abs() > 0.5  # clamped in the loss: no gradient either
    assert not raw.width_offset.grad[saturated].any()
    heavier = stage(offset_weight=10.0).loss(raw, targets)["loss/offsets"]
    assert torch.isclose(heavier, 2 * losses["loss/offsets"])
    # one label in a cell is not enough to make its angle class a positive
    single = (targets.angle > 0) & (targets.angle < 0.5)
    assert (
        single.any() and abs(float(targets.angle[single][0]) - (2 / (1 + math.exp(-1)) - 1)) < 1e-6
    )
    with pytest.raises(ContractError, match="do not match their targets"):
        encoder.loss(
            raw, encoder.encode(labels, torch.full((1, 4 * V, 4 * U), 0.5), (2 * U, 2 * V))
        )
