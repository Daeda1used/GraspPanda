"""HGGD's rotation anchors: decoding, label matching, targets, losses and the anchor shift."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grasppanda.models.detectors.hggd import AnchorHead, AnchorOptions, PartialGraspSet, RegionSet
from grasppanda.models.detectors.hggd.anchor_head import euler_quaternion, rotation_similarity
from grasppanda.structures import ContractError, Gripper, SampleKey

KEYS = [SampleKey("graspnet1b", 100, "realsense", f) for f in range(2)]
SQUARE = torch.tensor([[[100.0, 0.0, 63.5], [0.0, 100.0, 39.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
OBLONG = SQUARE.clone()
OBLONG[:, 1, 1] = 103.0


def head(**options) -> AnchorHead:
    return AnchorHead(AnchorOptions(**options))


def regions(frames: list[int], theta: float = 0.3, width: float = 20.0) -> RegionSet:
    count = len(frames)
    generator = torch.Generator().manual_seed(0)
    centre = torch.tensor([[0.02, -0.01, 0.5]], dtype=torch.float64).repeat(count, 1)
    centre[:, 0] += torch.arange(count) * 0.05
    grasps = PartialGraspSet(
        frame=torch.tensor(frames, dtype=torch.long),
        pixel=torch.zeros(count, 2, dtype=torch.long),
        theta=torch.full((count,), theta, dtype=torch.float64),
        width=torch.full((count,), width, dtype=torch.float64),
        depth_offset=torch.zeros(count, dtype=torch.float64),
        score=torch.ones(count, dtype=torch.float64),
        centre=centre,
        num_frames=2,
    )
    return RegionSet(
        points=torch.randn(count, 8, 5, generator=generator),
        grasps=grasps,
        kept=torch.ones(count, dtype=torch.bool),
        support=torch.full((count,), 8),
    )


def rotation(axis: str, angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        {
            "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
            "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
            "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        }[axis],
        dtype=torch.float64,
    )


# ------------------------------------------------------------------------- anchors


def test_anchors_start_evenly_spaced_and_index_gamma_first():
    anchors = head()
    expected = torch.tensor([-6, -4, -2, 0, 2, 4, 6], dtype=torch.float64) / 7
    assert torch.allclose(anchors.gamma, expected) and torch.allclose(anchors.beta, expected)
    assert anchors.gamma.dtype == torch.float64 and anchors.num_classes == 49
    assert {"gamma", "beta"} <= set(anchors.state_dict())  # they travel with the checkpoint
    angles = anchors.anchor_angles()
    assert tuple(angles.shape) == (49, 2)
    assert torch.allclose(angles[9], torch.tensor([expected[2], expected[1]]) * math.pi / 2)
    with pytest.raises(ValueError, match="top_k must be between 1 and"):
        head(num_anchors=2, top_k=5)
    with pytest.raises(ValueError, match="unknown frame 'camera'"):
        head(frame="camera")
    with pytest.raises(TypeError, match="has no option"):
        AnchorHead({"anchors": 7})


# ------------------------------------------------------------------------ decoding


def one_hot_logits(count: int, anchor: int) -> torch.Tensor:
    logits = torch.full((count, 49), -6.0)
    logits[:, anchor] = 4.0
    return logits


def test_decoded_rotation_is_the_euler_factorisation():
    """R = R0(theta) Rz(gamma) Ry(beta), with R0's approach axis on the optical axis."""
    anchors = head(top_k=1, frame="orthonormal")
    theta, anchor = 0.3, 4 + 7 * 5  # gamma index 4, beta index 5
    (group, other) = anchors.decode(
        one_hot_logits(1, anchor), torch.zeros(1, 49, 3), regions([0], theta), SQUARE, 2.0, KEYS
    )
    assert len(group) == 1 and len(other) == 0 and other.key == KEYS[1]
    gamma, beta = (float(v) for v in anchors.anchor_angles()[anchor])
    base = torch.tensor(
        [
            [0, -math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), -math.cos(theta)],
            [1, 0, 0],
        ],
        dtype=torch.float64,
    )
    expected = base @ rotation("z", gamma) @ rotation("y", beta)
    assert torch.allclose(group.rotation[0], expected, atol=1e-12)
    assert torch.allclose(
        group.rotation[0].T @ group.rotation[0], torch.eye(3, dtype=torch.float64), atol=1e-12
    )
    assert float(torch.linalg.det(group.rotation[0])) == pytest.approx(1.0)
    group.validate()
    # in-plane: approach along +z of the camera, closing axis in the image plane
    flat = anchors.decode(
        one_hot_logits(1, 3 + 7 * 3), torch.zeros(1, 49, 3), regions([0], theta), SQUARE, 2.0, KEYS
    )[0]
    assert torch.allclose(
        flat.rotation[0, :, 0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-12
    )
    assert abs(float(flat.rotation[0, 2, 1])) < 1e-12


def test_decoded_grasp_fields_follow_the_partial_grasp():
    anchors = head(top_k=1)
    offsets = torch.zeros(1, 49, 3)
    offsets[0, 24] = torch.tensor([0.5, -3.0, 0.25])  # clipped to [-1, 1], then times 0.02 m
    (group, _) = anchors.decode(
        one_hot_logits(1, 24), offsets, regions([0], 0.0, 20.0), SQUARE, 2.0, KEYS
    )
    centre = torch.tensor([0.02 + 0.01, -0.01 - 0.02, 0.5 + 0.005], dtype=torch.float64)
    assert torch.allclose(group.translation[0], centre, atol=1e-9)
    z = float(centre[2])
    assert float(group.width) == pytest.approx(z * 40 / 100.0)  # 20 px of the network image
    assert float(group.height) == pytest.approx(z * 20 / 100.0)  # 20 sensor pixels
    assert float(group.depth) == 0.02 and int(group.object_id) == -1
    assert float(group.score) == pytest.approx(float(torch.sigmoid(torch.tensor(4.0))))
    assert group.score_kind == "anchor_probability" and group.key == KEYS[0]
    assert group.gripper == Gripper() and group.frame == "camera"


def test_reference_frame_is_orthogonal_only_for_square_pixels():
    reference = head(top_k=1, frame="pixel_keypoints")
    exact = head(top_k=1, frame="orthonormal")
    arguments = (one_hot_logits(1, 24), torch.zeros(1, 49, 3), regions([0], 0.6))
    square = reference.decode(*arguments, SQUARE, 2.0, KEYS)[0]
    assert torch.allclose(
        square.rotation, exact.decode(*arguments, SQUARE, 2.0, KEYS)[0].rotation, atol=1e-7
    )
    oblong = reference.decode(*arguments, OBLONG, 2.0, KEYS)[0]
    gram = oblong.rotation[0].T @ oblong.rotation[0]
    deviation = float((gram - torch.eye(3, dtype=torch.float64)).abs().max())
    assert 1e-3 < deviation < 0.1  # fy / fx = 1.03: the in-plane axes are visibly not orthogonal
    with pytest.raises(ContractError, match="not orthonormal"):
        oblong.validate()
    exact.decode(*arguments, OBLONG, 2.0, KEYS)[0].validate()
    # the head knows how far its reference frame may be off for a camera, and no further
    bound = reference.rotation_tolerance(OBLONG[0])
    assert deviation < bound < 2 * (103.0**2 - 100.0**2) / (103.0**2 + 100.0**2)
    oblong.validate(atol=bound)
    assert reference.rotation_tolerance(SQUARE[0]) == exact.rotation_tolerance(OBLONG[0]) == 1e-4


def test_top_k_anchors_per_region_and_steep_gammas_are_dropped():
    anchors = head(top_k=10, max_gamma=60.0)
    with torch.no_grad():
        anchors.gamma[0], anchors.gamma[6] = -0.7, 0.65  # 63 degrees: never decoded; 58.5: kept
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(5, 49, generator=generator)
    groups = anchors.decode(
        logits, torch.zeros(5, 49, 3), regions([0, 0, 0, 1, 1]), SQUARE, 2.0, KEYS
    )
    top = torch.topk(logits, 10, dim=1).indices
    steep = top % 7 == 0
    assert [len(g) for g in groups] == [int((~steep[:3]).sum()), int((~steep[3:]).sum())]
    scores = torch.sigmoid(torch.topk(logits, 10, dim=1).values)
    assert torch.allclose(groups[0].score, scores[:3][~steep[:3]].double())  # region by region
    everything = head(top_k=49, max_gamma=90.0).decode(
        logits, torch.zeros(5, 49, 3), regions([0, 0, 0, 1, 1]), SQUARE, 2.0, KEYS
    )
    assert [len(g) for g in everything] == [3 * 49, 2 * 49]


def test_no_region_decodes_to_explicit_empty_groups():
    anchors = head()
    groups = anchors.decode(
        torch.zeros(0, 49), torch.zeros(0, 49, 3), regions([]), SQUARE, 2.0, KEYS
    )
    assert [len(g) for g in groups] == [0, 0] and [g.key for g in groups] == KEYS
    assert all(g.score_kind == "anchor_probability" for g in groups)
    with pytest.raises(ContractError, match="for 2 regions and 49 anchors"):
        anchors.decode(
            torch.zeros(3, 49), torch.zeros(3, 49, 3), regions([0, 1]), SQUARE, 2.0, KEYS
        )


# ------------------------------------------------------------------------- targets


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def test_euler_quaternion_is_the_product_of_axis_rotations():
    generator = torch.Generator().manual_seed(2)
    euler = (torch.rand(20, 3, generator=generator, dtype=torch.float64) - 0.5) * 6
    mine = euler_quaternion(euler).numpy()
    for row, (a, b, c) in enumerate(euler.tolist()):
        qx = np.array([math.cos(a / 2), math.sin(a / 2), 0, 0])
        qy = np.array([math.cos(b / 2), 0, math.sin(b / 2), 0])
        qz = np.array([math.cos(c / 2), 0, 0, math.sin(c / 2)])
        assert np.allclose(
            mine[row], quaternion_product(quaternion_product(qx, qy), qz), atol=1e-14
        )
    assert np.allclose(np.linalg.norm(mine, axis=1), 1.0)


def test_similarity_knows_the_gripper_symmetry():
    triple = torch.tensor([[0.4, 0.2, -0.3]], dtype=torch.float64)
    mirrored = torch.tensor([[0.4 + math.pi, -0.2, 0.3]], dtype=torch.float64)
    assert float(rotation_similarity(triple, triple)) == pytest.approx(1.0)
    assert float(rotation_similarity(triple, mirrored)) == pytest.approx(1.0)
    other = torch.tensor([[0.4, 0.9, -0.3]], dtype=torch.float64)
    assert float(rotation_similarity(triple, other)) == pytest.approx(math.cos(0.35), abs=1e-3)
    table = rotation_similarity(
        torch.randn(7, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    )
    assert (
        tuple(table.shape) == (7, 4) and float(table.min()) >= 0 and float(table.max()) <= 1 + 1e-12
    )


def labels_at(centres: torch.Tensor, intrinsics: torch.Tensor, **angles: float) -> dict:
    """Per-view labels whose 3-D centres are ``centres``."""
    count = len(centres)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    pixel = torch.stack(
        [centres[:, 0] / centres[:, 2] * fx + cx, centres[:, 1] / centres[:, 2] * fy + cy], 1
    )
    return {
        "centers": pixel.double(),
        "depth": centres[:, 2].double(),
        "theta": torch.full((count,), angles.get("theta", 0.3), dtype=torch.float64),
        "gamma": torch.full((count,), angles.get("gamma", 0.0), dtype=torch.float64),
        "beta": torch.full((count,), angles.get("beta", 0.0), dtype=torch.float64),
        "width": torch.full((count,), 40.0, dtype=torch.float64),
        "score": torch.ones(count, dtype=torch.float64),
    }


def test_labels_within_the_radius_are_matched_nearest_first():
    anchors = head(match_radius=0.02, max_matches=3)
    found = regions([0, 1])
    centre = found.grasps.centre
    near = centre[0] + torch.tensor(
        [[0.004, 0, 0], [0.019, 0, 0], [0.001, 0, 0], [0.012, 0, 0], [0.021, 0, 0]],
        dtype=torch.float64,
    )
    labels = [labels_at(near, SQUARE[0].double()), labels_at(centre[1:2] + 0.5, SQUARE[1].double())]
    matched = anchors.match(found, labels, SQUARE)
    assert [len(rows) for rows in matched] == [3, 0]  # four within 2 cm, the nearest three kept
    assert matched[0].dtype == torch.float32 and tuple(matched[0].shape) == (3, 8)
    assert torch.allclose(matched[0][:, 0], torch.tensor([0.001, 0.004, 0.012]), atol=1e-6)
    assert torch.allclose(
        matched[0][:, 3], torch.tensor(0.3)
    )  # theta, gamma, beta, width, score follow
    assert torch.allclose(matched[0][:, 6], torch.tensor(40.0))


def test_targets_are_the_best_similarity_and_its_labels_offset():
    anchors = head()
    theta = torch.tensor([0.3, 0.3], dtype=torch.float64)
    gamma, beta = (float(v) for v in anchors.anchor_angles()[30])
    exact = torch.tensor(
        [
            [0.001, 0.002, 0.003, 0.3, gamma, beta, 40.0, 1.0],
            [0.01, 0.0, 0.0, 0.3 + math.pi, -gamma, -beta, 40.0, 1.0],
            [0.0, 0.01, 0.0, 1.2, 0.5, -0.9, 40.0, 1.0],
        ]
    )
    similarity, offset = anchors.encode(theta, [exact, exact[:0]])
    assert tuple(similarity.shape) == (2, 49) and tuple(offset.shape) == (2, 49, 3)
    assert float(similarity[0, 30]) == pytest.approx(1.0, abs=1e-6)
    assert int(similarity[0].argmax()) == 30
    assert torch.allclose(offset[0, 30], exact[0, :3].double())  # the first of the two equal labels
    assert not similarity[1].any() and not offset[1].any()  # no label: all negatives
    assert int((similarity[0] > 0.99).sum()) >= 1
    table = rotation_similarity(
        torch.cat(
            [torch.full((49, 1), 0.3), anchors.anchor_angles().float().double()], dim=1
        ).double(),
        exact[:, 3:6].double(),
    )
    assert torch.allclose(similarity[0], table.amax(dim=1), atol=1e-6)


# -------------------------------------------------------------------------- losses


def test_loss_supervises_offsets_of_positive_anchors_only():
    anchors = head(offset_weight=1.0)
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(4, 49, generator=generator, requires_grad=True)
    offsets = torch.randn(4, 49, 3, generator=generator, requires_grad=True)
    similarity = torch.rand(4, 49, generator=generator, dtype=torch.float64) * 0.9
    similarity[0, 5] = similarity[2, 40] = 0.995
    target = torch.randn(4, 49, 3, generator=generator, dtype=torch.float64) * 0.01
    losses = anchors.loss(logits, offsets, similarity, target)
    assert set(losses) == {"loss/anchor", "loss/anchor_offset"}
    losses["loss/anchor_offset"].backward()
    touched = offsets.grad.abs().sum(dim=-1) > 0
    assert touched.nonzero().tolist() == [[0, 5], [2, 40]]
    expected = (
        sum(
            torch.nn.functional.smooth_l1_loss(
                offsets[i, j], (target[i, j] / 0.02).float(), reduction="sum"
            )
            for i, j in ((0, 5), (2, 40))
        )
        / 2
    )
    assert torch.isclose(losses["loss/anchor_offset"], expected, atol=1e-6)
    heavier = head(offset_weight=5.0).loss(logits, offsets, similarity, target)
    assert torch.isclose(heavier["loss/anchor_offset"], 5 * losses["loss/anchor_offset"])
    empty = anchors.loss(torch.zeros(0, 49), torch.zeros(0, 49, 3), similarity[:0], target[:0])
    assert all(float(value) == 0.0 for value in empty.values())
    with pytest.raises(ContractError, match="do not match their targets"):
        anchors.loss(logits, offsets, similarity[:2], target)


# --------------------------------------------------------------------- anchor shift


def loop_shift(anchors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    fraction = np.clip((angles.astype(np.float32) / np.float32(np.pi) * 2), -1 + 1e-6, 1 - 1e-6)
    edges = [-1.0, *((anchors[1:] + anchors[:-1]) / 2), 1.0]
    moved = anchors.copy()
    for index in range(len(anchors)):
        inside = [float(f) for f in fraction if edges[index] < f <= edges[index + 1]]
        if inside:
            moved[index] = sum(inside) / (len(inside) + 1)  # sic: the count plus one
    return (moved + anchors) / 2


def test_anchor_shift_moves_half_way_to_the_shrunk_bin_mean():
    anchors = head()
    generator = np.random.default_rng(4)
    gamma = generator.normal(0.2, 0.5, 3000).astype(np.float32)
    beta = np.clip(generator.normal(-0.4, 0.2, 3000), -1.2, 1.2).astype(
        np.float32
    )  # leaves bins empty
    before = anchors.gamma.clone().numpy(), anchors.beta.clone().numpy()
    anchors.shift(torch.from_numpy(gamma), torch.from_numpy(beta))
    assert np.allclose(anchors.gamma.numpy(), loop_shift(before[0], gamma), atol=1e-12)
    assert np.allclose(anchors.beta.numpy(), loop_shift(before[1], beta), atol=1e-12)
    assert (np.diff(anchors.gamma.numpy()) > 0).all()  # still ascending
    assert anchors.beta[-1] == before[1][-1]  # an empty bin keeps its anchor


def test_pool_shifts_only_when_full_and_starts_over():
    anchors = head(shift_pool=100)
    start = anchors.gamma.clone()
    rows = torch.zeros(40, 8)
    rows[:, 4], rows[:, 5] = 0.3, -0.2
    assert not anchors.pool([rows, rows[:0]]) and not anchors.pool([rows])
    assert torch.equal(anchors.gamma, start)  # 80 labels: not yet
    assert anchors.pool([rows])  # 120 > 100
    assert not torch.equal(anchors.gamma, start) and anchors._pooled == 0
    moved = anchors.gamma.clone()
    anchors.pool([rows])
    anchors.clear_pool()
    assert anchors._pooled == 0 and torch.equal(anchors.gamma, moved)
    with pytest.raises(ValueError, match="shift_epochs must not be negative"):
        head(shift_epochs=-1)
