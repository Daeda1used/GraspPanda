"""Rotation anchors of HGGD: decoding, targets, loss and the learned anchor positions.

A grasp rotation is factored as ``R = R0(theta) · Rz(gamma) · Ry(beta)``. ``R0`` puts the
approach axis on the optical axis and turns the jaws by the in-plane angle ``theta``,
which the image stage predicts. The region stage classifies ``(gamma, beta)`` over a
``n × n`` grid of *anchors*; anchor ``a`` is ``(gamma[a % n], beta[a // n])``, both stored
as fractions of ``pi / 2``. With every anchor comes a 3-D offset of the grasp centre.

The anchor positions are not gradient-trained. During the first epochs the labels
matched to the proposals are pooled, and whenever the pool is full each anchor moves
half-way towards the mean of the label angles that fall into its bin. The positions are
buffers of this module and travel with the checkpoint.

Reference behaviours kept on purpose, each a named option or stated here:

* ``frame: pixel_keypoints`` — ``R0`` is built from two pixel offsets back-projected with
  ``fx`` and ``fy`` separately, so its in-plane axes are orthogonal only when
  ``fx = fy``; with the GraspNet cameras they are off by a few 1e-4. ``orthonormal``
  builds an exact rotation.
* the label angles are compared with the anchors as Euler angles about x, y and z (not
  through ``R``), with the gripper symmetry ``(theta + pi, -gamma, -beta)``;
* the bin mean of the anchor update divides by the bin count **plus one**;
* an anchor whose ``|gamma|`` reaches ``max_gamma`` is never decoded.
"""

from __future__ import annotations

import math
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.structures import ContractError, GraspGroup, Gripper, SampleKey

from .heatmap import EPS, focal_loss
from .options import AnchorOptions, options_from
from .structures import RegionSet

__all__ = ["AnchorHead", "euler_quaternion", "rotation_similarity"]


def euler_quaternion(euler: Tensor) -> Tensor:
    """``(..., 3)`` angles about x, y, z → ``(..., 4)`` quaternions ``qx ⊗ qy ⊗ qz`` (w first)."""
    half = euler / 2
    cx, cy, cz = half.cos().unbind(-1)
    sx, sy, sz = half.sin().unbind(-1)
    # qx ⊗ qy
    w, x, y, z = cx * cy, sx * cy, cx * sy, sx * sy
    # (qx ⊗ qy) ⊗ qz with qz = (cz, 0, 0, sz)
    return torch.stack(
        [-z * sz + w * cz, y * sz + x * cz, -x * sz + y * cz, z * cz + w * sz], dim=-1
    )


def rotation_similarity(anchors: Tensor, labels: Tensor) -> Tensor:
    """``(A, L)`` similarity in ``[0, 1]`` of ``(theta, gamma, beta)`` triples.

    ``|<q_a, q_l>|`` of the Euler quaternions, maximised over the two descriptions of a
    symmetric gripper.
    """
    mirrored = anchors * anchors.new_tensor([1.0, -1.0, -1.0])
    mirrored[:, 0] = mirrored[:, 0] + math.pi
    target = euler_quaternion(labels).T
    return torch.maximum(
        (euler_quaternion(anchors) @ target).abs(), (euler_quaternion(mirrored) @ target).abs()
    )


class AnchorHead(nn.Module):
    """Decoder, target encoder and loss of the region stage; owns the anchor positions.

    Every option of :class:`~.options.AnchorOptions` is an attribute of the head.

    Args:
        options: See :class:`~.options.AnchorOptions`.
        gripper: Gripper the grasps refer to (injected by the detector).
    """

    #: Meaning of the scores of the decoded grasps.
    score_kind = "anchor_probability"

    def __init__(
        self, options: AnchorOptions | None = None, gripper: Gripper | None = None
    ) -> None:
        super().__init__()
        self.options = options_from(AnchorOptions, options)
        for field in fields(self.options):
            setattr(self, field.name, getattr(self.options, field.name))
        num_anchors = self.num_anchors
        if num_anchors < 1 or not 1 <= self.top_k <= num_anchors**2:
            raise ValueError(
                f"top_k must be between 1 and num_anchors ** 2 = {num_anchors**2}, got {self.top_k}"
            )
        if self.frame not in ("pixel_keypoints", "orthonormal"):
            raise ValueError(f"unknown frame '{self.frame}'")
        if self.shift_epochs < 0:
            raise ValueError(f"shift_epochs must not be negative, got {self.shift_epochs}")
        self.gripper = gripper if gripper is not None else Gripper()
        edges = torch.linspace(-1, 1, num_anchors + 1, dtype=torch.float64)
        centres = (edges[1:] + edges[:-1]) / 2
        #: Anchor positions as fractions of pi / 2, ascending.
        self.register_buffer("gamma", centres.clone())
        self.register_buffer("beta", centres.clone())
        self._pool: list[Tensor] = []
        self._pooled = 0

    @property
    def num_classes(self) -> int:
        return self.num_anchors**2

    def anchor_angles(self) -> Tensor:
        """``(A, 2)`` float64 ``(gamma, beta)`` of every anchor class, radians."""
        index = torch.arange(self.num_classes, device=self.gamma.device)
        gamma = self.gamma[index % self.num_anchors] * math.pi / 2
        beta = self.beta[torch.div(index, self.num_anchors, rounding_mode="floor")] * math.pi / 2
        return torch.stack([gamma, beta], dim=1)

    # --------------------------------------------------------------------- decoding

    def base_frame(
        self, theta: Tensor, width: Tensor, depth: Tensor, intrinsics: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``R0(theta)`` with the metric width and finger height of each grasp.

        Args:
            theta: ``(G,)`` in-plane angles.
            width: ``(G,)`` jaw openings in sensor pixels.
            depth: ``(G,)`` camera-z of the grasp centres, metres.
            intrinsics: ``(G, 3, 3)`` camera matrices.

        Returns:
            ``(rotation (G, 3, 3), width (G,), height (G,))``, float64, metres.
        """
        fx, fy = intrinsics[:, 0, 0].double(), intrinsics[:, 1, 1].double()
        cx, cy = intrinsics[:, 0, 2].double(), intrinsics[:, 1, 2].double()
        cos, sin = torch.cos(theta), torch.sin(theta)
        open_u = cx - width / 2 * cos
        open_v = cy + width / 2 * sin
        # The in-plane unit vector towards the open point, turned by a quarter.
        du, dv = open_u - cx, open_v - cy
        norm = torch.sqrt(du * du + dv * dv)
        upper_u = -(dv / norm) * self.finger_pixels / 2 + cx
        upper_v = (du / norm) * self.finger_pixels / 2 + cy
        zero = torch.zeros_like(depth)
        closing = torch.stack([depth / fx * (open_u - cx), depth / fy * (open_v - cy), zero], dim=1)
        across = torch.stack(
            [depth / fx * (upper_u - cx), depth / fy * (upper_v - cy), zero], dim=1
        )
        closing_norm = closing.norm(dim=1)
        across_norm = across.norm(dim=1)
        approach = torch.stack([zero, zero, torch.ones_like(depth)], dim=1)
        if self.frame == "pixel_keypoints":
            columns = [
                approach,
                closing / (closing_norm[:, None] + 1e-8),
                across / (across_norm[:, None] + 1e-8),
            ]
        else:
            columns = [
                approach,
                torch.stack([-cos, sin, zero], dim=1),
                torch.stack([-sin, -cos, zero], dim=1),
            ]
        return torch.stack(columns, dim=2), closing_norm * 2, across_norm * 2

    def decode(
        self,
        logits: Tensor,
        offsets: Tensor,
        regions: RegionSet,
        intrinsics: Tensor,
        scale: float,
        keys: list[SampleKey],
    ) -> list[GraspGroup]:
        """One :class:`GraspGroup` per frame: the ``top_k`` anchors of every region.

        Args:
            logits: ``(R, A)`` anchor logits.
            offsets: ``(R, A, 3)`` raw centre offsets.
            regions: The regions the predictions belong to.
            intrinsics: ``(B, 3, 3)`` camera matrices of the sensor images.
            scale: Sensor pixels per pixel of the network image.
            keys: The frames of the batch.
        """
        grasps = regions.grasps
        if logits.shape != (len(grasps), self.num_classes) or offsets.shape != (
            len(grasps),
            self.num_classes,
            3,
        ):
            raise ContractError(
                f"AnchorHead.decode got logits {tuple(logits.shape)} and offsets "
                f"{tuple(offsets.shape)} for {len(grasps)} regions and {self.num_classes} anchors"
            )
        device = logits.device
        if len(grasps) == 0:
            return [self._group(None, key, device) for key in keys]
        score, anchor = torch.topk(torch.sigmoid(logits), self.top_k, dim=1)
        moved = self.offset_scale * offsets.clamp(-1.0, 1.0)
        moved = torch.gather(moved, 1, anchor[..., None].expand(-1, -1, 3))
        angles = self.anchor_angles()[anchor]  # (R, k, 2)
        centre = grasps.centre.double()[:, None, :] + moved.double()
        keep = angles[..., 0].abs() < math.pi * self.max_gamma / 180.0

        frame = grasps.frame[:, None].expand_as(anchor)[keep]
        camera = intrinsics.double()[frame]
        fx, fy, cx, cy = camera[:, 0, 0], camera[:, 1, 1], camera[:, 0, 2], camera[:, 1, 2]
        centre = centre[keep]
        gamma, beta = angles[..., 0][keep], angles[..., 1][keep]
        theta = grasps.theta.double()[:, None].expand_as(anchor)[keep]
        width = (grasps.width.double() * scale)[:, None].expand_as(anchor)[keep]
        # Through the image and back, as the reference stores decoded grasps as pixels.
        pixel_u = centre[:, 0] / centre[:, 2] * fx + cx
        pixel_v = centre[:, 1] / centre[:, 2] * fy + cy
        z = centre[:, 2] * 1e3 / 1000
        translation = torch.stack([z / fx * (pixel_u - cx), z / fy * (pixel_v - cy), z], dim=1)
        rotation, metric_width, height = self.base_frame(theta, width, z, camera)
        cos, sin = torch.cos(gamma), torch.sin(gamma)
        zero, one = torch.zeros_like(cos), torch.ones_like(cos)
        about_height = torch.stack(
            [cos, -sin, zero, sin, cos, zero, zero, zero, one], dim=1
        ).reshape(-1, 3, 3)
        cos, sin = torch.cos(beta), torch.sin(beta)
        about_closing = torch.stack(
            [cos, zero, sin, zero, one, zero, -sin, zero, cos], dim=1
        ).reshape(-1, 3, 3)
        rotation = rotation @ about_height @ about_closing
        values = {
            "translation": translation,
            "rotation": rotation,
            "width": metric_width,
            "height": height,
            "depth": torch.full_like(metric_width, self.grasp_depth),
            "score": score[keep].double(),
        }
        return [
            self._group(
                {name: value[frame == index] for name, value in values.items()}, key, device
            )
            for index, key in enumerate(keys)
        ]

    def _group(
        self, values: dict[str, Tensor] | None, key: SampleKey, device: torch.device
    ) -> GraspGroup:
        meta = {"gripper": self.gripper, "score_kind": self.score_kind, "key": key}
        if values is None:
            return GraspGroup.empty(device=device, **meta)
        return GraspGroup(**values, **meta)

    def rotation_tolerance(self, intrinsics: Tensor) -> float:
        """Deviation from orthonormality that a rotation decoded for this camera may have.

        The in-plane axes of the ``pixel_keypoints`` frame enclose an angle whose cosine
        is at most ``|fx² - fy²| / (fx² + fy²)`` (2e-4 and 5e-4 for the GraspNet-1Billion
        cameras). That is the reference decode, not an error; anything beyond it is.
        """
        if self.frame == "orthonormal":
            return 1e-4
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        return 1e-4 + abs(fx * fx - fy * fy) / (fx * fx + fy * fy)

    # ---------------------------------------------------------------------- targets

    def match(
        self,
        regions: RegionSet,
        labels: list[dict[str, Tensor]],
        intrinsics: Tensor,
    ) -> list[Tensor]:
        """Per region, its labels as ``(M, 8)`` float32 rows, nearest first.

        Columns: centre relative to the region centre (3), ``theta``, ``gamma``, ``beta``,
        width in pixels, score.
        """
        centres = regions.grasps.centre.double()
        matched: list[Tensor] = []
        for frame, entry in enumerate(labels):
            rows = regions.grasps.rows(frame)
            camera = intrinsics[frame].double()
            fx, fy, cx, cy = camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2]
            pixel = entry["centers"].double()
            z = entry["depth"].double()
            table = torch.stack(
                [
                    z / fx * (pixel[:, 0] - cx),
                    z / fy * (pixel[:, 1] - cy),
                    z,
                    entry["theta"].double(),
                    entry["gamma"].double(),
                    entry["beta"].double(),
                    entry["width"].double(),
                    entry["score"].double(),
                ],
                dim=1,
            ).to(torch.float32)
            p1, p2 = centres[rows], table[:, :3].double()
            squared = -2 * (p1 @ p2.T) + p1.square().sum(1)[:, None] + p2.square().sum(1)[None, :]
            distance = squared.sqrt()
            near = distance < self.match_radius
            limit = min(self.max_matches, len(table))
            # Nearest first: labels outside the radius sort behind every label inside it.
            far = torch.full_like(distance, float("inf"))
            order = torch.argsort(torch.where(near, distance, far), dim=1, stable=True)[:, :limit]
            mine = table[order]  # (regions, limit, 8)
            relative = mine[..., :3].double() - p1[:, None, :]
            mine = torch.cat([relative.to(torch.float32), mine[..., 3:]], dim=-1)
            counts = near.sum(dim=1).clamp(max=limit).tolist()
            matched.extend(mine[row, :count] for row, count in enumerate(counts))
        return matched

    def encode(self, theta: Tensor, matched: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Targets of the regions: ``(R, A)`` best similarity and ``(R, A, 3)`` offsets.

        For every anchor of a region the label with the most similar rotation is
        looked up; its similarity is the classification target and its centre offset
        (metres) the regression target. A region without labels gets zeros.
        """
        device = self.gamma.device
        regions, classes = len(matched), self.num_classes
        similarity = torch.zeros(regions, classes, dtype=torch.float64, device=device)
        offset = torch.zeros(regions, classes, 3, dtype=torch.float64, device=device)
        counts = torch.tensor([len(rows) for rows in matched], dtype=torch.long, device=device)
        if regions == 0 or int(counts.max()) == 0:
            return similarity, offset
        # All regions at once: labels padded to the longest list, padding never chosen.
        labels = torch.nn.utils.rnn.pad_sequence(matched, batch_first=True).to(device).double()
        present = torch.arange(labels.shape[1], device=device)[None, :] < counts[:, None]
        angles = self.anchor_angles().to(torch.float32).double()  # single precision, as trained
        anchors = torch.cat(
            [
                theta.to(device, torch.float32).double()[:, None, None].expand(-1, classes, 1),
                angles[None].expand(regions, -1, -1),
            ],
            dim=-1,
        )
        mirrored = anchors * anchors.new_tensor([1.0, -1.0, -1.0])
        mirrored[..., 0] = mirrored[..., 0] + math.pi
        target = euler_quaternion(labels[..., 3:6]).transpose(1, 2)  # (R, 4, M)
        table = torch.maximum(
            (euler_quaternion(anchors) @ target).abs(), (euler_quaternion(mirrored) @ target).abs()
        )
        distance = (1 - table).masked_fill(~present[:, None, :], float("inf"))
        best = distance.argmin(dim=2)  # the first of equally similar labels: the nearest
        labelled = (counts > 0)[:, None]
        similarity = torch.where(
            labelled, 1 - distance.gather(2, best[..., None])[..., 0], similarity
        )
        chosen = labels[..., :3].gather(1, best[..., None].expand(-1, -1, 3))
        offset = torch.where(labelled[..., None], chosen, offset)
        return similarity, offset

    # ------------------------------------------------------------------------- loss

    def loss(
        self, logits: Tensor, offsets: Tensor, similarity: Tensor, offset: Tensor
    ) -> dict[str, Tensor]:
        """Focal loss on the anchor logits and smooth-L1 on the offsets of positive anchors."""
        if logits.shape != similarity.shape or offsets.shape != offset.shape:
            raise ContractError(
                f"anchor predictions {tuple(logits.shape)} / {tuple(offsets.shape)} do not match "
                f"their targets {tuple(similarity.shape)} / {tuple(offset.shape)}"
            )
        if len(logits) == 0:
            zero = logits.sum() + offsets.sum()
            return {"loss/anchor": zero, "loss/anchor_offset": zero.clone()}
        target = similarity.to(logits.dtype)
        wanted = offset.to(offsets.dtype) / self.offset_scale
        anchor = focal_loss(logits, target, positive_threshold=self.positive_similarity)
        positive = (target > self.positive_similarity)[..., None]
        moved = F.smooth_l1_loss(offsets * positive, wanted * positive, reduction="sum")
        moved = moved / (positive.sum() + EPS)
        return {"loss/anchor": anchor, "loss/anchor_offset": self.offset_weight * moved}

    # ------------------------------------------------------------------- anchor shift

    def pool(self, matched: list[Tensor]) -> bool:
        """Add matched labels to the pool; move the anchors when it is full.

        Returns whether the anchors moved.
        """
        angles = [labels[:, 4:6].detach().to("cpu", torch.float32) for labels in matched]
        angles = [a for a in angles if len(a)]
        if angles:
            self._pool.append(torch.cat(angles))
            self._pooled += len(self._pool[-1])
        if self._pooled <= self.shift_pool:
            return False
        pooled = torch.cat(self._pool)
        self.clear_pool()
        self.shift(pooled[:, 0], pooled[:, 1])
        return True

    def clear_pool(self) -> None:
        self._pool, self._pooled = [], 0

    @staticmethod
    def _shifted(anchors: Tensor, angles: Tensor) -> Tensor:
        """Anchors moved half-way to the (shrunk) mean of the angles in their bins."""
        # Label angles as fractions of pi / 2, in single precision like the labels.
        fraction = (angles.to(torch.float32) / math.pi * 2).clamp(-1 + 1e-6, 1 - 1e-6).double()
        anchors = anchors.detach().cpu().double()
        edges = torch.cat(
            [
                anchors.new_tensor([-1.0]),
                (anchors[1:] + anchors[:-1]) / 2,
                anchors.new_tensor([1.0]),
            ]
        )
        bins = torch.searchsorted(edges, fraction) - 1
        count = torch.bincount(bins, minlength=len(anchors)).double()
        total = torch.zeros(len(anchors), dtype=torch.float64).index_add_(0, bins, fraction)
        moved = torch.where(count > 0, total / (count + 1), anchors)
        return (moved + anchors) / 2

    @torch.no_grad()
    def shift(self, gamma: Tensor, beta: Tensor) -> None:
        """One anchor update from label angles in radians."""
        self.gamma.copy_(self._shifted(self.gamma, gamma).to(self.gamma.device))
        self.beta.copy_(self._shifted(self.beta, beta).to(self.beta.device))
