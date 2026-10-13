"""The heat-map stage of HGGD: from image maps to partial grasps, and its training targets.

Decoding (inference and training alike):

1. squash the heat-map to ``(0, 1)`` and blur it with a 9×9 Gaussian;
2. take the cells of a coarse grid with the largest maxima, and in each the brightest
   pixel of the cell grown by one pixel on every side;
3. drop pixels below the score threshold, read the best in-plane angle class of the
   pixel's cell with its three offsets, and
4. accept candidates greedily in grid order unless an accepted one is closer than
   ``min_distance`` pixels *and* within ``min_angle`` of it.

Targets are dense maps built from the per-view grasp labels: a Gaussian per (thinned)
label centre for the heat-map, and per cell and angle class a soft count with the mean
offsets of the labels that fall there.

Reference behaviours kept on purpose, each a named option or stated here:

* the candidate window of the last grid row and column stops one pixel short of the
  image border (the reference clips the window's end to ``size - 1``);
* a cell counts as a positive angle class only from two labels on (the soft count
  ``2 / (1 + exp(-n)) - 1`` is compared with ``positive_count_threshold = 0.5``), while
  the offsets are supervised from one label on;
* the width offset is clamped to ``±0.5`` in the loss and to ``±1`` in the decoder;
* a label on a pixel without depth does not contribute a depth offset, and rescales the
  running sum only when that sum is positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.structures import ContractError

from .options import ProposalOptions, options_from
from .structures import ImagePredictions, PartialGraspSet

__all__ = ["HeatmapStage", "HeatmapTargets", "focal_loss", "gaussian_blur"]

#: Bound that keeps probabilities away from 0 and 1 before a logarithm.
EPS = 1e-6


def gaussian_blur(image: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Blur ``(B, U, V)`` maps with a ``kernel_size``² Gaussian, reflecting at the border.

    The kernel is the outer product of two normalised 1-D Gaussians sampled at integer
    offsets, built in single precision: the arithmetic of torchvision's ``gaussian_blur``.
    """
    if kernel_size % 2 != 1 or kernel_size < 1:
        raise ValueError(f"kernel_size must be odd and positive, got {kernel_size}")
    half = (kernel_size - 1) * 0.5
    offsets = torch.linspace(-half, half, steps=kernel_size)
    density = torch.exp(-0.5 * (offsets / sigma).pow(2))
    line = (density / density.sum()).to(image.device, image.dtype)
    kernel = line[:, None] * line[None, :]
    pad = kernel_size // 2
    padded = F.pad(image[:, None], [pad, pad, pad, pad], mode="reflect")
    return F.conv2d(padded, kernel[None, None])[:, 0]


def focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    positive_threshold: float,
    alpha: float = 0.5,
    gamma: float = 2.0,
    suppress_near_positives: bool = False,
) -> Tensor:
    """Penalty-reduced focal loss on logits, summed and divided by the number of positives.

    Entries whose target reaches ``positive_threshold`` are positives. With
    ``suppress_near_positives`` the loss of a negative is scaled by ``(1 - target)^4``,
    which spares the surroundings of a peak. Without any positive the sum over the
    negatives is returned undivided.
    """
    probability = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
    positive = (target >= positive_threshold).to(probability.dtype)
    negative = (target < positive_threshold).to(probability.dtype)
    positive_loss = alpha * torch.log(probability) * (1 - probability).pow(gamma) * positive
    negative_loss = (1 - alpha) * torch.log(1 - probability) * probability.pow(gamma) * negative
    if suppress_near_positives:
        negative_loss = negative_loss * (1 - target).pow(4)
    count = positive.sum()
    if count == 0:
        return -negative_loss.sum()
    return -(positive_loss.sum() + negative_loss.sum()) / count


@dataclass(eq=False)
class HeatmapTargets:
    """Dense targets of the image network for a batch, in its ``(u, v)`` layout.

    Attributes:
        heatmap: ``(B, U, V)`` Gaussians at the label centres, peak 1.
        angle: ``(B, K, U/s, V/s)`` soft label count per cell and angle class, in ``[0, 1)``.
        angle_offset, depth_offset, width_offset: ``(B, K, U/s, V/s)`` mean offsets of
            the labels of a cell and class; zero where there is none.
    """

    heatmap: Tensor
    angle: Tensor
    angle_offset: Tensor
    depth_offset: Tensor
    width_offset: Tensor


class HeatmapStage(nn.Module):
    """Decoder, target encoder and loss of the image maps. Holds no parameters.

    It is the single owner of the proposal grid: the angle classes, the stride of the
    angle maps and the width and depth anchors appear here and nowhere else. Every
    option of :class:`~.options.ProposalOptions` is an attribute of the stage.
    """

    def __init__(self, options: ProposalOptions | None = None) -> None:
        super().__init__()
        self.options = options_from(ProposalOptions, options)
        for field in fields(self.options):
            setattr(self, field.name, getattr(self.options, field.name))
        if self.num_angles < 1 or self.stride < 1 or self.grid_size < 1:
            raise ValueError("num_angles, stride and grid_size must be positive")
        if self.max_proposals < 1 or self.train_max_proposals < 1 or self.candidate_factor < 1:
            raise ValueError("proposal counts and candidate_factor must be positive")
        if self.blur_kernel % 2 != 1:
            raise ValueError(f"blur_kernel must be odd, got {self.blur_kernel}")

    # --------------------------------------------------------------------- decoding

    @property
    def angle_step(self) -> float:
        return math.pi / self.num_angles

    def activate(self, raw: ImagePredictions) -> ImagePredictions:
        """Squash and blur the heat-map, squash the angle scores, clamp the offsets."""
        heat = torch.sigmoid(raw.heatmap.detach()).clamp(EPS, 1 - EPS)
        return ImagePredictions(
            heatmap=gaussian_blur(heat, self.blur_kernel, self.blur_sigma),
            angle=torch.sigmoid(raw.angle.detach()).clamp(EPS, 1 - EPS),
            angle_offset=raw.angle_offset.detach().clamp(-0.5, 0.5),
            depth_offset=raw.depth_offset.detach().clamp(-0.5, 0.5),
            width_offset=raw.width_offset.detach().clamp(-1.0, 1.0),
            features=raw.features,
        )

    def candidates(self, heat: Tensor, count: int) -> Tensor:
        """``(n, 2)`` pixels ``(u, v)``: the brightest pixel of each of the ``count`` best cells.

        Cells are ranked by their maximum. The pixel is searched in the cell grown by
        one pixel on each side; the grown window ends at ``size - 1`` (exclusive), so
        the last row and column of the image are never searched.
        """
        size_u, size_v = heat.shape
        grid = self.grid_size
        cells = F.max_pool2d(heat[None, None], grid)[0, 0]
        best = torch.topk(cells.reshape(-1), min(cells.numel(), count)).indices
        cell_u = torch.div(best, cells.shape[1], rounding_mode="floor")
        cell_v = best % cells.shape[1]
        top = (cell_u * grid - 1).clamp(min=0)
        bottom = ((cell_u + 1) * grid + 1).clamp(max=size_u - 1)
        left = (cell_v * grid - 1).clamp(min=0)
        right = ((cell_v + 1) * grid + 1).clamp(max=size_v - 1)
        span = grid + 2
        offsets = torch.arange(span, device=heat.device)
        window_u = top[:, None] + offsets
        window_v = left[:, None] + offsets
        inside = (window_u < bottom[:, None])[:, :, None] & (window_v < right[:, None])[:, None, :]
        values = heat[
            window_u.clamp(max=size_u - 1)[:, :, None], window_v.clamp(max=size_v - 1)[:, None, :]
        ]
        values = torch.where(inside, values, values.new_tensor(-1.0))
        # Row-major over the window; ``argmax`` returns the first maximum.
        brightest = values.reshape(len(best), -1).argmax(dim=1)
        pixel_u = top + torch.div(brightest, span, rounding_mode="floor")
        pixel_v = left + brightest % span
        return torch.stack([pixel_u, pixel_v], dim=1)

    def select(
        self,
        maps: ImagePredictions,
        frame: int,
        *,
        training: bool,
        candidates: Tensor | None = None,
    ) -> PartialGraspSet:
        """The partial grasps of one frame of activated ``maps``.

        ``candidates`` replaces step 2 by recorded pixels, for comparisons with a
        reference run.
        """
        limit = self.train_max_proposals if training else self.max_proposals
        threshold = self.train_score_threshold if training else self.score_threshold
        heat = maps.heatmap[frame]
        if candidates is None:
            candidates = self.candidates(heat, limit * self.candidate_factor)
        pixel = candidates.to(heat.device)
        quality = heat[pixel[:, 0], pixel[:, 1]]
        keep = quality > threshold
        pixel, quality = pixel[keep], quality[keep]
        cell = torch.div(pixel, self.stride, rounding_mode="floor")
        scores = maps.angle[frame][:, cell[:, 0], cell[:, 1]]  # (K, n)
        angle_class = scores.argmax(dim=0)
        picked = (angle_class, cell[:, 0], cell[:, 1])
        angle_offset = maps.angle_offset[frame][picked]
        depth_offset = maps.depth_offset[frame][picked]
        width_offset = maps.width_offset[frame][picked]
        # The offsets are single precision; everything derived from them is double.
        theta = self.angle_step * (angle_class + angle_offset.double() + 0.5) - math.pi / 2
        millimetres = depth_offset.double() * (self.anchor_depth * 1000.0) * 2
        width = self.anchor_width * torch.exp(width_offset).double()

        accepted = self._thin(pixel.cpu().numpy(), theta.cpu().numpy(), limit)
        index = torch.as_tensor(accepted, dtype=torch.long, device=heat.device)
        return PartialGraspSet(
            frame=torch.full_like(index, frame),
            pixel=pixel[index],
            theta=theta[index],
            width=width[index],
            depth_offset=millimetres[index] / 1000.0,
            score=quality[index].double(),
            num_frames=maps.heatmap.shape[0],
        )

    def _thin(self, pixel: np.ndarray, theta: np.ndarray, limit: int) -> list[int]:
        """Greedy selection in candidate order; see step 4 of the module docstring."""
        accepted: list[int] = []
        min_angle = math.pi * self.min_angle / 180.0
        centres = np.zeros((0, 2), dtype=np.float64)
        angles = np.zeros((0,), dtype=np.float64)
        for index in range(len(pixel)):
            if len(accepted) >= limit:
                break
            if accepted and self.min_distance > 0:
                distance = np.sqrt(np.square(centres - pixel[index]).sum(axis=1))
                turn = np.abs(angles - theta[index])
                turn = np.minimum(math.pi - turn, turn)
                if np.logical_and(distance < self.min_distance, turn < min_angle).any():
                    continue
            accepted.append(index)
            centres = np.vstack([centres, pixel[index][None].astype(np.float64)])
            angles = np.append(angles, theta[index])
        return accepted

    def forward(
        self,
        raw: ImagePredictions,
        *,
        training: bool | None = None,
        candidates: list[Tensor] | None = None,
    ) -> tuple[PartialGraspSet, ImagePredictions]:
        """Decode a batch; returns the partial grasps and the activated maps."""
        training = self.training if training is None else training
        maps = self.activate(raw)
        batch = maps.heatmap.shape[0]
        if candidates is not None and len(candidates) != batch:
            raise ContractError(
                f"got recorded candidates for {len(candidates)} frame(s) of a batch of {batch}"
            )
        sets = [
            self.select(
                maps,
                frame,
                training=training,
                candidates=None if candidates is None else candidates[frame],
            )
            for frame in range(batch)
        ]
        return PartialGraspSet.cat(sets, batch), maps

    # ---------------------------------------------------------------------- targets

    def _map_shapes(self, image_size: tuple[int, int]) -> tuple[int, int]:
        return image_size[0] // self.stride, image_size[1] // self.stride

    def encode_frame(
        self,
        labels: dict[str, np.ndarray],
        depth: np.ndarray,
        image_size: tuple[int, int],
        *,
        subset: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Dense targets of one frame, as float64 arrays in the ``(u, v)`` layout.

        Args:
            labels: Per-view grasp labels (``centers``, ``theta``, ``width``, ``height``,
                ``depth``, ``score``; see :mod:`grasppanda.datasets.view_grasps`).
            depth: ``(H, W)`` sensor depth in metres, zero where invalid.
            image_size: ``(U, V)`` of the network image.
            subset: Indices of the labels to use when there are more than
                ``max_labels``; drawn at random when omitted.
        """
        size_u, size_v = image_size
        height, width = depth.shape
        if width % size_u or height % size_v or width // size_u != height // size_v:
            raise ContractError(
                f"the network image {image_size} is not an integer reduction of the "
                f"{(width, height)} sensor image"
            )
        scale = width // size_u
        keep = (np.asarray(labels["width"]) >= 0) & (
            np.asarray(labels["score"]) >= self.label_min_score
        )
        centre = np.asarray(labels["centers"], dtype=np.float64)[keep]
        theta = np.asarray(labels["theta"], dtype=np.float64)[keep].copy()
        jaw = np.asarray(labels["width"], dtype=np.float64)[keep]
        finger = np.asarray(labels["height"], dtype=np.float64)[keep]
        surface = np.asarray(labels["depth"], dtype=np.float64)[keep] * 1000.0  # millimetres
        if len(theta) > self.max_labels:
            if subset is None:
                subset = torch.randperm(len(theta))[: self.max_labels].numpy()
            centre, theta, jaw = centre[subset], theta[subset], jaw[subset]
            finger, surface = finger[subset], surface[subset]
        # A parallel gripper is symmetric: fold the angle into [-pi/2, pi/2].
        outside = (theta > math.pi / 2) | (theta < -math.pi / 2)
        theta[outside] = (theta[outside] + math.pi / 2) % math.pi - math.pi / 2

        # Corners of the grasp rectangle in sensor pixels, truncated to integers, then
        # expressed in the network image. Centres and widths are read back from them.
        along = jaw[:, None] / 2 * np.stack([np.cos(theta), -np.sin(theta)], axis=1)
        across = np.stack([-along[:, 1], along[:, 0]], axis=1)
        norm = np.linalg.norm(across, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            across = across / norm * finger[:, None] / 2
        corners = np.stack(
            [
                centre + across + along,
                centre + across - along,
                centre - across - along,
                centre - across + along,
            ],
            axis=1,
        )
        corners = np.nan_to_num(corners, nan=0.0).astype(np.int32).astype(np.float64) / scale
        middle = corners.mean(axis=1).astype(np.int32)  # (L, 2) pixel of the network image

        heatmap = self._peaks(middle, image_size)
        maps_u, maps_v = self._map_shapes(image_size)
        shape = (self.num_angles, maps_u, maps_v)
        count = np.zeros(shape)
        angle_sum, depth_sum, width_sum = np.zeros(shape), np.zeros(shape), np.zeros(shape)

        cell = middle // self.stride
        inside = (
            (cell[:, 0] >= 0) & (cell[:, 1] >= 0) & (cell[:, 0] < maps_u) & (cell[:, 1] < maps_v)
        )
        cell, middle_in = cell[inside], middle[inside]
        theta_in, corners_in, surface_in = theta[inside], corners[inside], surface[inside]
        depth_offset = self._depth_offsets(middle_in * scale, surface_in, depth)
        with np.errstate(divide="ignore"):
            width_offset = np.log(
                np.sqrt(np.square(corners_in[:, 0] - corners_in[:, 1]).sum(axis=1))
                / self.anchor_width
            )
        clipped = np.clip(theta_in, -math.pi / 2 + 1e-8, math.pi / 2 - 1e-8)
        angle_class, remainder = np.divmod(clipped + math.pi / 2, self.angle_step)
        angle_offset = remainder / self.angle_step - 0.5
        where = (angle_class.astype(np.int64), cell[:, 0], cell[:, 1])
        np.add.at(count, where, 1.0)
        np.add.at(angle_sum, where, angle_offset)
        np.add.at(width_sum, where, width_offset)
        missing = np.isnan(depth_offset)
        if not missing.any():
            np.add.at(depth_sum, where, depth_offset)
        else:
            # Cells with a label on a pixel without depth are accumulated in label order,
            # because such a label rescales the running sum; the others do not care.
            flat = np.ravel_multi_index(where, shape)
            ordered = np.isin(flat, flat[missing])
            np.add.at(depth_sum, tuple(w[~ordered] for w in where), depth_offset[~ordered])
            running = np.zeros(shape)
            for row in np.flatnonzero(ordered):
                at = (where[0][row], where[1][row], where[2][row])
                running[at] += 1
                if not missing[row]:
                    depth_sum[at] += depth_offset[row]
                elif running[at] > 1 and depth_sum[at] > 0:
                    depth_sum[at] *= running[at] / (running[at] - 1)
        divisor = count + (count == 0)
        return {
            "heatmap": heatmap,
            "angle": 2 / (1 + np.exp(-count)) - 1,
            "angle_offset": angle_sum / divisor,
            "depth_offset": depth_sum / divisor,
            "width_offset": width_sum / divisor,
        }

    def _peaks(self, middle: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        """Heat-map of Gaussians at the label centres, thinned greedily in label order."""
        size_u, size_v = image_size
        radius = self.peak_radius
        diameter = 2 * radius + 1
        sigma = diameter / 6
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        peak = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * sigma * sigma))
        peak[peak < np.finfo(peak.dtype).eps * peak.max()] = 0
        heatmap = np.zeros((size_u, size_v))
        centres = np.zeros((0, 2))
        for u, v in middle.tolist():
            if (
                len(centres)
                and not (np.linalg.norm(centres - (u, v), axis=1) >= self.peak_min_distance).all()
            ):
                continue
            centres = np.vstack([centres, [[u, v]]])
            u0, u1 = max(u - radius, 0), min(u + radius + 1, size_u)
            v0, v1 = max(v - radius, 0), min(v + radius + 1, size_v)
            if u0 >= u1 or v0 >= v1:
                continue
            patch = peak[u0 - u + radius : u1 - u + radius, v0 - v + radius : v1 - v + radius]
            np.maximum(heatmap[u0:u1, v0:v1], patch, out=heatmap[u0:u1, v0:v1])
        return heatmap

    def _depth_offsets(
        self, pixel: np.ndarray, surface: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Depth offset of each label, ``NaN`` where the sensor has no depth below it.

        ``pixel`` is ``(n, 2)`` in the sensor image, ``surface`` the label depth in
        millimetres. The observed depth is the median of a ``2 depth_window`` square
        window whose end is clipped to ``size - 1`` (exclusive).
        """
        height, width = depth.shape
        half = self.depth_window
        millimetres = depth.astype(np.float32) * np.float32(1000.0)
        offsets = np.full(len(pixel), np.nan)
        for row, (u, v) in enumerate(pixel.tolist()):
            window = millimetres[
                max(0, v - half) : min(height - 1, v + half),
                max(0, u - half) : min(width - 1, u + half),
            ]
            if window.size == 0:
                continue
            observed = np.median(window)
            if observed > 0:
                offsets[row] = np.clip(
                    (surface[row] - observed) / (self.anchor_depth * 1000.0 * 2), -0.5, 0.5
                )
        return offsets

    def encode(
        self,
        labels: list[dict[str, Tensor]],
        depth: Tensor,
        image_size: tuple[int, int],
    ) -> HeatmapTargets:
        """Dense targets of a batch from its per-view labels and depth images."""
        if len(labels) != depth.shape[0]:
            raise ContractError(
                f"got labels of {len(labels)} frame(s) for a batch of {depth.shape[0]}"
            )
        frames = []
        for frame, entry in enumerate(labels):
            arrays = {name: value.detach().cpu().numpy() for name, value in entry.items()}
            frames.append(self.encode_frame(arrays, depth[frame].cpu().numpy(), image_size))

        def stack(name: str) -> Tensor:
            array = np.stack([frame[name] for frame in frames]).astype(np.float32)
            return torch.from_numpy(array).to(depth.device)

        return HeatmapTargets(
            heatmap=stack("heatmap"),
            angle=stack("angle"),
            angle_offset=stack("angle_offset"),
            depth_offset=stack("depth_offset"),
            width_offset=stack("width_offset"),
        )

    # ------------------------------------------------------------------------- loss

    def loss(self, raw: ImagePredictions, target: HeatmapTargets) -> dict[str, Tensor]:
        """The three weighted loss terms of the image network."""
        if raw.heatmap.shape != target.heatmap.shape or raw.angle.shape != target.angle.shape:
            raise ContractError(
                f"image maps {tuple(raw.heatmap.shape)} / {tuple(raw.angle.shape)} do not match "
                f"their targets {tuple(target.heatmap.shape)} / {tuple(target.angle.shape)}"
            )
        heat = focal_loss(
            raw.heatmap, target.heatmap, positive_threshold=0.99, suppress_near_positives=True
        )
        angle = focal_loss(
            raw.angle, target.angle, positive_threshold=self.positive_count_threshold, alpha=0.25
        )
        supervised = target.angle > 0
        pairs = (
            (raw.angle_offset, target.angle_offset),
            (raw.depth_offset, target.depth_offset),
            (raw.width_offset, target.width_offset),
        )
        offsets = raw.angle.new_zeros(())
        for predicted, wanted in pairs:
            offsets = offsets + F.smooth_l1_loss(
                predicted.clamp(-0.5, 0.5) * supervised, wanted * supervised, reduction="sum"
            ) / (supervised.sum() + EPS)
        return {
            "loss/heatmap": self.heatmap_weight * heat,
            "loss/angle": self.angle_weight * angle,
            "loss/offsets": self.offset_weight * offsets / len(pairs),
        }
