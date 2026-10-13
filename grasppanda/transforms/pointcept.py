# Copyright (c) 2023 Pointcept
# Adapted for GraspPanda from https://github.com/Pointcept/Pointcept @ 1342eda (MIT)
"""Point-cloud augmentations of Pointcept, fitted to the grasp data contract.

Pointcept's transforms work on a dictionary with ``coord`` and per-point arrays. A grasp
sample also carries object poses, to which the grasp labels are attached, and possibly
images. So only the augmentations that this contract can carry along are here:

==================  ==========================================================
``RandomJitter``    noise on the input points; poses and labels stay
``RandomDropout``   removes points with their per-point labels
``RandomShift``     translates points *and* object poses
==================  ==========================================================

Not adapted: ``RandomScale`` (label widths and the gripper are in metres and would not
follow), ``ElasticDistortion`` (not rigid: object poses cannot follow), the chromatic
transforms (the methods here feed no colour). Rotation and flip exist already
(:mod:`grasppanda.transforms.points`).

Randomness is the sample's own generator (``sample.rng``), not the global one, and each
step splits into ``draw`` and ``apply`` like the steps of ``points.py``.
"""

from __future__ import annotations

import numpy as np

from grasppanda.registry import TRANSFORMS
from grasppanda.structures import ContractError

from .base import PointTransform
from .frame import PointSample

__all__ = ["RandomDropout", "RandomJitter", "RandomShift"]


def _refuse_images(sample: PointSample, step: str) -> None:
    carried = [name for name in ("rgb", "depth") if getattr(sample, name) is not None]
    carried += sorted(sample.camera_bound & sample.extras.keys())
    if carried:
        raise ContractError(
            f"{sample.key}: {step} moves points, which would leave the image-space data of "
            f"this sample behind ({', '.join(carried)}). Remove the point-space augmentation, "
            "or do not ask the dataset for images and image-keyed labels."
        )


@TRANSFORMS.register()
class RandomJitter(PointTransform):
    """Add clipped Gaussian noise to every input point (sensor noise).

    Only ``points``, the network input, is perturbed: the clean cloud ``full_points``,
    the object poses and all labels stay, so the labels describe the clean scene.

    Args:
        sigma: Standard deviation of the noise per coordinate, in metres. Pointcept's
            default of 0.01 is for rooms; a table-top grasp cloud wants millimetres.
        clip: Largest absolute noise per coordinate, in metres.
    """

    status = "tested"
    source = (
        "Qi et al., PointNet++ (arXiv 1706.02413); adapted from Pointcept/Pointcept@1342eda, MIT"
    )

    def __init__(self, sigma: float = 0.001, clip: float = 0.005) -> None:
        if sigma < 0 or clip <= 0:
            raise ValueError(f"RandomJitter needs sigma >= 0 and clip > 0, got {sigma}, {clip}")
        self.sigma, self.clip = float(sigma), float(clip)

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(self.sigma * rng.standard_normal((count, 3)), -self.clip, self.clip)

    def apply(self, sample: PointSample, noise: np.ndarray) -> PointSample:
        _refuse_images(sample, "RandomJitter")
        sample.points = sample.points + noise  # a new array: full_points stays clean
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(len(sample), sample.rng))


@TRANSFORMS.register()
class RandomDropout(PointTransform):
    """With probability ``prob`` remove a fraction ``ratio`` of the points, labels included.

    Put it *before* the step that fixes the point count (``RandomSample``), which then
    draws from the thinned cloud: every frame of a batch must end with the same count.

    Args:
        ratio: Fraction of the points removed when the step applies.
        prob: Probability that it applies to a sample.
    """

    status = "tested"
    source = (
        "Qi et al., PointNet++ (arXiv 1706.02413); adapted from Pointcept/Pointcept@1342eda, MIT"
    )

    def __init__(self, ratio: float = 0.2, prob: float = 0.5) -> None:
        if not 0.0 <= ratio < 1.0 or not 0.0 <= prob <= 1.0:
            raise ValueError(
                f"RandomDropout needs 0 <= ratio < 1 and 0 <= prob <= 1, got {ratio}, {prob}"
            )
        self.ratio, self.prob = float(ratio), float(prob)

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray | None:
        """Sorted rows to keep, or ``None`` when the step does not apply."""
        if rng.random() >= self.prob:
            return None
        keep = max(int(count * (1.0 - self.ratio)), 1)
        return np.sort(rng.choice(count, keep, replace=False))

    def apply(self, sample: PointSample, index: np.ndarray | None) -> PointSample:
        if index is not None:
            sample.gather(index)
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(len(sample), sample.rng))


@TRANSFORMS.register()
class RandomShift(PointTransform):
    """Translate the scene by a uniform offset per camera axis; object poses move along.

    Args:
        shift: ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]`` in metres, in the
            camera frame (z is the viewing direction).
    """

    status = "tested"
    source = "Pointcept (Wu et al.), RandomShift; adapted from Pointcept/Pointcept@1342eda, MIT"

    def __init__(self, shift: list[list[float]]) -> None:
        bounds = np.asarray(shift, dtype=np.float64)
        if bounds.shape != (3, 2) or (bounds[:, 0] > bounds[:, 1]).any():
            raise ValueError(
                f"RandomShift.shift must be three [min, max] pairs with min <= max, got {shift}"
            )
        self.shift = [[float(low), float(high)] for low, high in bounds]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        bounds = np.asarray(self.shift)
        return rng.uniform(bounds[:, 0], bounds[:, 1])

    def apply(self, sample: PointSample, offset: np.ndarray) -> PointSample:
        _refuse_images(sample, "RandomShift")
        offset = np.asarray(offset, dtype=np.float64)
        shared = sample.full_points is sample.points
        sample.points = sample.points + offset
        if shared:
            sample.full_points = sample.points
        elif sample.full_points is not None:
            sample.full_points = sample.full_points + offset
        for annotation in sample.objects or ():
            pose = annotation.pose.astype(np.float64)
            pose[:3, 3] += offset
            annotation.pose = pose.astype(np.float32)
        return sample

    def __call__(self, sample: PointSample) -> PointSample:
        return self.apply(sample, self.draw(sample.rng))
