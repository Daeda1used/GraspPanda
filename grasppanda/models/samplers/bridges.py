"""Samplers that call an optional library: ``fpsample`` and ``torch_cluster``.

Each library is imported when the sampler is built, so the package imports without it
and a missing library is reported with the command that installs it.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from grasppanda.registry import SAMPLERS
from grasppanda.utils.imports import require

from .base import Sampler
from .builtin import distinct

__all__ = ["FpsampleFPS", "TorchClusterFPS"]

_FPSAMPLE_METHODS = ("bucket", "bucket_kdtree", "vanilla", "npdu", "npdu_kdtree")


@SAMPLERS.register()
class FpsampleFPS(Sampler):
    """Furthest point sampling on the CPU through ``fpsample``, with the bucket-based QuickFPS.

    Bucket-based sampling keeps the points in a KD-tree and only updates the buckets a
    new pick can affect. Measured with fpsample 1.0.2: ``vanilla`` returns exactly the
    picks of ``grasppanda.ops.furthest_point_sample``; the bucket methods are repeatable
    and spread the seeds as well (the smallest spacing of 256 picks among 5000 points was
    0.138 against 0.140), but they do not begin at the requested start point, so the
    seed set is another one; the ``npdu`` methods assume an ordered cloud and spread
    worse than random sampling on a shuffled one. The points travel to the CPU and back.

    Args:
        method: ``bucket`` (QuickFPS, ``bucket_fps_kdline_sampling``), ``bucket_kdtree``,
            ``vanilla``, ``npdu`` or ``npdu_kdtree`` (the last two are approximations for
            clouds whose points are stored in scan order).
        height: Height ``h`` of the KD-tree of ``bucket`` (the bucket size is ``2^h``).
            The library recommends 3 or 5 for small clouds, 7 for medium, 9 for very
            large ones. Lowered automatically for a frame with fewer than ``2^h`` points.
        start: ``first``: pass start index 0 (repeatable). ``random``: the library
            picks the start with NumPy's global generator.
    """

    status = "bridge"
    extras = ("fpsample",)
    source = (
        "QuickFPS (Han et al., IEEE TCAD 2023) via fpsample 1.0 "
        "(github.com/leonardodalinky/fpsample, MIT)"
    )

    def __init__(
        self,
        method: Literal["bucket", "bucket_kdtree", "vanilla", "npdu", "npdu_kdtree"] = "bucket",
        height: int = 5,
        start: Literal["first", "random"] = "first",
    ) -> None:
        if method not in _FPSAMPLE_METHODS:
            raise ValueError(
                f"FpsampleFPS: unknown method '{method}'; choose one of {_FPSAMPLE_METHODS}"
            )
        if start not in ("first", "random"):
            raise ValueError(f"FpsampleFPS: unknown start '{start}'; choose 'first' or 'random'")
        if height < 1:
            raise ValueError("FpsampleFPS: height must be at least 1")
        self.method = method
        self.height = height
        self.start = start
        self._library = require("fpsample", extra="fpsample", purpose="FpsampleFPS")

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        count = len(coords)
        num_samples = min(num_samples, count)
        cloud = coords.detach().cpu().numpy().astype("float32")
        start = 0 if self.start == "first" else None
        library = self._library
        if self.method == "bucket":
            height = min(self.height, max(count.bit_length() - 1, 1))
            if 2**height > count:  # one or two points: the tree cannot be built
                picks = library.fps_sampling(cloud, num_samples, start_idx=start)
            else:
                picks = library.bucket_fps_kdline_sampling(
                    cloud, num_samples, h=height, start_idx=start
                )
        elif self.method == "bucket_kdtree":
            picks = library.bucket_fps_kdtree_sampling(cloud, num_samples, start_idx=start)
        elif self.method == "vanilla":
            picks = library.fps_sampling(cloud, num_samples, start_idx=start)
        elif self.method == "npdu":
            picks = library.fps_npdu_sampling(cloud, num_samples, start_idx=start)
        else:
            picks = library.fps_npdu_kdtree_sampling(cloud, num_samples, start_idx=start)
        index = torch.as_tensor(picks.astype("int64"), device=coords.device)
        return distinct(index)


@SAMPLERS.register()
class TorchClusterFPS(Sampler):
    """Furthest point sampling through ``torch_cluster.fps``, the kernel PyTorch Geometric uses.

    Runs on the device of the points (a compiled CPU and CUDA kernel). The library takes
    a ratio, not a count, so the result is trimmed to the requested number.

    Args:
        random_start: Start at a random candidate (the library's default) instead of
            candidate 0. A random start draws from PyTorch's global generator.
    """

    status = "bridge"
    extras = ("torch-cluster",)
    source = (
        "PointNet++ (Qi et al., NeurIPS 2017) via torch_cluster.fps "
        "(github.com/rusty1s/pytorch_cluster, MIT; Fey & Lenssen, PyTorch Geometric)"
    )

    def __init__(self, random_start: bool = False) -> None:
        self.random_start = random_start
        self._library = require("torch_cluster", extra="torch-cluster", purpose="TorchClusterFPS")

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        count = len(coords)
        num_samples = min(num_samples, count)
        picks = self._library.fps(
            coords.float(), ratio=num_samples / count, random_start=self.random_start
        )
        return distinct(picks.long())[:num_samples]
