"""The contract of a sampler."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor

from grasppanda.registry import SAMPLERS

__all__ = ["Sampler", "check_inputs"]


@SAMPLERS.base
class Sampler:
    """Picks seeds among the candidate points of one frame.

    ``sample(coords, num_samples, scores=None, feats=None)`` receives the ``(N, 3)``
    coordinates of the candidates in metres (camera frame), optionally ``(N,)`` scores
    (never negative, higher is better) and ``(N, C)`` features, all detached. It returns
    **distinct** indices into them, at most ``num_samples`` and at most ``N``, as a
    ``long`` tensor on the device of ``coords``. Returning fewer is allowed (a voxel grid
    has as many points as occupied cells): the proposal that holds the sampler fills the
    frame up and marks what it added.

    ``needs`` names what the sampler reads besides coordinates (``"scores"``,
    ``"feats"``). A proposal that cannot provide it refuses the sampler in the config
    check.
    """

    needs: ClassVar[tuple[str, ...]] = ()

    def sample(
        self,
        coords: Tensor,
        num_samples: int,
        scores: Tensor | None = None,
        feats: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        options = ", ".join(
            f"{key}={value!r}" for key, value in vars(self).items() if key[0] != "_"
        )
        return f"{type(self).__name__}({options})"


def check_inputs(sampler: Sampler, scores: Tensor | None, feats: Tensor | None) -> None:
    """Raise when a sampler is called without something it declares in ``needs``."""
    given = {"scores": scores, "feats": feats}
    missing = [name for name in sampler.needs if given[name] is None]
    if missing:
        raise ValueError(
            f"{type(sampler).__name__} needs {' and '.join(missing)} per candidate point, but "
            "the proposal that calls it passed none"
        )
