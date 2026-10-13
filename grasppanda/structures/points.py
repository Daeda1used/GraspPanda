"""Point sets and the features bound to them."""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .common import ContractError

__all__ = ["PointFeatures", "PointSet"]

_ids = itertools.count(1)


def _new_id() -> int:
    """An identity that no other process of the run hands out.

    A data-loader worker is a copy of the main process and inherits its counter, so a
    set made in a worker and a set made in the main process would otherwise get the
    same number, and so would the sets of two workers. The process id keeps them apart.
    """
    return (os.getpid() << 32) | next(_ids)


@dataclass(eq=False)
class PointSet:
    """Points of a batch, concatenated, with an identity.

    Attributes:
        coords: ``(N, 3)`` coordinates in metres.
        batch_offsets: ``(B + 1,)`` boundaries; sample ``b`` owns ``coords[o[b]:o[b+1]]``.
            ``None`` means a single sample.
        frame: Reference frame of the coordinates.
        parent_id: ``id`` of the set this one was sampled from, if any.
        parent_index: ``(N,)`` index into the parent when this set is a true subset.
        id: Unique identity. Features are bound to it, so a feature tensor can never be
            silently reused with a different set.
    """

    coords: Tensor
    batch_offsets: Tensor | None = None
    frame: str = "camera"
    parent_id: int | None = None
    parent_index: Tensor | None = None
    id: int = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[-1] != 3:
            raise ContractError(f"PointSet.coords must be (N, 3), got {tuple(self.coords.shape)}")
        if self.batch_offsets is not None:
            offsets = self.batch_offsets
            if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(self.coords):
                raise ContractError("batch_offsets must be (B + 1,), start at 0 and end at N")
            if (offsets[1:] < offsets[:-1]).any():
                raise ContractError("batch_offsets must be non-decreasing")
        if self.parent_index is not None and len(self.parent_index) != len(self.coords):
            raise ContractError("parent_index must have one entry per point")

    def __len__(self) -> int:
        return self.coords.shape[0]

    @property
    def batch_size(self) -> int:
        return 1 if self.batch_offsets is None else len(self.batch_offsets) - 1

    @classmethod
    def from_dense(cls, coords: Tensor, *, frame: str = "camera") -> PointSet:
        """Build from ``(B, N, 3)`` clouds of equal size."""
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ContractError(f"expected (B, N, 3), got {tuple(coords.shape)}")
        batch, count, _ = coords.shape
        offsets = torch.arange(batch + 1, device=coords.device) * count
        return cls(coords.reshape(-1, 3), offsets, frame)

    @classmethod
    def cat(cls, sets: Sequence[PointSet]) -> PointSet:
        """Concatenate point sets of the same frame into one batch with a new identity."""
        if not sets:
            raise ValueError("cat() needs at least one PointSet")
        if len({s.frame for s in sets}) != 1:
            raise ContractError("cannot concatenate point sets of different reference frames")
        sizes = torch.cat([s.sizes for s in sets])
        offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
        return cls(torch.cat([s.coords for s in sets]), offsets, sets[0].frame)

    @property
    def sizes(self) -> Tensor:
        """``(B,)`` number of points of each sample."""
        if self.batch_offsets is None:
            return torch.tensor([len(self)], device=self.coords.device)
        return self.batch_offsets[1:] - self.batch_offsets[:-1]

    @property
    def batch_index(self) -> Tensor:
        """``(N,)`` sample index of every point."""
        sizes = self.sizes
        return torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), sizes)

    def dense(self) -> Tensor:
        """Return ``(B, N, 3)``; raises if samples differ in size."""
        return _dense(self.coords, self.batch_offsets, "PointSet")

    def subset(self, index: Tensor, batch_offsets: Tensor | None = None) -> PointSet:
        """A child set made of ``coords[index]``, remembering where it came from."""
        return PointSet(self.coords[index], batch_offsets, self.frame, self.id, index)

    def to(self, device: torch.device | str) -> PointSet:
        """Move to a device. Identity is preserved: it is still the same set."""
        return PointSet(
            self.coords.to(device),
            None if self.batch_offsets is None else self.batch_offsets.to(device),
            self.frame,
            self.parent_id,
            None if self.parent_index is None else self.parent_index.to(device),
            self.id,
        )


@dataclass(eq=False)
class PointFeatures:
    """Features that cannot exist without the point set they describe."""

    feats: Tensor
    points: PointSet

    def __post_init__(self) -> None:
        if self.feats.ndim != 2 or self.feats.shape[0] != len(self.points):
            raise ContractError(
                f"PointFeatures.feats must be (N, C) with N = {len(self.points)}, "
                f"got {tuple(self.feats.shape)}"
            )

    @property
    def channels(self) -> int:
        return self.feats.shape[1]

    def expect(self, points: PointSet, *, user: str) -> None:
        """Assert these features belong to ``points``; ``user`` names who is asking."""
        if self.points.id != points.id:
            relation = (
                " The features live on a subsample of that set; add an explicit propagation "
                "step instead of indexing across sets."
                if self.points.parent_id == points.id
                else ""
            )
            raise ContractError(
                f"{user} received features bound to point set #{self.points.id} but needs "
                f"features on point set #{points.id}.{relation}"
            )

    def dense(self) -> Tensor:
        """Return ``(B, N, C)``; raises if samples differ in size."""
        return _dense(self.feats, self.points.batch_offsets, "PointFeatures")

    def to(self, device: torch.device | str) -> PointFeatures:
        return PointFeatures(self.feats.to(device), self.points.to(device))


def _dense(values: Tensor, offsets: Tensor | None, what: str) -> Tensor:
    if offsets is None:
        return values.unsqueeze(0)
    sizes = offsets[1:] - offsets[:-1]
    if (sizes != sizes[0]).any():
        raise ContractError(
            f"{what}.dense() needs equally sized samples, got sizes {sizes.tolist()}"
        )
    return values.reshape(len(sizes), int(sizes[0]), values.shape[-1])
