"""A point set made in a data-loader worker must not share its identity with any other set.

Features are bound to the identity of their point set, and a seed-based detector asks
"do the backbone's features live on the observation's points?" by comparing identities.
A loader worker is a copy of the main process; if it counted identities the way the main
process does, a cloud made in a worker could carry the identity of the sub-sampled set
the backbone makes later, and the detector would take the seed features for scene
features. That happened: with four workers every fourth frame of a sub-sampling method
was predicted from the wrong crop support.
"""

from __future__ import annotations

import multiprocessing

import torch
from torch.utils.data import DataLoader, Dataset

from grasppanda.structures import PointSet


class _Clouds(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> PointSet:
        return PointSet(torch.zeros(4, 3))


def _first(items: list[PointSet]) -> PointSet:
    return items[0]


def test_identities_from_loader_workers_are_unique():
    context = multiprocessing.get_context("fork")
    loader = DataLoader(
        _Clouds(), batch_size=1, num_workers=2, collate_fn=_first, multiprocessing_context=context
    )
    from_workers = [points.id for points in loader]
    in_main = [PointSet(torch.zeros(4, 3)).id for _ in range(8)]
    assert len(set(from_workers)) == len(from_workers)  # two workers do not count alike
    assert not set(from_workers) & set(in_main)  # and never like the main process


def test_identity_survives_a_device_move_and_marks_a_subset():
    points = PointSet(torch.zeros(4, 3))
    child = points.subset(torch.tensor([0, 2]))
    assert points.to("cpu").id == points.id
    assert child.id != points.id and child.parent_id == points.id
