"""Base class of dataset readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import Dataset

from grasppanda.registry import DATASETS
from grasppanda.structures import Batch, SampleKey, collate

__all__ = ["BaseDataset"]


@DATASETS.base
class BaseDataset(Dataset):
    """A dataset addressed by :class:`~grasppanda.structures.SampleKey`.

    ``dataset[i]`` returns a :class:`~grasppanda.structures.Batch` holding one frame;
    its ``supervision`` is ``None`` unless the dataset was built with labels. Readers
    never write into the dataset tree: derived labels live in the asset store.
    """

    #: Dataset name used in :class:`SampleKey` and in result records.
    name: str = "unspecified"

    @property
    def keys(self) -> Sequence[SampleKey]:
        """Identity of every frame, in index order."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError

    #: Merges frames into a batch; point clouds may differ in size.
    collate = staticmethod(collate)

    @classmethod
    def served_supervision(cls, node: Mapping[str, Any]) -> frozenset[str] | None:
        """Fields of ``Supervision`` in the batches of a dataset configured with ``node``.

        Named as detectors name what they read (``needs_supervision``): ``objects``,
        ``objects.tolerance``, ``point_labels.<name>``, ``extras.<name>``. The config
        check refuses a training dataset that does not serve a field its model reads.
        ``None``, the default, means that the reader does not declare what it serves;
        nothing is compared then.
        """
        return None

    @classmethod
    def supervision_hint(cls, field: str, node: Mapping[str, Any]) -> str:
        """What ``node`` would have to say for the dataset to serve ``field``: a sentence
        for the message of the config check, or nothing when this reader cannot say."""
        return ""
