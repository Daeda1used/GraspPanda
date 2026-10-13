"""Base class of dataset readers."""

from __future__ import annotations

from collections.abc import Sequence

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
