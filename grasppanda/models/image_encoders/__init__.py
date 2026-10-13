"""Image encoders: the slot contract, HGGD's reference encoder and the library bridges."""

from .base import ImageEncoder, InputLayout
from .bridges import HFImageEncoder, TimmImageEncoder, TorchvisionImageEncoder
from .hggd_residual import HGGDResidualEncoder

__all__ = [
    "HFImageEncoder",
    "HGGDResidualEncoder",
    "ImageEncoder",
    "InputLayout",
    "TimmImageEncoder",
    "TorchvisionImageEncoder",
]
