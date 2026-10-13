"""The image-encoder slot: an upright image in, a short list of feature maps out.

An image encoder is the 2-D counterpart of a scene encoder. It knows nothing about
grasps: the stage that owns it (HGGD's image network, the point-painting adapter of the
seed-based family) turns its maps into heat-maps or per-point features.

Contract, for every registered encoder:

* ``forward(image)`` takes ``(B, in_channels, H, W)``, **upright** (rows are image rows),
  and returns a list of ``(B, out_channels[i], h_i, w_i)`` maps, finest first.
* ``input_layout`` says what the channels are: ``"rgb"`` is colour in ``[0, 1]``
  (the encoder normalises it itself), ``"rgbd"`` is the same followed by one depth
  channel in metres, ``"drgb"`` is HGGD's own order, depth first.
* ``out_strides[i]`` is the nominal stride of map ``i`` (``H / h_i`` rounded), and
  ``out_channels[i]`` its width. An encoder may round sizes its own way (a ViT with
  14-pixel patches drops the border); consumers resample to the size they need.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from grasppanda.registry import IMAGE_ENCODERS

__all__ = ["ImageEncoder", "InputLayout"]

InputLayout = Literal["rgb", "rgbd", "drgb"]


@IMAGE_ENCODERS.base
class ImageEncoder(nn.Module):
    """Image to feature maps; see the module docstring for the contract."""

    #: Channels of the image ``forward`` takes.
    in_channels: int
    #: Meaning of those channels.
    input_layout: InputLayout
    #: Width of every returned map, finest first.
    out_channels: tuple[int, ...]
    #: Nominal stride of every returned map, finest first.
    out_strides: tuple[int, ...]

    def forward(self, image: Tensor) -> list[Tensor]:
        raise NotImplementedError
