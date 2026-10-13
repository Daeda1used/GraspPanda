"""HGGD's image network: an image encoder with a decoder and five map heads.

The reference encoder (:class:`~grasppanda.models.image_encoders.HGGDResidualEncoder`)
has the *block layout* of ResNet-34 (basic blocks, ``[3, 4, 6, 3]``) and nothing else of
it: the stem keeps its stride-2 convolution but has no pooling, every stage halves the
resolution and doubles the width, the base width is 8 instead of 64, the input has four
channels and the activations are leaky. It is trained from scratch; ImageNet weights of
ResNet-34 do not fit it. Its decoder is the reference's chain of transposed convolutions.

Any other registered image encoder can take its place (``model.image_encoder``). Its
maps then go through a feature-pyramid decoder that does not depend on the encoder's
strides or widths: every map is projected to one width, merged top-down, resampled to
the proposal stride (this is the feature map of the region stage) and brought to full
resolution by transposed convolutions for the heat-map.

Five maps come out of the decoder: the grasp-centre heat-map at full resolution and, at
the proposal stride, the in-plane angle classes with three regressed offsets per class.
The decoder activation at the proposal stride is also the feature map of the region
stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.image_encoders import HGGDResidualEncoder, ImageEncoder
from grasppanda.structures import ContractError

from .structures import ImagePredictions

__all__ = ["ImageNet"]


class _Up(nn.Module):
    """A stride-2 transposed convolution, batch norm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: tuple[int, int] = (1, 1),
        output_padding: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, 4, 2, padding, output_padding, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class _MapHead(nn.Module):
    """5×5 convolution, batch norm, ReLU, then a 1×1 convolution to the output maps."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden, 5, 1, 2)
        self.norm = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU()
        self.out = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.act(self.norm(self.conv(x))))


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class _PyramidDecoder(nn.Module):
    """Top-down merge of any list of maps, read out at one stride and at full resolution.

    Args:
        in_channels: Width of every encoder map, finest first.
        channels: Width of the merged pyramid and of the map at the proposal stride.
        stride: The proposal stride, a power of two.
    """

    def __init__(self, in_channels: Sequence[int], channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.lateral = nn.ModuleList(_ConvNormAct(width, channels, 1) for width in in_channels)
        self.smooth = nn.ModuleList(_ConvNormAct(channels, channels, 3) for _ in in_channels)
        self.at_stride = _ConvNormAct(channels, channels, 3)
        ups, width = [], channels
        for _ in range(int(math.log2(stride))):
            ups.append(_Up(width, max(8, width // 2)))
            width = max(8, width // 2)
        self.ups = nn.Sequential(*ups)
        self.out_channels = width

    def forward(self, maps: Sequence[Tensor], size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """``size`` is the image size; returns the map at the stride and at full resolution."""
        x = self.smooth[-1](self.lateral[-1](maps[-1]))
        for level in range(len(maps) - 2, -1, -1):
            lateral = self.lateral[level](maps[level])
            x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[level](x + lateral)
        target = (size[0] // self.stride, size[1] // self.stride)
        if x.shape[-2:] != target:
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        at_stride = self.at_stride(x)
        return at_stride, self.ups(at_stride)


class ImageNet(nn.Module):
    """Encoder-decoder over the ``(B, 4, U, V)`` image; see the module docstring.

    With the reference encoder: it returns five maps at strides 2 to 32. The decoder
    walks back up with five transposed convolutions; before each, the encoder map of the
    same stride is added. (At the deepest level this adds the bottleneck to itself, as
    the reference does.) The two deepest decoder steps are padded for an image height of
    360, where the encoder rounds 45 → 23 → 12 rows.

    With another encoder: the image is turned upright (``(B, C, V, U)``), re-ordered to
    the encoder's ``input_layout`` (colour first, depth last or dropped), encoded, and
    the maps are turned back before the pyramid decoder.

    Args:
        in_channels: Channels of the input image.
        base_width: Channels of the stem of the reference encoder; stage ``i`` has
            ``base_width * 2**(i + 1)``.
        blocks: Residual blocks per stage of the reference encoder.
        head_channels: Hidden width of the five map heads.
        num_angles: In-plane angle classes per cell.
        stride: Stride of the angle maps and of the feature map; a power of two between
            2 and 16.
        detach_features: Hand the feature map on without gradient, so the region stage
            cannot train the image network (the reference recipe).
        encoder: A built image encoder to use instead of the reference encoder.
        decoder_channels: Width of the pyramid decoder of such an encoder (and of the
            feature map of the region stage).
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_width: int = 8,
        blocks: Sequence[int] = (3, 4, 6, 3),
        head_channels: int = 32,
        num_angles: int = 6,
        stride: int = 8,
        detach_features: bool = True,
        encoder: ImageEncoder | None = None,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        if stride not in (2, 4, 8, 16):
            raise ValueError(f"stride must be 2, 4, 8 or 16, got {stride}")
        if len(blocks) != 4:
            raise ValueError(f"blocks must list four stages, got {list(blocks)}")
        self.stride = stride
        self.detach_features = detach_features
        if encoder is None:
            encoder = HGGDResidualEncoder(in_channels, base_width, blocks)
        self.reference = isinstance(encoder, HGGDResidualEncoder)
        self.input_layout = encoder.input_layout
        if self.reference:
            if encoder.in_channels != in_channels:
                raise ContractError(
                    f"ImageNet: the encoder takes {encoder.in_channels} channels, the image "
                    f"has {in_channels}"
                )
            # The reference encoder keeps its parameter names (`stem`, `stages`), so
            # converted reference checkpoints load unchanged.
            self.stem = encoder.stem
            self.stages = encoder.stages
            base_width = encoder.base_width
            channels = encoder.out_channels[-1]
            #: Decoder widths, deepest first: the encoder widths mirrored, never below the stem.
            up_widths = [max(base_width, channels // 2 ** (step + 1)) for step in range(5)]
            ups = []
            for step, width in enumerate(up_widths):
                odd = step < 2  # 12 -> 23 -> 45 rows; see the class docstring
                ups.append(
                    _Up(channels, width, (1, 2) if odd else (1, 1), (0, 1) if odd else (0, 0))
                )
                channels = width
            self.ups = nn.ModuleList(ups)
            #: Index of the decoder step whose output has the proposal stride.
            self.feature_step = 4 - int(math.log2(stride))
            self.feature_channels = up_widths[self.feature_step]
            heat_channels = up_widths[-1]
        else:
            if encoder.input_layout not in ("rgb", "rgbd"):
                raise ContractError(
                    f"ImageNet: {type(encoder).__name__} declares the input layout "
                    f"'{encoder.input_layout}'; HGGD can serve 'rgb' and 'rgbd'"
                )
            self.encoder = encoder
            self.decoder = _PyramidDecoder(encoder.out_channels, decoder_channels, stride)
            self.feature_channels = decoder_channels
            heat_channels = self.decoder.out_channels
        self.heatmap = _MapHead(heat_channels, head_channels, 1)
        self.angle = _MapHead(self.feature_channels, head_channels, num_angles)
        self.angle_offset = _MapHead(self.feature_channels, head_channels, num_angles)
        self.width_offset = _MapHead(self.feature_channels, head_channels, num_angles)
        self.depth_offset = _MapHead(self.feature_channels, head_channels, num_angles)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """He initialisation by fan-out for convolutions (not the transposed ones).

        A foreign encoder keeps the weights it came with (pretrained or its own
        initialisation); only the decoder and the heads are initialised here.
        """
        foreign = set() if self.reference else set(self.encoder.modules())
        for module in self.modules():
            if module in foreign:
                continue
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, image: Tensor) -> list[Tensor]:
        """The encoder maps, finest first, in the ``(B, C, u, v)`` layout of the image.

        The reference encoder returns five, at strides 2, 4, 8, 16 and 32.
        """
        if self.reference:
            maps = [self.stem(image)]
            for stage in self.stages:
                maps.append(stage(maps[-1]))
            return maps
        upright = image.transpose(-1, -2)  # (B, [depth, R, G, B], V, U): rows are image rows
        colour = upright[:, 1:4]
        if self.input_layout == "rgbd":
            colour = torch.cat([colour, upright[:, :1]], dim=1)
        return [m.transpose(-1, -2) for m in self.encoder(colour)]

    def forward(self, image: Tensor) -> ImagePredictions:
        if image.ndim != 4:
            raise ContractError(f"expected a (B, C, U, V) image, got {tuple(image.shape)}")
        skips = self.encode(image)
        if not self.reference:
            if image.shape[-2] % self.stride or image.shape[-1] % self.stride:
                raise ContractError(
                    f"ImageNet: the image {tuple(image.shape[2:])} is not a multiple of the "
                    f"proposal stride {self.stride}"
                )
            at_stride, x = self.decoder(skips, (image.shape[-2], image.shape[-1]))
            return self._predictions(x, at_stride)
        x = skips[-1]
        at_stride = None
        for step, up in enumerate(self.ups):
            skip = skips[4 - step]
            if x.shape != skip.shape:
                raise ContractError(
                    f"ImageNet: decoder map {tuple(x.shape[2:])} does not meet the encoder map "
                    f"{tuple(skip.shape[2:])} of an image of {tuple(image.shape[2:])}. The "
                    "decoder is laid out for 640 x 360 images (and sizes that halve the same "
                    "way: the two deepest maps must have an even first and an odd second side)."
                )
            x = up(x + skip)
            if step == self.feature_step:
                at_stride = x
        assert at_stride is not None
        return self._predictions(x, at_stride)

    def _predictions(self, x: Tensor, at_stride: Tensor) -> ImagePredictions:
        return ImagePredictions(
            heatmap=self.heatmap(x)[:, 0],
            angle=self.angle(at_stride),
            angle_offset=self.angle_offset(at_stride),
            depth_offset=self.depth_offset(at_stride),
            width_offset=self.width_offset(at_stride),
            features=at_stride.detach() if self.detach_features else at_stride,
        )
