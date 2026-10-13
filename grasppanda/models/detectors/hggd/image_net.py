"""HGGD's image network: a small residual encoder with a transposed-convolution decoder.

The encoder has the *block layout* of ResNet-34 (basic blocks, ``[3, 4, 6, 3]``) and
nothing else of it: the stem keeps its stride-2 convolution but has no pooling, every
stage halves the resolution and doubles the width, the base width is 8 instead of 64,
the input has four channels and the activations are leaky. It is trained from scratch;
ImageNet weights of ResNet-34 do not fit it.

Five maps come out of the decoder: the grasp-centre heat-map at full resolution and, at
the proposal stride, the in-plane angle classes with three regressed offsets per class.
The decoder activation at the proposal stride is also the feature map of the region
stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch import Tensor, nn

from grasppanda.structures import ContractError

from .structures import ImagePredictions

__all__ = ["ImageNet"]


class _ResidualBlock(nn.Module):
    """Two 3×3 convolutions with a shortcut; leaky activations."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))
        return self.act(residual + (x if self.shortcut is None else self.shortcut(x)))


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


class ImageNet(nn.Module):
    """Encoder-decoder over the ``(B, C, U, V)`` image; see the module docstring.

    The encoder returns five maps at strides 2 to 32. The decoder walks back up with
    five transposed convolutions; before each, the encoder map of the same stride is
    added. (At the deepest level this adds the bottleneck to itself, as the reference
    does.) The two deepest decoder steps are padded for an image height of 360, where
    the encoder rounds 45 → 23 → 12 rows.

    Args:
        in_channels: Channels of the input image.
        base_width: Channels of the stem; stage ``i`` has ``base_width * 2**(i + 1)``.
        blocks: Residual blocks per stage.
        head_channels: Hidden width of the five map heads.
        num_angles: In-plane angle classes per cell.
        stride: Stride of the angle maps and of the feature map; a power of two between
            2 and 16.
        detach_features: Hand the feature map on without gradient, so the region stage
            cannot train the image network (the reference recipe).
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
    ) -> None:
        super().__init__()
        if stride not in (2, 4, 8, 16):
            raise ValueError(f"stride must be 2, 4, 8 or 16, got {stride}")
        if len(blocks) != 4:
            raise ValueError(f"blocks must list four stages, got {list(blocks)}")
        self.stride = stride
        self.detach_features = detach_features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.LeakyReLU(),
        )
        widths = [base_width * 2 ** (stage + 1) for stage in range(4)]
        stages, channels = [], base_width
        for width, count in zip(widths, blocks):
            layers = [_ResidualBlock(channels, width, 2)]
            layers += [_ResidualBlock(width, width, 1) for _ in range(count - 1)]
            stages.append(nn.Sequential(*layers))
            channels = width
        self.stages = nn.ModuleList(stages)

        #: Decoder widths, deepest first: the encoder widths mirrored, never below the stem.
        up_widths = [max(base_width, channels // 2 ** (step + 1)) for step in range(5)]
        ups = []
        for step, width in enumerate(up_widths):
            odd = step < 2  # 12 -> 23 -> 45 rows; see the class docstring
            ups.append(_Up(channels, width, (1, 2) if odd else (1, 1), (0, 1) if odd else (0, 0)))
            channels = width
        self.ups = nn.ModuleList(ups)
        #: Index of the decoder step whose output has the proposal stride.
        self.feature_step = 4 - int(math.log2(stride))
        self.feature_channels = up_widths[self.feature_step]
        self.heatmap = _MapHead(up_widths[-1], head_channels, 1)
        self.angle = _MapHead(self.feature_channels, head_channels, num_angles)
        self.angle_offset = _MapHead(self.feature_channels, head_channels, num_angles)
        self.width_offset = _MapHead(self.feature_channels, head_channels, num_angles)
        self.depth_offset = _MapHead(self.feature_channels, head_channels, num_angles)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """He initialisation by fan-out for convolutions (not the transposed ones)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, image: Tensor) -> list[Tensor]:
        """The five encoder maps, strides 2, 4, 8, 16 and 32."""
        maps = [self.stem(image)]
        for stage in self.stages:
            maps.append(stage(maps[-1]))
        return maps

    def forward(self, image: Tensor) -> ImagePredictions:
        if image.ndim != 4:
            raise ContractError(f"expected a (B, C, U, V) image, got {tuple(image.shape)}")
        skips = self.encode(image)
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
        return ImagePredictions(
            heatmap=self.heatmap(x)[:, 0],
            angle=self.angle(at_stride),
            angle_offset=self.angle_offset(at_stride),
            depth_offset=self.depth_offset(at_stride),
            width_offset=self.width_offset(at_stride),
            features=at_stride.detach() if self.detach_features else at_stride,
        )
