"""The encoder of HGGD's reference image network.

It has the *block layout* of ResNet-34 (basic blocks, ``[3, 4, 6, 3]``) and nothing else
of it: the stem keeps its stride-2 convolution but has no pooling, every stage halves the
resolution and doubles the width, the base width is 8 instead of 64, the input has four
channels and the activations are leaky. It is trained from scratch; ImageNet weights of
ResNet-34 do not fit it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch import Tensor, nn

from grasppanda.registry import IMAGE_ENCODERS

from .base import ImageEncoder

__all__ = ["HGGDResidualEncoder"]


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


@IMAGE_ENCODERS.register()
class HGGDResidualEncoder(ImageEncoder):
    """HGGD's own image encoder: ResNet-34 block layout at base width 8, four input channels.

    Five maps at strides 2, 4, 8, 16 and 32. The input is HGGD's ``[depth, R, G, B]``
    image as its detector lays it out (``(B, 4, U, V)`` with ``u`` along the width); the
    network is trained from scratch on it, so nothing is normalised or re-ordered. Built
    alone, the weights have PyTorch's default initialisation; HGGD's image network
    applies the He initialisation of the reference to it (:meth:`reset_parameters`).

    Args:
        in_channels: Channels of the input image.
        base_width: Channels of the stem; stage ``i`` has ``base_width * 2**(i + 1)``.
        blocks: Residual blocks per stage.
    """

    status = "tested"
    source = "HGGD (Chen et al., RA-L 2023)"
    extras = ()
    input_layout = "drgb"

    def __init__(
        self,
        in_channels: int = 4,
        base_width: int = 8,
        blocks: Sequence[int] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        if len(blocks) != 4:
            raise ValueError(f"blocks must list four stages, got {list(blocks)}")
        self.in_channels = in_channels
        self.base_width = base_width
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
        self.out_channels = (base_width, *widths)
        self.out_strides = (2, 4, 8, 16, 32)

    def reset_parameters(self) -> None:
        """He initialisation by fan-out, as the reference applies it."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: Tensor) -> list[Tensor]:
        maps = [self.stem(image)]
        for stage in self.stages:
            maps.append(stage(maps[-1]))
        return maps
