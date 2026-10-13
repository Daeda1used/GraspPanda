"""The image HGGD's first network sees.

A frame becomes a four-channel image ``[depth, R, G, B]`` at the network resolution, in
``(channel, u, v)`` order with ``u`` along the image width:

* colour: resized and divided by 255;
* depth: resized (in millimetres), converted to metres, shifted by the mean of the
  resized image and clipped to ``±depth_clip``.

The reference resizes with Pillow's default ``Image.resize``, a two-pass antialiased
bicubic filter that rounds colour images to 8 bits after *each* pass. A network trained
on such images is evaluated on the same arithmetic here, written in tensor form so that
it runs on any device and needs no image library: ``resize: pil_bicubic``.
``resize: torch_bicubic`` is PyTorch's antialiased bicubic interpolation, which differs
from it by the 8-bit rounding (about one grey level on a share of the pixels).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from grasppanda.structures import ContractError

__all__ = ["encode_image", "pil_bicubic_weights", "resize_colour", "resize_depth"]

#: Fractional bits of Pillow's fixed-point filter coefficients for 8-bit images.
_PRECISION_BITS = 32 - 8 - 2
#: Half-width of the bicubic kernel, in output pixels.
_BICUBIC_SUPPORT = 2.0


def _bicubic(x: float) -> float:
    """Keys' cubic convolution kernel with ``a = -0.5`` (Pillow's ``BICUBIC``)."""
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


@lru_cache(maxsize=16)
def _weights(in_size: int, out_size: int) -> tuple[tuple[int, tuple[float, ...]], ...]:
    """Per output pixel: first input pixel and the normalised filter taps."""
    scale = in_size / out_size
    filter_scale = max(scale, 1.0)
    support = _BICUBIC_SUPPORT * filter_scale
    rows = []
    for out in range(out_size):
        centre = (out + 0.5) * scale
        first = max(int(centre - support + 0.5), 0)
        last = min(int(centre + support + 0.5), in_size)
        taps = [_bicubic((x + first - centre + 0.5) / filter_scale) for x in range(last - first)]
        total = 0.0
        for tap in taps:
            total += tap
        if total != 0.0:
            taps = [tap / total for tap in taps]
        rows.append((first, tuple(taps)))
    return tuple(rows)


def pil_bicubic_weights(in_size: int, out_size: int, *, fixed_point: bool = False) -> Tensor:
    """``(out_size, in_size)`` float64 matrix of Pillow's antialiased bicubic filter.

    With ``fixed_point`` the taps are the integers Pillow uses for 8-bit images
    (coefficients scaled by ``2**22`` and rounded half away from zero).
    """
    if in_size <= 0 or out_size <= 0:
        raise ValueError(f"sizes must be positive, got {in_size} -> {out_size}")
    matrix = torch.zeros(out_size, in_size, dtype=torch.float64)
    for out, (first, taps) in enumerate(_weights(in_size, out_size)):
        values = [
            float(int(math.copysign(0.5, tap) + tap * (1 << _PRECISION_BITS)))
            if fixed_point
            else tap
            for tap in taps
        ]
        matrix[out, first : first + len(taps)] = torch.tensor(values, dtype=torch.float64)
    return matrix


def _pass_uint8(levels: Tensor, weights: Tensor, dim: int) -> Tensor:
    """One Pillow pass over 8-bit data held as float64 integers."""
    moved = levels.movedim(dim, -1)
    # Sums stay below 2**53, so float64 holds Pillow's int32 accumulator exactly.
    total = moved @ weights.T + float(1 << (_PRECISION_BITS - 1))
    out = torch.floor(total / float(1 << _PRECISION_BITS)).clamp_(0.0, 255.0)
    return out.movedim(-1, dim)


def resize_colour(levels: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize ``(..., H, W)`` 8-bit levels (integers in a float tensor) as Pillow does.

    Horizontal pass, round, vertical pass, round. Returns float64 integers in 0..255.
    """
    height, width = size
    levels = levels.double()
    if levels.shape[-1] != width:
        weights = pil_bicubic_weights(levels.shape[-1], width, fixed_point=True).to(levels.device)
        levels = _pass_uint8(levels, weights, -1)
    if levels.shape[-2] != height:
        weights = pil_bicubic_weights(levels.shape[-2], height, fixed_point=True).to(levels.device)
        levels = _pass_uint8(levels, weights, -2)
    return levels


def resize_depth(depth: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize a ``(..., H, W)`` float32 image as Pillow resizes mode ``F``.

    Each pass accumulates in double precision and stores single precision.
    """
    height, width = size
    out = depth.to(torch.float32)
    if out.shape[-1] != width:
        weights = pil_bicubic_weights(out.shape[-1], width).to(out.device)
        out = (out.double() @ weights.T).to(torch.float32)
    if out.shape[-2] != height:
        weights = pil_bicubic_weights(out.shape[-2], height).to(out.device)
        out = (out.double().movedim(-2, -1) @ weights.T).movedim(-1, -2).to(torch.float32)
    return out


def encode_image(
    rgb: Tensor,
    depth: Tensor,
    size: tuple[int, int],
    *,
    resize: Literal["pil_bicubic", "torch_bicubic"] = "pil_bicubic",
    depth_clip: float = 1.0,
) -> Tensor:
    """Build the ``(B, 4, U, V)`` network input from a batch of frames.

    Args:
        rgb: ``(B, 3, H, W)`` colour in ``[0, 1]``.
        depth: ``(B, H, W)`` metres, zero where invalid.
        size: ``(U, V)``: width and height of the network image.
        resize: See the module docstring.
        depth_clip: Bound of the mean-shifted depth channel, metres.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or depth.ndim != 3:
        raise ContractError(
            f"expected rgb (B, 3, H, W) and depth (B, H, W), got {tuple(rgb.shape)} and "
            f"{tuple(depth.shape)}"
        )
    if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
        raise ContractError(
            f"rgb {tuple(rgb.shape)} and depth {tuple(depth.shape)} are not images of the same "
            "frames at the same resolution"
        )
    width, height = size
    millimetres = depth.to(torch.float32) * 1000.0
    if resize == "pil_bicubic":
        colour = resize_colour(torch.round(rgb * 255.0), (height, width)).to(torch.float32)
        small = resize_depth(millimetres, (height, width))
    elif resize == "torch_bicubic":
        colour = F.interpolate(
            rgb * 255.0, size=(height, width), mode="bicubic", antialias=True, align_corners=False
        ).clamp(0.0, 255.0)
        small = F.interpolate(
            millimetres[:, None],
            size=(height, width),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        )[:, 0]
    else:
        raise ValueError(f"unknown resize '{resize}'")
    # True divisions (by tensors): CUDA multiplies by the reciprocal of a Python number,
    # which moves the last bit away from what the same code gives on the CPU.
    colour = colour / colour.new_tensor(255.0)
    small = small / small.new_tensor(1000.0)
    small = (small - small.mean(dim=(-2, -1), keepdim=True)).clamp(-depth_clip, depth_clip)
    image = torch.cat([small[:, None], colour], dim=1)  # (B, 4, V, U)
    return image.transpose(2, 3).contiguous()  # the network's (B, 4, U, V)
