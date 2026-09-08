"""Shared layer factories for configurable point and neighborhood encoders."""
import math
from torch import nn


def point_mlp(channels, dimension=1, activation='relu', normalization='batch'):
    convolution = nn.Conv1d if dimension == 1 else nn.Conv2d
    batch_norm = nn.BatchNorm1d if dimension == 1 else nn.BatchNorm2d
    nonlinear = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]
    layers = []
    for a, b in zip(channels, channels[1:]):
        layers.append(convolution(a, b, 1, bias=normalization == 'none'))
        if normalization == 'batch':
            layers.append(batch_norm(b))
        elif normalization == 'group':
            layers.append(nn.GroupNorm(math.gcd(8, b), b))
        layers.append(nonlinear())
    return nn.Sequential(*layers)
