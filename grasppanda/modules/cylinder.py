"""Configurable native cylindrical queries with learned neighborhood pooling.

This adapter exposes geometry and aggregation experiments. It is not a claim
to reproduce a named attention paper or the complete SBG architecture.
"""
import copy
import torch
from torch import nn
from .layers import point_mlp


class CylindricalAggregation(nn.Module):
    def __init__(self, native, protocol, hidden_channels=(64, 128),
                 radius_factors=(1.,), nsample=None, pooling='max',
                 activation='relu', normalization='batch'):
        super().__init__()
        self.protocol, self.pooling = protocol, pooling
        native_queries = native.groupers if protocol == 'baseline' else [native.grouper]
        self.depths = len(native_queries)
        self.groups = nn.ModuleList()
        for factor in radius_factors:
            for query in native_queries:
                group = copy.deepcopy(query)
                group.radius *= factor
                if nsample is not None:
                    group.nsample = nsample
                self.groups.append(group)
        in_channels = native.in_dim if protocol == 'baseline' else native.in_dim + 3
        self.encoder = point_mlp([in_channels, *hidden_channels, 256], 2, activation, normalization)
        self.attention = nn.Conv2d(256, 1, 1) if pooling == 'attention' else None
        self.fusion = point_mlp([256*len(radius_factors), 256], 2, activation, normalization)

    def _encode(self, grouped, group):
        return self.encoder(grouped)

    def forward(self, seeds, inputs, rotations):
        scales = []
        for offset in range(0, len(self.groups), self.depths):
            depths = []
            for group in self.groups[offset:offset+self.depths]:
                grouped = (group(inputs, seeds, rotations) if self.protocol == 'baseline'
                           else group(seeds, seeds, rotations, inputs))
                encoded = self._encode(grouped, group)
                if self.pooling == 'max':
                    pooled = encoded.amax(-1)
                elif self.pooling == 'mean':
                    pooled = encoded.mean(-1)
                else:
                    pooled = (encoded * self.attention(encoded).softmax(-1)).sum(-1)
                depths.append(pooled)
            scales.append(torch.stack(depths, dim=-1))
        output = self.fusion(torch.cat(scales, dim=1))
        return output if self.protocol == 'baseline' else output.squeeze(-1)
