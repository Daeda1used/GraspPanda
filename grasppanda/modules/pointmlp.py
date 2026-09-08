"""PointMLP's native point hierarchy and decoder with an XYZ grasp adapter."""
import importlib.util
import sys

import torch
from torch import nn


def native_module():
    from ..config import ROOT
    name = '_grasppanda_native_pointmlp'
    if name not in sys.modules:
        path = ROOT / 'environments/sources/cv/pointmlp/part_segmentation/model/pointMLP.py'
        if not path.is_file():
            raise ValueError('PointMLP source is missing; run the component installer')
        # Require the installed operator; do not enter the author's fallback
        # JIT build with its hard-coded legacy GPU architecture list.
        import pointnet2_ops._ext
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


class PointMLPBackbone(nn.Module):
    def __init__(self, embed_dim=64, dim_expansion=(2, 2, 2, 2),
                 pre_blocks=(2, 2, 2, 2), pos_blocks=(2, 2, 2, 2),
                 k_neighbors=(32, 32, 32, 16), stage_points=(1024, 256, 64, 16),
                 decoder_channels=(512, 256, 128, 128), decoder_blocks=(2, 2, 2, 2),
                 res_expansion=1., activation='relu', normalize='anchor'):
        super().__init__()
        native = native_module()
        self.embedding = native.ConvBNReLU1D(3, embed_dim, activation=activation)
        self.groupers = nn.ModuleList()
        self.pre_blocks = nn.ModuleList()
        self.pos_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.stage_points, self.k_neighbors = tuple(stage_points), tuple(k_neighbors)
        channels = [embed_dim]
        for i in range(4):
            out = channels[-1] * dim_expansion[i]
            self.groupers.append(native.LocalGrouper(channels[-1], stage_points[i],
                k_neighbors[i], use_xyz=True, normalize=normalize))
            self.pre_blocks.append(native.PreExtraction(channels[-1], out, pre_blocks[i],
                res_expansion=res_expansion, activation=activation, use_xyz=True))
            self.pos_blocks.append(native.PosExtraction(out, pos_blocks[i],
                res_expansion=res_expansion, activation=activation))
            channels.append(out)
        previous = channels[-1]
        for skip, out, blocks in zip(reversed(channels[:-1]), decoder_channels, decoder_blocks):
            self.decoder.append(native.PointNetFeaturePropagation(previous + skip, out,
                blocks=blocks, res_expansion=res_expansion, activation=activation))
            previous = out
        self.projection = nn.Conv1d(previous, 256, 1)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('PointMLP requires camera XYZ [B,N,3] with N >= 1024')
        sizes = (points.shape[1], *self.stage_points[:-1])
        if any(n > size or k > size for n, k, size in zip(self.stage_points, self.k_neighbors, sizes)):
            raise ValueError('PointMLP stage points and neighbors must fit each input stage')
        points = points.contiguous()
        xyz = points
        feature = self.embedding(points.transpose(1, 2))
        positions, features = [xyz], [feature]
        for group, pre, post in zip(self.groupers, self.pre_blocks, self.pos_blocks):
            xyz, local = group(xyz, feature.transpose(1, 2))
            feature = post(pre(local))
            positions.append(xyz)
            features.append(feature)
        # Native propagation restores dense features in the original row order.
        for i, decoder in enumerate(self.decoder):
            target = len(positions) - 2 - i
            feature = decoder(positions[target], positions[target + 1], features[target], feature)
        dense = self.projection(feature)
        indices = _ext.furthest_point_sampling(points, 1024)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(2, indices.long()[:, None, :].expand(-1, 256, -1)).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds,
                          fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points
