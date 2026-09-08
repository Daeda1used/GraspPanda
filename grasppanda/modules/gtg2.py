"""Native GtG2 graph regressor with compatible graph-convolution substitutions."""
import importlib.util

import torch
from torch import nn

from ..config import ROOT, catalogue


def native_module():
    path = ROOT / catalogue()['gtg2']['path'] / 'Training/models.py'
    spec = importlib.util.spec_from_file_location('_grasppanda_gtg2_model', path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


class GraphRegressor(nn.Module):
    def __init__(self, encoder, graph):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, SAGEConv
        source = native_module()
        width, layers = encoder['hidden'], encoder['layers']
        self.block = source.GNNBlock(4 if graph['encoding'] == 'binary' else 5, width, layers)
        if encoder['type'] == 'gtg_gatv2':
            self.block.convs = nn.ModuleList(GATv2Conv(
                width, width // encoder['heads'], heads=encoder['heads'], dropout=encoder['dropout'],
                add_self_loops=encoder['add_self_loops'], share_weights=encoder['share_weights'])
                for _ in range(layers))
        elif encoder['aggregation'] != 'max':
            self.block.convs = nn.ModuleList(SAGEConv(width, width, aggr=encoder['aggregation']) for _ in range(layers))

    def forward(self, data, native_augmentation=False):
        # Augmentation belongs to the graph dataset. The source block's in-place
        # half-turn would otherwise mutate shared inputs or augment twice.
        if native_augmentation:
            return self.block(data.clone())
        import torch.nn.functional as F
        from torch_geometric.nn import global_max_pool
        x = self.block.pre(data.x)
        for conv, bn in zip(self.block.convs, self.block.bns):
            x = F.relu(bn(conv(x, data.edge_index)))
        return self.block.head(global_max_pool(self.block.att(x), data.batch))


def build_graph(inside, outside, score, options, augmentation=None, seed=0):
    import numpy as np
    import fpsample
    from torch_geometric.data import Data
    from torch_geometric.nn import knn_graph
    from torch_geometric.utils import to_undirected
    rng = np.random.default_rng(seed)
    aug = augmentation or {}
    mode = aug.get('mode', 'none' if augmentation is None else 'custom')
    flip = aug.get('half_turn_probability', .5) if mode == 'custom' else 0.
    dropout = aug.get('point_dropout', 0.) if mode == 'custom' else 0.
    def sample(points):
        points = np.asarray(points, dtype=np.float32).copy()
        if dropout and len(points):
            count = max(min(len(points), options['k'] + 1), int(len(points) * (1 - dropout)))
            points = points[rng.choice(len(points), count, replace=False)]
        if len(points) > options['max_points']:
            points = points[fpsample.fps_sampling(points, options['max_points'], start_idx=0)]
        return points
    a = sample(inside)
    b = sample(outside) if options['include_outside'] else np.empty((0, 3), dtype=np.float32)
    pos = np.concatenate([a, b])
    if len(pos) <= options['k'] or not np.isfinite(pos).all(): raise ValueError('Invalid candidate graph points')
    if rng.random() < flip: pos[:, 1:] *= -1
    pos = torch.from_numpy(pos)
    flag = torch.cat([torch.ones(len(a)), torch.zeros(len(b))])
    types = flag[:, None] if options['encoding'] == 'binary' else torch.stack([flag, 1 - flag], dim=1)
    edge = to_undirected(knn_graph(pos, k=options['k'], loop=False), num_nodes=len(pos))
    return Data(x=torch.cat([pos, types], dim=1), pos=pos, edge_index=edge,
                y=torch.tensor([[score]], dtype=torch.float32))
