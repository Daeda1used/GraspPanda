"""Official OA-CNNs with configurable sparse stages and reversible grasp mappings."""
import ast
from functools import lru_cache
import hashlib
import sys
import types

import torch
from torch import nn
from .oacnns_options import resolve

SOURCE_HASHES = {
    'pointcept/models/oacnns/oacnns_v1m1_base.py': '60b5a1f934f081c0523fad0ca8b74030c49e27aff52bfeb351cdb35774668fff',
    'pointcept/models/utils/misc.py': 'abb01683e69f569921400251686247d2527eecc250ecdb7d073add65bd297b8b',
}


def relation_weights(logits, cluster, mode):
    from torch_geometric.utils import scatter
    if mode == 'native':
        weights = torch.exp(logits-logits.max())
        return weights/(scatter(weights, cluster, reduce='sum', dim=0)[cluster]+1e-6)
    # Independent softmax for each receptive-field cluster and feature channel.
    shift = scatter(logits.detach(), cluster, reduce='max', dim=0)[cluster]
    weights = (logits-shift).exp()
    return weights/scatter(weights, cluster, reduce='sum', dim=0)[cluster]


def sparse_shape(grid, stages, policy):
    extent = grid.amax(0)+1
    if policy == 'stride':
        factor = 2**stages
        extent = ((extent+factor-1)//factor)*factor
    if (extent < 2**stages).any():
        raise ValueError('OA-CNNs native sparse extent is too small for its stages; use sparse_padding: stride')
    return extent.tolist()


@lru_cache(None)
def native_module():
    from ..config import ROOT
    root = ROOT/'environments/sources/cv/pointcept'
    for name, expected in SOURCE_HASHES.items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('OA-CNNs source is missing or differs from the pinned revision; run ./panda install')
    name = '_grasppanda_oacnns'
    module = types.ModuleType(name); path = root/'pointcept/models/oacnns/oacnns_v1m1_base.py'
    module.__file__ = str(path)
    utility = {}; utility_path = root/'pointcept/models/utils/misc.py'
    exec(compile(utility_path.read_text(), str(utility_path), 'exec'), utility)
    module.offset2batch = utility['offset2batch']
    module.relation_weights = relation_weights; module.sparse_shape = sparse_shape
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    for node in tree.body:
        if not isinstance(node, ast.ClassDef): continue
        node.decorator_list = []
        forward = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        if node.name == 'BasicBlock':
            loop = next(n for n in forward.body if isinstance(n, ast.For))
            start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.value) == 'torch.exp(pw - pw.max())')
            loop.body[start:start+2] = ast.parse('pw = relation_weights(pw, cluster, self.relation_normalization)').body
        elif node.name == 'DonwBlock':
            for call in ast.walk(forward):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'voxel_grid':
                    call.keywords.append(ast.keyword(arg='start', value=ast.parse('0 if self.grid_origin == "zero" else None', mode='eval').body))
        elif node.name == 'OACNNs':
            for call in ast.walk(forward):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'SparseConvTensor':
                    for kw in call.keywords:
                        if kw.arg == 'spatial_shape':
                            kw.value = ast.parse('sparse_shape(discrete_coord, self.num_stages, self.sparse_padding)', mode='eval').body
            # Retain coordinates so the adapter can verify native row correspondence.
            forward.body[-1] = ast.Return(value=ast.Name(id='x', ctx=ast.Load()))
    sys.modules[name] = module
    try:
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class OACNNFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, voxel_size=.005, **options):
        super().__init__()
        from ..module_options import validate_options
        validate_options('graspnet_baseline', 'backbone', 'oacnns', options)
        p = resolve(options); native = native_module(); self.voxel_size = voxel_size
        count = len(p['enc_channels'])
        self.network = native.OACNNs(in_channels=in_channels, num_classes=out_channels,
            **{k:p[k] for k in ('embed_channels','enc_channels','enc_depth','dec_channels','point_grid_size')},
            groups=[1]*count, enc_num_ref=[16]*count, dec_depth=[2]*count)
        network = self.network; network.sparse_padding = p['sparse_padding']
        # Native stem and decoder widths remain intact; expose actual layer counts.
        stem = list(network.stem.children())[:p['stem_depth']*3]
        for _ in range(3, p['stem_depth']):
            conv = native.spconv.SubMConv3d(p['embed_channels'], p['embed_channels'], kernel_size=3,
                padding=1, indice_key='stem', bias=False)
            conv.apply(network._init_weights)
            stem.extend([conv, nn.BatchNorm1d(p['embed_channels']), nn.ReLU()])
        if p['stem_depth'] != 3: network.stem = native.spconv.SparseSequential(*stem)
        for i, decoder in enumerate(network.dec):
            layers = list(decoder.fuse.children())[:p['decoder_layers'][i]*3]
            for _ in range(2, p['decoder_layers'][i]):
                linear = nn.Linear(p['dec_channels'][i], p['dec_channels'][i]);linear.apply(network._init_weights)
                layers.extend([linear, nn.BatchNorm1d(p['dec_channels'][i]), nn.ReLU()])
            if p['decoder_layers'][i] != 2: decoder.fuse = nn.Sequential(*layers)
        for module in network.modules():
            if isinstance(module, native.BasicBlock): module.relation_normalization = p['relation_normalization']
            elif isinstance(module, native.DonwBlock): module.grid_origin = p['grid_origin']
            elif isinstance(module, nn.BatchNorm1d): module.eps = p['bn_eps']; module.momentum = p['bn_momentum']
        if p['activation'] != 'relu':
            activation = {'gelu': nn.GELU, 'silu': nn.SiLU}[p['activation']]
            for module in list(network.modules()):
                for key, child in list(module.named_children()):
                    if isinstance(child, nn.ReLU): setattr(module, key, activation())

    def forward(self, xyz, features, batch, grid=None):
        from .sonata import voxelize
        if xyz.ndim != 2 or xyz.shape[1] != 3 or features.ndim != 2 or len(features) != len(xyz) or batch.shape != (len(xyz),):
            raise ValueError('OA-CNNs expects aligned XYZ [N,3], features [N,C] and batch IDs [N]')
        if xyz.dtype != torch.float32 or features.dtype != torch.float32 or not xyz.is_cuda or features.device != xyz.device or batch.dtype != torch.long or batch.device != xyz.device:
            raise ValueError('OA-CNNs expects CUDA float32 coordinates/features and int64 batch IDs on the same device')
        if not len(batch) or batch.min() < 0 or batch.max() >= len(batch) or (batch.bincount() == 0).any():
            raise ValueError('OA-CNNs requires nonempty scenes with consecutive batch IDs')
        try:
            data, inverse = voxelize(xyz, features, batch, self.voxel_size, grid)
        except ValueError as error:
            raise ValueError(str(error).replace('PTv3', 'OA-CNNs')) from error
        data['offset'] = data['batch'].bincount().cumsum(0)
        with torch.cuda.device(xyz.device): result = self.network(data)
        expected = torch.cat([data['batch'][:, None], data['grid_coord']], 1).int()
        if not torch.equal(result.indices, expected):
            raise ValueError('OA-CNNs sparse decoder changed input row correspondence')
        return result.features[inverse]


class OACNNBackbone(nn.Module):
    def __init__(self, voxel_size=.005, **options):
        super().__init__(); self.features = OACNNFeatures(3, 256, voxel_size, **options)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('OA-CNNs requires camera XYZ [B,N,3] with N >= 1024')
        points = points.contiguous(); size, count, _ = points.shape
        batch = torch.arange(size, device=points.device).repeat_interleave(count); xyz = points.reshape(-1, 3)
        dense = self.features(xyz, xyz, batch).reshape(size, count, 256)
        indices = _ext.furthest_point_sampling(points, 1024)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256)).transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds, fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points


class SparseOACNNBackbone(nn.Module):
    def __init__(self, out_channels=512, voxel_size=.005, feature_channels=3, **options):
        super().__init__(); self.voxel_size = voxel_size
        self.features = OACNNFeatures(3+feature_channels, out_channels, voxel_size, **options)

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long(); xyz = coords[:, 1:].to(sparse.F.dtype)*self.voxel_size
        features = self.features(xyz, torch.cat([xyz, sparse.F], 1), coords[:, 0], grid=coords[:, 1:])
        return ME.SparseTensor(features, coordinate_map_key=sparse.coordinate_map_key, coordinate_manager=sparse.coordinate_manager)
