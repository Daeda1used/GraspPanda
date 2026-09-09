"""Native PTv2 grouped vector attention with original-row grasp feature adapters."""
import ast
import hashlib
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import types
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from grasppanda.modules.ptv2_options import resolve

SOURCE_HASHES = {'point_transformer_v2/point_transformer_v2m1_origin.py': '403ad195bb57f6ff01830f162c91a9e8e548f9e8d5e45f7e04347e1f7dcacb11', 'utils/misc.py': 'abb01683e69f569921400251686247d2527eecc250ecdb7d073add65bd297b8b'}


@contextmanager
def recompute_buffers(module):
    buffers = []
    try:
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                    value = getattr(child, name)
                    if value is not None:
                        buffers.append((child, name, value))
                        setattr(child, name, value.clone())
        yield
    finally:
        for child, name, value in buffers:
            setattr(child, name, value)


def checkpoint(module, *args):
    return torch_checkpoint(module, *args, use_reentrant=False,
        context_fn=lambda: (nullcontext(), recompute_buffers(module)))


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    import _grasppanda_pointops as ops
    indices, distance = ops.knn_query(k, xyz, offset, new_xyz, new_offset)
    valid = indices >= 0
    weights = valid / (distance + 1e-8)
    weights = weights / weights.sum(1, keepdim=True)
    # Missing neighbors must not gather the last point of another scene.
    return (feat[indices.long().clamp_min(0)] * weights[..., None]).sum(1)


@lru_cache(None)
def native_module():
    from ..config import ROOT
    try:
        import _grasppanda_pointops as ops
    except (ImportError, OSError) as error:
        raise ValueError('PTv2 pointops are missing or incompatible; run ./panda install') from error
    root = ROOT/'environments/sources/cv/pointcept/pointcept/models'
    path = root/'point_transformer_v2/point_transformer_v2m1_origin.py'
    if not path.is_file():
        raise ValueError('PTv2 source is missing; run ./panda install')
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('PTv2 source differs from the pinned revision; restore it with ./panda install')
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or '').startswith('pointcept.'))]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'pointops':
                    alias.name = '_grasppanda_pointops';alias.asname = 'pointops'
        if isinstance(node, ast.ClassDef) and node.name == 'PointTransformerV2':
            node.decorator_list = []
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)]
    if len(assertions) != 1:
        raise ValueError('PTv2 grouped-linear validation differs from the pinned source')
    assertions[0].op = ast.Mod()
    module = types.ModuleType('_grasppanda_ptv2')
    module.__file__ = str(path)
    namespace = module.__dict__
    exec(compile((root/'utils/misc.py').read_text(), str(root/'utils/misc.py'), 'exec'), namespace)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    namespace['checkpoint'] = checkpoint
    namespace['pointops'] = types.SimpleNamespace(knn_query=ops.knn_query, grouping=ops.grouping,
                                                  interpolation=interpolation)
    namespace['GroupedLinear'].extra_repr = lambda self: f'in_features={self.in_features}, groups={self.groups}'
    sys.modules[module.__name__] = module
    return module


class PTv2Features(nn.Module):
    def __init__(self, in_channels, out_channels, **options):
        super().__init__()
        options = resolve(options)
        self.network = native_module().PointTransformerV2(in_channels=in_channels, num_classes=0, **options)
        width = options['dec_channels'][0]
        self.projection = nn.Linear(width, out_channels)

    def forward(self, xyz, features, batch):
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or features.ndim != 2
                or len(features) != len(xyz) or batch.shape != (len(xyz),)):
            raise ValueError('PTv2 expects aligned coordinates [N,3], features [N,C] and batch IDs [N]')
        if (xyz.dtype != torch.float32 or features.dtype != torch.float32 or not xyz.is_cuda
                or features.device != xyz.device or batch.device != xyz.device or batch.dtype != torch.long):
            raise ValueError('PTv2 pointops require CUDA float32 coordinates and features')
        if not torch.isfinite(xyz).all() or not torch.isfinite(features).all():
            raise ValueError('PTv2 requires finite coordinates and features')
        order = torch.argsort(batch, stable=True)
        if not len(batch) or batch.min() < 0 or batch.max() >= len(batch):
            raise ValueError('PTv2 requires nonempty scenes with consecutive batch IDs')
        counts = batch.bincount()
        if not len(counts) or (counts == 0).any():
            raise ValueError('PTv2 requires nonempty scenes with consecutive batch IDs')
        data = dict(coord=xyz[order].contiguous(), feat=features[order].contiguous(),
                    offset=counts.cumsum(0).int())
        with torch.cuda.device(xyz.device):
            dense = self.projection(self.network(data))
        return dense.new_empty(dense.shape).index_copy(0, order, dense)


class PTv2Backbone(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.features = PTv2Features(3, 256, **options)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('PTv2 requires camera XYZ [B,N,3] with N >= 1024')
        points = points.contiguous();size, count, _ = points.shape
        batch = torch.arange(size, device=points.device).repeat_interleave(count)
        xyz = points.reshape(-1, 3)
        dense = self.features(xyz, xyz, batch).reshape(size, count, 256)
        indices = _ext.furthest_point_sampling(points, 1024)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256)).transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds, fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points


class SparsePTv2Backbone(nn.Module):
    def __init__(self, out_channels=512, voxel_size=.005, feature_channels=3, **options):
        super().__init__()
        self.voxel_size = voxel_size
        self.features = PTv2Features(3 + feature_channels, out_channels, **options)

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long()
        xyz = coords[:, 1:].to(sparse.F.dtype) * self.voxel_size
        features = self.features(xyz, torch.cat([xyz, sparse.F], 1), coords[:, 0])
        return ME.SparseTensor(features, coordinate_map_key=sparse.coordinate_map_key,
                              coordinate_manager=sparse.coordinate_manager)
