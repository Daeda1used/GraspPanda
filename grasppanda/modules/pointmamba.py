"""Native PointMamba token features with camera-aligned grasp seed decoding."""
import ast
from functools import lru_cache, partial
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
import types

import torch
from torch import nn


_PREFIX = '_grasppanda_pointmamba'


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for original, local in (('mamba_ssm', 'mamba'), ('causal_conv1d', 'causal')):
                    if node.module == original or node.module.startswith(original + '.'):
                        node.module = _PREFIX + '.' + local + node.module[len(original):]
                # The pinned model imports the old filename; its vendored source
                # actually supplies layer_norm.py, including the native RMSNorm.
                if node.module == _PREFIX + '.mamba.ops.triton.layernorm':
                    node.module = _PREFIX + '.mamba.ops.triton.layer_norm'
            if isinstance(node, ast.Import):
                for alias in node.names:
                    renamed = {'selective_scan_cuda': '_grasppanda_pointmamba_scan',
                               'causal_conv1d_cuda': '_grasppanda_causal_conv1d'}.get(alias.name)
                    if renamed:
                        alias.asname = alias.asname or alias.name
                        alias.name = renamed
        return compile(tree, self.path, 'exec')


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, roots):
        self.roots = roots

    def find_spec(self, fullname, path=None, target=None):
        for prefix, root in self.roots.items():
            if fullname.startswith(prefix + '.'):
                relative = fullname[len(prefix)+1:].replace('.', '/')
                module, package = root/(relative + '.py'), root/relative/'__init__.py'
                file = package if package.is_file() else module
                if file.is_file():
                    return importlib.util.spec_from_file_location(fullname, file,
                        loader=_Loader(fullname, str(file)),
                        submodule_search_locations=[str(file.parent)] if file == package else None)
        return None


@lru_cache(maxsize=1)
def native_module():
    from ..config import ROOT
    from ..compat import triton_driver
    source = ROOT/'environments/sources/cv/pointmamba'
    causal = ROOT/'environments/sources/cv/causal-conv1d/causal_conv1d'
    if not (source/'models/point_mamba_scan.py').is_file() or not causal.is_dir():
        raise ValueError('PointMamba sources are missing; run ./panda install')
    try:
        importlib.import_module('_grasppanda_pointmamba_scan')
        importlib.import_module('_grasppanda_causal_conv1d')
    except ImportError as error:
        raise ValueError('PointMamba native operators are missing; run ./panda install') from error
    triton_driver()
    roots = {_PREFIX+'.models': source/'models',
             _PREFIX+'.mamba': source/'mamba/mamba_ssm', _PREFIX+'.causal': causal}
    # Skip classification datasets and language-model package initializers.
    for name, directory in {_PREFIX: source, **roots,
            _PREFIX+'.mamba.ops': roots[_PREFIX+'.mamba']/'ops',
            _PREFIX+'.mamba.ops.triton': roots[_PREFIX+'.mamba']/'ops/triton',
            _PREFIX+'.mamba.modules': roots[_PREFIX+'.mamba']/'modules'}.items():
        package = types.ModuleType(name)
        package.__path__, package.__package__ = [str(directory)], name
        sys.modules[name] = package
    sys.meta_path.insert(0, _Finder(roots))
    interface = importlib.import_module(_PREFIX+'.causal.causal_conv1d_interface')
    for name in ('causal_conv1d_fn', 'causal_conv1d_update'):
        setattr(sys.modules[_PREFIX+'.causal'], name, getattr(interface, name))
    mamba = importlib.import_module(_PREFIX+'.mamba.modules.mamba_simple')
    block = importlib.import_module(_PREFIX+'.models.block_scan')
    norm = importlib.import_module(_PREFIX+'.mamba.ops.triton.layer_norm')
    serialization = importlib.import_module(_PREFIX+'.models.serialization')
    name = _PREFIX+'.features'
    native = types.ModuleType(name)
    native.__dict__.update(torch=torch, nn=nn, math=math, partial=partial,
        Mamba=mamba.Mamba, Block=block.Block, RMSNorm=norm.RMSNorm,
        layer_norm_fn=norm.layer_norm_fn, rms_norm_fn=norm.rms_norm_fn)
    path = source/'models/point_mamba_scan.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = {'Encoder', '_init_weights', 'create_block', 'MixerModel',
                'init_OrderScale', 'apply_OrderScale'}
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
    if {node.name for node in tree.body} != selected:
        raise RuntimeError('PointMamba feature definitions differ from the pinned source')
    exec(compile(tree, str(path), 'exec'), native.__dict__)
    native.Point = serialization.Point
    sys.modules[name] = native
    return native


def serialization_indices(centers, grid_size):
    """Author Hilbert orders and their inverses, including per-batch offsets."""
    native = native_module()
    batch, count, _ = centers.shape
    grid = torch.floor(centers / grid_size).long()
    grid = grid - grid.amin(dim=1, keepdim=True)
    depth = max(1, int(grid.max()).bit_length())
    if depth > 16 or 3 * depth + batch.bit_length() > 63:
        raise ValueError('PointMamba grid_size is too small for the point-cloud extent')
    points = native.Point(grid_coord=grid.flatten(0, 1),
        batch=torch.arange(batch, device=centers.device).repeat_interleave(count))
    points.serialization(order=['hilbert', 'hilbert-trans'], depth=depth)
    return points.serialized_order, points.serialized_inverse


def restore_orders(tokens, inverse, fusion):
    """Restore each sequence independently before combining matching centers."""
    halves = tokens.chunk(2, dim=1)
    aligned = [half.flatten(0, 1)[index].reshape_as(half)
               for half, index in zip(halves, inverse)]
    return (aligned[0] + aligned[1]) * .5 if fusion == 'mean' else torch.cat(aligned, dim=-1)


class PointMambaBackbone(nn.Module):
    def __init__(self, dim=384, depth=12, num_group=128, group_size=32,
                 grid_size=.02, d_state=16, d_conv=4, expand=2, dt_rank=None,
                 rms_norm=False, drop_path=.5, dropout=0., order_fusion='mean',
                 gradient_checkpointing=False):
        super().__init__()
        native = native_module()
        self.num_group, self.group_size, self.grid_size = num_group, group_size, grid_size
        self.order_fusion, self.gradient_checkpointing = order_fusion, gradient_checkpointing
        self.encoder = native.Encoder(dim)
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dim))
        self.blocks = native.MixerModel(d_model=dim, n_layer=depth,
            ssm_cfg={'d_state':d_state, 'd_conv':d_conv, 'expand':expand,
                     'dt_rank':'auto' if dt_rank is None else dt_rank, 'use_fast_path':True},
            rms_norm=rms_norm, drop_out=dropout,
            drop_path=torch.linspace(0, drop_path, depth).tolist())
        self.OrderScale_gamma_1, self.OrderScale_beta_1 = native.init_OrderScale(dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = native.init_OrderScale(dim)
        self.projection = nn.Conv1d(dim * (2 if order_fusion == 'concat' else 1), 256, 1)

    def grouped_tokens(self, points):
        from pointnet2_ops import pointnet2_utils as ops
        from pytorch3d.ops import knn_points, knn_gather
        indices = ops.furthest_point_sample(points, self.num_group).long()
        centers = points.gather(1, indices[..., None].expand(-1, -1, 3))
        # KNN replaces the author's separate KNN_CUDA packaging. It selects
        # Euclidean neighbors before the same centered native local encoder.
        neighbors = knn_points(centers, points, K=self.group_size).idx
        groups = knn_gather(points, neighbors) - centers.unsqueeze(2)
        return centers.contiguous(), self.encoder(groups.contiguous())

    def center_features(self, centers, features):
        native = native_module()
        batch, count, channels = features.shape
        orders, inverse = serialization_indices(centers, self.grid_size)
        positions = self.pos_embed(centers).flatten(0, 1)
        scales = ((self.OrderScale_gamma_1, self.OrderScale_beta_1),
                  (self.OrderScale_gamma_2, self.OrderScale_beta_2))
        tokens = [native.apply_OrderScale(features.flatten(0, 1)[order].reshape(batch, count, channels), *scale)
                  for order, scale in zip(orders, scales)]
        pos = torch.cat([positions[order].reshape(batch, count, channels) for order in orders], dim=1)
        tokens = torch.cat(tokens, dim=1)
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(self.blocks, tokens, pos, use_reentrant=False)
        else:
            output = self.blocks(tokens, pos)
        return restore_orders(output, inverse, self.order_fusion)

    def forward(self, points, end_points=None):
        from pointnet2_ops import pointnet2_utils as ops
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < max(1024, self.num_group, self.group_size):
            raise ValueError('PointMamba requires camera XYZ [B,N,3]; N must fit 1024 seeds, groups and neighbors')
        if not points.is_cuda or points.dtype != torch.float32:
            raise ValueError('PointMamba requires CUDA float32 camera coordinates')
        if not torch.isfinite(points).all():
            raise ValueError('PointMamba coordinates must be finite')
        points = points.contiguous()
        centers, tokens = self.grouped_tokens(points)
        feature = self.center_features(centers, tokens).transpose(1, 2).contiguous()
        from .sampling import sample_indices
        indices = sample_indices(points,1024,getattr(self,"seed_sampling","upstream"),native=ops.furthest_point_sample,training=self.training)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        distance, neighbors = ops.three_nn(seeds, centers)
        weights = 1. / (distance + 1.e-8)
        weights = weights / weights.sum(-1, keepdim=True)
        sampled = self.projection(ops.three_interpolate(feature, neighbors, weights)).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds,
                          fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points
