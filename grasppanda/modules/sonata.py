"""Native Sonata PTv3 encoder/decoder with reversible grasp point mappings."""
import importlib.util
import sys

import torch
from torch import nn


def native_module():
    from ..config import ROOT
    name = '_grasppanda_sonata'
    if name not in sys.modules:
        path = ROOT / 'environments/sources/cv/sonata/sonata/__init__.py'
        if not path.is_file():
            raise ValueError('Sonata source is missing; run ./panda install')
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            for key in list(sys.modules):
                if key == name or key.startswith(name + '.'):
                    sys.modules.pop(key, None)
            raise
    return sys.modules[name].model


def voxelize(xyz, features, batch, voxel_size, grid=None):
    """Mean-coalesce each batch's lattice cells; retain every input row's inverse."""
    if not torch.isfinite(xyz).all() or not torch.isfinite(features).all():
        raise ValueError('PTv3 requires finite coordinates and features')
    if grid is None:
        if (xyz.detach().abs() / voxel_size).max() >= 2 ** 62:
            raise ValueError('PTv3 coordinates exceed the integer lattice range')
        grid = torch.floor(xyz.detach() / voxel_size).long()
    else:
        grid = grid.long().clone()
    # Shift the lattice origin only. Camera coordinates and supervision stay in metres.
    for value in batch.unique():
        mask = batch == value
        grid[mask] -= grid[mask].amin(0)
    if grid.max() >= 65535:
        raise ValueError('PTv3 lattice exceeds its 16-bit serialization range; increase voxel size')
    keys, inverse, counts = torch.unique(torch.cat([batch[:, None], grid], dim=1),
        dim=0, sorted=True, return_inverse=True, return_counts=True)
    def mean(values):
        total = values.new_zeros((len(keys), values.shape[1]))
        return total.index_add(0, inverse, values) / counts[:, None].to(values.dtype)
    return dict(coord=mean(xyz), feat=mean(features), grid_coord=keys[:, 1:],
                batch=keys[:, 0]), inverse


def attention_cache(module, args):
    """Native encoder parents retain caches when reused by decoder stages."""
    point = args[0]
    if point.get('_grasppanda_patch_size') != module.patch_size_max:
        for key in list(point.keys()):
            if key in ('pad', 'unpad', 'cu_seqlens_key') or key.startswith('rel_pos_'):
                point.pop(key)
        point['_grasppanda_patch_size'] = module.patch_size_max


class SonataFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, voxel_size=.005,
                 enc_depths=(3, 3, 3, 12, 3), enc_channels=(48, 96, 192, 384, 512),
                 enc_num_head=(3, 6, 12, 24, 32), enc_patch_size=(128,) * 5,
                 dec_depths=(3, 3, 3, 3), dec_channels=(96, 96, 192, 384),
                 dec_num_head=(6, 6, 12, 32), dec_patch_size=(128,) * 4,
                 stride=(2, 2, 2, 2), order='z+z-trans', pooling='max',
                 mlp_ratio=4., drop_path=.3, attn_drop=0., proj_drop=0.,
                 qkv_bias=True, pre_norm=True, shuffle_orders=True,
                 enable_rpe=False, upcast_attention=False, upcast_softmax=False,
                 layer_scale=None):
        super().__init__()
        native = native_module()
        self.voxel_size = voxel_size
        self.network = native.PointTransformerV3(in_channels=in_channels,
            enc_depths=enc_depths, enc_channels=enc_channels, enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size, dec_depths=dec_depths, dec_channels=dec_channels,
            dec_num_head=dec_num_head, dec_patch_size=dec_patch_size, stride=stride,
            order=order.split('+'), mlp_ratio=mlp_ratio, drop_path=drop_path,
            attn_drop=attn_drop, proj_drop=proj_drop, qkv_bias=qkv_bias, pre_norm=pre_norm,
            shuffle_orders=shuffle_orders, enable_rpe=enable_rpe, enable_flash=False,
            upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
            layer_scale=layer_scale)
        for module in self.network.modules():
            if isinstance(module, native.GridPooling):
                module.shuffle_orders = shuffle_orders
                module.reduce = pooling
            elif isinstance(module, native.SerializedAttention):
                module.register_forward_pre_hook(attention_cache)
        self.projection = nn.Linear(dec_channels[0], out_channels)

    def forward(self, xyz, features, batch, grid=None):
        data, inverse = voxelize(xyz, features, batch, self.voxel_size, grid)
        point = self.network(data)
        return self.projection(point.feat)[inverse]


class SonataBackbone(nn.Module):
    def __init__(self, voxel_size=.005, **options):
        super().__init__()
        self.features = SonataFeatures(3, 256, voxel_size, **options)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('PTv3 requires camera XYZ [B,N,3] with N >= 1024')
        points = points.contiguous()
        batch_size, count, _ = points.shape
        batch = torch.arange(batch_size, device=points.device).repeat_interleave(count)
        xyz = points.reshape(-1, 3)
        dense = self.features(xyz, xyz, batch).reshape(batch_size, count, 256)
        indices = _ext.furthest_point_sampling(points, 1024)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256))
        sampled = sampled.transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds,
                          fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points


class SparseSonataBackbone(nn.Module):
    def __init__(self, out_channels=512, voxel_size=.005, feature_channels=3, **options):
        super().__init__()
        self.voxel_size = voxel_size
        self.features = SonataFeatures(3 + feature_channels, out_channels, voxel_size, **options)

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long()
        xyz = coords[:, 1:].to(sparse.F.dtype) * self.voxel_size
        features = self.features(xyz, torch.cat([xyz, sparse.F], dim=1),
                                 coords[:, 0], grid=coords[:, 1:])
        return ME.SparseTensor(features, coordinate_map_key=sparse.coordinate_map_key,
                              coordinate_manager=sparse.coordinate_manager)
