"""Native PointHR multi-resolution features with original-point grasp adapters."""
import ast
from functools import lru_cache
import hashlib
import sys
import types
import torch
from torch import nn
from .pointhr_options import resolve

SOURCE_HASHES = {
    'pcr/models/pointhr/pointhr_semseg.py': 'dcb8ad9ad3031697794ed654e399b54daf10735999894e76357fed618d2560d8',
    'pcr/models/utils.py': '3cd9dbb698c4bf9a71cdd6f99fe6b0a0fa8de90eb4c7a3181a0ba0f80f5f0576',
}


@lru_cache(None)
def native_module():
    from ..config import ROOT
    from .ptv2 import checkpoint, interpolation
    try: import _grasppanda_pointops as ops
    except (ImportError, OSError) as error:
        raise ValueError('PointHR shared pointops are missing; run ./panda install') from error
    root = ROOT/'environments/sources/cv/pointhr'
    for name, expected in SOURCE_HASHES.items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('PointHR source is missing or differs from its pin; restore it with ./panda install')
    path = root/'pcr/models/pointhr/pointhr_semseg.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module.startswith('pcr.'))]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'pointops': alias.name = '_grasppanda_pointops'; alias.asname = 'pointops'
        if isinstance(node, ast.ClassDef):
            node.decorator_list = [d for d in node.decorator_list if 'MODELS.' not in ast.unparse(d)]
    module = types.ModuleType('_grasppanda_pointhr'); module.__file__ = str(path)
    exec(compile((root/'pcr/models/utils.py').read_text(), str(root/'pcr/models/utils.py'), 'exec'), module.__dict__)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    module.pointops = types.SimpleNamespace(knn_query=ops.knn_query, grouping=ops.grouping, interpolation=interpolation)
    module.checkpoint = checkpoint
    sys.modules[module.__name__] = module
    return module


def mean_fusion(module, args, output):
    return [[coord, features/len(output), offset] for coord, features, offset in output]


class PointHRFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, **options):
        super().__init__(); p = resolve(options); native = native_module(); self.in_channels = in_channels
        kwargs = {k: v for k, v in p.items() if k not in ('gradient_checkpointing', 'fusion', 'bn_momentum', 'bn_eps')}
        kwargs['enable_checkpoint'] = p['gradient_checkpointing']
        native_widths = [p['patch_embed_channels']] + [p['enc_channels'][-1]*2**i for i in range(4)]
        if p['dec_channels'] != native_widths[:4]: kwargs['dec_groups'] = [1]*4
        self.network = native.PointHR(in_channels=in_channels, num_classes=0, **kwargs)
        # The author constructor derives decoder widths and ignores dec_channels.
        # Rebuild only an explicitly different decoder, using the actual skip widths.
        if p['dec_channels'] != native_widths[:4]:
            rates = torch.linspace(0, p['drop_path_rate'], sum(p['dec_depths'])).tolist()
            stages = []
            for i in range(4):
                stages.append(native.Decoder(in_channels=p['dec_channels'][i+1] if i < 3 else native_widths[4],
                    skip_channels=native_widths[i], embed_channels=p['dec_channels'][i], groups=p['dec_groups'][i],
                    depth=p['dec_depths'][i], neighbours=p['dec_neighbours'][i], qkv_bias=p['attn_qkv_bias'],
                    pe_multiplier=p['pe_multiplier'], pe_bias=p['pe_bias'], attn_drop_rate=p['attn_drop_rate'],
                    drop_path_rate=rates[sum(p['dec_depths'][:i]):sum(p['dec_depths'][:i+1])],
                    enable_checkpoint=p['gradient_checkpointing'], unpool_backend=p['unpool_backend']))
            self.network.dec_stages = nn.ModuleList(stages)
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm1d): module.momentum = p['bn_momentum']; module.eps = p['bn_eps']
            if p['fusion'] == 'mean' and isinstance(module, native.MRFusionBlock): module.register_forward_hook(mean_fusion)
        self.projection = nn.Linear(p['dec_channels'][0], out_channels)

    def forward(self, xyz, features, batch):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or features.shape != (len(xyz), self.in_channels) or batch.shape != (len(xyz),) or not len(xyz):
            raise ValueError('PointHR requires aligned nonempty coordinates, features and scene IDs')
        if not xyz.is_cuda or xyz.dtype != torch.float32 or features.dtype != torch.float32 or features.device != xyz.device or batch.device != xyz.device or batch.dtype != torch.long:
            raise ValueError('PointHR requires CUDA float32 coordinates/features and int64 scene IDs')
        if not torch.isfinite(xyz).all() or not torch.isfinite(features).all() or batch.min() < 0:
            raise ValueError('PointHR requires finite inputs and nonnegative scene IDs')
        _, scene = torch.unique(batch, sorted=True, return_inverse=True); order = torch.argsort(scene, stable=True)
        with torch.cuda.device(xyz.device):
            value = self.network(dict(coord=xyz[order].contiguous(), feat=features[order].contiguous(),
                offset=torch.bincount(scene).cumsum(0).int()))
        if len(value) != len(xyz): raise ValueError('PointHR did not restore every input point')
        return self.projection(value)[torch.argsort(order)]


class PointHRBackbone(nn.Module):
    def __init__(self, **options):
        super().__init__(); self.features = PointHRFeatures(3, 256, **options)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('PointHR requires camera XYZ [B,N,3] with N >= 1024')
        points = points.contiguous(); size, count, _ = points.shape; xyz = points.reshape(-1, 3)
        batch = torch.arange(size, device=points.device).repeat_interleave(count)
        dense = self.features(xyz, xyz, batch).reshape(size, count, 256)
        indices = sample_indices(points, 1024, getattr(self, 'seed_sampling', 'upstream'), native=_ext.furthest_point_sampling, training=self.training)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256)).transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds, fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points


class SparsePointHRBackbone(nn.Module):
    def __init__(self, out_channels=512, voxel_size=.005, feature_channels=3, **options):
        super().__init__(); self.voxel_size = voxel_size
        self.features = PointHRFeatures(3+feature_channels, out_channels, **options)

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long(); xyz = coords[:, 1:].to(sparse.F.dtype)*self.voxel_size
        features = self.features(xyz, torch.cat([xyz, sparse.F], 1), coords[:, 0])
        return ME.SparseTensor(features, coordinate_map_key=sparse.coordinate_map_key, coordinate_manager=sparse.coordinate_manager)
