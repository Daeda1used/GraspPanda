"""Native LitePT with reversible point mappings and selectable PointROPE kernels."""
import ast
from functools import lru_cache
import hashlib
import importlib.util
import sys
import types

import torch
from torch import nn
from grasppanda.modules.litept_options import resolve

SOURCE_HASHES = {'libs/pointrope/kernels.cu': 'c76dae3c016566a86d6915f35af3ea976e5d2858fbb91cdf6b24214c144db6ed',
 'libs/pointrope/pointrope.cpp': 'c8c3e609306da50eb15ee79b1850460879b42a8829e7a166555e134bed908675',
 'libs/pointrope/pointrope_torch.py': 'c063b084e20bb931c5d896b231b70f02f9c43df8fd697345409f9dc78f5dc419',
 'litept/model.py': '29375f64629b87fdded54fb4bf0fcb3023c67d83af938ba65bc76c749987e27c',
 'litept/serialization/__init__.py': 'e6461b37ccb5dcb24725943271259c4f74dac89d27b22becb97e48ce04c8a9b1',
 'litept/serialization/default.py': 'a6677332bb8e50916653801c610d98d5ba61f4f7a0c9529793117a80c48294bb',
 'litept/serialization/hilbert.py': '60cb8365656312bb49a7a4c45859d35c60d764b613fab53bb0d97178a9c2f25a',
 'litept/serialization/z_order.py': '81101b623cf80aedf376202d4b4cb9c5f8e74a320390bd33ee67af662e3e05a4'}


class RotaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, positions, base):
        import _grasppanda_pointrope_cuda as kernels
        ctx.save_for_backward(positions); ctx.base = base
        output = tokens.clone(memory_format=torch.contiguous_format)
        kernels.pointrope(output, positions, base, 1.)
        return output

    @staticmethod
    def backward(ctx, grad):
        import _grasppanda_pointrope_cuda as kernels
        positions, = ctx.saved_tensors
        output = grad.clone(memory_format=torch.contiguous_format)
        kernels.pointrope(output, positions, ctx.base, -1.)
        return output, None, None


class CudaPointROPE(nn.Module):
    def __init__(self, freq=100.):
        super().__init__(); self.base = freq
        try:
            import _grasppanda_pointrope_cuda
        except (ImportError, OSError) as error:
            raise ValueError('LitePT PointROPE kernels are missing or incompatible; run ./panda install') from error

    def forward(self, tokens, positions):
        if tokens.dtype != torch.float32 or not tokens.is_cuda:
            raise ValueError('LitePT PointROPE expects CUDA float32 tokens')
        if positions.dtype != torch.long or positions.device != tokens.device:
            raise ValueError('LitePT PointROPE positions must be int64 on the token device')
        return RotaryFunction.apply(tokens.transpose(1, 2).contiguous(), positions.contiguous(), self.base).transpose(1, 2)


@lru_cache(None)
def native_module():
    from ..config import ROOT
    root = ROOT/'environments/sources/cv/litept'
    for name, expected in SOURCE_HASHES.items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('LitePT source is missing or differs from the pinned revision; run ./panda install')
    try:
        import flash_attn
    except (ImportError, OSError) as error:
        raise ValueError('LitePT FlashAttention is missing or incompatible; run ./panda install') from error
    name = '_grasppanda_litept'
    package = types.ModuleType(name); package.__path__ = [str(root/'litept')]
    sys.modules[name] = package
    try:
        rope_path = root/'libs/pointrope/pointrope_torch.py'
        spec = importlib.util.spec_from_file_location(name+'.rope', rope_path)
        rope = importlib.util.module_from_spec(spec);sys.modules[spec.name] = rope;spec.loader.exec_module(rope)
        path = root/'litept/model.py';tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == 'libs.pointrope')]
        module = types.ModuleType(name+'.model');module.__file__ = str(path);module.__package__ = name
        module.PointROPE = rope.PointROPE
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        return module
    except Exception:
        for key in list(sys.modules):
            if key == name or key.startswith(name+'.'): sys.modules.pop(key, None)
        raise


class LitePTFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, voxel_size=.005, **options):
        super().__init__(); p = resolve(options); native = native_module()
        backend = p.pop('rope_backend'); pooling = p.pop('pooling')
        self.voxel_size = voxel_size
        self.order = p['order']; self.shuffle_orders = p['shuffle_orders']
        self.network = native.LitePT(in_channels=in_channels, **p)
        for module in self.network.modules():
            if isinstance(module, native.GridPooling):
                module.reduce = pooling;module.shuffle_orders = self.shuffle_orders
            elif isinstance(module, native.PointROPEAttention):
                if backend == 'cuda': module.rope = CudaPointROPE(module.rope.base)
                module.register_forward_pre_hook(self.attention_cache)
        self.projection = nn.Linear(p['dec_channels'][0] if p['dec_channels'] else p['enc_channels'][0], out_channels)

    def attention_cache(self, module, args):
        point = args[0]
        # A decoder may enable attention on a parent that used only convolution.
        if 'serialized_order' not in point:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        if point.get('_grasppanda_patch_size') != module.patch_size:
            for key in ('pad', 'unpad', 'cu_seqlens_key'): point.pop(key, None)
            point['_grasppanda_patch_size'] = module.patch_size

    def forward(self, xyz, features, batch, grid=None):
        from .sonata import voxelize
        if xyz.ndim != 2 or xyz.shape[1] != 3 or features.ndim != 2 or len(features) != len(xyz) or batch.shape != (len(xyz),):
            raise ValueError('LitePT expects aligned XYZ [N,3], features [N,C] and batch IDs [N]')
        if xyz.dtype != torch.float32 or features.dtype != torch.float32 or not xyz.is_cuda or features.device != xyz.device or batch.dtype != torch.long or batch.device != xyz.device:
            raise ValueError('LitePT expects CUDA float32 coordinates/features and int64 batch IDs on the same device')
        if not len(batch) or batch.min() < 0 or batch.max() >= len(batch) or (batch.bincount() == 0).any():
            raise ValueError('LitePT requires nonempty scenes with consecutive batch IDs')
        try:
            data, inverse = voxelize(xyz, features, batch, self.voxel_size, grid)
        except ValueError as error:
            raise ValueError(str(error).replace('PTv3', 'LitePT')) from error
        with torch.cuda.device(xyz.device):
            point = self.network(data)
        return self.projection(point.feat)[inverse]


class LitePTBackbone(nn.Module):
    def __init__(self, voxel_size=.005, **options):
        super().__init__();self.features = LitePTFeatures(3, 256, voxel_size, **options)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('LitePT requires camera XYZ [B,N,3] with N >= 1024')
        points = points.contiguous();size, count, _ = points.shape
        batch = torch.arange(size, device=points.device).repeat_interleave(count);xyz = points.reshape(-1, 3)
        dense = self.features(xyz, xyz, batch).reshape(size, count, 256)
        indices = _ext.furthest_point_sampling(points, 1024)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256)).transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds, fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points


class SparseLitePTBackbone(nn.Module):
    def __init__(self, out_channels=512, voxel_size=.005, feature_channels=3, **options):
        super().__init__();self.voxel_size = voxel_size
        self.features = LitePTFeatures(3+feature_channels, out_channels, voxel_size, **options)

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long();xyz = coords[:, 1:].to(sparse.F.dtype)*self.voxel_size
        features = self.features(xyz, torch.cat([xyz, sparse.F], 1), coords[:, 0], grid=coords[:, 1:])
        return ME.SparseTensor(features, coordinate_map_key=sparse.coordinate_map_key, coordinate_manager=sparse.coordinate_manager)
