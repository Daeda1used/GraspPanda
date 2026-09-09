"""Native Flash3D with reversible feature mappings and isolated scene attention."""
from functools import lru_cache, partial
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn

from .flash3d_options import resolve


def _save_precision_state(module, state, prefix, _metadata):
    """Encode TE's empty BF16 metadata as a weights-only compatible tensor."""
    import io
    value = state[prefix+'_extra_state']
    if not isinstance(value, io.BytesIO) or torch.load(io.BytesIO(value.getvalue()), map_location='cpu', weights_only=True) is not None:
        raise ValueError('Flash3D checkpointing requires BF16 without FP8 calibration state')
    state[prefix+'_extra_state'] = torch.empty(0, dtype=torch.uint8)


def _load_precision_state(module, state, prefix, _metadata, _strict, _missing, _unexpected, _errors):
    key = prefix+'_extra_state'
    if key not in state:
        return
    value = state[key]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.shape != (0,):
        raise ValueError('Flash3D requires its tensor-only BF16 precision metadata')
    state[key] = None


@lru_cache(None)
def native_module(record=None):
    from ..config import ROOT
    root = ROOT/'environments/native/flash3d'
    try:
        state = json.loads((Path(record) if record is not None else root/'state.json').read_text())
        if not state.get('files'):
            raise ValueError('Native build record has no source or binary hashes')
        for key in ('source', 'extension', 'engine_path'):
            if key in state and not (root/state[key]).resolve().is_relative_to(root.resolve()):
                raise ValueError('Native paths must remain inside their installation')
        required = [state['extension'], state.get('source', 'source/flash3dxfmr')+'/layers/flash3d.py']
        if 'engine_path' in state:
            required.append(state['engine_path']+'/transformer_engine/__init__.py')
        if any(name not in state['files'] for name in required):
            raise ValueError('Native source, engine and binary must be covered by the build record')
        if state['torch'] != torch.__version__ or state['python'] != list(sys.version_info[:2]):
            raise ValueError('Interpreter or PyTorch changed')
        if state['abi'] != int(torch._C._GLIBCXX_USE_CXX11_ABI):
            raise ValueError('PyTorch C++ ABI changed')
        if list(torch.cuda.get_device_capability()) not in state['capabilities']:
            raise ValueError('This GPU architecture has not been compiled')
        for name, expected in state['files'].items():
            path = root/name
            if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Native source or binary differs from its build record')
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        raise ValueError('Flash3D is missing or incompatible; run ./panda install with the selected GPU') from error
    name = '_grasppanda_flash3d'
    try:
        from ..compat import triton_driver
        triton_driver()
        if 'engine_path' in state:
            engine_path = root/state['engine_path']
            existing = sys.modules.get('transformer_engine')
            if existing is not None and not Path(existing.__file__).resolve().is_relative_to(engine_path.resolve()):
                raise ValueError('Another Transformer Engine is already imported; start a fresh experiment worker')
            sys.path.insert(0, str(engine_path))
        import transformer_engine.pytorch
        if 'engine_path' in state and not Path(transformer_engine.pytorch.__file__).resolve().is_relative_to(engine_path.resolve()):
            raise ValueError('Transformer Engine must load from the matching native build')
        for suffix, directory in (('', ''), ('.lib', '/lib'), ('.layers', '/layers')):
            package = types.ModuleType(name+suffix)
            package.__path__ = [str(root/(state.get('source', 'source/flash3dxfmr')+directory))]
            sys.modules[package.__name__] = package
        binary = root/state['extension']
        spec = importlib.util.spec_from_file_location(name+'.lib.pshattn', binary)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules[spec.name] = native
        sys.modules[name+'.lib'].pshattn = native
        return types.SimpleNamespace(
            model=importlib.import_module(name+'.layers.flash3d'),
            stage=importlib.import_module(name+'.layers.stage'),
            scopes=importlib.import_module(name+'.psh.bucket_scope'),
            scatter=importlib.import_module(name+'.psh.psh_main').batch_bucket_scatter,
            device=importlib.import_module(name+'.psh.dev_context'),
            te=transformer_engine.pytorch)
    except Exception:
        for key in list(sys.modules):
            if key == name or key.startswith(name+'.'): sys.modules.pop(key, None)
        raise


class ScopePlan:
    def __init__(self, native, bucket, size, kind, shift, stride):
        self.native, self.bucket, self.size = native, bucket, size
        self.kind, self.shift, self.stride = kind, shift, stride

    def gen_scopes_from_plan(self, count, device):
        alignment = self.bucket*self.size*(self.stride if self.kind == 'strided' else 1)
        if count % alignment:
            raise ValueError('Flash3D point count does not satisfy the selected attention scopes')
        if self.kind == 'strided':
            scopes = self.native.generate_stride_scopes(count, self.bucket, self.size, self.stride)
        else:
            scopes = self.native.generate_swin_scopes(count, self.bucket, self.size,
                self.shift if self.kind == 'shifted' else 0)
        if sorted(i for scope in scopes for i in scope) != list(range(count//self.bucket)):
            raise ValueError('Flash3D scopes must partition every bucket exactly once')
        return torch.tensor(scopes, device=device, dtype=torch.uint32)


class Flash3DFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, *, _native=None, **options):
        super().__init__()
        from ..module_options import validate_options
        validate_options('graspnet_baseline', 'backbone', 'flash3d', options)
        p = resolve(options)
        self.settings = p
        self.in_channels, self.out_channels = in_channels, out_channels
        self.native = native_module() if _native is None else _native
        self.alignment = p['alignment']
        self.minimum_points = max(self.alignment, 32*p['bucket_size'])
        norm = partial(getattr(self.native.te, 'LayerNorm' if p['normalization'] == 'layer' else 'RMSNorm'), eps=p['norm_eps'])
        counts = {'enc': 0, 'dec': 0}
        levels = []
        for level, width in enumerate(p['channels']):
            specs = {}
            for prefix in ('enc', 'dec'):
                specs[prefix] = []
                for _ in range(p[prefix+'_depths'][level]):
                    i = counts[prefix]
                    plan = ScopePlan(self.native.scopes, p['bucket_size'], p[prefix+'_scope_size'][i],
                        p[prefix+'_scope_plan'][i], p[prefix+'_scope_shift'][i], p[prefix+'_scope_stride'][i])
                    specs[prefix].append(self.native.stage.XFMRSpecs(width,
                        int(width*p[prefix+'_mlp_ratio'][i]), p[prefix+'_heads'][i], p[prefix+'_qkv_bias'][i], plan, norm))
                    counts[prefix] += 1
            levels.append(self.native.model.F3DLevelSpecs(specs['enc'], specs['dec'],
                p['pooling'][level] if level < len(p['pooling']) else 'mean',
                1, norm, nn.GELU, norm, nn.GELU))
        self.network = self.native.model.Flash3D(levels, p['bucket_size'], 1, p['hash_type'])
        if in_channels != 3:
            self.network.psh_scatter.lin = self.native.te.Linear(in_channels, p['channels'][0])
        self.projection = nn.Linear(p['channels'][0], out_channels)
        counts = {'enc': 0, 'dec': 0}
        stage = self.network.module_tree
        for level, width in enumerate(p['channels']):
            for prefix, blocks in (('enc', stage.encoder_blocks), ('dec', stage.decoder_blocks)):
                for block in blocks:
                    i = counts[prefix]
                    block.mlp.mlp = self.native.te.LayerNormMLP(width, int(width*p[prefix+'_mlp_ratio'][i]),
                        activation=p[prefix+'_mlp_activation'][i],
                        normalization='LayerNorm' if p['normalization'] == 'layer' else 'RMSNorm', eps=p['norm_eps'])
                    dropout = p[prefix+'_residual_dropout'][i]
                    if dropout:
                        block.bwa.proj = nn.Sequential(block.bwa.proj, nn.Dropout(dropout))
                        block.mlp.mlp = nn.Sequential(block.mlp.mlp, nn.Dropout(dropout))
                    counts[prefix] += 1
            if level+1 < len(p['channels']): stage = stage.middle.submodule
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
        for module in self.network.modules():
            if isinstance(module, TransformerEngineBaseModule):
                module.register_state_dict_post_hook(_save_precision_state)
                module.register_load_state_dict_pre_hook(_load_precision_state)

    def forward(self, xyz, features, batch, grid=None):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or features.shape != (len(xyz), self.in_channels) or batch.shape != (len(xyz),):
            raise ValueError('Flash3D requires aligned XYZ, input features and scene IDs')
        if not xyz.is_cuda or xyz.dtype != torch.float32 or features.dtype != torch.float32 or batch.dtype != torch.long or features.device != xyz.device or batch.device != xyz.device:
            raise ValueError('Flash3D requires CUDA float32 coordinates/features and int64 scene IDs on the same device')
        if not len(batch) or batch.min() < 0 or batch.max() >= len(batch) or (batch.bincount() == 0).any():
            raise ValueError('Flash3D requires nonempty scenes with consecutive IDs')
        if not torch.isfinite(xyz).all() or not torch.isfinite(features).all():
            raise ValueError('Flash3D input coordinates and features must be finite')
        output = features.new_zeros((len(xyz), self.out_channels))
        with torch.cuda.device(xyz.device):
            self.native.device.init_dev(xyz.device.index)
            for scene in range(int(batch.max())+1):
                rows = (batch == scene).nonzero().flatten()
                count = len(rows)
                padded = max(count, self.minimum_points)
                padded = (padded+self.alignment-1)//self.alignment*self.alignment
                repeat = torch.arange(padded, device=xyz.device) % count
                coords = xyz[rows][repeat].half().contiguous()
                if not torch.isfinite(coords).all():
                    raise ValueError('Flash3D coordinates exceed the native FP16 range')
                seps = torch.tensor([padded], device=xyz.device, dtype=torch.uint32)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    scattered, buffers = self.native.scatter(coords, seps, self.settings['bucket_size'], padded, self.settings['hash_type'])
                    starts = torch.nn.functional.pad(buffers.bucket_cumsum[0, :-1].long(), (1, 0))
                    destination = starts[buffers.bucket_id.long()]+buffers.bucket_offs.long()
                    if destination.min() < 0 or destination.max() >= padded or (destination.bincount(minlength=padded) != 1).any():
                        raise ValueError('Flash3D native hash did not preserve every input row')
                    repeated = features[rows][repeat]
                    scattered_features = torch.empty_like(repeated).index_copy(0, destination, repeated)
                    embedded = self.network.psh_scatter.lin(scattered_features.bfloat16())
                    encoded = self.network.module_tree(scattered, embedded, seps)
                    restored = encoded[destination[:count]]
                projected = self.projection(restored.float())
                output = output.index_copy(0, rows, projected)
        return output


class Flash3DBackbone(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.features = Flash3DFeatures(3, 256, **options)

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('Flash3D requires camera XYZ [B,N,3], N >= 1024')
        points = points.contiguous()
        size, count, _ = points.shape
        batch = torch.arange(size, device=points.device).repeat_interleave(count)
        xyz = points.reshape(-1, 3)
        dense = self.features(xyz, xyz, batch).reshape(size, count, 256)
        indices = sample_indices(points, 1024, getattr(self, 'seed_sampling', 'upstream'),
            native=_ext.furthest_point_sampling, training=self.training)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        sampled = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256)).transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds, fp2_features=sampled, fp2_inds=indices)
        return sampled, seeds, end_points


class SparseFlash3DBackbone(nn.Module):
    def __init__(self, out_channels=512, voxel_size=.005, feature_channels=3, **options):
        super().__init__()
        self.voxel_size = voxel_size
        self.features = Flash3DFeatures(3+feature_channels, out_channels, **options)

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long()
        xyz = coords[:, 1:].float()*self.voxel_size
        features = self.features(xyz, torch.cat([xyz, sparse.F], 1), coords[:, 0])
        return ME.SparseTensor(features, coordinate_map_key=sparse.coordinate_map_key, coordinate_manager=sparse.coordinate_manager)
