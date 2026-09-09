"""Native PCM hierarchy with camera-aligned serialization and grasp decoding."""
import ast
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from functools import lru_cache
from ..config import ROOT
from ..pcm_options import CTS_ORDERS as ORDERS, native_options, resolve, validate_points
from torch import nn
import torch
PREFIX = '_grasppanda_pcm'



class Loader(importlib.machinery.SourceFileLoader):

    def get_code(self, fullname):
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (not node.level) and node.module:
                for original, replacement in [('openpoints', PREFIX), ('mamba_ssm', PREFIX + '.mamba'), ('causal_conv1d', PREFIX + '.causal')]:
                    if node.module == original or node.module.startswith(original + '.'):
                        node.module = replacement + node.module[len(original):]
            if isinstance(node, ast.Import):
                for alias in node.names:
                    replacement = {'selective_scan_cuda': '_grasppanda_pcm_scan', 'causal_conv1d_cuda': '_grasppanda_pcm_causal'}.get(alias.name)
                    if replacement:
                        alias.asname = alias.asname or alias.name
                        alias.name = replacement
        if fullname == PREFIX + '.mamba.ops.triton.layernorm':
            # RMSNorm already calls the dedicated RMS function. The upstream
            # wrapper also passes a LayerNorm-only keyword that it cannot take.
            fixes = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'rms_norm_fn':
                    invalid = [k for k in node.keywords if k.arg == 'is_rms_norm']
                    if invalid:
                        if len(invalid) != 1 or not isinstance(invalid[0].value, ast.Constant) or invalid[0].value.value is not True:
                            raise RuntimeError('PCM RMSNorm wrapper differs from the pinned source')
                        node.keywords.remove(invalid[0])
                        fixes += 1
            if fixes != 1:
                raise RuntimeError('PCM RMSNorm wrapper differs from the pinned source')
        return compile(tree, self.path, 'exec')


class Finder(importlib.abc.MetaPathFinder):

    def __init__(self, roots):
        self.roots = sorted(roots.items(), key=lambda x: -len(x[0]))

    def find_spec(self, fullname, path=None, target=None):
        for prefix, root in self.roots:
            if fullname.startswith(prefix + '.'):
                relative = fullname[len(prefix) + 1:].replace('.', '/')
                file = root / (relative + '.py')
                package = root / relative / '__init__.py'
                if package.exists():
                    file = package
                if file.exists():
                    return importlib.util.spec_from_file_location(fullname, file, loader=Loader(fullname, str(file)), submodule_search_locations=[str(file.parent)] if file == package else None)
                return None


@lru_cache(maxsize=1)
def load():
    from ..compat import triton_driver
    triton_driver()
    root = ROOT / 'environments/sources/cv/pointcloudmamba/openpoints'
    pcm = root / 'models/PCM'
    if not (pcm / 'PCM.py').is_file():
        raise ValueError('PCM sources are missing; run ./panda install')
    try:
        extension = importlib.import_module('_grasppanda_openpoints_cuda')
        importlib.import_module('_grasppanda_pcm_scan')
        importlib.import_module('_grasppanda_pcm_causal')
    except ImportError as error:
        raise ValueError('PCM native operators are missing; run ./panda install') from error
    roots = {PREFIX: root, PREFIX + '.mamba': pcm / 'mamba/mamba_ssm', PREFIX + '.causal': pcm / 'causal-conv1d/causal_conv1d'}
    packages = {**roots, **{PREFIX + '.' + s: root / s.replace('.', '/') for s in ('models', 'models.layers', 'models.PCM', 'cpp', 'cpp.pointnet2_batch')}}
    packages.update({PREFIX + '.mamba.' + s: roots[PREFIX + '.mamba'] / s.replace('.', '/') for s in ('modules', 'ops', 'ops.triton')})
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module
    sys.modules[PREFIX + '.cpp.pointnet2_batch'].pointnet2_cuda = extension
    registry = types.ModuleType(PREFIX + '.models.build')
    registry.MODELS = types.SimpleNamespace(register_module=lambda: lambda cls: cls)
    sys.modules[registry.__name__] = registry
    sys.meta_path.insert(0, Finder(roots))
    causal = importlib.import_module(PREFIX + '.causal.causal_conv1d_interface')
    for name in ('causal_conv1d_fn', 'causal_conv1d_update'):
        setattr(sys.modules[PREFIX + '.causal'], name, getattr(causal, name))
    subsample = importlib.import_module(PREFIX + '.models.layers.subsample')
    sys.modules[PREFIX + '.models.layers'].furthest_point_sample = subsample.furthest_point_sample
    return importlib.import_module(PREFIX + '.models.PCM.PCM')


def cts_codes(grid, batch, order):
    """Zero-based serpentine CTS codes with disjoint batch ranges."""
    if order not in ORDERS:
        raise ValueError(order)
    if grid.dtype != torch.int64 or batch.dtype != torch.int64:
        raise ValueError('integer coordinates required')
    if (grid < 0).any():
        raise ValueError('shift grid coordinates before encoding')
    axes = ['xyz'.index(axis) for axis in order]
    coordinates = grid[:, axes]
    extents = coordinates.amax(0) + 1
    code = coordinates[:, 0]
    span = extents[0]
    for axis in (1, 2):
        row = coordinates[:, axis]
        code = row * span + torch.where(row % 2 == 0, code, span - 1 - code)
        span = span * extents[axis]
    return code + batch * span


def sort_indices(pos, order, grid_size):
    if not torch.isfinite(pos).all():
        raise ValueError('PCM coordinates must be finite')
    scaled = torch.floor(pos / grid_size).double()
    scaled = scaled - scaled.amin(dim=1, keepdim=True)
    if not torch.isfinite(scaled).all() or scaled.max() >= 65536:
        raise ValueError('PCM voxel extent exceeds 16 bits; increase grid_size')
    grid = scaled.long()
    batch, n, _ = pos.shape
    ids = torch.arange(batch, device=pos.device).repeat_interleave(n)
    if order in ORDERS:
        code = cts_codes(grid.flatten(0, 1), ids, order)
        permutation = code.argsort()
    else:
        native = importlib.import_module(PREFIX + '.models.PCM.serialization')
        depth = max(1, int(grid.max()).bit_length())
        point = native.Point(grid_coord=grid.flatten(0, 1), batch=ids)
        point.serialization(order=[order], depth=depth)
        permutation = point.serialized_order[0]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(batch * n, device=pos.device)
    return (permutation, inverse)


def reorder(value, indices):
    return None if value is None else value.flatten(0, 1)[indices].reshape_as(value).contiguous()


def corrected_serialization(pos, feat, x_res=None, order='z', layers_outputs=None, grid_size=0.02):
    permutation, _ = sort_indices(pos, order, grid_size)
    for i in range(len(layers_outputs or [])):
        layers_outputs[i] = reorder(layers_outputs[i], permutation)
    return (reorder(pos, permutation), reorder(feat, permutation), reorder(x_res, permutation))


def decoder_serialization(self, p, x, x_res, order, layers_outputs=None):
    if order == self.order or order == 'null':
        return (p, x, x_res)
    indices, _ = sort_indices(p, order, self.grid_size)
    if self.stage_indices is None:
        self.stage_indices = torch.arange(p.shape[0] * p.shape[1], device=p.device).reshape(p.shape[0], p.shape[1], 1)
    self.stage_indices = reorder(self.stage_indices, indices)
    self.order = order
    return (reorder(p, indices), reorder(x, indices), reorder(x_res, indices))


def restore_stage(self, x):
    """Restore each decoder resolution before the next coordinate interpolation."""
    if self.stage_indices is not None:
        indices = self.stage_indices.flatten()
        inverse = torch.empty_like(indices)
        inverse[indices] = torch.arange(len(indices), device=x.device)
        x = reorder(x.transpose(1, 2), inverse).transpose(1, 2).contiguous()
    self.stage_indices = None
    self.order = 'original'
    return x


def prune_scan_projections(encoder, counts):
    """Remove projections that cannot reach any active encoder scan."""
    active = [i for i, count in enumerate(counts) if count]
    first = min(active, default=4)
    encoder._last_scan_stage = max(active, default=-1)
    for i, count in enumerate(counts):
        if i <= first or i > encoder._last_scan_stage:
            encoder.residual_proj_blocks_list[i] = nn.Identity()
        if not count:
            if encoder.use_order_prompt:
                encoder.order_prompt_proj[i] = nn.Identity()
            if encoder.mamba_pos and encoder.pos_type == 'share' and not encoder.block_pos_share:
                encoder.pos_proj[i] = nn.Identity()


@lru_cache(maxsize=1)
def corrected_classes():
    """Load native layers with deterministic prompts and stage correspondence fixes."""
    native = load()
    file = ROOT / 'environments/sources/cv/pointcloudmamba/openpoints/models/PCM/PCM.py'
    tree = ast.parse(file.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ('PointMambaEncoder', 'PointMambaDecoder')]
    for cls in tree.body:
        cls.decorator_list = []
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                method.body = [n for n in method.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and (n.value.func.id == 'print'))]
        scans = 0
        for node in ast.walk(cls):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'MambaBlock':
                node.keywords.append(ast.keyword(arg='ssm_cfg', value=ast.parse(
                    "kwargs.get('_grasppanda_ssm', [{}] * sum(mamba_blocks))[mamba_layer_idx]", mode='eval').body))
                scans += 1
        if scans != 1:
            raise RuntimeError('PCM scan constructor differs from the pinned source')
        if cls.name == 'PointMambaEncoder':
            initializer = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
            initializer.body.extend(ast.parse('_grasppanda_prune(self, mamba_blocks)').body)
            projections = 0
            for node in ast.walk(cls):
                if isinstance(node, ast.Assign) and ast.unparse(node.value) == 'self.residual_proj_blocks_list[i](x_res)':
                    node.value = ast.parse('self.residual_proj_blocks_list[i](x_res) if x_res is not None and i <= self._last_scan_stage else None', mode='eval').body
                    projections += 1
            if projections != 2:
                raise RuntimeError('PCM residual path differs from the pinned source')
            count = 0
            for node in ast.walk(cls):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'set') and (len(node.args) == 1) and isinstance(node.args[0], ast.Name) and (node.args[0].id == 'mamba_layers_orders'):
                    node.func = ast.Attribute(value=ast.Name(id='dict', ctx=ast.Load()), attr='fromkeys', ctx=ast.Load())
                    count += 1
            if count != 1:
                raise RuntimeError('PCM prompt source differs from the pinned layout')
        else:
            forward = next((node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward'))
            loop = next((node for node in forward.body if isinstance(node, ast.For) and ast.unparse(node.iter) == 'range(len(self.decode_list))'))
            loop.body.extend(ast.parse('x = self.restore_stage(x)').body)
            forward.body[:0] = ast.parse('self.stage_indices = None').body
    namespace = dict(native.__dict__)
    namespace['serialization'] = corrected_serialization
    namespace['_grasppanda_prune'] = prune_scan_projections
    exec(compile(ast.fix_missing_locations(tree), str(file), 'exec'), namespace)
    decoder = namespace['PointMambaDecoder']
    decoder.serialize_func = decoder_serialization
    decoder.restore_stage = restore_stage
    return (namespace['PointMambaEncoder'], decoder)


class PCMBackbone(nn.Module):

    def __init__(self, **options):
        super().__init__()
        from ..module_options import validate_options
        validate_options('graspnet_baseline', 'backbone', 'pointcloud_mamba', options)
        self.options = resolve(options)
        encoder_options, decoder_options = native_options(self.options)
        encoder, decoder = corrected_classes()
        self.encoder = encoder(**encoder_options)
        self.decoder = decoder(**decoder_options)
        self.decoder.grid_size = self.encoder.grid_size
        self.decoder.stage_indices = None
        self.projection = nn.Conv1d(self.decoder.out_channels, 256, 1)

    def forward(self, points, end_points=None):
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('PCM requires camera XYZ [B,N,3] with N >= 1024')
        if not points.is_cuda or points.dtype != torch.float32:
            raise ValueError('PCM native point operators require CUDA float32 coordinates')
        if self.training and points.shape[0] < 2:
            raise ValueError('PCM training requires batch_size >= 2 for native global-context BatchNorm')
        if not torch.isfinite(points).all():
            raise ValueError('PCM coordinates must be finite')
        validate_points(self.options, points.shape[1])
        points = points.contiguous()
        p, f = self.encoder.forward_seg_feat(points)
        dense = self.projection(self.decoder(p.copy(), f.copy()))
        ids = load().furthest_point_sample(points, 1024)
        xyz = points.gather(1, ids.long()[..., None].expand(-1, -1, 3)).contiguous()
        features = dense.gather(2, ids.long()[:, None, :].expand(-1, 256, -1)).contiguous()
        ends = {} if end_points is None else end_points
        ends.update(input_xyz=points, input_features=None, fp2_xyz=xyz, fp2_features=features, fp2_inds=ids)
        return (features, xyz, ends)
