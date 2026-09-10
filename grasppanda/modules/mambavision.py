"""Released MambaVision blocks with camera-aligned RGB and depth features."""
import ast
from functools import lru_cache
import hashlib
import math
from types import MethodType, ModuleType

import torch
from torch import nn
from torch.nn import functional as F

from .mambavision_options import resolve


def definitions(path, sha256, names, namespace):
    if not path.is_file(): raise ValueError('MambaVision build inputs are missing; run ./panda install')
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != sha256:
        raise ValueError('MambaVision build input differs from its pinned source; restore it and reinstall')
    tree = ast.parse(source, filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in tree.body} != set(names): raise ValueError('Pinned MambaVision definitions are incomplete')
    if 'Attention' in names:
        # The released SDPA branch applies dropout in evaluation as well.
        # Keep training behavior and make evaluation follow module.eval().
        changed = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'scaled_dot_product_attention':
                for key in node.keywords:
                    if key.arg == 'dropout_p':
                        key.value = ast.IfExp(test=ast.parse('self.training', mode='eval').body,
                            body=key.value, orelse=ast.Constant(0.))
                        changed += 1
        if changed != 1: raise ValueError('MambaVision SDPA interface differs from the pinned implementation')
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)


@lru_cache(maxsize=1)
def native_module():
    from ..config import ROOT
    from einops import rearrange, repeat
    from timm.layers import DropPath, LayerNorm2d, trunc_normal_
    from timm.models.vision_transformer import Mlp
    try:
        import _grasppanda_pointmamba_scan as scan_cuda
    except ImportError as error:
        raise ValueError('The shared Mamba CUDA operator is missing; run ./panda install') from error
    native = ModuleType('_grasppanda_mambavision')
    native.__dict__.update(torch=torch, nn=nn, F=F, math=math, rearrange=rearrange, repeat=repeat,
        DropPath=DropPath, LayerNorm2d=LayerNorm2d, trunc_normal_=trunc_normal_, Mlp=Mlp,
        selective_scan_cuda=scan_cuda)
    definitions(ROOT/'environments/sources/cv/pointmamba/mamba/mamba_ssm/ops/selective_scan_interface.py',
        'fe606b4c7e81b47bb091cf59dc474aece1112a6ca01eb6f22309090e56030bc4',
        ('SelectiveScanFn', 'selective_scan_fn'), native.__dict__)
    definitions(ROOT/'environments/sources/cv/mambavision/mambavision/models/mamba_vision.py',
        '726d8eef113a94b2e71657d967abefec95719e47f1791643d2d42229651d9acf',
        ('window_partition', 'window_reverse', 'Downsample', 'PatchEmbed', 'ConvBlock',
         'MambaVisionMixer', 'Attention', 'Block', 'MambaVisionLayer', 'MambaVision'), native.__dict__)
    return native


def checkpointed_block(block, value):
    from torch.utils.checkpoint import checkpoint
    if block.training and torch.is_grad_enabled():
        return checkpoint(type(block).forward, block, value, use_reentrant=False)
    return type(block).forward(block, value)


def pretrained_state(path):
    from safetensors.torch import load_file
    payload = load_file(path)
    weights = {}
    for key, value in payload.items():
        if not key.startswith('model.'): raise ValueError('Unexpected MambaVision pretrained key prefix')
        key = key.removeprefix('model.')
        # The author HF export calls the same layer-scale parameters g_1/g_2.
        if key.endswith(('.g_1', '.g_2')): key = key[:-3] + 'gamma_' + key[-1]
        if key in weights: raise ValueError('Duplicate MambaVision pretrained parameter after name conversion')
        weights[key] = value
    return weights


class MambaVisionPyramid(nn.Module):
    def __init__(self, **options):
        super().__init__()
        p = resolve(options); native = native_module()
        self.pretrained_requested = p['pretrained']
        self.weight_id = 'mambavision_' + p['variant']
        self.trainable_stages = p['trainable_stages']
        self.freeze_norm_stats = p['freeze_norm_stats']
        self.encoder = native.MambaVision(dim=p['embed_dim'], in_dim=p['stem_dim'],
            depths=p['stage_depths'], num_heads=[1, 1, 1, 1],
            window_size=[8, 8, *p['window_sizes']], mlp_ratio=4.,
            drop_path_rate=p['drop_path'], layer_scale=p['layer_scale'],
            layer_scale_conv=p['conv_layer_scale'])
        # The grasp pyramid uses pre-downsample features; pooled classification
        # normalization and logits do not contribute to those features.
        del self.encoder.head, self.encoder.norm, self.encoder.avgpool
        rates = torch.linspace(0, p['drop_path'], sum(p['stage_depths'])).tolist()
        index = 0
        for stage in (2, 3):
            width = p['embed_dim'] * 2 ** stage
            for j in range(p['stage_depths'][stage]):
                attention = p['block_mixers'][index] == 'attention'
                block = native.Block(dim=width, num_heads=p['block_heads'][index],
                    counter=0, transformer_blocks=[0] if attention else [],
                    mlp_ratio=p['block_mlp_ratios'][index], qkv_bias=p['block_qkv_bias'][index],
                    qk_scale=p['block_qk_norm'][index], drop=p['mlp_dropout'],
                    attn_drop=p['attention_dropout'],
                    drop_path=rates[sum(p['stage_depths'][:stage]) + j], layer_scale=p['layer_scale'])
                if not attention:
                    block.mixer = native.MambaVisionMixer(d_model=width,
                        d_state=p['block_state_dims'][index], d_conv=p['block_conv_sizes'][index],
                        expand=p['block_expansions'][index], dt_rank=p['block_dt_ranks'][index])
                block.apply(self.encoder._init_weights)
                if p['gradient_checkpointing']: block.forward = MethodType(checkpointed_block, block)
                self.encoder.levels[stage].blocks[j] = block
                index += 1
        if self.trainable_stages < 4:
            self.encoder.requires_grad_(False)
            if self.trainable_stages: self.encoder.levels[-self.trainable_stages:].requires_grad_(True)
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        def projection(channels, target, kernel):
            layers = [nn.Conv2d(channels, target, kernel, padding=kernel//2, bias=p['projection_norm'] == 'none')]
            if p['projection_norm'] == 'batch': layers.append(nn.BatchNorm2d(target))
            elif p['projection_norm'] == 'group': layers.append(nn.GroupNorm(8, target))
            return nn.Sequential(*layers)
        self.rgb_projections = nn.ModuleList(projection(p['embed_dim'] * 2**i, 4*s, 1)
            for i, s in enumerate((4, 8, 16, 32)))
        self.depth_projections = nn.ModuleList(projection(1, 4*s, 3) for s in (4, 8, 16, 32))
        self.stem = nn.Sequential(nn.Conv2d(4, 8, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(8), nn.LeakyReLU(inplace=True))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.trainable_stages < 4:
            self.encoder.eval()
            if mode and self.trainable_stages: self.encoder.levels[-self.trainable_stages:].train()
        if mode and self.freeze_norm_stats:
            for layer in self.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm): layer.eval()
        return self

    def initialize_pretrained(self):
        if not self.pretrained_requested: return None
        from ..weights import fetch_component, component_records
        weights = pretrained_state(fetch_component(self.weight_id))
        excluded = {k for k in weights if k.startswith(('head.', 'norm.'))}
        expected = {'head.weight', 'head.bias', 'norm.weight', 'norm.bias',
                    'norm.running_mean', 'norm.running_var', 'norm.num_batches_tracked'}
        if excluded != expected: raise ValueError('MambaVision classification checkpoint layout differs from the registered weights')
        self.encoder.load_state_dict({k: v for k, v in weights.items() if k not in excluded}, strict=True)
        record = component_records()[self.weight_id]
        return dict(id=self.weight_id, source=record['source'], sha256=record['sha256'],
                    trainable_stages=self.trainable_stages)

    def rgb_features(self, rgb):
        outputs, handles = [None] * 4, []
        try:
            for i, level in enumerate(self.encoder.levels[:3]):
                handles.append(level.downsample.register_forward_pre_hook(
                    lambda module, args, i=i: outputs.__setitem__(i, args[0])))
            value = self.encoder.patch_embed(rgb)
            for level in self.encoder.levels: value = level(value)
            outputs[-1] = value
        finally:
            for handle in handles: handle.remove()
        return outputs

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (4, 640, 360):
            raise ValueError('MambaVision grasp adapters require native D,R,G,B [B,4,640,360] inputs')
        rgb = (x[:, 1:].transpose(-2, -1) - self.rgb_mean) / self.rgb_std
        features = self.rgb_features(rgb)
        result = [self.stem(x)]
        for value, project, depth_project, stride in zip(features, self.rgb_projections, self.depth_projections, (4, 8, 16, 32)):
            # Odd, symmetrically padded kernels retain the pixel-zero origin.
            value = value.transpose(-2, -1)
            shape = tuple(math.ceil(side / stride) for side in x.shape[-2:])
            if value.shape[-2:] != shape: raise ValueError('MambaVision lost the native image feature lattice')
            result.append(F.leaky_relu(project(value) + depth_project(x[:, :1, ::stride, ::stride])))
        return result
