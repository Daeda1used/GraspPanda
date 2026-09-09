"""Native rank-augmented attention on the HGGD/RNG image feature lattice."""
import ast
import hashlib
import sys
import types
from torch import nn

from .image_pyramid import ImagePyramid
from .rala_options import resolve

SOURCE_SHA256 = 'cd9210dd2dff6be6a17bbbe24c49ee5219aa80a75175872514223fa2be3525d2'


def native_module():
    from ..config import ROOT
    path = ROOT / 'environments/sources/cv/rala/segmentation/mmseg/models/backbones/RALA.py'
    if not path.is_file(): raise ValueError('RALA source is missing; run ./panda install')
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('RALA source differs from its pinned implementation; restore the source and reinstall')
    name = '_grasppanda_rala'
    if name in sys.modules:
        if sys.modules[name].__file__ != str(path): raise ValueError('A different RALA source is loaded; start a fresh worker')
        return sys.modules[name]
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
        and node.module in ('builder', 'mmcv_custom', 'mmseg.utils'))]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = [d for d in node.decorator_list if 'BACKBONES.' not in ast.unparse(d)]
            node.body = [m for m in node.body if not (isinstance(m, ast.FunctionDef) and m.name == 'init_weights')]
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    try: exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def checkpointed_block(block, value, sin, cos):
    import torch
    from torch.utils.checkpoint import checkpoint
    if block.training and torch.is_grad_enabled():
        return checkpoint(type(block).forward, block, value, sin, cos, use_reentrant=False)
    return type(block).forward(block, value, sin, cos)


class RALAPyramid(ImagePyramid):
    def __init__(self, **options):
        nn.Module.__init__(self); p = resolve(options); native = native_module()
        flags = []; offset = 0
        for depth in p['stage_depths']:
            flags.append([{'rala': 'l', 'softmax': 'v'}[k] for k in p['block_attention'][offset:offset+depth]])
            offset += depth
        self.encoder = native.GLTA(in_chans=4, out_indices=(0, 1, 2, 3),
            embed_dims=p['stage_channels'], depths=p['stage_depths'], num_heads=p['stage_heads'],
            mlp_ratios=p['mlp_ratios'], flagss=flags, drop_path_rate=p['drop_path'],
            layerscales=p['layer_scale'], layer_init_values=p['layer_scale_init'], norm_eval=False)
        if p['gradient_checkpointing']:
            for stage in self.encoder.layers:
                for block in stage.blocks: block.forward = types.MethodType(checkpointed_block, block)
        self.freeze_norm_stats = p['freeze_norm_stats']
        self.configure_pyramid('rala', p['stage_channels'], (4, 8, 16, 32),
            p['projection_norm'], pad_to_stride=False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_norm_stats:
            for layer in self.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm): layer.eval()
        return self
