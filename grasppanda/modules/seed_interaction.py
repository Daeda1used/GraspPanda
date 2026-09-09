"""Distance-biased interaction between grouped grasp seeds, isolated by scene and depth."""
import ast
import hashlib
import math
from functools import lru_cache

import torch
from torch import nn

from ..config import ROOT


@lru_cache(maxsize=1)
def native_class():
    path = ROOT/'environments/sources/cv/gcf-graphgrasp/models/modules_economicgrasp.py'
    if not path.is_file():
        raise ValueError('Seed interaction source is missing; run ./panda install')
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != '9afa1e1a9acba07c158534fcecc2f1497734a88a9b3d971186a321a9008441a1':
        raise ValueError('Seed interaction source differs from its pinned revision; restore it with ./panda install')
    tree = ast.parse(contents, filename=str(path))
    names = {'GraspGNN', 'MultiHeadAttn', 'AttentionModule'}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in tree.body} != names or len(tree.body) != len(names):
        raise ValueError('Seed interaction class layout differs from the pinned source')
    namespace = {'torch': torch, 'nn': nn, 'math': math, '__name__': __name__}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['GraspGNN']


class SeedInteraction(nn.Module):
    def __init__(self, heads=4, sigma=.05, layers=1, dropout=.1):
        super().__init__()
        from ..module_options import validate_options
        validate_options('economicgrasp', 'crop', 'upstream', dict(seed_interaction='gaussian',
            interaction_heads=heads, interaction_sigma=sigma,
            interaction_layers=layers, interaction_dropout=dropout))
        self.blocks = nn.ModuleList(native_class()(feat_dim=256, n_head=heads, sigma=sigma)
                                    for _ in range(layers))
        for block in self.blocks:
            block.attn.msa.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, features, xyz):
        if (xyz.ndim != 3 or xyz.shape[-1] != 3 or features.ndim not in (3, 4)
                or features.shape[1] != 256 or features.shape[0] != xyz.shape[0]
                or features.shape[2] != xyz.shape[1] or xyz.shape[1] == 0):
            raise ValueError('Seed interaction requires matching XYZ [B,N,3] and features [B,256,N] or [B,256,N,D]')
        if features.device != xyz.device or features.dtype != xyz.dtype or features.dtype != torch.float32:
            raise ValueError('Seed interaction requires float32 features and camera XYZ on the same device')
        if features.ndim == 4:
            b, c, n, d = features.shape
            if not d: raise ValueError('Seed interaction requires a nonempty depth dimension')
            x = features.permute(0, 3, 1, 2).reshape(b*d, c, n)
            points = xyz[:, None].expand(-1, d, -1, -1).reshape(b*d, n, 3)
        else:
            x, points = features, xyz
        for block in self.blocks:
            x = block(x, points)
        return x.reshape(b, d, c, n).permute(0, 2, 3, 1).contiguous() if features.ndim == 4 else x


def apply_interaction(module, inputs, output):
    return module.seed_interaction(output, inputs[0])


def install(module, **options):
    if hasattr(module, 'seed_interaction'):
        raise ValueError('Seed interaction is already installed on this grouping module')
    module.add_module('seed_interaction', SeedInteraction(**options))
    module.register_forward_hook(apply_interaction)
    return module
