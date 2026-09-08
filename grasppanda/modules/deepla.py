"""Native DeepLA ResLFE blocks inside method-specific oriented cylinders."""
import ast
import sys
import types

import torch
from torch import nn

from .cylinder import CylindricalAggregation
from .layers import point_mlp


def native_module():
    name = '_grasppanda_deepla'
    if name in sys.modules:
        return sys.modules[name]
    from ..config import ROOT
    root = ROOT/'environments/sources/cv/deepla'
    try:
        import _grasppanda_deepla_cuda as extension
    except ImportError as error:
        raise ValueError('DeepLA CUDA operators are missing; run ./panda install') from error
    module = types.ModuleType(name)
    module.__dict__.update(torch=torch, nn=nn, Function=torch.autograd.Function,
        custom_fwd=torch.cuda.amp.custom_fwd, custom_bwd=torch.cuda.amp.custom_bwd,
        cutils=extension)
    # Import the declared native blocks without initializing the S3DIS trainer,
    # global utils package or the author's import-time CUDA compilation.
    definitions = (
        ('utils/timm/models/layers/drop.py', {'drop_path', 'DropPath'}),
        ('utils/cutils/__init__.py', {'KEMP'}),
        ('S3DIS/deepla_semseg.py', {'VFR', 'FFN', 'ResLFE_Block'}),
    )
    for relative, names in definitions:
        path = root/relative
        if not path.is_file():
            raise ValueError('DeepLA source is missing; run ./panda install')
        tree = ast.parse(path.read_text(), filename=str(path))
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if {node.name for node in selected} != names:
            raise ValueError('DeepLA block definitions differ from the registered source')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), module.__dict__)
        if 'KEMP' in names:
            module.knn_edge_maxpooling = module.KEMP.apply
    sys.modules[name] = module
    return module


class ResLFENeighborhood(nn.Module):
    def __init__(self, in_channels, width, depth, local_neighbors, mlp_ratio,
                 drop_path, bn_momentum, activation, normalization):
        super().__init__()
        self.local_neighbors = local_neighbors
        self.embedding = point_mlp([in_channels, width], 2, activation, normalization)
        self.position = point_mlp([3, width], 2, activation, normalization)
        act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]
        self.blocks = native_module().ResLFE_Block(width, depth, drop_path,
                                                  mlp_ratio, bn_momentum, act)
        self.projection = point_mlp([width, 256], 2, activation, normalization)

    def forward(self, grouped):
        batch, _, seeds, count = grouped.shape
        # Each oriented cylinder is an independent local graph. Its padded
        # native query samples retain their original order and multiplicity.
        coordinates = grouped[:, :3].permute(0, 2, 3, 1).reshape(batch*seeds, count, 3)
        with torch.no_grad():
            distances = torch.cdist(coordinates.float(), coordinates.float())
            neighbors = distances.topk(self.local_neighbors, largest=False, sorted=True).indices.contiguous()
        feature = self.embedding(grouped).permute(0, 2, 3, 1).reshape(batch*seeds, count, -1).contiguous()
        position = self.position(grouped[:, :3]).permute(0, 2, 3, 1).reshape(batch*seeds, count, -1).contiguous()
        if feature.dtype not in (torch.float32, torch.float16):
            raise ValueError('DeepLA neighborhood operators support float32 and float16')
        feature = self.blocks(feature, position, neighbors, pts=[count])
        feature = feature.reshape(batch, seeds, count, -1).permute(0, 3, 1, 2).contiguous()
        return self.projection(feature)


class ResLFECylinder(CylindricalAggregation):
    def __init__(self, native, protocol, width=64, depth=4, local_neighbors=8,
                 nsample=16, radius_factors=(1.,), mlp_ratio=1., drop_path=.1,
                 bn_momentum=.02, pooling='max', activation='gelu', normalization='batch'):
        super().__init__(native, protocol, hidden_channels=(width,), nsample=nsample,
                         radius_factors=radius_factors, pooling=pooling,
                         activation=activation, normalization=normalization)
        if any(not group.use_xyz for group in self.groups):
            raise ValueError('DeepLA cylindrical aggregation requires grouped XYZ')
        in_channels = native.in_dim if protocol == 'baseline' else native.in_dim + 3
        self.encoder = ResLFENeighborhood(in_channels, width, depth, local_neighbors,
            mlp_ratio, drop_path, bn_momentum, activation, normalization)
