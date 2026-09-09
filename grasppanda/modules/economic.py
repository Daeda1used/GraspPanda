"""Configurable EconomicGrasp blocks retaining native sparse and grasp contracts."""
from torch import nn


def tdunet(native, channels=None, blocks=None, dilations=None, stem_channels=32,
           block='basic', bn_momentum=.1):
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
    import MinkowskiEngine as ME
    widths = tuple(channels or (32, 64, 128, 256, 192, 192, 192, 192))
    depths = tuple(blocks or (1,) * 8)
    rates = tuple(dilations or (1,) * 8)

    class ConfiguredTDUnet(type(native)):
        PLANES = widths
        LAYERS = depths
        DILATIONS = rates
        INIT_DIM = stem_channels
        BLOCK = BasicBlock if block == 'basic' else Bottleneck

        def __init__(self):
            self._stage_index = 0
            super().__init__(in_channels=3, out_channels=512, D=3)

        def _make_layer(self, *args, **kwargs):
            kwargs['dilation'] = rates[self._stage_index]
            self._stage_index += 1
            return super()._make_layer(*args, **kwargs)

    model = ConfiguredTDUnet()
    for module in model.modules():
        if isinstance(module, ME.MinkowskiBatchNorm):
            module.bn.momentum = bn_momentum
    return model


class NoInteraction(nn.Module):
    def forward(self, query, key, value, mask=None):
        return value


def cylinder(native, nsample=16, radius=.05, hmin=-.02, hmax=.04,
             attention_heads=1, attention_dropout=.05, local_attention=True):
    result = type(native)(nsample=nsample, seed_feature_dim=native.in_dim,
                          cylinder_radius=radius, hmin=hmin, hmax=hmax)
    if local_attention:
        attention = result.local_interaction_module.msa
        attention.nhead = attention_heads
        attention.head_dim = attention.dim // attention_heads
        attention.dropout = nn.Dropout(attention_dropout) if attention_dropout else None
    else:
        result.local_interaction_module = NoInteraction()
    return result


class TaskInteraction(nn.Module):
    """Apply native residual-normalized attention blocks to four task tokens."""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, query, key, value, mask=None):
        value = self.blocks[0](query, key, value, mask)
        for block in self.blocks[1:]:
            value = block(value, value, value, mask)
        return value


def head(native, feature_channels=64, branch_depths=None, activation='relu',
         interaction='attention', attention_layers=1, attention_heads=1, attention_dropout=.05):
    """Specialize the pinned native head, retaining branch order and prediction keys."""
    import ast
    import inspect
    import textwrap
    native_type = type(native)
    # The original forward fixes its task-token width in eight reshape operations.
    tree = ast.parse(textwrap.dedent(inspect.getsource(native_type.forward)))
    widths = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == 64]
    if len(widths) != 8:
        raise ValueError('Native interactive-head feature reshapes differ from the registered source')
    for node in widths:
        node.value = feature_channels
    namespace = dict(native_type.forward.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(native_type), 'exec'), namespace)
    configured = type('ConfiguredInteractiveHead', (native_type,), {'forward': namespace['forward']})
    result = configured(num_angle=native.num_angle, num_depth=native.num_depth)
    names = ('angle', 'depth', 'width', 'score')
    depths = branch_depths or [1]*4
    act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]
    for name, depth in zip(names, depths):
        projection = getattr(result, 'conv_' + name + '_feature')
        if feature_channels != 64:
            projection = nn.Conv1d(256, feature_channels, 1)
            output = getattr(result, 'conv_' + name)
            setattr(result, 'conv_' + name, nn.Conv1d(feature_channels, output.out_channels, 1))
        if depth > 1:
            projection = nn.Sequential(projection, *[layer for _ in range(depth-1)
                for layer in (act(), nn.Conv1d(feature_channels, feature_channels, 1))])
        setattr(result, 'conv_' + name + '_feature', projection)
    if interaction == 'none':
        result.global_interaction_module = NoInteraction()
    else:
        heads = attention_heads if isinstance(attention_heads, list) else [attention_heads]*attention_layers
        drops = attention_dropout if isinstance(attention_dropout, list) else [attention_dropout]*attention_layers
        attention_type = type(native.global_interaction_module)
        first = result.global_interaction_module
        if feature_channels != 64:
            first = attention_type(dim=feature_channels, n_head=heads[0], msa_dropout=drops[0])
        else:
            first.msa.nhead = heads[0]
            first.msa.head_dim = feature_channels // heads[0]
            first.msa.dropout = nn.Dropout(drops[0]) if drops[0] else None
        blocks = [first] + [attention_type(dim=feature_channels, n_head=n, msa_dropout=d)
                            for n, d in zip(heads[1:], drops[1:])]
        result.global_interaction_module = first if len(blocks) == 1 else TaskInteraction(blocks)
    return result
