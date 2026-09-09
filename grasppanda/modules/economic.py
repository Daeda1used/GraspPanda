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
