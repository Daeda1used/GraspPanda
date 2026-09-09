"""Configuration contracts for the native RALA image hierarchy."""
from copy import deepcopy
import math


DEFAULTS = dict(stage_channels=[64, 128, 256, 512], stage_depths=[2, 2, 6, 2],
    stage_heads=[1, 2, 4, 8], mlp_ratios=3.5, layer_scale=True,
    layer_scale_init=1., drop_path=.1, gradient_checkpointing=False,
    freeze_norm_stats=False, projection_norm='batch')


def schema():
    return dict(stage_channels=('int_list', 4, 16, 1024),
        stage_depths=('int_list', 4, 1, 24), stage_heads=('int_list', 4, 1, 64),
        mlp_ratios=('per_stage', 4, ('float', 1, 8)),
        layer_scale=('per_stage', 4, ('bool',)),
        layer_scale_init=('per_stage', 4, ('float', 1e-8, 1)),
        block_attention=('choice_list', 4, 96, ('rala', 'softmax')),
        drop_path=('float', 0, .8), gradient_checkpointing=('bool',),
        freeze_norm_stats=('bool',), projection_norm=('choice', ('batch', 'group', 'none')))


def resolve(options):
    unknown = set(options) - set(schema())
    if unknown: raise ValueError(f'Unknown RALA parameters: {sorted(unknown)}')
    p = {**deepcopy(DEFAULTS), **deepcopy(options)}
    for key, low, high in [('stage_channels', 16, 1024), ('stage_depths', 1, 24), ('stage_heads', 1, 64)]:
        values = p[key]
        if not isinstance(values, list) or len(values) != 4 or any(type(v) is not int or not low <= v <= high for v in values):
            raise ValueError(f'RALA {key} needs four integers in [{low}, {high}]')
    if any(c % (4 * h) for c, h in zip(p['stage_channels'], p['stage_heads'])):
        raise ValueError('Each RALA stage width must be divisible by four times its head count for 2D RoPE')
    for key in ('mlp_ratios', 'layer_scale', 'layer_scale_init'):
        value = p[key]; values = value if isinstance(value, list) else [value] * 4
        if len(values) != 4: raise ValueError(f'RALA {key} needs one value or four stage values')
        if key == 'layer_scale': valid = all(type(v) is bool for v in values)
        else:
            low, high = (1, 8) if key == 'mlp_ratios' else (1e-8, 1)
            valid = all(type(v) in (int, float) and math.isfinite(v) and low <= v <= high for v in values)
        if not valid: raise ValueError(f'Invalid RALA {key}')
        p[key] = values
    if 'layer_scale_init' in options and not any(p['layer_scale']):
        raise ValueError('RALA layer_scale_init requires at least one enabled layer scale')
    if 'block_attention' not in p:
        p['block_attention'] = [kind for i, depth in enumerate(p['stage_depths'])
            for kind in [('rala' if i < 2 else 'softmax')] * depth]
    if (not isinstance(p['block_attention'], list) or len(p['block_attention']) != sum(p['stage_depths'])
        or any(v not in ('rala', 'softmax') for v in p['block_attention'])):
        raise ValueError('RALA block_attention needs one rala/softmax value per block, in encoder stage order')
    if type(p['drop_path']) not in (int, float) or not math.isfinite(p['drop_path']) or not 0 <= p['drop_path'] <= .8:
        raise ValueError('RALA drop_path must be finite and in [0, .8]')
    if any(type(p[k]) is not bool for k in ('gradient_checkpointing', 'freeze_norm_stats')):
        raise ValueError('RALA checkpointing and normalization controls must be boolean')
    if p['projection_norm'] not in ('batch', 'group', 'none'): raise ValueError('Invalid RALA projection normalization')
    return p


def validate(options):
    resolve(options)
