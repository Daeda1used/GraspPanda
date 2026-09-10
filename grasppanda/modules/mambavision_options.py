"""Stage and block contracts for the released MambaVision image hierarchy."""
from copy import deepcopy
import math


VARIANTS = {
    'tiny': (80, 32, [1, 3, 8, 4]),
    'tiny2': (80, 32, [1, 3, 11, 4]),
    'small': (96, 64, [3, 3, 7, 5]),
    'base': (128, 64, [3, 3, 10, 5]),
    'large': (196, 64, [3, 3, 10, 5]),
    'large2': (196, 64, [3, 3, 12, 5]),
}


def schema():
    integer = lambda low, high: ('per_block', 64, ('int', low, high))
    boolean = ('per_block', 64, ('bool',))
    return dict(variant=('choice', tuple(VARIANTS)), pretrained=('bool',),
        embed_dim=('int', 16, 256), stem_dim=('int', 8, 128),
        stage_depths=('int_list', 4, 1, 32), stage_heads=('int_list', 2, 1, 64),
        window_sizes=('int_list', 2, 1, 64),
        block_mixers=('choice_list', 2, 64, ('mamba', 'attention')),
        block_heads=integer(1, 64), block_state_dims=integer(1, 256),
        block_conv_sizes=integer(1, 9), block_expansions=integer(1, 4),
        block_dt_ranks=integer(1, 128), block_qkv_bias=boolean, block_qk_norm=boolean,
        block_mlp_ratios=('per_stage', 64, ('float', 1, 8)),
        layer_scale=('float', 0, 1), conv_layer_scale=('float', 0, 1),
        attention_dropout=('float', 0, .8), mlp_dropout=('float', 0, .8),
        drop_path=('float', 0, .8), gradient_checkpointing=('bool',),
        freeze_norm_stats=('bool',), trainable_stages=('int', 0, 4),
        projection_norm=('choice', ('batch', 'group', 'none')))


def _resolve(options):
    variant = options.get('variant', 'tiny')
    if variant not in VARIANTS: raise ValueError('Unknown MambaVision variant')
    dim, stem, depths = VARIANTS[variant]
    p = dict(variant=variant, pretrained=True, embed_dim=dim, stem_dim=stem,
        stage_depths=deepcopy(depths), stage_heads=[16, 32] if dim == 196 else [8, 16],
        window_sizes=[14, 7], layer_scale=1e-5 if dim >= 128 else None,
        conv_layer_scale=None, attention_dropout=0., mlp_dropout=0., drop_path=.3 if dim >= 128 else .2,
        gradient_checkpointing=False, freeze_norm_stats=False, trainable_stages=4,
        projection_norm='batch')
    p.update(deepcopy(options))
    for key, value in options.items():
        rule = schema().get(key)
        if rule is None: raise ValueError('Unknown MambaVision parameter: ' + key)
        kind = rule[0]
        values = value if isinstance(value, list) else [value]
        if kind == 'choice': valid = isinstance(value, str) and value in rule[1]
        elif kind == 'choice_list':
            valid = isinstance(value, list) and rule[1] <= len(value) <= rule[2] and all(v in rule[3] for v in value)
        elif kind == 'bool': valid = type(value) is bool
        else:
            scalar = rule[2] if kind in ('per_block', 'per_stage') else ('int', rule[2], rule[3]) if kind == 'int_list' else rule
            valid = all(type(v) is bool if scalar[0] == 'bool' else
                type(v) in ((int,) if scalar[0] == 'int' else (int, float)) and
                math.isfinite(v) and scalar[1] <= v <= scalar[2] for v in values)
            if kind == 'int_list': valid &= isinstance(value, list) and len(value) == rule[1]
            elif kind in ('int', 'float'): valid &= not isinstance(value, list)
        if not valid: raise ValueError('Invalid MambaVision parameter: ' + key)
    widths = [p['embed_dim'] * 4, p['embed_dim'] * 8]
    count = sum(p['stage_depths'][2:])
    defaults = dict(block_heads=[h for h, d in zip(p['stage_heads'], p['stage_depths'][2:]) for _ in range(d)],
        block_state_dims=8, block_conv_sizes=3, block_expansions=1,
        block_dt_ranks=[math.ceil(w / 16) for w, d in zip(widths, p['stage_depths'][2:]) for _ in range(d)],
        block_qkv_bias=True, block_qk_norm=False, block_mlp_ratios=4.)
    p.setdefault('block_mixers', [kind for d in p['stage_depths'][2:]
        for kind in ['mamba'] * math.ceil(d / 2) + ['attention'] * (d // 2)])
    for key, default in defaults.items():
        value = p.get(key, default)
        p[key] = value if isinstance(value, list) else [value] * count
    if any(len(p[k]) != count for k in ('block_mixers', *defaults)):
        raise ValueError('MambaVision block vectors must match the combined depths of stages 3 and 4')
    block_widths = [w for w, d in zip(widths, p['stage_depths'][2:]) for _ in range(d)]
    for i, width in enumerate(block_widths):
        if p['block_mixers'][i] == 'attention' and width % p['block_heads'][i]:
            raise ValueError('MambaVision attention heads must divide their block width')
        if p['block_mixers'][i] == 'mamba' and (width * p['block_expansions'][i]) % 2:
            raise ValueError('MambaVision mixer expansion must split into equal scan and convolution branches')
    return p


def resolve(options):
    p = _resolve(options)
    if p['pretrained']:
        native = _resolve({'variant': p['variant']})
        structural = ('embed_dim', 'stem_dim', 'stage_depths', 'block_mixers',
            'block_mlp_ratios', 'layer_scale', 'conv_layer_scale')
        changed = [k for k in structural if p[k] != native[k]]
        if not changed:
            for i, kind in enumerate(p['block_mixers']):
                keys = ('block_qkv_bias', 'block_qk_norm') if kind == 'attention' else (
                    'block_state_dims', 'block_conv_sizes', 'block_expansions', 'block_dt_ranks')
                changed += [k for k in keys if p[k][i] != native[k][i]]
        if changed:
            raise ValueError('MambaVision pretrained weights require the released parameter shapes; '
                             'set pretrained: false for structural edits: ' + ', '.join(sorted(set(changed))))
    return p


def validate(options):
    resolve(options)
