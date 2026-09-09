"""Serializable hierarchy and individual transformer controls for Flash3D."""
from copy import deepcopy
from math import lcm


DEFAULTS = dict(channels=[32, 64], enc_depths=[2, 2], dec_depths=[1, 1],
    bucket_size=128, hash_type=1, pooling='mean', normalization='layer', norm_eps=1e-5)
for _prefix in ('enc', 'dec'):
    DEFAULTS.update({
        _prefix+'_heads': 2, _prefix+'_mlp_ratio': 2., _prefix+'_qkv_bias': True,
        _prefix+'_scope_plan': ['contiguous', 'shifted', 'strided'],
        _prefix+'_scope_size': 4, _prefix+'_scope_shift': 2, _prefix+'_scope_stride': 2,
        _prefix+'_mlp_activation': ['swiglu'], _prefix+'_residual_dropout': 0.,
    })


def schema():
    fields = dict(channels=('int_sequence', 1, 5, 16, 512),
        enc_depths=('int_sequence', 1, 5, 1, 12), dec_depths=('int_sequence', 1, 5, 1, 12),
        bucket_size=('int', 128, 512), hash_type=('int', 1, 4),
        pooling=('flash3d_pooling',),
        normalization=('choice', ('layer', 'rms')), norm_eps=('float', 1e-6, .01))
    for prefix in ('enc', 'dec'):
        fields.update({
            prefix+'_heads': ('per_block', 60, ('int', 1, 32)),
            prefix+'_mlp_ratio': ('per_stage', 60, ('float', .5, 8)),
            prefix+'_qkv_bias': ('per_block', 60, ('bool',)),
            prefix+'_scope_plan': ('choice_list', 1, 60, ('contiguous', 'shifted', 'strided')),
            prefix+'_scope_size': ('per_block', 60, ('int', 1, 16)),
            prefix+'_scope_shift': ('per_block', 60, ('int', 0, 15)),
            prefix+'_scope_stride': ('per_block', 60, ('int', 1, 8)),
            prefix+'_mlp_activation': ('choice_list', 1, 60, ('swiglu', 'gelu', 'relu')),
            prefix+'_residual_dropout': ('per_stage', 60, ('float', 0, .5)),
        })
    return fields


def resolve(options):
    p = {**deepcopy(DEFAULTS), **deepcopy(options)}
    stages = len(p['channels'])
    if any(len(p[key]) != stages for key in ('enc_depths', 'dec_depths')):
        raise ValueError('Flash3D channels and depth lists need one entry per hierarchy level')
    if p['bucket_size'] not in (128, 256, 512):
        raise ValueError('Flash3D bucket_size must be 128, 256 or 512')
    pools = p['pooling']
    if isinstance(pools, str): pools = [pools]*(stages-1)
    if not isinstance(pools, list) or len(pools) != stages-1 or any(v not in ('mean', 'sum', 'min', 'max') for v in pools):
        raise ValueError('Flash3D pooling must be one name, or one reduction per hierarchy transition')
    if stages == 1 and 'pooling' in options:
        raise ValueError('Flash3D pooling requires at least two hierarchy levels')
    p['pooling'] = pools
    alignment = p['bucket_size']
    for prefix in ('enc', 'dec'):
        count = sum(p[prefix+'_depths'])
        for key in ('heads', 'mlp_ratio', 'qkv_bias', 'scope_size', 'scope_shift', 'scope_stride', 'residual_dropout'):
            name = prefix+'_'+key
            if not isinstance(p[name], list): p[name] = [p[name]]*count
            if len(p[name]) != count:
                raise ValueError(f'Flash3D {name} needs a scalar or one value per {prefix} block in fine-to-coarse level order')
        for key in ('scope_plan', 'mlp_activation'):
            name = prefix+'_'+key
            if len(p[name]) > count:
                # The default plan is a cycle; shorter custom hierarchies may use its prefix.
                if name in options:
                    raise ValueError(f'Flash3D {name} pattern is longer than the {prefix} block count')
            p[name] = [p[name][i % len(p[name])] for i in range(count)]
        for parameter, plan in (('scope_shift', 'shifted'), ('scope_stride', 'strided')):
            if prefix+'_'+parameter in options and plan not in p[prefix+'_scope_plan']:
                raise ValueError(f'Flash3D {parameter} requires a {plan} attention block')
        index = 0
        for level, (width, depth) in enumerate(zip(p['channels'], p[prefix+'_depths'])):
            for _ in range(depth):
                heads = p[prefix+'_heads'][index]
                if width % heads or width//heads not in (16, 32, 64, 128):
                    raise ValueError('Flash3D each channel width must divide into heads of 16, 32, 64 or 128 features')
                hidden = width*p[prefix+'_mlp_ratio'][index]
                if int(hidden) != hidden or int(hidden) % 8:
                    raise ValueError('Flash3D MLP hidden widths must be integral multiples of 8')
                scope = p[prefix+'_scope_size'][index]
                kind = p[prefix+'_scope_plan'][index]
                if kind == 'shifted' and not 0 < p[prefix+'_scope_shift'][index] < scope:
                    raise ValueError('Flash3D shifted scopes require 0 < scope_shift < scope_size')
                stride = p[prefix+'_scope_stride'][index] if kind == 'strided' else 1
                alignment = lcm(alignment, p['bucket_size']*scope*stride*2**level)
                index += 1
    if alignment > 131072:
        raise ValueError('Flash3D scope settings require more than 131072 points of alignment; choose compatible scope sizes and strides')
    p['alignment'] = alignment
    return p


def validate(options):
    resolve(options)
