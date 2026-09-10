"""Effective architecture controls for the native PointHR hierarchy."""
from copy import deepcopy
import math

DEFAULTS = dict(patch_embed_depth=1, patch_embed_channels=32, patch_embed_groups=4,
    patch_embed_neighbours=8, enc_depths=[1, 1, 5, 4], enc_blocks=[2]*4,
    enc_channels=[64, 32, 32, 32], enc_groups=[8, 4, 4, 4], enc_neighbours=[16]*4,
    dec_depths=[1]*4, dec_groups=[4, 4, 8, 16], dec_neighbours=[16]*4,
    grid_sizes=[.005, .01, .02, .04], attn_qkv_bias=True, pe_multiplier=False,
    pe_bias=True, attn_drop_rate=0., drop_path_rate=0., gradient_checkpointing=False,
    unpool_backend='map', fusion='sum', bn_momentum=.1, bn_eps=1e-5)


def schema():
    return dict(patch_embed_depth=('int', 1, 8), patch_embed_channels=('int', 8, 256),
        patch_embed_groups=('int', 1, 64), patch_embed_neighbours=('int', 2, 128),
        enc_depths=('int_list', 4, 1, 12), enc_blocks=('int_list', 4, 1, 8),
        enc_channels=('int_list', 4, 8, 256), enc_groups=('int_list', 4, 1, 64),
        enc_neighbours=('int_list', 4, 2, 128), dec_depths=('int_list', 4, 1, 12),
        dec_channels=('int_list', 4, 8, 1024), dec_groups=('int_list', 4, 1, 128),
        dec_neighbours=('int_list', 4, 2, 128), grid_sizes=('float_list', 4, 4, .001, .5),
        attn_qkv_bias=('bool',), pe_multiplier=('bool',), pe_bias=('bool',),
        attn_drop_rate=('float', 0, .8), drop_path_rate=('float', 0, .8),
        gradient_checkpointing=('bool',), unpool_backend=('choice', ('map', 'interp')),
        fusion=('choice', ('sum', 'mean')), bn_momentum=('float', .001, 1), bn_eps=('float', 1e-6, .1))


def resolve(options):
    unknown = set(options) - set(schema())
    if unknown: raise ValueError(f'Unknown PointHR parameters: {sorted(unknown)}')
    p = {**deepcopy(DEFAULTS), **deepcopy(options)}
    if not isinstance(p['enc_channels'], list) or len(p['enc_channels']) != 4 or any(type(v) is not int for v in p['enc_channels']):
        raise ValueError('PointHR enc_channels requires four integer widths')
    p.setdefault('dec_channels', [p['patch_embed_channels']] + [p['enc_channels'][-1]*2**i for i in range(3)])
    for key, rule in schema().items():
        v = p[key]; kind = rule[0]
        if kind == 'int_list': valid = isinstance(v, list) and len(v) == rule[1] and all(type(x) is int and rule[2] <= x <= rule[3] for x in v)
        elif kind == 'float_list': valid = isinstance(v, list) and len(v) == 4 and all(type(x) in (int, float) and math.isfinite(x) and rule[3] <= x <= rule[4] for x in v)
        elif kind in ('int', 'float'): valid = type(v) in ((int,) if kind == 'int' else (int, float)) and math.isfinite(v) and rule[1] <= v <= rule[2]
        elif kind == 'bool': valid = type(v) is bool
        else: valid = isinstance(v, str) and v in rule[1]
        if not valid: raise ValueError(f'Invalid PointHR {key}: expected {rule}')
    if p['patch_embed_channels'] % p['patch_embed_groups']:
        raise ValueError('PointHR patch width must be divisible by patch groups')
    if any(c % g for c, g in zip(p['enc_channels'], p['enc_groups'])) or any(c % g for c, g in zip(p['dec_channels'], p['dec_groups'])):
        raise ValueError('PointHR stage widths must be divisible by their attention group counts')
    if any(a >= b for a, b in zip(p['grid_sizes'], p['grid_sizes'][1:])):
        raise ValueError('PointHR grid_sizes must increase strictly in metres')
    return p


def validate(options):
    resolve(options)
