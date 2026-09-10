"""CPU-only stage contracts for the native LitePT encoder and decoder."""
from copy import deepcopy

DEFAULTS = dict(order=['z', 'z-trans', 'hilbert', 'hilbert-trans'], stride=[2]*4,
    enc_depths=[2, 2, 2, 6, 2], enc_channels=[36, 72, 144, 252, 504],
    enc_num_head=[2, 4, 8, 14, 28], enc_patch_size=[1024]*5,
    enc_conv=[True, True, True, False, False], enc_attn=[False, False, False, True, True],
    enc_rope_freq=[100.]*5, dec_depths=[0]*4, dec_channels=[72, 72, 144, 252],
    dec_num_head=[4, 4, 8, 14], dec_patch_size=[1024]*4,
    dec_conv=[False]*4, dec_attn=[False]*4, dec_rope_freq=[100.]*4,
    mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
    drop_path=.3, pre_norm=True, shuffle_orders=True, pooling='max', rope_backend='cuda')


def schema():
    fields = dict(order=('choice_list', 1, 4, ('z', 'z-trans', 'hilbert', 'hilbert-trans')),
        stride=('int_sequence', 0, 5, 1, 8), mlp_ratio=('float', .5, 8),
        qkv_bias=('bool',), qk_scale=('float', .0001, 10), attn_drop=('float', 0, .8),
        proj_drop=('float', 0, .8), drop_path=('float', 0, .8), pre_norm=('bool',),
        shuffle_orders=('bool',), pooling=('choice', ('max', 'mean', 'min', 'sum')),
        rope_backend=('choice', ('cuda', 'torch')))
    for prefix, minimum, maximum in (('enc', 1, 6), ('dec', 0, 5)):
        for key, low, high in (('depths', 1 if prefix == 'enc' else 0, 24),
                ('channels', 8, 768), ('num_head', 1, 128), ('patch_size', 1, 4096)):
            fields[prefix+'_'+key] = ('int_sequence', minimum, maximum, low, high)
        fields[prefix+'_conv'] = ('per_stage', maximum, ('bool',))
        fields[prefix+'_attn'] = ('per_stage', maximum, ('bool',))
        fields[prefix+'_rope_freq'] = ('per_stage', maximum, ('float', 1., 10000.))
    return fields


def resolve(options):
    p = {**deepcopy(DEFAULTS), **deepcopy(options)}
    for prefix in ('enc', 'dec'):
        count = len(p[prefix+'_depths'])
        for key in ('conv', 'attn', 'rope_freq'):
            name = prefix+'_'+key
            if not isinstance(p[name], list): p[name] = [p[name]]*count
    return p


def validate(options):
    p = resolve(options); count = len(p['enc_depths'])
    if len(p['stride']) != count-1:
        raise ValueError('LitePT needs one pooling stride between each pair of encoder stages')
    for prefix, size in (('enc', count), ('dec', count-1)):
        if any(len(p[key]) != size for key in p if key.startswith(prefix+'_')):
            raise ValueError('LitePT encoder lists need one entry per encoder stage; decoder lists need one fewer')
        for width, heads, active, depth in zip(p[prefix+'_channels'], p[prefix+'_num_head'], p[prefix+'_attn'], p[prefix+'_depths']):
            if active and depth and (width % heads or (width//heads) % 6 or width//heads > 252):
                raise ValueError('Active LitePT attention needs channels divisible by heads and a head width divisible by 6, at most 252')


def selected(config):
    from grasppanda.module_options import unpack
    return unpack(config.modules.get('backbone', 'upstream'))[0] == 'litept'


def validate_config(config):
    if not selected(config): return
    if config.action in ('train', 'train_short') and config.batch_size < 2:
        raise ValueError('LitePT training requires batch_size >= 2 for native batch normalization after coarse pooling')
    if config.action == 'train_short' and config.frame+config.batch_size > 256:
        raise ValueError('LitePT short training uses consecutive frames within one scene; choose an earlier first frame')
    if config.action == 'train' and config.train_batch_limit and config.scene*256+config.frame+config.batch_size > 25600:
        raise ValueError('LitePT bounded training requires at least one full batch within the training split')
