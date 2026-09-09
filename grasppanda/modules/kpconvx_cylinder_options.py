"""Configuration contracts for kernel-point aggregation inside grasp cylinders."""
from copy import deepcopy

DEFAULTS=dict(channels=[64,64,64],local_neighbors=8,expansion=4,attention_groups=8,
    attention_activation='sigmoid',shell_sizes=[1,14,28],kernel_radius=1.,kernel_sigma=.5,
    influence='linear',block='kpconvx',modulation_norm=True,modulation_scope='cylinder',
    normalization='layer',bn_momentum=.1,activation='gelu',use_upcut=False,
    layer_scale=0.,drop_path=0.,nsample=16,radius_factors=[1.],pooling='max',
    chunk_size=128,checkpoint=True)


def schema():
    return dict(channels=('int_sequence',2,9,16,256),local_neighbors=('per_block',8,('int',1,128)),
        expansion=('per_block',8,('int',1,8)),attention_groups=('per_block',8,('int',-64,64)),
        attention_activation=('choice',('sigmoid','tanh','softmax','none')),
        shell_sizes=('int_sequence',2,4,1,64),kernel_radius=('per_stage',8,('float',.01,8)),
        kernel_sigma=('per_stage',8,('float',.01,8)),influence=('choice',('constant','linear','gaussian')),
        block=('choice',('kpconvx','kpconvd')),modulation_norm=('bool',),
        modulation_scope=('choice',('cylinder','point')),normalization=('choice',('layer','group','batch','none')),
        bn_momentum=('float',.001,1),activation=('choice',('leaky_relu','relu','gelu','silu')),
        use_upcut=('bool',),layer_scale=('per_stage',8,('float',0,1)),
        drop_path=('per_stage',8,('float',0,.8)),nsample=('int',4,128),radius_factors=('radii',),
        pooling=('choice',('max','mean','attention')),chunk_size=('int',0,4096),checkpoint=('bool',))


def resolve(options):
    p={**deepcopy(DEFAULTS),**deepcopy(options)};depth=len(p['channels'])-1
    for key in ('local_neighbors','expansion','attention_groups','kernel_radius','kernel_sigma','layer_scale','drop_path'):
        if not isinstance(p[key],list):p[key]=[p[key]]*depth
    return p


def validate(options):
    p=resolve(options);depth=len(p['channels'])-1
    for key in ('local_neighbors','expansion','attention_groups','kernel_radius','kernel_sigma','layer_scale','drop_path'):
        if len(p[key])!=depth:raise ValueError(f'KPConvX cylinder {key} needs one value per block, or a scalar')
    if any(c%8 for c in p['channels']):
        raise ValueError('KPConvX cylinder widths must be multiples of 8')
    if max(p['local_neighbors'])>p['nsample']:
        raise ValueError('KPConvX cylinder local_neighbors must not exceed nsample')
    if p['shell_sizes'][0]!=1 or sum(p['shell_sizes'])>128:
        raise ValueError('KPConvX cylinder shells must start with one center point and total at most 128')
    for c,g in zip(p['channels'][:-1],p['attention_groups']):
        if g and c%abs(g):raise ValueError('KPConvX cylinder attention groups must divide the block input width')
        if g and p['block']=='kpconvx' and p['modulation_norm'] and p['modulation_scope']=='point':
            if (c//g if g>0 else -g)<2:raise ValueError('Point modulation normalization needs at least two channels per kernel group')
    if p['use_upcut'] and len({c*e for c,e in zip(p['channels'][:-1],p['expansion'])})>1:
        raise ValueError('KPConvX expanded shortcuts require equal expanded input widths across consecutive blocks')
    if p['normalization']=='batch' and p['chunk_size']:
        raise ValueError('Batch normalization requires chunk_size: 0 to retain complete grouped-batch statistics')
