"""Contracts for the official KPConvX grid hierarchy."""
from copy import deepcopy
import math

DEFAULTS = dict(layer_blocks=[3, 3, 9, 12, 3], init_channels=64,
    channel_scaling=math.sqrt(2), neighbor_limits=[12, 16, 20, 20, 20],
    shell_sizes=[1, 14, 28], subsample_size=.02, radius_scaling=2.2,
    kp_radius=2.3, kp_sigma=2.3, kp_influence='linear', kp_aggregation='nearest',
    kp_mode='kpconvx', inv_groups=8, inv_act='sigmoid', inv_grp_norm=True,
    first_inv_layer=1, share_kp=False, kpx_upcut=False, decoder_layer=True,
    drop_path_rate=0., norm='batch', bn_momentum=.1, activation='leaky_relu',
    modulation_scope='native')


def schema():
    return dict(layer_blocks=('int_sequence', 1, 6, 1, 24), init_channels=('int', 16, 256),
        channel_scaling=('float', 1, 2), stage_channels=('int_sequence', 1, 6, 16, 1024),
        neighbor_limits=('int_sequence', 1, 6, 1, 128),
        shell_sizes=('int_sequence', 2, 4, 1, 64), subsample_size=('float', .001, .2),
        radius_scaling=('float', 1.1, 4), kp_radius=('float', .1, 8),
        kp_sigma=('float', .1, 8), kp_influence=('choice', ('constant', 'linear', 'gaussian')),
        kp_aggregation=('choice', ('nearest', 'sum')), kp_mode=('choice', ('kpconvx', 'kpconvd')),
        inv_groups=('int', -64, 64), inv_act=('choice', ('sigmoid', 'tanh', 'softmax', 'none')),
        inv_grp_norm=('bool',), first_inv_layer=('int', 0, 6), share_kp=('bool',),
        kpx_upcut=('bool',), decoder_layer=('bool',), drop_path_rate=('float', 0, .8),
        norm=('choice', ('batch', 'group', 'layer', 'none')), bn_momentum=('float', .001, 1),
        activation=('choice', ('leaky_relu', 'relu', 'gelu', 'silu')),
        modulation_scope=('choice', ('native', 'point')))


def resolve(options):
    return {**deepcopy(DEFAULTS), **deepcopy(options)}


def widths(p):
    if 'stage_channels' in p: return p['stage_channels']
    return [int(math.ceil((p['init_channels']*p['channel_scaling']**i-.1)/16)*16)
            for i in range(len(p['layer_blocks']))]


def validate(options):
    p=resolve(options); stages=len(p['layer_blocks'])
    if len(p['neighbor_limits']) != stages:
        raise ValueError('KPConvX requires one neighbor limit per encoder stage')
    if p['shell_sizes'][0] != 1 or sum(p['shell_sizes']) > 128:
        raise ValueError('KPConvX kernel shells must start with one center point and total at most 128 points')
    if p['first_inv_layer'] > stages:
        raise ValueError('KPConvX first_inv_layer must not exceed the encoder stage count')
    channels=widths(p)
    if len(channels)!=stages or any(c % 16 for c in channels):
        raise ValueError('KPConvX stage_channels must have one multiple-of-16 width per stage')
    if max(channels)>1024:
        raise ValueError('KPConvX derived stage widths must not exceed 1024 channels')
    groups=abs(p['inv_groups'])
    if groups and any(c % groups for c in channels):
        raise ValueError('KPConvX inv_groups must divide every derived stage width (negative means channels per group)')
    if p['modulation_scope']=='point' and p['inv_grp_norm'] and p['kp_mode']=='kpconvx' and groups:
        if min(c//groups if p['inv_groups']>0 else groups for c in channels)<2:
            raise ValueError('KPConvX point modulation normalization requires at least two channels per kernel group')


def validate_config(config):
    from ..module_options import unpack
    choice,options=unpack(config.modules.get('backbone','upstream'))
    if choice!='kpconvx': return
    if config.action in ('train','train_short') and config.batch_size<2:
        raise ValueError('KPConvX training requires batch_size >= 2 for coarse native batch normalization')
    if config.action == 'train_short' and config.frame+config.batch_size>256:
        raise ValueError('KPConvX short training requires consecutive frames within one scene')
    if config.action=='train' and config.train_batch_limit and config.scene*256+config.frame+config.batch_size>25600:
        raise ValueError('KPConvX bounded training requires a full batch within the training split')
