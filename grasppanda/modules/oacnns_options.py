"""Configuration contracts for the native omni-adaptive sparse hierarchy."""
from copy import deepcopy

DEFAULTS = dict(embed_channels=64, enc_channels=[64, 64, 128, 256],
    enc_depth=[2, 3, 6, 4], dec_channels=[96, 96, 128, 256],
    point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],
    stem_depth=3, decoder_layers=[2]*4, activation='relu', bn_eps=.001,
    bn_momentum=.01, relation_normalization='cluster', grid_origin='zero',
    sparse_padding='stride')


def schema():
    return dict(embed_channels=('int', 8, 512), enc_channels=('int_sequence', 1, 6, 8, 512),
        enc_depth=('int_sequence', 1, 6, 0, 24), dec_channels=('int_sequence', 1, 6, 8, 512),
        point_grid_size=('float_matrix', 1, 6, 1, 8, 1, 512),
        stem_depth=('int', 1, 6), decoder_layers=('int_sequence', 1, 6, 1, 6),
        activation=('choice', ('relu', 'gelu', 'silu')), bn_eps=('float', 1e-6, .1),
        bn_momentum=('float', .001, 1), relation_normalization=('choice', ('cluster', 'native')),
        grid_origin=('choice', ('zero', 'native')), sparse_padding=('choice', ('stride', 'native')))


def resolve(options):
    return {**deepcopy(DEFAULTS), **deepcopy(options)}


def validate(options):
    p = resolve(options); count = len(p['enc_channels'])
    if any(len(p[key]) != count for key in ('enc_depth', 'dec_channels', 'point_grid_size', 'decoder_layers')):
        raise ValueError('OA-CNNs needs matching encoder, decoder and grid lists: one entry per stage')


def selected(config):
    from ..module_options import unpack
    return unpack(config.modules.get('backbone', 'upstream'))[0] == 'oacnns'


def validate_config(config):
    if not selected(config): return
    if config.action in ('train', 'train_short') and config.batch_size < 2:
        raise ValueError('OA-CNNs training requires batch_size >= 2 for native batch normalization after coarse pooling')
    if config.action == 'train_short' and config.frame+config.batch_size > 256:
        raise ValueError('OA-CNNs short training uses consecutive frames within one scene; choose an earlier first frame')
    if config.action == 'train' and config.train_batch_limit and config.scene*256+config.frame+config.batch_size > 25600:
        raise ValueError('OA-CNNs bounded training requires at least one full batch within the training split')
