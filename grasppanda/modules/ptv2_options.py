"""CPU-only contracts for native grouped-vector Point Transformer V2."""
from copy import deepcopy

DEFAULTS = dict(patch_embed_depth=1, patch_embed_channels=48, patch_embed_groups=6,
    patch_embed_neighbours=8, enc_depths=[2, 2, 6, 2], enc_channels=[96, 192, 384, 512],
    enc_groups=[12, 24, 48, 64], enc_neighbours=[16]*4, dec_depths=[1]*4,
    dec_channels=[48, 96, 192, 384], dec_groups=[6, 12, 24, 48], dec_neighbours=[16]*4,
    grid_sizes=[.06, .12, .24, .48], attn_qkv_bias=True, pe_multiplier=False,
    pe_bias=True, attn_drop_rate=0., drop_path_rate=0., enable_checkpoint=False,
    unpool_backend='map')


def schema():
    fields = dict(patch_embed_depth=('int', 1, 12), patch_embed_channels=('int', 8, 768),
        patch_embed_groups=('int', 1, 128), patch_embed_neighbours=('int', 1, 128),
        grid_sizes=('float_list', 1, 6, .001, 2), attn_qkv_bias=('bool',),
        pe_multiplier=('bool',), pe_bias=('bool',), attn_drop_rate=('float', 0, .8),
        drop_path_rate=('float', 0, .8), enable_checkpoint=('bool',),
        unpool_backend=('choice', ('map', 'interp')))
    for prefix in ('enc', 'dec'):
        for key, low, high in (('depths', 1, 12), ('channels', 8, 768),
                                ('groups', 1, 128), ('neighbours', 1, 128)):
            fields[prefix+'_'+key] = ('int_sequence', 1, 6, low, high)
    return fields


def resolve(options):
    return {**deepcopy(DEFAULTS), **deepcopy(options)}


def validate(options):
    p = resolve(options)
    count = len(p['enc_depths'])
    if any(len(p[key]) != count for key in DEFAULTS if key.startswith(('enc_', 'dec_')) or key == 'grid_sizes'):
        raise ValueError('PTv2 encoder, decoder and grid lists must describe the same number of stages')
    pairs = [(p['patch_embed_channels'], p['patch_embed_groups'])]
    for prefix in ('enc', 'dec'):
        pairs.extend(zip(p[prefix+'_channels'], p[prefix+'_groups']))
    if any(channels % groups for channels, groups in pairs):
        raise ValueError('PTv2 patch and stage channels must be divisible by their attention groups')


def selected(config):
    from grasppanda.module_options import unpack
    return unpack(config.modules.get('backbone', 'upstream'))[0] == 'point_transformer_v2'


def validate_config(config):
    if not selected(config):
        return
    if config.action in ('train', 'train_check', 'train_smoke') and config.batch_size < 2:
        raise ValueError('PTv2 training requires batch_size >= 2 because coarse grid pooling can leave one point per scene; inference supports batch 1')
    if config.action in ('train_check', 'train_smoke') and config.frame + config.batch_size > 256:
        raise ValueError('PTv2 short training uses consecutive frames within one scene; choose an earlier first frame')
    if config.action == 'train' and config.train_batch_limit and config.scene * 256 + config.frame + config.batch_size > 25600:
        raise ValueError('PTv2 bounded training must include at least one full batch within the training split')
