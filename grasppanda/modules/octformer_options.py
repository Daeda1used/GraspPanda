"""CPU-only contracts for the native OctFormer grasp backbone."""
from copy import deepcopy

DEFAULTS = dict(depth=11, full_depth=2, channels=[96,192,384,384],
    num_blocks=[2,2,18,2], num_heads=[6,12,24,24], patch_size=32,
    dilation=4, drop_path=.5, nempty=True, stem_down=2, head_up=2,
    fpn_channel=168, head_drop=[0.,0.], use_dwconv=True, use_checkpoint=True,
    mlp_ratio=4., qkv_bias=True, attn_drop=0., proj_drop=0., use_rpe=True)
STAGE_FIELDS = dict(use_dwconv=('bool',), use_checkpoint=('bool',),
    mlp_ratio=('float',1,8), qkv_bias=('bool',), attn_drop=('float',0,.8),
    proj_drop=('float',0,.8), use_rpe=('bool',))


def schema():
    fields = dict(depth=('int',3,16), full_depth=('int',2,4),
        channels=('int_sequence',1,6,8,768), num_blocks=('int_sequence',1,6,0,24),
        num_heads=('int_sequence',1,6,1,64), patch_size=('int',2,256),
        dilation=('int',1,16), drop_path=('float',0,.8), nempty=('bool',),
        stem_down=('int',1,3), head_up=('int',0,3), fpn_channel=('int',8,512),
        head_drop=('float_list',2,2,0,.8))
    fields.update({key:('per_stage',6,rule) for key,rule in STAGE_FIELDS.items()})
    return fields


def resolve(options):
    result=deepcopy(DEFAULTS)
    result.update(deepcopy(options))
    return result


def stage_value(value,index):
    return value[index] if isinstance(value,list) else value


def validate(options):
    p=resolve(options)
    count=len(p['channels'])
    if len(p['num_blocks'])!=count or len(p['num_heads'])!=count:
        raise ValueError('OctFormer channels, num_blocks and num_heads must describe the same stages')
    minimum=p['depth']-p['stem_down']-count+1
    if minimum<2 or p['full_depth']>minimum:
        raise ValueError('OctFormer depth must accommodate stem_down and all stages, with full_depth no greater than the coarsest stage depth')
    if p['head_up']>p['stem_down']:
        raise ValueError('OctFormer head_up cannot exceed stem_down')
    for key in STAGE_FIELDS:
        if isinstance(p[key],list) and len(p[key])!=count:
            raise ValueError(f'OctFormer {key} requires one value per stage')
    for i,(width,heads) in enumerate(zip(p['channels'],p['num_heads'])):
        if width%heads:
            raise ValueError('OctFormer stage channels must be divisible by their attention heads')
        if not stage_value(p['use_dwconv'],i) and width%8:
            raise ValueError('OctFormer grouped positional convolution requires stage widths divisible by 8')
    if p['channels'][0]<2**p['stem_down']:
        raise ValueError('OctFormer first stage is too narrow for stem_down')
