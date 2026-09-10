"""Architecture controls for the released-code PointRWKV feature hierarchy."""
from copy import deepcopy
import math

DEFAULTS = dict(stage_channels=[384]*3, depths=[4]*3, stage_heads=[8]*3,
    num_points=[2048,1024,512], group_sizes=[32]*3, k_neighbors=[16,8,8],
    graph_iterations=[3]*3, ffn_ratios=[4]*3, patch_channels=[128,256,512],
    decoder_channels=[384]*3, decoder_depths=[2]*3, drop=0., drop_path_rate=.1,
    gradient_checkpointing=False, recurrence_backend='parallel', chunk_size=32)


def schema():
    return dict(stage_channels=('int_list',3,8,768), depths=('int_list',3,1,12),
        stage_heads=('int_list',3,1,32), num_points=('int_list',3,1,8192),
        group_sizes=('int_list',3,1,128), k_neighbors=('int_list',3,1,128),
        graph_iterations=('int_list',3,1,8), ffn_ratios=('int_list',3,1,8),
        patch_channels=('int_list',3,8,1024), decoder_channels=('int_list',3,8,1024),
        decoder_depths=('int_list',3,1,8), drop=('float',0,.8), drop_path_rate=('float',0,.8),
        gradient_checkpointing=('bool',), recurrence_backend=('choice',('native','parallel')),
        chunk_size=('int',1,128), block_heads=('int_sequence',3,36,1,32),
        block_neighbors=('int_sequence',3,36,1,128),
        block_graph_iterations=('int_sequence',3,36,1,8),
        block_ffn_ratios=('int_sequence',3,36,1,8),
        block_drop=('float_list',3,36,0,.8), block_drop_path=('float_list',3,36,0,.8))


def resolve(options):
    unknown=set(options)-set(schema())
    if unknown: raise ValueError(f'Unknown PointRWKV parameters: {sorted(unknown)}')
    p={**deepcopy(DEFAULTS),**deepcopy(options)}
    for key,rule in schema().items():
        if key not in p: continue
        value=p[key]
        if rule[0]=='int_list':valid=isinstance(value,list) and len(value)==rule[1] and all(type(v) is int and rule[2]<=v<=rule[3] for v in value)
        elif rule[0] in ('int_sequence','float_list'):
            allowed=(int,) if rule[0]=='int_sequence' else (int,float)
            valid=isinstance(value,list) and rule[1]<=len(value)<=rule[2] and all(type(v) in allowed and math.isfinite(v) and rule[3]<=v<=rule[4] for v in value)
        elif rule[0]=='int':valid=type(value) is int and rule[1]<=value<=rule[2]
        elif rule[0]=='float':valid=type(value) in (int,float) and math.isfinite(value) and rule[1]<=value<=rule[2]
        elif rule[0]=='bool':valid=type(value) is bool
        else:valid=isinstance(value,str) and value in rule[1]
        if not valid:raise ValueError(f'Invalid PointRWKV {key}: expected {rule}')
    if any(c%4 for c in p['stage_channels']):
        raise ValueError('PointRWKV stage widths must divide into four BQE quarters')
    if any(a<b for a,b in zip(p['num_points'],p['num_points'][1:])):
        raise ValueError('PointRWKV center counts must be nonincreasing')
    for key in schema():
        if key.startswith('block_') and key in p and len(p[key])!=sum(p['depths']):
            raise ValueError(f'PointRWKV {key} needs one value per encoder block: sum(depths)')
    index=0
    for stage,depth in enumerate(p['depths']):
        for _ in range(depth):
            settings=block_options(p,stage,index)
            if p['stage_channels'][stage]%settings['num_heads']:
                raise ValueError(f'PointRWKV block {index} heads must divide its stage width')
            if settings['k']>p['num_points'][stage]:
                raise ValueError(f'PointRWKV block {index} neighbors exceed its stage center count')
            index+=1
    return p


def block_options(options,stage,index):
    """Flat block lists override their stage or shared defaults, fine to coarse."""
    mappings={'num_heads':('block_heads','stage_heads'), 'k':('block_neighbors','k_neighbors'),
        'graph_iter':('block_graph_iterations','graph_iterations'),
        'ffn_ratio':('block_ffn_ratios','ffn_ratios')}
    result={name:options[override][index] if override in options else options[default][stage]
        for name,(override,default) in mappings.items()}
    result['drop']=options['block_drop'][index] if 'block_drop' in options else options['drop']
    return result


def validate(options):
    resolve(options)
