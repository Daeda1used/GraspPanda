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
        chunk_size=('int',1,128))


def resolve(options):
    unknown=set(options)-set(schema())
    if unknown: raise ValueError(f'Unknown PointRWKV parameters: {sorted(unknown)}')
    p={**deepcopy(DEFAULTS),**deepcopy(options)}
    for key,rule in schema().items():
        value=p[key]
        if rule[0]=='int_list':valid=isinstance(value,list) and len(value)==rule[1] and all(type(v) is int and rule[2]<=v<=rule[3] for v in value)
        elif rule[0]=='int':valid=type(value) is int and rule[1]<=value<=rule[2]
        elif rule[0]=='float':valid=type(value) in (int,float) and math.isfinite(value) and rule[1]<=value<=rule[2]
        elif rule[0]=='bool':valid=type(value) is bool
        else:valid=isinstance(value,str) and value in rule[1]
        if not valid:raise ValueError(f'Invalid PointRWKV {key}: expected {rule}')
    if any(c%4 or c%h for c,h in zip(p['stage_channels'],p['stage_heads'])):
        raise ValueError('PointRWKV stage widths must divide into four BQE quarters and complete attention heads')
    if any(a<b for a,b in zip(p['num_points'],p['num_points'][1:])):
        raise ValueError('PointRWKV center counts must be nonincreasing')
    if any(k>n for k,n in zip(p['k_neighbors'],p['num_points'])):
        raise ValueError('PointRWKV local graph neighbors cannot exceed stage center count')
    return p


def validate(options):
    resolve(options)
