"""Effective stage and block controls for native Swin3D grasp features."""
from copy import deepcopy
import math

DEFAULTS = dict(channels=[48,96,192,384,384], depths=[2,4,9,4,4], heads=[6,6,12,24,24],
    window_sizes=[5,7,7,7,7], quant_sizes=[4]*5, grid_size=.005, stem_transformer=True,
    drop_path_rate=.3, decoder_drop_path=.1, rpe_features='xyz', mlp_ratios=4.,
    qkv_bias=True, projection_dropout=0., mlp_dropout=0., activation='gelu',
    gradient_checkpointing=False, bn_eps=1e-5, bn_momentum=.1, norm_eps=1e-5)


def schema():
    result=dict(channels=('int_sequence',1,5,16,2048), depths=('int_sequence',1,5,0,24),
        heads=('int_sequence',1,5,2,64), window_sizes=('int_sequence',1,5,1,16),
        quant_sizes=('int_sequence',1,5,1,8), strides=('int_sequence',0,4,2,4),
        downsample=('choice_list',0,4,('knn','grid')), knn_neighbors=('int_sequence',0,4,1,64),
        decoder_depths=('int_sequence',0,4,0,12), up_neighbors=('int_sequence',0,4,1,64),
        grid_size=('float',.001,.1), stem_transformer=('bool',),
        drop_path_rate=('float',0,.8), decoder_drop_path=('float',0,.8),
        rpe_features=('choice',('xyz','xyz_normals')), gradient_checkpointing=('bool',),
        bn_eps=('float',1e-6,.1), bn_momentum=('float',.001,1), norm_eps=('float',1e-6,.1),
        mlp_ratios=('per_stage',5,('float',1,8)), qkv_bias=('per_stage',5,('bool',)),
        projection_dropout=('per_stage',5,('float',0,.8)), mlp_dropout=('per_stage',5,('float',0,.8)),
        activation=('choice',('gelu','relu','silu')), block_heads=('int_sequence',1,168,2,64))
    for key,bounds in [('block_mlp_ratios',(1,8)),('block_drop_path',(0,.8)),
                       ('block_projection_dropout',(0,.8)),('block_mlp_dropout',(0,.8))]:
        result[key]=('float_list',1,168,*bounds)
    return result


def resolve(options):
    if not isinstance(options,dict):raise ValueError('Swin3D options must be a mapping')
    unknown=set(options)-set(schema())
    if unknown:raise ValueError(f'Unknown Swin3D parameters: {sorted(unknown)}')
    p={**deepcopy(DEFAULTS),**deepcopy(options)}
    if not isinstance(p['channels'],list) or not 1<=len(p['channels'])<=5:
        raise ValueError('Swin3D requires one to five stage widths')
    levels=len(p['channels'])
    for key in ('depths','heads','window_sizes','quant_sizes'):
        if key not in options:p[key]=p[key][:levels]
    if not p['stem_transformer'] and 'depths' not in options:p['depths'][0]=0
    p.setdefault('strides',([3]+[2]*(levels-2)) if levels>1 else [])
    p.setdefault('downsample',['knn']*(levels-1));p.setdefault('knn_neighbors',[16]*(levels-1))
    p.setdefault('decoder_depths',[1]*(levels-1));p.setdefault('up_neighbors',[3]*(levels-1))
    for key,rule in schema().items():
        if key not in p:continue
        value=p[key];kind=rule[0]
        if kind=='int_sequence':valid=isinstance(value,list) and rule[1]<=len(value)<=rule[2] and all(type(v) is int and rule[3]<=v<=rule[4] for v in value)
        elif kind=='float_list':valid=isinstance(value,list) and rule[1]<=len(value)<=rule[2] and all(type(v) in (int,float) and math.isfinite(v) and rule[3]<=v<=rule[4] for v in value)
        elif kind=='choice_list':valid=isinstance(value,list) and rule[1]<=len(value)<=rule[2] and all(v in rule[3] for v in value)
        elif kind=='per_stage':
            values=value if isinstance(value,list) else [value]*levels;scalar=rule[2]
            valid=len(values)==levels and all(type(v) is bool if scalar[0]=='bool' else type(v) in (int,float) and math.isfinite(v) and scalar[1]<=v<=scalar[2] for v in values)
            p[key]=values
        elif kind=='bool':valid=type(value) is bool
        elif kind=='float':valid=type(value) in (int,float) and math.isfinite(value) and rule[1]<=value<=rule[2]
        else:valid=isinstance(value,str) and value in rule[1]
        if not valid:raise ValueError(f'Invalid Swin3D {key}: expected {rule}')
    for key in ('depths','heads','window_sizes','quant_sizes'):
        if len(p[key])!=levels:raise ValueError(f'Swin3D {key} must match the number of stages')
    for key in ('strides','downsample','knn_neighbors','decoder_depths','up_neighbors'):
        if len(p[key])!=levels-1:raise ValueError(f'Swin3D {key} needs one value per transition, in fine-to-coarse order')
    if not p['stem_transformer'] and (levels<2 or p['depths'][0]!=0):
        raise ValueError('A residual Swin3D stem needs at least two stages and depths[0]: 0')
    if any(d<1 for d in p['depths'][0 if p['stem_transformer'] else 1:]):
        raise ValueError('Every active Swin3D encoder stage needs a transformer block')
    for mode,k in zip(p['downsample'],p['knn_neighbors']):
        if mode=='grid' and k!=16:raise ValueError('knn_neighbors applies only to KNN transitions; retain 16 for grid entries')
    stages=block_stages(p)
    for key in ('block_heads','block_mlp_ratios','block_drop_path','block_projection_dropout','block_mlp_dropout'):
        if key in p and len(p[key])!=len(stages):raise ValueError(f'Swin3D {key} needs {len(stages)} values in encoder then decoder execution order')
    heads=p.get('block_heads',[p['heads'][stage] for stage in stages])
    for stage,h in zip(stages,heads):
        if h%2 or p['channels'][stage]%h or p['channels'][stage]//h not in (8,16,32):
            raise ValueError('Swin3D blocks require even head counts and 8, 16 or 32 channels per head')
    # Native construction also uses stage heads before optional block overrides.
    if any(h%2 or c%h or c//h not in (8,16,32) for c,h in zip(p['channels'],p['heads'])):
        raise ValueError('Swin3D stage heads must be even and divide widths into 8, 16 or 32 channels')
    ratios=p.get('block_mlp_ratios',[p['mlp_ratios'][stage] for stage in stages])
    if any(not float(p['channels'][stage]*r).is_integer() for stage,r in zip(stages,ratios)):
        raise ValueError('Swin3D MLP ratios must produce integral hidden widths')
    return p


def block_stages(p):
    return ([stage for stage,d in enumerate(p['depths']) for _ in range(d)] +
            [stage for stage in reversed(range(len(p['decoder_depths']))) for _ in range(p['decoder_depths'][stage])])


def validate(options):
    resolve(options)
