"""Serializable architecture and neighborhood controls for native PointCNN++."""
from copy import deepcopy
import math

DEFAULTS = dict(grid_size=.01, base_channels=32,
    channels=[32,64,128,256,256,128,96,96], depths=[2,3,4,6,2,2,2,2],
    normalize_features=False, bn_momentum=.01, bn_eps=.001,
    activation='relu', block_kernel_sizes=3, block_radius_scalers=2.5)


def schema():
    return dict(grid_size=('float', .001, .1), base_channels=('int', 8, 128),
        channels=('int_list', 8, 8, 512), depths=('int_list', 8, 1, 12),
        normalize_features=('bool',), bn_momentum=('float', .001, 1),
        bn_eps=('float', 1e-6, .1), activation=('choice', ('relu', 'gelu', 'silu')),
        block_kernel_sizes=('per_block', 96, ('int', 1, 5)),
        block_radius_scalers=('per_stage', 96, ('float', .1, 8)),
        block_activations=('choice_list', 1, 96, ('relu', 'gelu', 'silu')))


def resolve(options):
    unknown=set(options)-set(schema())
    if unknown: raise ValueError(f'Unknown PointCNN++ parameters: {sorted(unknown)}')
    p={**deepcopy(DEFAULTS), **deepcopy(options)}
    for key,low,high in [('channels',8,512),('depths',1,12)]:
        if not isinstance(p[key],(list,tuple)) or len(p[key])!=8 or any(type(x) is not int or not low<=x<=high for x in p[key]):
            raise ValueError(f'PointCNN++ {key} needs eight integers in [{low}, {high}]')
    if type(p['base_channels']) is not int or not 8<=p['base_channels']<=128:
        raise ValueError('PointCNN++ base_channels must be an integer in [8, 128]')
    for key,low,high in [('grid_size',.001,.1),('bn_momentum',.001,1),('bn_eps',1e-6,.1)]:
        if type(p[key]) not in (int,float) or not math.isfinite(p[key]) or not low<=p[key]<=high:
            raise ValueError(f'PointCNN++ {key} must be finite and in [{low}, {high}]')
    if type(p['normalize_features']) is not bool or p['activation'] not in ('relu','gelu','silu'):
        raise ValueError('Invalid PointCNN++ normalization or activation')
    count=sum(p['depths'])
    p.setdefault('block_activations',p['activation'])
    for key in ('block_kernel_sizes','block_radius_scalers','block_activations'):
        value=p[key]; value=list(value) if isinstance(value,(list,tuple)) else [value]*count
        if len(value)!=count: raise ValueError(f'PointCNN++ {key} needs one value per residual block ({count})')
        p[key]=value
    if any(type(v) is not int or v not in (1,3,5) for v in p['block_kernel_sizes']):
        raise ValueError('PointCNN++ residual kernel sizes must be 1, 3 or 5')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not .1<=v<=8 for v in p['block_radius_scalers']):
        raise ValueError('PointCNN++ residual radius scalers must be finite and in [.1, 8]')
    if any(v not in ('relu','gelu','silu') for v in p['block_activations']):
        raise ValueError('Invalid PointCNN++ block activation')
    return p


def validate(options):
    resolve(options)
