"""Serializable sampling policies for compatible dense point hierarchies."""
import math

DENSE_BACKBONES = ('pointnet', 'pointnext', 'pointvector', 'pointmeta', 'pointmlp',
    'pointmamba', 'pointcloud_mamba', 'octformer', 'sonata_ptv3',
    'point_transformer_v2', 'litept', 'pointcnnpp','pointhr', 'flash3d', 'oacnns', 'kpconvx', 'utonia', 'concerto')
HIERARCHIES = ('pointnext', 'pointvector', 'pointmeta')
CHOICES = ('upstream', 'uniform', 'fps', 'pointsp_wrs', 'pointsp_ffps')


def normalize(value, phased=True):
    if isinstance(value, str): value = {'type': value}
    if not isinstance(value, dict): raise ValueError('Sampling policy must be a name or mapping')
    if 'train' in value or 'eval' in value:
        if not phased or set(value) != {'train', 'eval'}:
            raise ValueError('Phased sampling requires exactly train and eval policies, without nesting')
        return {key: normalize(item, phased=False) for key, item in value.items()}
    kind = value.get('type', 'upstream')
    if not isinstance(kind, str) or kind not in CHOICES:
        raise ValueError(f'Sampling type must be one of {CHOICES}')
    fields = {'type'}
    if kind in ('pointsp_wrs', 'pointsp_ffps'): fields |= {'neighbors', 'density_quantile'}
    if kind in ('fps', 'pointsp_ffps'): fields.add('start')
    if kind == 'pointsp_ffps': fields.add('keep_ratio')
    if set(value) - fields: raise ValueError(f'Unknown {kind} sampling parameters: {sorted(set(value)-fields)}')
    if 'neighbors' in value and (type(value['neighbors']) is not int or not 1 <= value['neighbors'] <= 128):
        raise ValueError('Sampling neighbors must be an integer from 1 to 128')
    for key, lower in (('density_quantile', 0), ('keep_ratio', .1)):
        v = value.get(key, .5 if key == 'density_quantile' else .95)
        if type(v) not in (int, float) or not math.isfinite(v) or not lower <= v <= 1:
            raise ValueError(f'Sampling {key} must be in [{lower}, 1]')
    if value.get('start', 'first') not in ('first', 'random'):
        raise ValueError('Sampling start must be first or random')
    return dict(value, type=kind)
