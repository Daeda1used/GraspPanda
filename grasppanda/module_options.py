"""Serializable, method-specific component arguments; no arbitrary imports."""
import math


def unpack(value):
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, dict) or not isinstance(value.get('type'), str):
        raise ValueError('A component must be a name or a mapping with type and parameters')
    return value['type'], {k: v for k, v in value.items() if k != 'type'}


def schema(method, slot, choice):
    common = {'activation': ('choice', ('relu', 'gelu', 'silu')),
              'normalization': ('choice', ('batch', 'group', 'none'))}
    if slot == 'backbone' and choice == 'pointmlp':
        return {'embed_dim': ('int', 8, 128), 'dim_expansion': ('int_list', 4, 1, 4),
                'pre_blocks': ('int_list', 4, 1, 12), 'pos_blocks': ('int_list', 4, 1, 12),
                'k_neighbors': ('int_list', 4, 4, 128), 'stage_points': ('int_list', 4, 4, 2048),
                'decoder_channels': ('int_list', 4, 8, 2048), 'decoder_blocks': ('int_list', 4, 1, 12),
                'res_expansion': ('float', .25, 4), 'activation': common['activation'],
                'normalize': ('choice', ('anchor', 'center'))}
    if slot == 'backbone' and choice == 'pointnext':
        return {'width': ('int', 8, 128), 'blocks': ('blocks',),
                'nsample': ('int', 4, 128), 'radius': ('float', .005, .5),
                'radius_scaling': ('float', 1, 4), 'expansion': ('int', 1, 8),
                'activation': ('choice', ('relu', 'gelu', 'silu')),
                'reduction': ('choice', ('max', 'mean', 'sum')),
                'decoder_layers': ('int', 1, 4)}
    if slot == 'backbone' and choice == 'pointnet' and method != 'graspness':
        return {**common, 'local_channels': ('channels',), 'global_channels': ('channels',),
                'fusion_channels': ('channels',), 'dropout': ('float', 0, .8)}
    if slot == 'crop' and choice == 'multiscale':
        return {'radius_factors': ('radii',)}
    if method == 'graspness' and slot == 'crop' and choice == 'finegrasp':
        return {'nsample': ('int', 4, 128), 'radius_factors': ('radii',)}
    if slot == 'crop' and choice == 'cylinder':
        return {**common, 'hidden_channels': ('channels',), 'radius_factors': ('radii',),
                'nsample': ('int', 4, 256), 'pooling': ('choice', ('max', 'mean', 'attention'))}
    return {}


def validate_options(method, slot, choice, options):
    fields = schema(method, slot, choice)
    if set(options) - set(fields):
        raise ValueError(f'{method}/{slot}/{choice}: unknown parameters {sorted(set(options)-set(fields))}')
    for key, value in options.items():
        rule = fields[key]
        valid = False
        if rule[0] == 'choice':
            valid = isinstance(value, str) and value in rule[1]
        elif rule[0] in ('int', 'float'):
            valid = (type(value) in ((int,) if rule[0] == 'int' else (int, float))
                     and math.isfinite(value) and rule[1] <= value <= rule[2])
        elif rule[0] == 'channels':
            valid = isinstance(value, list) and 1 <= len(value) <= 8 and all(type(v) == int and 8 <= v <= 2048 for v in value)
        elif rule[0] == 'radii':
            valid = isinstance(value, list) and 1 <= len(value) <= 8 and all(type(v) in (int, float) and math.isfinite(v) and .1 <= v <= 4 for v in value)
        elif rule[0] == 'blocks':
            valid = isinstance(value, list) and len(value) == 5 and all(type(v) == int and 1 <= v <= 12 for v in value)
        elif rule[0] == 'int_list':
            valid = isinstance(value, list) and len(value) == rule[1] and all(type(v) == int and rule[2] <= v <= rule[3] for v in value)
        if not valid:
            raise ValueError(f'Invalid {method}/{slot}/{choice} parameter {key}: expected {rule}')
    if choice == 'pointmlp':
        sizes = options.get('stage_points', [1024, 256, 64, 16])
        neighbors = options.get('k_neighbors', [32, 32, 32, 16])
        if any(b > a or k > a for a, b, k in zip(sizes, sizes[1:], neighbors[1:])):
            raise ValueError('PointMLP stage points must decrease; neighbors must fit the preceding stage')
        width = options.get('embed_dim', 64)
        for factor in options.get('dim_expansion', [2, 2, 2, 2]):
            width *= factor
            if width > 2048:
                raise ValueError('PointMLP expanded stage width must not exceed 2048')
    return options
