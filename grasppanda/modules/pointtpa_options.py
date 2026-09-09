"""Configuration contracts for native PointTPA branches inside PTv3 encoders."""
import math

DEFAULTS = dict(bottleneck_channels=64, experts=4, group_size=100,
                group_mode='num', dynamic_down=True, dynamic_up=False, scale=1.)


def resolve(value, depths, max_level=4):
    if not isinstance(value, dict) or value.get('type') != 'pointtpa':
        raise ValueError('adaptation must be a mapping with type: pointtpa')
    unknown = set(value) - {'type', 'blocks', *DEFAULTS}
    if unknown:
        raise ValueError(f'Unknown PointTPA parameters: {sorted(unknown)}')
    active_count = sum(depths[:max_level+1])
    blocks = value.get('blocks', list(range(active_count)))
    if (not isinstance(blocks, list) or not blocks or
        any(type(index) is not int or not 0 <= index < active_count for index in blocks) or
        blocks != sorted(set(blocks))):
        raise ValueError('PointTPA blocks must be distinct increasing encoder block indices that contribute to selected feature levels')
    rules = dict(bottleneck_channels=('int',1,512), experts=('int',1,16),
                 group_size=('int',1,4096), group_mode=('choice','num','length'),
                 dynamic_down=('bool',), dynamic_up=('bool',), scale=('float',.001,10))
    result = {'blocks': blocks.copy()}
    for name, default in DEFAULTS.items():
        item = value.get(name, default)
        items = item if isinstance(item, list) else [item]*len(blocks)
        if len(items) != len(blocks):
            raise ValueError(f'PointTPA {name} requires one value or one value per selected block')
        rule = rules[name]
        for item in items:
            if rule[0] == 'choice': valid = item in rule[1:]
            elif rule[0] == 'bool': valid = type(item) is bool
            else:
                valid = type(item) in ((int,) if rule[0] == 'int' else (int,float)) and math.isfinite(item) and rule[1] <= item <= rule[2]
            if not valid: raise ValueError(f'Invalid PointTPA {name}: expected {rule}')
        result[name] = items.copy()
    return result
