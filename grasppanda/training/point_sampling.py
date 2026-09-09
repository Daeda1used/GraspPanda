"""Observation sampling with explicit point-row correspondence.

PointSP weighting and local/global removal follow its released sampling rules.
The CPU neighbor search and fixed-size padding are GraspPanda adaptations.
See docs/THIRD_PARTY.md for the upstream references and notices.
"""
import math


def validate(options):
    from .options import finite
    if not isinstance(options, dict):
        raise ValueError('augmentation.resampling must be a mapping')
    kind = options.get('type', 'uniform')
    choices = {'uniform': set(), 'pointsp_wrs': {'neighbors', 'density_quantile'},
               'pointsp_lgd': {'global_fraction'}}
    if not isinstance(kind, str) or kind not in choices:
        raise ValueError('resampling.type must be uniform, pointsp_wrs or pointsp_lgd')
    allowed = {'type', 'keep_ratio', 'probability'} | choices[kind]
    if set(options) - allowed:
        raise ValueError(f'Unknown {kind} resampling options: {sorted(set(options)-allowed)}')
    ratio = options.get('keep_ratio', [.5, 1.])
    if isinstance(ratio, list):
        valid = len(ratio) == 2 and all(finite(v, .1, 1.) for v in ratio) and ratio[0] <= ratio[1]
    else:
        valid = finite(ratio, .1, 1.)
    if not valid:
        raise ValueError('resampling.keep_ratio requires a value or ordered [min, max] in [0.1, 1]')
    if not finite(options.get('probability', 1.), 0, 1):
        raise ValueError('resampling.probability must be between 0 and 1')
    if 'neighbors' in options and (type(options['neighbors']) is not int or not 1 <= options['neighbors'] <= 128):
        raise ValueError('resampling.neighbors must be an integer from 1 to 128')
    if 'density_quantile' in options and not finite(options['density_quantile'], 0, 1):
        raise ValueError('resampling.density_quantile must be between 0 and 1')
    fraction = options.get('global_fraction', 'random')
    if fraction != 'random' and not finite(fraction, 0, 1):
        raise ValueError('resampling.global_fraction must be random or a value between 0 and 1')


def density_weights(points, neighbors=20, quantile=.5):
    """Count nearby neighbors using PointSP's quantile of mean squared distances.

    Self and repeated observations participate, as in the author implementation.
    Exact CPU KD-tree queries avoid materializing an N-by-N distance matrix.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Point sampling requires finite camera XYZ with shape [N, 3]')
    distance, _ = cKDTree(points).query(points, k=min(neighbors, len(points)), workers=1)
    squared = np.square(distance.reshape(len(points), -1))
    threshold = np.quantile(squared.mean(axis=1), quantile)
    weights = (squared <= threshold).sum(axis=1).astype(np.float64)
    return weights / weights.sum()


def retained_indices(points, options):
    """Select original rows, before the caller restores its fixed batch size."""
    import numpy as np
    validate(options)
    count = len(points)
    indices = np.arange(count)
    probability = options.get('probability', 1.)
    if count <= 1024 or probability == 0 or (probability < 1 and np.random.random() >= probability):
        return indices
    ratio = options.get('keep_ratio', [.5, 1.])
    if isinstance(ratio, list):
        ratio = math.exp(np.random.uniform(math.log(ratio[0]), math.log(ratio[1])))
    remove = min(int(count * (1 - ratio)), count - 1024)
    if not remove:
        return indices
    kind = options.get('type', 'uniform')
    if kind == 'pointsp_lgd':
        fraction = options.get('global_fraction', 'random')
        if fraction == 'random': fraction = np.random.random()
        center = points[np.random.randint(count)]
        order = np.argsort(np.square(points.astype(np.float64) - center).sum(axis=1), kind='stable')
        pool = remove + int((count - remove) * fraction)
        discarded = np.random.choice(order[:pool], remove, replace=False)
        mask = np.ones(count, dtype=bool)
        mask[discarded] = False
        return indices[mask]
    weights = (density_weights(points, options.get('neighbors', 20), options.get('density_quantile', .5))
               if kind == 'pointsp_wrs' else None)
    return np.sort(np.random.choice(count, count - remove, replace=False, p=weights))
