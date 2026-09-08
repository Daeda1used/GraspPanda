"""Training controls with explicit supervision and method contracts."""
import math

METHODS = ('graspnet_baseline', 'pointnet2_upgrade', 'graspness')
LOSS_TERMS = {
    'graspnet_baseline': {
        'objectness': ('loss/stage1_objectness_loss', 1.),
        'view': ('loss/stage1_view_loss', 1.),
        'score': ('loss/stage2_grasp_score_loss', .2),
        'angle': ('loss/stage2_grasp_angle_class_loss', .2),
        'width': ('loss/stage2_grasp_width_loss', .2),
        'tolerance': ('loss/stage2_grasp_tolerance_loss', .2),
    },
    'graspness': {
        'objectness': ('loss/stage1_objectness_loss', 1.),
        'graspness': ('loss/stage1_graspness_loss', 10.),
        'view': ('loss/stage2_view_loss', 100.),
        'score': ('loss/stage3_score_loss', 15.),
        'width': ('loss/stage3_width_loss', 10.),
    },
}
LOSS_TERMS['pointnet2_upgrade'] = LOSS_TERMS['graspnet_baseline']


def finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate_training_options(config):
    for name in ('loss', 'augmentation'):
        value = getattr(config, name)
        if not isinstance(value, dict):
            raise ValueError(f'{name} must be a mapping')
        if value and (config.method not in METHODS or config.action not in ('train', 'train_check')):
            raise ValueError(f'{name} overrides require a registered training adapter: {METHODS}')
    loss = config.loss
    if set(loss) - {'weights'} or not isinstance(loss.get('weights', {}), dict):
        raise ValueError('loss accepts a weights mapping')
    terms = LOSS_TERMS.get(config.method, {})
    weights = loss.get('weights', {})
    if set(weights) - set(terms) or not all(finite(v, 0, 1000) for v in weights.values()):
        raise ValueError(f'Loss weights must be finite, nonnegative and named from {tuple(terms)}')
    if weights and not any(weights.get(k, default) > 0 for k, (_, default) in terms.items()):
        raise ValueError('At least one loss component must retain positive weight')
    aug = config.augmentation
    allowed = {'mode', 'rotation_axis', 'rotation_degrees', 'translation', 'jitter_std', 'jitter_clip'}
    if set(aug) - allowed:
        raise ValueError(f'Unknown augmentation options: {sorted(set(aug)-allowed)}')
    mode = aug.get('mode', 'custom')
    if mode not in ('native', 'none', 'custom'):
        raise ValueError('augmentation.mode must be native, none or custom')
    if mode != 'custom' and set(aug) - {'mode'}:
        raise ValueError('Custom augmentation parameters require mode: custom')
    if aug.get('rotation_axis', 'x') not in ('x', 'y', 'z'):
        raise ValueError('rotation_axis must be x, y or z in camera coordinates')
    for key, maximum in (('rotation_degrees', 180), ('translation', .5), ('jitter_std', .02), ('jitter_clip', .1)):
        if key in aug and not finite(aug[key], 0, maximum):
            raise ValueError(f'Invalid augmentation {key}')


def weighted_loss(native_loss, end_points, config):
    """Keep native masks/reductions; override coefficients without double counting."""
    loss = native_loss
    for name, value in config.loss.get('weights', {}).items():
        key, original = LOSS_TERMS[config.method][name]
        if value != original:
            loss = loss + (value-original)*end_points[key]
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def transform_points(points, poses, options):
    """Rigidly transform observations and poses; noise affects observations only."""
    import numpy as np
    angle = np.deg2rad(np.random.uniform(-1, 1)*options.get('rotation_degrees', 0))
    axis = 'xyz'.index(options.get('rotation_axis', 'x'))
    other = [(axis+1) % 3, (axis+2) % 3]
    rotation = np.eye(3, dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    rotation[np.ix_(other, other)] = [[c, -s], [s, c]]
    delta = options.get('translation', 0)
    shift = np.random.uniform(-delta, delta, size=3).astype(np.float32)
    points = points @ rotation.T + shift
    transformed = []
    for pose in poses:
        new = pose.copy()
        new[:3, :3] = rotation @ pose[:3, :3]
        new[:3, 3] = rotation @ pose[:3, 3] + shift
        transformed.append(new.astype(np.float32))
    std = options.get('jitter_std', 0)
    if std:
        clip = options.get('jitter_clip', .01)
        points = points + np.clip(np.random.normal(0, std, points.shape), -clip, clip)
    return points.astype(np.float32), transformed


def configure_dataset(dataset, config):
    """Called only for training datasets; empty settings preserve native behavior."""
    from functools import partial
    if not config.augmentation:
        return dataset
    mode = config.augmentation.get('mode', 'custom')
    dataset.augment = mode != 'none'
    if mode == 'custom':
        dataset.augment_data = partial(transform_points, options=config.augmentation)
    return dataset


def augment_sample(sample, dataset, config):
    if not config.augmentation or config.augmentation.get('mode') == 'none':
        return sample
    fn = dataset.augment_data if config.augmentation.get('mode') == 'native' else None
    points, poses = (fn(sample['point_clouds'], sample['object_poses_list']) if fn
                     else transform_points(sample['point_clouds'], sample['object_poses_list'], config.augmentation))
    sample['point_clouds'], sample['object_poses_list'] = points, poses
    if 'coors' in sample:
        sample['coors'] = points / config.voxel_size
    return sample
