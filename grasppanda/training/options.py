"""Training controls with explicit supervision and method contracts."""
import math

IMAGE_METHODS = ('hggd', 'region_normalized_grasp')
METHODS = ('graspnet_baseline', 'pointnet2_upgrade', 'scale_balanced_grasp', 'graspness', 'finegrasp', 'gtg2', 'economicgrasp') + IMAGE_METHODS
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
LOSS_TERMS['scale_balanced_grasp'] = {
    'graspable': ('loss/stage1_graspable_loss', 1.),
    **{k: v for k, v in LOSS_TERMS['graspnet_baseline'].items() if k != 'objectness'}}
LOSS_TERMS['gtg2'] = {'score': ('score_loss', 1.)}
LOSS_TERMS['finegrasp'] = {name: (name + '_loss', weight) for name, weight in
    (('objectness', 1.), ('graspness', 10.), ('view', 100.), ('angle', 1.),
     ('depth', 1.), ('score', 1.), ('width', 10.))}
LOSS_TERMS['economicgrasp'] = {name: ('B: ' + name.capitalize() + ' Loss', weight)
    for name, weight in (('objectness', 1.), ('graspness', 10.), ('view', 100.),
                         ('angle', 1.), ('depth', 1.), ('score', 1.), ('width', 10.))}
LOSS_TERMS['hggd'] = {name: (name, weight) for name, weight in
    (('anchor_location', 1.), ('anchor_classification', 1.), ('anchor_theta', 5/3),
     ('anchor_depth', 5/3), ('anchor_width', 5/3), ('local_orientation', 1.), ('local_offset', 1.))}
LOSS_TERMS['region_normalized_grasp'] = {**LOSS_TERMS['hggd'],
    'local_theta_classification': ('local_theta_classification', 1.),
    'local_theta': ('local_theta', 5.), 'local_width': ('local_width', 1.)}


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
    if set(loss) - {'weights', 'functions'} or not isinstance(loss.get('weights', {}), dict):
        raise ValueError('loss accepts weights and functions mappings')
    from grasppanda.training.losses import validate
    validate(config.method, loss.get('functions', {}))
    terms = LOSS_TERMS.get(config.method, {})
    weights = loss.get('weights', {})
    if set(weights) - set(terms) or not all(finite(v, 0, 1000) for v in weights.values()):
        raise ValueError(f'Loss weights must be finite, nonnegative and named from {tuple(terms)}')
    if weights and not any(weights.get(k, default) > 0 for k, (_, default) in terms.items()):
        raise ValueError('At least one loss component must retain positive weight')
    if config.method == 'gtg2':
        from grasppanda.methods.gtg2_options import validate_augmentation
        validate_augmentation(config.augmentation)
        return
    if config.method in IMAGE_METHODS:
        if config.proposal_warmup_steps and not any(weights.get(k, v) > 0 for k, (_, v) in terms.items() if k.startswith('anchor_')):
            raise ValueError('Anchor warmup requires at least one positive anchor loss weight')
        from grasppanda.training.image_augmentation import validate
        validate(config.augmentation)
        return
    aug = config.augmentation
    allowed = {'mode', 'rotation_axis', 'rotation_degrees', 'translation', 'jitter_std', 'jitter_clip',
               'point_dropout', 'cutout_fraction', 'depth_noise_std', 'depth_noise_clip', 'resampling'}
    if set(aug) - allowed:
        raise ValueError(f'Unknown augmentation options: {sorted(set(aug)-allowed)}')
    mode = aug.get('mode', 'custom')
    if mode not in ('native', 'none', 'custom'):
        raise ValueError('augmentation.mode must be native, none or custom')
    if mode != 'custom' and set(aug) - {'mode'}:
        raise ValueError('Custom augmentation parameters require mode: custom')
    if 'resampling' in aug:
        from .point_sampling import validate
        validate(aug['resampling'])
    if aug.get('rotation_axis', 'x') not in ('x', 'y', 'z'):
        raise ValueError('rotation_axis must be x, y or z in camera coordinates')
    for key, maximum in (('rotation_degrees', 180), ('translation', .5), ('jitter_std', .02), ('jitter_clip', .1),
                         ('point_dropout', .8), ('cutout_fraction', .5), ('depth_noise_std', .01), ('depth_noise_clip', .1)):
        if key in aug and not finite(aug[key], 0, maximum):
            raise ValueError(f'Invalid augmentation {key}')


def weighted_loss(native_loss, end_points, config):
    """Keep native masks/reductions; override coefficients without double counting."""
    from grasppanda.module_options import unpack
    if any(unpack(value)[0] != 'upstream' for value in config.loss.get('functions', {}).values()):
        from grasppanda.training.losses import replace_losses
        replace_losses(end_points, config)
        loss = sum(config.loss.get('weights', {}).get(name, coefficient) * end_points[key]
                   for name, (key, coefficient) in LOSS_TERMS[config.method].items())
    else:
        loss = native_loss
        for name, value in config.loss.get('weights', {}).items():
            key, original = LOSS_TERMS[config.method][name]
            if value != original:
                loss = loss + (value-original)*end_points[key]
    end_points['loss/overall_loss'] = loss
    if config.method == 'economicgrasp': end_points['A: Overall Loss'] = loss
    return loss, end_points


def transform_points(points, poses, options, *, return_rotation=False):
    """Rigidly transform observations and poses; noise affects observations only."""
    import numpy as np
    depth_std = options.get('depth_noise_std', 0)
    if depth_std:
        points = points.copy()
        valid = points[:, 2] > 0
        depth = points[valid, 2]
        clip = options.get('depth_noise_clip', .01)
        noise = np.clip(np.random.normal(0, depth_std * depth ** 2), -clip, clip)
        new_depth = np.maximum(depth + noise, 1e-6)
        points[valid] *= (new_depth / depth)[:, None]
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
    result = (points.astype(np.float32), transformed)
    return (*result, rotation.T.copy()) if return_rotation else result


def configure_dataset(dataset, config):
    """Called only for training datasets; empty settings preserve native behavior."""
    from functools import partial
    if not config.augmentation:
        return dataset
    mode = config.augmentation.get('mode', 'custom')
    dataset.augment = mode != 'none'
    if mode == 'custom':
        dataset.augment_data = partial(transform_points, options=config.augmentation,
                                      return_rotation=config.method == 'scale_balanced_grasp')
        if config.augmentation.get('point_dropout', 0) or config.augmentation.get('cutout_fraction', 0) or 'resampling' in config.augmentation:
            return PointSamplingDataset(dataset, config)
    return dataset


class PointSamplingDataset:
    """Apply observation sampling after the native loader has assembled labels."""
    def __init__(self, dataset, config):
        self.dataset, self.config = dataset, config

    def __len__(self): return len(self.dataset)

    def __getattr__(self, name): return getattr(object.__getattribute__(self, 'dataset'), name)

    def __getitem__(self, index):
        return sample_points(self.dataset[index], self.config)


def sample_points(sample, config):
    """Resample observations and all registered per-point labels with one index map."""
    import numpy as np
    options = config.augmentation
    dropout, cutout = options.get('point_dropout', 0), options.get('cutout_fraction', 0)
    if not dropout and not cutout and 'resampling' not in options: return sample
    points = sample['point_clouds']
    count = len(points)
    retained = np.arange(count)
    minimum = min(1024, count)
    if cutout:
        center = points[np.random.randint(count)]
        remove = min(int(count * cutout), count - minimum)
        order = np.argsort(((points - center) ** 2).sum(1), kind='stable')
        retained = np.sort(order[remove:])
    if dropout:
        keep = max(minimum, int(np.ceil(len(retained) * (1 - dropout))))
        retained = np.sort(np.random.choice(retained, keep, replace=False))
    if 'resampling' in options:
        from .point_sampling import retained_indices
        retained = retained[retained_indices(points[retained], options['resampling'])]
    missing = np.ones(count, dtype=bool)
    missing[retained] = False
    indices = np.arange(count)
    indices[missing] = np.random.choice(retained, missing.sum(), replace=True)
    for key in ('point_clouds', 'cloud_colors', 'cloud_normal', 'feats', 'objectness_label', 'graspness_label', 'segmentation_label'):
        if key in sample:
            if len(sample[key]) != count: raise ValueError(f'Per-point field {key} has mismatched rows')
            sample[key] = sample[key][indices].copy()
    for key in ('coors', 'coordinates_for_voxel'):
        if key in sample: sample[key] = sample['point_clouds'] / config.voxel_size
    return sample


def augment_sample(sample, dataset, config):
    if not config.augmentation or config.augmentation.get('mode') == 'none':
        return sample
    fn = dataset.augment_data if config.augmentation.get('mode') == 'native' else None
    transformed = (fn(sample['point_clouds'], sample['object_poses_list']) if fn
                   else transform_points(sample['point_clouds'], sample['object_poses_list'], config.augmentation,
                                         return_rotation=config.method == 'scale_balanced_grasp'))
    points, poses = transformed[:2]
    if config.method == 'scale_balanced_grasp': sample['aug_trans'] = transformed[2]
    # Native loaders cast observations after augmentation; retain that contract here.
    points = points.astype(sample['point_clouds'].dtype, copy=False)
    sample['point_clouds'], sample['object_poses_list'] = points, poses
    for key in ('coors', 'coordinates_for_voxel'):
        if key in sample: sample[key] = points / config.voxel_size
    return sample_points(sample, config)


def snapshot_components(model, prefixes):
    """Distinguish graph-connected masked zeros from disconnected components."""
    import torch
    snapshots, zero_gradients = {}, []
    for prefix in prefixes:
        connected = [p for name, p in model.named_parameters()
                     if name.startswith(prefix) and p.grad is not None]
        if not connected:
            raise ValueError(f'No gradient graph reaches replacement component {prefix}')
        active = [p for p in connected if torch.count_nonzero(p.grad)]
        if not active:
            zero_gradients.append(prefix)
        parameter = (active or connected)[0]
        snapshots[prefix] = (parameter, parameter.detach().clone())
    return snapshots, zero_gradients
