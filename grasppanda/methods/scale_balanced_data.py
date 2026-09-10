"""Noisy-clean mixing with native grasp targets and explicit cache provenance."""
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=4096)
def _input_digest(path, stamp):
    from ..jobs import digest
    return digest(path)


def enabled(config):
    return config.method == 'scale_balanced_grasp' and config.trainer.get('noisy_clean', False)


def validate(config):
    from ..training.options import finite
    options = config.trainer
    if not isinstance(options, dict): raise ValueError('trainer must be a mapping')
    if options and config.action not in ('train', 'train_check'):
        raise ValueError('Scale-Balanced-Grasp trainer controls require training')
    if set(options) - {'noisy_clean', 'clean_probability'}: raise ValueError('Unknown Scale-Balanced-Grasp trainer controls')
    if type(options.get('noisy_clean', False)) is not bool: raise ValueError('trainer.noisy_clean must be boolean')
    if not finite(options.get('clean_probability', .25), 0, 1): raise ValueError('clean_probability must be between 0 and 1')
    if 'clean_probability' in options and not enabled(config): raise ValueError('clean_probability requires noisy_clean: true')


def clean_root(config):
    from ..config import ROOT
    return (ROOT/Path(config.label_root).expanduser()).resolve() if config.label_root else Path(config.dataset_root)/'clean_scenes'


def frame_paths(config, scene, frame):
    root = clean_root(config)/f'scene_{scene:04d}'/config.camera
    return root/'points'/f'{frame:04d}.npy', root/'seg'/f'{frame:04d}.npy', root/'manifests'/f'{frame:04d}.json'


def verify_frame(config, scene, frame):
    import json
    import numpy as np
    from ..jobs import digest
    points, labels, manifest = frame_paths(config, scene, frame)
    if not all(p.is_file() for p in (points, labels, manifest)):
        raise ValueError(f'Clean scene {scene}/{frame} is missing; run ./panda prepare-clean-scenes and set label_root to its output')
    record = json.loads(manifest.read_text())
    if record.get('format') != 'grasppanda-clean-v1' or (record.get('scene'), record.get('frame'), record.get('camera')) != (scene, frame, config.camera):
        raise ValueError('Clean-scene manifest has a different frame or camera')
    if digest(points) != record['points_sha256'] or digest(labels) != record['seg_sha256']:
        raise ValueError('Clean-scene arrays changed after preparation; use a new cache directory')
    for name, expected in record['inputs'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Invalid clean-scene source path')
        source = Path(config.dataset_root)/name
        stat = source.stat()
        if _input_digest(str(source), (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)) != expected:
            raise ValueError(f'Clean-scene source changed: {name}; prepare a new cache')
    meta = Path(config.dataset_root)/'scenes'/f'scene_{scene:04d}'/config.camera/'meta'/f'{frame:04d}.mat'
    if digest(meta) != record['meta_sha256']: raise ValueError('Clean scene belongs to different frame metadata')
    xyz, seg = np.load(points, allow_pickle=False), np.load(labels, allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Clean points must be nonempty finite camera XYZ')
    if seg.shape != (len(xyz),) or not np.isfinite(seg).all() or not np.equal(seg, seg.astype('int64')).all() or (seg < 0).any() or (seg > 88).any():
        raise ValueError('Clean segmentation must contain matching integer point rows, background 0 and object IDs 1..88')
    return digest(manifest)


def inventory(config):
    import hashlib
    if not enabled(config): return None
    if config.action == 'train':
        start = config.scene*256+config.frame if config.train_batch_limit else 0
        stop = min(start+config.train_batch_limit*config.batch_size, 25600) if config.train_batch_limit else 25600
    else:
        from ..components import requires_scene_batch
        from ..module_options import unpack
        multi = requires_scene_batch(config) or unpack(config.modules.get('backbone', 'upstream'))[0] in ('utonia', 'concerto')
        start = config.scene*256+config.frame; stop = start+(config.batch_size if multi else 1)
    digest = hashlib.sha256()
    for index in range(start, stop): digest.update(verify_frame(config, *divmod(index, 256)).encode())
    return digest.hexdigest()


def mix(noisy, noisy_labels, clean, clean_labels, *, probability):
    import numpy as np
    points, labels = [], []
    for object_id in np.unique(noisy_labels):
        use_clean = np.random.random() <= probability if 0 < probability < 1 else probability == 1
        source, ids = (clean, clean_labels) if use_clean else (noisy, noisy_labels)
        selected = ids == object_id
        points.append(source[selected]); labels.append(ids[selected])
    points, labels = np.concatenate(points), np.concatenate(labels)
    if not len(points): raise ValueError('No selected clean objects have sampled points; increase num_points or inspect the clean cache')
    return points, labels


def augment(sample, dataset, config):
    import numpy as np
    from ..training.options import transform_points, sample_points
    options = config.augmentation
    if not options or options.get('mode') == 'none': return sample
    keys = ('point_clouds', 'noise_point_clouds', 'clear_point_clouds')
    if options.get('mode') == 'native':
        a, b, c, poses, inverse = dataset.augment_data(*(sample[k] for k in keys), sample['object_poses_list'])
        clouds = (a, b, c)
    else:
        sizes = [len(sample[k]) for k in keys]
        cloud, poses, inverse = transform_points(np.concatenate([sample[k] for k in keys]), sample['object_poses_list'], options, return_rotation=True)
        clouds = np.split(cloud, np.cumsum(sizes)[:-1])
    sample.update({k:v.astype(np.float32) for k,v in zip(keys, clouds)})
    sample['object_poses_list'], sample['aug_trans'] = poses, inverse
    return sample_points(sample, config)


class NoisyCleanDataset:
    def __init__(self, dataset, config):
        from functools import partial
        self.dataset, self.config = dataset, config
        dataset.mix = partial(mix, probability=config.trainer.get('clean_probability', .25))
        for index, (scene, frame) in enumerate(zip(dataset.scenename, dataset.frameid)):
            points, labels, _ = frame_paths(config, int(scene.split('_')[1]), frame)
            dataset.pcdpath[index], dataset.segpath[index] = str(points), str(labels)
        custom = config.augmentation and config.augmentation.get('mode', 'custom') != 'native'
        if custom: dataset.augment = False

    def __getattr__(self, name): return getattr(object.__getattribute__(self, 'dataset'), name)
    def __setattr__(self, name, value):
        if name in ('dataset', 'config'): object.__setattr__(self, name, value)
        else: setattr(self.dataset, name, value)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        verify_frame(self.config, int(self.dataset.scenename[index].split('_')[1]), self.dataset.frameid[index])
        sample = self.dataset[index]
        if self.config.action == 'train' and self.config.augmentation and self.config.augmentation.get('mode', 'custom') not in ('native', 'none'):
            sample = augment(sample, self.dataset, self.config)
        return sample
